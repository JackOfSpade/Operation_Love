"""BigQuery backend — system-of-record + analytics, with batched writes.

Design (per the chosen 'BQ of-record + memory cache' approach):
- load_labels() runs ONE query at startup; the orchestrator holds labels in
  memory for fast inference.
- writes are buffered and flushed in batches (default every 25 rows, and on
  close), so the swipe loop never waits on a per-row cloud round-trip and we
  never read-back a just-streamed row (avoids BigQuery's streaming-buffer
  consistency gap).
- profile photos are archived as private bucket objects; BigQuery stores the
  image manifest, labels, and embeddings keyed by profile_id.

At personal scale this stays within BigQuery's free tier. The ``client`` is
injectable so tests run without the google-cloud-bigquery package or network.
"""
from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime, timezone

from ..costing import Usage

_UPLOAD_ATTEMPTS = 3        # bounded retry so a transient GCS blip doesn't drop a swipe
_UPLOAD_BACKOFF_S = 0.5

_TABLES = {
    "profiles": (
        "run_id STRING, app STRING, profile_id STRING, created_at TIMESTAMP, liked BOOL, "
        "source STRING, photo_count INT64"
    ),
    "profile_photos": (
        "run_id STRING, app STRING, profile_id STRING, created_at TIMESTAMP, "
        "photo_index INT64, gcs_uri STRING, sha256 STRING, byte_size INT64, content_type STRING"
    ),
    "labels": (
        "run_id STRING, app STRING, profile_id STRING, created_at TIMESTAMP, liked BOOL, "
        "source STRING, embedding ARRAY<FLOAT64>, photo_count INT64"
    ),
    "decisions": (
        "run_id STRING, app STRING, created_at TIMESTAMP, decision STRING, "
        "score FLOAT64, source STRING"
    ),
    "openers": "run_id STRING, app STRING, created_at TIMESTAMP, model STRING, opener STRING, referenced STRING",
    "spend": (
        "run_id STRING, created_at TIMESTAMP, model STRING, input_tokens INT64, output_tokens INT64, "
        "cache_read_tokens INT64, cache_write_tokens INT64, cost_usd FLOAT64"
    ),
}

_MIGRATIONS = (
    "ALTER TABLE `{labels}` ADD COLUMN IF NOT EXISTS profile_id STRING;",
    "ALTER TABLE `{decisions}` ADD COLUMN IF NOT EXISTS source STRING;",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy_labels(labels: list[tuple[bool, list[float]]]) -> list[tuple[bool, list[float]]]:
    return [(liked, list(embedding)) for liked, embedding in labels]


def _image_type(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", "jpg"
    return "application/octet-stream", "bin"


class BigQueryStore:
    def __init__(self, project_id: str, dataset: str = "operation_love",
                 location: str = "US", photo_bucket: str = "", flush_every: int = 25,
                 client=None, storage_client=None, ensure: bool = True):
        if not project_id:
            raise ValueError("Storage.bigquery.project_id is required for the BigQuery backend")
        if not photo_bucket:
            raise ValueError("Storage.bigquery.photo_bucket is required for the BigQuery backend")
        self.project_id = project_id
        self.dataset = dataset
        self.location = location
        self.photo_bucket_name = photo_bucket
        self.flush_every = max(1, int(flush_every))
        if client is None:
            from google.cloud import bigquery  # lazy: only needed for real use
            client = bigquery.Client(project=project_id)
        if storage_client is None:
            from google.cloud import storage  # lazy: only needed for real use
            storage_client = storage.Client(project=project_id)
        self.client = client
        self.storage_client = storage_client
        self._buf: dict[str, list[dict]] = {name: [] for name in _TABLES}
        self._written: dict[str, int] = {name: 0 for name in _TABLES}  # rows confirmed inserted this run
        self._label_count = 0
        self._labels_cache: list[tuple[bool, list[float]]] | None = None
        self._lock = threading.RLock()  # shared across worker threads
        self._photo_bucket = self._get_or_create_photo_bucket() if ensure else self.storage_client.bucket(photo_bucket)
        if ensure:
            self._ensure_tables()

    # --- setup ----------------------------------------------------------
    def _tid(self, name: str) -> str:
        return f"{self.project_id}.{self.dataset}.{name}"

    def _ensure_tables(self) -> None:
        # One multi-statement script = a single job submission instead of 5
        # sequential round-trips, so Start isn't gated on ~5 BigQuery job latencies.
        stmts = [f"CREATE SCHEMA IF NOT EXISTS `{self.project_id}.{self.dataset}` "
                 f"OPTIONS(location='{self.location}');"]
        for name, cols in _TABLES.items():
            stmts.append(f"CREATE TABLE IF NOT EXISTS `{self._tid(name)}` ({cols});")
        tids = {name: self._tid(name) for name in _TABLES}
        for stmt in _MIGRATIONS:
            stmts.append(stmt.format(**tids))
        self.client.query("\n".join(stmts)).result()

    def _get_or_create_photo_bucket(self):
        bucket = self.storage_client.bucket(self.photo_bucket_name)
        if bucket.exists():
            return bucket
        bucket = self.storage_client.create_bucket(bucket, location=self.location)
        # Private by default: uniform bucket-level access + enforced public-access-
        # prevention so a freshly created bucket can never be exposed publicly. Log
        # (don't silently swallow) if the lockdown patch fails — it's a privacy gap.
        try:
            bucket.iam_configuration.uniform_bucket_level_access_enabled = True
            bucket.iam_configuration.public_access_prevention = "enforced"
            bucket.patch()
        except Exception as exc:  # noqa: BLE001
            print(f"BigQuery store warning: could not lock down bucket "
                  f"{self.photo_bucket_name} (uniform access / public-access-prevention): {exc}")
        return bucket

    # --- reads ----------------------------------------------------------
    def load_labels(self) -> list[tuple[bool, list[float]]]:
        # Hold the lock across the whole fill so the cache-miss check, the SELECT,
        # and folding in buffered rows are atomic (no concurrent add_label can be
        # dropped). Called once at startup before workers spawn, so the in-lock
        # query doesn't contend in practice; RLock keeps it reentrancy-safe.
        with self._lock:
            if self._labels_cache is not None:
                return _copy_labels(self._labels_cache)
            rows = self.client.query(f"SELECT liked, embedding FROM `{self._tid('labels')}`").result()
            out = [(bool(r["liked"]), list(r["embedding"])) for r in rows]
            out.extend((bool(r["liked"]), list(r["embedding"])) for r in self._buf["labels"])
            self._labels_cache = _copy_labels(out)
            self._label_count = len(self._labels_cache)
            return _copy_labels(self._labels_cache)

    def load_labels_ordered(self) -> list[tuple[bool, list[float]]]:
        """Committed labels in swipe order (created_at asc) for the quality-trajectory
        chart. Deliberately does NOT take self._lock or read the in-memory buffer: it's
        called repeatedly by the hub's eval thread while workers are writing, so holding
        the lock across this (multi-second) query would stall swipes. The unflushed tail
        is therefore omitted here — the hub appends the live full-set point separately.
        Streaming-buffer rows may lag, which is fine for a historical trend."""
        rows = self.client.query(
            f"SELECT liked, embedding FROM `{self._tid('labels')}` ORDER BY created_at"
        ).result()
        return [(bool(r["liked"]), list(r["embedding"])) for r in rows]

    def label_count(self) -> int:
        with self._lock:
            return self._label_count

    def count_today(self, app: str) -> int:
        safe = app.replace("'", "").replace("\\", "")
        rows = self.client.query(
            f"SELECT COUNT(*) AS c FROM `{self._tid('decisions')}` "
            f"WHERE app='{safe}' AND created_at >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), DAY) "
            "AND source='auto'"
        ).result()
        for r in rows:
            return int(r["c"])
        return 0

    # --- writes (buffered, thread-safe) --------------------------------
    def record_profile(self, run_id, app, profile_id, liked, source="manual",
                       photos=None, photo_count=0) -> bool:
        """Archive the profile's images + manifest row. Returns True if recorded.

        Image archiving is the system of record, so it is mandatory but non-fatal
        to the swipe loop: each upload is retried; a transient GCS blip never
        raises. If any photo cannot be stored after retries, we record no manifest
        and return False so the worker skips that swipe's label instead of keeping
        a label without the complete profile image set.
        """
        photos = list(photos or [])
        if not photos:
            print(f"BigQuery store warning: captured 0 photos for profile {profile_id}; "
                  "skipping profile archive.")
            return False
        created_at = _now()
        photo_rows = self._upload_profile_photos(run_id, app, profile_id, created_at, photos)
        if len(photo_rows) != len(photos):
            if photo_rows:
                self._delete_profile_photo_rows(photo_rows)
            print(f"BigQuery store warning: archived {len(photo_rows)}/{len(photos)} photos for "
                  f"profile {profile_id}; skipping its label to keep image data complete.")
            return False
        with self._lock:
            self._buf["profiles"].append({
                "run_id": run_id, "app": app, "profile_id": profile_id, "created_at": created_at,
                "liked": bool(liked), "source": source, "photo_count": len(photo_rows),
            })
            self._buf["profile_photos"].extend(photo_rows)
            self._maybe_flush("profiles")
            self._maybe_flush("profile_photos")
        return True

    def _upload_blob(self, blob, data: bytes, content_type: str) -> bool:
        delay = _UPLOAD_BACKOFF_S
        for attempt in range(1, _UPLOAD_ATTEMPTS + 1):
            try:
                blob.upload_from_string(data, content_type=content_type)
                return True
            except Exception as exc:  # noqa: BLE001
                if attempt == _UPLOAD_ATTEMPTS:
                    print(f"BigQuery store photo upload failed after {_UPLOAD_ATTEMPTS} "
                          f"attempts ({getattr(blob, 'name', '?')}): {exc}")
                    return False
                time.sleep(delay)
                delay *= 2
        return False

    def _upload_profile_photos(self, run_id: str, app: str, profile_id: str,
                               created_at: str, photos: list[bytes]) -> list[dict]:
        rows = []
        for i, photo in enumerate(photos):
            digest = hashlib.sha256(photo).hexdigest()
            content_type, ext = _image_type(photo)
            object_name = f"profiles/{app}/{run_id}/{profile_id}/{i:02d}-{digest[:16]}.{ext}"
            blob = self._photo_bucket.blob(object_name)
            if not self._upload_blob(blob, photo, content_type):
                break
            rows.append({
                "run_id": run_id, "app": app, "profile_id": profile_id, "created_at": created_at,
                "photo_index": i, "gcs_uri": f"gs://{self.photo_bucket_name}/{object_name}",
                "sha256": digest, "byte_size": len(photo), "content_type": content_type,
            })
        return rows

    def _delete_profile_photo_rows(self, rows: list[dict]) -> None:
        prefix = f"gs://{self.photo_bucket_name}/"
        for row in rows:
            uri = str(row.get("gcs_uri") or "")
            if not uri.startswith(prefix):
                continue
            try:
                self._photo_bucket.blob(uri[len(prefix):]).delete()
            except Exception as exc:  # noqa: BLE001
                print(f"BigQuery store warning: could not delete partial photo {uri}: {exc}")

    def add_label(self, run_id, app, liked, embedding, source="manual", photo_count=0,
                  profile_id="", **_):
        liked = bool(liked)
        embedding_vec = [float(x) for x in embedding]
        with self._lock:
            self._buf["labels"].append({
                "run_id": run_id, "app": app, "profile_id": profile_id, "created_at": _now(), "liked": liked,
                "source": source, "embedding": embedding_vec, "photo_count": int(photo_count),
            })
            label = (liked, list(embedding_vec))
            if self._labels_cache is not None:
                self._labels_cache.append(label)
            self._label_count += 1
            self._maybe_flush("labels")

    def record_decision(self, run_id, app, decision, score, source="auto"):
        with self._lock:
            self._buf["decisions"].append({
                "run_id": run_id, "app": app, "created_at": _now(),
                "decision": decision, "score": float(score), "source": source,
            })
            self._maybe_flush("decisions")

    def record_opener(self, run_id, app, model, opener, referenced):
        with self._lock:
            self._buf["openers"].append({
                "run_id": run_id, "app": app, "created_at": _now(),
                "model": model, "opener": opener, "referenced": referenced,
            })
            self._maybe_flush("openers")

    def record_spend(self, run_id, model, usage: Usage, cost):
        with self._lock:
            self._buf["spend"].append({
                "run_id": run_id, "created_at": _now(), "model": model,
                "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_input_tokens,
                "cache_write_tokens": usage.cache_creation_input_tokens, "cost_usd": float(cost),
            })
            self._maybe_flush("spend")

    # --- flush ----------------------------------------------------------
    def _maybe_flush(self, table: str) -> None:  # caller holds self._lock
        if len(self._buf[table]) >= self.flush_every:
            self._flush_table(table)

    def _flush_table(self, table: str) -> None:  # caller holds self._lock
        rows = self._buf[table]
        if not rows:
            return
        errors = self.client.insert_rows_json(self._tid(table), rows)
        if errors:
            raise RuntimeError(f"BigQuery insert errors for {table}: {errors}")
        self._written[table] += len(rows)
        self._buf[table] = []

    def flush(self) -> None:
        with self._lock:
            for table in self._buf:
                self._flush_table(table)

    def close(self) -> None:
        self.flush()

    def saved_summary(self) -> str:
        """Human-readable tally of rows confirmed inserted to BigQuery this run, for the
        shutdown confirmation log. Empty buffers + a non-empty tally = everything landed."""
        with self._lock:
            parts = [f"{name}={self._written[name]}" for name in _TABLES if self._written[name]]
            pending = sum(len(self._buf[name]) for name in _TABLES)
        body = ", ".join(parts) if parts else "nothing new"
        return f"{body} (pending={pending})" if pending else body
