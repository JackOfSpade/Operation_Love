"""BigQuery backend — system-of-record + analytics, with batched writes.

Design (per the chosen 'BQ of-record + memory cache' approach):
- load_labels() runs ONE query at startup; the supervisor holds labels in
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
import json
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

from ..costing import Usage
from .store import local_midnight_epoch

_UPLOAD_ATTEMPTS = 3        # bounded retry so a transient GCS blip doesn't drop a swipe
_UPLOAD_BACKOFF_S = 0.5
# Shared with ranker/__init__.py's make_store(), so the config-parsing fallback (when
# storage.bigquery.flush_every isn't set) can't silently drift from this constructor's
# own default.
DEFAULT_FLUSH_EVERY = 25

_MAX_INSERT_ATTEMPTS = 5    # bounded retry so a row BigQuery keeps rejecting as invalid
                            # can't poison its table's buffer (and everything queued
                            # behind it) forever -- see _flush_table

_TABLES = {
    "profiles": (
        "run_id STRING, app STRING, profile_id STRING, created_at TIMESTAMP, liked BOOL, "
        "source STRING, photo_count INT64, capture_truncated BOOL"
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
    # `angle` is the model's own free-text description of what the opener is doing (a guess, a
    # tease, a connection between two things she wrote). Telemetry only, never read back at
    # runtime; it exists so "which opener shapes correlate with matches" becomes an answerable
    # query. See _MIGRATIONS below — this line alone does NOT add it to the live table.
    #
    # `item_description` is the model's own short description of the ITEM it picked to write
    # about and to like (ops/OPENER-REDESIGN.md 5.7). Deliberately NOT a duplicate of
    # `referenced`: that column is the DETAIL the opener reacts to (and what the redundancy
    # monitor compares the opener against), this one says what the item IS -- a photo, a
    # written prompt -- which is the coarse class doc 5.8's pre-flight cross-check works on.
    # Written in both auto and observe, always. Same _MIGRATIONS caveat as `angle`.
    "openers": ("run_id STRING, app STRING, created_at TIMESTAMP, model STRING, opener STRING, "
                "referenced STRING, angle STRING, item_description STRING"),
    # Every REJECTED opener attempt (OpenerParseError), not just the successes `openers`
    # above holds -- see opener/service.py's OpenerParseError handling and opener.py's
    # OpenerParseError docstring for reason_code/raw_opener semantics. `attempt` is the
    # 1-based retry count within maybe_opener()'s per-profile retry loop, so a run of
    # consecutive rejections that exhausted the loop (service.py's max_attempts) is
    # queryable as clearly as one that succeeded on the first retry.
    "opener_rejections": (
        "run_id STRING, app STRING, created_at TIMESTAMP, model STRING, attempt INT64, "
        "reason_code STRING, reason STRING, raw_opener STRING"
    ),
    "spend": (
        "run_id STRING, created_at TIMESTAMP, model STRING, input_tokens INT64, output_tokens INT64, "
        "cache_read_tokens INT64, cache_write_tokens INT64, cost_usd FLOAT64"
    ),
}

# Every column added to a table AFTER that table first shipped needs BOTH an entry in _TABLES
# above and a line here, because the two reach different databases. The _TABLES column list is
# only ever applied by CREATE TABLE IF NOT EXISTS, which is a silent no-op against a table that
# already exists — so on its own it reaches new/empty projects only. The live project's tables
# already exist and already hold real rows, so the ALTER below is the ONLY thing that puts the
# column there, and without it the first insert carrying the new field would be rejected as
# "no such field" in production while passing every local test. Additive + nullable in both
# directions, so old rows simply read back NULL.
_MIGRATIONS = (
    "ALTER TABLE `{labels}` ADD COLUMN IF NOT EXISTS profile_id STRING;",
    "ALTER TABLE `{decisions}` ADD COLUMN IF NOT EXISTS source STRING;",
    "ALTER TABLE `{profiles}` ADD COLUMN IF NOT EXISTS capture_truncated BOOL;",
    "ALTER TABLE `{openers}` ADD COLUMN IF NOT EXISTS angle STRING;",
    "ALTER TABLE `{openers}` ADD COLUMN IF NOT EXISTS item_description STRING;",
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


def _row_id(row: dict) -> str:
    """Deterministic insert id for a buffered row, derived from its own content (NOT a
    random uuid) so BigQuery's best-effort streaming dedup recognizes a row resent on
    retry as the same logical row instead of inserting it again."""
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def _day_start_job_config(start_dt: datetime):
    """QueryJobConfig binding `start_dt` as the `day_start` TIMESTAMP parameter shared by
    count_today/spend_today (see local_midnight_epoch). Falls back to a minimal duck-typed
    stand-in when the SDK isn't installed: the ``client`` is injectable (see module
    docstring) so tests run against a fake client and never touch the real BigQuery API."""
    try:
        from google.cloud import bigquery
    except ImportError:
        return SimpleNamespace(query_parameters=[
            SimpleNamespace(name="day_start", type_="TIMESTAMP", value=start_dt)])
    return bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("day_start", "TIMESTAMP", start_dt)])


class BigQueryStore:
    def __init__(self, project_id: str, dataset: str = "operation_love",
                 location: str = "US", photo_bucket: str = "", flush_every: int = DEFAULT_FLUSH_EVERY,
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
        self._dropped: dict[str, int] = {name: 0 for name in _TABLES}  # rows permanently given up on (never inserted)
        self._fail_counts: dict[str, int] = {}  # f"{table}:{row_id}" -> consecutive failed-insert attempts
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
            try:
                bucket.reload()             # fetch iam_configuration before inspecting it
            except Exception:  # noqa: BLE001
                pass
            self._lockdown_bucket(bucket, created=False)
            return bucket
        bucket = self.storage_client.create_bucket(bucket, location=self.location)
        self._lockdown_bucket(bucket, created=True)
        return bucket

    def _lockdown_bucket(self, bucket, *, created: bool) -> None:
        """Make the photo bucket private: uniform bucket-level access + enforced public-
        access-prevention so it can never be exposed publicly. Applied to EXISTING buckets
        too (idempotently), not just freshly created ones — a bucket created before this
        hardening would otherwise keep weaker defaults (e.g. fine-grained ACLs). Logs but
        never raises: a permissions hiccup must not break startup, but it's a privacy gap
        worth surfacing."""
        try:
            iam = bucket.iam_configuration
            if (getattr(iam, "uniform_bucket_level_access_enabled", False)
                    and getattr(iam, "public_access_prevention", "") == "enforced"):
                return                      # already locked down -> no needless patch
            iam.uniform_bucket_level_access_enabled = True
            iam.public_access_prevention = "enforced"
            bucket.patch()
            if not created:
                print(f"BigQuery store: hardened existing photo bucket {self.photo_bucket_name} "
                      "(uniform bucket-level access + public-access-prevention enforced).")
        except Exception as exc:  # noqa: BLE001
            print(f"BigQuery store warning: could not lock down bucket "
                  f"{self.photo_bucket_name} (uniform access / public-access-prevention): {exc}")

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

    def count_today(self, app: str) -> int:
        """Auto-mode swipes recorded today (LOCAL day — same boundary as
        SQLiteStore.count_today, via local_midnight_epoch(), NOT a UTC reporting day)."""
        safe = app.replace("'", "").replace("\\", "")
        start_dt = datetime.fromtimestamp(local_midnight_epoch(), tz=timezone.utc)
        rows = self.client.query(
            f"SELECT COUNT(*) AS c FROM `{self._tid('decisions')}` "
            f"WHERE app='{safe}' AND created_at >= @day_start AND source='auto'",
            job_config=_day_start_job_config(start_dt),
        ).result()
        for r in rows:
            return int(r["c"])
        return 0

    def spend_today(self) -> float:
        """Sum of cost_usd already committed to BigQuery today (LOCAL day — same
        boundary as SQLiteStore.spend_today, via local_midnight_epoch(), NOT a UTC
        reporting day). Falls back to 0.0 on any error — used to seed the daily budget
        floor."""
        try:
            start_dt = datetime.fromtimestamp(local_midnight_epoch(), tz=timezone.utc)
            rows = self.client.query(
                f"SELECT COALESCE(SUM(cost_usd), 0.0) AS total FROM `{self._tid('spend')}` "
                "WHERE created_at >= @day_start",
                job_config=_day_start_job_config(start_dt),
            ).result()
            for r in rows:
                return float(r["total"])
        except Exception:  # noqa: BLE001
            pass
        return 0.0

    # --- writes (buffered, thread-safe) --------------------------------
    def record_profile(self, run_id, app, profile_id, liked, source="manual",
                       photos=None, photo_count=0, capture_truncated: bool = False) -> bool:
        """Archive the profile's images + manifest row. Returns True if recorded.

        Image archiving is the system of record, so it is mandatory but non-fatal
        to the swipe loop: each upload is retried; a transient GCS blip never
        raises. If any photo cannot be stored after retries, we record no manifest
        and return False so the worker skips that swipe's label instead of keeping
        a label without the complete profile image set.

        `capture_truncated` records whether the driver hit its scroll ceiling
        (e.g. Hinge's scroll_captures) before reaching the profile's real bottom,
        i.e. the label attached to this profile was made from an INCOMPLETE read.
        That's already in the local debug log and Profile.meta, but this column is
        what lets a later BigQuery query actually ask "which labels came from a
        truncated read?" (useful if those turn out noisier) instead of the answer
        being dropped on the way into the system of record. Additive/nullable (see
        _MIGRATIONS) so it defaults to False for callers that don't pass it.
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
                "capture_truncated": bool(capture_truncated),
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
            self._maybe_flush("labels")

    def record_decision(self, run_id, app, decision, score, source="auto"):
        with self._lock:
            self._buf["decisions"].append({
                "run_id": run_id, "app": app, "created_at": _now(),
                "decision": decision, "score": float(score), "source": source,
            })
            self._maybe_flush("decisions")

    def record_opener(self, run_id, app, model, opener, referenced, angle="",
                      item_description=""):
        # `angle` and `item_description`: telemetry only (see the openers entry in _TABLES).
        # Both defaulted to "" so a caller that predates either still writes a valid row rather
        # than omitting the field. The live table already holds real rows, so each column got
        # to production via _MIGRATIONS, not via CREATE TABLE IF NOT EXISTS.
        with self._lock:
            self._buf["openers"].append({
                "run_id": run_id, "app": app, "created_at": _now(),
                "model": model, "opener": opener, "referenced": referenced, "angle": angle,
                "item_description": item_description,
            })
            self._maybe_flush("openers")

    def record_opener_rejection(self, run_id, app, model, attempt, reason_code, reason, raw_opener):
        with self._lock:
            self._buf["opener_rejections"].append({
                "run_id": run_id, "app": app, "created_at": _now(), "model": model,
                "attempt": int(attempt), "reason_code": reason_code, "reason": reason,
                "raw_opener": raw_opener,
            })
            self._maybe_flush("opener_rejections")

    def record_spend(self, run_id, model, usage: Usage, cost):
        # cost is None when the call's price couldn't be determined (e.g. no
        # budget.pricing entry for the model) — stored as NULL, distinct from a
        # genuinely free $0.00 call.
        with self._lock:
            self._buf["spend"].append({
                "run_id": run_id, "created_at": _now(), "model": model,
                "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_input_tokens,
                "cache_write_tokens": usage.cache_creation_input_tokens,
                "cost_usd": None if cost is None else float(cost),
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
        row_ids = [_row_id(row) for row in rows]
        # skip_invalid_rows is deliberately left unset (BigQuery's documented default,
        # False): per google.cloud.bigquery.Client.insert_rows_json's own docstring,
        # that means "if any invalid rows exist [...] the entire request [fails]" --
        # i.e. NOT partial acceptance. `errors` names only the offending row(s), but
        # rows absent from it were NOT written either; the whole request was rejected.
        errors = self.client.insert_rows_json(self._tid(table), rows, row_ids=row_ids)
        if errors:
            # Do NOT assume any row in this batch was accepted -- with skip_invalid_rows
            # unset, none were. Keep the whole batch buffered so it gets resent on the
            # next flush; this is safe (won't double-insert anything BigQuery actually
            # did ingest, e.g. via a transport-level retry) because row_ids are stable,
            # content-derived ids (see _row_id) that BigQuery's own best-effort streaming
            # dedup recognizes as "already seen".
            #
            # A row that is genuinely, permanently invalid would otherwise sit in the
            # buffer forever: every flush re-includes it, keeps failing, and blocks every
            # valid row queued behind it. Bound that -- after _MAX_INSERT_ATTEMPTS
            # consecutive failures naming the same row (tracked by its stable row_id, not
            # its position, since position shifts as the buffer changes across retries),
            # drop just that row, loudly, and keep the rest of the buffer intact.
            bad_ids = {row_ids[e["index"]] for e in errors if e.get("index") is not None}
            for rid in bad_ids:
                key = f"{table}:{rid}"
                self._fail_counts[key] = self._fail_counts.get(key, 0) + 1
            drop_ids = {rid for rid in bad_ids if self._fail_counts[f"{table}:{rid}"] >= _MAX_INSERT_ATTEMPTS}
            if drop_ids:
                kept, dropped = [], []
                for row, rid in zip(rows, row_ids):
                    (dropped if rid in drop_ids else kept).append(row)
                for row in dropped:
                    print(f"BigQuery store: DROPPING a row from '{table}' after "
                          f"{_MAX_INSERT_ATTEMPTS} consecutive failed insert attempts -- "
                          f"this row's data is LOST (never written to BigQuery). "
                          f"BigQuery errors: {errors}. Row: {row}")
                self._dropped[table] += len(dropped)
                for rid in drop_ids:
                    self._fail_counts.pop(f"{table}:{rid}", None)
                self._buf[table] = kept
            raise RuntimeError(f"BigQuery insert errors for {table}: {errors}")
        # Whole request succeeded -- every row in it was actually written.
        for rid in row_ids:
            self._fail_counts.pop(f"{table}:{rid}", None)
        self._written[table] += len(rows)
        self._buf[table] = []

    def flush(self) -> None:
        errors: list[Exception] = []
        with self._lock:
            for table in self._buf:
                try:
                    self._flush_table(table)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
                    print(f"BigQuery flush error for table '{table}': {exc}")
        if errors:
            raise RuntimeError(
                f"BigQuery flush failed for {len(errors)} table(s): "
                + "; ".join(str(e) for e in errors)
            )

    def close(self) -> None:
        self.flush()

    def saved_summary(self) -> str:
        """Human-readable tally of rows confirmed inserted to BigQuery this run, for the
        shutdown confirmation log. Empty buffers + a non-empty tally = everything landed."""
        with self._lock:
            parts = [f"{name}={self._written[name]}" for name in _TABLES if self._written[name]]
            dropped = [f"{name}={self._dropped[name]}" for name in _TABLES if self._dropped[name]]
            pending = sum(len(self._buf[name]) for name in _TABLES)
        body = ", ".join(parts) if parts else "nothing new"
        if dropped:
            body += f" (DROPPED, never written: {', '.join(dropped)})"
        return f"{body} (pending={pending})" if pending else body
