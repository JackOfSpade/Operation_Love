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
import re
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

_SCHEMA_UPDATE_ATTEMPTS = 5
_SCHEMA_UPDATE_BACKOFF_S = 1.0

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
        "source STRING, embedding ARRAY<FLOAT64>, photo_count INT64, profile_name STRING"
    ),
    "decisions": (
        "run_id STRING, app STRING, created_at TIMESTAMP, decision STRING, "
        "score FLOAT64, source STRING, profile_id STRING"
    ),
    "label_retractions": (
        "correction_id STRING, run_id STRING, app STRING, source STRING, profile_id STRING, "
        "label_created_at TIMESTAMP, decision_created_at TIMESTAMP, decision_fingerprint STRING, "
        "reason STRING, evidence_ref STRING, created_at TIMESTAMP"
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
                "referenced STRING, angle STRING, item_description STRING, profile_id STRING, "
                "decision STRING, decision_source STRING, decision_created_at TIMESTAMP, "
                "model_item_index INT64"),
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
    "opener_retractions": (
        "correction_id STRING, run_id STRING, app STRING, opener_created_at TIMESTAMP, model STRING, "
        "opener_fingerprint STRING, reason STRING, evidence_ref STRING, created_at TIMESTAMP"
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
    "ALTER TABLE `{labels}` ADD COLUMN IF NOT EXISTS profile_name STRING;",
    "ALTER TABLE `{decisions}` ADD COLUMN IF NOT EXISTS source STRING;",
    "ALTER TABLE `{decisions}` ADD COLUMN IF NOT EXISTS profile_id STRING;",
    "ALTER TABLE `{profiles}` ADD COLUMN IF NOT EXISTS capture_truncated BOOL;",
    "ALTER TABLE `{openers}` ADD COLUMN IF NOT EXISTS angle STRING;",
    "ALTER TABLE `{openers}` ADD COLUMN IF NOT EXISTS item_description STRING;",
    "ALTER TABLE `{openers}` ADD COLUMN IF NOT EXISTS profile_id STRING;",
    "ALTER TABLE `{openers}` ADD COLUMN IF NOT EXISTS decision STRING;",
    "ALTER TABLE `{openers}` ADD COLUMN IF NOT EXISTS decision_source STRING;",
    "ALTER TABLE `{openers}` ADD COLUMN IF NOT EXISTS decision_created_at TIMESTAMP;",
    "ALTER TABLE `{openers}` ADD COLUMN IF NOT EXISTS model_item_index INT64;",
)

_MIGRATION_RE = re.compile(
    r"^ALTER TABLE `\{(?P<table>[a-z_]+)\}` ADD COLUMN IF NOT EXISTS "
    r"(?P<column>[A-Za-z_][A-Za-z0-9_]*) (?P<type>[^;]+);$"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp(value) -> str:
    """Normalize a shared action timestamp for BigQuery's JSON TIMESTAMP representation."""
    if value is None:
        return _now()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    return str(value)


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


def _day_start_job_config(start_dt: datetime, *, app: str | None = None):
    """QueryJobConfig binding `start_dt` as the `day_start` TIMESTAMP parameter shared by
    count_today/spend_today (see local_midnight_epoch), and optionally the application name.
    Falls back to a minimal duck-typed stand-in when the SDK isn't installed: the ``client`` is
    injectable (see module docstring) so tests run against a fake client and never touch the
    real BigQuery API."""
    parameters = [("day_start", "TIMESTAMP", start_dt)]
    if app is not None:
        parameters.append(("app", "STRING", app))
    try:
        from google.cloud import bigquery
    except ImportError:
        return SimpleNamespace(query_parameters=[
            SimpleNamespace(name=name, type_=kind, value=value)
            for name, kind, value in parameters
        ])
    return bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter(name, kind, value)
        for name, kind, value in parameters
    ])


def _missing_optional_opener_retractions(exc: Exception, *, ensure: bool) -> bool:
    """Only the known optional-table NotFound is empty; every other read error is real."""
    return (not ensure and type(exc).__name__ == "NotFound"
            and "opener_retractions" in str(exc))


def _missing_dataset(exc: Exception) -> bool:
    """Recognize only BigQuery's dataset-not-found response during schema discovery."""
    return type(exc).__name__ == "NotFound" and "dataset" in str(exc).lower()


def _table_update_rate_limited(exc: Exception) -> bool:
    """The metadata-update quota is transient and BigQuery explicitly recommends retry."""
    message = str(exc).lower()
    return ("exceeded rate limits" in message
            and "too many table update operations" in message)


def _migration_parts(stmt: str) -> tuple[str, str, str]:
    """Return the table placeholder, column name, and type from a declared migration.

    Keeping ``_MIGRATIONS`` as executable SQL preserves the two-part schema rollout
    invariant documented above. Parsing our deliberately narrow internal format lets
    startup discover which migrations are actually needed and combine all additions for
    one table into one metadata update.
    """
    match = _MIGRATION_RE.fullmatch(stmt)
    if match is None:
        raise ValueError(f"Unsupported BigQuery migration statement: {stmt}")
    return match["table"], match["column"], match["type"]


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
        self._ensure = bool(ensure)
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
        self._retraction_ids: set[str] = set()  # closes the streaming-read visibility gap for retries
        self._opener_retraction_ids: set[tuple[str, str]] = set()
        self._lock = threading.RLock()  # shared across worker threads
        self._photo_bucket = self._get_or_create_photo_bucket() if ensure else self.storage_client.bucket(photo_bucket)
        if ensure:
            self._ensure_tables()

    # --- setup ----------------------------------------------------------
    def _tid(self, name: str) -> str:
        return f"{self.project_id}.{self.dataset}.{name}"

    def _ensure_tables(self) -> None:
        """Create missing tables and apply only missing additive migrations.

        BigQuery limits a standard table to five metadata updates per ten seconds.  The
        old startup script replayed every historical ``ALTER TABLE`` on every run and,
        after the opener telemetry additions, targeted ``openers`` seven times in one
        script.  ``IF NOT EXISTS`` made the DDL logically idempotent but did not keep those
        statements out of the metadata-update quota.

        One INFORMATION_SCHEMA read makes the common (already-current) startup entirely
        free of table DDL.  A legacy table receives one grouped ALTER regardless of how
        many nullable columns it is missing; a new table is created with the current full
        schema and therefore needs no follow-up ALTER.
        """
        schema_query = (
            "SELECT table_name, column_name "
            f"FROM `{self.project_id}.{self.dataset}.INFORMATION_SCHEMA.COLUMNS` "
            "WHERE table_name IN ("
            + ", ".join(f"'{name}'" for name in _TABLES)
            + ")"
        )
        try:
            rows = self.client.query(schema_query).result()
        except Exception as exc:  # noqa: BLE001 - SDK exception is optional at import time
            if not _missing_dataset(exc):
                raise
            self.client.query(
                f"CREATE SCHEMA IF NOT EXISTS `{self.project_id}.{self.dataset}` "
                f"OPTIONS(location='{self.location}');"
            ).result()
            rows = []

        existing: dict[str, set[str]] = {}
        for row in rows:
            table = str(row["table_name"])
            if table in _TABLES:
                existing.setdefault(table, set()).add(str(row["column_name"]).lower())

        statements: list[str] = []
        missing_tables = [name for name in _TABLES if name not in existing]
        for name in missing_tables:
            statements.append(
                f"CREATE TABLE IF NOT EXISTS `{self._tid(name)}` ({_TABLES[name]});")

        additions: dict[str, list[tuple[str, str]]] = {}
        for migration in _MIGRATIONS:
            table, column, column_type = _migration_parts(migration)
            if table not in missing_tables and column.lower() not in existing.get(table, set()):
                additions.setdefault(table, []).append((column, column_type))
        for table, columns in additions.items():
            clauses = ",\n".join(
                f"ADD COLUMN IF NOT EXISTS {column} {column_type}"
                for column, column_type in columns
            )
            statements.append(f"ALTER TABLE `{self._tid(table)}`\n{clauses};")

        if not statements:
            return

        script = "\n".join(statements)
        for attempt in range(_SCHEMA_UPDATE_ATTEMPTS):
            try:
                self.client.query(script).result()
                return
            except Exception as exc:  # noqa: BLE001 - preserve the original SDK error
                if (not _table_update_rate_limited(exc)
                        or attempt == _SCHEMA_UPDATE_ATTEMPTS - 1):
                    raise
                time.sleep(_SCHEMA_UPDATE_BACKOFF_S * (2 ** attempt))

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
            rows = self.client.query(
                f"SELECT l.liked, l.embedding FROM `{self._tid('labels')}` l WHERE NOT EXISTS "
                f"(SELECT 1 FROM `{self._tid('label_retractions')}` r WHERE r.run_id=l.run_id "
                "AND r.app=l.app AND r.source=l.source AND r.profile_id=l.profile_id "
                "AND r.label_created_at=l.created_at)").result()
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
            f"SELECT l.liked, l.embedding FROM `{self._tid('labels')}` l WHERE NOT EXISTS "
            f"(SELECT 1 FROM `{self._tid('label_retractions')}` r WHERE r.run_id=l.run_id "
            "AND r.app=l.app AND r.source=l.source AND r.profile_id=l.profile_id "
            "AND r.label_created_at=l.created_at) ORDER BY created_at"
        ).result()
        return [(bool(r["liked"]), list(r["embedding"])) for r in rows]

    def count_today(self, app: str) -> int:
        """Auto-mode swipes recorded today (LOCAL day — same boundary as
        SQLiteStore.count_today, via local_midnight_epoch(), NOT a UTC reporting day)."""
        start_dt = datetime.fromtimestamp(local_midnight_epoch(), tz=timezone.utc)
        rows = self.client.query(
            f"SELECT COUNT(*) AS c FROM `{self._tid('decisions')}` "
            "WHERE app=@app AND created_at >= @day_start AND source='auto'",
            job_config=_day_start_job_config(start_dt, app=app),
        ).result()
        for r in rows:
            return int(r["c"])
        return 0

    def observe_release_persistence_summary(self, run_id: str, app: str) -> dict[str, int]:
        """Read-only run-scoped complete-cycle counts for the Hinge AUTO release gate.

        This deliberately queries only aggregate counts with bound parameters: a verifier needs
        proof that the production store persisted *both* manual outcomes and a successful Hinge
        opener for the Worker run, never profile IDs, embeddings, opener text, or photos.
        Worker calls the stored pass decision ``dislike``; it maps to the operator-visible
        ``manual_pass_decisions`` field here. ``ensure=False`` construction lets the offline
        verifier call it without DDL/bucket changes.
        """
        from google.cloud import bigquery
        job = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
            bigquery.ScalarQueryParameter("app", "STRING", app),
        ])
        opener_visibility = (
            f"AND NOT EXISTS (SELECT 1 FROM `{self._tid('opener_retractions')}` r WHERE r.run_id=o.run_id "
            "AND r.app=o.app AND r.opener_created_at=o.created_at)"
        )
        query = (
            "SELECT "
            f"(SELECT COUNT(*) FROM `{self._tid('labels')}` l WHERE run_id=@run_id AND app=@app "
            f"AND source='manual' AND liked=FALSE AND NOT EXISTS (SELECT 1 FROM `{self._tid('label_retractions')}` r "
            "WHERE r.run_id=l.run_id AND r.app=l.app AND r.source=l.source AND r.profile_id=l.profile_id "
            "AND r.label_created_at=l.created_at)) AS pass_labels, "
            f"(SELECT COUNT(*) FROM `{self._tid('decisions')}` d WHERE run_id=@run_id AND app=@app "
            f"AND source='manual' AND decision='dislike' AND NOT EXISTS (SELECT 1 FROM `{self._tid('label_retractions')}` r "
            "WHERE r.run_id=d.run_id AND r.app=d.app AND r.source=d.source "
            "AND r.decision_created_at=d.created_at)) AS pass_decisions, "
            f"(SELECT COUNT(*) FROM `{self._tid('labels')}` l WHERE run_id=@run_id AND app=@app "
            f"AND source='manual' AND liked=TRUE AND NOT EXISTS (SELECT 1 FROM `{self._tid('label_retractions')}` r "
            "WHERE r.run_id=l.run_id AND r.app=l.app AND r.source=l.source AND r.profile_id=l.profile_id "
            "AND r.label_created_at=l.created_at)) AS like_labels, "
            f"(SELECT COUNT(*) FROM `{self._tid('decisions')}` d WHERE run_id=@run_id AND app=@app "
            f"AND source='manual' AND decision='like' AND NOT EXISTS (SELECT 1 FROM `{self._tid('label_retractions')}` r "
            "WHERE r.run_id=d.run_id AND r.app=d.app AND r.source=d.source "
            "AND r.decision_created_at=d.created_at)) AS like_decisions, "
            f"(SELECT COUNT(*) FROM `{self._tid('openers')}` o WHERE run_id=@run_id AND app=@app "
            f"{opener_visibility}) AS successful_hinge_openers"
        )
        try:
            rows = self.client.query(query, job_config=job).result()
        except Exception as exc:
            if not _missing_optional_opener_retractions(exc, ensure=self._ensure):
                raise
            rows = self.client.query(query.replace(opener_visibility, ""), job_config=job).result()
        for row in rows:
            return {
                "manual_pass_labels": int(row["pass_labels"]),
                "manual_like_labels": int(row["like_labels"]),
                "manual_pass_decisions": int(row["pass_decisions"]),
                "manual_like_decisions": int(row["like_decisions"]),
                "successful_hinge_openers": int(row["successful_hinge_openers"]),
            }
        return {
            "manual_pass_labels": 0,
            "manual_like_labels": 0,
            "manual_pass_decisions": 0,
            "manual_like_decisions": 0,
            "successful_hinge_openers": 0,
        }

    def ai_observe_release_persistence_summary(self, run_id: str, app: str,
                                               source: str) -> dict[str, int]:
        """Read-only complete-cycle summary for an honest non-manual evidence source."""
        if source not in {"external_ai_review", "automation"}:
            raise ValueError("AI observe release source must be external_ai_review or automation")
        from google.cloud import bigquery
        job = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
            bigquery.ScalarQueryParameter("app", "STRING", app),
            bigquery.ScalarQueryParameter("source", "STRING", source),
        ])
        opener_visibility = (
            f"AND NOT EXISTS (SELECT 1 FROM `{self._tid('opener_retractions')}` r WHERE r.run_id=o.run_id "
            "AND r.app=o.app AND r.opener_created_at=o.created_at)"
        )
        query = (
            "SELECT "
            f"(SELECT COUNT(*) FROM `{self._tid('labels')}` l WHERE run_id=@run_id AND app=@app "
            f"AND source=@source AND liked=FALSE AND NOT EXISTS (SELECT 1 FROM `{self._tid('label_retractions')}` r "
            "WHERE r.run_id=l.run_id AND r.app=l.app AND r.source=l.source AND r.profile_id=l.profile_id "
            "AND r.label_created_at=l.created_at)) AS pass_labels, "
            f"(SELECT COUNT(*) FROM `{self._tid('decisions')}` d WHERE run_id=@run_id AND app=@app "
            f"AND source=@source AND decision='dislike' AND NOT EXISTS (SELECT 1 FROM `{self._tid('label_retractions')}` r "
            "WHERE r.run_id=d.run_id AND r.app=d.app AND r.source=d.source "
            "AND r.decision_created_at=d.created_at)) AS pass_decisions, "
            f"(SELECT COUNT(*) FROM `{self._tid('labels')}` l WHERE run_id=@run_id AND app=@app "
            f"AND source=@source AND liked=TRUE AND NOT EXISTS (SELECT 1 FROM `{self._tid('label_retractions')}` r "
            "WHERE r.run_id=l.run_id AND r.app=l.app AND r.source=l.source AND r.profile_id=l.profile_id "
            "AND r.label_created_at=l.created_at)) AS like_labels, "
            f"(SELECT COUNT(*) FROM `{self._tid('decisions')}` d WHERE run_id=@run_id AND app=@app "
            f"AND source=@source AND decision='like' AND NOT EXISTS (SELECT 1 FROM `{self._tid('label_retractions')}` r "
            "WHERE r.run_id=d.run_id AND r.app=d.app AND r.source=d.source "
            "AND r.decision_created_at=d.created_at)) AS like_decisions, "
            f"(SELECT COUNT(*) FROM `{self._tid('openers')}` o WHERE run_id=@run_id AND app=@app "
            f"{opener_visibility}) AS successful_hinge_openers"
        )
        try:
            rows = self.client.query(query, job_config=job).result()
        except Exception as exc:
            if not _missing_optional_opener_retractions(exc, ensure=self._ensure):
                raise
            rows = self.client.query(query.replace(opener_visibility, ""), job_config=job).result()
        for row in rows:
            return {
                "ai_pass_labels": int(row["pass_labels"]),
                "ai_like_labels": int(row["like_labels"]),
                "ai_pass_decisions": int(row["pass_decisions"]),
                "ai_like_decisions": int(row["like_decisions"]),
                "successful_hinge_openers": int(row["successful_hinge_openers"]),
            }
        return {
            "ai_pass_labels": 0,
            "ai_like_labels": 0,
            "ai_pass_decisions": 0,
            "ai_like_decisions": 0,
            "successful_hinge_openers": 0,
        }

    def retraction_run_rows(self, run_id: str, app: str, source: str) -> dict:
        """Read the exact, minimal committed rows needed to plan a safe correction."""
        from google.cloud import bigquery
        job = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
            bigquery.ScalarQueryParameter("app", "STRING", app),
            bigquery.ScalarQueryParameter("source", "STRING", source),
        ])
        def fetch(table: str, columns: str, order: str) -> list[dict]:
            try:
                rows = self.client.query(
                    f"SELECT {columns} FROM `{self._tid(table)}` "
                    "WHERE run_id=@run_id AND app=@app AND source=@source " + order,
                    job_config=job).result()
            except Exception as exc:  # The first plan can precede creation of this new table.
                if table == "label_retractions" and type(exc).__name__ == "NotFound":
                    return []
                raise
            out = []
            for value in rows:
                row = dict(value)
                for key, item in list(row.items()):
                    if isinstance(item, datetime):
                        row[key] = item.astimezone(timezone.utc).isoformat()
                out.append(row)
            return out
        return {"run_id": run_id, "app": app, "source": source,
                "profiles": fetch("profiles", "profile_id", "ORDER BY profile_id"),
                "labels": fetch("labels", "profile_id, created_at, liked", "ORDER BY created_at, profile_id"),
                "decisions": fetch("decisions", "created_at, decision, score", "ORDER BY created_at, decision, score"),
                "retractions": fetch("label_retractions", "correction_id, profile_id, label_created_at, "
                                      "decision_created_at, decision_fingerprint", "ORDER BY created_at, correction_id")}

    def append_label_retraction(self, row: dict) -> bool:
        """Synchronously append one idempotent tombstone after bound existence checks.

        The first query protects retries by correction id; the second rejects a competing
        correction for the exact label/legacy-decision pair.  BigQuery streaming inserts do
        not expose database uniqueness constraints, so both checks are intentionally bound
        and happen under this store's lock before the deterministic insert id is submitted.
        """
        from .retractions import RetractionRefused
        from google.cloud import bigquery
        correction_id = row["correction_id"]
        # Query parameters of type TIMESTAMP are datetime objects in the BigQuery
        # client.  The portable correction document stores RFC-3339 text so it can be
        # hashed/reviewed; convert only at this transport boundary rather than relying
        # on SDK-version-dependent coercion of strings.
        try:
            label_time = datetime.fromisoformat(str(row["label_created_at"]).replace("Z", "+00:00"))
            decision_time = datetime.fromisoformat(str(row["decision_created_at"]).replace("Z", "+00:00"))
        except ValueError as exc:
            raise RetractionRefused("BigQuery correction timestamps must be RFC-3339") from exc
        job = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("correction_id", "STRING", correction_id)])
        target_job = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("run_id", "STRING", row["run_id"]),
            bigquery.ScalarQueryParameter("app", "STRING", row["app"]),
            bigquery.ScalarQueryParameter("source", "STRING", row["source"]),
            bigquery.ScalarQueryParameter("profile_id", "STRING", row["profile_id"]),
            bigquery.ScalarQueryParameter("label_created_at", "TIMESTAMP", label_time),
            bigquery.ScalarQueryParameter("decision_created_at", "TIMESTAMP", decision_time),
        ])
        with self._lock:
            if correction_id in self._retraction_ids:
                return False
            if any(item.get("correction_id") == correction_id for item in self._buf["label_retractions"]):
                return False
            existing = list(self.client.query(
                f"SELECT correction_id FROM `{self._tid('label_retractions')}` "
                "WHERE correction_id=@correction_id LIMIT 1", job_config=job).result())
            if existing:
                return False
            target = list(self.client.query(
                f"SELECT correction_id FROM `{self._tid('label_retractions')}` "
                "WHERE run_id=@run_id AND app=@app AND source=@source AND profile_id=@profile_id "
                "AND label_created_at=@label_created_at AND decision_created_at=@decision_created_at LIMIT 1",
                job_config=target_job).result())
            if target:
                raise RetractionRefused("target label/decision pair already has a different retraction")
            payload = dict(row)
            for key in ("label_created_at", "decision_created_at", "created_at"):
                payload[key] = datetime.fromisoformat(str(payload[key]).replace("Z", "+00:00")).isoformat()
            errors = self.client.insert_rows_json(self._tid("label_retractions"), [payload],
                                                  row_ids=[correction_id])
            if errors:
                raise RuntimeError(f"BigQuery insert errors for label_retractions: {errors}")
            self._written["label_retractions"] += 1
            self._retraction_ids.add(correction_id)
            self._labels_cache = None
            return True

    def advisory_opener_run_rows(self, run_id: str, app: str) -> dict:
        """Read exact no-text opener identities and zero-decision proof with bound values."""
        from .retractions import RetractionRefused, canonical_sha
        from google.cloud import bigquery
        job = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
            bigquery.ScalarQueryParameter("app", "STRING", app),
        ])
        def rows(query: str, *, optional_opener_retractions: bool = False) -> list[dict]:
            try:
                return [dict(value) for value in self.client.query(query, job_config=job).result()]
            except Exception as exc:
                # Read-only tools may predate this optional append-only table.  Do not turn an
                # absent tombstone table into a failed audit; do re-raise every other failure.
                if optional_opener_retractions and _missing_optional_opener_retractions(
                        exc, ensure=self._ensure):
                    return []
                raise
        opener_rows = rows(f"SELECT created_at,model,opener FROM `{self._tid('openers')}` "
                           "WHERE run_id=@run_id AND app=@app ORDER BY created_at")
        tombstones = rows(f"SELECT correction_id,opener_created_at,model,opener_fingerprint,reason,evidence_ref "
                          f"FROM `{self._tid('opener_retractions')}` "
                          "WHERE run_id=@run_id AND app=@app", optional_opener_retractions=True)
        counts = rows(
            "SELECT "
            f"(SELECT COUNT(*) FROM `{self._tid('profiles')}` WHERE run_id=@run_id AND app=@app) AS profiles, "
            f"(SELECT COUNT(*) FROM `{self._tid('profile_photos')}` WHERE run_id=@run_id AND app=@app) AS profile_photos, "
            f"(SELECT COUNT(*) FROM `{self._tid('labels')}` WHERE run_id=@run_id AND app=@app) AS labels, "
            f"(SELECT COUNT(*) FROM `{self._tid('decisions')}` WHERE run_id=@run_id AND app=@app) AS decisions, "
            f"(SELECT COUNT(*) FROM `{self._tid('labels')}` l WHERE run_id=@run_id AND app=@app AND liked=TRUE "
            f"AND NOT EXISTS (SELECT 1 FROM `{self._tid('label_retractions')}` r WHERE r.run_id=l.run_id "
            "AND r.app=l.app AND r.source=l.source AND r.profile_id=l.profile_id "
            "AND r.label_created_at=l.created_at)) AS like_labels, "
            f"(SELECT COUNT(*) FROM `{self._tid('labels')}` l WHERE run_id=@run_id AND app=@app AND liked=FALSE "
            f"AND NOT EXISTS (SELECT 1 FROM `{self._tid('label_retractions')}` r WHERE r.run_id=l.run_id "
            "AND r.app=l.app AND r.source=l.source AND r.profile_id=l.profile_id "
            "AND r.label_created_at=l.created_at)) AS pass_labels, "
            f"(SELECT COUNT(*) FROM `{self._tid('decisions')}` WHERE run_id=@run_id AND app=@app AND decision='like') AS like_decisions, "
            f"(SELECT COUNT(*) FROM `{self._tid('decisions')}` WHERE run_id=@run_id AND app=@app AND decision='dislike') AS pass_decisions")[0]
        def stamp(value) -> str:
            if not isinstance(value, datetime):
                raise RetractionRefused("BigQuery opener row has invalid created_at")
            return value.astimezone(timezone.utc).isoformat()
        openers = [{"created_at": stamp(row["created_at"]), "model": str(row["model"]),
                    "opener_fingerprint": canonical_sha({"run_id": run_id, "app": app,
                                                           "created_at": stamp(row["created_at"]),
                                                           "model": str(row["model"]), "opener": str(row["opener"])})}
                   for row in opener_rows]
        return {"run_id": run_id, "app": app, "openers": openers,
                "decisions": [{} for _ in range(int(counts["decisions"]))],
                "preference_counts": {key: int(counts[key]) for key in
                                      ("profiles", "profile_photos", "labels", "decisions")},
                "effective_counts": {key: int(counts[key]) for key in
                                     ("like_labels", "like_decisions", "pass_labels", "pass_decisions")},
                "retractions": [{"correction_id": str(row["correction_id"]),
                                  "opener_created_at": stamp(row["opener_created_at"]),
                                  "model": str(row["model"]),
                                  "opener_fingerprint": str(row["opener_fingerprint"]),
                                  "reason": str(row["reason"]),
                                  "evidence_ref": str(row["evidence_ref"])} for row in tombstones]}

    def append_opener_retraction(self, row: dict) -> bool:
        """Append one exact opener tombstone using bound idempotency checks."""
        from .retractions import RetractionRefused, canonical_sha
        from google.cloud import bigquery
        try:
            opener_at = datetime.fromisoformat(str(row["opener_created_at"]).replace("Z", "+00:00"))
        except ValueError as exc:
            raise RetractionRefused("BigQuery opener correction timestamp must be RFC-3339") from exc
        correction_id = row["correction_id"]
        by_key = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("correction_id", "STRING", correction_id),
            bigquery.ScalarQueryParameter("opener_created_at", "TIMESTAMP", opener_at)])
        target = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("run_id", "STRING", row["run_id"]),
            bigquery.ScalarQueryParameter("app", "STRING", row["app"]),
            bigquery.ScalarQueryParameter("opener_created_at", "TIMESTAMP", opener_at),
        ])
        key = (correction_id, row["opener_created_at"])
        with self._lock:
            if key in self._opener_retraction_ids:
                return False
            if any(item.get("correction_id") == correction_id and
                   item.get("opener_created_at") == row["opener_created_at"]
                   for item in self._buf["opener_retractions"]):
                return False
            source = list(self.client.query(
                f"SELECT created_at,model,opener FROM `{self._tid('openers')}` "
                "WHERE run_id=@run_id AND app=@app AND created_at=@opener_created_at", job_config=target).result())
            if len(source) != 1 or str(source[0]["model"]) != row["model"]:
                raise RetractionRefused("target opener row is missing or does not match its cleanup plan")
            source_at = source[0]["created_at"]
            if not isinstance(source_at, datetime):
                raise RetractionRefused("target opener row has an invalid created_at")
            source_stamp = source_at.astimezone(timezone.utc).isoformat()
            fingerprint = canonical_sha({"run_id": row["run_id"], "app": row["app"],
                                         "created_at": source_stamp, "model": str(source[0]["model"]),
                                         "opener": str(source[0]["opener"])})
            if source_stamp != str(row["opener_created_at"]).replace("Z", "+00:00") \
                    or fingerprint != row["opener_fingerprint"]:
                raise RetractionRefused("target opener fingerprint no longer matches its cleanup plan")
            by_same_plan = list(self.client.query(
                f"SELECT correction_id,model,opener_fingerprint,reason,evidence_ref "
                f"FROM `{self._tid('opener_retractions')}` WHERE correction_id=@correction_id "
                "AND opener_created_at=@opener_created_at LIMIT 1", job_config=by_key).result())
            if by_same_plan:
                actual = dict(by_same_plan[0])
                expected = {key: row[key] for key in actual}
                if actual == expected:
                    return False
                raise RetractionRefused("correction id/opener timestamp has different cleanup data")
            existing = list(self.client.query(
                f"SELECT correction_id,model,opener_fingerprint,reason,evidence_ref "
                f"FROM `{self._tid('opener_retractions')}` WHERE run_id=@run_id AND app=@app "
                "AND opener_created_at=@opener_created_at LIMIT 1", job_config=target).result())
            if existing:
                raise RetractionRefused("opener row already has a different cleanup tombstone")
            payload = dict(row)
            payload["opener_created_at"] = opener_at.isoformat()
            payload["created_at"] = datetime.now(timezone.utc).isoformat()
            errors = self.client.insert_rows_json(self._tid("opener_retractions"), [payload],
                                                  row_ids=[canonical_sha(row)])
            if errors:
                raise RuntimeError(f"BigQuery insert errors for opener_retractions: {errors}")
            self._written["opener_retractions"] += 1
            self._opener_retraction_ids.add(key)
            return True

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
                  profile_id="", profile_name="", **_):
        liked = bool(liked)
        embedding_vec = [float(x) for x in embedding]
        with self._lock:
            self._buf["labels"].append({
                "run_id": run_id, "app": app, "profile_id": profile_id, "created_at": _now(), "liked": liked,
                "source": source, "embedding": embedding_vec, "photo_count": int(photo_count),
                "profile_name": str(profile_name or ""),
            })
            label = (liked, list(embedding_vec))
            if self._labels_cache is not None:
                self._labels_cache.append(label)
            self._maybe_flush("labels")

    def clear_training_data(self) -> int:
        """Delete all labels which train the preference model.

        This is intentionally limited to the model dataset: decision and profile archives
        remain an audit of real app actions, while the next run starts the ranker cold.
        """
        with self._lock:
            self.flush()
            rows = self.client.query(f"SELECT COUNT(*) AS c FROM `{self._tid('labels')}`").result()
            count = next((int(row["c"]) for row in rows), 0)
            self.client.query(f"DELETE FROM `{self._tid('labels')}` WHERE TRUE").result()
            self.client.query(f"DELETE FROM `{self._tid('label_retractions')}` WHERE TRUE").result()
            self._labels_cache = []
        return count

    def remove_latest_training_label(self) -> dict | None:
        """Remove the latest visible label and return its operator-readable identity."""
        from google.cloud import bigquery
        with self._lock:
            self.flush()
            rows = self.client.query(
                f"SELECT l.profile_id, l.profile_name, l.created_at FROM `{self._tid('labels')}` l "
                f"WHERE NOT EXISTS (SELECT 1 FROM `{self._tid('label_retractions')}` r "
                "WHERE r.run_id=l.run_id AND r.app=l.app AND r.source=l.source "
                "AND r.profile_id=l.profile_id AND r.label_created_at=l.created_at) "
                "ORDER BY l.created_at DESC LIMIT 1").result()
            row = next(iter(rows), None)
            if row is None:
                return None
            profile_id, created_at = str(row["profile_id"] or ""), row["created_at"]
            job = bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("profile_id", "STRING", profile_id),
                bigquery.ScalarQueryParameter("created_at", "TIMESTAMP", created_at),
            ])
            self.client.query(f"DELETE FROM `{self._tid('labels')}` "
                              "WHERE profile_id=@profile_id AND created_at=@created_at", job_config=job).result()
            self._labels_cache = None
        return {"profile_name": str(row["profile_name"] or ""), "profile_id": profile_id}

    def record_decision(self, run_id, app, decision, score, source="auto", profile_id="",
                        created_at=None):
        with self._lock:
            self._buf["decisions"].append({
                "run_id": run_id, "app": app, "created_at": _timestamp(created_at),
                "decision": decision, "score": float(score), "source": source, "profile_id": profile_id,
            })
            self._maybe_flush("decisions")

    def record_opener(self, run_id, app, model, opener, referenced, angle="",
                      item_description="", *, profile_id="", decision="", decision_source="",
                      decision_created_at=None, model_item_index=None):
        # `angle` and `item_description`: telemetry only (see the openers entry in _TABLES).
        # Both defaulted to "" so a caller that predates either still writes a valid row rather
        # than omitting the field. The live table already holds real rows, so each column got
        # to production via _MIGRATIONS, not via CREATE TABLE IF NOT EXISTS.
        with self._lock:
            self._buf["openers"].append({
                "run_id": run_id, "app": app, "created_at": _now(),
                "model": model, "opener": opener, "referenced": referenced, "angle": angle,
                "item_description": item_description, "profile_id": profile_id,
                "decision": decision, "decision_source": decision_source,
                "decision_created_at": (None if decision_created_at is None
                                        else _timestamp(decision_created_at)),
                "model_item_index": model_item_index,
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
