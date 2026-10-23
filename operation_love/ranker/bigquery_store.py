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
import inspect
import json
import re
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

from ..bigquery_validation import (
    validate_bigquery_identifier,
    validate_bigquery_photo_bucket,
)
from ..costing import Usage
from .store import (
    AUTO_REVIEW_KIND_NEAR_CUT,
    AUTO_REVIEW_KIND_RANDOM,
    AUTO_REVIEW_SOURCE_PREFIX,
    DEFAULT_AUTO_REVIEW_WINDOW,
    REVIEW_ANSWER_IS_LATEST_MESSAGE,
    _require_auto_review_kind,
    _require_optional_app,
    _require_text,
    _require_window,
    auto_review_label_source,
    clean_auto_pass_photos,
    doomed_auto_passes,
    has_usable_photo_row,
    join_label_features,
    local_midnight_epoch,
    missing_photo_feature_entries,
    normalize_timestamp_datetime,
    normalize_timestamp_epoch,
    normalize_timestamp_iso,
    photo_feature_coverage_of,
    require_auto_pass_retention,
    require_photo_feature_version,
    stored_photo_vector,
    validate_auto_pass,
    validate_photo_features,
)

_UPLOAD_ATTEMPTS = 3  # bounded retry so a transient GCS blip doesn't drop a swipe
_UPLOAD_BACKOFF_S = 0.5
# Cloud client defaults are intentionally generous (and, depending on the transport, can be
# effectively unbounded). A Training action has already changed the phone by the time its
# archive starts, so an indefinitely blocked GCS/BQ call must become an explicit failed action
# rather than leaving the Hub's exact-card checkpoint at ``executing`` forever.
_GCS_UPLOAD_REQUEST_TIMEOUT_S = 20.0
_BQ_INSERT_REQUEST_TIMEOUT_S = 20.0
_CLOUD_RETRY_DEADLINE_S = 30.0
# ``insert_rows_json`` already retries transient requests inside one SDK call.  A short
# BigQuery incident can outlive that deadline, however, and the SDK then raises RetryError
# even though a fresh insertAll call succeeds (the reported production failure did exactly
# that during the supervisor's later flush).  Give the unchanged batch one fresh SDK retry
# cycle before failing the already-landed action.  Stable insert IDs make an uncertain first
# call safe to replay under BigQuery's best-effort streaming de-duplication contract.
_BQ_INSERT_ATTEMPTS = 2
_BQ_INSERT_BACKOFF_S = 1.0
# A full Hinge review can contain a dozen PNGs. Bound the complete serial archive as well as
# every request: per-request timeouts alone still allow a long sequence of transient failures
# to make the operator wait without a meaningful upper bound.
#
# DERIVED FROM THE WORK, NOT PICKED (found 2026-09-04). It used to be a flat 180.0, which is
# SMALLER than what the same code's own per-request budget permits for the archive it wraps:
# twelve photos at `_GCS_UPLOAD_REQUEST_TIMEOUT_S` each is 240s. A merely slow-but-working uplink
# -- every photo completing well inside its own 20s timeout, the serial total past 180 -- then
# tripped the aggregate deadline, and the consequence is not a retry. `record_profile` treats a
# short row set as a corrupt archive, DELETES every blob it already uploaded, and returns False;
# worker.py turns that into "the training profile could not be archived; no label was saved" and
# ends the run -- for a Hinge like or pass that has already physically landed on the phone and
# cannot be taken back. So the aggregate bound must never be the thing that fires first on a
# link that is merely slow, or even on one that is failing-then-succeeding within the retry
# ladder `_upload_blob` is allowed to walk; it exists ONLY to stop an unbounded stall. Sizing it
# as requests-times-attempts alone was still short, because that ladder also SLEEPS between
# attempts (0.5s then 1.0s), which is why the per-photo term below adds the backoff sum: the
# deadline can then only bind after every individual request has had its full allowance.
_PROFILE_UPLOAD_DEADLINE_FLOOR_S = 180.0


def _profile_upload_deadline_s(photo_count: int) -> float:
    """The whole-archive budget for `photo_count` photos, never below the historic floor."""
    # `_upload_blob` sleeps _UPLOAD_BACKOFF_S, then double it, between attempts, so the whole
    # ladder's sleeps sum to B*(2**(attempts-1) - 1) on top of the per-request timeouts.
    per_photo = _GCS_UPLOAD_REQUEST_TIMEOUT_S * _UPLOAD_ATTEMPTS + _UPLOAD_BACKOFF_S * (
        2 ** (_UPLOAD_ATTEMPTS - 1) - 1
    )
    return max(_PROFILE_UPLOAD_DEADLINE_FLOOR_S, max(0, int(photo_count)) * per_photo)


# Shared with ranker/__init__.py's make_store(), so the config-parsing fallback (when
# storage.bigquery.flush_every isn't set) can't silently drift from this constructor's
# own default.
DEFAULT_FLUSH_EVERY = 25

_MAX_INSERT_ATTEMPTS = 5  # bounded retry so a row BigQuery keeps rejecting as invalid
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
    #
    # `prompt_sha256` is the digest of the PROMPT ERA this row was generated under (see
    # prompt_stamp in opener/opener.py for what the digest covers and what it deliberately
    # excludes). It exists so an offline calibration pass can GROUP BY era instead of
    # reconstructing the 2026-08-11 / 2026-09-05 boundaries from created_at against
    # config.yaml's git commit dates. Nullable and never backfilled: rows written before
    # 2026-09-05 (b) read back NULL, meaning "predates the stamp". Same _MIGRATIONS caveat as
    # every column above -- this line alone does NOT add it to the live table.
    #
    # `profile_key` (2026-09-06) is the STABLE, cross-time attribution key -- see
    # ranker/profile_key.py's `profile_key_from_identity` for exactly what it hashes and why.
    # Unrelated to `profile_id` above, which stays a per-CARD lineage id regenerated every
    # swipe: `profile_key` is what an `opener_outcomes` row observed days later joins against
    # (see the `opener_outcomes` table below and `BigQueryStore.joined_opener_outcomes`).
    # Nullable and never backfilled, same _MIGRATIONS caveat as every column above.
    "openers": (
        "run_id STRING, app STRING, created_at TIMESTAMP, model STRING, opener STRING, "
        "referenced STRING, angle STRING, item_description STRING, profile_id STRING, "
        "decision STRING, decision_source STRING, decision_created_at TIMESTAMP, "
        "model_item_index INT64, prompt_sha256 STRING, profile_key STRING"
    ),
    # The OUTCOME half of the measurement gap `openers.profile_key` exists to close: what a
    # sent opener actually PERFORMED (a match, a reply, silence), as opposed to everything
    # else this schema records about what the prompt PRODUCED. Deliberately carries no
    # `run_id`/foreign key into `openers` at all -- an outcome is observed on a real
    # conversation, often well after any automation run ended and often by the owner looking
    # at the phone rather than by anything this codebase ran, so (`app`, `profile_key`) is the
    # WHOLE join key back to `openers` (see `BigQueryStore.joined_opener_outcomes`, which owns
    # that join so no caller has to reimplement it). `outcome` and `source` are free text
    # against the documented, unenforced vocabularies in ranker/__init__.py
    # (`KNOWN_OPENER_OUTCOMES` / `KNOWN_OPENER_OUTCOME_SOURCES`); `source` is what lets the
    # SAME table later be populated by an automated reader without any schema change, since it
    # is already a plain STRING rather than a closed type. `observed_at` is WHEN the outcome
    # happened/was noticed (analogous to `openers.decision_created_at`); `created_at` is when
    # this row was WRITTEN, stamped by this store like every other table here. Brand-new
    # table: it needs only this CREATE entry, never a `_MIGRATIONS` line for itself -- that
    # rule is for a COLUMN added to a table that already exists elsewhere (see the comment
    # above `_MIGRATIONS`), and a whole new table reaches an existing project exactly the way
    # `opener_rejections` once did, through `_ensure_tables`' `missing_tables` branch.
    "opener_outcomes": (
        "app STRING, profile_key STRING, outcome STRING, observed_at TIMESTAMP, "
        "source STRING, note STRING, created_at TIMESTAMP"
    ),
    # One queryable row per verified-landed AUTO opener. The PNG itself belongs in the same
    # private Cloud Storage bucket as profile images; BigQuery holds its URI, exact opener,
    # target, hashes, and outcome. Keeping this separate from ``openers`` preserves that table's
    # long-lived analytics contract while making pre-send evidence independently inspectable.
    "opener_send_evidence": (
        "run_id STRING, app STRING, profile_id STRING, created_at TIMESTAMP, "
        "decision_source STRING, decision_created_at TIMESTAMP, outcome STRING, "
        "model_item_index INT64, opener STRING, evidence_id STRING, opener_sha256 STRING, "
        "frame_sha256 STRING, gcs_uri STRING, byte_size INT64, content_type STRING"
    ),
    # Every REJECTED opener attempt -- not just OpenerParseError (the majority, and the only
    # kind that carries a `model`/billed usage), but also (as of 2026-09-06) maybe_opener's
    # OpenerError / HTTP-400 / generic-transient-exception branches, none of which have a model
    # to report (see opener/service.py's OpenerError/bad-request/transient branches and
    # opener.py's REASON_OPENER_ERROR/REASON_BAD_REQUEST/REASON_TRANSIENT_ERROR). Not just the
    # successes `openers` above holds, either -- see opener.py's OpenerParseError docstring for
    # reason_code/raw_opener semantics. `attempt` is the 1-based retry count within
    # maybe_opener()'s per-profile retry loop, so a run of consecutive rejections that exhausted
    # the loop (service.py's max_attempts) is queryable as clearly as one that succeeded on the
    # first retry.
    #
    # `prompt_sha256` is the same prompt-era digest the `openers` entry above describes,
    # carried here so a guard's firing rate is attributable to the prompt that provoked it
    # rather than only to a date range. It was this table's FIRST added column, so it was also
    # this table's first _MIGRATIONS line -- that line, not this one, is what reached the live
    # table.
    "opener_rejections": (
        "run_id STRING, app STRING, created_at TIMESTAMP, model STRING, attempt INT64, "
        "reason_code STRING, reason STRING, raw_opener STRING, prompt_sha256 STRING"
    ),
    "opener_retractions": (
        "correction_id STRING, run_id STRING, app STRING, opener_created_at TIMESTAMP, model STRING, "
        "opener_fingerprint STRING, reason STRING, evidence_ref STRING, created_at TIMESTAMP"
    ),
    # AUTO pass archive + owner review (the recall-target feature). Three brand-new tables, so
    # they need only these CREATE entries and never a _MIGRATIONS line (see the comment above
    # `_MIGRATIONS`): they reach an existing project through `_ensure_tables`' `missing_tables`
    # branch. Deliberately NOT in the retraction/archive snapshot machinery (retraction_run_rows,
    # advisory_opener_run_rows): an archived pass is not a label until the owner answers it, and
    # the answer is written to `labels` (source 'auto_review_random' / 'auto_review_near_cut')
    # through add_label like any other label.
    #
    # `auto_passes` is the manifest (one row per archived pass: the score and threshold in
    # force, the embedding the answer will train on, and whether it was drawn into the
    # unbiased `random_sample` pool). `auto_pass_photos` has EXACTLY profile_photos' columns;
    # the PNGs live in the same private bucket under `auto_passes/{app}/{run_id}/{profile_id}/`.
    # `auto_pass_reviews` is the owner's answer; its presence is what makes a pass "reviewed".
    "auto_passes": (
        "run_id STRING, app STRING, profile_id STRING, created_at TIMESTAMP, score FLOAT64, "
        "threshold FLOAT64, embedding ARRAY<FLOAT64>, photo_count INT64, profile_name STRING, "
        "random_sample BOOL, capture_truncated BOOL"
    ),
    "auto_pass_photos": (
        "run_id STRING, app STRING, profile_id STRING, created_at TIMESTAMP, "
        "photo_index INT64, gcs_uri STRING, sha256 STRING, byte_size INT64, content_type STRING"
    ),
    "auto_pass_reviews": (
        "profile_id STRING, app STRING, reviewed_at TIMESTAMP, would_like BOOL, kind STRING"
    ),
    # Whole-photo features (store.py's "whole-photo features" helpers): one profile's pooled
    # whole-photo vector per (app, profile_id, feature_version), append-only, the latest
    # created_at wins. No zero-card row is written; one an earlier build stored has
    # `card_count = 0` and an EMPTY embedding array (BigQuery ARRAYs cannot be NULL). Brand-new
    # table, so a CREATE entry and no _MIGRATIONS line; it is optional to readers (see
    # `_missing_optional_photo_features`).
    "photo_features": (
        "run_id STRING, app STRING, profile_id STRING, created_at TIMESTAMP, "
        "feature_version STRING, embedding ARRAY<FLOAT64>, card_count INT64, "
        "frames_sha256 STRING, source STRING"
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
    "ALTER TABLE `{openers}` ADD COLUMN IF NOT EXISTS prompt_sha256 STRING;",
    "ALTER TABLE `{opener_rejections}` ADD COLUMN IF NOT EXISTS prompt_sha256 STRING;",
    "ALTER TABLE `{openers}` ADD COLUMN IF NOT EXISTS profile_key STRING;",
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
    return normalize_timestamp_iso(value)


def _copy_label_rows(
    labels: list[tuple[bool, list[float], str]],
) -> list[tuple[bool, list[float], str]]:
    return [(liked, list(embedding), source) for liked, embedding, source in labels]


def _align_identities(
    cache: list[tuple[bool, list[float], str]],
    rows: list[tuple[bool, list[float], str]],
    keys: list[tuple[str, str]],
) -> list[tuple[str, str]] | None:
    """Each ``cache`` row's (app, profile_id), taken from a fresh read (``rows``/``keys``).

    Rows are matched by their full content (liked, embedding, source), in read order among
    equal contents.  Rows with equal content are interchangeable training rows, so which of
    their identities each one gets cannot change the feature rows as a multiset.  None --
    refill the cache from the read -- when a cached row is absent from the read (a retraction
    or a permanently dropped row since the fill).  Rows the read has beyond the cache (written
    by another process since the fill) are ignored, as load_label_rows ignores them.
    """
    pool: dict[tuple, list[tuple[str, str]]] = {}
    for (liked, embedding, source), key in zip(rows, keys, strict=True):
        pool.setdefault((liked, tuple(embedding), source), []).append(key)
    aligned = []
    for liked, embedding, source in cache:
        candidates = pool.get((liked, tuple(embedding), source))
        if not candidates:
            return None
        aligned.append(candidates.pop(0))
    return aligned


_AUTO_REVIEW_TABLES = ("auto_passes", "auto_pass_photos", "auto_pass_reviews")
# Passes removed per DML statement by `prune_auto_passes`: each id is a bound scalar parameter
# (`@p0..@pN`), and BigQuery caps a query at 10,000 parameters, so this is nowhere near a limit
# -- it just keeps one failed statement from rolling back an entire backlog.
_PRUNE_CHUNK = 100


def _job_config(parameters: list[tuple[str, str, object]]):
    """QueryJobConfig binding ``(name, type, value)`` scalar parameters.

    Same duck-typed fallback as `_day_start_job_config`: the client is injectable, so a test
    double must never need the SDK. Every auto-review query binds its values through this --
    only table ids and constant SQL keywords are ever interpolated.
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        return SimpleNamespace(
            query_parameters=[
                SimpleNamespace(name=name, type_=kind, value=value)
                for name, kind, value in parameters
            ]
        )
    return bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(name, kind, value)
            for name, kind, value in parameters
        ]
    )


def _missing_optional_auto_review(exc: Exception, *, ensure: bool) -> bool:
    """Same accommodation as `_missing_optional_opener_outcomes`, for the pass-review tables.

    A read-only (`ensure=False`) store may open against a project that no `ensure=True` store
    has touched since these tables were added; a NotFound naming one of THEM is "nothing has
    been archived yet", not a failure. Anything else still propagates.
    """
    return (
        not ensure
        and type(exc).__name__ == "NotFound"
        and any(name in str(exc) for name in _AUTO_REVIEW_TABLES)
    )


def _missing_optional_photo_features(exc: Exception, *, ensure: bool) -> bool:
    """Same accommodation as `_missing_optional_auto_review`, for `photo_features`.

    A read-only store against a project no `ensure=True` store has touched since the table was
    added reads it as empty, which makes every label "missing" its features -- the loud
    not-ready answer, never a silent one.
    """
    return (
        not ensure and type(exc).__name__ == "NotFound" and "photo_features" in str(exc)
    )


def _row_epoch(value, label: str) -> float:
    """BigQuery TIMESTAMP (datetime) or a buffered ISO/epoch value -> epoch seconds."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    return normalize_timestamp_epoch(value, label=label)


def _epoch(value) -> float:
    """BigQuery TIMESTAMP (datetime) or an already-epoch/ISO value -> epoch seconds."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    return normalize_timestamp_epoch(value, label="auto pass created_at")


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


def _deadline_retry():
    """Return an SDK retry policy with an absolute deadline when google-api-core is present.

    The store's injected test doubles deliberately do not require Google packages. Importing
    lazily keeps that contract while production calls still receive a real retry deadline.
    """
    try:
        from google.api_core.retry import Retry
    except ImportError:
        return None
    return Retry(deadline=_CLOUD_RETRY_DEADLINE_S)


def _retryable_bq_insert_exception(exc: Exception) -> bool:
    """Whether an insert exception (or its SDK wrapper cause) is transient.

    google-api-core wraps the last retryable failure in ``RetryError`` when its deadline
    expires.  ``if_transient_error`` does not classify that wrapper itself, nor explicit
    502/504 and HTTP request timeouts, so include those narrow transport cases and walk
    the wrapper's public ``cause`` as well as normal Python exception chaining.  Imports
    stay lazy to preserve this module's injectable-client/no-Google-SDK test contract.
    """
    try:
        from google.api_core import exceptions as google_exceptions
        from google.api_core.retry import if_transient_error
    except ImportError:
        return False
    try:
        from requests import exceptions as requests_exceptions

        request_timeout_types = (requests_exceptions.Timeout,)
    except ImportError:
        request_timeout_types = ()
    additional_transient_types = (
        google_exceptions.BadGateway,
        google_exceptions.GatewayTimeout,
        *request_timeout_types,
    )

    pending: list[Exception] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if if_transient_error(current) or isinstance(
            current, additional_transient_types
        ):
            return True
        pending.extend(
            candidate
            for candidate in (
                getattr(current, "cause", None),
                getattr(current, "__cause__", None),
            )
            if isinstance(candidate, Exception)
        )
    return False


def _call_with_supported_keywords(method, *args, **kwargs):
    """Call an SDK method with deadline kwargs, omitting ones a small injected fake lacks.

    Google SDK methods accept ``retry`` and ``timeout``. Existing tests and local integrations
    often provide tiny functions with only their positional production payload, though. Inspect
    before calling rather than catching ``TypeError`` afterwards: a TypeError raised *inside* a
    real cloud call is a real failure and must never be mistaken for an old fake signature.
    """
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        # Cython/proxy methods may not expose a signature; the production Google methods support
        # these kwargs, so retain the bounded call in that uncommon case.
        return method(*args, **kwargs)
    parameters = signature.parameters.values()
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters):
        return method(*args, **kwargs)
    allowed = set(signature.parameters)
    return method(
        *args, **{key: value for key, value in kwargs.items() if key in allowed}
    )


def _day_start_job_config(
    start_dt: datetime, *, app: str | None = None, source: str | None = None
):
    """QueryJobConfig binding `start_dt` as the `day_start` TIMESTAMP parameter shared by
    count_today/spend_today (see local_midnight_epoch), plus optional application/source filters.
    Falls back to a minimal duck-typed stand-in when the SDK isn't installed: the ``client`` is
    injectable (see module docstring) so tests run against a fake client and never touch the
    real BigQuery API."""
    parameters = [("day_start", "TIMESTAMP", start_dt)]
    if app is not None:
        parameters.append(("app", "STRING", app))
    if source is not None:
        parameters.append(("source", "STRING", source))
    try:
        from google.cloud import bigquery
    except ImportError:
        return SimpleNamespace(
            query_parameters=[
                SimpleNamespace(name=name, type_=kind, value=value)
                for name, kind, value in parameters
            ]
        )
    return bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(name, kind, value)
            for name, kind, value in parameters
        ]
    )


def _missing_optional_opener_retractions(exc: Exception, *, ensure: bool) -> bool:
    """Only the known optional-table NotFound is empty; every other read error is real."""
    return (
        not ensure
        and type(exc).__name__ == "NotFound"
        and "opener_retractions" in str(exc)
    )


def _missing_optional_opener_send_evidence(exc: Exception, *, ensure: bool) -> bool:
    """Same accommodation as `_missing_optional_opener_retractions`, for the send-evidence archive.

    Only the Training release summary reads ``opener_send_evidence``; a read-only store against a
    project that has never archived a landed opener counts no evidence rather than failing.
    """
    return (
        not ensure
        and type(exc).__name__ == "NotFound"
        and "opener_send_evidence" in str(exc)
    )


def _missing_optional_opener_outcomes(exc: Exception, *, ensure: bool) -> bool:
    """Same accommodation as `_missing_optional_opener_retractions`, for the newer table.

    `opener_outcomes` is added by this change; an `ensure=False` reader (e.g. an offline
    calibration pass) can run against a live project before any `ensure=True` store has ever
    created it. A NotFound naming this specific table is "no outcomes recorded yet," not a
    real failure; anything else still propagates.
    """
    return (
        not ensure
        and type(exc).__name__ == "NotFound"
        and "opener_outcomes" in str(exc)
    )


def _missing_dataset(exc: Exception) -> bool:
    """Recognize only BigQuery's dataset-not-found response during schema discovery."""
    return type(exc).__name__ == "NotFound" and "dataset" in str(exc).lower()


def _table_update_rate_limited(exc: Exception) -> bool:
    """The metadata-update quota is transient and BigQuery explicitly recommends retry."""
    message = str(exc).lower()
    return (
        "exceeded rate limits" in message
        and "too many table update operations" in message
    )


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
    def __init__(
        self,
        project_id: str,
        dataset: str = "operation_love",
        location: str = "US",
        photo_bucket: str = "",
        flush_every: int = DEFAULT_FLUSH_EVERY,
        client=None,
        storage_client=None,
        ensure: bool = True,
    ):
        self.project_id = validate_bigquery_identifier(project_id, "project_id")
        self.dataset = validate_bigquery_identifier(dataset, "dataset")
        self.location = validate_bigquery_identifier(location, "location")
        photo_bucket = validate_bigquery_photo_bucket(photo_bucket)
        if type(flush_every) is not int or flush_every <= 0:
            raise ValueError(
                f"BigQuery flush_every must be a positive integer (got {flush_every!r})"
            )
        if type(ensure) is not bool:
            raise ValueError(f"BigQuery ensure must be true or false (got {ensure!r})")
        self.photo_bucket_name = photo_bucket
        self._ensure = ensure
        self.flush_every = flush_every
        owns_client = client is None
        owns_storage_client = storage_client is None
        # Keep the resources in locals until both constructors have succeeded. In particular,
        # a credentials/transport failure while creating the Storage client must still close a
        # BigQuery client this constructor just created moments earlier.
        client_resource = client
        storage_resource = storage_client
        try:
            if client_resource is None:
                from google.cloud import bigquery  # lazy: only needed for real use

                client_resource = bigquery.Client(project=project_id)
            if storage_resource is None:
                from google.cloud import storage  # lazy: only needed for real use

                storage_resource = storage.Client(project=project_id)
            self.client = client_resource
            self.storage_client = storage_resource
            self._buf: dict[str, list[dict]] = {name: [] for name in _TABLES}
            # Rows confirmed inserted this run.
            self._written: dict[str, int] = {name: 0 for name in _TABLES}
            # Rows permanently given up on (never inserted).
            self._dropped: dict[str, int] = {name: 0 for name in _TABLES}
            # f"{table}:{row_id}" -> consecutive failed-insert attempts.
            self._fail_counts: dict[str, int] = {}
            # (liked, embedding, source) -- the source rides along so load_label_rows can
            # serve the ranker from the SAME cache load_labels projects; a second query per
            # retrain (training retrains after every label) would be far too slow.
            self._labels_cache: list[tuple[bool, list[float], str]] | None = None
            # Each `_labels_cache` row's (app, profile_id), in the same order, for
            # load_label_feature_rows; `_label_keys_for` is the cache LIST they were read
            # with, so any refill or invalidation of the cache (a new list, or None) makes
            # them stale by identity rather than by convention.
            self._label_keys: list[tuple[str, str]] | None = None
            self._label_keys_for: list | None = None
            # feature_version -> {(app, profile_id): winning vector or None (zero cards)}.
            self._photo_features_cache: dict[
                str, dict[tuple[str, str], list[float] | None]
            ] = {}
            # Process-local memory of what this run already archived / answered. The buffered
            # rows and BigQuery's streaming buffer both lag a write by a flush or a few
            # seconds; these sets make "archived once, answered once" hold regardless.
            self._auto_pass_keys: set[tuple[str, str]] = set()
            self._reviewed_auto_passes: set[tuple[str, str]] = set()
            # Close the streaming-read visibility gap for retries.
            self._retraction_ids: set[str] = set()
            self._opener_retraction_ids: set[tuple[str, str]] = set()
            self._lock = threading.RLock()  # shared across worker threads
            self._writes_drained = threading.Condition(self._lock)
            # ``close`` is the supervisor's final persistence boundary.  A worker which missed
            # its bounded join may still return from a device/API call afterwards; without an
            # explicit gate, its next record_* call would append to an in-memory buffer after the
            # final flush and the row would disappear when the process exits.  Image-backed writes
            # have a network phase outside the buffer lock, so close waits for operations that
            # already began before flushing while rejecting newly-started ones.
            self._closing = False
            self._closed = False
            self._active_async_writes = 0
            self._photo_bucket_private_verified = False
            self._photo_bucket = (
                self._get_or_create_photo_bucket()
                if ensure
                else self.storage_client.bucket(photo_bucket)
            )
            self._photo_bucket_private_verified = ensure
            if ensure:
                self._ensure_tables()
        except BaseException:
            # SDK clients created here own HTTP resources. Constructor failure produces no
            # usable store, so close only those owned clients and retain the setup exception.
            for owned, resource in (
                (owns_storage_client, storage_resource),
                (owns_client, client_resource),
            ):
                close = getattr(resource, "close", None)
                if owned and callable(close):
                    try:
                        close()
                    except Exception:  # noqa: BLE001
                        pass
            raise

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
            "WHERE table_name IN (" + ", ".join(f"'{name}'" for name in _TABLES) + ")"
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
        statements.extend(
            f"CREATE TABLE IF NOT EXISTS `{self._tid(name)}` ({_TABLES[name]});"
            for name in missing_tables
        )

        additions: dict[str, list[tuple[str, str]]] = {}
        for migration in _MIGRATIONS:
            table, column, column_type = _migration_parts(migration)
            if table not in missing_tables and column.lower() not in existing.get(
                table, set()
            ):
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
                if (
                    not _table_update_rate_limited(exc)
                    or attempt == _SCHEMA_UPDATE_ATTEMPTS - 1
                ):
                    raise
                time.sleep(_SCHEMA_UPDATE_BACKOFF_S * (2**attempt))

    def _get_or_create_photo_bucket(self):
        bucket = self.storage_client.bucket(self.photo_bucket_name)
        if bucket.exists():
            self._lockdown_bucket(bucket, created=False)
            return bucket
        bucket = self.storage_client.create_bucket(bucket, location=self.location)
        self._lockdown_bucket(bucket, created=True)
        return bucket

    def _lockdown_bucket(self, bucket, *, created: bool) -> None:
        """Make the photo bucket private: uniform bucket-level access + enforced public-
        access-prevention so it can never be exposed publicly. Applied to EXISTING buckets
        too (idempotently), not just freshly created ones — a bucket created before this
        hardening would otherwise keep weaker defaults (e.g. fine-grained ACLs). Any read,
        patch, or verification failure aborts startup before profile photos can be uploaded."""
        try:
            if not created:
                bucket.reload()  # fetch authoritative IAM state before inspecting it
            iam = bucket.iam_configuration
            if (
                getattr(iam, "uniform_bucket_level_access_enabled", False)
                and getattr(iam, "public_access_prevention", "") == "enforced"
            ):
                return  # already locked down -> no needless patch
            iam.uniform_bucket_level_access_enabled = True
            iam.public_access_prevention = "enforced"
            bucket.patch()
            bucket.reload()
            iam = bucket.iam_configuration
            if (
                not getattr(iam, "uniform_bucket_level_access_enabled", False)
                or getattr(iam, "public_access_prevention", "") != "enforced"
            ):
                raise RuntimeError(
                    "bucket IAM did not retain the required private settings"
                )
            if not created:
                print(
                    f"BigQuery store: hardened existing photo bucket {self.photo_bucket_name} "
                    "(uniform bucket-level access + public-access-prevention enforced)."
                )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Refusing BigQuery photo storage: could not verify private bucket "
                f"{self.photo_bucket_name} (uniform access and public-access prevention)"
            ) from exc

    # --- reads ----------------------------------------------------------
    def _label_not_retracted(self) -> str:
        """The ONE predicate that answers "is this label retracted?" for this backend.

        Every read path that must hide a tombstoned label (load_labels, both release
        summaries, the opener cleanup advisory, remove_latest_training_label) substitutes this
        exact text against a ``labels`` row aliased ``l``, so a release summary can never count
        a label load_labels has already dropped from the training set.

        A tombstone is keyed to a label by (run_id, app, source, profile_id, label_created_at)
        against labels' (run_id, app, source, profile_id, created_at); correction_id identifies
        the CORRECTION rather than its target, so it is deliberately not part of the join.

        profile_id is IFNULL'd on BOTH sides because a label can legitimately carry NO profile
        identity, spelled two different ways: `labels.profile_id` reached the live table through
        _MIGRATIONS, so every row written before that ALTER reads back NULL, while add_label's
        own default writes "". BigQuery's `NULL = <anything>` is NULL and never TRUE, so a
        NULL-profile label could not be matched by ANY tombstone -- permanently in the training
        set, removable (remove_latest_training_label) but never retractable. Folding NULL and ""
        into one comparable value fixes that in the PREDICATE, never by rewriting stored rows.

        Precision: the join still carries run_id/app/source AND label_created_at, so a tombstone
        with no profile identity hides only the label at its own timestamp rather than every
        legacy label in the run. Residual, stated rather than papered over: nothing in this
        schema makes created_at unique, so two identity-less labels in one run/app/source
        sharing an exact created_at cannot be told apart here and a single tombstone would hide
        both. No other column could separate them, and the correction planner independently
        refuses any run whose labels lack strict unique created_at ordering
        (retractions._strict), so the tools path cannot reach that case.

        The decisions half of a correction (see the release summaries below) keys on
        (run_id, app, source, decision_created_at) and never on profile_id, so it carries no
        NULL exposure and is intentionally left as plain equality.
        """
        return (
            f"NOT EXISTS (SELECT 1 FROM `{self._tid('label_retractions')}` r "
            "WHERE r.run_id=l.run_id AND r.app=l.app AND r.source=l.source "
            "AND IFNULL(r.profile_id,'')=IFNULL(l.profile_id,'') "
            "AND r.label_created_at=l.created_at)"
        )

    def load_labels(self) -> list[tuple[bool, list[float]]]:
        # A projection of load_label_rows (and so of the same cache), never a second query:
        # the two must return identical rows in identical order.
        return [(liked, emb) for liked, emb, _source in self.load_label_rows()]

    def load_label_rows(self) -> list[tuple[bool, list[float], str]]:
        """Exactly the rows load_labels() returns plus each label's `source` ("" for NULL).

        The source lets ranker/threshold.py tell MANUAL calibration labels from auto-review
        answers. Cache-backed like load_labels, and add_label appends the new triple to the
        cache, so a retrain after every label stays coherent without re-querying BigQuery.
        """
        # Hold the lock across the whole fill so the cache-miss check, the SELECT,
        # and folding in buffered rows are atomic (no concurrent add_label can be
        # dropped). Called once at startup before workers spawn, so the in-lock
        # query doesn't contend in practice; RLock keeps it reentrancy-safe.
        with self._lock:
            if self._labels_cache is not None:
                return _copy_label_rows(self._labels_cache)
            rows = self.client.query(
                f"SELECT l.liked, l.embedding, l.source FROM `{self._tid('labels')}` l "
                f"WHERE {self._label_not_retracted()}"
            ).result()
            # .get, not []: `source` was ALTERed onto old tables and reads back NULL there.
            out = [
                (bool(r["liked"]), list(r["embedding"]), str(r.get("source") or ""))
                for r in rows
            ]
            out.extend(
                (
                    bool(r["liked"]),
                    list(r["embedding"]),
                    str(r.get("source") or ""),
                )
                for r in self._buf["labels"]
            )
            self._labels_cache = _copy_label_rows(out)
            return _copy_label_rows(self._labels_cache)

    # --- whole-photo features -------------------------------------------------------------
    def _label_identity_rows_locked(
        self,
    ) -> tuple[list[tuple[bool, list[float], str]], list[tuple[str, str]]]:
        """``_labels_cache`` and each row's (app, profile_id), in the cache's own order.

        Caller holds the lock.  The rows ARE the label cache, so they are exactly what
        load_label_rows serves, in its order.  When the identities are unknown -- the cache was
        filled by load_label_rows, or replaced or invalidated since -- one SELECT reads them:
        load_label_rows' own query plus two columns, folded with the buffered labels the same
        way.  A cache that already exists keeps its rows and order and only gains identities
        (BigQuery promises no row order, so adopting the new read's order would make a v1
        read and a whole-photo read of the same labels disagree); see ``_align_identities``
        for when that is impossible and the cache is refilled from the read instead, exactly
        as an invalidated cache is.  Only the whole-photo feature set and the tools reach
        this; the face_crop_v1 path never does.
        """
        cache = self._labels_cache
        if (
            cache is not None
            and self._label_keys is not None
            and self._label_keys_for is cache
            and len(self._label_keys) == len(cache)
        ):
            return _copy_label_rows(cache), list(self._label_keys)
        rows = self.client.query(
            "SELECT l.liked, l.embedding, l.source, l.app, l.profile_id "
            f"FROM `{self._tid('labels')}` l WHERE {self._label_not_retracted()}"
        ).result()
        out: list[tuple[bool, list[float], str]] = []
        keys: list[tuple[str, str]] = []
        for r in [*rows, *self._buf["labels"]]:
            out.append(
                (bool(r["liked"]), list(r["embedding"]), str(r.get("source") or ""))
            )
            keys.append((str(r.get("app") or ""), str(r.get("profile_id") or "")))
        aligned = None if cache is None else _align_identities(cache, out, keys)
        if aligned is None:
            self._labels_cache = _copy_label_rows(out)
        else:
            keys = aligned
        self._label_keys = keys
        self._label_keys_for = self._labels_cache
        return _copy_label_rows(self._labels_cache), list(keys)

    def _photo_features_locked(
        self, version: str
    ) -> dict[tuple[str, str], list[float] | None]:
        """(app, profile_id) -> winning vector of ``version`` (None = zero cards); cached.

        One query per version per process; add_photo_features keeps the map coherent, the same
        way add_label keeps the label cache.  Rows still in the buffer are folded in.  Caller
        holds the lock.
        """
        cached = self._photo_features_cache.get(version)
        if cached is not None:
            return cached
        try:
            rows = list(
                self.client.query(
                    "SELECT app, profile_id, created_at, embedding, card_count "
                    f"FROM `{self._tid('photo_features')}` "
                    "WHERE feature_version = @feature_version",
                    job_config=_job_config([("feature_version", "STRING", version)]),
                ).result()
            )
        except Exception as exc:  # noqa: BLE001 - classified by name; SDK optional at import
            if not _missing_optional_photo_features(exc, ensure=self._ensure):
                raise
            rows = []
        rows.extend(
            row
            for row in self._buf["photo_features"]
            if row["feature_version"] == version
        )
        ordered = sorted(
            enumerate(rows),
            key=lambda item: (
                _row_epoch(item[1]["created_at"], "photo_features created_at"),
                item[0],
            ),
        )
        features: dict[tuple[str, str], list[float] | None] = {}
        for _order, row in ordered:
            app, profile_id = str(row["app"]), str(row["profile_id"])
            features[(app, profile_id)] = stored_photo_vector(
                row.get("embedding"), row.get("card_count"), f"{app}/{profile_id}"
            )
        self._photo_features_cache[version] = features
        return features

    def add_photo_features(
        self,
        *,
        run_id,
        app,
        profile_id,
        feature_version,
        embedding,
        card_count,
        frames_sha256,
        source,
        best_effort=False,
    ) -> bool:
        """Buffer (or, ``best_effort``, insert) one profile's whole-photo feature row.

        Validated by ``store.validate_photo_features`` (ValueError otherwise, in either mode).
        By default the row is buffered and flushed in batches like a label (True once
        buffered): the required write of face_crop_whole_photo_v1 mode, inside the Training
        persist's flush boundary.  ``best_effort`` is for face_crop_v1 mode, where nothing reads
        the row: it is inserted at once, NEVER buffered, so a rejected row cannot fail a later
        ``flush()`` or ``close()`` of rows that matter; a rejection or an insert error is a loud
        WARNING and False.  Either way a written row is applied to the in-memory feature map, so
        a retrain right after a Training label sees it.
        """
        row = validate_photo_features(
            run_id=run_id,
            app=app,
            profile_id=profile_id,
            feature_version=feature_version,
            embedding=embedding,
            card_count=card_count,
            frames_sha256=frames_sha256,
            source=source,
        )
        vector = list(row["embedding"])
        payload = {
            "run_id": row["run_id"],
            "app": row["app"],
            "profile_id": row["profile_id"],
            "created_at": _now(),
            "feature_version": row["feature_version"],
            "embedding": vector,
            "card_count": row["card_count"],
            "frames_sha256": row["frames_sha256"],
            "source": row["source"],
        }
        with self._lock:
            # Same footing as add_label: the follow-on row of a Training archive close may
            # already be waiting for.
            self._require_open_for_write(allow_closing=True)
            if best_effort:
                try:
                    errors = self._insert_rows_json(
                        "photo_features", [payload], row_ids=[_row_id(payload)]
                    )
                except Exception as exc:  # noqa: BLE001 - best effort: reported, never raised
                    errors = f"{type(exc).__name__}: {exc}"
                if errors:
                    print(
                        "WARNING: BigQuery store: the best-effort photo_features row for "
                        f"{row['app']}/{row['profile_id']} was NOT saved ({errors})"
                    )
                    return False
                self._written["photo_features"] += 1
            else:
                self._buf["photo_features"].append(payload)
            features = self._photo_features_cache.get(row["feature_version"])
            if features is not None:
                features[(row["app"], row["profile_id"])] = list(vector)
            if not best_effort:
                self._maybe_flush("photo_features")
        return True

    def load_label_feature_rows(
        self, feature_version: str
    ) -> list[tuple[bool, list[float], str, list[float] | None, bool]]:
        """Exactly load_label_rows()' rows, in its order, plus (photo_vector, has_photo_row).

        Served from the label cache (see ``_label_identity_rows_locked``) and the cached
        feature map, matched by (app, profile_id).  ``has_photo_row`` False means no row
        exists; True with a None vector is a zero-card row an earlier build stored.
        """
        version = require_photo_feature_version(feature_version)
        with self._lock:
            rows, keys = self._label_identity_rows_locked()
            features = self._photo_features_locked(version)
            return join_label_features(rows, keys, features)

    def photo_feature_coverage(self, feature_version: str) -> dict:
        """``{"labels","with_features","missing","zero_cards"}`` (see photo_feature_coverage_of)."""
        return photo_feature_coverage_of(self.load_label_feature_rows(feature_version))

    def labels_missing_photo_features(self, feature_version: str) -> list[dict]:
        """The backfill's work list (see ``store.missing_photo_feature_entries``).

        Frames are the archived ``profile_photos`` set of a manual label and the
        ``auto_pass_photos`` set of an ``auto_review*`` label, by (run_id, app, profile_id), in
        photo_index order.  An AUTO pass pruned by retention has no frames left, so its entry
        comes back with empty lists.  Read-only: queries, never a flush or a write.
        """
        version = require_photo_feature_version(feature_version)
        with self._lock:
            labels = [
                dict(row)
                for row in self.client.query(
                    "SELECT l.run_id, l.app, l.profile_id, l.source "
                    f"FROM `{self._tid('labels')}` l "
                    f"WHERE {self._label_not_retracted()} ORDER BY l.created_at"
                ).result()
            ]
            labels.extend(dict(row) for row in self._buf["labels"])
            features = self._photo_features_locked(version)
            # Read an archive only when a label that lacks a usable row needs frames from it.
            pending_review = {
                str(label.get("source") or "").startswith(AUTO_REVIEW_SOURCE_PREFIX)
                for label in labels
                if label.get("profile_id")
                and not has_usable_photo_row(
                    features, (str(label.get("app") or ""), str(label["profile_id"]))
                )
            }
            manual_frames = (
                self._archived_frames_locked("profile_photos", optional=False)
                if False in pending_review
                else {}
            )
            review_frames = (
                self._archived_frames_locked("auto_pass_photos", optional=True)
                if True in pending_review
                else {}
            )
        return missing_photo_feature_entries(
            labels, features, manual_frames, review_frames
        )

    def _archived_frames_locked(
        self, table: str, *, optional: bool
    ) -> dict[tuple[str, str, str], list[tuple[float, int, str, str]]]:
        """(run_id, app, profile_id) -> [(created_at epoch, photo_index, gcs_uri, sha256)]."""
        try:
            rows = list(
                self.client.query(
                    "SELECT run_id, app, profile_id, created_at, photo_index, gcs_uri, sha256 "
                    f"FROM `{self._tid(table)}`"
                ).result()
            )
        except Exception as exc:  # noqa: BLE001 - classified by name; SDK optional at import
            if not optional or not _missing_optional_auto_review(
                exc, ensure=self._ensure
            ):
                raise
            rows = []
        frames: dict[tuple[str, str, str], list[tuple[float, int, str, str]]] = {}
        for row in [*rows, *self._buf[table]]:
            frames.setdefault(
                (
                    str(row["run_id"] or ""),
                    str(row["app"] or ""),
                    str(row["profile_id"] or ""),
                ),
                [],
            ).append(
                (
                    _row_epoch(row["created_at"], f"{table} created_at"),
                    row["photo_index"],
                    str(row["gcs_uri"] or ""),
                    str(row["sha256"] or ""),
                )
            )
        return frames

    def count_today(self, app: str, *, source: str = "auto") -> int:
        """One validated decision source's local-day count."""
        if source not in {"auto", "manual"}:
            raise ValueError("count_today source must be 'auto' or 'manual'")
        start_dt = datetime.fromtimestamp(local_midnight_epoch(), tz=timezone.utc)
        rows = self.client.query(
            f"SELECT COUNT(*) AS c FROM `{self._tid('decisions')}` "
            "WHERE app=@app AND created_at >= @day_start AND source=@source",
            job_config=_day_start_job_config(start_dt, app=app, source=source),
        ).result()
        for r in rows:
            return int(r["c"])
        return 0

    def observe_release_persistence_summary(
        self, run_id: str, app: str
    ) -> dict[str, int]:
        """Read-only run-scoped complete-cycle counts for the Hinge AUTO release gate.

        This deliberately queries only aggregate counts with bound parameters: a verifier needs
        proof that the production store persisted *both* manual outcomes and a successful Hinge
        opener for the Worker run, never profile IDs, embeddings, opener text, or photos.
        Worker calls the stored pass decision ``dislike``; it maps to the operator-visible
        ``manual_pass_decisions`` field here. ``ensure=False`` construction lets the offline
        verifier call it without DDL/bucket changes.
        """
        from google.cloud import bigquery

        job = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
                bigquery.ScalarQueryParameter("app", "STRING", app),
            ]
        )
        opener_visibility = (
            f"AND NOT EXISTS (SELECT 1 FROM `{self._tid('opener_retractions')}` r WHERE r.run_id=o.run_id "
            "AND r.app=o.app AND r.opener_created_at=o.created_at)"
        )
        query = (
            "SELECT "
            f"(SELECT COUNT(*) FROM `{self._tid('labels')}` l WHERE run_id=@run_id AND app=@app "
            f"AND source='manual' AND liked=FALSE AND {self._label_not_retracted()}) AS pass_labels, "
            f"(SELECT COUNT(*) FROM `{self._tid('decisions')}` d WHERE run_id=@run_id AND app=@app "
            f"AND source='manual' AND decision='dislike' AND NOT EXISTS (SELECT 1 FROM `{self._tid('label_retractions')}` r "
            "WHERE r.run_id=d.run_id AND r.app=d.app AND r.source=d.source "
            "AND r.decision_created_at=d.created_at)) AS pass_decisions, "
            f"(SELECT COUNT(*) FROM `{self._tid('labels')}` l WHERE run_id=@run_id AND app=@app "
            f"AND source='manual' AND liked=TRUE AND {self._label_not_retracted()}) AS like_labels, "
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
            rows = self.client.query(
                query.replace(opener_visibility, ""), job_config=job
            ).result()
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

    def ai_observe_release_persistence_summary(
        self, run_id: str, app: str, source: str
    ) -> dict[str, int]:
        """Read-only complete-cycle summary for an honest non-manual evidence source."""
        if source not in {"external_ai_review", "automation"}:
            raise ValueError(
                "AI observe release source must be external_ai_review or automation"
            )
        from google.cloud import bigquery

        job = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
                bigquery.ScalarQueryParameter("app", "STRING", app),
                bigquery.ScalarQueryParameter("source", "STRING", source),
            ]
        )
        opener_visibility = (
            f"AND NOT EXISTS (SELECT 1 FROM `{self._tid('opener_retractions')}` r WHERE r.run_id=o.run_id "
            "AND r.app=o.app AND r.opener_created_at=o.created_at)"
        )
        query = (
            "SELECT "
            f"(SELECT COUNT(*) FROM `{self._tid('labels')}` l WHERE run_id=@run_id AND app=@app "
            f"AND source=@source AND liked=FALSE AND {self._label_not_retracted()}) AS pass_labels, "
            f"(SELECT COUNT(*) FROM `{self._tid('decisions')}` d WHERE run_id=@run_id AND app=@app "
            f"AND source=@source AND decision='dislike' AND NOT EXISTS (SELECT 1 FROM `{self._tid('label_retractions')}` r "
            "WHERE r.run_id=d.run_id AND r.app=d.app AND r.source=d.source "
            "AND r.decision_created_at=d.created_at)) AS pass_decisions, "
            f"(SELECT COUNT(*) FROM `{self._tid('labels')}` l WHERE run_id=@run_id AND app=@app "
            f"AND source=@source AND liked=TRUE AND {self._label_not_retracted()}) AS like_labels, "
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
            rows = self.client.query(
                query.replace(opener_visibility, ""), job_config=job
            ).result()
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

    def training_release_persistence_summary(
        self,
        run_id: str,
        app: str,
        *,
        like_profile_id: str,
        pass_profile_id: str,
        like_send_evidence_id: str,
    ) -> dict[str, int]:
        """Read-only per-profile counts for the Hinge supervised-Training release verifier.

        One aggregate statement with bound parameters; counts only. Each count is filtered to
        the profile id the verified debug chain persisted, and the send evidence also to the
        landed resumed-send evidence id, so a Disliked or never-sent draft (Training writes both
        to ``openers``) or another profile's rows cannot stand in for the Like or Pass the
        release names. The label, decision and opener visibility clauses are the observe
        summaries'. A read-only store may meet a project that has no ``opener_retractions`` or
        ``opener_send_evidence`` table yet: the first then hides nothing and the second counts 0.
        """
        from google.cloud import bigquery

        job = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
                bigquery.ScalarQueryParameter("app", "STRING", app),
                bigquery.ScalarQueryParameter(
                    "like_profile_id", "STRING", like_profile_id
                ),
                bigquery.ScalarQueryParameter(
                    "pass_profile_id", "STRING", pass_profile_id
                ),
                bigquery.ScalarQueryParameter(
                    "like_send_evidence_id", "STRING", like_send_evidence_id
                ),
            ]
        )
        opener_visibility = (
            f"AND NOT EXISTS (SELECT 1 FROM `{self._tid('opener_retractions')}` r WHERE r.run_id=o.run_id "
            "AND r.app=o.app AND r.opener_created_at=o.created_at)"
        )
        send_evidence = (
            f"(SELECT COUNT(*) FROM `{self._tid('opener_send_evidence')}` e WHERE run_id=@run_id "
            "AND app=@app AND decision_source='manual' AND outcome='like_landed' "
            "AND profile_id=@like_profile_id AND evidence_id=@like_send_evidence_id)"
        )

        def decisions(decision: str, profile_param: str) -> str:
            return (
                f"(SELECT COUNT(*) FROM `{self._tid('decisions')}` d WHERE run_id=@run_id AND app=@app "
                f"AND source='manual' AND decision='{decision}' AND profile_id=@{profile_param} "
                f"AND NOT EXISTS (SELECT 1 FROM `{self._tid('label_retractions')}` r "
                "WHERE r.run_id=d.run_id AND r.app=d.app AND r.source=d.source "
                "AND r.decision_created_at=d.created_at))"
            )

        query = (
            "SELECT "
            f"(SELECT COUNT(*) FROM `{self._tid('labels')}` l WHERE run_id=@run_id AND app=@app "
            "AND source='manual' AND liked=TRUE AND profile_id=@like_profile_id "
            f"AND {self._label_not_retracted()}) AS like_labels, "
            f"{decisions('like', 'like_profile_id')} AS like_decisions, "
            f"(SELECT COUNT(*) FROM `{self._tid('openers')}` o WHERE run_id=@run_id AND app=@app "
            "AND decision='like' AND decision_source='manual' AND profile_id=@like_profile_id "
            f"{opener_visibility}) AS like_openers, "
            f"{send_evidence} AS like_send_evidence, "
            f"(SELECT COUNT(*) FROM `{self._tid('labels')}` l WHERE run_id=@run_id AND app=@app "
            "AND source='manual' AND liked=FALSE AND profile_id=@pass_profile_id "
            f"AND {self._label_not_retracted()}) AS pass_labels, "
            f"{decisions('dislike', 'pass_profile_id')} AS pass_decisions"
        )
        # Each optional table can be missing independently. Every retry removes one table's
        # clause from the statement, so this ends after at most two retries.
        while True:
            try:
                rows = self.client.query(query, job_config=job).result()
                break
            except Exception as exc:
                if opener_visibility in query and _missing_optional_opener_retractions(
                    exc, ensure=self._ensure
                ):
                    query = query.replace(opener_visibility, "")
                elif send_evidence in query and _missing_optional_opener_send_evidence(
                    exc, ensure=self._ensure
                ):
                    query = query.replace(send_evidence, "0")
                else:
                    raise
        keys = (
            "like_labels",
            "like_decisions",
            "like_openers",
            "like_send_evidence",
            "pass_labels",
            "pass_decisions",
        )
        for row in rows:
            return {key: int(row[key]) for key in keys}
        return dict.fromkeys(keys, 0)

    def retraction_run_rows(self, run_id: str, app: str, source: str) -> dict:
        """Read the exact, minimal committed rows needed to plan a safe correction."""
        from google.cloud import bigquery

        job = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
                bigquery.ScalarQueryParameter("app", "STRING", app),
                bigquery.ScalarQueryParameter("source", "STRING", source),
            ]
        )

        def fetch(table: str, columns: str, order: str) -> list[dict]:
            try:
                rows = self.client.query(
                    f"SELECT {columns} FROM `{self._tid(table)}` "
                    "WHERE run_id=@run_id AND app=@app AND source=@source " + order,
                    job_config=job,
                ).result()
            except (
                Exception
            ) as exc:  # The first plan can precede creation of this new table.
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

        decisions = fetch(
            "decisions",
            "profile_id, created_at, decision, score",
            "ORDER BY created_at, decision, score",
        )
        for decision in decisions:
            if decision.get("profile_id") in {None, ""}:
                decision.pop("profile_id", None)
        return {
            "run_id": run_id,
            "app": app,
            "source": source,
            "profiles": fetch("profiles", "profile_id", "ORDER BY profile_id"),
            "labels": fetch(
                "labels",
                "profile_id, created_at, liked",
                "ORDER BY created_at, profile_id",
            ),
            "decisions": decisions,
            "retractions": fetch(
                "label_retractions",
                "correction_id, profile_id, label_created_at, "
                "decision_created_at, decision_fingerprint",
                "ORDER BY created_at, correction_id",
            ),
        }

    def append_label_retraction(self, row: dict) -> bool:
        """Synchronously append one idempotent tombstone after bound existence checks.

        The first query protects retries by correction id; the second rejects a competing
        correction for the exact label/legacy-decision pair.  BigQuery streaming inserts do
        not expose database uniqueness constraints, so both checks are intentionally bound
        and happen under this store's lock before the deterministic insert id is submitted.

        Tombstones are the one write this class never buffers: ``_insert_rows_json`` below
        publishes them immediately, so ``self._retraction_ids`` plus these bound queries are
        the whole of the local idempotency story -- there is no pending ``_buf`` entry to scan.
        """
        from .retractions import RetractionRefused
        from google.cloud import bigquery

        correction_id = row["correction_id"]
        # Query parameters of type TIMESTAMP are datetime objects in the BigQuery
        # client.  The portable correction document stores RFC-3339 text so it can be
        # hashed/reviewed; convert only at this transport boundary rather than relying
        # on SDK-version-dependent coercion of strings.
        try:
            label_time = normalize_timestamp_datetime(
                row["label_created_at"], label="label_created_at"
            )
            decision_time = normalize_timestamp_datetime(
                row["decision_created_at"], label="decision_created_at"
            )
        except (KeyError, ValueError) as exc:
            raise RetractionRefused(
                "BigQuery correction timestamps must be RFC-3339"
            ) from exc
        job = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("correction_id", "STRING", correction_id)
            ]
        )
        target_job = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id", "STRING", row["run_id"]),
                bigquery.ScalarQueryParameter("app", "STRING", row["app"]),
                bigquery.ScalarQueryParameter("source", "STRING", row["source"]),
                bigquery.ScalarQueryParameter(
                    "profile_id", "STRING", row["profile_id"]
                ),
                bigquery.ScalarQueryParameter(
                    "label_created_at", "TIMESTAMP", label_time
                ),
                bigquery.ScalarQueryParameter(
                    "decision_created_at", "TIMESTAMP", decision_time
                ),
            ]
        )
        with self._lock:
            self._require_open_for_write()
            if correction_id in self._retraction_ids:
                return False
            existing = list(
                self.client.query(
                    f"SELECT correction_id FROM `{self._tid('label_retractions')}` "
                    "WHERE correction_id=@correction_id LIMIT 1",
                    job_config=job,
                ).result()
            )
            if existing:
                return False
            target = list(
                self.client.query(
                    f"SELECT correction_id FROM `{self._tid('label_retractions')}` "
                    "WHERE run_id=@run_id AND app=@app AND source=@source AND profile_id=@profile_id "
                    "AND label_created_at=@label_created_at AND decision_created_at=@decision_created_at LIMIT 1",
                    job_config=target_job,
                ).result()
            )
            if target:
                raise RetractionRefused(
                    "target label/decision pair already has a different retraction"
                )
            payload = dict(row)
            try:
                for key in ("label_created_at", "decision_created_at", "created_at"):
                    payload[key] = normalize_timestamp_iso(payload[key], label=key)
            except (KeyError, ValueError) as exc:
                raise RetractionRefused(
                    "BigQuery correction timestamps must be RFC-3339"
                ) from exc
            errors = self._insert_rows_json(
                "label_retractions", [payload], row_ids=[correction_id]
            )
            if errors:
                raise RuntimeError(
                    f"BigQuery insert errors for label_retractions: {errors}"
                )
            self._written["label_retractions"] += 1
            self._retraction_ids.add(correction_id)
            self._labels_cache = None
            return True

    def advisory_opener_run_rows(self, run_id: str, app: str) -> dict:
        """Read exact no-text opener identities and zero-decision proof with bound values."""
        from .retractions import RetractionRefused, canonical_sha
        from google.cloud import bigquery

        job = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
                bigquery.ScalarQueryParameter("app", "STRING", app),
            ]
        )

        def rows(
            query: str, *, optional_opener_retractions: bool = False
        ) -> list[dict]:
            try:
                return [
                    dict(value)
                    for value in self.client.query(query, job_config=job).result()
                ]
            except Exception as exc:
                # Read-only tools may predate this optional append-only table.  Do not turn an
                # absent tombstone table into a failed audit; do re-raise every other failure.
                if optional_opener_retractions and _missing_optional_opener_retractions(
                    exc, ensure=self._ensure
                ):
                    return []
                raise

        opener_rows = rows(
            f"SELECT created_at,model,opener FROM `{self._tid('openers')}` "
            "WHERE run_id=@run_id AND app=@app ORDER BY created_at"
        )
        tombstones = rows(
            f"SELECT correction_id,opener_created_at,model,opener_fingerprint,reason,evidence_ref "
            f"FROM `{self._tid('opener_retractions')}` "
            "WHERE run_id=@run_id AND app=@app",
            optional_opener_retractions=True,
        )
        counts = rows(
            "SELECT "
            f"(SELECT COUNT(*) FROM `{self._tid('profiles')}` WHERE run_id=@run_id AND app=@app) AS profiles, "
            f"(SELECT COUNT(*) FROM `{self._tid('profile_photos')}` WHERE run_id=@run_id AND app=@app) AS profile_photos, "
            f"(SELECT COUNT(*) FROM `{self._tid('labels')}` WHERE run_id=@run_id AND app=@app) AS labels, "
            f"(SELECT COUNT(*) FROM `{self._tid('decisions')}` WHERE run_id=@run_id AND app=@app) AS decisions, "
            # An answer to an archived AUTO pass is stored under the pass's own run_id with source
            # `auto_review_*`; it is not a swipe the owner made in that run, so it is not Like
            # evidence for it (one tap on the review card must not make the cleanup planner
            # refuse a run it had nothing to say about).
            f"(SELECT COUNT(*) FROM `{self._tid('labels')}` l WHERE run_id=@run_id AND app=@app AND liked=TRUE "
            f"AND {self._label_not_retracted()} "
            f"AND NOT STARTS_WITH(IFNULL(l.source,''), '{AUTO_REVIEW_SOURCE_PREFIX}')) AS like_labels, "
            f"(SELECT COUNT(*) FROM `{self._tid('labels')}` l WHERE run_id=@run_id AND app=@app AND liked=FALSE "
            f"AND {self._label_not_retracted()} "
            f"AND NOT STARTS_WITH(IFNULL(l.source,''), '{AUTO_REVIEW_SOURCE_PREFIX}')) AS pass_labels, "
            f"(SELECT COUNT(*) FROM `{self._tid('decisions')}` WHERE run_id=@run_id AND app=@app AND decision='like') AS like_decisions, "
            f"(SELECT COUNT(*) FROM `{self._tid('decisions')}` WHERE run_id=@run_id AND app=@app AND decision='dislike') AS pass_decisions"
        )[0]

        def stamp(value) -> str:
            if not isinstance(value, datetime):
                raise RetractionRefused("BigQuery opener row has invalid created_at")
            return value.astimezone(timezone.utc).isoformat()

        openers = [
            {
                "created_at": stamp(row["created_at"]),
                "model": str(row["model"]),
                "opener_fingerprint": canonical_sha(
                    {
                        "run_id": run_id,
                        "app": app,
                        "created_at": stamp(row["created_at"]),
                        "model": str(row["model"]),
                        "opener": str(row["opener"]),
                    }
                ),
            }
            for row in opener_rows
        ]
        return {
            "run_id": run_id,
            "app": app,
            "openers": openers,
            "decisions": [{} for _ in range(int(counts["decisions"]))],
            "preference_counts": {
                key: int(counts[key])
                for key in ("profiles", "profile_photos", "labels", "decisions")
            },
            "effective_counts": {
                key: int(counts[key])
                for key in (
                    "like_labels",
                    "like_decisions",
                    "pass_labels",
                    "pass_decisions",
                )
            },
            "retractions": [
                {
                    "correction_id": str(row["correction_id"]),
                    "opener_created_at": stamp(row["opener_created_at"]),
                    "model": str(row["model"]),
                    "opener_fingerprint": str(row["opener_fingerprint"]),
                    "reason": str(row["reason"]),
                    "evidence_ref": str(row["evidence_ref"]),
                }
                for row in tombstones
            ],
        }

    def append_opener_retraction(self, row: dict) -> bool:
        """Append one exact opener tombstone using bound idempotency checks.

        Published synchronously like its label counterpart, so ``_opener_retraction_ids``
        and these bound queries carry idempotency alone; nothing queues in ``_buf``.
        """
        from .retractions import RetractionRefused, canonical_sha
        from google.cloud import bigquery

        try:
            opener_at = normalize_timestamp_datetime(
                row["opener_created_at"], label="opener_created_at"
            )
        except (KeyError, ValueError) as exc:
            raise RetractionRefused(
                "BigQuery opener correction timestamp must be RFC-3339"
            ) from exc
        correction_id = row["correction_id"]
        by_key = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("correction_id", "STRING", correction_id),
                bigquery.ScalarQueryParameter(
                    "opener_created_at", "TIMESTAMP", opener_at
                ),
            ]
        )
        target = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id", "STRING", row["run_id"]),
                bigquery.ScalarQueryParameter("app", "STRING", row["app"]),
                bigquery.ScalarQueryParameter(
                    "opener_created_at", "TIMESTAMP", opener_at
                ),
            ]
        )
        opener_stamp = opener_at.isoformat()
        key = (correction_id, opener_stamp)
        with self._lock:
            self._require_open_for_write()
            if key in self._opener_retraction_ids:
                return False
            source = list(
                self.client.query(
                    f"SELECT created_at,model,opener FROM `{self._tid('openers')}` "
                    "WHERE run_id=@run_id AND app=@app AND created_at=@opener_created_at",
                    job_config=target,
                ).result()
            )
            if len(source) != 1 or str(source[0]["model"]) != row["model"]:
                raise RetractionRefused(
                    "target opener row is missing or does not match its cleanup plan"
                )
            source_at = source[0]["created_at"]
            if not isinstance(source_at, datetime):
                raise RetractionRefused("target opener row has an invalid created_at")
            source_stamp = source_at.astimezone(timezone.utc).isoformat()
            fingerprint = canonical_sha(
                {
                    "run_id": row["run_id"],
                    "app": row["app"],
                    "created_at": source_stamp,
                    "model": str(source[0]["model"]),
                    "opener": str(source[0]["opener"]),
                }
            )
            if source_stamp != opener_stamp or fingerprint != row["opener_fingerprint"]:
                raise RetractionRefused(
                    "target opener fingerprint no longer matches its cleanup plan"
                )
            by_same_plan = list(
                self.client.query(
                    f"SELECT correction_id,model,opener_fingerprint,reason,evidence_ref "
                    f"FROM `{self._tid('opener_retractions')}` WHERE correction_id=@correction_id "
                    "AND opener_created_at=@opener_created_at LIMIT 1",
                    job_config=by_key,
                ).result()
            )
            if by_same_plan:
                actual = dict(by_same_plan[0])
                expected = {key: row[key] for key in actual}
                if actual == expected:
                    return False
                raise RetractionRefused(
                    "correction id/opener timestamp has different cleanup data"
                )
            existing = list(
                self.client.query(
                    f"SELECT correction_id,model,opener_fingerprint,reason,evidence_ref "
                    f"FROM `{self._tid('opener_retractions')}` WHERE run_id=@run_id AND app=@app "
                    "AND opener_created_at=@opener_created_at LIMIT 1",
                    job_config=target,
                ).result()
            )
            if existing:
                raise RetractionRefused(
                    "opener row already has a different cleanup tombstone"
                )
            payload = dict(row)
            payload["opener_created_at"] = opener_stamp
            payload["created_at"] = datetime.now(timezone.utc).isoformat()
            errors = self._insert_rows_json(
                "opener_retractions", [payload], row_ids=[canonical_sha(row)]
            )
            if errors:
                raise RuntimeError(
                    f"BigQuery insert errors for opener_retractions: {errors}"
                )
            self._written["opener_retractions"] += 1
            self._opener_retraction_ids.add(key)
            return True

    def spend_today(self) -> float:
        """Sum of cost_usd already committed to BigQuery today (LOCAL day — same
        boundary as SQLiteStore.spend_today, via local_midnight_epoch(), NOT a UTC
        reporting day). Query/permission failures propagate: treating an unreadable spend
        ledger as $0 would silently disable the configured daily-cost ceiling."""
        start_dt = datetime.fromtimestamp(local_midnight_epoch(), tz=timezone.utc)
        rows = self.client.query(
            f"SELECT COALESCE(SUM(cost_usd), 0.0) AS total FROM `{self._tid('spend')}` "
            "WHERE created_at >= @day_start",
            job_config=_day_start_job_config(start_dt),
        ).result()
        for r in rows:
            return float(r["total"])
        return 0.0

    # --- writes (buffered, thread-safe) --------------------------------
    def _require_open_for_write(self, *, allow_closing: bool = False) -> None:
        """Reject a late worker write after the supervisor's final flush.

        Caller must hold ``self._lock``. ``allow_closing`` is reserved for an image-backed write
        that registered before close began; close waits for it, so its completed row belongs in
        the final flush. This remains a RuntimeError rather than a silent no-op: a physical
        action may already have landed, and losing its audit row must be visible in the wedged
        worker's error log.
        """
        if self._closed or (self._closing and not allow_closing):
            raise RuntimeError(
                "BigQueryStore is closed; refusing a late write after the final flush"
            )

    def _begin_async_write(self) -> None:
        """Register an image-backed write before its upload starts."""
        with self._lock:
            self._require_open_for_write()
            self._active_async_writes += 1

    def _finish_async_write(self) -> None:
        """Release one registered image-backed write and wake a waiting close."""
        with self._lock:
            self._active_async_writes -= 1
            if (
                self._active_async_writes < 0
            ):  # internal invariant; never hide a lifecycle bug
                self._active_async_writes = 0
                raise RuntimeError("BigQueryStore async-write accounting underflow")
            if self._active_async_writes == 0:
                self._writes_drained.notify_all()

    def archive_writes_in_flight(self) -> bool:
        """Whether a registered image-backed archive write is still running.

        Public seam for the supervisor's shutdown join, which must not call a worker wedged
        while it is mid-archive: this reports the exact counter ``close`` blocks on, so a
        caller can never be asked to wait for work close would not itself have waited for.
        """
        with self._lock:
            return self._active_async_writes > 0

    def _insert_rows_json(self, table: str, rows: list[dict], *, row_ids: list[str]):
        """Insert one buffered batch with bounded recovery beyond the SDK retry cycle."""
        delay = _BQ_INSERT_BACKOFF_S
        for attempt in range(1, _BQ_INSERT_ATTEMPTS + 1):
            retry = _deadline_retry()
            kwargs = {"row_ids": row_ids, "timeout": _BQ_INSERT_REQUEST_TIMEOUT_S}
            if retry is not None:
                kwargs["retry"] = retry
            try:
                return _call_with_supported_keywords(
                    self.client.insert_rows_json, self._tid(table), rows, **kwargs
                )
            except Exception as exc:  # noqa: BLE001 - classify optional SDK exceptions lazily
                if attempt == _BQ_INSERT_ATTEMPTS or not _retryable_bq_insert_exception(
                    exc
                ):
                    raise
                print(
                    f"BigQuery transient insert failure for table '{table}' after SDK retries "
                    f"(batch attempt {attempt}/{_BQ_INSERT_ATTEMPTS}); retrying the unchanged "
                    f"batch in {delay:.1f}s: {exc}"
                )
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable BigQuery insert retry state")

    @staticmethod
    def _report_profile_progress(
        progress, stage: str, completed: int, total: int
    ) -> None:
        """Best-effort persistence telemetry; observers must never affect durability."""
        if not callable(progress):
            return
        try:
            progress(stage, completed, total)
        except Exception:  # noqa: BLE001 -- status plumbing must not affect an already-landed action
            pass

    def record_profile(
        self,
        run_id,
        app,
        profile_id,
        liked,
        source="manual",
        photos=None,
        photo_count=0,
        capture_truncated: bool = False,
        progress=None,
    ) -> bool:
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
            print(
                f"BigQuery store warning: captured 0 photos for profile {profile_id}; "
                "skipping profile archive."
            )
            return False
        if not self._photo_bucket_private_verified:
            raise RuntimeError(
                "Refusing profile-photo upload: this BigQueryStore was created with "
                "ensure=False and its bucket privacy policy was not verified"
            )
        self._begin_async_write()
        try:
            created_at = _now()
            self._report_profile_progress(progress, "profile_upload", 0, len(photos))
            photo_rows = self._upload_profile_photos(
                run_id, app, profile_id, created_at, photos, progress=progress
            )
            if len(photo_rows) != len(photos):
                if photo_rows:
                    self._delete_profile_photo_rows(photo_rows)
                print(
                    f"BigQuery store warning: archived {len(photo_rows)}/{len(photos)} photos for "
                    f"profile {profile_id}; skipping its label to keep image data complete."
                )
                return False
            with self._lock:
                # close waits for this registered write, so its finished manifest must be
                # buffered before close performs its final flush.
                self._require_open_for_write(allow_closing=True)
                self._buf["profiles"].append(
                    {
                        "run_id": run_id,
                        "app": app,
                        "profile_id": profile_id,
                        "created_at": created_at,
                        "liked": bool(liked),
                        "source": source,
                        "photo_count": len(photo_rows),
                        "capture_truncated": bool(capture_truncated),
                    }
                )
                self._buf["profile_photos"].extend(photo_rows)
                self._maybe_flush("profiles")
                self._maybe_flush("profile_photos")
            self._report_profile_progress(
                progress, "profile_uploaded", len(photo_rows), len(photos)
            )
            return True
        finally:
            self._finish_async_write()

    def _upload_blob(
        self, blob, data: bytes, content_type: str, *, deadline: float | None = None
    ) -> bool:
        delay = _UPLOAD_BACKOFF_S
        for attempt in range(1, _UPLOAD_ATTEMPTS + 1):
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                print(
                    "BigQuery store photo upload deadline expired before the next attempt "
                    f"({getattr(blob, 'name', '?')})"
                )
                return False
            try:
                retry = _deadline_retry()
                kwargs = {
                    "content_type": content_type,
                    "timeout": min(_GCS_UPLOAD_REQUEST_TIMEOUT_S, remaining)
                    if remaining is not None
                    else _GCS_UPLOAD_REQUEST_TIMEOUT_S,
                }
                if retry is not None:
                    kwargs["retry"] = retry
                _call_with_supported_keywords(blob.upload_from_string, data, **kwargs)
                return True
            except Exception as exc:  # noqa: BLE001
                if attempt == _UPLOAD_ATTEMPTS:
                    print(
                        f"BigQuery store photo upload failed after {_UPLOAD_ATTEMPTS} "
                        f"attempts ({getattr(blob, 'name', '?')}): {exc}"
                    )
                    return False
                if remaining is not None:
                    delay = min(delay, max(0.0, deadline - time.monotonic()))
                    if delay <= 0:
                        continue
                time.sleep(delay)
                delay *= 2
        return False

    def _upload_profile_photos(
        self,
        run_id: str,
        app: str,
        profile_id: str,
        created_at: str,
        photos: list[bytes],
        *,
        progress=None,
        prefix: str = "profiles",
    ) -> list[dict]:
        # `prefix` is the top-level object-name folder: training archives use "profiles", the
        # AUTO pass archive "auto_passes" -- one upload/deadline/privacy path for both, and the
        # rows come back in exactly profile_photos' column set (auto_pass_photos shares it).
        rows = []
        deadline = time.monotonic() + _profile_upload_deadline_s(len(photos))
        for i, photo in enumerate(photos):
            if time.monotonic() >= deadline:
                print(
                    f"BigQuery store profile upload deadline expired after {len(rows)}/{len(photos)} "
                    f"photos for profile {profile_id}"
                )
                break
            digest = hashlib.sha256(photo).hexdigest()
            content_type, ext = _image_type(photo)
            object_name = (
                f"{prefix}/{app}/{run_id}/{profile_id}/{i:02d}-{digest[:16]}.{ext}"
            )
            blob = self._photo_bucket.blob(object_name)
            # Called directly, NOT through _call_with_supported_keywords: that shim exists for
            # third-party SDK entry points whose injected fakes may lack `retry`/`timeout`, and
            # routing our own method through it would silently drop `deadline` for any override
            # instead of failing. The thing being dropped is the bound that keeps a partial
            # photo set (and the refused label behind it) from following an already-landed
            # Hinge action, so a replacement that cannot take it must break loudly.
            if not self._upload_blob(blob, photo, content_type, deadline=deadline):
                break
            rows.append(
                {
                    "run_id": run_id,
                    "app": app,
                    "profile_id": profile_id,
                    "created_at": created_at,
                    "photo_index": i,
                    "gcs_uri": f"gs://{self.photo_bucket_name}/{object_name}",
                    "sha256": digest,
                    "byte_size": len(photo),
                    "content_type": content_type,
                }
            )
            self._report_profile_progress(
                progress, "profile_upload", len(rows), len(photos)
            )
        return rows

    def _delete_profile_photo_rows(self, rows: list[dict]) -> None:
        prefix = f"gs://{self.photo_bucket_name}/"
        for row in rows:
            uri = str(row.get("gcs_uri") or "")
            if not uri.startswith(prefix):
                continue
            try:
                retry = _deadline_retry()
                kwargs = {"timeout": _GCS_UPLOAD_REQUEST_TIMEOUT_S}
                if retry is not None:
                    kwargs["retry"] = retry
                _call_with_supported_keywords(
                    self._photo_bucket.blob(uri[len(prefix) :]).delete, **kwargs
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"BigQuery store warning: could not delete partial photo {uri}: {exc}"
                )

    def add_label(
        self,
        run_id,
        app,
        liked,
        embedding,
        source="manual",
        photo_count=0,
        profile_id="",
        profile_name="",
        **_,
    ):
        liked = bool(liked)
        embedding_vec = [float(x) for x in embedding]
        with self._lock:
            # Same footing as record_profile: this is the follow-on row of an archive close is
            # already waiting for, so refusing it would discard the label of a swipe that has
            # physically landed while close is still holding the buffer open for its manifest.
            self._require_open_for_write(allow_closing=True)
            self._buf["labels"].append(
                {
                    "run_id": run_id,
                    "app": app,
                    "profile_id": profile_id,
                    "created_at": _now(),
                    "liked": liked,
                    "source": source,
                    "embedding": embedding_vec,
                    "photo_count": int(photo_count),
                    "profile_name": str(profile_name or ""),
                }
            )
            label = (liked, list(embedding_vec), str(source or ""))
            if self._labels_cache is not None:
                if (
                    self._label_keys is not None
                    and self._label_keys_for is self._labels_cache
                ):
                    self._label_keys.append((str(app or ""), str(profile_id or "")))
                self._labels_cache.append(label)
            self._maybe_flush("labels")

    def remove_latest_training_label(self) -> dict | None:
        """Remove the latest visible label and return its operator-readable identity.

        When that newest label is an answer to an archived AUTO pass (source
        ``auto_review...``) this REFUSES with ``ValueError`` and deletes nothing: removing the
        answer would leave its ``auto_pass_reviews`` row marked answered with no label behind
        it, and reaching past it to delete the newest older MANUAL label (what an earlier
        version did) silently destroyed an unrelated swipe while the mistaken answer stayed in
        the training set.  ``undo_latest_auto_review`` is the undo for an answer.

        Deletes by the FULL identity the visibility predicate above selects on, not by
        profile_id alone, and binds profile_id NULL-safely: `labels.profile_id` arrived via
        _MIGRATIONS against a table that already held rows, so a pre-migration label reads back
        NULL, and BigQuery's `NULL = ''` is NULL rather than TRUE. That combination used to
        delete nothing while this method still returned a success dict, which the hub prints as
        "Training label removed for profile: X" over a label that is still there.

        The affected-row count is checked for exactly that reason -- it catches every zero-row
        cause, not just the one found. Anything other than a single row means the predicate did
        not name the row the SELECT chose, and the operator has to see that (hub/state.py turns
        the raise into a visible error) instead of a removal that never happened.
        """
        from google.cloud import bigquery

        with self._lock:
            self._require_open_for_write()
            self.flush()
            rows = self.client.query(
                f"SELECT l.run_id, l.app, l.source, l.profile_id, l.profile_name, l.created_at "
                f"FROM `{self._tid('labels')}` l WHERE {self._label_not_retracted()} "
                "ORDER BY l.created_at DESC LIMIT 1"
            ).result()
            row = next(iter(rows), None)
            if row is None:
                return None
            if str(row["source"] or "").startswith(AUTO_REVIEW_SOURCE_PREFIX):
                raise ValueError(REVIEW_ANSWER_IS_LATEST_MESSAGE)
            created_at = row["created_at"]
            job = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("run_id", "STRING", row["run_id"]),
                    bigquery.ScalarQueryParameter("app", "STRING", row["app"]),
                    bigquery.ScalarQueryParameter("source", "STRING", row["source"]),
                    bigquery.ScalarQueryParameter(
                        "profile_id", "STRING", row["profile_id"]
                    ),
                    bigquery.ScalarQueryParameter(
                        "created_at", "TIMESTAMP", created_at
                    ),
                ]
            )
            delete_job = self.client.query(
                f"DELETE FROM `{self._tid('labels')}` WHERE run_id=@run_id AND app=@app "
                "AND source=@source AND created_at=@created_at "
                "AND (profile_id=@profile_id OR (profile_id IS NULL AND @profile_id IS NULL))",
                job_config=job,
            )
            delete_job.result()
            affected = getattr(delete_job, "num_dml_affected_rows", None)
            if affected is None or int(affected) != 1:
                raise RuntimeError(
                    "BigQuery removed "
                    f"{'an unknown number of' if affected is None else affected} training label "
                    "rows for the label it selected; refusing to report a removal"
                )
            self._labels_cache = None
        return {
            "profile_name": str(row["profile_name"] or ""),
            "profile_id": str(row["profile_id"] or ""),
        }

    def record_decision(
        self,
        run_id,
        app,
        decision,
        score,
        source="auto",
        profile_id="",
        created_at=None,
    ):
        with self._lock:
            # allow_closing: see add_label -- an already-landed decision's audit row must not be
            # refused while close is still draining that same profile's archive.
            self._require_open_for_write(allow_closing=True)
            self._buf["decisions"].append(
                {
                    "run_id": run_id,
                    "app": app,
                    "created_at": _timestamp(created_at),
                    "decision": decision,
                    "score": float(score),
                    "source": source,
                    "profile_id": profile_id,
                }
            )
            self._maybe_flush("decisions")

    def record_opener(
        self,
        run_id,
        app,
        model,
        opener,
        referenced,
        angle="",
        item_description="",
        *,
        profile_id="",
        decision="",
        decision_source="",
        decision_created_at=None,
        model_item_index=None,
        prompt_sha256: str | None = None,
        profile_key: str = "",
    ):
        # `angle` and `item_description`: telemetry only (see the openers entry in _TABLES).
        # Both defaulted to "" so a caller that predates either still writes a valid row rather
        # than omitting the field. The live table already holds real rows, so each column got
        # to production via _MIGRATIONS, not via CREATE TABLE IF NOT EXISTS.
        #
        # `prompt_sha256` is the prompt-era digest (see _TABLES and opener/opener.py's
        # prompt_stamp), defaulted to None so an unstamped caller writes NULL -- "predates the
        # stamp" -- rather than a wrong era. flush_every=1 is a live configuration, so a row
        # carrying a field the live table does not declare is rejected on the FIRST insert and
        # takes its whole batch with it (skip_invalid_rows is unset); that is precisely why
        # this column needs its _MIGRATIONS line and not just its _TABLES entry.
        #
        # `profile_key` is the STABLE cross-time attribution key (ranker/profile_key.py),
        # unrelated to `profile_id`'s per-card lineage. Defaulted to "" -- matching
        # `profile_id`'s own convention -- so a caller that predates it, or one whose identity
        # capture was unknown for this card, still writes a complete row.
        with self._lock:
            # allow_closing: see add_label -- the opener belongs to a like that already landed.
            self._require_open_for_write(allow_closing=True)
            self._buf["openers"].append(
                {
                    "run_id": run_id,
                    "app": app,
                    "created_at": _now(),
                    "model": model,
                    "opener": opener,
                    "referenced": referenced,
                    "angle": angle,
                    "item_description": item_description,
                    "profile_id": profile_id,
                    "decision": decision,
                    "decision_source": decision_source,
                    "decision_created_at": (
                        None
                        if decision_created_at is None
                        else _timestamp(decision_created_at)
                    ),
                    "model_item_index": model_item_index,
                    "prompt_sha256": prompt_sha256,
                    "profile_key": profile_key,
                }
            )
            self._maybe_flush("openers")

    def record_opener_outcome(
        self,
        app: str,
        profile_key: str,
        outcome: str,
        *,
        observed_at=None,
        source: str = "owner",
        note: str = "",
    ) -> None:
        # See ranker/__init__.py's Store.record_opener_outcome for the full design rationale
        # (no run_id by design, profile_key accepted unvalidated including "", observed_at
        # distinct from this row's own created_at). Buffered like every other write here --
        # flushed on the normal batch/close cadence, not published synchronously.
        with self._lock:
            self._require_open_for_write(allow_closing=True)
            self._buf["opener_outcomes"].append(
                {
                    "app": app,
                    "profile_key": profile_key,
                    "outcome": outcome,
                    "observed_at": _timestamp(observed_at),
                    "source": source,
                    "note": note,
                    "created_at": _now(),
                }
            )
            self._maybe_flush("opener_outcomes")

    def joined_opener_outcomes(
        self, app: str, *, prompt_sha256: str | None = None
    ) -> list[dict]:
        """Every `opener_outcomes` row for `app` joined to its `openers` row by profile_key.

        See ranker/__init__.py's Store.joined_opener_outcomes for the full contract (why the
        join excludes '' and NULL profile_key on both sides, why it keeps only SENT openers
        (`decision = 'like'`), why `prompt_sha256=None` means "no filter" rather than "match
        NULL"). `opener_outcomes` is a table this change adds;
        a project whose live BigQuery dataset has not yet been touched by an `ensure=True`
        store (this method's own class does that in `_ensure_tables`) may not have it yet, so
        a NotFound naming this specific optional table reads as "no outcomes exist" rather
        than a real failure -- the same accommodation `opener_retractions` already has.
        """
        from google.cloud import bigquery

        parameters = [bigquery.ScalarQueryParameter("app", "STRING", app)]
        query = (
            "SELECT o.run_id AS run_id, o.model AS model, o.opener AS opener, "
            "o.prompt_sha256 AS prompt_sha256, o.profile_key AS profile_key, "
            "o.created_at AS opener_created_at, oc.outcome AS outcome, "
            "oc.observed_at AS observed_at, oc.source AS source, oc.note AS note, "
            "oc.created_at AS outcome_created_at "
            f"FROM `{self._tid('openers')}` o "
            f"JOIN `{self._tid('opener_outcomes')}` oc "
            "ON oc.app = o.app AND oc.profile_key = o.profile_key "
            "WHERE o.app = @app "
            "AND o.profile_key IS NOT NULL AND o.profile_key != '' "
            "AND oc.profile_key IS NOT NULL AND oc.profile_key != '' "
            # Only openers that were actually SENT can have earned an outcome. 'like' is the
            # ONE decision value that has ever meant "this opener reached the device" --
            # literally tools/opener_outcome_recorder.py's DECISION_SENT. Without this, a
            # profile drafted-then-DISLIKED in one run and LIKED in a later one returns the
            # owner's single recorded match TWICE, crediting the earlier prompt era with a
            # match earned by a draft nobody ever saw. NULL/'' (predates decision tracking)
            # and 'synthetic_replay' are excluded by the same literal, which is correct: none
            # of them proves a send, and this must never guess one.
            "AND o.decision = 'like' "
        )
        if prompt_sha256 is not None:
            query += "AND o.prompt_sha256 = @prompt_sha256 "
            parameters.append(
                bigquery.ScalarQueryParameter("prompt_sha256", "STRING", prompt_sha256)
            )
        query += "ORDER BY oc.created_at"
        job = bigquery.QueryJobConfig(query_parameters=parameters)
        try:
            rows = self.client.query(query, job_config=job).result()
        except Exception as exc:
            if not _missing_optional_opener_outcomes(exc, ensure=self._ensure):
                raise
            return []
        out = []
        for row in rows:
            entry = dict(row)
            for key in ("opener_created_at", "observed_at", "outcome_created_at"):
                value = entry.get(key)
                if isinstance(value, datetime):
                    entry[key] = value.astimezone(timezone.utc).isoformat()
            out.append(entry)
        return out

    def record_opener_send_evidence(
        self,
        run_id,
        app,
        opener,
        *,
        profile_id="",
        decision_source="auto",
        decision_created_at=None,
        model_item_index=None,
        evidence=None,
    ) -> bool:
        """Archive one verified-landed AUTO opener's pre-send PNG and queryable linkage.

        Raw images stay in the private GCS bucket; BigQuery stores only scalar metadata and the
        ``gs://`` reference. Evidence failure is deliberately non-fatal after a real Like: the
        ordinary opener/decision rows remain truthful, while a loud warning names the missing
        diagnostic artifact.
        """
        evidence = evidence if isinstance(evidence, dict) else {}
        frame = evidence.get("frame")
        if not isinstance(frame, bytes) or not frame:
            print(
                "BigQuery store warning: AUTO opener evidence had no pre-send frame; "
                "skipping its evidence row."
            )
            return False
        if not self._photo_bucket_private_verified:
            print(
                "BigQuery store warning: refusing AUTO opener evidence upload because this "
                "store has not verified the bucket privacy policy."
            )
            return False
        self._begin_async_write()
        try:
            opener_bytes = str(opener).encode("utf-8")
            opener_sha256 = hashlib.sha256(opener_bytes).hexdigest()
            frame_sha256 = hashlib.sha256(frame).hexdigest()
            evidence_id = hashlib.sha256(frame + b"\0" + opener_bytes).hexdigest()
            content_type, ext = _image_type(frame)
            object_name = (
                f"opener-evidence/{app}/{run_id}/{profile_id or 'unknown-profile'}/"
                f"{evidence_id}.{ext}"
            )
            blob = self._photo_bucket.blob(object_name)
            if not self._upload_blob(blob, frame, content_type):
                print(
                    "BigQuery store warning: pre-send AUTO opener screenshot was not archived; "
                    "skipping its evidence row."
                )
                return False

            expected_id = evidence.get("evidence_id")
            expected_frame_hash = evidence.get("frame_sha256")
            expected_opener_hash = evidence.get("opener_sha256")
            if any(
                (
                    expected_id is not None and expected_id != evidence_id,
                    expected_frame_hash is not None
                    and expected_frame_hash != frame_sha256,
                    expected_opener_hash is not None
                    and expected_opener_hash != opener_sha256,
                )
            ):
                print(
                    "BigQuery store warning: driver-supplied AUTO opener evidence hashes did not "
                    "match the content; persisted authoritative hashes computed at storage."
                )

            row = {
                "run_id": run_id,
                "app": app,
                "profile_id": profile_id,
                "created_at": _now(),
                "decision_source": decision_source,
                "decision_created_at": (
                    None
                    if decision_created_at is None
                    else _timestamp(decision_created_at)
                ),
                "outcome": "like_landed",
                "model_item_index": model_item_index,
                "opener": str(opener),
                "evidence_id": evidence_id,
                "opener_sha256": opener_sha256,
                "frame_sha256": frame_sha256,
                "gcs_uri": f"gs://{self.photo_bucket_name}/{object_name}",
                "byte_size": len(frame),
                "content_type": content_type,
            }
            with self._lock:
                self._require_open_for_write(allow_closing=True)
                self._buf["opener_send_evidence"].append(row)
                self._maybe_flush("opener_send_evidence")
            return True
        finally:
            self._finish_async_write()

    def record_opener_rejection(
        self,
        run_id,
        app,
        model,
        attempt,
        reason_code,
        reason,
        raw_opener,
        *,
        prompt_sha256: str | None = None,
    ):
        # `prompt_sha256`: the same prompt-era digest record_opener writes, keyword-only so the
        # seven existing parameters stay positional. This table's first added column, so its
        # _MIGRATIONS line is what puts it on the live table -- see the comment there.
        with self._lock:
            self._require_open_for_write()
            self._buf["opener_rejections"].append(
                {
                    "run_id": run_id,
                    "app": app,
                    "created_at": _now(),
                    "model": model,
                    "attempt": int(attempt),
                    "reason_code": reason_code,
                    "reason": reason,
                    "raw_opener": raw_opener,
                    "prompt_sha256": prompt_sha256,
                }
            )
            self._maybe_flush("opener_rejections")

    # --- auto pass archive + owner review ---------------------------------------------
    def record_auto_pass(
        self,
        *,
        run_id,
        app,
        profile_id,
        score,
        threshold,
        embedding,
        photos,
        profile_name="",
        random_sample,
        capture_truncated=False,
    ) -> bool:
        """Archive one AUTO pass: the screenshots the model saw, then the manifest row.

        Same contract as record_profile's archive: mandatory-but-non-fatal. Each upload is
        retried; if any photo cannot be stored, NO manifest is written, the blobs already
        uploaded are deleted, and False is returned (loudly) -- a pass the owner cannot see
        cannot be judged. Never writes a label, a decision, or a profiles/profile_photos row:
        the training archive is what retraction planning reasons about, and an unanswered
        pass is not training data. Malformed arguments raise ValueError.
        """
        args = validate_auto_pass(
            run_id=run_id,
            app=app,
            profile_id=profile_id,
            score=score,
            threshold=threshold,
            embedding=embedding,
            random_sample=random_sample,
        )
        key = (args["app"], args["profile_id"])
        photo_list = clean_auto_pass_photos(photos, args["profile_id"])
        if photo_list is None:
            return False
        if not self._photo_bucket_private_verified:
            raise RuntimeError(
                "Refusing auto-pass photo upload: this BigQueryStore was created with "
                "ensure=False and its bucket privacy policy was not verified"
            )
        with self._lock:
            already = key in self._auto_pass_keys
        if already:
            print(
                f"BigQuery store warning: auto pass {args['profile_id']} ({args['app']}) "
                "was already archived this run; not writing it again."
            )
            return False
        self._begin_async_write()
        try:
            created_at = _now()
            photo_rows = self._upload_profile_photos(
                args["run_id"],
                args["app"],
                args["profile_id"],
                created_at,
                photo_list,
                prefix="auto_passes",
            )
            if len(photo_rows) != len(photo_list):
                if photo_rows:
                    self._delete_profile_photo_rows(photo_rows)
                print(
                    f"BigQuery store warning: archived {len(photo_rows)}/{len(photo_list)} "
                    f"photos for auto pass {args['profile_id']}; skipping its pass archive "
                    "to keep image data complete."
                )
                return False
            with self._lock:
                # close waits for this registered write, so its finished manifest must be
                # buffered before close performs its final flush.
                self._require_open_for_write(allow_closing=True)
                self._buf["auto_passes"].append(
                    {
                        "run_id": args["run_id"],
                        "app": args["app"],
                        "profile_id": args["profile_id"],
                        "created_at": created_at,
                        "score": args["score"],
                        "threshold": args["threshold"],
                        "embedding": args["embedding"],
                        "photo_count": len(photo_rows),
                        "profile_name": str(profile_name or ""),
                        "random_sample": args["random_sample"],
                        "capture_truncated": bool(capture_truncated),
                    }
                )
                self._buf["auto_pass_photos"].extend(photo_rows)
                self._auto_pass_keys.add(key)
                self._maybe_flush("auto_pass_photos")
                self._maybe_flush("auto_passes")
            return True
        finally:
            self._finish_async_write()

    def _flush_auto_review_buffers(
        self,
        tables: tuple[str, ...] = (
            "auto_pass_photos",
            "auto_passes",
            "auto_pass_reviews",
        ),
    ) -> None:
        """Best-effort flush so a review read sees this process's own recent writes.

        Photos before the manifest and reviews last. A failure is printed, not raised: the
        read then serves what BigQuery already has, which is stale but never wrong (the
        process-local sets and the buffer scans in the write path still hold the invariants).
        """
        with self._lock:
            for table in tables:
                try:
                    self._flush_table(table)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"BigQuery store: could not flush '{table}' before a read: {exc}"
                    )

    def _auto_pass_pool_sql(self, *, random: bool, app: bool) -> str:
        """FROM/JOIN/WHERE of the passes a review can be shown: complete, unreviewed, in pool.

        "Complete" means every photo the manifest promises is already stored: a manifest whose
        photo rows were lost (a dropped insert, a failed flush) must never be offered, because
        an answer given without the screenshots would train the ranker on a guess. The pool
        flag is spliced in as a constant keyword (the kind was validated before this is
        built); every VALUE is a bound parameter (@app).
        """
        return (
            f"FROM `{self._tid('auto_passes')}` p "
            "JOIN (SELECT app, profile_id, COUNT(DISTINCT photo_index) AS stored "
            f"FROM `{self._tid('auto_pass_photos')}` GROUP BY app, profile_id) ph "
            "ON ph.app = p.app AND ph.profile_id = p.profile_id "
            f"WHERE p.random_sample = {'TRUE' if random else 'FALSE'} "
            "AND p.photo_count > 0 AND ph.stored >= p.photo_count "
            + ("AND p.app = @app " if app else "")
            + f"AND NOT EXISTS (SELECT 1 FROM `{self._tid('auto_pass_reviews')}` r "
            "WHERE r.app = p.app AND r.profile_id = p.profile_id)"
        )

    def next_auto_pass_for_review(self, *, kind, app=None) -> dict | None:
        """One unreviewed, fully archived pass of ``kind``, or None when that pool is empty.

        ``random``: uniform among the random_sample passes (the unbiased sample the live
        false-pass rate is measured on). ``near_cut``: among the rest, the largest
        (score - threshold), i.e. the pass closest below its own threshold.
        """
        kind = _require_auto_review_kind(kind)
        app = _require_optional_app(app)
        self._flush_auto_review_buffers()
        random_pool = kind == AUTO_REVIEW_KIND_RANDOM
        order = (
            "RAND()"
            if random_pool
            else "(p.score - p.threshold) DESC, p.created_at, p.profile_id"
        )
        parameters = [] if app is None else [("app", "STRING", app)]
        query = (
            "SELECT p.profile_id, p.app, p.run_id, p.created_at, p.score, p.threshold, "
            "p.photo_count, p.profile_name "
            + self._auto_pass_pool_sql(random=random_pool, app=app is not None)
            + f" ORDER BY {order} LIMIT 1"
        )
        try:
            rows = self.client.query(query, job_config=_job_config(parameters)).result()
        except Exception as exc:  # noqa: BLE001 - classified by name; SDK optional at import
            if not _missing_optional_auto_review(exc, ensure=self._ensure):
                raise
            return None
        row = next(iter(rows), None)
        if row is None:
            return None
        return {
            "profile_id": str(row["profile_id"]),
            "app": str(row["app"]),
            "run_id": str(row["run_id"]),
            "created_at": _epoch(row["created_at"]),
            "score": float(row["score"]),
            "threshold": float(row["threshold"]),
            "photo_count": int(row["photo_count"]),
            "profile_name": str(row["profile_name"] or ""),
            "kind": kind,
        }

    def auto_pass_photo(self, *, profile_id, index) -> bytes | None:
        """The archived bytes of one pass photo, or None when there is no such photo.

        Reads only from the configured private bucket and checks the bytes against the stored
        sha256: a mismatch is corruption the owner must hear about, not a picture to judge a
        person by, so it raises instead of returning the bytes.
        """
        _require_text(profile_id, "profile_id")
        if type(index) is not int or index < 0:
            raise ValueError(
                f"auto pass photo index must be a non-negative integer (got {index!r})"
            )
        uri = digest = None
        with self._lock:
            for row in self._buf["auto_pass_photos"]:
                if row["profile_id"] == profile_id and row["photo_index"] == index:
                    uri, digest = row["gcs_uri"], row["sha256"]
                    break
        if uri is None:
            try:
                rows = self.client.query(
                    "SELECT gcs_uri, sha256 "
                    f"FROM `{self._tid('auto_pass_photos')}` "
                    "WHERE profile_id = @profile_id AND photo_index = @photo_index "
                    "ORDER BY created_at LIMIT 1",
                    job_config=_job_config(
                        [
                            ("profile_id", "STRING", profile_id),
                            ("photo_index", "INT64", index),
                        ]
                    ),
                ).result()
            except Exception as exc:  # noqa: BLE001
                if not _missing_optional_auto_review(exc, ensure=self._ensure):
                    raise
                return None
            row = next(iter(rows), None)
            if row is None:
                return None
            uri, digest = str(row["gcs_uri"]), str(row["sha256"])
        prefix = f"gs://{self.photo_bucket_name}/"
        if not str(uri).startswith(prefix):
            raise RuntimeError(
                f"Refusing to read auto pass photo {uri!r}: it is outside the configured "
                f"private bucket {self.photo_bucket_name!r}"
            )
        blob = self._photo_bucket.blob(str(uri)[len(prefix) :])
        retry = _deadline_retry()
        kwargs = {"timeout": _GCS_UPLOAD_REQUEST_TIMEOUT_S}
        if retry is not None:
            kwargs["retry"] = retry
        try:
            data = _call_with_supported_keywords(blob.download_as_bytes, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "NotFound":
                print(
                    f"BigQuery store warning: auto pass photo {uri} is missing: {exc}"
                )
                return None
            raise
        data = bytes(data)
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(
                f"Auto pass photo {uri} does not match its stored sha256; refusing to show it"
            )
        return data

    def _find_auto_pass(self, profile_id: str) -> dict:
        """The one archived pass for ``profile_id`` (buffered or stored), else ValueError."""
        found: list[dict] = [
            dict(row)
            for row in self._buf["auto_passes"]
            if row["profile_id"] == profile_id
        ]
        try:
            rows = self.client.query(
                "SELECT run_id, app, profile_id, embedding, photo_count, profile_name, "
                "random_sample "
                f"FROM `{self._tid('auto_passes')}` WHERE profile_id = @profile_id "
                "ORDER BY created_at",
                job_config=_job_config([("profile_id", "STRING", profile_id)]),
            ).result()
        except Exception as exc:  # noqa: BLE001
            if not _missing_optional_auto_review(exc, ensure=self._ensure):
                raise
            rows = []
        found.extend(
            {
                "run_id": str(row["run_id"]),
                "app": str(row["app"]),
                "profile_id": str(row["profile_id"]),
                "embedding": [float(x) for x in row["embedding"]],
                "photo_count": int(row["photo_count"]),
                "profile_name": str(row["profile_name"] or ""),
                "random_sample": row["random_sample"],
            }
            for row in rows
        )
        if not found:
            raise ValueError(f"no archived auto pass for profile {profile_id!r}")
        # The same pass can legitimately appear twice (buffered AND already flushed, or a
        # re-flushed batch); two DIFFERENT (app, run) identities cannot be told apart by the
        # profile_id the caller gave, so refuse rather than label the wrong one.
        identities = {(row["app"], row["run_id"]) for row in found}
        if len(identities) > 1:
            raise ValueError(
                f"profile {profile_id!r} matches {len(identities)} archived auto passes; "
                "refusing to guess which one is being answered"
            )
        return found[0]

    def _auto_pass_reviewed(self, app: str, profile_id: str) -> bool:
        if (app, profile_id) in self._reviewed_auto_passes:
            return True
        if any(
            row["app"] == app and row["profile_id"] == profile_id
            for row in self._buf["auto_pass_reviews"]
        ):
            return True
        try:
            rows = self.client.query(
                f"SELECT 1 AS one FROM `{self._tid('auto_pass_reviews')}` "
                "WHERE app = @app AND profile_id = @profile_id LIMIT 1",
                job_config=_job_config(
                    [("app", "STRING", app), ("profile_id", "STRING", profile_id)]
                ),
            ).result()
        except Exception as exc:  # noqa: BLE001
            if not _missing_optional_auto_review(exc, ensure=self._ensure):
                raise
            return False
        return next(iter(rows), None) is not None

    def _existing_review_label(self, pass_row: dict, source: str) -> bool | None:
        """The ``liked`` of a label this pass's answer already produced, else None.

        This is what makes a retry safe: the label is written (and flushed) BEFORE the review
        row, so a failure in between leaves a label with no review. The owner then answers the
        same pass again, and that must find the existing label rather than add a second one.
        """
        for row in self._buf["labels"]:
            if (
                row["run_id"] == pass_row["run_id"]
                and row["app"] == pass_row["app"]
                and row["profile_id"] == pass_row["profile_id"]
                and row["source"] == source
            ):
                return bool(row["liked"])
        rows = self.client.query(
            f"SELECT l.liked FROM `{self._tid('labels')}` l "
            "WHERE l.app = @app AND l.run_id = @run_id AND l.profile_id = @profile_id "
            "AND l.source = @source ORDER BY l.created_at LIMIT 1",
            job_config=_job_config(
                [
                    ("app", "STRING", pass_row["app"]),
                    ("run_id", "STRING", pass_row["run_id"]),
                    ("profile_id", "STRING", pass_row["profile_id"]),
                    ("source", "STRING", source),
                ]
            ),
        ).result()
        row = next(iter(rows), None)
        return None if row is None else bool(row["liked"])

    def record_auto_pass_review(self, *, profile_id, would_like) -> dict:
        """Record the owner's answer to one archived pass: a training label AND a review row.

        ValueError for an unknown, ambiguous or already-reviewed pass (an answer is final).
        Ordering is the crash-safety argument, since BigQuery has no multi-table transaction:
        the LABEL is flushed first, the review row second. A failure between them leaves a
        label with no review -- never a review with no label -- and the retry finds that label
        (`_existing_review_label`) and writes only the missing review row, so a pass can never
        get a second label. If the label already exists with a different answer, the REVIEW
        row follows the label (it is what training already used) and a notice says so.
        """
        _require_text(profile_id, "profile_id")
        if type(would_like) is not bool:
            raise ValueError(f"would_like must be a bool (got {would_like!r})")
        with self._lock:
            self._require_open_for_write()
            pass_row = self._find_auto_pass(profile_id)
            app = pass_row["app"]
            if self._auto_pass_reviewed(app, profile_id):
                raise ValueError(f"auto pass {profile_id!r} was already reviewed")
            random_sample = pass_row["random_sample"]
            if type(random_sample) is not bool:
                raise ValueError(
                    f"archived auto pass {profile_id!r} has no valid random_sample flag"
                )
            kind = (
                AUTO_REVIEW_KIND_RANDOM if random_sample else AUTO_REVIEW_KIND_NEAR_CUT
            )
            source = auto_review_label_source(kind)
            existing = self._existing_review_label(pass_row, source)
            if existing is None:
                self.add_label(
                    run_id=pass_row["run_id"],
                    app=app,
                    liked=would_like,
                    embedding=pass_row["embedding"],
                    source=source,
                    photo_count=pass_row["photo_count"],
                    profile_id=profile_id,
                    profile_name=pass_row["profile_name"],
                )
                recorded = would_like
            else:
                recorded = existing
                print(
                    f"BigQuery store: auto pass {profile_id} already has a {source} label "
                    f"(liked={existing}) from an earlier, interrupted answer; reusing it "
                    "instead of writing a second label."
                    + (
                        f" Your new answer ({would_like}) differs and is NOT applied."
                        if existing != would_like
                        else ""
                    )
                )
            # Label durable BEFORE the review row exists anywhere. A raise here leaves the
            # label buffered (and the review unwritten); the retry above finds it.
            self._flush_table("labels")
            self._buf["auto_pass_reviews"].append(
                {
                    "profile_id": profile_id,
                    "app": app,
                    "reviewed_at": _now(),
                    "would_like": recorded,
                    "kind": kind,
                }
            )
            self._reviewed_auto_passes.add((app, profile_id))
            try:
                self._flush_table("auto_pass_reviews")
            except Exception as exc:  # noqa: BLE001
                # The label is already durable and the review row stays buffered (it is
                # re-sent by the next flush and by close). Raising would invite a retry that
                # can only answer "already reviewed" -- say so loudly instead.
                print(
                    f"BigQuery store: the review row for auto pass {profile_id} is queued "
                    f"but not yet written ({exc}); it will be re-sent on the next flush."
                )
        return {"kind": kind, "label_source": source}

    def _dml(self, sql: str, parameters: list[tuple[str, str, object]]) -> int:
        """Run one bound DML statement and return the rows it affected (raises if unknown)."""
        job = self.client.query(sql, job_config=_job_config(parameters))
        job.result()
        affected = getattr(job, "num_dml_affected_rows", None)
        if affected is None:
            raise RuntimeError(
                "BigQuery did not report how many rows a DML statement changed; refusing to "
                "assume it did what was intended"
            )
        return int(affected)

    def undo_latest_auto_review(self) -> dict | None:
        """Undo the newest answer to an archived pass: delete its review row AND its label.

        The pass goes back to the unreviewed pool and can be answered again.  Returns the
        undone answer (``profile_id``, ``profile_name``, ``kind``, ``would_like``), or ``None``
        when there is no answer to undo.

        BigQuery has no multi-table transaction, so the ORDER is the safety argument -- the
        mirror image of ``record_auto_pass_review`` (label first, review second): the REVIEW
        row goes first, then the label.  A failure between them leaves a label with no review
        row (the retry path of ``record_auto_pass_review`` already knows how to adopt that),
        never a review row with no label.  The same streaming-buffer limit as
        ``remove_latest_training_label`` applies: BigQuery refuses to DELETE rows it received
        by streaming in roughly the last half hour, and that refusal is raised, not hidden.
        """
        with self._lock:
            self._require_open_for_write()
            # The answer being undone may still sit in this process's buffers; an undo that
            # could not see it would undo the PREVIOUS answer instead.  A failing flush raises.
            self._flush_table("labels")
            self._flush_table("auto_pass_reviews")
            try:
                rows = self.client.query(
                    "SELECT profile_id, app, kind, would_like "
                    f"FROM `{self._tid('auto_pass_reviews')}` "
                    "ORDER BY reviewed_at DESC, profile_id DESC LIMIT 1"
                ).result()
            except Exception as exc:  # noqa: BLE001
                if not _missing_optional_auto_review(exc, ensure=self._ensure):
                    raise
                return None
            review = next(iter(rows), None)
            if review is None:
                return None
            profile_id, app = str(review["profile_id"]), str(review["app"])
            kind = _require_auto_review_kind(review["kind"])
            source = auto_review_label_source(kind)
            identity = [
                ("app", "STRING", app),
                ("profile_id", "STRING", profile_id),
                ("source", "STRING", source),
            ]
            label_rows = self.client.query(
                "SELECT l.run_id, l.profile_name, l.created_at "
                f"FROM `{self._tid('labels')}` l "
                "WHERE l.app = @app AND l.profile_id = @profile_id AND l.source = @source "
                "ORDER BY l.created_at DESC LIMIT 1",
                job_config=_job_config(identity),
            ).result()
            label = next(iter(label_rows), None)
            name = str(label["profile_name"] or "") if label is not None else ""
            if not name:
                manifest = next(
                    iter(
                        self.client.query(
                            "SELECT profile_name "
                            f"FROM `{self._tid('auto_passes')}` "
                            "WHERE app = @app AND profile_id = @profile_id LIMIT 1",
                            job_config=_job_config(identity[:2]),
                        ).result()
                    ),
                    None,
                )
                name = (
                    str(manifest["profile_name"] or "") if manifest is not None else ""
                )
            removed = self._dml(
                f"DELETE FROM `{self._tid('auto_pass_reviews')}` "
                "WHERE app = @app AND profile_id = @profile_id",
                identity[:2],
            )
            if removed != 1:
                raise RuntimeError(
                    f"BigQuery removed {removed} review rows for the answer it selected; "
                    "refusing to report an undo (rows streamed in the last ~30 minutes cannot "
                    "be deleted yet -- try again later)"
                )
            self._reviewed_auto_passes.discard((app, profile_id))
            if label is not None:
                removed = self._dml(
                    f"DELETE FROM `{self._tid('labels')}` "
                    "WHERE app = @app AND profile_id = @profile_id AND source = @source "
                    "AND run_id = @run_id AND created_at = @created_at",
                    [
                        *identity,
                        ("run_id", "STRING", str(label["run_id"])),
                        ("created_at", "TIMESTAMP", label["created_at"]),
                    ],
                )
                if removed != 1:
                    raise RuntimeError(
                        f"BigQuery removed {removed} label rows for the answer it undid; the "
                        "review row is already gone, so re-answering the pass will adopt "
                        "whatever label is left"
                    )
            self._labels_cache = None
        return {
            "profile_id": profile_id,
            "profile_name": name,
            "kind": kind,
            "would_like": bool(review["would_like"]),
        }

    def prune_auto_passes(self, *, max_passes=None, max_age_days=None, now=None) -> int:
        """Delete archived passes (manifest, photo rows AND the GCS objects) beyond retention.

        A pass goes when it is older than ``max_age_days`` or not among the ``max_passes``
        newest, answered or not; ``None``/0 leaves that bound off.  Returns how many passes
        were removed.  The owner's ANSWERS are untouched (labels, review rows, decisions).

        Each pass is removed object-first so every step can be repeated: GCS objects, then the
        photo rows, then the manifest.  A crash leaves a manifest that the next prune finds
        again; it never leaves objects nothing points at.  Only rows old enough to be out of
        BigQuery's streaming buffer are normally selected; a refusal for that reason is raised
        after the passes already removed, and the next prune retries.
        """
        max_passes, max_age_days = require_auto_pass_retention(max_passes, max_age_days)
        if max_passes is None and max_age_days is None:
            return 0
        clock = (
            time.time() if now is None else normalize_timestamp_epoch(now, label="now")
        )
        with self._lock:
            self._require_open_for_write()
            self._flush_auto_review_buffers(("auto_pass_photos", "auto_passes"))
        # The lock is released for the rest: deleting a backlog is hundreds of Cloud Storage
        # calls and several DML round trips, and holding the store lock through them would stall
        # every decision/label write behind housekeeping.  Safe because the doomed set is a
        # snapshot of specific (app, profile_id) pairs: a pass archived meanwhile is newer than
        # every one of them and is not in it.
        try:
            rows = self.client.query(
                f"SELECT app, profile_id, created_at FROM `{self._tid('auto_passes')}`"
            ).result()
            passes = [
                (str(r["app"]), str(r["profile_id"]), _epoch(r["created_at"]))
                for r in rows
            ]
        except Exception as exc:  # noqa: BLE001
            if not _missing_optional_auto_review(exc, ensure=self._ensure):
                raise
            return 0
        doomed = doomed_auto_passes(passes, max_passes, max_age_days, clock)
        by_app: dict[str, list[str]] = {}
        for app, profile_id in doomed:
            by_app.setdefault(app, []).append(profile_id)
        removed = 0
        for app, ids in by_app.items():
            for start in range(0, len(ids), _PRUNE_CHUNK):
                chunk = ids[start : start + _PRUNE_CHUNK]
                names = [f"p{i}" for i in range(len(chunk))]
                params = [("app", "STRING", app)] + [
                    (n, "STRING", pid) for n, pid in zip(names, chunk, strict=True)
                ]
                where = (
                    "WHERE app = @app AND profile_id IN ("
                    + ", ".join(f"@{n}" for n in names)
                    + ")"
                )
                photo_rows = self.client.query(
                    f"SELECT gcs_uri FROM `{self._tid('auto_pass_photos')}` {where}",
                    job_config=_job_config(params),
                ).result()
                self._delete_profile_photo_rows(
                    [{"gcs_uri": str(r["gcs_uri"])} for r in photo_rows]
                )
                self._dml(
                    f"DELETE FROM `{self._tid('auto_pass_photos')}` {where}", params
                )
                self._dml(f"DELETE FROM `{self._tid('auto_passes')}` {where}", params)
                with self._lock:
                    for pid in chunk:
                        self._auto_pass_keys.discard((app, pid))
                removed += len(chunk)
        return removed

    def auto_review_stats(self, *, app=None, window=DEFAULT_AUTO_REVIEW_WINDOW) -> dict:
        """One-query tally for the ranker's live false-pass estimate and the hub's card.

        ``random_reviewed`` / ``random_would_like`` cover the most recent ``window`` random
        reviews (a rolling estimate that tracks the current model); near-cut counts are
        cumulative. ``auto_likes`` / ``auto_passes`` count AUTO decisions since the first
        archived pass's own decision and are 0/0 while nothing has been archived;
        ``auto_model_passes`` is the part of ``auto_passes`` that entered the archive (the rest
        are likes the session policy or like-ratio ceiling demoted, which the random pool never
        samples). This process's own
        buffered passes, reviews and decisions are flushed first (best effort) so the counts
        include them -- otherwise the first flush_every passes of a run would be invisible to
        the very estimate they feed. That is one round trip per table with pending rows, which
        is why callers on the training path reuse the result for a short TTL
        (ranker/decider.py's LIVE_REVIEW_TTL_S) instead of asking after every label.
        """
        app = _require_optional_app(app)
        window = _require_window(window)
        self._flush_auto_review_buffers(
            ("decisions", "auto_pass_photos", "auto_passes", "auto_pass_reviews")
        )
        app_clause = "" if app is None else " AND app = @app"
        # `review_window`, not `window`: WINDOW is a reserved word in GoogleSQL, and a
        # parameter name is the one place nobody would think to check.
        parameters: list[tuple[str, str, object]] = [("review_window", "INT64", window)]
        if app is not None:
            parameters.append(("app", "STRING", app))
        reviews = f"`{self._tid('auto_pass_reviews')}`"

        def recent_random(select: str) -> str:
            return (
                f"(SELECT {select} FROM (SELECT would_like FROM {reviews} "
                f"WHERE kind = 'random'{app_clause} "
                "ORDER BY reviewed_at DESC LIMIT @review_window))"
            )

        def near_cut(select: str) -> str:
            return (
                f"(SELECT {select} FROM {reviews} WHERE kind = 'near_cut'{app_clause})"
            )

        # The window opens at the DECISION row of the earliest archived pass, not at that
        # pass's own auto_passes row: the worker records the decision first and archives
        # afterwards, so counting from auto_passes.created_at would drop the very decision that
        # started the window and bias pass_share by one pass.  The join is on the pass's lineage
        # profile_id (exact, no time tolerance); a pass whose decision row is missing falls back
        # to its own timestamp.  With no archived pass the MIN is NULL, so the count is 0.
        passes = f"`{self._tid('auto_passes')}`"
        window_start = (
            "(SELECT MIN(ts) FROM ("
            f"SELECT created_at AS ts FROM {passes} WHERE TRUE{app_clause} "
            "UNION ALL "
            "SELECT d.created_at AS ts "
            f"FROM `{self._tid('decisions')}` d JOIN {passes} p "
            "ON p.app = d.app AND p.profile_id = d.profile_id "
            "WHERE d.source = 'auto' AND d.decision = 'dislike'"
            f"{app_clause.replace('app =', 'd.app =')}))"
        )

        def decisions(decision: str) -> str:
            return (
                f"(SELECT COUNT(*) FROM `{self._tid('decisions')}` "
                f"WHERE source = 'auto' AND decision = '{decision}'{app_clause} "
                f"AND created_at >= {window_start})"
            )

        def model_passes() -> str:
            # AUTO passes that actually entered the archive, matched by the pass's lineage
            # profile_id (exact, like the window start above).  EXISTS, not a join: BigQuery
            # streaming dedup is best effort, and a duplicated archive row must not count a
            # decision twice (that would push this above auto_passes, which the live-review
            # parser rejects as a store fault).
            return (
                f"(SELECT COUNT(*) FROM `{self._tid('decisions')}` d "
                "WHERE d.source = 'auto' AND d.decision = 'dislike' "
                f"AND EXISTS (SELECT 1 FROM {passes} p "
                "WHERE p.app = d.app AND p.profile_id = d.profile_id)"
                f"{app_clause.replace('app =', 'd.app =')} "
                f"AND d.created_at >= {window_start})"
            )

        query = (
            "SELECT "
            f"(SELECT COUNT(*) {self._auto_pass_pool_sql(random=True, app=app is not None)}) "
            "AS pending_random, "
            f"(SELECT COUNT(*) {self._auto_pass_pool_sql(random=False, app=app is not None)}) "
            "AS pending_near_cut, "
            f"{recent_random('COUNT(*)')} AS random_reviewed, "
            f"{recent_random('COUNTIF(would_like)')} AS random_would_like, "
            f"{near_cut('COUNT(*)')} AS near_cut_reviewed, "
            f"{near_cut('COUNTIF(would_like)')} AS near_cut_would_like, "
            f"{decisions('like')} AS auto_likes, "
            f"{decisions('dislike')} AS auto_passes, "
            f"{model_passes()} AS auto_model_passes"
        )
        names = (
            "pending_random",
            "pending_near_cut",
            "random_reviewed",
            "random_would_like",
            "near_cut_reviewed",
            "near_cut_would_like",
            "auto_likes",
            "auto_passes",
            "auto_model_passes",
        )
        try:
            rows = self.client.query(query, job_config=_job_config(parameters)).result()
        except Exception as exc:  # noqa: BLE001
            if not _missing_optional_auto_review(exc, ensure=self._ensure):
                raise
            return dict.fromkeys(names, 0)
        row = next(iter(rows), None)
        if row is None:
            return dict.fromkeys(names, 0)
        return {name: int(row[name] or 0) for name in names}

    def record_spend(self, run_id, model, usage: Usage, cost):
        # cost is None when the call's price couldn't be determined (e.g. no
        # budget.pricing entry for the model) — stored as NULL, distinct from a
        # genuinely free $0.00 call.
        with self._lock:
            self._require_open_for_write()
            self._buf["spend"].append(
                {
                    "run_id": run_id,
                    "created_at": _now(),
                    "model": model,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cache_read_tokens": usage.cache_read_input_tokens,
                    "cache_write_tokens": usage.cache_creation_input_tokens,
                    "cost_usd": None if cost is None else float(cost),
                }
            )
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
        errors = self._insert_rows_json(table, rows, row_ids=row_ids)
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
            bad_ids = {
                row_ids[e["index"]] for e in errors if e.get("index") is not None
            }
            for rid in bad_ids:
                key = f"{table}:{rid}"
                self._fail_counts[key] = self._fail_counts.get(key, 0) + 1
            drop_ids = {
                rid
                for rid in bad_ids
                if self._fail_counts[f"{table}:{rid}"] >= _MAX_INSERT_ATTEMPTS
            }
            if drop_ids:
                kept, dropped = [], []
                for row, rid in zip(rows, row_ids, strict=True):
                    (dropped if rid in drop_ids else kept).append(row)
                for row in dropped:
                    print(
                        f"BigQuery store: DROPPING a row from '{table}' after "
                        f"{_MAX_INSERT_ATTEMPTS} consecutive failed insert attempts -- "
                        f"this row's data is LOST (never written to BigQuery). "
                        f"BigQuery errors: {errors}. Row: {row}"
                    )
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
        errors: list[tuple[str, Exception]] = []
        with self._lock:
            for table in self._buf:
                try:
                    self._flush_table(table)
                except Exception as exc:  # noqa: BLE001
                    errors.append((table, exc))
                    print(f"BigQuery flush error for table '{table}': {exc}")
        if errors:
            raise RuntimeError(
                f"BigQuery flush failed for {len(errors)} table(s): "
                + "; ".join(f"{table}: {exc}" for table, exc in errors)
            )

    def close(self) -> None:
        # First fence new writes, then drain uploads that registered before the fence. Their rows
        # become part of the final flush; uploads that have not started yet are rejected before
        # touching Cloud Storage. Mark closed even when flushing raises: the supervisor has ended
        # ownership of this store, and accepting more rows into an already-failed shutdown buffer
        # would only make the loss larger.
        with self._lock:
            if self._closed:
                return
            if self._closing:
                while not self._closed:
                    self._writes_drained.wait()
                return
            self._closing = True
            while self._active_async_writes:
                self._writes_drained.wait()
            try:
                self.flush()
            finally:
                self._closed = True
                self._writes_drained.notify_all()

    def saved_summary(self) -> str:
        """Human-readable tally of rows confirmed inserted to BigQuery this run, for the
        shutdown confirmation log. Empty buffers + a non-empty tally = everything landed."""
        with self._lock:
            parts = [
                f"{name}={self._written[name]}"
                for name in _TABLES
                if self._written[name]
            ]
            dropped = [
                f"{name}={self._dropped[name]}"
                for name in _TABLES
                if self._dropped[name]
            ]
            pending = sum(len(self._buf[name]) for name in _TABLES)
        body = ", ".join(parts) if parts else "nothing new"
        if dropped:
            body += f" (DROPPED, never written: {', '.join(dropped)})"
        return f"{body} (pending={pending})" if pending else body

    def dropped_rows(self) -> dict[str, int]:
        """Per-table tally of rows this run PERMANENTLY GAVE UP ON — data that is LOST.

        The same fact `saved_summary` already renders as prose, exposed as DATA so a caller can
        branch on it instead of grepping a sentence. That distinction is the whole point: a row
        BigQuery rejects permanently (bad UTF-8 in raw_opener, an over-length field) is dropped
        by `_flush_table` after _MAX_INSERT_ATTEMPTS so the rest of the buffer can drain, the
        RuntimeError it raises is caught and merely WARNED about by opener/service.py, and the
        final shutdown flush then succeeds with an empty buffer. Every terminal signal the
        supervisor had — flush returned, phase is "stopped" — said the run saved everything,
        and it printed "✅ all data saved" over the top of real data loss (see supervisor.py's
        dropped-rows branch, which exists because of exactly that).

        Only nonzero tables appear, so an empty dict means "nothing was lost" and `if
        store.dropped_rows():` is the whole test a caller needs. See ranker/__init__.py's
        Store.dropped_rows for the contract both backends share.
        """
        with self._lock:
            return {
                name: self._dropped[name] for name in _TABLES if self._dropped[name]
            }

    def pending_opener_rejections(self) -> list[dict]:
        """(2026-09-17) Snapshot of `opener_rejections` rows still sitting in the buffer.

        Exists for exactly one caller: supervisor.py's shutdown path
        (_deadletter_stranded_opener_rejections). A wire failure that only ever surfaces at
        the FINAL store.flush() -- because storage.bigquery.flush_every kept every row below it
        buffered without ever raising synchronously inside record_opener_rejection -- would
        otherwise be reported as a bare exception with no row-level evidence at all. This lets
        that shutdown path dead-letter each surviving row with its own reason_code/reason/
        raw_opener, the same way a row that DID raise synchronously already gets dead-lettered
        by opener/service.py. See _write_opener_rejection_deadletter's "THE flush_every
        DEPENDENCY" docstring section for the full picture.

        Deliberately NOT added to ranker/__init__.py's Store Protocol: SQLiteStore has no
        buffer a row can be silently lost from (see its own dropped_rows docstring) and so has
        nothing to report here -- the shutdown caller duck-types this via getattr(...) and
        treats its absence as "nothing to recover", not an error.

        Returns a snapshot, not a live view -- one dict copy per row -- so a caller mutating the
        result cannot corrupt a row this store might still successfully flush on a later retry.
        """
        with self._lock:
            return [dict(row) for row in self._buf["opener_rejections"]]
