"""Local SQLite storage backend.

Two backends implement the same ``Store`` surface defined in ranker/__init__.py
(see bigquery_store.py for the cloud one). The supervisor loads labels once at
startup, keeps them in memory for fast inference, and appends new rows through
the store — so the hot path never blocks on per-row I/O regardless of backend.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import numbers
import sqlite3
import threading
import time
from pathlib import Path

from ..costing import Usage
from ..private_files import (
    ensure_private_dir,
    tighten_private_file,
    write_private_bytes,
)


def local_midnight_epoch() -> float:
    """Epoch seconds for local midnight — the start of "today" in the host's own
    timezone. The daily anti-ban limits (limits.max_per_day, budget.day_budget_usd)
    are human-scale, tied to the owner's own day, not a UTC reporting boundary, so
    BOTH store backends derive "today" from this one function and can't drift apart
    again (see BigQueryStore.count_today / spend_today)."""
    return (
        datetime.datetime.now()
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .timestamp()
    )


def _invalid_timestamp(label: str) -> ValueError:
    """One stable public error for timestamp values neither store can represent."""
    return ValueError(
        f"{label} must be a finite epoch number or a timezone-aware ISO-8601 timestamp"
    )


def _timestamp_parts(
    value: object, *, label: str, allow_epoch_text: bool = False
) -> tuple[float, datetime.datetime]:
    """Validate a portable timestamp and return its exact epoch plus UTC datetime.

    Numeric epoch values remain exact (important for SQLite's equality-bound correction
    records).  Normal action/correction documents use timezone-aware ISO text; SQLite's
    correction planner additionally emits a ``.17g`` decimal epoch spelling, enabled only
    by callers that need that documented round-trip contract.
    """
    if isinstance(value, bool):
        raise _invalid_timestamp(label)

    numeric: float | None = None
    if isinstance(value, (int, float)):
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _invalid_timestamp(label) from exc
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise _invalid_timestamp(label)
        # A terminal Z is the RFC-3339 spelling for UTC; avoid a broad replace that
        # could turn malformed text into something datetime.fromisoformat accepts.
        iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            parsed = datetime.datetime.fromisoformat(iso_text)
        except ValueError as iso_error:
            if not allow_epoch_text:
                raise _invalid_timestamp(label) from iso_error
            try:
                numeric = float(text)
            except (TypeError, ValueError, OverflowError) as numeric_error:
                raise _invalid_timestamp(label) from numeric_error
        else:
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise _invalid_timestamp(label)
            try:
                epoch = parsed.timestamp()
            except (OSError, OverflowError, ValueError) as exc:
                raise _invalid_timestamp(label) from exc
            if not math.isfinite(epoch):
                raise _invalid_timestamp(label)
            return epoch, parsed.astimezone(datetime.timezone.utc)
    else:
        raise _invalid_timestamp(label)

    assert numeric is not None
    if not math.isfinite(numeric):
        raise _invalid_timestamp(label)
    try:
        parsed = datetime.datetime.fromtimestamp(numeric, tz=datetime.timezone.utc)
    except (OSError, OverflowError, ValueError) as exc:
        raise _invalid_timestamp(label) from exc
    return numeric, parsed


def normalize_timestamp_epoch(
    value: object, *, label: str = "timestamp", allow_epoch_text: bool = False
) -> float:
    """Return a strict, SQLite-safe epoch timestamp without lossy coercion."""
    return _timestamp_parts(value, label=label, allow_epoch_text=allow_epoch_text)[0]


def normalize_timestamp_iso(
    value: object, *, label: str = "timestamp", allow_epoch_text: bool = False
) -> str:
    """Return a strict UTC ISO timestamp suitable for BigQuery JSON TIMESTAMP fields."""
    return _timestamp_parts(value, label=label, allow_epoch_text=allow_epoch_text)[
        1
    ].isoformat()


def normalize_timestamp_datetime(
    value: object, *, label: str = "timestamp", allow_epoch_text: bool = False
) -> datetime.datetime:
    """Return the strict UTC ``datetime`` needed for bound BigQuery TIMESTAMP parameters."""
    return _timestamp_parts(value, label=label, allow_epoch_text=allow_epoch_text)[1]


# --- auto pass review ------------------------------------------------------------------------
# AUTO archives every profile the model PASSES (screenshots + score + embedding) so the owner
# can later say "I would have liked her" or "correct pass". Every answer becomes a training
# label. Both backends share the vocabulary and the argument validation below, so SQLiteStore
# and BigQueryStore can never disagree about what a valid archive or review is.
#
# `random` passes are drawn uniformly at record time (worker's `auto_review_random_share`) and
# are the UNBIASED sample the live false-pass rate is measured on; `near_cut` passes are the
# rest, reviewed closest-to-the-threshold first because those are the labels that move the
# decision boundary the most.
AUTO_REVIEW_KIND_RANDOM = "random"
AUTO_REVIEW_KIND_NEAR_CUT = "near_cut"
AUTO_REVIEW_KINDS = (AUTO_REVIEW_KIND_RANDOM, AUTO_REVIEW_KIND_NEAR_CUT)
# `labels.source` of an answer. The shared prefix is what ranker/threshold.py keys on to tell a
# review label from a MANUAL calibration label (a review label must never calibrate the recall
# threshold it is also being used to measure).
AUTO_REVIEW_SOURCE_PREFIX = "auto_review"
DEFAULT_AUTO_REVIEW_WINDOW = 200
# What "Remove last saved label" says when the newest label is an answer to an auto-pass review
# (both backends raise ValueError with it; the hub shows it to the owner as it is).
REVIEW_ANSWER_IS_LATEST_MESSAGE = (
    "the latest saved label is an answer to an auto-pass review, so nothing was removed "
    "(an older swipe is not touched either); use 'Undo last answer' on the review card"
)


def auto_review_label_source(kind: str) -> str:
    """The `labels.source` an answer to a pass of this kind is stored under."""
    return f"{AUTO_REVIEW_SOURCE_PREFIX}_{_require_auto_review_kind(kind)}"


def _require_auto_review_kind(kind: object) -> str:
    if kind not in AUTO_REVIEW_KINDS:
        raise ValueError(
            f"auto pass review kind must be one of {AUTO_REVIEW_KINDS!r} (got {kind!r})"
        )
    return str(kind)


def _require_text(value: object, name: str) -> str:
    """A non-empty string identity field.

    An archive keyed by an empty profile_id could never be found again by the review that has
    to answer it, so an empty/non-string identity is refused up front instead of stored.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"auto pass {name} must be a non-empty string (got {value!r})")
    return value


def _require_optional_app(app: object) -> str | None:
    return None if app is None else _require_text(app, "app")


def _require_window(window: object) -> int:
    if type(window) is not int or window <= 0:
        raise ValueError(
            f"auto review window must be a positive integer (got {window!r})"
        )
    return window


def require_auto_pass_retention(
    max_passes: object, max_age_days: object
) -> tuple[int | None, float | None]:
    """Normalise ``prune_auto_passes``' two bounds; ``None`` (or 0) means unbounded.

    Both backends share this so a typo in a retention knob raises the same ValueError
    everywhere instead of silently meaning "keep everything" in one of them.
    """
    kept: int | None = None
    if max_passes is not None:
        if type(max_passes) is not int or max_passes < 0:
            raise ValueError(
                f"auto pass retention max_passes must be a non-negative integer or None "
                f"(got {max_passes!r})"
            )
        kept = max_passes or None
    age: float | None = None
    if max_age_days is not None:
        if isinstance(max_age_days, bool) or not isinstance(max_age_days, numbers.Real):
            raise ValueError(
                f"auto pass retention max_age_days must be a finite number >= 0 or None "
                f"(got {max_age_days!r})"
            )
        try:
            age = float(max_age_days)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                f"auto pass retention max_age_days must be a finite number >= 0 or None "
                f"(got {max_age_days!r})"
            ) from exc
        if not math.isfinite(age) or age < 0:
            raise ValueError(
                f"auto pass retention max_age_days must be a finite number >= 0 or None "
                f"(got {max_age_days!r})"
            )
        age = age or None
    return kept, age


def doomed_auto_passes(
    passes: list[tuple[str, str, float]],
    max_passes: int | None,
    max_age_days: float | None,
    now: float,
) -> list[tuple[str, str]]:
    """The ``(app, profile_id)`` of every archived pass retention removes.

    ``passes`` is ``(app, profile_id, created_at epoch)``, any order.  A pass goes when it is
    older than ``max_age_days`` OR is not among the ``max_passes`` newest (oldest first), whether
    or not the owner has answered it: the photos and embedding of a stranger have no use once
    the answer (a label, which keeps its own embedding) is recorded or the pass has gone stale.
    """
    ordered = sorted(passes, key=lambda p: (p[2], p[1]), reverse=True)  # newest first
    horizon = None if max_age_days is None else now - max_age_days * 86400.0
    doomed = []
    for rank, (app, profile_id, created_at) in enumerate(ordered):
        too_many = max_passes is not None and rank >= max_passes
        too_old = horizon is not None and created_at < horizon
        if too_many or too_old:
            doomed.append((app, profile_id))
    return doomed


def _finite_float(value: object, name: str) -> float:
    # numbers.Real (not int/float) because the embedding and score arrive as numpy scalars
    # (np.float32 is not a float subclass); bool is excluded because True is not a score.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"auto pass {name} must be a finite number (got {value!r})")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            f"auto pass {name} must be a finite number (got {value!r})"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"auto pass {name} must be a finite number (got {value!r})")
    return number


def validate_auto_pass(
    *,
    run_id: object,
    app: object,
    profile_id: object,
    score: object,
    threshold: object,
    embedding: object,
    random_sample: object,
) -> dict:
    """Validate and normalise the scalar half of a pass archive; raises ValueError.

    Everything here is a CALLER bug rather than a transient failure, so it raises instead of
    returning False: the False return is reserved for "the photos could not be archived".
    ``random_sample`` must be a real bool -- it decides which pool the pass is reviewed in and
    therefore which statistic its answer feeds, so a truthy string must not slip through.
    """
    if type(random_sample) is not bool:
        raise ValueError(
            f"auto pass random_sample must be a bool (got {random_sample!r})"
        )
    if isinstance(embedding, (str, bytes)) or not hasattr(embedding, "__iter__"):
        raise ValueError("auto pass embedding must be a sequence of finite numbers")
    vector = [_finite_float(x, "embedding element") for x in embedding]
    if not vector:
        raise ValueError("auto pass embedding must not be empty")
    return {
        "run_id": _require_text(run_id, "run_id"),
        "app": _require_text(app, "app"),
        "profile_id": _require_text(profile_id, "profile_id"),
        "score": _finite_float(score, "score"),
        "threshold": _finite_float(threshold, "threshold"),
        "embedding": vector,
        "random_sample": random_sample,
    }


def clean_auto_pass_photos(photos: object, profile_id: str) -> list[bytes] | None:
    """The photo list to archive, or None (after a loud print) when nothing valid can be.

    A pass the owner cannot SEE cannot be judged, and an answer given without the screenshots
    would train the ranker on a guess. So one unusable photo spoils the whole archive: None
    means "do not record this pass", exactly like record_profile's short-photo-set refusal.
    """
    if photos is None:
        photos = []
    try:
        photo_list = list(photos)  # type: ignore[call-overload]
    except TypeError:
        print(
            f"Auto pass archive warning: photos for profile {profile_id} were not a list; "
            "skipping the pass archive."
        )
        return None
    if not photo_list:
        print(
            f"Auto pass archive warning: captured 0 photos for profile {profile_id}; "
            "skipping the pass archive."
        )
        return None
    out: list[bytes] = []
    for index, photo in enumerate(photo_list):
        if not isinstance(photo, (bytes, bytearray)) or not photo:
            print(
                f"Auto pass archive warning: photo {index} of profile {profile_id} is not a "
                "non-empty byte string; skipping the pass archive."
            )
            return None
        out.append(bytes(photo))
    return out


def photo_content_type(data: bytes) -> str:
    """Sniffed MIME type (same rules as bigquery_store._image_type, which needs the ext too)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return "application/octet-stream"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,
    liked INTEGER, source TEXT, embedding TEXT, photo_count INTEGER, profile_id TEXT
);
CREATE TABLE IF NOT EXISTS training_label_names (
    label_id INTEGER PRIMARY KEY, profile_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,
    decision TEXT, score REAL, source TEXT, profile_id TEXT
);
CREATE TABLE IF NOT EXISTS label_retractions (
    correction_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, app TEXT NOT NULL, source TEXT NOT NULL,
    profile_id TEXT NOT NULL, label_created_at REAL NOT NULL, decision_created_at REAL NOT NULL,
    decision_fingerprint TEXT NOT NULL, reason TEXT NOT NULL, evidence_ref TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS openers (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,
    model TEXT, opener TEXT, referenced TEXT, angle TEXT, item_description TEXT,
    profile_id TEXT, decision TEXT, decision_source TEXT, decision_created_at REAL,
    model_item_index INTEGER, prompt_sha256 TEXT, profile_key TEXT
);
CREATE TABLE IF NOT EXISTS opener_rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,
    model TEXT, attempt INTEGER, reason_code TEXT, reason TEXT, raw_opener TEXT,
    prompt_sha256 TEXT
);
-- The OUTCOME half of the measurement gap `openers.profile_key` exists to close: what a sent
-- opener actually PERFORMED (a match, a reply, silence), as opposed to everything else this
-- schema records about what the prompt PRODUCED. No run_id/foreign key to `openers` at all --
-- an outcome is observed on a real conversation, often well after any automation run ended, and
-- (app, profile_key) is the whole join key back to `openers` (see SQLiteStore.joined_opener_outcomes).
-- Brand-new table: CREATE TABLE IF NOT EXISTS alone reaches an existing db file with no ALTER
-- needed (unlike a new COLUMN on a table that already exists elsewhere -- see the two-part rule
-- documented above the ALTER loop in _initialize_schema, which is what a new column DOES need).
CREATE TABLE IF NOT EXISTS opener_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, app TEXT, profile_key TEXT, outcome TEXT,
    observed_at REAL, source TEXT, note TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS opener_retractions (
    correction_id TEXT NOT NULL, run_id TEXT NOT NULL, app TEXT NOT NULL,
    opener_created_at REAL NOT NULL, model TEXT NOT NULL, opener_fingerprint TEXT NOT NULL,
    reason TEXT NOT NULL, evidence_ref TEXT NOT NULL, created_at REAL NOT NULL,
    PRIMARY KEY (correction_id, opener_created_at)
);
-- AUTO pass archive + owner review (see the "auto pass review" helpers above). Brand-new
-- tables: CREATE ... IF NOT EXISTS alone reaches an existing db file, no ALTER needed. They are
-- deliberately NOT part of `labels`/`decisions` or the retraction machinery: an archived pass
-- is not a label until the owner answers it, and the answer is written through add_label.
-- (app, profile_id) is unique: profile_id is a per-card uuid, so a repeat is a bug, and the
-- UNIQUE index on reviews is what makes "one answer per pass" hold across processes.
CREATE TABLE IF NOT EXISTS auto_passes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, app TEXT NOT NULL,
    profile_id TEXT NOT NULL, created_at REAL NOT NULL, score REAL NOT NULL,
    threshold REAL NOT NULL, embedding TEXT NOT NULL, photo_count INTEGER NOT NULL,
    profile_name TEXT, random_sample INTEGER NOT NULL, capture_truncated INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS auto_passes_identity ON auto_passes (app, profile_id);
CREATE TABLE IF NOT EXISTS auto_pass_photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, profile_id TEXT,
    created_at REAL, photo_index INTEGER, gcs_uri TEXT, sha256 TEXT, byte_size INTEGER,
    content_type TEXT, photo BLOB NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS auto_pass_photos_identity
    ON auto_pass_photos (app, profile_id, photo_index);
CREATE TABLE IF NOT EXISTS auto_pass_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id TEXT NOT NULL, app TEXT NOT NULL,
    reviewed_at REAL NOT NULL, would_like INTEGER NOT NULL, kind TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS auto_pass_reviews_identity
    ON auto_pass_reviews (app, profile_id);
CREATE TABLE IF NOT EXISTS spend (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, created_at REAL, model TEXT,
    input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER,
    cache_write_tokens INTEGER, cost_usd REAL
);
"""

# The ONE predicate that answers "is this label retracted?" for this backend. Every read path
# that must hide a tombstoned label (load_labels, remove_latest_training_label, both release
# summaries, the opener cleanup advisory) substitutes this exact text, so a summary can never
# report a label that load_labels has already dropped from the training set.
#
# A tombstone is keyed to a label by (run_id, app, source, profile_id, label_created_at) against
# labels' (run_id, app, source, profile_id, created_at) -- correction_id identifies the CORRECTION,
# not its target, so it is deliberately not part of the join.
#
# profile_id is COALESCEd on BOTH sides because a label can legitimately carry NO profile
# identity, spelled two different ways: `labels.profile_id` was added by ALTER (see
# _initialize_schema), so rows written before that migration read back NULL, while add_label's
# own default writes ''. SQL's `NULL = <anything>` is NULL and never TRUE, so a NULL-profile
# label was invisible to EVERY tombstone -- permanently in the training set, removable but not
# retractable. Folding NULL and '' together makes "no profile identity" a single comparable
# value in the predicate WITHOUT rewriting a single stored row.
#
# Precision: the join still carries run_id/app/source AND label_created_at, so a tombstone with
# no profile identity hides only the label at its own timestamp, not every legacy label in the
# run. Residual, stated rather than papered over: nothing in this schema makes created_at unique,
# so two identity-less labels in one run/app/source sharing an exact created_at cannot be told
# apart here and one tombstone would hide both. There is no other column that could separate
# them; the correction planner independently refuses any run whose labels lack strict unique
# created_at ordering (retractions._strict), so the tools path cannot reach that case.
#
# The decisions half of a correction (see the release summaries below) keys on
# (run_id, app, source, decision_created_at) and never on profile_id, so it has no NULL exposure
# and is intentionally left as plain equality.
_LABEL_NOT_RETRACTED = (
    "NOT EXISTS (SELECT 1 FROM label_retractions r WHERE r.run_id=l.run_id AND r.app=l.app "
    "AND r.source=l.source AND COALESCE(r.profile_id,'')=COALESCE(l.profile_id,'') "
    "AND r.label_created_at=l.created_at)"
)


class SQLiteStore:
    """Local, offline, zero-dependency backend. Good default / fallback."""

    def __init__(self, db_file: str | Path):
        db_text = str(db_file)
        self._db_path: Path | None = None
        if db_text != ":memory:":
            db_path = Path(db_file)
            parent = db_path.parent
            # Only the configured leaf data directory is tightened. A bare filename means the
            # caller chose the current (potentially shared repository) directory, whose mode is
            # outside this store's authority; root is likewise never a chmod target.
            if parent != Path(".") and parent != Path(parent.anchor):
                ensure_private_dir(parent)
            else:
                parent.mkdir(parents=True, exist_ok=True)
            self._db_path = db_path
            # Refuse a planted DB or sidecar symlink before SQLite can follow it. Pre-creating a
            # new DB with O_EXCL closes the default-umask exposure window before sqlite3 opens it.
            self._tighten_private_files()
            if not db_path.exists():
                write_private_bytes(db_path, b"", parent=parent)
            else:
                tighten_private_file(db_path, parent=parent)
        # check_same_thread=False + a lock: safe to share across worker threads.
        self.con = sqlite3.connect(db_text, check_same_thread=False)
        try:
            self._initialize_schema()
        except BaseException:
            # A failed migration/privacy-sidecar check must not leave a live connection (and
            # therefore SQLite file handles) behind. Closing is cleanup only; retain the actual
            # initialization failure if a test double or damaged driver also fails to close.
            try:
                self.con.close()
            except Exception:  # noqa: BLE001
                pass
            raise
        self._lock = threading.Lock()

    def _initialize_schema(self) -> None:
        """Create/migrate the schema after connect; caller owns failure cleanup."""
        self._tighten_private_files()
        self.con.executescript(_SCHEMA)
        try:
            self.con.execute("ALTER TABLE decisions ADD COLUMN source TEXT")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
        try:
            self.con.execute("ALTER TABLE decisions ADD COLUMN profile_id TEXT")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
        try:
            self.con.execute("ALTER TABLE labels ADD COLUMN profile_id TEXT")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
        # openers.angle: the two statements above are BOTH needed for every added column, and
        # for the same reason — CREATE TABLE IF NOT EXISTS is a no-op against a db file that
        # already has the table, so the _SCHEMA entry only ever reaches a FRESH database and
        # the ALTER is what carries the column into an existing one. Same shape as the two
        # migrations above: swallow only "duplicate column name" (the already-migrated case)
        # and re-raise anything else rather than starting up with a schema we can't write to.
        try:
            self.con.execute("ALTER TABLE openers ADD COLUMN angle TEXT")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
        # openers.item_description: the model's own short description of the ITEM it picked
        # (ops/OPENER-REDESIGN.md 5.7 -- auto logs it, observe displays it). Same two-part
        # migration as every column above, for the same reason spelled out there: the _SCHEMA
        # line only reaches a FRESH database, this ALTER is what carries the column into one
        # that already exists.
        try:
            self.con.execute("ALTER TABLE openers ADD COLUMN item_description TEXT")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
        # prompt_sha256 rides in this loop for openers (see prompt_stamp in opener/opener.py):
        # the digest of the prompt era that produced the row, so calibration groups by era
        # instead of reconstructing boundaries from created_at against git dates. Nullable and
        # never backfilled -- a NULL here means "written before 2026-09-05 (b)".
        #
        # profile_key (2026-09-06) rides in the same loop for the same structural reason: the
        # STABLE cross-time attribution key (ranker/profile_key.py) that lets an outcome
        # observed days later join back to the opener that earned it (see `openers.profile_id`,
        # unrelated and unchanged, for the per-card lineage id this is NOT). Nullable and never
        # backfilled -- a NULL here means "written before this column existed" (distinct from a
        # post-migration row that legitimately had no derivable key, which writes "").
        for column, decl in (
            ("profile_id", "TEXT"),
            ("decision", "TEXT"),
            ("decision_source", "TEXT"),
            ("decision_created_at", "REAL"),
            ("model_item_index", "INTEGER"),
            ("prompt_sha256", "TEXT"),
            ("profile_key", "TEXT"),
        ):
            try:
                self.con.execute(f"ALTER TABLE openers ADD COLUMN {column} {decl}")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
        # opener_rejections.prompt_sha256: the FIRST column this table has ever gained, so it
        # gets its own stanza rather than joining a loop. Both statements are needed for the
        # same reason spelled out for openers.angle above -- CREATE TABLE IF NOT EXISTS is a
        # no-op against a db file that already has the table, so the _SCHEMA entry only ever
        # reaches a FRESH database and this ALTER is what carries the column into an existing
        # one. Swallow only "duplicate column name" (the already-migrated case) and re-raise
        # anything else rather than starting up with a schema we can't write to.
        try:
            self.con.execute(
                "ALTER TABLE opener_rejections ADD COLUMN prompt_sha256 TEXT"
            )
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
        self._commit()

    def _private_paths(self) -> tuple[Path, ...]:
        if self._db_path is None:
            return ()
        text = str(self._db_path)
        return (
            self._db_path,
            *(Path(text + suffix) for suffix in ("-wal", "-shm", "-journal")),
        )

    def _tighten_private_files(self) -> None:
        """Tighten the DB and any live SQLite sidecars, rejecting symlink leaves."""
        for path in self._private_paths():
            tighten_private_file(path, parent=path.parent, missing_ok=True)

    def _commit(self) -> None:
        self.con.commit()
        self._tighten_private_files()

    def load_labels(self) -> list[tuple[bool, list[float]]]:
        # A projection of load_label_rows, never a second query: the two must return the
        # same rows in the same order (same retraction filter) or a caller zipping them
        # against each other would silently mislabel.
        return [(liked, emb) for liked, emb, _source in self.load_label_rows()]

    def load_label_rows(self) -> list[tuple[bool, list[float], str]]:
        """Exactly the rows load_labels() returns, plus each label's `source` ("" for NULL).

        The source is what lets the ranker split MANUAL calibration labels from auto-review
        answers (ranker/threshold.py): a review answer is training data but must never set the
        recall threshold it is also used to audit.
        """
        with self._lock:
            rows = self.con.execute(
                "SELECT l.liked, l.embedding, l.source FROM labels l "
                f"WHERE {_LABEL_NOT_RETRACTED}"
            ).fetchall()
        return [
            (bool(liked), json.loads(emb), str(source or ""))
            for liked, emb, source in rows
        ]

    def _visible_openers_count(self, run_id: str, app: str) -> int:
        """Count non-tombstoned openers, retaining compatibility with old test/local schemas."""
        columns = {row[1] for row in self.con.execute("PRAGMA table_info(openers)")}
        if "created_at" not in columns:
            return int(
                self.con.execute(
                    "SELECT COUNT(*) FROM openers WHERE run_id=? AND app=?",
                    (run_id, app),
                ).fetchone()[0]
            )
        return int(
            self.con.execute(
                """SELECT COUNT(*) FROM openers AS o WHERE o.run_id=? AND o.app=?
            AND NOT EXISTS (SELECT 1 FROM opener_retractions AS r WHERE r.run_id=o.run_id
            AND r.app=o.app AND r.opener_created_at=o.created_at)""",
                (run_id, app),
            ).fetchone()[0]
        )

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
    ) -> bool:
        # SQLite is the offline, labels-only fallback: it doesn't archive images,
        # so there's nothing that can fail here — always "recorded". capture_truncated
        # is accepted-and-ignored for the same reason: there's no profiles manifest
        # row here to hang it off of (see BigQueryStore.record_profile, the system of
        # record, for where "was this label made from an incomplete profile read?"
        # is actually kept queryable).
        return True

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
        with self._lock:
            self._insert_label_locked(
                run_id,
                app,
                liked,
                embedding,
                source,
                photo_count,
                profile_id,
                profile_name,
            )
            self._commit()

    def _insert_label_locked(
        self,
        run_id,
        app,
        liked,
        embedding,
        source,
        photo_count,
        profile_id,
        profile_name,
    ) -> None:
        """INSERT one label (+ its display-name row) WITHOUT committing.

        Caller holds ``self._lock`` and owns the commit/rollback. Split out of add_label so
        record_auto_pass_review can put the label and its review row in ONE transaction.
        """
        self.con.execute(
            "INSERT INTO labels (run_id, app, created_at, liked, source, embedding,"
            " photo_count, profile_id) VALUES (?,?,?,?,?,?,?,?)",
            (
                run_id,
                app,
                time.time(),
                int(liked),
                source,
                json.dumps(embedding, allow_nan=False),
                photo_count,
                profile_id,
            ),
        )
        label_id = self.con.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.con.execute(
            "INSERT INTO training_label_names (label_id, profile_name) VALUES (?,?)",
            (label_id, str(profile_name or "")),
        )

    def remove_latest_training_label(self) -> dict | None:
        """Delete and identify the newest currently-active training label.

        When that newest label is an answer to an archived AUTO pass (source
        ``auto_review...``) this REFUSES with ``ValueError`` instead of deleting anything:
        deleting the answer would leave its ``auto_pass_reviews`` row marked answered with no
        label behind it, and reaching past it to delete the newest older MANUAL label (what an
        earlier version did) silently destroyed an unrelated swipe while the mistaken answer
        stayed in the training set.  ``undo_latest_auto_review`` is the undo for an answer.
        """
        with self._lock:
            row = self.con.execute(
                "SELECT l.id, n.profile_name, l.profile_id, l.source FROM labels AS l "
                "LEFT JOIN training_label_names AS n ON n.label_id=l.id "
                f"WHERE {_LABEL_NOT_RETRACTED} "
                "ORDER BY l.created_at DESC, l.id DESC LIMIT 1",
            ).fetchone()
            if row is None:
                return None
            if str(row[3] or "").startswith(AUTO_REVIEW_SOURCE_PREFIX):
                raise ValueError(REVIEW_ANSWER_IS_LATEST_MESSAGE)
            self.con.execute("DELETE FROM labels WHERE id=?", (row[0],))
            self.con.execute(
                "DELETE FROM training_label_names WHERE label_id=?", (row[0],)
            )
            self._commit()
        return {"profile_name": str(row[1] or ""), "profile_id": str(row[2] or "")}

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
        timestamp = (
            time.time()
            if created_at is None
            else normalize_timestamp_epoch(created_at, label="created_at")
        )
        with self._lock:
            self.con.execute(
                "INSERT INTO decisions (run_id, app, created_at, decision, score, source, profile_id) "
                "VALUES (?,?,?,?,?,?,?)",
                (run_id, app, timestamp, decision, score, source, profile_id),
            )
            self._commit()

    def retraction_run_rows(self, run_id: str, app: str, source: str) -> dict:
        """Exact minimal run rows used by the correction planner; no embeddings/photos."""
        with self._lock:
            labels = self.con.execute(
                """SELECT profile_id, created_at, liked
                FROM labels WHERE run_id=? AND app=? AND source=? ORDER BY created_at, id""",
                (run_id, app, source),
            ).fetchall()
            decisions = self.con.execute(
                """SELECT profile_id, created_at, decision, score
                FROM decisions WHERE run_id=? AND app=? AND source=? ORDER BY created_at, id""",
                (run_id, app, source),
            ).fetchall()
            retractions = self.con.execute(
                """SELECT correction_id, profile_id, label_created_at,
                decision_created_at, decision_fingerprint FROM label_retractions
                WHERE run_id=? AND app=? AND source=? ORDER BY created_at, correction_id""",
                (run_id, app, source),
            ).fetchall()

        def exact_epoch(value):
            """A decimal spelling which round-trips SQLite's stored IEEE-754 REAL exactly.

            Retraction predicates use equality, by design.  An ISO timestamp has only
            microsecond precision and can silently leave a label live when `time.time()`
            supplied sub-microsecond data.  ``.17g`` is the shortest safe portable
            precision for a binary64 round-trip; retractions._timestamp / append parser
            intentionally accept this SQLite-only planner representation.
            """
            return format(float(value), ".17g")

        return {
            "run_id": run_id,
            "app": app,
            "source": source,
            "profiles": [],
            "labels": [
                {"profile_id": p, "created_at": exact_epoch(t), "liked": bool(liked)}
                for p, t, liked in labels
            ],
            "decisions": [
                (
                    {"created_at": exact_epoch(t), "decision": d, "score": s}
                    | ({"profile_id": p} if p not in {None, ""} else {})
                )
                for p, t, d, s in decisions
            ],
            "retractions": [
                {
                    "correction_id": c,
                    "profile_id": p,
                    "label_created_at": exact_epoch(label_time),
                    "decision_created_at": exact_epoch(d),
                    "decision_fingerprint": f,
                }
                for c, p, label_time, d, f in retractions
            ],
        }

    def append_label_retraction(self, row: dict) -> bool:
        """Append one correction using bound values.

        A repeat of the exact correction id is a no-op.  A different correction aimed at
        the same label/legacy-decision pair is refused instead of creating competing audit
        records; the planner performs the same check, while this closes the check/append
        window for another local process.
        """
        from .retractions import RetractionRefused

        try:
            values = (
                row["correction_id"],
                row["run_id"],
                row["app"],
                row["source"],
                row["profile_id"],
                normalize_timestamp_epoch(
                    row["label_created_at"],
                    label="label_created_at",
                    allow_epoch_text=True,
                ),
                normalize_timestamp_epoch(
                    row["decision_created_at"],
                    label="decision_created_at",
                    allow_epoch_text=True,
                ),
                row["decision_fingerprint"],
                row["reason"],
                row["evidence_ref"],
                time.time(),
            )
        except (KeyError, ValueError) as exc:
            raise RetractionRefused("SQLite correction timestamps are invalid") from exc
        with self._lock:
            existing = self.con.execute(
                """SELECT correction_id FROM label_retractions
                WHERE run_id=? AND app=? AND source=? AND profile_id=?
                AND label_created_at=? AND decision_created_at=?""",
                values[1:7],
            ).fetchone()
            if existing:
                if existing[0] == row["correction_id"]:
                    return False
                raise RetractionRefused(
                    "target label/decision pair already has a different retraction"
                )
            cur = self.con.execute(
                """INSERT OR IGNORE INTO label_retractions
                (correction_id,run_id,app,source,profile_id,label_created_at,decision_created_at,
                decision_fingerprint,reason,evidence_ref,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                values,
            )
            self._commit()
        return cur.rowcount == 1

    def observe_release_persistence_summary(
        self, run_id: str, app: str
    ) -> dict[str, int]:
        """Read-only run-scoped counts for the Hinge AUTO release verifier.

        Return counts only — no embeddings, profile ids, or opener text leave the configured
        store.  The release evidence is deliberately outcome-specific: a complete supervised
        cycle must have persisted both a manual PASS and a manual LIKE, as both labels and
        decisions, plus a successful Hinge opener.  Do not replace these with broad totals:
        one LIKE duplicated twice is not evidence that the pass half of the cycle survived.

        Worker persists a manual pass decision as ``dislike`` (the historical storage term),
        while its label is ``liked=0``.  The public aggregate field calls this a ``pass`` so
        release policy stays aligned with the operator-visible action.
        """
        with self._lock:
            pass_labels = self.con.execute(
                "SELECT COUNT(*) FROM labels l WHERE run_id=? AND app=? AND source='manual' "
                f"AND liked=0 AND {_LABEL_NOT_RETRACTED}",
                (run_id, app),
            ).fetchone()[0]
            like_labels = self.con.execute(
                "SELECT COUNT(*) FROM labels l WHERE run_id=? AND app=? AND source='manual' "
                f"AND liked=1 AND {_LABEL_NOT_RETRACTED}",
                (run_id, app),
            ).fetchone()[0]
            pass_decisions = self.con.execute(
                """SELECT COUNT(*) FROM decisions d WHERE run_id=? AND app=? AND source='manual' AND decision='dislike'
                AND NOT EXISTS (SELECT 1 FROM label_retractions r WHERE r.run_id=d.run_id AND r.app=d.app
                AND r.source=d.source AND r.decision_created_at=d.created_at)""",
                (run_id, app),
            ).fetchone()[0]
            like_decisions = self.con.execute(
                """SELECT COUNT(*) FROM decisions d WHERE run_id=? AND app=? AND source='manual' AND decision='like'
                AND NOT EXISTS (SELECT 1 FROM label_retractions r WHERE r.run_id=d.run_id AND r.app=d.app
                AND r.source=d.source AND r.decision_created_at=d.created_at)""",
                (run_id, app),
            ).fetchone()[0]
            openers = self._visible_openers_count(run_id, app)
        return {
            "manual_pass_labels": int(pass_labels),
            "manual_like_labels": int(like_labels),
            "manual_pass_decisions": int(pass_decisions),
            "manual_like_decisions": int(like_decisions),
            "successful_hinge_openers": int(openers),
        }

    def ai_observe_release_persistence_summary(
        self, run_id: str, app: str, source: str
    ) -> dict[str, int]:
        """Read-only complete-cycle counts for a declared non-manual release source."""
        if source not in {"external_ai_review", "automation"}:
            raise ValueError(
                "AI observe release source must be external_ai_review or automation"
            )
        with self._lock:
            pass_labels = self.con.execute(
                "SELECT COUNT(*) FROM labels l WHERE run_id=? AND app=? AND source=? "
                f"AND liked=0 AND {_LABEL_NOT_RETRACTED}",
                (run_id, app, source),
            ).fetchone()[0]
            like_labels = self.con.execute(
                "SELECT COUNT(*) FROM labels l WHERE run_id=? AND app=? AND source=? "
                f"AND liked=1 AND {_LABEL_NOT_RETRACTED}",
                (run_id, app, source),
            ).fetchone()[0]
            pass_decisions = self.con.execute(
                """SELECT COUNT(*) FROM decisions d WHERE run_id=? AND app=? AND source=? AND decision='dislike'
                AND NOT EXISTS (SELECT 1 FROM label_retractions r WHERE r.run_id=d.run_id AND r.app=d.app
                AND r.source=d.source AND r.decision_created_at=d.created_at)""",
                (run_id, app, source),
            ).fetchone()[0]
            like_decisions = self.con.execute(
                """SELECT COUNT(*) FROM decisions d WHERE run_id=? AND app=? AND source=? AND decision='like'
                AND NOT EXISTS (SELECT 1 FROM label_retractions r WHERE r.run_id=d.run_id AND r.app=d.app
                AND r.source=d.source AND r.decision_created_at=d.created_at)""",
                (run_id, app, source),
            ).fetchone()[0]
            openers = self._visible_openers_count(run_id, app)
        return {
            "ai_pass_labels": int(pass_labels),
            "ai_like_labels": int(like_labels),
            "ai_pass_decisions": int(pass_decisions),
            "ai_like_decisions": int(like_decisions),
            "successful_hinge_openers": int(openers),
        }

    def advisory_opener_run_rows(self, run_id: str, app: str) -> dict:
        """Exact no-text opener identities plus effective Like/Pass proof for cleanup planning."""
        from .retractions import canonical_sha

        with self._lock:
            raw = self.con.execute(
                "SELECT created_at,model,opener FROM openers WHERE run_id=? AND app=? "
                "ORDER BY created_at",
                (run_id, app),
            ).fetchall()
            existing = self.con.execute(
                "SELECT correction_id,opener_created_at,model,opener_fingerprint,reason,evidence_ref "
                "FROM opener_retractions "
                "WHERE run_id=? AND app=?",
                (run_id, app),
            ).fetchall()

            def count(table):
                return int(
                    self.con.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE run_id=? AND app=?",
                        (run_id, app),
                    ).fetchone()[0]
                )

            def effective_labels(liked: int):
                # An answer to an archived AUTO pass is stored under the pass's own run_id with
                # source `auto_review_*`.  It is not a swipe the owner made in that run, so it
                # is not "Like evidence" for it: counting it would let one tap on the review
                # card make the cleanup planner refuse a run it had nothing to say about.
                return int(
                    self.con.execute(
                        "SELECT COUNT(*) FROM labels l WHERE run_id=? AND app=? AND liked=? "
                        f"AND {_LABEL_NOT_RETRACTED} "
                        "AND substr(COALESCE(l.source,''), 1, ?) <> ?",
                        (
                            run_id,
                            app,
                            liked,
                            len(AUTO_REVIEW_SOURCE_PREFIX),
                            AUTO_REVIEW_SOURCE_PREFIX,
                        ),
                    ).fetchone()[0]
                )

            def decisions(decision: str):
                return int(
                    self.con.execute(
                        "SELECT COUNT(*) FROM decisions WHERE run_id=? AND app=? AND decision=?",
                        (run_id, app, decision),
                    ).fetchone()[0]
                )

            # Every one of these closures touches self.con, so they are CALLED here rather than
            # from the returned dict below: this class shares one connection across threads on
            # check_same_thread=False + this lock (see __init__), and a query issued after the
            # release would be exactly the unsynchronized use that combination forbids. Only the
            # pure-CPU fingerprinting below is done outside.
            decision_rows = count("decisions")
            label_rows = count("labels")
            like_labels, pass_labels = effective_labels(1), effective_labels(0)
            like_decisions, pass_decisions = decisions("like"), decisions("dislike")
        openers = [
            {
                "created_at": format(float(created), ".17g"),
                "model": str(model),
                "opener_fingerprint": canonical_sha(
                    {
                        "run_id": run_id,
                        "app": app,
                        "created_at": format(float(created), ".17g"),
                        "model": str(model),
                        "opener": str(opener),
                    }
                ),
            }
            for created, model, opener in raw
        ]
        return {
            "run_id": run_id,
            "app": app,
            "openers": openers,
            "decisions": [{} for _ in range(decision_rows)],
            # SQLite is explicitly labels-only: profiles/photos are never archived here.
            "preference_counts": {
                "profiles": 0,
                "profile_photos": 0,
                "labels": label_rows,
                "decisions": decision_rows,
            },
            "effective_counts": {
                "like_labels": like_labels,
                "like_decisions": like_decisions,
                "pass_labels": pass_labels,
                "pass_decisions": pass_decisions,
            },
            "retractions": [
                {
                    "correction_id": row[0],
                    "opener_created_at": format(float(row[1]), ".17g"),
                    "model": row[2],
                    "opener_fingerprint": row[3],
                    "reason": row[4],
                    "evidence_ref": row[5],
                }
                for row in existing
            ],
        }

    def append_opener_retraction(self, row: dict) -> bool:
        from .retractions import RetractionRefused, canonical_sha

        try:
            opener_at = normalize_timestamp_epoch(
                row["opener_created_at"],
                label="opener_created_at",
                allow_epoch_text=True,
            )
            values = (
                row["correction_id"],
                row["run_id"],
                row["app"],
                opener_at,
                row["model"],
                row["opener_fingerprint"],
                row["reason"],
                row["evidence_ref"],
                time.time(),
            )
        except (KeyError, ValueError) as exc:
            raise RetractionRefused(
                "SQLite opener correction timestamp is invalid"
            ) from exc
        with self._lock:
            source = self.con.execute(
                "SELECT model,opener FROM openers WHERE run_id=? AND app=? AND created_at=?",
                (row["run_id"], row["app"], opener_at),
            ).fetchall()
            if len(source) != 1 or str(source[0][0]) != row["model"]:
                raise RetractionRefused(
                    "target opener row is missing or does not match its cleanup plan"
                )
            fingerprint = canonical_sha(
                {
                    "run_id": row["run_id"],
                    "app": row["app"],
                    "created_at": format(opener_at, ".17g"),
                    "model": str(source[0][0]),
                    "opener": str(source[0][1]),
                }
            )
            if fingerprint != row["opener_fingerprint"]:
                raise RetractionRefused(
                    "target opener fingerprint no longer matches its cleanup plan"
                )
            existing = self.con.execute(
                "SELECT correction_id,model,opener_fingerprint,reason,evidence_ref FROM opener_retractions "
                "WHERE run_id=? AND app=? AND opener_created_at=?",
                (row["run_id"], row["app"], opener_at),
            ).fetchone()
            if existing:
                actual = {
                    "correction_id": existing[0],
                    "model": existing[1],
                    "opener_fingerprint": existing[2],
                    "reason": existing[3],
                    "evidence_ref": existing[4],
                }
                expected = {key: row[key] for key in actual}
                if actual == expected:
                    return False
                raise RetractionRefused(
                    "opener row already has a different cleanup tombstone"
                )
            cur = self.con.execute(
                "INSERT OR IGNORE INTO opener_retractions "
                "(correction_id,run_id,app,opener_created_at,model,opener_fingerprint,reason,evidence_ref,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                values,
            )
            self._commit()
        return cur.rowcount == 1

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
        # `angle` is the model's own free-text label for what this opener is doing. Telemetry
        # only — nothing reads it back at runtime; it's here so the question "which opener
        # shapes correlate with matches" becomes answerable offline later. Defaulted so
        # callers that predate it (and any store used positionally) still work.
        #
        # `item_description` is the model's own short description of the ITEM it chose to write
        # about and to like (ops/OPENER-REDESIGN.md 5.7). Also telemetry here, and distinct
        # from `referenced`: that column holds the DETAIL the opener reacts to, this one holds
        # what the item IS. Persisted in both modes so a wrong-item report can later be checked
        # against what the model believed it picked. Same trailing-default rule as `angle`.
        #
        # `prompt_sha256` is the digest of the prompt era this row was generated under (see
        # prompt_stamp in opener/opener.py). Telemetry too, and keyword-only with a None
        # default: a caller that predates it writes NULL, which reads as "predates the stamp",
        # not as "no era".
        #
        # `profile_key` is the STABLE cross-time attribution key (ranker/profile_key.py) --
        # unrelated to `profile_id`'s per-card lineage above. Keyword-only with a "" default,
        # matching `profile_id`'s own convention: "" means "no key could be derived for this
        # card," NULL (a caller that predates the column entirely) means "this row predates
        # profile_key," and only the paired ALTER in _initialize_schema produces the latter.
        timestamp = (
            None
            if decision_created_at is None
            else normalize_timestamp_epoch(
                decision_created_at, label="decision_created_at"
            )
        )
        with self._lock:
            self.con.execute(
                "INSERT INTO openers (run_id, app, created_at, model, opener, referenced, angle,"
                " item_description, profile_id, decision, decision_source, decision_created_at,"
                " model_item_index, prompt_sha256, profile_key) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    app,
                    time.time(),
                    model,
                    opener,
                    referenced,
                    angle,
                    item_description,
                    profile_id,
                    decision,
                    decision_source,
                    timestamp,
                    model_item_index,
                    prompt_sha256,
                    profile_key,
                ),
            )
            self._commit()

    def record_opener_outcome(
        self,
        app: str,
        profile_key: str,
        outcome: str,
        *,
        observed_at: object | None = None,
        source: str = "owner",
        note: str = "",
    ) -> None:
        # See ranker/__init__.py's Store.record_opener_outcome for the full design rationale
        # (why there is no run_id, why profile_key is accepted unvalidated including "",
        # why observed_at is distinct from this row's own created_at). `outcome`/`source` are
        # stored exactly as given -- free text against a documented, unenforced vocabulary
        # (see ranker/__init__.py's KNOWN_OPENER_OUTCOMES / KNOWN_OPENER_OUTCOME_SOURCES).
        timestamp = (
            time.time()
            if observed_at is None
            else normalize_timestamp_epoch(observed_at, label="observed_at")
        )
        with self._lock:
            self.con.execute(
                "INSERT INTO opener_outcomes (app, profile_key, outcome, observed_at, source,"
                " note, created_at) VALUES (?,?,?,?,?,?,?)",
                (app, profile_key, outcome, timestamp, source, note, time.time()),
            )
            self._commit()

    def joined_opener_outcomes(
        self, app: str, *, prompt_sha256: str | None = None
    ) -> list[dict]:
        # The join predicate excludes '' and NULL profile_key on BOTH sides -- see
        # ranker/__init__.py's Store.joined_opener_outcomes for why two unattributable rows
        # must never join to EACH OTHER just because both happen to carry the same "no key"
        # spelling. prompt_sha256=None means "no filter" (every era), never "match NULL" --
        # same convention as every other prompt_sha256 consumer in this store.
        #
        # `o.decision = 'like'` keeps only openers that were actually SENT: 'like' is the ONE
        # decision value that has ever meant "this opener reached the device" (literally
        # tools/opener_outcome_recorder.py's DECISION_SENT). Without it, a profile that was
        # drafted-then-DISLIKED in one run and LIKED in a later one returns the owner's single
        # recorded match TWICE -- crediting the earlier prompt era with a match earned by a
        # draft that was never sent. 'dislike'/'never_sent', 'synthetic_replay', and NULL/''
        # (predates decision tracking) all fail this predicate, which is the point: none of
        # them proves a send, and a read this feeds must never guess one.
        query = (
            "SELECT o.run_id, o.model, o.opener, o.prompt_sha256, o.profile_key, o.created_at,"
            " oc.outcome, oc.observed_at, oc.source, oc.note, oc.created_at"
            " FROM openers o JOIN opener_outcomes oc"
            " ON oc.app = o.app AND oc.profile_key = o.profile_key"
            " WHERE o.app = ? AND o.profile_key IS NOT NULL AND o.profile_key != ''"
            " AND oc.profile_key IS NOT NULL AND oc.profile_key != ''"
            " AND o.decision = 'like'"
        )
        params: list = [app]
        if prompt_sha256 is not None:
            query += " AND o.prompt_sha256 = ?"
            params.append(prompt_sha256)
        query += " ORDER BY oc.created_at, oc.id"
        with self._lock:
            rows = self.con.execute(query, params).fetchall()
        return [
            {
                "run_id": r[0],
                "app": app,
                "model": r[1],
                "opener": r[2],
                "prompt_sha256": r[3],
                "profile_key": r[4],
                "opener_created_at": r[5],
                "outcome": r[6],
                "observed_at": r[7],
                "source": r[8],
                "note": r[9],
                "outcome_created_at": r[10],
            }
            for r in rows
        ]

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
        # `prompt_sha256`: same era digest the successes carry (see record_opener above), so a
        # guard's firing rate is attributable to the prompt that provoked it. Keyword-only, so
        # the seven existing parameters stay positional for every caller and test double.
        with self._lock:
            self.con.execute(
                "INSERT INTO opener_rejections (run_id, app, created_at, model, attempt,"
                " reason_code, reason, raw_opener, prompt_sha256)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    app,
                    time.time(),
                    model,
                    attempt,
                    reason_code,
                    reason,
                    raw_opener,
                    prompt_sha256,
                ),
            )
            self._commit()

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
        """Archive one AUTO pass (manifest + the exact screenshots the model saw).

        Never writes a label, a decision, or a `profiles`/`profile_photos` row: the training
        archive is what retraction planning reasons about, and an unanswered pass is not
        training data. Returns False (after a loud print) when the photos cannot be archived or
        the pass was already archived; raises ValueError for malformed arguments. The manifest
        and every photo commit in ONE transaction, so a crash can never leave a pass the owner
        is shown without its screenshots.
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
        photo_list = clean_auto_pass_photos(photos, args["profile_id"])
        if photo_list is None:
            return False
        embedding_json = json.dumps(args["embedding"], allow_nan=False)
        created_at = time.time()
        with self._lock:
            try:
                self.con.execute(
                    "INSERT INTO auto_passes (run_id, app, profile_id, created_at, score,"
                    " threshold, embedding, photo_count, profile_name, random_sample,"
                    " capture_truncated) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        args["run_id"],
                        args["app"],
                        args["profile_id"],
                        created_at,
                        args["score"],
                        args["threshold"],
                        embedding_json,
                        len(photo_list),
                        str(profile_name or ""),
                        int(args["random_sample"]),
                        int(bool(capture_truncated)),
                    ),
                )
                for index, photo in enumerate(photo_list):
                    self.con.execute(
                        "INSERT INTO auto_pass_photos (run_id, app, profile_id, created_at,"
                        " photo_index, gcs_uri, sha256, byte_size, content_type, photo)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (
                            args["run_id"],
                            args["app"],
                            args["profile_id"],
                            created_at,
                            index,
                            "",  # no object store on this backend; the bytes live in `photo`
                            hashlib.sha256(photo).hexdigest(),
                            len(photo),
                            photo_content_type(photo),
                            photo,
                        ),
                    )
                self._commit()
            except sqlite3.IntegrityError:
                self.con.rollback()
                print(
                    f"Auto pass archive warning: profile {args['profile_id']} ({args['app']}) "
                    "was already archived; not writing it again."
                )
                return False
            except BaseException:
                self.con.rollback()
                raise
        return True

    def next_auto_pass_for_review(self, *, kind, app=None) -> dict | None:
        """One unreviewed pass of ``kind``, or None when that pool is empty.

        ``random``: uniform among the random_sample passes (the unbiased sample).
        ``near_cut``: among the rest, the one closest below its own threshold, i.e. the
        largest (score - threshold) -- the answers that teach the boundary the most.
        """
        kind = _require_auto_review_kind(kind)
        app = _require_optional_app(app)
        where = [
            "p.random_sample = ?",
            "NOT EXISTS (SELECT 1 FROM auto_pass_reviews r "
            "WHERE r.app = p.app AND r.profile_id = p.profile_id)",
        ]
        params: list = [1 if kind == AUTO_REVIEW_KIND_RANDOM else 0]
        if app is not None:
            where.append("p.app = ?")
            params.append(app)
        order = (
            "RANDOM()"
            if kind == AUTO_REVIEW_KIND_RANDOM
            else "(p.score - p.threshold) DESC, p.created_at, p.id"
        )
        with self._lock:
            row = self.con.execute(
                "SELECT p.profile_id, p.app, p.run_id, p.created_at, p.score, p.threshold,"
                " p.photo_count, p.profile_name FROM auto_passes p WHERE "
                + " AND ".join(where)
                + f" ORDER BY {order} LIMIT 1",
                params,
            ).fetchone()
        if row is None:
            return None
        return {
            "profile_id": row[0],
            "app": row[1],
            "run_id": row[2],
            "created_at": float(row[3]),
            "score": float(row[4]),
            "threshold": float(row[5]),
            "photo_count": int(row[6]),
            "profile_name": str(row[7] or ""),
            "kind": kind,
        }

    def auto_pass_photo(self, *, profile_id, index) -> bytes | None:
        """The stored PNG bytes of one archived pass photo, or None when there is none."""
        _require_text(profile_id, "profile_id")
        if type(index) is not int or index < 0:
            raise ValueError(
                f"auto pass photo index must be a non-negative integer (got {index!r})"
            )
        with self._lock:
            row = self.con.execute(
                "SELECT photo FROM auto_pass_photos WHERE profile_id=? AND photo_index=?"
                " ORDER BY id LIMIT 1",
                (profile_id, index),
            ).fetchone()
        return None if row is None else bytes(row[0])

    def record_auto_pass_review(self, *, profile_id, would_like) -> dict:
        """Record the owner's answer to one archived pass: a review row AND a training label.

        Raises ValueError for an unknown, ambiguous or already-reviewed pass (an answer is
        final: a second tap must not become a second label). The label and the review row
        commit in ONE transaction -- a crash leaves neither, never a review without its label
        or a label without its review.
        """
        _require_text(profile_id, "profile_id")
        if type(would_like) is not bool:
            raise ValueError(f"would_like must be a bool (got {would_like!r})")
        with self._lock:
            passes = self.con.execute(
                "SELECT run_id, app, embedding, photo_count, profile_name, random_sample"
                " FROM auto_passes WHERE profile_id=?",
                (profile_id,),
            ).fetchall()
            if not passes:
                raise ValueError(f"no archived auto pass for profile {profile_id!r}")
            if len(passes) > 1:
                raise ValueError(
                    f"profile {profile_id!r} matches {len(passes)} archived auto passes; "
                    "refusing to guess which one is being answered"
                )
            run_id, app, embedding_json, photo_count, profile_name, random_sample = (
                passes[0]
            )
            if random_sample not in (0, 1):
                raise ValueError(
                    f"archived auto pass {profile_id!r} has no valid random_sample flag"
                )
            kind = (
                AUTO_REVIEW_KIND_RANDOM if random_sample else AUTO_REVIEW_KIND_NEAR_CUT
            )
            source = auto_review_label_source(kind)
            if self.con.execute(
                "SELECT 1 FROM auto_pass_reviews WHERE app=? AND profile_id=?",
                (app, profile_id),
            ).fetchone():
                raise ValueError(f"auto pass {profile_id!r} was already reviewed")
            try:
                self._insert_label_locked(
                    run_id,
                    app,
                    would_like,
                    json.loads(embedding_json),
                    source,
                    int(photo_count),
                    profile_id,
                    profile_name,
                )
                self.con.execute(
                    "INSERT INTO auto_pass_reviews (profile_id, app, reviewed_at,"
                    " would_like, kind) VALUES (?,?,?,?,?)",
                    (profile_id, app, time.time(), int(would_like), kind),
                )
                self._commit()
            except sqlite3.IntegrityError as exc:
                # Another process answered between our check and our insert; the UNIQUE
                # index is the authority. Roll the label back with it.
                self.con.rollback()
                raise ValueError(
                    f"auto pass {profile_id!r} was already reviewed"
                ) from exc
            except BaseException:
                self.con.rollback()
                raise
        return {"kind": kind, "label_source": source}

    def undo_latest_auto_review(self) -> dict | None:
        """Undo the newest answer to an archived pass: delete its label AND its review row.

        The pass goes back to the unreviewed pool and can be answered again.  Returns the
        undone answer (``profile_id``, ``profile_name``, ``kind``, ``would_like``), or ``None``
        when there is no answer to undo.  One transaction, so a crash can never leave a review
        row without its label or a label without its review row.
        """
        with self._lock:
            review = self.con.execute(
                "SELECT id, profile_id, app, kind, would_like FROM auto_pass_reviews"
                " ORDER BY reviewed_at DESC, id DESC LIMIT 1"
            ).fetchone()
            if review is None:
                return None
            review_id, profile_id, app, kind, would_like = review
            source = auto_review_label_source(kind)
            label = self.con.execute(
                "SELECT l.id, n.profile_name FROM labels AS l"
                " LEFT JOIN training_label_names AS n ON n.label_id=l.id"
                " WHERE l.app=? AND l.profile_id=? AND l.source=?"
                " ORDER BY l.id DESC LIMIT 1",
                (app, profile_id, source),
            ).fetchone()
            name = label[1] if label is not None else None
            if name is None:
                manifest = self.con.execute(
                    "SELECT profile_name FROM auto_passes WHERE app=? AND profile_id=?",
                    (app, profile_id),
                ).fetchone()
                name = manifest[0] if manifest is not None else None
            try:
                self.con.execute(
                    "DELETE FROM auto_pass_reviews WHERE id=?", (review_id,)
                )
                if label is not None:
                    self.con.execute("DELETE FROM labels WHERE id=?", (label[0],))
                    self.con.execute(
                        "DELETE FROM training_label_names WHERE label_id=?", (label[0],)
                    )
                self._commit()
            except BaseException:
                self.con.rollback()
                raise
        return {
            "profile_id": str(profile_id),
            "profile_name": str(name or ""),
            "kind": kind,
            "would_like": bool(would_like),
        }

    def prune_auto_passes(self, *, max_passes=None, max_age_days=None, now=None) -> int:
        """Delete archived passes (manifest + embedding + screenshots) beyond the retention.

        A pass goes when it is older than ``max_age_days`` or not among the ``max_passes``
        newest, answered or not; ``None``/0 leaves that bound off.  Returns how many passes
        were removed.  The owner's ANSWERS are untouched: labels, review rows and the decisions
        that the live false-pass estimate reads all stay.  One transaction.  SQLite does not
        shrink the file, but freed pages are reused, so the database stops growing.
        """
        max_passes, max_age_days = require_auto_pass_retention(max_passes, max_age_days)
        if max_passes is None and max_age_days is None:
            return 0
        clock = (
            time.time() if now is None else normalize_timestamp_epoch(now, label="now")
        )
        with self._lock:
            passes = [
                (row[0], row[1], float(row[2]))
                for row in self.con.execute(
                    "SELECT app, profile_id, created_at FROM auto_passes"
                ).fetchall()
            ]
            doomed = doomed_auto_passes(passes, max_passes, max_age_days, clock)
            if not doomed:
                return 0
            try:
                for app, profile_id in doomed:
                    self.con.execute(
                        "DELETE FROM auto_pass_photos WHERE app=? AND profile_id=?",
                        (app, profile_id),
                    )
                    self.con.execute(
                        "DELETE FROM auto_passes WHERE app=? AND profile_id=?",
                        (app, profile_id),
                    )
                self._commit()
            except BaseException:
                self.con.rollback()
                raise
        return len(doomed)

    def auto_review_stats(self, *, app=None, window=DEFAULT_AUTO_REVIEW_WINDOW) -> dict:
        """Counts the ranker's live false-pass estimate and the hub's review card read.

        ``random_reviewed`` / ``random_would_like`` cover the most recent ``window`` random
        reviews only (a rolling estimate that tracks the current model); near-cut counts are
        cumulative. ``auto_likes`` / ``auto_passes`` count AUTO decisions made since the first
        archived pass's own decision -- the denominator of "how much of what AUTO decides is a
        pass" -- and are 0/0 while nothing has been archived.  ``auto_model_passes`` is the part
        of ``auto_passes`` that is a genuine MODEL pass (it entered the archive); the rest are
        likes the session policy or like-ratio ceiling demoted, which the random pool never
        samples.
        """
        app = _require_optional_app(app)
        window = _require_window(window)
        app_clause = "" if app is None else " AND app=?"
        app_params: list = [] if app is None else [app]
        with self._lock:

            def pending(random_flag: int) -> int:
                clause = "" if app is None else " AND p.app=?"
                return int(
                    self.con.execute(
                        "SELECT COUNT(*) FROM auto_passes p WHERE p.random_sample=?"
                        + clause
                        + " AND NOT EXISTS (SELECT 1 FROM auto_pass_reviews r"
                        " WHERE r.app=p.app AND r.profile_id=p.profile_id)",
                        [random_flag, *app_params],
                    ).fetchone()[0]
                )

            recent_random = self.con.execute(
                "SELECT COUNT(*), COALESCE(SUM(would_like), 0) FROM ("
                "SELECT would_like FROM auto_pass_reviews WHERE kind=?"
                + app_clause
                + " ORDER BY reviewed_at DESC, id DESC LIMIT ?)",
                [AUTO_REVIEW_KIND_RANDOM, *app_params, window],
            ).fetchone()
            near_cut = self.con.execute(
                "SELECT COUNT(*), COALESCE(SUM(would_like), 0) FROM auto_pass_reviews"
                " WHERE kind=?" + app_clause,
                [AUTO_REVIEW_KIND_NEAR_CUT, *app_params],
            ).fetchone()

            # The window opens at the DECISION row of the earliest archived pass, not at that
            # pass's own auto_passes row: the worker records the decision first and archives
            # afterwards, so counting from auto_passes.created_at would silently drop the very
            # decision that started the window and bias pass_share by one pass (3 archived
            # passes read back as auto_passes == 2).  The join is by the pass's lineage
            # profile_id, so it is exact rather than a time tolerance; a pass whose decision row
            # is missing falls back to its own timestamp.
            opened_at = [
                row[0]
                for row in (
                    self.con.execute(
                        "SELECT MIN(created_at) FROM auto_passes WHERE 1=1"
                        + app_clause,
                        app_params,
                    ).fetchone(),
                    self.con.execute(
                        "SELECT MIN(d.created_at) FROM decisions d JOIN auto_passes p"
                        " ON p.app=d.app AND p.profile_id=d.profile_id"
                        " WHERE d.source='auto' AND d.decision='dislike'"
                        + app_clause.replace("app", "d.app"),
                        app_params,
                    ).fetchone(),
                )
                if row is not None and row[0] is not None
            ]
            window_start = min(opened_at, default=None)

            def decisions(decision: str) -> int:
                if window_start is None:
                    return 0
                return int(
                    self.con.execute(
                        "SELECT COUNT(*) FROM decisions WHERE source='auto' AND decision=?"
                        + app_clause
                        + " AND created_at >= ?",
                        [decision, *app_params, window_start],
                    ).fetchone()[0]
                )

            def model_passes() -> int:
                # The AUTO passes that actually entered the archive: matched by the pass's
                # lineage profile_id, exactly like the window start above (EXISTS, not a
                # join, so a duplicated archive row can never count a decision twice).
                # `auto_passes` also counts a like the session policy or the like-ratio
                # ceiling demoted to a pass -- a decision row, but never archived, so the
                # random pool that measures the false-pass rate cannot have sampled it.
                if window_start is None:
                    return 0
                return int(
                    self.con.execute(
                        "SELECT COUNT(*) FROM decisions d"
                        " WHERE d.source='auto' AND d.decision='dislike'"
                        " AND EXISTS (SELECT 1 FROM auto_passes p"
                        " WHERE p.app=d.app AND p.profile_id=d.profile_id)"
                        + app_clause.replace("app", "d.app")
                        + " AND d.created_at >= ?",
                        [*app_params, window_start],
                    ).fetchone()[0]
                )

            stats = {
                "pending_random": pending(1),
                "pending_near_cut": pending(0),
                "random_reviewed": int(recent_random[0]),
                "random_would_like": int(recent_random[1]),
                "near_cut_reviewed": int(near_cut[0]),
                "near_cut_would_like": int(near_cut[1]),
                "auto_likes": decisions("like"),
                "auto_passes": decisions("dislike"),
                "auto_model_passes": model_passes(),
            }
        return stats

    def record_spend(self, run_id, model, usage: Usage, cost):
        with self._lock:
            self.con.execute(
                "INSERT INTO spend (run_id, created_at, model, input_tokens, output_tokens,"
                " cache_read_tokens, cache_write_tokens, cost_usd) VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    time.time(),
                    model,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cache_read_input_tokens,
                    usage.cache_creation_input_tokens,
                    cost,
                ),
            )
            self._commit()

    def count_today(self, app: str, *, source: str = "auto") -> int:
        """Decisions from one validated source since local midnight."""
        if source not in {"auto", "manual"}:
            raise ValueError("count_today source must be 'auto' or 'manual'")
        start = local_midnight_epoch()
        with self._lock:
            return self.con.execute(
                "SELECT COUNT(*) FROM decisions WHERE app=? AND created_at>=? AND source=?",
                (app, start, source),
            ).fetchone()[0]

    def spend_today(self) -> float:
        """Sum of cost_usd recorded today (LOCAL day, i.e. since local midnight)."""
        start = local_midnight_epoch()
        with self._lock:
            row = self.con.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM spend WHERE created_at >= ?",
                (start,),
            ).fetchone()
        return float(row[0]) if row else 0.0

    def dropped_rows(self) -> dict[str, int]:
        """Always empty here — this backend has no buffer a row can be permanently lost from.

        Present so both backends answer the same question with the same shape (see
        ranker/__init__.py's Store.dropped_rows, and BigQueryStore.dropped_rows for the loss
        this exists to make visible): every write above goes straight into SQLite inside the
        caller's own call, and a rejected INSERT raises out of that call rather than being
        parked, retried, and eventually given up on. There is therefore no such thing as a row
        this store dropped silently -- a caller reading {} back from a SQLiteStore is reading a
        fact, not a missing implementation.
        """
        return {}

    def flush(self) -> None:
        with self._lock:
            self._commit()

    def close(self) -> None:
        with self._lock:
            try:
                self.con.close()
            finally:
                self._tighten_private_files()
