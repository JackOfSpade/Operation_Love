"""Local SQLite storage backend.

Two backends implement the same ``Store`` surface defined in ranker/__init__.py
(see bigquery_store.py for the cloud one). The supervisor loads labels once at
startup, keeps them in memory for fast inference, and appends new rows through
the store — so the hot path never blocks on per-row I/O regardless of backend.
"""
from __future__ import annotations

import datetime
import json
import math
import sqlite3
import threading
import time
from pathlib import Path

from ..costing import Usage
from ..private_files import ensure_private_dir, tighten_private_file, write_private_bytes


def local_midnight_epoch() -> float:
    """Epoch seconds for local midnight — the start of "today" in the host's own
    timezone. The daily anti-ban limits (limits.max_per_day, budget.day_budget_usd)
    are human-scale, tied to the owner's own day, not a UTC reporting boundary, so
    BOTH store backends derive "today" from this one function and can't drift apart
    again (see BigQueryStore.count_today / spend_today)."""
    return datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def _invalid_timestamp(label: str) -> ValueError:
    """One stable public error for timestamp values neither store can represent."""
    return ValueError(
        f"{label} must be a finite epoch number or a timezone-aware ISO-8601 timestamp")


def _timestamp_parts(value: object, *, label: str,
                     allow_epoch_text: bool = False) -> tuple[float, datetime.datetime]:
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


def normalize_timestamp_epoch(value: object, *, label: str = "timestamp",
                              allow_epoch_text: bool = False) -> float:
    """Return a strict, SQLite-safe epoch timestamp without lossy coercion."""
    return _timestamp_parts(value, label=label, allow_epoch_text=allow_epoch_text)[0]


def normalize_timestamp_iso(value: object, *, label: str = "timestamp",
                            allow_epoch_text: bool = False) -> str:
    """Return a strict UTC ISO timestamp suitable for BigQuery JSON TIMESTAMP fields."""
    return _timestamp_parts(value, label=label, allow_epoch_text=allow_epoch_text)[1].isoformat()


def normalize_timestamp_datetime(value: object, *, label: str = "timestamp",
                                 allow_epoch_text: bool = False) -> datetime.datetime:
    """Return the strict UTC ``datetime`` needed for bound BigQuery TIMESTAMP parameters."""
    return _timestamp_parts(value, label=label, allow_epoch_text=allow_epoch_text)[1]

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
    model_item_index INTEGER
);
CREATE TABLE IF NOT EXISTS opener_rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,
    model TEXT, attempt INTEGER, reason_code TEXT, reason TEXT, raw_opener TEXT
);
CREATE TABLE IF NOT EXISTS opener_retractions (
    correction_id TEXT NOT NULL, run_id TEXT NOT NULL, app TEXT NOT NULL,
    opener_created_at REAL NOT NULL, model TEXT NOT NULL, opener_fingerprint TEXT NOT NULL,
    reason TEXT NOT NULL, evidence_ref TEXT NOT NULL, created_at REAL NOT NULL,
    PRIMARY KEY (correction_id, opener_created_at)
);
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
        for column, decl in (("profile_id", "TEXT"), ("decision", "TEXT"),
                             ("decision_source", "TEXT"), ("decision_created_at", "REAL"),
                             ("model_item_index", "INTEGER")):
            try:
                self.con.execute(f"ALTER TABLE openers ADD COLUMN {column} {decl}")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
        self._commit()

    def _private_paths(self) -> tuple[Path, ...]:
        if self._db_path is None:
            return ()
        text = str(self._db_path)
        return (self._db_path, *(Path(text + suffix) for suffix in ("-wal", "-shm", "-journal")))

    def _tighten_private_files(self) -> None:
        """Tighten the DB and any live SQLite sidecars, rejecting symlink leaves."""
        for path in self._private_paths():
            tighten_private_file(path, parent=path.parent, missing_ok=True)

    def _commit(self) -> None:
        self.con.commit()
        self._tighten_private_files()

    def load_labels(self) -> list[tuple[bool, list[float]]]:
        with self._lock:
            rows = self.con.execute(
                f"SELECT l.liked, l.embedding FROM labels l WHERE {_LABEL_NOT_RETRACTED}").fetchall()
        return [(bool(liked), json.loads(emb)) for liked, emb in rows]

    def _visible_openers_count(self, run_id: str, app: str) -> int:
        """Count non-tombstoned openers, retaining compatibility with old test/local schemas."""
        columns = {row[1] for row in self.con.execute("PRAGMA table_info(openers)")}
        if "created_at" not in columns:
            return int(self.con.execute("SELECT COUNT(*) FROM openers WHERE run_id=? AND app=?",
                                        (run_id, app)).fetchone()[0])
        return int(self.con.execute("""SELECT COUNT(*) FROM openers AS o WHERE o.run_id=? AND o.app=?
            AND NOT EXISTS (SELECT 1 FROM opener_retractions AS r WHERE r.run_id=o.run_id
            AND r.app=o.app AND r.opener_created_at=o.created_at)""", (run_id, app)).fetchone()[0])

    def record_profile(self, run_id, app, profile_id, liked, source="manual",
                       photos=None, photo_count=0, capture_truncated: bool = False) -> bool:
        # SQLite is the offline, labels-only fallback: it doesn't archive images,
        # so there's nothing that can fail here — always "recorded". capture_truncated
        # is accepted-and-ignored for the same reason: there's no profiles manifest
        # row here to hang it off of (see BigQueryStore.record_profile, the system of
        # record, for where "was this label made from an incomplete profile read?"
        # is actually kept queryable).
        return True

    def add_label(self, run_id, app, liked, embedding, source="manual", photo_count=0,
                  profile_id="", profile_name="", **_):
        with self._lock:
            self.con.execute(
                "INSERT INTO labels (run_id, app, created_at, liked, source, embedding,"
                " photo_count, profile_id) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, app, time.time(), int(liked), source,
                 json.dumps(embedding, allow_nan=False),
                 photo_count, profile_id),
            )
            label_id = self.con.execute("SELECT last_insert_rowid()").fetchone()[0]
            self.con.execute("INSERT INTO training_label_names (label_id, profile_name) VALUES (?,?)",
                             (label_id, str(profile_name or "")))
            self._commit()

    def remove_latest_training_label(self) -> dict | None:
        """Delete and identify the newest currently-active training label."""
        with self._lock:
            row = self.con.execute(
                "SELECT l.id, n.profile_name, l.profile_id FROM labels AS l "
                "LEFT JOIN training_label_names AS n ON n.label_id=l.id "
                f"WHERE {_LABEL_NOT_RETRACTED} "
                "ORDER BY l.created_at DESC, l.id DESC LIMIT 1").fetchone()
            if row is None:
                return None
            self.con.execute("DELETE FROM labels WHERE id=?", (row[0],))
            self.con.execute("DELETE FROM training_label_names WHERE label_id=?", (row[0],))
            self._commit()
        return {"profile_name": str(row[1] or ""), "profile_id": str(row[2] or "")}

    def record_decision(self, run_id, app, decision, score, source="auto", profile_id="",
                        created_at=None):
        timestamp = (time.time() if created_at is None else normalize_timestamp_epoch(
            created_at, label="created_at"))
        with self._lock:
            self.con.execute(
                "INSERT INTO decisions (run_id, app, created_at, decision, score, source, profile_id) "
                "VALUES (?,?,?,?,?,?,?)",
                (run_id, app, timestamp, decision, score,
                 source, profile_id),
            )
            self._commit()

    def retraction_run_rows(self, run_id: str, app: str, source: str) -> dict:
        """Exact minimal run rows used by the correction planner; no embeddings/photos."""
        with self._lock:
            labels = self.con.execute("""SELECT profile_id, created_at, liked
                FROM labels WHERE run_id=? AND app=? AND source=? ORDER BY created_at, id""",
                (run_id, app, source)).fetchall()
            decisions = self.con.execute("""SELECT profile_id, created_at, decision, score
                FROM decisions WHERE run_id=? AND app=? AND source=? ORDER BY created_at, id""",
                (run_id, app, source)).fetchall()
            retractions = self.con.execute("""SELECT correction_id, profile_id, label_created_at,
                decision_created_at, decision_fingerprint FROM label_retractions
                WHERE run_id=? AND app=? AND source=? ORDER BY created_at, correction_id""",
                (run_id, app, source)).fetchall()
        def exact_epoch(value):
            """A decimal spelling which round-trips SQLite's stored IEEE-754 REAL exactly.

            Retraction predicates use equality, by design.  An ISO timestamp has only
            microsecond precision and can silently leave a label live when `time.time()`
            supplied sub-microsecond data.  ``.17g`` is the shortest safe portable
            precision for a binary64 round-trip; retractions._timestamp / append parser
            intentionally accept this SQLite-only planner representation.
            """
            return format(float(value), ".17g")
        return {"run_id": run_id, "app": app, "source": source, "profiles": [],
                "labels": [{"profile_id": p, "created_at": exact_epoch(t), "liked": bool(liked)}
                           for p, t, liked in labels],
                "decisions": [({"created_at": exact_epoch(t), "decision": d, "score": s}
                               | ({"profile_id": p} if p not in {None, ""} else {}))
                              for p, t, d, s in decisions],
                "retractions": [{"correction_id": c, "profile_id": p,
                                  "label_created_at": exact_epoch(label_time),
                                  "decision_created_at": exact_epoch(d),
                                  "decision_fingerprint": f}
                                 for c, p, label_time, d, f in retractions]}

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
                row["correction_id"], row["run_id"], row["app"], row["source"], row["profile_id"],
                normalize_timestamp_epoch(row["label_created_at"], label="label_created_at",
                                          allow_epoch_text=True),
                normalize_timestamp_epoch(row["decision_created_at"], label="decision_created_at",
                                          allow_epoch_text=True),
                row["decision_fingerprint"], row["reason"], row["evidence_ref"], time.time())
        except (KeyError, ValueError) as exc:
            raise RetractionRefused("SQLite correction timestamps are invalid") from exc
        with self._lock:
            existing = self.con.execute("""SELECT correction_id FROM label_retractions
                WHERE run_id=? AND app=? AND source=? AND profile_id=?
                AND label_created_at=? AND decision_created_at=?""", values[1:7]).fetchone()
            if existing:
                if existing[0] == row["correction_id"]:
                    return False
                raise RetractionRefused("target label/decision pair already has a different retraction")
            cur = self.con.execute("""INSERT OR IGNORE INTO label_retractions
                (correction_id,run_id,app,source,profile_id,label_created_at,decision_created_at,
                decision_fingerprint,reason,evidence_ref,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""", values)
            self._commit()
        return cur.rowcount == 1

    def observe_release_persistence_summary(self, run_id: str, app: str) -> dict[str, int]:
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
                (run_id, app)).fetchone()[0]
            like_labels = self.con.execute(
                "SELECT COUNT(*) FROM labels l WHERE run_id=? AND app=? AND source='manual' "
                f"AND liked=1 AND {_LABEL_NOT_RETRACTED}",
                (run_id, app)).fetchone()[0]
            pass_decisions = self.con.execute(
                """SELECT COUNT(*) FROM decisions d WHERE run_id=? AND app=? AND source='manual' AND decision='dislike'
                AND NOT EXISTS (SELECT 1 FROM label_retractions r WHERE r.run_id=d.run_id AND r.app=d.app
                AND r.source=d.source AND r.decision_created_at=d.created_at)""",
                (run_id, app)).fetchone()[0]
            like_decisions = self.con.execute(
                """SELECT COUNT(*) FROM decisions d WHERE run_id=? AND app=? AND source='manual' AND decision='like'
                AND NOT EXISTS (SELECT 1 FROM label_retractions r WHERE r.run_id=d.run_id AND r.app=d.app
                AND r.source=d.source AND r.decision_created_at=d.created_at)""",
                (run_id, app)).fetchone()[0]
            openers = self._visible_openers_count(run_id, app)
        return {
            "manual_pass_labels": int(pass_labels),
            "manual_like_labels": int(like_labels),
            "manual_pass_decisions": int(pass_decisions),
            "manual_like_decisions": int(like_decisions),
            "successful_hinge_openers": int(openers),
        }

    def ai_observe_release_persistence_summary(self, run_id: str, app: str,
                                               source: str) -> dict[str, int]:
        """Read-only complete-cycle counts for a declared non-manual release source."""
        if source not in {"external_ai_review", "automation"}:
            raise ValueError("AI observe release source must be external_ai_review or automation")
        with self._lock:
            pass_labels = self.con.execute(
                "SELECT COUNT(*) FROM labels l WHERE run_id=? AND app=? AND source=? "
                f"AND liked=0 AND {_LABEL_NOT_RETRACTED}",
                (run_id, app, source)).fetchone()[0]
            like_labels = self.con.execute(
                "SELECT COUNT(*) FROM labels l WHERE run_id=? AND app=? AND source=? "
                f"AND liked=1 AND {_LABEL_NOT_RETRACTED}",
                (run_id, app, source)).fetchone()[0]
            pass_decisions = self.con.execute(
                """SELECT COUNT(*) FROM decisions d WHERE run_id=? AND app=? AND source=? AND decision='dislike'
                AND NOT EXISTS (SELECT 1 FROM label_retractions r WHERE r.run_id=d.run_id AND r.app=d.app
                AND r.source=d.source AND r.decision_created_at=d.created_at)""",
                (run_id, app, source)).fetchone()[0]
            like_decisions = self.con.execute(
                """SELECT COUNT(*) FROM decisions d WHERE run_id=? AND app=? AND source=? AND decision='like'
                AND NOT EXISTS (SELECT 1 FROM label_retractions r WHERE r.run_id=d.run_id AND r.app=d.app
                AND r.source=d.source AND r.decision_created_at=d.created_at)""",
                (run_id, app, source)).fetchone()[0]
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
            raw = self.con.execute("SELECT created_at,model,opener FROM openers WHERE run_id=? AND app=? "
                                   "ORDER BY created_at", (run_id, app)).fetchall()
            existing = self.con.execute("SELECT correction_id,opener_created_at,model,opener_fingerprint,reason,evidence_ref "
                                        "FROM opener_retractions "
                                        "WHERE run_id=? AND app=?", (run_id, app)).fetchall()
            def count(table):
                return int(self.con.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id=? AND app=?",
                                            (run_id, app)).fetchone()[0])
            def effective_labels(liked: int):
                return int(self.con.execute(
                    "SELECT COUNT(*) FROM labels l WHERE run_id=? AND app=? AND liked=? "
                    f"AND {_LABEL_NOT_RETRACTED}", (run_id, app, liked)).fetchone()[0])
            def decisions(decision: str):
                return int(self.con.execute(
                    "SELECT COUNT(*) FROM decisions WHERE run_id=? AND app=? AND decision=?",
                    (run_id, app, decision)).fetchone()[0])
            # Every one of these closures touches self.con, so they are CALLED here rather than
            # from the returned dict below: this class shares one connection across threads on
            # check_same_thread=False + this lock (see __init__), and a query issued after the
            # release would be exactly the unsynchronized use that combination forbids. Only the
            # pure-CPU fingerprinting below is done outside.
            decision_rows = count("decisions")
            label_rows = count("labels")
            like_labels, pass_labels = effective_labels(1), effective_labels(0)
            like_decisions, pass_decisions = decisions("like"), decisions("dislike")
        openers = [{"created_at": format(float(created), ".17g"), "model": str(model),
                    "opener_fingerprint": canonical_sha({"run_id": run_id, "app": app,
                                                           "created_at": format(float(created), ".17g"),
                                                           "model": str(model), "opener": str(opener)})}
                   for created, model, opener in raw]
        return {"run_id": run_id, "app": app, "openers": openers,
                "decisions": [{} for _ in range(decision_rows)],
                # SQLite is explicitly labels-only: profiles/photos are never archived here.
                "preference_counts": {"profiles": 0, "profile_photos": 0,
                                      "labels": label_rows, "decisions": decision_rows},
                "effective_counts": {"like_labels": like_labels,
                                     "like_decisions": like_decisions,
                                     "pass_labels": pass_labels,
                                     "pass_decisions": pass_decisions},
                "retractions": [{"correction_id": row[0], "opener_created_at": format(float(row[1]), ".17g"),
                                "model": row[2], "opener_fingerprint": row[3], "reason": row[4],
                                "evidence_ref": row[5]}
                               for row in existing]}

    def append_opener_retraction(self, row: dict) -> bool:
        from .retractions import RetractionRefused, canonical_sha
        try:
            opener_at = normalize_timestamp_epoch(row["opener_created_at"],
                                                  label="opener_created_at",
                                                  allow_epoch_text=True)
            values = (row["correction_id"], row["run_id"], row["app"], opener_at,
                      row["model"], row["opener_fingerprint"], row["reason"],
                      row["evidence_ref"], time.time())
        except (KeyError, ValueError) as exc:
            raise RetractionRefused("SQLite opener correction timestamp is invalid") from exc
        with self._lock:
            source = self.con.execute(
                "SELECT model,opener FROM openers WHERE run_id=? AND app=? AND created_at=?",
                (row["run_id"], row["app"], opener_at)).fetchall()
            if len(source) != 1 or str(source[0][0]) != row["model"]:
                raise RetractionRefused("target opener row is missing or does not match its cleanup plan")
            fingerprint = canonical_sha({"run_id": row["run_id"], "app": row["app"],
                                         "created_at": format(opener_at, ".17g"),
                                         "model": str(source[0][0]), "opener": str(source[0][1])})
            if fingerprint != row["opener_fingerprint"]:
                raise RetractionRefused("target opener fingerprint no longer matches its cleanup plan")
            existing = self.con.execute(
                "SELECT correction_id,model,opener_fingerprint,reason,evidence_ref FROM opener_retractions "
                "WHERE run_id=? AND app=? AND opener_created_at=?",
                (row["run_id"], row["app"], opener_at)).fetchone()
            if existing:
                actual = {"correction_id": existing[0], "model": existing[1],
                          "opener_fingerprint": existing[2], "reason": existing[3],
                          "evidence_ref": existing[4]}
                expected = {key: row[key] for key in actual}
                if actual == expected:
                    return False
                raise RetractionRefused("opener row already has a different cleanup tombstone")
            cur = self.con.execute("INSERT OR IGNORE INTO opener_retractions "
                "(correction_id,run_id,app,opener_created_at,model,opener_fingerprint,reason,evidence_ref,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)", values)
            self._commit()
        return cur.rowcount == 1

    def record_opener(self, run_id, app, model, opener, referenced, angle="",
                      item_description="", *, profile_id="", decision="", decision_source="",
                      decision_created_at=None, model_item_index=None):
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
        timestamp = (None if decision_created_at is None else normalize_timestamp_epoch(
            decision_created_at, label="decision_created_at"))
        with self._lock:
            self.con.execute(
                "INSERT INTO openers (run_id, app, created_at, model, opener, referenced, angle,"
                " item_description, profile_id, decision, decision_source, decision_created_at,"
                " model_item_index) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, app, time.time(), model, opener, referenced, angle, item_description,
                 profile_id, decision, decision_source, timestamp, model_item_index),
            )
            self._commit()

    def record_opener_rejection(self, run_id, app, model, attempt, reason_code, reason, raw_opener):
        with self._lock:
            self.con.execute(
                "INSERT INTO opener_rejections (run_id, app, created_at, model, attempt,"
                " reason_code, reason, raw_opener) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, app, time.time(), model, attempt, reason_code, reason, raw_opener),
            )
            self._commit()

    def record_spend(self, run_id, model, usage: Usage, cost):
        with self._lock:
            self.con.execute(
                "INSERT INTO spend (run_id, created_at, model, input_tokens, output_tokens,"
                " cache_read_tokens, cache_write_tokens, cost_usd) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, time.time(), model, usage.input_tokens, usage.output_tokens,
                 usage.cache_read_input_tokens, usage.cache_creation_input_tokens, cost),
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

    def flush(self) -> None:
        with self._lock:
            self._commit()

    def close(self) -> None:
        with self._lock:
            try:
                self.con.close()
            finally:
                self._tighten_private_files()
