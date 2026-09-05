"""SQLiteStore source-tagging and daily-limit behavior."""
import math
import os
import sqlite3
import stat
import time
from pathlib import Path

import pytest

from operation_love.ranker import Store
from operation_love.ranker import store as store_module
from operation_love.ranker.store import SQLiteStore
from operation_love.private_files import UnsafePrivatePathError


def test_sqlite_store_conforms_to_store_protocol(tmp_path):
    s = SQLiteStore(tmp_path / "store.db")
    try:
        assert isinstance(s, Store)
    finally:
        s.close()


@pytest.mark.parametrize("value", [True, math.nan, math.inf, -math.inf, 1e300, 10 ** 10_000,
                                   "2026-01-01T00:00:00"],
                         ids=["bool", "nan", "inf", "negative-inf", "out-of-range-float",
                              "huge-int", "naive-iso"])
def test_sqlite_action_writes_reject_invalid_timestamp_values_without_overflow(tmp_path, value):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        with pytest.raises(ValueError, match="created_at"):
            store.record_decision("r", "hinge", "like", 1.0, created_at=value)
        with pytest.raises(ValueError, match="decision_created_at"):
            store.record_opener(
                "r", "hinge", "model", "opener", "reference",
                decision_created_at=value,
            )
        assert store.con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
        assert store.con.execute("SELECT COUNT(*) FROM openers").fetchone()[0] == 0
    finally:
        store.close()


def test_sqlite_action_timestamps_accept_timezone_aware_iso_text(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_decision("r", "hinge", "like", 1.0,
                              created_at="2026-01-01T00:00:00Z")
        assert store.con.execute("SELECT created_at FROM decisions").fetchone()[0] == 1767225600.0
    finally:
        store.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_sqlite_store_tightens_only_its_leaf_data_dir_database_and_sidecars(tmp_path):
    broad_parent = tmp_path / "shared"
    data_dir = broad_parent / "operation-love"
    broad_parent.mkdir(mode=0o755)
    data_dir.mkdir(mode=0o755)
    broad_parent.chmod(0o755)
    data_dir.chmod(0o755)
    db = data_dir / "store.db"

    store = SQLiteStore(db)
    try:
        assert stat.S_IMODE(broad_parent.stat().st_mode) == 0o755
        assert stat.S_IMODE(data_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(db.stat().st_mode) == 0o600

        sidecars = [Path(str(db) + suffix) for suffix in ("-wal", "-shm", "-journal")]
        for sidecar in sidecars:
            sidecar.write_bytes(b"")
            sidecar.chmod(0o644)
        store.flush()
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in sidecars)
    finally:
        store.close()


@pytest.mark.skipif(os.name != "posix", reason="symlink and POSIX permission semantics")
@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_sqlite_store_rejects_symlink_database_artifacts_without_chmodding_target(
        tmp_path, suffix):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = data_dir / "store.db"
    unrelated = tmp_path / f"unrelated{suffix or '-db'}"
    unrelated.write_bytes(b"do not touch")
    unrelated.chmod(0o644)
    Path(str(db) + suffix).symlink_to(unrelated)

    with pytest.raises(UnsafePrivatePathError):
        SQLiteStore(db)

    assert unrelated.read_bytes() == b"do not touch"
    assert stat.S_IMODE(unrelated.stat().st_mode) == 0o644


def test_sqlite_memory_database_does_not_create_or_chmod_a_filesystem_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = stat.S_IMODE(tmp_path.stat().st_mode)

    store = SQLiteStore(":memory:")
    try:
        assert store.con.execute("SELECT COUNT(*) FROM labels").fetchone() == (0,)
    finally:
        store.close()

    assert not (tmp_path / ":memory:").exists()
    assert stat.S_IMODE(tmp_path.stat().st_mode) == before


def test_sqlite_initialization_failure_closes_connection_without_masking_cause(monkeypatch):
    class BrokenConnection:
        closed = False

        def executescript(self, _schema):
            raise sqlite3.DatabaseError("schema exploded")

        def close(self):
            self.closed = True
            raise OSError("cleanup also failed")

    connection = BrokenConnection()
    monkeypatch.setattr(store_module.sqlite3, "connect", lambda *_args, **_kwargs: connection)

    with pytest.raises(sqlite3.DatabaseError, match="schema exploded"):
        SQLiteStore(":memory:")

    assert connection.closed is True


def test_sqlite_count_today_counts_the_requested_decision_source(tmp_path):
    db = tmp_path / "store.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE decisions ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,"
        "decision TEXT, score REAL)"
    )
    con.execute(
        "INSERT INTO decisions (run_id, app, created_at, decision, score) VALUES (?,?,?,?,?)",
        ("legacy", "bumble", time.time(), "like", 1.0),
    )
    con.commit()
    con.close()

    store = SQLiteStore(db)
    try:
        store.record_decision("r", "bumble", "like", 0.9, source="manual")
        store.record_decision("r", "bumble", "dislike", 0.1, source="auto")
        store.record_decision("r", "hinge", "like", 0.8, source="auto")

        assert store.count_today("bumble") == 1
        assert store.count_today("bumble", source="manual") == 1
        with pytest.raises(ValueError, match="source"):
            store.count_today("bumble", source="automation")
        rows = store.con.execute(
            "SELECT decision, source FROM decisions WHERE app='bumble' ORDER BY id"
        ).fetchall()
        assert rows == [("like", None), ("like", "manual"), ("dislike", "auto")]
    finally:
        store.close()


def test_sqlite_add_label_persists_profile_id(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.add_label("r", "bumble", True, [0.1, 0.2], profile_id="profile-1")

        row = store.con.execute("SELECT profile_id FROM labels").fetchone()
        assert row == ("profile-1",)
    finally:
        store.close()


def test_sqlite_observe_release_summary_requires_distinct_manual_pass_and_like_records(tmp_path):
    """Release proof is aggregate-only but cannot conflate two likes with a complete cycle."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.add_label("r", "hinge", False, [0.1], source="manual")
        store.add_label("r", "hinge", True, [0.2], source="manual")
        store.record_decision("r", "hinge", "dislike", 0.0, source="manual")
        store.record_decision("r", "hinge", "like", 1.0, source="manual")
        store.record_opener("r", "hinge", "gemini-x", "hi", "photo")

        assert store.observe_release_persistence_summary("r", "hinge") == {
            "manual_pass_labels": 1,
            "manual_like_labels": 1,
            "manual_pass_decisions": 1,
            "manual_like_decisions": 1,
            "successful_hinge_openers": 1,
        }
    finally:
        store.close()


def test_sqlite_retraction_excludes_only_bound_label_and_decision_from_loads_and_release(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.con.execute("INSERT INTO labels VALUES (NULL,?,?,?,?,?,?,?,?)",
                          ("r", "hinge", 10.0, 0, "manual", "[0.1]", 0, "false-profile"))
        store.con.execute("INSERT INTO labels VALUES (NULL,?,?,?,?,?,?,?,?)",
                          ("r", "hinge", 20.0, 1, "manual", "[0.2]", 0, "good-profile"))
        # Name columns so this historical fixture remains valid as optional lineage fields are
        # appended to the decisions schema.
        store.con.execute("INSERT INTO decisions (run_id,app,created_at,decision,score,source) "
                          "VALUES (?,?,?,?,?,?)",
                          ("r", "hinge", 11.0, "dislike", 0.0, "manual"))
        store.con.execute("INSERT INTO decisions (run_id,app,created_at,decision,score,source) "
                          "VALUES (?,?,?,?,?,?)",
                          ("r", "hinge", 21.0, "like", 1.0, "manual"))
        store.con.execute("""INSERT INTO label_retractions VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                          ("correction", "r", "hinge", "manual", "false-profile", 10.0, 11.0,
                           "fingerprint", "false controller action", "debug#225", 30.0))
        store.con.commit()
        assert store.load_labels() == [(True, [0.2])]
        assert store.observe_release_persistence_summary("r", "hinge") == {
            "manual_pass_labels": 0, "manual_like_labels": 1,
            "manual_pass_decisions": 0, "manual_like_decisions": 1,
            "successful_hinge_openers": 0}
    finally:
        store.close()


def _seed_identity_less_labels(store):
    """Two labels carrying NO profile identity, in both spellings, plus one that has one.

    NULL is how a row written before `labels.profile_id` was ALTERed in reads back (see
    _initialize_schema); '' is what add_label's own default writes. Both mean "no profile
    identity", and a tombstone has to be able to reach either one.
    """
    for created_at, liked, embedding, profile_id in (
            (10.0, 0, "[0.1]", None),
            (20.0, 0, "[0.2]", ""),
            (30.0, 1, "[0.3]", "modern-profile"),
    ):
        store.con.execute("INSERT INTO labels VALUES (NULL,?,?,?,?,?,?,?,?)",
                          ("r", "hinge", created_at, liked, "manual", embedding, 0, profile_id))
    for created_at, decision in ((9.0, "dislike"), (19.0, "dislike"), (29.0, "like")):
        store.con.execute("INSERT INTO decisions (run_id,app,created_at,decision,score,source) "
                          "VALUES (?,?,?,?,?,?)",
                          ("r", "hinge", created_at, decision, 0.0, "manual"))
    store.con.commit()


def _seed_tombstone(store, *, correction_id, profile_id, label_created_at, decision_created_at):
    store.con.execute("INSERT INTO label_retractions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (correction_id, "r", "hinge", "manual", profile_id, label_created_at,
                       decision_created_at, "fingerprint", "false pass", "debug#225", 99.0))
    store.con.commit()


def test_sqlite_tombstone_retracts_a_label_that_carries_no_profile_id(tmp_path):
    """A legacy NULL-profile_id label must be retractable, not merely removable.

    `r.profile_id=l.profile_id` is NULL (never TRUE) for such a label, so no tombstone could
    ever reach it: it stayed in the training set permanently, and the only way to get rid of it
    was remove_latest_training_label, which destroys the row instead of correcting it.
    """
    store = SQLiteStore(tmp_path / "store.db")
    try:
        _seed_identity_less_labels(store)
        assert store.load_labels() == [(False, [0.1]), (False, [0.2]), (True, [0.3])]

        _seed_tombstone(store, correction_id="c-null", profile_id="",
                        label_created_at=10.0, decision_created_at=9.0)
        assert store.load_labels() == [(False, [0.2]), (True, [0.3])]

        # The '' spelling of the same absent identity has to be reachable by its own tombstone
        # too, and only by its own: the join still carries label_created_at.
        _seed_tombstone(store, correction_id="c-empty", profile_id="",
                        label_created_at=20.0, decision_created_at=19.0)
        assert store.load_labels() == [(True, [0.3])]
    finally:
        store.close()


def test_sqlite_tombstone_for_another_label_never_hides_an_identity_less_one(tmp_path):
    """Folding NULL and '' together must not turn a tombstone into a wildcard."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        _seed_identity_less_labels(store)
        # Same timestamp as the NULL-profile label but a different profile identity.
        _seed_tombstone(store, correction_id="c-other-identity", profile_id="modern-profile",
                        label_created_at=10.0, decision_created_at=9.0)
        # No profile identity, but another label's timestamp.
        _seed_tombstone(store, correction_id="c-other-time", profile_id="",
                        label_created_at=99.0, decision_created_at=9.0)
        # Right label, wrong run.
        store.con.execute("INSERT INTO label_retractions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                          ("c-other-run", "other-run", "hinge", "manual", "", 10.0, 9.0,
                           "fingerprint", "false pass", "debug#225", 99.0))
        store.con.commit()

        assert store.load_labels() == [(False, [0.1]), (False, [0.2]), (True, [0.3])]
    finally:
        store.close()


def test_sqlite_every_read_path_agrees_about_a_retracted_identity_less_label(tmp_path):
    """load_labels, both release summaries and the cleanup advisory share one predicate.

    They are separate SQL statements, so a fix applied to only some of them lets a release
    summary count a label the ranker has already stopped training on.
    """
    store = SQLiteStore(tmp_path / "store.db")
    try:
        _seed_identity_less_labels(store)
        _seed_tombstone(store, correction_id="c-null", profile_id="",
                        label_created_at=10.0, decision_created_at=9.0)

        assert store.load_labels() == [(False, [0.2]), (True, [0.3])]
        assert store.observe_release_persistence_summary("r", "hinge") == {
            "manual_pass_labels": 1, "manual_like_labels": 1,
            "manual_pass_decisions": 1, "manual_like_decisions": 1,
            "successful_hinge_openers": 0}
        assert store.ai_observe_release_persistence_summary(
            "r", "hinge", "external_ai_review") == {
            "ai_pass_labels": 0, "ai_like_labels": 0,
            "ai_pass_decisions": 0, "ai_like_decisions": 0,
            "successful_hinge_openers": 0}
        assert store.advisory_opener_run_rows("r", "hinge")["effective_counts"] == {
            "like_labels": 1, "like_decisions": 1,
            "pass_labels": 1, "pass_decisions": 2}
        # remove_latest_training_label reads the same predicate, so it must not offer the
        # operator a label the correction has already retracted.
        assert store.remove_latest_training_label() == {"profile_name": "", "profile_id": "modern-profile"}
    finally:
        store.close()


def test_sqlite_migrates_legacy_labels_profile_id_column(tmp_path):
    db = tmp_path / "store.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE labels ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,"
        "liked INTEGER, source TEXT, embedding TEXT, photo_count INTEGER)"
    )
    con.commit()
    con.close()

    store = SQLiteStore(db)
    try:
        cols = [row[1] for row in store.con.execute("PRAGMA table_info(labels)").fetchall()]
        assert "profile_id" in cols

        store.add_label("r", "bumble", False, [0.3], profile_id="profile-legacy")
        row = store.con.execute("SELECT profile_id FROM labels").fetchone()
        assert row == ("profile-legacy",)
    finally:
        store.close()


def test_sqlite_add_label_rejects_nan_embedding(tmp_path):
    """add_label() must raise (not silently store) NaN/Inf values in embeddings."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        try:
            store.add_label("r", "bumble", True, [0.1, float("nan"), 0.3])
            raise AssertionError("expected ValueError on NaN embedding")
        except ValueError as e:
            # json.dumps(..., allow_nan=False) is what actually rejects it (store.py),
            # and its real message is "Out of range float values are not JSON
            # compliant" — it never literally says "nan"/"allow_nan".
            assert "not json compliant" in str(e).lower()
    finally:
        store.close()


def test_sqlite_record_opener_persists_angle_alongside_the_opener(tmp_path):
    """`angle` is the model's own free-text words for what the opener is DOING (see the
    Store protocol's comment in ranker/__init__.py). It is telemetry only -- nothing reads
    it back at runtime -- which is exactly why it needs a test: a silent drop here would
    never surface as a failure anywhere else, it would just quietly make the "which opener
    shapes correlate with matches" question unanswerable again."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("r", "hinge", "gemini-x", "I bet you were freezing out there",
                            "Photo of her on a ridgeline", "guess the world")

        row = store.con.execute(
            "SELECT run_id, app, model, opener, referenced, angle FROM openers"
        ).fetchone()
        assert row == ("r", "hinge", "gemini-x", "I bet you were freezing out there",
                       "Photo of her on a ridgeline", "guess the world")
    finally:
        store.close()


def test_sqlite_record_opener_defaults_angle_to_empty_string_for_positional_callers(tmp_path):
    """`angle` is strictly trailing with a "" default so every pre-existing 5-positional-arg
    caller (and every test double implementing the Store protocol) keeps working untouched.
    Pinned as an empty string rather than NULL: a row written AFTER this column landed but
    without an angle is a "the model returned no angle" row, which is a different fact from
    a legacy row that predates the column entirely (that one reads back NULL -- see
    test_sqlite_migrates_legacy_openers_angle_column)."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("r", "hinge", "gemini-x", "hi there", "Her travel prompt")

        row = store.con.execute("SELECT opener, angle FROM openers").fetchone()
        assert row == ("hi there", "")
    finally:
        store.close()


def test_sqlite_openers_table_has_the_expected_columns_in_order(tmp_path):
    """A FRESH database gets `angle` and `item_description` from _SCHEMA (the ALTERs below it
    are the no-op path there). Pinning the full column list catches _SCHEMA drifting away from
    the ALTER migrations, which would leave fresh and migrated databases with different
    shapes."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        cols = [row[1] for row in store.con.execute("PRAGMA table_info(openers)").fetchall()]
        assert cols == ["id", "run_id", "app", "created_at", "model", "opener",
                        "referenced", "angle", "item_description", "profile_id", "decision",
                        "decision_source", "decision_created_at", "model_item_index"]
    finally:
        store.close()


def test_sqlite_record_opener_persists_item_description_as_its_own_column(tmp_path):
    """`item_description` is the model's own description of the ITEM it picked to write about
    and to like (ops/OPENER-REDESIGN.md 5.7). Stored SEPARATELY from `referenced` (the detail
    the opener reacts to) and `angle` (what the opener is doing) because the doc is explicit
    that none of the three replaces another -- folding any two together would blank a
    telemetry column that nothing else in the system would notice was gone."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("r", "hinge", "gemini-x", "I bet you were freezing out there",
                            "Photo of her on a ridgeline", "guess the world",
                            "a photo, her on a ridge")

        row = store.con.execute(
            "SELECT referenced, angle, item_description FROM openers").fetchone()
        assert row == ("Photo of her on a ridgeline", "guess the world",
                       "a photo, her on a ridge")
    finally:
        store.close()


def test_sqlite_opener_lineage_binds_the_landed_like_and_model_item(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_decision("run", "hinge", "like", 1.0, source="manual",
                              profile_id="profile-1", created_at=12.5)
        store.record_opener("run", "hinge", "gemini-x", "hello", "photo", profile_id="profile-1",
                            decision="like", decision_source="manual", decision_created_at=12.5,
                            model_item_index=3)
        assert store.con.execute(
            "SELECT profile_id,decision,decision_source,decision_created_at,model_item_index FROM openers"
        ).fetchone() == ("profile-1", "like", "manual", 12.5, 3)
        assert store.con.execute("SELECT profile_id,created_at FROM decisions").fetchone() == ("profile-1", 12.5)
    finally:
        store.close()


def test_sqlite_persists_opener_action_lineage_with_the_landed_like(tmp_path):
    """A committed opener has an explicit, queryable decision identity rather than a heuristic."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        action_at = 1234.5
        store.record_decision("r", "hinge", "like", 0.9, source="auto",
                              profile_id="action-1", created_at=action_at)
        store.record_opener("r", "hinge", "gemini-x", "hello", "her photo", "teasing",
                            "a photo", profile_id="action-1", decision="like",
                            decision_source="auto", decision_created_at=action_at,
                            model_item_index=3)

        decision = store.con.execute(
            "SELECT profile_id, created_at, decision FROM decisions").fetchone()
        opener = store.con.execute(
            "SELECT profile_id, decision, decision_source, decision_created_at, model_item_index "
            "FROM openers").fetchone()
        assert decision == ("action-1", action_at, "like")
        assert opener == ("action-1", "like", "auto", action_at, 3)
    finally:
        store.close()


def test_sqlite_record_opener_defaults_item_description_for_positional_callers(tmp_path):
    """Trailing with a "" default, exactly like `angle` before it and for the same reason: a
    six-positional-arg caller (or a model response that carried no description) must still
    write a complete row. "" here means "this generation produced none"; NULL means "this row
    predates the column" -- see test_sqlite_migrates_legacy_openers_item_description_column."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("r", "hinge", "gemini-x", "hi there", "Her travel prompt", "know")

        row = store.con.execute("SELECT opener, item_description FROM openers").fetchone()
        assert row == ("hi there", "")
    finally:
        store.close()


def test_sqlite_migrates_legacy_openers_item_description_column(tmp_path):
    """The production case for the newer column: a db file written when `openers` already had
    `angle` but not `item_description`. CREATE TABLE IF NOT EXISTS is a no-op against it, so
    the ALTER is the ONLY thing that carries the column in -- without it record_opener's INSERT
    fails on every single opener sent, which surfaces only as a per-profile "failed to persist"
    warning."""
    db = tmp_path / "store.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE openers ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,"
        "model TEXT, opener TEXT, referenced TEXT, angle TEXT)"
    )
    con.execute(
        "INSERT INTO openers (run_id, app, created_at, model, opener, referenced, angle)"
        " VALUES (?,?,?,?,?,?,?)",
        ("legacy", "hinge", 1.0, "gemini-old", "an opener from before item_description",
         "ref", "guess"),
    )
    con.commit()
    con.close()

    store = SQLiteStore(db)
    try:
        cols = [row[1] for row in store.con.execute("PRAGMA table_info(openers)").fetchall()]
        assert "item_description" in cols

        store.record_opener("r", "hinge", "gemini-x", "a new opener", "ref", "imagine",
                            "a prompt card")
        rows = store.con.execute(
            "SELECT opener, item_description FROM openers ORDER BY id").fetchall()
        # The pre-existing row keeps NULL (ALTER ... ADD COLUMN backfills nothing), the new
        # row round-trips its description. Asserted together so the two can't be confused.
        assert rows == [("an opener from before item_description", None),
                        ("a new opener", "a prompt card")]
    finally:
        store.close()


def test_sqlite_openers_item_description_migration_reraises_unexpected_operational_errors(
        tmp_path, monkeypatch):
    """Fail loud, same contract as the angle ALTER directly below this in store.py: swallow
    ONLY "duplicate column name". Pinned per column rather than once for the file because each
    ALTER has its own try/except, so a new one can easily be written with a bare `except
    sqlite3.OperationalError: pass` and nothing else would catch it."""
    real_connect = sqlite3.connect
    opened = []

    class _ItemDescriptionAlterFails:
        def __init__(self, con):
            self._con = con

        def execute(self, sql, *args, **kwargs):
            if "ALTER TABLE openers ADD COLUMN item_description" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return self._con.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._con, name)

    def fake_connect(*args, **kwargs):
        con = real_connect(*args, **kwargs)
        opened.append(con)
        return _ItemDescriptionAlterFails(con)

    monkeypatch.setattr(sqlite3, "connect", fake_connect)

    try:
        SQLiteStore(tmp_path / "store.db")
        raise AssertionError("expected the non-duplicate-column OperationalError to propagate")
    except sqlite3.OperationalError as e:
        assert "disk i/o error" in str(e).lower()
    finally:
        for con in opened:
            con.close()


def test_sqlite_openers_angle_migration_is_idempotent_across_reopens(tmp_path):
    """Every startup re-runs the ALTER, so the SECOND (and third) SQLiteStore against the
    same file must not raise -- that "duplicate column name" guard in __init__ is the only
    thing standing between an added column and a crash on every subsequent boot."""
    db = tmp_path / "store.db"
    first = SQLiteStore(db)
    try:
        first.record_opener("r1", "hinge", "gemini-x", "first", "ref", "guess")
    finally:
        first.close()

    # Reopened twice: once against a file this process just closed, and once more while
    # that second connection is still open, since production reopens are not serialized.
    second = SQLiteStore(db)
    third = SQLiteStore(db)
    try:
        cols = [row[1] for row in second.con.execute("PRAGMA table_info(openers)").fetchall()]
        assert cols.count("angle") == 1

        third.record_opener("r2", "hinge", "gemini-x", "second", "ref", "know")
        rows = second.con.execute("SELECT opener, angle FROM openers ORDER BY id").fetchall()
        assert rows == [("first", "guess"), ("second", "know")]
    finally:
        third.close()
        second.close()


def test_sqlite_migrates_legacy_openers_angle_column(tmp_path):
    """The production case: a db file written before `angle` existed. CREATE TABLE IF NOT
    EXISTS is a no-op against it, so the ALTER is the ONLY thing that carries the column in
    -- if it were missing, record_opener's INSERT would fail on every single opener sent."""
    db = tmp_path / "store.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE openers ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,"
        "model TEXT, opener TEXT, referenced TEXT)"
    )
    con.execute(
        "INSERT INTO openers (run_id, app, created_at, model, opener, referenced)"
        " VALUES (?,?,?,?,?,?)",
        ("legacy", "hinge", 1.0, "gemini-old", "an opener from before angle", "ref"),
    )
    con.commit()
    con.close()

    store = SQLiteStore(db)
    try:
        cols = [row[1] for row in store.con.execute("PRAGMA table_info(openers)").fetchall()]
        assert "angle" in cols

        store.record_opener("r", "hinge", "gemini-x", "a new opener", "ref", "imagine")
        rows = store.con.execute("SELECT opener, angle FROM openers ORDER BY id").fetchall()
        # The pre-existing row keeps NULL (ALTER ... ADD COLUMN backfills nothing), while the
        # new row round-trips its angle. Asserted together so the two cases can't be confused.
        assert rows == [("an opener from before angle", None), ("a new opener", "imagine")]
    finally:
        store.close()


def test_sqlite_openers_angle_migration_reraises_unexpected_operational_errors(tmp_path,
                                                                               monkeypatch):
    """Fail loud: the guard swallows ONLY "duplicate column name" (the already-migrated
    case). Any other OperationalError means we are about to start up with a schema we can't
    write openers to, and that must crash at startup rather than surface later as a per-
    profile "failed to persist opener spend record" warning that nobody acts on."""
    real_connect = sqlite3.connect
    opened = []

    class _AngleAlterFails:
        """Passes everything through except the openers.angle ALTER, which fails with an
        error that is NOT the benign duplicate-column one."""

        def __init__(self, con):
            self._con = con

        def execute(self, sql, *args, **kwargs):
            if "ALTER TABLE openers ADD COLUMN angle" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return self._con.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._con, name)

    def fake_connect(*args, **kwargs):
        con = real_connect(*args, **kwargs)
        opened.append(con)
        return _AngleAlterFails(con)

    monkeypatch.setattr(sqlite3, "connect", fake_connect)

    try:
        SQLiteStore(tmp_path / "store.db")
        raise AssertionError("expected the non-duplicate-column OperationalError to propagate")
    except sqlite3.OperationalError as e:
        assert "disk i/o error" in str(e).lower()
    finally:
        for con in opened:
            con.close()


def test_sqlite_record_opener_rejection_persists_all_columns(tmp_path):
    """opener_rejections is the durable record of every REJECTED opener attempt (not just
    the successes `openers` holds) -- see ranker/store.py's schema and opener/service.py's
    OpenerParseError handling."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener_rejection("r", "hinge", "gemini-x", 3, "scaffolding",
                                      "Gemini's opener contained scaffolding text",
                                      "Here's: hi")

        row = store.con.execute(
            "SELECT run_id, app, model, attempt, reason_code, reason, raw_opener "
            "FROM opener_rejections"
        ).fetchone()
        assert row == ("r", "hinge", "gemini-x", 3, "scaffolding",
                       "Gemini's opener contained scaffolding text", "Here's: hi")
    finally:
        store.close()


def test_sqlite_record_opener_rejection_accepts_none_raw_opener_and_reason_code(tmp_path):
    """no_text/max_tokens rejections have no candidate opener text to show; None must
    round-trip as NULL, not raise."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener_rejection("r", "hinge", "gemini-x", 1, None, "no text content", None)

        row = store.con.execute(
            "SELECT reason_code, raw_opener FROM opener_rejections"
        ).fetchone()
        assert row == (None, None)
    finally:
        store.close()


def test_sqlite_spend_today_sums_only_local_today(tmp_path):
    """spend_today() gates budget.day_budget_usd -- it must sum only rows since local
    midnight, excluding anything from a prior local day."""
    from operation_love.costing import Usage
    store = SQLiteStore(tmp_path / "store.db")
    try:
        usage = Usage(input_tokens=1, output_tokens=1, cache_read_input_tokens=0,
                      cache_creation_input_tokens=0)
        store.record_spend("r", "gemini-x", usage, 0.05)
        store.record_spend("r", "gemini-x", usage, 0.02)
        # Backdate one row well outside any local day (25h) so it must be excluded.
        store.con.execute(
            "UPDATE spend SET created_at = ? WHERE id = (SELECT MIN(id) FROM spend)",
            (time.time() - 90_000,),
        )
        store.con.commit()

        assert abs(store.spend_today() - 0.02) < 1e-9
    finally:
        store.close()


def test_sqlite_record_profile_accepts_capture_truncated(tmp_path):
    """SQLiteStore doesn't archive images (see record_profile's own comment) so there's
    no profiles manifest row here to persist capture_truncated on -- this just proves
    the keyword (which the worker now always passes) doesn't blow up with a TypeError
    against this backend."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        assert store.record_profile("r", "hinge", "profile-1", True, capture_truncated=True) is True
    finally:
        store.close()


def test_sqlite_remove_latest_training_label_reports_its_profile_name(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.add_label("r", "hinge", True, [1.0], profile_id="one", profile_name="Ada")
        store.add_label("r", "hinge", False, [2.0], profile_id="two", profile_name="Bea")

        assert store.remove_latest_training_label() == {"profile_name": "Bea", "profile_id": "two"}
        assert store.load_labels() == [(True, [1.0])]
    finally:
        store.close()


class _LockWatchingConnection:
    """Forwards to the real connection, counting executes issued without the store lock."""

    def __init__(self, con, lock):
        self._con = con
        self._store_lock = lock
        self.unlocked_executes = []

    def execute(self, sql, *args, **kwargs):
        if not self._store_lock.locked():
            self.unlocked_executes.append(" ".join(str(sql).split()))
        return self._con.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._con, name)


def test_sqlite_advisory_opener_run_rows_reads_entirely_under_the_store_lock(tmp_path):
    """Every read of the shared connection must happen while self._lock is held.

    This class shares ONE sqlite3 connection across worker threads on
    check_same_thread=False, and the lock is the whole of that safety argument (see
    __init__). A count issued after the with-block released it is therefore not a
    style detail: it is the unsynchronized use that combination forbids.
    """
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.add_label("r", "hinge", True, [0.1], profile_id="one", profile_name="Ada")
        store.add_label("r", "hinge", False, [0.2], profile_id="two", profile_name="Bea")
        store.record_decision("r", "hinge", "like", 0.9, source="manual", profile_id="one")
        store.record_opener("r", "hinge", "m", "hello there", "a book")

        watched = _LockWatchingConnection(store.con, store._lock)
        store.con = watched
        rows = store.advisory_opener_run_rows("r", "hinge")
        store.con = watched._con

        assert watched.unlocked_executes == []
        # The counts still have to be the real ones, so the assertion above cannot be
        # satisfied by simply not reading anything.
        assert rows["preference_counts"] == {"profiles": 0, "profile_photos": 0,
                                             "labels": 2, "decisions": 1}
        assert rows["effective_counts"] == {"like_labels": 1, "like_decisions": 1,
                                            "pass_labels": 1, "pass_decisions": 0}
        assert len(rows["openers"]) == 1
    finally:
        store.close()


def test_stats_show_uses_read_only_store(tmp_path, monkeypatch):
    """stats.show() only reads; it must ask make_store() for ensure=False so it doesn't
    run BigQuery table DDL / bucket IAM patching just to print a readout (hub.py and
    tools/eval_aggregation.py already pass ensure=False for their read-only paths)."""
    from operation_love import config as cfg_mod
    from operation_love import stats

    cfg = cfg_mod.Config(
        enabled_apps=["bumble"], mode="auto", apps={}, limits={},
        data_dir=tmp_path, db_file=tmp_path / "s.db",
        ranker=cfg_mod.RankerCfg(), quality_filter=cfg_mod.QualityCfg(),
        opener=cfg_mod.OpenerCfg(enabled=False),
        budget=cfg_mod.BudgetCfg(pricing={}),
        pacing=cfg_mod.PacingCfg(),
        storage=cfg_mod.StorageCfg(backend="sqlite"),
    )
    monkeypatch.setattr(cfg_mod, "load", lambda path: cfg)

    seen = {}
    real_store = SQLiteStore(tmp_path / "s.db")

    def fake_make_store(passed_cfg, ensure=True):
        seen["ensure"] = ensure
        return real_store

    monkeypatch.setattr(stats, "make_store", fake_make_store)

    stats.show("unused.yaml")

    assert seen["ensure"] is False
