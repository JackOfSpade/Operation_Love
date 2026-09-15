"""SQLiteStore source-tagging and daily-limit behavior."""
import inspect
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
                        "decision_source", "decision_created_at", "model_item_index",
                        "prompt_sha256", "profile_key"]
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


def test_sqlite_record_opener_persists_the_prompt_era_stamp(tmp_path):
    """`prompt_sha256` is the digest of the prompt era the row was generated under (see
    prompt_stamp in opener/opener.py). Without it, splitting rows by era means reading
    created_at against config.yaml's git commit dates by hand -- a reconstruction, not a
    record. Keyword-only with a None default, so the seven-positional call shape is unchanged
    and a caller that predates the stamp writes NULL rather than a wrong era."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("r", "hinge", "gemini-x", "hello", "her ridgeline photo", "guess",
                            "a photo", prompt_sha256="a" * 64)

        assert store.con.execute("SELECT prompt_sha256 FROM openers").fetchone() == ("a" * 64,)
    finally:
        store.close()


def test_sqlite_record_opener_leaves_the_prompt_stamp_null_when_the_caller_omits_it(tmp_path):
    """NULL means "this row predates the stamp", which is exactly what an unstamped caller
    should record. "" would claim an era whose digest is the empty string."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("r", "hinge", "gemini-x", "hello", "her ridgeline photo")

        assert store.con.execute("SELECT prompt_sha256 FROM openers").fetchone() == (None,)
    finally:
        store.close()


def test_sqlite_record_opener_rejection_persists_the_prompt_era_stamp(tmp_path):
    """Rejections carry the same era digest the successes do, so "how often does this guard
    fire" is attributable to the prompt that provoked it rather than to a date range."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener_rejection("r", "hinge", "gemini-x", 2, "scaffolding",
                                      "Gemini's opener contained scaffolding text",
                                      "Here's: hi", prompt_sha256="b" * 64)

        row = store.con.execute(
            "SELECT attempt, reason_code, prompt_sha256 FROM opener_rejections").fetchone()
        assert row == (2, "scaffolding", "b" * 64)
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


def test_sqlite_migrates_legacy_openers_prompt_sha256_column(tmp_path):
    """The production case for the era stamp: a db file written before 2026-09-05 (b), when
    `openers` had every column up to model_item_index but not `prompt_sha256`. CREATE TABLE IF
    NOT EXISTS is a no-op against it, so the ALTER is the ONLY thing that carries the column
    in -- without it record_opener's INSERT fails on every opener generated, surfacing only as
    a per-profile "failed to persist" warning."""
    db = tmp_path / "store.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE openers ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,"
        "model TEXT, opener TEXT, referenced TEXT, angle TEXT, item_description TEXT,"
        "profile_id TEXT, decision TEXT, decision_source TEXT, decision_created_at REAL,"
        "model_item_index INTEGER)"
    )
    con.execute(
        "INSERT INTO openers (run_id, app, created_at, model, opener, referenced, angle)"
        " VALUES (?,?,?,?,?,?,?)",
        ("legacy", "hinge", 1.0, "gemini-old", "an opener from before the stamp", "ref",
         "guess"),
    )
    con.commit()
    con.close()

    store = SQLiteStore(db)
    try:
        cols = [row[1] for row in store.con.execute("PRAGMA table_info(openers)").fetchall()]
        assert "prompt_sha256" in cols

        store.record_opener("r", "hinge", "gemini-x", "a new opener", "ref", "imagine",
                            "a prompt card", prompt_sha256="c" * 64)
        rows = store.con.execute(
            "SELECT opener, prompt_sha256 FROM openers ORDER BY id").fetchall()
        # The pre-existing row keeps NULL (ALTER ... ADD COLUMN backfills nothing) and NULL is
        # the load-bearing value: it means "predates the stamp", so an offline pass knows to
        # date that row against git rather than to group it with any era.
        assert rows == [("an opener from before the stamp", None),
                        ("a new opener", "c" * 64)]
    finally:
        store.close()


def test_sqlite_migrates_legacy_opener_rejections_prompt_sha256_column(tmp_path):
    """opener_rejections' FIRST added column ever, so this is also the first test that the
    table is migrated at all rather than only created. Same two-part rule as every openers
    column: the _SCHEMA line reaches a fresh database, the ALTER reaches an existing one, and
    only the ALTER is what a live db file ever sees."""
    db = tmp_path / "store.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE opener_rejections ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,"
        "model TEXT, attempt INTEGER, reason_code TEXT, reason TEXT, raw_opener TEXT)"
    )
    con.execute(
        "INSERT INTO opener_rejections (run_id, app, created_at, model, attempt, reason_code,"
        " reason, raw_opener) VALUES (?,?,?,?,?,?,?,?)",
        ("legacy", "hinge", 1.0, "gemini-old", 1, "scaffolding", "scaffolding text",
         "Here's: hi"),
    )
    con.commit()
    con.close()

    store = SQLiteStore(db)
    try:
        cols = [row[1] for row in
                store.con.execute("PRAGMA table_info(opener_rejections)").fetchall()]
        assert "prompt_sha256" in cols

        store.record_opener_rejection("r", "hinge", "gemini-x", 3, "too_many_sentences",
                                      "three sentences", "one. two. three.",
                                      prompt_sha256="d" * 64)
        rows = store.con.execute(
            "SELECT reason_code, prompt_sha256 FROM opener_rejections ORDER BY id").fetchall()
        assert rows == [("scaffolding", None), ("too_many_sentences", "d" * 64)]
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


def test_sqlite_opener_rejections_prompt_sha256_migration_reraises_unexpected_operational_errors(
        tmp_path, monkeypatch):
    """Same fail-loud contract as the item_description pin directly above, pinned separately for
    the reason that pin's own docstring gives: each ALTER carries its own try/except, so a NEW
    stanza can be written with a bare `except sqlite3.OperationalError: pass` and no existing
    test would notice. This one is the new stanza (2026-09-05 (b), opener_rejections' first
    migration ever); the openers half of the same change rides the existing ALTER loop, whose
    guard is pinned by test_sqlite_openers_alter_loop_reraises_unexpected_operational_errors
    directly below -- the angle and item_description pins do NOT reach it, because those are
    standalone ALTERs with try/excepts of their own and the loop's handler is a third one.
    Swallowing a disk I/O
    error here would start the store up against a schema it cannot write prompt_sha256 to, and
    every subsequent insert would fail one row at a time instead of once at boot."""
    real_connect = sqlite3.connect
    opened = []

    class _RejectionStampAlterFails:
        def __init__(self, con):
            self._con = con

        def execute(self, sql, *args, **kwargs):
            if "ALTER TABLE opener_rejections ADD COLUMN prompt_sha256" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return self._con.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._con, name)

    def fake_connect(*args, **kwargs):
        con = real_connect(*args, **kwargs)
        opened.append(con)
        return _RejectionStampAlterFails(con)

    monkeypatch.setattr(sqlite3, "connect", fake_connect)

    try:
        SQLiteStore(tmp_path / "store.db")
        raise AssertionError("expected the non-duplicate-column OperationalError to propagate")
    except sqlite3.OperationalError as e:
        assert "disk i/o error" in str(e).lower()
    finally:
        for con in opened:
            con.close()


def test_sqlite_openers_alter_loop_reraises_unexpected_operational_errors(tmp_path, monkeypatch):
    """The `openers` ALTER LOOP's own handler, which no other pin in this file reaches.

    The angle and item_description pins above each intercept a STANDALONE ALTER with its own
    try/except; the loop that adds profile_id, decision, decision_source, decision_created_at,
    model_item_index and prompt_sha256 has a THIRD handler, and rewriting that one as a bare
    `except sqlite3.OperationalError: pass` left this file entirely green before this test
    existed. So the loop is pinned here through one of the columns it actually emits
    (model_item_index), and the claim in the rejection pin above -- that the openers half of
    the 2026-09-05 (b) stamp inherits a guard that is already covered -- is true because of
    this test rather than in spite of the missing one.

    Same contract as every sibling: swallow ONLY "duplicate column name" (the already-migrated
    case). Swallowing a disk I/O error instead would start the store against a schema it cannot
    write those columns to, turning one loud failure at boot into a silent failure per insert.
    """
    real_connect = sqlite3.connect
    opened = []

    class _LoopAlterFails:
        def __init__(self, con):
            self._con = con

        def execute(self, sql, *args, **kwargs):
            if "ALTER TABLE openers ADD COLUMN model_item_index" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return self._con.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._con, name)

    def fake_connect(*args, **kwargs):
        con = real_connect(*args, **kwargs)
        opened.append(con)
        return _LoopAlterFails(con)

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


def test_sqlite_advisory_opener_run_rows_keep_the_prompt_stamp_out_of_the_fingerprint(tmp_path):
    """`prompt_sha256` must NEVER enter the opener fingerprint or the advisory projection.

    Five on-disk cleanup plans in ops/corrections/ carry fingerprints computed from exactly
    five fields (run_id, app, created_at, model, opener) and a three-key emitted row. Adding
    the new column to either side would re-key every one of those plans against rows they
    already name, and append_opener_retraction would then refuse them as "fingerprint no
    longer matches its cleanup plan" -- a stored plan that can never be applied again.

    Both halves are asserted together: the emitted keys are exactly the three, and a STAMPED
    row fingerprints identically to the same logical row written unstamped."""
    stamped = SQLiteStore(tmp_path / "stamped.db")
    plain = SQLiteStore(tmp_path / "plain.db")
    try:
        stamped.record_opener("r", "hinge", "gemini-x", "hello there", "a book",
                              prompt_sha256="e" * 64)
        rows = stamped.advisory_opener_run_rows("r", "hinge")["openers"]
        assert len(rows) == 1
        assert set(rows[0]) == {"created_at", "model", "opener_fingerprint"}

        # Same logical row, no stamp, and its created_at forced to the stamped row's so the
        # only difference left between the two is the new column.
        plain.record_opener("r", "hinge", "gemini-x", "hello there", "a book")
        plain.con.execute("UPDATE openers SET created_at=?",
                          (float(rows[0]["created_at"]),))
        plain.con.commit()
        unstamped = plain.advisory_opener_run_rows("r", "hinge")["openers"]
        assert unstamped[0]["opener_fingerprint"] == rows[0]["opener_fingerprint"]
    finally:
        stamped.close()
        plain.close()


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


# =====================================================================================
# profile_key / opener_outcomes (2026-09-06): the outcome-signal data layer.
#
# `profile_key` is the STABLE cross-time attribution key (ranker/profile_key.py); `openers`'
# per-card `profile_id` stays exactly what it always was and is untouched by any test below.
# `opener_outcomes` is a brand-new table joined back to `openers` by (app, profile_key) --
# see ranker/store.py's `joined_opener_outcomes`.
# =====================================================================================

def test_sqlite_record_opener_persists_profile_key(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("r", "hinge", "gemini-x", "hi there", "her photo",
                            profile_key="k" * 64)

        assert store.con.execute("SELECT profile_key FROM openers").fetchone() == ("k" * 64,)
    finally:
        store.close()


def test_sqlite_record_opener_defaults_profile_key_to_empty_string_for_callers_that_omit_it(
        tmp_path):
    """"" (not NULL) is "no key could be derived for this card," matching `profile_id`'s own
    empty-string convention right next to it. NULL is reserved for rows that predate this
    column entirely -- see test_sqlite_migrates_legacy_openers_profile_key_column."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("r", "hinge", "gemini-x", "hi there", "her photo")

        assert store.con.execute("SELECT profile_key FROM openers").fetchone() == ("",)
    finally:
        store.close()


def test_sqlite_migrates_legacy_openers_profile_key_column(tmp_path):
    """The production case for the attribution key: a db file written before 2026-09-06, when
    `openers` had every column up to `prompt_sha256` but not `profile_key`. CREATE TABLE IF NOT
    EXISTS is a no-op against it, so the ALTER is the ONLY thing that carries the column in --
    without it, record_opener's INSERT fails on every opener generated once a caller starts
    passing profile_key, surfacing only as a per-profile "failed to persist" warning."""
    db = tmp_path / "store.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE openers ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,"
        "model TEXT, opener TEXT, referenced TEXT, angle TEXT, item_description TEXT,"
        "profile_id TEXT, decision TEXT, decision_source TEXT, decision_created_at REAL,"
        "model_item_index INTEGER, prompt_sha256 TEXT)"
    )
    con.execute(
        "INSERT INTO openers (run_id, app, created_at, model, opener, referenced, angle)"
        " VALUES (?,?,?,?,?,?,?)",
        ("legacy", "hinge", 1.0, "gemini-old", "an opener from before profile_key", "ref",
         "guess"),
    )
    con.commit()
    con.close()

    store = SQLiteStore(db)
    try:
        cols = [row[1] for row in store.con.execute("PRAGMA table_info(openers)").fetchall()]
        assert "profile_key" in cols

        store.record_opener("r", "hinge", "gemini-x", "a new opener", "ref", "imagine",
                            "a prompt card", profile_key="k" * 64)
        rows = store.con.execute(
            "SELECT opener, profile_key FROM openers ORDER BY id").fetchall()
        # The pre-existing row keeps NULL (ALTER ... ADD COLUMN backfills nothing) and NULL is
        # the load-bearing value: it means "predates this column," distinct from a post-
        # migration row that legitimately had no derivable key (which writes "").
        assert rows == [("an opener from before profile_key", None),
                        ("a new opener", "k" * 64)]
    finally:
        store.close()


def test_sqlite_openers_profile_key_migration_reraises_unexpected_operational_errors(
        tmp_path, monkeypatch):
    """Same fail-loud contract as every other stanza in the ALTER loop (see
    test_sqlite_openers_alter_loop_reraises_unexpected_operational_errors, which pins the
    loop's shared handler through `model_item_index`): swallow ONLY "duplicate column name."
    Pinned separately through `profile_key` specifically because it is the newest column added
    to the loop and the easiest one for a future edit to accidentally exempt from the shared
    handler by pulling it out into its own bare `except sqlite3.OperationalError: pass`."""
    real_connect = sqlite3.connect
    opened = []

    class _ProfileKeyAlterFails:
        def __init__(self, con):
            self._con = con

        def execute(self, sql, *args, **kwargs):
            if "ALTER TABLE openers ADD COLUMN profile_key" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return self._con.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._con, name)

    def fake_connect(*args, **kwargs):
        con = real_connect(*args, **kwargs)
        opened.append(con)
        return _ProfileKeyAlterFails(con)

    monkeypatch.setattr(sqlite3, "connect", fake_connect)

    try:
        SQLiteStore(tmp_path / "store.db")
        raise AssertionError("expected the non-duplicate-column OperationalError to propagate")
    except sqlite3.OperationalError as e:
        assert "disk i/o error" in str(e).lower()
    finally:
        for con in opened:
            con.close()


def test_sqlite_record_opener_persists_profile_key_as_the_hash_never_the_raw_fingerprint(
        tmp_path):
    """`profile_key` must be exactly the hex SHA-256 `ranker.profile_key` computes -- never the
    raw `ProfileIdentity.fingerprint` those pixels came from. A stored fingerprint would be a
    real, if low-resolution, rendering of a real person's name sitting in a database whose
    whole purpose is aggregate calibration (see ranker/profile_key.py's own docstring)."""
    from operation_love.drivers.item_identity import ProfileIdentity
    from operation_love.ranker.profile_key import profile_key_from_identity

    identity = ProfileIdentity(fingerprint=(10, 20, 30, 40), band=(0.0, 0.0, 1.0, 1.0),
                               grid=(64, 16), frame_index=0, scroll_top_distance=0.0,
                               reason="settled header")
    key = profile_key_from_identity(identity)

    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("r", "hinge", "gemini-x", "hi there", "her photo", profile_key=key)

        stored = store.con.execute("SELECT profile_key FROM openers").fetchone()[0]
        assert stored == key
        assert len(stored) == 64 and all(c in "0123456789abcdef" for c in stored)
        assert stored != str(identity.fingerprint)
    finally:
        store.close()


def test_sqlite_opener_outcomes_table_has_the_expected_columns_in_order(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        cols = [row[1] for row in
                store.con.execute("PRAGMA table_info(opener_outcomes)").fetchall()]
        assert cols == ["id", "app", "profile_key", "outcome", "observed_at", "source", "note",
                        "created_at"]
    finally:
        store.close()


def test_sqlite_creates_the_opener_outcomes_table_in_a_preexisting_database_that_lacks_it(
        tmp_path):
    """`opener_outcomes` is a BRAND NEW table (2026-09-06): CREATE TABLE IF NOT EXISTS alone
    must reach a database file that predates it entirely, with no ALTER needed at all -- unlike
    a new COLUMN on a table that already exists elsewhere (contrast
    test_sqlite_migrates_legacy_openers_profile_key_column just above). Simulates the real
    upgrade case: an existing store.db with `openers` but no `opener_outcomes` table."""
    db = tmp_path / "store.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE openers ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT, created_at REAL,"
        "model TEXT, opener TEXT, referenced TEXT)"
    )
    con.commit()
    con.close()

    store = SQLiteStore(db)
    try:
        tables = {row[0] for row in
                  store.con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "opener_outcomes" in tables

        store.record_opener_outcome("hinge", "k" * 64, "match")
        assert store.con.execute("SELECT COUNT(*) FROM opener_outcomes").fetchone()[0] == 1
    finally:
        store.close()


def test_sqlite_record_opener_outcome_round_trips_all_fields(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener_outcome("hinge", "k" * 64, "reply", observed_at=1000.0,
                                    source="owner", note="she asked about the trail")

        row = store.con.execute(
            "SELECT app, profile_key, outcome, observed_at, source, note "
            "FROM opener_outcomes").fetchone()
        assert row == ("hinge", "k" * 64, "reply", 1000.0, "owner", "she asked about the trail")
    finally:
        store.close()


def test_sqlite_record_opener_outcome_defaults_source_note_and_stamps_observed_at_now(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        before = time.time()
        store.record_opener_outcome("hinge", "k" * 64, "match")
        after = time.time()

        row = store.con.execute(
            "SELECT source, note, observed_at, created_at FROM opener_outcomes").fetchone()
        assert row[0] == "owner"
        assert row[1] == ""
        assert before <= row[2] <= after
        assert before <= row[3] <= after
    finally:
        store.close()


def test_sqlite_record_opener_outcome_observed_at_is_independent_of_created_at(tmp_path):
    """An owner backfilling a match noticed three days ago must be able to say WHEN it
    happened without lying about WHEN the database learned it."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        three_days_ago = time.time() - (3 * 86400)
        before_write = time.time()
        store.record_opener_outcome("hinge", "k" * 64, "match", observed_at=three_days_ago)

        row = store.con.execute(
            "SELECT observed_at, created_at FROM opener_outcomes").fetchone()
        assert row[0] == three_days_ago
        assert row[1] >= before_write
        assert row[1] != row[0]
    finally:
        store.close()


def test_sqlite_record_opener_outcome_stores_an_unattributable_observation_rather_than_dropping_it(
        tmp_path):
    """An owner who observed a real outcome but could not pin down which captured profile it
    belongs to must still have the observation LAND. Silently dropping it would be worse than
    storing it unattributed: a dropped row leaves no trace an observation was ever made at
    all (see ranker/profile_key.py's own docstring)."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener_outcome("hinge", "", "unknown", note="lost track of which profile")

        row = store.con.execute(
            "SELECT app, profile_key, outcome, note FROM opener_outcomes").fetchone()
        assert row == ("hinge", "", "unknown", "lost track of which profile")
    finally:
        store.close()


def test_sqlite_joined_opener_outcomes_joins_by_app_and_profile_key(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("r", "hinge", "gemini-x", "hi there", "her photo",
                            decision="like", prompt_sha256="a" * 64, profile_key="k" * 64)
        store.record_opener_outcome("hinge", "k" * 64, "match", note="mutual like")

        rows = store.joined_opener_outcomes("hinge")
        assert len(rows) == 1
        row = rows[0]
        assert row["run_id"] == "r"
        assert row["model"] == "gemini-x"
        assert row["opener"] == "hi there"
        assert row["prompt_sha256"] == "a" * 64
        assert row["profile_key"] == "k" * 64
        assert row["outcome"] == "match"
        assert row["source"] == "owner"
        assert row["note"] == "mutual like"
        assert set(row) == {"run_id", "app", "model", "opener", "prompt_sha256", "profile_key",
                            "opener_created_at", "outcome", "observed_at", "source", "note",
                            "outcome_created_at"}
    finally:
        store.close()


def test_sqlite_joined_opener_outcomes_finds_nothing_when_no_outcome_was_recorded(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("r", "hinge", "gemini-x", "hi there", "her photo",
                            decision="like", profile_key="k" * 64)

        assert store.joined_opener_outcomes("hinge") == []
    finally:
        store.close()


def test_sqlite_joined_opener_outcomes_filters_by_prompt_era(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("r1", "hinge", "gemini-x", "old era opener", "ref",
                            decision="like", prompt_sha256="a" * 64, profile_key="k" * 64)
        store.record_opener("r2", "hinge", "gemini-x", "new era opener", "ref",
                            decision="like", prompt_sha256="b" * 64, profile_key="k" * 64)
        store.record_opener_outcome("hinge", "k" * 64, "match")

        matched = store.joined_opener_outcomes("hinge", prompt_sha256="a" * 64)
        assert [row["opener"] for row in matched] == ["old era opener"]

        no_match = store.joined_opener_outcomes("hinge", prompt_sha256="c" * 64)
        assert no_match == []

        # prompt_sha256=None (the default) means "no filter, every era" -- never "match NULL."
        unfiltered = store.joined_opener_outcomes("hinge")
        assert {row["opener"] for row in unfiltered} == {"old era opener", "new era opener"}
    finally:
        store.close()


def test_sqlite_joined_opener_outcomes_excludes_empty_string_profile_key_on_both_sides(
        tmp_path):
    """An unattributable opener and an unattributable outcome must never join to EACH OTHER
    just because both happen to carry the same "no key" spelling (""). That would attribute a
    real observation to an unrelated opener that also lacked identity -- a worse failure than
    the outcome staying unjoined."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("r", "hinge", "gemini-x", "unattributed opener", "ref",
                            decision="like", profile_key="")
        store.record_opener_outcome("hinge", "", "unknown")

        assert store.joined_opener_outcomes("hinge") == []
    finally:
        store.close()


def test_sqlite_joined_opener_outcomes_excludes_null_profile_key_openers(tmp_path):
    """A NULL `openers.profile_key` means "this row predates the column" (see the ALTER
    migration) -- it must never satisfy the join just because SQL lets a comparison against
    NULL be worked around. Simulates a legacy row inserted before this migration ever ran."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.con.execute(
            "INSERT INTO openers (run_id, app, created_at, model, opener, referenced,"
            " decision, profile_key) VALUES (?,?,?,?,?,?,?,?)",
            # decision='like' so this row fails the join on its NULL profile_key ALONE -- the
            # rule under test -- rather than on the sent-only predicate it would otherwise trip.
            ("legacy", "hinge", 1.0, "gemini-old", "predates profile_key", "ref", "like", None),
        )
        store.con.commit()
        store.record_opener_outcome("hinge", "", "unknown")

        assert store.joined_opener_outcomes("hinge") == []
    finally:
        store.close()


def test_sqlite_joined_opener_outcomes_does_not_cross_app_boundaries(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("r", "hinge", "gemini-x", "hinge opener", "ref",
                            decision="like", profile_key="k" * 64)
        store.record_opener_outcome("bumble", "k" * 64, "match")  # same key, different app

        assert store.joined_opener_outcomes("hinge") == []
        assert store.joined_opener_outcomes("bumble") == []
    finally:
        store.close()


def test_sqlite_joined_opener_outcomes_returns_every_outcome_in_created_order(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("r", "hinge", "gemini-x", "hi there", "ref", decision="like",
                            profile_key="k" * 64)
        store.record_opener_outcome("hinge", "k" * 64, "match", observed_at=100.0)
        store.record_opener_outcome("hinge", "k" * 64, "reply", observed_at=200.0)

        rows = store.joined_opener_outcomes("hinge")
        assert [row["outcome"] for row in rows] == ["match", "reply"]
        assert [row["observed_at"] for row in rows] == [100.0, 200.0]
    finally:
        store.close()


def test_sqlite_record_opener_signature_ends_with_profile_key_as_a_trailing_keyword(tmp_path):
    """`profile_key` is threaded exactly like `prompt_sha256` immediately before it: keyword-
    only, trailing, defaulted, so no existing positional caller (or Protocol-conforming test
    double) changes shape."""
    params = inspect.signature(SQLiteStore.record_opener).parameters
    assert list(params)[-1] == "profile_key"
    assert params["profile_key"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["profile_key"].default == ""
    assert list(params) == list(inspect.signature(Store.record_opener).parameters)


def test_sqlite_record_opener_outcome_signature_matches_the_store_protocol():
    params = inspect.signature(SQLiteStore.record_opener_outcome).parameters
    assert list(params) == list(inspect.signature(Store.record_opener_outcome).parameters)
    assert params["app"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["profile_key"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["outcome"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("observed_at", "source", "note"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_sqlite_joined_opener_outcomes_signature_matches_the_store_protocol():
    params = inspect.signature(SQLiteStore.joined_opener_outcomes).parameters
    assert list(params) == list(inspect.signature(Store.joined_opener_outcomes).parameters)
    assert params["prompt_sha256"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["prompt_sha256"].default is None


def test_sqlite_joined_opener_outcomes_credits_only_the_opener_that_was_actually_sent(tmp_path):
    """One outcome, one credited opener -- the SENT one.

    The incident this pins: a profile is drafted-then-DISLIKED in one run (a real `openers`
    row, real profile_key, decision='dislike', nothing ever typed on the phone), reappears in a
    later run under a NEWER prompt era, is LIKED, and the owner records ONE match. The join used
    to return that match TWICE, the extra copy crediting the older era with a match earned by a
    draft nobody ever saw -- which is exactly the measurement the era comparison exists to make.
    'like' is the one decision value that means "this opener reached the device" (the literal
    tools/opener_outcome_recorder.py calls DECISION_SENT).
    """
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_opener("run-a", "hinge", "gemini-x", "never sent draft", "ref",
                            decision="dislike", prompt_sha256="a" * 64, profile_key="k" * 64)
        store.record_opener("run-b", "hinge", "gemini-x", "the one she got", "ref",
                            decision="like", prompt_sha256="b" * 64, profile_key="k" * 64)
        store.record_opener_outcome("hinge", "k" * 64, "match")

        rows = store.joined_opener_outcomes("hinge")
        assert [row["opener"] for row in rows] == ["the one she got"]
        assert [row["prompt_sha256"] for row in rows] == ["b" * 64]
        # ...and the drafted era can no longer claim it even when asked for by name.
        assert store.joined_opener_outcomes("hinge", prompt_sha256="a" * 64) == []
    finally:
        store.close()


@pytest.mark.parametrize("decision", ["dislike", "never_sent", "synthetic_replay", "", None],
                         ids=["dislike", "never-sent", "synthetic-replay", "empty", "null"])
def test_sqlite_joined_opener_outcomes_excludes_every_unsent_decision_value(tmp_path, decision):
    """Everything that is not 'like' means "never sent," including the NULL/'' spellings that
    predate decision tracking. A read that feeds prompt-era comparisons must never guess a send
    out of a row that does not record one."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.con.execute(
            "INSERT INTO openers (run_id, app, created_at, model, opener, referenced,"
            " decision, profile_key) VALUES (?,?,?,?,?,?,?,?)",
            ("r", "hinge", 1.0, "gemini-x", "not sent", "ref", decision, "k" * 64),
        )
        store.con.commit()
        store.record_opener_outcome("hinge", "k" * 64, "match")

        assert store.joined_opener_outcomes("hinge") == []
    finally:
        store.close()


def test_sqlite_dropped_rows_is_empty_because_this_backend_cannot_lose_a_row(tmp_path):
    """`{}` here is a FACT, not an unimplemented stub: SQLite writes inside the caller's own
    call and a rejected INSERT raises out of it, so there is no buffer for a row to be parked
    in, retried, and eventually given up on (which is exactly what BigQueryStore does, and why
    the supervisor has to ask). Both backends answer, so no caller needs to know which it holds.
    """
    store = SQLiteStore(tmp_path / "store.db")
    try:
        assert store.dropped_rows() == {}
        store.record_opener("r", "hinge", "gemini-x", "hi there", "ref", decision="like",
                            profile_key="k" * 64)
        store.flush()
        assert store.dropped_rows() == {}
    finally:
        store.close()
