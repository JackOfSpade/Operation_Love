"""tools/opener_outcome_recorder.py -- the owner-facing manual outcome-recording CLI.

No network, ever: the BigQuery tests inject a fake client exactly as
tests/test_opener_corpus_report.py's own read_bigquery_openers tests do. NEVER make a real API
call from a test.
"""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from operation_love.ranker.store import SQLiteStore
from tools import opener_outcome_recorder as m

# ---------------------------------------------------------------------------------------
# local helpers (no conftest.py in this repo)
# ---------------------------------------------------------------------------------------


def _row(*, run_id="r1", app="hinge", created_at=1000.0, model="gemini-x", opener="Hi there.",
        prompt_sha256="era-a", profile_id="pid-1", profile_key="k" * 64, profile_name="",
        decision="like"):
    return m.OpenerRow(run_id=run_id, app=app, created_at=created_at, model=model, opener=opener,
                      prompt_sha256=prompt_sha256, profile_id=profile_id, profile_key=profile_key,
                      profile_name=profile_name, decision=decision)


class _FakeStore:
    """Spies on record_opener_outcome; also a valid flush()/close() target."""

    def __init__(self):
        self.calls = []
        self.flushed = False
        self.closed = False

    def record_opener_outcome(self, app, profile_key, outcome, *, observed_at=None,
                              source="owner", note=""):
        self.calls.append({"app": app, "profile_key": profile_key, "outcome": outcome,
                          "observed_at": observed_at, "source": source, "note": note})

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


class _RaisingStore(_FakeStore):
    def record_opener_outcome(self, *a, **k):
        raise RuntimeError("boom: the store is unavailable")


def _fake_cfg(*, backend="sqlite", db_file="", bigquery=None):
    return SimpleNamespace(storage=SimpleNamespace(backend=backend, bigquery=bigquery or {}),
                          db_file=db_file)


# ---------------------------------------------------------------------------------------
# is_sent / list_sent_openers -- listing shows only sent openers, at their TRUE window position
# ---------------------------------------------------------------------------------------


def test_is_sent_is_true_only_for_the_literal_like_decision():
    assert m.is_sent(_row(decision="like")) is True
    assert m.is_sent(_row(decision="dislike")) is False
    assert m.is_sent(_row(decision="never_sent")) is False
    assert m.is_sent(_row(decision="synthetic_replay")) is False
    assert m.is_sent(_row(decision=None)) is False
    assert m.is_sent(_row(decision="")) is False


def test_list_sent_openers_filters_to_sent_and_keeps_true_window_position():
    rows = [_row(opener="sent one", decision="like"),
           _row(opener="draft, never sent", decision="never_sent"),
           _row(opener="sent two", decision="like"),
           _row(opener="rejected in training", decision="dislike")]

    sent = m.list_sent_openers(rows)

    # Position 2 (the never_sent draft) and position 4 (the dislike) must never appear --
    # this is the pin for "listing shows only sent openers."
    assert [(pos, r.opener) for pos, r in sent] == [(1, "sent one"), (3, "sent two")]


def test_format_list_text_shows_only_sent_openers_with_name_and_era():
    rows = [_row(opener="Where was that taken?", profile_name="Alex", decision="like"),
           _row(opener="a draft nobody sent", profile_name="Should Never Appear",
               decision="never_sent")]

    text = m.format_list_text(rows, app="hinge", registry=None)

    assert "Where was that taken?" in text
    assert "Alex" in text
    assert "Should Never Appear" not in text
    assert "a draft nobody sent" not in text
    assert "[1]" in text


def test_format_list_text_reports_when_nothing_was_sent():
    rows = [_row(decision="never_sent")]
    text = m.format_list_text(rows, app="hinge", registry=None)
    assert "none of the 1 most recent opener row(s)" in text


# ---------------------------------------------------------------------------------------
# resolve_selection -- the one place a write can be refused BEFORE the store is ever touched
# ---------------------------------------------------------------------------------------


def test_resolve_selection_returns_the_row_for_a_valid_sent_index():
    rows = [_row(opener="first", decision="like"), _row(opener="second", decision="like")]
    assert m.resolve_selection(rows, 2).opener == "second"


def test_resolve_selection_refuses_an_index_outside_the_window():
    rows = [_row(decision="like")]
    with pytest.raises(m.SelectionError, match="no opener at position 5"):
        m.resolve_selection(rows, 5)


def test_resolve_selection_refuses_index_zero_and_negative():
    rows = [_row(decision="like")]
    with pytest.raises(m.SelectionError):
        m.resolve_selection(rows, 0)
    with pytest.raises(m.SelectionError):
        m.resolve_selection(rows, -1)


def test_resolve_selection_refuses_a_draft_that_was_never_sent():
    rows = [_row(decision="like"), _row(decision="never_sent")]
    with pytest.raises(m.SelectionError, match="NEVER SENT"):
        m.resolve_selection(rows, 2)


def test_resolve_selection_refusal_names_the_actual_decision_value():
    rows = [_row(decision="dislike")]
    with pytest.raises(m.SelectionError, match="decision='dislike'"):
        m.resolve_selection(rows, 1)


def test_resolve_selection_refusal_for_no_recorded_decision_says_so():
    rows = [_row(decision=None)]
    with pytest.raises(m.SelectionError, match=r"\(no decision recorded\)"):
        m.resolve_selection(rows, 1)


# ---------------------------------------------------------------------------------------
# record_outcome -- writes the RESOLVED row's profile_key, source always "owner"
# ---------------------------------------------------------------------------------------


def test_record_outcome_writes_the_resolved_profile_key_and_owner_source():
    store = _FakeStore()
    row = _row(profile_key="deadbeef" * 8, decision="like")

    m.record_outcome(store, row, app="hinge", outcome="match", note="she messaged back")

    assert store.calls == [{"app": "hinge", "profile_key": "deadbeef" * 8, "outcome": "match",
                           "observed_at": None, "source": "owner", "note": "she messaged back"}]


def test_record_outcome_passes_through_an_empty_profile_key_rather_than_refusing():
    """An unattributable observation is still worth storing -- see
    ranker/__init__.py's Store.record_opener_outcome docstring."""
    store = _FakeStore()
    row = _row(profile_key="", decision="like")

    m.record_outcome(store, row, app="hinge", outcome="no_response")

    assert store.calls[0]["profile_key"] == ""


def test_record_outcome_forwards_observed_at():
    store = _FakeStore()
    row = _row(decision="like")

    m.record_outcome(store, row, app="hinge", outcome="unmatch", observed_at=123.0)

    assert store.calls[0]["observed_at"] == 123.0


# ---------------------------------------------------------------------------------------
# _parse_observed_at
# ---------------------------------------------------------------------------------------


def test_parse_observed_at_none_passes_through_none():
    assert m._parse_observed_at(None) is None


def test_parse_observed_at_numeric_string_becomes_a_float():
    assert m._parse_observed_at("1757000000") == 1757000000.0


def test_parse_observed_at_non_numeric_string_passes_through_unchanged():
    assert m._parse_observed_at("2026-09-06T12:00:00+00:00") == "2026-09-06T12:00:00+00:00"


# ---------------------------------------------------------------------------------------
# read_recent_openers_sqlite -- direct read, name resolution, missing db/table
# ---------------------------------------------------------------------------------------


def test_read_recent_openers_sqlite_resolves_the_training_label_name(tmp_path):
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        # Mirrors worker.py's Training path: the SAME freshly-minted profile_id is used for
        # both the committed opener and the label carrying her name.
        store.record_opener("run1", "hinge", "gemini-x", "Nice trail shot.", "the ridge",
                            profile_id="pid-alex", decision="like", profile_key="k" * 64)
        store.add_label("run1", "hinge", True, [0.1, 0.2], source="manual",
                        profile_id="pid-alex", profile_name="Alex")
    finally:
        store.close()

    rows, notes = m.read_recent_openers_sqlite(db_path, "hinge", limit=10)

    assert notes == []
    assert len(rows) == 1
    assert rows[0].profile_name == "Alex"
    assert rows[0].profile_key == "k" * 64
    assert rows[0].decision == "like"


def test_read_recent_openers_sqlite_leaves_name_empty_for_auto_mode_openers(tmp_path):
    """AUTO never archives training data (worker.py's own comment), so a committed AUTO opener
    has no label to join to and must read back with no name -- never a wrong or borrowed one."""
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("run1", "hinge", "gemini-x", "Where was this?", "the view",
                            profile_id="pid-auto", decision="like", profile_key="k" * 64)
    finally:
        store.close()

    rows, _notes = m.read_recent_openers_sqlite(db_path, "hinge", limit=10)

    assert rows[0].profile_name == ""


def test_read_recent_openers_sqlite_orders_most_recent_first(tmp_path):
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("run1", "hinge", "m", "older", "x", decision="like")
        store.record_opener("run1", "hinge", "m", "newer", "x", decision="like")
    finally:
        store.close()
    # created_at is `time.time()` at insert; force a deterministic order regardless of clock
    # resolution on a fast machine.
    con = sqlite3.connect(db_path)
    con.execute("UPDATE openers SET created_at = 100.0 WHERE opener = 'older'")
    con.execute("UPDATE openers SET created_at = 200.0 WHERE opener = 'newer'")
    con.commit()
    con.close()

    rows, _notes = m.read_recent_openers_sqlite(db_path, "hinge", limit=10)

    assert [r.opener for r in rows] == ["newer", "older"]


def test_read_recent_openers_sqlite_respects_limit(tmp_path):
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        for i in range(5):
            store.record_opener("run1", "hinge", "m", f"opener {i}", "x", decision="like")
    finally:
        store.close()

    rows, _notes = m.read_recent_openers_sqlite(db_path, "hinge", limit=2)

    assert len(rows) == 2


def test_read_recent_openers_sqlite_scopes_to_the_given_app(tmp_path):
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("run1", "hinge", "m", "hinge opener", "x", decision="like")
        store.record_opener("run1", "bumble", "m", "bumble opener", "x", decision="like")
    finally:
        store.close()

    rows, _notes = m.read_recent_openers_sqlite(db_path, "hinge", limit=10)

    assert [r.opener for r in rows] == ["hinge opener"]


def test_read_recent_openers_sqlite_missing_db_reports_a_note_not_a_crash(tmp_path):
    rows, notes = m.read_recent_openers_sqlite(tmp_path / "does_not_exist.db", "hinge", limit=10)
    assert rows == []
    assert any("no database file" in n for n in notes)


def test_read_recent_openers_sqlite_missing_openers_table_reports_a_note(tmp_path):
    db_path = tmp_path / "empty.db"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE decisions (id INTEGER PRIMARY KEY)")
    con.commit()
    con.close()

    rows, notes = m.read_recent_openers_sqlite(db_path, "hinge", limit=10)

    assert rows == []
    assert any("no 'openers' table" in n for n in notes)


def test_read_recent_openers_sqlite_legacy_schema_without_decision_column_reads_as_never_sent(
        tmp_path):
    db_path = tmp_path / "legacy.db"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE openers (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, "
               "app TEXT, created_at REAL, model TEXT, opener TEXT, referenced TEXT)")
    con.execute("INSERT INTO openers (run_id, app, created_at, model, opener, referenced) "
               "VALUES ('r','hinge',1.0,'m','a legacy opener','ref')")
    con.commit()
    con.close()

    rows, notes = m.read_recent_openers_sqlite(db_path, "hinge", limit=10)

    assert len(rows) == 1
    assert rows[0].decision is None
    assert m.is_sent(rows[0]) is False
    assert any("predates the decision column" in n for n in notes)


# ---------------------------------------------------------------------------------------
# read_recent_openers_bigquery -- fake client, no network (matches
# tests/test_opener_corpus_report.py's own BigQuery test convention)
# ---------------------------------------------------------------------------------------


class _FakeBQJob:
    def __init__(self, rows):
        self._rows = rows

    def result(self):
        return self._rows


class _Row(dict):
    def __getitem__(self, key):
        return super().__getitem__(key)


class _FakeBQClient:
    def __init__(self, rows=None, *, error=None):
        self.rows = rows if rows is not None else []
        self.error = error
        self.calls = []

    def query(self, sql, job_config=None):
        self.calls.append({"sql": sql, "job_config": job_config})
        if self.error is not None:
            raise self.error
        return _FakeBQJob([_Row(r) for r in self.rows])


def test_read_recent_openers_bigquery_reads_rows_with_a_fake_client():
    client = _FakeBQClient([
        {"run_id": "r1", "app": "hinge", "created_at": 200.0, "model": "gemini-x",
         "opener": "Sent one.", "prompt_sha256": "era-a", "profile_id": "pid-1",
         "profile_key": "k" * 64, "profile_name": "Alex", "decision": "like"},
    ])

    rows, notes = m.read_recent_openers_bigquery("proj-1", "operation_love", "hinge", limit=25,
                                                 client=client)

    assert notes == []
    assert len(rows) == 1
    assert rows[0].opener == "Sent one."
    assert rows[0].profile_name == "Alex"
    assert rows[0].decision == "like"
    assert client.calls[0]["job_config"] is not None


def test_read_recent_openers_bigquery_missing_table_reports_a_note_not_a_crash():
    client = _FakeBQClient(error=RuntimeError("NotFound: table not found"))

    rows, notes = m.read_recent_openers_bigquery("proj-1", "operation_love", "hinge", limit=25,
                                                 client=client)

    assert rows == []
    assert any("could not read" in n for n in notes)


def test_read_recent_openers_bigquery_rejects_an_invalid_project_id():
    rows, notes = m.read_recent_openers_bigquery("not a valid id!", "operation_love", "hinge",
                                                 limit=25, client=_FakeBQClient([]))
    assert rows == []
    assert any("invalid BigQuery config" in n for n in notes)


def test_read_recent_openers_dispatches_to_bigquery_when_backend_is_bigquery():
    client = _FakeBQClient([
        {"run_id": "r1", "app": "hinge", "created_at": 200.0, "model": "gemini-x",
         "opener": "BQ opener.", "prompt_sha256": None, "profile_id": "", "profile_key": "",
         "profile_name": "", "decision": "like"},
    ])
    cfg = _fake_cfg(backend="bigquery", bigquery={"project_id": "proj-1", "dataset": "operation_love"})

    rows, notes = m.read_recent_openers(cfg, "hinge", limit=25, db_path=None, backend="auto",
                                        bigquery_client=client)

    assert notes == []
    assert rows[0].opener == "BQ opener."


def test_read_recent_openers_bigquery_without_project_id_reports_a_note():
    cfg = _fake_cfg(backend="bigquery", bigquery={})
    rows, notes = m.read_recent_openers(cfg, "hinge", limit=25, db_path=None, backend="auto")
    assert rows == []
    assert any("project_id" in n for n in notes)


# ---------------------------------------------------------------------------------------
# main() -- end to end against a real SQLite file
# ---------------------------------------------------------------------------------------


def _seeded_store(db_path):
    # Insert the never-sent draft FIRST and the sent opener SECOND, so the sent one (index 1,
    # the one every happy-path test wants) is unambiguously the most recent by created_at --
    # both use time.time() at insert, so ordering must not depend on within-a-test clock ties.
    store = SQLiteStore(db_path)
    store.record_opener("run1", "hinge", "gemini-x", "A draft nobody sent.", "her bio",
                        profile_id="pid-draft", decision="never_sent")
    store.record_opener("run1", "hinge", "gemini-x", "Sent and landed.", "her photo",
                        profile_id="pid-sent", decision="like", profile_key="k" * 64,
                        prompt_sha256="era-a")
    store.add_label("run1", "hinge", True, [0.1], profile_id="pid-sent", profile_name="Alex")
    store.close()
    con = sqlite3.connect(db_path)
    con.execute("UPDATE openers SET created_at = 100.0 WHERE opener = 'A draft nobody sent.'")
    con.execute("UPDATE openers SET created_at = 200.0 WHERE opener = 'Sent and landed.'")
    con.commit()
    con.close()


def _base_argv(tmp_path, *command):
    return [*command, "--db", str(tmp_path / "store.db"), "--backend", "sqlite",
           "--eras-file", str(tmp_path / "no-such-eras.json")]


def test_main_list_shows_only_sent_openers_end_to_end(tmp_path, capsys):
    _seeded_store(tmp_path / "store.db")

    code = m.main(_base_argv(tmp_path, "list"), cfg=_fake_cfg(backend="sqlite"))

    out = capsys.readouterr().out
    assert code == 0
    assert "Sent and landed." in out
    assert "Alex" in out
    assert "A draft nobody sent." not in out


def test_main_list_json_mode_only_includes_sent_openers(tmp_path, capsys):
    import json
    _seeded_store(tmp_path / "store.db")

    code = m.main(_base_argv(tmp_path, "list") + ["--json"], cfg=_fake_cfg(backend="sqlite"))

    doc = json.loads(capsys.readouterr().out)
    assert code == 0
    assert doc["sent_count"] == 1
    assert doc["sent"][0]["opener"] == "Sent and landed."
    assert doc["sent"][0]["profile_name"] == "Alex"


def test_main_record_writes_outcome_with_right_profile_key_and_source_end_to_end(tmp_path, capsys):
    _seeded_store(tmp_path / "store.db")

    code = m.main(_base_argv(tmp_path, "record")
                 + ["--index", "1", "--outcome", "match", "--note", "she replied", "--yes"],
                 cfg=_fake_cfg(backend="sqlite"))

    assert code == 0
    out = capsys.readouterr().out
    assert "Recorded outcome='match'" in out

    con = sqlite3.connect(tmp_path / "store.db")
    rows = con.execute(
        "SELECT app, profile_key, outcome, source, note FROM opener_outcomes").fetchall()
    con.close()
    assert rows == [("hinge", "k" * 64, "match", "owner", "she replied")]


def test_main_record_refuses_an_unsent_draft_and_writes_no_orphan_row(tmp_path, capsys):
    _seeded_store(tmp_path / "store.db")

    # Position 2 in the recent-openers window is "A draft nobody sent." (never_sent).
    code = m.main(_base_argv(tmp_path, "record")
                 + ["--index", "2", "--outcome", "match", "--yes"],
                 cfg=_fake_cfg(backend="sqlite"))

    assert code == 1
    err = capsys.readouterr().err
    assert "NEVER SENT" in err

    con = sqlite3.connect(tmp_path / "store.db")
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    count = (con.execute("SELECT COUNT(*) FROM opener_outcomes").fetchone()[0]
            if "opener_outcomes" in tables else 0)
    con.close()
    assert count == 0


def test_main_record_refuses_an_unknown_index_and_writes_no_orphan_row(tmp_path, capsys):
    _seeded_store(tmp_path / "store.db")

    code = m.main(_base_argv(tmp_path, "record")
                 + ["--index", "999", "--outcome", "match", "--yes"],
                 cfg=_fake_cfg(backend="sqlite"))

    assert code == 1
    err = capsys.readouterr().err
    assert "no opener at position 999" in err

    con = sqlite3.connect(tmp_path / "store.db")
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    count = (con.execute("SELECT COUNT(*) FROM opener_outcomes").fetchone()[0]
            if "opener_outcomes" in tables else 0)
    con.close()
    assert count == 0


def test_main_record_without_yes_asks_for_confirmation_and_aborts_on_no(tmp_path, capsys):
    _seeded_store(tmp_path / "store.db")

    code = m.main(_base_argv(tmp_path, "record") + ["--index", "1", "--outcome", "match"],
                 cfg=_fake_cfg(backend="sqlite"), confirm=lambda _prompt: False)

    assert code == 1
    assert "Aborted" in capsys.readouterr().err

    con = sqlite3.connect(tmp_path / "store.db")
    count = con.execute("SELECT COUNT(*) FROM opener_outcomes").fetchone()[0]
    con.close()
    assert count == 0  # declining the confirmation must write nothing at all


def test_main_record_confirm_yes_writes_the_row(tmp_path, capsys):
    _seeded_store(tmp_path / "store.db")

    code = m.main(_base_argv(tmp_path, "record") + ["--index", "1", "--outcome", "reply"],
                 cfg=_fake_cfg(backend="sqlite"), confirm=lambda _prompt: True)

    assert code == 0
    con = sqlite3.connect(tmp_path / "store.db")
    count = con.execute("SELECT COUNT(*) FROM opener_outcomes").fetchone()[0]
    con.close()
    assert count == 1


def test_main_record_with_injected_store_never_builds_its_own_and_flushes_it(tmp_path, capsys):
    _seeded_store(tmp_path / "store.db")
    store = _FakeStore()

    code = m.main(_base_argv(tmp_path, "record") + ["--index", "1", "--outcome", "unknown", "--yes"],
                 cfg=_fake_cfg(backend="sqlite"), store=store)

    assert code == 0
    assert store.calls[0]["outcome"] == "unknown"
    assert store.flushed is True
    assert store.closed is False  # only main()'s OWN store is closed, never an injected one


def test_main_record_reports_a_store_failure_loudly_rather_than_swallowing_it(tmp_path, capsys):
    _seeded_store(tmp_path / "store.db")
    store = _RaisingStore()

    code = m.main(_base_argv(tmp_path, "record") + ["--index", "1", "--outcome", "match", "--yes"],
                 cfg=_fake_cfg(backend="sqlite"), store=store)

    assert code == 1
    assert "could not record the outcome" in capsys.readouterr().err


def test_main_rejects_an_unknown_outcome_value(tmp_path, capsys):
    _seeded_store(tmp_path / "store.db")

    with pytest.raises(SystemExit):
        m.main(_base_argv(tmp_path, "record")
              + ["--index", "1", "--outcome", "definitely_matched_bigly", "--yes"],
              cfg=_fake_cfg(backend="sqlite"))

    assert "invalid choice" in capsys.readouterr().err


def test_main_record_uses_the_explicit_db_flag_not_the_config_db_file(tmp_path, capsys):
    """--db must win over cfg.db_file for BOTH the read and the write side, so `list` and
    `record` can never disagree about which database is in play (see _build_store's docstring)."""
    _seeded_store(tmp_path / "store.db")
    other_cfg_db = tmp_path / "unrelated_config_db.db"  # deliberately never created

    code = m.main(_base_argv(tmp_path, "record") + ["--index", "1", "--outcome", "match", "--yes"],
                 cfg=_fake_cfg(backend="sqlite", db_file=str(other_cfg_db)))

    assert code == 0
    assert not other_cfg_db.exists()
