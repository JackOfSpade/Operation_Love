"""tools/opener_corpus_report.py -- offline measurement of what the opener prompt produces.

No device, no live provider call, no real BigQuery ever: every test either drives this tool's
pure functions directly, builds a real (tmp_path) SQLite store via SQLiteStore, or injects a
fake BigQuery client -- matching tests/test_gemini_model_probe.py and test_bigquery_store.py's
own no-network conventions.
"""
from __future__ import annotations

import json
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

import pytest

from operation_love.opener import replay_corpus as rc
from operation_love.opener.opener import prompt_stamp
from operation_love.ranker.store import SQLiteStore
from tools import opener_corpus_report as m

# ---------------------------------------------------------------------------------------
# GRADE DETECTOR -- the four cases the task pins by name, plus a few adjacent behaviours.
# ---------------------------------------------------------------------------------------
# Both "must flag" lines are quoted VERBATIM from ops/OPENER-REDESIGN.md's 2026-09-06 "a
# compliment moved off her is still a score" addendum -- the sushi opener that triggered the
# owner objection, and the matcha-tiramisu opener the addendum cites as the predicate's other
# verbatim occurrence in the local corpus.

_SUSHI = "Starting the new year with a massive spread of sushi is an elite move."
_MATCHA = "Matcha tiramisu for a birthday cake is such an elite move."
_HYPERBOLE = "That heavy bag stood zero chance."


def test_grade_detector_flags_the_sushi_opener_verbatim_from_the_addendum():
    assert m.is_grade_shaped(_SUSHI) is True


def test_grade_detector_flags_the_matcha_opener_verbatim_from_the_addendum():
    assert m.is_grade_shaped(_MATCHA) is True


def test_grade_detector_does_not_flag_a_hyperbole_line():
    assert m.is_grade_shaped(_HYPERBOLE) is False


def test_grade_detector_does_not_flag_a_question_with_the_same_copula_shape():
    # Same copula + article + evaluative-noun-phrase shape as the sushi line above, but as a
    # question rather than an assertion -- must not be flagged: a verdict has to actually be
    # asserted, not merely raised as a question, to be a grade.
    assert m.is_grade_shaped("That spread is an elite move?") is False


def test_grade_detector_does_not_flag_a_plain_question_with_no_grade_shape():
    assert m.is_grade_shaped("What's the story behind that hike?") is False


def test_grade_detector_does_not_flag_a_negated_verdict():
    assert m.is_grade_shaped("That is not an elite move.") is False


def test_grade_detector_requires_the_complement_to_be_in_the_lexicon():
    # A real copula ("is") with a complement that is NOT in the evaluative lexicon must not be
    # flagged -- the detector is gated on the lexicon, not on "any copula sentence".
    assert m.is_grade_shaped("This snack is delicious.") is False


def test_grade_detector_only_looks_at_the_first_beat():
    # The evaluative shape is real but lives in the SECOND sentence; grade_rate (measured via
    # era_metrics, exercised below) must not count it, even though is_grade_shaped applied
    # directly to that second sentence alone would say True.
    two_sentence_opener = "Where was this taken? That trail is an elite move."
    sentences = m.split_sentences(two_sentence_opener)
    assert len(sentences) == 2
    assert m.is_grade_shaped(sentences[0]) is False
    assert m.is_grade_shaped(sentences[1]) is True


# ---------------------------------------------------------------------------------------
# split_sentences / ends_in_question / opening_trigram / has_apostrophe / has_looks_like
# ---------------------------------------------------------------------------------------

def test_split_sentences_keeps_abbreviation_periods_intact():
    assert m.split_sentences("Dr. Dolittle energy. What's the story?") == [
        "Dr. Dolittle energy.", "What's the story?"]


def test_split_sentences_single_sentence_opener():
    assert m.split_sentences(_SUSHI) == [_SUSHI]


def test_ends_in_question_true_and_false():
    assert m.ends_in_question("Where is this from?") is True
    assert m.ends_in_question('Where is this from?"') is True  # trailing quote tolerated
    assert m.ends_in_question("That is a nice view.") is False


def test_opening_trigram_normalizes_case_and_punctuation():
    assert m.opening_trigram("That LOOKS like, a great day.") == "that looks like"


def test_has_apostrophe_detects_straight_and_curly_marks():
    assert m.has_apostrophe("That's a good dog.") is True
    assert m.has_apostrophe("That’s a good dog.") is True
    assert m.has_apostrophe("That is a good dog.") is False


def test_has_looks_like_is_word_bounded():
    assert m.has_looks_like("That looks like a fun trip.") is True
    assert m.has_looks_like("That is a fun trip.") is False


# ---------------------------------------------------------------------------------------
# era_metrics -- trigram + sentence metrics on a small, hand-verified fixture
# ---------------------------------------------------------------------------------------

_FIXTURE_TEXTS = [
    "That looks like a fun hike. Where was it taken?",  # 2 sent, question-final, looks-like, trigram "that looks like"
    "That looks like a cozy cabin. Did you build it?",  # 2 sent, question-final, looks-like, trigram "that looks like"
    "That's an elite move.",                             # 1 sent, grade-shaped, apostrophe
    "What's your favorite trail?",                       # 1 sent, question-final, apostrophe
    "Quiet morning here.",                               # 1 sent, plain
    "Bold choice for brunch.",                           # 1 sent, plain
    "Solid pick for movie night.",                       # 1 sent, plain
]


def test_era_metrics_sentence_and_trigram_fixture():
    metrics = m.era_metrics("fixture", _FIXTURE_TEXTS)
    assert metrics.n == 7
    # Deliberately asymmetric (2 two-sentence vs 5 one-sentence) so a detector that counted the
    # wrong bucket size would not accidentally land on the same number by coincidence.
    assert metrics.sentence_counts == {1: 5, 2: 2}
    assert metrics.two_sentence_count == 2
    assert metrics.two_sentence_rate == pytest.approx(2 / 7)
    assert metrics.question_final_count == 3
    assert metrics.question_final_rate == pytest.approx(3 / 7)
    # Only "that looks like" repeats (2 openers); the other five each open on a unique trigram,
    # so top-5 coverage excludes exactly one (the sixth, least-recent distinct trigram).
    assert metrics.distinct_trigrams == 6
    assert metrics.total_trigrams == 7
    assert metrics.trigram_diversity == pytest.approx(6 / 7)
    assert metrics.top5_trigrams[0] == ("that looks like", 2)
    assert metrics.top5_trigram_coverage == pytest.approx(6 / 7)
    assert metrics.looks_like_count == 2
    assert metrics.looks_like_rate == pytest.approx(2 / 7)
    assert metrics.apostrophe_count == 2
    assert metrics.apostrophe_rate == pytest.approx(2 / 7)
    # Only "That's an elite move." is grade-shaped (single sentence, first beat == whole text).
    assert metrics.grade_count == 1
    assert metrics.grade_rate == pytest.approx(1 / 7)
    lengths = [len(t) for t in _FIXTURE_TEXTS]
    assert metrics.mean_chars == pytest.approx(statistics.mean(lengths))
    assert metrics.median_chars == pytest.approx(statistics.median(lengths))


def test_era_metrics_grade_rate_only_counts_the_first_beat():
    # The evaluative shape lives in the SECOND sentence only; grade_rate must not count it.
    texts = ["Where was this taken? That trail is an elite move."]
    metrics = m.era_metrics("x", texts)
    assert metrics.grade_count == 0
    assert metrics.grade_rate == 0.0


def test_era_metrics_empty_bucket_never_divides_by_zero():
    metrics = m.era_metrics("empty", [])
    assert metrics.n == 0
    assert metrics.grade_rate == 0.0
    assert metrics.two_sentence_rate == 0.0
    assert metrics.question_final_rate == 0.0
    assert metrics.trigram_diversity == 0.0
    assert metrics.top5_trigram_coverage == 0.0
    assert metrics.apostrophe_rate == 0.0
    assert metrics.looks_like_rate == 0.0
    assert metrics.mean_chars == 0.0
    assert metrics.median_chars == 0.0


# ---------------------------------------------------------------------------------------
# Era grouping
# ---------------------------------------------------------------------------------------

def test_build_era_metrics_groups_by_era_and_buckets_none_as_unknown():
    rows = [
        m.OpenerRow(text="alpha one is nice.", source="sqlite", era="era-a"),
        m.OpenerRow(text="alpha two is nice.", source="sqlite", era="era-a"),
        m.OpenerRow(text="beta one is nice.", source="sqlite", era="era-b"),
        m.OpenerRow(text="gamma one is nice.", source="jsonl", era=None),
    ]
    grouped = m.build_era_metrics(rows)
    assert set(grouped) == {"era-a", "era-b", m.UNKNOWN_ERA, "ALL"}
    assert grouped["era-a"].n == 2
    assert grouped["era-b"].n == 1
    assert grouped[m.UNKNOWN_ERA].n == 1
    assert grouped["ALL"].n == 4


def test_sort_era_keys_orders_named_then_unknown_then_all():
    keys = {"zzz-era", "aaa-era", m.UNKNOWN_ERA, "ALL"}
    assert m.sort_era_keys(keys) == ["aaa-era", "zzz-era", m.UNKNOWN_ERA, "ALL"]


def test_dedupe_openers_prefers_a_row_carrying_a_known_era():
    jsonl_row = m.OpenerRow(text="same text.", source="jsonl", era=None)
    sqlite_row = m.OpenerRow(text="same text.", source="sqlite", era="era-a")
    deduped = m.dedupe_openers([jsonl_row, sqlite_row])
    assert len(deduped) == 1
    assert deduped[0].era == "era-a"
    assert deduped[0].source == "sqlite"


def test_dedupe_openers_keeps_distinct_text_separate():
    rows = [
        m.OpenerRow(text="one.", source="jsonl", era=None),
        m.OpenerRow(text="two.", source="jsonl", era=None),
    ]
    assert len(m.dedupe_openers(rows)) == 2


# ---------------------------------------------------------------------------------------
# resolve_era + compare mode
# ---------------------------------------------------------------------------------------

def test_resolve_era_exact_and_prefix_match():
    available = ["abcdef0123456789", "fedcba9876543210", m.UNKNOWN_ERA, "ALL"]
    assert m.resolve_era(available, "abcdef0123456789") == "abcdef0123456789"
    assert m.resolve_era(available, "abcdef") == "abcdef0123456789"
    assert m.resolve_era(available, "unknown") == m.UNKNOWN_ERA


def test_resolve_era_raises_on_no_match():
    with pytest.raises(ValueError, match="no era matches"):
        m.resolve_era(["abc123", m.UNKNOWN_ERA], "zzz")


def test_resolve_era_raises_on_ambiguous_prefix():
    with pytest.raises(ValueError, match="ambiguous"):
        m.resolve_era(["abc111", "abc222"], "abc")


def test_compare_eras_reports_signed_deltas():
    a = m.era_metrics("a", ["That's an elite move."])            # grade, 1 sentence, apostrophe
    b = m.era_metrics("b", ["Where was this taken? Nice spot."])  # not graded, 2 sentences
    rows = dict((label, (va, vb, delta)) for label, va, vb, delta in m.compare_eras(a, b))
    assert rows["grade_rate"] == (1.0, 0.0, -1.0)
    assert rows["n"] == (1, 1, 0)


def test_format_compare_includes_both_era_labels_and_ns():
    a = m.era_metrics("a", [_SUSHI])
    b = m.era_metrics("b", [_HYPERBOLE])
    text = m.format_compare("era-a", "era-b", a, b)
    assert "era-a" in text and "era-b" in text
    assert "n=1" in text


# ---------------------------------------------------------------------------------------
# find_opener_strings -- recursive walk, including a depth > 0 case the local corpus never
# actually exercises today (see this tool's own docstring on why the walk stays generic).
# ---------------------------------------------------------------------------------------

def test_find_opener_strings_top_level():
    assert m.find_opener_strings({"opener": "hello", "other": 1}) == ["hello"]


def test_find_opener_strings_nested_inside_list_and_dict():
    nested = {"action": "auto_opener_pre_send",
             "payload": {"attempts": [{"opener": "first try"}, {"opener": "second try"}]}}
    assert m.find_opener_strings(nested) == ["first try", "second try"]


def test_find_opener_strings_ignores_non_string_and_blank_values():
    assert m.find_opener_strings({"opener": 42}) == []
    assert m.find_opener_strings({"opener": "   "}) == []
    assert m.find_opener_strings({"opener": None}) == []


# ---------------------------------------------------------------------------------------
# read_jsonl_openers
# ---------------------------------------------------------------------------------------

def _write_actions(run_dir, lines):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "actions.jsonl").open("w") as fh:
        for line in lines:
            fh.write(line + "\n")


def test_read_jsonl_openers_walks_populated_and_skips_empty_run_dirs(tmp_path):
    debug_dir = tmp_path / "hinge_debug"
    _write_actions(debug_dir / "run_a", [
        json.dumps({"action": "auto_opener_pre_send", "opener": "Draft one."}),
        json.dumps({"action": "auto_opener_resumed_send", "opener": "Draft one."}),  # duplicate
        json.dumps({"action": "capture"}),  # no opener key at all
    ])
    _write_actions(debug_dir / "run_b", [
        json.dumps({"action": "auto_opener_pre_send", "opener": "Draft two."}),
    ])
    _write_actions(debug_dir / "run_c_unpopulated", [
        json.dumps({"action": "capture"}),
        json.dumps({"action": "dislike"}),
    ])  # has a real actions.jsonl, but no line in it ever carries an "opener" key
    (debug_dir / "run_empty").mkdir()  # no actions.jsonl at all

    rows, stats = m.read_jsonl_openers(debug_dir)

    assert stats.run_dirs_total == 4
    assert stats.run_dirs_with_actions_file == 3
    # run_c_unpopulated has an actions.jsonl but contributes no opener, so populated (2) must
    # stay strictly below with_actions_file (3) -- proves this counts the FOUND-opener case,
    # not merely "an actions.jsonl exists".
    assert stats.run_dirs_populated == 2
    assert stats.opener_occurrences == 3       # 2 in run_a (incl. the duplicate) + 1 in run_b
    assert {row.text for row in rows} == {"Draft one.", "Draft two."}
    assert all(row.source == "jsonl" and row.era is None for row in rows)


def test_read_jsonl_openers_counts_malformed_lines_without_crashing(tmp_path):
    debug_dir = tmp_path / "hinge_debug"
    run_dir = debug_dir / "run_a"
    run_dir.mkdir(parents=True)
    with (run_dir / "actions.jsonl").open("w") as fh:
        fh.write("{not valid json\n")
        fh.write(json.dumps({"opener": "Fine draft."}) + "\n")

    rows, stats = m.read_jsonl_openers(debug_dir)

    assert stats.malformed_lines == 1
    assert [row.text for row in rows] == ["Fine draft."]


def test_read_jsonl_openers_missing_debug_dir_reports_zero_not_an_error(tmp_path):
    rows, stats = m.read_jsonl_openers(tmp_path / "does_not_exist")
    assert rows == []
    assert stats.run_dirs_total == 0


# ---------------------------------------------------------------------------------------
# read_sqlite_openers -- built through the real SQLiteStore, not a hand-rolled schema
# ---------------------------------------------------------------------------------------

def test_read_sqlite_openers_reads_openers_and_rejections_with_era(tmp_path):
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("run1", "hinge", "gemini-x", "Nice view opener.", "the view",
                            prompt_sha256="era-a")
        store.record_opener("run1", "hinge", "gemini-x", "Second opener.", "the dog",
                            prompt_sha256="era-b")
        store.record_opener_rejection("run1", "hinge", "gemini-x", 1, "too_many_sentences",
                                      "reason text", "raw", prompt_sha256="era-a")
    finally:
        store.close()

    rows, rejections, stats = m.read_sqlite_openers(db_path)

    assert stats.available is True
    assert stats.openers_row_count == 2
    assert {(r.text, r.era) for r in rows} == {
        ("Nice view opener.", "era-a"), ("Second opener.", "era-b")}
    assert len(rejections) == 1
    assert rejections[0].reason_code == "too_many_sentences"
    assert rejections[0].era == "era-a"


def test_read_sqlite_openers_missing_file_reports_a_note_not_an_error(tmp_path):
    rows, rejections, stats = m.read_sqlite_openers(tmp_path / "nope.db")
    assert rows == []
    assert rejections == []
    assert stats.available is False
    assert any("no database file" in note for note in stats.notes)


def test_read_sqlite_openers_pre_redesign_schema_reports_missing_tables(tmp_path):
    # The real, checked-in local db is exactly this: a file that exists, is a valid sqlite
    # database, but predates the openers/opener_rejections tables entirely.
    db_path = tmp_path / "old.db"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE labels (id INTEGER PRIMARY KEY)")
    con.commit()
    con.close()

    rows, rejections, stats = m.read_sqlite_openers(db_path)

    assert rows == []
    assert rejections == []
    assert stats.available is True
    assert any("no 'openers' table" in note for note in stats.notes)
    assert any("no 'opener_rejections' table" in note for note in stats.notes)


# ---------------------------------------------------------------------------------------
# HIGH-severity fix: the `decision` column -- read, legacy NULL/empty bucketing, and the
# per-decision metrics breakdown (sent / not_sent / unknown) this tool now reports as a first
# class axis instead of silently pooling sent and discarded drafts. See this module's docstring
# BE HONEST caveat 4 for the OpenerService.discard_opener boundary this whole section pins.
# ---------------------------------------------------------------------------------------

def test_read_sqlite_openers_reads_the_decision_column(tmp_path):
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("run1", "hinge", "gemini-x", "Sent opener.", "the view",
                            prompt_sha256="era-a", decision="like")
        store.record_opener("run1", "hinge", "gemini-x", "Discarded opener.", "the dog",
                            prompt_sha256="era-a", decision="never_sent")
    finally:
        store.close()

    rows, _rejections, _stats = m.read_sqlite_openers(db_path)

    by_text = {r.text: r.decision for r in rows}
    assert by_text["Sent opener."] == "like"
    assert by_text["Discarded opener."] == "never_sent"


def test_read_sqlite_openers_legacy_null_decision_is_none_not_like(tmp_path):
    # record_opener's own `decision` default is "" (see store.py) -- a caller that predates
    # decision-tracking entirely and never passes the keyword writes exactly this. The reader
    # must normalize that to None (unknown), never to "like".
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("run1", "hinge", "gemini-x", "Legacy opener.", "the view")
    finally:
        store.close()

    rows, _rejections, _stats = m.read_sqlite_openers(db_path)

    assert len(rows) == 1
    assert rows[0].decision is None  # NOT "like", NOT "" -- must never be guessed as sent


def test_read_sqlite_openers_missing_decision_column_notes_and_lands_unknown(tmp_path):
    # A schema that has the `openers` table but predates the `decision` column entirely (older
    # than the pre-redesign snapshot exercised by the sibling missing-tables test above).
    db_path = tmp_path / "old.db"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE openers (id INTEGER PRIMARY KEY, run_id TEXT, app TEXT, "
                "created_at REAL, model TEXT, opener TEXT, referenced TEXT)")
    con.execute("INSERT INTO openers (run_id, app, created_at, model, opener, referenced) "
                "VALUES ('run1','hinge',0,'gemini-x','No-decision-column opener.','x')")
    con.commit()
    con.close()

    rows, _rejections, stats = m.read_sqlite_openers(db_path)

    assert len(rows) == 1
    assert rows[0].decision is None
    assert any("predates the decision column" in note for note in stats.notes)


def test_read_bigquery_openers_reads_the_decision_column():
    client = _FakeBQClient({
        "openers": [{"opener": "Sent BQ opener.", "prompt_sha256": "era-a", "decision": "like"},
                   {"opener": "Discarded BQ opener.", "prompt_sha256": "era-a",
                    "decision": "dislike"},
                   {"opener": "Legacy BQ opener.", "prompt_sha256": None, "decision": None}],
    })

    rows, _rejections, _stats = m.read_bigquery_openers("proj-1", "operation_love", client=client)

    by_text = {r.text: r.decision for r in rows}
    assert by_text["Sent BQ opener."] == "like"
    assert by_text["Discarded BQ opener."] == "dislike"
    assert by_text["Legacy BQ opener."] is None


def test_decision_bucket_classifies_like_dislike_never_sent_and_legacy():
    assert m.decision_bucket("like") == m.DECISION_SENT
    assert m.decision_bucket("dislike") == m.DECISION_NOT_SENT
    assert m.decision_bucket("never_sent") == m.DECISION_NOT_SENT
    assert m.decision_bucket(None) == m.DECISION_UNKNOWN
    assert m.decision_bucket("") == m.DECISION_UNKNOWN


def test_decision_bucket_never_counts_legacy_none_as_sent():
    # The exact regression this whole fix exists to prevent: a legacy row with no decision
    # recorded must land in "unknown", never silently join "sent".
    assert m.decision_bucket(None) != m.DECISION_SENT
    assert m.decision_bucket("") != m.DECISION_SENT


def test_build_decision_metrics_groups_sent_not_sent_and_unknown_separately():
    rows = [
        m.OpenerRow(text="alpha is nice.", source="sqlite", era="era-a", decision="like"),
        m.OpenerRow(text="beta is nice.", source="sqlite", era="era-a", decision="like"),
        m.OpenerRow(text="gamma is nice.", source="sqlite", era="era-a", decision="dislike"),
        m.OpenerRow(text="delta is nice.", source="sqlite", era="era-a", decision="never_sent"),
        m.OpenerRow(text="epsilon is nice.", source="sqlite", era="era-a", decision=None),
    ]
    grouped = m.build_decision_metrics(rows)
    assert set(grouped) == {m.DECISION_SENT, m.DECISION_NOT_SENT, m.DECISION_UNKNOWN, "ALL"}
    assert grouped[m.DECISION_SENT].n == 2
    assert grouped[m.DECISION_NOT_SENT].n == 2
    assert grouped[m.DECISION_UNKNOWN].n == 1
    assert grouped["ALL"].n == 5


def test_sort_decision_keys_orders_sent_then_not_sent_then_unknown_then_all():
    keys = {"ALL", m.DECISION_UNKNOWN, m.DECISION_NOT_SENT, m.DECISION_SENT}
    assert m.sort_decision_keys(keys) == [
        m.DECISION_SENT, m.DECISION_NOT_SENT, m.DECISION_UNKNOWN, "ALL"]


def _build_decision_corpus(tmp_path):
    """One sent and one never-sent row, same era, no jsonl -- the minimal fixture for
    exercising the decision breakdown end to end through main()."""
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("run1", "hinge", "gemini-x", "Sent one.", "the view",
                            prompt_sha256="era-a", decision="like")
        store.record_opener("run1", "hinge", "gemini-x", "Discarded one.", "the dog",
                            prompt_sha256="era-a", decision="never_sent")
    finally:
        store.close()
    return db_path


def test_main_json_mode_carries_the_decision_breakdown(tmp_path, capsys):
    db_path = _build_decision_corpus(tmp_path)
    code = m.main(["--debug-dir", str(tmp_path / "no_such_debug_dir"), "--db", str(db_path),
                  "--json"])
    out = capsys.readouterr().out
    assert code == 0
    doc = json.loads(out)
    assert set(doc["by_decision"]) == {"sent", "not_sent", "ALL"}
    assert doc["by_decision"]["sent"]["n"] == 1
    assert doc["by_decision"]["not_sent"]["n"] == 1
    assert doc["by_decision"]["ALL"]["n"] == 2


def test_main_compare_mode_still_carries_the_decision_breakdown_in_text(tmp_path, capsys):
    # --compare must not suppress the decision breakdown: it is unconditional per this module's
    # OUTPUT docstring section, regardless of which era pair is being compared.
    db_path = _build_decision_corpus(tmp_path)
    code = m.main(["--debug-dir", str(tmp_path / "no_such_debug_dir"), "--db", str(db_path),
                  "--compare", "era-a", "ALL"])
    out = capsys.readouterr().out
    assert code == 0
    assert "=== METRICS BY DECISION" in out
    assert "decision: sent (n=1)" in out
    assert "decision: not_sent (n=1)" in out


def test_caveat_four_describes_partial_bias_and_discard_opener_boundary():
    caveat_four = m._CAVEATS[3]
    assert "PARTIAL" in caveat_four
    assert "discard_opener" in caveat_four
    assert "NOT retroactive" in caveat_four
    assert "never_sent" in caveat_four
    assert "dislike" in caveat_four
    # The stale, now-false unconditional claim this caveat used to make must be gone: it must
    # not claim every store-backed number describes only sent openers.
    assert "store-backed numbers describe openers that were SENT" not in caveat_four


def test_main_text_mode_prints_the_corrected_caveat_and_decision_section(tmp_path, capsys):
    db_path = _build_decision_corpus(tmp_path)
    code = m.main(["--debug-dir", str(tmp_path / "no_such_debug_dir"), "--db", str(db_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "PARTIAL" in out
    assert "discard_opener" in out
    assert "=== METRICS BY DECISION" in out
    assert "decision: sent (n=1)" in out
    assert "decision: not_sent (n=1)" in out


# ---------------------------------------------------------------------------------------
# read_bigquery_openers -- fake client, no google-cloud-bigquery / network needed
# ---------------------------------------------------------------------------------------

class _FakeBQJob:
    def __init__(self, rows):
        self._rows = rows

    def result(self):
        return self._rows


class _FakeBQClient:
    def __init__(self, table_rows):
        self._table_rows = table_rows  # {table_name_suffix: [row_dict, ...]}
        self.queries = []

    def query(self, sql):
        self.queries.append(sql)
        for suffix, rows in self._table_rows.items():
            if sql.rstrip().endswith(f"{suffix}`"):
                return _FakeBQJob([_Row(r) for r in rows])
        raise RuntimeError(f"NotFound: table referenced in {sql!r} does not exist")


class _Row(dict):
    def __getitem__(self, key):
        return super().__getitem__(key)


class _FakeBQOutcomesClient:
    """Minimal fake client for read_bigquery_opener_outcomes()'s single JOIN query. Unlike
    _FakeBQClient above (which routes by which single TABLE a query names, matched by a
    trailing backtick), this reader issues one query joining two tables with a WHERE clause
    after it, so this fake just returns whatever rows it was configured with (or raises, to
    simulate a missing table/permissions failure) regardless of the exact SQL text -- the join
    reading function's own behaviour on the returned rows is what these tests exercise."""
    def __init__(self, rows=None, raises=None):
        self._rows = rows or []
        self._raises = raises
        self.queries = []

    def query(self, sql):
        self.queries.append(sql)
        if self._raises is not None:
            raise self._raises
        return _FakeBQJob([_Row(r) for r in self._rows])


def test_read_bigquery_openers_reads_both_tables_with_a_fake_client():
    client = _FakeBQClient({
        "openers": [{"opener": "BQ draft one.", "prompt_sha256": "era-a", "decision": "like"},
                   {"opener": "BQ draft two.", "prompt_sha256": None, "decision": None}],
        "opener_rejections": [{"reason_code": "scaffolding", "prompt_sha256": "era-a"}],
    })

    rows, rejections, stats = m.read_bigquery_openers("proj-1", "operation_love", client=client)

    assert stats.available is True
    assert {(r.text, r.era) for r in rows} == {
        ("BQ draft one.", "era-a"), ("BQ draft two.", None)}
    assert len(rejections) == 1
    assert rejections[0].reason_code == "scaffolding"


def test_read_bigquery_openers_missing_table_reports_a_note_not_a_crash():
    client = _FakeBQClient({})  # neither table exists

    rows, rejections, stats = m.read_bigquery_openers("proj-1", "operation_love", client=client)

    assert rows == []
    assert stats.available is False
    assert any("openers" in note for note in stats.notes)


def test_read_bigquery_openers_rejects_an_invalid_project_id():
    rows, rejections, stats = m.read_bigquery_openers("not a valid id!", "operation_love",
                                                       client=_FakeBQClient({}))
    assert rows == []
    assert any("invalid BigQuery config" in note for note in stats.notes)


# ---------------------------------------------------------------------------------------
# main() -- end to end, text and --json, plus --compare success/failure
# ---------------------------------------------------------------------------------------

def _build_corpus(tmp_path):
    debug_dir = tmp_path / "hinge_debug"
    _write_actions(debug_dir / "run_a", [
        json.dumps({"action": "auto_opener_pre_send", "opener": _SUSHI}),
    ])
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("run1", "hinge", "gemini-x", "Where was this taken?", "the view",
                            prompt_sha256="era-b")
    finally:
        store.close()
    return debug_dir, db_path


def test_main_text_mode_reports_sources_and_both_eras(tmp_path, capsys):
    debug_dir, db_path = _build_corpus(tmp_path)
    code = m.main(["--debug-dir", str(debug_dir), "--db", str(db_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "=== SOURCES ===" in out
    assert "era: era-b" in out
    assert "era: unknown" in out
    assert "DRAFTS, not sent messages" in out


def test_main_json_mode_emits_parseable_document(tmp_path, capsys):
    debug_dir, db_path = _build_corpus(tmp_path)
    code = m.main(["--debug-dir", str(debug_dir), "--db", str(db_path), "--json"])
    out = capsys.readouterr().out
    assert code == 0
    doc = json.loads(out)
    assert doc["sources"]["combined_unique"] == 2
    assert set(doc["by_era"]) == {"era-b", m.UNKNOWN_ERA, "ALL"}


def test_main_compare_valid_prefix_succeeds(tmp_path, capsys):
    debug_dir, db_path = _build_corpus(tmp_path)
    code = m.main(["--debug-dir", str(debug_dir), "--db", str(db_path),
                  "--compare", "era-b", "unknown"])
    out = capsys.readouterr().out
    assert code == 0
    assert "COMPARE:" in out


def test_main_compare_unknown_token_fails_loudly(tmp_path, capsys):
    debug_dir, db_path = _build_corpus(tmp_path)
    code = m.main(["--debug-dir", str(debug_dir), "--db", str(db_path),
                  "--compare", "does-not-exist", "unknown"])
    err = capsys.readouterr().err
    assert code == 1
    assert "no era matches" in err


# =========================================================================================
# THE PROMPT ERA REGISTRY (tools/backfill_prompt_eras.py's ops/prompt-eras.json) -- loading,
# label resolution, --eras, --rule, and label-aware --compare.
# =========================================================================================

def _registry_doc(*eras, working_tree=None):
    """eras: (digest, label, first_date, last_date, rules) tuples, oldest first. `working_tree`,
    when given, is written verbatim as the top level 'working_tree' key -- see
    _working_tree_entry() below for a ready-made one shaped like
    tools/backfill_prompt_eras.py's build_working_tree_entry() output."""
    doc = {
        "schema_version": 1,
        "eras": [
            {"prompt_sha256": digest, "label": label, "first_commit": digest[:8] + "-first",
             "first_date": first_date, "first_subject": "first subject",
             "last_commit": digest[:8] + "-last", "last_date": last_date,
             "last_subject": "last subject", "commit_count": 1, "commits": [digest[:8]],
             "rules": rules}
            for digest, label, first_date, last_date, rules in eras
        ],
        "unevaluable_revisions": [],
    }
    if working_tree is not None:
        doc["working_tree"] = working_tree
    return doc


def _write_registry(tmp_path, *eras, working_tree=None):
    path = tmp_path / "prompt-eras.json"
    path.write_text(json.dumps(_registry_doc(*eras, working_tree=working_tree)))
    return path


_ERA_A = ("a" * 64, "era A label", "2026-01-01T00:00:00", "2026-01-01T00:00:00", ["RULE ONE"])
_ERA_B = ("b" * 64, "era B label", "2026-02-01T00:00:00", "2026-02-01T00:00:00",
         ["RULE ONE", "RULE TWO"])
_ERA_C = ("c" * 64, "era C label", "2026-03-01T00:00:00", "2026-03-01T00:00:00", ["RULE TWO"])


def _working_tree_entry(digest="d" * 64, label=None, rules=("NO GRADING",),
                        matches=None, reasons=None):
    """A ready-made 'working_tree' registry entry, shaped like
    tools/backfill_prompt_eras.py's build_working_tree_entry() output, for tests that only care
    about how opener_corpus_report.py CONSUMES it (not how backfill_prompt_eras.py builds it --
    that is tests/test_backfill_prompt_eras.py's job). `digest=None` simulates an UNEVALUABLE
    working tree (see that module's own shape for this case)."""
    if digest is None:
        return {
            "provisional": True, "prompt_sha256": None,
            "label": label or "UNCOMMITTED WORKING TREE -- PROVISIONAL, not a shipped era "
                             "(UNEVALUABLE, see reasons)",
            "commit": None, "rules": [], "reasons": reasons or ["opener.py does not parse"],
            "matches_known_committed_era": None, "no_metrics_recorded_here": True, "note": "n/a",
        }
    return {
        "provisional": True, "prompt_sha256": digest,
        "label": label or ("UNCOMMITTED WORKING TREE -- PROVISIONAL, not a shipped era "
                          f"(+{rules[0]})" if rules else
                          "UNCOMMITTED WORKING TREE -- PROVISIONAL, not a shipped era (baseline)"),
        "commit": None, "rules": list(rules), "reasons": [],
        "matches_known_committed_era": matches, "no_metrics_recorded_here": True, "note": "n/a",
    }


# ---------------------------------------------------------------------------------------
# load_era_registry
# ---------------------------------------------------------------------------------------

def test_load_era_registry_reports_a_missing_file_honestly(tmp_path):
    registry = m.load_era_registry(tmp_path / "does_not_exist.json")
    assert registry.loaded is False
    assert "no era registry at" in registry.note
    assert registry.eras == {} and registry.order == []


def test_load_era_registry_reports_invalid_json_honestly(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    registry = m.load_era_registry(path)
    assert registry.loaded is False
    assert "could not parse" in registry.note


def test_load_era_registry_loads_eras_in_stored_order(tmp_path):
    path = _write_registry(tmp_path, _ERA_A, _ERA_B)
    registry = m.load_era_registry(path)
    assert registry.loaded is True
    assert registry.order == [_ERA_A[0], _ERA_B[0]]
    assert registry.eras[_ERA_A[0]]["label"] == "era A label"


def test_load_era_registry_reads_the_working_tree_entry_when_present(tmp_path):
    wt = _working_tree_entry()
    path = _write_registry(tmp_path, _ERA_A, working_tree=wt)
    registry = m.load_era_registry(path)
    assert registry.working_tree == wt


def test_load_era_registry_working_tree_defaults_to_none_when_absent(tmp_path):
    # A registry generated before this feature existed (or with --no-working-tree) -- must
    # degrade cleanly, never crash on a missing key.
    path = _write_registry(tmp_path, _ERA_A)
    registry = m.load_era_registry(path)
    assert registry.working_tree is None


def test_load_era_registry_ignores_a_non_dict_working_tree_value(tmp_path):
    # A malformed/hand-edited registry -- 'working_tree' must degrade to None rather than being
    # handed to a formatter as a string/list and crashing on .get().
    path = tmp_path / "prompt-eras.json"
    doc = _registry_doc(_ERA_A)
    doc["working_tree"] = "not a dict"
    path.write_text(json.dumps(doc))
    registry = m.load_era_registry(path)
    assert registry.working_tree is None


# ---------------------------------------------------------------------------------------
# resolve_era_label
# ---------------------------------------------------------------------------------------

def test_resolve_era_label_returns_the_registered_label(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A))
    assert m.resolve_era_label(_ERA_A[0], registry) == "era A label"


def test_resolve_era_label_marks_an_unregistered_digest_clearly(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A))
    label = m.resolve_era_label("f" * 64, registry)
    assert "f" * 64 in label
    assert "unregistered" in label


def test_resolve_era_label_handles_unknown_and_all_pseudo_buckets(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A))
    assert m.resolve_era_label(m.UNKNOWN_ERA, registry) == "(not a single stamped era)"
    assert m.resolve_era_label("ALL", registry) == "(not a single stamped era)"


def test_resolve_era_label_with_no_registry_loaded_falls_back():
    label = m.resolve_era_label("f" * 64, None)
    assert "f" * 64 in label
    assert "no era registry" in label


def test_resolve_era_label_falls_back_to_the_working_tree_entry_when_it_matches(tmp_path):
    # A measured corpus row can carry the working tree's OWN digest (a local test run before
    # committing) -- resolve_era_label must recognize it via registry.working_tree rather than
    # reporting it as merely "unregistered".
    wt = _working_tree_entry(digest="d" * 64)
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, working_tree=wt))
    label = m.resolve_era_label("d" * 64, registry)
    assert label == wt["label"]
    assert "UNCOMMITTED" in label


def test_resolve_era_label_unregistered_digest_stays_unregistered_when_no_working_tree_matches(tmp_path):
    wt = _working_tree_entry(digest="d" * 64)
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, working_tree=wt))
    label = m.resolve_era_label("f" * 64, registry)  # neither a known era nor the working tree
    assert "unregistered" in label


# ---------------------------------------------------------------------------------------
# resolve_compare_token -- resolve_era() extended to accept a registered human label
# ---------------------------------------------------------------------------------------

def test_resolve_compare_token_accepts_a_human_label(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, _ERA_B))
    available = [_ERA_A[0], _ERA_B[0]]
    assert m.resolve_compare_token(available, "era B label", registry) == _ERA_B[0]


def test_resolve_compare_token_label_match_is_case_insensitive(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A))
    available = [_ERA_A[0]]
    assert m.resolve_compare_token(available, "ERA A LABEL", registry) == _ERA_A[0]


def test_resolve_compare_token_still_accepts_a_digest_prefix(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A))
    available = [_ERA_A[0]]
    assert m.resolve_compare_token(available, _ERA_A[0][:10], registry) == _ERA_A[0]


def test_resolve_compare_token_raises_on_a_label_matching_multiple_eras(tmp_path):
    dup_a = ("a" * 64, "same label", "2026-01-01T00:00:00", "2026-01-01T00:00:00", ["R"])
    dup_b = ("b" * 64, "same label", "2026-02-01T00:00:00", "2026-02-01T00:00:00", ["R"])
    registry = m.load_era_registry(_write_registry(tmp_path, dup_a, dup_b))
    available = [dup_a[0], dup_b[0]]
    with pytest.raises(ValueError, match="matches multiple registered eras"):
        m.resolve_compare_token(available, "same label", registry)


def test_resolve_compare_token_with_no_registry_behaves_like_resolve_era():
    available = ["abcdef0123456789", m.UNKNOWN_ERA]
    assert m.resolve_compare_token(available, "abcdef", None) == "abcdef0123456789"


# ---------------------------------------------------------------------------------------
# format_era_metrics / format_rejections / format_compare -- label resolution wired in
# ---------------------------------------------------------------------------------------

def test_format_era_metrics_appends_the_registry_label_on_the_era_axis(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A))
    metrics = m.era_metrics(_ERA_A[0], [_SUSHI])
    text = m.format_era_metrics(metrics, label="era", registry=registry)
    assert f"era: {_ERA_A[0]} [era A label]" in text


def test_format_era_metrics_decision_axis_never_attempts_era_label_resolution(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A))
    metrics = m.era_metrics("sent", [_SUSHI])
    text = m.format_era_metrics(metrics, label="decision", registry=registry)
    assert "[" not in text.splitlines()[0]


def test_format_rejections_appends_the_registry_label(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A))
    rejections = {_ERA_A[0]: Counter({"scaffolding": 2})}
    text = m.format_rejections(rejections, registry=registry)
    assert f"era {_ERA_A[0]} [era A label]" in text


def test_format_compare_appends_registered_labels_when_a_registry_is_given(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, _ERA_B))
    a = m.era_metrics(_ERA_A[0], [_SUSHI])
    b = m.era_metrics(_ERA_B[0], [_HYPERBOLE])
    text = m.format_compare(_ERA_A[0], _ERA_B[0], a, b, registry=registry)
    assert "[era A label]" in text and "[era B label]" in text


# ---------------------------------------------------------------------------------------
# build_report -- era_labels side table, compare era_a_label/era_b_label
# ---------------------------------------------------------------------------------------

def test_build_report_carries_an_era_labels_side_table(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A))
    rows = [m.OpenerRow(text=_SUSHI, source="sqlite", era=_ERA_A[0], decision="like")]
    source_report = m.SourceReport(m.JsonlStats(), 0, m.SqliteStats(), 1, m.BigQueryStats(), 0, 1)
    doc = m.build_report(rows, [], source_report, registry=registry)
    assert doc["era_labels"] == {_ERA_A[0]: "era A label"}


def test_build_report_era_labels_covers_a_working_tree_digest_with_real_rows(tmp_path):
    # THE BUG THIS PINS: build_report() used to build era_labels from registry.order alone
    # (registry.eras[digest].get("label", digest)), never consulting registry.working_tree. A
    # working-tree digest carrying real, measured `by_era` rows (e.g. a local test run under the
    # current uncommitted prompt edit, before it is ever committed) then showed up in `by_era`
    # with a real `n` but had NO entry in `era_labels` at all -- exactly the ambiguity the
    # provisional marker exists to prevent. Every TEXT-mode formatter already resolved this
    # correctly via resolve_era_label(); only the JSON side table dropped it.
    wt_digest = "d" * 64
    wt = _working_tree_entry(digest=wt_digest, rules=("NO GRADING",))
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, working_tree=wt))
    rows = [m.OpenerRow(text=_SUSHI, source="sqlite", era=wt_digest, decision="like")]
    source_report = m.SourceReport(m.JsonlStats(), 0, m.SqliteStats(), 1, m.BigQueryStats(), 0, 1)
    doc = m.build_report(rows, [], source_report, registry=registry)

    # The working-tree digest actually has real rows in by_era ...
    assert doc["by_era"][wt_digest]["n"] == 1
    # ... and must therefore also carry its provisional label in era_labels, resolved the exact
    # same way resolve_era_label() (and every text-mode formatter) already does.
    assert wt_digest in doc["era_labels"]
    assert doc["era_labels"][wt_digest] == m.resolve_era_label(wt_digest, registry)
    assert doc["era_labels"][wt_digest].startswith("UNCOMMITTED WORKING TREE")
    # The committed era A, which carries no rows in this run's corpus, is still resolvable and
    # still reported -- this fix must not narrow era_labels to ONLY the working-tree digest.
    assert doc["era_labels"] == {
        wt_digest: m.resolve_era_label(wt_digest, registry),
    }


def test_build_report_era_labels_includes_working_tree_digest_even_with_zero_rows(tmp_path):
    # The working-tree digest belongs in era_labels once the registry names it as provisional,
    # even on a run whose corpus happens not to contain any row under that exact digest yet --
    # mirroring "plus the working-tree digest when EraRegistry.working_tree is set" rather than
    # only ever keying off of what this particular run's by_era happened to contain.
    wt_digest = "e" * 64
    wt = _working_tree_entry(digest=wt_digest, rules=("NO GRADING",))
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, working_tree=wt))
    rows = [m.OpenerRow(text=_SUSHI, source="sqlite", era=_ERA_A[0], decision="like")]
    source_report = m.SourceReport(m.JsonlStats(), 0, m.SqliteStats(), 1, m.BigQueryStats(), 0, 1)
    doc = m.build_report(rows, [], source_report, registry=registry)

    assert wt_digest not in doc["by_era"]
    assert doc["era_labels"][wt_digest] == m.resolve_era_label(wt_digest, registry)
    assert doc["era_labels"][wt_digest].startswith("UNCOMMITTED WORKING TREE")


def test_build_report_era_labels_never_carries_the_unknown_or_all_pseudo_buckets(tmp_path):
    # "unknown" and "ALL" are pseudo-buckets in by_era/metrics, not real prompt_sha256 digests --
    # era_labels must never grow entries for either one, even though resolve_era_label() itself
    # is happy to resolve them (to "(not a single stamped era)").
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A))
    rows = [
        m.OpenerRow(text=_SUSHI, source="sqlite", era=_ERA_A[0], decision="like"),
        m.OpenerRow(text=_HYPERBOLE, source="sqlite", era=None, decision="like"),
    ]
    source_report = m.SourceReport(m.JsonlStats(), 0, m.SqliteStats(), 2, m.BigQueryStats(), 0, 2)
    doc = m.build_report(rows, [], source_report, registry=registry)

    assert m.UNKNOWN_ERA in doc["by_era"] and "ALL" in doc["by_era"]  # sanity: both are present
    assert m.UNKNOWN_ERA not in doc["era_labels"]
    assert "ALL" not in doc["era_labels"]


def test_build_report_compare_includes_resolved_labels(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, _ERA_B))
    rows = [
        m.OpenerRow(text=_SUSHI, source="sqlite", era=_ERA_A[0], decision="like"),
        m.OpenerRow(text=_HYPERBOLE, source="sqlite", era=_ERA_B[0], decision="like"),
    ]
    source_report = m.SourceReport(m.JsonlStats(), 0, m.SqliteStats(), 2, m.BigQueryStats(), 0, 2)
    doc = m.build_report(rows, [], source_report, compare=(_ERA_A[0], _ERA_B[0]),
                         registry=registry)
    assert doc["compare"]["era_a_label"] == "era A label"
    assert doc["compare"]["era_b_label"] == "era B label"


# ---------------------------------------------------------------------------------------
# format_eras_listing (--eras)
# ---------------------------------------------------------------------------------------

def test_format_eras_listing_reports_an_unloaded_registry():
    registry = m.EraRegistry(path=Path("nope.json"), loaded=False, eras={}, order=[],
                             note="no era registry at nope.json")
    text = m.format_eras_listing(registry)
    assert "no era registry" in text


def test_format_eras_listing_reports_an_empty_registry():
    registry = m.EraRegistry(path=Path("x.json"), loaded=True, eras={}, order=[])
    text = m.format_eras_listing(registry)
    assert "zero eras" in text


def test_format_eras_listing_shows_oldest_first_with_added_and_removed(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, _ERA_B, _ERA_C))
    text = m.format_eras_listing(registry)
    idx_a = text.index("era A label")
    idx_b = text.index("era B label")
    idx_c = text.index("era C label")
    assert idx_a < idx_b < idx_c
    assert "baseline era" in text  # era A has nothing before it
    # era B added RULE TWO on top of era A's RULE ONE
    b_section = text[idx_b:idx_c]
    assert "added        : ['RULE TWO']" in b_section
    assert "removed      : (none)" in b_section
    # era C dropped RULE ONE (kept only RULE TWO)
    c_section = text[idx_c:]
    assert "removed      : ['RULE ONE']" in c_section


def test_format_eras_listing_appends_the_working_tree_section_after_every_committed_era(tmp_path):
    wt = _working_tree_entry(digest="d" * 64, rules=("NO GRADING", "SPOKEN REGISTER"))
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, working_tree=wt))
    text = m.format_eras_listing(registry)
    idx_a = text.index("era A label")
    idx_wt = text.index("UNCOMMITTED WORKING TREE")
    assert idx_a < idx_wt  # after every committed era, never interleaved
    assert "d" * 64 in text
    assert "not yet shipped" in text  # matches_known_committed_era is None here


def test_format_eras_listing_reports_a_working_tree_that_matches_a_known_era(tmp_path):
    wt = _working_tree_entry(digest=_ERA_A[0], matches=_ERA_A[0])
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, working_tree=wt))
    text = m.format_eras_listing(registry)
    assert f"matches known committed era: {_ERA_A[0]}" in text


def test_format_eras_listing_reports_an_unevaluable_working_tree(tmp_path):
    wt = _working_tree_entry(digest=None, reasons=["opener.py does not parse: SyntaxError"])
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, working_tree=wt))
    text = m.format_eras_listing(registry)
    assert "(unevaluable)" in text
    assert "opener.py does not parse: SyntaxError" in text


def test_format_eras_listing_omits_the_working_tree_section_when_absent(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A))
    text = m.format_eras_listing(registry)
    assert "UNCOMMITTED WORKING TREE" not in text


def test_format_eras_listing_shows_working_tree_even_with_zero_committed_eras(tmp_path):
    wt = _working_tree_entry(digest="d" * 64)
    registry = m.EraRegistry(path=Path("x.json"), loaded=True, eras={}, order=[],
                             working_tree=wt)
    text = m.format_eras_listing(registry)
    assert "zero eras" in text
    assert "UNCOMMITTED WORKING TREE" in text


# ---------------------------------------------------------------------------------------
# format_rule_lookup (--rule)
# ---------------------------------------------------------------------------------------

def test_format_rule_lookup_reports_an_unloaded_registry():
    registry = m.EraRegistry(path=Path("nope.json"), loaded=False, eras={}, order=[],
                             note="no era registry at nope.json")
    text = m.format_rule_lookup("RULE ONE", registry)
    assert "no era registry" in text


def test_format_rule_lookup_says_so_plainly_when_never_found(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A))
    text = m.format_rule_lookup("NEVER SHIPPED RULE", registry)
    assert "was never found on the wire" in text


def test_format_rule_lookup_matches_case_insensitively(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A))
    text = m.format_rule_lookup("rule one", registry)
    assert "was never found" not in text
    assert "first appeared" in text


def test_format_rule_lookup_reports_still_on_the_wire(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, _ERA_B))
    text = m.format_rule_lookup("RULE TWO", registry)  # only in era B, the newest
    assert "first appeared : 2026-02-01T00:00:00 (era B label)" in text
    assert "still on the wire as of the newest known era (era B label)" in text
    assert "present in 1/2 known era(s)" in text


def test_format_rule_lookup_reports_disappearance(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, _ERA_B, _ERA_C))
    # RULE ONE: present in A and B, absent in C.
    text = m.format_rule_lookup("RULE ONE", registry)
    assert "first appeared : 2026-01-01T00:00:00 (era A label)" in text
    assert "last present in era B label" in text
    assert "absent starting era C label" in text
    assert "present in 2/3 known era(s)" in text


def test_format_rule_lookup_includes_measured_metrics_when_available(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A))
    metrics_by_era = {_ERA_A[0]: m.era_metrics(_ERA_A[0], [_SUSHI])}
    text = m.format_rule_lookup("RULE ONE", registry, metrics_by_era)
    assert "measured (DRAFTS only" in text
    assert "n=1" in text


def test_format_rule_lookup_reports_no_local_openers_when_metrics_are_empty(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A))
    metrics_by_era = {_ERA_A[0]: m.era_metrics(_ERA_A[0], [])}
    text = m.format_rule_lookup("RULE ONE", registry, metrics_by_era)
    assert "no local openers recorded" in text


def test_format_rule_lookup_reports_metrics_not_checked_when_none_supplied(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A))
    text = m.format_rule_lookup("RULE ONE", registry, None)
    assert "not checked" in text


# ---------------------------------------------------------------------------------------
# format_rule_lookup -- the working tree section. This is the exact "have we tried this before"
# gap named by the task: a rule that is NOT in any committed era but IS in the current
# uncommitted working tree (e.g. NO GRADING, COMPLIMENT AS REMARK) must be reported as present,
# never silently absorbed into "never found on the wire".
# ---------------------------------------------------------------------------------------

def test_format_rule_lookup_reports_a_rule_present_only_in_the_working_tree(tmp_path):
    wt = _working_tree_entry(digest="d" * 64, rules=("NO GRADING",))
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, working_tree=wt))
    text = m.format_rule_lookup("NO GRADING", registry)
    # Never found in COMMITTED history...
    assert "was never found on the wire in any of the 1 known COMMITTED prompt era(s)" in text
    # ...but IS on the wire in the uncommitted working tree.
    assert "UNCOMMITTED WORKING TREE" in text
    assert "'NO GRADING' IS present" in text
    assert "d" * 64 in text


def test_format_rule_lookup_reports_a_rule_absent_from_the_working_tree_too(tmp_path):
    wt = _working_tree_entry(digest="d" * 64, rules=("NO GRADING",))
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, working_tree=wt))
    text = m.format_rule_lookup("NEVER SHIPPED OR WRITTEN", registry)
    assert "was never found on the wire" in text
    assert "is NOT present in the current uncommitted working tree" in text


def test_format_rule_lookup_working_tree_section_present_even_when_rule_ships_in_history(tmp_path):
    # A rule that DID ship AND is still in the working tree -- both sections must agree.
    wt = _working_tree_entry(digest="d" * 64, rules=("RULE ONE",))
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, working_tree=wt))
    text = m.format_rule_lookup("RULE ONE", registry)
    assert "present in 1/1 known era(s)" in text
    assert "'RULE ONE' IS present" in text  # also still in the working tree


def test_format_rule_lookup_reports_no_working_tree_entry_when_registry_predates_the_feature(tmp_path):
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A))  # no working_tree key
    text = m.format_rule_lookup("RULE ONE", registry)
    assert "not evaluated" in text
    assert "no 'working_tree' entry" in text


def test_format_rule_lookup_reports_an_unevaluable_working_tree(tmp_path):
    wt = _working_tree_entry(digest=None, reasons=["opener.py does not parse: SyntaxError"])
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, working_tree=wt))
    text = m.format_rule_lookup("RULE ONE", registry)
    assert "UNEVALUABLE: opener.py does not parse: SyntaxError" in text


def test_format_rule_lookup_measures_the_working_tree_digest_when_corpus_rows_exist(tmp_path):
    wt = _working_tree_entry(digest="d" * 64, rules=("NO GRADING",))
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, working_tree=wt))
    metrics_by_era = {"d" * 64: m.era_metrics("d" * 64, [_SUSHI])}
    text = m.format_rule_lookup("NO GRADING", registry, metrics_by_era)
    assert "measured (DRAFTS only" in text
    assert "n=1" in text


def test_format_rule_lookup_never_claims_measurement_for_an_empty_working_tree_bucket(tmp_path):
    wt = _working_tree_entry(digest="d" * 64, rules=("NO GRADING",))
    registry = m.load_era_registry(_write_registry(tmp_path, _ERA_A, working_tree=wt))
    metrics_by_era = {"d" * 64: m.era_metrics("d" * 64, [])}
    text = m.format_rule_lookup("NO GRADING", registry, metrics_by_era)
    assert "no local openers recorded" in text


# ---------------------------------------------------------------------------------------
# CLI end to end: --eras, --rule, --compare with a human label
# ---------------------------------------------------------------------------------------

def test_main_eras_mode_lists_known_eras(tmp_path, capsys):
    eras_file = _write_registry(tmp_path, _ERA_A, _ERA_B)
    code = m.main(["--eras-file", str(eras_file), "--eras"])
    out = capsys.readouterr().out
    assert code == 0
    assert "era A label" in out and "era B label" in out


def test_main_eras_mode_fails_loudly_without_a_registry(tmp_path, capsys):
    code = m.main(["--eras-file", str(tmp_path / "missing.json"), "--eras"])
    err = capsys.readouterr().err
    assert code == 1
    assert "no era registry" in err


def test_main_rule_mode_reports_measured_metrics_from_the_real_corpus(tmp_path, capsys):
    eras_file = _write_registry(tmp_path, _ERA_A)
    debug_dir = tmp_path / "hinge_debug"
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("run1", "hinge", "gemini-x", _SUSHI, "the view",
                            prompt_sha256=_ERA_A[0])
    finally:
        store.close()

    code = m.main(["--eras-file", str(eras_file), "--debug-dir", str(debug_dir),
                  "--db", str(db_path), "--rule", "RULE ONE"])
    out = capsys.readouterr().out
    assert code == 0
    assert "first appeared" in out
    assert "measured (DRAFTS only" in out
    assert "n=1" in out


def test_main_rule_mode_fails_loudly_without_a_registry(tmp_path, capsys):
    debug_dir, db_path = _build_corpus(tmp_path)
    code = m.main(["--eras-file", str(tmp_path / "missing.json"), "--debug-dir", str(debug_dir),
                  "--db", str(db_path), "--rule", "ANYTHING"])
    err = capsys.readouterr().err
    assert code == 1
    assert "no era registry" in err


def test_main_eras_mode_shows_the_working_tree_section(tmp_path, capsys):
    wt = _working_tree_entry(digest="d" * 64, rules=("NO GRADING",))
    eras_file = _write_registry(tmp_path, _ERA_A, working_tree=wt)
    code = m.main(["--eras-file", str(eras_file), "--eras"])
    out = capsys.readouterr().out
    assert code == 0
    assert "UNCOMMITTED WORKING TREE" in out
    assert "d" * 64 in out


def test_main_rule_mode_answers_have_we_tried_this_for_a_working_tree_only_rule(tmp_path, capsys):
    """The end-to-end reproduction of the task's own scenario: a rule that is ONLY in the
    uncommitted working tree (never shipped in any committed era) must be reported as present,
    not as 'never found on the wire'."""
    wt = _working_tree_entry(digest="d" * 64, rules=("NO GRADING", "COMPLIMENT AS REMARK"))
    eras_file = _write_registry(tmp_path, _ERA_A, working_tree=wt)
    debug_dir, db_path = _build_corpus(tmp_path)

    code = m.main(["--eras-file", str(eras_file), "--debug-dir", str(debug_dir),
                  "--db", str(db_path), "--rule", "NO GRADING"])
    out = capsys.readouterr().out
    assert code == 0
    assert "was never found on the wire in any of the 1 known COMMITTED prompt era(s)" in out
    assert "UNCOMMITTED WORKING TREE" in out
    assert "'NO GRADING' IS present" in out


def test_main_compare_by_human_label_end_to_end(tmp_path, capsys):
    eras_file = _write_registry(tmp_path, _ERA_A, _ERA_B)
    debug_dir = tmp_path / "hinge_debug"
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("run1", "hinge", "gemini-x", _SUSHI, "the view",
                            prompt_sha256=_ERA_A[0])
        store.record_opener("run2", "hinge", "gemini-x", _HYPERBOLE, "the view",
                            prompt_sha256=_ERA_B[0])
    finally:
        store.close()

    code = m.main(["--eras-file", str(eras_file), "--debug-dir", str(debug_dir),
                  "--db", str(db_path), "--compare", "era A label", "era B label"])
    out = capsys.readouterr().out
    assert code == 0
    assert "[era A label]" in out and "[era B label]" in out


def test_main_default_text_report_annotates_eras_with_registry_labels(tmp_path, capsys):
    eras_file = _write_registry(tmp_path, _ERA_A)
    debug_dir, db_path = _build_corpus(tmp_path)  # writes rows under era "era-b" (not in registry)
    code = m.main(["--eras-file", str(eras_file), "--debug-dir", str(debug_dir),
                  "--db", str(db_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "unregistered digest" in out  # "era-b" is not a digest in this registry


# =========================================================================================
# OUTCOMES AXIS -- did the openers written under prompt era X actually PERFORM better. Covers
# read_sqlite_opener_outcomes/read_bigquery_opener_outcomes (the join), outcome_metrics/
# build_outcome_metrics (the HARD HONESTY GUARD), the replay/unattributed exclusions, and the
# axis's presence in --json/--compare. See this module's OUTCOMES section and BE HONEST caveats
# 5-8 for the full design rationale each test below is pinning.
# =========================================================================================

# ---------------------------------------------------------------------------------------
# read_sqlite_opener_outcomes -- the join
# ---------------------------------------------------------------------------------------

def test_read_sqlite_opener_outcomes_joins_by_app_and_profile_key(tmp_path):
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("run1", "hinge", "gemini-x", "Opener one.", "the view",
                            prompt_sha256="era-a", profile_key="pk1")
        store.record_opener_outcome("hinge", "pk1", "reply")
    finally:
        store.close()

    rows, stats = m.read_sqlite_opener_outcomes(db_path)

    assert stats.available is True
    assert stats.joined_row_count == 1
    assert stats.replay_excluded_count == 0
    assert rows == [m.OutcomeRow(era="era-a", outcome="reply", read_source="sqlite")]


def test_read_sqlite_opener_outcomes_never_joins_across_an_app_boundary(tmp_path):
    # THE (app, profile_key) REQUIREMENT: the same profile_key under a DIFFERENT app must never
    # join -- proves the join predicate actually compares `oc.app = o.app`, not profile_key alone.
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("run1", "hinge", "gemini-x", "Hinge opener.", "the view",
                            prompt_sha256="era-a", profile_key="shared-key")
        store.record_opener_outcome("bumble", "shared-key", "match")  # different app, same key
    finally:
        store.close()

    rows, stats = m.read_sqlite_opener_outcomes(db_path)

    assert rows == []
    assert stats.joined_row_count == 0


def test_read_sqlite_opener_outcomes_excludes_unattributed_rows_on_both_sides(tmp_path):
    # THE BUG THIS PINS: an `openers` row with no derivable profile_key ("") and an
    # `opener_outcomes` row with no derivable profile_key ("") must NEVER join to each other just
    # because both happen to be "" -- that would attribute a real observation to an unrelated,
    # equally-unattributed opener. See BE HONEST caveat 6.
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("run1", "hinge", "gemini-x", "Unattributed opener.", "the view",
                            prompt_sha256="era-a", profile_key="")
        store.record_opener_outcome("hinge", "", "match")
    finally:
        store.close()

    rows, stats = m.read_sqlite_opener_outcomes(db_path)

    assert rows == []
    assert stats.joined_row_count == 0


def test_read_sqlite_opener_outcomes_excludes_synthetic_replay_rows(tmp_path):
    # THE BUG THIS PINS (task requirement 3): a synthetic replay row's decision is
    # DECISION_REPLAY -- even if it somehow carried a REAL, non-empty profile_key (defense in
    # depth: today opener_replay.py never gives one a real profile_key at all, but this
    # exclusion must not depend on that staying true), it must never be counted as a real
    # outcome, and never as "no_response" by omission either -- it must be dropped outright.
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("replay_run", "hinge", "gemini-x", "Replay opener.", "the view",
                            prompt_sha256="era-a", profile_key="pk-replay",
                            decision=m.DECISION_REPLAY)
        store.record_opener_outcome("hinge", "pk-replay", "reply")
    finally:
        store.close()

    rows, stats = m.read_sqlite_opener_outcomes(db_path)

    assert rows == []
    assert stats.joined_row_count == 0
    assert stats.replay_excluded_count == 1
    # MUTATION CHECK: delete the `if decision == DECISION_REPLAY: ... continue` branch inside
    # read_sqlite_opener_outcomes() and re-run this test -- it fails (rows == [OutcomeRow(era=
    # "era-a", outcome="reply", read_source="sqlite")], stats.joined_row_count == 1,
    # stats.replay_excluded_count == 0), proving this assertion is load-bearing. Restored exactly
    # afterward.


def test_read_sqlite_opener_outcomes_real_row_alongside_an_excluded_replay_row(tmp_path):
    # A real, attributed outcome must still be counted even when a replay row sits in the same
    # database -- the exclusion must be PER ROW, never an all-or-nothing fail-closed on the table.
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("replay_run", "hinge", "gemini-x", "Replay opener.", "the view",
                            prompt_sha256="era-a", profile_key="pk-replay",
                            decision=m.DECISION_REPLAY)
        store.record_opener_outcome("hinge", "pk-replay", "reply")
        store.record_opener("run1", "hinge", "gemini-x", "Real opener.", "the view",
                            prompt_sha256="era-a", profile_key="pk-real", decision="like")
        store.record_opener_outcome("hinge", "pk-real", "match")
    finally:
        store.close()

    rows, stats = m.read_sqlite_opener_outcomes(db_path)

    assert rows == [m.OutcomeRow(era="era-a", outcome="match", read_source="sqlite")]
    assert stats.joined_row_count == 1
    assert stats.replay_excluded_count == 1


def test_read_sqlite_opener_outcomes_missing_file_reports_a_note_not_an_error(tmp_path):
    rows, stats = m.read_sqlite_opener_outcomes(tmp_path / "nope.db")
    assert rows == []
    assert stats.available is False
    assert any("no database file" in note for note in stats.notes)


def test_read_sqlite_opener_outcomes_missing_table_reports_a_note(tmp_path):
    # A schema that predates this axis entirely -- no 'opener_outcomes' table at all.
    db_path = tmp_path / "old.db"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE openers (id INTEGER PRIMARY KEY)")
    con.commit()
    con.close()

    rows, stats = m.read_sqlite_opener_outcomes(db_path)

    assert rows == []
    assert stats.available is True
    assert any("no 'opener_outcomes' table" in note for note in stats.notes)


def test_read_sqlite_opener_outcomes_no_openers_table_reports_a_note(tmp_path):
    db_path = tmp_path / "outcomes_only.db"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE opener_outcomes (id INTEGER PRIMARY KEY)")
    con.commit()
    con.close()

    rows, stats = m.read_sqlite_opener_outcomes(db_path)

    assert rows == []
    assert any("'openers' does not exist" in note or "nothing to join" in note
              for note in stats.notes)


# ---------------------------------------------------------------------------------------
# read_bigquery_opener_outcomes -- fake client, no google-cloud-bigquery / network needed
# ---------------------------------------------------------------------------------------

def test_read_bigquery_opener_outcomes_reads_and_joins():
    client = _FakeBQOutcomesClient(rows=[
        {"prompt_sha256": "era-a", "outcome": "reply", "decision": "like"},
    ])

    rows, stats = m.read_bigquery_opener_outcomes("proj-1", "operation_love", client=client)

    assert stats.available is True
    assert stats.joined_row_count == 1
    assert rows == [m.OutcomeRow(era="era-a", outcome="reply", read_source="bigquery")]


def test_read_bigquery_opener_outcomes_excludes_synthetic_replay_rows():
    client = _FakeBQOutcomesClient(rows=[
        {"prompt_sha256": "era-a", "outcome": "reply", "decision": m.DECISION_REPLAY},
        {"prompt_sha256": "era-a", "outcome": "match", "decision": "like"},
    ])

    rows, stats = m.read_bigquery_opener_outcomes("proj-1", "operation_love", client=client)

    assert rows == [m.OutcomeRow(era="era-a", outcome="match", read_source="bigquery")]
    assert stats.joined_row_count == 1
    assert stats.replay_excluded_count == 1


def test_read_bigquery_opener_outcomes_missing_table_reports_a_note_not_a_crash():
    client = _FakeBQOutcomesClient(raises=RuntimeError("NotFound: opener_outcomes"))

    rows, stats = m.read_bigquery_opener_outcomes("proj-1", "operation_love", client=client)

    assert rows == []
    assert stats.available is False
    assert any("opener_outcomes" in note for note in stats.notes)


def test_read_bigquery_opener_outcomes_rejects_an_invalid_project_id():
    rows, stats = m.read_bigquery_opener_outcomes(
        "not a valid id!", "operation_love", client=_FakeBQOutcomesClient())
    assert rows == []
    assert any("invalid BigQuery config" in note for note in stats.notes)


# ---------------------------------------------------------------------------------------
# outcome_metrics / build_outcome_metrics -- the HARD HONESTY GUARD
# ---------------------------------------------------------------------------------------

def test_outcome_metrics_counts_by_kind_and_response_rate_at_the_threshold():
    # Exactly MIN_SAMPLE_FOR_OUTCOME_RATE (10): the rate must be COMPUTED, not blocked.
    outcomes = ["match"] * 3 + ["reply"] * 4 + ["no_response"] * 3
    metrics = m.outcome_metrics("era-a", outcomes)
    assert metrics.n == 10
    assert metrics.counts == {"match": 3, "no_response": 3, "reply": 4}
    assert metrics.response_count == 7  # match + reply, never no_response
    assert metrics.rate_blocked is False
    assert metrics.response_rate == pytest.approx(0.7)


def test_outcome_metrics_below_threshold_blocks_the_rate_but_keeps_the_counts():
    # One BELOW the threshold (9): counts must still be exact and complete; the rate must be
    # withheld, never silently computed anyway.
    outcomes = ["reply"] * 9
    metrics = m.outcome_metrics("era-a", outcomes)
    assert metrics.n == 9
    assert metrics.counts == {"reply": 9}
    assert metrics.response_count == 9
    assert metrics.rate_blocked is True
    assert metrics.response_rate is None
    # MUTATION CHECK: change `rate_blocked = n < MIN_SAMPLE_FOR_OUTCOME_RATE` to
    # `n <= MIN_SAMPLE_FOR_OUTCOME_RATE` (off-by-one in the other direction) -- this test still
    # passes (9 < 10 either way), but test_outcome_metrics_counts_by_kind_and_response_rate_at_
    # the_threshold above then FAILS (10 <= 10 would wrongly block a rate at exactly the
    # threshold) -- the two tests together pin the exact boundary. Change the comparison to
    # `n < MIN_SAMPLE_FOR_OUTCOME_RATE - 1` instead and THIS test fails (9 is no longer < 9),
    # proving this assertion is load-bearing. Restored exactly afterward.


def test_outcome_metrics_empty_bucket_blocks_the_rate_without_dividing_by_zero():
    metrics = m.outcome_metrics("empty", [])
    assert metrics.n == 0
    assert metrics.counts == {}
    assert metrics.response_count == 0
    assert metrics.rate_blocked is True
    assert metrics.response_rate is None


def test_outcome_metrics_unmatch_is_reported_but_never_counted_as_a_response():
    # "unmatch" is reported in `counts` in full, but is deliberately excluded from
    # `response_count`/`response_rate` -- see this module's _RESPONSE_OUTCOME_KINDS comment.
    outcomes = ["match"] * 5 + ["unmatch"] * 5
    metrics = m.outcome_metrics("era-a", outcomes)
    assert metrics.n == 10
    assert metrics.counts == {"match": 5, "unmatch": 5}
    assert metrics.response_count == 5  # unmatch does NOT count
    assert metrics.response_rate == pytest.approx(0.5)
    # MUTATION CHECK: add OPENER_OUTCOME_UNMATCH to _RESPONSE_OUTCOME_KINDS -- this test then
    # fails (response_count becomes 10, response_rate becomes 1.0), proving unmatch's exclusion
    # is load-bearing and actually enforced, not just documented. Restored exactly afterward.


def test_outcome_metrics_unknown_and_none_outcome_kinds_never_count_as_a_response():
    outcomes = ["unknown"] * 5 + ["(none)"] * 5
    metrics = m.outcome_metrics("era-a", outcomes)
    assert metrics.response_count == 0
    assert metrics.response_rate == pytest.approx(0.0)


def test_build_outcome_metrics_groups_by_era_and_buckets_none_as_unknown():
    rows = [
        m.OutcomeRow(era="era-a", outcome="reply", read_source="sqlite"),
        m.OutcomeRow(era="era-a", outcome="match", read_source="sqlite"),
        m.OutcomeRow(era="era-b", outcome="no_response", read_source="sqlite"),
        m.OutcomeRow(era=None, outcome="unknown", read_source="sqlite"),
    ]
    grouped = m.build_outcome_metrics(rows)
    assert set(grouped) == {"era-a", "era-b", m.UNKNOWN_ERA, "ALL"}
    assert grouped["era-a"].n == 2
    assert grouped["era-b"].n == 1
    assert grouped[m.UNKNOWN_ERA].n == 1
    assert grouped["ALL"].n == 4


def test_build_outcome_metrics_empty_rows_still_has_an_all_bucket():
    grouped = m.build_outcome_metrics([])
    assert set(grouped) == {"ALL"}
    assert grouped["ALL"].n == 0


def test_outcome_bucket_or_empty_returns_a_real_bucket_when_present():
    grouped = m.build_outcome_metrics([m.OutcomeRow(era="era-a", outcome="reply",
                                                    read_source="sqlite")])
    bucket = m.outcome_bucket_or_empty(grouped, "era-a")
    assert bucket.n == 1


def test_outcome_bucket_or_empty_returns_an_empty_bucket_for_a_missing_era():
    grouped = m.build_outcome_metrics([])
    bucket = m.outcome_bucket_or_empty(grouped, "era-never-seen")
    assert bucket.n == 0
    assert bucket.rate_blocked is True
    assert bucket.response_rate is None


# ---------------------------------------------------------------------------------------
# format_outcome_metrics -- the guard's printed message
# ---------------------------------------------------------------------------------------

def test_format_outcome_metrics_prints_the_small_n_refusal_message(tmp_path):
    metrics = m.outcome_metrics("era-a", ["reply"] * 3)
    text = m.format_outcome_metrics(metrics)
    assert "n too small for a rate" in text
    assert "n=3" in text
    assert str(m.MIN_SAMPLE_FOR_OUTCOME_RATE) in text
    # The raw counts must still be printed even though the rate is withheld.
    assert "reply" in text
    assert "3" in text
    # Never a bare percentage sign standing in for the withheld rate.
    assert "response_rate          : 100.0%" not in text


def test_format_outcome_metrics_prints_the_rate_at_the_threshold(tmp_path):
    metrics = m.outcome_metrics("era-a", ["reply"] * 10)
    text = m.format_outcome_metrics(metrics)
    assert "n too small for a rate" not in text
    assert "100.0%" in text


def test_format_outcome_metrics_empty_bucket_says_so():
    metrics = m.outcome_metrics("era-a", [])
    text = m.format_outcome_metrics(metrics)
    assert "no owner-observed outcomes joined" in text


# ---------------------------------------------------------------------------------------
# compare_outcome_eras / format_compare_outcomes
# ---------------------------------------------------------------------------------------

def test_compare_outcome_eras_computes_a_delta_when_both_sides_pass_the_guard():
    a = m.outcome_metrics("a", ["reply"] * 5 + ["no_response"] * 5)     # 50%
    b = m.outcome_metrics("b", ["reply"] * 8 + ["no_response"] * 2)     # 80%
    cmp = m.compare_outcome_eras(a, b)
    assert cmp["rate_blocked"] is False
    assert cmp["a_response_rate"] == pytest.approx(0.5)
    assert cmp["b_response_rate"] == pytest.approx(0.8)
    assert cmp["delta_response_rate"] == pytest.approx(0.3)


def test_compare_outcome_eras_blocks_the_delta_when_either_side_is_below_threshold():
    # THE BUG THIS PINS: a delta must never be computed from a withheld rate -- if the guard
    # blocks EITHER side, the whole comparison must say so, never substitute 0.0 or silently use
    # only the side that happened to pass.
    a = m.outcome_metrics("a", ["reply"] * 9)             # below threshold, blocked
    b = m.outcome_metrics("b", ["reply"] * 10)            # at threshold, would compute 1.0
    cmp = m.compare_outcome_eras(a, b)
    assert cmp["rate_blocked"] is True
    assert cmp["delta_response_rate"] is None
    # MUTATION CHECK: change `blocked = a.rate_blocked or b.rate_blocked` to
    # `blocked = a.rate_blocked and b.rate_blocked` -- this test then fails (blocked becomes
    # False since only `a` is blocked, and delta_response_rate becomes a real number computed
    # from b.response_rate minus a withheld a.response_rate, i.e. a TypeError or a nonsense
    # value), proving the `or` is load-bearing. Restored exactly afterward.


def test_format_compare_outcomes_prints_the_blocked_message(tmp_path):
    a = m.outcome_metrics("a", ["reply"] * 2)
    b = m.outcome_metrics("b", ["reply"] * 2)
    text = m.format_compare_outcomes("era-a", "era-b", a, b)
    assert "n too small for a rate on at least one side" in text


def test_format_compare_outcomes_prints_the_delta_when_both_sides_pass(tmp_path):
    a = m.outcome_metrics("a", ["reply"] * 5 + ["no_response"] * 5)
    b = m.outcome_metrics("b", ["reply"] * 10)
    text = m.format_compare_outcomes("era-a", "era-b", a, b)
    assert "50.0% -> 100.0%" in text
    assert "delta +50.0 pts" in text


# ---------------------------------------------------------------------------------------
# build_report / format_text_report -- the axis carried through --json and --compare
# ---------------------------------------------------------------------------------------

def test_build_report_carries_the_by_outcome_axis(tmp_path):
    source_report = m.SourceReport(m.JsonlStats(), 0, m.SqliteStats(), 0, m.BigQueryStats(), 0, 0)
    outcome_rows = [m.OutcomeRow(era="era-a", outcome="reply", read_source="sqlite")]
    doc = m.build_report([], [], source_report, outcome_rows=outcome_rows)
    assert "by_outcome" in doc
    assert doc["by_outcome"]["era-a"]["n"] == 1
    assert doc["by_outcome"]["era-a"]["counts"] == {"reply": 1}
    assert "ALL" in doc["by_outcome"]


def test_build_report_defaults_to_an_empty_outcome_axis_when_not_supplied(tmp_path):
    # Every existing build_report(...) call across this test file omits outcome_rows entirely --
    # this pins that the default degrades to an empty, present axis, never a KeyError/crash.
    source_report = m.SourceReport(m.JsonlStats(), 0, m.SqliteStats(), 0, m.BigQueryStats(), 0, 0)
    doc = m.build_report([], [], source_report)
    assert doc["by_outcome"]["ALL"]["n"] == 0


def test_build_report_compare_includes_an_outcomes_key(tmp_path):
    source_report = m.SourceReport(m.JsonlStats(), 0, m.SqliteStats(), 2, m.BigQueryStats(), 0, 2)
    rows = [
        m.OpenerRow(text=_SUSHI, source="sqlite", era="era-a", decision="like"),
        m.OpenerRow(text=_HYPERBOLE, source="sqlite", era="era-b", decision="like"),
    ]
    outcome_rows = [m.OutcomeRow(era="era-a", outcome="reply", read_source="sqlite")]
    doc = m.build_report(rows, [], source_report, outcome_rows=outcome_rows,
                         compare=("era-a", "era-b"))
    assert "outcomes" in doc["compare"]
    assert doc["compare"]["outcomes"]["a_counts"] == {"reply": 1}
    # era-b has zero joined outcomes -- outcome_bucket_or_empty() must supply an empty bucket
    # rather than this key being absent.
    assert doc["compare"]["outcomes"]["b_counts"] == {}


def _build_outcome_corpus(tmp_path):
    """One opener/outcome pair under era-a, real profile_key, decision='like' -- the minimal
    fixture for exercising the outcomes axis end to end through main()."""
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("run1", "hinge", "gemini-x", "Sent one.", "the view",
                            prompt_sha256="era-a", profile_key="pk1", decision="like")
        store.record_opener_outcome("hinge", "pk1", "reply")
    finally:
        store.close()
    return db_path


def test_main_json_mode_carries_the_outcome_axis(tmp_path, capsys):
    db_path = _build_outcome_corpus(tmp_path)
    code = m.main(["--debug-dir", str(tmp_path / "no_such_debug_dir"), "--db", str(db_path),
                  "--json"])
    out = capsys.readouterr().out
    assert code == 0
    doc = json.loads(out)
    assert doc["by_outcome"]["era-a"]["n"] == 1
    assert doc["by_outcome"]["era-a"]["counts"] == {"reply": 1}
    assert doc["sources"]["outcomes"]["sqlite"]["joined_row_count"] == 1


def test_main_text_mode_prints_the_outcome_section(tmp_path, capsys):
    db_path = _build_outcome_corpus(tmp_path)
    code = m.main(["--debug-dir", str(tmp_path / "no_such_debug_dir"), "--db", str(db_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "=== METRICS BY OUTCOME" in out
    assert "counts by outcome kind : {'reply': 1}" in out


def test_main_compare_mode_prints_the_compare_outcomes_section(tmp_path, capsys):
    db_path = _build_outcome_corpus(tmp_path)
    code = m.main(["--debug-dir", str(tmp_path / "no_such_debug_dir"), "--db", str(db_path),
                  "--compare", "era-a", "ALL"])
    out = capsys.readouterr().out
    assert code == 0
    assert "=== COMPARE OUTCOMES:" in out


# ---------------------------------------------------------------------------------------
# BE HONEST caveats 5-8 -- the new OUTCOMES-axis caveats
# ---------------------------------------------------------------------------------------

def test_caveats_state_outcomes_are_owner_observed_and_incomplete():
    assert any("OWNER-OBSERVED" in c and "incomplete" in c for c in m._CAVEATS)


def test_caveats_state_unattributed_rows_are_excluded_not_counted_as_failure():
    caveat = next(c for c in m._CAVEATS if "unattributed" in c.lower())
    assert "EXCLUDED" in caveat
    assert "failure" in caveat


def test_caveats_state_replay_rows_are_excluded():
    caveat = next(c for c in m._CAVEATS if "synthetic_replay" in c)
    assert "excluded" in caveat.lower()
    assert "opener_replay" in caveat


def test_caveats_state_correlation_is_not_causation():
    caveat = next(c for c in m._CAVEATS if "CORRELATION" in c)
    assert "not a" in caveat.lower()
    assert "causal" in caveat.lower()


def test_caveat_four_still_intact_after_appending_new_caveats():
    # The pre-existing caveat-4 test pins m._CAVEATS[3] by index -- confirm the new caveats were
    # APPENDED (never inserted earlier), so that index is still stable.
    caveat_four = m._CAVEATS[3]
    assert "PARTIAL" in caveat_four
    assert "discard_opener" in caveat_four


# =========================================================================================
# MEASUREMENT-HONESTY FIX: synthetic replay rows (tools/opener_replay.py's DECISION_REPLAY ==
# "synthetic_replay") were already excluded from the OUTCOMES axis, but INVISIBLE and POOLED on
# the PRODUCED-side axes: decision_bucket() folded a replay row into DECISION_NOT_SENT
# indistinguishably from a real human "dislike", and METRICS BY PROMPT ERA pooled replay rows
# into whichever era they were replayed under with no marker anywhere. The fix (NOT exclusion --
# opener_replay.py's whole reason for existing is a produced-side diff across eras): replay rows
# get their own decision bucket, a SOURCES count, a dedicated caveat, and a visible marker on the
# era axis (text and --json/--compare), while remaining fully counted in every produced-side
# metric. The OUTCOMES axis's existing exclusion (tested extensively above, e.g.
# test_read_sqlite_opener_outcomes_excludes_synthetic_replay_rows) is untouched by any of this.
# =========================================================================================

# ---------------------------------------------------------------------------------------
# decision_bucket() / build_decision_metrics() / sort_decision_keys() -- the new bucket
# ---------------------------------------------------------------------------------------

def test_decision_bucket_classifies_synthetic_replay_as_its_own_bucket():
    # THE BUG THIS PINS: before this fix, a replay row fell through decision_bucket()'s
    # catch-all and came back DECISION_NOT_SENT, indistinguishable from a real human "dislike".
    assert m.decision_bucket(m.DECISION_REPLAY) == m.DECISION_REPLAY
    assert m.decision_bucket(m.DECISION_REPLAY) != m.DECISION_NOT_SENT
    assert m.decision_bucket(m.DECISION_REPLAY) != m.DECISION_SENT
    assert m.decision_bucket(m.DECISION_REPLAY) != m.DECISION_UNKNOWN
    # MUTATION CHECK: delete the `if decision == DECISION_REPLAY: return DECISION_REPLAY` branch
    # inside decision_bucket() -- re-run: decision_bucket(m.DECISION_REPLAY) comes back
    # "not_sent", both inequality assertions above fail. Verified by hand, then restored exactly.


def test_build_decision_metrics_never_pools_replay_rows_into_not_sent():
    rows = [
        m.OpenerRow(text="sent one.", source="sqlite", era="era-a", decision="like"),
        m.OpenerRow(text="rejected one.", source="sqlite", era="era-a", decision="dislike"),
        m.OpenerRow(text="replayed one.", source="sqlite", era="era-a",
                   decision=m.DECISION_REPLAY),
        m.OpenerRow(text="legacy one.", source="sqlite", era="era-a", decision=None),
    ]
    grouped = m.build_decision_metrics(rows)
    assert set(grouped) == {m.DECISION_SENT, m.DECISION_NOT_SENT, m.DECISION_REPLAY,
                            m.DECISION_UNKNOWN, "ALL"}
    assert grouped[m.DECISION_SENT].n == 1
    # THE BUG THIS PINS: not_sent must hold ONLY the real "dislike" row -- never the replay row.
    assert grouped[m.DECISION_NOT_SENT].n == 1
    assert grouped[m.DECISION_REPLAY].n == 1
    assert grouped[m.DECISION_UNKNOWN].n == 1
    assert grouped["ALL"].n == 4
    # MUTATION CHECK: revert decision_bucket() to its pre-fix 3-branch form (drop the
    # DECISION_REPLAY check) -- re-run: grouped[m.DECISION_NOT_SENT].n becomes 2 (the replay row
    # joins the real dislike), grouped[m.DECISION_REPLAY] no longer exists at all (KeyError on
    # the `set(grouped) ==` assertion), and this test fails. Verified by hand, then restored
    # exactly.


def test_sort_decision_keys_places_synthetic_replay_between_not_sent_and_unknown():
    keys = {"ALL", m.DECISION_UNKNOWN, m.DECISION_NOT_SENT, m.DECISION_SENT, m.DECISION_REPLAY}
    assert m.sort_decision_keys(keys) == [
        m.DECISION_SENT, m.DECISION_NOT_SENT, m.DECISION_REPLAY, m.DECISION_UNKNOWN, "ALL"]
    # MUTATION CHECK: remove DECISION_REPLAY from _DECISION_BUCKET_ORDER -- re-run: the returned
    # list drops "synthetic_replay" entirely (sort_decision_keys only ever emits buckets present
    # in its own order tuple), so the equality assertion fails. Verified by hand, restored.


def test_build_decision_metrics_threads_replay_count_into_its_own_bucket():
    # A consistency guarantee for JSON consumers of `by_decision`: the synthetic_replay bucket's
    # own replay_count must equal its own `n` (every row in it IS a replay row, by
    # decision_bucket()'s construction), never the era_metrics() default of 0 -- so
    # doc["by_decision"]["synthetic_replay"] never looks like a bucket with zero replay rows in
    # it despite being nothing BUT replay rows.
    rows = [
        m.OpenerRow(text="sent one.", source="sqlite", era="era-a", decision="like"),
        m.OpenerRow(text="replayed one.", source="sqlite", era="era-a",
                   decision=m.DECISION_REPLAY),
    ]
    grouped = m.build_decision_metrics(rows)
    assert grouped[m.DECISION_REPLAY].n == 1
    assert grouped[m.DECISION_REPLAY].replay_count == 1
    assert grouped[m.DECISION_SENT].replay_count == 0
    # MUTATION CHECK: drop the `replay_count=bucket_replay_counts.get(bucket, 0)` keyword (and
    # the ALL bucket's own replay_count=...) from build_decision_metrics()'s era_metrics() calls
    # -- re-run: grouped[m.DECISION_REPLAY].replay_count reads back 0 (era_metrics()'s own
    # default), so that assertion fails. Verified by hand, restored exactly.


# ---------------------------------------------------------------------------------------
# SOURCES: the synthetic-replay row count, mirroring the OUTCOMES path's own
# "N synthetic-replay row(s) excluded" line -- except these are INCLUDED, never excluded.
# ---------------------------------------------------------------------------------------

def test_read_sqlite_openers_counts_synthetic_replay_rows_separately(tmp_path):
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("run1", "hinge", "gemini-x", "Real sent opener.", "the view",
                            prompt_sha256="era-a", decision="like")
        store.record_opener("replay_run", "hinge", "gemini-x", "Replayed opener.", "the dog",
                            prompt_sha256="era-a", decision=m.DECISION_REPLAY)
    finally:
        store.close()

    rows, _rejections, stats = m.read_sqlite_openers(db_path)

    assert stats.openers_row_count == 2
    assert stats.replay_row_count == 1
    by_text = {r.text: r.decision for r in rows}
    assert by_text["Replayed opener."] == m.DECISION_REPLAY
    # MUTATION CHECK: delete the `if normalized_decision == DECISION_REPLAY:
    # stats.replay_row_count += 1` line inside read_sqlite_openers() -- re-run:
    # stats.replay_row_count stays 0, the count assertion fails. Verified by hand, restored.


def test_read_bigquery_openers_counts_synthetic_replay_rows_separately():
    client = _FakeBQClient({
        "openers": [{"opener": "Real BQ opener.", "prompt_sha256": "era-a", "decision": "like"},
                   {"opener": "Replayed BQ opener.", "prompt_sha256": "era-a",
                    "decision": m.DECISION_REPLAY}],
    })

    rows, _rejections, stats = m.read_bigquery_openers("proj-1", "operation_love", client=client)

    assert stats.openers_row_count == 2
    assert stats.replay_row_count == 1
    # MUTATION CHECK: same deletion as the sqlite sibling test above, in
    # read_bigquery_openers() -- re-run: stats.replay_row_count stays 0, fails. Restored.


def test_format_source_report_prints_the_synthetic_replay_row_count():
    source_report = m.SourceReport(
        m.JsonlStats(), 0,
        m.SqliteStats(available=True, openers_row_count=3, replay_row_count=1), 3,
        m.BigQueryStats(attempted=True, available=True, openers_row_count=1, replay_row_count=1),
        1, 3)
    text = m.format_source_report(source_report)
    lines = text.splitlines()
    sqlite_line = next(line for line in lines if line.strip().startswith("sqlite"))
    bigquery_line = next(line for line in lines if line.strip().startswith("bigquery"))
    assert "1 synthetic-replay row(s) included" in sqlite_line
    assert "1 synthetic-replay row(s) included" in bigquery_line
    # MUTATION CHECK: revert the sqlite/bigquery lines in format_source_report() to their
    # pre-fix text (drop the "synthetic-replay row(s) included" clause) -- re-run: both
    # `next(...)` substring assertions fail. Verified by hand, restored exactly.


# ---------------------------------------------------------------------------------------
# BE HONEST: the new caveat -- a replayed era is GENERATED, not captured from live use
# ---------------------------------------------------------------------------------------

def test_caveats_state_a_replayed_era_is_generated_not_live():
    caveat = next(c for c in m._CAVEATS if "GENERATED" in c)
    assert "not captured from live use" in caveat
    assert "no outcomes by construction" in caveat
    assert "opener_replay.py" in caveat


def test_caveat_four_still_intact_after_the_replay_caveat_too():
    # The pre-existing caveat-4 tests re-check index 3 after caveats 5-8 were appended; this
    # re-checks it once more now that a NINTH caveat has been appended, confirming the new one
    # was appended (never inserted earlier) so every existing index -- caveat 4 included --
    # stays stable.
    assert "PARTIAL" in m._CAVEATS[3]
    assert len(m._CAVEATS) == 9
    # MUTATION CHECK: delete the new caveat tuple entry from _CAVEATS -- re-run:
    # test_caveats_state_a_replayed_era_is_generated_not_live above fails with StopIteration
    # (next() finds nothing containing "GENERATED"), and this test's `len(...) == 9` assertion
    # fails too (back to 8). Verified by hand, restored exactly.


# ---------------------------------------------------------------------------------------
# synthetic_status() / era_synthetic_marker() -- the wholly/partial/none verdict and its header
# ---------------------------------------------------------------------------------------

def test_synthetic_status_classifies_none_partial_and_wholly():
    assert m.synthetic_status(0, 0) == m.SYNTHETIC_NONE
    assert m.synthetic_status(4, 0) == m.SYNTHETIC_NONE
    assert m.synthetic_status(4, 4) == m.SYNTHETIC_WHOLLY
    assert m.synthetic_status(4, 2) == m.SYNTHETIC_PARTIAL


def test_era_synthetic_marker_empty_when_nothing_is_synthetic():
    metrics = m.era_metrics("era-a", [_SUSHI])
    assert m.era_synthetic_marker(metrics) == ""


def test_era_synthetic_marker_names_wholly_synthetic_eras():
    metrics = m.era_metrics("era-a", [_SUSHI], replay_count=1)
    marker = m.era_synthetic_marker(metrics)
    assert "WHOLLY SYNTHETIC" in marker
    assert "opener_replay.py" in marker


def test_era_synthetic_marker_names_partial_with_the_exact_counts():
    metrics = m.era_metrics("era-a", [_SUSHI, _HYPERBOLE], replay_count=1)
    marker = m.era_synthetic_marker(metrics)
    assert "PARTIALLY SYNTHETIC" in marker
    assert "1/2" in marker
    # MUTATION CHECK (all three era_synthetic_marker tests): make era_synthetic_marker() always
    # `return ""` -- re-run: the wholly/partial tests above both fail (empty string has neither
    # "WHOLLY SYNTHETIC" nor "PARTIALLY SYNTHETIC" in it); the "empty when nothing is synthetic"
    # test still passes (a false negative it can't catch alone, which is exactly why the other
    # two exist). Verified by hand, restored exactly.


# ---------------------------------------------------------------------------------------
# build_era_metrics() threads replay_count per era -- the ERA axis becomes marker-able
# ---------------------------------------------------------------------------------------

def test_build_era_metrics_threads_replay_count_per_era():
    rows = [
        m.OpenerRow(text="alpha.", source="sqlite", era="era-a", decision="like"),
        m.OpenerRow(text="beta.", source="sqlite", era="era-a", decision=m.DECISION_REPLAY),
        m.OpenerRow(text="gamma.", source="sqlite", era="era-b", decision="like"),
    ]
    grouped = m.build_era_metrics(rows)
    assert grouped["era-a"].n == 2
    assert grouped["era-a"].replay_count == 1
    assert m.synthetic_status(grouped["era-a"].n, grouped["era-a"].replay_count) == (
        m.SYNTHETIC_PARTIAL)
    assert grouped["era-b"].replay_count == 0
    assert grouped["ALL"].replay_count == 1
    # MUTATION CHECK: drop the `replay_count=replay_counts.get(era, 0)` keyword (and the ALL
    # bucket's own replay_count=...) from build_era_metrics()'s era_metrics() calls -- re-run:
    # every replay_count above reads back 0 (era_metrics()'s own default), so
    # grouped["era-a"].replay_count == 1 fails. Verified by hand, restored exactly.


# ---------------------------------------------------------------------------------------
# format_era_metrics() / format_compare() -- the marker is printed on the ERA axis (and BOTH
# --compare sides), never on the DECISION axis (which already has its own unmistakable bucket).
# ---------------------------------------------------------------------------------------

def test_format_era_metrics_era_axis_shows_the_synthetic_marker():
    metrics = m.era_metrics("era-a", [_SUSHI], replay_count=1)
    text = m.format_era_metrics(metrics, label="era")
    assert "WHOLLY SYNTHETIC" in text.splitlines()[0]
    # MUTATION CHECK: remove the `if label == "era": header += era_synthetic_marker(m)` line
    # from format_era_metrics() -- re-run: the header no longer contains the marker, fails.
    # Verified by hand, restored exactly.


def test_format_era_metrics_decision_axis_never_shows_the_synthetic_marker():
    # Even a bucket that IS wholly replay (n == replay_count) must never print the bracketed
    # marker on the DECISION axis: it already has its own unmistakable bucket name
    # ("synthetic_replay") for exactly this, and this axis's existing header-bracket contract
    # (see the pre-existing sibling test
    # test_format_era_metrics_decision_axis_never_attempts_era_label_resolution) is "no '[' on
    # the first line at all".
    metrics = m.era_metrics(m.DECISION_REPLAY, [_SUSHI], replay_count=1)
    text = m.format_era_metrics(metrics, label="decision")
    assert "[" not in text.splitlines()[0]
    # MUTATION CHECK: change the `if label == "era":` guard around the marker append to run
    # unconditionally -- re-run: the decision-axis header now contains "[WHOLLY SYNTHETIC...]",
    # so "[" appears on the first line and this assertion fails. Verified by hand, restored.


def test_format_compare_marks_a_wholly_synthetic_side_but_not_a_live_one():
    a = m.era_metrics("era-a", [_SUSHI], replay_count=1)    # wholly synthetic
    b = m.era_metrics("era-b", [_HYPERBOLE])                 # live, no replay rows at all
    text = m.format_compare("era-a", "era-b", a, b)
    header = text.splitlines()[0]
    assert "WHOLLY SYNTHETIC" in header
    assert header.count("SYNTHETIC") == 1  # era-b's side must not be marked
    # MUTATION CHECK: drop the `{era_synthetic_marker(a)}`/`{era_synthetic_marker(b)}` f-string
    # insertions from format_compare()'s header line -- re-run: "SYNTHETIC" no longer appears in
    # the header at all, both assertions fail. Verified by hand, restored exactly.


# ---------------------------------------------------------------------------------------
# build_report() -- by_era / --compare carry replay_count + synthetic_status through --json
# ---------------------------------------------------------------------------------------

def test_build_report_by_era_carries_replay_count_and_synthetic_status():
    source_report = m.SourceReport(m.JsonlStats(), 0, m.SqliteStats(), 1, m.BigQueryStats(), 0, 1)
    rows = [m.OpenerRow(text=_SUSHI, source="sqlite", era="era-a", decision=m.DECISION_REPLAY)]
    doc = m.build_report(rows, [], source_report)
    assert doc["by_era"]["era-a"]["replay_count"] == 1
    assert doc["by_era"]["era-a"]["synthetic_status"] == m.SYNTHETIC_WHOLLY


def test_build_report_compare_carries_synthetic_status_for_both_sides():
    source_report = m.SourceReport(m.JsonlStats(), 0, m.SqliteStats(), 2, m.BigQueryStats(), 0, 2)
    rows = [
        m.OpenerRow(text=_SUSHI, source="sqlite", era="era-a", decision=m.DECISION_REPLAY),
        m.OpenerRow(text=_HYPERBOLE, source="sqlite", era="era-b", decision="like"),
    ]
    doc = m.build_report(rows, [], source_report, compare=("era-a", "era-b"))
    assert doc["compare"]["era_a_synthetic_status"] == m.SYNTHETIC_WHOLLY
    assert doc["compare"]["era_a_replay_count"] == 1
    assert doc["compare"]["era_b_synthetic_status"] == m.SYNTHETIC_NONE
    assert doc["compare"]["era_b_replay_count"] == 0
    # MUTATION CHECK (both build_report tests above): delete the four
    # era_a_replay_count/era_b_replay_count/era_a_synthetic_status/era_b_synthetic_status keys
    # from build_report()'s `doc["compare"] = {...}` block -- re-run: KeyError on every access,
    # both tests fail. The by_era test is unaffected by that particular deletion (it reads
    # EraMetrics.to_dict() directly, not the compare block), which is exactly why both tests
    # exist rather than one. Verified by hand, restored exactly.


# ---------------------------------------------------------------------------------------
# main() end to end -- text, --json, and --compare all carrying the fix together against a
# real SQLite store with sent / dislike / synthetic_replay rows sharing one era.
# ---------------------------------------------------------------------------------------

def _build_replay_corpus(tmp_path):
    """One sent, one dislike, and one synthetic-replay row, all under era-a -- the minimal
    fixture for exercising this whole fix end to end through main()."""
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("run1", "hinge", "gemini-x", "Sent one.", "the view",
                            prompt_sha256="era-a", decision="like")
        store.record_opener("run1", "hinge", "gemini-x", "Rejected one.", "the dog",
                            prompt_sha256="era-a", decision="dislike")
        store.record_opener("replay_run", "hinge", "gemini-x", "Replayed one.", "the trail",
                            prompt_sha256="era-a", decision=m.DECISION_REPLAY)
    finally:
        store.close()
    return db_path


def test_main_text_mode_reports_synthetic_replay_end_to_end(tmp_path, capsys):
    db_path = _build_replay_corpus(tmp_path)
    code = m.main(["--debug-dir", str(tmp_path / "no_such_debug_dir"), "--db", str(db_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "1 synthetic-replay row(s) included" in out
    assert "decision: synthetic_replay (n=1)" in out
    # The real "dislike" alone -- never the replay row -- lands in not_sent.
    assert "decision: not_sent (n=1)" in out
    assert "PARTIALLY SYNTHETIC" in out  # era-a: 1 of 3 rows is replay
    assert "1/3" in out


def test_main_json_mode_carries_synthetic_replay_end_to_end(tmp_path, capsys):
    db_path = _build_replay_corpus(tmp_path)
    code = m.main(["--debug-dir", str(tmp_path / "no_such_debug_dir"), "--db", str(db_path),
                  "--json"])
    out = capsys.readouterr().out
    assert code == 0
    doc = json.loads(out)
    assert doc["by_decision"]["synthetic_replay"]["n"] == 1
    assert doc["by_decision"]["not_sent"]["n"] == 1
    assert doc["by_era"]["era-a"]["replay_count"] == 1
    assert doc["by_era"]["era-a"]["synthetic_status"] == "partial"
    assert doc["sources"]["sqlite"]["replay_row_count"] == 1


def test_main_compare_mode_marks_a_partially_synthetic_era_in_text(tmp_path, capsys):
    db_path = _build_replay_corpus(tmp_path)
    code = m.main(["--debug-dir", str(tmp_path / "no_such_debug_dir"), "--db", str(db_path),
                  "--compare", "era-a", "ALL"])
    out = capsys.readouterr().out
    assert code == 0
    assert "PARTIALLY SYNTHETIC" in out


def test_main_compare_json_mode_carries_synthetic_status(tmp_path, capsys):
    db_path = _build_replay_corpus(tmp_path)
    code = m.main(["--debug-dir", str(tmp_path / "no_such_debug_dir"), "--db", str(db_path),
                  "--compare", "era-a", "ALL", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    doc = json.loads(out)
    assert doc["compare"]["era_a_synthetic_status"] == "partial"
    assert doc["compare"]["era_a_replay_count"] == 1


def test_main_outcomes_axis_still_excludes_replay_rows_alongside_the_new_decision_bucket(
        tmp_path, capsys):
    # THE FULL-STACK GUARANTEE: giving replay rows their own PRODUCED-side bucket must never leak
    # them into the OUTCOMES axis, which excludes them by a completely separate mechanism
    # (read_sqlite_opener_outcomes()'s own decision check, unit-tested in isolation above by
    # test_read_sqlite_opener_outcomes_excludes_synthetic_replay_rows). Confirm both hold at once
    # against one real corpus with a real joined outcome sitting right next to a replay row with
    # ITS OWN joined outcome -- the replay row's outcome must never be counted anywhere.
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        store.record_opener("run1", "hinge", "gemini-x", "Sent one.", "the view",
                            prompt_sha256="era-a", profile_key="pk1", decision="like")
        store.record_opener_outcome("hinge", "pk1", "reply")
        store.record_opener("replay_run", "hinge", "gemini-x", "Replayed one.", "the trail",
                            prompt_sha256="era-a", profile_key="pk-replay",
                            decision=m.DECISION_REPLAY)
        store.record_opener_outcome("hinge", "pk-replay", "reply")
    finally:
        store.close()

    code = m.main(["--debug-dir", str(tmp_path / "no_such_debug_dir"), "--db", str(db_path),
                  "--json"])
    out = capsys.readouterr().out
    assert code == 0
    doc = json.loads(out)
    assert doc["by_decision"]["synthetic_replay"]["n"] == 1   # produced-side: visible
    assert doc["by_outcome"]["era-a"]["n"] == 1                # outcomes-side: still excludes it
    assert doc["sources"]["outcomes"]["sqlite"]["replay_excluded_count"] == 1


# ---------------------------------------------------------------------------------------
# REPLAY CORPUS section -- read_replay_corpus_stats() / format_replay_corpus_report(), reading
# the on-disk corpus entirely through operation_love.opener.replay_corpus's own API.
# ---------------------------------------------------------------------------------------

def test_read_replay_corpus_stats_missing_dir_reports_zero_not_an_error(tmp_path):
    stats = m.read_replay_corpus_stats(tmp_path / "does-not-exist")
    assert stats.capture_count == 0
    assert stats.distinct_prompt_eras == 0
    assert any("no captures found" in note for note in stats.notes)


def test_read_replay_corpus_stats_counts_captures_date_range_and_eras(tmp_path):
    corpus_dir = tmp_path / "corpus"
    rc.write_replay_capture(corpus_dir, items=(b"item-a",), name="A",
                            prompt_sha256="era-1", captured_at=100.0)
    rc.write_replay_capture(corpus_dir, items=(b"item-b",), name="B",
                            prompt_sha256="era-2", captured_at=200.0)
    rc.write_replay_capture(corpus_dir, items=(b"item-c",), name="C",
                            prompt_sha256="era-1", captured_at=300.0)

    stats = m.read_replay_corpus_stats(corpus_dir)

    assert stats.capture_count == 3
    assert stats.earliest_captured_at == 100.0
    assert stats.latest_captured_at == 300.0
    # THE COUNT THIS TEST PINS: two distinct captured-at prompt_sha256 VALUES ("era-1", "era-2"),
    # never three -- a naive "count every capture" implementation would report 3 here instead.
    assert stats.distinct_prompt_eras == 2
    assert stats.unknown_era_count == 0


def test_read_replay_corpus_stats_counts_captures_with_no_era_stamp_separately(tmp_path):
    corpus_dir = tmp_path / "corpus"
    rc.write_replay_capture(corpus_dir, items=(b"item-a",), name="A",
                            prompt_sha256=None, captured_at=100.0)
    rc.write_replay_capture(corpus_dir, items=(b"item-b",), name="B",
                            prompt_sha256="era-1", captured_at=200.0)

    stats = m.read_replay_corpus_stats(corpus_dir)

    assert stats.capture_count == 2
    # THE BUG THIS PINS: a None prompt_sha256 must never be folded into distinct_prompt_eras (as
    # if "no era" were itself one more distinct era) -- it lands in unknown_era_count instead.
    assert stats.distinct_prompt_eras == 1
    assert stats.unknown_era_count == 1


def test_format_replay_corpus_report_empty_corpus_reports_the_note(tmp_path):
    stats = m.read_replay_corpus_stats(tmp_path / "empty")
    text = m.format_replay_corpus_report(stats)
    assert "REPLAY CORPUS" in text
    assert "captures on disk: 0" in text
    assert "no captures found" in text


def test_format_replay_corpus_report_nonempty_corpus_reports_range_and_eras(tmp_path):
    corpus_dir = tmp_path / "corpus"
    rc.write_replay_capture(corpus_dir, items=(b"item-a",), name="A",
                            prompt_sha256="era-1", captured_at=100.0)
    rc.write_replay_capture(corpus_dir, items=(b"item-b",), name="B",
                            prompt_sha256="era-1", captured_at=9_999_999.0)

    stats = m.read_replay_corpus_stats(corpus_dir)
    text = m.format_replay_corpus_report(stats)

    assert "captures on disk: 2" in text
    assert "distinct prompt eras spanned: 1" in text
    assert "date range" in text


def test_replay_corpus_stats_to_dict_carries_every_field(tmp_path):
    corpus_dir = tmp_path / "corpus"
    rc.write_replay_capture(corpus_dir, items=(b"item-a",), name="A",
                            prompt_sha256="era-1", captured_at=100.0)
    stats = m.read_replay_corpus_stats(corpus_dir)
    d = stats.to_dict()
    assert d["capture_count"] == 1
    assert d["earliest_captured_at"] == 100.0
    assert d["latest_captured_at"] == 100.0
    assert d["distinct_prompt_eras"] == 1
    assert d["unknown_era_count"] == 0
    assert d["notes"] == []
    assert d["load_errors"] == []


# ---------------------------------------------------------------------------------------
# PRE-REGISTERED CHECK -- single_sentence_rate / compute_current_prompt_era /
# build_pre_registered_check / format_pre_registered_check
# (ops/OPENER-REDESIGN.md's 2026-09-06 (d) addendum's pre-registered NO GRADING prediction).
# ---------------------------------------------------------------------------------------

def test_single_sentence_rate_derives_from_sentence_counts():
    metrics = m.era_metrics("era", ["One sentence only.", "Two sentences. Right here."])
    assert metrics.sentence_counts == {1: 1, 2: 1}
    assert m.single_sentence_rate(metrics) == pytest.approx(0.5)


def test_single_sentence_rate_empty_bucket_is_zero_not_a_crash():
    metrics = m.era_metrics("era", [])
    assert m.single_sentence_rate(metrics) == 0.0


def test_compute_current_prompt_era_matches_prompt_stamp(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text('opener:\n  models: ["gemini-x"]\n  style: "Be warm."\n')

    era, error = m.compute_current_prompt_era(str(config_path))

    assert error is None
    assert era == prompt_stamp("Be warm.")


def test_compute_current_prompt_era_reports_an_error_for_a_missing_config(tmp_path):
    era, error = m.compute_current_prompt_era(str(tmp_path / "nope.yaml"))
    assert era is None
    assert error is not None


# The exact number of drafts needed to satisfy PRE_REGISTERED_MIN_DRAFTS's own predicate --
# every "at threshold" fixture below builds exactly this many texts so the test still pins the
# real constant rather than a hand-copied "40".
_AT_THRESHOLD = m.PRE_REGISTERED_MIN_DRAFTS


def test_build_pre_registered_check_below_threshold_reports_shortfall_and_no_verdict():
    metrics_map = {"era-a": m.era_metrics("era-a", ["Nice opener."] * 10)}

    check = m.build_pre_registered_check(
        metrics_map, current_era="era-a", current_era_error=None, replay_captures_on_disk=3)

    assert check.drafts_recorded == 10
    assert check.min_drafts_required == _AT_THRESHOLD
    assert check.shortfall == _AT_THRESHOLD - 10
    assert check.checkable is False
    assert check.predictions == []
    assert check.falsified is None


def test_build_pre_registered_check_unknown_current_era_reports_the_error_and_zero_progress():
    check = m.build_pre_registered_check(
        {}, current_era=None, current_era_error="boom", replay_captures_on_disk=0)

    assert check.current_era is None
    assert check.current_era_error == "boom"
    assert check.checkable is False
    assert check.drafts_recorded == 0
    assert check.shortfall == _AT_THRESHOLD


def test_build_pre_registered_check_current_era_absent_from_metrics_map_is_zero_drafts():
    # The era exists (no error), but nothing has been recorded under it yet -- must read as 0
    # drafts, never a KeyError.
    check = m.build_pre_registered_check(
        {"some-other-era": m.era_metrics("some-other-era", ["x."] * 50)},
        current_era="era-a", current_era_error=None, replay_captures_on_disk=0)
    assert check.drafts_recorded == 0
    assert check.checkable is False


def test_build_pre_registered_check_one_short_of_threshold_is_not_checkable():
    texts = [f"Item number {i} looks calm today." for i in range(_AT_THRESHOLD - 1)]
    metrics_map = {"era-a": m.era_metrics("era-a", texts)}

    check = m.build_pre_registered_check(
        metrics_map, current_era="era-a", current_era_error=None, replay_captures_on_disk=0)

    assert check.checkable is False
    assert check.shortfall == 1
    assert check.predictions == []


def test_build_pre_registered_check_at_threshold_is_checkable_with_four_predictions():
    texts = [f"Item number {i} looks calm today." for i in range(_AT_THRESHOLD)]
    metrics_map = {"era-a": m.era_metrics("era-a", texts)}

    check = m.build_pre_registered_check(
        metrics_map, current_era="era-a", current_era_error=None, replay_captures_on_disk=0)

    assert check.checkable is True
    assert check.shortfall == 0
    assert {p.metric for p in check.predictions} == {
        "grade_rate", "single_sentence_rate", "two_sentence_rate", "trigram_diversity"}
    assert check.falsified is not None


def test_build_pre_registered_check_evaluates_all_four_predictions_as_passing():
    # Every text: single sentence, a unique opening trigram, and no grade-shaped copula clause.
    texts = [f"Item number {i} looks calm today." for i in range(_AT_THRESHOLD)]
    metrics_map = {"era-a": m.era_metrics("era-a", texts)}

    check = m.build_pre_registered_check(
        metrics_map, current_era="era-a", current_era_error=None, replay_captures_on_disk=0)

    by_metric = {p.metric: p for p in check.predictions}
    assert by_metric["grade_rate"].value == 0.0
    assert by_metric["grade_rate"].passed is True
    assert by_metric["single_sentence_rate"].value == 1.0
    assert by_metric["single_sentence_rate"].passed is True
    assert by_metric["two_sentence_rate"].value == 0.0
    assert by_metric["two_sentence_rate"].passed is True
    assert by_metric["trigram_diversity"].value == 1.0
    assert by_metric["trigram_diversity"].passed is True
    assert check.falsified is False


def test_build_pre_registered_check_falsified_when_grade_high_and_single_sentence_low():
    # THE FALSIFICATION CRITERION verbatim from ops/OPENER-REDESIGN.md 2026-09-06 (d): grade_rate
    # >= 10% AND single_sentence_rate < 8%. 5/40 grade-shaped first beats is 12.5% (>= 10%); zero
    # single-sentence openers is 0% (< 8%).
    texts = (["That spread is an elite move. Where was it from?"] * 5
            + ["That trail looks calm today. Where was it taken?"] * 35)
    assert len(texts) == _AT_THRESHOLD
    metrics_map = {"era-a": m.era_metrics("era-a", texts)}

    check = m.build_pre_registered_check(
        metrics_map, current_era="era-a", current_era_error=None, replay_captures_on_disk=0)

    by_metric = {p.metric: p for p in check.predictions}
    assert by_metric["grade_rate"].value == pytest.approx(5 / 40)
    assert by_metric["single_sentence_rate"].value == 0.0
    assert check.falsified is True


def test_build_pre_registered_check_not_falsified_when_only_one_criterion_leg_is_met():
    # grade_rate is high (12.5%, >= 10%) but single_sentence_rate is ALSO high (not < 8%) --
    # the falsification criterion is an AND, so this must NOT be falsified even though one leg
    # alone would meet its own bar.
    texts = (["That spread is an elite move."] * 5
            + [f"What's the story behind hike number {i}?" for i in range(35)])
    assert len(texts) == _AT_THRESHOLD
    metrics_map = {"era-a": m.era_metrics("era-a", texts)}

    check = m.build_pre_registered_check(
        metrics_map, current_era="era-a", current_era_error=None, replay_captures_on_disk=0)

    by_metric = {p.metric: p for p in check.predictions}
    assert by_metric["grade_rate"].value >= m.PRE_REGISTERED_FALSIFY_GRADE_RATE_MIN
    assert by_metric["single_sentence_rate"].value >= m.PRE_REGISTERED_FALSIFY_SINGLE_SENTENCE_RATE_MAX
    assert check.falsified is False


def test_format_pre_registered_check_reports_shortfall_and_suppresses_the_verdict():
    metrics_map = {"era-a": m.era_metrics("era-a", ["Nice opener."] * 5)}
    check = m.build_pre_registered_check(
        metrics_map, current_era="era-a", current_era_error=None, replay_captures_on_disk=2)

    text = m.format_pre_registered_check(check)

    assert "NOT YET CHECKABLE" in text
    assert f"{_AT_THRESHOLD - 5} more draft(s) needed" in text
    assert "PASS" not in text
    assert "FAIL" not in text
    assert "VERDICT" not in text


def test_format_pre_registered_check_renders_pass_fail_and_verdict_at_threshold():
    texts = [f"Item number {i} looks calm today." for i in range(_AT_THRESHOLD)]
    metrics_map = {"era-a": m.era_metrics("era-a", texts)}
    check = m.build_pre_registered_check(
        metrics_map, current_era="era-a", current_era_error=None, replay_captures_on_disk=0)

    text = m.format_pre_registered_check(check)

    assert "THRESHOLD MET" in text
    assert "PASS" in text
    assert "VERDICT: NOT FALSIFIED" in text


def test_format_pre_registered_check_renders_falsified_verdict():
    texts = (["That spread is an elite move. Where was it from?"] * 5
            + ["That trail looks calm today. Where was it taken?"] * 35)
    metrics_map = {"era-a": m.era_metrics("era-a", texts)}
    check = m.build_pre_registered_check(
        metrics_map, current_era="era-a", current_era_error=None, replay_captures_on_disk=0)

    text = m.format_pre_registered_check(check)

    assert "VERDICT: FALSIFIED" in text
    assert "falsification criterion" in text


def test_format_pre_registered_check_no_current_era_reports_the_error():
    check = m.build_pre_registered_check(
        {}, current_era=None, current_era_error="boom", replay_captures_on_disk=0)
    text = m.format_pre_registered_check(check)
    assert "could not determine the current prompt era" in text
    assert "boom" in text


def test_format_pre_registered_check_notes_when_drafts_and_captures_differ():
    metrics_map = {"era-a": m.era_metrics("era-a", ["Nice opener."] * 5)}
    check = m.build_pre_registered_check(
        metrics_map, current_era="era-a", current_era_error=None, replay_captures_on_disk=2)

    text = m.format_pre_registered_check(check)

    assert "differ" in text
    assert "5" in text and "2" in text


def test_format_pre_registered_check_no_note_when_drafts_and_captures_are_equal():
    metrics_map = {"era-a": m.era_metrics("era-a", ["Nice opener."] * 5)}
    check = m.build_pre_registered_check(
        metrics_map, current_era="era-a", current_era_error=None, replay_captures_on_disk=5)

    text = m.format_pre_registered_check(check)

    assert "differ" not in text


# ---------------------------------------------------------------------------------------
# Wiring into build_report() / format_text_report() / main() -- including --json
# ---------------------------------------------------------------------------------------

def test_build_report_carries_replay_corpus_and_pre_registered_check(tmp_path):
    source_report = m.SourceReport(m.JsonlStats(), 0, m.SqliteStats(), 0, m.BigQueryStats(), 0, 0)
    replay_stats = m.ReplayCorpusStats(corpus_dir=tmp_path, capture_count=2)

    doc = m.build_report([], [], source_report, replay_corpus_stats=replay_stats,
                         current_era="era-a", current_era_error=None)

    assert doc["replay_corpus"]["capture_count"] == 2
    assert doc["pre_registered_check"]["current_era"] == "era-a"
    assert doc["pre_registered_check"]["checkable"] is False
    assert doc["pre_registered_check"]["replay_captures_on_disk"] == 2


def test_build_report_defaults_replay_corpus_and_pre_registered_check_when_omitted():
    # Every pre-existing build_report(...) call across this test file omits these new
    # keyword-only params entirely -- this pins that the default degrades to an empty,
    # present axis, never a KeyError/crash.
    source_report = m.SourceReport(m.JsonlStats(), 0, m.SqliteStats(), 0, m.BigQueryStats(), 0, 0)
    doc = m.build_report([], [], source_report)
    assert doc["replay_corpus"]["capture_count"] == 0
    assert doc["pre_registered_check"]["current_era"] is None
    assert doc["pre_registered_check"]["checkable"] is False


def _write_config(tmp_path, style="Be warm, specific, and brief."):
    path = tmp_path / "config.yaml"
    path.write_text(f'opener:\n  models: ["gemini-x"]\n  style: "{style}"\n')
    return path


def test_main_text_mode_reports_replay_corpus_and_progress_below_threshold(tmp_path, capsys):
    config_path = _write_config(tmp_path)
    era = prompt_stamp("Be warm, specific, and brief.")

    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        for i in range(5):
            store.record_opener(f"run{i}", "hinge", "gemini-x", f"Nice opener number {i}.",
                                "the view", prompt_sha256=era, decision="like")
    finally:
        store.close()

    corpus_dir = tmp_path / "replay"
    rc.write_replay_capture(corpus_dir, items=(b"item-a",), name="A", prompt_sha256=era)

    code = m.main(["--debug-dir", str(tmp_path / "debug"), "--db", str(db_path),
                  "--config", str(config_path), "--replay-corpus-dir", str(corpus_dir)])

    assert code == 0
    out = capsys.readouterr().out
    assert "=== REPLAY CORPUS" in out
    assert "captures on disk: 1" in out
    assert "=== PRE-REGISTERED CHECK" in out
    assert "drafts recorded under this era" in out
    assert "NOT YET CHECKABLE" in out
    assert "VERDICT" not in out


def test_main_text_mode_renders_verdict_once_threshold_is_met(tmp_path, capsys):
    config_path = _write_config(tmp_path)
    era = prompt_stamp("Be warm, specific, and brief.")

    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    try:
        for i in range(_AT_THRESHOLD):
            store.record_opener(f"run{i}", "hinge", "gemini-x",
                                f"Item number {i} looks calm today.", "the view",
                                prompt_sha256=era, decision="like")
    finally:
        store.close()

    code = m.main(["--debug-dir", str(tmp_path / "debug"), "--db", str(db_path),
                  "--config", str(config_path)])

    assert code == 0
    out = capsys.readouterr().out
    assert "THRESHOLD MET" in out
    assert "VERDICT" in out


def test_main_json_mode_carries_replay_corpus_and_pre_registered_check(tmp_path, capsys):
    config_path = _write_config(tmp_path)
    era = prompt_stamp("Be warm, specific, and brief.")
    corpus_dir = tmp_path / "replay"
    rc.write_replay_capture(corpus_dir, items=(b"item-a",), name="A", prompt_sha256=era)

    code = m.main(["--debug-dir", str(tmp_path / "debug"), "--db", str(tmp_path / "store.db"),
                  "--config", str(config_path), "--replay-corpus-dir", str(corpus_dir), "--json"])

    assert code == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["replay_corpus"]["capture_count"] == 1
    assert doc["pre_registered_check"]["current_era"] == era
    assert doc["pre_registered_check"]["replay_captures_on_disk"] == 1


def test_main_reports_current_era_error_when_config_is_unreadable(tmp_path, capsys):
    code = m.main(["--debug-dir", str(tmp_path / "debug"), "--db", str(tmp_path / "store.db"),
                  "--config", str(tmp_path / "nope.yaml")])

    assert code == 0
    out = capsys.readouterr().out
    assert "could not determine the current prompt era" in out


# =========================================================================================
# PRE-REGISTERED CHECK: verdict PROVENANCE
#
# An adversarial review found format_pre_registered_check() rendering a full THRESHOLD MET /
# PASS-FAIL / FALSIFIED verdict without ever disclosing whether the drafts behind it were
# tools/opener_replay.py synthetic replays or live Training/AUTO sends -- the "a replayed era
# reads as a live one" failure mode. Replayed drafts are genuine model output under the current
# prompt, so they ARE valid produced-side evidence and are deliberately still counted (making
# the prediction checkable offline is why replay exists); what must never happen is a
# replay-derived verdict presented as a live batch. These pin the disclosure.
# =========================================================================================


def test_verdict_basis_is_live_when_no_draft_is_synthetic():
    check = m.build_pre_registered_check(
        {"era-a": m.era_metrics("era-a", ["Nice opener."] * _AT_THRESHOLD, replay_count=0)},
        current_era="era-a", current_era_error=None, replay_captures_on_disk=0)
    assert check.replay_drafts == 0
    assert check.verdict_basis() == "live"


def test_verdict_basis_is_replay_when_every_draft_is_synthetic():
    check = m.build_pre_registered_check(
        {"era-a": m.era_metrics("era-a", ["Nice opener."] * _AT_THRESHOLD,
                                replay_count=_AT_THRESHOLD)},
        current_era="era-a", current_era_error=None, replay_captures_on_disk=0)
    assert check.replay_drafts == _AT_THRESHOLD
    assert check.verdict_basis() == "replay"


def test_verdict_basis_is_mixed_when_only_some_drafts_are_synthetic():
    check = m.build_pre_registered_check(
        {"era-a": m.era_metrics("era-a", ["Nice opener."] * _AT_THRESHOLD, replay_count=5)},
        current_era="era-a", current_era_error=None, replay_captures_on_disk=0)
    assert check.verdict_basis() == "mixed"


def test_pre_registered_check_reports_synthetic_count_and_labels_a_mixed_verdict():
    check = m.build_pre_registered_check(
        {"era-a": m.era_metrics("era-a", ["Nice opener."] * _AT_THRESHOLD, replay_count=5)},
        current_era="era-a", current_era_error=None, replay_captures_on_disk=0)
    out = m.format_pre_registered_check(check)
    assert "of which SYNTHETIC" in out
    assert "basis: MIXED" in out
    assert "PROVENANCE:" in out
    # The verdict line itself must carry the basis -- a reader who skims to VERDICT alone must
    # still see that it is not a live result.
    verdict_line = [ln for ln in out.splitlines() if "VERDICT:" in ln]
    assert verdict_line and "MIXED-BASED" in verdict_line[0]
    assert f"5/{_AT_THRESHOLD}" in verdict_line[0]


def test_pre_registered_check_leaves_a_live_verdict_unlabelled():
    check = m.build_pre_registered_check(
        {"era-a": m.era_metrics("era-a", ["Nice opener."] * _AT_THRESHOLD, replay_count=0)},
        current_era="era-a", current_era_error=None, replay_captures_on_disk=0)
    out = m.format_pre_registered_check(check)
    assert "basis: LIVE" in out
    assert "PROVENANCE:" not in out
    verdict_line = [ln for ln in out.splitlines() if "VERDICT:" in ln]
    assert verdict_line
    assert "-BASED" not in verdict_line[0]


def test_pre_registered_check_to_dict_carries_provenance():
    check = m.build_pre_registered_check(
        {"era-a": m.era_metrics("era-a", ["Nice opener."] * _AT_THRESHOLD, replay_count=5)},
        current_era="era-a", current_era_error=None, replay_captures_on_disk=0)
    doc = check.to_dict()
    assert doc["replay_drafts"] == 5
    assert doc["verdict_basis"] == "mixed"
