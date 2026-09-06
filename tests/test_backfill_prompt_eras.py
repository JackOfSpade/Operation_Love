"""tools/backfill_prompt_eras.py -- reconstructing prompt eras from git history.

Two kinds of test live here, deliberately kept apart:

  1. Tests against a SYNTHETIC git repo built fresh in tmp_path (see _repo() below). These cover
     era grouping, labeling, rule diffing, and the BE HONEST unevaluable-revision reporting --
     they must never depend on this actual repo's history, which keeps growing and would make a
     hardcoded commit count or digest go stale the moment someone else's unrelated commit lands.

  2. Exactly ONE test against the real, current working tree
     (test_reconstruction_matches_live_prompt_stamp_for_the_current_working_tree) -- the single
     case the task names as directly verifiable: this tool's own extraction and digest functions,
     applied to the files sitting on disk right now, must reproduce the real, imported
     operation_love.opener.opener.prompt_stamp() exactly. This is intentionally decoupled from
     git entirely (it reads the files directly), since the working tree can carry uncommitted
     edits that are not yet "history" at all (see tools/backfill_prompt_eras.py's own docstring
     on why the registry is git-history-only).

No network, no BigQuery, no imported opener.py module anywhere in these tests except the one
place above that explicitly needs the real, live prompt_stamp() to compare against.
"""
from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

from tools import backfill_prompt_eras as m

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------------------
# Synthetic git fixture repo -- local helper, no conftest.py fixture for this (this tool is the
# only consumer of a throwaway git repo in the whole suite).
# ---------------------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                            check=False)  # returncode asserted on the next line
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result.stdout


def _init_repo(repo: Path) -> None:
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")


def _commit(repo: Path, files: dict[str, str], message: str, when: str) -> str:
    """Write `files` (path -> content, relative to `repo`), stage everything, and commit with a
    fixed author/committer date `when` (e.g. "2026-01-01T00:00:00" -- git accepts a bare local
    time here) so era ordering is deterministic rather than depending on wall-clock speed."""
    for rel_path, content in files.items():
        full = repo / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
    env = dict(os.environ, GIT_AUTHOR_DATE=when, GIT_COMMITTER_DATE=when)
    _git(repo, "add", "-A")
    result = subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo,
                            capture_output=True, text=True, env=env,
                            check=False)  # returncode asserted on the next line
    assert result.returncode == 0, result.stderr
    return _git(repo, "rev-parse", "HEAD").strip()


# A minimal but structurally real opener.py: module docstring, an unrelated import (so a test
# that accidentally tried to exec/import this would fail loudly rather than silently working),
# and the six constants in the exact literal shapes prompt_stamp() covers.
def _opener_py(system: str, schema: dict, item_preamble="P", item_preamble_context="PC",
              item_label="=== ITEM {number} ===", context_label="=== CONTEXT ===",
              extra: str = "") -> str:
    return textwrap.dedent(f'''\
        """A fixture opener module."""
        import os  # unrelated import; must never be exec'd or imported by this tool

        _SYSTEM = {system!r}
        _SCHEMA = {schema!r}
        _ITEM_PREAMBLE = {item_preamble!r}
        _ITEM_PREAMBLE_CONTEXT = {item_preamble_context!r}
        _ITEM_LABEL = {item_label!r}
        _CONTEXT_LABEL = {context_label!r}
        {extra}
        ''')


def _config_yaml(style: str | None) -> str:
    if style is None:
        return "opener:\n  enabled: true\n"
    return "opener:\n  style: |\n" + textwrap.indent(style, "    ") + "\n"


FIXTURE_PATHS = ("config.yaml", "opener.py")


# ---------------------------------------------------------------------------------------
# extract_constants -- AST literal_eval, no exec, no import
# ---------------------------------------------------------------------------------------

def test_extract_constants_reads_every_named_constant():
    src = _opener_py("RULE ONE: text.", {"type": "object"})
    values, failures = m.extract_constants(src)
    assert failures == {}
    assert values["_SYSTEM"] == "RULE ONE: text."
    assert values["_SCHEMA"] == {"type": "object"}
    assert values["_ITEM_PREAMBLE"] == "P"
    assert values["_ITEM_LABEL"] == "=== ITEM {number} ==="


def test_extract_constants_reports_a_missing_constant_as_simply_absent():
    # No _ITEM_PREAMBLE/_ITEM_PREAMBLE_CONTEXT/_ITEM_LABEL/_CONTEXT_LABEL at all -- simulates a
    # pre-item-crop-redesign revision. Absent, not a failure: evaluate_revision is the one that
    # turns "absent" into a reason.
    src = textwrap.dedent('''\
        _SYSTEM = "RULE ONE: text."
        _SCHEMA = {"type": "object"}
        ''')
    values, failures = m.extract_constants(src)
    assert failures == {}
    assert set(values) == {"_SYSTEM", "_SCHEMA"}


def test_extract_constants_records_a_non_literal_assignment_as_a_failure_not_a_crash():
    src = textwrap.dedent('''\
        _HELPER = "x"
        _SYSTEM = "RULE ONE: " + _HELPER  # not a literal -- BinOp, literal_eval must reject it
        _SCHEMA = {"type": "object"}
        _ITEM_PREAMBLE = "P"
        _ITEM_PREAMBLE_CONTEXT = "PC"
        _ITEM_LABEL = "L"
        _CONTEXT_LABEL = "C"
        ''')
    values, failures = m.extract_constants(src)
    assert "_SYSTEM" not in values
    assert "_SYSTEM" in failures
    assert "not a plain literal" in failures["_SYSTEM"]


def test_extract_constants_on_a_syntax_error_fails_every_name_with_a_reason():
    values, failures = m.extract_constants("def broken(:\n    pass")
    assert values == {}
    assert set(failures) == set(m.CONST_NAMES)
    assert all("does not parse" in reason for reason in failures.values())


def test_extract_constants_last_top_level_assignment_wins():
    # Mirrors real Python module execution: a SECOND module-level assignment to the same name
    # really does overwrite the first when the module is actually imported. Getting this
    # backwards would silently reconstruct the wrong (stale) value for any revision that ever
    # reassigns one of these names.
    src = textwrap.dedent('''\
        _SYSTEM = "OLD RULE: stale."
        _SCHEMA = {"type": "object"}
        _ITEM_PREAMBLE = "P"
        _ITEM_PREAMBLE_CONTEXT = "PC"
        _ITEM_LABEL = "L"
        _CONTEXT_LABEL = "C"
        _SYSTEM = "NEW RULE: current."
        ''')
    values, failures = m.extract_constants(src)
    assert failures == {}
    assert values["_SYSTEM"] == "NEW RULE: current."


def test_extract_constants_ignores_a_same_named_local_variable_inside_a_function():
    src = textwrap.dedent('''\
        _SYSTEM = "MODULE RULE: real."
        _SCHEMA = {"type": "object"}
        _ITEM_PREAMBLE = "P"
        _ITEM_PREAMBLE_CONTEXT = "PC"
        _ITEM_LABEL = "L"
        _CONTEXT_LABEL = "C"

        def build():
            _SYSTEM = "LOCAL: should never be picked up"
            return _SYSTEM
        ''')
    values, failures = m.extract_constants(src)
    assert values["_SYSTEM"] == "MODULE RULE: real."


# ---------------------------------------------------------------------------------------
# extract_style -- plain yaml.safe_load, matching config.py's real loader
# ---------------------------------------------------------------------------------------

def test_extract_style_reads_the_style_string():
    style, reason = m.extract_style("opener:\n  style: |\n    Be kind.\n")
    assert reason is None
    assert style == "Be kind.\n"


def test_extract_style_reports_missing_opener_section():
    style, reason = m.extract_style("ranker:\n  like_threshold: 0.5\n")
    assert style is None
    assert "no top level 'opener:' section" in reason


def test_extract_style_reports_missing_style_key():
    style, reason = m.extract_style("opener:\n  enabled: true\n")
    assert style is None
    assert "no 'style' key" in reason


def test_extract_style_reports_a_non_string_style():
    style, reason = m.extract_style("opener:\n  style: 42\n")
    assert style is None
    assert "not a string" in reason


def test_extract_style_reports_invalid_yaml():
    style, reason = m.extract_style("opener: [unterminated\n")
    assert style is None
    assert "does not parse as YAML" in reason


def test_extract_style_reports_a_non_mapping_document():
    style, reason = m.extract_style("- just\n- a\n- list\n")
    assert style is None
    assert "did not parse to a mapping" in reason


# ---------------------------------------------------------------------------------------
# compute_prompt_stamp -- must match operation_love.opener.opener.prompt_stamp()'s algorithm
# exactly: NUL-joined, canonical _SCHEMA, sha256 hex.
# ---------------------------------------------------------------------------------------

def test_compute_prompt_stamp_matches_a_hand_built_reference_digest():
    import hashlib
    style, system = "STYLE", "SYSTEM"
    schema = {"b": 1, "a": 2}
    payload = "\x00".join((
        style, system,
        json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        "PREAMBLE", "PREAMBLE_CTX", "LABEL", "CTX_LABEL",
    ))
    expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    actual = m.compute_prompt_stamp(style, system, schema, "PREAMBLE", "PREAMBLE_CTX",
                                    "LABEL", "CTX_LABEL")
    assert actual == expected


def test_compute_prompt_stamp_is_sensitive_to_schema_key_order_being_a_non_issue():
    # {"a":1,"b":2} and {"b":2,"a":1} are the SAME canonical JSON (sort_keys=True), so they must
    # produce the SAME digest -- a dict literal reordering that changes no text changes no
    # digest, exactly like prompt_stamp()'s own docstring states.
    d1 = m.compute_prompt_stamp("s", "y", {"a": 1, "b": 2}, "p", "pc", "l", "c")
    d2 = m.compute_prompt_stamp("s", "y", {"b": 2, "a": 1}, "p", "pc", "l", "c")
    assert d1 == d2


def test_compute_prompt_stamp_changes_when_any_single_component_changes():
    base = ("style", "system", {"a": 1}, "preamble", "preamble_ctx", "label", "ctx_label")
    baseline = m.compute_prompt_stamp(*base)
    for i in range(len(base)):
        mutated = list(base)
        mutated[i] = {"a": 2} if i == 2 else str(mutated[i]) + "!"
        assert m.compute_prompt_stamp(*mutated) != baseline, f"component {i} did not move the digest"


def test_reconstruction_matches_live_prompt_stamp_for_the_current_working_tree():
    """The one case the task singles out as directly verifiable: this tool's own reconstruction
    functions, applied to the files ACTUALLY ON DISK right now (not via git show -- the working
    tree may carry uncommitted edits, which are not yet "history"), must reproduce the real,
    imported prompt_stamp()'s output exactly. This is what proves compute_prompt_stamp() and
    extract_constants()/extract_style() are byte-for-byte faithful to the real algorithm,
    independent of anything about git history.
    """
    from operation_love import config as cfg_mod
    from operation_love.opener.opener import prompt_stamp

    opener_src = (REPO_ROOT / "operation_love" / "opener" / "opener.py").read_text(encoding="utf-8")
    config_src = (REPO_ROOT / "config.yaml").read_text(encoding="utf-8")

    values, failures = m.extract_constants(opener_src)
    assert failures == {}, failures
    assert set(values) == set(m.CONST_NAMES)

    style, style_reason = m.extract_style(config_src)
    assert style_reason is None, style_reason

    reconstructed = m.compute_prompt_stamp(
        style, values["_SYSTEM"], values["_SCHEMA"], values["_ITEM_PREAMBLE"],
        values["_ITEM_PREAMBLE_CONTEXT"], values["_ITEM_LABEL"], values["_CONTEXT_LABEL"])

    cfg = cfg_mod.load(str(REPO_ROOT / "config.yaml"))
    live_digest = prompt_stamp(cfg.opener.style)

    assert reconstructed == live_digest


# ---------------------------------------------------------------------------------------
# extract_rule_names
# ---------------------------------------------------------------------------------------

def test_extract_rule_names_finds_multi_word_all_caps_names():
    text = "SHARED CONTEXT RULE: do this. Also NO GRADING: never that."
    assert m.extract_rule_names(text) == ["SHARED CONTEXT RULE", "NO GRADING"]


def test_extract_rule_names_keeps_internal_commas_and_apostrophes():
    text = "HEDGE THE CLAIM, NEVER YOURSELF: word it carefully."
    assert m.extract_rule_names(text) == ["HEDGE THE CLAIM, NEVER YOURSELF"]


def test_extract_rule_names_rejects_a_single_all_caps_word():
    assert m.extract_rule_names("URL: not a rule name.") == []


def test_extract_rule_names_ignores_prose_with_no_colon():
    assert m.extract_rule_names("NO GRADING is mentioned here but never as a header.") == []


def test_extract_rule_names_dedupes_keeping_first_appearance_order():
    text = "HARD RULE: first thing. Some prose. HARD RULE: second thing."
    assert m.extract_rule_names(text) == ["HARD RULE"]


def test_extract_rule_names_preserves_appearance_order_not_alphabetical():
    text = "ZEBRA RULE: z. APPLE RULE: a."
    assert m.extract_rule_names(text) == ["ZEBRA RULE", "APPLE RULE"]


# ---------------------------------------------------------------------------------------
# evaluate_revision -- BE HONEST reasons for every kind of historical gap
# ---------------------------------------------------------------------------------------

def _ref(sha="deadbeef", date="2026-01-01T00:00:00-00:00", subject="a commit"):
    return m.CommitRef(sha=sha, date=date, subject=subject)


def test_evaluate_revision_computes_a_digest_when_everything_is_present(tmp_path):
    _init_repo(tmp_path)
    sha = _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE TWO: base style."),
        "opener.py": _opener_py("RULE ONE: base system.", {"type": "object"}),
    }, "initial", "2026-01-01T00:00:00")
    ev = m.evaluate_revision(tmp_path, _ref(sha=sha, date="2026-01-01T00:00:00-00:00"),
                             FIXTURE_PATHS)
    assert ev.digest is not None
    assert ev.reasons == []
    # rule order is _SYSTEM's own text first, then style's (evaluate_revision joins
    # _SYSTEM + "\n" + style before extracting -- see its own source).
    assert ev.rules == ["RULE ONE", "RULE TWO"]


def test_evaluate_revision_reports_missing_constants_by_name(tmp_path):
    _init_repo(tmp_path)
    sha = _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE TWO: base style."),
        "opener.py": '_SYSTEM = "RULE ONE: base."\n_SCHEMA = {"type": "object"}\n',
    }, "no item constants yet", "2026-01-01T00:00:00")
    ev = m.evaluate_revision(tmp_path, _ref(sha=sha), FIXTURE_PATHS)
    assert ev.digest is None
    assert any("missing constant(s)" in r and "_ITEM_PREAMBLE" in r for r in ev.reasons)


def test_evaluate_revision_reports_a_missing_file(tmp_path):
    _init_repo(tmp_path)
    sha = _commit(tmp_path, {"config.yaml": _config_yaml("x")}, "no opener.py at all",
                  "2026-01-01T00:00:00")
    ev = m.evaluate_revision(tmp_path, _ref(sha=sha), FIXTURE_PATHS)
    assert ev.digest is None
    assert any("opener.py does not exist" in r for r in ev.reasons)


def test_evaluate_revision_reports_a_missing_style_key(tmp_path):
    _init_repo(tmp_path)
    sha = _commit(tmp_path, {
        "config.yaml": _config_yaml(None),
        "opener.py": _opener_py("RULE ONE: base.", {"type": "object"}),
    }, "no style key", "2026-01-01T00:00:00")
    ev = m.evaluate_revision(tmp_path, _ref(sha=sha), FIXTURE_PATHS)
    assert ev.digest is None
    assert any("no 'style' key" in r for r in ev.reasons)


# ---------------------------------------------------------------------------------------
# evaluate_working_tree -- the CURRENT uncommitted working tree (no git history involved at
# all: reads bytes straight off disk via _evaluate_sources(), the exact same core
# evaluate_revision() uses). These tests write files directly under tmp_path -- deliberately
# WITHOUT ever committing them -- since the entire point is that this function must work on
# bytes that are not (and may never be) in git history at all.
# ---------------------------------------------------------------------------------------

def test_evaluate_working_tree_computes_a_digest_from_uncommitted_disk_files(tmp_path):
    (tmp_path / "config.yaml").write_text(_config_yaml("RULE TWO: base style."), encoding="utf-8")
    (tmp_path / "opener.py").write_text(
        _opener_py("RULE ONE: base system.", {"type": "object"}), encoding="utf-8")
    ev = m.evaluate_working_tree(tmp_path, FIXTURE_PATHS)
    assert ev.digest is not None
    assert ev.reasons == []
    assert ev.rules == ["RULE ONE", "RULE TWO"]


def test_evaluate_working_tree_reports_a_missing_file_without_touching_git(tmp_path):
    # No git repo at tmp_path at all -- proves this function never shells out to git, unlike
    # evaluate_revision (which needs a real commit to read `git show` against).
    (tmp_path / "config.yaml").write_text(_config_yaml("x"), encoding="utf-8")
    ev = m.evaluate_working_tree(tmp_path, FIXTURE_PATHS)
    assert ev.digest is None
    assert any("opener.py does not exist" in r for r in ev.reasons)


def test_evaluate_working_tree_reports_missing_constants_the_same_way_as_evaluate_revision(tmp_path):
    (tmp_path / "config.yaml").write_text(_config_yaml("RULE TWO: base."), encoding="utf-8")
    (tmp_path / "opener.py").write_text(
        '_SYSTEM = "RULE ONE: base."\n_SCHEMA = {"type": "object"}\n', encoding="utf-8")
    ev = m.evaluate_working_tree(tmp_path, FIXTURE_PATHS)
    assert ev.digest is None
    assert any("missing constant(s)" in r and "_ITEM_PREAMBLE" in r for r in ev.reasons)


def test_evaluate_working_tree_agrees_with_evaluate_revision_for_identical_bytes(tmp_path):
    """The whole premise of _evaluate_sources() being shared: pointed at the SAME bytes (here,
    a clean checkout with nothing uncommitted), evaluate_working_tree() and evaluate_revision()
    must compute the IDENTICAL digest -- proving evaluate_working_tree is not a second,
    independently maintained copy of the algorithm that could silently drift."""
    _init_repo(tmp_path)
    sha = _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base."),
        "opener.py": _opener_py("RULE ONE: base.", {"type": "object"}),
    }, "initial", "2026-01-01T00:00:00")
    rev_eval = m.evaluate_revision(tmp_path, _ref(sha=sha), FIXTURE_PATHS)
    wt_eval = m.evaluate_working_tree(tmp_path, FIXTURE_PATHS)
    assert rev_eval.digest is not None
    assert wt_eval.digest == rev_eval.digest
    assert wt_eval.rules == rev_eval.rules


def test_evaluate_working_tree_matches_live_prompt_stamp_for_the_real_repo():
    """The exact case the task names as directly verifiable, exercised through the actual
    evaluate_working_tree() function (not by hand-calling extract_constants/extract_style/
    compute_prompt_stamp the way the older
    test_reconstruction_matches_live_prompt_stamp_for_the_current_working_tree does): applied to
    THIS repo's real files, right now, its digest must equal the real, imported
    operation_love.opener.opener.prompt_stamp()'s output exactly."""
    from operation_love import config as cfg_mod
    from operation_love.opener.opener import prompt_stamp

    ev = m.evaluate_working_tree(REPO_ROOT, m.REPO_PATHS)
    assert ev.reasons == [], ev.reasons
    assert ev.digest is not None

    cfg = cfg_mod.load(str(REPO_ROOT / "config.yaml"))
    live_digest = prompt_stamp(cfg.opener.style)
    assert ev.digest == live_digest


# ---------------------------------------------------------------------------------------
# build_registry -- grouping, labeling, rule diffing, determinism
# ---------------------------------------------------------------------------------------

def _make_evals(*rows):
    """rows: (commit, date, subject, digest_or_None, rules) -> list[RevisionEval]."""
    out = []
    for commit, date, subject, digest, rules in rows:
        reasons = [] if digest is not None else ["synthetic gap"]
        out.append(m.RevisionEval(commit, date, subject, digest, reasons, rules))
    return out


def test_build_registry_groups_identical_digests_into_one_era():
    evals = _make_evals(
        ("c1", "2026-01-01T00:00:00", "first", "digestA", ["RULE ONE"]),
        ("c2", "2026-01-02T00:00:00", "second, no prompt change", "digestA", ["RULE ONE"]),
    )
    doc = m.build_registry(evals)
    assert len(doc["eras"]) == 1
    era = doc["eras"][0]
    assert era["commit_count"] == 2
    assert era["commits"] == ["c1", "c2"]
    assert era["first_commit"] == "c1" and era["last_commit"] == "c2"
    assert era["first_date"] == "2026-01-01T00:00:00"
    assert era["last_date"] == "2026-01-02T00:00:00"


def test_build_registry_baseline_era_label_names_the_rule_count():
    evals = _make_evals(("c1", "2026-01-01T00:00:00", "first", "digestA", ["A", "B", "C"]))
    doc = m.build_registry(evals)
    assert "baseline: 3 rule(s) on wire" in doc["eras"][0]["label"]


def test_build_registry_labels_a_prose_only_change_when_rules_are_unchanged():
    evals = _make_evals(
        ("c1", "2026-01-01T00:00:00", "first", "digestA", ["RULE ONE"]),
        ("c2", "2026-01-02T00:00:00", "second", "digestB", ["RULE ONE"]),
    )
    doc = m.build_registry(evals)
    assert "prose only" in doc["eras"][1]["label"]


def test_build_registry_label_names_added_and_removed_rules():
    evals = _make_evals(
        ("c1", "2026-01-01T00:00:00", "first", "digestA", ["RULE ONE", "RULE TWO"]),
        ("c2", "2026-01-02T00:00:00", "second", "digestB", ["RULE ONE", "RULE THREE"]),
    )
    doc = m.build_registry(evals)
    label = doc["eras"][1]["label"]
    assert "+RULE THREE" in label
    assert "-RULE TWO" in label


def test_build_registry_reports_unevaluable_revisions_with_their_reasons():
    evals = _make_evals(
        ("c1", "2026-01-01T00:00:00", "unevaluable", None, []),
        ("c2", "2026-01-02T00:00:00", "evaluable", "digestA", ["RULE ONE"]),
    )
    doc = m.build_registry(evals)
    assert doc["generated_from"] == {
        "total_revisions": 2, "evaluated_revisions": 1, "unevaluable_revisions": 1,
    }
    assert doc["unevaluable_revisions"] == [
        {"commit": "c1", "date": "2026-01-01T00:00:00", "subject": "unevaluable",
         "reasons": ["synthetic gap"]},
    ]


def test_build_registry_eras_are_oldest_first():
    evals = _make_evals(
        ("c1", "2026-01-01T00:00:00", "first", "digestA", ["RULE ONE"]),
        ("c2", "2026-02-01T00:00:00", "second", "digestB", ["RULE ONE", "RULE TWO"]),
        ("c3", "2026-03-01T00:00:00", "third", "digestC", ["RULE TWO"]),
    )
    doc = m.build_registry(evals)
    assert [era["prompt_sha256"] for era in doc["eras"]] == ["digestA", "digestB", "digestC"]


def test_build_registry_is_deterministic_across_repeated_runs():
    evals = _make_evals(
        ("c1", "2026-01-01T00:00:00", "first", "digestA", ["RULE ONE"]),
        ("c2", "2026-01-02T00:00:00", "second", "digestB", ["RULE ONE", "RULE TWO"]),
    )
    doc1 = m.render_registry(m.build_registry(evals))
    doc2 = m.render_registry(m.build_registry(evals))
    assert doc1 == doc2


def test_build_registry_omits_working_tree_key_when_not_given():
    # The default (working_tree=None) -- every synthetic-repo test above relies on this to keep
    # exercising committed history only, so this pins it directly rather than by omission.
    evals = _make_evals(("c1", "2026-01-01T00:00:00", "first", "digestA", ["RULE ONE"]))
    doc = m.build_registry(evals)
    assert "working_tree" not in doc


# ---------------------------------------------------------------------------------------
# build_working_tree_entry / build_registry(working_tree=...) -- the provisional entry for the
# CURRENT uncommitted working tree. See tools/backfill_prompt_eras.py's WORKING TREE docstring
# section for why this is a separate top level key, never appended to `eras`.
# ---------------------------------------------------------------------------------------

def test_build_working_tree_entry_is_unmistakably_provisional_with_no_commit():
    doc = m.build_registry(
        _make_evals(("c1", "2026-01-01T00:00:00", "first", "digestA", ["RULE ONE"])),
        working_tree=m.WorkingTreeEval(digest="digestB", reasons=[], rules=["RULE ONE", "RULE TWO"]),
    )
    entry = doc["working_tree"]
    assert entry["provisional"] is True
    assert entry["commit"] is None
    assert "UNCOMMITTED" in entry["label"]
    assert "PROVISIONAL" in entry["label"]


def test_build_working_tree_entry_diffs_against_the_most_recent_known_committed_era():
    doc = m.build_registry(
        _make_evals(
            ("c1", "2026-01-01T00:00:00", "first", "digestA", ["RULE ONE"]),
            ("c2", "2026-01-02T00:00:00", "second", "digestB", ["RULE ONE", "RULE TWO"]),
        ),
        working_tree=m.WorkingTreeEval(digest="digestC", reasons=[],
                                       rules=["RULE ONE", "RULE THREE"]),
    )
    entry = doc["working_tree"]
    assert entry["matches_known_committed_era"] is None
    assert "+RULE THREE" in entry["label"]
    assert "-RULE TWO" in entry["label"]  # diffed against digestB (the LAST known era), not digestA


def test_build_working_tree_entry_flags_when_it_matches_a_known_committed_era():
    # A clean checkout: the working tree's digest is EXACTLY an already-known committed era's --
    # i.e. no uncommitted prompt changes at all.
    doc = m.build_registry(
        _make_evals(("c1", "2026-01-01T00:00:00", "first", "digestA", ["RULE ONE"])),
        working_tree=m.WorkingTreeEval(digest="digestA", reasons=[], rules=["RULE ONE"]),
    )
    entry = doc["working_tree"]
    assert entry["matches_known_committed_era"] == "digestA"
    assert "identical to already-known era" in entry["label"]


def test_build_working_tree_entry_reports_unevaluable_with_reasons_and_no_digest():
    doc = m.build_registry(
        _make_evals(("c1", "2026-01-01T00:00:00", "first", "digestA", ["RULE ONE"])),
        working_tree=m.WorkingTreeEval(digest=None, reasons=["opener.py does not parse"], rules=[]),
    )
    entry = doc["working_tree"]
    assert entry["prompt_sha256"] is None
    assert entry["reasons"] == ["opener.py does not parse"]
    assert "UNEVALUABLE" in entry["label"]


def test_build_working_tree_entry_never_claims_measured_metrics():
    # See this module's docstring's WORKING TREE section: this registry never records measured
    # metrics for ANY entry -- pinned here directly so a future edit can't quietly start implying
    # otherwise for the provisional entry specifically.
    doc = m.build_registry(
        _make_evals(("c1", "2026-01-01T00:00:00", "first", "digestA", ["RULE ONE"])),
        working_tree=m.WorkingTreeEval(digest="digestB", reasons=[], rules=["RULE ONE"]),
    )
    assert doc["working_tree"]["no_metrics_recorded_here"] is True


# ---------------------------------------------------------------------------------------
# End-to-end: a real synthetic git repo through discover_revisions -> evaluate_revision ->
# build_registry (backfill()), exercising the actual git plumbing rather than hand-built
# RevisionEval rows.
# ---------------------------------------------------------------------------------------

def test_backfill_end_to_end_against_a_synthetic_repo(tmp_path):
    _init_repo(tmp_path)

    # Commit 1: pre-redesign shape -- no item-crop constants at all. Unevaluable.
    c1 = _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base."),
        "opener.py": '_SYSTEM = "RULE ONE: base."\n_SCHEMA = {"type": "object"}\n',
    }, "pre redesign", "2026-01-01T00:00:00")

    # Commit 2: redesign lands -- all six constants now present. First evaluable era.
    c2 = _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base."),
        "opener.py": _opener_py("RULE ONE: base.", {"type": "object"}),
    }, "redesign lands", "2026-01-02T00:00:00")

    # Commit 3: unrelated whitespace-only opener.py comment change -- SAME digest as c2 (tests
    # multi-commit grouping into one era).
    c3 = _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base."),
        "opener.py": _opener_py("RULE ONE: base.", {"type": "object"}, extra="# a harmless comment"),
    }, "harmless comment", "2026-01-03T00:00:00")

    # Commit 4: a real rule addition -- new era.
    c4 = _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base.\n\nRULE TWO: new."),
        "opener.py": _opener_py("RULE ONE: base.", {"type": "object"}),
    }, "add rule two", "2026-01-04T00:00:00")

    # Commit 5: opener.py fails to parse at all -- unevaluable with a syntax-error reason.
    c5 = _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base.\n\nRULE TWO: new."),
        "opener.py": "def broken(:\n    pass\n",
    }, "broken syntax", "2026-01-05T00:00:00")

    # include_working_tree=False: this test is about the committed-history pipeline only (see
    # this test module's own docstring's "kept apart" rule) -- the working tree's own behavior
    # is covered separately below.
    doc, evaluations = m.backfill(tmp_path, FIXTURE_PATHS, include_working_tree=False)

    assert doc["generated_from"] == {
        "total_revisions": 5, "evaluated_revisions": 3, "unevaluable_revisions": 2,
    }
    assert [ev.commit for ev in evaluations] == [c1, c2, c3, c4, c5]  # oldest first

    unevaluable_commits = {u["commit"] for u in doc["unevaluable_revisions"]}
    assert unevaluable_commits == {c1, c5}

    eras = doc["eras"]
    assert len(eras) == 2
    assert eras[0]["commits"] == [c2, c3]
    assert eras[0]["commit_count"] == 2
    assert eras[1]["commits"] == [c4]
    assert "+RULE TWO" in eras[1]["label"]
    assert "working_tree" not in doc

    # Determinism against the SAME repo state.
    doc_again, _ = m.backfill(tmp_path, FIXTURE_PATHS, include_working_tree=False)
    assert m.render_registry(doc) == m.render_registry(doc_again)


def test_discover_revisions_raises_a_clear_error_outside_a_git_checkout(tmp_path):
    with pytest.raises(RuntimeError, match="git log failed"):
        m.discover_revisions(tmp_path)  # tmp_path is not a git repo at all


# ---------------------------------------------------------------------------------------
# backfill() with the (default-on) working tree included -- a real synthetic repo, exercising
# evaluate_working_tree() through the full pipeline rather than calling it directly.
# ---------------------------------------------------------------------------------------

def test_backfill_working_tree_defaults_to_included_and_matches_a_clean_checkout(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base."),
        "opener.py": _opener_py("RULE ONE: base.", {"type": "object"}),
    }, "initial", "2026-01-01T00:00:00")

    doc, _ = m.backfill(tmp_path, FIXTURE_PATHS)  # include_working_tree defaults to True

    assert "working_tree" in doc
    entry = doc["working_tree"]
    # A clean checkout: nothing uncommitted, so the working tree's digest is EXACTLY the one
    # known committed era's.
    assert entry["matches_known_committed_era"] == doc["eras"][0]["prompt_sha256"]


def test_backfill_working_tree_reflects_an_uncommitted_edit(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base."),
        "opener.py": _opener_py("RULE ONE: base.", {"type": "object"}),
    }, "initial", "2026-01-01T00:00:00")

    # Dirty the tree WITHOUT committing -- exactly the gap this feature closes.
    (tmp_path / "config.yaml").write_text(
        _config_yaml("RULE ONE: base.\n\nNO GRADING: never grade."), encoding="utf-8")

    doc, _ = m.backfill(tmp_path, FIXTURE_PATHS)

    entry = doc["working_tree"]
    assert entry["matches_known_committed_era"] is None
    assert "NO GRADING" in entry["rules"]
    assert "+NO GRADING" in entry["label"]
    # The committed `eras` list itself must be completely unaffected by the uncommitted edit.
    assert "NO GRADING" not in doc["eras"][0]["rules"]


def test_backfill_no_working_tree_omits_the_key(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base."),
        "opener.py": _opener_py("RULE ONE: base.", {"type": "object"}),
    }, "initial", "2026-01-01T00:00:00")
    doc, _ = m.backfill(tmp_path, FIXTURE_PATHS, include_working_tree=False)
    assert "working_tree" not in doc


# ---------------------------------------------------------------------------------------
# CLI: main() -- write, --check
# ---------------------------------------------------------------------------------------

def test_main_writes_the_registry_file(tmp_path, capsys):
    _init_repo(tmp_path)
    _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base."),
        "opener.py": _opener_py("RULE ONE: base.", {"type": "object"}),
    }, "initial", "2026-01-01T00:00:00")
    out_path = tmp_path / "eras.json"

    code = m.main(["--repo-root", str(tmp_path), "--out", str(out_path),
                  "--paths", "config.yaml", "opener.py"])
    assert code == 0
    doc = json.loads(out_path.read_text())
    assert doc["generated_from"]["evaluated_revisions"] == 1
    err = capsys.readouterr().err
    assert "Evaluated 1/1" in err


def test_main_check_mode_passes_when_up_to_date(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base."),
        "opener.py": _opener_py("RULE ONE: base.", {"type": "object"}),
    }, "initial", "2026-01-01T00:00:00")
    out_path = tmp_path / "eras.json"
    assert m.main(["--repo-root", str(tmp_path), "--out", str(out_path),
                  "--paths", "config.yaml", "opener.py"]) == 0
    assert m.main(["--repo-root", str(tmp_path), "--out", str(out_path),
                  "--paths", "config.yaml", "opener.py", "--check"]) == 0
    # --check must never write: content is untouched by the check-mode run.
    before = out_path.read_text()
    assert m.main(["--repo-root", str(tmp_path), "--out", str(out_path),
                  "--paths", "config.yaml", "opener.py", "--check"]) == 0
    assert out_path.read_text() == before


def test_main_check_mode_fails_loudly_on_a_stale_file(tmp_path, capsys):
    _init_repo(tmp_path)
    _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base."),
        "opener.py": _opener_py("RULE ONE: base.", {"type": "object"}),
    }, "initial", "2026-01-01T00:00:00")
    out_path = tmp_path / "eras.json"
    out_path.write_text("{}")

    code = m.main(["--repo-root", str(tmp_path), "--out", str(out_path),
                  "--paths", "config.yaml", "opener.py", "--check"])
    assert code == 1
    assert "is stale" in capsys.readouterr().err
    assert out_path.read_text() == "{}"  # --check never writes even on a mismatch


# ---------------------------------------------------------------------------------------
# CLI: --no-working-tree, and --check tolerating a dirty working tree (task requirement:
# "make --check tolerate it so a dirty tree does not spuriously fail an up-to-date check").
# ---------------------------------------------------------------------------------------

def test_main_writes_a_working_tree_entry_by_default(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base."),
        "opener.py": _opener_py("RULE ONE: base.", {"type": "object"}),
    }, "initial", "2026-01-01T00:00:00")
    out_path = tmp_path / "eras.json"
    assert m.main(["--repo-root", str(tmp_path), "--out", str(out_path),
                  "--paths", "config.yaml", "opener.py"]) == 0
    doc = json.loads(out_path.read_text())
    assert "working_tree" in doc


def test_main_no_working_tree_flag_omits_the_key(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base."),
        "opener.py": _opener_py("RULE ONE: base.", {"type": "object"}),
    }, "initial", "2026-01-01T00:00:00")
    out_path = tmp_path / "eras.json"
    assert m.main(["--repo-root", str(tmp_path), "--out", str(out_path),
                  "--paths", "config.yaml", "opener.py", "--no-working-tree"]) == 0
    doc = json.loads(out_path.read_text())
    assert "working_tree" not in doc


def test_main_check_mode_tolerates_a_dirty_working_tree(tmp_path, capsys):
    _init_repo(tmp_path)
    _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base."),
        "opener.py": _opener_py("RULE ONE: base.", {"type": "object"}),
    }, "initial", "2026-01-01T00:00:00")
    out_path = tmp_path / "eras.json"
    assert m.main(["--repo-root", str(tmp_path), "--out", str(out_path),
                  "--paths", "config.yaml", "opener.py"]) == 0
    before = out_path.read_text()

    # Dirty the tree WITHOUT committing -- the exact scenario --check must not treat as
    # staleness (only the provisional 'working_tree' entry can possibly differ here, since
    # nothing was committed).
    (tmp_path / "config.yaml").write_text(
        _config_yaml("RULE ONE: base.\n\nNO GRADING: never grade."), encoding="utf-8")

    code = m.main(["--repo-root", str(tmp_path), "--out", str(out_path),
                  "--paths", "config.yaml", "opener.py", "--check"])
    err = capsys.readouterr().err
    assert code == 0
    assert "up to date" in err
    assert out_path.read_text() == before  # --check never writes, dirty tree or not


def test_main_check_mode_still_fails_when_committed_history_actually_changed(tmp_path, capsys):
    _init_repo(tmp_path)
    _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base."),
        "opener.py": _opener_py("RULE ONE: base.", {"type": "object"}),
    }, "initial", "2026-01-01T00:00:00")
    out_path = tmp_path / "eras.json"
    assert m.main(["--repo-root", str(tmp_path), "--out", str(out_path),
                  "--paths", "config.yaml", "opener.py"]) == 0

    # A REAL new commit lands (not just an uncommitted edit) -- the registry is now genuinely
    # stale relative to git history, and --check must still catch that even though the working
    # tree feature now also drifts the 'working_tree' entry with the very same edit.
    _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base.\n\nNO GRADING: never grade."),
        "opener.py": _opener_py("RULE ONE: base.", {"type": "object"}),
    }, "add NO GRADING", "2026-01-02T00:00:00")

    code = m.main(["--repo-root", str(tmp_path), "--out", str(out_path),
                  "--paths", "config.yaml", "opener.py", "--check"])
    err = capsys.readouterr().err
    assert code == 1
    assert "is stale" in err


def test_main_check_mode_with_no_working_tree_behaves_like_before_the_feature(tmp_path, capsys):
    _init_repo(tmp_path)
    _commit(tmp_path, {
        "config.yaml": _config_yaml("RULE ONE: base."),
        "opener.py": _opener_py("RULE ONE: base.", {"type": "object"}),
    }, "initial", "2026-01-01T00:00:00")
    out_path = tmp_path / "eras.json"
    assert m.main(["--repo-root", str(tmp_path), "--out", str(out_path),
                  "--paths", "config.yaml", "opener.py", "--no-working-tree"]) == 0
    assert m.main(["--repo-root", str(tmp_path), "--out", str(out_path),
                  "--paths", "config.yaml", "opener.py", "--no-working-tree", "--check"]) == 0
    err = capsys.readouterr().err
    assert "up to date" in err
