"""Reconstruct every historical opener PROMPT ERA from git history and write the registry
tools/opener_corpus_report.py needs to turn an opaque prompt_sha256 back into something a human
can reason about.

WHY THIS EXISTS. opener/opener.py's prompt_stamp() stamps every generated opener with the
SHA-256 of the exact prompt bytes it was sent under (see that function's own docstring for the
seven components it covers). tools/opener_corpus_report.py already groups measured openers by
that digest, and --compare already diffs two of them -- but until this tool existed, nothing
mapped a digest to "this is the era where NO GRADING shipped" or answered "have we tried this
rule before, and what happened". A hash with no era attached to it is not a measurement tool,
it is a lookup key with nothing on the other end. This tool builds the other end: it walks every
commit that ever touched config.yaml or operation_love/opener/opener.py, recomputes
prompt_stamp()'s digest for each one using ONLY the historical bytes at that revision, groups
commits that produced the same digest into one era, and records each era's label, date range,
and the named ALL CAPS rules that were on the wire in it.

HOW A REVISION IS EVALUATED, deliberately WITHOUT importing the live module. opener.py has real
package-relative imports (``from ..costing import Usage``, ...) that only resolve inside the
installed package, so `exec`-ing an arbitrary historical revision's source text as a standalone
module would fail on every one of them for a reason that has nothing to do with whether that
revision's PROMPT actually changed. Every constant this tool needs (_SYSTEM, _SCHEMA,
_ITEM_PREAMBLE, _ITEM_PREAMBLE_CONTEXT, _ITEM_LABEL, _CONTEXT_LABEL) is a plain module level
literal assignment in every revision that has ever defined it, so this tool AST-parses the
historical source text and ast.literal_eval()s just those seven assignments' right hand sides
-- string/dict/list literals only, no execution, no import, no side effect. config.yaml's
opener.style is loaded the same way config.py's real loader ultimately reads it: plain
``yaml.safe_load()`` of the historical file text, no config.py import needed either (config.py
itself does nothing to opener.style beyond handing back exactly what yaml.safe_load parsed --
see operation_love/config.py's _section()).

BE HONEST about what cannot be evaluated. Before the 2026-08-12 "Opener redesign: model picks
the item" commit, opener.py had no per-item crop request shape at all, so _ITEM_PREAMBLE,
_ITEM_PREAMBLE_CONTEXT, _ITEM_LABEL and _CONTEXT_LABEL simply do not exist in any earlier
revision -- there is no digest to compute because three of the digest's seven components have
no historical value to put there. Rather than silently skipping those commits (which would make
"22 known eras" look like the whole history), every such revision is recorded, by commit, in
this tool's own UNEVALUABLE bucket with the concrete reason (a named missing constant, a parse
failure, a missing file, or a config.yaml with no opener.style at that revision) -- see
evaluate_revision() and RevisionEval.reasons. main() always prints how many revisions were
evaluated and how many were not; the registry file also carries both counts and the full
unevaluable list.

DETERMINISM (of the `eras` list). Regenerating the `eras` list from the same git history must
produce the same bytes: every input this half of the tool reads (git log, git show, the
historical file text) is a pure function of the commit graph, and every transform (AST
literal_eval, yaml.safe_load, sha256) is a pure function of its input. "History" here means
commits: an uncommitted edit is not history, is not the same from one run to the next, and
would make two runs on the same commit disagree if it were folded into `eras` itself -- so it
never is. See tests/test_backfill_prompt_eras.py for the mutation-tested proof this stays true.

WORKING TREE (provisional, NOT part of the deterministic `eras` list above). The `eras` list
answers "what has SHIPPED"; it was, until now, blind to whatever a developer has sitting
uncommitted on disk -- so a rule added only in the working tree (not yet committed) looked
IDENTICAL to a rule that was never written at all, which is exactly backwards for a tool whose
entire purpose is answering "have we tried this before". evaluate_working_tree() closes that gap:
it runs the IDENTICAL extraction/digest pipeline as evaluate_revision() (same extract_constants,
extract_style, compute_prompt_stamp, extract_rule_names -- see _evaluate_sources(), the shared
core both now call), the only difference being that it reads bytes straight off disk
(Path.read_text) instead of through `git show <rev>:<path>`. The result is written to the
registry as a SEPARATE top level `working_tree` key -- never appended to `eras` -- and is
unmistakably marked `"provisional": true` with `"commit": null` and a label that leads with
"UNCOMMITTED WORKING TREE" so no reader of the JSON, --eras, or --rule output can mistake an
unshipped rule for a shipped one. It is included by default (see backfill()'s
`include_working_tree` and main()'s `--no-working-tree`) precisely because the tool that exists
to answer "have we tried this" should answer it correctly out of the box, even when the honest
answer is "not yet, but it is sitting in the working tree right now" -- and because leaving it
opt-in would mean the common case (an operator adding a new rule and immediately asking `--rule`
about it) silently gets the WRONG answer by default. `--check` (see main()) tolerates the
`working_tree` entry changing between runs -- a dirty tree is expected to make it drift on every
edit, and that must never be confused with the `eras` list itself going stale. This registry
never carries measured metrics for any entry, committed or provisional -- see
tools/opener_corpus_report.py's --rule/--eras for whatever has actually been measured against a
real corpus; a rule appearing here (in a committed era OR in `working_tree`) is a fact about what
was ON THE WIRE, never a claim that it was measured. tests/test_backfill_prompt_eras.py verifies
that evaluate_working_tree(), applied to the files actually on disk right now, reproduces the
live prompt_stamp()'s output exactly -- see that test module's own docstring.

LABELS are generated, not hand written, so a re-run never has to remember to relabel a shifted
era. Each era's label names its first commit's date and short SHA (which is unique by
construction) plus a short digest of what changed in the named-rule set versus the era right
before it ("+RULE_ADDED; -RULE_REMOVED; +N more change(s)"), or "baseline: N rule(s) on wire" for
the very first known era, or "prose only (no rule added or removed)" when the prompt text
changed but no named rule was added or removed. The label is a mnemonic, never the source of
truth: the full added/removed rule sets are recomputed straight from each era's own stored
`rules` list every time (see tools/opener_corpus_report.py's --eras), never parsed back out of
the label string.

OUTPUT: ops/prompt-eras.json, one entry per DISTINCT digest, oldest era first. Run this file
directly (`python tools/backfill_prompt_eras.py`) to regenerate it, or with `--check` to verify
the checked-in file is still current without writing anything (useful in CI once this tool's
output is expected to track ongoing prompt edits).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

# The two paths whose commit history defines a "prompt era". Neither has ever moved (verified
# with `git log --diff-filter=A --follow` against both -- each was created at this exact path in
# the repo's very first commit and never renamed), so a plain pathspec suffices; no --follow
# needed.
REPO_PATHS: tuple[str, str] = ("config.yaml", "operation_love/opener/opener.py")

# The seven-component digest's constant names, exactly as named in opener.py's prompt_stamp()
# docstring and payload tuple -- kept as a tuple (not re-derived from the live module) because
# deriving it FROM the module we are explicitly not importing would defeat the point.
CONST_NAMES: tuple[str, ...] = (
    "_SYSTEM", "_SCHEMA", "_ITEM_PREAMBLE", "_ITEM_PREAMBLE_CONTEXT", "_ITEM_LABEL",
    "_CONTEXT_LABEL",
)

DEFAULT_OUT = Path("ops/prompt-eras.json")

_FIELD_SEP = "\x1f"  # ASCII unit separator; never appears in a commit subject line in practice.


# =========================================================================================
# git plumbing
# =========================================================================================

@dataclass(frozen=True)
class CommitRef:
    sha: str
    date: str       # ISO 8601, from `git log --date=iso-strict`
    subject: str


def discover_revisions(repo_root: Path, paths: Sequence[str] = REPO_PATHS) -> list[CommitRef]:
    """Every commit that touched any of `paths`, OLDEST FIRST (git log itself prints newest
    first; this reverses it, since every downstream consumer -- era grouping, label diffing --
    wants to walk history forward)."""
    result = subprocess.run(
        ["git", "log", f"--format=%H{_FIELD_SEP}%ad{_FIELD_SEP}%s", "--date=iso-strict",
         "--", *paths],
        cwd=repo_root, capture_output=True, text=True,
        check=False)  # a nonzero exit is reported below, not raised
    if result.returncode != 0:
        raise RuntimeError(
            f"git log failed (is {repo_root} a git checkout?): {result.stderr.strip()}")
    refs: list[CommitRef] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        sha, date, subject = line.split(_FIELD_SEP, 2)
        refs.append(CommitRef(sha=sha, date=date, subject=subject))
    refs.reverse()
    return refs


def git_show(repo_root: Path, rev: str, path: str) -> str | None:
    """The text of `path` as it existed at `rev`, or None when it did not exist there (a
    nonzero exit from `git show` -- e.g. the path was added later -- is reported this way rather
    than raising, since a missing historical file is exactly the kind of thing this tool's BE
    HONEST reasons are for, not a crash)."""
    result = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=repo_root,
                            capture_output=True,
                            check=False)  # a missing historical path returns None below
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", errors="replace")


# =========================================================================================
# Historical constant / style extraction -- AST literal_eval only, no exec, no import.
# =========================================================================================

def extract_constants(source: str) -> tuple[dict[str, Any], dict[str, str]]:
    """AST-parse `source` (a historical opener.py revision's full text) and literal_eval() the
    right hand side of the FIRST top level assignment to each name in CONST_NAMES.

    Returns (values, failures): `values` maps every successfully evaluated name to its Python
    value (a str for the five string constants, a dict for _SCHEMA); `failures` maps a name that
    WAS assigned at this revision but whose value could not be literal_eval()'d (e.g. it
    references a variable, calls a function, or otherwise is not a plain literal) to the reason.
    A name simply absent from this revision appears in neither dict -- the caller (
    evaluate_revision) is responsible for treating "not in values and not in failures" as
    "missing constant at this revision", since that is a legitimate, expected state for every
    commit before the item-crop shape existed at all, not a parse failure.

    THE LAST top-level assignment to a name wins, matching ordinary Python module execution (a
    second `_SYSTEM = ...` at module scope really does overwrite the first when the module is
    actually imported) -- not the first one found. No revision in this repo's history has ever
    reassigned one of these names twice, so this has never changed a real digest, but getting it
    backwards would silently compute the WRONG revision's prompt for the rare file that does.

    Deliberately only walks `tree.body` (true module level statements) rather than every Assign
    anywhere in the file, so a same-named local variable inside some unrelated function can never
    be mistaken for the module constant.
    """
    values: dict[str, Any] = {}
    failures: dict[str, str] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        # A syntax error blocks every constant equally; record it against each name CONST_NAMES
        # cares about so evaluate_revision's per-name reasons stay uniform rather than needing a
        # separate "whole file unparseable" code path.
        for name in CONST_NAMES:
            failures[name] = f"opener.py at this revision does not parse: {type(exc).__name__}: {exc}"
        return values, failures
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name not in CONST_NAMES:
            continue
        try:
            values[name] = ast.literal_eval(node.value)
            failures.pop(name, None)  # a later successful assignment clears an earlier failure
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
            failures[name] = f"{name} is not a plain literal at this revision: {type(exc).__name__}: {exc}"
            values.pop(name, None)  # a later failing assignment shadows an earlier good value
    return values, failures


def extract_style(source: str) -> tuple[str | None, str | None]:
    """The historical config.yaml's opener.style, loaded exactly the way config.py's real
    loader ultimately hands it to prompt_stamp(): plain yaml.safe_load(), then
    doc["opener"]["style"] with no further transform (see operation_love/config.py's
    `_section()` -- OpenerCfg.style is just whatever the YAML mapping held).

    Returns (style, reason): `style` is the string on success; on any failure `style` is None
    and `reason` names exactly what was wrong (parse error, non-mapping document, missing
    `opener` section, missing `style` key, or a `style` that parsed to something other than a
    string), never a bare exception with no context.
    """
    try:
        doc = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        return None, f"config.yaml does not parse as YAML at this revision: {type(exc).__name__}: {exc}"
    if not isinstance(doc, dict):
        return None, f"config.yaml did not parse to a mapping at this revision (got {type(doc).__name__})"
    opener_section = doc.get("opener")
    if opener_section is None:
        return None, "config.yaml has no top level 'opener:' section at this revision"
    if not isinstance(opener_section, dict):
        return None, (f"config.yaml's 'opener' section is not a mapping at this revision "
                      f"(got {type(opener_section).__name__})")
    style = opener_section.get("style")
    if style is None:
        return None, "config.yaml's 'opener' section has no 'style' key at this revision"
    if not isinstance(style, str):
        return None, f"config.yaml's opener.style is not a string at this revision (got {type(style).__name__})"
    return style, None


def compute_prompt_stamp(style: str, system: str, schema: dict, item_preamble: str,
                          item_preamble_context: str, item_label: str, context_label: str) -> str:
    """Byte-for-byte the SAME algorithm as operation_love.opener.opener.prompt_stamp(): a NUL
    join of the seven components in the SAME order, canonicalized the SAME way, sha256 hex
    digested. Deliberately re-implemented here rather than imported (see this module's docstring
    for why nothing here imports the live opener module) -- kept honest by
    tests/test_backfill_prompt_eras.py, which asserts this function and the real prompt_stamp()
    agree on the CURRENT working tree, the one case where both can be computed and compared
    directly.
    """
    payload = "\x00".join((
        str(style), system,
        json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        item_preamble, item_preamble_context, item_label, context_label,
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# =========================================================================================
# Named ALL CAPS rule extraction (from _SYSTEM and opener.style text ONLY -- never _SCHEMA;
# see the task this tool was built for: "extract them from the historical _SYSTEM and style
# text").
# =========================================================================================

# Matches a run of one-or-more ALL CAPS words (each an uppercase-letter run, optionally holding
# an apostrophe or a comma, as in "HEDGE THE CLAIM, NEVER YOURSELF" and "GUESS THE WORLD, NOT
# HER IDENTITY"), joined by single spaces, immediately followed by a colon -- the exact shape
# every named rule in _SYSTEM and opener.style uses to introduce itself. Requires at least two
# words (a lone capitalized abbreviation followed by a colon is not a rule name) -- enforced
# below by rejecting any match with no interior space, rather than in the pattern itself, so the
# pattern stays a single readable alternation.
_RULE_NAME_RE = re.compile(r"\b[A-Z]{2,}(?:[A-Z'’,]|\s(?=[A-Z]))*[A-Z]:")


def extract_rule_names(text: str) -> list[str]:
    """Every distinct named ALL CAPS rule introduced in `text`, in FIRST APPEARANCE order (not
    alphabetical -- reading order is what makes a diff against the previous era legible, and
    determinism only requires a fixed order, not a sorted one). A rule name that appears more
    than once (e.g. "HARD RULE:" is reused for two separate hard rules in the live prompt) is
    recorded once, at its first occurrence.
    """
    names: list[str] = []
    seen: set[str] = set()
    for match in _RULE_NAME_RE.finditer(text):
        name = match.group(0)[:-1]  # drop the trailing ':'
        if " " not in name:
            continue  # single all-caps word: not a named rule (e.g. a stray acronym)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


# =========================================================================================
# Per-revision evaluation
# =========================================================================================

@dataclass
class RevisionEval:
    commit: str
    date: str
    subject: str
    digest: str | None          # None means unevaluable -- see `reasons`
    reasons: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)   # empty when digest is None


@dataclass
class WorkingTreeEval:
    """The result of evaluating the CURRENT uncommitted working tree (see
    evaluate_working_tree()). Deliberately NOT a RevisionEval: there is no commit, no date, no
    subject behind these bytes -- a distinct shape makes "this has no commit" a type level fact
    rather than a None smuggled into a field that means something else for every other caller.
    """
    digest: str | None          # None means unevaluable -- see `reasons`
    reasons: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)   # empty when digest is None


def _evaluate_sources(opener_path: str, config_path: str, opener_src: str | None,
                       config_src: str | None, *,
                       context: str = "this revision") -> tuple[str | None, list[str], list[str]]:
    """The reasons/digest/rules core shared by evaluate_revision() (bytes from `git show`) and
    evaluate_working_tree() (bytes read straight off disk) -- factored out so the two can never
    silently drift into computing the digest two different ways. `context` names what "does not
    exist at ..." / "missing constant(s) at ..." refers to in the returned reason strings (a
    historical revision for the former, the working tree for the latter) -- purely cosmetic, it
    changes no computation.

    Returns (digest_or_None, reasons, rules); `rules` is always [] when digest is None, matching
    RevisionEval/WorkingTreeEval's own "empty when digest is None" contract.
    """
    reasons: list[str] = []
    if opener_src is None:
        reasons.append(f"{opener_path} does not exist at {context}")
    if config_src is None:
        reasons.append(f"{config_path} does not exist at {context}")

    values: dict[str, Any] = {}
    if opener_src is not None:
        values, failures = extract_constants(opener_src)
        reasons.extend(f"{name}: {reason}" for name, reason in sorted(failures.items()))
        missing = [name for name in CONST_NAMES if name not in values and name not in failures]
        if missing:
            reasons.append(f"missing constant(s) at {context}: {missing}")

    style: str | None = None
    if config_src is not None:
        style, style_reason = extract_style(config_src)
        if style_reason:
            reasons.append(style_reason)

    if reasons:
        return None, reasons, []

    digest = compute_prompt_stamp(
        style, values["_SYSTEM"], values["_SCHEMA"], values["_ITEM_PREAMBLE"],
        values["_ITEM_PREAMBLE_CONTEXT"], values["_ITEM_LABEL"], values["_CONTEXT_LABEL"])
    rules = extract_rule_names(values["_SYSTEM"] + "\n" + style)
    return digest, [], rules


def evaluate_revision(repo_root: Path, ref: CommitRef,
                       paths: tuple[str, str] = REPO_PATHS) -> RevisionEval:
    """Evaluate one commit: read both historical files, extract every needed constant, and
    either compute its digest (recording the named rules on the wire) or record every concrete
    reason it could not be computed. Never raises for an ordinary historical gap (missing file,
    missing constant, unparseable YAML/py) -- those are exactly what `reasons` is for; only a
    git plumbing failure (see git_show/discover_revisions) is allowed to propagate as an
    exception, since that is an environment problem, not a fact about the prompt's history.
    """
    config_path, opener_path = paths
    opener_src = git_show(repo_root, ref.sha, opener_path)
    config_src = git_show(repo_root, ref.sha, config_path)
    digest, reasons, rules = _evaluate_sources(opener_path, config_path, opener_src, config_src)
    return RevisionEval(ref.sha, ref.date, ref.subject, digest, reasons, rules)


def evaluate_working_tree(repo_root: Path,
                          paths: tuple[str, str] = REPO_PATHS) -> WorkingTreeEval:
    """Evaluate the CURRENT files on disk at `repo_root` -- the uncommitted working tree -- using
    the exact same extraction/digest pipeline as evaluate_revision() (see _evaluate_sources()),
    the only difference being that bytes come from Path.read_text() rather than `git show
    <rev>:<path>`. This is the one place in this tool that looks at anything other than committed
    git history -- see this module's docstring's WORKING TREE section for why that is deliberate
    and is never folded into the deterministic `eras` list build_registry() assembles.

    A path that does not exist on disk at all is reported exactly like a historical revision that
    never had the file (see _evaluate_sources) rather than raising -- a fresh checkout missing
    one of the two tracked paths is a legitimate, if unusual, state to report on honestly.
    """
    config_path, opener_path = paths
    opener_file = repo_root / opener_path
    config_file = repo_root / config_path
    opener_src = opener_file.read_text(encoding="utf-8") if opener_file.is_file() else None
    config_src = config_file.read_text(encoding="utf-8") if config_file.is_file() else None
    digest, reasons, rules = _evaluate_sources(
        opener_path, config_path, opener_src, config_src, context="the current working tree")
    return WorkingTreeEval(digest=digest, reasons=reasons, rules=rules)


# =========================================================================================
# Era grouping, labeling, and registry assembly
# =========================================================================================

def _rule_diff_descriptor(rules: list[str], previous_rules: list[str] | None) -> str:
    """The "what changed in the named rule set" half of a label -- e.g. "+RULE_ADDED;
    -RULE_REMOVED; +N more change(s)", "baseline: N rule(s) on wire", or "prose only (no rule
    added or removed)". Factored out of _label_for_era() so build_working_tree_entry() can reuse
    the IDENTICAL diffing logic for the provisional working-tree entry's label -- a second,
    separately maintained copy could silently drift from what committed eras use and describe the
    same kind of change two different ways."""
    if previous_rules is None:
        return f"baseline: {len(rules)} rule(s) on wire"
    prev_set = set(previous_rules)
    cur_set = set(rules)
    added = [r for r in rules if r not in prev_set]
    removed = [r for r in previous_rules if r not in cur_set]
    changes = [f"+{r}" for r in added] + [f"-{r}" for r in removed]
    if not changes:
        return "prose only (no rule added or removed)"
    shown = changes[:3]
    more = len(changes) - len(shown)
    descriptor = "; ".join(shown)
    if more:
        descriptor += f"; +{more} more change(s)"
    return descriptor


def _label_for_era(rules: list[str], previous_rules: list[str] | None,
                    first_date: str, first_commit: str) -> str:
    """A short, deterministic, unique-by-construction label: the era's first commit's date and
    short SHA (unique -- two eras never share a first commit) followed by a short digest of what
    changed in the named-rule set versus the era immediately before it in history. See this
    module's docstring's LABELS section for the full rationale."""
    date_short = first_date[:10]
    sha_short = first_commit[:8]
    return f"{date_short} {sha_short}: {_rule_diff_descriptor(rules, previous_rules)}"


# The label prefix every provisional working-tree entry starts with -- deliberately loud and
# repeated in both words ("UNCOMMITTED" and "PROVISIONAL") so truncated or out-of-context display
# (a log line, a narrow terminal) still can't be mistaken for a shipped era's date-and-sha label.
_WORKING_TREE_LABEL_PREFIX = "UNCOMMITTED WORKING TREE -- PROVISIONAL, not a shipped era"

_WORKING_TREE_NOTE = (
    "Computed from the files on disk right now (Path.read_text), not from git history -- there "
    "is no commit behind this digest, and it can change on the very next edit or `git commit`. "
    "Regenerate this registry (python tools/backfill_prompt_eras.py) to refresh it. This "
    "registry never records measured metrics for ANY entry, committed or provisional -- see "
    "tools/opener_corpus_report.py's --rule/--eras for whatever has actually been measured "
    "against a real corpus; a rule appearing here is a fact about what was ON THE WIRE, never a "
    "claim that it was measured."
)


def build_working_tree_entry(evaluation: WorkingTreeEval, era_order: list[str],
                              eras: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the JSON-serializable `working_tree` registry entry from a WorkingTreeEval. This is
    a SEPARATE top level key from `eras` (see build_registry()), never appended to that list:
    mixing an uncommitted, possibly-transient digest into the same list as immutable committed
    eras would make `eras[-1]` an unreliable "the current shipped prompt" answer the moment
    anyone's working tree is dirty. Every field here is either `None`/empty or unmistakably
    prefixed/labeled so a reader can never mistake this for a committed era: `"provisional"` is
    always `True`, `"commit"` is always `None`, and `"label"` always starts with
    _WORKING_TREE_LABEL_PREFIX.

    `era_order`/`eras` are the same structures build_registry() assembles for the committed
    `eras` list -- used here only to (a) diff the working tree's rule set against the most recent
    committed era's (exactly the same +/- diffing _label_for_era uses, via
    _rule_diff_descriptor(), so the two label styles read consistently) and (b) detect the
    (perfectly legal) case where the working tree's digest is IDENTICAL to an already-known
    committed era's -- i.e. a clean checkout with no uncommitted prompt edits at all.
    """
    if evaluation.digest is None:
        return {
            "provisional": True,
            "prompt_sha256": None,
            "label": f"{_WORKING_TREE_LABEL_PREFIX} (UNEVALUABLE, see reasons)",
            "commit": None,
            "rules": [],
            "reasons": list(evaluation.reasons),
            "matches_known_committed_era": None,
            "no_metrics_recorded_here": True,
            "note": _WORKING_TREE_NOTE,
        }
    matches = evaluation.digest if evaluation.digest in eras else None
    if matches is not None:
        label = f"{_WORKING_TREE_LABEL_PREFIX} (identical to already-known era: {eras[matches]['label']!r})"
    else:
        previous_rules = eras[era_order[-1]]["rules"] if era_order else None
        label = f"{_WORKING_TREE_LABEL_PREFIX} ({_rule_diff_descriptor(evaluation.rules, previous_rules)})"
    return {
        "provisional": True,
        "prompt_sha256": evaluation.digest,
        "label": label,
        "commit": None,
        "rules": list(evaluation.rules),
        "reasons": [],
        "matches_known_committed_era": matches,
        "no_metrics_recorded_here": True,
        "note": _WORKING_TREE_NOTE,
    }


def build_registry(evaluations: Sequence[RevisionEval],
                   paths: tuple[str, str] = REPO_PATHS,
                   working_tree: WorkingTreeEval | None = None) -> dict[str, Any]:
    """Assemble the full JSON-serializable registry document from every revision's evaluation
    (oldest first, as discover_revisions/evaluate_revision produce them). Pure and deterministic
    with respect to `evaluations`: the same sequence always produces a byte-identical `eras` /
    `unevaluable_revisions` (given identical json.dumps settings at the caller), which is what
    lets --check compare a fresh run's committed-history portion against the checked-in file.

    `working_tree`, when given, is attached as a SEPARATE top level `working_tree` key (see
    build_working_tree_entry()) -- never merged into `eras`. Passing None (the default) omits the
    key entirely, which is what every caller that only cares about committed history (including
    every synthetic-repo test in tests/test_backfill_prompt_eras.py) still gets.
    """
    evaluated = [e for e in evaluations if e.digest is not None]
    unevaluated = [e for e in evaluations if e.digest is None]

    era_order: list[str] = []
    eras: dict[str, dict[str, Any]] = {}
    for ev in evaluated:  # oldest first
        entry = eras.get(ev.digest)
        if entry is None:
            entry = {
                "prompt_sha256": ev.digest,
                "label": "",  # filled in below, once every era's rule list is known
                "first_commit": ev.commit, "first_date": ev.date, "first_subject": ev.subject,
                "last_commit": ev.commit, "last_date": ev.date, "last_subject": ev.subject,
                "commit_count": 0,
                "commits": [],
                "rules": ev.rules,
            }
            eras[ev.digest] = entry
            era_order.append(ev.digest)
        entry["last_commit"], entry["last_date"], entry["last_subject"] = (
            ev.commit, ev.date, ev.subject)
        entry["commits"].append(ev.commit)
        entry["commit_count"] += 1

    previous_rules: list[str] | None = None
    for digest in era_order:
        entry = eras[digest]
        entry["label"] = _label_for_era(entry["rules"], previous_rules,
                                        entry["first_date"], entry["first_commit"])
        previous_rules = entry["rules"]

    doc: dict[str, Any] = {
        "schema_version": 1,
        "digest_components": [
            "opener.style (config.yaml, the owner's user-turn style guide)",
            "_SYSTEM (operation_love/opener/opener.py systemInstruction)",
            "canonical json.dumps(_SCHEMA, sort_keys=True, separators=(',',':'), "
            "ensure_ascii=True) (operation_love/opener/opener.py responseJsonSchema)",
            "_ITEM_PREAMBLE (operation_love/opener/opener.py)",
            "_ITEM_PREAMBLE_CONTEXT (operation_love/opener/opener.py)",
            "_ITEM_LABEL (operation_love/opener/opener.py)",
            "_CONTEXT_LABEL (operation_love/opener/opener.py)",
        ],
        "digest_algorithm": "sha256(NUL.join(digest_components), utf-8); must match "
                            "operation_love.opener.opener.prompt_stamp() exactly -- see "
                            "tests/test_backfill_prompt_eras.py",
        "source_paths": list(paths),
        "generated_from": {
            "total_revisions": len(evaluations),
            "evaluated_revisions": len(evaluated),
            "unevaluable_revisions": len(unevaluated),
        },
        "eras": [eras[digest] for digest in era_order],
        "unevaluable_revisions": [
            {"commit": ev.commit, "date": ev.date, "subject": ev.subject, "reasons": ev.reasons}
            for ev in unevaluated
        ],
    }
    if working_tree is not None:
        # A SEPARATE top level key, deliberately never merged into "eras" -- see
        # build_working_tree_entry()'s docstring and this module's docstring's WORKING TREE
        # section for why mixing an uncommitted digest into the committed list would be wrong.
        doc["working_tree"] = build_working_tree_entry(working_tree, era_order, eras)
    return doc


def render_registry(doc: dict[str, Any]) -> str:
    """The registry's canonical on-disk text: indent=2, insertion order preserved (already
    deterministic -- see build_registry), one trailing newline."""
    return json.dumps(doc, indent=2, sort_keys=False) + "\n"


def backfill(repo_root: Path, paths: tuple[str, str] = REPO_PATHS, *,
            include_working_tree: bool = True
            ) -> tuple[dict[str, Any], list[RevisionEval]]:
    """Run the full pipeline against a real (or test-fixture) git checkout at `repo_root`:
    discover every touching commit, evaluate each one, and build the registry document. Returns
    (doc, evaluations) so a caller (main(), or a test) can report on the evaluations directly
    without re-deriving them from the assembled doc.

    `include_working_tree` (default True -- see main()'s `--no-working-tree` for the opt-out)
    additionally evaluates the CURRENT files on disk (evaluate_working_tree()) and attaches them
    to the returned doc as its provisional `working_tree` entry -- see this module's docstring's
    WORKING TREE section for why this is on by default.
    """
    revisions = discover_revisions(repo_root, paths)
    evaluations = [evaluate_revision(repo_root, ref, paths) for ref in revisions]
    working_tree = evaluate_working_tree(repo_root, paths) if include_working_tree else None
    doc = build_registry(evaluations, paths, working_tree=working_tree)
    return doc, evaluations


# =========================================================================================
# CLI
# =========================================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/backfill_prompt_eras.py",
        description="Reconstruct every historical opener prompt era from git history and write "
                    "the ops/prompt-eras.json registry tools/opener_corpus_report.py reads.")
    parser.add_argument("--repo-root", default=".",
                        help="git checkout to read history from (default: current directory)")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"registry file to write (default {DEFAULT_OUT})")
    parser.add_argument("--paths", nargs=2, metavar=("CONFIG_YAML", "OPENER_PY"),
                        default=list(REPO_PATHS),
                        help="override the two tracked paths (default: this repo's real ones; "
                             "tests use this to point at a fixture repo's own file names)")
    parser.add_argument("--check", action="store_true",
                        help="do not write --out; exit 1 if regenerating it would change its "
                             "committed-history contents (prints a diff-free staleness verdict "
                             "only; a dirty working tree changing only the provisional "
                             "'working_tree' entry is NOT staleness -- see --no-working-tree)")
    parser.add_argument("--no-working-tree", action="store_true",
                        help="omit the provisional 'working_tree' entry (the CURRENT files on "
                             "disk, evaluated the same way as history but with no commit behind "
                             "them) from the output. By default it IS included, unmistakably "
                             "marked provisional, so this registry answers 'have we tried this' "
                             "correctly even for a rule that only exists in the working tree so "
                             "far -- see this module's docstring's WORKING TREE section")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    repo_root = Path(args.repo_root)
    paths = (args.paths[0], args.paths[1])
    include_working_tree = not args.no_working_tree

    try:
        doc, evaluations = backfill(repo_root, paths, include_working_tree=include_working_tree)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    total = doc["generated_from"]["total_revisions"]
    evaluated_n = doc["generated_from"]["evaluated_revisions"]
    unevaluable_n = doc["generated_from"]["unevaluable_revisions"]
    print(f"Evaluated {evaluated_n}/{total} revision(s) touching {' and '.join(paths)}; "
         f"{unevaluable_n} unevaluable (see the registry's 'unevaluable_revisions').",
         file=sys.stderr)
    print(f"Found {len(doc['eras'])} distinct prompt era(s) among the evaluated revisions.",
         file=sys.stderr)

    working_tree = doc.get("working_tree")
    if working_tree is None:
        print("Working tree: not evaluated (--no-working-tree).", file=sys.stderr)
    elif working_tree["prompt_sha256"] is None:
        print("Working tree: UNEVALUABLE -- see doc['working_tree']['reasons']: "
             f"{working_tree['reasons']}", file=sys.stderr)
    elif working_tree["matches_known_committed_era"] is not None:
        print(f"Working tree: matches already-known committed era "
             f"{working_tree['matches_known_committed_era'][:12]}... (no uncommitted prompt "
             "changes).", file=sys.stderr)
    else:
        print(f"Working tree: NEW provisional era, digest "
             f"{working_tree['prompt_sha256'][:12]}... (uncommitted -- see doc['working_tree']).",
             file=sys.stderr)

    rendered = render_registry(doc)
    out_path = Path(args.out)
    if args.check:
        existing = out_path.read_text(encoding="utf-8") if out_path.exists() else None
        if existing == rendered:
            print(f"{out_path} is up to date.", file=sys.stderr)
            return 0
        if existing is None:
            print(f"ERROR: {out_path} does not exist; run without --check to create it.",
                 file=sys.stderr)
            return 1
        # An exact match failed, but the provisional 'working_tree' entry is EXPECTED to drift on
        # every uncommitted edit -- that must never count as the registry being stale. Retry the
        # comparison with both documents' 'working_tree' entries removed: only a mismatch in the
        # committed-history portion (schema_version, digest_components, eras,
        # unevaluable_revisions, ...) is real staleness.
        try:
            existing_doc = json.loads(existing)
        except json.JSONDecodeError as exc:
            print(f"ERROR: {out_path} is stale relative to git history (and does not even parse "
                 f"as JSON: {exc}); re-run without --check to regenerate it.", file=sys.stderr)
            return 1
        existing_hist = {k: v for k, v in existing_doc.items() if k != "working_tree"}
        fresh_hist = {k: v for k, v in doc.items() if k != "working_tree"}
        if existing_hist == fresh_hist:
            print(f"{out_path} is up to date (committed-history portion matches; only the "
                 "provisional 'working_tree' entry differs, which is expected in a dirty "
                 "checkout and is NOT treated as staleness).", file=sys.stderr)
            return 0
        print(f"ERROR: {out_path} is stale relative to git history; re-run without --check "
             "to regenerate it.", file=sys.stderr)
        return 1

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
