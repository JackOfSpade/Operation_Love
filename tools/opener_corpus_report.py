"""Offline measurement of what the opener prompt actually PRODUCES, grouped by prompt era.

WHY THIS EXISTS. ops/OPENER-REDESIGN.md records eight prompt-only fixes to date. Of those
eight: ZERO were verified working, three failed live or had to be escalated to a deterministic
guard (the location-followup hard stop, the "HERE'S" scaffolding collision, and the entropy
guard's earlier live failure), and five shipped with no outcome measurement at all. Every one
of those five addenda ends with some spelling of "UNMEASURED until a live batch runs" or
"compliance is unmeasured; cheapest prediction to check: ...". Prompt rewrite N+1 has therefore
always been argued from memory and a handful of manually re-read Training drafts, never from a
repeatable before/after comparison. This tool is the missing measurement: run it once before a
prompt edit and once after, then `--compare` the two `prompt_sha256` eras it reports.

READ FIRST (this tool's own prior art): tools/gemini_model_probe.py for this repo's CLI/argparse
conventions; operation_love/ranker/store.py and bigquery_store.py for the `openers` /
`opener_rejections` table schemas this tool reads; ops/OPENER-REDESIGN.md 3.6/3.7 and the
2026-09-05/2026-09-06 addenda for the exact corpus numbers this tool's own detectors are
calibrated against (94.7% two sentences, 98.9% question-final, 13.7% hyperbole first beats,
~20/95 grade-shaped, 0/188 apostrophes pre-register-rewrite, 12/40 "looks like" under VARY THE
SHAPE) -- this tool should reproduce those figures on the same local corpus, not invent new ones.

SOURCES, each clearly labeled in the output (see read_all_sources()):
  - data/hinge_debug/<run_id>/actions.jsonl -- walked recursively for any dict key literally
    named "opener" at ANY nesting depth (currently always top-level per run, but the walk does
    not assume that). These rows carry NO prompt era stamp at all: the debug log was never
    wired to prompt_stamp(), so every jsonl opener lands in the "unknown era" bucket regardless
    of when it was actually generated.
  - the SQLite store (data/operation_love.db by default; override with --db). Rows written
    since 2026-09-05 (b) carry `prompt_sha256`; earlier rows read back NULL and also land in
    "unknown era". Every row also carries `decision` (see DECISION GROUPING below): NULL/empty
    for a row written before decision-tracking existed at all, "like" for a landed Like via
    OpenerService.commit_opener, a non-like reason (today "dislike" / "never_sent") for a
    drafted-but-never-sent row via OpenerService.discard_opener, or the literal marker
    "synthetic_replay" (tools/opener_replay.py's DECISION_REPLAY) for an offline-generated row
    that was never shown to a human or AUTO to decide anything about at all -- see DECISION
    GROUPING below for why that marker gets its own bucket rather than folding into the
    not-sent one. A stale pre-redesign schema (no `openers`/`opener_rejections` table, or an
    empty one) is reported as exactly that, never silently treated as zero.
  - BigQuery, ONLY when --bigquery is passed (OFF by default -- see main()'s argument help).
    This tool must be useful with no credentials configured at all, so nothing here ever
    contacts BigQuery unless that flag is explicit. Carries the same `prompt_sha256` and
    `decision` columns as the SQLite store above (see operation_love/ranker/bigquery_store.py's
    `openers` schema), read and bucketed identically.
An opener seen in more than one source (e.g. a live run's jsonl debug line and its own
BigQuery `openers` row) is the same physical draft, so DEDUPE runs across all sources by exact
opener text before any metric is computed (see dedupe_openers()) -- while every per-source raw
count is still printed, so "how many did each source contribute" stays answerable even though
the metrics below are computed on the deduped, cross-source corpus.

METRICS are computed on two independent axes, both printed in full every run (never gated
behind a flag, and never only in --json):
  - by prompt_sha256 ERA (see build_era_metrics()), plus one "unknown" bucket for rows with no
    era -- answers "did the prompt change what gets produced". This axis POOLS every decision
    together within each era, so it says nothing on its own about what a human chose to send.
  - by DECISION (see build_decision_metrics()/decision_bucket()): "sent" (decision == "like"),
    "not_sent" (every other REAL not-sent decision -- today "dislike" and "never_sent"),
    "synthetic_replay" (decision == tools/opener_replay.py's DECISION_REPLAY marker -- its own
    bucket, never folded into "not_sent", because nobody ever decided not to send a replay row:
    it was never shown to a human or AUTO to send in the first place -- see the synthetic-replay
    BE HONEST caveat below), and "unknown" (NULL/empty -- a row written before decision-tracking
    existed). This is the axis that answers "what did the model produce that a human actually
    chose to send" -- see BE HONEST caveat 4 below for why this axis exists and the exact
    sent/not_sent/unknown boundary it is built around.
Both axes report the same per-bucket metric set: grade_rate (see the GRADE DETECTOR section
below), sentence_count distribution and the exactly-two-sentences share, question_final rate,
opening trigram diversity (distinct/total, top-5 coverage), apostrophe/contraction rate, "looks
like" rate, mean/median character length, and -- new -- how many of the bucket's rows are
synthetic replay rows plus the wholly/partially/none verdict that count implies (see
synthetic_status()), so an ERA populated by offline replay can never be mistaken for one
captured live. opener_rejections counts by reason_code remain store-only (never from jsonl) and
grouped by era only, since a rejected attempt was never sent by construction and carries no
`decision` column to group by.

BE HONEST. This tool prints (and main() always prints, not just on request) four caveats that
apply to every number it produces:
  1. These are DRAFTS, not sent messages. Nothing here proves a single one of them reached the
     device screen, let alone that Hinge delivered it.
  2. There is no reply, match, or any other conversational-outcome signal anywhere in this
     system. "Did it work" in the dating sense is permanently out of scope for this tool --
     it only ever answers "what did the prompt produce".
  3. The jsonl corpus carries no era stamp (see SOURCES above), so its numbers are a mixture of
     however many prompt eras happen to be represented in the local run directories and cannot
     be attributed to any single one of them.
  4. The store corpus's survivorship bias is now PARTIAL, not total, and the boundary is
     OpenerService.discard_opener (operation_love/opener/service.py): before that method
     existed, `record_opener` was only ever called from commit_opener at a successful Like
     commit, so every row from that era carries decision == "like" and a store-backed number
     from that era describes SENT openers only. Since discard_opener started being called, the
     store also durably records drafts nobody sent -- decision == "dislike" for an explicit
     Training reject, decision == "never_sent" for an AUTO stop/refusal/exception discard (see
     discard_opener's own docstring for the full lifecycle). This is NOT retroactive: a row
     written before discard_opener existed is still sent-only, never reclassified. A row with
     no `decision` recorded at all (NULL or empty -- predates decision-tracking entirely) is
     bucketed "unknown" and is never assumed to be sent either way. Because pooling all of this
     together would repeat exactly the mistake this caveat used to make, every number in this
     report is broken out by decision ("sent" / "not_sent" / "unknown") in addition to by era --
     see METRICS BY DECISION in the output, and never read a by-era number alone as being about
     sent openers. The `opener_rejections` counts remain the (also store-only, also partial)
     record of drafts a deterministic GUARD rejected before a human ever saw them, distinct from
     `openers.decision`, which records what a human (or AUTO) chose to do with a draft the guard
     let through.
If a metric cannot be computed for a source (missing table, empty table, no era/decision
column, no credentials), this tool prints WHY, in the output itself, rather than silently
omitting a row.

THE GRADE DETECTOR'S LEXICON LIVES ONLY IN THIS FILE. It must NEVER be imported into, copy-
pasted into, or otherwise referenced by the model-facing prompt (config.yaml opener.style,
opener.py's _SYSTEM/_SCHEMA/_ITEM_* constants) or by any online guard in opener.py. Two
existing repo decisions say why: the 2026-08-11 scaffolding-defense decision (deterministic
detectors stay OFFLINE measurement, never an online judge/classifier -- see
ops/OPENER-REDESIGN.md Sec. "opener scaffolding defense") and the 2026-08-16 de-templating rule
(no example verdict vocabulary ships on any model-facing surface, because a model that is shown
the very words being screened for learns to route around them -- see the NO GRADING addendum's
own "DELIBERATELY NOT CHANGED" paragraph). An offline measurement tool that nothing on the wire
ever reads has NO false-rejection cost -- a miss here costs nothing but a slightly stale count,
while the same lexicon reachable from the live guard would cost a real send -- which is exactly
why this lexicon is acceptable here and nowhere else in the codebase.

OUTPUT. A readable text table to stdout by default; --json for a machine-readable document;
--compare <era_a> <era_b> (either may be a unique prefix of a full prompt_sha256, a registered
era's human label, or the literal string "unknown"/"ALL") for a side-by-side delta of every
metric -- this is the tool's headline capability, since a repeatable N vs N+1 diff is the entire
reason it exists. Both output modes always include the METRICS BY DECISION breakdown (see
caveat 4) alongside the by-era one -- in text as its own "=== METRICS BY DECISION ===" section,
in --json as the top-level `by_decision` key -- regardless of whether --compare or --json is
passed; it is never a footnote and never conditional.

THE PROMPT ERA REGISTRY (--eras-file, default ops/prompt-eras.json, generated by
tools/backfill_prompt_eras.py -- see that module's docstring for how) is what turns every
prompt_sha256 this tool prints into something a human can reason about, and answers the
"have we tried this before" question the era grouping alone cannot. Every place an era is
printed -- METRICS BY PROMPT ERA, OPENER REJECTIONS, --compare's header, and --json's
`era_labels` side table -- resolves it to its registered label, falling back to the bare digest
(CLEARLY MARKED as such) when the registry is missing or the digest is unregistered; see
resolve_era_label(). --eras lists every known era oldest first with its label, date range, and
the named rules added/removed versus the era before it -- reading ONLY the registry, never the
corpus sources, so it works with no debug-dir or database at all. --rule NAME is the actual
"have we tried this" query: every known era NAME was on the wire in, when it first and last
appeared, and -- since it also reads the corpus sources -- whatever measured metrics exist for
those eras, so a stale "we tried that already" can be checked against what the prompt actually
produced rather than just when it shipped.

REPLAY CORPUS + THE PRE-REGISTERED CHECK (--replay-corpus-dir, default
operation_love.opener.replay_corpus.DEFAULT_CORPUS_DIR) are two more unconditional sections,
printed in both output modes exactly like METRICS BY DECISION above -- never gated behind a
flag. REPLAY CORPUS reports how many requests (operation_love.opener.replay_corpus captures,
read via that module's own list_replay_ids/load_replay_capture, never a re-implemented directory
walk) exist on disk, their captured_at date range, and how many distinct prompt eras they span --
see read_replay_corpus_stats(). THE PRE-REGISTERED CHECK reports progress toward, and (once
reached) the result of, ops/OPENER-REDESIGN.md's 2026-09-06 (d) pre-registered NO GRADING
prediction: PRE_REGISTERED_MIN_DRAFTS drafts (openers rows -- explicitly NOT the same count as a
replay CAPTURE above; see build_pre_registered_check()'s own docstring for why the two can
differ and which one the threshold applies to) generated under the CURRENT prompt era
(compute_current_prompt_era(), the same digest tools/opener_replay.py would write its next rows
under). Below threshold, this prints only the shortfall -- no verdict is ever rendered. At or
above it, it prints the four pre-registered metrics next to their predicted direction, whether
each passed, and the falsification verdict, with the exact thresholds pinned to named
module-level constants (PRE_REGISTERED_GRADE_RATE_MAX and siblings, PRE_REGISTERED_FALSIFY_*)
that comment-reference the addendum that set them, so the numbers can never quietly drift from
the pre-registered record. Both sections are carried through --json under the `replay_corpus`
and `pre_registered_check` top-level keys.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# `_leading_ngram` is opener.py's own leading-n-gram normalizer for the entropy guard (doc
# 3.6): lowercase, punctuation dropped, apostrophes kept, whitespace collapsed, first `n` words.
# Reused here (with n=3) rather than re-implemented so "opening trigram" means exactly the same
# tokenization the live near-duplicate guard already uses -- a second, slightly different
# tokenizer would make this tool's diversity numbers uncheckable against that guard's own
# behaviour. This is a plain, private, pure-string helper, not a prompt or a guard: importing it
# does not touch the lexicon-isolation rule in this module's docstring above, which is about the
# GRADE lexicon specifically, not about reusing an existing tokenizer.
from operation_love.opener.opener import _leading_ngram
# `OPENER_OUTCOME_MATCH`/`OPENER_OUTCOME_REPLY`: the two outcome kinds this tool's OUTCOMES axis
# (see build_outcome_metrics() below) counts as "a response" -- reused from the store's own
# documented vocabulary (ranker/__init__.py's KNOWN_OPENER_OUTCOMES) rather than re-spelling the
# literal strings "match"/"reply" here, so the two can never quietly drift apart.
from operation_love.ranker import OPENER_OUTCOME_MATCH, OPENER_OUTCOME_REPLY
# `DECISION_REPLAY`: the exact decision-column marker tools/opener_replay.py stamps on every
# synthetic offline-replay row ("synthetic_replay" -- see that module's own constant docstring).
# Imported, not re-spelled, for the same drift-proofing reason as the outcome constants above --
# this is the literal value read_sqlite_opener_outcomes()/read_bigquery_opener_outcomes() below
# check `openers.decision` against before ever counting a joined outcome row (see this module's
# OUTCOMES section and BE HONEST caveat on synthetic replay rows for why this exclusion exists at
# all, even though the join's own profile_key exclusion already keeps these rows out today).
# Importing tools.opener_replay adds no new dependency: it imports operation_love.opener.opener,
# which this module already imports (immediately above, for _leading_ngram) -- and it does not
# import this module back, so there is no import cycle.
# `DEFAULT_CORPUS_DIR`/`list_replay_ids`/`load_replay_capture`: the on-disk replay corpus's OWN
# read API (operation_love.opener.replay_corpus), reused here for the REPLAY CORPUS section
# (see read_replay_corpus_stats() below) so this tool never re-implements a second directory
# walk over the same on-disk format tools/opener_replay.py already reads.
from operation_love.opener.replay_corpus import (
    DECISION_REPLAY,
    DEFAULT_CORPUS_DIR,
    list_replay_ids,
    load_replay_capture,
)

DEFAULT_DEBUG_DIR = "data/hinge_debug"
# Matches config.py's own db_file default (data_dir/"operation_love.db") so running this tool
# with no flags at all inspects the same file a default `config.yaml` would actually write to.
DEFAULT_DB_FILE = "data/operation_love.db"

UNKNOWN_ERA = "unknown"


# ---------------------------------------------------------------------------------------
# Row shapes
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class OpenerRow:
    text: str
    source: str            # "jsonl" | "sqlite" | "bigquery"
    era: str | None        # prompt_sha256, or None (predates the stamp / no era column)
    decision: str | None = None
    # Raw `decision` column value, normalized so NULL and "" both read as None -- "like" for a
    # landed Like (OpenerService.commit_opener), a non-like reason such as "dislike" or
    # "never_sent" for a drafted-but-never-sent row (OpenerService.discard_opener), or None for
    # a row written before decision-tracking existed at all (jsonl NEVER carries this field --
    # see this module's docstring's SOURCES section -- so every jsonl row is also None here).
    # See decision_bucket() for how this becomes one of the three report buckets.


@dataclass(frozen=True)
class RejectionRow:
    reason_code: str       # "(none)" when the stored reason_code was NULL/empty
    source: str            # "sqlite" | "bigquery" -- jsonl never carries rejections, see below
    era: str | None


@dataclass(frozen=True)
class OutcomeRow:
    """One `opener_outcomes` row already joined to its `openers` row by (app, profile_key) --
    see read_sqlite_opener_outcomes()/read_bigquery_opener_outcomes() below, which perform this
    exact join (mirroring Store.joined_opener_outcomes' own predicate) before this dataclass is
    ever constructed. A row whose owning opener's `decision` was the synthetic-replay marker
    (DECISION_REPLAY) or whose `profile_key` was empty/NULL on either side of the join never
    reaches this dataclass at all -- see this module's OUTCOMES section and BE HONEST caveats."""
    era: str | None        # the SENT opener's prompt_sha256, or None (predates the stamp)
    outcome: str            # raw opener_outcomes.outcome text; "(none)" when NULL/empty
    read_source: str        # "sqlite" | "bigquery" -- which STORE this joined row was read from.
                            # NOT the same axis as opener_outcomes.source ("owner"/"automated",
                            # i.e. WHO recorded the observation -- see ranker/__init__.py's
                            # KNOWN_OPENER_OUTCOME_SOURCES); this tool does not currently group
                            # by that axis, only by era.


# ---------------------------------------------------------------------------------------
# SOURCE 1: data/hinge_debug/<run_id>/actions.jsonl
# ---------------------------------------------------------------------------------------

def find_opener_strings(value: Any) -> list[str]:
    """Recursively walk a decoded JSON value for every dict key literally named "opener" whose
    value is a non-empty string, at ANY nesting depth.

    Every run directory examined while building this tool carries "opener" as a top-level key
    of the action dict (`auto_opener_pre_send`, `auto_opener_resumed_send`), never nested --
    but the task this tool is built for is repeatable measurement, not a snapshot of today's
    shape, so the walk does not assume the field stays at depth 0 the next time the debug log
    is extended.
    """
    found: list[str] = []
    if isinstance(value, dict):
        for key, sub in value.items():
            if key == "opener" and isinstance(sub, str) and sub.strip():
                found.append(sub)
            found.extend(find_opener_strings(sub))
    elif isinstance(value, list):
        for item in value:
            found.extend(find_opener_strings(item))
    return found


@dataclass
class JsonlStats:
    run_dirs_total: int = 0
    run_dirs_with_actions_file: int = 0
    run_dirs_populated: int = 0     # >=1 opener key found
    opener_occurrences: int = 0     # every matching key, including repeats (e.g. a resumed send
                                    # re-logging the same draft) -- NOT deduped
    malformed_lines: int = 0


def read_jsonl_openers(debug_dir: Path) -> tuple[list[OpenerRow], JsonlStats]:
    """Every ``opener`` string found under ``debug_dir``/<run_id>/actions.jsonl.

    Never carries an era: see this module's docstring. A missing ``debug_dir`` is not an
    error -- it is reported as zero run directories, exactly like an empty one -- since a fresh
    checkout with no local debug data is a legitimate (if uninformative) thing to report on.
    """
    stats = JsonlStats()
    rows: list[OpenerRow] = []
    if not debug_dir.exists():
        return rows, stats
    for run_dir in sorted(p for p in debug_dir.iterdir() if p.is_dir()):
        stats.run_dirs_total += 1
        actions_file = run_dir / "actions.jsonl"
        if not actions_file.is_file():
            continue
        stats.run_dirs_with_actions_file += 1
        found_here = False
        with actions_file.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    stats.malformed_lines += 1
                    continue
                for text in find_opener_strings(obj):
                    normalized = " ".join(text.split())
                    if not normalized:
                        continue
                    rows.append(OpenerRow(text=normalized, source="jsonl", era=None))
                    stats.opener_occurrences += 1
                    found_here = True
        if found_here:
            stats.run_dirs_populated += 1
    return rows, stats


# ---------------------------------------------------------------------------------------
# SOURCE 2: the SQLite store (operation_love/ranker/store.py's schema)
# ---------------------------------------------------------------------------------------

@dataclass
class SqliteStats:
    available: bool = False
    notes: list[str] = field(default_factory=list)
    openers_row_count: int = 0
    rejections_row_count: int = 0
    # How many of openers_row_count's rows carry decision == DECISION_REPLAY (tools/
    # opener_replay.py's "synthetic_replay" marker) -- counted here, on the PRODUCED side, so
    # SOURCES can report it directly rather than a reader having to infer it from METRICS BY
    # DECISION's synthetic_replay bucket. These rows are NOT excluded from the produced-side
    # corpus (see this module's docstring and decision_bucket()) -- unlike the OUTCOMES axis's
    # own replay_excluded_count (SqliteOutcomeStats below), which mirrors this field's name but
    # counts an EXCLUSION, not an inclusion; see format_source_report() for the two lines side
    # by side.
    replay_row_count: int = 0


def read_sqlite_openers(db_path: Path) -> tuple[list[OpenerRow], list[RejectionRow], SqliteStats]:
    """Every ``openers``/``opener_rejections`` row in the SQLite store at ``db_path``.

    Opened read-only (SQLite's ``mode=ro`` URI) so a report run can never create, migrate, or
    otherwise mutate a database file that happens not to exist yet or to be on an older schema
    -- this tool only ever reads. A missing file, an unreadable file, or a file predating the
    `openers`/`opener_rejections` tables (see ranker/store.py's own migration comments -- this
    is exactly the state of the checked-in data/operation_love.db as of 2026-09-05) is reported
    in ``notes`` rather than raising, per this module's "print why" honesty rule.
    """
    stats = SqliteStats()
    opener_rows: list[OpenerRow] = []
    rejection_rows: list[RejectionRow] = []
    if not db_path.exists():
        stats.notes.append(f"no database file at {db_path}")
        return opener_rows, rejection_rows, stats
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError as exc:
        stats.notes.append(f"could not open {db_path} read-only: {exc}")
        return opener_rows, rejection_rows, stats
    stats.available = True
    try:
        tables = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}

        if "openers" not in tables:
            stats.notes.append(
                "no 'openers' table (pre-redesign schema, or a fresh/empty database)")
        else:
            columns = {row[1] for row in con.execute("PRAGMA table_info(openers)")}
            era_expr = "prompt_sha256" if "prompt_sha256" in columns else "NULL"
            if "prompt_sha256" not in columns:
                stats.notes.append(
                    "'openers' table predates the prompt_sha256 column; every row here lands "
                    "in the unknown-era bucket")
            decision_expr = "decision" if "decision" in columns else "NULL"
            if "decision" not in columns:
                stats.notes.append(
                    "'openers' table predates the decision column; every row here lands in "
                    "the unknown-decision bucket (see caveat 4)")
            for opener_text, era, decision in con.execute(
                    f"SELECT opener, {era_expr}, {decision_expr} FROM openers"):
                stats.openers_row_count += 1
                if not isinstance(opener_text, str) or not opener_text.strip():
                    continue
                normalized = " ".join(opener_text.split())
                # NULL and "" both read as "no decision recorded" -- see OpenerRow.decision and
                # decision_bucket() -- never guessed as either sent or not-sent.
                normalized_decision = decision if decision else None
                if normalized_decision == DECISION_REPLAY:
                    stats.replay_row_count += 1
                opener_rows.append(OpenerRow(
                    text=normalized, source="sqlite", era=era,
                    decision=normalized_decision))

        if "opener_rejections" not in tables:
            stats.notes.append("no 'opener_rejections' table (pre-redesign schema)")
        else:
            columns = {row[1] for row in con.execute("PRAGMA table_info(opener_rejections)")}
            era_expr = "prompt_sha256" if "prompt_sha256" in columns else "NULL"
            if "prompt_sha256" not in columns:
                stats.notes.append(
                    "'opener_rejections' table predates the prompt_sha256 column; every row "
                    "here lands in the unknown-era bucket")
            for reason_code, era in con.execute(
                    f"SELECT reason_code, {era_expr} FROM opener_rejections"):
                stats.rejections_row_count += 1
                rejection_rows.append(RejectionRow(
                    reason_code=reason_code if reason_code else "(none)",
                    source="sqlite", era=era))
    finally:
        con.close()
    return opener_rows, rejection_rows, stats


# ---------------------------------------------------------------------------------------
# SOURCE 2b: the SQLite store's `opener_outcomes` table, JOINED to `openers` -- the OUTCOMES
# axis (see this module's docstring's OUTCOMES section). A brand-new table (added the same
# session as this axis), so a database that predates it is reported via `notes`, never raised.
# ---------------------------------------------------------------------------------------

@dataclass
class SqliteOutcomeStats:
    available: bool = False
    notes: list[str] = field(default_factory=list)
    joined_row_count: int = 0          # outcome rows actually counted (post every exclusion)
    replay_excluded_count: int = 0     # rows dropped because their opener's decision was
                                       # DECISION_REPLAY (see this module's imports)


def read_sqlite_opener_outcomes(db_path: Path) -> tuple[list[OutcomeRow], SqliteOutcomeStats]:
    """Every `opener_outcomes` row in the SQLite store at `db_path`, joined to its `openers` row
    by (app, profile_key) -- the SAME join predicate SQLiteStore.joined_opener_outcomes uses
    (empty/NULL profile_key excluded on BOTH sides; see that method's own docstring in
    ranker/store.py for why two unattributable rows must never join to each other), issued here
    directly against a read-only connection rather than through a SQLiteStore instance: this
    tool's read-only contract (see read_sqlite_openers' own docstring immediately above) forbids
    ever constructing a real store, since SQLiteStore.__init__ runs schema migrations as a side
    effect and this tool must never create or migrate a database file it only means to inspect.

    Deliberately NOT scoped to one `app` the way Store.joined_opener_outcomes itself is (that
    method takes a mandatory `app` argument) -- every other reader in this module
    (read_sqlite_openers, read_bigquery_openers) reads across every app the local corpus happens
    to contain, and this one matches that convention: the join predicate still compares
    `oc.app = o.app` row by row, so outcomes never cross an app boundary, but no single app value
    is required up front.

    REPLAY ROWS: a row whose owning `openers.decision` is the synthetic-replay marker
    (DECISION_REPLAY, imported from tools/opener_replay.py) is dropped here and counted in
    `replay_excluded_count`, never in `joined_row_count` -- defensively, in ADDITION to the
    join's own profile_key exclusion. Today every row opener_replay.py writes already carries
    `profile_key=""` (see that module), so this exclusion is currently redundant with the join
    predicate above; it stays as an explicit, second, independent guard because a synthetic
    replay draft was never sent to a real person and can never have earned a real owner-observed
    outcome, so it must never be silently joined to somebody else's genuine outcome even if a
    future change to opener_replay.py ever let a replay row carry a real profile_key.
    """
    stats = SqliteOutcomeStats()
    rows: list[OutcomeRow] = []
    if not db_path.exists():
        stats.notes.append(f"no database file at {db_path}")
        return rows, stats
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError as exc:
        stats.notes.append(f"could not open {db_path} read-only: {exc}")
        return rows, stats
    stats.available = True
    try:
        tables = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "opener_outcomes" not in tables:
            stats.notes.append(
                "no 'opener_outcomes' table (predates this axis, or a fresh/empty database)")
            return rows, stats
        if "openers" not in tables:
            stats.notes.append(
                "'opener_outcomes' exists but 'openers' does not -- nothing to join it to")
            return rows, stats
        columns = {row[1] for row in con.execute("PRAGMA table_info(openers)")}
        if "profile_key" not in columns:
            stats.notes.append(
                "'openers' table predates the profile_key column; no outcome can be joined")
            return rows, stats
        era_expr = "prompt_sha256" if "prompt_sha256" in columns else "NULL"
        if "prompt_sha256" not in columns:
            stats.notes.append(
                "'openers' table predates the prompt_sha256 column; every joined outcome row "
                "here lands in the unknown-era bucket")
        decision_expr = "decision" if "decision" in columns else "NULL"
        if "decision" not in columns:
            stats.notes.append(
                "'openers' table predates the decision column; synthetic replay rows cannot be "
                "identified and excluded by decision here (the join's own profile_key exclusion "
                "still applies)")
        query = (
            f"SELECT o.{era_expr}, oc.outcome, o.{decision_expr} FROM openers o "
            "JOIN opener_outcomes oc ON oc.app = o.app AND oc.profile_key = o.profile_key "
            "WHERE o.profile_key IS NOT NULL AND o.profile_key != '' "
            "AND oc.profile_key IS NOT NULL AND oc.profile_key != ''"
        )
        for era, outcome, decision in con.execute(query):
            if decision == DECISION_REPLAY:
                stats.replay_excluded_count += 1
                continue
            stats.joined_row_count += 1
            rows.append(OutcomeRow(
                era=era, outcome=outcome if outcome else "(none)", read_source="sqlite"))
    finally:
        con.close()
    return rows, stats


# ---------------------------------------------------------------------------------------
# SOURCE 3: BigQuery -- OFF by default, only touched when --bigquery is passed
# ---------------------------------------------------------------------------------------

@dataclass
class BigQueryStats:
    attempted: bool = False
    available: bool = False
    notes: list[str] = field(default_factory=list)
    openers_row_count: int = 0
    rejections_row_count: int = 0
    # See SqliteStats.replay_row_count immediately above -- same meaning, same source table,
    # different backend.
    replay_row_count: int = 0


def read_bigquery_openers(project_id: str, dataset: str,
                          client=None) -> tuple[list[OpenerRow], list[RejectionRow], BigQueryStats]:
    """Every ``openers``/``opener_rejections`` row in BigQuery's ``project_id.dataset``.

    ``client`` is injectable (matching operation_love.ranker.bigquery_store's own testability
    contract) so tests exercise this against a fake and never need the google-cloud-bigquery
    package or network access. Real callers (main() with --bigquery) leave it None, which
    lazily imports and builds a real ``bigquery.Client`` -- lazy so importing this whole tool,
    or running it against jsonl/sqlite only, never requires the ``bq`` extra to be installed.
    """
    from operation_love.bigquery_validation import validate_bigquery_identifier

    stats = BigQueryStats(attempted=True)
    opener_rows: list[OpenerRow] = []
    rejection_rows: list[RejectionRow] = []
    try:
        project_id = validate_bigquery_identifier(project_id, "project_id")
        dataset = validate_bigquery_identifier(dataset, "dataset")
    except ValueError as exc:
        stats.notes.append(f"invalid BigQuery config: {exc}")
        return opener_rows, rejection_rows, stats

    if client is None:
        try:
            from google.cloud import bigquery
        except ImportError:
            stats.notes.append(
                "google-cloud-bigquery is not installed (install this project's 'bq' extra, "
                "or drop --bigquery to measure jsonl/sqlite only)")
            return opener_rows, rejection_rows, stats
        try:
            client = bigquery.Client(project=project_id)
        except Exception as exc:  # noqa: BLE001 -- a credentials/transport failure is data, not a crash
            stats.notes.append(f"could not create a BigQuery client: {type(exc).__name__}: {exc}")
            return opener_rows, rejection_rows, stats

    openers_table = f"{project_id}.{dataset}.openers"
    rejections_table = f"{project_id}.{dataset}.opener_rejections"
    try:
        result = client.query(
            f"SELECT opener, prompt_sha256, decision FROM `{openers_table}`").result()
        for row in result:
            stats.openers_row_count += 1
            text = row["opener"]
            if not isinstance(text, str) or not text.strip():
                continue
            decision = row["decision"]
            # NULL and "" both read as "no decision recorded" -- see OpenerRow.decision and
            # decision_bucket() -- never guessed as either sent or not-sent.
            normalized_decision = decision if decision else None
            if normalized_decision == DECISION_REPLAY:
                stats.replay_row_count += 1
            opener_rows.append(OpenerRow(
                text=" ".join(text.split()), source="bigquery", era=row["prompt_sha256"],
                decision=normalized_decision))
        stats.available = True
    except Exception as exc:  # noqa: BLE001 -- missing table/dataset/permissions is data, not a crash
        stats.notes.append(f"could not read {openers_table}: {type(exc).__name__}: {exc}")

    try:
        result = client.query(
            f"SELECT reason_code, prompt_sha256 FROM `{rejections_table}`").result()
        for row in result:
            stats.rejections_row_count += 1
            reason_code = row["reason_code"]
            rejection_rows.append(RejectionRow(
                reason_code=reason_code if reason_code else "(none)",
                source="bigquery", era=row["prompt_sha256"]))
    except Exception as exc:  # noqa: BLE001 -- same as above
        stats.notes.append(f"could not read {rejections_table}: {type(exc).__name__}: {exc}")

    return opener_rows, rejection_rows, stats


# ---------------------------------------------------------------------------------------
# SOURCE 3b: BigQuery's `opener_outcomes` table, JOINED to `openers` -- the OUTCOMES axis. OFF
# by default, exactly like SOURCE 3 above: only read when --bigquery is passed.
# ---------------------------------------------------------------------------------------

@dataclass
class BigQueryOutcomeStats:
    attempted: bool = False
    available: bool = False
    notes: list[str] = field(default_factory=list)
    joined_row_count: int = 0
    replay_excluded_count: int = 0


def read_bigquery_opener_outcomes(project_id: str, dataset: str,
                                  client=None) -> tuple[list[OutcomeRow], BigQueryOutcomeStats]:
    """Every `opener_outcomes` row in BigQuery's `project_id.dataset`, joined to its `openers`
    row by (app, profile_key) -- the SAME join predicate BigQueryStore.joined_opener_outcomes
    uses (empty/NULL profile_key excluded on both sides), issued directly here rather than
    through a BigQueryStore instance, mirroring read_bigquery_openers' own pattern immediately
    above (this tool never constructs a live store; see that function's docstring). Deliberately
    NOT scoped to one `app`, for the exact same reason read_sqlite_opener_outcomes() above isn't:
    every other reader in this module reads across every app the local corpus happens to
    contain.

    REPLAY ROWS: see read_sqlite_opener_outcomes()'s own docstring for the full rationale --
    the same DECISION_REPLAY exclusion applies here, defensively, in addition to the join's own
    profile_key exclusion.
    """
    from operation_love.bigquery_validation import validate_bigquery_identifier

    stats = BigQueryOutcomeStats(attempted=True)
    rows: list[OutcomeRow] = []
    try:
        project_id = validate_bigquery_identifier(project_id, "project_id")
        dataset = validate_bigquery_identifier(dataset, "dataset")
    except ValueError as exc:
        stats.notes.append(f"invalid BigQuery config: {exc}")
        return rows, stats

    if client is None:
        try:
            from google.cloud import bigquery
        except ImportError:
            stats.notes.append(
                "google-cloud-bigquery is not installed (install this project's 'bq' extra, "
                "or drop --bigquery to measure jsonl/sqlite only)")
            return rows, stats
        try:
            client = bigquery.Client(project=project_id)
        except Exception as exc:  # noqa: BLE001 -- a credentials/transport failure is data, not a crash
            stats.notes.append(f"could not create a BigQuery client: {type(exc).__name__}: {exc}")
            return rows, stats

    openers_table = f"{project_id}.{dataset}.openers"
    outcomes_table = f"{project_id}.{dataset}.opener_outcomes"
    query = (
        f"SELECT o.prompt_sha256 AS prompt_sha256, oc.outcome AS outcome, "
        "o.decision AS decision "
        f"FROM `{openers_table}` o JOIN `{outcomes_table}` oc "
        "ON oc.app = o.app AND oc.profile_key = o.profile_key "
        "WHERE o.profile_key IS NOT NULL AND o.profile_key != '' "
        "AND oc.profile_key IS NOT NULL AND oc.profile_key != ''"
    )
    try:
        result = client.query(query).result()
        for row in result:
            if row["decision"] == DECISION_REPLAY:
                stats.replay_excluded_count += 1
                continue
            outcome = row["outcome"]
            rows.append(OutcomeRow(
                era=row["prompt_sha256"], outcome=outcome if outcome else "(none)",
                read_source="bigquery"))
            stats.joined_row_count += 1
        stats.available = True
    except Exception as exc:  # noqa: BLE001 -- missing table/dataset/permissions is data, not a crash
        stats.notes.append(
            f"could not read joined {openers_table}/{outcomes_table}: {type(exc).__name__}: {exc}")

    return rows, stats


# ---------------------------------------------------------------------------------------
# Cross-source dedup
# ---------------------------------------------------------------------------------------

def dedupe_openers(rows: Iterable[OpenerRow]) -> list[OpenerRow]:
    """Collapse rows with identical opener text into one, keeping per-source raw counts
    meaningful (reported separately, before this runs) while every METRIC below is computed
    once per physical draft.

    A row carrying a known era always wins over one that doesn't for the same text (an
    era-stamped store row is strictly more informative than a jsonl duplicate of the same
    draft), and ties keep whichever copy was seen first -- callers are expected to pass jsonl
    rows first, then sqlite, then bigquery (see read_all_sources()), so a first-seen tie is
    already the least-authoritative source and stable regardless.
    """
    best: dict[str, OpenerRow] = {}
    for row in rows:
        existing = best.get(row.text)
        if existing is None:
            best[row.text] = row
        elif existing.era is None and row.era is not None:
            best[row.text] = row
    return list(best.values())


@dataclass
class SourceReport:
    jsonl_stats: JsonlStats
    jsonl_unique: int
    sqlite_stats: SqliteStats
    sqlite_unique: int
    bigquery_stats: BigQueryStats
    bigquery_unique: int
    combined_unique: int
    # Outcome-axis stats, appended with defaults so every existing positional SourceReport(...)
    # construction (tests included) keeps working unchanged -- same compatibility rule this
    # module already applies to every other trailing, defaulted parameter (see e.g.
    # Store.record_opener's own `prompt_sha256`/`profile_key` comments in ranker/__init__.py).
    sqlite_outcome_stats: SqliteOutcomeStats = field(default_factory=SqliteOutcomeStats)
    bigquery_outcome_stats: BigQueryOutcomeStats = field(default_factory=BigQueryOutcomeStats)


def read_all_sources(*, debug_dir: Path, db_path: Path, use_bigquery: bool,
                     bigquery_project: str | None = None, bigquery_dataset: str = "operation_love",
                     bigquery_client=None
                     ) -> tuple[list[OpenerRow], list[RejectionRow], list[OutcomeRow],
                               SourceReport]:
    """Read every configured source, dedupe the opener corpus, and return the combined result
    plus a SourceReport describing exactly what each source contributed (see this module's
    docstring's SOURCES section for what each field means).

    Outcome rows (the new OUTCOMES axis) are combined by simple concatenation, like rejection
    rows already are just below -- never cross-source deduped the way opener TEXT is: dedupe_
    openers() collapses the same DRAFT seen twice (e.g. a live run's jsonl line and its own store
    row), which has no equivalent concept for an owner-observed outcome row.
    """
    jsonl_rows, jsonl_stats = read_jsonl_openers(debug_dir)
    sqlite_rows, sqlite_rejections, sqlite_stats = read_sqlite_openers(db_path)
    sqlite_outcome_rows, sqlite_outcome_stats = read_sqlite_opener_outcomes(db_path)
    bigquery_rows: list[OpenerRow] = []
    bigquery_rejections: list[RejectionRow] = []
    bigquery_stats = BigQueryStats()
    bigquery_outcome_rows: list[OutcomeRow] = []
    bigquery_outcome_stats = BigQueryOutcomeStats()
    if use_bigquery:
        if not bigquery_project:
            bigquery_stats.attempted = True
            bigquery_stats.notes.append(
                "--bigquery was passed but no storage.bigquery.project_id was found in config "
                "(pass --config to point at the right config.yaml)")
            bigquery_outcome_stats.attempted = True
            bigquery_outcome_stats.notes.append(
                "--bigquery was passed but no storage.bigquery.project_id was found in config "
                "(pass --config to point at the right config.yaml)")
        else:
            bigquery_rows, bigquery_rejections, bigquery_stats = read_bigquery_openers(
                bigquery_project, bigquery_dataset, client=bigquery_client)
            bigquery_outcome_rows, bigquery_outcome_stats = read_bigquery_opener_outcomes(
                bigquery_project, bigquery_dataset, client=bigquery_client)

    combined = dedupe_openers(jsonl_rows + sqlite_rows + bigquery_rows)
    rejections = sqlite_rejections + bigquery_rejections
    outcomes = sqlite_outcome_rows + bigquery_outcome_rows

    report = SourceReport(
        jsonl_stats=jsonl_stats, jsonl_unique=len({r.text for r in jsonl_rows}),
        sqlite_stats=sqlite_stats, sqlite_unique=len({r.text for r in sqlite_rows}),
        bigquery_stats=bigquery_stats, bigquery_unique=len({r.text for r in bigquery_rows}),
        combined_unique=len(combined),
        sqlite_outcome_stats=sqlite_outcome_stats,
        bigquery_outcome_stats=bigquery_outcome_stats)
    return combined, rejections, outcomes, report


# ---------------------------------------------------------------------------------------
# REPLAY CORPUS (operation_love/opener/replay_corpus.py) -- how many captured requests exist ON
# DISK, independent of whether any of them has ever been replayed into the `openers` table (see
# tools/opener_replay.py). Read entirely through that module's OWN read API
# (list_replay_ids/load_replay_capture) rather than a second directory walk, mirroring this
# module's existing SOURCES convention above.
#
# A CAPTURE IS NOT A DRAFT. A capture is a request INPUT (the numbered item crops a live Hinge
# session once sent to Gemini, saved so it can be replayed later); a draft is an actual generated
# opener, one `openers` table row. The two counts can differ in either direction -- the same
# capture replayed twice under the same prompt era yields two drafts from one capture, and a live
# send needs no capture at all -- so the PRE-REGISTERED CHECK section immediately below keeps
# them separate rather than conflating "how many requests are on disk" with "how many drafts have
# been generated", which is the exact count ops/OPENER-REDESIGN.md's 2026-09-06 (d) pre-registered
# threshold applies to.
# ---------------------------------------------------------------------------------------

@dataclass
class ReplayCorpusStats:
    corpus_dir: Path
    capture_count: int = 0
    earliest_captured_at: float | None = None
    latest_captured_at: float | None = None
    distinct_prompt_eras: int = 0        # distinct non-null captured-at prompt_sha256 values
    unknown_era_count: int = 0           # captures whose manifest carries no prompt_sha256 at all
    notes: list[str] = field(default_factory=list)
    load_errors: list[str] = field(default_factory=list)  # "<replay_id>: <error>" per bad capture

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_dir": str(self.corpus_dir),
            "capture_count": self.capture_count,
            "earliest_captured_at": self.earliest_captured_at,
            "latest_captured_at": self.latest_captured_at,
            "distinct_prompt_eras": self.distinct_prompt_eras,
            "unknown_era_count": self.unknown_era_count,
            "notes": list(self.notes),
            "load_errors": list(self.load_errors),
        }


def read_replay_corpus_stats(corpus_dir: Path) -> ReplayCorpusStats:
    """How many captures exist on disk under ``corpus_dir``, their ``captured_at`` date range,
    and how many distinct prompt eras (captured-at ``prompt_sha256`` values) they span -- read
    entirely via ``replay_corpus.list_replay_ids``/``load_replay_capture`` (never a re-implemented
    walk; see this section's own header comment). A missing or empty ``corpus_dir`` is reported
    via ``notes`` with ``capture_count == 0``, never raised -- matching every other read_* function
    in this module's honesty convention. A capture whose manifest fails to load (corrupt/partial
    write) is counted in ``load_errors``, not ``capture_count``, and does not stop the rest of the
    corpus from being read.
    """
    stats = ReplayCorpusStats(corpus_dir=corpus_dir)
    try:
        replay_ids = list_replay_ids(corpus_dir)
    except Exception as exc:  # noqa: BLE001 -- a corrupt corpus root must not crash the report
        stats.notes.append(
            f"could not list the replay corpus at {corpus_dir}: {type(exc).__name__}: {exc}")
        return stats
    if not replay_ids:
        stats.notes.append(f"no captures found under {corpus_dir} (empty or nonexistent)")
        return stats
    eras: set[str] = set()
    captured_ats: list[float] = []
    for replay_id in replay_ids:
        try:
            capture = load_replay_capture(corpus_dir, replay_id)
        except Exception as exc:  # noqa: BLE001 -- one corrupt capture must not sink the report
            stats.load_errors.append(f"{replay_id}: {type(exc).__name__}: {exc}")
            continue
        stats.capture_count += 1
        if capture.captured_at is not None:
            captured_ats.append(capture.captured_at)
        if capture.prompt_sha256:
            eras.add(capture.prompt_sha256)
        else:
            stats.unknown_era_count += 1
    stats.distinct_prompt_eras = len(eras)
    if captured_ats:
        stats.earliest_captured_at = min(captured_ats)
        stats.latest_captured_at = max(captured_ats)
    return stats


def _format_epoch(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def format_replay_corpus_report(stats: ReplayCorpusStats) -> str:
    """The REPLAY CORPUS section: how many captures exist on disk, their date range, and how many
    distinct prompt eras they span. See this module's docstring and ops/OPENER-REDESIGN.md's
    2026-09-06 (i) addendum for what this corpus is; see this section's own header comment above
    for why ``capture_count`` here is deliberately never read as the same number as METRICS BY
    PROMPT ERA's ``n`` for any given era.
    """
    lines = [f"=== REPLAY CORPUS ({stats.corpus_dir}) ==="]
    lines.append(f"  captures on disk: {stats.capture_count}")
    if stats.capture_count == 0:
        lines.extend(f"  NOTE: {note}" for note in stats.notes)
        return "\n".join(lines)
    if stats.earliest_captured_at is not None:
        lines.append(f"  date range       : {_format_epoch(stats.earliest_captured_at)} -> "
                     f"{_format_epoch(stats.latest_captured_at)}")
    else:
        lines.append("  date range       : unknown (no capture recorded a captured_at)")
    era_note = (f", {stats.unknown_era_count} with no era stamp"
               if stats.unknown_era_count else "")
    lines.append(f"  distinct prompt eras spanned: {stats.distinct_prompt_eras}{era_note}")
    if stats.load_errors:
        lines.append(f"  {len(stats.load_errors)} capture(s) failed to load:")
        lines.extend(f"    {err}" for err in stats.load_errors)
    lines.extend(f"  NOTE: {note}" for note in stats.notes)
    return "\n".join(lines)


# =========================================================================================
# GRADE DETECTOR -- see this module's docstring for why this lexicon lives ONLY here.
# =========================================================================================
#
# Detects the "grade-shaped predicate" named in ops/OPENER-REDESIGN.md's 2026-09-06 addendum
# ("a compliment moved off her is still a score"): a FIRST beat built as a copula whose
# complement is an unconditional evaluative verdict ("... is an elite move.", "... is such an
# elite move."), which reads as a grade no matter what noun the subject names. Implemented as a
# DETERMINISTIC pattern match, per the 2026-08-11 scaffolding-defense decision, never an LLM
# judge or classifier.
#
# THIS IS A LOWER BOUND, NOT A MEASUREMENT -- exactly like opener.py's own redundancy monitor
# (see _redundant_description_markers's docstring there for the same shape of caveat). A
# synonym absent from the two lexicon tuples below is a MISS (a real grade, nothing reported),
# never a false alarm on a clean opener: that is the direction a measurement tool should fail
# in, since this number only ever informs a human reading a report, and never gates a send.
# Extend the lexicon as new grade wordings are observed in a fresh corpus; do not read an
# unmatched wording as proof the model stopped grading.
#
# Seeded directly from the worked examples in that addendum: "is an elite move" (the sushi
# opener quoted verbatim), "is such an elite move" (the matcha-tiramisu opener, also quoted
# verbatim), and the ~20-of-95 near-miss tally ("is a bold move", "is iconic weekend energy",
# "is top tier apres ski energy", "is unmatched", "is a masterpiece", "is unreal", "is
# impressively professional", "a great look", "the perfect ...", "an incredible/amazing
# experience", "is pretty fantastic").
_EVALUATIVE_ADJECTIVES: tuple[str, ...] = (
    "elite", "unreal", "unmatched", "iconic", "impressive", "amazing", "incredible",
    "fantastic", "phenomenal", "legendary", "flawless", "perfect", "immaculate", "unbeatable",
    "unstoppable", "stunning", "gorgeous", "epic", "insane", "wild", "impeccable",
    "unforgettable", "magical", "majestic", "glorious", "surreal", "spectacular",
    "impressively professional",
)
# Short evaluative NOUN PHRASES that grade something as a whole ("is a masterpiece") rather
# than via a bare adjective. Kept separate from the adjective tuple above because a noun phrase
# only counts as this predicate's complement when it follows an article (see
# _grade_complement_after_copula), while a bare adjective does not need one.
_EVALUATIVE_NOUN_PHRASES: tuple[str, ...] = (
    "elite move", "bold move", "power move", "masterpiece", "great look", "iconic energy",
    "weekend energy", "main character energy", "top tier", "chef's kiss", "work of art",
)

# The copula forms this detector matches, per spec: is / was / are / the 's contraction. A
# lookbehind requires a word character immediately before "'s" so a bare possessive apostrophe
# floating alone (never occurs in practice, but cheap to guard) can't match.
_COPULA_RE = re.compile(r"\bis\b|\bwas\b|\bare\b|(?<=[A-Za-z0-9])'s\b", re.IGNORECASE)

# One optional degree/intensifier word directly after the copula ("is SUCH an elite move"), and
# one optional article directly after that -- both stripped, in this order, before the
# remaining head phrase is compared against the lexicon above. Order matters: "such an elite
# move" only reduces to "elite move" by removing the intensifier BEFORE the article.
_GRADE_INTENSIFIER_RE = re.compile(
    r"^(?:such|so|really|very|truly|absolutely|totally|honestly|pretty)\s+", re.IGNORECASE)
_GRADE_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)

_TRAILING_PUNCTUATION_RE = re.compile(r"[.!?,;:\"'”’]+$")


def _grade_complement_after_copula(remainder: str) -> str:
    """Strip one leading intensifier, then one leading article, from a copula's complement."""
    stripped = _GRADE_INTENSIFIER_RE.sub("", remainder, count=1)
    stripped = _GRADE_ARTICLE_RE.sub("", stripped, count=1)
    return stripped.strip()


def _complement_is_evaluative(remainder: str) -> bool:
    """True when ``remainder`` (the copula's complement, after stripping one intensifier and
    one article) BEGINS with a lexicon phrase or adjective. A prefix match, not an exact match,
    on purpose: "elite move for the office party" still opens with "elite move", and a grade
    doesn't stop being one because a modifier trails it."""
    lowered = _TRAILING_PUNCTUATION_RE.sub("", remainder.strip().lower())
    if not lowered:
        return False
    for phrase in _EVALUATIVE_NOUN_PHRASES + _EVALUATIVE_ADJECTIVES:
        if lowered == phrase or lowered.startswith(phrase + " "):
            return True
    return False


def is_grade_shaped(sentence: str) -> bool:
    """Whether ``sentence`` (expected to be one already-split sentence -- see split_sentences)
    is this detector's grade-shaped predicate: a declarative copula clause whose complement is
    a bare evaluative noun phrase or adjective from the module-level lexicon above.

    Requires the sentence NOT end in "?": an interrogative use of the same copula ("Is that
    spread an elite move?") is a question, not an assertion, and a verdict has to actually be
    asserted to be a grade. Every copula occurrence in the sentence is tried in turn (a
    sentence has at most one or two in practice) and the first one whose complement matches the
    lexicon settles it.

    A negated copula ("That is not an elite move.") is never flagged, with no dedicated carve-
    out needed: "not"/"never"/an "n't" contraction are neither an intensifier nor an article, so
    _grade_complement_after_copula strips neither one, and the complement is then compared
    starting from "not ...", which is never equal to (or a prefix of) any lexicon entry.
    """
    text = sentence.strip()
    if not text or text.endswith("?"):
        return False
    for match in _COPULA_RE.finditer(text):
        remainder = text[match.end():].lstrip()
        if not remainder:
            continue
        complement = _grade_complement_after_copula(remainder)
        if _complement_is_evaluative(complement):
            return True
    return False


# =========================================================================================
# Sentence splitting, question-final, trigram, apostrophe, "looks like", length metrics
# =========================================================================================

# A small, self-contained sentence splitter for MEASUREMENT only -- deliberately NOT the same
# code as operation_love.opener.opener._sentence_count (the live hard two-sentence send guard).
# That function only needs a COUNT; this tool also needs the actual first-sentence TEXT for the
# grade detector above, and coupling an offline analysis tool to the exact regex behind a live
# send guard would make this tool's numbers silently drift the moment that guard is tuned for
# an unrelated reason. Handles the common title abbreviations (Mr./Dr./etc.) so "Dr. Dolittle
# energy. What's the story?" doesn't split into three sentences; openers are short enough
# (two-sentence hard cap) that nothing more elaborate is needed.
_ABBREVIATION_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Jr|Sr|St|Mt|vs|etc)\.", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, each still carrying its own terminal punctuation."""
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return []
    # Swap the abbreviation's period for a placeholder that can't be mistaken for a sentence
    # end, split, then swap it back -- so "Dr." never becomes a sentence boundary.
    protected = _ABBREVIATION_RE.sub(lambda m: m.group(0).replace(".", "․"), cleaned)
    parts = [p.replace("․", ".") for p in _SENTENCE_SPLIT_RE.split(protected) if p]
    return parts or [cleaned]


def ends_in_question(text: str) -> bool:
    """True when ``text``, after trimming trailing whitespace/quotes/parens, ends in "?"."""
    trimmed = str(text).rstrip()
    trimmed = trimmed.rstrip("\"'”’)")
    return trimmed.endswith("?")


_APOSTROPHE_CHARS = ("'", "’", "‘")


def has_apostrophe(text: str) -> bool:
    return any(ch in text for ch in _APOSTROPHE_CHARS)


_LOOKS_LIKE_RE = re.compile(r"\blooks like\b", re.IGNORECASE)


def has_looks_like(text: str) -> bool:
    return bool(_LOOKS_LIKE_RE.search(text))


def opening_trigram(text: str) -> str:
    """The normalized opening trigram (first three words), via opener.py's own leading-n-gram
    normalizer -- see this module's import comment for why n=3 here is the only difference from
    the entropy guard's own n=4 use of the same function."""
    return _leading_ngram(text, n=3)


# ---------------------------------------------------------------------------------------
# Per-era metrics
# ---------------------------------------------------------------------------------------

@dataclass
class EraMetrics:
    era: str
    n: int
    grade_count: int
    grade_rate: float
    sentence_counts: dict[int, int]
    two_sentence_count: int
    two_sentence_rate: float
    question_final_count: int
    question_final_rate: float
    distinct_trigrams: int
    total_trigrams: int
    trigram_diversity: float
    top5_trigrams: list[tuple[str, int]]
    top5_trigram_coverage: float
    apostrophe_count: int
    apostrophe_rate: float
    looks_like_count: int
    looks_like_rate: float
    mean_chars: float
    median_chars: float
    # How many of this bucket's `n` rows carry decision == DECISION_REPLAY (tools/
    # opener_replay.py's "synthetic_replay" marker) -- 0 by default so every existing positional/
    # keyword era_metrics(era, texts) call (tests included) keeps working unchanged and simply
    # reports "no synthetic rows here", exactly the same trailing-defaulted-field convention
    # SourceReport's own outcome-stats fields already use above. On the DECISION axis this is
    # redundant with the bucket itself (the synthetic_replay bucket's every row IS replay, by
    # construction of decision_bucket()) but is still populated for that axis too (see
    # build_decision_metrics()) so a JSON consumer never sees an inconsistent 0 there. See
    # synthetic_status() below for how a bucket's `n`/`replay_count` become a wholly/partial/none
    # verdict, and format_era_metrics()/format_compare() for where that verdict is surfaced on
    # the ERA axis specifically (see this module's docstring and the synthetic-replay caveat for
    # why the ERA axis, not the DECISION axis, is where a reader needs this most: a whole ERA
    # populated by replay looks exactly like a live one unless something marks it).
    replay_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "era": self.era, "n": self.n,
            "grade_count": self.grade_count, "grade_rate": self.grade_rate,
            "sentence_counts": dict(self.sentence_counts),
            "two_sentence_count": self.two_sentence_count,
            "two_sentence_rate": self.two_sentence_rate,
            "question_final_count": self.question_final_count,
            "question_final_rate": self.question_final_rate,
            "distinct_trigrams": self.distinct_trigrams, "total_trigrams": self.total_trigrams,
            "trigram_diversity": self.trigram_diversity,
            "top5_trigrams": [list(pair) for pair in self.top5_trigrams],
            "top5_trigram_coverage": self.top5_trigram_coverage,
            "apostrophe_count": self.apostrophe_count, "apostrophe_rate": self.apostrophe_rate,
            "looks_like_count": self.looks_like_count, "looks_like_rate": self.looks_like_rate,
            "mean_chars": self.mean_chars, "median_chars": self.median_chars,
            "replay_count": self.replay_count,
            "synthetic_status": synthetic_status(self.n, self.replay_count),
        }


# The three verdicts synthetic_status() below can return, for a bucket's `n` total rows and
# `replay_count` of them carrying tools/opener_replay.py's DECISION_REPLAY marker.
SYNTHETIC_NONE = "none"
SYNTHETIC_PARTIAL = "partial"
SYNTHETIC_WHOLLY = "wholly"


def synthetic_status(n: int, replay_count: int) -> str:
    """Classify a bucket as SYNTHETIC_WHOLLY (every row is a replay row), SYNTHETIC_PARTIAL (some
    but not all), or SYNTHETIC_NONE (n==0, or replay_count==0) -- the verdict
    format_era_metrics()/format_compare() print on the ERA axis so a reader can never mistake a
    replayed era for a live one (see this module's docstring and the synthetic-replay caveat).
    Never divides by zero: n==0 is SYNTHETIC_NONE regardless of replay_count (an empty bucket has
    no rows to be wholly or partly anything)."""
    if not n or not replay_count:
        return SYNTHETIC_NONE
    if replay_count >= n:
        return SYNTHETIC_WHOLLY
    return SYNTHETIC_PARTIAL


def era_synthetic_marker(metrics: EraMetrics) -> str:
    """The " [...]" header suffix naming whether ``metrics`` is wholly or partially populated by
    tools/opener_replay.py's synthetic replay rows -- "" (nothing to append) when it is not, so
    every caller can unconditionally concatenate this rather than branching on synthetic_status()
    itself. See format_era_metrics() (ERA axis only) and format_compare() (both sides, always
    ERA-axis buckets) for where this is actually printed."""
    status = synthetic_status(metrics.n, metrics.replay_count)
    if status == SYNTHETIC_WHOLLY:
        return (" [WHOLLY SYNTHETIC REPLAY ERA -- GENERATED by tools/opener_replay.py, not "
                "captured from live use; see caveats]")
    if status == SYNTHETIC_PARTIAL:
        return (f" [PARTIALLY SYNTHETIC REPLAY: {metrics.replay_count}/{metrics.n} row(s) "
                "generated by tools/opener_replay.py; see caveats]")
    return ""


def era_metrics(era: str, texts: Sequence[str], *, replay_count: int = 0) -> EraMetrics:
    """Every corpus metric this tool defines, computed once over ``texts`` (one prompt era's,
    or the combined corpus's, deduped opener drafts). Never raises on an empty ``texts``: every
    rate is 0.0 and every distribution is empty, which callers print as "n=0" rather than
    dividing by zero.

    ``replay_count`` (default 0) is metadata the caller already knows from the same rows
    ``texts`` was extracted from (see build_era_metrics()/build_decision_metrics()) -- it does
    not affect any metric computed below, it only rides along on the returned EraMetrics so
    synthetic_status()/era_synthetic_marker() can later classify this bucket without a second
    pass over the original OpenerRow objects."""
    n = len(texts)

    grade_count = 0
    sentence_counts: Counter[int] = Counter()
    two_sentence_count = 0
    question_final_count = 0
    trigram_counts: Counter[str] = Counter()
    apostrophe_count = 0
    looks_like_count = 0
    lengths: list[int] = []

    for text in texts:
        sentences = split_sentences(text)
        if sentences and is_grade_shaped(sentences[0]):
            grade_count += 1
        count = len(sentences)
        sentence_counts[count] += 1
        if count == 2:
            two_sentence_count += 1
        if ends_in_question(text):
            question_final_count += 1
        trigram_counts[opening_trigram(text)] += 1
        if has_apostrophe(text):
            apostrophe_count += 1
        if has_looks_like(text):
            looks_like_count += 1
        lengths.append(len(text))

    top5 = trigram_counts.most_common(5)
    top5_coverage = (sum(c for _, c in top5) / n) if n else 0.0

    return EraMetrics(
        era=era, n=n,
        grade_count=grade_count, grade_rate=(grade_count / n) if n else 0.0,
        sentence_counts=dict(sorted(sentence_counts.items())),
        two_sentence_count=two_sentence_count,
        two_sentence_rate=(two_sentence_count / n) if n else 0.0,
        question_final_count=question_final_count,
        question_final_rate=(question_final_count / n) if n else 0.0,
        distinct_trigrams=len(trigram_counts), total_trigrams=n,
        trigram_diversity=(len(trigram_counts) / n) if n else 0.0,
        top5_trigrams=top5, top5_trigram_coverage=top5_coverage,
        apostrophe_count=apostrophe_count, apostrophe_rate=(apostrophe_count / n) if n else 0.0,
        looks_like_count=looks_like_count, looks_like_rate=(looks_like_count / n) if n else 0.0,
        mean_chars=statistics.mean(lengths) if lengths else 0.0,
        median_chars=statistics.median(lengths) if lengths else 0.0,
        replay_count=replay_count,
    )


def build_era_metrics(rows: Sequence[OpenerRow]) -> dict[str, EraMetrics]:
    """Group ``rows`` by era (None -> UNKNOWN_ERA) and compute EraMetrics for each, plus one
    combined "ALL" bucket over every row regardless of era.

    Also tallies, per era, how many of its rows carry decision == DECISION_REPLAY and threads
    that count into era_metrics()'s own ``replay_count`` -- so a prompt era populated in whole or
    in part by tools/opener_replay.py's offline replay is visible on the ERA axis itself (see
    synthetic_status()/era_synthetic_marker() and this module's docstring/caveats), rather than
    only inferable by cross-referencing METRICS BY DECISION's separate synthetic_replay bucket.
    """
    by_era: dict[str, list[str]] = defaultdict(list)
    replay_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        era = row.era or UNKNOWN_ERA
        by_era[era].append(row.text)
        if row.decision == DECISION_REPLAY:
            replay_counts[era] += 1
    result = {era: era_metrics(era, texts, replay_count=replay_counts.get(era, 0))
             for era, texts in by_era.items()}
    result["ALL"] = era_metrics(
        "ALL", [row.text for row in rows],
        replay_count=sum(1 for row in rows if row.decision == DECISION_REPLAY))
    return result


def rejections_by_era(rows: Sequence[RejectionRow]) -> dict[str, Counter[str]]:
    """reason_code counts, grouped the same way build_era_metrics groups openers."""
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        result[row.era or UNKNOWN_ERA][row.reason_code] += 1
    return result


# ---------------------------------------------------------------------------------------
# Decision grouping -- sent vs not-sent vs synthetic-replay vs unknown/legacy. See this module's
# docstring's BE HONEST caveat 4 for what the sent/not_sent/unknown boundary means
# (OpenerService.discard_opener), and the caveat appended for synthetic replay rows for what the
# fourth (DECISION_REPLAY) bucket means -- a replay row is not a human/AUTO decision at all, so it
# must never be pooled into DECISION_NOT_SENT alongside a real rejected draft. METRICS above
# explains why this whole grouping is a SEPARATE axis from era.
# ---------------------------------------------------------------------------------------

DECISION_SENT = "sent"
DECISION_NOT_SENT = "not_sent"
# The synthetic-replay bucket's name is DECISION_REPLAY ITSELF ("synthetic_replay", imported
# from tools/opener_replay.py -- see this module's top-of-file import comment) rather than a
# second, independently-spelled bucket constant: reusing the raw marker string as the bucket
# name both makes the label unmistakable wherever it is printed (the exact string a reader could
# grep opener_replay.py for is the exact string that shows up in this report) and guarantees the
# bucket name can never drift from the marker decision_bucket() below is testing against.
DECISION_UNKNOWN = "unknown"
_DECISION_BUCKET_ORDER = (DECISION_SENT, DECISION_NOT_SENT, DECISION_REPLAY, DECISION_UNKNOWN)


def decision_bucket(decision: str | None) -> str:
    """Classify one row's raw ``decision`` column value into exactly one of FOUR buckets.

    None/"" (NULL or empty in the store, or any jsonl row -- see OpenerRow.decision) is a row
    with no decision recorded at all -- bucketed DECISION_UNKNOWN, never guessed as either sent
    or not-sent, per this module's honesty rule. "like" is OpenerService.commit_opener's own
    decision value for a landed Like (see operation_love/opener/service.py) -- historically the
    ONLY value the `openers` table ever held, and the only one that has ever meant "this
    reached the device" -- bucketed DECISION_SENT. DECISION_REPLAY ("synthetic_replay",
    tools/opener_replay.py's own marker for an offline-generated row) gets its OWN bucket,
    checked before the catch-all below: a replay row is not a human (or AUTO) decision at all --
    nobody chose not to send it, it was simply never sent to anyone -- so it must never sit
    indistinguishably next to a real "dislike"/"never_sent" row inside DECISION_NOT_SENT (this
    was exactly this function's defect before this bucket existed: every non-"like" decision,
    replay included, folded into the same not_sent bucket a human's own rejected draft landed
    in). Every OTHER non-empty value (today: "dislike" from Training's explicit reject,
    "never_sent" from AUTO's stop/refusal/exception discard -- see OpenerService.discard_opener)
    means a draft that was generated but never sent, bucketed DECISION_NOT_SENT -- deliberately a
    catch-all rather than an enumerated allowlist, so a future discard reason this tool has never
    seen still lands in the correct bucket instead of silently falling out of every group or
    being miscounted as sent.
    """
    if not decision:
        return DECISION_UNKNOWN
    if decision == "like":
        return DECISION_SENT
    if decision == DECISION_REPLAY:
        return DECISION_REPLAY
    return DECISION_NOT_SENT


def build_decision_metrics(rows: Sequence[OpenerRow]) -> dict[str, EraMetrics]:
    """Group ``rows`` by decision_bucket() and compute EraMetrics for each bucket that actually
    occurs, plus one combined "ALL" bucket over every row regardless of decision -- the same
    pooled number caveat 4 warns against reading as "sent openers" on its own.

    Reuses era_metrics()/EraMetrics verbatim: every metric this tool defines is computed once,
    over a sequence of texts, regardless of which axis (era or decision) grouped them -- see
    build_era_metrics() for the era-axis equivalent of this function.

    Also threads a ``replay_count`` into each bucket exactly like build_era_metrics() does --
    trivially every row of the DECISION_REPLAY bucket itself (a replay row's decision IS the
    replay marker, by decision_bucket()'s own construction) and 0 for every other bucket (a
    replay row can never land in sent/not_sent/unknown) -- purely so a JSON consumer of
    ``by_decision`` never sees an inconsistent 0 on the one bucket where it should read `n`.
    """
    by_bucket: dict[str, list[str]] = defaultdict(list)
    bucket_replay_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        bucket = decision_bucket(row.decision)
        by_bucket[bucket].append(row.text)
        if row.decision == DECISION_REPLAY:
            bucket_replay_counts[bucket] += 1
    result = {bucket: era_metrics(bucket, texts, replay_count=bucket_replay_counts.get(bucket, 0))
             for bucket, texts in by_bucket.items()}
    result["ALL"] = era_metrics(
        "ALL", [row.text for row in rows],
        replay_count=sum(1 for row in rows if row.decision == DECISION_REPLAY))
    return result


def sort_decision_keys(buckets: Iterable[str]) -> list[str]:
    """Deterministic display order: sent, then not_sent, then unknown, then the combined total
    -- mirrors sort_era_keys()'s own ordering rationale, but only over the (small, fixed)
    decision vocabulary rather than an alphabetical sort of arbitrary digests."""
    present = set(buckets)
    ordered = [bucket for bucket in _DECISION_BUCKET_ORDER if bucket in present]
    if "ALL" in present:
        ordered = ordered + ["ALL"]
    return ordered


def sort_era_keys(eras: Iterable[str]) -> list[str]:
    """Deterministic display order: real prompt_sha256 digests first (alphabetically, so a
    re-run's output diffs cleanly), then the unknown-era bucket, then the combined total."""
    named = sorted(era for era in eras if era not in (UNKNOWN_ERA, "ALL"))
    ordered = named
    if UNKNOWN_ERA in eras:
        ordered = ordered + [UNKNOWN_ERA]
    if "ALL" in eras:
        ordered = ordered + ["ALL"]
    return ordered


def resolve_era(available: Sequence[str], token: str) -> str:
    """Resolve a --compare argument to exactly one era key in ``available``.

    Accepts an exact key (a full prompt_sha256, "unknown", or "ALL") or a unique case-
    insensitive PREFIX of a real digest -- typing a full 64-character sha256 on a command line
    is exactly the friction this convenience exists to remove. Raises ValueError with a message
    naming either "no era matches" or the full list of ambiguous matches, since a silent
    arbitrary pick would make --compare's whole point (a trustworthy before/after diff) unsafe.
    """
    if token in available:
        return token
    lowered = token.lower()
    matches = [era for era in available if era.lower().startswith(lowered)]
    if not matches:
        raise ValueError(
            f"no era matches {token!r}; available eras: {sort_era_keys(available)}")
    if len(matches) > 1:
        raise ValueError(f"{token!r} is ambiguous; matches: {sorted(matches)}")
    return matches[0]


# ---------------------------------------------------------------------------------------
# OUTCOMES -- did the openers written under prompt era X actually PERFORM better. Built from
# OutcomeRow (already joined to its era by read_sqlite_opener_outcomes()/
# read_bigquery_opener_outcomes() above), grouped by era exactly like build_era_metrics() groups
# OpenerRow -- see that function for the shape this one deliberately mirrors. See this module's
# docstring's OUTCOMES section and BE HONEST caveats 5-8 for what this axis does and does not
# prove.
# ---------------------------------------------------------------------------------------

# The two outcome kinds counted as "a response" for OutcomeMetrics.response_rate below: the
# recipient re-engaged in some observable way. Deliberately excludes OPENER_OUTCOME_UNMATCH (a
# prior match/conversation that later disappeared) from the numerator -- an unmatch means she DID
# respond at some point, but folding it in either direction (as a response, or as a non-response)
# would quietly redefine what "response rate" means depending on how attrition is counted; it is
# reported in full in `counts` below, just never in this one summary number. Also excludes
# OPENER_OUTCOME_NO_RESPONSE and OPENER_OUTCOME_UNKNOWN, per their own definitions in
# ranker/__init__.py. Not a closed set enforced anywhere else: an outcome string this tool has
# never seen simply is not a response, exactly like decision_bucket()'s catch-all NOT_SENT
# bucket handles a novel decision value it has never seen either.
_RESPONSE_OUTCOME_KINDS = frozenset({OPENER_OUTCOME_MATCH, OPENER_OUTCOME_REPLY})

# HARD HONESTY GUARD (see this module's docstring / the task this axis ships under). Below this
# many joined outcome rows in a bucket, a single new observation swings the reported response
# rate by more than ten percentage points (1/9 -> 2/10 is +11.1 points; 9/9 -> 9/10 is -10
# points) -- printing a number that unstable, to one decimal place, invites reading it as a real
# measurement when it is closer to noise. This is a documented JUDGMENT CALL, not a statistically
# derived confidence bound: the corpus this axis measures (real, owner-observed conversation
# outcomes) will be small for a long time (see this module's docstring), and printing some number
# under the guise of a rate is worse than printing none and showing the raw counts instead (see
# OutcomeMetrics.rate_blocked / format_outcome_metrics()). Raise this constant if experience
# shows 10 still reads as more precise than it is; never lower it just to make a rate appear
# sooner.
MIN_SAMPLE_FOR_OUTCOME_RATE = 10


@dataclass
class OutcomeMetrics:
    era: str
    n: int                          # total joined outcome rows counted in this bucket
    counts: dict[str, int]           # raw outcome string -> count (e.g. "match", "reply",
                                     # "no_response", "unmatch", "unknown", "(none)")
    response_count: int              # sum of counts[k] for k in _RESPONSE_OUTCOME_KINDS
    response_rate: float | None      # None exactly when rate_blocked is True
    rate_blocked: bool                # True when n < MIN_SAMPLE_FOR_OUTCOME_RATE

    def to_dict(self) -> dict[str, Any]:
        return {
            "era": self.era, "n": self.n, "counts": dict(self.counts),
            "response_count": self.response_count, "response_rate": self.response_rate,
            "rate_blocked": self.rate_blocked,
            "min_sample_for_rate": MIN_SAMPLE_FOR_OUTCOME_RATE,
        }


def outcome_metrics(era: str, outcomes: Sequence[str]) -> OutcomeMetrics:
    """Every OUTCOMES-axis metric this tool defines, computed once over ``outcomes`` (one era's,
    or the combined corpus's, raw outcome strings). Never raises on an empty ``outcomes``: n=0
    is itself below MIN_SAMPLE_FOR_OUTCOME_RATE, so response_rate is None and rate_blocked is
    True exactly like any other too-small bucket -- no separate zero-division branch needed."""
    n = len(outcomes)
    counts: Counter[str] = Counter(outcomes)
    response_count = sum(c for kind, c in counts.items() if kind in _RESPONSE_OUTCOME_KINDS)
    rate_blocked = n < MIN_SAMPLE_FOR_OUTCOME_RATE
    response_rate = None if rate_blocked else (response_count / n)
    return OutcomeMetrics(
        era=era, n=n, counts=dict(sorted(counts.items())), response_count=response_count,
        response_rate=response_rate, rate_blocked=rate_blocked)


def build_outcome_metrics(rows: Sequence[OutcomeRow]) -> dict[str, OutcomeMetrics]:
    """Group ``rows`` by era (None -> UNKNOWN_ERA) and compute OutcomeMetrics for each, plus one
    combined "ALL" bucket over every row regardless of era -- the exact same shape
    build_era_metrics() builds for OpenerRow, so the two axes can be printed side by side."""
    by_era: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_era[row.era or UNKNOWN_ERA].append(row.outcome)
    result = {era: outcome_metrics(era, outcomes) for era, outcomes in by_era.items()}
    result["ALL"] = outcome_metrics("ALL", [row.outcome for row in rows])
    return result


def outcome_bucket_or_empty(outcome_metrics_map: dict[str, OutcomeMetrics], era: str
                            ) -> OutcomeMetrics:
    """The outcome bucket for ``era``, or an explicit empty one (n=0, rate_blocked=True) when no
    outcome row has ever joined under that exact era -- so --compare can always show an outcomes
    delta for whichever two eras the PRODUCED-side axis is comparing, even when one or both sides
    have zero recorded outcomes so far, rather than raising a KeyError."""
    return outcome_metrics_map.get(era) or outcome_metrics(era, [])


def compare_outcome_eras(a: OutcomeMetrics, b: OutcomeMetrics) -> dict[str, Any]:
    """response_rate delta between two outcome buckets, honoring the same small-n guard as
    OutcomeMetrics itself: if EITHER side's rate is blocked, the delta is also blocked -- this
    never substitutes 0.0 (or any other value) for a rate that was withheld, since a delta built
    from a withheld rate would smuggle the exact number the guard exists to refuse back in."""
    blocked = a.rate_blocked or b.rate_blocked
    delta = None if blocked else (b.response_rate - a.response_rate)
    return {
        "a_n": a.n, "b_n": b.n, "a_counts": dict(a.counts), "b_counts": dict(b.counts),
        "a_response_rate": a.response_rate, "b_response_rate": b.response_rate,
        "delta_response_rate": delta, "rate_blocked": blocked,
        "min_sample_for_rate": MIN_SAMPLE_FOR_OUTCOME_RATE,
    }


# =========================================================================================
# THE PROMPT ERA REGISTRY (tools/backfill_prompt_eras.py's ops/prompt-eras.json) -- turns a
# prompt_sha256 back into something a human can reason about: a short label, a date range, and
# the named ALL CAPS rules that were on the wire in that era. See that tool's own module
# docstring for how the registry is built and why it is git-history-only (never the current
# uncommitted working tree). Everything below is READ ONLY: this module never writes the
# registry, only resolves against it.
# =========================================================================================

DEFAULT_ERAS_FILE = "ops/prompt-eras.json"


@dataclass
class EraRegistry:
    """The era registry, loaded once per run. `eras` maps a full prompt_sha256 to its JSON
    entry (label, date range, commits, rules, ...); `order` preserves the registry file's own
    oldest-first ordering, which --eras and --rule both rely on rather than re-sorting by date
    (a re-sort would silently paper over a registry that was ever hand-edited out of order).
    `loaded` is False, with `note` explaining why, for a missing file or one that fails to
    parse -- never silently treated as "zero known eras" without saying so.

    `working_tree` is tools/backfill_prompt_eras.py's provisional entry for the CURRENT
    uncommitted working tree (its `build_working_tree_entry()`'s output), or None when the
    loaded registry predates that feature or was generated with --no-working-tree. It is
    DELIBERATELY never folded into `eras`/`order`: those two describe what has SHIPPED (a
    committed, immutable history), while `working_tree` describes whatever happens to be on
    disk right now and can change on the next edit -- see format_eras_listing()/
    format_rule_lookup() for how each is presented so a reader can never mistake one for the
    other."""
    path: Path
    loaded: bool
    eras: dict[str, dict[str, Any]]
    order: list[str]
    note: str | None = None
    working_tree: dict[str, Any] | None = None


def load_era_registry(path: Path) -> EraRegistry:
    """Load the era registry at `path`. A missing file or a file that fails to parse as JSON is
    reported via `loaded=False` + `note` (per this tool's BE HONEST convention), never raised --
    every caller in this module treats "no registry" as a degraded-but-usable state (labels fall
    back to the bare digest), since a report run should still be useful before anyone has ever
    run tools/backfill_prompt_eras.py."""
    if not path.exists():
        return EraRegistry(path=path, loaded=False, eras={}, order=[],
                           note=f"no era registry at {path} (run tools/backfill_prompt_eras.py "
                                "to generate one)")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return EraRegistry(path=path, loaded=False, eras={}, order=[],
                           note=f"could not parse {path} as JSON: {exc}")
    eras: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for entry in doc.get("eras", []) if isinstance(doc, dict) else []:
        digest = entry.get("prompt_sha256") if isinstance(entry, dict) else None
        if not digest:
            continue
        eras[digest] = entry
        order.append(digest)
    working_tree = doc.get("working_tree") if isinstance(doc, dict) else None
    if not isinstance(working_tree, dict):
        working_tree = None
    return EraRegistry(path=path, loaded=True, eras=eras, order=order, working_tree=working_tree)


def resolve_era_label(era: str, registry: EraRegistry | None) -> str:
    """The human label for `era` (a prompt_sha256, or the UNKNOWN_ERA/"ALL" pseudo-buckets),
    falling back to the bare hash -- CLEARLY MARKED as such -- when it is not in the registry.
    Never raises: an unresolvable era is exactly the case this exists to report honestly rather
    than paper over.

    Also checks the registry's provisional `working_tree` entry (see EraRegistry.working_tree):
    if a measured era digest happens to be exactly the current uncommitted working tree's (e.g. a
    local test run under it before it was committed), this returns THAT entry's own label --
    which always starts with "UNCOMMITTED WORKING TREE" -- rather than misreporting a real,
    just-not-yet-shipped digest as merely "unregistered".
    """
    if era in (UNKNOWN_ERA, "ALL"):
        return "(not a single stamped era)"
    if registry is None or not registry.loaded:
        return f"{era} (no era registry loaded)"
    entry = registry.eras.get(era)
    if entry is not None:
        return str(entry.get("label", era))
    working_tree = registry.working_tree
    if working_tree is not None and working_tree.get("prompt_sha256") == era:
        return str(working_tree.get("label", era))
    return f"{era} (unregistered digest; run tools/backfill_prompt_eras.py)"


def resolve_compare_token(available: Sequence[str], token: str,
                          registry: EraRegistry | None) -> str:
    """resolve_era(), extended to also accept a human era LABEL from the registry (exact,
    case-insensitive match) -- per the task's requirement that --compare take a label as well as
    a digest or digest prefix. Tried FIRST, since a label match is unambiguous by construction
    (each registered digest carries exactly one label) and a registry miss falls straight
    through to resolve_era()'s existing exact/prefix digest matching, so passing registry=None
    (or an unloaded registry) reproduces resolve_era()'s old behaviour exactly."""
    if registry is not None and registry.loaded:
        lowered = token.lower()
        label_matches = [digest for digest in available
                         if digest in registry.eras
                         and str(registry.eras[digest].get("label", "")).lower() == lowered]
        if len(label_matches) == 1:
            return label_matches[0]
        if len(label_matches) > 1:
            raise ValueError(f"label {token!r} matches multiple registered eras: {label_matches}")
    return resolve_era(available, token)


# =========================================================================================
# THE PRE-REGISTERED CHECK -- ops/OPENER-REDESIGN.md's 2026-09-06 (d) addendum ("three owner
# decisions and a pre registered prediction"). Read that addendum before touching any constant
# below: these are the exact, already-committed thresholds, not this tool's own invention, and
# pre-registration's whole point is that a number written before the data exists cannot be moved
# once it does.
# =========================================================================================

# "Predictions for the first NO GRADING era batch of at least 40 drafts." -- the minimum count of
# DRAFTS (openers rows -- see PreRegisteredCheck.drafts_recorded, and this module's REPLAY CORPUS
# section above for why a draft is NOT the same count as a replay capture) generated under the
# CURRENT prompt era before the prediction below is checkable at all.
PRE_REGISTERED_MIN_DRAFTS = 40

# The four directional predictions from that same addendum, each compared against the CURRENT
# era's pooled METRICS BY PROMPT ERA bucket:
#   - grade_rate FALLS below 5% -- the primary endpoint.
#   - single_sentence_rate RISES above 15% -- NO GRADING's escape hatch (one question as the
#     whole message) is actually being taken.
#   - two_sentence_rate falls below 85% -- the mirror of the above.
#   - trigram_diversity does NOT fall below 80% -- guards against the fix narrowing the space
#     rather than redirecting it.
# No prediction is registered for looks_like_rate or apostrophe_rate (the addendum's own baseline
# for both is era-mixed and therefore not attributable), so this tool makes none either.
PRE_REGISTERED_GRADE_RATE_MAX = 0.05
PRE_REGISTERED_SINGLE_SENTENCE_RATE_MIN = 0.15
PRE_REGISTERED_TWO_SENTENCE_RATE_MAX = 0.85
PRE_REGISTERED_TRIGRAM_DIVERSITY_MIN = 0.80

# "THE FALSIFICATION CRITERION, stated up front. If after at least 40 drafts in the new era
# grade_rate is still at or above 10% AND single_sentence_rate is still below 8%, the
# prohibition-only approach is FALSIFIED and the positive specification restructure named under
# DECISION 3 above ships without further argument." -- a SEPARATE pair of thresholds from the four
# directional predictions above (see build_pre_registered_check()'s own docstring: a metric can
# fail its own directional prediction without the falsification criterion being met -- these are
# two different bars, not one restated).
PRE_REGISTERED_FALSIFY_GRADE_RATE_MIN = 0.10
PRE_REGISTERED_FALSIFY_SINGLE_SENTENCE_RATE_MAX = 0.08


def single_sentence_count(m: EraMetrics) -> int:
    return m.sentence_counts.get(1, 0)


def single_sentence_rate(m: EraMetrics) -> float:
    """single_sentence_rate is not itself a field on EraMetrics (its sibling two_sentence_rate
    is), because the corpus's own historical baseline table names it as a metric in its own right
    (ops/OPENER-REDESIGN.md 2026-09-06 (d): 5.3% baseline, n=95) -- derived here from
    ``sentence_counts`` rather than duplicated as a field computed twice."""
    return (single_sentence_count(m) / m.n) if m.n else 0.0


def compute_current_prompt_era(config_path: str) -> tuple[str | None, str | None]:
    """The prompt_sha256 tools/opener_replay.py's own ``current_prompt_sha256`` would compute
    from ``config_path``'s LIVE ``opener.style`` right now -- the era the PRE-REGISTERED CHECK
    tracks progress against, always the era a fresh replay run would write its rows under, never
    a stale one. Returns ``(era, None)`` on success, or ``(None, error)`` when the config could
    not be read/parsed -- never raises, matching this module's honesty convention (a bad
    ``--config`` is reported, never silently treated as zero progress)."""
    try:
        from operation_love import config as cfg_mod
        from operation_love.opener.opener import prompt_stamp
        cfg = cfg_mod.load(config_path)
        return prompt_stamp(cfg.opener.style), None
    except Exception as exc:  # noqa: BLE001 -- a bad/missing config must not crash the report
        return None, f"{type(exc).__name__}: {exc}"


@dataclass
class PreRegisteredMetricCheck:
    """One of the four directional predictions (see the module-level constants above), evaluated
    against the CURRENT era's measured value."""
    metric: str
    predicted: str
    value: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "predicted": self.predicted, "value": self.value,
                "passed": self.passed}


@dataclass
class PreRegisteredCheck:
    """Progress toward, and (once PRE_REGISTERED_MIN_DRAFTS is reached) the result of,
    ops/OPENER-REDESIGN.md's 2026-09-06 (d) pre-registered NO GRADING prediction.

    ``current_era``/``current_era_error`` come from compute_current_prompt_era() -- ``current_era``
    is None (with ``current_era_error`` explaining why) when it could not be computed at all, e.g.
    an unreadable --config; this is reported honestly, never silently treated as zero progress.

    ``drafts_recorded`` is METRICS BY PROMPT ERA's own ``n`` for ``current_era`` -- the count
    PRE_REGISTERED_MIN_DRAFTS actually applies to: a DRAFT is a generated opener row, never a
    replay CAPTURE (a request input still waiting to be replayed -- see this module's REPLAY
    CORPUS section). ``replay_captures_on_disk`` is that section's own capture count, carried here
    purely so a reader sees both numbers side by side and never conflates them; the two can differ
    in either direction (see that section's header comment for why).
    """
    current_era: str | None
    current_era_error: str | None
    drafts_recorded: int
    # How many of `drafts_recorded` are SYNTHETIC (decision == DECISION_REPLAY), i.e. generated
    # by tools/opener_replay.py rather than captured from a live Training/AUTO run. Carried here
    # so the verdict can DISCLOSE ITS PROVENANCE. A replayed draft is genuine model output under
    # the current prompt, so it is legitimate produced-side evidence and is deliberately NOT
    # excluded -- making the prediction checkable offline is exactly why replay exists. What
    # would be misleading is presenting a replay-derived verdict as a LIVE batch, which
    # verdict_basis() below prevents.
    replay_drafts: int
    replay_captures_on_disk: int
    min_drafts_required: int
    checkable: bool
    shortfall: int
    predictions: list[PreRegisteredMetricCheck] = field(default_factory=list)
    falsified: bool | None = None  # None until checkable -- see build_pre_registered_check()

    def verdict_basis(self) -> str:
        """Where this era's drafts came from: "live", "replay", or "mixed".

        The pre-registration in ops/OPENER-REDESIGN.md 2026-09-06 (d) was written expecting a
        live Training batch. Replayed drafts are real model output under the current prompt and
        so are valid produced-side evidence, but a reader deciding whether to ship the
        positive-specification restructure must be able to see which they are looking at.
        """
        if self.drafts_recorded <= 0 or self.replay_drafts <= 0:
            return "live"
        if self.replay_drafts >= self.drafts_recorded:
            return "replay"
        return "mixed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_era": self.current_era,
            "current_era_error": self.current_era_error,
            "drafts_recorded": self.drafts_recorded,
            "replay_drafts": self.replay_drafts,
            "verdict_basis": self.verdict_basis(),
            "replay_captures_on_disk": self.replay_captures_on_disk,
            "min_drafts_required": self.min_drafts_required,
            "checkable": self.checkable,
            "shortfall": self.shortfall,
            "predictions": [p.to_dict() for p in self.predictions],
            "falsified": self.falsified,
        }


def build_pre_registered_check(era_metrics_map: dict[str, EraMetrics], *,
                               current_era: str | None, current_era_error: str | None,
                               replay_captures_on_disk: int) -> PreRegisteredCheck:
    """Assemble the PreRegisteredCheck for ``current_era`` against ``era_metrics_map``
    (build_era_metrics()'s own output). Below PRE_REGISTERED_MIN_DRAFTS, ``predictions`` stays
    empty and ``falsified`` stays None -- a caller renders a verdict exactly when ``checkable`` is
    True and refuses one otherwise, per this task's own "do not soften it" / "refuse to render a
    verdict" requirement."""
    if current_era is None and current_era_error is None:
        current_era_error = "not computed for this report (no --config was read)"
    _era_present = current_era is not None and current_era in era_metrics_map
    drafts_recorded = era_metrics_map[current_era].n if _era_present else 0
    # EraMetrics already tallies this for the synthetic-era marker; reuse it rather than
    # recounting, so the verdict's provenance and the era axis can never disagree.
    replay_drafts = era_metrics_map[current_era].replay_count if _era_present else 0
    shortfall = max(0, PRE_REGISTERED_MIN_DRAFTS - drafts_recorded)
    checkable = current_era is not None and drafts_recorded >= PRE_REGISTERED_MIN_DRAFTS
    check = PreRegisteredCheck(
        current_era=current_era, current_era_error=current_era_error,
        drafts_recorded=drafts_recorded, replay_drafts=replay_drafts,
        replay_captures_on_disk=replay_captures_on_disk,
        min_drafts_required=PRE_REGISTERED_MIN_DRAFTS, checkable=checkable, shortfall=shortfall)
    if not checkable:
        return check

    m = era_metrics_map[current_era]
    ssr = single_sentence_rate(m)
    check.predictions = [
        PreRegisteredMetricCheck(
            "grade_rate", f"falls below {_pct(PRE_REGISTERED_GRADE_RATE_MAX)}",
            m.grade_rate, m.grade_rate < PRE_REGISTERED_GRADE_RATE_MAX),
        PreRegisteredMetricCheck(
            "single_sentence_rate",
            f"rises above {_pct(PRE_REGISTERED_SINGLE_SENTENCE_RATE_MIN)}",
            ssr, ssr > PRE_REGISTERED_SINGLE_SENTENCE_RATE_MIN),
        PreRegisteredMetricCheck(
            "two_sentence_rate", f"falls below {_pct(PRE_REGISTERED_TWO_SENTENCE_RATE_MAX)}",
            m.two_sentence_rate, m.two_sentence_rate < PRE_REGISTERED_TWO_SENTENCE_RATE_MAX),
        PreRegisteredMetricCheck(
            "trigram_diversity",
            f"does not fall below {_pct(PRE_REGISTERED_TRIGRAM_DIVERSITY_MIN)}",
            m.trigram_diversity, m.trigram_diversity >= PRE_REGISTERED_TRIGRAM_DIVERSITY_MIN),
    ]
    # THE FALSIFICATION CRITERION is its own pair of thresholds, never derived from the four
    # PASS/FAIL predictions above (see the module-level constants' own comment).
    check.falsified = (m.grade_rate >= PRE_REGISTERED_FALSIFY_GRADE_RATE_MIN
                      and ssr < PRE_REGISTERED_FALSIFY_SINGLE_SENTENCE_RATE_MAX)
    return check


def format_pre_registered_check(check: PreRegisteredCheck, *,
                                registry: EraRegistry | None = None) -> str:
    """The PRE-REGISTERED CHECK section: progress toward PRE_REGISTERED_MIN_DRAFTS for the
    current era, and -- ONLY once that threshold is met -- the pre-registered metrics next to
    their predicted values, whether each passed, and the falsification verdict. Below threshold,
    this prints the shortfall and NOTHING else: no PASS/FAIL, no verdict, per this task's own
    "refuse to render a verdict" requirement."""
    lines = [f"=== PRE-REGISTERED CHECK (ops/OPENER-REDESIGN.md 2026-09-06 (d): first NO GRADING "
             f"era batch of at least {check.min_drafts_required} drafts) ==="]
    if check.current_era is None:
        lines.append(f"  could not determine the current prompt era: {check.current_era_error}")
        return "\n".join(lines)
    lines.append(f"  current prompt era (prompt_sha256): {check.current_era} "
                f"[{resolve_era_label(check.current_era, registry)}]")
    lines.append(f"  drafts recorded under this era (openers rows -- the threshold applies "
                f"HERE): {check.drafts_recorded}")
    _basis = check.verdict_basis()
    lines.append(f"    of which SYNTHETIC (tools/opener_replay.py offline replay): "
                f"{check.replay_drafts}  -> basis: {_basis.upper()}")
    if _basis != "live":
        lines.append(
            "    PROVENANCE: this era's drafts are wholly or partly GENERATED by offline replay, "
            "not captured from a live Training or AUTO run. They are real model output under the "
            "current prompt, so they are valid PRODUCED-side evidence and are counted -- making "
            "this prediction checkable without further device time is precisely why replay "
            "exists. They are NOT evidence about live behaviour, and they carry no outcomes.")
    lines.append(f"  replay captures on disk (request inputs, NOT drafts): "
                f"{check.replay_captures_on_disk}")
    if check.drafts_recorded != check.replay_captures_on_disk:
        lines.append(
            f"  NOTE: drafts recorded ({check.drafts_recorded}) and replay captures on disk "
            f"({check.replay_captures_on_disk}) differ -- a single capture can be replayed more "
            "than once under the same era, and a live send needs no capture at all, so these two "
            "counts are not interchangeable; PRE_REGISTERED_MIN_DRAFTS applies to DRAFTS only.")
    if not check.checkable:
        lines.append(
            f"  progress: {check.drafts_recorded}/{check.min_drafts_required} drafts -- "
            f"NOT YET CHECKABLE ({check.shortfall} more draft(s) needed before the pre-registered "
            "prediction can be checked)")
        return "\n".join(lines)
    lines.append(f"  progress: {check.drafts_recorded}/{check.min_drafts_required} drafts -- "
                "THRESHOLD MET; the pre-registered prediction is checkable")
    lines.append("  --- pre-registered metrics vs. predicted direction ---")
    lines.extend(f"    {p.metric:<22}: {_pct(p.value)}  (predicted: {p.predicted})  "
                f"{'PASS' if p.passed else 'FAIL'}"
                for p in check.predictions)
    lines.append(
        f"  falsification criterion (grade_rate >= "
        f"{_pct(PRE_REGISTERED_FALSIFY_GRADE_RATE_MIN)} AND single_sentence_rate < "
        f"{_pct(PRE_REGISTERED_FALSIFY_SINGLE_SENTENCE_RATE_MAX)}): "
        f"{'MET' if check.falsified else 'NOT MET'}")
    _suffix = ("" if _basis == "live"
               else f" [{_basis.upper()}-BASED: {check.replay_drafts}/{check.drafts_recorded} "
                    "draft(s) generated by offline replay, not live capture]")
    if check.falsified:
        lines.append(
            f"  VERDICT: FALSIFIED{_suffix} -- the prohibition-only approach did not hold; the "
            "positive-specification restructure (DECISION 3) ships without further argument.")
    else:
        lines.append(f"  VERDICT: NOT FALSIFIED{_suffix}")
    return "\n".join(lines)


# =========================================================================================
# Output
# =========================================================================================

_CAVEATS = (
    "These are DRAFTS, not sent messages -- nothing here proves any opener reached the device "
    "screen, let alone that Hinge delivered it.",
    "The PRODUCED-side metrics (METRICS BY PROMPT ERA / METRICS BY DECISION) still carry no "
    "reply, match, or other conversational-outcome signal at all -- they only ever answer what "
    "the prompt produced, never whether it worked. METRICS BY OUTCOME below is the one place "
    "this report measures performance, and it comes with its own caveats (this list, items 5-8) "
    "that matter at least as much as the numbers themselves.",
    "The jsonl corpus (data/hinge_debug/<run_id>/actions.jsonl) carries no prompt-era stamp, so "
    "its rows all land in the 'unknown' era bucket and mix however many real prompt eras are "
    "represented locally -- they cannot be attributed to any one rewrite.",
    "The store corpus's survivorship bias is now PARTIAL, not total: the boundary is "
    "OpenerService.discard_opener. A row with decision=='like' is SENT (commit_opener's only "
    "value, and the only thing every store row meant before discard_opener existed). A row "
    "with a non-like decision ('dislike', 'never_sent') is a draft that was generated but "
    "NEVER sent (discard_opener) -- this is NOT retroactive, so a row written before "
    "discard_opener existed remains sent-only, never reclassified. A row with no decision "
    "recorded at all (NULL/empty) is bucketed 'unknown', never assumed sent. See METRICS BY "
    "DECISION below for the sent/not_sent/unknown breakdown this pools into if read alone; the "
    "opener_rejections counts remain the (also store-only, also partial) record of drafts a "
    "GUARD rejected before a human ever saw them, distinct from this decision breakdown.",
    "OUTCOMES (match/reply/no_response/unmatch/unknown) are OWNER-OBSERVED, not automatically "
    "detected: a row exists only because someone looked at a real conversation and recorded it. "
    "This axis is therefore incomplete by construction, and can be biased toward whichever "
    "outcomes were easiest, most memorable, or most rewarding to go note down -- silence in this "
    "axis means 'nobody recorded an outcome for this opener', never 'nothing happened'.",
    "An opener with no derivable profile_key (an unattributed row, empty on either side of the "
    "join) is EXCLUDED from every outcome count and every outcome denominator here, never "
    "counted as a failure or as 'no_response' -- see read_sqlite_opener_outcomes() / "
    "read_bigquery_opener_outcomes()'s join predicate, which drops empty/NULL profile_key on "
    "BOTH sides for exactly this reason.",
    "Synthetic replay rows (tools/opener_replay.py's decision=='synthetic_replay') can never "
    "have earned a real outcome -- they were never sent to a real person -- and are excluded "
    "from every outcome count and denominator here, by an explicit decision check in addition "
    "to the profile_key exclusion above (every replay row is also unattributed by construction "
    "today, but this exclusion does not depend on that staying true).",
    "A higher or lower response rate under one prompt era than another is a CORRELATION, not a "
    "causal claim about whichever single rule changed: two eras almost always differ in more "
    "than one rule at once (see --eras), and this axis has no control for who was swiped on, "
    "when, or how much time each opener has had to accumulate a response.",
    "An era populated in whole or in part by tools/opener_replay.py's offline replay (see the "
    "synthetic-replay row count in SOURCES above, the 'synthetic_replay' decision bucket in "
    "METRICS BY DECISION, and the bracketed synthetic-era marker on METRICS BY PROMPT ERA / "
    "--compare below) is GENERATED, not captured from live use. Its produced-side metrics remain "
    "directly comparable to another era's on the same produced-side axis -- diffing exactly that "
    "is opener_replay.py's entire reason for existing -- but a replayed era is NOT evidence about "
    "live behaviour, and it carries no outcomes by construction: a replay row was never sent to a "
    "real person, so it can never earn one (see the synthetic-replay caveat above, under METRICS "
    "BY OUTCOME).",
)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def format_source_report(report: SourceReport) -> str:
    lines = ["=== SOURCES ==="]
    js = report.jsonl_stats
    lines.append(
        f"  jsonl   : {js.run_dirs_total} run dir(s), {js.run_dirs_with_actions_file} with an "
        f"actions.jsonl, {js.run_dirs_populated} populated (>=1 opener) -> "
        f"{js.opener_occurrences} occurrence(s), {report.jsonl_unique} unique"
        + (f" ({js.malformed_lines} malformed line(s) skipped)" if js.malformed_lines else ""))
    sq = report.sqlite_stats
    lines.append(
        f"  sqlite  : {'available' if sq.available else 'unavailable'}, "
        f"{sq.openers_row_count} openers row(s) -> {report.sqlite_unique} unique "
        f"({sq.replay_row_count} synthetic-replay row(s) included -- see caveats; PRODUCED-side "
        f"only, never OUTCOMES), {sq.rejections_row_count} opener_rejections row(s)")
    lines.extend(f"            NOTE: {note}" for note in sq.notes)
    bq = report.bigquery_stats
    if not bq.attempted:
        lines.append("  bigquery: not queried (pass --bigquery to include it)")
    else:
        lines.append(
            f"  bigquery: {'available' if bq.available else 'unavailable'}, "
            f"{bq.openers_row_count} openers row(s) -> {report.bigquery_unique} unique "
            f"({bq.replay_row_count} synthetic-replay row(s) included -- see caveats; "
            f"PRODUCED-side only, never OUTCOMES), {bq.rejections_row_count} "
            "opener_rejections row(s)")
        lines.extend(f"            NOTE: {note}" for note in bq.notes)
    lines.append(
        f"  combined corpus after cross-source de-duplication: {report.combined_unique} "
        "unique opener(s)")
    sqo = report.sqlite_outcome_stats
    lines.append(
        f"  sqlite opener_outcomes  : {'available' if sqo.available else 'unavailable'}, "
        f"{sqo.joined_row_count} joined outcome row(s) "
        f"({sqo.replay_excluded_count} synthetic-replay row(s) excluded)")
    lines.extend(f"            NOTE: {note}" for note in sqo.notes)
    bqo = report.bigquery_outcome_stats
    if not bqo.attempted:
        lines.append("  bigquery opener_outcomes: not queried (pass --bigquery to include it)")
    else:
        lines.append(
            f"  bigquery opener_outcomes: {'available' if bqo.available else 'unavailable'}, "
            f"{bqo.joined_row_count} joined outcome row(s) "
            f"({bqo.replay_excluded_count} synthetic-replay row(s) excluded)")
        lines.extend(f"            NOTE: {note}" for note in bqo.notes)
    return "\n".join(lines)


def format_era_metrics(m: EraMetrics, *, label: str = "era",
                       registry: EraRegistry | None = None) -> str:
    """Render one EraMetrics bucket. ``label`` names the axis this bucket came from ("era" for
    build_era_metrics()'s prompt_sha256 buckets, "decision" for build_decision_metrics()'s
    sent/not_sent/unknown buckets) -- EraMetrics itself is axis-agnostic (see
    build_decision_metrics()'s docstring), only the header needs to say which axis is which.
    On the "era" axis only, the bucket's registry label (see resolve_era_label) is appended in
    brackets so a prompt_sha256 is never printed with nothing to identify it by, and -- also era
    axis only -- era_synthetic_marker() appends a second, unmistakable bracketed marker when this
    era is wholly or partially populated by tools/opener_replay.py's synthetic replay rows (see
    that function and this module's docstring/caveats): the DECISION axis never needs this same
    marker, since a replay row already sits in its own unmistakably-named synthetic_replay
    bucket there."""
    header = f"--- {label}: {m.era}"
    if label == "era":
        header += f" [{resolve_era_label(m.era, registry)}]"
    header += f" (n={m.n})"
    if label == "era":
        header += era_synthetic_marker(m)
    header += " ---"
    lines = [header]
    if m.n == 0:
        lines.append("  (no openers in this bucket)")
        return "\n".join(lines)
    lines.append(f"  grade_rate            : {_pct(m.grade_rate)} ({m.grade_count}/{m.n})")
    lines.append(f"  two_sentence_rate     : {_pct(m.two_sentence_rate)} "
                 f"({m.two_sentence_count}/{m.n})")
    lines.append(f"  sentence_counts       : {m.sentence_counts}")
    lines.append(f"  question_final_rate   : {_pct(m.question_final_rate)} "
                 f"({m.question_final_count}/{m.n})")
    lines.append(f"  trigram_diversity     : {_pct(m.trigram_diversity)} "
                 f"({m.distinct_trigrams}/{m.total_trigrams} distinct)")
    lines.append(f"  top5_trigram_coverage : {_pct(m.top5_trigram_coverage)} -> "
                 f"{m.top5_trigrams}")
    lines.append(f"  apostrophe_rate       : {_pct(m.apostrophe_rate)} "
                 f"({m.apostrophe_count}/{m.n})")
    lines.append(f"  looks_like_rate       : {_pct(m.looks_like_rate)} "
                 f"({m.looks_like_count}/{m.n})")
    lines.append(f"  mean/median chars     : {m.mean_chars:.1f} / {m.median_chars:.1f}")
    return "\n".join(lines)


def format_rejections(rejections: dict[str, Counter[str]],
                      registry: EraRegistry | None = None) -> str:
    if not rejections:
        return ("=== OPENER REJECTIONS ===\n  no opener_rejections rows available from any "
                "queried store (see the SOURCES notes above for why)")
    lines = ["=== OPENER REJECTIONS (store only; jsonl carries none) ==="]
    for era in sort_era_keys(rejections.keys()):
        counts = rejections[era]
        total = sum(counts.values())
        lines.append(f"  era {era} [{resolve_era_label(era, registry)}] ({total} total):")
        for reason_code, count in counts.most_common():
            lines.append(f"    {reason_code}: {count}")
    return "\n".join(lines)


_COMPARE_FIELDS: tuple[tuple[str, str], ...] = (
    ("n", "n"),
    ("grade_rate", "grade_rate"),
    ("two_sentence_rate", "two_sentence_rate"),
    ("question_final_rate", "question_final_rate"),
    ("trigram_diversity", "trigram_diversity"),
    ("top5_trigram_coverage", "top5_trigram_coverage"),
    ("apostrophe_rate", "apostrophe_rate"),
    ("looks_like_rate", "looks_like_rate"),
    ("mean_chars", "mean_chars"),
    ("median_chars", "median_chars"),
)


def compare_eras(a: EraMetrics, b: EraMetrics) -> list[tuple[str, float, float, float]]:
    """(field, value_a, value_b, delta=b-a) for every field in _COMPARE_FIELDS."""
    out = []
    for label, attr in _COMPARE_FIELDS:
        va, vb = getattr(a, attr), getattr(b, attr)
        out.append((label, va, vb, vb - va))
    return out


def format_compare(era_a: str, era_b: str, a: EraMetrics, b: EraMetrics, *,
                   registry: EraRegistry | None = None) -> str:
    """--compare's side-by-side delta. Both ``a`` and ``b`` are always ERA-axis buckets (--compare
    only ever operates on build_era_metrics()'s output -- see main()/build_report()), so
    era_synthetic_marker() is appended to BOTH headers unconditionally (never gated behind a
    ``label`` the way format_era_metrics() gates it) -- exactly the case this whole marker exists
    for: a reader diffing two eras must never be able to mistake one populated by
    tools/opener_replay.py's offline replay for a live one just because --compare's header alone
    doesn't say so."""
    rows = compare_eras(a, b)
    label_w = max(len(label) for label, *_ in rows)
    header_a, header_b = repr(era_a), repr(era_b)
    if registry is not None:
        header_a += f" [{resolve_era_label(era_a, registry)}]"
        header_b += f" [{resolve_era_label(era_b, registry)}]"
    lines = [f"=== COMPARE: {header_a} (n={a.n}){era_synthetic_marker(a)} vs "
            f"{header_b} (n={b.n}){era_synthetic_marker(b)} ==="]
    for label, va, vb, delta in rows:
        if label in ("n",):
            lines.append(f"  {label.ljust(label_w)} : {va:g} -> {vb:g}  (delta {delta:+g})")
        else:
            lines.append(f"  {label.ljust(label_w)} : {_pct(va)} -> {_pct(vb)}  "
                         f"(delta {delta * 100:+.1f} pts)"
                         if "rate" in label or "coverage" in label or "diversity" in label
                         else f"  {label.ljust(label_w)} : {va:.1f} -> {vb:.1f}  "
                              f"(delta {delta:+.1f})")
    return "\n".join(lines)


def format_outcome_metrics(m: OutcomeMetrics, *, registry: EraRegistry | None = None) -> str:
    """Render one OutcomeMetrics bucket -- the OUTCOMES-axis counterpart of format_era_metrics()
    above, always on the "era" axis (there is no decision-axis equivalent: an outcome is observed
    on a real conversation, never on a draft that was or wasn't sent). Enforces the HARD HONESTY
    GUARD at print time: below MIN_SAMPLE_FOR_OUTCOME_RATE, the raw counts still print in full,
    but response_rate is replaced with an explicit "n too small for a rate" message rather than a
    number -- see OutcomeMetrics.rate_blocked / MIN_SAMPLE_FOR_OUTCOME_RATE."""
    header = f"--- era: {m.era} [{resolve_era_label(m.era, registry)}] (n={m.n}) ---"
    lines = [header]
    if m.n == 0:
        lines.append("  (no owner-observed outcomes joined to this era yet)")
        return "\n".join(lines)
    lines.append(f"  counts by outcome kind : {m.counts}")
    if m.rate_blocked:
        lines.append(
            f"  response_rate          : n too small for a rate (n={m.n} < "
            f"{MIN_SAMPLE_FOR_OUTCOME_RATE} -- see MIN_SAMPLE_FOR_OUTCOME_RATE); the raw counts "
            "above are the only honest number here")
    else:
        lines.append(f"  response_rate          : {_pct(m.response_rate)} "
                     f"({m.response_count}/{m.n})")
    return "\n".join(lines)


def format_compare_outcomes(era_a: str, era_b: str, a: OutcomeMetrics, b: OutcomeMetrics, *,
                            registry: EraRegistry | None = None) -> str:
    """The OUTCOMES-axis counterpart of format_compare() above -- same era tokens, same small-n
    guard as format_outcome_metrics(): a delta is only ever printed when NEITHER side's rate is
    blocked."""
    cmp = compare_outcome_eras(a, b)
    header_a, header_b = repr(era_a), repr(era_b)
    if registry is not None:
        header_a += f" [{resolve_era_label(era_a, registry)}]"
        header_b += f" [{resolve_era_label(era_b, registry)}]"
    lines = [f"=== COMPARE OUTCOMES: {header_a} (n={a.n}) vs {header_b} (n={b.n}) ==="]
    lines.append(f"  counts a : {cmp['a_counts']}")
    lines.append(f"  counts b : {cmp['b_counts']}")
    if cmp["rate_blocked"]:
        lines.append(
            f"  response_rate : n too small for a rate on at least one side (min sample "
            f"{MIN_SAMPLE_FOR_OUTCOME_RATE}) -- see the raw counts above instead")
    else:
        lines.append(f"  response_rate : {_pct(a.response_rate)} -> {_pct(b.response_rate)}  "
                     f"(delta {cmp['delta_response_rate'] * 100:+.1f} pts)")
    return "\n".join(lines)


def build_report(rows: Sequence[OpenerRow], rejections: Sequence[RejectionRow],
                 source_report: SourceReport, *,
                 outcome_rows: Sequence[OutcomeRow] = (),
                 compare: tuple[str, str] | None = None,
                 registry: EraRegistry | None = None,
                 replay_corpus_stats: ReplayCorpusStats | None = None,
                 current_era: str | None = None,
                 current_era_error: str | None = None) -> dict[str, Any]:
    """Assemble the full JSON-serializable report document shared by text and --json output.

    ``outcome_rows``, the OUTCOMES axis (see build_outcome_metrics()), is keyword-only with an
    empty-tuple default so every existing positional/keyword build_report(...) call -- tests
    included -- keeps working unchanged and simply reports zero outcomes everywhere.

    ``replay_corpus_stats``/``current_era``/``current_era_error`` are the REPLAY CORPUS and
    PRE-REGISTERED CHECK sections' own inputs (see read_replay_corpus_stats()/
    compute_current_prompt_era()), all keyword-only with defaults so every existing
    build_report(...) call keeps working unchanged: an omitted ``replay_corpus_stats`` reports an
    empty corpus at DEFAULT_CORPUS_DIR, and an omitted ``current_era`` reports the era as
    uncomputed rather than guessing at zero progress.

    ``registry``, when loaded, resolves every prompt_sha256 this report prints to its human
    label -- rather than duplicating a resolved label onto every by_era/rejections_by_era row,
    this document carries one side table, ``era_labels`` (digest -> label, for every real
    prompt_sha256 digest actually measured in ``by_era``, resolved through resolve_era_label() --
    the SAME function every text-mode formatter already calls for this, so the two outputs can
    never silently diverge). This explicitly includes a digest that is not in the registry's
    committed `eras` at all but happens to match its provisional `working_tree` entry (e.g. real
    corpus rows recorded under the current uncommitted prompt before that edit was ever
    committed) -- resolve_era_label() already knows to check `registry.working_tree` for exactly
    that case, so this side table reports it too, with its own "UNCOMMITTED WORKING TREE" label,
    rather than silently dropping a digest that has real measured data. A digest that is
    genuinely unregistered (no committed entry AND no working-tree match) still gets an entry
    here -- resolve_era_label()'s own "(unregistered digest; ...)" fallback -- never a silent
    omission. The "unknown"/"ALL" pseudo-buckets are deliberately excluded from this table (they
    are not prompt_sha256 digests a consumer would ever look up here)."""
    metrics = build_era_metrics(rows)
    decision_metrics = build_decision_metrics(rows)
    outcome_metrics_map = build_outcome_metrics(outcome_rows)
    rej_by_era = rejections_by_era(rejections)
    era_labels: dict[str, str] = {}
    if registry is not None and registry.loaded:
        # Every real digest this run actually measured (by_era's keys, minus the "unknown"/"ALL"
        # pseudo-buckets), plus the registry's provisional working-tree digest even when it
        # happens not to appear in by_era for this particular run -- see the docstring above.
        # Resolved via resolve_era_label(), never by reading registry.eras directly, so this side
        # table and every text-mode formatter can never disagree about what a digest means.
        digests = {era for era in metrics if era not in (UNKNOWN_ERA, "ALL")}
        working_tree_digest = (registry.working_tree or {}).get("prompt_sha256")
        if working_tree_digest is not None:
            digests.add(working_tree_digest)
        era_labels = {digest: resolve_era_label(digest, registry) for digest in digests}
    replay_stats = replay_corpus_stats or ReplayCorpusStats(corpus_dir=Path(DEFAULT_CORPUS_DIR))
    pre_registered_check = build_pre_registered_check(
        metrics, current_era=current_era, current_era_error=current_era_error,
        replay_captures_on_disk=replay_stats.capture_count)
    doc: dict[str, Any] = {
        "caveats": list(_CAVEATS),
        # The corpus-on-disk and progress-toward-the-pre-registered-check sections, carried
        # through --json unconditionally exactly like every other axis in this document -- see
        # ReplayCorpusStats/PreRegisteredCheck's own docstrings for what each key means.
        "replay_corpus": replay_stats.to_dict(),
        "pre_registered_check": pre_registered_check.to_dict(),
        "sources": {
            "jsonl": vars(source_report.jsonl_stats) | {"unique": source_report.jsonl_unique},
            "sqlite": vars(source_report.sqlite_stats) | {"unique": source_report.sqlite_unique},
            "bigquery": vars(source_report.bigquery_stats)
                       | {"unique": source_report.bigquery_unique},
            "combined_unique": source_report.combined_unique,
            # Outcome-axis source stats, kept in their OWN nested key rather than folded into
            # the sqlite/bigquery entries above -- those describe the openers/opener_rejections
            # read, this describes the SEPARATE opener_outcomes join (see SourceReport).
            "outcomes": {
                "sqlite": vars(source_report.sqlite_outcome_stats),
                "bigquery": vars(source_report.bigquery_outcome_stats),
            },
        },
        "by_era": {era: metrics[era].to_dict() for era in sort_era_keys(metrics.keys())},
        # See this module's docstring's METRICS section and BE HONEST caveat 4: this axis is
        # first class, not a footnote, because it is the only one that answers "sent vs not".
        "by_decision": {bucket: decision_metrics[bucket].to_dict()
                        for bucket in sort_decision_keys(decision_metrics.keys())},
        # The OUTCOMES axis (this task): did the openers written under prompt era X actually
        # PERFORM better -- see build_outcome_metrics()/OutcomeMetrics and BE HONEST caveats 5-8.
        # Carried through --json unconditionally, exactly like by_era/by_decision above, never
        # gated behind a flag.
        "by_outcome": {era: outcome_metrics_map[era].to_dict()
                      for era in sort_era_keys(outcome_metrics_map.keys())},
        "rejections_by_era": {era: dict(rej_by_era[era]) for era in sort_era_keys(rej_by_era.keys())},
        "era_labels": era_labels,
    }
    if compare is not None:
        era_a_token, era_b_token = compare
        available = list(metrics.keys())
        era_a = resolve_compare_token(available, era_a_token, registry)
        era_b = resolve_compare_token(available, era_b_token, registry)
        doc["compare"] = {
            "era_a": era_a, "era_b": era_b,
            "era_a_label": resolve_era_label(era_a, registry),
            "era_b_label": resolve_era_label(era_b, registry),
            # Whether EITHER side being compared is wholly/partially populated by
            # tools/opener_replay.py's synthetic replay rows -- carried through --json exactly
            # like format_compare()'s own header marker is carried through the text report,
            # so a JSON consumer of --compare can never mistake a replayed era for a live one
            # either. See synthetic_status()/era_synthetic_marker() and this module's caveats.
            "era_a_replay_count": metrics[era_a].replay_count,
            "era_b_replay_count": metrics[era_b].replay_count,
            "era_a_synthetic_status": synthetic_status(
                metrics[era_a].n, metrics[era_a].replay_count),
            "era_b_synthetic_status": synthetic_status(
                metrics[era_b].n, metrics[era_b].replay_count),
            "deltas": [
                {"field": label, "a": va, "b": vb, "delta": delta}
                for label, va, vb, delta in compare_eras(metrics[era_a], metrics[era_b])
            ],
            # Same era tokens, carried through to the OUTCOMES axis -- outcome_bucket_or_empty()
            # supplies an explicit empty (n=0, rate_blocked=True) bucket for a side that has no
            # joined outcome rows yet, so this key is always present, never conditionally absent.
            "outcomes": compare_outcome_eras(
                outcome_bucket_or_empty(outcome_metrics_map, era_a),
                outcome_bucket_or_empty(outcome_metrics_map, era_b)),
        }
    return doc


def format_text_report(rows: Sequence[OpenerRow], rejections: Sequence[RejectionRow],
                       source_report: SourceReport, *,
                       outcome_rows: Sequence[OutcomeRow] = (),
                       compare: tuple[str, str] | None = None,
                       registry: EraRegistry | None = None,
                       replay_corpus_stats: ReplayCorpusStats | None = None,
                       current_era: str | None = None,
                       current_era_error: str | None = None) -> str:
    metrics = build_era_metrics(rows)
    decision_metrics = build_decision_metrics(rows)
    outcome_metrics_map = build_outcome_metrics(outcome_rows)
    replay_stats = replay_corpus_stats or ReplayCorpusStats(corpus_dir=Path(DEFAULT_CORPUS_DIR))
    pre_registered_check = build_pre_registered_check(
        metrics, current_era=current_era, current_era_error=current_era_error,
        replay_captures_on_disk=replay_stats.capture_count)
    parts = ["=== CAVEATS (read before trusting any number below) ==="]
    parts.extend(f"  - {caveat}" for caveat in _CAVEATS)
    parts.append("")
    parts.append(format_source_report(source_report))
    if registry is None or not registry.loaded:
        parts.append(f"  NOTE: {registry.note if registry is not None else 'no era registry loaded'} "
                     "-- era labels below fall back to the bare digest")
    parts.append("")
    parts.append(format_replay_corpus_report(replay_stats))
    parts.append("")
    parts.append(format_pre_registered_check(pre_registered_check, registry=registry))
    parts.append("")
    # First class per this module's docstring's METRICS section and caveat 4 -- printed before
    # the by-era breakdown, not after it, and never gated behind --json or --compare.
    parts.append("=== METRICS BY DECISION (sent vs not_sent vs unknown/legacy -- see caveat 4) ===")
    parts.extend(format_era_metrics(decision_metrics[bucket], label="decision")
                 for bucket in sort_decision_keys(decision_metrics.keys()))
    parts.append("")
    parts.append("=== METRICS BY PROMPT ERA (pools every decision together -- see METRICS BY "
                 "DECISION above) ===")
    parts.extend(format_era_metrics(metrics[era], registry=registry)
                 for era in sort_era_keys(metrics.keys()))
    parts.append("")
    # Printed directly after the produced-side era breakdown -- "so a reader can see production
    # and performance side by side" -- and, like METRICS BY DECISION above, unconditional: never
    # gated behind --json or --compare. See BE HONEST caveats 5-8 for what this axis does and
    # does not prove.
    parts.append("=== METRICS BY OUTCOME (owner-observed; PER PROMPT ERA -- see BE HONEST "
                 "caveats 5-8) ===")
    parts.extend(format_outcome_metrics(outcome_metrics_map[era], registry=registry)
                 for era in sort_era_keys(outcome_metrics_map.keys()))
    parts.append("")
    parts.append(format_rejections(rejections_by_era(rejections), registry=registry))
    if compare is not None:
        era_a = resolve_compare_token(list(metrics.keys()), compare[0], registry)
        era_b = resolve_compare_token(list(metrics.keys()), compare[1], registry)
        parts.append("")
        parts.append(format_compare(era_a, era_b, metrics[era_a], metrics[era_b],
                                    registry=registry))
        parts.append("")
        parts.append(format_compare_outcomes(
            era_a, era_b,
            outcome_bucket_or_empty(outcome_metrics_map, era_a),
            outcome_bucket_or_empty(outcome_metrics_map, era_b),
            registry=registry))
    return "\n".join(parts)


# =========================================================================================
# --eras and --rule -- the "have we tried this before" queries the registry exists to answer.
# Both read ONLY the registry (never the corpus sources) for their era/label/rule-list data;
# --rule additionally folds in measured metrics when the corpus sources were also read (see
# main()), since "what happened" is a corpus question, not a registry one.
# =========================================================================================

def _working_tree_listing_lines(working_tree: dict[str, Any] | None) -> list[str]:
    """Render the registry's provisional `working_tree` entry (tools/backfill_prompt_eras.py's
    build_working_tree_entry()) for --eras, in its own clearly headed section so it can never be
    read as one more row of the committed `eras` list above it. Returns [] (nothing to append)
    when the loaded registry has no `working_tree` entry at all (predates the feature, or was
    generated with --no-working-tree) -- format_eras_listing() just extends with this, so an
    empty list is a silent no-op rather than a special case its caller has to handle."""
    if working_tree is None:
        return []
    lines = ["", "=== UNCOMMITTED WORKING TREE (provisional -- NOT a shipped era; current files "
                 "on disk, see tools/backfill_prompt_eras.py) ==="]
    lines.append(f"  label        : {working_tree.get('label')}")
    digest = working_tree.get("prompt_sha256")
    if digest is None:
        lines.append("  digest       : (unevaluable)")
        lines.extend(f"    reason: {reason}"
                     for reason in working_tree.get("reasons") or [])
        return lines
    lines.append(f"  digest       : {digest}")
    matches = working_tree.get("matches_known_committed_era")
    lines.append("  matches known committed era: "
                 f"{matches if matches else '(none -- new, not yet shipped)'}")
    rules = working_tree.get("rules", [])
    lines.append(f"  rules on wire: {len(rules)}")
    lines.append("  metrics      : not tracked by this listing -- run --rule NAME against your "
                 "local corpus to check whether anything has actually been measured under this "
                 "exact digest")
    return lines


def format_eras_listing(registry: EraRegistry) -> str:
    """--eras: every known prompt era, OLDEST FIRST, with its label, date range, short digest,
    and the named rules that changed versus the era immediately before it in the registry's own
    stored order -- recomputed here from each era's stored `rules` list, never parsed back out
    of the (deliberately lossy) label string. The registry's provisional `working_tree` entry
    (see EraRegistry.working_tree), when present, is appended as its own clearly headed section
    AFTER every committed era -- never interleaved into the oldest-first list above, and never
    counted in "N known" -- so a reader can never mistake an unshipped rule for a shipped one.
    """
    if not registry.loaded:
        return f"(no era registry: {registry.note})"
    if not registry.order:
        lines = ["(era registry loaded but contains zero eras)"]
        lines.extend(_working_tree_listing_lines(registry.working_tree))
        return "\n".join(lines)
    lines = [f"=== KNOWN PROMPT ERAS (oldest first, {len(registry.order)} known) ==="]
    previous_rules: list[str] | None = None
    for digest in registry.order:
        entry = registry.eras[digest]
        rules = list(entry.get("rules", []))
        lines.append(f"--- {entry.get('label', '(no label)')} ---")
        lines.append(f"  digest       : {digest}")
        lines.append(f"  date range   : {entry.get('first_date')} -> {entry.get('last_date')}")
        lines.append(f"  commits      : {entry.get('commit_count')} "
                     f"(first {str(entry.get('first_commit'))[:10]}, "
                     f"last {str(entry.get('last_commit'))[:10]})")
        lines.append(f"  rules on wire: {len(rules)}")
        if previous_rules is None:
            lines.append("  vs previous  : (baseline era; nothing earlier in this registry)")
        else:
            prev_set = set(previous_rules)
            cur_set = set(rules)
            added = [r for r in rules if r not in prev_set]
            removed = [r for r in previous_rules if r not in cur_set]
            lines.append(f"  added        : {added if added else '(none)'}")
            lines.append(f"  removed      : {removed if removed else '(none)'}")
        previous_rules = rules
    lines.extend(_working_tree_listing_lines(registry.working_tree))
    return "\n".join(lines)


def _format_rule_metrics_line(digest: str, metrics_by_era: dict[str, EraMetrics] | None) -> str:
    """The one "measured: ..." line shared by every era row (committed or the provisional
    working tree) in format_rule_lookup() -- factored out so the two can never silently report
    "measured" under different conditions. Never claims a rule was measured unless a real,
    non-empty corpus bucket exists under this exact digest (see this module's docstring's BE
    HONEST section) -- the whole reason this helper, and --rule itself, exist."""
    metrics = (metrics_by_era or {}).get(digest)
    if metrics is not None and metrics.n:
        return (f"      measured (DRAFTS only, see caveats): n={metrics.n} "
               f"grade_rate={_pct(metrics.grade_rate)} "
               f"two_sentence_rate={_pct(metrics.two_sentence_rate)} "
               f"question_final_rate={_pct(metrics.question_final_rate)}")
    if metrics_by_era is not None:
        return ("      measured: no local openers recorded under this era digest "
               "(see --debug-dir/--db)")
    return "      measured: not checked (no corpus sources were read)"


def _format_rule_lookup_working_tree(target: str, working_tree: dict[str, Any] | None,
                                     metrics_by_era: dict[str, EraMetrics] | None) -> list[str]:
    """The trailing "UNCOMMITTED WORKING TREE" section of --rule's output: whether `target` is on
    the wire in the CURRENT uncommitted working tree, regardless of whether it ever appeared in
    committed history -- this is what lets --rule give the correct answer for a rule that was
    only just added and never shipped (see this module's docstring and
    tools/backfill_prompt_eras.py's WORKING TREE section for why this can never be answered from
    the committed `eras` list alone). Reuses _format_rule_metrics_line() so a rule present only
    in the working tree gets the exact same measured/no-local-openers/not-checked honesty as one
    long since shipped, never a special-cased "obviously unmeasured" shortcut -- a working tree
    digest CAN already have real corpus rows under it (a local test run before committing).
    """
    lines = ["", "UNCOMMITTED WORKING TREE (current files on disk; not a shipped era):"]
    if working_tree is None:
        lines.append("  not evaluated -- this registry has no 'working_tree' entry (regenerate "
                     "it with tools/backfill_prompt_eras.py to get one)")
        return lines
    digest = working_tree.get("prompt_sha256")
    if digest is None:
        lines.append("  UNEVALUABLE: " + ("; ".join(working_tree.get("reasons") or [])
                                          or "no reason recorded"))
        return lines
    if target in working_tree.get("rules", []):
        lines.append(f"  {target!r} IS present (label: {working_tree.get('label')}) "
                     f"digest={digest}")
        lines.append(_format_rule_metrics_line(digest, metrics_by_era))
    else:
        lines.append(f"  {target!r} is NOT present in the current uncommitted working tree.")
    return lines


def format_rule_lookup(name: str, registry: EraRegistry,
                       metrics_by_era: dict[str, EraMetrics] | None = None) -> str:
    """--rule NAME: "have we tried this before, and what happened" for one named ALL CAPS rule.

    Matched case-insensitively against the registry's exact stored rule spelling (upper-cased
    for comparison; a rule name with internal punctuation, like "HEDGE THE CLAIM, NEVER
    YOURSELF", must still be typed with that punctuation). If the rule is not on the wire in any
    known COMMITTED era, this says so plainly rather than printing an empty section. When it IS
    known, reports its first appearance, its most recent disappearance (or that it is still on
    the wire as of the newest known era), every era it appeared in, and -- when ``metrics_by_era``
    was supplied (main() only reads the corpus when it is available) -- whatever measured metrics
    exist for each of those eras, so a stale "we already tried this" claim can be checked against
    what the prompt actually produced rather than just when it shipped.

    ALWAYS also reports, in a separate trailing section, whether `name` is on the wire in the
    registry's provisional `working_tree` entry (see EraRegistry.working_tree) -- the current
    uncommitted files on disk -- regardless of whether it was ever found in committed history.
    This is what makes "have we tried this before" answer correctly for a rule that exists only
    in an uncommitted edit: without it, a brand new rule with no committed history would be
    reported as "never found on the wire" with nothing to contradict that, even though it is
    sitting on the wire right now.
    """
    target = name.strip().upper()
    lines = [f"=== RULE LOOKUP: {name!r} ==="]
    if not registry.loaded:
        lines.append(f"(no era registry: {registry.note})")
        return "\n".join(lines)
    matches = [digest for digest in registry.order
              if target in registry.eras[digest].get("rules", [])]
    if not matches:
        lines.append(f"{target!r} was never found on the wire in any of the "
                     f"{len(registry.order)} known COMMITTED prompt era(s).")
    else:
        first_entry = registry.eras[matches[0]]
        lines.append(f"first appeared : {first_entry.get('first_date')} "
                     f"({first_entry.get('label')})")
        newest_digest = registry.order[-1]
        last_seen_digest = matches[-1]
        if last_seen_digest == newest_digest:
            lines.append(f"disappeared    : still on the wire as of the newest known era "
                         f"({registry.eras[newest_digest].get('label')})")
        else:
            next_digest = registry.order[registry.order.index(last_seen_digest) + 1]
            next_entry = registry.eras[next_digest]
            last_entry = registry.eras[last_seen_digest]
            lines.append(f"disappeared    : last present in {last_entry.get('label')} "
                         f"(through {last_entry.get('last_date')}); absent starting "
                         f"{next_entry.get('label')} ({next_entry.get('first_date')})")
        lines.append(f"present in {len(matches)}/{len(registry.order)} known era(s):")
        for digest in matches:
            entry = registry.eras[digest]
            lines.append(f"  - {entry.get('label')}  "
                         f"[{entry.get('first_date')} .. {entry.get('last_date')}]  "
                         f"digest={digest}")
            lines.append(_format_rule_metrics_line(digest, metrics_by_era))
    lines.extend(_format_rule_lookup_working_tree(target, registry.working_tree, metrics_by_era))
    return "\n".join(lines)


# =========================================================================================
# CLI
# =========================================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/opener_corpus_report.py",
        description="Measure what the opener prompt actually produces (jsonl + sqlite by "
                    "default, BigQuery only with --bigquery), grouped by prompt_sha256 era.")
    parser.add_argument("--debug-dir", default=DEFAULT_DEBUG_DIR,
                        help=f"data/hinge_debug-style directory to walk (default {DEFAULT_DEBUG_DIR})")
    parser.add_argument("--db", default=DEFAULT_DB_FILE,
                        help=f"SQLite store file to read (default {DEFAULT_DB_FILE})")
    parser.add_argument("--bigquery", action="store_true",
                        help="also read BigQuery's openers/opener_rejections tables using "
                             "storage.bigquery from --config. OFF by default so this tool "
                             "never needs credentials to be useful.")
    parser.add_argument("--config", default="config.yaml",
                        help="config.yaml to read storage.bigquery.{project_id,dataset} from "
                             "when --bigquery is passed (default config.yaml; never modified)")
    parser.add_argument("--json", action="store_true",
                        help="emit one JSON report to stdout instead of the readable text one")
    parser.add_argument("--compare", nargs=2, metavar=("ERA_A", "ERA_B"), default=None,
                        help="print a side-by-side delta of every metric between two eras "
                             "(each may be a full prompt_sha256, a unique prefix of one, a "
                             "registered era's human label, or the literal 'unknown' or 'ALL')")
    parser.add_argument("--eras-file", default=DEFAULT_ERAS_FILE,
                        help="prompt era registry JSON, from tools/backfill_prompt_eras.py "
                             f"(default {DEFAULT_ERAS_FILE}); used to resolve every printed "
                             "prompt_sha256 to a human label, and by --eras/--rule below")
    parser.add_argument("--eras", action="store_true",
                        help="list every known prompt era (oldest first) with its label, date "
                             "range, digest, and the rules that changed versus the previous "
                             "era; reads ONLY --eras-file, never the corpus sources")
    parser.add_argument("--rule", metavar="NAME", default=None,
                        help="'have we tried this before': print every known era NAME was on "
                             "the wire in, when it first/last appeared, and any measured "
                             "metrics for those eras (reads --eras-file and the corpus sources)")
    parser.add_argument("--replay-corpus-dir", default=DEFAULT_CORPUS_DIR,
                        help="operation_love.opener.replay_corpus on-disk corpus directory to "
                             f"report on (default {DEFAULT_CORPUS_DIR}); read for the REPLAY "
                             "CORPUS section and the PRE-REGISTERED CHECK's capture count")
    return parser


def main(argv: list[str] | None = None, *, bigquery_client=None) -> int:
    """CLI entry point. ``bigquery_client`` is the injection seam tests use to exercise
    --bigquery without the google-cloud-bigquery package or network (see read_bigquery_openers).
    """
    args = _build_arg_parser().parse_args(argv)
    registry = load_era_registry(Path(args.eras_file))

    if args.eras:
        if not registry.loaded:
            print(f"ERROR: --eras: {registry.note}", file=sys.stderr)
            return 1
        print(format_eras_listing(registry))
        return 0

    bigquery_project: str | None = None
    bigquery_dataset = "operation_love"
    if args.bigquery:
        try:
            from operation_love import config as cfg_mod
            cfg = cfg_mod.load(args.config)
            bigquery_project = cfg.storage.bigquery.get("project_id") or None
            bigquery_dataset = cfg.storage.bigquery.get("dataset", "operation_love")
        except Exception as exc:  # noqa: BLE001 -- config is informational here, never fatal
            print(f"WARNING: could not read {args.config!r} for BigQuery settings "
                 f"({type(exc).__name__}: {exc}); --bigquery will report why it found nothing.",
                 file=sys.stderr)

    rows, rejections, outcome_rows, source_report = read_all_sources(
        debug_dir=Path(args.debug_dir), db_path=Path(args.db), use_bigquery=args.bigquery,
        bigquery_project=bigquery_project, bigquery_dataset=bigquery_dataset,
        bigquery_client=bigquery_client)

    # REPLAY CORPUS + PRE-REGISTERED CHECK -- read unconditionally (never gated behind --bigquery
    # or any other flag), matching every other axis in this report.
    replay_stats = read_replay_corpus_stats(Path(args.replay_corpus_dir))
    current_era, current_era_error = compute_current_prompt_era(args.config)

    if args.rule is not None:
        if not registry.loaded:
            print(f"ERROR: --rule: {registry.note}", file=sys.stderr)
            return 1
        metrics_by_era = build_era_metrics(rows)
        print(format_rule_lookup(args.rule, registry, metrics_by_era))
        return 0

    compare = tuple(args.compare) if args.compare else None
    if compare is not None:
        try:
            available = list(build_era_metrics(rows).keys())
            resolve_compare_token(available, compare[0], registry)
            resolve_compare_token(available, compare[1], registry)
        except ValueError as exc:
            print(f"ERROR: --compare: {exc}", file=sys.stderr)
            return 1

    if args.json:
        doc = build_report(rows, rejections, source_report, outcome_rows=outcome_rows,
                          compare=compare, registry=registry, replay_corpus_stats=replay_stats,
                          current_era=current_era, current_era_error=current_era_error)
        print(json.dumps(doc, indent=2, sort_keys=False))
        return 0

    print(format_text_report(rows, rejections, source_report, outcome_rows=outcome_rows,
                             compare=compare, registry=registry, replay_corpus_stats=replay_stats,
                             current_era=current_era, current_era_error=current_era_error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
