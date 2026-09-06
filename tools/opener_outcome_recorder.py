"""Owner-facing CLI: record an OBSERVED OUTCOME against an opener that was actually SENT.

WHY THIS EXISTS. `ranker/profile_key.py` and the `opener_outcomes` table (ops/OPENER-REDESIGN.md's
2026-09-06 (i) addendum) close the data-layer half of "what did a sent opener actually PERFORM" --
but nothing writes to that table yet. Collection is deliberately MANUAL FIRST (see that addendum's
DELIBERATELY NOT BUILT section: no automated Hinge match/chat screen reader -- a new screen-reading
surface carries anti-bot exposure this project's own research log is the canonical place to weigh,
and it would introduce attribution errors of its own). This is that manual entry point: the owner
looks at Hinge, sees a match, a reply, or silence, and tells this tool which SENT opener it belongs
to. It never reads a screen, never touches a device, and never calls out to Hinge or any network
endpoint other than the store this project already reads/writes everywhere else.

READ FIRST (this tool's own prior art, per this repo's convention of citing precedent rather than
re-deriving CLI shape from scratch): tools/gemini_model_probe.py and tools/opener_corpus_report.py
for argparse/testability conventions; tools/label_retraction.py for the closest existing precedent
of "select an existing row, then append a narrowly-bound record against it, never silently"; and
ranker/profile_key.py + ranker/__init__.py's `Store.record_opener_outcome`/`joined_opener_outcomes`
for the data layer this tool is the one manual writer of.

THE SELECTION DESIGN, stated because it is the one thing this tool exists to get right. `list`
reads the last `--limit` opener rows for `--app` (ANY decision, most-recent first -- the "recent
openers window"), but only PRINTS the ones that were actually SENT (`decision == "like"`, the only
value `OpenerService.commit_opener` has ever written -- see ranker/store.py's own comment on that
column), each tagged with its own TRUE 1-based position in that same window. `record --index N`
re-reads the identical window (same --app/--limit/--config/--backend/--db must be given) and
resolves N against it via `resolve_selection()`, which is the ONE place this tool decides whether an
outcome may be recorded at all:
  - N outside the window                   -> refused: "no opener at position N ..."
  - N inside the window but NOT sent       -> refused: "... is a DRAFT THAT WAS NEVER SENT ..."
  - N inside the window and sent           -> the row, and only then does this tool touch the store
Both refusals happen BEFORE anything is written -- `record_outcome()` is only ever called with an
already-resolved, already-verified-sent row, so there is no code path in this module that can turn
an unknown or never-sent selection into a written row (see this module's own tests for the
mutation-checked proof: removing the `is_sent` check makes the never-sent case start writing).

WHAT THIS TOOL NEVER DOES: it never picks WHICH opener a match/reply belongs to on the owner's
behalf (that would be exactly the kind of silent substitution
`ops/OPENER-REDESIGN.md`'s "never substitute the liked item" family of rules exists to forbid, one
layer over); it never lets the caller choose `source` (every row this tool writes is stamped
`OPENER_OUTCOME_SOURCE_OWNER` -- "owner-entered" is the one thing this tool's whole existence
guarantees about its own output, so it is not a flag); and it never requires the owner to type,
read, or even see a raw `profile_key` hash -- `record_outcome()` reads it off the resolved row and
passes it to the store directly, exactly the "resolving to that row's stored profile_key rather
than making the owner type a hash" the owner asked this tool to do.

BACKEND. Production storage for this project is BigQuery (config.yaml's `storage.backend`), so
unlike tools/opener_replay.py (which deliberately writes ONLY to local SQLite, because its rows are
synthetic offline calibration data) this tool reads and writes WHATEVER `--config` configures by
default (`--backend auto`), so `list` actually sees the openers a live run produced. `--backend
sqlite --db PATH` and `--backend bigquery` are available to override for testing or a secondary
local database, matching tools/opener_corpus_report.py's own `--db`/`--bigquery` overrides. Every
BigQuery write here is flushed (not merely buffered) before this process exits -- `BigQueryStore`
buffers up to `flush_every` (25) rows before an automatic flush, and a one-shot CLI writing a single
row would otherwise leave it stranded in memory forever.

OUTCOME VOCABULARY. `--outcome` is constrained to `ranker.KNOWN_OPENER_OUTCOMES` (match / reply /
no_response / unmatch / unknown) via argparse `choices` -- a CLI typed by a human benefits from a
typo being caught immediately, even though the underlying store column stays free TEXT against an
unenforced vocabulary (ranker/__init__.py's own documented design: `angle` and `outcome` are both
"suggestions, not restrictions"). This tool narrows ITS OWN input, not the store's contract.

TESTABILITY, exactly gemini_model_probe.py's/opener_replay.py's pattern: `main()` accepts
`cfg`/`store`/`bigquery_client`/`confirm` injection seams so tests never touch a real config.yaml,
a real BigQuery client, real stdin, or (unless a test explicitly wants one) a real database file.
NEVER make a real network call from a test.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from operation_love import config as cfg_mod
from operation_love.ranker import KNOWN_OPENER_OUTCOMES, OPENER_OUTCOME_SOURCE_OWNER, make_store
from tools.opener_corpus_report import DEFAULT_ERAS_FILE, EraRegistry, load_era_registry, resolve_era_label

DEFAULT_APP = "hinge"
# Matches tools/opener_corpus_report.py's own DEFAULT_DB_FILE, so an unconfigured --db lands on
# the same file every other local-store tool in this project defaults to.
DEFAULT_DB_FILE = "data/operation_love.db"
DEFAULT_LIMIT = 25

# The ONE decision value that has ever meant "this opener actually reached the device" --
# `OpenerService.commit_opener`'s own literal (see ranker/store.py's `openers.decision` column
# comment and tools/opener_corpus_report.py's `decision_bucket()`, which draws the identical line).
# Every other value -- "dislike"/"never_sent" (OpenerService.discard_opener), "synthetic_replay"
# (tools/opener_replay.py), NULL/"" (predates decision-tracking) -- means "never sent," and this
# tool must never guess otherwise (see is_sent()).
DECISION_SENT = "like"


@dataclass(frozen=True)
class OpenerRow:
    """One row out of the "recent openers window" this tool reads -- ANY decision, not only sent
    ones (see this module's docstring's THE SELECTION DESIGN section for why the window itself is
    unfiltered while `list`'s DISPLAY is not)."""
    run_id: str
    app: str
    created_at: float
    model: str
    opener: str
    prompt_sha256: str | None
    profile_id: str
    profile_key: str
    profile_name: str          # "" when no Training label ever recorded her name for this card
    decision: str | None       # raw column value; None and "" both mean "no decision recorded"


def is_sent(row: OpenerRow) -> bool:
    """Whether ``row`` was actually sent. See DECISION_SENT above for why this is a single exact
    equality check and never a guess for an empty/NULL/unrecognized decision value."""
    return row.decision == DECISION_SENT


# ---------------------------------------------------------------------------------------
# READ: the recent-openers window, per backend. Neither backend exposes a generic "list recent
# openers" method on the Store protocol (ranker/__init__.py) -- this tool reads the tables
# directly, exactly the way tools/opener_corpus_report.py's read_sqlite_openers/
# read_bigquery_openers already do, rather than adding one more narrowly-scoped method to that
# shared interface for a single manual tool.
# ---------------------------------------------------------------------------------------

def read_recent_openers_sqlite(db_path: Path, app: str, *, limit: int) -> tuple[list[OpenerRow], list[str]]:
    """Every recent `openers` row for `app`, most-recent first, capped at `limit`. Opened
    read-only (SQLite's `mode=ro` URI) so this read path can never create, migrate, or mutate a
    database file that happens not to exist yet or predates a column this tool wants -- exactly
    tools/opener_corpus_report.py's read_sqlite_openers convention, including its "print why,
    never silently degrade" rule for a missing table/column.
    """
    notes: list[str] = []
    if not db_path.exists():
        return [], [f"no database file at {db_path}"]
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError as exc:
        return [], [f"could not open {db_path} read-only: {exc}"]
    try:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "openers" not in tables:
            return [], ["no 'openers' table (pre-redesign schema, or a fresh/empty database)"]
        columns = {row[1] for row in con.execute("PRAGMA table_info(openers)")}
        decision_expr = "o.decision" if "decision" in columns else "NULL"
        if "decision" not in columns:
            notes.append("'openers' table predates the decision column; every row here reads "
                         "as never-sent (see ranker/store.py's decision column comment)")
        profile_key_expr = "o.profile_key" if "profile_key" in columns else "NULL"
        if "profile_key" not in columns:
            notes.append("'openers' table predates the profile_key column; any outcome recorded "
                         "against these rows will be unattributable")
        prompt_expr = "o.prompt_sha256" if "prompt_sha256" in columns else "NULL"
        has_profile_id = "profile_id" in columns
        profile_id_expr = "o.profile_id" if has_profile_id else "NULL"
        has_names = "labels" in tables and "training_label_names" in tables and has_profile_id
        # A correlated scalar subquery (never a JOIN) so a row can contribute at most one name
        # regardless of how many labels might ever share a profile_id -- profile_id is a fresh
        # uuid4 per card (worker.py), so this is defensive rather than expected, but it keeps
        # this read from ever duplicating an opener row the way a plain LEFT JOIN could.
        name_expr = (
            "(SELECT n.profile_name FROM labels l JOIN training_label_names n ON n.label_id = l.id "
            "WHERE l.app = o.app AND l.profile_id = o.profile_id AND o.profile_id IS NOT NULL "
            "AND o.profile_id != '' ORDER BY l.created_at DESC LIMIT 1)"
        ) if has_names else "NULL"
        query = (
            f"SELECT o.run_id, o.app, o.created_at, o.model, o.opener, {prompt_expr}, "
            f"{profile_id_expr}, {profile_key_expr}, {name_expr}, {decision_expr} "
            "FROM openers o WHERE o.app = ? ORDER BY o.created_at DESC LIMIT ?"
        )
        raw = con.execute(query, (app, limit)).fetchall()
    finally:
        con.close()
    rows = [
        OpenerRow(run_id=str(r[0] or ""), app=str(r[1] or ""), created_at=float(r[2]),
                 model=str(r[3] or ""), opener=str(r[4] or ""),
                 prompt_sha256=(str(r[5]) if r[5] else None), profile_id=str(r[6] or ""),
                 profile_key=str(r[7] or ""), profile_name=str(r[8] or ""),
                 decision=(str(r[9]) if r[9] else None))
        for r in raw
    ]
    return rows, notes


def _query_job_config(parameters):
    """A QueryJobConfig binding `parameters` (name, type, value), degrading to a duck-typed
    stand-in when google-cloud-bigquery is not installed.

    WHY THE FALLBACK. `client` is injectable here so tests can run against a fake client with no
    SDK and no network (see read_recent_openers_bigquery's docstring). Building the parameter
    objects from the real SDK defeated that: the import ran even when a caller had injected a
    client, so these tests passed only on a machine that happened to have the package and FAILED
    IN CI, which installs no BigQuery extra. This mirrors the identical fallback in
    operation_love/ranker/bigquery_store.py::_day_start_job_config -- the same problem, already
    solved once in this repo.
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        return SimpleNamespace(query_parameters=[
            SimpleNamespace(name=name, type_=kind, value=value)
            for name, kind, value in parameters
        ])
    return bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter(name, kind, value)
        for name, kind, value in parameters
    ])


def read_recent_openers_bigquery(project_id: str, dataset: str, app: str, *, limit: int,
                                 client=None) -> tuple[list[OpenerRow], list[str]]:
    """BigQuery counterpart of read_recent_openers_sqlite. `client` is injectable exactly like
    tools/opener_corpus_report.py's read_bigquery_openers, so tests never need the
    google-cloud-bigquery package or network access."""
    from operation_love.bigquery_validation import validate_bigquery_identifier

    try:
        project_id = validate_bigquery_identifier(project_id, "project_id")
        dataset = validate_bigquery_identifier(dataset, "dataset")
    except ValueError as exc:
        return [], [f"invalid BigQuery config: {exc}"]

    if client is None:
        try:
            from google.cloud import bigquery
        except ImportError:
            return [], ["google-cloud-bigquery is not installed (install this project's 'bq' "
                       "extra, or pass --backend sqlite)"]
        try:
            client = bigquery.Client(project=project_id)
        except Exception as exc:  # noqa: BLE001 -- a credentials/transport failure is data here
            return [], [f"could not create a BigQuery client: {type(exc).__name__}: {exc}"]

    openers_table = f"{project_id}.{dataset}.openers"
    labels_table = f"{project_id}.{dataset}.labels"
    query = (
        "SELECT o.run_id AS run_id, o.app AS app, o.created_at AS created_at, o.model AS model, "
        "o.opener AS opener, o.prompt_sha256 AS prompt_sha256, o.profile_id AS profile_id, "
        "o.profile_key AS profile_key, "
        f"(SELECT l.profile_name FROM `{labels_table}` l WHERE l.app = o.app "
        "AND l.profile_id = o.profile_id AND o.profile_id IS NOT NULL AND o.profile_id != '' "
        "ORDER BY l.created_at DESC LIMIT 1) AS profile_name, "
        "o.decision AS decision "
        f"FROM `{openers_table}` o WHERE o.app = @app ORDER BY o.created_at DESC LIMIT @row_limit"
    )
    try:
        job = _query_job_config([("app", "STRING", app), ("row_limit", "INT64", limit)])
        result = client.query(query, job_config=job).result()
    except Exception as exc:  # noqa: BLE001 -- missing table/dataset/permissions is data here
        return [], [f"could not read {openers_table}: {type(exc).__name__}: {exc}"]

    rows: list[OpenerRow] = []
    for row in result:
        created_at = row["created_at"]
        created_epoch = (created_at.timestamp() if hasattr(created_at, "timestamp")
                         else float(created_at))
        prompt_sha256 = row["prompt_sha256"]
        decision = row["decision"]
        rows.append(OpenerRow(
            run_id=str(row["run_id"] or ""), app=str(row["app"] or ""), created_at=created_epoch,
            model=str(row["model"] or ""), opener=str(row["opener"] or ""),
            prompt_sha256=(str(prompt_sha256) if prompt_sha256 else None),
            profile_id=str(row["profile_id"] or ""), profile_key=str(row["profile_key"] or ""),
            profile_name=str(row["profile_name"] or ""),
            decision=(str(decision) if decision else None)))
    return rows, []


def _resolved_backend(cfg, backend: str) -> str:
    return cfg.storage.backend if backend == "auto" else backend


def read_recent_openers(cfg, app: str, *, limit: int, db_path: Path | None, backend: str = "auto",
                        bigquery_client=None) -> tuple[list[OpenerRow], list[str]]:
    """Read the recent-openers window from whichever backend `backend` resolves to (default
    "auto" = `cfg.storage.backend`, matching production -- see this module's docstring's BACKEND
    section for why this tool, unlike tools/opener_replay.py, does not default to local SQLite)."""
    resolved = _resolved_backend(cfg, backend)
    if resolved == "bigquery":
        bq_cfg = cfg.storage.bigquery or {}
        project_id = bq_cfg.get("project_id") or ""
        dataset = bq_cfg.get("dataset") or "operation_love"
        if not project_id:
            return [], ["storage.backend resolves to 'bigquery' but no "
                        "storage.bigquery.project_id is configured"]
        return read_recent_openers_bigquery(project_id, dataset, app, limit=limit,
                                            client=bigquery_client)
    if resolved != "sqlite":
        return [], [f"unknown backend {resolved!r} (expected 'sqlite' or 'bigquery')"]
    path = db_path if db_path is not None else Path(cfg.db_file)
    return read_recent_openers_sqlite(path, app, limit=limit)


# ---------------------------------------------------------------------------------------
# SELECTION -- the one place this tool decides whether an outcome may be recorded at all.
# See this module's docstring's THE SELECTION DESIGN section.
# ---------------------------------------------------------------------------------------

class SelectionError(ValueError):
    """Raised by resolve_selection() -- an index outside the recent-openers window, or one that
    refers to a draft that was never sent. Both are refused BEFORE any store write is attempted."""


def list_sent_openers(rows: Sequence[OpenerRow]) -> list[tuple[int, OpenerRow]]:
    """(position, row) for every SENT row in `rows`, preserving each row's TRUE 1-based position
    in the FULL `rows` window -- so the number this prints is exactly the --index `record` needs,
    with no separate renumbering step that could drift out of sync with resolve_selection()."""
    return [(position, row) for position, row in enumerate(rows, start=1) if is_sent(row)]


def resolve_selection(rows: Sequence[OpenerRow], index: int) -> OpenerRow:
    """Resolve a 1-based `index` against the FULL recent-openers window `rows` (any decision --
    the same window `list_sent_openers` filters for display), refusing rather than guessing in
    either failure direction. Never returns a row this tool has not verified was actually sent."""
    if index < 1 or index > len(rows):
        raise SelectionError(
            f"no opener at position {index} -- the current recent-openers window holds "
            f"{len(rows)} row(s) (valid positions: 1..{len(rows)}). Run `list` again with the "
            "same --app/--limit/--config/--backend/--db to refresh, and raise --limit if the "
            "one you want has scrolled out of the window.")
    row = rows[index - 1]
    if not is_sent(row):
        shown = row.decision if row.decision else "(no decision recorded)"
        raise SelectionError(
            f"position {index} is a DRAFT THAT WAS NEVER SENT (decision={shown!r}) -- an "
            "outcome can only be recorded against an opener that actually reached the device. "
            "Run `list` (which only ever prints sent openers) to pick a valid position.")
    return row


def record_outcome(store, row: OpenerRow, *, app: str, outcome: str, note: str = "",
                   observed_at: object | None = None) -> None:
    """Durably record `outcome` against `row`'s stored profile_key, source always
    OPENER_OUTCOME_SOURCE_OWNER. `row` MUST already be a verified-sent row -- this function does
    not re-check `decision` itself; every caller (main() below, and every test) resolves through
    resolve_selection() first, so this stays a single, obvious, one-line store call rather than a
    second copy of the check resolve_selection() already owns. Passes `row.profile_key` through
    even when it is "" -- an unattributable observation is still worth storing (see
    ranker/__init__.py's own Store.record_opener_outcome docstring for why silently dropping it
    would be the worse failure), never a reason for this tool to refuse.
    """
    store.record_opener_outcome(app, row.profile_key, outcome, source=OPENER_OUTCOME_SOURCE_OWNER,
                               note=note, observed_at=observed_at)


# ---------------------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------------------

def _format_when(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M local")


def _preview(opener: str, width: int = 70) -> str:
    text = " ".join(opener.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _era_label(row: OpenerRow, registry: EraRegistry | None) -> str:
    if not row.prompt_sha256:
        return "(unstamped)"
    return resolve_era_label(row.prompt_sha256, registry)


def format_list_text(rows: Sequence[OpenerRow], *, app: str, registry: EraRegistry | None) -> str:
    sent = list_sent_openers(rows)
    lines = [f"=== RECENT SENT OPENERS (app={app!r}) ==="]
    if not sent:
        lines.append(f"  (none of the {len(rows)} most recent opener row(s) for this app were sent)")
        return "\n".join(lines)
    for position, row in sent:
        name = row.profile_name or "(name not recorded)"
        lines.append(f"  [{position}] {_format_when(row.created_at)} -- {name} -- "
                     f"era {_era_label(row, registry)}")
        lines.append(f"        {_preview(row.opener)!r}  (model={row.model})")
    lines.append(f"\n({len(rows)} recent opener row(s) scanned, {len(sent)} sent -- pass "
                 "--index N from the [N] shown above to `record`)")
    return "\n".join(lines)


def _list_json_doc(rows: Sequence[OpenerRow], *, app: str, registry: EraRegistry | None) -> dict:
    sent = list_sent_openers(rows)
    return {
        "app": app,
        "window_size": len(rows),
        "sent_count": len(sent),
        "sent": [
            {"index": position, "created_at": row.created_at, "when": _format_when(row.created_at),
             "profile_name": row.profile_name, "opener": row.opener, "model": row.model,
             "prompt_sha256": row.prompt_sha256, "era_label": _era_label(row, registry),
             "profile_key_present": bool(row.profile_key)}
            for position, row in sent
        ],
    }


def _parse_observed_at(raw: str | None) -> object | None:
    """--observed-at accepts either an epoch number or an ISO-8601 timestamp, matching
    ranker/store.py's own accepted timestamp shapes -- try numeric first (a CLI string like
    "1757000000" is not itself a float), fall through to the raw string otherwise so the store's
    own ISO parser (which requires a timezone-aware string) reports any real parse failure."""
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return raw


# ---------------------------------------------------------------------------------------
# Store construction (write side)
# ---------------------------------------------------------------------------------------

def _build_store(cfg, *, backend: str, db_path: Path | None):
    """The store `record` writes through -- consistent with whatever `read_recent_openers` just
    read from, so `list` and `record` can never disagree about which database is in play. Built
    directly from `--db` when the resolved backend is sqlite and an explicit path was given
    (make_store() only ever reads `cfg.db_file`, which may differ from an explicit --db);
    otherwise this is exactly `ranker.make_store(cfg, ensure=True)`."""
    resolved = _resolved_backend(cfg, backend)
    if resolved == "sqlite" and db_path is not None:
        from operation_love.ranker.store import SQLiteStore
        return SQLiteStore(db_path)
    return make_store(cfg, ensure=True)


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------

def _interactive_confirm(prompt: str) -> bool:
    try:
        reply = input(prompt)
    except EOFError:
        return False
    return reply.strip().lower() in ("y", "yes")


def _common_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--config", default="config.yaml",
                        help="config.yaml to read storage.backend/storage.bigquery/db_file from "
                             "(default config.yaml; never modified)")
    parent.add_argument("--app", default=DEFAULT_APP,
                        help=f"app label to scope both the read and the write to (default "
                             f"{DEFAULT_APP!r})")
    parent.add_argument("--backend", choices=("auto", "sqlite", "bigquery"), default="auto",
                        help="which store to read/write (default 'auto' = --config's "
                             "storage.backend, matching production)")
    parent.add_argument("--db", default=None,
                        help="SQLite file to use when the resolved backend is sqlite (default: "
                             f"--config's paths.db_file, normally {DEFAULT_DB_FILE})")
    parent.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help="how many of the most recent opener rows (ANY decision) make up "
                             "the recent-openers window -- both `list`'s display and `record`'s "
                             f"--index positions are numbered within this window (default "
                             f"{DEFAULT_LIMIT})")
    parent.add_argument("--eras-file", default=DEFAULT_ERAS_FILE,
                        help="prompt era registry (tools/backfill_prompt_eras.py) used to "
                             f"resolve each row's prompt_sha256 to a human label (default "
                             f"{DEFAULT_ERAS_FILE})")
    return parent


def _build_arg_parser() -> argparse.ArgumentParser:
    parent = _common_parent()
    parser = argparse.ArgumentParser(
        prog="python tools/opener_outcome_recorder.py",
        description="Record an OWNER-OBSERVED outcome (match/reply/no_response/unmatch/unknown) "
                    "against a Hinge opener that actually reached the device. Collection is "
                    "deliberately manual first -- see ops/OPENER-REDESIGN.md's 2026-09-06 (i) "
                    "addendum.")
    sub = parser.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", parents=[parent],
                            help="list recent SENT openers and their --index position")
    list_p.add_argument("--json", action="store_true",
                        help="emit one JSON document instead of narrative text")

    record_p = sub.add_parser("record", parents=[parent],
                              help="record an outcome against one sent opener")
    record_p.add_argument("--index", type=int, required=True,
                          help="the [N] position printed by `list` -- pass the SAME "
                               "--app/--limit/--config/--backend/--db so the identical window "
                               "is read again")
    record_p.add_argument("--outcome", required=True, choices=sorted(KNOWN_OPENER_OUTCOMES),
                          help="the documented outcome vocabulary (ranker.KNOWN_OPENER_OUTCOMES)")
    record_p.add_argument("--note", default="", help="optional free-text note")
    record_p.add_argument("--observed-at", default=None,
                          help="when the outcome happened/was noticed -- an epoch number or an "
                               "ISO-8601 timestamp (default: now)")
    record_p.add_argument("--yes", action="store_true",
                          help="skip the interactive confirmation before writing")
    return parser


def main(argv: list[str] | None = None, *, cfg: Any = None, store: Any = None,
         bigquery_client: Any = None, confirm: Callable[[str], bool] | None = None) -> int:
    """CLI entry point. `cfg`/`store`/`bigquery_client`/`confirm` are the injection seams tests
    use to keep this hermetic (no real config.yaml, no real BigQuery client, no real stdin, and --
    when `store` is given -- no store this function constructs or closes itself). Real usage (the
    `__main__` block below) passes none of them."""
    args = _build_arg_parser().parse_args(argv)

    try:
        active_cfg = cfg if cfg is not None else cfg_mod.load(args.config)
    except Exception as exc:  # noqa: BLE001 -- a bad config file must be a clear CLI error
        print(f"ERROR: could not read {args.config!r}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1

    db_path = Path(args.db) if args.db else None
    rows, notes = read_recent_openers(active_cfg, args.app, limit=args.limit, db_path=db_path,
                                      backend=args.backend, bigquery_client=bigquery_client)
    for note in notes:
        print(f"NOTE: {note}", file=sys.stderr)

    registry = load_era_registry(Path(args.eras_file))
    if not registry.loaded:
        print(f"NOTE: {registry.note} -- era labels below fall back to the bare digest",
              file=sys.stderr)

    if args.command == "list":
        if args.json:
            print(json.dumps(_list_json_doc(rows, app=args.app, registry=registry), indent=2))
        else:
            print(format_list_text(rows, app=args.app, registry=registry))
        return 0

    # record
    try:
        row = resolve_selection(rows, args.index)
    except SelectionError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    print(f"About to record outcome={args.outcome!r} against the opener sent "
         f"{_format_when(row.created_at)} to {row.profile_name or '(name not recorded)'}:")
    print(f"  {row.opener!r}  (model={row.model})")
    if not row.profile_key:
        print("  WARNING: this row has no stored profile_key -- the outcome will be recorded "
             "UNATTRIBUTABLE (never dropped, but it cannot join back to this opener in any "
             "report -- see ranker/profile_key.py).")

    if not args.yes:
        ask = confirm or _interactive_confirm
        if not ask("Proceed? [y/N] "):
            print("Aborted -- nothing was recorded.", file=sys.stderr)
            return 1

    active_store = store if store is not None else _build_store(
        active_cfg, backend=args.backend, db_path=db_path)
    try:
        record_outcome(active_store, row, app=args.app, outcome=args.outcome, note=args.note,
                       observed_at=_parse_observed_at(args.observed_at))
        # Flush explicitly rather than relying on an eventual auto-flush: BigQueryStore buffers
        # up to flush_every (25) rows before writing, and a one-shot CLI recording a single
        # outcome would otherwise leave it stranded in memory when this process exits. Never
        # skipped even for an injected test store, so a test can inspect the write immediately
        # after main() returns without needing to know which backend it targeted.
        flush = getattr(active_store, "flush", None)
        if callable(flush):
            flush()
    except Exception as exc:  # noqa: BLE001 -- this tool is the owner's only record of a manual
        # observation, not a fire-and-forget telemetry path (contrast OpenerService's own
        # best-effort writes on the live send path) -- a save failure must be reported loudly.
        print(f"ERROR: could not record the outcome: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    finally:
        if store is None:
            close = getattr(active_store, "close", None)
            if callable(close):
                close()

    print(f"Recorded outcome={args.outcome!r} (source={OPENER_OUTCOME_SOURCE_OWNER!r}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
