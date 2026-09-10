"""Offline replay harness: re-run captured opener requests through the CURRENT prompt.

WHY THIS EXISTS. Until operation_love/opener/replay_corpus.py, the numbered item crops a live
Hinge capture actually sent to Gemini were never persisted anywhere -- so testing a prompt edit
always meant running a fresh live batch on the phone and waiting to see what came back (see that
module's own docstring for the full argument). This tool is the other half: given a corpus of
previously captured requests (operation_love.opener.replay_corpus), it re-generates an opener
for each one through WHATEVER config.yaml currently configures -- never the prompt that was live
when the request was originally captured -- and writes every result to the local store under the
CURRENT prompt_sha256 (operation_love.opener.opener.prompt_stamp), so
tools/opener_corpus_report.py --compare can diff this offline pass's era against a prior live
era. The whole point is to run one live batch, then iterate on the prompt offline forever, with
no phone and no owner time. Unnumbered context crops are retained in the corpus as forensic and
research data, but this tool deliberately omits them from the current Gemini request, exactly as
live opener generation does.

DRY RUN IS THE DEFAULT (see main()'s --live flag) and never depends on GEMINI_API_KEY being set
at all: it builds the exact request payload for each selected capture locally (the same
GeminiOpener._payload() this project's own request path uses -- see gemini_model_probe.py for
the established precedent of a tool reaching into that method directly) and reports its size,
without ever calling the configured transport. WHY DRY RUN IS THE DEFAULT: the API key this
project runs against is a Gemini FREE-TIER key with per-model daily caps as low as 20
requests/day (see config.yaml's opener.models comment) -- a careless replay across a whole
corpus, through however many models the cascade tries per capture, could burn a meaningful
fraction of a day's quota for what is supposed to be a zero-cost offline calibration pass.
Spending real quota therefore requires the explicit --live flag, plus (mirroring
gemini_model_probe.py's own COST TRANSPARENCY gate) either --yes or an interactive "yes" naming
exactly how many billed requests are about to be issued.

STORE: this tool writes ONLY to the local SQLite store (--db, default matching
tools/opener_corpus_report.py's own DEFAULT_DB_FILE), regardless of config.yaml's
storage.backend. Deliberately never BigQuery: a replay run's rows are synthetic offline
calibration data, not production telemetry, and folding them into a shared BigQuery dataset
other tools/dashboards read by default would need its own considered opt-in the way --bigquery
already gates tools/opener_corpus_report.py's reads -- something this tool does not attempt to
provide. Every row this tool writes carries decision=DECISION_REPLAY (see that constant) rather
than a real human decision value, so it can never be counted as a landed "like" by any consumer
that groups the openers table by decision.

TESTABILITY, exactly tools/gemini_model_probe.py's pattern: every network call goes through an
injected GeminiTransport, and main() accepts transport/env/confirm/store injection seams so
tests never touch the real network, the real .env, or a real database file. NEVER make a real
API call from a test -- use a fake transport exactly as tests/test_gemini_opener.py does.

--PURGE: a wholly separate action (see main()'s early dispatch to _run_purge()) for bounding or
clearing the on-disk corpus by hand, independent of OpenerService's own automatic retention bound
(operation_love.opener.replay_corpus's config-driven prune-after-every-capture). Removes the
WHOLE corpus by default, or only captures older than --purge-older-than-days when that flag is
given. DRY RUN IS THE DEFAULT here too (mirrors the replay side's own default): --purge alone only
reports what it would remove; --delete is required to actually touch disk. See _run_purge()'s own
docstring for exactly how each mode is implemented and why age-based purges reuse
operation_love.opener.replay_corpus.prune_replay_corpus directly rather than a second,
independently-written deletion path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from operation_love import config as cfg_mod
from operation_love.opener.opener import (
    GeminiAPIError,
    GeminiCapacityExhausted,
    GeminiOpener,
    GeminiTransport,
    ItemRequest,
    OpenerAborted,
    OpenerError,
    OpenerParseError,
    _stdlib_gemini_transport,
    prompt_stamp,
)
# `prune_replay_corpus`: the SAME safety-audited, oldest-first, symlink-safe deletion helper
# OpenerService's own automatic retention bound uses (see replay_corpus.py's module docstring's
# PRIVACY section) -- reused here, never re-implemented, for --purge's age-based mode (see
# _run_age_based_purge() below). `list_replay_ids`/`load_replay_capture` are that same module's
# read-only discovery API, reused for --purge's whole-corpus mode and for the dry-run preview's
# read side (see _run_purge()'s own docstring for why the two --purge modes are handled
# differently).
from operation_love.opener.replay_corpus import (
    DECISION_REPLAY as _DECISION_REPLAY,
    DEFAULT_CORPUS_DIR,
    delete_replay_captures,
    list_replay_ids,
    load_replay_capture,
    load_replay_corpus,
    prune_replay_corpus,
)
from operation_love.perception.capture import Profile
from operation_love.private_files import load_private_dotenv
from operation_love.ranker.store import SQLiteStore

# Matches tools/opener_corpus_report.py's own DEFAULT_DB_FILE exactly, so a replay run and a
# report run against an unconfigured --db both land on the same file by default.
DEFAULT_DB_FILE = "data/operation_love.db"

# The decision-column marker every row this tool writes carries (ranker/store.py's `openers`
# table `decision` column is free TEXT, grouped by tools/opener_corpus_report.py's
# decision_bucket()). Deliberately a value no real caller has ever written or will ever write:
# OpenerService.commit_opener writes "like" for a landed Like and discard_opener writes
# "dislike"/"never_sent" for a drafted-but-never-sent row -- this value is neither, on purpose,
# so a synthetic replay row can never be miscounted as a real human "sent" decision by any
# consumer that groups on this column. (decision_bucket() today folds any non-"like" value into
# its NOT_SENT bucket alongside real "dislike"/"never_sent" rows -- this marker cannot change
# that grouping, which lives in tools/opener_corpus_report.py, not here; what it guarantees is
# that a replay row can never land in the SENT bucket, and that a raw query can always separate
# replay rows from real ones with `WHERE decision = 'synthetic_replay'`.)
# Re-exported from operation_love.opener.replay_corpus so this CLI and
# tools/opener_corpus_report.py share one spelling; see that constant for why it lives in
# the installed package rather than here.
DECISION_REPLAY = _DECISION_REPLAY

DEFAULT_APP = "hinge"


def _load_dotenv() -> None:
    """Load this project's own .env, exactly like tools/gemini_model_probe.py's own
    _load_dotenv (see that module for the full rationale: no parent-directory walk, symlink
    leaves rejected, tightened to owner-only mode before python-dotenv reads it)."""
    load_private_dotenv(Path.cwd() / ".env")


def _interactive_confirm(prompt: str) -> bool:
    try:
        reply = input(prompt)
    except EOFError:
        return False
    return reply.strip().lower() in ("y", "yes")


def _redact(text: str, secret: str | None) -> str:
    """Never let a real API key reach stdout/stderr, even on an error path."""
    if not text or not secret:
        return text
    return text.replace(secret, "<redacted-api-key>")


def estimate_request_bytes(opener: GeminiOpener, profile: Profile, style: str, model: str,
                          item_request: ItemRequest) -> int:
    """The EXACT local size (UTF-8 bytes of the serialized JSON body) of the request this
    capture would produce against ``model``, computed with zero network I/O by building the
    real request payload (``GeminiOpener._payload`` -- the same private method
    tools/gemini_model_probe.py already calls directly for the same reason: it is this
    project's own proven-correct request builder, so an estimate built any other way could
    drift from what actually gets sent). Never calls ``opener.transport`` -- ``_payload`` does
    not touch it.
    """
    payload = opener._payload(profile, style, model, items=item_request)
    return len(json.dumps(payload).encode("utf-8"))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay captured opener requests (operation_love.opener.replay_corpus) "
                    "through the prompt CURRENTLY configured in config.yaml, so a prompt "
                    "revision can be measured offline against real historical inputs -- no "
                    "phone, no live batch. Defaults to a DRY RUN that spends no API quota; "
                    "pass --live to actually generate and persist openers. The configured "
                    "Gemini key is a FREE-TIER key with per-model daily caps as low as 20 "
                    "requests/day, so spending it on a whole corpus replay is never the "
                    "default.")
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR,
                        help=f"replay corpus directory to read (default {DEFAULT_CORPUS_DIR})")
    parser.add_argument("--config", default="config.yaml",
                        help="config.yaml to read opener.models/opener.style/opener.thinking "
                             "from (default config.yaml; never modified)")
    parser.add_argument("--db", default=DEFAULT_DB_FILE,
                        help=f"local SQLite store to write replay rows to in --live mode "
                             f"(default {DEFAULT_DB_FILE}); never BigQuery, regardless of "
                             "config.yaml's storage.backend -- see this module's docstring")
    parser.add_argument("--app", default=DEFAULT_APP,
                        help=f"app label recorded on every written row (default {DEFAULT_APP})")
    parser.add_argument("--run-id", default=None,
                        help="run_id recorded on every written row (default: a fresh generated "
                             "id, printed at the start of the run)")
    parser.add_argument("--replay-ids", default=None,
                        help="comma-separated list of specific replay_id(s) to use, IN THIS "
                             "ORDER (default: every capture in the corpus, in the corpus's own "
                             "deterministic captured_at/replay_id order). Passing the exact "
                             "same list twice makes two replay runs directly comparable.")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N selected captures (applied AFTER "
                             "--replay-ids selection/ordering, never before)")
    parser.add_argument("--live", action="store_true",
                        help="actually call the Gemini API and persist results (spends real "
                             "free-tier quota). Without this flag, nothing is sent over the "
                             "network and nothing is written to the store.")
    parser.add_argument("--yes", action="store_true",
                        help="skip the interactive confirmation before a --live run")
    parser.add_argument("--json", action="store_true",
                        help="emit a single JSON report on stdout instead of narrative text "
                             "(narrative still goes to stderr)")
    parser.add_argument("--purge", action="store_true",
                        help="remove captures from --corpus-dir instead of replaying anything: "
                             "the WHOLE corpus by default, or only captures older than "
                             "--purge-older-than-days when that flag is also given. DRY RUN BY "
                             "DEFAULT (reports exactly what would be removed; deletes nothing); "
                             "pass --delete to actually remove them. Every other flag above is "
                             "ignored in this mode.")
    parser.add_argument("--purge-older-than-days", type=float, default=None,
                        help="with --purge, remove only captures whose captured_at is older "
                             "than this many days (age-based purge, via "
                             "operation_love.opener.replay_corpus.prune_replay_corpus); omit to "
                             "purge the WHOLE corpus instead")
    parser.add_argument("--delete", action="store_true",
                        help="with --purge, ACTUALLY remove the selected captures. Without "
                             "this flag, --purge only reports what it would remove -- nothing "
                             "on disk is touched.")
    return parser


# ---------------------------------------------------------------------------------------
# --purge: remove captures from the on-disk replay corpus. A completely separate action from
# replaying (see main()'s early dispatch) -- it never touches config.yaml, the Gemini transport,
# or the SQLite store.
# ---------------------------------------------------------------------------------------

PURGE_REASON_WHOLE_CORPUS = "whole_corpus"


def _format_epoch(ts: float | None) -> str:
    if ts is None:
        return "(unknown captured_at)"
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _mirror_manifests_for_dry_run_preview(corpus_dir: Path, replay_ids: list[str],
                                          shadow_root: Path) -> None:
    """Populate ``shadow_root`` with one bare-bones manifest.json per real capture in
    ``replay_ids`` -- ONLY the ``captured_at`` field ``prune_replay_corpus`` actually reads to
    decide what is too old, never the real crop images or her name. Used exclusively so an
    age-based --purge dry run can preview prune_replay_corpus's REAL decision without that
    function ever touching the real corpus (see _run_age_based_purge()'s own docstring for why
    this exists instead of a second, hand-written "which ones are older than N days" check)."""
    for replay_id in replay_ids:
        try:
            capture = load_replay_capture(corpus_dir, replay_id)
        except Exception:  # noqa: BLE001 -- a real capture that fails to load is simply not
            # mirrored/previewed; prune_replay_corpus would also be unable to read its manifest.
            continue
        capture_dir = shadow_root / replay_id
        capture_dir.mkdir(parents=True, exist_ok=True)
        (capture_dir / "manifest.json").write_text(
            json.dumps({"captured_at": capture.captured_at}), encoding="utf-8")


def _run_age_based_purge(corpus_dir: Path, older_than_days: float, *, delete: bool
                         ) -> tuple[bool, list[tuple[str, float | None, str]], str | None]:
    """Age-based purge: (ok, removed[(replay_id, captured_at, reason)], error). ALWAYS goes
    through ``prune_replay_corpus`` -- the exact same call, for real, against ``corpus_dir``
    when ``delete`` is True. When ``delete`` is False (the default), that same function is
    instead run against a THROWAWAY mirror directory holding only a minimal manifest.json per
    real capture (see _mirror_manifests_for_dry_run_preview) so the dry run learns prune_
    replay_corpus's own real answer without ever calling it against -- or deleting anything
    from -- the actual corpus. Never a second, independently-written "is this older than N
    days" check: whichever branch runs, the removal decision is made by that one function.
    """
    if delete:
        result = prune_replay_corpus(corpus_dir, max_age_days=older_than_days)
        if not result.ok:
            return False, [], result.error
        return True, [(r.replay_id, r.captured_at, r.reason) for r in result.removed], None

    try:
        replay_ids = list_replay_ids(corpus_dir)
    except Exception as exc:  # noqa: BLE001 -- a corrupt corpus root must not crash the preview
        return False, [], f"{type(exc).__name__}: {exc}"
    with tempfile.TemporaryDirectory(prefix="opener_replay_purge_preview_") as tmp:
        shadow_root = Path(tmp)
        _mirror_manifests_for_dry_run_preview(corpus_dir, replay_ids, shadow_root)
        result = prune_replay_corpus(shadow_root, max_age_days=older_than_days)
    if not result.ok:
        return False, [], result.error
    return True, [(r.replay_id, r.captured_at, r.reason) for r in result.removed], None


def _run_whole_corpus_purge(corpus_dir: Path, *, delete: bool
                            ) -> tuple[bool, list[tuple[str, float | None, str]], str | None]:
    """Whole-corpus purge: every capture ``list_replay_ids`` finds under ``corpus_dir``,
    unconditionally. Deliberately NOT routed through ``prune_replay_corpus``:
    ``max_captures``/``max_age_days`` are both documented as "0 means unlimited" (see that
    function's own docstring) with no way to express "keep zero" -- forcing "everything" through
    an age cutoff could only reliably catch captures at least an instant older than "now", which
    cannot honestly promise to remove a capture written moments before this command runs. "The
    whole corpus" means exactly what it says instead: every capture this read-only listing finds,
    removed through the package's descriptor-relative capture deletion helper.  The root itself
    is deliberately retained: recursively deleting an arbitrary command-line pathname would also
    delete stray/non-corpus files and would reintroduce a root-path replacement race.
    """
    try:
        replay_ids = list_replay_ids(corpus_dir)
    except Exception as exc:  # noqa: BLE001 -- a corrupt corpus root must not crash the purge
        return False, [], f"{type(exc).__name__}: {exc}"
    removed: list[tuple[str, float | None, str]] = []
    for replay_id in replay_ids:
        captured_at: float | None = None
        try:
            captured_at = load_replay_capture(corpus_dir, replay_id).captured_at
        except Exception:  # noqa: BLE001 -- still reported by id even if its manifest is corrupt
            pass
        removed.append((replay_id, captured_at, PURGE_REASON_WHOLE_CORPUS))
    if delete and corpus_dir.is_dir():
        try:
            actually_removed = set(delete_replay_captures(corpus_dir, replay_ids))
            # Do not claim a concurrently replaced/unsafe entry was removed.
            removed = [entry for entry in removed if entry[0] in actually_removed]
        except Exception as exc:  # noqa: BLE001 -- report, don't traceback, on a delete failure
            return False, removed, f"{type(exc).__name__}: {exc}"
    return True, removed, None


def _run_purge(args: argparse.Namespace) -> int:
    """--purge's own CLI handler: remove the whole replay corpus, or only captures older than
    --purge-older-than-days, from --corpus-dir. DRY RUN BY DEFAULT -- with no --delete, this only
    REPORTS what it would remove; --delete is required to actually touch disk. Either way, what
    was (or would be) removed is printed explicitly, one capture per line -- never just a count.
    See _run_age_based_purge()/_run_whole_corpus_purge() for why the two modes are implemented
    differently, and this module's docstring for why --purge is a wholly separate action from
    replaying (no config.yaml, no transport, no store).
    """
    corpus_dir = Path(args.corpus_dir)
    older_than_days = args.purge_older_than_days
    log: Callable[..., None] = (
        (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print)

    if older_than_days is not None:
        ok, removed, error = _run_age_based_purge(corpus_dir, older_than_days, delete=args.delete)
        scope_desc = f"captures older than {older_than_days} day(s)"
    else:
        ok, removed, error = _run_whole_corpus_purge(corpus_dir, delete=args.delete)
        scope_desc = "the ENTIRE replay corpus"

    if not ok:
        print(f"ERROR: purge of {str(corpus_dir)!r} failed: {error}", file=sys.stderr)
        return 1

    mode = "DELETE" if args.delete else "DRY RUN"
    verb = "removed" if args.delete else "would remove"
    log(f"=== PURGE ({mode}) -- {scope_desc} under {str(corpus_dir)!r} ===")
    if not removed:
        log("  nothing matched -- 0 capture(s) affected.")
    else:
        log(f"  {verb} {len(removed)} capture(s):")
        for replay_id, captured_at, reason in removed:
            log(f"    {replay_id}  captured_at={_format_epoch(captured_at)}  reason={reason}")
    if not args.delete:
        log("\nDRY RUN -- nothing was deleted. Pass --delete to actually remove these.")

    if args.json:
        print(json.dumps({
            "mode": "delete" if args.delete else "dry_run",
            "corpus_dir": str(corpus_dir),
            "purge_older_than_days": older_than_days,
            "removed": [{"replay_id": rid, "captured_at": cat, "reason": reason}
                       for rid, cat, reason in removed],
        }, indent=2))
    return 0


def _parse_replay_ids(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def main(argv: list[str] | None = None, *, transport: GeminiTransport | None = None,
         env: Mapping[str, str] | None = None,
         confirm: Callable[[str], bool] | None = None,
         store: Any | None = None) -> int:
    """CLI entry point. ``transport``/``env``/``confirm``/``store`` are the injection seams
    tests use to keep this hermetic (no network, no real .env, no interactive stdin, no real
    database file) -- see this module's docstring's TESTABILITY paragraph. Real usage (the
    __main__ block below) passes none of them.
    """
    args = _build_arg_parser().parse_args(argv)

    if args.purge:
        # A wholly separate action from replaying -- see _run_purge()'s own docstring. Dispatched
        # before any of the replay-specific setup below (config.yaml, transport, store) so --purge
        # never needs any of it.
        return _run_purge(args)

    log: Callable[..., None] = (
        (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print)

    replay_ids = _parse_replay_ids(args.replay_ids)
    try:
        captures = load_replay_corpus(args.corpus_dir, replay_ids=replay_ids, limit=args.limit)
    except Exception as exc:  # noqa: BLE001 -- a bad --replay-ids entry or a corrupt corpus
        # entry must not traceback; report it and stop.
        print(f"ERROR: could not load the replay corpus from {args.corpus_dir!r}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not captures:
        log(f"No captures found under {args.corpus_dir!r} (nothing selected). Nothing to do.")
        if args.json:
            print(json.dumps({"mode": "live" if args.live else "dry_run", "captures": []},
                             indent=2))
        return 0

    try:
        cfg = cfg_mod.load(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not read {args.config!r}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1

    models = list(cfg.opener.effective_models)
    style = cfg.opener.style
    current_prompt_sha256 = prompt_stamp(style)

    if args.live:
        if env is None:
            _load_dotenv()
            environment: Mapping[str, str] = os.environ
        else:
            environment = env
        api_key = environment.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY is not set (checked the environment and .env). "
                  "Required for --live; a dry run does not need it.", file=sys.stderr)
            return 1
    else:
        # _payload() never reads this value and no network call is made in dry-run mode, so a
        # placeholder is safe here and lets dry runs work with zero credential setup.
        api_key = "dry-run-no-network-placeholder"

    live_transport = transport or _stdlib_gemini_transport
    try:
        gemini = GeminiOpener(models, max_tokens=cfg.opener.max_tokens,
                              request_timeout_s=cfg.opener.request_timeout_s, api_key=api_key,
                              transport=live_transport, thinking=cfg.opener.thinking)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not build a GeminiOpener from {args.config!r}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    profile = Profile()  # her profile TEXT (bio/prompts) is always empty on Hinge, the only
                          # live driver as of this writing -- see replay_corpus.py's module
                          # docstring and opener.opener.GeminiOpener.generate()'s own comment.

    # Context crops remain in every ReplayCapture for forensic inspection and future research,
    # but must never be reintroduced to the model by an offline replay after live generation
    # stopped sending them. The numbered images are the complete model-visible image set.
    item_requests = [
        (capture, ItemRequest(items=capture.items, name=capture.name,
                              truncated=capture.truncated))
        for capture in captures
    ]

    log(f"Corpus: {args.corpus_dir!r} -- {len(captures)} capture(s) selected: "
        f"{[c.replay_id for c in captures]}")
    log(f"Configured model cascade (opener.models): {models}")
    log(f"Current prompt era (prompt_sha256): {current_prompt_sha256}")

    sizing_model = models[0]
    per_capture: list[dict[str, Any]] = []
    for capture, item_request in item_requests:
        entry: dict[str, Any] = {
            "replay_id": capture.replay_id,
            "item_count": item_request.item_count,
            "context_count": len(capture.context),
            "sent_context_count": item_request.context_count,
            "truncated": item_request.truncated,
            "name_present": bool(capture.name),  # never echo her literal name in this tool's
                                                  # output (text or JSON) -- see this comment;
                                                  # the manifest itself still has it if needed.
            "captured_prompt_sha256": capture.prompt_sha256,
        }
        try:
            entry["estimated_request_bytes"] = estimate_request_bytes(
                gemini, profile, style, sizing_model, item_request)
        except Exception as exc:  # noqa: BLE001 -- an oversized/corrupt capture must not abort
            # sizing every other one.
            entry["estimated_request_bytes"] = None
            entry["size_error"] = f"{type(exc).__name__}: {exc}"
        per_capture.append(entry)

    if not args.live:
        log("\n=== DRY RUN (default; no network call, no store write) -- pass --live to "
            "actually spend quota ===")
        for entry in per_capture:
            size = entry["estimated_request_bytes"]
            size_text = f"~{size} byte(s)" if size is not None else f"ERROR: {entry['size_error']}"
            log(f"  {entry['replay_id']}: item_count={entry['item_count']} "
                f"context_count={entry['context_count']} sent_context_count="
                f"{entry['sent_context_count']} truncated={entry['truncated']} "
                f"estimated_request_bytes={size_text} "
                f"(sized against {sizing_model!r})")
        worst_case = len(captures) * len(models)
        log(f"\nWorst case billed request count if --live were used: {len(captures)} "
            f"capture(s) x {len(models)} configured model(s) = {worst_case} (actual is "
            "usually far fewer -- generate() stops at the first model that answers 2xx per "
            "capture).")
        if args.json:
            print(json.dumps({
                "mode": "dry_run",
                "corpus_dir": str(args.corpus_dir),
                "prompt_sha256": current_prompt_sha256,
                "models": models,
                "worst_case_total_requests": worst_case,
                "captures": per_capture,
            }, indent=2))
        return 0

    # --live from here on: real generateContent calls, real store writes.
    worst_case = len(captures) * len(models)
    log("\n=== COST TRANSPARENCY ===")
    log(f"About to replay {len(captures)} capture(s) through the model cascade {models}. "
        f"Worst case total billed request(s): {worst_case} (each capture stops at the first "
        "model that answers 2xx; a capture that hits transient failures on earlier models in "
        "the cascade spends more than one request against that capture alone).")

    if not args.yes:
        ask = confirm or _interactive_confirm
        if not ask("Proceed with these billed generateContent calls? [y/N] "):
            log("Aborted -- no requests were issued.")
            return 1

    run_id = args.run_id or f"opener_replay_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    log(f"run_id: {run_id}")

    active_store = store if store is not None else SQLiteStore(args.db)

    results: list[dict[str, Any]] = []
    for capture, item_request in item_requests:
        outcome: dict[str, Any] = {"replay_id": capture.replay_id}
        try:
            result = gemini.generate(profile, style, items=item_request)
        except OpenerParseError as exc:
            outcome.update(outcome_kind="rejected", reason_code=exc.reason_code,
                          message=_redact(str(exc), api_key))
            log(f"  {capture.replay_id}: REJECTED ({exc.reason_code}): {outcome['message']}")
            try:
                active_store.record_opener_rejection(
                    run_id, args.app, exc.model, 1, exc.reason_code, str(exc), exc.raw_opener,
                    prompt_sha256=current_prompt_sha256)
            except Exception as store_exc:  # noqa: BLE001 -- telemetry must never abort a replay
                log(f"    warning: failed to persist rejection record: {store_exc}")
        except (OpenerAborted, GeminiCapacityExhausted, GeminiAPIError, OpenerError) as exc:
            outcome.update(outcome_kind="error", message=_redact(str(exc), api_key))
            log(f"  {capture.replay_id}: ERROR ({type(exc).__name__}): {outcome['message']}")
        except Exception as exc:  # noqa: BLE001 -- an operator tool must never traceback over
            # one capture.
            outcome.update(outcome_kind="error",
                          message=f"unexpected {type(exc).__name__}: "
                                  f"{_redact(str(exc), api_key)}")
            log(f"  {capture.replay_id}: ERROR (unexpected {type(exc).__name__}): "
                f"{outcome['message']}")
        else:
            outcome.update(outcome_kind="ok", model=result.model, opener=result.opener,
                          item_index=result.item_index)
            log(f"  {capture.replay_id}: ok (model={result.model}, "
                f"item_index={result.item_index}): {result.opener!r}")
            try:
                active_store.record_opener(
                    run_id, args.app, result.model, result.opener, result.referenced,
                    result.angle, result.item_description, profile_id=capture.replay_id,
                    decision=DECISION_REPLAY, decision_source="opener_replay",
                    model_item_index=result.item_index, prompt_sha256=current_prompt_sha256)
            except Exception as store_exc:  # noqa: BLE001 -- telemetry must never abort a replay
                log(f"    warning: failed to persist opener record: {store_exc}")
        results.append(outcome)

    ok_count = sum(1 for r in results if r["outcome_kind"] == "ok")
    log(f"\n=== SUMMARY === {ok_count}/{len(results)} succeeded and were written under "
        f"decision={DECISION_REPLAY!r}, prompt_sha256={current_prompt_sha256}")

    if args.json:
        print(json.dumps({
            "mode": "live",
            "run_id": run_id,
            "prompt_sha256": current_prompt_sha256,
            "models": models,
            "results": results,
        }, indent=2))
    # A partial success (some captures rejected/errored, others fine) still exits 0 -- exactly
    # gemini_model_probe.py's own convention of reporting per-item outcomes without failing the
    # whole process over one bad item. But EVERY capture failing is not partial: nothing this
    # run was asked to do actually happened, so a caller scripting this (or --live in CI against
    # a corpus) can tell "ran and produced nothing" apart from "ran and worked" via the exit code.
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
