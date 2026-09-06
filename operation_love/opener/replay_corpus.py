"""On-disk REPLAY CORPUS format for opener item-crop requests (ops/OPENER-REDESIGN.md 5.2/5.7).

THE PROBLEM THIS EXISTS TO FIX. The numbered item crops actually sent to Gemini
(opener.opener.ItemRequest -- her name as text, the numbered item crops in order, the
unnumbered context crops, and the truncation flag) are not persisted anywhere today.
``data/hinge_debug/<run_id>/`` holds navigation and verification screenshots
(``capture_before``, ``verify_sheet_item``, ``navigate_to_item`` and so on), not the request
INPUTS. So no historical draft can ever be re-run through a revised prompt: every prompt
revision needs a fresh live batch to measure, which is exactly why no prompt change has ever
been measured before/after (see tools/opener_corpus_report.py's own WHY THIS EXISTS section
for the eight prompt-only fixes that shipped with no outcome measurement at all). This module
is the missing persistence layer: a captured request, written once, can be replayed through as
many future prompt revisions as needed with no phone and no owner time.

THE FORMAT. One directory per captured request, named after its own ``replay_id``, holding the
crops as plain image files plus one ``manifest.json``:

    <root>/<replay_id>/
        manifest.json
        item_001.png            (or .jpg / .webp / .bin -- see _guess_extension)
        item_002.jpg
        ...
        context_001.png
        ...

``manifest.json`` carries exactly the fields needed to reconstruct an ``ItemRequest``-equivalent
request, plus bookkeeping, and NOTHING else (see PRIVACY below for why "nothing else" is a
deliberate constraint, not laziness):

    {
      "format_version": 1,
      "replay_id": "<sha256 hex digest, content-derived -- see compute_replay_id>",
      "captured_at": 1757000000.0,           # epoch seconds, wall clock at write time
      "name": "Alex",                        # her name as text ("" if not read -- see
                                              # opener.ItemRequest.name / opener._NAME_UNAVAILABLE)
      "truncated": false,                    # opener.ItemRequest.truncated
      "prompt_sha256": "<hex digest or null>",  # the prompt ERA current at CAPTURE time (see
                                              # opener.prompt_stamp) -- historical metadata only;
                                              # a replay tool re-generating through the CURRENT
                                              # prompt computes its own era stamp and does not
                                              # read this field to build a request
      "items": ["item_001.png", "item_002.jpg"],   # ordered filenames; items[k-1] is item k
      "context": ["context_001.png"]                # ordered filenames, unnumbered tier
    }

Deliberately NOT captured: the style guide text itself (the replay tool always re-generates
through whatever config.yaml currently configures, never the style that was live at capture --
storing it here would invite a replay tool to accidentally use a stale copy instead of the
current one), her profile bio/prompt text (``Profile.text_blob()`` is always empty on Hinge,
the only live driver as of this writing -- see opener.opener.GeminiOpener.generate()'s own
comment: "profile.text_blob(), empty on Hinge but potentially populated by a future browser
driver"), and any run/profile identifier beyond the content-derived ``replay_id`` itself (no
``run_id``, no ``profile_id`` -- see PRIVACY below).

PRIVACY. This corpus is REAL PEOPLE'S PHOTOS -- the exact crops a live Hinge capture sent to a
third-party model. It is LOCAL ONLY BY DEFAULT: this module never uploads, never calls out to
BigQuery or any network endpoint, and never imports anything that could (no
``operation_love.ranker.bigquery_store``, no HTTP client). ``DEFAULT_CORPUS_DIR`` sits under
``data/``, which project-wide ``.gitignore`` already excludes wholesale (see
``data/hinge_debug`` for the existing precedent this format follows). RETENTION is BOUNDED, from
the very first capture, by ``prune_replay_corpus`` (below): it removes captures oldest-first once
their count exceeds a configured ``max_captures`` and/or their age exceeds a configured
``max_age_days`` (0 means unlimited for either -- see that function's own docstring for the full
contract, including its delete-path safety guarantees and its documented handling of a manifest
that fails to parse). The two bounds are owner-tunable via ``opener.replay_corpus_max_captures``
/ ``opener.replay_corpus_max_age_days`` in ``config.yaml``; ``OpenerService`` calls the prune
after every successful capture (see ``OpenerService._capture_replay_corpus``), so the corpus can
never grow unbounded even for a fully-enabled, long-running deployment.

WHAT THIS MODULE DELIBERATELY DOES NOT DO. It has NO dependency on ``operation_love.opener.
opener`` or ``operation_love.opener.service`` -- not even to import ``ItemRequest`` -- and no
dependency on any store or provider client. It is pure, generic I/O over plain values (bytes,
str, bool, float), by design: the live opener path (wired up from
``OpenerService._capture_replay_corpus``, see that method's own docstring) calls the writer with
the exact four fields ``opener.opener.ItemRequest`` already carries (``items``, ``name``,
``context``, ``truncated``) plus its own already-computed ``prompt_sha256``, without this module
ever needing to know what an ``ItemRequest`` is. ``tools/opener_replay.py`` is the one place that
bridges this format back to ``ItemRequest`` for an actual replay.

SAFE TO FAIL. ``write_replay_capture`` never raises -- every exception (a full disk, a
permissions error, a malformed argument) is caught and reported through its return value
instead (see ``ReplayWriteResult``), so a caller on the opener/decision/send/refusal path (this
project's own hard rule: telemetry must be best-effort and must never raise into that path) can
call it unconditionally with no try/except of its own. The READ side (``load_replay_capture``,
``load_replay_corpus``) is the opposite on purpose: a tool reading the corpus back wants to know
loudly about a missing file or a corrupt manifest, not have it silently skipped, so those DO
raise.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Where the corpus lives when a caller doesn't override it -- matches this project's existing
# data/hinge_debug convention (tools/opener_corpus_report.py's DEFAULT_DEBUG_DIR), and, like
# it, sits under data/ which project-wide .gitignore already excludes wholesale.
DEFAULT_CORPUS_DIR = "data/opener_replay_corpus"

_MANIFEST_NAME = "manifest.json"
# Bumped only if the on-disk shape changes in a way a reader must know about (a field renamed
# or reinterpreted, not merely a field added -- an added optional field is forward-compatible
# and does not need a bump, since every reader here uses .get() with a default).
_FORMAT_VERSION = 1

# Every ``replay_id`` this module has ever produced is a SHA-256 hex digest (see
# ``compute_replay_id``): exactly 64 lowercase hex characters, nothing else. ``prune_replay_corpus``
# uses this as its FIRST delete-path safety gate -- a directory name is never even considered as
# a removal candidate unless it matches this shape, which by construction can never contain a
# path separator (``/`` or, on a platform where ``os.sep``/``os.altsep`` differ, either of those),
# a ``.``/``..`` traversal segment, or anything else that could steer a filesystem join outside
# ``root``. See ``_is_safe_capture_id`` for the (redundant, deliberately so) second check applied
# again immediately before any actual deletion.
_REPLAY_ID_RE = re.compile(r"[0-9a-f]{64}")


def _is_safe_capture_id(name: object) -> bool:
    """Whether ``name`` is safe to treat as a ``replay_id`` path component under the corpus
    root -- i.e. it is EXACTLY the shape ``compute_replay_id`` produces and nothing more. Used
    both when discovering prune candidates and, redundantly, a second time immediately before
    ``prune_replay_corpus`` actually deletes anything, so a bug in one call site can never be the
    only thing standing between a bad value and a real ``shutil.rmtree``.
    """
    return isinstance(name, str) and bool(_REPLAY_ID_RE.fullmatch(name))


def _guess_extension(data: bytes) -> str:
    """A human-readable file extension for ``data``'s image format, purely for on-disk
    friendliness (so a directory listing itself hints at what a file is). Never load-bearing --
    every reader here reads bytes off the filename recorded in the manifest, not off the
    extension, so a format this function doesn't recognize still round-trips correctly under
    the generic ".bin" fallback."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    return ".bin"


def compute_replay_id(*, name: str, items: Sequence[bytes], context: Sequence[bytes] = (),
                      truncated: bool = False) -> str:
    """The STABLE identifier for one captured request: a SHA-256 digest of its own content
    (her name, the truncation flag, and every item/context crop's own digest, in order), never
    of wall-clock time or a random value.

    STABLE means two things, both deliberate: (1) writing the exact same capture twice (a
    caller retrying after a transient failure, say) lands in the exact same directory rather
    than creating a duplicate copy of someone's photos on disk, and (2) two independent replay
    runs over "the same set" of captures can refer to entries by this id and know they mean the
    same underlying request -- see tools/opener_replay.py's --replay-ids selection.

    Per-image content is folded in via each image's OWN sha256 digest rather than the raw bytes
    directly, so this function never has to hold two full copies of a multi-megabyte crop in
    memory at once (the image's digest, then this function's own running hash) -- only one.
    """
    hasher = hashlib.sha256()
    hasher.update(str(name or "").strip().encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(b"1" if truncated else b"0")
    for image in items:
        hasher.update(b"\x00")
        hasher.update(hashlib.sha256(bytes(image)).digest())
    hasher.update(b"\x00\x00")  # a fixed double-NUL boundary between the items tier and the
                                # context tier, so an items-heavy capture and a context-heavy
                                # one can never hash the same way merely by shifting crops
                                # across that boundary.
    for image in context:
        hasher.update(b"\x00")
        hasher.update(hashlib.sha256(bytes(image)).digest())
    return hasher.hexdigest()


@dataclass(frozen=True)
class ReplayWriteResult:
    """The outcome of one ``write_replay_capture`` call. Always returned, never raised (see
    this module's docstring's SAFE TO FAIL section) -- a caller checks ``ok`` and moves on
    either way."""
    ok: bool
    replay_id: str | None = None
    path: Path | None = None
    error: str | None = None


def write_replay_capture(root: str | Path, *, items: Sequence[bytes], name: str = "",
                         context: Sequence[bytes] = (), truncated: bool = False,
                         prompt_sha256: str | None = None,
                         captured_at: float | None = None) -> ReplayWriteResult:
    """Write one captured item-crop request to the corpus under ``root``. PURE I/O: this
    function has no dependency on ``operation_love.opener`` or any service internals -- it
    takes the same four fields ``ItemRequest`` carries (``items``, ``name``, ``context``,
    ``truncated``) as plain bytes/str/bool, plus the caller's own already-computed
    ``prompt_sha256`` (this module never computes one itself -- see the module docstring).

    WIRED INTO THE OPENER PATH from ``operation_love.opener.service.OpenerService.
    _capture_replay_corpus``, which calls this once per profile with the exact ``ItemRequest``
    fields it just built, right before making the actual provider call (see that method's own
    docstring). It remains a plain, standalone API otherwise -- a test or a manual script can
    still call it directly with no service involved.

    NEVER RAISES (see this module's docstring's SAFE TO FAIL section) -- every failure, from a
    bad argument to a full disk, comes back as ``ReplayWriteResult(ok=False, error=...)``, so a
    caller on the opener/decision/send/refusal path can call this unconditionally with no
    try/except of its own, exactly as this project's telemetry rule requires.

    IDEMPOTENT BY CONTENT: ``replay_id`` is derived purely from ``name``/``items``/``context``/
    ``truncated`` (see ``compute_replay_id``), so writing the identical capture twice is a
    cheap no-op the second time (detected via the existing directory already having a
    ``manifest.json``) rather than a duplicate copy of someone's photos on disk.

    ATOMICITY, to the extent a plain filesystem allows it without cross-directory renames: the
    manifest is written to a temporary sibling file and ``Path.replace``'d into place LAST, only
    after every crop file has already been written successfully. A crash or exception partway
    through therefore leaves at most an INCOMPLETE directory with no ``manifest.json`` in it --
    every read function in this module treats "no manifest.json" as "not a capture" and skips
    it, so a partial write can never be read back as a corrupt-but-present entry. It can leave
    orphaned crop files on disk with nothing pointing at them, which is an acceptable residual
    for a local-only, best-effort corpus (see this module's docstring's PRIVACY section) rather
    than a correctness problem: no reader will ever surface them.
    """
    try:
        items_t = tuple(bytes(image) for image in items)
        context_t = tuple(bytes(image) for image in context)
        name_s = str(name or "").strip()
        truncated_b = bool(truncated)
        if not items_t:
            raise ValueError(
                "write_replay_capture requires at least one numbered item crop; a request "
                "with none is not something a replay could ever reconstruct meaningfully")
        for position, image in enumerate(items_t + context_t):
            if not image:
                raise ValueError(f"replay capture image at position {position} is empty")

        replay_id = compute_replay_id(name=name_s, items=items_t, context=context_t,
                                      truncated=truncated_b)
        directory = Path(root) / replay_id
        manifest_path = directory / _MANIFEST_NAME
        if manifest_path.is_file():
            # Already captured (same content -> same id -- see compute_replay_id): a no-op
            # success rather than rewriting bytes already on disk.
            return ReplayWriteResult(ok=True, replay_id=replay_id, path=directory)

        directory.mkdir(parents=True, exist_ok=True)

        item_files = []
        for position, image in enumerate(items_t, start=1):
            filename = f"item_{position:03d}{_guess_extension(image)}"
            (directory / filename).write_bytes(image)
            item_files.append(filename)

        context_files = []
        for position, image in enumerate(context_t, start=1):
            filename = f"context_{position:03d}{_guess_extension(image)}"
            (directory / filename).write_bytes(image)
            context_files.append(filename)

        manifest = {
            "format_version": _FORMAT_VERSION,
            "replay_id": replay_id,
            "captured_at": float(captured_at) if captured_at is not None else time.time(),
            "name": name_s,
            "truncated": truncated_b,
            "prompt_sha256": str(prompt_sha256) if prompt_sha256 else None,
            "items": item_files,
            "context": context_files,
        }
        tmp_path = directory / (_MANIFEST_NAME + ".tmp")
        tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(manifest_path)
        return ReplayWriteResult(ok=True, replay_id=replay_id, path=directory)
    except Exception as exc:  # noqa: BLE001 -- see docstring: this function must never raise
        return ReplayWriteResult(ok=False, error=f"{type(exc).__name__}: {exc}")


@dataclass(frozen=True)
class ReplayCapture:
    """One captured request, read back off disk -- the read-side mirror of the four fields
    ``write_replay_capture`` took in, plus the bookkeeping fields every manifest carries.

    ``items``/``context`` are populated (actual crop bytes, in order) rather than left as
    filenames, so a caller (tools/opener_replay.py) can build an equivalent request directly
    without a second pass over the filesystem.
    """
    replay_id: str
    path: Path
    name: str
    items: tuple[bytes, ...]
    context: tuple[bytes, ...]
    truncated: bool
    prompt_sha256: str | None
    captured_at: float | None


def list_replay_ids(root: str | Path) -> list[str]:
    """Every replay_id under ``root`` with a readable, parseable manifest, in a DETERMINISTIC
    default order: ascending ``captured_at`` (a capture made earlier sorts first), tie-broken by
    ``replay_id`` itself so two entries captured in the same wall-clock instant still sort the
    same way on every call. A directory that is not a capture at all (no manifest.json, or a
    manifest.json that fails to parse) is silently skipped rather than raising -- discovery must
    survive a stray file or an in-progress write sitting under the same root; see
    ``load_replay_capture`` for the loud, raising counterpart used once a specific id is chosen.

    Returns ``[]`` for a root that does not exist yet -- an empty corpus, not an error.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    entries: list[tuple[float, str]] = []
    for child in root_path.iterdir():
        if not child.is_dir():
            continue
        manifest_path = child / _MANIFEST_NAME
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        captured_at = manifest.get("captured_at")
        sort_key = float(captured_at) if isinstance(captured_at, (int, float)) else float("inf")
        entries.append((sort_key, child.name))
    entries.sort(key=lambda pair: (pair[0], pair[1]))
    return [replay_id for _sort_key, replay_id in entries]


def load_replay_capture(root: str | Path, replay_id: str) -> ReplayCapture:
    """Read one capture back off disk. RAISES (``FileNotFoundError``, ``json.JSONDecodeError``,
    ``ValueError``) on anything wrong -- a missing directory, a missing/corrupt manifest, a
    manifest naming files that are not there, or an internally inconsistent replay_id -- rather
    than returning a partial result, because a tool consuming a chosen entry needs to know
    immediately rather than silently replay against truncated or wrong data.
    """
    directory = Path(root) / replay_id
    manifest_path = directory / _MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"replay corpus manifest at {manifest_path} is not a JSON object")
    manifest_replay_id = manifest.get("replay_id")
    if manifest_replay_id != replay_id:
        raise ValueError(
            f"replay corpus manifest at {manifest_path} declares replay_id "
            f"{manifest_replay_id!r}, which does not match its own directory name {replay_id!r}")

    item_files = manifest.get("items")
    if not isinstance(item_files, list) or not item_files:
        raise ValueError(
            f"replay corpus manifest at {manifest_path} has no numbered items")
    context_files = manifest.get("context")
    if context_files is None:
        context_files = []
    if not isinstance(context_files, list):
        raise ValueError(
            f"replay corpus manifest at {manifest_path} has a non-list 'context' field")

    items = tuple((directory / str(fname)).read_bytes() for fname in item_files)
    context = tuple((directory / str(fname)).read_bytes() for fname in context_files)
    captured_at = manifest.get("captured_at")
    prompt_sha256 = manifest.get("prompt_sha256")
    return ReplayCapture(
        replay_id=replay_id,
        path=directory,
        name=str(manifest.get("name") or ""),
        items=items,
        context=context,
        truncated=bool(manifest.get("truncated", False)),
        prompt_sha256=str(prompt_sha256) if prompt_sha256 else None,
        captured_at=float(captured_at) if isinstance(captured_at, (int, float)) else None,
    )


def load_replay_corpus(root: str | Path, *, replay_ids: Sequence[str] | None = None,
                       limit: int | None = None) -> list[ReplayCapture]:
    """Load a whole selection from the corpus, in order.

    ``replay_ids``, when given, is BOTH the subset AND the exact order -- passing the same list
    twice (e.g. across two prompt-revision replay runs meant to be compared against each other)
    guarantees the same captures in the same order both times, which ``list_replay_ids``'s own
    default (captured_at, replay_id) ordering also guarantees on its own but a caller may still
    want to pin explicitly. When omitted, defaults to every id ``list_replay_ids`` finds under
    ``root``, in its deterministic order.

    ``limit``, when given, keeps only the first ``limit`` entries of whichever ordering was
    already chosen above -- applied AFTER subset/order selection, never before, so
    ``replay_ids=[...]`` plus ``limit=5`` means "the first 5 of exactly this list", not
    "5 arbitrary ids from this list".
    """
    ids = list(replay_ids) if replay_ids is not None else list_replay_ids(root)
    if limit is not None:
        ids = ids[:limit]
    return [load_replay_capture(root, replay_id) for replay_id in ids]


@dataclass(frozen=True)
class PruneRemoval:
    """One capture ``prune_replay_corpus`` actually removed -- enough to log without a second
    filesystem read (the directory is already gone by the time a caller sees this)."""
    replay_id: str
    captured_at: float | None
    reason: str   # PRUNE_REASON_MAX_CAPTURES or PRUNE_REASON_MAX_AGE_DAYS


PRUNE_REASON_MAX_CAPTURES = "max_captures"
PRUNE_REASON_MAX_AGE_DAYS = "max_age_days"


@dataclass(frozen=True)
class PruneResult:
    """The outcome of one ``prune_replay_corpus`` call. Always returned, never raised (see that
    function's own SAFE TO FAIL section) -- a caller checks ``ok`` and moves on either way,
    exactly like ``ReplayWriteResult``."""
    ok: bool
    removed: tuple[PruneRemoval, ...] = ()
    kept: int = 0
    skipped_unparseable: int = 0
    error: str | None = None


def prune_replay_corpus(root: str | Path, *, max_captures: int = 0,
                        max_age_days: float = 0) -> PruneResult:
    """Bound the corpus under ``root`` by deleting whole capture directories, OLDEST-FIRST by
    the manifest's own ``captured_at``, until it satisfies both ``max_captures`` (an absolute
    cap on how many captures may exist) and ``max_age_days`` (no surviving capture is older than
    this many days). **0 MEANS UNLIMITED for either bound** -- ``max_captures=0`` never removes
    anything for being over-count, ``max_age_days=0`` never removes anything for being too old,
    and ``prune_replay_corpus(root)`` with both left at their defaults is a documented no-op.

    ORDER OF OPERATIONS: age-based removal is applied FIRST, then count-based removal is applied
    to whatever survives it. This is deliberate, not incidental -- it means the two bounds
    compose the way an operator would expect ("nothing older than N days, and no more than M
    captures even within that window") rather than either bound alone deciding survivors
    independently of the other. A capture matches at most one reason in the returned
    ``PruneResult.removed`` (age-removed captures are excluded from the count-removal pool by
    construction, so the two groups can never overlap).

    A CORRUPT OR UNREADABLE MANIFEST IS A DELIBERATE, DOCUMENTED "KEEP", NEVER A DELETE. A
    directory under ``root`` whose ``manifest.json`` is missing, unreadable, not valid JSON, or
    not a JSON object is treated EXACTLY the way ``list_replay_ids`` already treats it: not a
    (complete) capture at all. This function therefore never counts it toward ``max_captures``,
    never compares it against ``max_age_days``, and never deletes it -- it is silently excluded
    from every decision this function makes, the same way it is silently excluded from every
    listing ``list_replay_ids`` produces. Three reasons this is the right default, not merely the
    easy one: (1) this function's own delete-path safety mandate is to make it impossible to
    remove the wrong thing, and a directory whose manifest cannot even be parsed is one this
    function cannot verify is genuinely a spent, replaceable capture rather than, say, a
    mid-external-copy directory or hand-edited evidence -- guessing wrong in the "prunable"
    direction is irreversible, guessing wrong in the "keep" direction only costs disk; (2) reader
    and pruner sharing the exact same definition of "what counts as a capture" means a tool built
    against ``list_replay_ids`` can never see a directory that ``prune_replay_corpus`` silently
    disagreed with it about; (3) ``write_replay_capture``'s own atomicity guarantee (the manifest
    is written LAST, via an atomic replace, only after every crop file already landed) means a
    present-but-corrupt ``manifest.json`` can only happen from something OUTSIDE this module
    entirely -- external disk corruption or manual tampering -- which is rare enough that an
    unbounded-but-rare residual is an acceptable, explicitly accepted risk (the same shape of
    trade-off ``write_replay_capture`` itself already documents for orphaned crop files after a
    crash). ``PruneResult.skipped_unparseable`` counts these so the residual is at least visible
    to a caller who logs it, even though it is never acted on.

    A capture whose manifest DOES parse but omits/mistypes ``captured_at`` (unusual -- every
    manifest this module itself writes always sets it via ``time.time()``) is handled
    differently, on purpose: it is still a candidate (a parseable, well-formed capture), but with
    an unknown age. It sorts as `+inf` for ordering, MATCHING ``list_replay_ids``'s own
    ``captured_at`` tie-break convention exactly, which has two effects: it is NEVER removed for
    age (an unknown age can never be compared against a cutoff), and it is the LAST candidate
    ever removed for count (oldest-first ordering means "sorts newest" removes last), so this
    function never preferentially punishes a capture merely because its age cannot be determined.

    DELETE-PATH SAFETY, the highest-risk part of this function's contract: it must be impossible
    for this function to delete anything outside ``root``, or anything a caller did not mean for
    it to manage. Every one of these must hold before a single ``shutil.rmtree`` call is made for
    a given candidate:

      * ``root`` itself is resolved once, up front (``Path(root).resolve()``), and every later
        comparison is against that resolved path, never the original possibly-relative one;
      * the candidate is an IMMEDIATE child of ``root`` discovered via ``root.iterdir()`` --
        never a name built from string concatenation or user input;
      * the candidate's directory NAME must independently match ``_is_safe_capture_id`` (exactly
        64 lowercase hex characters -- the shape ``compute_replay_id`` produces and the ONLY
        shape this module has ever written to disk), checked once at discovery and AGAIN,
        redundantly, immediately before deletion;
      * the candidate must not be a symlink (``Path.is_symlink()``) -- checked at discovery, so a
        symlink is never even parsed as a capture, let alone considered for removal; this module
        never follows a symlink to decide what to delete;
      * immediately before deletion, the candidate is resolved again and its resolved path must
        (a) be a direct child of resolved ``root`` (``resolved.parent == root`` and
        ``resolved.name`` equal to the id being removed) AND (b) satisfy
        ``resolved.is_relative_to(root)`` -- belt-and-suspenders confirmation that nothing in
        between discovery and deletion (nor any symlink component this function did not
        anticipate) could have moved the real target outside ``root``.

    Every one of the checks above is a `continue`, never a raise: a candidate that fails ANY of
    them is simply left alone (kept), and the loop moves on to the next one. See
    tests/test_replay_corpus_prune.py for a test that attempts each escape explicitly and proves
    it is refused rather than merely "not currently exploited".

    NEVER RAISES (same SAFE TO FAIL contract as ``write_replay_capture``): the ENTIRE body,
    including argument validation and the directory walk itself, runs inside one try/except, so
    a bad argument, a permissions error on ``root``, a full disk, or any other unexpected failure
    comes back as ``PruneResult(ok=False, error=...)`` rather than propagating -- a caller on the
    opener/decision/send/refusal path (this project's own hard rule: telemetry must be
    best-effort and must never raise into that path) can call this unconditionally with no
    try/except of its own, exactly like ``write_replay_capture``. A per-directory deletion
    failure (e.g. one directory becomes unwritable mid-run) is handled the same way one level
    down: that single candidate is skipped (left in place, NOT counted as removed) and every
    other candidate is still attempted -- one bad directory can never abort the whole prune.
    """
    try:
        if isinstance(max_captures, bool) or not isinstance(max_captures, int) or max_captures < 0:
            raise ValueError(
                "prune_replay_corpus max_captures must be a non-negative integer "
                f"(0 = unlimited), got {max_captures!r}")
        if (isinstance(max_age_days, bool) or not isinstance(max_age_days, (int, float))
                or max_age_days < 0 or max_age_days != max_age_days  # NaN check, no math import
                or max_age_days in (float("inf"), float("-inf"))):
            raise ValueError(
                "prune_replay_corpus max_age_days must be a non-negative, finite number of days "
                f"(0 = unlimited), got {max_age_days!r}")

        root_path = Path(root).resolve()
        if not root_path.is_dir():
            # An empty/nonexistent corpus is nothing to prune, not an error -- matches
            # list_replay_ids's own "[] for a root that does not exist yet" contract.
            return PruneResult(ok=True)

        candidates: list[tuple[str, float | None]] = []
        skipped_unparseable = 0
        for child in root_path.iterdir():
            if child.is_symlink():
                # Never follow a symlink to decide what is (or is deleted as) a capture.
                continue
            if not child.is_dir():
                continue
            name = child.name
            if not _is_safe_capture_id(name):
                # Not shaped like anything this module has ever written; not ours to manage.
                continue
            manifest_path = child / _MANIFEST_NAME
            if not manifest_path.is_file():
                # No manifest -- matches list_replay_ids: not a (complete) capture, never touched.
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict):
                    raise ValueError("replay corpus manifest is not a JSON object")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                # DECISION: corrupt/unreadable manifest -> KEEP, never delete. See this
                # function's own docstring for the full three-part justification.
                skipped_unparseable += 1
                continue
            captured_at = manifest.get("captured_at")
            captured_at_f = (float(captured_at)
                             if isinstance(captured_at, (int, float)) else None)
            candidates.append((name, captured_at_f))

        # Oldest-first: ascending captured_at, missing treated as +inf (sorts last / "newest"),
        # tie-broken by id -- the exact same convention list_replay_ids uses, so this function's
        # notion of "capture order" never diverges from what a reader sees.
        candidates.sort(key=lambda pair: (pair[1] if pair[1] is not None else float("inf"),
                                          pair[0]))

        age_ids: set[str] = set()
        if max_age_days > 0:
            cutoff = time.time() - (max_age_days * 86400.0)
            for cid, cat in candidates:
                if cat is not None and cat < cutoff:
                    age_ids.add(cid)

        remaining = [pair for pair in candidates if pair[0] not in age_ids]
        count_ids: set[str] = set()
        if max_captures > 0 and len(remaining) > max_captures:
            excess = len(remaining) - max_captures
            for cid, _cat in remaining[:excess]:
                count_ids.add(cid)

        # A single ascending pass over `candidates` keeps the removal list itself oldest-first,
        # tagging each with whichever bound actually removed it (the two id sets are disjoint by
        # construction: count_ids is only ever drawn from `remaining`, which already excludes
        # every id in age_ids).
        removal_plan = []
        for cid, cat in candidates:
            if cid in age_ids:
                removal_plan.append((cid, cat, PRUNE_REASON_MAX_AGE_DAYS))
            elif cid in count_ids:
                removal_plan.append((cid, cat, PRUNE_REASON_MAX_CAPTURES))

        removed: list[PruneRemoval] = []
        for cid, cat, reason in removal_plan:
            # Redundant re-check immediately before deletion (see docstring) -- cheap insurance
            # against a bug anywhere upstream of this point.
            if not _is_safe_capture_id(cid):
                continue
            candidate_dir = root_path / cid
            try:
                if candidate_dir.is_symlink():
                    continue
                resolved = candidate_dir.resolve()
                if (resolved.parent != root_path or resolved.name != cid
                        or not resolved.is_relative_to(root_path)):
                    continue
                if not resolved.is_dir():
                    continue
                shutil.rmtree(candidate_dir)
            except Exception:  # noqa: BLE001 -- one bad directory must never abort the prune
                continue
            removed.append(PruneRemoval(replay_id=cid, captured_at=cat, reason=reason))

        kept = len(candidates) - len(removed)
        return PruneResult(ok=True, removed=tuple(removed), kept=kept,
                           skipped_unparseable=skipped_unparseable)
    except Exception as exc:  # noqa: BLE001 -- see docstring: this function must never raise
        return PruneResult(ok=False, error=f"{type(exc).__name__}: {exc}")


# The `decision` column marker stamped on every openers row written by an OFFLINE REPLAY
# (tools/opener_replay.py) rather than captured from a live run.
#
# IT LIVES HERE, NOT IN THE CLI, for an import-path reason worth stating: `operation_love` is an
# installed package and importable from anywhere, while `tools/` is only importable when the repo
# root happens to be on sys.path. A consumer that did `from tools.opener_replay import ...` broke
# `python tools/opener_corpus_report.py` outright (ModuleNotFoundError: No module named 'tools'),
# because a direct script run puts tools/ on sys.path rather than the repo root. Both the writer
# (tools/opener_replay.py) and the reader (tools/opener_corpus_report.py) import it from here so
# the value is spelled exactly once and neither depends on the other.
#
# It is deliberately neither "like" (a landed send) nor "dislike"/"never_sent" (a real human or
# AUTO decision not to send), so a synthetic row can never be miscounted as either. A raw query
# can always separate replay rows from real ones with `WHERE decision = 'synthetic_replay'`.
DECISION_REPLAY = "synthetic_replay"
