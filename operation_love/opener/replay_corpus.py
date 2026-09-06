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
``data/hinge_debug`` for the existing precedent this format follows). RETENTION -- how long a
captured directory is kept, and whether it is ever deleted -- is the owner's call; this module
provides no automatic expiry or pruning of its own.

WHAT THIS MODULE DELIBERATELY DOES NOT DO. It has NO dependency on ``operation_love.opener.
opener`` or ``operation_love.opener.service`` -- not even to import ``ItemRequest`` -- and no
dependency on any store or provider client. It is pure, generic I/O over plain values (bytes,
str, bool, float), by design: a future caller in the live opener path (NOT wired up by this
module -- see ``write_replay_capture``'s docstring for why) can call the writer with the exact
four fields ``opener.opener.ItemRequest`` already carries (``items``, ``name``, ``context``,
``truncated``) plus its own already-computed ``prompt_sha256``, without this module ever needing
to know what an ``ItemRequest`` is. ``tools/opener_replay.py`` is the one place that bridges
this format back to ``ItemRequest`` for an actual replay.

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

    NOT WIRED INTO THE OPENER PATH BY THIS MODULE. Some later change will call this from
    wherever an ``ItemRequest`` is actually built; until then this is a standalone API a test or
    a manual script can call directly.

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
