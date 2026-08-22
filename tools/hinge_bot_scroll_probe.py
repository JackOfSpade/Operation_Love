"""Bot-driven Hinge scroll probe — measures whether like-hearts can be tracked frame to frame
at the DRIVER's own bounded scroll cadence.

This is the calibration this project needs before ops/OPENER-REDESIGN.md Part B (5.5/5.10) can
be built. Read those two sections first. The prior calibration (`tools/hinge_scroll_capture.py`,
115 frames of a HUMAN-scrolled profile) found that naive heart-chaining fabricated a phantom
item, because human scroll steps ranged 0-787px against an 1800px content band. The hypothesis
this tool tests is that the driver's own BOUNDED scroll step (`read_scroll_frac`, humanized but
narrow) makes tracking reliable where uncontrolled human scrolling was not. This tool only
measures that hypothesis — it does NOT implement any part of Part B (no cropping, no indexing,
no card-snapping, no targeting).

    python -m tools.hinge_bot_scroll_probe                        # uses config.yaml
    python -m tools.hinge_bot_scroll_probe --config x.yaml
    python -m tools.hinge_bot_scroll_probe --out ops/calibration/botscroll_manual/
    python -m tools.hinge_bot_scroll_probe --match-tolerance-px 20 --min-response 0.4
    python -m tools.hinge_bot_scroll_probe --read-scroll-frac 0.16 --scroll-captures 48

--read-scroll-frac / --scroll-captures: a first bot-scroll capture (config.yaml's default
read_scroll_frac: 0.55) MEASURED the driver's own read-scroll step and the profile's
heart-to-heart spacing as the same magnitude (both ~1000-1200px) -- an ALIASING failure, not
an estimator-noise failure: a heart translated by one step lands almost exactly where the
next item's heart already was, so no delta estimator can tell "same heart moved" from "next
heart arrived" apart. These two flags let one capture run test the fix -- a step MATERIALLY
SMALLER than item spacing (e.g. read_scroll_frac ~0.15-0.18, i.e. roughly a third of the
default) -- without editing config.yaml (which would also change every OTHER Hinge action:
auto's own read-scrolls, `_scroll_to_top`'s mirror distance, etc). Both flags override the
DRIVER'S OWN attribute (`driver.read_scroll_frac` / `driver.scroll_captures` -- exactly what
`_sample_read_scroll`/`_capture_limit_for_profile` read in the no-auto-policy path this tool
always runs under), so `_capture_current` -> `_sample_read_scroll` -> `_scroll_down_one` ->
`_scroll` all still run completely normally: same jitter, same forbidden-zone guard, same
ledger bookkeeping. A smaller step needs proportionally more captures to reach the same
profile depth (see render_verdict's own step/spacing ratio math for what this is testing),
hence --scroll-captures alongside it.

WHAT THIS DOES to the phone: opens a session (same as every other tool here — launches the
Hinge app if it isn't already foreground) and then calls the real `HingeDriver.current_profile()`
EXACTLY ONCE. That method is production's own observe-mode capture path
(operation_love/drivers/hinge.py:2347) — it screencaps, humanized-read-scrolls a bounded number
of times (`self.spec.scroll_captures`, humanized dwell/step sampled from config, same as a real
run), and scrolls back to a confirmed top afterwards. Nothing about that scrolling is
reimplemented here; this tool calls the shipped method and nothing else on the driver's action
surface. See "SAFETY" below for how taps are made structurally impossible on top of that.

WHAT THIS NEVER DOES: tap, like, pass, send, or type anything. No comment sheet is ever opened.
See `neutralize_unsafe_methods` below — every decision method (`like`/`dislike`) and every
raw-input method (`tap`/`text`/any `keyevent`) reachable from the live driver and its
transports is monkeypatched to raise before `current_profile()` is ever called, on top of the
fact that `current_profile()` itself never calls any of them in the first place. Swipe/scroll is
left untouched, since bot-driven scrolling is the one action the owner approved measuring.

OUTPUT: every captured frame, plus `manifest.json` (the raw capture record — frame list, the
driver's own scroll ledger, content_band, screen size) and `analysis.json` (the heart-tracking
measurement: per-frame heart positions, per-pair phase-correlation scroll deltas, per-pair
match/tracking-failure verdicts, and the overall summary + verdict), under
`ops/calibration/botscroll_<UTC timestamp>/`. That directory is gitignored (see .gitignore's
`ops/calibration/` line) — these are a real person's dating profile photos. Nothing here
uploads, copies outside the repo, or transmits a frame anywhere.

Reads config.yaml the same way every other Hinge tool does (`operation_love.config.load` ->
`HingeDriver(cfg)`), so every humanization parameter (dwell distribution, read_scroll_frac,
forbidden zones, touch backend) matches a real run exactly — nothing here is a second,
independently-drifting copy of a timing constant.
"""
from __future__ import annotations

import argparse
from itertools import pairwise
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from operation_love import config as cfg_mod
from operation_love.drivers import hinge
from operation_love.drivers.hinge import HingeDriver
from operation_love.private_files import (
    atomic_write_private_bytes,
    atomic_write_private_text,
    ensure_private_dir,
)

_TOOL_VERSION = "1"

# How close a predicted heart position (a frame-N heart shifted by the pair's estimated scroll
# delta) must land to an actual frame-N+1 heart to count as the SAME item continuing. This is
# this tool's own matching knob, not a production constant — ops/OPENER-REDESIGN.md 5.10
# measured a residual std of 2.4px on well-matched pairs under an independently estimated
# offset; 15px is ~6x that, generous headroom for a different (bot) scroll cadence and a
# different offset estimator (phase correlation here, mean-abs-diff shift-search there).
_DEFAULT_MATCH_TOLERANCE_PX = 15.0

# cv2.phaseCorrelate's own response (roughly a 0..1 confidence: how much of the shifted crop's
# energy is explained by a single consistent translation) below which this tool refuses to
# trust the estimated delta at all, rather than risk chaining hearts under a wrong offset — see
# build_pair_result's docstring for why that refusal, not a best-effort match, is the point.
_DEFAULT_MIN_CORRELATION_RESPONSE = 0.5

# Sanity ceiling for --scroll-captures, purely to catch a typo (e.g. "480" meant as "48")
# before it turns into an hours-long capture against a real device. A read_scroll_frac around
# _READ_SCROLL_FRAC_MIN (hinge.py's own floor, 0.10) covering the longest plausible Hinge
# profile needs nowhere near this many frames; 200 is generous headroom above the ~48 the
# module docstring's own worked example (frac 0.16) needs, not a value anyone should expect to
# actually hit.
_MAX_SCROLL_CAPTURES = 200

# When measured_median_scroll_step_px / measured_median_heart_spacing_px falls in this band,
# render_verdict calls the failure step/spacing ALIASING rather than "estimator noise" — see
# its own docstring. Picked from the two regimes this tool exists to tell apart: the FIRST
# bot-scroll capture (config.yaml's default read_scroll_frac: 0.55) measured step and spacing
# within ~10% of each other (ratio ~1.0 — textbook aliasing), and the hypothesis this tool's
# --read-scroll-frac flag tests is a step around a third of that (frac ~0.16, ratio ~0.3 —
# comfortably clear of aliasing). 0.5..2.0 sits strictly between those two measured regimes on
# a log scale (a factor of 2 either side of equal), rather than requiring near-exact equality,
# because the failure mode this flags — a step and a spacing close enough in magnitude that a
# heart-shifted-by-one-step is hard to tell from a-different-heart-arrived — degrades well
# before the two are pixel-identical.
_ALIASING_RATIO_LOW = 0.5
_ALIASING_RATIO_HIGH = 2.0


# =====================================================================================
# SAFETY: make taps/likes/passes/typing structurally impossible before touching the phone
# =====================================================================================

class ScrollProbeGuardTripped(RuntimeError):
    """Raised by a neutralised method the instant this tool ever calls it.

    This must never actually happen: the only driver method this tool calls is
    `current_profile()`, and that method's own implementation never taps, likes, passes, or
    types (it screencaps and read-scrolls only). This guard is belt-and-braces on top of that
    fact, not a substitute for it — see `neutralize_unsafe_methods`'s docstring. If this
    exception ever surfaces, something is structurally wrong with this tool (or with
    `current_profile()` itself) and must be treated as a bug, not retried around.
    """


def _guarded(qualified_name: str) -> Callable:
    """Returns a function that raises ScrollProbeGuardTripped, standing in for a neutralised
    method. Takes *args/**kwargs so it can replace any method regardless of signature."""

    def _raise(*_args, **_kwargs):
        raise ScrollProbeGuardTripped(
            f"{qualified_name} was called. This tool only ever calls driver.current_profile() "
            f"on the live driver, which never taps/likes/passes/types — so this call means "
            f"something outside that contract ran. Refusing rather than letting it reach the "
            f"phone. See tools/hinge_bot_scroll_probe.py's module docstring.")

    return _raise


def neutralize_unsafe_methods(driver: HingeDriver) -> list[str]:
    """Monkeypatch every decision (like/pass) and raw-input (tap/text/keyevent) entry point
    reachable from the live driver and its transports, so each one RAISES immediately if this
    tool ever calls it — belt-and-braces on top of the fact that the one method this tool
    calls, `current_profile()`, never reaches any of them in its own implementation.

    Call this AFTER `driver.open_session()` (the transports — `driver.adb` / `driver.touch` —
    don't exist until then; both are properties that raise DriverClosed before open_session,
    per hinge.py:1241-1251) and BEFORE calling `current_profile()`.

    Neutralised, when present (checked with hasattr, so this degrades to "found nothing to
    neutralise there" rather than raising if a surface doesn't exist, instead of silently
    skipping the check altogether):

      - driver.like() / driver.dislike() — the only two decision methods DatingAppDriver
        declares (operation_love/drivers/base.py's abstract `like`/`dislike`); "pass" in
        Hinge's own vocabulary is `dislike`, not a method literally named `pass_`.
      - driver.adb.tap() / driver.adb.text() / driver.adb.keyevent() — the raw ADB transport.
        `text()` matters even though scrolling never calls it: Hinge's guarded `_text()`
        choke point ultimately types through `self.adb.text(...)`, not the touch transport.
      - driver.touch.tap() / driver.touch.text() / driver.touch.keyevent() — the humanized
        touch transport actually used for gestures (hinge.py's `_tap`/`_swipe`/`_scroll`
        route through `self.touch`, not always `self.adb` — see `_make_touch`). This can be
        UhidTouch (genuine virtual touchscreen) or, when `touch_backend: adb`, the SAME Adb
        object as `driver.adb` — skipped here when it is the same object, so the startup
        banner doesn't double-report one neutralisation as two.

    No method on this codebase is actually named `keyevent` (grep confirms — Adb/UhidTouch
    expose tap/swipe/scroll_up/text only), so that hasattr check is expected to find nothing
    today; it stays in the loop so a future transport method by that name is caught
    automatically rather than requiring someone to remember to update this list.

    swipe()/scroll_up() are deliberately left untouched on every transport: bounded,
    humanized bot-driven scrolling is the one action the owner approved this tool measuring.

    Returns the list of qualified names actually neutralised, printed at startup so the
    guard's presence is visible, not just asserted in a docstring.
    """
    neutralized: list[str] = []

    for name in ("like", "dislike"):
        if hasattr(driver, name):
            setattr(driver, name, _guarded(f"driver.{name}()"))
            neutralized.append(f"driver.{name}()")

    transports: list[tuple[str, object]] = []
    adb = getattr(driver, "adb", None)
    if adb is not None:
        transports.append(("driver.adb", adb))
    touch = getattr(driver, "touch", None)
    if touch is not None and touch is not adb:
        transports.append(("driver.touch", touch))

    for label, transport in transports:
        for method in ("tap", "text", "keyevent"):
            if hasattr(transport, method):
                setattr(transport, method, _guarded(f"{label}.{method}()"))
                neutralized.append(f"{label}.{method}()")

    return neutralized


# =====================================================================================
# MEASUREMENT INSTRUMENTATION: capture the driver's own read-scroll ledger before
# current_profile()'s own trailing _scroll_to_top() consumes/resets it.
# =====================================================================================

def install_ledger_capture(driver: HingeDriver) -> list[list[tuple[float, float]]]:
    """Wrap `driver._scroll_to_top` so the read-scroll ledger it is about to consume/reset
    gets snapshotted first, and return the (empty, mutated-in-place) list those snapshots get
    appended to.

    Why this is needed: `current_profile()` ends with `self._scroll_to_top(should_stop)`
    (hinge.py ~2365) whenever a profile was actually captured, and `_scroll_to_top` itself
    unconditionally resets `self._capture_scroll_ledger = []` once it confirms (or
    ceiling-bounds) the top (hinge.py ~2055) — that is how it hands back a clean slate for the
    NEXT profile's capture. So by the time `driver.current_profile()` returns to `main()`
    below, `driver._capture_scroll_ledger` already reads `[]`: reading it after the call, the
    way `save_capture` used to, silently reports "zero scrolling happened" for a capture that
    demonstrably scrolled — the exact "ships an empty field that looks like data" trap this
    tool must not fall into for the one number that is independent ground truth for the actual
    step size (everything else this tool measures is inferred from pixels).

    This wrapper does not change `_scroll_to_top`'s behavior at all: it reads
    `driver._capture_scroll_ledger` (never writes it, never touches `driver.adb`/`driver.touch`)
    and then calls straight through to the ORIGINAL bound method, unchanged — same swipes, same
    per-swipe jitter, same settle checks, same `should_stop` handling. It is not a substitute
    for `_scroll_down_one` doing the actual scrolling (it never scrolls anything itself), only
    a way to observe state `_scroll_to_top` is about to discard.

    `_scroll_to_top` can run TWICE inside one `current_profile()` call — once from
    `_ensure_session_top` (before the capture, on the first call in a session; ledger is `[]`
    at that point, since nothing has scrolled yet) and once from `current_profile()` itself
    (right after `_capture_current`, with the real per-profile ledger) — so the snapshot list
    can hold up to two entries. The caller wants the LAST one: whichever call happened most
    recently before `current_profile()` returned is the one that immediately followed this
    tool's single `_capture_current` read, regardless of how many `_scroll_to_top` calls came
    before it.
    """
    snapshots: list[list[tuple[float, float]]] = []
    original = driver._scroll_to_top

    def _wrapped(should_stop=None):
        snapshots.append(list(driver._capture_scroll_ledger))
        return original(should_stop)

    driver._scroll_to_top = _wrapped
    return snapshots


# =====================================================================================
# Vision: heart detection (reuses the production template/threshold/band, no reimplementation)
# =====================================================================================

def detect_hearts(frame: bytes, *, template, content_band: tuple[float, float]) -> list[tuple[int, int]]:
    """Locate like-hearts in `frame` using the CURRENT production recipe verbatim:
    `hinge._match_glyph` with `side="right"`, `threshold=hinge._LIKE_MATCH_THRESHOLD`, and
    `y_band=content_band` — the exact call shape `_locate_button("like")` uses
    (hinge.py:1461-1464). Returns (x, y) centers sorted top-to-bottom, in real screen pixels."""
    return hinge._match_glyph(frame, template, side="right",
                               threshold=hinge._LIKE_MATCH_THRESHOLD, y_band=content_band)


def compute_heart_spacings_px(hearts_per_frame: list[list[tuple[int, int]]]) -> list[float]:
    """Every consecutive-heart vertical gap measured WITHIN a single frame (two or more hearts
    visible on screen at once), collected across every frame. This is a ground-truth
    measurement of ITEM SPACING that depends on nothing but `detect_hearts`'s own per-frame
    output (sorted top-to-bottom, its own documented contract) — no delta estimator, no
    frame-to-frame matching, no assumption about scroll direction or step size. It exists so
    `render_verdict` can test the aliasing hypothesis (ops/OPENER-REDESIGN.md 5.10 / this
    tool's module docstring) — whether the measured scroll step and the measured item spacing
    are close enough in magnitude that no frame-to-frame estimator could ever tell "the same
    heart, moved" apart from "the next heart, arrived" — using two independently-measured
    pixel quantities rather than one derived from the other.

    Returns [] when no single frame ever showed 2+ hearts at once (a real possibility when the
    content band is shorter than one item's spacing) — callers must treat that as "not
    measured this run", not as "spacing is zero".
    """
    spacings: list[float] = []
    for hearts in hearts_per_frame:
        ys = sorted(y for _x, y in hearts)
        spacings.extend(b - a for a, b in pairwise(ys))
    return spacings


# =====================================================================================
# Scroll delta estimation: phase correlation (our own measurement, independent of the
# driver's ledger and of hinge.py's _vertical_shift_match, which 5.5 demoted to corroboration)
# =====================================================================================

def estimate_vertical_delta(frame_a: bytes, frame_b: bytes, *,
                             content_band: tuple[float, float]) -> tuple[float, float]:
    """Independently estimate the vertical scroll offset between two consecutive frames via 2D
    phase correlation (`cv2.phaseCorrelate`), restricted to `content_band`'s rows.

    Restricted to content_band for the same reason `_vertical_shift_match`/`_match_glyph`'s
    `y_band` are (hinge.py:588's own docstring, MEASURED 2026-08-10): the status bar, sticky
    header, floating like/pass buttons, and bottom nav do not translate when the content
    scrolls, and would dominate a whole-frame correlation with a shift of ~0 regardless of how
    well the actual content lines up underneath them.

    Returns `(delta_px, response)`.

    `delta_px` is POSITIVE for an ordinary forward/down read-scroll: a piece of content at row
    y in frame_a sits at approximately row `y - delta_px` in frame_b (content moves UP on
    screen as you scroll down through a profile). Sign convention EMPIRICALLY VERIFIED against
    `cv2.phaseCorrelate`'s own `(dx, dy)` return (see tests/test_hinge_bot_scroll_probe.py):
    `dy` comes back NEGATIVE for a downward scroll, so `delta_px = -dy`.

    `response` is phaseCorrelate's own confidence, roughly in `[0, 1]` — how much of the
    shifted crop's energy is explained by a single consistent translation. Low response is
    exactly what a "large-jump tracking failure" (ops/OPENER-REDESIGN.md 5.10) looks like from
    this method: too large a jump leaves too little genuinely-shared content between the two
    crops for any one translation to explain them well. `build_pair_result` below is what
    turns a low response into a reported tracking failure instead of a silently-wrong match.

    Raises RuntimeError if either frame fails to decode (corrupt/truncated screencap) — this
    is an offline analysis tool over already-captured bytes, so a decode failure is a data
    problem to report, not a case to guess through.
    """
    import cv2
    import numpy as np
    img_a = cv2.imdecode(np.frombuffer(frame_a, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imdecode(np.frombuffer(frame_b, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img_a is None or img_b is None:
        raise RuntimeError("estimate_vertical_delta: a frame failed to decode as an image")
    r0, r1 = hinge._content_rows(content_band, img_a.shape[0])
    crop_a = img_a[r0:r1].astype(np.float32)
    crop_b = img_b[r0:r1].astype(np.float32)
    window = cv2.createHanningWindow((crop_a.shape[1], crop_a.shape[0]), cv2.CV_32F)
    (_dx, dy), response = cv2.phaseCorrelate(crop_a, crop_b, window)
    return -dy, float(response)


# =====================================================================================
# Tracking / chaining analysis — pure geometry, no image decoding, hence directly unit
# testable against KNOWN deltas and KNOWN heart positions (see the test file).
# =====================================================================================

@dataclass
class PairResult:
    a_index: int
    b_index: int
    hearts_a: list[tuple[int, int]]
    hearts_b: list[tuple[int, int]]
    delta_px: float
    response: float
    reliable: bool
    # (index into hearts_a, index into hearts_b) for every pair matched within tolerance,
    # under `delta_px`, when `reliable` — [] when not reliable (see build_pair_result: an
    # unreliable pair is never used to attempt matching at all, on purpose).
    matched: list[tuple[int, int]] = field(default_factory=list)
    # hearts_a indices predicted to have scrolled OUT of content_band by frame b — an
    # expected, unremarkable disappearance, not a tracking problem.
    left_band_a: list[int] = field(default_factory=list)
    # hearts_a indices still expected to be visible in frame b (in-band prediction) but with
    # no matching heart found — a genuine miss, distinct from left_band_a.
    unmatched_a: list[int] = field(default_factory=list)
    # True iff this pair is reliable AND every hearts_a heart still expected in view found a
    # match — i.e. the literal "does every heart in frame N map to frame N+1" check.
    all_matched: bool = False


def build_pair_result(a_index: int, b_index: int, hearts_a, hearts_b, delta_px: float,
                       response: float, *, content_band_px: tuple[int, int],
                       tolerance_px: float = _DEFAULT_MATCH_TOLERANCE_PX,
                       min_response: float = _DEFAULT_MIN_CORRELATION_RESPONSE,
                       max_delta_px: float | None = None) -> PairResult:
    """Match hearts_a[i] to hearts_b[j] under a single shared `delta_px`, for one frame pair.

    A pair is `reliable` only when BOTH: `response >= min_response` (phase correlation itself
    trusts its own estimate) AND `abs(delta_px) <= max_delta_px` (when given — `max_delta_px`
    defaults to None, meaning no ceiling, for callers with no natural bound to hand; the real
    pipeline in `analyze_frames` passes the content band's own pixel height, since a shift at
    least that large guarantees ZERO row overlap between the two crops, making the phase
    correlation that produced it meaningless by construction, independent of what `response`
    happened to read).

    When a pair is NOT reliable, this deliberately returns `matched=[]` rather than a
    best-effort match — attempting to match hearts under a delta we already know not to trust
    is exactly how ops/OPENER-REDESIGN.md 5.10's naive counting fabricated a phantom item (a
    large-jump tracking failure silently accepted as "item left, new item entered"). Refusing
    to match anything for that pair, and letting `chain_items` report it as a named tracking
    failure instead, is the fix this tool measures whether it's even NEEDED for (bounded bot
    scrolling may simply never produce an unreliable pair at all — that is the hypothesis).

    Matching itself, when reliable: for every (hearts_a[i], hearts_b[j]) whose Euclidean
    distance from the PREDICTED position `(ax, ay - delta_px)` to `(bx, by)` is within
    `tolerance_px`, greedily assign the globally closest candidates first, one-to-one (no
    heart matched twice). `content_band_px` (a `(r0, r1)` row range in real screen pixels,
    e.g. `hinge._content_rows(content_band, screen_height)`) decides whether an unmatched
    hearts_a item is `left_band_a` (predicted position above `r0`: scrolled off the top,
    expected) or a genuine `unmatched_a` miss.
    """
    reliable = response >= min_response and (max_delta_px is None or abs(delta_px) <= max_delta_px)
    r0, _r1 = content_band_px

    matched: list[tuple[int, int]] = []
    if reliable and hearts_a and hearts_b:
        candidates = []
        for ai, (ax, ay) in enumerate(hearts_a):
            predicted_y = ay - delta_px
            for bi, (bx, by) in enumerate(hearts_b):
                dist = math.hypot(bx - ax, by - predicted_y)
                if dist <= tolerance_px:
                    candidates.append((dist, ai, bi))
        candidates.sort(key=lambda t: t[0])
        used_a: set[int] = set()
        used_b: set[int] = set()
        for _dist, ai, bi in candidates:
            if ai in used_a or bi in used_b:
                continue
            matched.append((ai, bi))
            used_a.add(ai)
            used_b.add(bi)
    else:
        used_a = set()

    left_band_a = []
    unmatched_a = []
    for ai, (_ax, ay) in enumerate(hearts_a):
        if ai in used_a:
            continue
        predicted_y = ay - delta_px
        if predicted_y < r0:
            left_band_a.append(ai)
        else:
            unmatched_a.append(ai)

    all_matched = reliable and not unmatched_a

    return PairResult(a_index=a_index, b_index=b_index, hearts_a=list(hearts_a),
                       hearts_b=list(hearts_b), delta_px=delta_px, response=response,
                       reliable=reliable, matched=matched, left_band_a=left_band_a,
                       unmatched_a=unmatched_a, all_matched=all_matched)


@dataclass
class ChainResult:
    # Distinct items chaining recovers WITH CONFIDENCE: frame-0 hearts, plus every heart that
    # first appeared behind a RELIABLE pair with no matching predecessor. This is the number
    # to compare against the true item count — it must never include a heart whose "new-ness"
    # is only apparent because the pair that would have matched it was unreliable.
    distinct_confirmed: int
    # Hearts that appeared right after an UNRELIABLE pair. Each MIGHT be a genuine new item,
    # or might be the very heart that went "unmatched" on the other side of that same
    # unreliable pair — i.e. a potential phantom, exactly like 5.10's spurious 10th item.
    # Deliberately kept OUT of distinct_confirmed and reported separately instead of guessed
    # either way.
    ambiguous_new: int
    # Confirmed tracks that hit a genuine unmatched_a miss (reliable pair, in-view prediction,
    # no candidate found) rather than a clean left-band disappearance — also not folded
    # silently into "the item left", since a miss here is itself the failure mode being
    # measured, not evidence the item was never there.
    orphaned_tracks: int
    # (a_index, b_index) of every pair build_pair_result marked unreliable.
    tracking_failures: list[tuple[int, int]]


def chain_items(pair_results: list[PairResult]) -> ChainResult:
    """Walk a sequence of consecutive-pair PairResults and count distinct likeable items,
    the way an index built on frame-to-frame heart tracking would have to.

    Every frame-0 heart starts a confirmed track (there is no "before" to compare it to). A
    track EXTENDS across a pair only when that pair is reliable and the specific heart has a
    recorded match — see PairResult.matched. A track ends CLEANLY (no ambiguity, not counted
    as a failure) when its predicted position has scrolled out of the content band
    (PairResult.left_band_a). A track is ORPHANED — flagged, not silently resolved either
    way — when: (a) the pair covering it is unreliable (we do not know whether it continues
    into some heart on the far side of that same pair, or genuinely left), or (b) it was
    still expected in view under a RELIABLE pair but no match was found anyway (a real, rarer
    miss). Symmetrically, a heart appearing on the far side of an unreliable pair is
    `ambiguous_new`, not `distinct_confirmed` — see ChainResult's own docstring for why:
    guessing either direction here is exactly how 5.10's naive counting fabricated a phantom.
    """
    tracking_failures: list[tuple[int, int]] = []
    orphaned = 0
    ambiguous_new = 0
    confirmed = 0

    if not pair_results:
        return ChainResult(distinct_confirmed=0, ambiguous_new=0, orphaned_tracks=0,
                            tracking_failures=[])

    n_first_frame_hearts = len(pair_results[0].hearts_a)
    confirmed += n_first_frame_hearts
    # active: heart-index-in-current-frame -> "this heart belongs to a track that is still
    # confirmed" (True) or "...still open but only ambiguously accounted for" (False). Only
    # the boolean matters going forward — chain_items never needs to name individual tracks,
    # only count how many started confidently vs. ambiguously and how many ended in doubt.
    active: dict[int, bool] = {i: True for i in range(n_first_frame_hearts)}

    for pair in pair_results:
        new_active: dict[int, bool] = {}
        if not pair.reliable:
            tracking_failures.append((pair.a_index, pair.b_index))
            # Every currently active track touching this pair becomes orphaned (unless it
            # was already ambiguous, which stays ambiguous rather than being double-counted
            # as newly orphaned): we cannot honestly say whether it continues. Every heart on
            # the B side is ambiguous-new for the identical reason.
            for was_confirmed in active.values():
                if was_confirmed:
                    orphaned += 1
            ambiguous_new += len(pair.hearts_b)
            active = {}
            continue

        matched_a = {ai for ai, _bi in pair.matched}
        matched_b = {bi for _ai, bi in pair.matched}

        for ai, bi in pair.matched:
            new_active[bi] = active.get(ai, True)

        for ai, was_confirmed in active.items():
            if ai in matched_a or ai in pair.left_band_a:
                continue   # continued (handled above) or left cleanly — not orphaned
            if was_confirmed:
                orphaned += 1
            # an already-ambiguous track that also misses here is simply dropped: it was
            # never added to distinct_confirmed, so there is nothing left to un-count.

        for bi in range(len(pair.hearts_b)):
            if bi in matched_b:
                continue
            # Genuinely new under a reliable pair: confident.
            new_active[bi] = True
            confirmed += 1

        active = new_active

    return ChainResult(distinct_confirmed=confirmed, ambiguous_new=ambiguous_new,
                        orphaned_tracks=orphaned, tracking_failures=tracking_failures)


# =====================================================================================
# Top-level analysis: glue detect_hearts + estimate_vertical_delta + build_pair_result +
# chain_items over a real captured frame sequence.
# =====================================================================================

@dataclass
class AnalysisReport:
    frame_count: int
    hearts_per_frame: list[list[tuple[int, int]]]
    pair_results: list[PairResult]
    scroll_deltas_px: list[float]
    match_rate: float | None   # matched / (matched + unmatched_a), across reliable pairs only
    chain: ChainResult
    # Every consecutive-heart Y-gap measured WITHIN a single frame — see
    # compute_heart_spacings_px's docstring. [] when no frame ever showed 2+ hearts at once.
    heart_spacings_px: list[float]
    # median(abs(delta_px)) across EVERY pair in scroll_deltas_px (reliable or not — an
    # unreliable pair's estimated delta is still a measurement, not noise to discard before
    # comparing it against spacing; see render_verdict). None when there were <2 frames.
    median_scroll_step_px: float | None
    # median(heart_spacings_px). None when heart_spacings_px is [].
    median_heart_spacing_px: float | None
    # median_scroll_step_px / median_heart_spacing_px. None when either input above is None or
    # median_heart_spacing_px is 0. This is the number render_verdict tests against
    # _ALIASING_RATIO_LOW/_ALIASING_RATIO_HIGH to tell "step/spacing aliasing" apart from an
    # ordinary noisy-estimator failure — see render_verdict's own docstring.
    step_spacing_ratio: float | None

    def to_json_dict(self) -> dict:
        d = asdict(self)
        return d


def _step_spacing_stats(scroll_deltas_px: list[float],
                         heart_spacings_px: list[float]
                         ) -> tuple[float | None, float | None, float | None]:
    """`(median_scroll_step_px, median_heart_spacing_px, step_spacing_ratio)` — see
    AnalysisReport's field docstrings for exactly what each measures and why `scroll_deltas_px`
    is used un-filtered (every pair, reliable or not). Any missing input yields `None`s rather
    than raising or silently reporting a ratio of 0/1: `render_verdict` treats a `None` ratio
    as "the aliasing hypothesis was not testable this run", which is a different, honestly
    reported outcome from "tested and not aliased"."""
    median_step = (statistics.median(abs(d) for d in scroll_deltas_px)
                   if scroll_deltas_px else None)
    median_spacing = statistics.median(heart_spacings_px) if heart_spacings_px else None
    ratio = (median_step / median_spacing
             if median_step is not None and median_spacing else None)
    return median_step, median_spacing, ratio


def analyze_frames(frames: list[bytes], *, screen_size: tuple[int, int],
                    content_band: tuple[float, float],
                    tolerance_px: float = _DEFAULT_MATCH_TOLERANCE_PX,
                    min_response: float = _DEFAULT_MIN_CORRELATION_RESPONSE) -> AnalysisReport:
    """The real pipeline: detect hearts in every frame (production template/threshold/band),
    estimate each consecutive pair's scroll delta by phase correlation, match hearts under
    that delta, and chain the matches into a distinct-item count. This is what `main()` runs
    against a real capture; the pure geometry pieces above (`build_pair_result`/`chain_items`)
    are what the test suite exercises directly against known, hand-built inputs."""
    template = hinge._load_template(hinge.HINGE_SPEC.templates["like"])
    if template is None:
        raise RuntimeError(
            "could not load the 'like' glyph template (hinge_like_button.png) — cv2 is "
            "likely missing. Install the extra: pip install -e '.[hinge]'")

    _w, h = screen_size
    r0, r1 = hinge._content_rows(content_band, h)
    max_delta_px = float(r1 - r0)   # a shift this large shares zero guaranteed row overlap

    hearts_per_frame = [detect_hearts(f, template=template, content_band=content_band)
                        for f in frames]

    pair_results: list[PairResult] = []
    for i in range(len(frames) - 1):
        delta_px, response = estimate_vertical_delta(frames[i], frames[i + 1],
                                                      content_band=content_band)
        pair_results.append(build_pair_result(
            i, i + 1, hearts_per_frame[i], hearts_per_frame[i + 1], delta_px, response,
            content_band_px=(r0, r1), tolerance_px=tolerance_px, min_response=min_response,
            max_delta_px=max_delta_px))

    total_matched = sum(len(p.matched) for p in pair_results)
    total_eligible = total_matched + sum(len(p.unmatched_a) for p in pair_results)
    match_rate = (total_matched / total_eligible) if total_eligible else None

    chain = chain_items(pair_results)

    scroll_deltas_px = [p.delta_px for p in pair_results]
    heart_spacings_px = compute_heart_spacings_px(hearts_per_frame)
    median_step, median_spacing, ratio = _step_spacing_stats(scroll_deltas_px, heart_spacings_px)

    return AnalysisReport(
        frame_count=len(frames),
        hearts_per_frame=hearts_per_frame,
        pair_results=pair_results,
        scroll_deltas_px=scroll_deltas_px,
        match_rate=match_rate,
        chain=chain,
        heart_spacings_px=heart_spacings_px,
        median_scroll_step_px=median_step,
        median_heart_spacing_px=median_spacing,
        step_spacing_ratio=ratio,
    )


def render_verdict(report: AnalysisReport) -> str:
    """Plain-language verdict — never softened. A single tracking failure or a single
    unresolved match miss is disqualifying, matching ops/OPENER-REDESIGN.md 5.6's own bar
    ("never substitute a different item"): an index this project would actually tap against
    needs every heart accounted for, not "almost all".

    ALIASING vs estimator noise: a first bot-scroll capture (read_scroll_frac 0.55) measured
    the driver's own scroll step and the profile's heart-to-heart spacing as the SAME
    magnitude, and BOTH delta estimators tried against it (2D phase correlation here, a 1D
    row-profile cross-correlation elsewhere) failed on it. "Large jump / low correlation
    response" describes the SYMPTOM of that failure, not its cause, and reporting only the
    symptom invites the wrong fix (tune the estimator) for a problem no estimator can fix: when
    a heart translated by one scroll step lands almost exactly where the next item's heart
    already was, "the same heart moved" and "the next heart arrived" are geometrically
    indistinguishable from the pixels alone, regardless of which algorithm looks at them. This
    function tests that directly — `report.step_spacing_ratio` compares the MEASURED median
    scroll step (`estimate_vertical_delta`'s own output, every pair, independent of whether
    that pair was later judged reliable) against the MEASURED median heart-to-heart spacing
    (`compute_heart_spacings_px` — hearts seen together in one frame, independent of any
    estimator). When that ratio falls in `_ALIASING_RATIO_LOW.._ALIASING_RATIO_HIGH`, the
    verdict below names the root cause explicitly as aliasing instead of folding it into the
    same "reasons" list as ordinary tracking symptoms.
    """
    if report.frame_count < 2:
        return ("INCONCLUSIVE: fewer than 2 frames were captured, so there is nothing to "
                "compare frame-to-frame.")

    ratio = report.step_spacing_ratio
    if ratio is not None:
        step_spacing_line = (
            f"measured median scroll step {report.median_scroll_step_px:.1f}px vs measured "
            f"median heart-to-heart spacing {report.median_heart_spacing_px:.1f}px (ratio "
            f"{ratio:.2f})")
        aliasing = _ALIASING_RATIO_LOW <= ratio <= _ALIASING_RATIO_HIGH
    else:
        step_spacing_line = (
            "step/spacing ratio: NOT MEASURABLE this run (no single frame showed 2+ hearts at "
            "once, so there is no within-frame spacing to compare the measured scroll step "
            "against) — the aliasing hypothesis below could not be directly tested; treat "
            "absence of this measurement as untested, not as ruled out")
        aliasing = False

    reasons = []
    if report.chain.tracking_failures:
        reasons.append(f"{len(report.chain.tracking_failures)} pair(s) flagged as tracking "
                        f"failures (unreliable scroll-delta estimate): "
                        f"{report.chain.tracking_failures}")
    if report.chain.orphaned_tracks:
        reasons.append(f"{report.chain.orphaned_tracks} heart(s) went unmatched despite a "
                        f"reliable pair (a genuine tracking miss)")
    if report.chain.ambiguous_new:
        reasons.append(f"{report.chain.ambiguous_new} heart(s) could not be confidently "
                        f"classified as new-vs-continuing (they sit behind a tracking "
                        f"failure) — this is exactly how a phantom item gets fabricated")
    if report.match_rate is not None and report.match_rate < 1.0:
        reasons.append(f"frame-to-frame match rate was {report.match_rate:.1%}, not 100%")

    if not reasons:
        return ("RELIABLE (this profile): every heart tracked cleanly frame-to-frame at the "
                "driver's own bounded scroll cadence — zero tracking failures, zero orphaned "
                "matches, zero ambiguous items, 100% match rate. " + step_spacing_line + ". "
                "Bounded bot-driven scrolling looks safe to build a heart-tracking index on. "
                "This is ONE profile — do not treat one clean run as proof; validate over more "
                "profiles before building on it.")

    if aliasing:
        return (
            "NOT RELIABLE (this profile): ROOT CAUSE is step/spacing ALIASING, not estimator "
            "noise. " + step_spacing_line + " — the scroll step and the item spacing are the "
            "SAME magnitude, so a heart translated by one step lands almost exactly where the "
            "next item's heart already was: 'same heart moved' and 'next heart arrived' are "
            "geometrically indistinguishable from the pixels alone. No better delta estimator "
            "fixes this by construction — shrink the scroll step well below the item spacing "
            "(or use a different indexing strategy entirely). Symptoms observed, all downstream "
            "of that one root cause, not independent bugs: " + "; ".join(reasons) + ".")

    return ("NOT RELIABLE (this profile): " + "; ".join(reasons) + ". " + step_spacing_line +
            ". Heart tracking is NOT safe to build an index on at this cadence without further "
            "work — the same failure mode ops/OPENER-REDESIGN.md 5.10 found under human "
            "scrolling (a large-jump tracking failure fabricating a phantom item) can still "
            "occur under bot-driven scrolling.")


# =====================================================================================
# Capture, save, and report
# =====================================================================================

def _default_out_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("ops/calibration") / f"botscroll_{stamp}"


def save_capture(photos: list[bytes], profile_meta: dict, driver: HingeDriver,
                  out_dir: Path, *,
                  driver_scroll_ledger: list[tuple[float, float]] | None) -> dict:
    """Write every frame as `NNNNN.png` (capture order) plus `manifest.json` — the raw
    capture record, separate from analysis.json's measurement output. Includes the driver's
    OWN scroll ledger (the (frac, x_frac) it actually issued for this capture) purely for
    cross-reference against this tool's independently phase-correlation-estimated deltas — the
    two are not expected to match exactly (the ledger is a fraction-of-screen intent before
    humanization jitter; the phase-correlation number is what the pixels actually did), but a
    wildly divergent pair would itself be a finding worth seeing.

    `driver_scroll_ledger` must be the snapshot `install_ledger_capture` took BEFORE
    `current_profile()`'s own trailing `_scroll_to_top()` reset `driver._capture_scroll_ledger`
    to `[]` — reading `driver._capture_scroll_ledger` directly here (the old implementation)
    always wrote an empty list after a successful capture, which reads as "zero scrolling
    happened" for a capture that demonstrably scrolled. Pass `None` (never `[]`, which would
    silently claim "recorded, and empty") when the snapshot genuinely could not be taken — the
    manifest then records that honestly via `driver_scroll_ledger_note` instead of shipping an
    empty field that looks like data."""
    ensure_private_dir(out_dir)
    frames = []
    for i, png in enumerate(photos, start=1):
        name = f"{i:05d}.png"
        atomic_write_private_bytes(out_dir / name, png, parent=out_dir)
        frames.append({"file": name})
    manifest = {
        "tool_version": _TOOL_VERSION,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "frame_count": len(photos),
        "frames": frames,
        "profile_meta": profile_meta,
        "driver_scroll_ledger": (
            [{"frac": frac, "x_frac": x_frac} for frac, x_frac in driver_scroll_ledger]
            if driver_scroll_ledger is not None else None),
        "driver_scroll_ledger_note": (
            None if driver_scroll_ledger is not None else
            "NOT CAPTURED: the ledger could not be snapshotted before current_profile()'s own "
            "_scroll_to_top() reset it. This is a measurement gap, not evidence of zero "
            "scrolling — see install_ledger_capture in tools/hinge_bot_scroll_probe.py."),
        "read_scroll_frac": driver.read_scroll_frac,
        "scroll_captures": driver.scroll_captures,
        "content_band": list(driver.content_band),
        "screen_size": list(driver.adb.screen_size()),
    }
    atomic_write_private_text(
        out_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n", parent=out_dir)
    return manifest


def save_analysis(report: AnalysisReport, out_dir: Path) -> None:
    ensure_private_dir(out_dir)
    atomic_write_private_text(
        out_dir / "analysis.json", json.dumps(report.to_json_dict(), indent=2) + "\n",
        parent=out_dir,
    )


def print_report(report: AnalysisReport, *,
                  driver_scroll_ledger: list[tuple[float, float]] | None) -> None:
    """`driver_scroll_ledger` — see save_capture's docstring: the pre-reset snapshot from
    `install_ledger_capture`, or `None` when that snapshot could not be taken. Printed
    separately from the phase-correlation-derived scroll_deltas_px below because it is the
    only INDEPENDENT ground truth this tool has for the actual step size — everything else
    (scroll_deltas_px, heart_spacings_px, the whole aliasing ratio) is inferred from pixels."""
    print("\n--- Per-frame heart detections ---")
    for i, hearts in enumerate(report.hearts_per_frame):
        ys = [y for _x, y in hearts]
        print(f"  frame {i:2d}: {len(hearts)} heart(s) at y={ys}")

    print("\n--- Per-pair scroll delta + tracking ---")
    for p in report.pair_results:
        status = "OK" if p.all_matched else ("TRACKING FAILURE" if not p.reliable else "MISS")
        print(f"  frame {p.a_index:2d} -> {p.b_index:2d}: delta={p.delta_px:7.1f}px  "
              f"response={p.response:.3f}  hearts {len(p.hearts_a)}->{len(p.hearts_b)}  "
              f"matched={len(p.matched)}  left_band={len(p.left_band_a)}  "
              f"unmatched={len(p.unmatched_a)}  [{status}]")

    print("\n--- Driver's own scroll ledger (ground truth intent — ADB gesture fraction, "
          "before pixel measurement) ---")
    if driver_scroll_ledger:
        fracs = [frac for frac, _x in driver_scroll_ledger]
        print(f"  {len(driver_scroll_ledger)} read-scroll(s) issued — frac min={min(fracs):.3f} "
              f"max={max(fracs):.3f} mean={sum(fracs) / len(fracs):.3f}")
        for i, (frac, x_frac) in enumerate(driver_scroll_ledger):
            print(f"    scroll {i:2d}: frac={frac:.3f}  x_frac={x_frac:.3f}")
    elif driver_scroll_ledger is None:
        print("  NOT CAPTURED this run (the pre-reset snapshot could not be taken — see "
              "manifest.json's driver_scroll_ledger_note). This is a measurement gap, not "
              "evidence that zero scrolling happened.")
    else:
        print("  empty (0 read-scrolls issued — the profile fit in a single frame)")

    deltas = report.scroll_deltas_px
    print("\n--- Summary ---")
    if deltas:
        print(f"  bot scroll deltas (px, phase-correlation estimate): min={min(deltas):.1f}  "
              f"max={max(deltas):.1f}  mean={sum(deltas) / len(deltas):.1f}  n={len(deltas)}")
    else:
        print("  bot scroll deltas: none (fewer than 2 frames)")
    mr = f"{report.match_rate:.1%}" if report.match_rate is not None else "n/a (no eligible hearts)"
    print(f"  frame-to-frame heart match rate: {mr}")
    print(f"  distinct items chaining recovers (confirmed): {report.chain.distinct_confirmed}")
    print(f"  ambiguous new items (behind a tracking failure): {report.chain.ambiguous_new}")
    print(f"  orphaned tracks (reliable pair, genuine miss): {report.chain.orphaned_tracks}")
    print(f"  tracking failures: {report.chain.tracking_failures or 'none'}")
    if report.step_spacing_ratio is not None:
        print(f"  step/spacing: median scroll step {report.median_scroll_step_px:.1f}px vs "
              f"median heart spacing {report.median_heart_spacing_px:.1f}px "
              f"(ratio {report.step_spacing_ratio:.2f}, n={len(report.heart_spacings_px)} "
              f"within-frame spacing sample(s))")
    else:
        print("  step/spacing: not measurable this run (no frame showed 2+ hearts at once)")

    print("\n--- Verdict ---")
    print("  " + render_verdict(report))


# =====================================================================================
# main
# =====================================================================================

from tools._devicelock import holding_the_device


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Bot-driven Hinge scroll probe: one real HingeDriver.current_profile() "
                     "read, measuring whether like-hearts track reliably frame to frame at "
                     "the driver's own bounded scroll cadence (ops/OPENER-REDESIGN.md 5.5/"
                     "5.10). Never taps, likes, passes, or types.")
    ap.add_argument("--config", default="config.yaml", help="config.yaml path")
    ap.add_argument("--out", default=None,
                     help="output directory (default ops/calibration/botscroll_<UTC ts>/)")
    ap.add_argument("--match-tolerance-px", type=float, default=_DEFAULT_MATCH_TOLERANCE_PX,
                     help="max px a predicted heart position may miss an actual heart by and "
                          f"still count as the same item (default {_DEFAULT_MATCH_TOLERANCE_PX})")
    ap.add_argument("--min-response", type=float, default=_DEFAULT_MIN_CORRELATION_RESPONSE,
                     help="minimum cv2.phaseCorrelate response to trust a pair's estimated "
                          f"delta at all (default {_DEFAULT_MIN_CORRELATION_RESPONSE})")
    ap.add_argument("--read-scroll-frac", type=float, default=None,
                     help="override the driver's read_scroll_frac (fraction of screen height "
                          "per read-scroll) for THIS capture only — config.yaml is not "
                          "touched. Must be in "
                          f"[{hinge._READ_SCROLL_FRAC_MIN}, {hinge._READ_SCROLL_FRAC_MAX}], the "
                          "same bound hinge.py enforces on a policy-sampled fraction. Default: "
                          "whatever config.yaml's apps.hinge.read_scroll_frac resolves to "
                          "(0.55 in the shipped config). The aliasing hypothesis this flag "
                          "exists to test wants a step MATERIALLY SMALLER than card spacing — "
                          "around 0.15-0.18 — see the module docstring.")
    ap.add_argument("--scroll-captures", type=int, default=None,
                     help="override the driver's scroll_captures (max screencaps while "
                          "reading one profile) for THIS capture only — config.yaml is not "
                          f"touched. Must be an integer in [1, {_MAX_SCROLL_CAPTURES}]. "
                          "Default: whatever config.yaml's apps.hinge.scroll_captures "
                          "resolves to (12 in the shipped config). A smaller "
                          "--read-scroll-frac needs proportionally more captures to reach the "
                          "same profile depth — roughly (default_frac / new_frac) x the "
                          "current scroll_captures; e.g. frac 0.16 against the shipped "
                          "0.55/12 needs on the order of 40-48.")
    args = ap.parse_args(argv)

    # Validate the two override flags before anything touches config.yaml or the device --
    # a typo here must fail loudly at the command line, not turn into a bogus gesture on a
    # real phone or a silently-truncated capture.
    if args.read_scroll_frac is not None and not (
            math.isfinite(args.read_scroll_frac)
            and hinge._READ_SCROLL_FRAC_MIN <= args.read_scroll_frac <= hinge._READ_SCROLL_FRAC_MAX):
        print(f"ERROR: --read-scroll-frac must be a finite number in "
              f"[{hinge._READ_SCROLL_FRAC_MIN}, {hinge._READ_SCROLL_FRAC_MAX}] (the same bound "
              f"hinge.py's own _sample_read_scroll enforces on a policy-sampled fraction) — "
              f"got {args.read_scroll_frac!r}.", file=sys.stderr)
        sys.exit(1)
    if args.scroll_captures is not None and not (1 <= args.scroll_captures <= _MAX_SCROLL_CAPTURES):
        print(f"ERROR: --scroll-captures must be an integer in [1, {_MAX_SCROLL_CAPTURES}] — "
              f"got {args.scroll_captures!r}.", file=sys.stderr)
        sys.exit(1)

    print("=" * 78)
    print("hinge_bot_scroll_probe: SAFETY")
    print("=" * 78)
    print("WILL: open a Hinge session (launch the app if needed) and call the real "
          "HingeDriver.current_profile() exactly ONCE — production's own humanized, "
          "bounded read-scroll capture. Frames are saved locally under ops/calibration/ "
          "(gitignored) for offline heart-tracking measurement only.")
    print("WILL NOT: tap, like, pass, send, or type ANYTHING. No comment sheet is ever "
          "opened. Every decision method and every raw-input method this tool can reach is "
          "monkeypatched to raise before current_profile() is called — see the neutralised-"
          "method list printed below once the session is open.")
    print("=" * 78 + "\n")

    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:
        print(f"ERROR: this tool requires opencv-python + numpy ({exc}). Install the extra: "
              "pip install -e '.[hinge]'", file=sys.stderr)
        sys.exit(1)

    # Device lock (tools/_devicelock.py): the probe opens a real session and drives
    # production's own read-scroll, so it must never share the phone with a hub run.
    with holding_the_device(args.config):
        _probe(args)


def _probe(args: argparse.Namespace) -> None:
    cfg = cfg_mod.load(args.config)
    driver = HingeDriver(cfg)
    out_dir = Path(args.out) if args.out else _default_out_dir()

    print(f"Connecting to the Hinge phone over ADB (config: {args.config})...")
    try:
        driver.open_session()
    except Exception as exc:  # noqa: BLE001 — report clearly, exit non-zero, never retry blind
        print(f"ERROR: open_session() raised {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    # Override the DRIVER'S OWN attribute, not a second copy of the value — this is exactly
    # what _sample_read_scroll/_capture_limit_for_profile read (self.read_scroll_frac /
    # self.scroll_captures) in the no-auto-policy path this standalone tool always runs under
    # (nothing here ever calls driver.set_auto_session_policy(), so
    # driver._auto_behavior_policy() is None the whole run — see hinge.py's own
    # _sample_read_scroll/_capture_limit_for_profile for that fallback). So
    # _capture_current -> _sample_read_scroll -> _scroll_down_one -> _scroll all still run
    # completely unmodified: same jitter, same forbidden-zone guard, same ledger bookkeeping —
    # only the NUMBER fed into that path changes. Applied AFTER open_session() (harmless either
    # way — nothing in open_session() reads either attribute) so a driver that fails to open
    # never needs to have these attributes at all.
    if args.read_scroll_frac is not None:
        driver.read_scroll_frac = args.read_scroll_frac
    if args.scroll_captures is not None:
        driver.scroll_captures = args.scroll_captures
    print(f"Effective read_scroll_frac: {driver.read_scroll_frac} "
          f"({'--read-scroll-frac override' if args.read_scroll_frac is not None else f'from {args.config}'})")
    print(f"Effective scroll_captures: {driver.scroll_captures} "
          f"({'--scroll-captures override' if args.scroll_captures is not None else f'from {args.config}'})")

    # THIS PROBE MEASURES THE ORDINARY READ CADENCE, so it must not be turned into an item
    # ENUMERATION read. Until doc 5.9's observe inversion that was implicit: enumeration ran in
    # AUTO sessions only, and this tool never calls set_auto_session_policy. Observe enumerates
    # now, so the remaining session-level gate is "will an opener actually be requested" -- and
    # the honest answer for a standalone measurement tool is no. Said explicitly rather than
    # inherited, because the two knobs above (read_scroll_frac / scroll_captures) are exactly
    # what an enumeration read ignores in favour of its own per-frame step and 48-frame ceiling:
    # leaving this out would silently make every printed "effective" value above a fiction.
    #
    # Optional-hook shape (getattr + callable), matching every other caller of this family in the
    # repo: a driver double, or any future driver with no enumeration subsystem to gate, simply
    # has nothing to switch off.
    disable_enumeration = getattr(driver, "set_opener_enabled", None)
    if callable(disable_enumeration):
        disable_enumeration(False)

    try:
        neutralized = neutralize_unsafe_methods(driver)
        print(f"\nSafety guard installed — {len(neutralized)} method(s) neutralised (any "
              f"call raises {ScrollProbeGuardTripped.__name__} immediately):")
        for name in neutralized:
            print(f"  - {name}")
        if not any(m.endswith("keyevent()") for m in neutralized):
            print("  (no keyevent()-named method exists on this driver's transports today; "
                  "nothing to neutralise there)")

        # See install_ledger_capture's docstring: current_profile()'s own trailing
        # _scroll_to_top() resets driver._capture_scroll_ledger to [] before this function
        # ever gets to look at it, so the snapshot has to be taken from INSIDE that call. This
        # wraps driver._scroll_to_top only — every scroll gesture itself still goes through
        # the real, unmodified _capture_current -> _sample_read_scroll -> _scroll_down_one ->
        # _scroll chain.
        ledger_snapshots = install_ledger_capture(driver)

        try:
            reason = driver.blocked_reason()
        except Exception:  # noqa: BLE001 — blocked_reason itself must never raise; belt+braces
            reason = None
        if reason:
            print(f"\nERROR: the deck is blocked: {reason}", file=sys.stderr)
            sys.exit(1)

        # The confirmation is a convenience for a human at a terminal, not a safety control --
        # the safety here is the neutralised-method guard above plus only ever calling
        # current_profile(). So when stdin is not a TTY (piped, or driven by an agent harness),
        # skip it rather than dying on EOFError. The blocked_reason() check above already
        # refuses a paywalled/unrecognised deck, and current_profile() raises rather than
        # guessing if what is on screen is not a profile card.
        if sys.stdin.isatty():
            input("\nPress ENTER once a real profile card is on screen "
                  "(NOT the comment/Send-Like sheet) -> ")
        else:
            print("\nstdin is not a TTY -- skipping the ENTER confirmation. Proceeding on the "
                  "assumption a profile card is already on screen.")

        print("\nReading the profile now (bot-driven, humanized, bounded scrolling)...")
        try:
            profile = driver.current_profile()
        except Exception as exc:  # noqa: BLE001 — report clearly, exit non-zero, never retry blind
            print(f"\nERROR: current_profile() raised {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            sys.exit(1)

        if profile is None or not profile.photos:
            print("\nERROR: current_profile() returned no frames (deck empty, a comment "
                  "sheet was already open, or the capture was otherwise abandoned). Nothing "
                  "to measure.", file=sys.stderr)
            sys.exit(1)

        print(f"Captured {len(profile.photos)} frame(s). Saving to {out_dir} ...")
        # ledger_snapshots[-1]: the LAST _scroll_to_top() call before current_profile()
        # returned is the one that immediately followed this tool's single _capture_current
        # read (see install_ledger_capture's docstring for why there can be an earlier,
        # irrelevant, pre-capture snapshot too). [] means current_profile() never called
        # _scroll_to_top() at all this run -- reaching this line means profile.photos is
        # non-empty, and current_profile() always calls _scroll_to_top() after a non-None
        # profile, so that should not happen; treated as "not captured" rather than assumed
        # empty, per the same "don't ship an empty field that looks like data" rule.
        pre_reset_ledger = ledger_snapshots[-1] if ledger_snapshots else None
        save_capture(profile.photos, profile.meta, driver, out_dir,
                     driver_scroll_ledger=pre_reset_ledger)

        screen_size = driver.adb.screen_size()
        content_band = driver.content_band
        report = analyze_frames(profile.photos, screen_size=screen_size,
                                content_band=content_band,
                                tolerance_px=args.match_tolerance_px,
                                min_response=args.min_response)
        save_analysis(report, out_dir)
        print_report(report, driver_scroll_ledger=pre_reset_ledger)
        print(f"\nFrames + manifest.json + analysis.json written to {out_dir}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
