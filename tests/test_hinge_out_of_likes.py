"""Hinge's out-of-free-likes paywall -- the screen that hung the 2026-08-11 observe run for
2m26s (data/hinge_debug/run_20260811_011416; see hinge.py's own "BLOCKED deck" module comment,
just above _PAYWALL_MATCH_THRESHOLD, for the full incident writeup). At 01:42:36 the owner
tapped the heart, wrote "Was the water freezing?" and tapped Send Like; Hinge refused it (out of
free likes for the day) and swapped the deck for its Hinge+ upgrade screen instead, which
_await_like_resolved then polled forever because nothing in the codebase recognised it.

Two defects, both pinned here, matching hinge.py's own framing:

  D1. The paywall itself is now recognised -- _paywall_visible (glyph + position),
      _paywall_headline (best-effort OCR refinement), _deck_blocked_reason (the operator-facing
      sentence), and blocked_reason() (the memoized, never-raising, worker-facing entry point).
      Detecting it is STRICTLY PASSIVE: zero taps, swipes, or typed text, because it is a
      purchase screen and observe mode never acts on the owner's behalf.
  D2. The deeper bail-out: _observe_stuck_bail, armed once per profile-wait by
      wait_for_decision and shared by _await_like_resolved, stops ANY screen the driver cannot
      positively recognise after at least _OBSERVE_STUCK_S -- the floor of a humanized budget
      drawn fresh on every (re)arm by _observe_stuck_budget(), never the fixed constant itself
      (see that function's docstring for the measured distribution) -- the paywall is only
      today's instance of that hole. It must never fire on a human genuinely deliberating on a
      real, ready deck (the incident run's own ~5-minute read on one profile was normal,
      legitimate use).

Ground truth: ops/calibration/hinge_out_of_likes_20260811.png, a real screencap taken live on
the Pixel 7a (1080x2400) on 2026-08-11 (plus a one-off read-only uiautomator dump used only to
measure geometry -- see HINGE_SPEC's templates comment in hinge.py). Everything else here
follows tests/test_hinge_observe.py / tests/test_android_upsell_protection.py's own FakeAdb +
synthetic-frame conventions: paste the REAL shipped glyph templates onto a noise canvas so
_paywall_visible genuinely finds them via cv2, rather than mocking the driver itself.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from operation_love.drivers import hinge
from operation_love.drivers.hinge import AndroidDriver, HINGE_SPEC

_REFERENCE_SCREENSHOT = (
    Path(__file__).resolve().parent.parent / "ops" / "calibration"
    / "hinge_out_of_likes_20260811.png"
)


class FakeAdb:
    """Records what actually reached the phone. Same shape every other driver test file in
    this suite uses -- see tests/test_android_upsell_protection.py's docstring for why each
    file keeps its own copy rather than sharing one."""

    def __init__(self, frames):
        self.frames = list(frames) or [b""]
        self.i = 0
        self.taps = []
        self.swipes = []
        self.scrolls = 0
        self.texts = []

    def screen_size(self):
        return (1080, 2400)

    def devices(self):
        return ["dev"]

    def shell(self, command="", **_):
        return ""

    def screencap(self):
        return self.frames[min(self.i, len(self.frames) - 1)]

    def tap(self, x, y):
        self.taps.append((x, y))

    def swipe(self, x1, y1, x2, y2, **_k):
        self.swipes.append((x1, y1, x2, y2))

    def scroll_up(self, *_a, **_k):
        self.scrolls += 1

    def text(self, s):
        self.texts.append(s)


def _drv(adb, **overrides):
    # halt_on_error False, same reasoning as the sibling upsell-protection test module: these
    # tests are about the NEW paywall/watchdog guards, not about _verify_progress's separate
    # halt_on_error-gated path.
    class C:
        apps = {"hinge": {"halt_on_error": False, **overrides}}
    d = AndroidDriver(C(), HINGE_SPEC)
    d._adb = adb
    d._touch = adb
    return d


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)


# --- frame builders: real, template-detectable glyphs on synthetic screens -------------------

def _paywall_frame(cx=810, cy=292, seed=7):
    """A decodable frame carrying the real "HingeX" tab-wordmark glyph pasted at (cx, cy).

    The default (810, 292) IS the real tab position: HINGE_SPEC's templates comment records the
    template as cropped from x 705..915, y 254..330 of the 2026-08-11 reference screenshot, and
    (810, 292) is the centre of that rectangle -- not an arbitrary stand-in.
    """
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    canvas = rng.integers(60, 200, size=(2400, 1080), dtype=np.uint8)
    t = hinge._load_template("hinge_upgrade_tab.png")
    th, tw = t.shape
    canvas[cy - th // 2: cy - th // 2 + th, cx - tw // 2: cx - tw // 2 + tw] = t
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


def _no_paywall_frame(seed=8):
    """A decodable frame carrying NEITHER the paywall glyph nor the deck glyphs -- stands in
    for an unrecognised screen (a system dialog, an app update prompt, anything the driver has
    never seen before)."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    canvas = rng.integers(60, 200, size=(2400, 1080), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


def _deck_ready_frame(heart_xy=(930, 1600), pass_xy=(130, 2030), seed=41):
    """A decodable frame carrying BOTH the like-heart and pass-X glyphs -- an ordinary,
    positively-confirmed swipe deck (same builder shape as
    tests/test_android_upsell_protection.py's own _deck_ready_frame).

    The heart glyph is HINGE_SPEC.templates["like"] (hinge_like_button.png), not a hardcoded
    filename -- see hinge.py's templates dict comment for why hinge_heart.png (a different,
    non-like control) would be the wrong stand-in here."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    canvas = rng.integers(60, 200, size=(2400, 1080), dtype=np.uint8)
    for name, (cx, cy) in ((HINGE_SPEC.templates["like"], heart_xy), ("hinge_pass_x.png", pass_xy)):
        t = hinge._load_template(name)
        th, tw = t.shape
        canvas[cy - th // 2: cy - th // 2 + th, cx - tw // 2: cx - tw // 2 + tw] = t
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


def _blank_frame():
    """A solid-black frame -- device asleep / on the keyguard (_is_blank_frame's own
    definition: near-black AND near-uniform)."""
    import cv2
    import numpy as np
    canvas = np.zeros((2400, 1080), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


# ===============================================================================================
# 1. _paywall_visible: glyph AND position both have to be right.
# ===============================================================================================

def test_paywall_visible_true_at_the_real_tab_position():
    """Pins the positive case: the real template, pasted at the real measured tab geometry,
    must be recognised. Without this, a refactor of the matcher could silently stop detecting
    the one screen this whole file exists to catch."""
    drv = _drv(FakeAdb([b""]))
    assert drv._paywall_visible(_paywall_frame()) is True


def test_paywall_visible_false_with_no_glyph_at_all():
    """An ordinary unrecognised screen (no paywall tab chrome anywhere) must not be mistaken
    for the paywall -- a false positive here would report the wrong operator-facing reason for
    an unrelated stuck screen."""
    drv = _drv(FakeAdb([b""]))
    assert drv._paywall_visible(_no_paywall_frame()) is False


def test_paywall_visible_false_when_glyph_sits_below_the_y_gate():
    """Proves the _PAYWALL_MAX_Y_FRAC position gate is REAL, not decorative: the identical
    glyph, at the identical match score, is rejected purely because it sits low on the screen
    (y_frac ~0.50, far past the 0.30 gate) -- exactly the defensive idiom
    _observe_like_sheet_visible's own y-gate already uses. Without this gate, some unrelated
    screen that happens to carry Hinge's tab-bar-shaped chrome anywhere in its lower two-thirds
    would misfire as the paywall."""
    drv = _drv(FakeAdb([b""]))
    frame = _paywall_frame(cx=810, cy=1200)     # 1200 / 2400 = 0.50, well past the 0.30 gate
    assert drv._paywall_visible(frame) is False


# ===============================================================================================
# 2. GROUND TRUTH: the committed reference screenshot -- the real screen the owner actually hit.
# This is the single most valuable test in this file: it is the only one that would catch a
# future refactor silently breaking detection on the ACTUAL screen, as opposed to a synthetic
# stand-in built from the same template it is meant to be testing.
# ===============================================================================================

def test_paywall_visible_true_on_the_real_committed_reference_screenshot():
    """MEASURED ground truth (2026-08-11): _paywall_visible must be True, unmodified, on
    ops/calibration/hinge_out_of_likes_20260811.png -- the exact screen Hinge showed the owner
    at 01:42:43 when the "Was the water freezing?" like was refused. Every other test in this
    file builds a SYNTHETIC frame from the same template this checks against; only this one
    proves detection still works on the real device pixels."""
    pytest.importorskip("cv2")
    if not _REFERENCE_SCREENSHOT.exists():
        pytest.skip(f"reference screenshot missing at {_REFERENCE_SCREENSHOT}")
    frame = _REFERENCE_SCREENSHOT.read_bytes()
    drv = _drv(FakeAdb([frame]))
    assert drv._paywall_visible(frame) is True


# ===============================================================================================
# 3. Template discrimination as a MEASURED property (not just a location count).
# ===============================================================================================

def test_reference_screenshot_scores_well_above_threshold():
    """The reference screenshot's cv2.TM_CCOEFF_NORMED score against the "paywall" template
    must clear _PAYWALL_MATCH_THRESHOLD (0.75) with a wide margin -- MEASURED 2026-08-11 at
    1.000 live, 0.965..1.000 under gain/bias perturbation. A score that merely limped over 0.75
    would be one rendering-mode change away from a false negative on the real screen."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    if not _REFERENCE_SCREENSHOT.exists():
        pytest.skip(f"reference screenshot missing at {_REFERENCE_SCREENSHOT}")
    frame = _REFERENCE_SCREENSHOT.read_bytes()
    image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    template = hinge._load_template("hinge_upgrade_tab.png")
    score = float(cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED).max())
    assert score > 0.9, f"reference screenshot only scored {score}, expected near-perfect"


def test_plain_noise_frame_scores_well_below_threshold():
    """The negative-side half of the same measured property: a frame with no paywall chrome at
    all must score far under 0.75 -- MEASURED 2026-08-11 at a maximum of 0.4903 across all 88
    real non-paywall frames of the hung incident run, so this pins the same wide margin on the
    other side of the threshold, not just a bare pass/fail."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    image = cv2.imdecode(np.frombuffer(_no_paywall_frame(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    template = hinge._load_template("hinge_upgrade_tab.png")
    score = float(cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED).max())
    assert score < 0.5, f"noise frame scored {score}, too close to the 0.75 threshold"


# ===============================================================================================
# 4. blocked_reason(): the memoized, worker-facing, never-raising entry point.
# ===============================================================================================

def test_blocked_reason_mentions_likes_on_the_real_paywall():
    """On the real reference screenshot, blocked_reason() must return a non-empty string that
    actually names the problem ("likes") -- the OCR-refined message the operator reads on the
    hub, rather than a generic "something is blocking the deck" that would have looked no more
    informative than the 2.5-minute hang this fixes."""
    pytest.importorskip("cv2")
    if not _REFERENCE_SCREENSHOT.exists():
        pytest.skip(f"reference screenshot missing at {_REFERENCE_SCREENSHOT}")
    frame = _REFERENCE_SCREENSHOT.read_bytes()
    drv = _drv(FakeAdb([frame]))
    reason = drv.blocked_reason()
    assert reason
    assert "likes" in reason.lower()


def test_blocked_reason_none_on_an_ordinary_deck_frame():
    """A normal, healthy swipe deck must never be reported as blocked -- the failure mode
    _deck_blocked_reason's own docstring calls out for a false positive would stop a perfectly
    healthy run."""
    drv = _drv(FakeAdb([_deck_ready_frame()]))
    assert drv.blocked_reason() is None


def test_blocked_reason_none_on_a_blank_asleep_frame():
    """blocked_reason()'s own documented contract: on_blank="none" -- a blank/asleep screen is
    the owner having stepped away, not a blocked deck. Reporting one as blocked would stop a
    healthy run the moment the phone's screen timed out."""
    adb = FakeAdb([_blank_frame()])
    drv = _drv(adb)
    assert drv.blocked_reason() is None


def test_blocked_reason_never_raises_on_undecodable_bytes():
    """blocked_reason()'s contract is that it NEVER raises -- worker.py calls it on every loop
    iteration of both the observe and auto loops. Undecodable garbage (an ADB hiccup, a partial
    frame) must degrade to None, not propagate an exception into the worker's hot loop."""
    adb = FakeAdb([b"not a real png at all"])
    drv = _drv(adb)
    assert drv.blocked_reason() is None


# ===============================================================================================
# 5. PASSIVITY: the paywall is a PURCHASE screen. Detecting it must never touch the phone.
# ===============================================================================================

def test_detecting_the_paywall_issues_zero_taps_swipes_or_text():
    """The assertion that stops a future "helpfully dismiss the paywall" change. Observe mode
    is strictly passive, and this screen doubly so -- it is Hinge's own purchase flow, and the
    owner rule is that paid controls are manual, always. Running the full wait_for_decision
    path against a persistent paywall frame must leave the phone completely untouched."""
    adb = FakeAdb([_paywall_frame()])
    drv = _drv(adb)
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 5      # a few polls, then end the wait deterministically

    drv.wait_for_decision(timeout=None, should_stop=should_stop)

    assert adb.taps == []
    assert adb.swipes == []
    assert adb.texts == []


# ===============================================================================================
# 6. wait_for_decision returns None -- never True, never False -- while the paywall is up.
# ===============================================================================================

def test_wait_for_decision_never_returns_true_or_false_on_a_persistent_paywall():
    """Pins the exact bug this fix targets: in the incident, the like Hinge REFUSED must never
    be recorded as sent (True) and the profile must never be recorded as passed (False) just
    because the screen changed size/shape enough to look like SOME kind of advance. A phantom
    label here would corrupt the taste model with a decision that never actually happened."""
    adb = FakeAdb([_paywall_frame()])
    drv = _drv(adb)
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 5

    result = drv.wait_for_decision(timeout=None, should_stop=should_stop)
    assert result is None


# ===============================================================================================
# 7. THE WATCHDOG, both directions. Clock driven deterministically via monkeypatched
# hinge.time.monotonic -- never via real sleeping.
# ===============================================================================================

def test_unrecognized_static_screen_bails_out_once_the_stuck_budget_is_exceeded(monkeypatch):
    """The fix for the deeper defect: before _observe_stuck_bail existed, worker.py's
    wait_for_decision(timeout=None) meant ANY screen the driver could not classify -- a
    paywall, a system dialog, an app update prompt, a crash to the launcher -- hung the run
    forever. A static, unrecognisable screen (no deck glyphs at all, nothing moving) must now
    give up once its drawn stuck budget has elapsed with nothing positively recognised, and must
    publish a reason through blocked_reason() naming the ACTUAL elapsed time (the budget is a
    human_cooldown draw now -- see hinge._observe_stuck_budget -- not the fixed _OBSERVE_STUCK_S
    constant, so the message can no longer name a constant; monkeypatched here to a deterministic
    value so the bound and the message are both exactly pinned rather than racing real
    randomness)."""
    clock = [10_000.0]
    monkeypatch.setattr(hinge.time, "monotonic", lambda: clock[0])
    stuck_budget = 48.0
    monkeypatch.setattr(hinge, "_observe_stuck_budget", lambda: stuck_budget)
    adb = FakeAdb([_no_paywall_frame()])
    drv = _drv(adb)
    stride = stuck_budget / 3
    calls = {"n": 0}

    def should_stop():
        # Advance the clock in strides comfortably bigger than the watchdog's re-probe cadence
        # (_OBSERVE_STUCK_CHECK_S, still humanized around that anchor even with the budget
        # pinned above) so every poll still triggers exactly one deck-ready reprobe -- keeping
        # the real cv2 cost proportional to iteration COUNT, not to how much simulated time each
        # iteration covers -- while reaching the stuck budget in only a handful of real polls.
        # should_stop itself never fires (always False): only the watchdog is allowed to end
        # this wait.
        calls["n"] += 1
        clock[0] += stride
        return False

    result = drv.wait_for_decision(timeout=None, should_stop=should_stop)
    assert result is None
    # should_stop's FIRST call happens inside _await_live_frame, BEFORE the watchdog is even
    # armed (this frame builder is always immediately decodable and non-blank, so that call
    # happens exactly once, ahead of the main wait loop); every later call happens inside the
    # main loop, one per poll, each stride AFTER the watchdog armed. So elapsed-SINCE-ARM -- the
    # value the driver itself measures and formats into the message -- is (total calls - 1)
    # strides, not (clock[0] - the test's own start time), which would double count the pre-arm
    # call and never match what got printed.
    elapsed_since_arm = (calls["n"] - 1) * stride
    assert elapsed_since_arm > stuck_budget
    reason = drv.blocked_reason()
    # The generic stuck-screen message now names the ACTUAL elapsed time (it varies per draw in
    # production), not the _OBSERVE_STUCK_S constant -- see hinge._format_stall_duration for the
    # "45s" / "1m37s" style this reuses from bugreport.py.
    assert reason and hinge.format_duration(elapsed_since_arm) in reason


def test_deck_ready_static_screen_never_bails_even_well_past_the_stuck_budget(monkeypatch):
    """The regression test that protects normal use. The incident run had a legitimate ~5
    minute deliberation on one profile before anything went wrong; the watchdog must not
    interrupt a human doing exactly that. A STATIC frame that positively shows a ready deck
    (both like and pass glyphs visible, nothing moving) re-arms the budget (a FRESH
    human_cooldown draw each time -- see hinge._observe_stuck_budget -- left un-mocked here on
    purpose) every ~_OBSERVE_STUCK_CHECK_S via the no_change fast path's deck-ready probe, so it
    must survive FAR longer than _OBSERVE_STUCK_S -- the drawn budget's floor, and therefore a
    lower bound on every possible draw -- without the watchdog ever firing."""
    clock = [20_000.0]
    monkeypatch.setattr(hinge.time, "monotonic", lambda: clock[0])
    adb = FakeAdb([_deck_ready_frame()])
    drv = _drv(adb)
    calls = {"n": 0}
    _survive_past = 3 * hinge._OBSERVE_STUCK_S

    def should_stop():
        calls["n"] += 1
        # A stride equal to _OBSERVE_STUCK_S is still comfortably below ANY possible drawn
        # budget (which can never fall below that same floor -- human_cooldown's guarantee), and
        # still comfortably above the ~_OBSERVE_STUCK_CHECK_S probe cadence, so this reaches
        # several multiples of the floor in a handful of real polls instead of dozens, without
        # changing what's being proven: every poll re-arms the budget because the deck stays
        # positively confirmed ready.
        clock[0] += hinge._OBSERVE_STUCK_S
        return clock[0] - 20_000.0 > _survive_past

    result = drv.wait_for_decision(timeout=None, should_stop=should_stop)
    assert result is None                 # ended by should_stop, not by any decision
    assert clock[0] - 20_000.0 > _survive_past, (
        "the wait must have survived multiple stuck-budgets for this to prove anything")
    assert drv.blocked_reason() is None   # the watchdog never fired at any point in the wait
    assert adb.taps == [] and adb.swipes == []


# ===============================================================================================
# 8. _await_like_resolved's deliberate ASYMMETRY -- like_sheet (human composing) never times
# out; like_sending (Hinge's own network/animation state) does. This is exactly where the
# 2026-08-11 incident hung.
# ===============================================================================================

def test_like_sheet_wait_does_not_time_out_while_the_human_is_composing(monkeypatch):
    """A persistent open like sheet (the human writing a comment) must re-arm the stuck-screen
    budget -- a FRESH human_cooldown draw every poll, see hinge._observe_stuck_budget, left
    un-mocked here on purpose since the floor guarantee alone is what the assertion below relies
    on -- and never time out on its own -- composing is human-paced, and the audited run
    measured a genuine 3m44s compose window. Driven well past 3x _OBSERVE_STUCK_S (the floor,
    and therefore a lower bound on every possible draw of the actual budget) to prove the
    watchdog never fires while the sheet stays open."""
    clock = [30_000.0]
    monkeypatch.setattr(hinge.time, "monotonic", lambda: clock[0])
    adb = FakeAdb([b"sheet"])
    drv = _drv(adb)
    drv._observe_last_recognized = clock[0]
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda frame: True)
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        clock[0] += hinge._OBSERVE_POLL_S
        return clock[0] - 30_000.0 > 3 * hinge._OBSERVE_STUCK_S

    sent, _notified = drv._await_like_resolved(b"base", None, should_stop)
    assert sent is None                   # ended by should_stop, never fabricated True/False
    assert clock[0] - 30_000.0 > 3 * hinge._OBSERVE_STUCK_S, (
        "must have survived past 3x the stuck budget floor with the sheet open the whole time")


def test_like_sending_wait_times_out_and_returns_none_not_a_fabricated_like(monkeypatch):
    """The exact state the incident hung in: the sheet has closed but Hinge has neither
    returned to the current card nor shown a ready next deck -- 'like_sending' -- and this is
    the APP working, not a human, so it must NOT re-arm the budget the way like_sheet does (no
    _observe_recognized() call anywhere on this path, so no fresh draw ever happens once armed).
    It must give up once its drawn stuck budget elapses and return (None, ...), never
    (True, ...): a fabricated True here is exactly the bug -- Hinge had REFUSED the like and
    this loop must never invent one that was sent. hinge._observe_stuck_budget is monkeypatched
    to a fixed value here (mirroring what _observe_last_recognized's manual arm below stands in
    for) so the exact bound is pinned rather than racing real randomness."""
    clock = [40_000.0]
    monkeypatch.setattr(hinge.time, "monotonic", lambda: clock[0])
    stuck_budget = 55.0
    monkeypatch.setattr(hinge, "_observe_stuck_budget", lambda: stuck_budget)
    adb = FakeAdb([b"sending"])
    drv = _drv(adb)
    drv._observe_last_recognized = clock[0]
    drv._observe_stuck_budget_s = stuck_budget
    monkeypatch.setattr(drv, "_observe_like_sheet_visible", lambda frame: False)
    monkeypatch.setattr(drv, "_is_current_profile_frame", lambda frame, **_k: False)
    monkeypatch.setattr(drv, "_observe_deck_ready", lambda frame: False)

    def should_stop():
        # Would run far past the stuck budget if should_stop were the only way out --
        # the watchdog inside _await_like_resolved must end it first.
        clock[0] += hinge._OBSERVE_POLL_S
        return clock[0] - 40_000.0 > 3 * stuck_budget

    sent, _notified = drv._await_like_resolved(b"base", None, should_stop)
    assert sent is None
    elapsed = clock[0] - 40_000.0
    assert elapsed <= stuck_budget + hinge._OBSERVE_POLL_S, (
        f"like_sending must time out at ~{stuck_budget}s via the watchdog, not run "
        f"to the 3x-budget should_stop ceiling (elapsed {elapsed}s)"
    )


# ===============================================================================================
# 9. _observe_stuck_budget itself: the drawn watchdog budget, pinned as a genuine distribution
# property rather than a single sample -- the owner's uniform-humanization rule (see hinge.py's
# _OBSERVE_STUCK_S comment) applies here exactly like it does everywhere else this repo
# randomizes a timing parameter, e.g.
# test_dismiss_via_zone_point_stays_inside_the_measured_safe_band_across_many_draws in
# tests/test_android_upsell_protection.py.
# ===============================================================================================

def test_observe_stuck_budget_never_below_the_floor_and_genuinely_spread():
    """The floor guarantee is the ENTIRE safety argument for humanizing this particular budget
    (see hinge._observe_stuck_budget's own docstring): human_cooldown never returns below its
    anchor, so the watchdog can never fire earlier than _OBSERVE_STUCK_S and cut short a
    legitimate human deliberation -- exactly the failure mode the two tests above this section
    guard against. Pinned across 300 independent draws, both directions: every single draw
    clears the floor, AND the draws are not the same fixed number 300 times over (which would
    just be _OBSERVE_STUCK_S wearing a randomizer-shaped costume)."""
    draws = [hinge._observe_stuck_budget() for _ in range(300)]
    assert all(d >= hinge._OBSERVE_STUCK_S for d in draws)
    assert len({round(d, 3) for d in draws}) > 5
