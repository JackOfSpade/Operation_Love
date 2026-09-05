"""The automated bound campaign must never spend a gesture on a profile it measured nothing on.

Four properties carry the weight here.

The first is the CLI refusal matrix: this tool spends a real Pass or a real, permanent Like on
every profile, so the circular-risk phrase, the per-run advance choice and the separate like
phrase must each be demanded before a line of config is read or a device is opened.

The second is the label rule, exercised as a truth table and then over frames.

The third is the 2026-08-21 REGRESSION.  That run captured five real profiles, persisted zero
frames, and passed all five anyway, one per minute, because the frame source was hooked on
`HingeDriver._index_captured_items` -- which `_capture_current` only calls on an ENUMERATION
read, which `_item_enumeration_blocker` disables whenever `apps.hinge.targeting_calibration` is
unconfigured.  The real `actions.jsonl` from that run is the fixture shape used below
(`_REAL_*`): nine-frame ordinary-cadence reads, `items = 0`, `ranker_photos = null`, and the
exact `items_unavailable` sentence.  Two things are asserted about it: the card rects now come
from per-frame segmentation and no longer depend on that gate at all, and -- belt and braces --
a profile that still ends up with no persisted frame or no measurable card HALTS before any
advance, with no strike allowance.

The fourth is that a signal writes the incomplete manifest, because that run was `pkill`ed and
lost everything.

No test here opens a device, runs adb, or sleeps; the driver is mocked in full.  The one test
that reads the real screencaps of the failed run skips itself when that gitignored directory is
absent, so a clean checkout is unaffected.
"""
from __future__ import annotations

import hashlib
import json
import signal
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from operation_love import targeting_policy
from operation_love.private_files import ensure_private_dir
from tools import hinge_video_bound as bound
from tools import hinge_video_bound_auto as auto

# Geometry of the real Pixel 7a frames in data/hinge_debug/run_20260821_163736, kept as
# constants so the synthetic fixtures below have the same shape as what actually failed.
_W, _H = 1080, 2400
_CARD_A = (53, 697, 1027, 1671)
_CARD_B = (53, 937, 1027, 1911)
_BAND = (0.125, 0.875)
# A rect far enough below the content-band centre to sit outside the autoplay trigger zone.
_CARD_OFF_CENTER = (53, 1800, 1027, 2300)
# What one fake read-scroll moves the content, in px. Big enough that a centred card leaves the
# 0.15 autoplay band (which is 270px of the 1800px content band here) and that the corrective
# stroke back is a legal read-scroll rather than one the envelope refuses, so a re-attach probe
# can actually complete against this fake and its round trip nets to zero displacement.
_FAKE_SCROLL_PX = 600
# Verbatim from that run's five `capture` records: the product-policy gate that made the old
# frame source unreachable. Any future frame source that depends on it will fail this file.
_REAL_ITEMS_UNAVAILABLE = (
    "apps.hinge.targeting_calibration is unavailable (not configured in config.yaml), so a "
    "model-selected item could not be verified or targeted and no numbered item list would have "
    "a usable consumer")
_REAL_FRAMES_PER_PROFILE = 9
# Absolute, resolved at import: the `repo` fixture chdirs into a tmp dir, and a relative path
# here would make the real-pixel replay below skip itself silently -- which is the one test that
# actually exercises live perception.
_DEBUG_RUN = Path(__file__).resolve().parent.parent / "data" / "hinge_debug" / "run_20260821_163736"


# =====================================================================================
# synthetic frames and a driver that behaves like a scrolling phone
# =====================================================================================

def _frame(fill: int = 40, *, marks=()) -> bytes:
    canvas = np.full((_H, _W, 3), fill, np.uint8)
    for row, col, value in marks:
        canvas[row, col] = value
    ok, buffer = cv2.imencode(".png", canvas)
    assert ok
    return buffer.tobytes()


def _tiny(fill: int = 40, *, marks=()) -> bytes:
    """A card-rect crop as this tool persists them: the crop IS the content, band (0, 1)."""
    canvas = np.full((32, 16, 3), fill, np.uint8)
    for row, col, value in marks:
        canvas[row, col] = value
    ok, buffer = cv2.imencode(".png", canvas)
    assert ok
    return buffer.tobytes()


def _still_page(fill: int = 40):
    """A page whose pixels never change: a still photo under C2."""
    frame = _frame(fill)
    return lambda _read: frame


def _video_page(fill: int = 60):
    """A page that emits a new frame on every read: a video playing under the dwell."""
    return lambda read: _frame(fill, marks=[(1000, 500, (read * 29) % 255)])


def _clean_matcher(_frame, _rect):
    """A mute screen that RAN over a complete ROI and matched nothing."""
    return True, 0.10


def _hit_matcher(_frame, _rect):
    """A near-perfect app-UI match: affirmative video evidence."""
    return True, 0.995


class _FakeDriver:
    """Only the guarded driver surface this tool may use, plus a call/gesture ledger.

    `_screencap` answers from the CURRENT SCROLL DEPTH rather than from a call counter, which is
    how a real phone behaves: holding still returns the same page (or a playing video's next
    frame), and `_scroll_down_one` is what moves to the next one.
    """

    dwell_s = 1.0
    content_band = (0.125, 0.875)

    def __init__(self, *, profiles, matcher=_clean_matcher, blocked=(), top_reasons=(),
                 session_top=True, like_template="LIKE-TEMPLATE", on_gesture=None,
                 deck_ready="always"):
        self._profiles = list(profiles)      # [[page, page, ...], ...] one list per profile
        self._match_video_mute = matcher
        self._blocked = list(blocked)
        self._top_reasons = list(top_reasons)
        self._session_top = session_top
        # "always" | "top" (only at scroll top, like the real floating heart) | "never"
        self._deck_ready = deck_ready
        self._like_template = like_template
        self._on_gesture = on_gesture
        self._pages = []
        self._depth = 0
        self._reads = 0
        # Where the content sits, and where it sat for every frame handed out. The real
        # `estimate_shift` answers about a PAIR of frames, so the fake has to remember positions
        # rather than accumulate a running delta: a probe measures its exit against the station
        # anchor, and a running delta would silently fold in the read-scroll that came before it.
        self._content_px = 0
        self._screencaps: list[tuple[bytes, int]] = []
        self.calls: list[str] = []
        self.gestures: list[str] = []
        self.profile_no = 0

    # --- perception -------------------------------------------------
    def _ensure_session_top(self, should_stop=None) -> bool:
        self.calls.append("ensure_session_top")
        self._load_next_profile()
        return self._session_top

    def _load_next_profile(self) -> None:
        self._pages = self._profiles.pop(0) if self._profiles else []
        self._depth = 0
        self._reads = 0
        self._content_px = 0
        self._screencaps = []
        self.profile_no += 1

    def _observe_deck_ready(self, frame) -> bool:
        """The same predicate `_require_deck_confirmed` uses, asked passively."""
        self.calls.append("observe_deck_ready")
        if self._deck_ready == "always":
            return True
        if self._deck_ready == "never":
            return False
        return self._depth == 0

    def blocked_reason(self):
        self.calls.append("blocked_reason")
        return self._blocked.pop(0) if self._blocked else None

    def _confirm_enumeration_top(self) -> str:
        self.calls.append("confirm_top")
        return self._top_reasons.pop(0) if self._top_reasons else ""

    def _template(self, role):
        return self._like_template if role == "like" else None

    def _screencap(self, *, on_blank="raise"):
        self.calls.append("screencap")
        if not self._pages:
            raise AssertionError("the fake driver ran out of pages; the test script is wrong")
        page = self._pages[min(max(self._depth, 0), len(self._pages) - 1)]
        self._reads += 1
        payload = page(self._reads) if callable(page) else page
        self._screencaps.append((payload, self._content_px))
        return payload

    def _sample_read_step(self, depth, complexity_hint):
        return 0.05, 0.55, 0.5

    def _scroll_down_one(self, frac=None, x_frac=None):
        self.calls.append("scroll_down_one")
        self._depth += 1
        self._reads = 0
        self._content_px += _FAKE_SCROLL_PX

    def _scroll_up_one(self, frac, x_frac):
        # Deliberately NOT clamped at zero: a re-attach probe scrolls up and straight back down,
        # and a clamp would turn that round trip into a net page advance, which is the one thing
        # the probe must not do. Only the page LOOKUP is clamped, in `_screencap`.
        self.calls.append("scroll_up_one")
        self._depth -= 1
        self._reads = 0
        self._content_px -= _FAKE_SCROLL_PX

    def _measured_page_shift(self, before, after):
        """The driver's frameshift wrapper: how far the page moved between two frames.

        Answered from the recorded positions of the two frames, so a chained measurement adds up
        the way the real estimator's does. A frame this phone never handed out is `None`, which
        is the estimator's own "I cannot tell" and is what every caller fails closed on.
        """
        self.calls.append("measured_page_shift")
        at_after = next((px for payload, px in reversed(self._screencaps) if payload == after),
                        None)
        earlier = self._screencaps[:-1]
        at_before = next((px for payload, px in reversed(earlier) if payload == before), None)
        if at_after is None or at_before is None:
            return None
        return at_after - at_before

    def _scroll_to_top(self, should_stop=None) -> bool:
        self.calls.append("scroll_to_top")
        self._depth = 0
        self._reads = 0
        self._content_px = 0
        return True

    # --- the only two gestures ---------------------------------------
    def dislike(self):
        # The real driver refuses here, before any tap, when the deck is not confirmed.
        if not self._observe_deck_ready(None):
            from operation_love.drivers.hinge import UnconfirmedScreenError

            raise UnconfirmedScreenError(
                "hinge: refusing to decide -- the swipe deck (like heart + pass X) is not "
                "positively confirmed on screen.")
        if self._on_gesture is not None:
            self._on_gesture("pass")
        self.gestures.append("pass")
        self._load_next_profile()

    def like(self, *_a, **_k):
        if self._on_gesture is not None:
            self._on_gesture("like")
        self.gestures.append("like")
        self._load_next_profile()

    def close(self):
        self.calls.append("close")


class _FakeTime:
    """A fake clock whose SLEEPS ACTUALLY PASS TIME.

    A clock that advances per read regardless of sleeping cannot test a schedule at all: the
    2026-08-21 burst-span defect is precisely a question of whether the sleeps happened, so the
    double has to model them. `step` is the small cost of reading the clock itself, which keeps
    stamps strictly increasing the way a real monotonic clock does.
    """

    def __init__(self, step: float = 0.001):
        self.now, self.step = 0.0, step
        self.slept: list[float] = []

    def clock(self) -> float:
        value = self.now
        self.now += self.step
        return value

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0, "a scheduler must never ask to sleep backwards"
        self.slept.append(float(seconds))
        self.now += float(seconds)


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ops" / "calibration").mkdir(parents=True)
    return tmp_path


def _out(repo: Path, name: str = "cap") -> Path:
    return ensure_private_dir(repo / "ops" / "calibration" / name)


def _one_card(_frame):
    return [_CARD_A]


def _no_card(_frame):
    return []


def _run(repo, driver, *, advance="pass", max_profiles=1, classify=lambda _c: "photo",
         target_videos=999, target_photos=999, out=None, segment=_one_card,
         signal_module=None, fake_time=None):
    fake_time = fake_time or _FakeTime()
    return auto.run_capture(
        out_dir=out or _out(repo), driver=driver, advance=advance, serial="PIXEL7A",
        config_sha256=None, target_videos=target_videos, target_photos=target_photos,
        max_profiles=max_profiles, hinge_version_name="10.0.1",
        driver_content_band=driver.content_band, print_fn=lambda *_a: None,
        sleep_fn=fake_time.sleep, clock=fake_time.clock, classify=classify, segment=segment,
        signal_module=signal_module or types.SimpleNamespace())


# =====================================================================================
# structural
# =====================================================================================

_FORBIDDEN_SOURCE = ("input tap", "input swipe", "input keyevent", "input text", "sendevent",
                     "exec-out", "adb shell", "subprocess")


def test_module_never_reaches_the_phone_except_through_the_driver():
    source = Path(auto.__file__).read_text()
    for token in _FORBIDDEN_SOURCE:
        assert token not in source, token
    assert "driver._screencap" in source and "driver._scroll_down_one" in source


def test_the_frame_source_no_longer_touches_the_enumeration_hook():
    """The 2026-08-21 root cause, pinned as CALL shapes.

    The prose in this module deliberately discusses the old hook at length, so the check is on
    what the code actually invokes: nothing may call the enumeration hook or the read that gates
    it, and the rects must come from the per-frame segmenter.
    """
    source = Path(auto.__file__).read_text()
    for call in ("_index_captured_items(", "._capture_current(", "_current_item_index",
                 "still_photo_dwell_evidence("):
        assert call not in source, call
    assert "segment_frame(" in source and "dwell_exact_over_rect(" in source


def test_circular_phrases_agree_with_the_policy_module_and_the_sibling_tool():
    assert auto.CIRCULAR_CHANNEL == bound.CIRCULAR_GROUND_TRUTH_CHANNEL
    assert auto.CIRCULAR_ACCEPTANCE == bound.CIRCULAR_ACCEPTANCE
    for name, value in (("STILL_PHOTO_BOUND_CIRCULAR_CHANNEL", auto.CIRCULAR_CHANNEL),
                        ("STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE", auto.CIRCULAR_ACCEPTANCE)):
        if hasattr(targeting_policy, name):
            assert getattr(targeting_policy, name) == value, name


def test_like_confirmation_is_the_same_phrase_the_calibration_tool_demands():
    calibrate = pytest.importorskip("tools.hinge_calibrate")
    assert auto.LIKE_CONFIRMATION == calibrate._SEND_LIKE_CONFIRMATION


# =====================================================================================
# CLI refusal matrix
# =====================================================================================

@pytest.fixture()
def no_device(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("config/device access happened before the confirmation check")

    monkeypatch.setattr(auto, "build_driver", _boom)
    monkeypatch.setattr(bound, "_load_config_mapping", _boom)
    monkeypatch.setattr(bound, "_resolve_serial", _boom)
    monkeypatch.setattr(bound, "_private_out_dir", _boom)
    # found+fixed 2026-08-22: `_refuse_unconfirmed` used to run INSIDE the device lock
    # (`run_holding_the_device`), so an unconfirmed invocation took the operator's real lock
    # first and only then refused -- a live campaign already holding it turned every such mistake
    # into "Android device is already in use..." instead of the confirmation error the operator
    # actually needed. Booming the lock function itself, not just what runs after it acquires
    # the lock, is what makes this fixture (and every parametrized case below that uses it) catch
    # a regression of that ordering rather than only the driver/config calls deeper inside.
    monkeypatch.setattr(auto, "run_holding_the_device", _boom)
    return monkeypatch


_BASE = ["capture", "--out", "ops/calibration/x"]


@pytest.mark.parametrize("argv, expected", [
    (_BASE + ["--advance", "pass"], "confirmation"),
    (_BASE + ["--advance", "pass", "--confirmation", "yes"], "confirmation"),
    (_BASE + ["--advance", "pass", "--confirmation",
              auto.CIRCULAR_ACCEPTANCE.lower()], "confirmation"),
    (_BASE + ["--advance", "like", "--confirmation", auto.CIRCULAR_ACCEPTANCE], "like"),
    (_BASE + ["--advance", "like", "--confirmation", auto.CIRCULAR_ACCEPTANCE,
              "--like-confirmation", "sure"], "like"),
])
def test_capture_refuses_before_touching_anything(argv, expected, no_device, capsys):
    with pytest.raises(SystemExit) as exit_info:
        auto.main(argv)
    assert exit_info.value.code != 0
    assert expected in capsys.readouterr().err


def test_capture_refuses_before_taking_the_device_lock(capsys, monkeypatch):
    """found+fixed 2026-08-22: `_capture_command` used to acquire the device lock and only THEN
    call `_refuse_unconfirmed` (inside `_capture_command_unlocked`), so a live campaign already
    holding the lock made an unconfirmed invocation print the lock's contention error instead of
    the confirmation error the operator actually needed to see. Pin the order directly, without
    `no_device`'s broader net, by making the lock function itself the only thing that can fail:
    if `capture` ever reaches it before refusing, this raises instead of exiting cleanly."""
    def _boom(*_a, **_kw):
        raise AssertionError("the device lock was acquired before the confirmation check")

    monkeypatch.setattr(auto, "run_holding_the_device", _boom)

    with pytest.raises(SystemExit) as exit_info:
        auto.main(_BASE + ["--advance", "pass"])
    assert exit_info.value.code != 0
    assert "confirmation" in capsys.readouterr().err


def test_capture_refuses_a_missing_advance_action(no_device, capsys):
    """The advance is chosen per run and has no default, so argparse itself must refuse."""
    with pytest.raises(SystemExit) as exit_info:
        auto.main(_BASE + ["--confirmation", auto.CIRCULAR_ACCEPTANCE])
    assert exit_info.value.code != 0
    assert "--advance" in capsys.readouterr().err


def test_capture_refuses_an_advance_action_that_is_not_pass_or_like(no_device, capsys):
    with pytest.raises(SystemExit) as exit_info:
        auto.main(_BASE + ["--advance", "superlike", "--confirmation", auto.CIRCULAR_ACCEPTANCE])
    assert exit_info.value.code != 0
    assert "--advance" in capsys.readouterr().err


def test_a_config_that_will_not_validate_never_buys_an_unlocked_capture(repo, monkeypatch,
                                                                       capsys):
    """A confirmed capture must yield the phone to a run holding the lock -- config or not.

    ``holding_the_device`` has a documented escape hatch: a config path that will not load
    yields an UN-HELD context, on the assumption the caller is about to load the same file and
    report the problem in its own words. Passing ``args.config`` armed that hatch here for no
    benefit -- this command revalidates the config itself either way (``build_driver`` ->
    ``cfg_mod.load``), and everything before that point, including ``_resolve_serial``'s
    ``adb devices``, would have run UNLOCKED for a config that parses as YAML but fails
    validation. An unlocked device-enumeration window beside a live hub run is not what the
    lock was added for (run ``a01fbcd1e9a0``'s false Pass was two drivers on one phone).

    The holder takes the lock WITH a config, the way a production run does, while the tool takes
    it with none -- so this also pins that the config-free path lands on the same file.
    """
    from operation_love import config as config_mod
    from operation_love import supervisor as sup

    monkeypatch.setattr(sup, "_ANDROID_LOCK_ROOT", repo / "locks")
    # Valid YAML, rejected by config.load: a config caught mid-edit, on a key neither this tool
    # nor the lock reads. Asserted rather than assumed -- the hazard only exists for configs
    # that parse but do not validate, so a file that started loading cleanly would make this
    # test pass vacuously.
    half_repaired = repo / "half-repaired.yaml"
    half_repaired.write_text("enabled_apps: [hinge]\napps: {hinge: {serial: PIXEL7A}}\n"
                             "opener: {max_chars: -5}\n")
    with pytest.raises(Exception):
        config_mod.load(str(half_repaired))
    good = repo / "config.yaml"
    good.write_text("enabled_apps: [hinge]\napps: {hinge: {serial: PIXEL7A}}\n")
    # The first device reach inside the lock, and the one the old escape hatch left exposed.
    monkeypatch.setattr(bound, "_resolve_serial",
                        lambda *a, **k: pytest.fail("the capture enumerated devices while "
                                                    "another run held the phone"))
    monkeypatch.setattr(auto, "build_driver",
                        lambda *a, **k: pytest.fail("a driver session was opened while another "
                                                    "run held the phone"))

    with sup.exclusive_android_device(config_mod.load(str(good)), "hinge"):
        with pytest.raises(SystemExit) as exit_info:
            auto.main(_BASE + ["--advance", "pass", "--confirmation", auto.CIRCULAR_ACCEPTANCE,
                               "--config", str(half_repaired)])

    assert exit_info.value.code == 1
    assert "already in use by another Operation Love run" in capsys.readouterr().err


def test_both_phrases_together_pass_the_gate_for_a_real_like_run():
    args = types.SimpleNamespace(
        confirmation=auto.CIRCULAR_ACCEPTANCE, advance="like",
        like_confirmation=auto.LIKE_CONFIRMATION, target_videos=60, target_photos=60,
        max_profiles=None)
    assert auto._refuse_unconfirmed(args) is None


@pytest.mark.parametrize("field, value", [("target_videos", 0), ("target_photos", 0),
                                          ("max_profiles", 0)])
def test_degenerate_budgets_are_refused(field, value):
    args = types.SimpleNamespace(
        confirmation=auto.CIRCULAR_ACCEPTANCE, advance="pass", like_confirmation="",
        target_videos=60, target_photos=60, max_profiles=None)
    setattr(args, field, value)
    with pytest.raises(auto.AutoBoundRefused):
        auto._refuse_unconfirmed(args)


# =====================================================================================
# perception preflight: refuse before spending, not after five profiles
# =====================================================================================

def test_preflight_refuses_a_build_that_cannot_segment_a_card():
    no_template = _FakeDriver(profiles=[], like_template=None)
    with pytest.raises(auto.AutoBoundRefused, match="'like' glyph template"):
        auto.preflight_perception(no_template)
    no_band = _FakeDriver(profiles=[])
    no_band.content_band = None
    with pytest.raises(auto.AutoBoundRefused, match="content_band"):
        auto.preflight_perception(no_band)


def test_preflight_does_not_require_the_gate_that_broke_the_live_run():
    """targeting_calibration / openers gate numbering for a MODEL; this campaign numbers nothing."""
    driver = _FakeDriver(profiles=[])
    driver.targeting_calibration = None
    driver._openers_enabled = False
    assert auto.preflight_perception(driver) is None


# =====================================================================================
# the label rule, as a truth table
# =====================================================================================

# Every re-attach leg passing, so one at a time can be spoiled below.
_PROBED = dict(reattach_probe_ran=True, reattach_dwell_exact=True,
               reattach_mute_screens_complete=True, reattach_centered=True)


@pytest.mark.parametrize("glyph, exact, clean, verdict, centered, expected", [
    (True, None, None, None, None, "video"),            # a glyph hit outranks everything
    (True, True, True, "photo", True, "video"),
    (False, False, True, "photo", True, "video"),        # moved with nothing touching the screen
    (False, False, False, "unknown", False, "video"),
    (False, True, True, "photo", True, "photo"),         # the only combination that yields photo
    (False, True, True, "written", True, "written"),     # a prompt card, terminal
    (False, True, True, "written", False, "written"),    # ...and it needs no autoplay argument
    (False, True, True, "unknown", True, "unsure"),
    (False, True, True, "photo", False, "unsure"),       # off-centre: autoplay precondition fails
    (False, True, True, "photo", None, "unsure"),
    (False, True, False, "photo", True, "unsure"),       # a mute screen that did not run clean
    (False, None, True, "photo", True, "unsure"),        # no dwell observed this card
])
def test_label_truth_table(glyph, exact, clean, verdict, centered, expected):
    label, reason = auto.label_card(glyph_hit=glyph, dwell_exact=exact,
                                    mute_screens_complete=clean, classifier_verdict=verdict,
                                    centered=centered, **_PROBED)
    assert label == expected
    assert reason


@pytest.mark.parametrize("spoiled, expected, phrase", [
    ({}, "photo", "BOTH bursts"),
    ({"reattach_dwell_exact": False}, "video", "when Hinge re-attached it"),
    ({"reattach_probe_ran": None, "reattach_dwell_exact": None,
      "reattach_mute_screens_complete": None, "reattach_centered": None},
     "unsure", "never asked to start"),
    ({"reattach_probe_ran": False}, "unsure", "never asked to start"),
    ({"reattach_dwell_exact": None}, "unsure", "second look"),
    ({"reattach_mute_screens_complete": False}, "unsure", "re-attach burst frame"),
    ({"reattach_mute_screens_complete": None}, "unsure", "re-attach burst frame"),
    ({"reattach_centered": False}, "unsure", "back inside Hinge's autoplay trigger zone"),
    ({"reattach_centered": None}, "unsure", "back inside Hinge's autoplay trigger zone"),
], ids=["both-bursts-exact", "motion-in-burst-two", "no-probe-at-all", "probe-did-not-run",
        "burst-two-unmeasured", "burst-two-screen-dirty", "burst-two-screen-missing",
        "re-centering-failed", "re-centering-unmeasured"])
def test_the_reattach_probe_truth_table(spoiled, expected, phrase):
    """The residual this probe exists for: a video that was NOT PLAYING while the first burst
    watched -- stalled, buffering, unloaded, or ended without looping -- holds byte-exact and
    reads as a photograph.  Motion in the second burst is video by the same rung motion in the
    first burst is; every other missing or failed leg is unsure, never photo."""
    label, reason = auto.label_card(
        glyph_hit=False, dwell_exact=True, mute_screens_complete=True,
        classifier_verdict="photo", centered=True, **{**_PROBED, **spoiled})

    assert label == expected
    assert phrase in reason


def test_a_mute_glyph_seen_only_in_the_reattach_burst_is_still_a_video():
    """The control Hinge redraws when it re-attaches media is affirmative evidence too, and it
    outranks every stillness argument exactly as a first-burst glyph does."""
    label, reason = auto.label_card(
        glyph_hit=True, dwell_exact=True, mute_screens_complete=True,
        classifier_verdict="photo", centered=True, **_PROBED)

    assert label == "video" and "mute-control match" in reason


def test_the_off_centre_refusal_names_the_autoplay_reason():
    label, reason = auto.label_card(glyph_hit=False, dwell_exact=True,
                                    mute_screens_complete=True, classifier_verdict="photo",
                                    centered=False, **_PROBED)
    assert label == "unsure"
    assert "autoplay trigger zone" in reason and "start playing" in reason


@pytest.mark.parametrize("field", ["dwell_exact", "mute_screens_complete", "centered"])
def test_three_valued_inputs_are_never_read_for_truthiness(field):
    """A truthy placeholder must not buy an observation nothing made."""
    kwargs = {"glyph_hit": False, "dwell_exact": True, "mute_screens_complete": True,
              "classifier_verdict": "photo", "centered": True}
    kwargs[field] = 1
    assert auto.label_card(**kwargs)[0] == "unsure"


def test_the_centre_band_constant_has_exactly_one_home():
    from operation_love.drivers import item_crops

    assert auto.CENTER_BAND_FRAC is item_crops.STILL_PHOTO_AUTOPLAY_CENTER_BAND_FRAC
    # No second copy of the number anywhere in either tool.
    for module in (auto, bound):
        source = Path(module.__file__).read_text()
        assert "0.15" not in source, module.__name__


# =====================================================================================
# observe_card over frames
# =====================================================================================

def _observe(sequence, *, matcher=_clean_matcher, classify=lambda _c: "photo", rect=_CARD_A,
             reattach=..., reattach_shift=0):
    """One card observed over a first burst and, by default, a re-attach burst that agrees.

    `reattach=None` is the no-probe case: every `reattach_*` leg stays unset and the card cannot
    be a photograph, which is the fail-closed default a station that could not probe lands on.
    """
    if reattach is ...:
        reattach = list(sequence)
    return auto.observe_card(rect, sequence, dwell_span_s=12.0, matcher=matcher,
                             frame_height=_H, content_band=_BAND, classify=classify,
                             reattach_sequence=reattach, reattach_span_s=12.0,
                             reattach_page_shift_px=reattach_shift)


def test_a_mute_glyph_match_in_any_frame_labels_the_card_video():
    still = _frame(40)
    card = _observe([still] * 5, matcher=_hit_matcher)
    assert card.label == "video" and card.glyph_hit is True
    assert card.max_glyph_score == pytest.approx(0.995)


def test_a_card_rect_that_moves_across_the_burst_labels_the_card_video():
    sequence = [_frame(40)] + [_frame(40, marks=[(1000, 500, (s * 31) % 255)])
                               for s in range(1, 6)]
    card = _observe(sequence)
    assert card.label == "video" and card.dwell_exact is False


def test_byte_exact_clean_and_photographic_labels_the_card_photo():
    still = _frame(40)
    card = _observe([still] * 6)
    assert card.label == "photo"
    assert (card.dwell_exact, card.mute_screens_complete) == (True, True)
    assert card.classifier_verdict == "photo" and card.centered is True
    assert abs(card.center_offset_frac) <= auto.CENTER_BAND_FRAC
    assert len(card.crops) == 6


def test_two_rects_on_one_anchor_become_two_independent_cards(repo, small):
    """One burst, two cards: the crops are what keep their verdicts apart."""
    moving = _frame(40, marks=[(1800, 500, 9)])            # inside CARD_B only (CARD_A ends 1671)

    def _both(_frame_bytes):
        return [_CARD_A, _CARD_B]

    driver = _FakeDriver(profiles=[[lambda read: (moving if read % 2 else _frame(40))]])
    manifest = _run(repo, driver, max_profiles=1, segment=_both)
    labels = sorted(card["label"] for card in manifest["cards"])
    assert labels == ["photo", "video"]


def test_a_crop_the_classifier_will_not_call_photographic_is_unsure():
    still = _frame(40)
    assert _observe([still] * 6, classify=lambda _c: "unknown").label == "unsure"


def test_a_written_prompt_card_gets_its_own_terminal_label():
    """`unsure` has to keep meaning "cannot tell"; a text card is a thing we CAN tell."""
    still = _frame(40)
    card = _observe([still] * 6, classify=lambda _c: "written")
    assert card.label == "written"
    assert "written prompt card" in card.label_reason


# --- the autoplay centring precondition -----------------------------------------------------

def test_an_off_centre_byte_exact_card_stays_unsure_with_the_autoplay_reason():
    """The defect this rung exists for: an off-centre video never plays and holds byte-exact."""
    still = _frame(40)
    card = _observe([still] * 6, rect=_CARD_OFF_CENTER)
    assert card.centered is False
    assert abs(card.center_offset_frac) > auto.CENTER_BAND_FRAC
    assert card.dwell_exact is True                 # it really did hold still...
    assert card.label == "unsure"                   # ...and that still proves nothing
    assert "autoplay trigger zone" in card.label_reason


def test_a_glyph_hit_off_centre_is_still_a_video():
    """Glyph and motion are valid from ANY screen position; only stillness needs the centre."""
    still = _frame(40)
    card = _observe([still] * 6, rect=_CARD_OFF_CENTER, matcher=_hit_matcher)
    assert card.centered is False and card.label == "video"


def test_motion_off_centre_is_still_a_video():
    sequence = [_frame(40)] + [_frame(40, marks=[(2000, 500, (s * 31) % 255)])
                               for s in range(1, 6)]
    card = _observe(sequence, rect=_CARD_OFF_CENTER)
    assert card.centered is False and card.dwell_exact is False and card.label == "video"


def test_movement_outside_the_card_rect_does_not_label_that_card_video():
    """The whole reason the persisted card frames are card-rect crops, not whole screens."""
    sequence = [_frame(40)] + [_frame(40, marks=[(2000, 500, (s * 31) % 255)])
                               for s in range(1, 6)]
    assert _observe(sequence).label == "photo"


def test_the_default_classifier_returns_the_shipped_three_way_verdict():
    from operation_love.drivers import item_type_preflight

    verdict = auto._classify(_frame(40))
    assert verdict in (item_type_preflight.PHOTO, item_type_preflight.WRITTEN,
                       item_type_preflight.UNKNOWN)
    assert verdict != item_type_preflight.PHOTO      # a flat grey field is not a photograph


# =====================================================================================
# the station read
# =====================================================================================

def test_read_stations_scrolls_with_the_drivers_humanized_step_and_stops_at_the_bottom(monkeypatch):
    monkeypatch.setattr(auto, "_STATIONS_SPAN", (4, 4))
    monkeypatch.setattr(bound, "_BURST_FRAMES_SPAN", (2, 2))
    driver = _FakeDriver(profiles=[[_still_page(40), _still_page(80)]])
    driver._load_next_profile()
    fake_time = _FakeTime()
    stations, notes = auto.read_stations(driver, rnd=bound.random.Random(3),
                                         sleep_fn=fake_time.sleep, clock=fake_time.clock,
                                         origin=0.0, content_band=_BAND, segment=_one_card,
                                         print_fn=lambda *_a: None)
    # Page 2 repeats once the depth runs past the script, so the third read sees the bottom.
    assert notes["reached_bottom"] is True
    assert len(stations) == 2
    # Two read-scrolls between stations, plus one return stroke per re-attach probe. Every one
    # of them is a guarded, ledger-keeping driver primitive; nothing here touches a transport.
    assert driver.calls.count("scroll_down_one") == 4
    assert driver.calls.count("scroll_up_one") == 2
    assert notes["reattach_probes_ran"] == 2 and notes["reattach_probes_refused"] == 0
    assert all(station.planned_frames == 2 for station in stations)


def test_every_station_takes_a_second_burst_after_a_measured_round_trip(monkeypatch):
    """The stimulus, end to end: out of the autoplay band, back into it, and burst again.

    The page displacement is MEASURED at every leg and nets to zero here, which is what lets the
    second burst be judged over the card the first one measured rather than over whatever now
    occupies those rows.
    """
    monkeypatch.setattr(auto, "_STATIONS_SPAN", (1, 1))
    monkeypatch.setattr(bound, "_BURST_FRAMES_SPAN", (2, 2))
    driver = _FakeDriver(profiles=[[_still_page(40), _still_page(80)]])
    driver._load_next_profile()
    fake_time = _FakeTime()

    stations, notes = auto.read_stations(driver, rnd=bound.random.Random(3),
                                         sleep_fn=fake_time.sleep, clock=fake_time.clock,
                                         origin=0.0, content_band=_BAND, segment=_one_card,
                                         print_fn=lambda *_a: None)

    station = stations[0]
    assert notes["reattach_probes_ran"] == 1
    assert station.reattach_anchor is not None
    assert len(station.reattach_burst) == 2
    assert station.reattach_span_s > 0
    assert station.reattach_page_shift_px == 0, "the round trip must not move the page"
    # The exit really left, and the return really came back: one stroke each, both guarded.
    assert driver.calls.count("scroll_up_one") == 1
    assert driver.calls.count("scroll_down_one") == 1


def test_a_probe_the_phone_will_not_move_leaves_the_station_without_a_second_burst(monkeypatch):
    """Fail closed: a page that does not move detaches nothing, so nothing can re-attach, and
    every card at that station stays unsure rather than inheriting the first burst's verdict."""
    monkeypatch.setattr(auto, "_STATIONS_SPAN", (1, 1))
    monkeypatch.setattr(bound, "_BURST_FRAMES_SPAN", (2, 2))
    driver = _FakeDriver(profiles=[[_still_page(40), _still_page(80)]])
    driver._load_next_profile()
    driver._measured_page_shift = lambda _before, _after: 0
    fake_time = _FakeTime()

    stations, notes = auto.read_stations(driver, rnd=bound.random.Random(3),
                                         sleep_fn=fake_time.sleep, clock=fake_time.clock,
                                         origin=0.0, content_band=_BAND, segment=_one_card,
                                         print_fn=lambda *_a: None)

    assert notes["reattach_probes_refused"] == 1 and notes["reattach_probes_ran"] == 0
    assert stations[0].reattach_anchor is None and stations[0].reattach_burst is None
    sequence = [stations[0].anchor] + [payload for payload, _t in stations[0].burst]
    observation = auto.observe_card(
        _CARD_A, sequence, dwell_span_s=1.0, matcher=_clean_matcher,
        frame_height=_H, content_band=_BAND, classify=lambda _c: "photo",
        reattach_sequence=None, reattach_span_s=None,
        reattach_page_shift_px=stations[0].reattach_page_shift_px)
    assert observation.dwell_exact is True, "the first burst really did hold still"
    assert observation.reattach_probe_ran is False
    assert observation.label == "unsure"


def test_a_station_with_no_complete_card_takes_no_burst(monkeypatch):
    monkeypatch.setattr(auto, "_STATIONS_SPAN", (2, 2))
    driver = _FakeDriver(profiles=[[_still_page(40), _still_page(80)]])
    driver._load_next_profile()
    fake_time = _FakeTime()
    stations, notes = auto.read_stations(driver, rnd=bound.random.Random(3),
                                         sleep_fn=fake_time.sleep, clock=fake_time.clock,
                                         origin=0.0, content_band=_BAND, segment=_no_card,
                                         print_fn=lambda *_a: None)
    assert stations == [] and notes["stations_without_a_complete_card"] == 2


def test_a_segmentation_refusal_is_counted_not_swallowed(monkeypatch):
    from operation_love.drivers.segment import SegmentationError

    monkeypatch.setattr(auto, "_STATIONS_SPAN", (2, 2))

    def _refuse(_frame):
        raise SegmentationError("no like-glyph template")

    driver = _FakeDriver(profiles=[[_still_page(40), _still_page(80)]])
    driver._load_next_profile()
    fake_time = _FakeTime()
    stations, notes = auto.read_stations(driver, rnd=bound.random.Random(3),
                                         sleep_fn=fake_time.sleep, clock=fake_time.clock,
                                         origin=0.0, content_band=_BAND, segment=_refuse,
                                         print_fn=lambda *_a: None)
    assert stations == [] and notes["segmentation_failures"] == 2


# =====================================================================================
# THE REGRESSION: nothing persisted => halt before any advance
# =====================================================================================

def test_a_profile_that_persists_nothing_halts_before_the_advance(repo, monkeypatch):
    """The exact 2026-08-21 failure: five profiles, zero frames, five real Passes."""
    monkeypatch.setattr(auto, "_STATIONS_SPAN", (2, 2))
    driver = _FakeDriver(profiles=[[_still_page(40)], [_still_page(50)]])
    out = _out(repo)
    manifest = _run(repo, driver, max_profiles=5, out=out, segment=_no_card)
    assert driver.gestures == []                       # nothing was spent, at all
    assert manifest["completed"] is False
    assert manifest["ended"] == "halted_empty_profile"
    assert "persisted ZERO frames" in manifest["halt_reason"]
    assert manifest["counts"]["passes"] == 0
    assert len(manifest["profiles"]) == 1              # and it stopped after the FIRST one
    assert json.loads((out / "manifest.json").read_text())["ended"] == "halted_empty_profile"


def test_there_is_no_strike_allowance_for_empty_profiles(repo, monkeypatch):
    monkeypatch.setattr(auto, "_STATIONS_SPAN", (2, 2))
    driver = _FakeDriver(profiles=[[_still_page(40)]] * 5)
    manifest = _run(repo, driver, max_profiles=5, segment=_no_card)
    assert manifest["counts"]["profiles_seen"] == 1 and driver.gestures == []


def test_the_advance_cannot_be_reached_without_a_harvest_licence(repo):
    """`advance_deck` takes the licence and checks it first; there is no other door."""
    driver = _FakeDriver(profiles=[[_still_page(40)]])
    empty = auto.ProfileHarvest(profile_id="profile_0001", root=_out(repo), frame_records=(),
                                measurable_cards=0)
    with pytest.raises(auto.EmptyProfileHalt, match="persisted ZERO frames"):
        auto.advance_deck(driver, "pass", harvest=empty)
    assert driver.gestures == []


def test_a_harvest_with_frames_but_no_measurable_card_is_still_refused(repo):
    out = _out(repo)
    records = auto._write_frames(out / "x", [(_tiny(9), 0.0)], relative="x")
    driver = _FakeDriver(profiles=[[_still_page(40)]])
    harvest = auto.ProfileHarvest(profile_id="profile_0001", root=out,
                                  frame_records=tuple(records), measurable_cards=0)
    with pytest.raises(auto.EmptyProfileHalt, match="no measurable card"):
        auto.advance_deck(driver, "pass", harvest=harvest)
    assert driver.gestures == []


def test_a_harvest_whose_frames_are_not_on_disk_is_refused(repo):
    out = _out(repo)
    records = auto._write_frames(out / "x", [(_tiny(9), 0.0)], relative="x")
    (out / records[0]["path"]).unlink()
    driver = _FakeDriver(profiles=[[_still_page(40)]])
    harvest = auto.ProfileHarvest(profile_id="profile_0001", root=out,
                                  frame_records=tuple(records), measurable_cards=1)
    with pytest.raises(auto.EmptyProfileHalt, match="not on disk"):
        auto.advance_deck(driver, "pass", harvest=harvest)
    assert driver.gestures == []


def test_a_harvest_whose_frames_changed_on_disk_is_refused(repo):
    out = _out(repo)
    records = auto._write_frames(out / "x", [(_tiny(9), 0.0)], relative="x")
    (out / records[0]["path"]).write_bytes(_tiny(200))
    driver = _FakeDriver(profiles=[[_still_page(40)]])
    harvest = auto.ProfileHarvest(profile_id="profile_0001", root=out,
                                  frame_records=tuple(records), measurable_cards=1)
    with pytest.raises(auto.EmptyProfileHalt, match="does not match the digest"):
        auto.advance_deck(driver, "pass", harvest=harvest)
    assert driver.gestures == []


def test_the_gesture_fires_only_after_the_frames_are_already_on_disk(repo, monkeypatch):
    """Ordering, observed from inside the gesture itself."""
    monkeypatch.setattr(auto, "_STATIONS_SPAN", (2, 2))
    monkeypatch.setattr(bound, "_BURST_FRAMES_SPAN", (2, 2))
    out = _out(repo)
    seen: list[int] = []

    def _on_gesture(_kind):
        seen.append(len(list((out / "cards").rglob("*.png"))))

    driver = _FakeDriver(profiles=[[_still_page(40), _still_page(80)]] * 2,
                         on_gesture=_on_gesture)
    manifest = _run(repo, driver, max_profiles=1, out=out)
    assert seen and seen[0] > 0                        # crops existed BEFORE the pass
    assert manifest["counts"]["passes"] == 1
    assert manifest["profiles"][0]["advanced"] is True


# =====================================================================================
# manifest schema and persistence
# =====================================================================================

def _good_driver(profiles=2, on_gesture=None):
    return _FakeDriver(profiles=[[_still_page(40), _still_page(80)]] * profiles,
                       on_gesture=on_gesture)


@pytest.fixture()
def small(monkeypatch):
    monkeypatch.setattr(auto, "_STATIONS_SPAN", (2, 2))
    monkeypatch.setattr(bound, "_BURST_FRAMES_SPAN", (3, 3))


def test_manifest_records_the_circular_channel_and_never_claims_human_ground_truth(repo, small):
    out = _out(repo)
    manifest = _run(repo, _good_driver(), out=out)
    assert manifest["ground_truth_channel"] == auto.CIRCULAR_CHANNEL
    assert manifest["human_ground_truth"] is False
    assert manifest["accepted_circular_risk"] == auto.CIRCULAR_ACCEPTANCE
    assert manifest["mute_matcher_is_observational_not_ground_truth"] is False
    assert manifest["label_blind_spot"] == bound.LABEL_BLIND_SPOT
    assert manifest["kind"] == bound._CAMPAIGN_KIND
    assert manifest["content_band"] == [0.0, 1.0]
    assert manifest["card_frames_are_card_rect_crops"] is True
    assert "segment_frame" in manifest["card_rect_source"]
    assert manifest["advance_action"] == "pass" and manifest["like_confirmation"] is None
    assert json.loads((out / "manifest.json").read_text()) == manifest


def test_every_persisted_frame_matches_its_manifest_digest(repo, small):
    out = _out(repo)
    manifest = _run(repo, _good_driver(), out=out)
    records = [frame for card in manifest["cards"] for frame in card["frames"]]
    for station in manifest["profiles"][0]["stations"]:
        records.append(station["anchor_frame"])
        records.extend(station["burst_frames"])
    assert records
    for record in records:
        payload = (out / record["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]


def test_card_records_use_the_sibling_schema_and_close_the_burst_before_labeling(repo, small):
    manifest = _run(repo, _good_driver())
    card = manifest["cards"][0]
    assert card["card_id"] == "card_0001" and card["label"] == "photo"
    assert all(set(frame) == {"path", "sha256", "t"} for frame in card["frames"])
    times = [frame["t"] for frame in card["frames"]]
    assert times == sorted(times)
    assert card["label_prompted_t"] >= times[-1] >= card["burst_completed_t"] - 1e-9
    assert len(card["mute_matcher_observations"]) == len(card["frames"])
    assert all(r["observational_only"] is True for r in card["mute_matcher_observations"])
    assert card["label_evidence"]["circular"] is True
    assert card["label_evidence"]["channel"] == auto.CIRCULAR_CHANNEL
    assert card["settled"] is None and card["settle_skipped_reason"]


def test_a_repeated_card_crop_within_one_profile_is_not_counted_twice(repo, small):
    """Two stations that both show the same physical card must not double the denominator.

    The two frames differ OUTSIDE the card rect (otherwise the read would call it the profile
    bottom and never reach the second station), while the card rect itself is byte-identical --
    which is exactly the "same card seen twice" the dedup exists for.
    """
    first = _frame(40)
    second = _frame(40, marks=[(2100, 500, 200)])          # a change below the card only
    driver = _FakeDriver(profiles=[[lambda _r, f=first: f, lambda _r, f=second: f]])
    manifest = _run(repo, driver, max_profiles=1)
    assert manifest["counts"]["cards"] == 1
    assert manifest["profiles"][0]["duplicate_card_crops_skipped"] == 1


def test_each_advance_is_followed_by_the_drivers_guarded_rewind(repo, small):
    """The next profile opens with a scroll-top gate, so something has to restore that."""
    driver = _good_driver(profiles=3)
    manifest = _run(repo, driver, max_profiles=2)
    assert driver.calls.count("scroll_to_top") == 2
    assert all(profile["rewound_to_top"] is True for profile in manifest["profiles"])


def test_a_profile_that_halts_is_never_rewound_or_advanced(repo, small):
    driver = _good_driver()
    driver._top_reasons = ["not a confirmed scroll top"]
    manifest = _run(repo, driver, max_profiles=2)
    assert "scroll_to_top" not in driver.calls and driver.gestures == []
    assert manifest["ended"] == "halted"


def test_pass_mode_spends_exactly_one_pass_per_completed_profile(repo, small):
    driver = _good_driver(profiles=3)
    manifest = _run(repo, driver, max_profiles=2)
    assert driver.gestures == ["pass", "pass"]
    assert all(profile["advanced"] is True for profile in manifest["profiles"])
    assert manifest["counts"]["passes"] == 2 and manifest["counts"]["likes"] == 0
    assert manifest["ended"] == "max_profiles" and manifest["completed"] is True


def test_like_mode_uses_the_drivers_ordinary_like_and_records_its_confirmation(repo, small):
    driver = _good_driver()
    manifest = _run(repo, driver, advance="like")
    assert driver.gestures == ["like"]
    assert manifest["like_confirmation"] == auto.LIKE_CONFIRMATION


def test_a_playing_video_is_labeled_and_still_advances(repo, small):
    driver = _FakeDriver(profiles=[[_video_page(), _still_page(80)]])
    manifest = _run(repo, driver, max_profiles=1)
    assert "video" in {card["label"] for card in manifest["cards"]}
    assert driver.gestures == ["pass"]


def test_targets_reached_completes_the_campaign(repo, small):
    driver = _FakeDriver(profiles=[[_video_page(), _still_page(80)]] * 3)
    manifest = _run(repo, driver, max_profiles=5, target_videos=1, target_photos=1)
    assert manifest["ended"] == "targets_reached" and manifest["completed"] is True


def test_an_unrecognized_screen_halts_and_spends_nothing_further(repo, small):
    driver = _FakeDriver(profiles=[[_still_page(40), _still_page(80)]] * 3,
                         blocked=[None, "paywall: Hinge is showing an upgrade sheet"])
    manifest = _run(repo, driver, max_profiles=4)
    assert manifest["ended"] == "halted" and "paywall" in manifest["halt_reason"]
    assert driver.gestures == ["pass"]


def test_an_unconfirmed_deck_top_halts_before_the_read(repo, small):
    driver = _good_driver()
    driver._top_reasons = ["the first frame is not a confirmed scroll top"]
    manifest = _run(repo, driver, max_profiles=2)
    assert manifest["ended"] == "halted"
    assert "confirmed scroll top" in manifest["halt_reason"]
    assert driver.gestures == []


def test_a_session_start_that_cannot_confirm_the_scroll_top_halts_immediately(repo, small):
    driver = _FakeDriver(profiles=[[_still_page(40)]], session_top=False)
    manifest = _run(repo, driver, max_profiles=2)
    assert manifest["ended"] == "halted" and "scroll top at session start" in manifest["halt_reason"]
    assert driver.gestures == []


def test_a_driver_refusal_during_the_advance_is_a_halt_that_keeps_the_profile(repo, small,
                                                                             monkeypatch):
    from operation_love.drivers.hinge import UnconfirmedScreenError

    driver = _good_driver()

    def _refuse():
        driver.gestures.append("pass_attempt")
        raise UnconfirmedScreenError("hinge: refusing to decide")

    monkeypatch.setattr(driver, "dislike", _refuse)
    manifest = _run(repo, driver, max_profiles=2)
    assert manifest["ended"] == "halted" and "refused the pass advance" in manifest["halt_reason"]
    assert driver.gestures == ["pass_attempt"]
    # Both stations of the profile were already read, labeled and persisted before the advance
    # was attempted, so the refusal must not throw that work away.
    assert [card["label"] for card in manifest["cards"]] == ["photo", "photo"]
    assert manifest["profiles"][0]["advanced"] is False
    assert manifest["counts"]["passes"] == 0


def test_the_run_summary_names_the_spend_and_repeats_the_caveat(repo, small):
    manifest = _run(repo, _good_driver())
    lines: list[str] = []
    auto.print_run_summary(manifest, print_fn=lines.append)
    joined = "\n".join(lines)
    assert "profiles seen: 1" in joined and "passes spent: 1" in joined
    assert bound.LABEL_BLIND_SPOT in joined


# =====================================================================================
# THE SECOND REGRESSION: a burst must SPAN its drawn window
# =====================================================================================
#
# Campaign 2 recorded nine burst frames whose on-screen video countdown advanced exactly ONE
# second. File mtimes are batch-write artefacts and prove nothing; the countdown was the ground
# truth. A one-second look cannot bound the exact-run tail a production dwell spanning the full
# window will meet, and `measure` would have thrown the whole run away at the window check after
# every profile had already been spent.

def test_a_burst_spans_its_drawn_window_under_the_real_scheduler():
    """The schedule is absolute deadlines, so a slow capture shortens the next gap, not the total."""
    for seed in range(6):
        plan = bound.plan_burst(bound.random.Random(seed))
        fake_time = _FakeTime()

        def _slow_capture(_time=fake_time):
            _time.now += 0.4                  # every screencap costs real time
            return b"frame"

        frames = bound.record_spanning_burst(_slow_capture, plan, sleep_fn=fake_time.sleep,
                                             clock=fake_time.clock)
        span = frames[-1][1] - frames[0][1]
        assert len(frames) == plan.frames
        assert bound._BURST_WINDOW_S_SPAN[0] <= plan.window_s <= bound._BURST_WINDOW_S_SPAN[1]
        assert span >= 8.0, "a dwell burst must span at least the low end of the window"
        assert span == pytest.approx(plan.window_s, abs=bound.BURST_SPAN_TOLERANCE_S)
        assert bound.burst_span_shortfall(frames, plan) <= bound.BURST_SPAN_TOLERANCE_S


def test_recorded_stamps_equal_the_sleep_schedule_and_are_taken_at_screencap_time():
    """Timestamps come from the monotonic clock at capture, never from a file mtime."""
    plan = bound.BurstPlan(frames=4, window_s=9.0, gaps_s=(2.0, 3.0, 4.0))
    fake_time = _FakeTime(step=0.0)
    frames = bound.record_spanning_burst(lambda: b"f", plan, sleep_fn=fake_time.sleep,
                                         clock=fake_time.clock)
    stamps = [stamp for _payload, stamp in frames]
    assert [round(stamp - stamps[0], 6) for stamp in stamps] == [0.0, 2.0, 5.0, 9.0]
    assert fake_time.slept == [2.0, 3.0, 4.0]
    assert stamps == sorted(stamps)


def test_the_gaps_are_hazard_drawn_and_never_uniform():
    """Owner rule: no timing parameter may be a fixed constant, a regular cadence is a signature."""
    plans = [bound.plan_burst(bound.random.Random(seed)) for seed in range(20)]
    assert len({round(plan.window_s, 6) for plan in plans}) > 1
    assert len({plan.frames for plan in plans}) > 1
    for plan in plans:
        assert len(set(round(gap, 6) for gap in plan.gaps_s)) > 1, "gaps must not be uniform"
        assert sum(plan.gaps_s) == pytest.approx(plan.window_s)


def test_a_burst_that_does_not_span_its_window_halts_the_campaign(repo, small, monkeypatch):
    """The guard, exercised: one profile lost instead of a whole campaign of wasted passes."""
    def _compressed(capture_fn, plan, *, sleep_fn, clock):
        return [(capture_fn(), 0.1 * index) for index in range(plan.frames)]

    monkeypatch.setattr(bound, "record_spanning_burst", _compressed)
    driver = _good_driver()
    manifest = _run(repo, driver, max_profiles=3)
    assert manifest["ended"] == "halted"
    assert "does not span its window" in manifest["halt_reason"]
    assert driver.gestures == []


def test_measure_refuses_a_campaign_whose_bursts_only_spanned_a_second(repo):
    """Pins exactly what campaign 2 would have produced: a window nothing was watched for.

    Refused per CARD now, against the window that card was drawn for, rather than downstream at
    the safety-factor margin: the margin guard only caught this shape when a video's last frame
    happened to differ, and a corpus of compressed bursts whose frames all differ used to pass.
    """
    one_second = (0.0, 0.33, 0.66, 1.0)
    held = _tiny(40, marks=[(12, 7, 3)])
    videos = [_card("video", [held, held, held, _tiny(40, marks=[(12, 7, 200)])])
              for _ in range(bound.MIN_VIDEO_CARDS)]
    photos = [_still_card() for _ in range(bound.MIN_PHOTO_CARDS)]
    campaign = _write_campaign(repo, videos + photos, _times=one_second)
    with pytest.raises(bound.VideoBoundRefused, match="did not span its window"):
        auto.measure(campaign)


def test_the_manifest_records_how_the_burst_was_scheduled(repo, small):
    manifest = _run(repo, _good_driver())
    assert "absolute deadlines" in manifest["burst_schedule"]
    assert "never file mtimes" in manifest["burst_schedule"]
    station = manifest["profiles"][0]["stations"][0]
    assert station["burst_span_s"] == pytest.approx(station["planned_window_s"],
                                                    abs=bound.BURST_SPAN_TOLERANCE_S)


def test_every_station_card_reports_its_own_dwell_span_and_not_the_profile_clock(repo,
                                                                                monkeypatch):
    """EVERY station, not just the first: the anchor is measured on the burst's own clock.

    A card's frame list opens with its station's anchor crop and continues with the burst crops,
    and `measure` reads that one list as the dwell.  Stamping the anchor with a constant while
    the burst carries profile-relative stamps splices two timelines into it, so every station
    after the first reports a dwell inflated by the whole time since the profile started -- and
    that number is what `observed_window_s`, `max_video_exact_run_s` and `accepted()` are all
    computed from.  The station at position 0 is the ONE station where the two agree, which is
    why checking only `stations[0]` let this survive.
    """
    monkeypatch.setattr(auto, "_STATIONS_SPAN", (4, 4))
    monkeypatch.setattr(bound, "_BURST_FRAMES_SPAN", (3, 3))
    out = _out(repo)
    driver = _FakeDriver(profiles=[[_still_page(40), _still_page(80),
                                    _still_page(120), _still_page(160)]])
    manifest = _run(repo, driver, max_profiles=1, out=out)

    stations = manifest["profiles"][0]["stations"]
    assert len(stations) == 4, "the walk must reach every station for this to prove anything"
    cards = {card["card_id"]: card for card in manifest["cards"]}
    band = (manifest["content_band"][0], manifest["content_band"][1])
    anchors = [station["anchor_frame"]["t"] for station in stations]
    # A measured anchor moves down the profile clock with its station; a constant does not.
    assert anchors == sorted(anchors) and anchors[0] < anchors[-1]
    for station in stations:
        assert station["cards"], f"station {station['station']} produced no card"
        for card_id in station["cards"]:
            card = cards[card_id]
            # The frame manifest and the card manifest must say the same thing about the anchor.
            assert card["frames"][0]["t"] == station["anchor_frame"]["t"]
            stat = bound._card_stat(out, card, band)
            assert stat.burst_span_s == pytest.approx(station["burst_span_s"], abs=0.05), (
                f"station {station['station']} card {card_id} reports "
                f"{stat.burst_span_s}s for a {station['burst_span_s']}s dwell")
            assert stat.longest_exact_run_s == pytest.approx(stat.burst_span_s, abs=0.05)


# =====================================================================================
# the autoplay centring precondition, through the whole walk
# =====================================================================================

def test_an_off_centre_card_is_never_labelled_photo_by_the_campaign(repo, small):
    """Even byte-exact, even clean, even photographic: off-centre it stays unsure."""
    driver = _FakeDriver(profiles=[[_still_page(40), _still_page(80)]])
    manifest = _run(repo, driver, max_profiles=1,
                    segment=lambda _f: [_CARD_OFF_CENTER])
    labels = {card["label"] for card in manifest["cards"]}
    assert labels == {"unsure"}
    for card in manifest["cards"]:
        assert card["centered"] is False
        assert abs(card["center_offset_frac"]) > auto.CENTER_BAND_FRAC
        assert card["center_band_frac"] == auto.CENTER_BAND_FRAC
        assert "autoplay trigger zone" in card["label_evidence"]["reason"]


def test_the_walk_scrolls_to_seat_an_off_centre_candidate(repo, small):
    """A corrective scroll is issued through the guarded humanized primitives, not raw input."""
    driver = _FakeDriver(profiles=[[_still_page(40), _still_page(80)]])
    _run(repo, driver, max_profiles=1, segment=lambda _f: [_CARD_OFF_CENTER])
    assert driver.calls.count("scroll_down_one") >= 1


def test_a_candidate_already_centred_costs_no_corrective_scroll(repo, small):
    driver = _FakeDriver(profiles=[[_still_page(40), _still_page(80)]])
    manifest = _run(repo, driver, max_profiles=1)
    assert all(station["centering_steps"] == 0
               for station in manifest["profiles"][0]["stations"])
    assert all(card["centered"] is True for card in manifest["cards"])


def test_an_offset_too_small_to_express_as_a_legal_gesture_is_not_faked(repo):
    """Best humanized interaction or fail loudly: no sub-minimum stroke is invented."""
    driver = _FakeDriver(profiles=[[_still_page(40)]])
    driver._load_next_profile()
    near = (53, 1150, 1027, 1300)                  # a few rows off centre
    moved = auto.center_candidate(driver, near, frame_height=_H, content_band=_BAND,
                                  rnd=bound.random.Random(1), sleep_fn=lambda _s: None)
    assert moved is False
    assert "scroll_down_one" not in driver.calls and "scroll_up_one" not in driver.calls


def test_a_card_above_centre_is_brought_down_with_the_reverse_read_scroll(repo):
    driver = _FakeDriver(profiles=[[_still_page(40)]])
    driver._load_next_profile()
    high = (53, 100, 1027, 600)
    assert auto.center_candidate(driver, high, frame_height=_H, content_band=_BAND,
                                 rnd=bound.random.Random(1), sleep_fn=lambda _s: None) is True
    assert driver.calls.count("scroll_up_one") == 1


# =====================================================================================
# the `written` bucket
# =====================================================================================

def test_written_cards_get_their_own_bucket_and_leave_unsure_alone(repo, small):
    driver = _FakeDriver(profiles=[[_still_page(40), _still_page(80)]])
    manifest = _run(repo, driver, max_profiles=1, classify=lambda _c: "written")
    assert {card["label"] for card in manifest["cards"]} == {"written"}
    assert manifest["counts"]["written"] == len(manifest["cards"])
    assert manifest["counts"]["unsure"] == 0
    lines: list[str] = []
    auto.print_run_summary(manifest, print_fn=lines.append)
    assert "written" in "\n".join(lines)


def test_written_cards_are_excluded_from_both_denominators(repo):
    cards = ([_moving_card() for _ in range(bound.MIN_VIDEO_CARDS)]
             + [_still_card() for _ in range(bound.MIN_PHOTO_CARDS)]
             + [_card("written", [_tiny(70)] * 4) for _ in range(7)])
    result = auto.measure(_write_campaign(repo, cards))
    assert result.video_cards == bound.MIN_VIDEO_CARDS
    assert result.photo_cards == bound.MIN_PHOTO_CARDS
    assert result.written_cards == 7 and result.unsure_cards == 0
    lines: list[str] = []
    bound.print_bound_report(result, print_fn=lines.append)
    assert "7 written" in "\n".join(lines)


@pytest.mark.parametrize("from_label, to_label", [("written", "photo"), ("written", "video"),
                                                  ("unsure", "written"), ("photo", "written")])
def test_adjudication_may_not_touch_a_written_card(repo, from_label, to_label):
    """`written` is terminal: any video affordance would already have been caught upstream."""
    seed = {"written": lambda: _card("written", [_tiny(70)] * 4),
            "unsure": _unsure_card, "photo": _still_card}[from_label]
    cards = ([_moving_card() for _ in range(bound.MIN_VIDEO_CARDS)]
             + [_still_card() for _ in range(bound.MIN_PHOTO_CARDS)] + [seed()])
    campaign = _write_campaign(repo, cards)
    target = f"card_{len(cards):04d}"
    _adjudication(campaign, _entry(campaign, target, from_label, to_label))
    with pytest.raises(bound.VideoBoundRefused, match="not one of the legal transitions"):
        auto.measure(campaign)


def test_the_artifact_freezes_the_centre_band_it_was_measured_under(repo, monkeypatch):
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: [])
    monkeypatch.setattr(bound, "_device_version_name", lambda *a, **k: None)
    campaign = _write_campaign(repo, _passing_cards(),
                               autoplay_center_band_frac=auto.CENTER_BAND_FRAC)
    artifact, _paste = auto.emit(campaign, config_path=str(_config(repo)))
    assert artifact["autoplay_center_band_frac"] == auto.CENTER_BAND_FRAC
    assert bound.verify_artifact_digest(artifact) is True
    tampered = dict(artifact)
    tampered["autoplay_center_band_frac"] = 0.9
    assert bound.verify_artifact_digest(tampered) is False


# =====================================================================================
# signals write the incomplete manifest
# =====================================================================================

def test_the_signal_handler_raises_a_halt_naming_the_signal():
    with pytest.raises(auto.CampaignSignalled, match="SIGTERM"):
        auto._raise_on_signal(signal.SIGTERM, None)
    with pytest.raises(auto.CampaignSignalled, match="SIGINT"):
        auto._raise_on_signal(signal.SIGINT, None)


def test_handlers_are_installed_for_both_signals_and_restored_afterwards():
    installed: dict = {}
    restored: list = []

    class _Signals:
        SIGINT, SIGTERM = signal.SIGINT, signal.SIGTERM

        def signal(self, signum, handler):
            previous = installed.get(signum, "ORIGINAL")
            installed[signum] = handler
            if handler != auto._raise_on_signal:
                restored.append(signum)
            return previous

    module = _Signals()
    with auto.halt_on_signals(signal_module=module, print_fn=lambda *_a: None) as trapped:
        assert set(trapped) == {signal.SIGINT, signal.SIGTERM}
        assert all(handler is auto._raise_on_signal for handler in installed.values())
    assert sorted(restored) == sorted([signal.SIGINT, signal.SIGTERM])


def test_a_signal_mid_run_writes_the_incomplete_manifest_and_spends_nothing(repo, small):
    out = _out(repo)
    driver = _good_driver(profiles=3)
    original = driver._screencap
    state = {"reads": 0}

    def _screencap(**kwargs):
        state["reads"] += 1
        if state["reads"] > 3:
            auto._raise_on_signal(signal.SIGTERM, None)
        return original(**kwargs)

    driver._screencap = _screencap
    manifest = _run(repo, driver, max_profiles=3, out=out)
    assert manifest["ended"] == "signalled" and manifest["completed"] is False
    assert "SIGTERM" in manifest["halt_reason"]
    assert driver.gestures == []
    assert json.loads((out / "manifest.json").read_text())["ended"] == "signalled"


def test_a_run_that_cannot_trap_signals_says_so_rather_than_pretending(repo):
    lines: list[str] = []

    class _Refusing:
        SIGINT, SIGTERM = signal.SIGINT, signal.SIGTERM

        def signal(self, *_a):
            raise ValueError("signal only works in main thread")

    with auto.halt_on_signals(signal_module=_Refusing(), print_fn=lines.append) as trapped:
        assert trapped == ()
    assert sum(1 for line in lines if "could not be trapped" in line) == 2


# =====================================================================================
# replay of the REAL failed run's record shape
# =====================================================================================

def test_the_real_capture_record_shape_no_longer_gates_the_frame_source(repo, small):
    """Structural replay: enumeration OFF, no item index, the real `items_unavailable` string.

    This is the shape data/hinge_debug/run_20260821_163736/actions.jsonl records for all five
    profiles that silently produced nothing.  A driver in exactly that state must now still
    persist frames, because the rects come from per-frame segmentation instead.
    """
    class _EnumerationBlockedDriver(_FakeDriver):
        # Everything the old frame source needed, present and useless, exactly as it was live.
        targeting_calibration = None
        _openers_enabled = True
        _current_item_index = None

        def _item_enumeration_blocker(self) -> str:
            return _REAL_ITEMS_UNAVAILABLE

        def _index_captured_items(self, photos):
            raise AssertionError("the enumeration hook must never be the frame source again")

        def _capture_current(self, should_stop=None):
            raise AssertionError("the campaign must not read through the enumeration path")

    pages = [_still_page(40 + step * 7) for step in range(_REAL_FRAMES_PER_PROFILE)]
    driver = _EnumerationBlockedDriver(profiles=[pages, list(pages)])
    manifest = _run(repo, driver, max_profiles=1)
    assert manifest["counts"]["cards"] >= 1
    assert manifest["profiles"][0]["persisted_frames"] > 0
    assert driver.gestures == ["pass"]
    assert manifest["completed"] is True


def test_the_real_frames_of_the_failed_run_segment_into_measurable_cards(repo, monkeypatch):
    """End to end over the ACTUAL screencaps of the failed run, with real perception.

    Skipped when the gitignored debug directory is absent (a clean checkout, CI).  When it is
    present this is the only test that would have caught the live failure: real segmentation,
    the real mute template, and the real crop classifier over real Hinge pixels.
    """
    if not _DEBUG_RUN.is_dir():
        pytest.skip("the 2026-08-21 debug run is not on this machine")
    frames = sorted(_DEBUG_RUN.glob("*_capture_before.png"))
    video_frame = _DEBUG_RUN / "00002_dislike_before.png"
    if len(frames) < 2 or not video_frame.exists():
        pytest.skip("the debug run does not carry the frames this replay needs")

    from operation_love.drivers import hinge as hinge_mod

    monkeypatch.setattr(auto, "_STATIONS_SPAN", (2, 2))
    monkeypatch.setattr(bound, "_BURST_FRAMES_SPAN", (2, 2))
    _real_frames_time = _FakeTime()
    still = frames[0].read_bytes()
    playing = video_frame.read_bytes()
    driver = _FakeDriver(
        profiles=[[lambda _r, f=still: f, lambda _r, f=playing: f]],
        matcher=hinge_mod.HingeDriver._match_video_mute,
        like_template=hinge_mod._load_template(hinge_mod.HINGE_SPEC.templates["like"]))
    # segment=None -> the REAL selectable_card_rects over the driver's real band and template.
    manifest = auto.run_capture(
        out_dir=_out(repo), driver=driver, advance="pass", serial="PIXEL7A",
        config_sha256=None, target_videos=999, target_photos=999, max_profiles=1,
        hinge_version_name="10.0.1", driver_content_band=driver.content_band,
        print_fn=lambda *_a: None, sleep_fn=_real_frames_time.sleep,
        clock=_real_frames_time.clock, classify=None, segment=None,
        signal_module=types.SimpleNamespace())

    assert manifest["counts"]["cards"] >= 2, "real frames must yield real card rects"
    assert manifest["profiles"][0]["persisted_frames"] > 0
    assert driver.gestures == ["pass"], "the advance must fire, and only after persistence"
    labels = {card["label"] for card in manifest["cards"]}
    assert "video" in labels, "the real mute-control frame must be labeled video"
    assert manifest["frame_size_px"] == [_W, _H]
    for card in manifest["cards"]:
        rect = card["card_rect"]
        assert rect[2] - rect[0] > 400 and rect[3] - rect[1] > 400


def test_the_centre_band_fits_the_real_card_geometry_of_the_failed_run():
    """The band has to be reachable in the field, or every card stays unsure forever.

    Measured over the 15 real screencaps of the 2026-08-21 run: every complete card rect the
    segmenter finds already sits inside the zone (worst |offset| ~0.124 against a 0.15 limit), so
    the centring walk is a no-op on ordinary cards and only spends gestures on genuinely
    off-centre ones. Skipped when that gitignored directory is absent.
    """
    if not _DEBUG_RUN.is_dir():
        pytest.skip("the 2026-08-21 debug run is not on this machine")
    from operation_love.drivers import hinge as hinge_mod
    from operation_love.drivers.item_crops import card_center_offset_frac

    band = hinge_mod.HINGE_SPEC.content_band
    template = hinge_mod._load_template(hinge_mod.HINGE_SPEC.templates["like"])
    offsets = []
    for path in sorted(_DEBUG_RUN.glob("*.png")):
        payload = path.read_bytes()
        height = bound._frame_size(payload)[1]
        offsets.extend(
            card_center_offset_frac(rect, frame_height=height, content_band=band)
            for rect in auto.selectable_card_rects(
                payload, content_band=band, like_template=template,
                like_threshold=hinge_mod._LIKE_MATCH_THRESHOLD))
    assert len(offsets) >= 10, "the real frames must yield real card rects"
    assert max(abs(offset) for offset in offsets) <= auto.CENTER_BAND_FRAC, (
        "the autoplay band must be reachable on real Hinge geometry, or nothing is ever "
        "photo-eligible")


# =====================================================================================
# measure / emit round trip
# =====================================================================================

_TIMES = (0.0, 4.0, 8.0, 12.0)


def _card(label: str, crops: list[bytes], **extra) -> dict:
    return {"label": label, "payloads": crops, "extra": extra}


def _still_card(label: str = "photo") -> dict:
    return _card(label, [_tiny(40)] * 4)


def _moving_card(label: str = "video") -> dict:
    return _card(label, [_tiny(40, marks=[(12, 7, (step * 17) % 255)]) for step in range(4)])


def _write_campaign(root: Path, cards: list[dict], _times=None, **overrides) -> Path:
    campaign = root / "ops" / "calibration" / "videoauto_test"
    (campaign / "cards").mkdir(parents=True, exist_ok=True)
    records = []
    for ordinal, card in enumerate(cards, start=1):
        card_id = f"card_{ordinal:04d}"
        card_dir = campaign / "cards" / card_id
        card_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for position, payload in enumerate(card["payloads"]):
            name = f"frame_{position:03d}.png"
            (card_dir / name).write_bytes(payload)
            frames.append({"path": f"cards/{card_id}/{name}",
                           "sha256": hashlib.sha256(payload).hexdigest(),
                           "t": float((_times or _TIMES)[position])})
        record = {"card_id": card_id, "label": card["label"], "frames": frames,
                  "settled": None, "settle_reads": 0, "planned_frames": len(frames),
                  "planned_window_s": 12.0,
                  "burst_completed_t": float((_times or _TIMES)[len(frames) - 1]),
                  "label_prompted_t": float((_times or _TIMES)[len(frames) - 1]) + 0.5,
                  "mute_matcher_observations": [{"screened": True, "score": 0.1,
                                                 "observational_only": True} for _ in frames],
                  "label_evidence": {"circular": True}}
        record.update(card["extra"])
        records.append(record)
    manifest = {
        "schema_version": 1, "kind": bound._CAMPAIGN_KIND, "tool_version": "2",
        "campaign_mode": "ai_labeled_automated_v2",
        "ground_truth_channel": auto.CIRCULAR_CHANNEL, "human_ground_truth": False,
        "accepted_circular_risk": auto.CIRCULAR_ACCEPTANCE,
        "label_blind_spot": bound.LABEL_BLIND_SPOT,
        "mute_matcher_is_observational_not_ground_truth": False,
        "advance_action": "pass", "completed": True, "ended": "targets_reached",
        "halt_reason": None, "device": "PIXEL7A", "hinge_version_name": "10.0.1",
        "frame_size_px": [16, 32], "content_band": [0.0, 1.0], "config_sha256": None,
        "captured_at": "2026-08-21T00:00:00+00:00", "cards": records,
    }
    manifest.update(overrides)
    (campaign / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return campaign


def _passing_cards() -> list[dict]:
    return ([_moving_card() for _ in range(bound.MIN_VIDEO_CARDS)]
            + [_still_card() for _ in range(bound.MIN_PHOTO_CARDS)])


def _config(root: Path) -> Path:
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump({"apps": {"hinge": {"serial": "PIXEL7A"}}}))
    return path


def test_measure_reads_a_circular_campaign_and_prints_the_vacuity_caveat(repo):
    result = auto.measure(_write_campaign(repo, _passing_cards()))
    assert result.video_accepts == 0 and result.photo_false_refusals == 0
    lines: list[str] = []
    bound.print_bound_report(result, print_fn=lines.append)
    joined = "\n".join(lines)
    assert "STRUCTURAL VACUITY CAVEAT" in joined
    assert "zero BY CONSTRUCTION" in joined
    assert "max_video_exact_run_s" in joined
    assert "false-refusal" in joined and "corpus" in joined


def test_measure_refuses_a_campaign_that_does_not_carry_the_acceptance_phrase(repo):
    campaign = _write_campaign(repo, _passing_cards(), accepted_circular_risk="I_GUESS")
    with pytest.raises(bound.VideoBoundRefused, match="acceptance phrase"):
        auto.measure(campaign)


def test_measure_refuses_a_circular_campaign_claiming_human_ground_truth(repo):
    campaign = _write_campaign(repo, _passing_cards(), human_ground_truth=True)
    with pytest.raises(bound.VideoBoundRefused, match="human_ground_truth false"):
        auto.measure(campaign)


def test_measure_refuses_an_owner_labeled_campaign(repo):
    campaign = _write_campaign(repo, _passing_cards(),
                               ground_truth_channel=bound.GROUND_TRUTH_CHANNEL,
                               human_ground_truth=True, accepted_circular_risk=None)
    with pytest.raises(auto.AutoBoundRefused, match="hinge_video_bound measure"):
        auto.measure(campaign)


def test_measure_refuses_a_session_that_did_not_stop_at_a_recognized_halt(repo):
    """`completed: false` with no recognized safety halt is still not evidence."""
    campaign = _write_campaign(repo, _passing_cards(), completed=False,
                               ended="aborted_VideoBoundRefused")
    with pytest.raises(bound.VideoBoundRefused, match="not completed"):
        auto.measure(campaign)


def test_a_video_that_held_one_frame_is_still_refused_on_this_channel(repo):
    cards = ([_moving_card() for _ in range(bound.MIN_VIDEO_CARDS - 1)] + [_still_card("video")]
             + [_still_card() for _ in range(bound.MIN_PHOTO_CARDS)])
    with pytest.raises(bound.VideoBoundRefused, match="would have been ACCEPTED"):
        auto.measure(_write_campaign(repo, cards))


def _emit(repo, monkeypatch):
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: [])
    monkeypatch.setattr(bound, "_device_version_name", lambda *a, **k: None)
    campaign = _write_campaign(repo, _passing_cards())
    return campaign, auto.emit(campaign, config_path=str(_config(repo)))


def test_emit_freezes_the_caveat_and_the_acceptance_inside_the_artifact(repo, monkeypatch):
    campaign, (artifact, _paste) = _emit(repo, monkeypatch)
    assert set(artifact) == bound.BOUND_ARTIFACT_KEYS | bound.CIRCULAR_ARTIFACT_KEYS
    assert artifact["ground_truth_channel"] == auto.CIRCULAR_CHANNEL
    assert artifact["human_ground_truth"] is False
    assert artifact["accepted_circular_risk"] == auto.CIRCULAR_ACCEPTANCE
    assert "zero BY CONSTRUCTION" in artifact["label_blind_spot"]
    assert bound.verify_artifact_digest(artifact) is True
    assert json.loads((campaign / "bound.json").read_text()) == artifact


def test_stripping_the_caveat_invalidates_the_artifact_digest(repo, monkeypatch):
    _campaign, (artifact, _paste) = _emit(repo, monkeypatch)
    tampered = dict(artifact)
    tampered["label_blind_spot"] = "looks fine to me"
    assert bound.verify_artifact_digest(tampered) is False


def test_emit_paste_block_gains_accepted_circular_risk(repo, monkeypatch):
    campaign, (artifact, paste) = _emit(repo, monkeypatch)
    assert tuple(paste) == bound.PASTE_KEYS + ("accepted_circular_risk",)
    assert paste["accepted_circular_risk"] == auto.CIRCULAR_ACCEPTANCE
    payload = (campaign / "bound.json").read_bytes()
    assert paste["artifact_sha256"] == hashlib.sha256(payload).hexdigest()
    assert paste["artifact_path"] == "ops/calibration/videoauto_test/bound.json"
    block = yaml.safe_load(yaml.safe_dump(
        {"apps": {"hinge": {"still_photo_bound_evidence": paste}}}, sort_keys=False))
    assert block["apps"]["hinge"]["still_photo_bound_evidence"] == paste


def test_the_owner_channel_paste_block_is_untouched_by_this_extension(repo, monkeypatch):
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: [])
    monkeypatch.setattr(bound, "_device_version_name", lambda *a, **k: None)
    campaign = _write_campaign(repo, _passing_cards(),
                               ground_truth_channel=bound.GROUND_TRUTH_CHANNEL,
                               human_ground_truth=True, accepted_circular_risk=None)
    artifact, paste = bound.emit(campaign, config_path=str(_config(repo)))
    assert set(artifact) == bound.BOUND_ARTIFACT_KEYS
    assert tuple(paste) == bound.PASTE_KEYS
    assert artifact["human_ground_truth"] is True


def test_emitted_circular_paste_block_satisfies_config_validation(repo, monkeypatch):
    from operation_love import config as config_mod

    validate = getattr(config_mod, "_validate_hinge_still_photo_bound_evidence", None)
    clear = getattr(targeting_policy, "clear_installed_still_photo_bound", None)
    if validate is None or clear is None:
        pytest.skip("config-side still-photo bound validation is not present in this tree")
    _campaign, (artifact, paste) = _emit(repo, monkeypatch)
    cfg = types.SimpleNamespace(
        enabled_apps=["hinge"],
        apps={"hinge": {"serial": artifact["device"], "still_photo_bound_evidence": paste}})
    try:
        validate(cfg)
        installed = targeting_policy.installed_still_photo_bound()
        assert installed is not None
        assert installed.ground_truth_channel == auto.CIRCULAR_CHANNEL
        assert installed.human_ground_truth is False
    finally:
        clear()


# =====================================================================================
# offline adjudication of the campaign's own labels
# =====================================================================================
#
# The owner's rule is "videos all have a mute icon, no mute icon means picture". These tests do
# not encode that rule -- they encode what the tool will and will not let it do: move a card
# toward caution, cite only frames the campaign really recorded, and never demote a video.

def _unsure_card(label: str = "unsure") -> dict:
    """A card whose burst was byte-exact but which the capture would not call photographic."""
    return _card(label, [_tiny(70)] * 4)


def _seed_card(from_label: str, to_label: str) -> dict:
    """A card labelled `from_label` whose DWELL behaviour is consistent with `to_label`.

    Promoting a byte-exact card to `video` legitimately makes the campaign refuse (that card
    would have been accepted, which is the whole failure mode), so a card destined for the video
    denominator has to be one that actually moved. That refusal is pinned separately below.
    """
    payloads = ([_tiny(70, marks=[(12, 7, (step * 17) % 255)]) for step in range(4)]
                if to_label == "video" else [_tiny(70)] * 4)
    return _card(from_label, payloads)


def _card_digests(campaign: Path, card_id: str) -> list[str]:
    manifest = json.loads((campaign / "manifest.json").read_text())
    card = next(c for c in manifest["cards"] if c["card_id"] == card_id)
    return [frame["sha256"] for frame in card["frames"]]


def _adjudication(campaign: Path, *entries, model="claude-vision", process="offline_frame_review"):
    document = {"adjudicator": {"model": model, "process": process}, "entries": list(entries)}
    (campaign / bound.ADJUDICATION_FILENAME).write_text(json.dumps(document, indent=2) + "\n")
    return document


def _entry(campaign: Path, card_id: str, from_label: str, to_label: str, **overrides) -> dict:
    entry = {"card_id": card_id, "from_label": from_label, "to_label": to_label,
             "frame_sha256s": _card_digests(campaign, card_id)[:1],
             "rationale": "no mute icon in any reviewed frame"}
    entry.update(overrides)
    return entry


def _campaign_with_unsures(repo, *, unsures: int = 3, photos_short: int = 0):
    """A passing campaign whose photo denominator is `photos_short` cards below the minimum."""
    cards = ([_moving_card() for _ in range(bound.MIN_VIDEO_CARDS)]
             + [_still_card() for _ in range(bound.MIN_PHOTO_CARDS - photos_short)]
             + [_unsure_card() for _ in range(unsures)])
    return _write_campaign(repo, cards)


def test_no_adjudications_file_leaves_the_campaign_byte_identical(repo, monkeypatch):
    """Requirement 4: absence changes nothing at all, digest included."""
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: [])
    monkeypatch.setattr(bound, "_device_version_name", lambda *a, **k: None)
    campaign = _write_campaign(repo, _passing_cards())
    artifact, paste = auto.emit(campaign, config_path=str(_config(repo)))
    assert "adjudications" not in artifact
    assert set(artifact) == bound.BOUND_ARTIFACT_KEYS | bound.CIRCULAR_ARTIFACT_KEYS
    assert tuple(paste) == bound.PASTE_KEYS + ("accepted_circular_risk",)
    result = auto.measure(campaign)
    assert result.adjudications is None
    lines: list[str] = []
    bound.print_bound_report(result, print_fn=lines.append)
    assert not any("adjudicated" in line for line in lines)


@pytest.mark.parametrize("from_label, to_label", list(bound.LEGAL_ADJUDICATIONS))
def test_every_legal_transition_is_applied(repo, from_label, to_label):
    cards = ([_moving_card() for _ in range(bound.MIN_VIDEO_CARDS)]
             + [_still_card() for _ in range(bound.MIN_PHOTO_CARDS)]
             + [_seed_card(from_label, to_label)])
    campaign = _write_campaign(repo, cards)
    target = f"card_{len(cards):04d}"
    _adjudication(campaign, _entry(campaign, target, from_label, to_label))
    result = auto.measure(campaign)
    moved = next(stat for stat in result.stats if stat.card_id == target)
    assert moved.label == to_label
    assert result.adjudications["entries"][0]["card_id"] == target


@pytest.mark.parametrize("from_label, to_label", [
    ("video", "photo"),        # the transition that could erase the bound's own population
    ("video", "unsure"),
    ("video", "video"),
    ("photo", "photo"),        # unchanged is pointless
    ("photo", "unsure"),
    ("unsure", "unsure"),
])
def test_every_illegal_transition_refuses(repo, from_label, to_label):
    seed = {"video": _moving_card, "photo": _still_card, "unsure": _unsure_card}[from_label]
    cards = ([_moving_card() for _ in range(bound.MIN_VIDEO_CARDS)]
             + [_still_card() for _ in range(bound.MIN_PHOTO_CARDS)]
             + [seed(from_label)])
    campaign = _write_campaign(repo, cards)
    target = f"card_{len(cards):04d}"
    _adjudication(campaign, _entry(campaign, target, from_label, to_label))
    with pytest.raises(bound.VideoBoundRefused, match="not one of the legal transitions"):
        auto.measure(campaign)


def test_a_video_can_never_be_demoted_however_it_is_dressed_up(repo):
    campaign = _campaign_with_unsures(repo)
    _adjudication(campaign, _entry(campaign, "card_0001", "video", "photo"))
    with pytest.raises(bound.VideoBoundRefused, match="toward caution"):
        auto.measure(campaign)


def test_adjudication_cannot_be_applied_to_the_owner_labeled_channel(repo):
    """Otherwise the artifact would still claim human ground truth it no longer had."""
    campaign = _write_campaign(repo, _passing_cards(),
                               ground_truth_channel=bound.GROUND_TRUTH_CHANNEL,
                               human_ground_truth=True, accepted_circular_risk=None)
    _adjudication(campaign, _entry(campaign, "card_0001", "video", "photo"))
    with pytest.raises(bound.VideoBoundRefused, match="owner's own tap-to-play verdicts"):
        bound.measure(campaign)


def test_an_unknown_card_id_refuses(repo):
    campaign = _campaign_with_unsures(repo)
    entry = _entry(campaign, "card_0001", "video", "photo")
    entry.update({"card_id": "card_9999", "from_label": "unsure", "to_label": "photo"})
    _adjudication(campaign, entry)
    with pytest.raises(bound.VideoBoundRefused, match="which this campaign does not contain"):
        auto.measure(campaign)


def test_a_wrong_from_label_refuses(repo):
    campaign = _campaign_with_unsures(repo)
    # card_0001 is a video; claiming it was unsure means the adjudicator saw a different card.
    _adjudication(campaign, _entry(campaign, "card_0001", "unsure", "photo"))
    with pytest.raises(bound.VideoBoundRefused, match="a different card"):
        auto.measure(campaign)


def test_a_frame_sha_that_is_not_the_cards_own_refuses(repo):
    campaign = _campaign_with_unsures(repo)
    target = f"card_{bound.MIN_VIDEO_CARDS + bound.MIN_PHOTO_CARDS + 1:04d}"
    # A digest that really exists in the campaign, but on a DIFFERENT card.
    foreign = _card_digests(campaign, "card_0001")[:1]
    _adjudication(campaign, _entry(campaign, target, "unsure", "photo", frame_sha256s=foreign))
    with pytest.raises(bound.VideoBoundRefused, match="not among card .* recorded frames"):
        auto.measure(campaign)


def test_a_fabricated_frame_sha_refuses(repo):
    campaign = _campaign_with_unsures(repo)
    target = f"card_{bound.MIN_VIDEO_CARDS + bound.MIN_PHOTO_CARDS + 1:04d}"
    _adjudication(campaign, _entry(campaign, target, "unsure", "photo",
                                   frame_sha256s=["0" * 64]))
    with pytest.raises(bound.VideoBoundRefused, match="not among card"):
        auto.measure(campaign)


@pytest.mark.parametrize("mutate, expected", [
    (lambda e: e.update({"frame_sha256s": []}), "at least one frame sha256"),
    (lambda e: e.update({"rationale": "  "}), "nonempty rationale"),
    (lambda e: e.pop("rationale"), "must carry exactly"),
    (lambda e: e.update({"extra": 1}), "must carry exactly"),
])
def test_malformed_entries_refuse(repo, mutate, expected):
    campaign = _campaign_with_unsures(repo)
    target = f"card_{bound.MIN_VIDEO_CARDS + bound.MIN_PHOTO_CARDS + 1:04d}"
    entry = _entry(campaign, target, "unsure", "photo")
    mutate(entry)
    _adjudication(campaign, entry)
    with pytest.raises(bound.VideoBoundRefused, match=expected):
        auto.measure(campaign)


def test_an_anonymous_or_empty_adjudication_refuses(repo):
    campaign = _campaign_with_unsures(repo)
    _adjudication(campaign, model="   ")
    with pytest.raises(bound.VideoBoundRefused, match="name its adjudicator"):
        auto.measure(campaign)
    _adjudication(campaign)
    with pytest.raises(bound.VideoBoundRefused, match="carries no entries"):
        auto.measure(campaign)


def test_one_card_gets_one_verdict(repo):
    campaign = _campaign_with_unsures(repo)
    target = f"card_{bound.MIN_VIDEO_CARDS + bound.MIN_PHOTO_CARDS + 1:04d}"
    _adjudication(campaign, _entry(campaign, target, "unsure", "photo"),
                  _entry(campaign, target, "unsure", "video"))
    with pytest.raises(bound.VideoBoundRefused, match="second verdict"):
        auto.measure(campaign)


def test_tallies_report_the_captures_own_labels_beside_the_adjudicated_ones(repo):
    """A photo denominator that only reaches the minimum THROUGH adjudication must show it."""
    short = 2
    campaign = _campaign_with_unsures(repo, unsures=3, photos_short=short)
    first_unsure = bound.MIN_VIDEO_CARDS + bound.MIN_PHOTO_CARDS - short + 1
    entries = [_entry(campaign, f"card_{first_unsure + n:04d}", "unsure", "photo")
               for n in range(short)]
    _adjudication(campaign, *entries)
    result = auto.measure(campaign)
    assert result.photo_cards == bound.MIN_PHOTO_CARDS
    assert result.original_label_counts["photo"] == bound.MIN_PHOTO_CARDS - short
    assert result.original_label_counts["unsure"] == 3
    assert result.unsure_cards == 1                       # one was left alone
    lines: list[str] = []
    bound.print_bound_report(result, print_fn=lines.append)
    joined = "\n".join(lines)
    assert f"{bound.MIN_PHOTO_CARDS} photo (+{short} adjudicated)" in joined
    assert f"{short} x unsure -> photo" in joined
    assert "claude-vision" in joined and "offline_frame_review" in joined
    assert f"{bound.MIN_PHOTO_CARDS - short} photo" in joined
    # The circularity is not repaired by a second reader, so the caveat stays put.
    assert "STRUCTURAL VACUITY CAVEAT" in joined


def test_without_the_adjudication_the_same_campaign_refuses_for_too_few_photos(repo):
    campaign = _campaign_with_unsures(repo, unsures=3, photos_short=2)
    with pytest.raises(bound.VideoBoundRefused, match="still-photo cards"):
        auto.measure(campaign)


def test_adjudicating_an_unsure_to_video_grows_the_video_denominator(repo):
    cards = ([_moving_card() for _ in range(bound.MIN_VIDEO_CARDS)]
             + [_still_card() for _ in range(bound.MIN_PHOTO_CARDS)]
             + [_seed_card("unsure", "video")])
    campaign = _write_campaign(repo, cards)
    target = f"card_{len(cards):04d}"
    _adjudication(campaign, _entry(campaign, target, "unsure", "video",
                                   rationale="mute icon visible in the cited frame"))
    result = auto.measure(campaign)
    assert result.video_cards == bound.MIN_VIDEO_CARDS + 1
    assert result.original_label_counts["video"] == bound.MIN_VIDEO_CARDS
    lines: list[str] = []
    bound.print_bound_report(result, print_fn=lines.append)
    assert "(+1 adjudicated)" in "\n".join(lines)


def test_promoting_a_byte_exact_card_to_video_makes_the_campaign_refuse(repo):
    """Adjudication cannot smuggle a card past the accept rule in EITHER direction.

    A card that held one frame for its whole burst, re-labelled `video`, is by definition a video
    that would have been ACCEPTED -- the single failure mode section 4 exists to exclude. The
    threshold sees the post-adjudication labels, so it refuses rather than reporting a bound the
    adjudication just invalidated.
    """
    campaign = _campaign_with_unsures(repo, unsures=1)
    target = f"card_{bound.MIN_VIDEO_CARDS + bound.MIN_PHOTO_CARDS + 1:04d}"
    _adjudication(campaign, _entry(campaign, target, "unsure", "video"))
    with pytest.raises(bound.VideoBoundRefused, match="would have been ACCEPTED"):
        auto.measure(campaign)


def test_an_adjudicated_card_that_behaves_like_a_video_is_still_refused(repo):
    """unsure -> photo cannot launder a card past the accept rule; the dwell still decides."""
    cards = ([_moving_card() for _ in range(bound.MIN_VIDEO_CARDS)]
             + [_still_card() for _ in range(bound.MIN_PHOTO_CARDS)]
             + [_card("unsure", [_tiny(40, marks=[(12, 7, s * 40)]) for s in range(4)])])
    campaign = _write_campaign(repo, cards)
    target = f"card_{len(cards):04d}"
    _adjudication(campaign, _entry(campaign, target, "unsure", "photo"))
    result = auto.measure(campaign)
    # It joins the photo denominator and is counted as a false refusal, never as an accept.
    assert result.photo_cards == bound.MIN_PHOTO_CARDS + 1
    assert result.photo_false_refusals == 1
    assert result.video_accepts == 0


# --- emit: the verdicts are inside the artifact's own digest --------------------------------

def _emit_adjudicated(repo, monkeypatch, *, to_label="photo"):
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: [])
    monkeypatch.setattr(bound, "_device_version_name", lambda *a, **k: None)
    campaign = _campaign_with_unsures(repo, unsures=1)
    target = f"card_{bound.MIN_VIDEO_CARDS + bound.MIN_PHOTO_CARDS + 1:04d}"
    _adjudication(campaign, _entry(campaign, target, "unsure", to_label))
    return campaign, target, auto.emit(campaign, config_path=str(_config(repo)))


def test_emit_freezes_the_applied_verdicts_and_the_adjudicator(repo, monkeypatch):
    campaign, target, (artifact, paste) = _emit_adjudicated(repo, monkeypatch)
    assert set(artifact) == (bound.BOUND_ARTIFACT_KEYS | bound.CIRCULAR_ARTIFACT_KEYS
                             | bound.ADJUDICATION_ARTIFACT_KEYS)
    assert artifact["adjudications"]["adjudicator"] == {"model": "claude-vision",
                                                        "process": "offline_frame_review"}
    entries = artifact["adjudications"]["entries"]
    assert [e["card_id"] for e in entries] == [target]
    assert entries[0]["from_label"] == "unsure" and entries[0]["to_label"] == "photo"
    assert entries[0]["frame_sha256s"] and entries[0]["rationale"]
    # The artifact's counts are the POST-adjudication ones, and the card carries its new label.
    assert artifact["photo_cards"] == bound.MIN_PHOTO_CARDS + 1
    assert next(c for c in artifact["cards"] if c["card_id"] == target)["label"] == "photo"
    assert bound.verify_artifact_digest(artifact) is True
    assert json.loads((campaign / "bound.json").read_text()) == artifact


def test_editing_an_adjudication_inside_the_artifact_invalidates_its_digest(repo, monkeypatch):
    _campaign, _target, (artifact, _paste) = _emit_adjudicated(repo, monkeypatch)
    for mutate in (
            lambda a: a["adjudications"]["entries"][0].update({"to_label": "video"}),
            lambda a: a["adjudications"]["entries"][0].update({"rationale": "trust me"}),
            lambda a: a["adjudications"]["adjudicator"].update({"model": "somebody else"}),
            lambda a: a["adjudications"]["entries"].clear()):
        tampered = json.loads(json.dumps(artifact))
        mutate(tampered)
        assert bound.verify_artifact_digest(tampered) is False
    stripped = dict(artifact)
    stripped.pop("adjudications")
    assert bound.verify_artifact_digest(stripped) is False


def test_the_paste_block_shape_is_unchanged_by_adjudication(repo, monkeypatch):
    campaign, _target, (artifact, paste) = _emit_adjudicated(repo, monkeypatch)
    assert tuple(paste) == bound.PASTE_KEYS + ("accepted_circular_risk",)
    # Its numbers are the post-adjudication ones, and config re-verifies each against the
    # artifact, so a mapping cannot quote a count the adjudicated artifact does not carry.
    assert paste["photo_cards"] == artifact["photo_cards"] == bound.MIN_PHOTO_CARDS + 1
    payload = (campaign / "bound.json").read_bytes()
    assert paste["artifact_sha256"] == hashlib.sha256(payload).hexdigest()


def test_an_adjudicated_artifact_still_satisfies_config_validation(repo, monkeypatch):
    from operation_love import config as config_mod

    validate = getattr(config_mod, "_validate_hinge_still_photo_bound_evidence", None)
    clear = getattr(targeting_policy, "clear_installed_still_photo_bound", None)
    if validate is None or clear is None:
        pytest.skip("config-side still-photo bound validation is not present in this tree")
    _campaign, _target, (artifact, paste) = _emit_adjudicated(repo, monkeypatch)
    cfg = types.SimpleNamespace(
        enabled_apps=["hinge"],
        apps={"hinge": {"serial": artifact["device"], "still_photo_bound_evidence": paste}})
    try:
        validate(cfg)
        installed = targeting_policy.installed_still_photo_bound()
        assert installed is not None and installed.photo_cards == bound.MIN_PHOTO_CARDS + 1
    finally:
        clear()


# =====================================================================================
# merging several sittings into one bound
# =====================================================================================
#
# 60 video cards is ~300 profiles at the observed ~1-in-5 rate, which is several sittings. A
# campaign directory is the right unit for a sitting -- it is what can be interrupted, halted and
# re-read -- so the BOUND has to span them. What it must never do is span two different worlds,
# which is what the agreement check is for.

def _named_campaign(root: Path, name: str, cards: list[dict], **overrides) -> Path:
    campaign = root / "ops" / "calibration" / name
    (campaign / "cards").mkdir(parents=True, exist_ok=True)
    records = []
    for ordinal, card in enumerate(cards, start=1):
        card_id = f"card_{ordinal:04d}"
        card_dir = campaign / "cards" / card_id
        card_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for position, payload in enumerate(card["payloads"]):
            (card_dir / f"frame_{position:03d}.png").write_bytes(payload)
            frames.append({"path": f"cards/{card_id}/frame_{position:03d}.png",
                           "sha256": hashlib.sha256(payload).hexdigest(),
                           "t": float(_TIMES[position])})
        records.append({"card_id": card_id, "label": card["label"], "frames": frames,
                        "settled": None, "settle_reads": 0, "planned_frames": len(frames),
                        "planned_window_s": 12.0,
                        "burst_completed_t": float(_TIMES[len(frames) - 1]),
                        "label_prompted_t": float(_TIMES[len(frames) - 1]) + 0.5,
                        "mute_matcher_observations": [
                            {"screened": True, "score": 0.1, "observational_only": True}
                            for _ in frames]})
    manifest = {
        "schema_version": 1, "kind": bound._CAMPAIGN_KIND, "tool_version": "2",
        "campaign_mode": "ai_labeled_automated_v2",
        "ground_truth_channel": auto.CIRCULAR_CHANNEL, "human_ground_truth": False,
        "accepted_circular_risk": auto.CIRCULAR_ACCEPTANCE,
        "label_blind_spot": bound.LABEL_BLIND_SPOT,
        "mute_matcher_is_observational_not_ground_truth": False,
        "advance_action": "pass", "completed": True, "ended": "targets_reached",
        "halt_reason": None, "device": "PIXEL7A", "hinge_version_name": "10.0.1",
        "frame_size_px": [16, 32], "content_band": [0.0, 1.0], "config_sha256": None,
        "autoplay_center_band_frac": auto.CENTER_BAND_FRAC,
        "dwell_parameter_space": {"burst_frames_span": [6, 10]},
        "captured_at": "2026-08-21T00:00:00+00:00", "cards": records,
    }
    manifest.update(overrides)
    (campaign / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return campaign


def _half_and_half(repo, *, first_videos, first_photos, second_videos, second_photos, **over):
    """Two sittings, neither of which reaches the thresholds on its own."""
    a = _named_campaign(repo, "sitting_a",
                        [_moving_card() for _ in range(first_videos)]
                        + [_still_card() for _ in range(first_photos)])
    b = _named_campaign(repo, "sitting_b",
                        [_moving_card() for _ in range(second_videos)]
                        + [_still_card() for _ in range(second_photos)],
                        captured_at="2026-08-22T00:00:00+00:00", **over)
    return a, b


def test_two_sittings_reach_the_thresholds_only_jointly(repo):
    half_v, half_p = bound.MIN_VIDEO_CARDS // 2, bound.MIN_PHOTO_CARDS // 2
    a, b = _half_and_half(repo, first_videos=half_v, first_photos=half_p,
                          second_videos=bound.MIN_VIDEO_CARDS - half_v,
                          second_photos=bound.MIN_PHOTO_CARDS - half_p)
    for single in (a, b):
        with pytest.raises(bound.VideoBoundRefused, match="section 4 requires"):
            auto.measure(single)
    result = auto.measure([a, b])
    assert result.video_cards == bound.MIN_VIDEO_CARDS
    assert result.photo_cards == bound.MIN_PHOTO_CARDS
    assert [record["dir"] for record in result.campaigns] == ["sitting_a", "sitting_b"]
    assert sum(record["cards"] for record in result.campaigns) == len(result.stats)


def test_merged_card_ids_are_namespaced_by_campaign_directory(repo):
    a, b = _half_and_half(repo, first_videos=bound.MIN_VIDEO_CARDS // 2,
                          first_photos=bound.MIN_PHOTO_CARDS // 2,
                          second_videos=bound.MIN_VIDEO_CARDS - bound.MIN_VIDEO_CARDS // 2,
                          second_photos=bound.MIN_PHOTO_CARDS - bound.MIN_PHOTO_CARDS // 2)
    result = auto.measure([a, b])
    ids = [stat.card_id for stat in result.stats]
    assert "sitting_a/card_0001" in ids and "sitting_b/card_0001" in ids
    assert len(set(ids)) == len(ids), "namespacing must make colliding bare ids unique"


def test_the_merged_artifact_names_every_sitting_inside_its_digest(repo, monkeypatch):
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: [])
    monkeypatch.setattr(bound, "_device_version_name", lambda *a, **k: None)
    a, b = _half_and_half(repo, first_videos=bound.MIN_VIDEO_CARDS // 2,
                          first_photos=bound.MIN_PHOTO_CARDS // 2,
                          second_videos=bound.MIN_VIDEO_CARDS - bound.MIN_VIDEO_CARDS // 2,
                          second_photos=bound.MIN_PHOTO_CARDS - bound.MIN_PHOTO_CARDS // 2)
    artifact, paste = auto.emit([a, b], config_path=str(_config(repo)))
    assert "campaigns" in artifact
    assert [record["dir"] for record in artifact["campaigns"]] == ["sitting_a", "sitting_b"]
    for record, campaign in zip(artifact["campaigns"], (a, b), strict=True):
        on_disk = (campaign / "manifest.json").read_bytes()
        assert record["manifest_sha256"] == hashlib.sha256(on_disk).hexdigest()
    assert artifact["video_cards"] == bound.MIN_VIDEO_CARDS
    assert {card["card_id"].split("/")[0] for card in artifact["cards"]} == {"sitting_a",
                                                                             "sitting_b"}
    # The earliest sitting dates the artifact; each sitting keeps its own stamp.
    assert artifact["captured_at"] == "2026-08-21T00:00:00+00:00"
    assert bound.verify_artifact_digest(artifact) is True
    tampered = json.loads(json.dumps(artifact))
    tampered["campaigns"][1]["manifest_sha256"] = "0" * 64
    assert bound.verify_artifact_digest(tampered) is False
    # The paste block's shape is untouched by merging.
    assert tuple(paste) == bound.PASTE_KEYS + ("accepted_circular_risk",)
    assert (a / "bound.json").is_file() and not (b / "bound.json").exists()


@pytest.mark.parametrize("field, value", [
    ("device", "OTHERPHONE"),
    ("hinge_version_name", "10.2.0"),
    ("frame_size_px", [16, 33]),
    ("config_sha256", "a" * 64),
    ("content_band", [0.1, 0.9]),
    ("autoplay_center_band_frac", 0.25),
    ("dwell_parameter_space", {"burst_frames_span": [4, 4]}),
])
def test_every_disagreement_field_refuses_by_name(repo, field, value):
    a, b = _half_and_half(repo, first_videos=bound.MIN_VIDEO_CARDS,
                          first_photos=bound.MIN_PHOTO_CARDS,
                          second_videos=1, second_photos=1, **{field: value})
    with pytest.raises(bound.VideoBoundRefused, match=f"campaigns disagree on {field}"):
        auto.measure([a, b])


def test_a_channel_or_acceptance_disagreement_refuses_before_the_merge(repo):
    """These two are load-bearing enough that the single-dir loader already refuses them."""
    a = _named_campaign(repo, "sitting_a", _passing_cards())
    b = _named_campaign(repo, "sitting_b", _passing_cards(),
                        accepted_circular_risk="I_GUESS")
    with pytest.raises(bound.VideoBoundRefused, match="acceptance phrase"):
        auto.measure([a, b])


def test_two_directories_with_the_same_basename_refuse(repo):
    a = _named_campaign(repo, "sitting_a", _passing_cards())
    nested = repo / "ops" / "calibration" / "later"
    nested.mkdir(parents=True, exist_ok=True)
    b = _named_campaign(nested, "sitting_a", _passing_cards())
    with pytest.raises(bound.VideoBoundRefused, match="share the basename"):
        auto.measure([a, b])


def test_passing_the_same_directory_twice_refuses_rather_than_double_counting(repo):
    a = _named_campaign(repo, "sitting_a", _passing_cards())
    with pytest.raises(bound.VideoBoundRefused, match="share the basename"):
        auto.measure([a, a])


def test_an_adjudication_cannot_reach_a_card_in_another_campaign(repo):
    """Bare card ids collide across sittings; scoping is what keeps them apart."""
    a, b = _half_and_half(repo, first_videos=bound.MIN_VIDEO_CARDS // 2,
                          first_photos=bound.MIN_PHOTO_CARDS // 2,
                          second_videos=bound.MIN_VIDEO_CARDS - bound.MIN_VIDEO_CARDS // 2,
                          second_photos=bound.MIN_PHOTO_CARDS - bound.MIN_PHOTO_CARDS // 2)
    # card_0001 exists in BOTH. In sitting_a it is a video; sitting_b's own card_0001 is too.
    # An entry placed in sitting_b that cites sitting_a's frame digest must not resolve.
    foreign_digest = _card_digests(a, "card_0001")[:1]
    _adjudication(b, {"card_id": "card_0001", "from_label": "video", "to_label": "photo",
                      "frame_sha256s": foreign_digest, "rationale": "wrong campaign"})
    with pytest.raises(bound.VideoBoundRefused, match="not one of the legal transitions"):
        auto.measure([a, b])


def test_each_campaigns_adjudication_applies_only_to_its_own_cards(repo):
    unsure = [_unsure_card()]
    a = _named_campaign(repo, "sitting_a",
                        [_moving_card() for _ in range(bound.MIN_VIDEO_CARDS)]
                        + [_still_card() for _ in range(bound.MIN_PHOTO_CARDS - 1)] + unsure)
    b = _named_campaign(repo, "sitting_b", [_moving_card(), _still_card()],
                        captured_at="2026-08-22T00:00:00+00:00")
    target = f"card_{bound.MIN_VIDEO_CARDS + bound.MIN_PHOTO_CARDS:04d}"
    _adjudication(a, _entry(a, target, "unsure", "photo"))
    result = auto.measure([a, b])
    assert result.photo_cards == bound.MIN_PHOTO_CARDS + 1     # 59 + 1 adjudicated + b's 1
    entries = result.adjudications["entries"]
    assert [entry["card_id"] for entry in entries] == [f"sitting_a/{target}"]
    assert result.unsure_cards == 0


def test_the_report_lists_every_merged_sitting(repo):
    a, b = _half_and_half(repo, first_videos=bound.MIN_VIDEO_CARDS // 2,
                          first_photos=bound.MIN_PHOTO_CARDS // 2,
                          second_videos=bound.MIN_VIDEO_CARDS - bound.MIN_VIDEO_CARDS // 2,
                          second_photos=bound.MIN_PHOTO_CARDS - bound.MIN_PHOTO_CARDS // 2)
    lines: list[str] = []
    bound.print_bound_report(auto.measure([a, b]), print_fn=lines.append)
    joined = "\n".join(lines)
    assert "campaign directories in this bound: 2" in joined
    assert "sitting_a" in joined and "sitting_b" in joined


def test_a_single_directory_is_byte_identical_to_the_unmerged_artifact(repo, monkeypatch):
    """The golden: passing one directory must produce exactly the pre-merge artifact.

    Digest included, so any accidental namespacing, extra key or changed timestamp shows up as a
    different evidence_sha256 rather than as a silently different artifact.
    """
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: [])
    monkeypatch.setattr(bound, "_device_version_name", lambda *a, **k: None)
    campaign = _write_campaign(repo, _passing_cards())
    config = str(_config(repo))
    as_path, _paste = bound.emit(campaign, config_path=config)
    as_list, _paste2 = bound.emit([campaign], config_path=config)
    assert as_path == as_list
    assert "campaigns" not in as_path
    assert set(as_path) == bound.BOUND_ARTIFACT_KEYS | bound.CIRCULAR_ARTIFACT_KEYS
    assert all("/" not in card["card_id"] for card in as_path["cards"])
    assert as_path["captured_at"] == "2026-08-21T00:00:00+00:00"
    assert bound.verify_artifact_digest(as_path) is True
    result = bound.measure(campaign)
    assert result.campaigns is None


def test_an_empty_directory_list_refuses(repo):
    with pytest.raises(bound.VideoBoundRefused, match="no campaign directory"):
        auto.measure([])


def test_a_merged_artifact_still_satisfies_config_validation(repo, monkeypatch):
    """config.py pins the numbers it quotes, not the artifact's full key set, so `campaigns`
    riding along must not disturb installation."""
    from operation_love import config as config_mod

    validate = getattr(config_mod, "_validate_hinge_still_photo_bound_evidence", None)
    clear = getattr(targeting_policy, "clear_installed_still_photo_bound", None)
    if validate is None or clear is None:
        pytest.skip("config-side still-photo bound validation is not present in this tree")
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: [])
    monkeypatch.setattr(bound, "_device_version_name", lambda *a, **k: None)
    a, b = _half_and_half(repo, first_videos=bound.MIN_VIDEO_CARDS // 2,
                          first_photos=bound.MIN_PHOTO_CARDS // 2,
                          second_videos=bound.MIN_VIDEO_CARDS - bound.MIN_VIDEO_CARDS // 2,
                          second_photos=bound.MIN_PHOTO_CARDS - bound.MIN_PHOTO_CARDS // 2)
    artifact, paste = auto.emit([a, b], config_path=str(_config(repo)))
    cfg = types.SimpleNamespace(
        enabled_apps=["hinge"],
        apps={"hinge": {"serial": artifact["device"], "still_photo_bound_evidence": paste}})
    try:
        validate(cfg)
        installed = targeting_policy.installed_still_photo_bound()
        assert installed is not None
        assert installed.video_cards == bound.MIN_VIDEO_CARDS
        assert installed.ground_truth_channel == auto.CIRCULAR_CHANNEL
    finally:
        clear()


# =====================================================================================
# repositioning before the advance (campaign 3, profile 14)
# =====================================================================================
#
# The walk left the page deep inside a profile -- centring the LAST station puts it at an
# arbitrary offset -- and Hinge draws the floating pass X there but not the like heart.
# `_require_deck_confirmed` needs both, so it refused, left the screen untouched and halted.
# The guard was right. The walk simply never put the page back.

def _deck_only_at_top(profiles=3, **kwargs):
    return _FakeDriver(profiles=[[_still_page(40), _still_page(80)]] * profiles,
                       deck_ready="top", **kwargs)


def test_the_walk_repositions_then_advances_when_the_deck_only_confirms_at_top(repo, small):
    driver = _deck_only_at_top()
    manifest = _run(repo, driver, max_profiles=2)
    assert driver.gestures == ["pass", "pass"]
    assert manifest["completed"] is True and manifest["ended"] == "max_profiles"
    for profile in manifest["profiles"]:
        assert profile["repositioning_gestures"] >= 1
        assert profile["deck_confirmed_before_advance"] is True
        assert profile["advanced"] is True


def test_the_repositioning_rewind_happens_before_the_gesture_and_after_the_read(repo, small):
    """Gesture ORDER, not just presence: read, then rewind, then decide."""
    driver = _deck_only_at_top(profiles=2)
    _run(repo, driver, max_profiles=1)
    order = [call for call in driver.calls
             if call in ("scroll_down_one", "scroll_to_top", "observe_deck_ready")]
    # The read scrolls first...
    assert order[0] == "scroll_down_one"
    first_probe = order.index("observe_deck_ready")
    first_rewind = order.index("scroll_to_top")
    assert first_probe < first_rewind, "the deck is probed before anything is rewound"
    assert "scroll_down_one" in order[:first_probe]
    # ...and the confirming probe comes after a rewind.
    assert order[first_rewind + 1] == "observe_deck_ready"


def test_a_deck_that_never_confirms_halts_with_the_drivers_own_message_and_no_advance(repo,
                                                                                      small):
    driver = _FakeDriver(profiles=[[_still_page(40), _still_page(80)]] * 3, deck_ready="never")
    manifest = _run(repo, driver, max_profiles=3)
    assert driver.gestures == [], "no advance may be delivered against an unconfirmed deck"
    assert manifest["ended"] == "halted"
    assert "refused the pass advance" in manifest["halt_reason"]
    assert "not positively confirmed" in manifest["halt_reason"]
    assert manifest["counts"]["passes"] == 0
    profile = manifest["profiles"][0]
    assert profile["deck_confirmed_before_advance"] is False
    assert profile["repositioning_gestures"] >= 1
    assert profile["advanced"] is False
    # The cards it did read survive the halt.
    assert manifest["cards"], "a halted profile keeps the evidence it finished writing"


def test_repositioning_is_bounded_rather_than_retried_forever(repo, small):
    driver = _FakeDriver(profiles=[[_still_page(40), _still_page(80)]], deck_ready="never")
    manifest = _run(repo, driver, max_profiles=1)
    spent = manifest["profiles"][0]["repositioning_gestures"]
    assert auto._REPOSITION_ATTEMPTS_SPAN[0] <= spent <= auto._REPOSITION_ATTEMPTS_SPAN[1]


def test_a_deck_already_confirmable_costs_no_repositioning(repo, small):
    driver = _good_driver()
    manifest = _run(repo, driver)
    assert manifest["profiles"][0]["repositioning_gestures"] == 0
    assert manifest["profiles"][0]["deck_confirmed_before_advance"] is True
    assert "scroll_to_top" not in driver.calls[:driver.calls.index("observe_deck_ready")]


def test_an_empty_profile_halts_before_a_single_repositioning_gesture(repo, small):
    """The licence still gates everything, including the rewind."""
    driver = _FakeDriver(profiles=[[_still_page(40)]] * 2, deck_ready="top")
    manifest = _run(repo, driver, max_profiles=2, segment=_no_card)
    assert manifest["ended"] == "halted_empty_profile"
    assert driver.gestures == [] and "scroll_to_top" not in driver.calls
    assert manifest["profiles"][0]["repositioning_gestures"] == 0
    assert manifest["profiles"][0]["deck_confirmed_before_advance"] is None


# =====================================================================================
# a cleanly halted campaign is still measurable
# =====================================================================================

def _halted_campaign(repo, name, cards, **overrides):
    return _named_campaign(repo, name, cards, completed=False, ended="halted",
                           halt_reason="the driver refused the pass advance", **overrides)


def test_a_halted_campaign_with_intact_cards_measures(repo):
    campaign = _halted_campaign(repo, "sitting_halted", _passing_cards())
    result = auto.measure(campaign)
    assert result.video_cards == bound.MIN_VIDEO_CARDS
    assert result.photo_cards == bound.MIN_PHOTO_CARDS
    assert result.campaigns is not None, "a halted sitting must carry its provenance"
    record = result.campaigns[0]
    assert record["ended"] == "halted"
    assert record["halt_reason"] == "the driver refused the pass advance"
    assert record["dropped_trailing_cards"] == 0


@pytest.mark.parametrize("break_it", [
    lambda card_dir: next(card_dir.glob("frame_*.png")).unlink(),
    lambda card_dir: next(card_dir.glob("frame_*.png")).write_bytes(b"truncated"),
], ids=["missing-frame", "changed-bytes"])
def test_a_halted_campaign_drops_a_truncated_trailing_card_and_measures_the_rest(repo, break_it):
    cards = _passing_cards() + [_still_card()]
    campaign = _halted_campaign(repo, "sitting_halted", cards)
    break_it(campaign / "cards" / f"card_{len(cards):04d}")
    result = auto.measure(campaign)
    assert result.photo_cards == bound.MIN_PHOTO_CARDS       # the partial card is gone
    assert len(result.stats) == len(cards) - 1
    assert result.campaigns[0]["dropped_trailing_cards"] == 1


def test_a_failure_in_the_MIDDLE_of_a_halted_campaign_is_corruption_and_refuses(repo):
    cards = _passing_cards()
    campaign = _halted_campaign(repo, "sitting_halted", cards)
    next((campaign / "cards" / "card_0002").glob("frame_*.png")).unlink()
    with pytest.raises(bound.VideoBoundRefused, match="corrupted record"):
        auto.measure(campaign)


def test_a_completed_campaign_never_tolerates_a_missing_frame(repo):
    """The drop is licensed by the halt, not by convenience."""
    campaign = _named_campaign(repo, "sitting_done", _passing_cards() + [_still_card()])
    next((campaign / "cards" / f"card_{bound.MIN_VIDEO_CARDS + bound.MIN_PHOTO_CARDS + 1:04d}")
         .glob("frame_*.png")).unlink()
    with pytest.raises(bound.VideoBoundRefused, match="is missing"):
        auto.measure(campaign)


def test_a_halted_campaign_whose_cards_all_fail_refuses(repo):
    campaign = _halted_campaign(repo, "sitting_halted", [_still_card(), _still_card()])
    for card_dir in (campaign / "cards").iterdir():
        for frame in card_dir.glob("frame_*.png"):
            frame.unlink()
    with pytest.raises(bound.VideoBoundRefused, match="no card that survives verification"):
        auto.measure(campaign)


def test_a_halted_sitting_merges_with_a_completed_one(repo):
    half_v, half_p = bound.MIN_VIDEO_CARDS // 2, bound.MIN_PHOTO_CARDS // 2
    done = _named_campaign(repo, "sitting_done",
                           [_moving_card() for _ in range(half_v)]
                           + [_still_card() for _ in range(half_p)])
    halted = _halted_campaign(
        repo, "sitting_halted",
        [_moving_card() for _ in range(bound.MIN_VIDEO_CARDS - half_v)]
        + [_still_card() for _ in range(bound.MIN_PHOTO_CARDS - half_p)],
        captured_at="2026-08-22T00:00:00+00:00")
    result = auto.measure([done, halted])
    assert result.video_cards == bound.MIN_VIDEO_CARDS
    assert [record["ended"] for record in result.campaigns] == ["targets_reached", "halted"]
    lines: list[str] = []
    bound.print_bound_report(result, print_fn=lines.append)
    assert "HALTED: halted" in "\n".join(lines)


def test_the_artifact_records_each_campaigns_ended_state_inside_its_digest(repo, monkeypatch):
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: [])
    monkeypatch.setattr(bound, "_device_version_name", lambda *a, **k: None)
    campaign = _halted_campaign(repo, "sitting_halted", _passing_cards())
    artifact, _paste = auto.emit(campaign, config_path=str(_config(repo)))
    assert [record["ended"] for record in artifact["campaigns"]] == ["halted"]
    assert artifact["campaigns"][0]["halt_reason"] == "the driver refused the pass advance"
    assert bound.verify_artifact_digest(artifact) is True
    for mutate in (lambda a: a["campaigns"][0].update({"ended": "targets_reached"}),
                   lambda a: a["campaigns"][0].update({"halt_reason": None}),
                   lambda a: a["campaigns"][0].update({"dropped_trailing_cards": 5})):
        tampered = json.loads(json.dumps(artifact))
        mutate(tampered)
        assert bound.verify_artifact_digest(tampered) is False
    # Bare card ids: one directory is still one directory, halted or not.
    assert all("/" not in card["card_id"] for card in artifact["cards"])


def test_a_completed_single_campaign_artifact_is_unchanged_by_all_of_this(repo, monkeypatch):
    """The golden, re-asserted: no campaigns block, no namespacing, same digest."""
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: [])
    monkeypatch.setattr(bound, "_device_version_name", lambda *a, **k: None)
    campaign = _write_campaign(repo, _passing_cards())
    artifact, paste = bound.emit(campaign, config_path=str(_config(repo)))
    assert "campaigns" not in artifact
    assert set(artifact) == bound.BOUND_ARTIFACT_KEYS | bound.CIRCULAR_ARTIFACT_KEYS
    assert all("/" not in card["card_id"] for card in artifact["cards"])
    assert tuple(paste) == bound.PASTE_KEYS + ("accepted_circular_risk",)
    assert bound.verify_artifact_digest(artifact) is True
