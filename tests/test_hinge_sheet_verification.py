"""The driver half of doc 5.6: verify the open comment sheet, THEN type. Never the other order.

`tests/test_item_verify.py` covers the comparison itself. What this file pins is the thing that
makes it worth having — that there is no path through `hinge._like_comment_sheet` from a tapped
heart to `adb.text(...)` that does not pass a match first, and that every refusal leaves the run
stopped with the screen where it was rather than sending something.

Every frame is SYNTHESISED. The captures the constants were measured against are real people's
profiles and are gitignored, so only geometry and grey levels from them appear here: two painted
"cards" and a painted comment sheet at the sheet geometry item_verify's docstring records.
"""
import dataclasses

import cv2
import numpy as np
import pytest

from operation_love.drivers import (
    hinge, item_crops, item_identity, item_nav, item_verify, scroll_step)
from operation_love.drivers.frameshift import ShiftEstimationError
from operation_love.drivers.base import (ActionCancelled, OBSERVE_ITEM_INCONCLUSIVE,
                                         OBSERVE_ITEM_MATCH, OBSERVE_ITEM_MISMATCH)
from operation_love.drivers.hinge import HingeActionError, HingeDriver, HingeTargetingError
from operation_love.drivers.like_composer import ComposerSurface, Rect
from operation_love.drivers.scroll_top import band_fingerprint
from operation_love.drivers.segment import SegmentationError

_W, _H = 1080, 2400
_CARD_W, _CARD_H = 974, 900
_SHEET_BG, _PREVIEW_X0, _PREVIEW_W, _PREVIEW_Y0 = 250, 95, 890, 650
_COMMENT_RECT = Rect(95, 1597, 985, 1775)
_SEND_RECT = Rect(390, 1807, 985, 1916)
_CONFIRM_POINT = (695, 1856)
_COMPOSER_SURFACE = ComposerSurface(
    layout_id="hinge_inline_v1", comment_rect=_COMMENT_RECT, send_rect=_SEND_RECT,
    confirm_point=_CONFIRM_POINT)
_CONFIRM_TEMPLATE = hinge._load_template("hinge_send_like.png")
assert _CONFIRM_TEMPLATE is not None, "the inline-composer fixture needs Hinge's shipped glyph"


def _calibration(**overrides):
    values = {
        "schema_version": 3,
        "hinge_version_name": "9.134.0",
        "frame_size_px": [_W, _H],
        "composer_layout_id": "hinge_inline_v1",
        "item_selection_policy_id": "hinge_photos_only_v2",
        "identity_match_max_dist": 2.0,
        "inline_item_max_dist": 10.0,
        "device": "pixel",
        "calibrated_at": "2026-08-12",
        "identity_band": list(hinge.HINGE_SPEC.identity_band),
        "content_band": list(hinge.HINGE_SPEC.content_band),
    }
    values.update(overrides)
    return values


def _card(seed: int) -> bytes:
    """One painted card, as the PNG bytes an `ItemCrop` carries."""
    rows, cols = np.mgrid[0:_CARD_H, 0:_CARD_W]
    pattern = (55 * np.sin(2 * np.pi * rows / (70 + 30 * seed))
               * np.cos(2 * np.pi * cols / (110 + 50 * seed)))
    gray = np.clip(100 + pattern + np.linspace(-30, 30, _CARD_H)[:, None]
                   + np.random.default_rng(seed).integers(-20, 21, size=(_CARD_H, _CARD_W)),
                   0, 210).astype(np.uint8)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    assert ok
    return buf.tobytes()


def _sheet(card: bytes) -> bytes:
    """A selected item plus Hinge's structurally verified inline composer.

    This leaves the identity strip and item pixels intact while drawing the two independent
    controls the driver now requires before it may tap text/send coordinates.
    """
    image = cv2.imdecode(np.frombuffer(card, np.uint8), cv2.IMREAD_COLOR)
    height = int(round(image.shape[0] * _PREVIEW_W / image.shape[1]))
    canvas = np.full((_H, _W, 3), _SHEET_BG, dtype=np.uint8)
    canvas[_PREVIEW_Y0:_PREVIEW_Y0 + height, _PREVIEW_X0:_PREVIEW_X0 + _PREVIEW_W] = cv2.resize(
        image, (_PREVIEW_W, height), interpolation=cv2.INTER_AREA)
    canvas[_COMMENT_RECT.y0:_COMMENT_RECT.y0 + 2,
           _COMMENT_RECT.x0:_COMMENT_RECT.x1] = 222
    canvas[_COMMENT_RECT.y1 - 2:_COMMENT_RECT.y1,
           _COMMENT_RECT.x0:_COMMENT_RECT.x1] = 222
    canvas[_SEND_RECT.y0:_SEND_RECT.y1, _SEND_RECT.x0:_SEND_RECT.x1] = 228
    glyph_h, glyph_w = _CONFIRM_TEMPLATE.shape
    x, y = _CONFIRM_POINT
    canvas[y - glyph_h // 2:y - glyph_h // 2 + glyph_h,
           x - glyph_w // 2:x - glyph_w // 2 + glyph_w] = cv2.cvtColor(
               _CONFIRM_TEMPLATE, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", canvas)
    assert ok
    return buf.tobytes()


_CARDS = [_card(1), _card(2)]
_SHEETS = [_sheet(c) for c in _CARDS]


def _payload(drift: float = 0.0) -> item_crops.ItemPayload:
    """A two-item payload built from the painted cards, with real signatures.

    Assembled directly rather than through `build_item_payload`: what is under test here is the
    DRIVER's ordering, and a whole synthetic scroll world would be a second copy of
    tests/test_item_crops.py's fixture with nothing extra to say.
    """
    crops = [
        item_crops.ItemCrop(
            kind=item_crops.CROP_ITEM, number=i + 1, heart_ordinal=i + 1, page_y0=0,
            page_y1=_CARD_H, x0=53, x1=53 + _CARD_W, frame_index=i, frame_y0=0,
            frame_y1=_CARD_H, image=card,
            signature=item_crops.signature_of(card, y0=0, y1=_CARD_H, x0=0, x1=_CARD_W),
            signature_drift=drift, drift_frames=(i,), nearest_item_distance=None,
            reason="painted")
        for i, card in enumerate(_CARDS)]
    item_crops._fill_nearest_item_distance(crops)
    return item_crops.ItemPayload(crops=tuple(crops), truncated=False, at_scroll_top=True,
                                  signature_grid=(32, 32), failures=())


class SheetAdb:
    """A phone that only ever shows the comment sheet, and records what was done to it."""

    def __init__(self, sheet: bytes):
        self.sheet = sheet
        self.taps: list[tuple[int, int]] = []
        self.texts: list[str] = []
        self.calls: list[str] = []

    def screen_size(self):
        return (_W, _H)

    def devices(self):
        return ["pixel"]

    def shell(self, command="", **_):
        if "dumpsys package" in command:
            return "versionName=9.134.0\n"
        return ""

    def screencap(self):
        return self.sheet

    def tap(self, x, y):
        self.taps.append((int(x), int(y)))
        self.calls.append("tap")

    def swipe(self, *_a, **_k):
        pass

    def scroll_up(self, *_a, **_k):
        pass

    def text(self, s):
        self.texts.append(s)
        self.calls.append("text")


class SwitchingSheetAdb(SheetAdb):
    """Shows the indexed profile until the comment sheet, then a foreign header."""

    def __init__(self, frames):
        super().__init__(frames[-1])
        self._frames = iter(frames)

    def screencap(self):
        return next(self._frames)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)


class _FakeIndex:
    """Just enough of an `ItemIndex` for the identity gate: the fingerprint of the profile these
    crops were cut from. The real thing is built by `build_item_index` and has its own tests."""

    def __init__(self, identity, translation=(1, 2)):
        self.identity = identity
        self.translation = tuple(translation)


def _identity_of(frame: bytes, *, foreign: bool = False) -> item_identity.ProfileIdentity:
    """The identity an index built from `frame`'s profile would carry.

    `foreign=True` shifts every cell instead, which is what an index belonging to a DIFFERENT
    person looks like to this comparison — the two calibration profiles measure 7.287 grey levels
    apart at this grid against a 3.0 bound, and the closest of six real profiles 2.565, so the
    shift here is deliberately large enough to be unambiguous rather than a reproduction of
    either.
    """
    band = hinge.HINGE_SPEC.identity_band
    grid = item_identity._IDENTITY_GRID
    fingerprint = band_fingerprint(frame, identity_band=band, grid=grid)
    if foreign:
        fingerprint = tuple(max(0, value - 40) for value in fingerprint)
    return item_identity.ProfileIdentity(
        fingerprint=fingerprint, band=tuple(band), grid=grid, frame_index=0,
        scroll_top_distance=99.0, reason="painted", agreeing_frames=3)


def _fake_navigate(recorder, *, point):
    """A stand-in for `item_nav.navigate_to_item` that records its call and returns a target.

    Only `.point` and `.reason` are read by the driver, so only those are supplied: this file is
    about the ORDER of the driver's steps, and navigation itself has a closed synthetic loop of
    its own in tests/test_item_nav.py."""
    class _Target:
        def __init__(self):
            self.point = point
            self.heart_ordinal = 1
            self.scrolls = 3
            self.climbed_px = 742
            self.agreement_px = 0
            self.hearts_counted = 2
            self.frame = b"landing frame"
            self.reason = "walked up to it"

    def navigate(driver, index, model_index, *, entry_reference, **_kw):
        recorder.append((driver, index, model_index, entry_reference, _kw))
        return _Target()
    return navigate


def _driver(adb, monkeypatch, *, payload=None, unavailable="", identity_frame=None,
            foreign_identity=False, anchor=None, targeting_calibration=True):
    class C:
        apps = {"hinge": {
            "serial": "pixel", "halt_on_error": False,
            **({"targeting_calibration": _calibration()} if targeting_calibration else {}),
        }}

    driver = HingeDriver(C())
    driver._adb = adb
    driver._touch = adb
    driver._targeting_runtime_version_name = "9.134.0"
    driver._targeting_runtime_frame_size = (_W, _H)
    driver._current_item_payload = payload
    driver._current_items_unavailable = unavailable
    # The index behind the crops, carrying whose profile they came from. Fingerprinted off the
    # very screen the fake phone will show, because in the ordinary case the like starts on the
    # profile that was just enumerated -- see `_confirm_payload_profile`.
    driver._current_item_index = None if payload is None else _FakeIndex(
        _identity_of(identity_frame if identity_frame is not None else adb.sheet,
                     foreign=foreign_identity))
    # The entry anchor bottom-up navigation measures its page offset against. Left None by
    # default so the "half a table is a missing table" refusal stays the DEFAULT shape for a
    # fixture that never opted into navigation, and set explicitly by the tests that do.
    driver._current_item_anchor = anchor
    # Everything the like flow does BESIDE verification, stubbed: this file is about the order of
    # the remaining two steps, not about scrolling, glyph matching or the upsell modal, each of
    # which has its own tests.
    monkeypatch.setattr(HingeDriver, "_scroll_to_top",
                        lambda self, should_stop=None: adb.calls.append("scroll_to_top"))
    # Returns a bare point since 2026-08-12: _locate_target_heart either lands on the item it was
    # asked for or raises, so there is no second "and this is actually the wrong one" flag left.
    monkeypatch.setattr(HingeDriver, "_locate_target_heart",
                        lambda self, index, should_stop=None: (540, 1200))
    def await_composer(self, tries=5):
        adb.calls.append("await_sheet")
        return _COMPOSER_SURFACE
    monkeypatch.setattr(HingeDriver, "_await_sheet_open", await_composer)
    monkeypatch.setattr(HingeDriver, "_handle_rose_upsell", lambda self, tries=2: False)
    # Model-item tests in this file exercise verification/order rather than the closed-loop
    # navigator (which has its own synthetic suite).  Supplying an anchor opts into this simple
    # successful navigation seam; tests that need a navigation failure override it afterwards.
    if anchor is not None:
        monkeypatch.setattr(hinge, "navigate_to_item", _fake_navigate([], point=(540, 1200)))
    return driver


# =====================================================================================
# THE HEADLINE: verify, then type
# =====================================================================================

def test_a_matching_sheet_is_verified_before_the_opener_is_typed():
    """Order is the guarantee. The tap opens the sheet, the sheet is checked, and only then does
    a character reach the phone."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload(), anchor=b"fixture anchor")
        driver.like("two sentences, no dashes", model_item_index=1)
    assert adb.texts == ["two sentences, no dashes"]
    assert adb.calls.index("await_sheet") < adb.calls.index("text")
    assert adb.taps[1:] == [_COMMENT_RECT.center, _CONFIRM_POINT]


def test_missing_targeting_calibration_refuses_a_model_item_before_any_gesture():
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload(), anchor=b"fixture anchor",
                         targeting_calibration=False)
        # The absent-mapping reason is operator-facing (hub banner, startup notice, bug
        # reports): plain prose only, never the RUNBOOK's literal `apps.<app>.…` schema
        # notation, which renders as a failed-interpolation bug (reported 2026-08-21).
        assert driver._targeting_calibration_unavailable == "not configured in config.yaml"
        assert "<app>" not in driver._targeting_calibration_unavailable
        with pytest.raises(HingeTargetingError, match="targeting_calibration"):
            driver.like("an opener", model_item_index=1)
    assert adb.calls == [] and adb.taps == [] and adb.texts == []


def test_absent_calibration_reasons_match_the_hub_banner_trim_byte_for_byte():
    """The two run-level reasons are pinned EXACTLY because hub.html depends on their bytes.

    renderSwipe's targeting-setup branch strips these sentences' internal consequence clauses
    ("…usable consumer" / "no opener text is offered") before showing the reason to the
    operator as fine print — a literal-string trim that degrades harmlessly (full sentence
    shown) if the driver wording drifts. Harmless must not also be silent: the operator-jargon
    leak is the exact class the 2026-08-21 bug report was about. Rewording either sentence
    here must fail this test, whose fix is to update the trim literals in hub.html's
    renderSwipe alongside the driver.
    """
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload(), anchor=b"fixture anchor",
                         targeting_calibration=False)
        assert driver._item_enumeration_blocker() == (
            "apps.hinge.targeting_calibration is unavailable (not configured in config.yaml), "
            "so a model-selected item could not be verified or targeted and no numbered item "
            "list would have a usable consumer")
        assert driver.targeted_suggestion_blocker() == (
            "targeted suggestion is unavailable because apps.hinge.targeting_calibration is "
            "unavailable (not configured in config.yaml); no opener text is offered")


def test_targeting_calibration_is_parsed_and_used_for_model_item_likes():
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload(), anchor=b"fixture anchor")
        assert driver.targeting_calibration is not None
        assert driver.targeting_calibration.identity_match_max_dist == 2.0
        assert driver.targeting_calibration.inline_item_max_dist == 10.0
        driver.like("an opener", model_item_index=1)
    assert adb.texts == ["an opener"]


def test_stop_after_navigation_prevents_the_heart_tap():
    """A Stop in navigation's return-to-tap window must not turn into one last like."""
    adb = SheetAdb(_SHEETS[0])
    stopped = {"now": False}
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload(), anchor=b"the read's last frame")

        def navigate(_self, _index, *, should_stop=None):
            assert should_stop is not None
            stopped["now"] = True
            return (540, 1200)

        mp.setattr(HingeDriver, "_navigate_to_model_item", navigate)
        with pytest.raises(ActionCancelled):
            driver.like("an opener", model_item_index=1,
                        should_stop=lambda: stopped["now"])
    assert adb.taps == [] and adb.texts == []


def test_stop_after_the_sheet_opens_prevents_text_and_send():
    """Once a sheet is open Stop leaves it for inspection; it cannot type or send."""
    adb = SheetAdb(_SHEETS[0])
    stopped = {"now": False}
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload(), anchor=b"fixture anchor")

        def sheet_opened(_self, tries=5):
            adb.calls.append("await_sheet")
            stopped["now"] = True
            return _COMPOSER_SURFACE

        mp.setattr(HingeDriver, "_await_sheet_open", sheet_opened)
        with pytest.raises(ActionCancelled):
            driver.like("an opener", model_item_index=1,
                        should_stop=lambda: stopped["now"])
    # The only tap is the pre-stop heart; no comment-box or Send Like input follows it.
    assert len(adb.taps) == 1 and adb.texts == []


def test_navigation_translates_model_item_through_the_payload_heart_ordinal():
    """An excluded selectable block makes model numbering differ from ItemIndex numbering."""
    adb = SheetAdb(_SHEETS[0])
    seen = []
    payload = _payload()
    crops = list(payload.crops)
    # Model item 1 is actually the second selectable heart in the full index.
    crops[0] = dataclasses.replace(crops[0], heart_ordinal=2)
    remapped_payload = dataclasses.replace(payload, crops=tuple(crops))
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=remapped_payload, anchor=b"entry")
        driver._current_item_index.translation = (1, 2)

        def navigate(_driver, _index, navigation_index, *, entry_reference, **_kw):
            seen.append((navigation_index, entry_reference, _kw.get("selected_model_item_index")))
            return type("Target", (), {
                "point": (540, 1200), "heart_ordinal": 2, "scrolls": 0,
                "climbed_px": 0, "agreement_px": 0, "hearts_counted": 2,
                "frame": b"landing", "reason": "synthetic",
            })()

        mp.setattr(hinge, "navigate_to_item", navigate)
        assert driver._navigate_to_model_item(1) == (540, 1200)
    assert seen == [(2, b"entry", 1)]


def test_shannon_style_photo_and_video_policy_translation_targets_heart_seven():
    """Policy demotions cannot make model item 2 tap page heart 2.

    This is the successful Shannon manifest shape in synthetic form: photos survive at hearts
    1/7/9, written-or-unknown cards are readable context, and confirmed or unscreenable video
    cards are withheld entirely.  The navigator still counts every physical selectable heart.
    """
    def crop(kind, heart, *, number=None, image=None, reason):
        signature = (item_crops.signature_of(
            image, y0=0, y1=_CARD_H, x0=0, x1=_CARD_W)
                     if kind == item_crops.CROP_ITEM else None)
        return item_crops.ItemCrop(
            kind=kind, number=number, heart_ordinal=heart,
            page_y0=heart * 1000, page_y1=heart * 1000 + _CARD_H,
            x0=53, x1=53 + _CARD_W, frame_index=heart, frame_y0=0, frame_y1=_CARD_H,
            image=image, signature=signature, signature_drift=None, drift_frames=(),
            nearest_item_distance=None, reason=reason)

    photo_1, photo_7, photo_9 = _card(21), _card(22), _card(23)
    payload = item_crops.ItemPayload(crops=(
        crop(item_crops.CROP_ITEM, 1, number=1, image=photo_1, reason="photo"),
        crop(item_crops.CROP_CONTEXT, 2, image=_card(24), reason="photo_only: written"),
        crop(item_crops.CROP_EXCLUDED, 3, reason="video_mute_v1: exact mute match"),
        crop(item_crops.CROP_EXCLUDED, 4, reason="video_mute_v1: screen failed closed"),
        crop(item_crops.CROP_CONTEXT, 5, image=_card(25), reason="photo_only: written"),
        crop(item_crops.CROP_EXCLUDED, 6, reason="video_mute_v1: screen failed closed"),
        crop(item_crops.CROP_ITEM, 7, number=2, image=photo_7, reason="photo"),
        crop(item_crops.CROP_CONTEXT, 8, image=_card(26), reason="photo_only: unknown"),
        crop(item_crops.CROP_ITEM, 9, number=3, image=photo_9, reason="photo"),
    ), truncated=False, at_scroll_top=True, signature_grid=(32, 32), failures=())
    assert payload.usable and payload.translation == (1, 7, 9)
    assert [crop.heart_ordinal for crop in payload.excluded] == [3, 4, 6]
    assert [crop.heart_ordinal for crop in payload.context] == [2, 5, 8]

    adb = SheetAdb(_SHEETS[0])
    seen = []
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=payload, anchor=b"entry")
        driver._current_item_index.translation = tuple(range(1, 10))

        def navigate(_driver, _index, navigation_index, *, entry_reference, **_kw):
            seen.append((navigation_index, entry_reference, _kw.get("selected_model_item_index")))
            return type("Target", (), {
                "point": (540, 1200), "heart_ordinal": 7, "scrolls": 0,
                "climbed_px": 0, "agreement_px": 0, "hearts_counted": 7,
                "frame": b"landing", "reason": "synthetic",
            })()

        mp.setattr(hinge, "navigate_to_item", navigate)
        assert driver._navigate_to_model_item(2) == (540, 1200)
        assert driver._navigate_to_model_item(3) == (540, 1200)

    assert seen == [(7, b"entry", 2), (9, b"entry", 3)]
    assert seen[0][0] not in {2, 3, 4, 5, 6, 8}


def test_runtime_parser_rejects_a_known_unsafe_sheet_ceiling_even_without_config_validation():
    """Embedding callers can construct a Config-like object directly; the gesture gate repeats
    the known false-accept caps rather than trusting that config.validate ran elsewhere."""
    class C:
        apps = {"hinge": {"serial": "pixel", "targeting_calibration": _calibration(
            inline_item_max_dist=14.91)}}

    driver = HingeDriver(C())
    assert driver.targeting_calibration is None
    assert "14.91" in driver._targeting_calibration_unavailable


def test_runtime_parser_refuses_a_calibration_for_another_adb_serial():
    class C:
        apps = {"hinge": {"serial": "other-pixel", "targeting_calibration": _calibration()}}

    driver = HingeDriver(C())
    assert driver.targeting_calibration is None
    assert "does not exactly match" in driver._targeting_calibration_unavailable


def test_runtime_parser_refuses_calibration_when_effective_geometry_changes():
    """Direct construction must not retain bounds after a crop override changes."""
    class C:
        apps = {"hinge": {
            "serial": "pixel",
            "identity_band": [0.11, 0.048, 0.80, 0.094],
            "targeting_calibration": _calibration(),
        }}

    driver = HingeDriver(C())
    assert driver.targeting_calibration is None
    assert "identity_band does not exactly match" in driver._targeting_calibration_unavailable


def test_runtime_parser_accepts_calibration_bound_to_effective_geometry_override():
    class C:
        apps = {"hinge": {
            "serial": "pixel",
            "identity_band": [0.11, 0.048, 0.80, 0.094],
            "content_band": [0.13, 0.87],
            "targeting_calibration": _calibration(
                identity_band=[0.11, 0.048, 0.80, 0.094], content_band=[0.13, 0.87]),
        }}

    driver = HingeDriver(C())
    assert driver.targeting_calibration is not None
    assert driver.targeting_calibration.identity_band == (0.11, 0.048, 0.80, 0.094)
    assert driver.targeting_calibration.content_band == (0.13, 0.87)


def test_post_tap_foreign_identity_stops_even_when_the_item_verifier_would_match():
    """A deck race must not attach text to a lookalike preview on another profile's sheet."""
    foreign = cv2.imdecode(np.frombuffer(_SHEETS[0], np.uint8), cv2.IMREAD_COLOR)
    x0, y0, x1, y1 = hinge.HINGE_SPEC.identity_band
    foreign[int(y0 * _H):int(y1 * _H), int(x0 * _W):int(x1 * _W)] = 0
    ok, foreign_png = cv2.imencode(".png", foreign)
    assert ok
    foreign_sheet = foreign_png.tobytes()
    assert item_verify.verify_sheet_item(foreign_sheet, _payload(), 1).matched

    # _confirm_payload_profile and _snap see the indexed profile; only the post-tap sheet races.
    adb = SwitchingSheetAdb([_SHEETS[0], foreign_sheet])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload(), identity_frame=_SHEETS[0],
                         anchor=b"fixture anchor")
        with pytest.raises(HingeTargetingError, match="like sheet is not on the profile"):
            driver.like("an opener", model_item_index=1)
    assert adb.texts == [] and "text" not in adb.calls


def test_a_sheet_showing_a_different_item_stops_the_run_with_nothing_typed():
    """The failure this whole redesign exists to stop: the opener was written about item 1 and
    the sheet that opened is item 2. Doc 5.6 -- do not type, do not send, never a commentless
    like, record intended and actual, leave the screen for debugging."""
    adb = SheetAdb(_SHEETS[1])                       # the sheet shows item 2...
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload(), anchor=b"fixture anchor")
        with pytest.raises(HingeActionError) as exc:
            driver.like("an opener about item one", model_item_index=1)   # ...we asked for 1
    assert adb.texts == []                           # nothing typed
    assert "text" not in adb.calls
    message = str(exc.value)
    assert "intended model item 1" in message and "actual 2" in message
    assert "NOT typed" in message and "NOT sent" in message
    assert "\n" not in message, "the hub shows the last line of the traceback, so keep it to one"


def test_a_sheet_that_is_not_any_indexed_item_stops_the_run_too():
    """A sheet nearest to item 1 but nowhere near it. Doc 5.6's bound is what catches this, and
    without it a nearest-match alone would confirm whichever item happened to be least unlike a
    screen that is none of them."""
    adb = SheetAdb(_sheet(_card(9)))
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload(), anchor=b"fixture anchor")
        with pytest.raises(HingeActionError):
            driver.like("an opener", model_item_index=1)
    assert adb.texts == []


def test_a_sheet_that_cannot_be_read_at_all_stops_the_run_rather_than_typing():
    """"Could not look" and "looked and it is wrong" call for the same action one tap away from
    typing; only the diagnosis differs."""
    flat = np.full((_H, _W), _SHEET_BG, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", flat)
    assert ok
    adb = SheetAdb(buf.tobytes())
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload(), anchor=b"fixture anchor")
        with pytest.raises(HingeActionError) as exc:
            driver.like("an opener", model_item_index=1)
    assert adb.texts == []
    assert "inline composer changed" in str(exc.value)


# =====================================================================================
# Refused BEFORE the tap
# =====================================================================================

def test_a_model_item_with_no_crops_behind_it_is_refused_without_touching_the_screen():
    """Doc 5.3: treat a missing table as a hard stop, never as a reason to fall back to a fixed
    coordinate. The invalidation reason is quoted, because "the deck advanced" and "this capture
    could not be indexed" call for different next moves."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=None, anchor=b"fixture anchor",
                         unavailable="the deck advanced after this like")
        with pytest.raises(HingeActionError) as exc:
            driver.like("an opener", model_item_index=1)
    assert adb.calls == [] and adb.taps == [] and adb.texts == []
    assert "the deck advanced after this like" in str(exc.value)


def test_an_item_number_outside_the_list_is_refused_without_touching_the_screen():
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload(), anchor=b"fixture anchor")
        with pytest.raises(HingeActionError) as exc:
            driver.like("an opener", model_item_index=7)
    assert adb.calls == [] and adb.texts == []
    assert "outside 1..2" in str(exc.value)


def test_crops_belonging_to_a_different_profile_are_refused_before_any_gesture():
    """THE STALE-TABLE CASE, and doc 5.3's argument about it was measured FALSE.

    5.3 called a stale translation table "a reliability bug rather than a safety one" because
    "the stored crops are stale too, so the post-tap signature check compares the opened sheet
    against the wrong reference, fails, and stops the run". A validation pass drove exactly that
    -- a payload for one profile, a sheet showing another's card -- through this method and got a
    MATCH, a typed opener and a SENT like, 10 times in 540 comparisons: `item_verify` is a
    closed-set test over one payload's items with no absolute ceiling, so out-of-payload content
    only has to beat that payload's own internal spacing.

    So whose profile this is gets asked FIRST, while the sticky header is still readable, and it
    is asked with the same primitive `item_nav.navigate_to_item` uses at the same point in its
    own sequence. Note what this test does NOT claim: the sheet here would have verified. The
    identity gate is what refused it.
    """
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload(), foreign_identity=True,
                         anchor=b"fixture anchor")
        with pytest.raises(HingeTargetingError) as exc:
            driver.like("an opener about item one", model_item_index=1)
    assert adb.calls == [] and adb.taps == [] and adb.texts == []
    message = str(exc.value)
    assert "not the one model item 1's crops were cut from" in message
    assert "NOT sent" in message
    # ...and the very same call passes the gate when the crops belong to the profile on screen,
    # which is what makes it a gate rather than a blanket refusal.
    adb2 = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb2, mp, payload=_payload(), anchor=b"fixture anchor")
        driver.like("an opener about item one", model_item_index=1)
    assert adb2.texts == ["an opener about item one"]


def test_a_screen_whose_identity_cannot_be_read_stops_rather_than_passing():
    """"Cannot tell" is never "yes". At a scroll top the identity strip is Hinge's own
    filter-chips row -- byte-identical for two different people -- so there is nothing there to
    tell anyone apart, and a gate that read silence as agreement would pass for everybody."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload(), anchor=b"fixture anchor")
        driver._current_item_index = _FakeIndex(item_identity.ProfileIdentity(
            fingerprint=None, band=tuple(hinge.HINGE_SPEC.identity_band),
            grid=item_identity._IDENTITY_GRID, frame_index=None, scroll_top_distance=None,
            reason="this capture never showed a sticky header"))
        with pytest.raises(HingeTargetingError) as exc:
            driver.like("an opener", model_item_index=1)
    assert adb.calls == [] and adb.taps == [] and adb.texts == []
    assert "NOT sent" in str(exc.value)


def test_a_model_item_number_now_routes_to_counting_navigation_and_never_rewinds():
    """THE HANDOVER, and this test used to pin its refusal.

    `model_item_index=2` with `item_index` left at its default is the shape counting navigation
    produces. Until 2026-08-12 it was a hard stop naming `item_nav.navigate_to_item` as the
    missing piece -- before that it was worse, falling through to `_locate_target_heart(None)`,
    which reads that None as "nobody named an item" and taps the TOPMOST HEART. It is now the
    navigation call, and the three things worth pinning are what it does NOT do:

      * it does not call `_locate_target_heart` at all -- the capture-order search is a different
        index space and cannot resolve a model item number;
      * it does not call `_scroll_to_top`. Doc 5.5's bottom-up design walks up from where the
        read left the card, and a rewind here would both cost ~51 gestures and put the identity
        strip on a screen where it carries no identity;
      * and the point it taps is the one navigation returned, not a fallback.
    """
    adb = SheetAdb(_SHEETS[1])                # the sheet really is showing item 2
    located, navigated = [], []
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload(), anchor=b"the read's last frame")
        mp.setattr(HingeDriver, "_locate_target_heart",
                   lambda self, index, should_stop=None: (located.append(index), (540, 1200))[1])
        mp.setattr(hinge, "navigate_to_item", _fake_navigate(navigated, point=(931, 1477)))
        index = driver._current_item_index        # `like()` clears it in a finally, by design
        driver.like("an opener about item two", model_item_index=2)

    assert located == [], "the capture-order heart search must not be entered"
    assert "scroll_to_top" not in adb.calls, "bottom-up navigation must not rewind"
    assert adb.taps[0] == (931, 1477), "the tap is the point navigation returned"
    assert adb.texts == ["an opener about item two"]
    # ...and navigation got the driver, the index, the model's number and the anchor frame.
    (driver_arg, index_arg, model_index_arg, reference_arg, navigation_kwargs), = navigated
    assert driver_arg is driver and index_arg is index
    assert (model_index_arg, reference_arg) == (2, b"the read's last frame")
    assert navigation_kwargs["identity_match_max_dist"] == 2.0


def test_half_a_translation_table_is_a_missing_one_and_is_refused_before_any_gesture():
    """Doc 5.3: "treat a missing table as a hard stop, never as a reason to fall back to a fixed
    coordinate". Bottom-up navigation needs TWO things from the driver -- the index and the entry
    anchor frame the page offset is measured against -- and they are set and cleared together, so
    a payload present without an anchor is not a state `_invalidate_item_index` can produce. It is
    checked anyway, because half a table is a missing table and the alternative to checking is a
    page origin nobody measured."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload())     # index set, anchor NOT set
        assert driver._current_item_index is not None and driver._current_item_anchor is None
        with pytest.raises(HingeTargetingError) as exc:
            driver.like("an opener about item two", model_item_index=2)
    assert adb.calls == [] and adb.taps == [] and adb.texts == []
    message = str(exc.value)
    assert "entry anchor frame" in message and "NOT sent" in message


def test_a_navigation_refusal_becomes_a_targeting_stop_with_nothing_typed():
    """Doc 9's blocker 4, closed at the driver boundary. `item_nav` refuses with a coded
    `ItemNavigationError`, and three of its dependencies refuse UNCODED by design
    (`ScrollStepError`, `SegmentationError`, `ShiftEstimationError`, plus `IdentityError`). All
    of them mean the same thing to the operator and to `worker.py` -- the like was not put on the
    item the opener was written about, so it was not put anywhere -- so all of them become
    `HingeTargetingError`, which is the one type the worker routes to a stop."""
    failures = [
        item_nav.ItemNavigationError(item_nav.NAV_COUNT_DISAGREES, "the count disagrees"),
        scroll_step.ScrollStepError("no legal gesture respects this spacing"),
        SegmentationError("the frame will not segment"),
        ShiftEstimationError("no page space spans these frames"),
        item_identity.IdentityError("the identity band could not be read"),
    ]
    for failure in failures:
        adb = SheetAdb(_SHEETS[0])
        with pytest.MonkeyPatch.context() as mp:
            driver = _driver(adb, mp, payload=_payload(), anchor=b"ref")

            def _raise(*_a, failure=failure, **_kw):
                raise failure
            mp.setattr(hinge, "navigate_to_item", _raise)
            with pytest.raises(HingeTargetingError) as exc:
                driver.like("an opener about item two", model_item_index=2)
        assert adb.taps == [] and adb.texts == [], type(failure).__name__
        assert exc.value.stage == "navigate" and exc.value.intended == 2
        assert exc.value.index_space == "model_items"
        assert "NOT sent" in str(exc.value)


def test_that_refusal_does_not_depend_on_there_being_an_opener_to_misplace():
    """A commentless like on a model item number is still a like spent on an item nobody could
    aim at. The opener-less variant of the guard above it is legal precisely because nothing was
    chosen; here something WAS chosen and cannot be reached."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload())
        with pytest.raises(HingeTargetingError):
            driver.like(None, model_item_index=2)
    assert adb.calls == [] and adb.taps == []


def test_mixed_capture_and_model_item_indices_are_refused_before_any_gesture():
    """A public caller cannot make two incompatible declarations of the target item."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload())
        with pytest.raises(HingeTargetingError, match="both capture-order"):
            driver.like("an opener about item one", 0, model_item_index=1)
    assert adb.calls == [] and adb.taps == [] and adb.texts == []


def test_an_item_whose_crop_cannot_serve_as_a_reference_is_refused_before_the_tap():
    """Doc 5.4's animated-card class. The measured case is a card whose own frame-to-frame drift
    (25.009 grey levels) exceeds half its distance to its nearest neighbour (47.861), so the
    reference is compromised and no tolerance can separate "correct but noisy" from "wrong". The
    run stops with the screen untouched rather than tapping and finding out."""
    payload = _payload()
    drifted = _payload(drift=payload.item(1).nearest_item_distance)
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=drifted, anchor=b"fixture anchor")
        with pytest.raises(HingeActionError) as exc:
            driver.like("an opener", model_item_index=1)
    assert adb.calls == [] and adb.taps == [] and adb.texts == []
    assert "cannot be verified on the like sheet" in str(exc.value)
    assert "NOT sent" in str(exc.value)


# =====================================================================================
# The legacy path, and the boundary between them
# =====================================================================================

def test_capture_order_opener_is_retired_before_any_gesture():
    """A frame index cannot license text: it bypasses the calibrated model-item verifier."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload())
        with pytest.raises(HingeTargetingError, match="Capture-order targeting is retired"):
            driver.like("an opener", 3)
    assert adb.calls == [] and adb.taps == [] and adb.texts == []


def test_capture_order_opener_does_not_enter_legacy_target_lookup():
    """The retired path is refused before its old scroll/search machinery can run."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload())
        mp.setattr(HingeDriver, "_locate_target_heart",
                   lambda *_a, **_k: pytest.fail("retired lookup was called"))
        with pytest.raises(HingeTargetingError, match="model item index"):
            driver.like("an opener", 3)
    assert adb.texts == []
    assert adb.taps == []


def test_the_anchored_reask_parameter_is_gone_from_the_driver_entirely():
    """The repair hatch is REMOVED, not merely bypassed once a model item number is present. A
    caller still passing one has to fail loudly at the call, rather than have its callback
    silently ignored -- which would be indistinguishable from a repair that was never needed."""
    adb = SheetAdb(_SHEETS[1])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload())
        with pytest.raises(TypeError):
            driver.like("an opener", 0, anchored_opener=lambda a: "repaired",
                        model_item_index=1)
    assert adb.texts == []


def test_the_item_table_is_still_dropped_after_a_verification_stop():
    """`like()`'s finally invalidates the table whatever happened, and a stop is exactly when a
    stale one would survive longest (doc 5.3)."""
    adb = SheetAdb(_SHEETS[1])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload(), anchor=b"fixture anchor")
        with pytest.raises(HingeActionError):
            driver.like("an opener", model_item_index=1)
    assert driver._current_item_payload is None
    assert driver._current_items_unavailable


def test_verification_cannot_be_skipped_by_omitting_the_model_item_number():
    """Any text without a model item number is refused before the legacy path can run."""
    adb = SheetAdb(_SHEETS[1])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload(), anchor=b"fixture anchor")
        with pytest.raises(HingeActionError):
            driver.like("an opener", model_item_index=1)
    # Omitting the number now refuses rather than creating an unverifiable legacy send.
    adb2 = SheetAdb(_SHEETS[1])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb2, mp, payload=_payload())
        with pytest.raises(HingeTargetingError, match="model item index"):
            driver.like("an opener", 0)
    assert adb2.calls == [] and adb2.texts == []


def test_the_verifier_and_the_driver_agree_on_which_item_the_painted_sheet_shows():
    """A fixture guard: if the painter and the verifier ever disagree the driver tests above would
    pass or fail for reasons that have nothing to do with the driver."""
    payload = _payload()
    assert item_verify.verify_sheet_item(_SHEETS[0], payload, 1).matched
    assert item_verify.verify_sheet_item(_SHEETS[1], payload, 2).matched
    assert not item_verify.verify_sheet_item(_SHEETS[0], payload, 2).matched


# =====================================================================================
# DOC 5.9's OBSERVE-SIDE MISMATCH GUARD
#
# `observe_item_mismatch` is the same pair of comparisons the like path above makes -- whose
# profile, then which item -- run on a sheet a HUMAN opened. What differs is only what a refusal
# costs: AUTO stops the run, OBSERVE refuses to show text. So these tests are the mirror of the
# ones above, asserting a SENTENCE where those assert an exception.
# =====================================================================================

def _observing(adb, monkeypatch, **kwargs):
    """The driver as observe holds it: crops for the profile on screen, no like in flight."""
    return _driver(adb, monkeypatch, payload=_payload(), **kwargs)


def test_the_item_the_human_opened_matching_the_suggestion_is_confirmed_with_no_reason():
    """"" is the ONLY value that lets the hub show text, so it is the only value a match may
    produce -- an empty-ish sentence would read as a warning and cost a real suggestion."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _observing(adb, mp)
        assert driver.observe_item_check(_SHEETS[0], 1).state == OBSERVE_ITEM_MATCH
        assert driver.observe_item_mismatch(_SHEETS[0], 1) == ""


def test_observe_check_allows_a_structurally_confirmed_priority_like_surface(monkeypatch):
    """Priority Like changes only the CTA label, never the passive item-check contract.

    The lower template threshold is deliberately reached only after strict
    action-level detection fails.  The returned surface still has to pass the
    same profile and selected-item comparisons before advice is shown.
    """
    adb = SheetAdb(_SHEETS[0])
    calls = []
    with pytest.MonkeyPatch.context() as mp:
        driver = _observing(adb, mp)

        # `image=` accepted and ignored: `_locate_observed_inline_composer` now decodes the frame
        # once and passes it to both the strict and fallback calls below (2026-08-23 perf pass;
        # see that method's docstring) -- this test is about the threshold retry sequence, not
        # about the shared decode.
        def priority_surface(_frame, _template, *, threshold, **_kw):
            calls.append(threshold)
            if threshold == 0.8:
                raise hinge.ComposerDetectionError("literal Send Like glyph is a Priority Like")
            assert threshold == 0.68
            return _COMPOSER_SURFACE

        mp.setattr(hinge, "locate_inline_composer", priority_surface)
        assert driver.observe_item_check(_SHEETS[0], 1).state == OBSERVE_ITEM_MATCH
        assert driver._observe_like_sheet_detection == "priority_variant"
    assert calls == [0.8, 0.68]
    assert adb.calls == []


def test_a_human_opening_a_different_item_is_reported_with_both_numbers():
    """DOC 5.9's headline case. The suggestion was written about item 1 and the human hearted
    item 2. The sentence has to name BOTH -- what they opened and what the text was for --
    because "these do not match" tells an operator nothing about which of the two to trust."""
    adb = SheetAdb(_SHEETS[1])
    with pytest.MonkeyPatch.context() as mp:
        driver = _observing(adb, mp)
        assert driver.observe_item_check(_SHEETS[1], 1).state == OBSERVE_ITEM_MISMATCH
        reason = driver.observe_item_mismatch(_SHEETS[1], 1)
    assert "item 2" in reason and "item 1" in reason


def test_observe_does_not_name_a_reachable_neighbour_when_the_intended_item_was_unmeasurable():
    """A nearest candidate is not proof of the item a human opened.

    This is the reporting half of Alex's portrait reframe regression: item 3 was beyond the
    previous reframe envelope, so item 6 was merely the nearest comparison that remained.  The
    hub must ask for a fresh check, never assert that item 6 was selected.
    """
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _observing(adb, mp)
        baseline = item_verify.verify_sheet_item(
            _SHEETS[0], driver._current_item_payload, 1, composer_surface=_COMPOSER_SURFACE,
            absolute_max_dist=10.0)
        unavailable = dataclasses.replace(
            baseline, state=item_verify.VERIFY_MISMATCH, nearest_index=2, distance=None,
            comparisons=tuple(
                dataclasses.replace(c, distance=None,
                                  reason="inline composer preview is still settling")
                if c.number == 1 else c
                for c in baseline.comparisons))
        mp.setattr(hinge, "verify_sheet_item", lambda *_a, **_k: unavailable)
        assert driver.observe_item_check(_SHEETS[0], 1).state == OBSERVE_ITEM_INCONCLUSIVE
        reason = driver.observe_item_mismatch(_SHEETS[0], 1)
    assert "could not yet be confirmed as model item 1" in reason
    assert "settling" in reason
    assert "item 2" not in reason


def test_a_sheet_on_a_different_profile_is_refused_on_IDENTITY_not_on_card_pixels():
    """THE DECK-ADVANCE RACE, answered the way doc 5.9 asks. A human who passes and then hearts
    the NEXT card leaves the worker holding a suggestion about somebody else, and the comment
    sheet does not occlude the sticky header (identity_band cuts rows 115..226; the sheet's
    preview starts at 236), so the answer is on the frame already in hand.

    It has to be identity rather than the card comparison because the card comparison CANNOT
    answer it: `verify_sheet_item` is a closed-set test over one payload with no absolute accept
    ceiling, and was measured accepting a foreign card 10 times in 540. Here the foreign sheet is
    the payload's OWN item 1, i.e. the case that comparison is guaranteed to call a match --
    which is exactly why this must refuse anyway."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _observing(adb, mp, foreign_identity=True)
        reason = driver.observe_item_mismatch(_SHEETS[0], 1)
    assert reason and "profile" in reason
    # The control: the identical call with the identity agreeing is a confirmation, so the
    # refusal above is the gate doing its job rather than a blanket no.
    with pytest.MonkeyPatch.context() as mp:
        assert _observing(SheetAdb(_SHEETS[0]), mp).observe_item_mismatch(_SHEETS[0], 1) == ""


def test_no_crops_for_the_profile_on_screen_is_a_reason_and_never_a_confirmation():
    """The state an observe run reaches whenever enumeration refused. There is nothing to check
    against, and doc 5.9's rule is that "could not look" must never render as "looked and it is
    fine" -- the operator gets the driver's own recorded sentence."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=None,
                         unavailable="the scroll top was never confirmed")
        reason = driver.observe_item_mismatch(_SHEETS[0], 1)
    assert "the scroll top was never confirmed" in reason


def test_observe_with_no_targeting_calibration_returns_a_warning_not_a_confirmation():
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload(), targeting_calibration=False)
        reason = driver.observe_item_mismatch(_SHEETS[0], 1)
    assert reason and "targeting_calibration" in reason and adb.calls == []


def test_an_unreadable_sheet_is_a_reason_rather_than_an_exception():
    """Observe must not be able to crash on a bad frame: this runs on a labelling session whose
    whole purpose is the human's decisions, and the suggestion is cosmetic beside them. Every
    "could not look" -- undecodable bytes, no preview on the frame, missing vision extras --
    comes back as a sentence, on the same terms as a genuine mismatch."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _observing(adb, mp)
        check = driver.observe_item_check(b"not an image at all", 1)
        assert check.state == OBSERVE_ITEM_INCONCLUSIVE
        reason = driver.observe_item_mismatch(b"not an image at all", 1)
    assert reason


def test_an_item_number_outside_the_payload_is_a_reason_rather_than_an_exception():
    """A model item number the payload has no crop for cannot be checked and must not be shown.
    `verification_blocker` raises `SheetVerificationError` for it (doc 5.6: an out-of-range
    number is not a verdict), which on the AUTO path is a stop; here it is a sentence."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _observing(adb, mp)
        assert driver.observe_item_mismatch(_SHEETS[0], 9)


def test_the_guard_never_touches_the_transport_or_the_debug_log():
    """Doc 5.9 runs this from the worker's suggestion THREAD (when the model's answer lands while
    the sheet is already open) while the worker thread is inside wait_for_decision screencapping
    and writing its own debug records. Pure vision over a frame already in hand is the only shape
    that is safe there: a second screencap would race the transport, and DebugLog appends to one
    file from one thread by design."""
    adb = SheetAdb(_SHEETS[0])
    written = []
    with pytest.MonkeyPatch.context() as mp:
        driver = _observing(adb, mp)
        driver._dbg = type("Dbg", (), {"action": lambda self, *a, **k: written.append(a)})()
        mp.setattr(HingeDriver, "_screencap",
                   lambda self, **_k: pytest.fail("observe_item_mismatch took a screencap"))
        assert driver.observe_item_mismatch(_SHEETS[0], 1) == ""
    assert adb.calls == [] and written == []
