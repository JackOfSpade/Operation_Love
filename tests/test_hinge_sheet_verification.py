"""The driver half of doc 5.6: verify the open comment sheet, THEN type. Never the other order.

`tests/test_item_verify.py` covers the comparison itself. What this file pins is the thing that
makes it worth having — that there is no path through `hinge._like_comment_sheet` from a tapped
heart to `adb.text(...)` that does not pass a match first, and that every refusal leaves the run
stopped with the screen where it was rather than sending something.

Every frame is SYNTHESISED. The captures the constants were measured against are real people's
profiles and are gitignored, so only geometry and grey levels from them appear here: two painted
"cards" and a painted comment sheet at the sheet geometry item_verify's docstring records.
"""
import cv2
import numpy as np
import pytest

from operation_love.drivers import (
    hinge, item_crops, item_identity, item_nav, item_verify, scroll_step)
from operation_love.drivers.frameshift import ShiftEstimationError
from operation_love.drivers.hinge import HingeActionError, HingeDriver, HingeTargetingError
from operation_love.drivers.scroll_top import band_fingerprint
from operation_love.drivers.segment import SegmentationError

_W, _H = 1080, 2400
_CARD_W, _CARD_H = 974, 900
_SHEET_BG, _PREVIEW_X0, _PREVIEW_W, _PREVIEW_Y0 = 250, 95, 890, 236


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
    """A comment sheet rendering `card` at the measured preview geometry."""
    image = cv2.imdecode(np.frombuffer(card, np.uint8), cv2.IMREAD_COLOR)
    height = int(round(image.shape[0] * _PREVIEW_W / image.shape[1]))
    canvas = np.full((_H, _W, 3), _SHEET_BG, dtype=np.uint8)
    canvas[_PREVIEW_Y0:_PREVIEW_Y0 + height, _PREVIEW_X0:_PREVIEW_X0 + _PREVIEW_W] = cv2.resize(
        image, (_PREVIEW_W, height), interpolation=cv2.INTER_AREA)
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


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)


class _FakeIndex:
    """Just enough of an `ItemIndex` for the identity gate: the fingerprint of the profile these
    crops were cut from. The real thing is built by `build_item_index` and has its own tests."""

    def __init__(self, identity):
        self.identity = identity


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
        recorder.append((driver, index, model_index, entry_reference))
        return _Target()
    return navigate


def _driver(adb, monkeypatch, *, payload=None, unavailable="", identity_frame=None,
            foreign_identity=False, anchor=None):
    class C:
        apps = {"hinge": {"serial": "pixel", "halt_on_error": False}}

    driver = HingeDriver(C())
    driver._adb = adb
    driver._touch = adb
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
                        lambda self: adb.calls.append("scroll_to_top"))
    # Returns a bare point since 2026-08-12: _locate_target_heart either lands on the item it was
    # asked for or raises, so there is no second "and this is actually the wrong one" flag left.
    monkeypatch.setattr(HingeDriver, "_locate_target_heart",
                        lambda self, index: (540, 1200))
    monkeypatch.setattr(HingeDriver, "_await_sheet_open",
                        lambda self, tries=5: adb.calls.append("await_sheet"))
    monkeypatch.setattr(HingeDriver, "_handle_rose_upsell", lambda self, tries=2: False)
    return driver


# =====================================================================================
# THE HEADLINE: verify, then type
# =====================================================================================

def test_a_matching_sheet_is_verified_before_the_opener_is_typed():
    """Order is the guarantee. The tap opens the sheet, the sheet is checked, and only then does
    a character reach the phone."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload())
        driver.like("two sentences, no dashes", 0, model_item_index=1)
    assert adb.texts == ["two sentences, no dashes"]
    assert adb.calls.index("await_sheet") < adb.calls.index("text")


def test_a_sheet_showing_a_different_item_stops_the_run_with_nothing_typed():
    """The failure this whole redesign exists to stop: the opener was written about item 1 and
    the sheet that opened is item 2. Doc 5.6 -- do not type, do not send, never a commentless
    like, record intended and actual, leave the screen for debugging."""
    adb = SheetAdb(_SHEETS[1])                       # the sheet shows item 2...
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload())
        with pytest.raises(HingeActionError) as exc:
            driver.like("an opener about item one", 0, model_item_index=1)   # ...we asked for 1
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
        driver = _driver(adb, mp, payload=_payload())
        with pytest.raises(HingeActionError):
            driver.like("an opener", 0, model_item_index=1)
    assert adb.texts == []


def test_a_sheet_that_cannot_be_read_at_all_stops_the_run_rather_than_typing():
    """"Could not look" and "looked and it is wrong" call for the same action one tap away from
    typing; only the diagnosis differs."""
    flat = np.full((_H, _W), _SHEET_BG, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", flat)
    assert ok
    adb = SheetAdb(buf.tobytes())
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload())
        with pytest.raises(HingeActionError) as exc:
            driver.like("an opener", 0, model_item_index=1)
    assert adb.texts == []
    assert "could not be checked" in str(exc.value)


# =====================================================================================
# Refused BEFORE the tap
# =====================================================================================

def test_a_model_item_with_no_crops_behind_it_is_refused_without_touching_the_screen():
    """Doc 5.3: treat a missing table as a hard stop, never as a reason to fall back to a fixed
    coordinate. The invalidation reason is quoted, because "the deck advanced" and "this capture
    could not be indexed" call for different next moves."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=None,
                         unavailable="the deck advanced after this like")
        with pytest.raises(HingeActionError) as exc:
            driver.like("an opener", 0, model_item_index=1)
    assert adb.calls == [] and adb.taps == [] and adb.texts == []
    assert "the deck advanced after this like" in str(exc.value)


def test_an_item_number_outside_the_list_is_refused_without_touching_the_screen():
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload())
        with pytest.raises(HingeActionError) as exc:
            driver.like("an opener", 0, model_item_index=7)
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
        driver = _driver(adb, mp, payload=_payload(), foreign_identity=True)
        with pytest.raises(HingeTargetingError) as exc:
            driver.like("an opener about item one", 0, model_item_index=1)
    assert adb.calls == [] and adb.taps == [] and adb.texts == []
    message = str(exc.value)
    assert "not the one model item 1's crops were cut from" in message
    assert "NOT sent" in message
    # ...and the very same call passes the gate when the crops belong to the profile on screen,
    # which is what makes it a gate rather than a blanket refusal.
    adb2 = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb2, mp, payload=_payload())
        driver.like("an opener about item one", 0, model_item_index=1)
    assert adb2.texts == ["an opener about item one"]


def test_a_screen_whose_identity_cannot_be_read_stops_rather_than_passing():
    """"Cannot tell" is never "yes". At a scroll top the identity strip is Hinge's own
    filter-chips row -- byte-identical for two different people -- so there is nothing there to
    tell anyone apart, and a gate that read silence as agreement would pass for everybody."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload())
        driver._current_item_index = _FakeIndex(item_identity.ProfileIdentity(
            fingerprint=None, band=tuple(hinge.HINGE_SPEC.identity_band),
            grid=item_identity._IDENTITY_GRID, frame_index=None, scroll_top_distance=None,
            reason="this capture never showed a sticky header"))
        with pytest.raises(HingeTargetingError) as exc:
            driver.like("an opener", 0, model_item_index=1)
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
                   lambda self, index: (located.append(index), (540, 1200))[1])
        mp.setattr(hinge, "navigate_to_item", _fake_navigate(navigated, point=(931, 1477)))
        index = driver._current_item_index        # `like()` clears it in a finally, by design
        driver.like("an opener about item two", model_item_index=2)

    assert located == [], "the capture-order heart search must not be entered"
    assert "scroll_to_top" not in adb.calls, "bottom-up navigation must not rewind"
    assert adb.taps[0] == (931, 1477), "the tap is the point navigation returned"
    assert adb.texts == ["an opener about item two"]
    # ...and navigation got the driver, the index, the model's number and the anchor frame.
    (driver_arg, index_arg, model_index_arg, reference_arg), = navigated
    assert driver_arg is driver and index_arg is index
    assert (model_index_arg, reference_arg) == (2, b"the read's last frame")


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

            def _raise(*_a, **_kw):
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


def test_a_capture_order_target_beside_the_model_item_number_is_still_accepted():
    """The guard is about having nothing to navigate BY, not about `model_item_index` itself.
    `item_index=0` is a real capture-order target -- the topmost heart at the scroll top -- so
    the tap is aimed rather than defaulted, and doc 5.6's post-tap check then confirms which item
    it landed on. Whoever wires `item_nav.navigate_to_item` replaces the refusal with the
    navigation call; until then this is the only shape that runs."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload())
        driver.like("an opener about item one", 0, model_item_index=1)
    assert adb.texts == ["an opener about item one"]


def test_an_item_whose_crop_cannot_serve_as_a_reference_is_refused_before_the_tap():
    """Doc 5.4's animated-card class. The measured case is a card whose own frame-to-frame drift
    (25.009 grey levels) exceeds half its distance to its nearest neighbour (47.861), so the
    reference is compromised and no tolerance can separate "correct but noisy" from "wrong". The
    run stops with the screen untouched rather than tapping and finding out."""
    payload = _payload()
    drifted = _payload(drift=payload.item(1).nearest_item_distance)
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=drifted)
        with pytest.raises(HingeActionError) as exc:
            driver.like("an opener", 0, model_item_index=1)
    assert adb.calls == [] and adb.taps == [] and adb.texts == []
    assert "cannot be verified on the like sheet" in str(exc.value)
    assert "NOT sent" in str(exc.value)


# =====================================================================================
# The legacy path, and the boundary between them
# =====================================================================================

def test_without_a_model_item_number_the_old_capture_order_path_still_types_its_opener():
    """Nothing hands this driver a model item number until doc 5.6's counting navigation is
    wired, so the capture-order path has to keep working: targeting reached the item it was asked
    for, so the ORIGINAL opener is typed and no crop check is performed (there is no model item
    number to check against).

    This test used to assert something else entirely -- that a MISS here called an
    `anchored_opener` re-ask and typed its replacement. That callback was removed on 2026-08-12:
    it repaired the TEXT while the LIKE still landed on an item the model never chose, which is
    the substitution the owner rule forbids. A miss on this path is now a stop, pinned by the
    test below."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload())
        driver.like("an opener", 3)
    assert adb.texts == ["an opener"]


def test_a_targeting_miss_on_the_capture_order_path_stops_instead_of_repairing_the_text():
    """Doc 5.6: "no falling back to `hearts[0]`, no 'closest reachable item', no rewriting the
    opener to match whatever we hit". The legacy path gets the same treatment as the verified one
    -- a heart the driver cannot reach is a stop, with nothing typed and nothing sent, and there
    is no second billed call to re-word the message around the miss."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload())
        mp.setattr(HingeDriver, "_locate_target_heart",
                   lambda self, index: (_ for _ in ()).throw(
                       HingeTargetingError("could not reach it", stage="navigate", intended=3)))
        with pytest.raises(HingeTargetingError):
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
        driver = _driver(adb, mp, payload=_payload())
        with pytest.raises(HingeActionError):
            driver.like("an opener", 0, model_item_index=1)
    assert driver._current_item_payload is None
    assert driver._current_items_unavailable


def test_verification_is_not_reachable_by_forgetting_to_pass_the_payload():
    """The gate cannot be skipped by omission: the only way to reach the typing with a model item
    number is through a MATCH, and the only way to reach it without one is the legacy path, which
    has no crops to check against in the first place. There is no third state where an item number
    was given and the check quietly did not happen."""
    adb = SheetAdb(_SHEETS[1])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb, mp, payload=_payload())
        with pytest.raises(HingeActionError):
            driver.like("an opener", 0, model_item_index=1)
    # ...and the same call with no number typed happily, which is what makes the above a gate
    # rather than an unconditional refusal.
    adb2 = SheetAdb(_SHEETS[1])
    with pytest.MonkeyPatch.context() as mp:
        driver = _driver(adb2, mp, payload=_payload())
        driver.like("an opener", 0)
    assert adb2.texts == ["an opener"]


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
        assert driver.observe_item_mismatch(_SHEETS[0], 1) == ""


def test_a_human_opening_a_different_item_is_reported_with_both_numbers():
    """DOC 5.9's headline case. The suggestion was written about item 1 and the human hearted
    item 2. The sentence has to name BOTH -- what they opened and what the text was for --
    because "these do not match" tells an operator nothing about which of the two to trust."""
    adb = SheetAdb(_SHEETS[1])
    with pytest.MonkeyPatch.context() as mp:
        driver = _observing(adb, mp)
        reason = driver.observe_item_mismatch(_SHEETS[1], 1)
    assert "item 2" in reason and "item 1" in reason


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


def test_an_unreadable_sheet_is_a_reason_rather_than_an_exception():
    """Observe must not be able to crash on a bad frame: this runs on a labelling session whose
    whole purpose is the human's decisions, and the suggestion is cosmetic beside them. Every
    "could not look" -- undecodable bytes, no preview on the frame, missing vision extras --
    comes back as a sentence, on the same terms as a genuine mismatch."""
    adb = SheetAdb(_SHEETS[0])
    with pytest.MonkeyPatch.context() as mp:
        driver = _observing(adb, mp)
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
