"""operation_love/drivers/web/ -- the PlaywrightDriver base / BumbleWebDriver
subclass split, and the platform-availability guard.

Offline only: no browser, no network. Bumble web has no live target
(operation_love/platforms.py registers "bumble_web" with available=False), so
these tests double as the regression guard that a driver built directly (not
through the supervisor) refuses to actually start, while still allowing
construction/introspection -- e.g. tools/bumble_inspect.py's --help path and this
file's own selector-wiring tests -- to keep working.
"""
from operation_love import platforms
from operation_love.drivers import make_driver
from operation_love.drivers.base import DatingAppDriver
from operation_love.drivers.web.bumble_web import DEFAULT_SELECTORS, BumbleWebDriver
from operation_love.drivers.web.playwright_base import PlatformUnavailable, PlaywrightDriver


# `apps.bumble` is the ANDROID block now; the web driver's settings live under its own
# registry id, `apps.bumble_web`.
class _Cfg:
    apps = {"bumble_web": {}}


class _CfgWithSelectorOverride:
    apps = {"bumble_web": {"selectors": {"like": "custom-like-selector"}}}


# --- base/subclass wiring ---------------------------------------------------

def test_bumble_web_driver_is_a_playwright_driver_is_a_dating_app_driver():
    drv = BumbleWebDriver(_Cfg())
    assert isinstance(drv, PlaywrightDriver)
    assert isinstance(drv, DatingAppDriver)


def test_bumble_web_driver_declares_its_registry_id_and_hud_label():
    # These two class attrs are how the generic base (platform_app for the
    # availability guard, hud_label for the HUD/log text) gets Bumble-flavored
    # without PlaywrightDriver itself knowing anything about Bumble.
    assert BumbleWebDriver.platform_app == "bumble_web"
    assert BumbleWebDriver.hud_label == "bumble"
    assert PlaywrightDriver.platform_app is None       # base itself declares no platform
    assert PlaywrightDriver.hud_label == "web"


def test_selectors_default_to_bumble_defaults():
    drv = BumbleWebDriver(_Cfg())
    assert drv.selectors == DEFAULT_SELECTORS
    assert drv.selectors is not DEFAULT_SELECTORS      # a copy, not the shared dict


def test_selectors_merge_config_override_with_defaults():
    drv = BumbleWebDriver(_CfgWithSelectorOverride())
    assert drv.selectors["like"] == "custom-like-selector"     # overridden
    assert drv.selectors["pass"] == DEFAULT_SELECTORS["pass"]  # untouched default kept


def test_make_driver_wires_bumble_web_to_the_subclass():
    drv = make_driver("bumble_web", _Cfg())
    assert isinstance(drv, BumbleWebDriver)


def test_accepts_opener_false_carried_over():
    # Bumble matches first, then messages -- no swipe-time opener. This lived on
    # the old monolithic BumbleDriver; confirm the split didn't drop it.
    assert BumbleWebDriver.accepts_opener is False


# --- platform-availability guard --------------------------------------------

def test_registry_marks_bumble_web_unavailable_with_a_reason():
    # Sanity check on the fixture this whole file leans on: if the registry ever
    # marks bumble_web available again, the guard tests below would trivially
    # "pass" by doing nothing, so pin the precondition explicitly.
    reason = platforms.unavailable_reason("bumble_web")
    assert reason and "web" in reason.lower()


def test_constructing_unavailable_driver_does_not_raise():
    # Only open_session() -- actually starting a browser -- refuses to run.
    # Construction must keep working: it's what lets a driver be introspected
    # (.selectors, its class hierarchy) without ever touching Playwright, which is
    # exactly what every test in this file and test_bumble_observe.py relies on.
    BumbleWebDriver(_Cfg())


def test_open_session_refuses_when_platform_unavailable():
    drv = BumbleWebDriver(_Cfg())

    def _must_not_be_called():
        raise AssertionError(
            "open_session() must refuse before ever calling _import_playwright() "
            "-- an unavailable platform must not launch so much as the driver "
            "subprocess, let alone a real browser"
        )

    drv._import_playwright = _must_not_be_called   # non-data descriptor -> instance wins
    try:
        drv.open_session()
    except PlatformUnavailable as exc:
        assert str(exc) == platforms.unavailable_reason("bumble_web")
    else:
        raise AssertionError("expected PlatformUnavailable")


def test_open_session_guard_carries_the_registry_reason_verbatim():
    drv = BumbleWebDriver(_Cfg())
    drv._import_playwright = lambda: (_ for _ in ()).throw(AssertionError("unreachable"))
    try:
        drv.open_session()
        raise AssertionError("expected PlatformUnavailable")
    except PlatformUnavailable as exc:
        assert "Bumble discontinued its web app" in str(exc)


def test_base_driver_with_no_platform_app_set_has_no_guard():
    # A hypothetical future subclass that hasn't set platform_app yet (or a base
    # driver used directly in a test) must not be blocked by a guard that has
    # nothing to check -- platform_app=None means "no registry entry to check",
    # not "always unavailable".
    class _Bare(PlaywrightDriver):
        def next_profile(self):
            return None

        def like(self, opener=None, item_index=0):
            pass

        def dislike(self):
            pass

        def out_of_profiles(self):
            return True

    drv = _Bare(
        _Cfg(),
        config_key="bumble",
        default_url="https://example.invalid",
        default_user_data_dir="./data/_bare_profile",
        default_debug_dir="./data/_bare_debug",
    )
    assert drv.platform_app is None
    drv._check_platform_unavailable()   # must not raise
