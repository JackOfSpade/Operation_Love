"""Platform registry safety and selection behaviour."""
import pytest

from operation_love import platforms


@pytest.fixture(autouse=True)
def _restore_registry():
    saved = platforms._PLATFORMS
    saved_modes = dict(platforms._AVAILABLE_MODES)
    yield
    platforms._PLATFORMS = saved
    platforms._BY_APP = {p.app: p for p in saved}
    platforms._AVAILABLE_MODES = saved_modes


def test_only_android_targets_are_registered():
    assert platforms.kinds() == (platforms.KIND_ANDROID,)
    assert {p.app for p in platforms.all_platforms()} == {"hinge", "bumble"}
    assert platforms.get("hinge").kind == platforms.KIND_ANDROID
    assert platforms.get("bumble").kind == platforms.KIND_ANDROID


def test_both_targets_exclusively_hold_the_phone():
    assert platforms.get("hinge").exclusive_resource == platforms.RESOURCE_ANDROID_DEVICE
    assert platforms.get("bumble").exclusive_resource == platforms.RESOURCE_ANDROID_DEVICE
    assert "one app in the foreground" in platforms.check_selection(["hinge", "bumble"])


def test_unknown_platform_is_rejected():
    assert "Unknown app 'tinder'" in platforms.check_selection(["tinder"])
    with pytest.raises(ValueError, match="Unknown app 'tinder'"):
        platforms.get("tinder")


def test_bumble_supports_auto_only():
    platforms._apply_calibration({"bumble": {"observe": False, "auto": True}})
    assert "Auto only" in platforms.unavailable_reason("bumble", "observe")
    assert platforms.unavailable_reason("bumble", "auto") is None


def test_calibration_changes_availability_without_changing_registered_apps():
    before = set(platforms.KNOWN_APPS)
    platforms._apply_calibration({"bumble": True})
    assert platforms.unavailable_reason("bumble") is None
    assert set(platforms.KNOWN_APPS) == before


def test_config_may_name_bumble_and_start_its_auto_mode():
    platforms._apply_calibration({"bumble": {"observe": False, "auto": True}})
    assert platforms.check_selection(["bumble"]) is None
    assert platforms.check_runnable(["bumble"], modes="auto") is None
    assert "Auto only" in platforms.check_runnable(["bumble"], modes="observe")
