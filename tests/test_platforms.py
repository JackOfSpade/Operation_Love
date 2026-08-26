"""Platform registry safety and selection behaviour."""
import importlib
import threading

import pytest

from operation_love import platforms
from operation_love import targeting_policy as tp

# Liveness bound, not a performance bound: it exists only so a genuine hang fails this test
# instead of hanging the whole suite forever. Widened 2026-08-22 when `python -m pytest` moved
# to one worker per core (pyproject.toml addopts `-n auto --dist loadgroup`), which measured a
# ~15x slowdown (0.33s idle vs 5.06s under load) on tests/test_concurrency.py -- the same kind
# of positive liveness wait used here. Nothing about the property under test (did the reader
# thread reach the expected state?) depends on the exact number, so widening it loses nothing.
_LIVENESS_TIMEOUT_S = 15.0


@pytest.fixture(autouse=True)
def _restore_registry():
    saved = platforms._PLATFORMS
    saved_modes = dict(platforms._AVAILABLE_MODES)
    saved_loaded = platforms._calibration_loaded
    saved_loading_thread = platforms._calibration_loading_thread
    yield
    platforms._PLATFORMS = saved
    platforms._BY_APP = {p.app: p for p in saved}
    platforms._AVAILABLE_MODES = saved_modes
    platforms._calibration_loaded = saved_loaded
    platforms._calibration_loading_thread = saved_loading_thread


@pytest.fixture(autouse=True)
def _still_photo_readiness_is_never_inherited():
    """Numbering readiness is process-global: never let one test license the next one."""
    tp._reset_installed_still_photo_bound_for_tests()
    yield
    tp._reset_installed_still_photo_bound_for_tests()


@pytest.fixture
def installed_bound():
    """A verified still-photo bound, installed exactly as config.validate() would install it."""
    tp.install_verified_still_photo_bound(tp.StillPhotoBoundSummary(
        ground_truth_channel=tp.STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL,
        human_ground_truth=True, video_cards=60, video_accepts=0, photo_cards=60,
        photo_false_refusals=3, max_video_exact_run_s=1.5, artifact_sha256="a" * 64,
        device="synthetic-pixel", hinge_version_name="10.0.1"))


@pytest.fixture
def installed_circular_bound():
    """The same, on the opted-in circular AI-labelled channel (owner decision 2026-08-21)."""
    tp.install_verified_still_photo_bound(tp.StillPhotoBoundSummary(
        ground_truth_channel=tp.STILL_PHOTO_BOUND_CIRCULAR_CHANNEL,
        human_ground_truth=False, video_cards=60, video_accepts=0, photo_cards=60,
        photo_false_refusals=3, max_video_exact_run_s=1.5, artifact_sha256="a" * 64,
        device="synthetic-pixel", hinge_version_name="10.0.1",
        accepted_circular_risk=tp.STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE))


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


def test_bumble_fails_closed_in_both_modes():
    assert platforms.get("bumble").available is False
    for mode in ("training", "auto"):
        reason = platforms.unavailable_reason("bumble", mode)
        assert reason and "not calibrated" in reason


def test_hinge_training_stays_targeting_gated_like_auto():
    assert platforms.mode_available("hinge", "training") is False
    training_reason = platforms.unavailable_reason("hinge", "training")
    assert training_reason and "Hinge Training is blocked" in training_reason
    reason = platforms.unavailable_reason("hinge", "auto")
    assert reason and "Hinge Auto is blocked" in reason
    assert "positive still-photo discriminator unavailable" in reason


def test_a_verified_still_photo_bound_licenses_hinge_driver_readiness(installed_bound):
    """The registry owns mechanical/policy readiness; config owns release-artifact validation."""
    assert tp.hinge_targeting_unavailable_reason() is None
    assert platforms.mode_available("hinge", "training") is True

    assert platforms.unavailable_reason("hinge", "auto") is None
    assert platforms.mode_available("hinge", "auto") is True


def test_an_accepted_circular_bound_reaches_the_same_driver_readiness(installed_circular_bound):
    """The second channel changes the evidence provenance, not driver geometry."""
    assert tp.hinge_targeting_unavailable_reason() is None
    assert platforms.mode_available("hinge", "training") is True

    assert platforms.unavailable_reason("hinge", "auto") is None
    assert platforms.mode_available("hinge", "auto") is True


def test_the_android_registry_registers_hinge_auto_from_calibrated_mechanics(installed_bound):
    """The driver package registers mechanical readiness, not config release evidence.

    Re-executing the package body is the only way to observe what it registers: the module is
    already imported by the time any test runs, so a lazy _ensure_calibration() would find it
    in sys.modules and never re-apply. The registry fixture restores the tables afterwards.
    """
    importlib.reload(importlib.import_module("operation_love.drivers.android"))

    assert platforms._AVAILABLE_MODES["hinge"] == frozenset(
        {"training", "auto"})
    assert platforms.unavailable_reason("hinge", "auto") is None
    assert platforms.unavailable_reason("hinge", "training") is None


def test_mode_unavailable_reason_is_directional_for_training_only_platform():
    platforms._apply_calibration({"bumble": {"training": True, "auto": False}})

    assert platforms.unavailable_reason("bumble", "training") is None
    assert platforms.unavailable_reason("bumble", "auto") == (
        "Bumble supports Training only; Auto is not available.")


def test_calibration_changes_availability_without_changing_registered_apps():
    before = set(platforms.KNOWN_APPS)
    platforms._apply_calibration({"bumble": True})
    assert platforms.unavailable_reason("bumble") is None
    assert set(platforms.KNOWN_APPS) == before


def test_config_may_name_bumble_but_cannot_start_until_calibrated():
    assert platforms.check_selection(["bumble"]) is None
    for mode in ("training", "auto"):
        reason = platforms.check_runnable(["bumble"], modes=mode)
        assert reason and "not calibrated" in reason


@pytest.mark.parametrize("state", ["false", 1, [], object()])
def test_historic_scalar_calibration_accepts_exact_booleans_only(state):
    with pytest.raises(ValueError, match="exact boolean"):
        platforms._apply_calibration({"hinge": state})


@pytest.mark.parametrize("value", [1, 0, "true", None, []])
def test_mode_calibration_values_accept_exact_booleans_only(value):
    with pytest.raises(ValueError, match="exact booleans"):
        platforms._apply_calibration({"hinge": {"training": value, "auto": False}})


def test_lazy_calibration_serializes_concurrent_readers(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    second_done = threading.Event()
    calls = []

    def load():
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(_LIVENESS_TIMEOUT_S)
        platforms._apply_calibration({
            "hinge": {"training": True, "auto": False},
            "bumble": {"training": False, "auto": False},
        })

    monkeypatch.setattr(platforms, "_load_android_calibration", load)
    platforms._calibration_loaded = False
    platforms._calibration_loading_thread = None

    first = threading.Thread(target=platforms._ensure_calibration)

    def second_reader():
        second_started.set()
        platforms._ensure_calibration()
        second_done.set()

    second = threading.Thread(target=second_reader)
    first.start()
    assert entered.wait(_LIVENESS_TIMEOUT_S)
    second.start()
    assert second_started.wait(_LIVENESS_TIMEOUT_S)
    assert not second_done.wait(0.05), "second reader bypassed in-progress registration"
    release.set()
    first.join(_LIVENESS_TIMEOUT_S)
    second.join(_LIVENESS_TIMEOUT_S)

    assert not first.is_alive() and not second.is_alive()
    assert len(calls) == 1
    assert platforms._calibration_loaded is True
    assert platforms.mode_available("hinge", "training") is False
    assert platforms.mode_available("hinge", "auto") is False


def test_lazy_calibration_failure_rolls_back_and_retries(monkeypatch):
    calls = 0

    def load():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("broken driver import")
        platforms._apply_calibration({"hinge": True, "bumble": False})

    monkeypatch.setattr(platforms, "_load_android_calibration", load)
    platforms._calibration_loaded = False
    platforms._calibration_loading_thread = None

    with pytest.raises(RuntimeError, match="broken driver import"):
        platforms._ensure_calibration()
    assert platforms._calibration_loaded is False
    assert platforms._calibration_loading_thread is None

    platforms._ensure_calibration()
    assert calls == 2
    assert platforms._calibration_loaded is True


def test_lazy_calibration_allows_same_thread_import_reentry(monkeypatch):
    calls = 0

    def load():
        nonlocal calls
        calls += 1
        platforms._ensure_calibration()
        platforms._apply_calibration({"hinge": True, "bumble": False})

    monkeypatch.setattr(platforms, "_load_android_calibration", load)
    platforms._calibration_loaded = False
    platforms._calibration_loading_thread = None

    platforms._ensure_calibration()

    assert calls == 1
    assert platforms._calibration_loaded is True
