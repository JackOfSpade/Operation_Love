"""The platform registry: the single source of truth for Android app targets.

Availability is checked before any driver is constructed or any touch reaches the
phone. An uncalibrated target fails closed rather than acting at guessed coordinates.
"""
from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass

from .targeting_policy import hinge_targeting_unavailable_reason

# How we drive a platform. The hub shows one button per kind, in this order.
KIND_ANDROID = "android"

KIND_ORDER = (KIND_ANDROID,)
KIND_LABELS = {KIND_ANDROID: "App-based"}

# Every Android target contends for the one physical Pixel: Android shows exactly one
# app in the foreground, `adb exec-out screencap` captures whatever is on top, and the
# UHID virtual touchscreen delivers to whatever holds focus. So two Android platforms
# can never run at once -- see _EXCLUSIVE below and supervisor's device lock.
RESOURCE_ANDROID_DEVICE = "android-device"

@dataclass(frozen=True)
class Platform:
    """One thing we can (or cannot yet) drive."""

    app: str                      # registry id; also the config key under `apps:`
    label: str                    # what the hub shows
    kind: str                     # KIND_ANDROID
    available: bool               # False => refuse to start, with `reason`
    reason: str | None = None     # why it cannot run; shown verbatim in the hub

    @property
    def exclusive_resource(self) -> str | None:
        """A resource only one running platform may hold, or None if unconstrained."""
        return RESOURCE_ANDROID_DEVICE if self.kind == KIND_ANDROID else None


# The registry. Adding a platform means adding a row here plus a driver; every other
# layer (worker, store, status, opener, costing, limits) already treats `app` as an
# opaque string and needs no change.
_PLATFORMS: tuple[Platform, ...] = (
    Platform(
        app="hinge",
        label="Hinge",
        kind=KIND_ANDROID,
        available=True,
    ),
    Platform(
        app="bumble",
        label="Bumble",
        kind=KIND_ANDROID,
        # Availability is recomputed from the driver spec's `calibrated` flag at
        # registry build time (see _apply_calibration below) so a placeholder spec can
        # never be started by accident.
        available=False,
        reason=(
            "Bumble on Android is not calibrated yet. Its card-drag coordinates are not "
            "release-licensed and the required paid-upsell detection template is absent, "
            "so both Training and Auto fail closed. Calibrate against the Pixel first."
        ),
    ),
)

# The hand-written entries above, captured before any calibration is applied. _apply_calibration
# rebuilds from THIS rather than from the live tuple, so a calibrate -> de-calibrate round trip
# restores the original `reason` text instead of leaving it None (rebuilding from the mutated
# entry lost the specific "placeholder coordinates" explanation permanently, degrading the hub
# message to a generic "X is not available").
_ORIGINAL: dict[str, Platform] = {p.app: p for p in _PLATFORMS}

_BY_APP = {p.app: p for p in _PLATFORMS}

# Mode-specific readiness is separate from the coarse Platform.available bit.  The latter
# answers whether the hub may offer a platform at all; this table answers whether the exact
# requested run mode is licensed. Production registration derives both bits from each
# Android spec; implementation code or unit coverage alone never makes a mode runnable.
_AVAILABLE_MODES: dict[str, frozenset[str]] = {
    "hinge": frozenset({"training", "auto"}),
    "bumble": frozenset(),
}

# Every registered id, for config validation. Replaces config._KNOWN_APPS.
# Deliberately assigned exactly once: calibration can flip a platform's AVAILABILITY but
# never changes which platforms exist, so this stays safe to `from platforms import
# KNOWN_APPS`. Availability must always be read through unavailable_reason()/get().
KNOWN_APPS = frozenset(_BY_APP)


def all_platforms() -> tuple[Platform, ...]:
    """Every registered platform, grouped by kind in KIND_ORDER."""
    _ensure_calibration()
    return tuple(p for kind in KIND_ORDER for p in _PLATFORMS if p.kind == kind)


def get(app: str) -> Platform:
    _ensure_calibration()
    try:
        return _BY_APP[app]
    except KeyError:
        known = ", ".join(sorted(KNOWN_APPS))
        raise ValueError(f"Unknown app '{app}'. Supported: {known}.") from None


def kinds() -> tuple[str, ...]:
    """Kinds that have at least one registered platform, in display order."""
    _ensure_calibration()
    present = {p.kind for p in _PLATFORMS}
    return tuple(k for k in KIND_ORDER if k in present)


def for_kind(kind: str) -> tuple[Platform, ...]:
    _ensure_calibration()
    return tuple(p for p in all_platforms() if p.kind == kind)


def unavailable_reason(app: str, mode: str | None = None) -> str | None:
    """Why `app` cannot run right now, or None if it can.

    Callers must check this before constructing a driver: it is the guard that keeps an
    uncalibrated spec from ever reaching the phone.
    """
    p = get(app)
    if mode is not None:
        if mode not in {"training", "auto"}:
            if mode in {"observe", "auto_testing"}:
                return (f"Mode {mode!r} was retired; choose 'training' for Hub-reviewed "
                        "Like/Dislike labels or 'auto' for ranker decisions.")
            return f"Unsupported mode {mode!r}; choose 'training' or 'auto'."
        if app == "hinge" and mode in {"auto", "training"}:
            blocker = hinge_targeting_unavailable_reason()
            if blocker is not None:
                return f"Hinge {mode.title()} is blocked: {blocker}."
            # This registry can assess driver geometry and still-photo readiness, but release
            # evidence is config-bound (artifact path/hash, device/build, calibration, and
            # production run). `supervisor.load_effective_config()` validates that exact gate
            # synchronously before a Hub thread or any device driver is created.
        available_modes = _AVAILABLE_MODES.get(app, frozenset())
        if mode in available_modes:
            return None
        if p.available:
            supported = " and ".join(m.replace("_", " ").title() for m in ("training", "auto")
                                     if m in available_modes)
            if supported:
                return f"{p.label} supports {supported} only; {mode.title()} is not available."
            return f"{p.label} does not support {mode.title()}."
    return None if p.available else (p.reason or f"{p.label} is not available.")


def mode_available(app: str, mode: str) -> bool:
    """Whether the exact app/mode pair can be started by the hub."""
    return unavailable_reason(app, mode) is None


def check_selection(apps: list[str]) -> str | None:
    """Structural validation only: are these real platforms, and can they coexist?

    Deliberately says nothing about AVAILABILITY. Whether a platform can run today is a
    property of the world (is it calibrated? does the target still exist?) that changes
    without the config file changing, so refusing to even LOAD a config that mentions an
    uncalibrated platform would be wrong -- you must be able to configure Bumble's
    coordinates before Bumble is calibrated. Config validation calls this; start-time
    calls check_runnable() below.
    """
    _ensure_calibration()
    if not apps:
        return "Select a platform to run."

    for app in apps:
        if app not in _BY_APP:
            known = ", ".join(sorted(KNOWN_APPS))
            return f"Unknown app '{app}'. Supported: {known}."

    held: dict[str, str] = {}
    for app in apps:
        resource = get(app).exclusive_resource
        if resource is None:
            continue
        if resource in held:
            first, second = get(held[resource]).label, get(app).label
            return (
                f"{first} and {second} both need the phone, and Android runs one app in "
                f"the foreground at a time, so they cannot run together. Run one, then "
                f"the other."
            )
        held[resource] = app
    return None


def check_runnable(apps: list[str], modes: dict[str, str] | str | None = None) -> str | None:
    """Everything check_selection() checks, PLUS whether each platform can run right now.

    This is the start-time gate -- the last thing between a selection and a driver that
    would launch a browser or touch the phone. Availability is checked AFTER the
    structural checks so an unknown id reports as an unknown id, and BEFORE the
    resource-contention check so an uncalibrated platform explains its real problem
    instead of sending you chasing a device conflict.
    """
    for app in apps:
        if app not in _BY_APP:
            continue          # let check_selection own the wording for unknown ids
        mode = modes.get(app) if isinstance(modes, dict) else modes
        reason = unavailable_reason(app, mode=mode)
        if reason:
            return reason
    return check_selection(apps)


_calibration_loaded = False
_calibration_loading_thread: int | None = None
_calibration_lock = threading.RLock()


def _load_android_calibration() -> None:
    """Import the driver package whose module body registers the current specs."""
    from .drivers import android  # noqa: F401 -- imported for its registration side effect


def _ensure_calibration() -> None:
    """Pull each Android spec's per-mode readiness into the registry on first query.

    The specs are the source of truth for whether a platform's coordinates have been
    verified on a real device, but they live in the driver package -- which this module
    cannot import at module scope, because that package imports this one.

    It used to be wired the other way round: drivers/android/__init__.py called
    _apply_calibration on import, and every caller was assumed to have imported that
    package first. Nothing in a real run did. supervisor.run() and HubState.start() both
    ask check_runnable() BEFORE constructing any driver, and make_driver()'s Hinge branch
    imports .hinge, not .android -- so availability came from the hand-written literals in
    this file and the specs were never consulted at all. The two happened to agree, which
    is precisely why it went unnoticed: flipping a spec's `calibrated` flag (the documented,
    intended way to enable a platform) would have changed nothing in a fresh process.

    The whole test suite masked it too -- test_android_spec.py imports the driver package at
    collection time, reconciling the registry as a side effect before any test body runs.
    """
    global _calibration_loaded, _calibration_loading_thread
    if _calibration_loaded:
        return
    current = threading.get_ident()
    with _calibration_lock:
        if _calibration_loaded:
            return
        # The lazy driver import can traverse modules that consult the registry. An RLock
        # prevents deadlock; this owner check prevents the re-entrant call from starting a
        # second import and falsely marking a partially initialized registry complete.
        if _calibration_loading_thread == current:
            return
        _calibration_loading_thread = current
        try:
            _load_android_calibration()
        except BaseException:
            # Import failures must remain retryable. The old eager flag permanently latched
            # success before registration had actually completed.
            _calibration_loaded = False
            raise
        else:
            _calibration_loaded = True
        finally:
            _calibration_loading_thread = None


def _apply_calibration(calibrated: Mapping[str, bool | Mapping[str, bool]]) -> None:
    """Re-derive Android availability from the driver specs' per-mode readiness.

    Keeping those values on the spec (next to the coordinates they describe) rather than
    duplicating them here means registry literals cannot bypass an uncalibrated binding.
    Config-bound Hinge AUTO release evidence is validated separately at run start.
    Invoked via _ensure_calibration() above.

    Rebuilds each entry from `_ORIGINAL`, never from the current (possibly already-mutated)
    one. Reading the live entry's `reason` meant a calibrate -> de-calibrate round trip in
    one process permanently lost the specific "placeholder coordinates" explanation --
    calibrating cleared it to None, and de-calibrating then copied that None forward, so the
    hub degraded to a generic "Bumble is not available."
    """
    global _PLATFORMS, _BY_APP, _calibration_loaded
    if not isinstance(calibrated, Mapping):
        raise ValueError("Calibration must be a mapping of app id to readiness")
    unknown = set(calibrated) - set(_ORIGINAL)
    if unknown:
        raise ValueError(
            f"Calibration reported for unregistered app(s): "
            f"{', '.join(sorted(map(repr, unknown)))}. "
            f"Add them to the registry in this module first."
        )
    mode_updates: dict[str, frozenset[str]] = {}
    # Derive and validate every record before mutating either live registry table. A malformed
    # later app must not leave an earlier app's modes half-applied.
    for app, state in calibrated.items():
        if isinstance(state, Mapping):
            unexpected_modes = set(state) - {"training", "auto"}
            if unexpected_modes:
                raise ValueError(
                    f"Calibration for {app!r} named unsupported mode(s): "
                    f"{', '.join(sorted(map(repr, unexpected_modes)))}")
            malformed = {
                mode: value for mode, value in state.items()
                if type(value) is not bool
            }
            if malformed:
                raise ValueError(
                    f"Calibration readiness for {app!r} must use exact booleans; "
                    f"got {malformed!r}")
            # Training opens a targeted composer, types an opener, then performs a verified
            # Like or Dislike.  It therefore needs the same mechanical calibration as AUTO.
            # Existing specs predate this key, so it inherits explicit AUTO readiness.
            effective = dict(state)
            if "training" not in effective:
                effective["training"] = effective.get("auto", False)
            modes = frozenset(
                mode for mode in ("training", "auto")
                if effective.get(mode) is True)
        elif type(state) is bool:
            # Backwards-compatible test/tool API: the historic single flag licensed both
            # modes. Production Android registration now supplies the explicit mapping.
            modes = frozenset({"training", "auto"}) if state else frozenset()
        else:
            raise ValueError(
                f"Calibration for {app!r} must be an exact boolean or a mapping of "
                f"training/auto to exact booleans (got {state!r})")
        mode_updates[app] = modes

    updated = []
    for p in _PLATFORMS:
        pristine = _ORIGINAL[p.app]
        if p.kind == KIND_ANDROID and p.app in mode_updates:
            modes = mode_updates[p.app]
            is_cal = bool(modes)
            updated.append(Platform(
                app=pristine.app,
                label=pristine.label,
                kind=pristine.kind,
                available=is_cal,
                reason=None if is_cal else pristine.reason,
            ))
        else:
            updated.append(p)
    _AVAILABLE_MODES.update(mode_updates)
    _PLATFORMS = tuple(updated)
    _BY_APP = {p.app: p for p in _PLATFORMS}
    # Direct calibration callers (the supported test/tool seam) have supplied the live
    # readiness table, so a subsequent availability read must not lazy-import and overwrite
    # it with the packaged defaults before it can be observed.
    _calibration_loaded = True
    # KNOWN_APPS is intentionally NOT reassigned -- see its definition above.
    if set(_BY_APP) != set(KNOWN_APPS):     # not an assert: must survive `python -O`
        raise AssertionError("calibration must not change the platform set")
