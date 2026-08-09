"""The platform registry: the single source of truth for what we can drive, and how.

Two things used to be conflated: *which dating app* (Bumble, Hinge) and *how we drive
it* (a browser, or the phone). That was fine while the two happened to line up one to
one, and it stopped being fine in August 2026 when Bumble discontinued its web app
(https://support.bumble.com/hc/en-us/articles/30996192802973-An-update-on-Bumble-web).
Bumble is now an Android target like Hinge, and "web" is a delivery mechanism with no
live target behind it rather than a synonym for Bumble.

So a platform is a (kind, app) pair:

  kind  = how we drive it. "android" = host side ADB + vision on the Pixel;
          "web" = Playwright/patchright against a real Chrome.
  app   = which dating app, e.g. hinge / bumble.

The hub renders exactly this structure: one button per kind, expanding to a single
choice among that kind's targets.

Availability is deliberately a property of the registry rather than something each
driver discovers at runtime, so an unrunnable platform is rejected BEFORE any driver
is constructed, any browser launches, or any tap reaches the phone. Two things make a
platform unavailable:

  * no live target      -- the web kind, since Bumble web shut down. The driver code is
                           kept (see operation_love/drivers/web/) because the hardening
                           in it is site agnostic and worth inheriting if we ever pick
                           up another web based platform.
  * not yet calibrated  -- an Android target whose tap coordinates and glyph templates
                           are still placeholders. Running it would fire real touches at
                           guessed coordinates on a real account, so it fails closed.

Note there is deliberately NO per-platform storage-bucket field here; see the comment in
`Platform` for why the one that existed was removed rather than left unread.
"""
from __future__ import annotations

from dataclasses import dataclass

# How we drive a platform. The hub shows one button per kind, in this order.
KIND_ANDROID = "android"
KIND_WEB = "web"

KIND_ORDER = (KIND_ANDROID, KIND_WEB)
KIND_LABELS = {KIND_ANDROID: "App-based", KIND_WEB: "Web-based"}

# Every Android target contends for the one physical Pixel: Android shows exactly one
# app in the foreground, `adb exec-out screencap` captures whatever is on top, and the
# UHID virtual touchscreen delivers to whatever holds focus. So two Android platforms
# can never run at once -- see _EXCLUSIVE below and supervisor's device lock.
RESOURCE_ANDROID_DEVICE = "android-device"

_WEB_NO_TARGET = (
    "Additional work needed to get this to run. Bumble discontinued its web app in "
    "August 2026, so the web path has no live target right now. The driver logic is "
    "kept and generalised, ready for a future web based platform."
)


@dataclass(frozen=True)
class Platform:
    """One thing we can (or cannot yet) drive."""

    app: str                      # registry id; also the config key under `apps:`
    label: str                    # what the hub shows
    kind: str                     # KIND_ANDROID | KIND_WEB
    available: bool               # False => refuse to start, with `reason`
    reason: str | None = None     # why it cannot run; shown verbatim in the hub

    # NOT here: a `store_key`/`bucket` that would pool Bumble-web and Bumble-app history
    # under one id. That existed briefly and was DEAD CODE -- nothing outside this module
    # ever read it. What actually reaches the store is the raw registry id, because
    # supervisor.run() passes `app` straight into Worker(), which passes self.app into
    # store.add_label/record_decision/record_profile/count_today.
    #
    # Removed rather than left in place, because a property that merely LOOKS like it
    # groups history is worse than none: it invites the belief that daily rate-limit
    # counting and photo archival already pool across a dating app's transports when they
    # do not. (Ranker TRAINING is unaffected either way -- store.load_labels() has no
    # per-app filter and already pools every app into one training set.)
    #
    # If a web platform ever goes live again alongside its Android twin, the pooling has to
    # be wired at the store boundary -- supervisor/Worker would need to carry a storage id
    # distinct from the display/registry id -- not re-added as an unread property here.

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
            "Bumble on Android is not calibrated yet. Its tap coordinates and glyph "
            "templates are placeholders, so running it would fire real touches at "
            "guessed points. Calibrate against the Pixel first."
        ),
    ),
    Platform(
        app="bumble_web",
        label="Bumble (web)",
        kind=KIND_WEB,
        available=False,
        reason=_WEB_NO_TARGET,
    ),
)

# The hand-written entries above, captured before any calibration is applied. _apply_calibration
# rebuilds from THIS rather than from the live tuple, so a calibrate -> de-calibrate round trip
# restores the original `reason` text instead of leaving it None (rebuilding from the mutated
# entry lost the specific "placeholder coordinates" explanation permanently, degrading the hub
# message to a generic "X is not available").
_ORIGINAL: dict[str, Platform] = {p.app: p for p in _PLATFORMS}

_BY_APP = {p.app: p for p in _PLATFORMS}

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


def unavailable_reason(app: str) -> str | None:
    """Why `app` cannot run right now, or None if it can.

    Callers must check this before constructing a driver: it is the guard that keeps an
    uncalibrated spec from ever reaching the phone.
    """
    p = get(app)
    return None if p.available else (p.reason or f"{p.label} is not available.")


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


def check_runnable(apps: list[str]) -> str | None:
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
        reason = unavailable_reason(app)
        if reason:
            return reason
    return check_selection(apps)


_calibration_loaded = False


def _ensure_calibration() -> None:
    """Pull each Android spec's `calibrated` flag into the registry, once, on first query.

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
    global _calibration_loaded
    if _calibration_loaded:
        return
    _calibration_loaded = True      # set BEFORE the import: it re-enters this module
    from .drivers import android    # noqa: F401 -- imported for its registration side effect


def _apply_calibration(calibrated: dict[str, bool]) -> None:
    """Re-derive Android availability from the driver specs' `calibrated` flags.

    Keeping the flag on the spec (next to the coordinates it describes) rather than
    duplicated here means a spec cannot claim to be calibrated in one file and be
    placeholders in another. Invoked via _ensure_calibration() above.

    Rebuilds each entry from `_ORIGINAL`, never from the current (possibly already-mutated)
    one. Reading the live entry's `reason` meant a calibrate -> de-calibrate round trip in
    one process permanently lost the specific "placeholder coordinates" explanation --
    calibrating cleared it to None, and de-calibrating then copied that None forward, so the
    hub degraded to a generic "Bumble is not available."
    """
    global _PLATFORMS, _BY_APP
    unknown = set(calibrated) - set(_ORIGINAL)
    if unknown:
        raise ValueError(
            f"Calibration reported for unregistered app(s): {', '.join(sorted(unknown))}. "
            f"Add them to the registry in this module first."
        )
    updated = []
    for p in _PLATFORMS:
        pristine = _ORIGINAL[p.app]
        if p.kind == KIND_ANDROID and p.app in calibrated:
            is_cal = calibrated[p.app]
            updated.append(Platform(
                app=pristine.app,
                label=pristine.label,
                kind=pristine.kind,
                available=is_cal,
                reason=None if is_cal else pristine.reason,
            ))
        else:
            updated.append(p)
    _PLATFORMS = tuple(updated)
    _BY_APP = {p.app: p for p in _PLATFORMS}
    # KNOWN_APPS is intentionally NOT reassigned -- see its definition above.
    if set(_BY_APP) != set(KNOWN_APPS):     # not an assert: must survive `python -O`
        raise AssertionError("calibration must not change the platform set")
