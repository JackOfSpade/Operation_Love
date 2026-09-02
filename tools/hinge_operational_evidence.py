"""Read-only recorder for Hinge's supervised operational-check evidence.

This is deliberately separate from ``hinge_calibrate capture``: it lets an operator document
the RUNBOOK's item-1 and gesture-transport checks while a calibration capture is safely paused
at its final evidence prompts.  It never constructs a driver or touch transport.  Its only
device calls are read-only model/framebuffer/build queries and ``screencap``; every heart and
profile advance in this workflow is performed by the owner by hand.  The live Hinge build recorded
in the evidence auto-focuses its inline composer as soon as the heart is pressed, so this
recorder deliberately does not invent a separate unfocused/focus transition.

The output is private, hashed frame evidence below gitignored ``ops/calibration/``.  It is not a
targeting calibration and cannot emit configuration: it proves only the supervised operational
states it records.  In particular, a single item-1 pre-frame is not a complete ``ItemPayload``,
so running ``verify_sheet_item`` on it would pretend to make a closed-set item proof that this
small read-only tool does not possess.  The full calibration capture remains the place for that
item verifier; this recorder uses the pure checks it can prove honestly: scroll-top state,
inline-composer topology, and profile identity fingerprints.

Frame role labels preserve the operator's reported action (for example, that the first heart was
item 1 or that a second heart moved the composer).  The pixels can prove only their own visible
state: top/header verdicts and structurally valid composer geometry.  They do not independently
identify a card ordinal or reconstruct which physical gesture produced a frame.

    python -m tools.hinge_operational_evidence --config config.yaml

Privacy: these frames contain real dating profiles.  Do not copy them outside the private,
gitignored calibration directory or upload them anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from operation_love import config as cfg_mod
from operation_love.drivers import hinge
from operation_love.drivers.adb import Adb, quote_android_package_id
from operation_love.drivers.hinge import HINGE_SPEC
from operation_love.drivers.item_identity import _IDENTITY_GRID, capture_profile_identity
from operation_love.drivers.like_composer import ComposerDetectionError, ComposerSurface, locate_inline_composer
from operation_love.drivers.scroll_top import (
    ScrollTopError, band_fingerprint, confirm_scroll_top, fingerprint_distance)
from operation_love.private_files import (
    atomic_write_private_bytes,
    atomic_write_private_text,
    ensure_private_dir,
)

_TOOL_VERSION = "2"
_LAYOUT_ID = "hinge_inline_v1"
_NEW_PROFILE_MIN_IDENTITY_DISTANCE = 2.565
_PACKAGE = HINGE_SPEC.package
_ROLES = (
    "confirmed_top_item1_pre",
    "composer_initial_autofocused",
    "composer_stable_autofocused",
    "other_item_composer_moved",
    "new_profile_top_clear",
    "new_sticky_identity",
)


class EvidenceRefused(RuntimeError):
    """A required supervised state was not affirmatively visible in its captured frame."""


def _canonical_json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _default_out_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("ops/calibration") / f"operational_evidence_{stamp}"


def _out_dir(raw: str | None) -> Path:
    root = Path("ops/calibration").resolve()
    destination = (Path(raw) if raw else _default_out_dir()).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise EvidenceRefused(
            f"--out must stay below private gitignored {root}; got {destination}") from exc
    if destination.exists() and not destination.is_dir():
        raise EvidenceRefused(f"output {destination} exists and is not a directory")
    if destination.exists() and any(destination.iterdir()):
        raise EvidenceRefused(
            f"output {destination} is not empty; use a fresh directory so evidence cannot merge")
    ensure_private_dir(destination)
    return destination


def _settings(cfg) -> tuple[str, str, tuple[float, float, float, float], object]:
    app_cfg = (getattr(cfg, "apps", {}) or {}).get("hinge", {}) or {}
    serial = app_cfg.get("serial")
    if not isinstance(serial, str) or not serial.strip():
        raise EvidenceRefused("apps.hinge.serial must name the physical phone to record evidence")
    identity_band = app_cfg.get("identity_band", HINGE_SPEC.identity_band)
    if identity_band is None:
        raise EvidenceRefused("effective apps.hinge.identity_band is missing")
    try:
        band = tuple(float(v) for v in identity_band)
    except (TypeError, ValueError) as exc:
        raise EvidenceRefused("effective apps.hinge.identity_band is not a readable rectangle") from exc
    if len(band) != 4 or not (0 <= band[0] < band[2] <= 1 and 0 <= band[1] < band[3] <= 1):
        raise EvidenceRefused("effective apps.hinge.identity_band is not an ordered normalised rectangle")
    template_name = HINGE_SPEC.templates.get("confirm")
    template = hinge._load_template(template_name) if template_name else None
    if template is None:
        raise EvidenceRefused("the shipped Hinge Send Like confirmation template could not be loaded")
    adb_path = app_cfg.get("adb_path", "adb")
    if not isinstance(adb_path, str) or not adb_path:
        raise EvidenceRefused("apps.hinge.adb_path must be a nonempty ADB executable path")
    return serial.strip(), adb_path, band, template


def _read_device_evidence(adb: Adb, *, serial: str, package: str = _PACKAGE) -> dict:
    """Return the read-only device/build facts that bind this trace's pixel geometry.

    Shared with `tools/hinge_calibrate.py`'s `_device_evidence` -- that tool wraps this with a
    thin HingeDriver-shaped adapter rather than reimplementing the probe. Found 2026-09-02: the
    two files had independently hand-rolled this identical ~20-line "getprop/wm density/dumpsys
    package, then parse versionName" probe, and had already drifted (hinge_calibrate.py's copy
    was missing the density check below). The versionName parse also now reuses the same
    anchored multiline regex `operation_love/drivers/hinge.py`'s own
    `_refresh_targeting_calibration_binding` uses to extract "versionName=..." out of a raw
    `dumpsys package` dump, instead of a hand-rolled line-scan loop -- the anchored end-of-line
    match refuses a value with trailing garbage on the same line that a bare
    `.split("=", 1)[1].strip()` would have silently accepted.
    """
    model = adb.shell("getprop ro.product.model").strip()
    width, height = adb.screen_size()
    density = adb.shell("wm density").strip()
    package_dump = adb.shell(
        f"dumpsys package {quote_android_package_id(package)} | grep versionName")
    match = re.search(r"(?m)^\s*versionName=(\S+)\s*$", package_dump)
    version_name = match.group(1) if match else None
    if not model or not density or not version_name:
        raise EvidenceRefused(
            f"could not read complete device/build evidence (model={model!r}, "
            f"density={density!r}, versionName={version_name!r} from `dumpsys package "
            f"{package}`); refusing an operational trace that cannot be bound to one phone "
            "and Hinge version")
    return {
        "serial": serial,
        "model": model,
        "display_w": width,
        "display_h": height,
        "density": density,
        "hinge_package": package,
        "hinge_version_name": version_name,
    }


def _surface_record(surface: ComposerSurface) -> dict:
    return {
        "layout_id": surface.layout_id,
        "comment_rect": asdict(surface.comment_rect),
        "send_rect": asdict(surface.send_rect),
        "confirm_point": list(surface.confirm_point),
    }


def _top_record(verdict) -> dict:
    return {
        "state": verdict.state,
        "distance": verdict.distance,
        "reason": verdict.reason,
        "grid": list(verdict.grid),
    }


def record(*, adb, serial: str, identity_band: tuple[float, float, float, float],
           confirm_template, out_dir: Path, device: dict, ask=input) -> dict:
    """Prompt through one supervised proof and write its immutable-at-rest frame record.

    ``adb`` is intentionally duck-typed to just ``screencap()``.  This function has no route to
    a gesture or text-injection method, which keeps the safety guarantee mechanically simple to
    test.  A failure still writes a manifest with ``completed: false`` for audit, but callers must
    treat it as unusable evidence and start a fresh directory.
    """
    frames: list[dict] = []
    analyses: dict[str, dict] = {}
    start = datetime.now(timezone.utc)
    completed = False
    interrupted = False
    failure: str | None = None
    expected_device_keys = {
        "serial", "model", "display_w", "display_h", "density", "hinge_package",
        "hinge_version_name",
    }
    if (not isinstance(device, dict) or set(device) != expected_device_keys
            or device.get("serial") != serial or device.get("hinge_package") != _PACKAGE
            or not isinstance(device.get("display_w"), int) or device["display_w"] <= 0
            or not isinstance(device.get("display_h"), int) or device["display_h"] <= 0
            or any(not isinstance(device.get(key), str) or not device[key].strip()
                   for key in ("model", "density", "hinge_version_name"))):
        raise EvidenceRefused("device/build evidence is incomplete or does not bind this recorder")

    def take(role: str) -> bytes:
        try:
            png = adb.screencap()
        except Exception as exc:  # noqa: BLE001 -- an unreadable device state is a hard stop
            raise EvidenceRefused(f"could not capture {role}: {type(exc).__name__}: {exc}") from exc
        if not isinstance(png, bytes) or not png:
            raise EvidenceRefused(f"could not capture {role}: screencap returned no PNG bytes")
        name = f"{len(frames) + 1:05d}.png"
        atomic_write_private_bytes(out_dir / name, png, parent=out_dir)
        frames.append({
            "file": name,
            "sha256": hashlib.sha256(png).hexdigest(),
            "role": role,
            "captured_utc": datetime.now(timezone.utc).isoformat(),
        })
        return png

    def require_surface(role: str, frame: bytes) -> ComposerSurface:
        try:
            surface = locate_inline_composer(frame, confirm_template, threshold=0.8)
        except ComposerDetectionError as exc:
            raise EvidenceRefused(f"{role} did not structurally prove an inline composer: {exc}") from exc
        if surface.layout_id != _LAYOUT_ID:
            raise EvidenceRefused(f"{role} reported unsupported composer layout {surface.layout_id!r}")
        analyses[role] = {"composer": _surface_record(surface)}
        return surface

    try:
        ask(
            "On a NEW profile, put item 1 at its confirmed top. Do not heart anything yet. "
            "Press ENTER to record the item-1 pre frame > ")
        top_pre = take("confirmed_top_item1_pre")
        try:
            top_verdict = confirm_scroll_top(top_pre, identity_band=identity_band)
        except ScrollTopError as exc:
            raise EvidenceRefused(f"item-1 pre frame could not confirm scroll top: {exc}") from exc
        if not top_verdict.confirmed:
            raise EvidenceRefused(f"item-1 pre frame is not a confirmed profile top: {top_verdict.reason}")
        analyses["confirmed_top_item1_pre"] = {"scroll_top": _top_record(top_verdict)}

        ask(
            "Manually heart PHOTO item 1 now. Hinge auto-focuses the composer immediately; do "
            "not type. Press ENTER once its inline composer and sticky header are visible > ")
        initial = take("composer_initial_autofocused")
        require_surface("composer_initial_autofocused", initial)
        try:
            initial_top = confirm_scroll_top(initial, identity_band=identity_band)
        except ScrollTopError as exc:
            raise EvidenceRefused(f"initial auto-focused composer could not read identity band: {exc}") from exc
        if not initial_top.refuted:
            raise EvidenceRefused(
                "initial auto-focused composer did not positively reveal the sticky identity "
                "header: " + initial_top.reason)
        analyses["composer_initial_autofocused"]["scroll_top"] = _top_record(initial_top)

        ask(
            "Do not touch the phone. Wait for the auto-focused composer/header to settle, then "
            "press ENTER to record an independent stable reading > ")
        stable = take("composer_stable_autofocused")
        require_surface("composer_stable_autofocused", stable)
        try:
            stable_top = confirm_scroll_top(stable, identity_band=identity_band)
        except ScrollTopError as exc:
            raise EvidenceRefused(f"stable auto-focused composer could not read identity band: {exc}") from exc
        if not stable_top.refuted:
            raise EvidenceRefused(
                "stable auto-focused composer did not positively reveal the sticky identity "
                "header: " + stable_top.reason)
        analyses["composer_stable_autofocused"]["scroll_top"] = _top_record(stable_top)

        ask(
            "Manually heart another PHOTO item on this same profile. Do not send. Press ENTER "
            "once the persistent inline composer has visibly moved to that item > ")
        moved = take("other_item_composer_moved")
        require_surface("other_item_composer_moved", moved)

        # This is the strongest identity statement these frames can honestly make: confirmed
        # top followed by two independently captured, auto-focused sticky-header readings on
        # the same profile.  The moved-composer state is transport evidence, not an identity
        # sample: its role is an operator claim that pixels alone cannot establish.
        identity = capture_profile_identity(
            [top_pre, initial, stable], identity_band=identity_band, grid=_IDENTITY_GRID)
        if not identity.known or identity.fingerprint is None:
            raise EvidenceRefused("item-1 proof did not yield a corroborated profile identity: "
                                  + identity.reason)
        analyses["item1_profile_identity"] = {
            "fingerprint": list(identity.fingerprint),
            "grid": list(identity.grid),
            "frame_index": identity.frame_index,
            "reason": identity.reason,
        }

        ask(
            "Manually advance to a NEW profile's confirmed top. Do not send the composer. "
            "Press ENTER only after the prior composer is gone and filter chips are visible > ")
        new_top = take("new_profile_top_clear")
        try:
            new_top_verdict = confirm_scroll_top(new_top, identity_band=identity_band)
        except ScrollTopError as exc:
            raise EvidenceRefused(f"new-profile top could not be read: {exc}") from exc
        if not new_top_verdict.confirmed:
            raise EvidenceRefused("new-profile frame is not a confirmed top: " + new_top_verdict.reason)
        try:
            locate_inline_composer(new_top, confirm_template, threshold=0.8)
        except ComposerDetectionError:
            pass
        else:
            raise EvidenceRefused("new-profile top still structurally contains an inline composer")
        analyses["new_profile_top_clear"] = {"scroll_top": _top_record(new_top_verdict),
                                               "composer_absent": True}

        ask(
            "On that new profile, manually scroll just enough for its sticky name header to be "
            "stable. Press ENTER to record the new-profile identity > ")
        new_sticky = take("new_sticky_identity")
        try:
            new_sticky_top = confirm_scroll_top(new_sticky, identity_band=identity_band)
            new_fingerprint = band_fingerprint(new_sticky, identity_band=identity_band,
                                               grid=_IDENTITY_GRID)
        except ScrollTopError as exc:
            raise EvidenceRefused(f"new sticky identity could not be read: {exc}") from exc
        if not new_sticky_top.refuted:
            raise EvidenceRefused("new sticky identity was not positively visible: "
                                  + new_sticky_top.reason)
        distance = fingerprint_distance(identity.fingerprint, new_fingerprint)
        if distance <= _NEW_PROFILE_MIN_IDENTITY_DISTANCE:
            raise EvidenceRefused(
                f"new sticky identity is only {distance:.3f} from the item-1 profile (must be "
                f"> {_NEW_PROFILE_MIN_IDENTITY_DISTANCE}); do not certify a same-profile scroll "
                "as a profile advance")
        analyses["new_sticky_identity"] = {
            "scroll_top": _top_record(new_sticky_top),
            "fingerprint": list(new_fingerprint),
            "grid": list(_IDENTITY_GRID),
            "distance_from_item1_profile": distance,
        }
        completed = True
    except KeyboardInterrupt:
        interrupted = True
        failure = "interrupted by Ctrl-C"
    except EvidenceRefused as exc:
        failure = str(exc)
    finally:
        manifest = {
            "tool_version": _TOOL_VERSION,
            "completed": completed,
            "interrupted": interrupted,
            "device": device,
            "frame_size_px": [device["display_w"], device["display_h"]],
            "identity_band": list(identity_band),
            "composer_layout_id": _LAYOUT_ID,
            "config_binding": {
                "serial": serial,
                "identity_band": list(identity_band),
                "composer_layout_id": _LAYOUT_ID,
                "hinge_package": _PACKAGE,
            },
            "new_profile_min_identity_distance": _NEW_PROFILE_MIN_IDENTITY_DISTANCE,
            "start_utc": start.isoformat(),
            "end_utc": datetime.now(timezone.utc).isoformat(),
            "frame_count": len(frames),
            "frames": frames,
            "analyses": analyses,
            "failure": failure,
        }
        # Canonically bind the outer manifest too: per-frame hashes catch a swapped screenshot,
        # while this digest catches unaccompanied corruption of a role or recorded verdict.  It
        # is not a signature and therefore does not claim to defend against a filesystem owner
        # who changes both the JSON and its digest.
        manifest["evidence_sha256"] = _canonical_json_digest(manifest)
        atomic_write_private_text(
            out_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n", parent=out_dir)

    if interrupted:
        raise EvidenceRefused("interrupted by Ctrl-C; evidence manifest is incomplete")
    if failure is not None:
        raise EvidenceRefused(failure)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Read-only Hinge operational-evidence recorder. The owner performs every "
                    "heart and profile advance by hand; this tool only screencaps.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=None,
                        help="private output under ops/calibration/ (default timestamped directory)")
    args = parser.parse_args(argv)

    try:
        cfg = cfg_mod.load(args.config)
        serial, adb_path, identity_band, template = _settings(cfg)
        out_dir = _out_dir(args.out)
        print("READ-ONLY: this tool only queries device evidence and captures screenshots. "
              "Perform every phone action by hand.")
        print(f"Frames remain local under {out_dir}; configured Hinge serial: {serial}")
        adb = Adb(serial, adb_path=adb_path)
        device = _read_device_evidence(adb, serial=serial)
        manifest = record(adb=adb, serial=serial,
                          identity_band=identity_band, confirm_template=template,
                          out_dir=out_dir, device=device)
    except EvidenceRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"SUCCESS: wrote {manifest['frame_count']} hashed evidence frame(s) to {out_dir}")


if __name__ == "__main__":
    main()
