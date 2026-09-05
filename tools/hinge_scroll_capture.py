"""READ-ONLY dense-frame capture of a human-driven Hinge scroll, for card-geometry
calibration.

Run while a HUMAN scrolls a real Hinge profile by hand. It repeatedly calls
`operation_love.drivers.adb.Adb.screencap()` — the exact `adb exec-out screencap -p` pipeline
production perception uses (drivers/adb.py:231-232) — and saves every frame whose bytes differ
from the previously saved one, building a dense, in-order sequence of the scroll. Those frames
are for measuring the card geometry ops/OPENER-REDESIGN.md §5.4 and §5.5 need before Part B can
be built: gutters between cards, heart-button positions, and whether a card can exceed the
viewport (the open question at OPENER-REDESIGN.md §9). This tool does not itself measure or
segment anything — that analysis happens later, offline, against the saved frames. It does NOT
implement any part of Part B.

This never taps, swipes, or sends any input to the phone — you stay in physical control the
whole time, exactly like tools/hinge_inspect.py's button probe. It goes further, though:
hinge_inspect.py drives a real HingeDriver (input-free in what it does, but the class itself
owns swipe/tap capability). This tool never constructs any app driver at all, never imports the
touch transport (adb.py's tap/swipe/text methods, uhid.py), and the one `Adb` instance it
builds has exactly one method called on it anywhere in this file: `screencap()`. Device
enumeration (for --serial resolution) goes through a raw `adb devices` subprocess call parsed
by adb.py's own pure `parse_devices_output`, deliberately kept off that instance, so the
"only ever calls screencap()" claim has nothing to explain away. tests/test_hinge_scroll_capture
enforces this mechanically: it scans this file's own source for a list of input-injection
substrings (see that test file for the exact list) and fails if any is present, so the
guarantee can't silently rot.

    python -m tools.hinge_scroll_capture                          # 120s, single connected device
    python -m tools.hinge_scroll_capture --seconds 60 --serial 33111JEHN04475
    python -m tools.hinge_scroll_capture --out ops/calibration/scroll_manual/

Privacy: frames are real people's dating profiles. The default output lives under
ops/calibration/, which is gitignored (see .gitignore's comment above that line) — local-only
by rule, not by accident. Nothing in this tool uploads, copies outside the repo, or transmits a
frame anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from operation_love import config as cfg_mod
from operation_love.drivers.adb import Adb, parse_devices_output
from operation_love.private_files import (
    atomic_write_private_bytes,
    atomic_write_private_text,
    ensure_private_dir,
)
from tools._devicelock import run_holding_the_device

_TOOL_VERSION = "1"
# Matches the driver's own observe-mode sampling cadence (operation_love/drivers/hinge.py:208
# `_OBSERVE_POLL_S = 0.35`). Not imported from there on purpose: this tool has no other reason
# to import hinge.py at all, and staying independent of it keeps "structurally incapable of
# sending input" a property of this file alone, not one that depends on hinge.py never growing
# a module-level side effect underneath it.
_DEFAULT_INTERVAL_S = 0.35
_DEFAULT_SECONDS = 120.0


def _list_ready_devices(adb_path: str = "adb") -> list[str]:
    """Enumerate ready device serials via a raw `adb devices` call, parsed by adb.py's own
    `parse_devices_output` — a pure function, not a method call on an `Adb` instance. Device
    enumeration sends no input to any device, but is kept off the `Adb` object this tool later
    builds for capture, so that object's call surface stays exactly `screencap()`, always."""
    try:
        result = subprocess.run(
            [adb_path, "devices"], capture_output=True, timeout=10, check=False)
    except FileNotFoundError:
        print(f"ERROR: adb binary not found ({adb_path!r} not on PATH).", file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        # A path can exist but still be non-executable (or be denied by host policy).  This is
        # an operator-facing preflight, so turn every launch failure into the same clean refusal
        # rather than exposing a traceback before any capture begins.
        print(f"ERROR: could not run `adb devices`: {exc}", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("ERROR: `adb devices` timed out.", file=sys.stderr)
        sys.exit(1)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        print(f"ERROR: `adb devices` failed with exit code {result.returncode}{suffix}",
              file=sys.stderr)
        sys.exit(1)
    return parse_devices_output(result.stdout.decode("utf-8", errors="replace"))


def _resolve_serial(requested: str | None, adb_path: str = "adb") -> str:
    """--serial if given (and connected); the single connected device otherwise. Exits
    non-zero with a clear message if adb reports no device, or reports more than one and
    --serial was not given to disambiguate."""
    devices = _list_ready_devices(adb_path)
    if not devices:
        print("ERROR: no ADB device connected (`adb devices` reports none ready).",
              file=sys.stderr)
        sys.exit(1)
    if requested:
        if requested not in devices:
            print(f"ERROR: device {requested!r} not connected. Ready devices: {devices}",
                  file=sys.stderr)
            sys.exit(1)
        return requested
    if len(devices) > 1:
        print(f"ERROR: multiple ADB devices connected {devices} and no --serial given. "
              "Pass --serial to pick one.", file=sys.stderr)
        sys.exit(1)
    return devices[0]


def _default_out_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("ops/calibration") / f"scroll_{stamp}"


def capture(*, adb, seconds: float, interval: float, out_dir: Path,
            serial: str, progress: bool = True) -> dict:
    """The capture loop. Writes `out_dir/manifest.json` and returns it as a dict.

    `adb` only needs a `screencap() -> bytes` method (a real `Adb`, or a fake in tests) — this
    function never calls anything else on it. Saves a frame ONLY when its sha256 digest differs
    from the previously SAVED frame's digest, the same dedupe-by-hash approach
    `DebugLog._save_shot` uses (operation_love/drivers/debuglog.py:48-69): a human deliberating
    between scrolls produces long runs of byte-identical polls, and saving every one of them
    would bury the frames that actually show new content. Frames are named `00001.png`,
    `00002.png`, ... in save order, so scroll order is unambiguous and there are never gaps.

    Ctrl-C (KeyboardInterrupt) stops the loop cleanly; the manifest is still written, with
    `interrupted: true` and whatever frames were captured before the interrupt.
    """
    # Reject invalid timing before creating a directory or touching the device.  A negative
    # interval otherwise turns the loop into a busy poll, while a non-positive/NaN duration
    # produces an empty, misleadingly successful capture manifest.
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be a finite value greater than zero")
    if not math.isfinite(interval) or interval < 0:
        raise ValueError("interval must be a finite value greater than or equal to zero")

    ensure_private_dir(out_dir)
    frames: list[dict] = []
    last_digest: str | None = None
    start_wall = time.monotonic()
    start_utc = datetime.now(timezone.utc)
    interrupted = False
    capture_error: str | None = None

    try:
        while (time.monotonic() - start_wall) < seconds:
            loop_start = time.monotonic()
            try:
                png = adb.screencap()
            except Exception as exc:  # noqa: BLE001 — report and stop; manifest still written
                print(f"\nERROR: screencap failed: {exc}", file=sys.stderr)
                # A partial capture is still useful evidence, but it is not a successful one.
                # The prior manifest marked this identical to a time-complete run and printed
                # "Done", inviting a later measurement to treat a device-loss truncation as an
                # intentional manual scroll. Keep the typed failure in the local manifest so an
                # offline reviewer can distinguish it from Ctrl-C and normal completion.
                detail = str(exc)
                capture_error = (f"{type(exc).__name__}: {detail}" if detail
                                 else type(exc).__name__)
                break
            digest = hashlib.sha256(png).hexdigest()
            if digest != last_digest:
                idx = len(frames) + 1
                name = f"{idx:05d}.png"
                atomic_write_private_bytes(out_dir / name, png, parent=out_dir)
                offset_s = time.monotonic() - start_wall
                frames.append({"file": name, "sha256": digest, "offset_s": round(offset_s, 3)})
                last_digest = digest
            if progress:
                elapsed = time.monotonic() - start_wall
                print(f"\r  frames captured: {len(frames):4d}  |  elapsed: {elapsed:6.1f}s "
                      f"/ {seconds:.0f}s", end="", flush=True)
            remaining = interval - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if progress:
            print()  # newline after the \r progress line

    end_utc = datetime.now(timezone.utc)
    manifest = {
        "tool_version": _TOOL_VERSION,
        "device_serial": serial,
        "interval_s": interval,
        "seconds_requested": seconds,
        "start_utc": start_utc.isoformat(),
        "end_utc": end_utc.isoformat(),
        "interrupted": interrupted,
        "capture_error": capture_error,
        "frame_count": len(frames),
        "frames": frames,
    }
    atomic_write_private_text(
        out_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n", parent=out_dir)
    verb = ("Stopped early (Ctrl-C)" if interrupted else
            "Stopped after screencap error" if capture_error is not None else "Done")
    print(f"{verb}. Wrote {len(frames)} frame(s) + manifest to {out_dir}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="READ-ONLY dense-frame capture of a human-driven Hinge scroll, for "
                     "card-geometry calibration (ops/OPENER-REDESIGN.md 5.4/5.5/9). Never "
                     "sends input to the phone.")
    ap.add_argument("--serial", default=None,
                     help="ADB device serial; defaults to the single connected device")
    ap.add_argument("--seconds", type=float, default=_DEFAULT_SECONDS,
                     help=f"how long to capture, in seconds (default {_DEFAULT_SECONDS:.0f})")
    ap.add_argument("--interval", type=float, default=_DEFAULT_INTERVAL_S,
                     help="poll cadence in seconds, matching the driver's own observe poll "
                          f"cadence (default {_DEFAULT_INTERVAL_S})")
    ap.add_argument("--out", default=None,
                     help="output directory (default ops/calibration/scroll_<UTC timestamp>/)")
    ap.add_argument("--config", default=None,
                     help="optional config file, accepted for symmetry with the other phone "
                          "tools; this capture reads no setting from it and takes the run's "
                          "device lock with or without one")
    args = ap.parse_args(argv)

    out_dir = Path(args.out) if args.out else _default_out_dir()

    print("READ-ONLY: this tool never taps or swipes the phone. You stay in control the whole "
          "time; it only reads screencaps.")
    print("Frames are LOCAL-ONLY: ops/calibration/ is gitignored — these are real people's "
          "dating profiles, so do not upload, copy outside the repo, or transmit them anywhere.")

    # This capture is NOT gated on a config that validates, and that is deliberate. The device
    # lock is one file per user for "the Android phone" and reads nothing out of a config
    # (supervisor._android_lock_path ignores both its arguments, on purpose), so the tool holds
    # exactly the lock a run holds with no config in the picture at all. An earlier version
    # loaded config.yaml first and exited on any error -- which let an unrelated `opener.max_chars`
    # complaint block a read-only screencap loop, i.e. the instrument the owner reaches for
    # precisely BECAUSE the config/calibration is mid-repair. An interlock on a value the lock
    # never consults protects nothing and refuses the tool the job it exists for.
    #
    # An EXPLICIT --config that will not load is still worth saying out loud (it is usually a
    # typo'd path, and the operator clearly expected that file to matter), but it is a WARNING:
    # nothing below reads it, so there is no decision it could make wrong.
    if args.config is not None:
        try:
            cfg_mod.load(args.config)
        except Exception as exc:  # noqa: BLE001 -- one operator-facing warning, never fatal
            print(f"WARNING: could not load {args.config}: {exc}", file=sys.stderr)
            print("WARNING: continuing -- this capture reads nothing from the config, and "
                  "still takes the run's device lock.", file=sys.stderr)

    def _capture_holding_the_device() -> None:
        # The lock spans the WHOLE capture, not just the session open: this reads only, but a
        # hub run walking the deck underneath the owner's hand would silently corrupt the
        # geometry evidence rather than fail, which is the worst way for a calibration to go
        # wrong (tools/_devicelock.py; run a01fbcd1e9a0 is what an honour system cost).
        serial = _resolve_serial(args.serial)
        adb = Adb(serial)

        print(f"\nCapturing from {serial} for {args.seconds:.0f}s at ~{args.interval:.2f}s "
              f"intervals -> {out_dir}")
        print("Scroll the profile by hand now. Press Ctrl-C to stop early.\n")

        capture(adb=adb, seconds=args.seconds, interval=args.interval, out_dir=out_dir,
                serial=serial)

    # `None` for the config path is the config-FREE acquisition, not a fallback: it takes the
    # same lock via the same class and path (see tools/_devicelock.py). Passing `args.config`
    # here would be worse than useless -- a config that will not load makes `holding_the_device`
    # yield an UN-HELD context, so a typo'd path would turn a 120s capture into an unlocked one,
    # exactly the honour system the lock exists to end.
    run_holding_the_device(None, _capture_holding_the_device)


if __name__ == "__main__":
    main()
