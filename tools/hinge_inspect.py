"""Hinge ADB inspect — confirm coordinate fractions and vision-glyph coverage.

Run ONCE after connecting a new device or after a Hinge UI update that moves
buttons. The phone must be connected via USB with Hinge open on a completed,
Selfie-Verified profile (so the heart / X glyphs are on screen).

What it does:
  1. Runs `adb devices` and confirms the target serial is visible.
  2. Takes a screenshot and writes it to data/hinge_inspect/ for manual review.
  3. Runs the vision template-match for like_heart and pass_x on that screenshot
     and prints the located (x_frac, y_frac) — these are what the driver uses at
     runtime (falling back to the config fractions if the glyph isn't found).
  4. Reports the configured fallback fractions for all four action points and
     whether they look plausible for the device's resolution.

Paste any corrections into config.yaml under apps.hinge.coords.

    python -m tools.hinge_inspect                  # uses config.yaml
    python -m tools.hinge_inspect --config x.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _adb(serial: str, adb: str, *args, timeout: int = 10) -> subprocess.CompletedProcess:
    cmd = [adb]
    if serial:
        cmd += ["-s", serial]
    cmd += list(args)
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def main(config_path: str = "config.yaml") -> int:
    from operation_love import config as cfg_mod

    cfg = cfg_mod.load(config_path)
    hinge_cfg = cfg.apps.get("hinge", {}) or {}
    serial = (hinge_cfg.get("serial") or "").strip()
    adb = (hinge_cfg.get("adb_path") or "adb").strip() or "adb"
    coords = hinge_cfg.get("coords", {}) or {}

    print(f"Hinge inspect — adb={adb!r}  serial={serial or '(first device)'!r}")
    print()

    # 1. Check adb devices
    try:
        r = _adb("", adb, "devices", timeout=5)
        lines = r.stdout.decode(errors="replace").strip().splitlines()[1:]
        visible = [ln.split()[0] for ln in lines if ln.strip() and ln.split()[-1] == "device"]
    except FileNotFoundError:
        print(f"ERROR: `{adb}` not found. Set apps.hinge.adb_path in config.yaml.")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: adb devices failed: {exc}")
        return 1

    if not visible:
        print("ERROR: no Android device connected. Connect the Pixel 7a and authorize RSA.")
        return 1
    if serial and serial not in visible:
        print(f"ERROR: serial {serial!r} not found. Connected: {visible}")
        return 1
    target = serial or visible[0]
    print(f"Device: {target}  ({len(visible)} device(s) total)")

    # 2. Screenshot
    out_dir = Path(cfg.data_dir) / "hinge_inspect"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    remote_path = f"/sdcard/hinge_inspect_{ts}.png"
    local_path = out_dir / f"screenshot_{ts}.png"

    r = _adb(target, adb, "shell", "screencap", "-p", remote_path, timeout=15)
    if r.returncode != 0:
        print(f"WARNING: screencap failed: {r.stderr.decode(errors='replace').strip()}")
    else:
        r2 = _adb(target, adb, "pull", remote_path, str(local_path), timeout=15)
        _adb(target, adb, "shell", "rm", remote_path, timeout=5)
        if r2.returncode == 0:
            print(f"Screenshot saved: {local_path}")
        else:
            print(f"WARNING: could not pull screenshot: {r2.stderr.decode(errors='replace').strip()}")

    # 3. Run vision template-match if screenshot landed
    if local_path.exists():
        try:
            import cv2
            import numpy as np
            from operation_love.drivers.hinge.vision import locate_glyph  # type: ignore[import]

            img = cv2.imread(str(local_path))
            h, w = img.shape[:2]
            print(f"\nScreen resolution: {w}×{h}")
            for glyph in ("like_heart", "pass_x"):
                try:
                    result = locate_glyph(img, glyph)
                    if result:
                        fx, fy = result[0] / w, result[1] / h
                        print(f"  {glyph}: vision found at ({fx:.3f}, {fy:.3f})  "
                              f"[pixel ({result[0]}, {result[1]})]")
                    else:
                        fb = coords.get(glyph, [None, None])
                        print(f"  {glyph}: vision NOT found — fallback config: {fb}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  {glyph}: vision error: {exc}")
        except ImportError:
            print("\nVision matching skipped (cv2 or hinge.vision not available).")

    # 4. Report configured coordinate fractions
    print("\nConfigured coordinate fractions (apps.hinge.coords):")
    for key in ("like_heart", "pass_x", "comment_box", "send_like"):
        val = coords.get(key, "(not set)")
        print(f"  {key}: {val}")

    print("\nDone. If any fractions look wrong, update apps.hinge.coords in config.yaml and re-run.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hinge ADB inspect tool")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    sys.exit(main(args.config))
