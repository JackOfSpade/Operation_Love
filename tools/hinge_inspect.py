"""Hinge live bring-up inspector — confirm the vision-located like/pass buttons.

Run ONCE on the machine with your Hinge phone connected over ADB. It screencaps the
current Hinge screen and runs the SAME template-matching the real driver uses at
runtime (operation_love.drivers.hinge._load_template / _match_glyph / _locate_button)
to locate the like-heart and pass-X glyphs, then saves an annotated screenshot so you
can eyeball whether the located point actually lands on the button.

This never taps, swipes, or installs anything on the phone — you stay in physical
control the whole time. Unlike a uiautomator2/uiautodev inspector, it never puts an
on-device automation helper on the phone; it only ever calls `adb exec-out screencap`
(see ops/HINGE-PIXEL-RUNBOOK.md §5 on why that helper is the one thing this project
refuses to install).

    python -m tools.hinge_inspect                  # uses config.yaml
    python -m tools.hinge_inspect --config x.yaml
    python -m tools.hinge_inspect --watch 3         # confirm observe-mode detection on 3 swipes
"""
from __future__ import annotations

import argparse
from pathlib import Path

from operation_love import config as cfg_mod
from operation_love.drivers import hinge
from operation_love.drivers.hinge import HingeDriver

_BUTTON_TEMPLATES = {"like": "hinge_heart.png", "pass": "hinge_pass_x.png"}
_SHEET_TEMPLATES = {
    "send_like sheet": "hinge_send_like.png",
    "rose-upsell modal": "hinge_send_like_anyway.png",
}


def _check_templates_load() -> bool:
    print("\n--- Glyph templates (shipped with the package) ---")
    ok = True
    for label, name in {**_BUTTON_TEMPLATES, **_SHEET_TEMPLATES}.items():
        found = hinge._load_template(name) is not None
        ok = ok and found
        print(f"  [{'OK  ' if found else 'MISS'}] {name:28} ({label})")
    if not ok:
        print("  MISS means cv2 isn't installed (`pip install -e '.[hinge]'`) or the "
              "asset is missing from operation_love/drivers/assets/.")
    return ok


def _annotate(frame: bytes, point: tuple[int, int], out_path: Path) -> None:
    """Save the screencap with a crosshair at `point`, so accuracy can be eyeballed
    instead of trusted blind. Best-effort: a broken decode just skips the save."""
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        cv2.drawMarker(img, (int(point[0]), int(point[1])), (0, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=40, thickness=3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
    except Exception as exc:  # noqa: BLE001 — annotation is a bonus, never fatal
        print(f"  (could not save annotated screenshot: {exc})")


def _probe_buttons(driver: HingeDriver, out_dir: Path) -> None:
    print("\n--- Button probe (vision-hit vs fallback-coord on the current screen) ---")
    w, h = driver.adb.screen_size()
    for which, name in _BUTTON_TEMPLATES.items():
        frame = driver.adb.screencap()
        hits = hinge._match_glyph(frame, hinge._load_template(name),
                                  side="right" if which == "like" else "left")
        if hits:
            pt = hits[0]
            print(f"  [HIT     ] {which:5} glyph at pixel {pt} "
                  f"({pt[0] / w:.3f}, {pt[1] / h:.3f} of screen)")
        else:
            frac = driver.coords["like_heart" if which == "like" else "pass_x"]
            pt = (int(frac[0] * w), int(frac[1] * h))
            print(f"  [FALLBACK] {which:5} glyph not found by vision; using "
                  f"apps.hinge.coords {frac} -> pixel {pt}")
        shot = out_dir / f"hinge_inspect_{which}.png"
        _annotate(frame, pt, shot)
        print(f"             annotated screenshot: {shot}")


def _probe_sheet_templates(driver: HingeDriver, out_dir: Path) -> None:
    print("\n--- Sheet/modal probe (only visible during a like — optional) ---")
    ans = input("  Trigger a like by hand so the 'Send Like' sheet is showing, "
               "then press ENTER (or 's' + ENTER to skip) ➔ ")
    if ans.strip().lower() == "s":
        print("  Skipped.")
        return
    frame = driver.adb.screencap()
    for label, name in _SHEET_TEMPLATES.items():
        hits = hinge._match_glyph(frame, hinge._load_template(name), side="any", threshold=0.6)
        if hits:
            print(f"  [HIT ] {label:18} at pixel {hits[0]}")
            _annotate(frame, hits[0], out_dir / f"hinge_inspect_{name}")
        else:
            print(f"  [MISS] {label:18} not found — expected if that screen isn't showing")


def _watch(driver: HingeDriver, rounds: int) -> None:
    print(f"\n--- Observe detection: manually like/pass {rounds} cards ---")
    for i in range(1, rounds + 1):
        liked = driver.wait_for_decision(timeout=120.0)
        if liked is None:
            print(f"  {i}. No decision (timeout / deck empty) — stopping watch.")
            return
        print(f"  {i}. Detected: {'LIKE  👍' if liked else 'PASS  👎'}")
    print("  Observe detection works ✔  (this is exactly what mode:observe records)")


def _paste_block(driver: HingeDriver) -> None:
    print("\n--- Paste-ready (drop into config.yaml only if you changed any) ---")
    print("apps:\n  hinge:\n    coords:")
    for name, frac in driver.coords.items():
        print(f"      {name}: [{frac[0]}, {frac[1]}]")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Confirm Hinge's vision-located buttons + observe detection.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--watch", type=int, default=3,
                    help="how many manual swipes to confirm detection (0 to skip)")
    ap.add_argument("--out-dir", default="./data/hinge_inspect",
                    help="where to save annotated screenshots")
    args = ap.parse_args()

    cfg = cfg_mod.load(args.config)
    driver = HingeDriver(cfg)
    out_dir = Path(args.out_dir)
    print("Connecting to your Hinge phone over ADB…")
    driver.open_session()
    try:
        if not _check_templates_load():
            print("\nCan't vision-locate anything without cv2 + the templates — fix that first.")
            return
        input("\nPress ENTER once a profile card is on screen ➔ ")
        _probe_buttons(driver, out_dir)
        _probe_sheet_templates(driver, out_dir)
        if args.watch > 0:
            _watch(driver, args.watch)
        _paste_block(driver)
        print("\nDone. If everything is OK, set mode:observe and run "
              "`python -m operation_love` to start seeding your taste.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
