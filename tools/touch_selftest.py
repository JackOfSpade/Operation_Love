"""Touch-watcher self-test — prove the phone's own touch stream reaches this machine.

Observe mode corroborates every recorded PASS against the human's OWN gestures, read
read-only off the phone's touchscreen via `adb shell getevent` (operation_love/drivers/
touchwatch.py). That signal has one failure mode that is completely silent from the host
side: if the stream delivers nothing, "the human never touched anything" and "the pipe is
broken" look identical. The driver already self-disables and warns when it sees zero
events for a whole run, but the honest way to know is to watch your own finger show up.

So: run this, touch the screen, and see the taps and drags print.

    python -m tools.touch_selftest                 # uses config.yaml
    python -m tools.touch_selftest --seconds 40
    python -m tools.touch_selftest --config x.yaml

This NEVER taps, swipes, types, or installs anything — it only reads. (It could not
cheat even if it wanted to: `adb shell input tap` injects above the kernel, so injected
taps never appear on /dev/input at all. Only a real finger on the glass shows up here,
which is exactly the property that makes this signal worth having.)

What to look for:
  * "selected device" naming your real touchscreen (goodix_ts0 on a Pixel 7a), NOT a
    device whose name starts with og_touch_ (that would be our own virtual digitizer).
  * TAP lines when you tap, DRAG lines when you scroll. If a deliberate scroll prints as
    TAP, the tap-slop threshold needs re-measuring on this screen.
  * coordinates that match where you actually touched — the reported (x, y) is in screen
    pixels, origin top-left.
If nothing prints at all while you are touching, the corroboration signal is dead on this
device and observe mode is running on the identity anchor alone; say so in a bug report.
"""
from __future__ import annotations

import argparse
import time

from operation_love import config as cfg_mod
from operation_love.drivers.hinge import HINGE_SPEC
from operation_love.drivers.touchwatch import TouchWatcher, TouchWatchUnavailable


def _unparsed_stream_verdict_lines(watcher) -> list[str]:
    """The "lines arrived, none parsed" verdict — printed with the evidence already in hand.

    The watcher keeps a redacted, capped sample of the exact lines this parser rejected
    (TouchWatcher.unparsed_line_samples: every numeric/hex token replaced with `<n>`, so no
    touch coordinate can ride out in it). Printing those means an unsupported Android toolbox
    spelling can be fixed straight from a bug report, instead of asking the operator to
    reproduce the failure in a terminal and transcribe raw lines that DO carry coordinates.
    """
    head = (f"VERDICT: {watcher.raw_line_count} line(s) arrived but NONE parsed — the stream "
            "works and this parser does not understand its format. That IS a bug here: "
            "compare the spelling against _LT_LINE_RE in "
            "operation_love/drivers/touchwatch.py.")
    samples = watcher.unparsed_line_samples
    if not samples:
        # A line whose every token redacts to nothing is dropped rather than stored, so the
        # deque can still be empty here. Only then is the manual round-trip worth asking for.
        return [head, "  No sample was retained. Capture one by hand with "
                      "`adb shell getevent -lt <device>`."]
    return [head,
            f"  {len(samples)} rejected line(s) below, redacted (every number replaced with "
            "<n>) and safe to paste into an incident report as-is:"
            ] + [f"    {sample}" for sample in samples]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seconds", type=float, default=30.0, help="how long to watch (default 30)")
    ap.add_argument("--app", default=HINGE_SPEC.app, help="which apps.<app> block to read")
    args = ap.parse_args()

    cfg = cfg_mod.load(args.config)
    app_cfg = (getattr(cfg, "apps", {}) or {}).get(args.app, {})
    adb_path = app_cfg.get("adb_path", "adb")
    serial = app_cfg.get("serial") or None

    # Screen geometry: ask the device rather than assuming, so the printed pixel
    # coordinates mean the same thing they will mean inside the driver.
    from operation_love.drivers.adb import Adb
    adb = Adb(serial, adb_path=adb_path)
    if not adb.devices():
        print("No ADB device connected. Plug the phone in and enable USB debugging.")
        return 2
    size = adb.screen_size()

    watcher = TouchWatcher(adb_path, serial, size)
    try:
        watcher.start()
    except TouchWatchUnavailable as exc:
        print(f"Touch watcher unavailable: {exc}")
        return 2

    slop = max(12.0, min(0.02 * size[1], 90.0))     # mirrors AndroidDriver._observe_tap_slop_px
    print(f"screen         : {size[0]}x{size[1]}")
    print(f"selected device: {watcher.device_path} {watcher.device_name!r}")
    print(f"tap slop       : {slop:.0f}px (travel at or below this counts as a tap)")
    print()
    print(f">>> Touch the screen now — tap a few times, then scroll. Watching {args.seconds:.0f}s. <<<")
    print()

    start = time.monotonic()
    seen = 0
    try:
        while time.monotonic() - start < args.seconds:
            time.sleep(0.25)
            gestures = watcher.gestures_since(start)
            for g in gestures[seen:]:
                kind = "TAP " if g.is_tap(slop) else "DRAG"
                print(f"  {kind} down={g.down} up={g.up} "
                      f"travel={g.travel_px:7.1f}px  held={g.t_up - g.t_down:.3f}s")
            seen = len(gestures)
            if not watcher.alive:
                print("  !! the getevent stream died mid-watch")
                break
    finally:
        watcher.close()

    print()
    print(f"raw lines read   : {watcher.raw_line_count}")
    print(f"events parsed    : {watcher.event_count}")
    print(f"gestures         : {seen}")
    if watcher.raw_line_count == 0:
        print("VERDICT: the device sent NOTHING. `getevent` opens the touchscreen (it can read "
              "its capabilities) but delivers no events to the adb shell user — Android "
              "withholds the input stream from unprivileged readers on this build. This is a "
              "platform restriction, not a bug in this code, and no amount of parsing fixes "
              "it. Observe mode runs on the identity anchor alone, which is what actually "
              "fixes the scroll-recorded-as-PASS bug; gesture corroboration is the optional "
              "extra layer. Keep apps.<app>.observe_touch_watch: false.")
        return 1
    if watcher.event_count == 0:
        for line in _unparsed_stream_verdict_lines(watcher):
            print(line)
        return 1
    if seen == 0:
        print("VERDICT: the stream is alive (raw events arrived) but no complete DOWN..UP "
              "gesture was assembled. If you were touching, the parser needs a look.")
        return 1
    print("VERDICT: gesture corroboration is working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
