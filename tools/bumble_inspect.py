"""Bumble live bring-up inspector — confirm selectors + observe detection.

Run ONCE on the machine that has your Bumble login. It opens the real Bumble web
app in a backgrounded window, waits for you to reach a profile card, then:

  1. probes every selector and reports how many elements each matched,
  2. watches your manual like/pass and prints what it detected — the exact
     signal observe mode uses (see BumbleDriver.wait_for_decision),
  3. prints a paste-ready `apps.bumble.selectors` block.

It never swipes for you; you stay in control the whole time. If a selector shows
MISS, open DevTools, find the right CSS, drop it into config.yaml under
apps.bumble.selectors, and re-run.

    python -m tools.bumble_inspect                 # uses config.yaml
    python -m tools.bumble_inspect --config x.yaml
    python -m tools.bumble_inspect --watch 5       # confirm detection on 5 swipes
"""
from __future__ import annotations

import argparse

from operation_love import config as cfg_mod
from operation_love.drivers.bumble import BumbleDriver

_CARD_SELECTORS = {"photo", "like", "pass"}     # expected to match while a card is shown


def _probe(driver: BumbleDriver) -> None:
    page = driver.page
    print("\n--- selector probe (matches on the current screen) ---")
    for name, sel in driver.selectors.items():
        try:
            n = len(page.query_selector_all(sel))
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERR ] {name:6} {sel!r}: {exc}")
            continue
        # photo/like/pass should hit while a card is up; empty should hit only when out.
        good = (n > 0) if name in _CARD_SELECTORS else True
        flag = "OK  " if (n > 0 and good) else ("MISS" if name in _CARD_SELECTORS else "—   ")
        sample = ""
        if n and name in ("bio", "prompt"):
            el = page.query_selector(sel)
            sample = "  | " + (el.inner_text() or "").strip().replace("\n", " ")[:60]
        print(f"  [{flag}] {name:6} x{n:<3} {sel}{sample}")
    print("  (photo/like/pass should be OK with a card shown; `empty` matches only "
          "when the deck is out.)")


def _watch(driver: BumbleDriver, rounds: int) -> None:
    print(f"\n--- observe detection: manually like/pass {rounds} cards ---")
    for i in range(1, rounds + 1):
        liked = driver.wait_for_decision(timeout=120.0)
        if liked is None:
            print(f"  {i}. no decision (timeout / deck empty) — stopping watch.")
            return
        print(f"  {i}. detected: {'LIKE  👍' if liked else 'PASS  👎'}")
    print("  observe detection works ✔  (this is exactly what mode:observe records)")


def _paste_block(driver: BumbleDriver) -> None:
    print("\n--- paste-ready (drop into config.yaml only if you changed any) ---")
    print("apps:\n  bumble:\n    selectors:")
    for name, sel in driver.selectors.items():
        print(f"      {name}: '{sel}'")


def main() -> None:
    ap = argparse.ArgumentParser(description="Confirm Bumble selectors + observe detection.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--watch", type=int, default=3,
                    help="how many manual swipes to confirm detection (0 to skip)")
    args = ap.parse_args()

    cfg = cfg_mod.load(args.config)
    driver = BumbleDriver(cfg)
    driver.headless = False                      # always headful for inspection
    print("Opening Bumble… sign in if needed, then navigate to a profile card (Encounters).")
    driver.open_session()
    try:
        input("\nPress ENTER once a profile card is on screen ➜ ")
        _probe(driver)
        if args.watch > 0:
            _watch(driver, args.watch)
        _paste_block(driver)
        print("\nDone. If everything is OK, set mode:observe and run "
              "`python -m operation_love` to start seeding your taste.")
    finally:
        input("\nPress ENTER to close the browser ➜ ")
        driver.close()


if __name__ == "__main__":
    main()
