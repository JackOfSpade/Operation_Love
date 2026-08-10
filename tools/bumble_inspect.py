"""Bumble live bring-up inspector — confirm selectors + observe detection.

Run ONCE on the machine that has your Bumble login. It opens the real Bumble web
app in a backgrounded window, waits for you to reach a profile card, then:

  1. probes every selector and reports how many elements each matched,
  2. watches your manual like/pass and prints what it detected — the exact
     signal observe mode uses (see BumbleDriver.wait_for_decision),
  3. prints a paste-ready `apps.bumble_web.selectors` block.

It never swipes for you; you stay in control the whole time. If a selector shows
MISS, open DevTools, find the right CSS, drop it into config.yaml under
apps.bumble_web.selectors, and re-run.

    python -m tools.bumble_inspect                 # uses config.yaml
    python -m tools.bumble_inspect --config x.yaml
    python -m tools.bumble_inspect --watch 5       # confirm detection on 5 swipes
"""
from __future__ import annotations

import argparse
import sys

from operation_love import config as cfg_mod
from operation_love.drivers.bumble import BumbleDriver
from operation_love.drivers.web import PlatformUnavailable

_CARD_SELECTORS = {"photo", "like", "pass"}     # expected to match while a card is shown


def _probe(driver: BumbleDriver) -> bool:
    page = driver.page
    any_miss = False
    print("\n--- Selector probe (matches on the current screen) ---")
    for name, sel in driver.selectors.items():
        try:
            n = len(page.query_selector_all(sel))
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERR ] {name:6} {sel!r}: {exc}")
            continue
        # photo/like/pass should hit while a card is up; empty should hit only when out.
        good = (n > 0) if name in _CARD_SELECTORS else True
        flag = "OK  " if (n > 0 and good) else ("MISS" if name in _CARD_SELECTORS else "—   ")
        any_miss = any_miss or flag.strip() == "MISS"
        sample = ""
        if n and name in ("bio", "prompt"):
            el = page.query_selector(sel)
            sample = "  | " + (el.inner_text() or "").strip().replace("\n", " ")[:60]
        print(f"  [{flag}] {name:6} x{n:<3} {sel}{sample}")
    print("  (Photo/like/pass should be OK with a card shown; `empty` matches only "
          "when the deck is out.)")
    return any_miss


def _discover(driver: BumbleDriver) -> None:
    """When a selector MISSes, dump raw DOM signals so the right CSS can be
    derived without DevTools: Bumble's [data-qa-role] inventory (the stable
    hooks — that's how like/pass match) and the actual large media elements."""
    page = driver.page
    print("\n--- Discovery: raw DOM signals to derive the MISSing selectors ---")
    roles = page.evaluate(r"""() => {
        const map = {};
        document.querySelectorAll('[data-qa-role]').forEach(e => {
            const r = e.getAttribute('data-qa-role');
            const txt = (e.innerText || '').trim().replace(/\s+/g, ' ');
            if (!map[r]) map[r] = {count: 0, sample: ''};
            map[r].count++;
            if (!map[r].sample && txt) map[r].sample = txt.slice(0, 70);
        });
        return map;
    }""")
    print("  Data-qa-role inventory ([data-qa-role], role  xCount  | sample text):")
    for r in sorted(roles):
        info = roles[r]
        s = ("  | " + info["sample"]) if info["sample"] else ""
        print(f"      {r:44} x{info['count']}{s}")

    media = page.evaluate(r"""() => {
        const out = [], seen = new Set();
        const push = (o) => { const k = o.tag + '|' + o.cls; if (!seen.has(k)) { seen.add(k); out.push(o); } };
        document.querySelectorAll('img').forEach(e => {
            const r = e.getBoundingClientRect();
            if (r.width * r.height > 8000)
                push({tag: 'img', cls: (typeof e.className === 'string' ? e.className : ''),
                      w: Math.round(r.width), h: Math.round(r.height), ref: (e.src || '').slice(0, 50)});
        });
        document.querySelectorAll('div,span,section,figure,picture,a').forEach(e => {
            const bg = getComputedStyle(e).backgroundImage;
            if (bg && bg.indexOf('url(') > -1) {
                const r = e.getBoundingClientRect();
                if (r.width * r.height > 8000)
                    push({tag: e.tagName.toLowerCase(), cls: (typeof e.className === 'string' ? e.className : ''),
                          w: Math.round(r.width), h: Math.round(r.height), ref: 'bg-image'});
            }
        });
        return out.slice(0, 20);
    }""")
    print("\n  Large media elements (likely profile photos — tag.class  WxH  src):")
    for m in media:
        cls = ("." + ".".join(m["cls"].split())) if m["cls"] else ""
        print(f"      {m['tag']}{cls}  {m['w']}x{m['h']}  {m['ref']}")
    print("\n  ➜ Paste this whole discovery block back; I'll derive the photo/bio/prompt selectors.")


def _watch(driver: BumbleDriver, rounds: int) -> None:
    print(f"\n--- Observe detection: manually like/pass {rounds} cards ---")
    for i in range(1, rounds + 1):
        liked = driver.wait_for_decision(timeout=120.0)
        if liked is None:
            print(f"  {i}. No decision (timeout / deck empty) — stopping watch.")
            return
        print(f"  {i}. Detected: {'LIKE  👍' if liked else 'PASS  👎'}")
    print("  Observe detection works ✔  (this is exactly what mode:observe records)")


def _paste_block(driver: BumbleDriver) -> None:
    print("\n--- Paste-ready (drop into config.yaml only if you changed any) ---")
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
    try:
        driver.open_session()
    except PlatformUnavailable as exc:
        # Bumble's web app is permanently gone (discontinued Aug 2026): the registry
        # (operation_love/platforms.py) marks "bumble_web" unavailable, and
        # PlaywrightDriver.open_session() correctly refuses to start for it -- that
        # availability gate is not bypassed. This tool is a dead-legacy inspector kept
        # only as a reference (see the module docstring); without this handler, that
        # correct refusal surfaced as a raw traceback instead of the clean explanation
        # the registry already computed. No browser was ever opened, so there's nothing
        # to close -- just report why and exit non-zero.
        print(f"\nCan't run: {exc}")
        sys.exit(1)
    try:
        input("\nPress ENTER once a profile card is on screen ➜ ")
        if _probe(driver):                 # a card selector MISSed -> show raw DOM signals
            _discover(driver)
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
