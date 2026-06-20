"""Main loop — replaces main.ahk. Skeleton wired to the budget guard.

Per profile: capture -> quality filter -> embed -> rank -> like/dislike ->
(if like) write opener with Claude under the budget cap -> record everything.
Vision/ranker/driver internals land in later phases; the control flow and the
cost-control behaviour the user asked about are implemented here.
"""
from __future__ import annotations

import random
import time
import uuid

from . import config as cfg_mod
from .costing import CostTracker, is_out_of_credit
from .opener.opener import AnthropicOpener
from .ranker.store import Store
from .runtime import Capabilities


def run(config_path: str = "config.yaml") -> None:
    cfg = cfg_mod.load(config_path)
    store = Store(cfg.db_file)
    run_id = uuid.uuid4().hex[:12]

    # Inspect this machine and auto-adjust to what's installed (any OS, GPU or not).
    caps = Capabilities.detect()
    print(caps.banner())

    opener_enabled = cfg.opener.enabled
    if opener_enabled and caps.missing("anthropic"):
        print("[degrade] anthropic SDK not installed -> openers disabled")
        opener_enabled = False
    if cfg.quality_filter.enabled and caps.missing("quality"):
        print("[degrade] pyiqa not installed -> quality pre-filter skipped")

    tracker = CostTracker(cfg.budget.pricing, cfg.budget.run_budget_usd)
    opener_client = AnthropicOpener(cfg.opener.model, cfg.opener.max_tokens) if opener_enabled else None
    openers_disabled = False  # flips on budget reached / out of credit

    driver = _make_driver(cfg)          # Phase 1/5
    driver.open_session()

    try:
        while True:
            profile = driver.next_profile()
            if profile is None or driver.out_of_profiles():
                break

            # decision = rank(profile)   # Phase 3; manual swipe until enough labels
            decision, score = _decide(cfg, store, profile)

            if decision != "like":
                driver.dislike()
                _pace(cfg)
                continue

            opener_text = None
            if opener_client and not openers_disabled:
                if tracker.budget_reached():
                    openers_disabled = True
                    _on_exhausted(cfg, reason="run budget reached")
                else:
                    try:
                        result = opener_client.generate(profile, cfg.opener.style)
                        cost = tracker.record(result.model, result.usage)
                        store.record_spend(run_id, result.model, result.usage, cost)
                        opener_text = result.opener
                    except Exception as e:  # noqa: BLE001
                        if is_out_of_credit(e):
                            openers_disabled = True
                            _on_exhausted(cfg, reason="Claude credit exhausted")
                        else:
                            raise

            if openers_disabled and cfg.budget.on_exhausted == "stop":
                break

            driver.like(opener_text)
            _pace(cfg)
    finally:
        driver.close()
        print(
            f"[run {run_id}] openers={tracker.calls} "
            f"spend=${tracker.run_spend_usd:.4f} lifetime=${store.lifetime_spend():.4f}"
        )
        store.close()


def _on_exhausted(cfg, reason: str) -> None:
    action = "stopping" if cfg.budget.on_exhausted == "stop" else "swiping without openers"
    print(f"[budget] {reason} -> {action}")


def _pace(cfg) -> None:
    time.sleep(random.uniform(cfg.pacing.min_delay_s, cfg.pacing.max_delay_s))


# --- placeholders filled in later phases ---------------------------------
def _make_driver(cfg):
    raise NotImplementedError("Driver wired in Phase 1 (Bumble) / Phase 5 (Hinge)")


def _decide(cfg, store, profile):
    raise NotImplementedError("Ranker wired in Phase 3; returns (decision, score)")


if __name__ == "__main__":
    run()
