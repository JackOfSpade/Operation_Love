"""Shared, budget-aware opener generation across all app workers.

One OpenerService is shared by every worker (Bumble, Hinge, ...) so the per-run
spend cap is GLOBAL, not per-app. Thread-safe. Handles the two ways spending
ends: the configured run budget, and the actual Anthropic out-of-credit error —
either flips the service to disabled and (if on_exhausted="stop") asks the
supervisor to stop all workers.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from ..costing import CostTracker, is_out_of_credit
from ..perception.capture import Profile
from .opener import OpenerClient


@dataclass
class OpenerPick:
    """An opener plus which profile item (0-based index, capture order) it is about, so the
    driver can attach the comment to the RIGHT photo/prompt instead of always the first."""
    text: str
    index: int = 0


class OpenerService:
    def __init__(self, client: OpenerClient | None, tracker: CostTracker, store,
                 style: str, on_exhausted: str = "stop"):
        self.client = client
        self.tracker = tracker
        self.store = store
        self.style = style
        self.on_exhausted = on_exhausted
        self.disabled = client is None
        self.stop_requested = False     # set when budget/credit is exhausted and on_exhausted="stop"
        self._lock = threading.RLock()

    def maybe_opener(self, run_id: str, app: str, profile: Profile) -> "OpenerPick | None":
        """Return an OpenerPick (text + referenced item index), or None (disabled / budget
        reached / out of credit / over-budget).

        The budget check, provider call, and spend recording are serialized so
        concurrent app workers cannot all pass the pre-call budget check and
        overspend the shared run budget at once.
        """
        with self._lock:
            if self.disabled:
                return None
            if self.tracker.budget_reached():
                self._exhaust("run budget reached")
                return None
            try:
                result = self.client.generate(profile, self.style)
            except Exception as e:  # noqa: BLE001
                if is_out_of_credit(e):
                    self._exhaust("Claude credit exhausted")
                    return None
                raise

            try:
                cost = self.tracker.record(result.model, result.usage)
            except KeyError:
                # The API already ran (real credits spent) but its response echoed a model
                # string with no budget.pricing entry, so spend can't be accounted for.
                # Degrade the same way as budget-reached/out-of-credit rather than crash —
                # continuing to spend with no way to track it would silently break the
                # budget-enforcement contract the rest of this service is built around.
                # cost=None (not 0.0): the real cost was nonzero, just unrecoverable, and
                # a fabricated $0.00 would misreport actual spend in the stored record.
                self._exhaust(f"no budget.pricing entry for model '{result.model}'; "
                              "spend can no longer be tracked")
                cost = None
            self.store.record_spend(run_id, result.model, result.usage, cost)
            self.store.record_opener(run_id, app, result.model, result.opener, result.referenced)
            if self.tracker.budget_reached():
                self._exhaust("run budget reached")
            return OpenerPick(result.opener, getattr(result, "referenced_index", 0))

    def _exhaust(self, reason: str) -> None:
        with self._lock:
            if not self.disabled:
                action = "stopping all workers" if self.on_exhausted == "stop" else "swiping without openers"
                print(f"Budget: {reason} -> {action}")
            self.disabled = True
            if self.on_exhausted == "stop":
                self.stop_requested = True
