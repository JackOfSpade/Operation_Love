"""Storage interface + factory — picks the backend from config."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..costing import Usage


@runtime_checkable
class Store(Protocol):
    """Shared storage surface both SQLiteStore and BigQueryStore implement."""

    def load_labels(self) -> list[tuple[bool, list[float]]]: ...
    # capture_truncated: the bot hit its per-profile screencap ceiling without ever reaching
    # the profile's true bottom, i.e. this label was made from an INCOMPLETE read. Carried
    # all the way to the store (rather than left in the local, rotating debug log) so the
    # system of record can answer "which labels came from a partial read?" later — the exact
    # query you want if truncated-read labels turn out to be noisier than complete ones.
    def record_profile(self, run_id: str, app: str, profile_id: str, liked: bool,
                       source: str = "manual", photos: list[bytes] | None = None,
                       photo_count: int = 0, capture_truncated: bool = False,
                       progress=None) -> bool: ...
    def add_label(self, run_id: str, app: str, liked: bool, embedding: list[float],
                  source: str = "manual", photo_count: int = 0,
                  profile_id: str = "", profile_name: str = "") -> None: ...
    def remove_latest_training_label(self) -> dict | None: ...
    def record_decision(self, run_id: str, app: str, decision: str, score: float,
                        source: str = "auto", profile_id: str = "",
                        created_at: object | None = None) -> None: ...
    def retraction_run_rows(self, run_id: str, app: str, source: str) -> dict: ...
    def append_label_retraction(self, row: dict) -> bool: ...
    def advisory_opener_run_rows(self, run_id: str, app: str) -> dict: ...
    def append_opener_retraction(self, row: dict) -> bool: ...
    # `angle` is the model's own free-text words for what the opener is DOING (guess / know /
    # imagine / tease / connect and anything else it invents) — telemetry only, it never
    # constrains generation. Deliberately not an enum: the opener design's move list is
    # explicitly non-binding, and a closed set would force the shoehorning it's meant to avoid.
    # It exists so we can eventually ask which opener shapes correlate with matches, a question
    # that is unaskable today. Trailing with a "" default so every existing positional caller
    # (and every test double implementing this Protocol) keeps working unchanged.
    #
    # `item_description` is the model's own short description of the ITEM it picked to write
    # about and to like (ops/OPENER-REDESIGN.md 5.7), recorded in auto and observe alike. Not
    # a second copy of `referenced`: that is the DETAIL the opener reacts to, this is what the
    # item IS (a photo, a written prompt), which is the coarse class doc 5.8's pre-flight
    # cross-check compares against our own crop. Telemetry here too -- nothing reads it back at
    # runtime -- and trailing with a "" default for the same compatibility reason as `angle`.
    def record_opener(self, run_id: str, app: str, model: str, opener: str, referenced: str,
                      angle: str = "", item_description: str = "", *, profile_id: str = "",
                      decision: str = "", decision_source: str = "",
                      decision_created_at: object | None = None,
                      model_item_index: int | None = None) -> None: ...
    # Durable record of a REJECTED opener attempt (OpenerParseError -- see opener/service.py's
    # OpenerParseError handling), captured for every attempt including the final one that
    # exhausts a profile's retries. Without this, only SUCCEEDED openers were ever recorded
    # (record_opener above), so there was no way to ask "how often does each guard fire" or
    # "is the scaffolding/sentence-cap detector too strict" -- the rejected text and reason
    # were printed and then lost forever. reason_code/raw_opener may be None (see
    # OpenerParseError's own docstring for exactly when each is None vs populated).
    def record_opener_rejection(self, run_id: str, app: str, model: str, attempt: int,
                                reason_code: str | None, reason: str,
                                raw_opener: str | None) -> None: ...
    def record_spend(self, run_id: str, model: str, usage: Usage, cost: float | None) -> None: ...
    def count_today(self, app: str, *, source: str = "auto") -> int: ...
    def spend_today(self) -> float: ...
    def observe_release_persistence_summary(self, run_id: str, app: str) -> dict[str, int]:
        """Return only aggregate, run-scoped manual pass/like and successful-opener counts.

        The Hinge AUTO release verifier uses these exact fields to prove a whole supervised
        manual cycle persisted, without ever exporting profile, embedding, or opener content.
        """
        ...
    def ai_observe_release_persistence_summary(self, run_id: str, app: str,
                                               source: str) -> dict[str, int]:
        """Return run-scoped PASS/LIKE persistence for an explicitly non-manual source.

        This deliberately has a different method and field names from the manual release
        summary so callers cannot relabel manual rows as autonomous evidence by accident.
        """
        ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


def make_store(cfg, ensure: bool = True) -> Store:
    """Build the configured store. ``ensure=False`` skips BigQuery table/bucket setup.

    Evaluation uses that for pure reads, while correction tools may still append non-photo
    tombstones. Profile-photo upload always refuses on an unverified ``ensure=False`` instance.
    """
    s = cfg.storage
    if s.backend == "bigquery":
        from .bigquery_store import DEFAULT_FLUSH_EVERY, BigQueryStore
        bq = s.bigquery
        return BigQueryStore(
            project_id=bq.get("project_id", ""),
            dataset=bq.get("dataset", "operation_love"),
            location=bq.get("location", "US"),
            photo_bucket=bq.get("photo_bucket", ""),
            flush_every=bq.get("flush_every", DEFAULT_FLUSH_EVERY),
            ensure=ensure,
        )
    from .store import SQLiteStore
    return SQLiteStore(cfg.db_file)
