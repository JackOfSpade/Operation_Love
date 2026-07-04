"""Storage interface + factory — picks the backend from config."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..costing import Usage


@runtime_checkable
class Store(Protocol):
    """Shared storage surface both SQLiteStore and BigQueryStore implement."""

    def load_labels(self) -> list[tuple[bool, list[float]]]: ...
    def load_labels_ordered(self) -> list[tuple[bool, list[float]]]: ...   # chronological (created_at)
    def record_profile(self, run_id: str, app: str, profile_id: str, liked: bool,
                       source: str = "manual", photos: list[bytes] | None = None,
                       photo_count: int = 0) -> bool: ...
    def add_label(self, run_id: str, app: str, liked: bool, embedding: list[float],
                  source: str = "manual", photo_count: int = 0,
                  profile_id: str = "") -> None: ...
    def record_decision(self, run_id: str, app: str, decision: str, score: float,
                        source: str = "auto") -> None: ...
    def record_opener(self, run_id: str, app: str, model: str, opener: str, referenced: str) -> None: ...
    def record_spend(self, run_id: str, model: str, usage: Usage, cost: float | None) -> None: ...
    def label_count(self) -> int: ...
    def count_today(self, app: str) -> int: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


def make_store(cfg, ensure: bool = True) -> Store:
    """Build the configured store. ensure=False skips BigQuery table/bucket setup —
    use it for read-only paths (e.g. evaluation) so we don't run DDL just to read."""
    s = cfg.storage
    if s.backend == "bigquery":
        from .bigquery_store import DEFAULT_FLUSH_EVERY, BigQueryStore
        bq = s.bigquery
        return BigQueryStore(
            project_id=bq.get("project_id", ""),
            dataset=bq.get("dataset", "operation_love"),
            location=bq.get("location", "US"),
            photo_bucket=bq.get("photo_bucket", ""),
            flush_every=int(bq.get("flush_every", DEFAULT_FLUSH_EVERY)),
            ensure=ensure,
        )
    from .store import SQLiteStore
    return SQLiteStore(cfg.db_file)
