"""Storage factory — picks the backend from config."""
from __future__ import annotations


def make_store(cfg, ensure: bool = True):
    """Build the configured store. ensure=False skips BigQuery table/bucket setup —
    use it for read-only paths (e.g. evaluation) so we don't run DDL just to read."""
    s = cfg.storage
    if s.backend == "bigquery":
        from .bigquery_store import BigQueryStore
        bq = s.bigquery
        return BigQueryStore(
            project_id=bq.get("project_id", ""),
            dataset=bq.get("dataset", "operation_love"),
            location=bq.get("location", "US"),
            photo_bucket=bq.get("photo_bucket", ""),
            flush_every=int(bq.get("flush_every", 25)),
            ensure=ensure,
        )
    from .store import SQLiteStore
    return SQLiteStore(cfg.db_file)
