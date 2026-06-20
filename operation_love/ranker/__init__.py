"""Storage factory — picks the backend from config."""
from __future__ import annotations


def make_store(cfg):
    s = cfg.storage
    if s.backend == "bigquery":
        from .bigquery_store import BigQueryStore
        bq = s.bigquery
        return BigQueryStore(
            project_id=bq.get("project_id", ""),
            dataset=bq.get("dataset", "operation_love"),
            location=bq.get("location", "US"),
            flush_every=int(bq.get("flush_every", 25)),
        )
    from .store import SQLiteStore
    return SQLiteStore(cfg.db_file)
