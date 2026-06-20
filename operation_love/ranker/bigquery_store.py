"""BigQuery backend — system-of-record + analytics, with batched writes.

Design (per the chosen 'BQ of-record + memory cache' approach):
- load_labels() runs ONE query at startup; the orchestrator holds labels in
  memory for fast inference.
- writes are buffered and flushed in batches (default every 25 rows, and on
  close), so the swipe loop never waits on a per-row cloud round-trip and we
  never read-back a just-streamed row (avoids BigQuery's streaming-buffer
  consistency gap).

At personal scale this stays within BigQuery's free tier. The ``client`` is
injectable so tests run without the google-cloud-bigquery package or network.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone

from ..costing import Usage

_TABLES = {
    "labels": (
        "run_id STRING, app STRING, created_at TIMESTAMP, liked BOOL, source STRING, "
        "embedding ARRAY<FLOAT64>, bio STRING, prompts STRING, photo_count INT64"
    ),
    "decisions": "run_id STRING, app STRING, created_at TIMESTAMP, decision STRING, score FLOAT64",
    "openers": "run_id STRING, app STRING, created_at TIMESTAMP, model STRING, opener STRING, referenced STRING",
    "spend": (
        "run_id STRING, created_at TIMESTAMP, model STRING, input_tokens INT64, output_tokens INT64, "
        "cache_read_tokens INT64, cache_write_tokens INT64, cost_usd FLOAT64"
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BigQueryStore:
    def __init__(self, project_id: str, dataset: str = "operation_love",
                 location: str = "US", flush_every: int = 25, client=None, ensure: bool = True):
        if not project_id:
            raise ValueError("storage.bigquery.project_id is required for the BigQuery backend")
        self.project_id = project_id
        self.dataset = dataset
        self.location = location
        self.flush_every = max(1, int(flush_every))
        if client is None:
            from google.cloud import bigquery  # lazy: only needed for real use
            client = bigquery.Client(project=project_id)
        self.client = client
        self._buf: dict[str, list[dict]] = {name: [] for name in _TABLES}
        self._label_count = 0
        self._lock = threading.RLock()  # shared across worker threads
        if ensure:
            self._ensure_tables()

    # --- setup ----------------------------------------------------------
    def _tid(self, name: str) -> str:
        return f"{self.project_id}.{self.dataset}.{name}"

    def _ensure_tables(self) -> None:
        self.client.query(
            f"CREATE SCHEMA IF NOT EXISTS `{self.project_id}.{self.dataset}` "
            f"OPTIONS(location='{self.location}')"
        ).result()
        for name, cols in _TABLES.items():
            self.client.query(f"CREATE TABLE IF NOT EXISTS `{self._tid(name)}` ({cols})").result()

    # --- reads ----------------------------------------------------------
    def load_labels(self) -> list[tuple[bool, list[float]]]:
        rows = self.client.query(f"SELECT liked, embedding FROM `{self._tid('labels')}`").result()
        out = [(bool(r["liked"]), list(r["embedding"])) for r in rows]
        with self._lock:
            self._label_count = len(out)
        return out

    def label_count(self) -> int:
        with self._lock:
            return self._label_count

    def count_today(self, app: str) -> int:
        safe = app.replace("'", "").replace("\\", "")
        rows = self.client.query(
            f"SELECT COUNT(*) AS c FROM `{self._tid('decisions')}` "
            f"WHERE app='{safe}' AND created_at >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), DAY)"
        ).result()
        for r in rows:
            return int(r["c"])
        return 0

    # --- writes (buffered, thread-safe) --------------------------------
    def add_label(self, run_id, app, liked, embedding, source="manual", bio="", prompts="", photo_count=0):
        with self._lock:
            self._buf["labels"].append({
                "run_id": run_id, "app": app, "created_at": _now(), "liked": bool(liked),
                "source": source, "embedding": [float(x) for x in embedding],
                "bio": bio, "prompts": prompts, "photo_count": int(photo_count),
            })
            self._label_count += 1
            self._maybe_flush("labels")

    def record_decision(self, run_id, app, decision, score):
        with self._lock:
            self._buf["decisions"].append({
                "run_id": run_id, "app": app, "created_at": _now(),
                "decision": decision, "score": float(score),
            })
            self._maybe_flush("decisions")

    def record_opener(self, run_id, app, model, opener, referenced):
        with self._lock:
            self._buf["openers"].append({
                "run_id": run_id, "app": app, "created_at": _now(),
                "model": model, "opener": opener, "referenced": referenced,
            })
            self._maybe_flush("openers")

    def record_spend(self, run_id, model, usage: Usage, cost):
        with self._lock:
            self._buf["spend"].append({
                "run_id": run_id, "created_at": _now(), "model": model,
                "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_input_tokens,
                "cache_write_tokens": usage.cache_creation_input_tokens, "cost_usd": float(cost),
            })
            self._maybe_flush("spend")

    # --- flush ----------------------------------------------------------
    def _maybe_flush(self, table: str) -> None:  # caller holds self._lock
        if len(self._buf[table]) >= self.flush_every:
            self._flush_table(table)

    def _flush_table(self, table: str) -> None:  # caller holds self._lock
        rows = self._buf[table]
        if not rows:
            return
        errors = self.client.insert_rows_json(self._tid(table), rows)
        if errors:
            raise RuntimeError(f"BigQuery insert errors for {table}: {errors}")
        self._buf[table] = []

    def flush(self) -> None:
        with self._lock:
            for table in self._buf:
                self._flush_table(table)

    def close(self) -> None:
        self.flush()
