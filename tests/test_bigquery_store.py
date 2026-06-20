"""BigQueryStore tests with a fake client — no google SDK or network required."""
from operation_love.costing import Usage
from operation_love.ranker.bigquery_store import BigQueryStore


class _FakeJob:
    def __init__(self, rows=None):
        self._rows = rows or []

    def result(self):
        return self._rows


class _FakeBQ:
    """Minimal stand-in for google.cloud.bigquery.Client."""

    def __init__(self, label_rows=None, insert_errors=None):
        self.queries = []
        self.inserted: dict[str, list[dict]] = {}
        self._label_rows = label_rows or []
        self._insert_errors = insert_errors or []

    def query(self, sql):
        self.queries.append(sql)
        if sql.strip().upper().startswith("SELECT"):
            return _FakeJob(self._label_rows)
        return _FakeJob()  # DDL

    def insert_rows_json(self, table_id, rows):
        self.inserted.setdefault(table_id, []).extend(rows)
        return self._insert_errors


def _store(client, flush_every=25):
    return BigQueryStore("proj", "ds", flush_every=flush_every, client=client, ensure=False)


def test_load_labels_parses_and_counts():
    client = _FakeBQ(label_rows=[{"liked": True, "embedding": [0.1, 0.2]},
                                 {"liked": False, "embedding": [0.3]}])
    s = _store(client)
    labels = s.load_labels()
    assert labels == [(True, [0.1, 0.2]), (False, [0.3])]
    assert s.label_count() == 2


def test_buffer_flushes_at_threshold():
    client = _FakeBQ()
    s = _store(client, flush_every=2)
    s.add_label("r", "bumble", True, [0.1])
    assert "proj.ds.labels" not in client.inserted          # buffered, not yet sent
    s.add_label("r", "bumble", False, [0.2])
    assert len(client.inserted["proj.ds.labels"]) == 2       # flushed at threshold
    assert s.label_count() == 2


def test_flush_on_close():
    client = _FakeBQ()
    s = _store(client, flush_every=100)
    s.record_spend("r", "claude-opus-4-8", Usage(input_tokens=10, output_tokens=5), 0.0001)
    assert "proj.ds.spend" not in client.inserted
    s.close()
    assert len(client.inserted["proj.ds.spend"]) == 1
    row = client.inserted["proj.ds.spend"][0]
    assert row["input_tokens"] == 10 and row["cost_usd"] == 0.0001


def test_insert_errors_raise():
    client = _FakeBQ(insert_errors=[{"index": 0, "errors": ["bad row"]}])
    s = _store(client, flush_every=1)
    try:
        s.add_label("r", "bumble", True, [0.1])
    except RuntimeError as e:
        assert "BigQuery insert errors" in str(e)
    else:
        raise AssertionError("expected RuntimeError on insert errors")


def test_requires_project_id():
    try:
        BigQueryStore("", "ds", client=_FakeBQ(), ensure=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError without project_id")


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
