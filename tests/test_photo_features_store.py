"""The `photo_features` table on both stores (whole-photo feature set, contract section B).

`add_photo_features`, `load_label_feature_rows`, `photo_feature_coverage` and
`labels_missing_photo_features`, on SQLite and on BigQuery through a test double that EXECUTES
the store's SQL on SQLite (so the queries themselves are exercised, not just their callers).
Synthetic vectors only.
"""

import hashlib
import inspect
import json
import sqlite3
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import operation_love.ranker.bigquery_store as bq_module
from operation_love.ranker import Store
from operation_love.ranker.bigquery_store import _MIGRATIONS, _TABLES, BigQueryStore
from operation_love.ranker.store import (
    SQLiteStore,
    resolve_label_frames,
    validate_photo_features,
)
from operation_love.vision.photo_cards import (
    WHOLE_PHOTO_DIMS,
    WHOLE_PHOTO_FEATURE_VERSION,
)

V = WHOLE_PHOTO_FEATURE_VERSION
NEW_METHODS = (
    "add_photo_features",
    "load_label_feature_rows",
    "photo_feature_coverage",
    "labels_missing_photo_features",
)


def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def _vec(value: float) -> list[float]:
    return [value] * WHOLE_PHOTO_DIMS


def _features(**overrides) -> dict:
    kwargs = {
        "run_id": "r1",
        "app": "hinge",
        "profile_id": "p1",
        "feature_version": V,
        "embedding": _vec(0.5),
        "card_count": 3,
        "frames_sha256": _sha("frames"),
        "source": "live",
    }
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------------------------
# A BigQuery double that runs the store's SELECTs on SQLite.
# --------------------------------------------------------------------------------------------
class NotFound(Exception):  # the store classifies by type NAME, like the SDK's NotFound
    pass


def _columns(table: str) -> list[str]:
    return [c.strip().split()[0] for c in _TABLES[table].split(",")]


_TIMESTAMP_COLUMNS = {
    table: {
        c.strip().split()[0]
        for c in spec.split(",")
        if c.strip().split()[1].upper() == "TIMESTAMP"
    }
    for table, spec in _TABLES.items()
}


class _Job:
    def __init__(self, rows):
        self._rows = rows

    def result(self):
        return self._rows


class _SqlBQ:
    def __init__(self):
        self.con = sqlite3.connect(":memory:", check_same_thread=False)
        for table in _TABLES:
            self.con.execute(
                f'CREATE TABLE "proj.ds.{table}" ({", ".join(_columns(table))})'
            )
        self.queries: list[str] = []
        self.missing_tables: set[str] = set()
        # BigQuery promises no row order: reads whose SQL contains this come back reversed.
        self.reverse_reads_containing: str | None = None

    def query(self, sql, job_config=None):
        self.queries.append(sql)
        if "INFORMATION_SCHEMA.COLUMNS" in sql:
            return _Job(
                [
                    {"table_name": table, "column_name": col}
                    for table in _TABLES
                    for col in _columns(table)
                ]
            )
        for table in self.missing_tables:
            if f".{table}`" in sql:
                raise NotFound(f"Not found: Table proj:ds.{table}")
        assert sql.lstrip().upper().startswith("SELECT"), sql
        params = {
            p.name: p.value
            for p in (getattr(job_config, "query_parameters", None) or [])
        }
        cur = self.con.execute(sql, params)
        names = [d[0] for d in cur.description]
        rows = []
        for raw in cur.fetchall():
            row = dict(zip(names, raw, strict=True))
            for key, value in row.items():
                if key == "embedding" and isinstance(value, str):
                    row[key] = json.loads(value)
                elif key == "created_at" and isinstance(value, float):
                    row[key] = datetime.fromtimestamp(value, tz=timezone.utc)
                elif key == "liked" and value is not None:
                    row[key] = bool(value)
            rows.append(row)
        if self.reverse_reads_containing and self.reverse_reads_containing in sql:
            rows.reverse()
        return _Job(rows)

    def insert_rows_json(self, table_id, rows, row_ids=None):
        table = table_id.rsplit(".", 1)[1]
        for row in rows:
            self.insert(table, row)
        return []

    def insert(self, table, row):
        values = {}
        for key, value in row.items():
            if isinstance(value, list):
                value = json.dumps(value)
            elif isinstance(value, str) and key in _TIMESTAMP_COLUMNS[table]:
                value = datetime.fromisoformat(value).timestamp()
            values[key] = value
        cols = ", ".join(values)
        marks = ", ".join(f":{k}" for k in values)
        self.con.execute(
            f'INSERT INTO "proj.ds.{table}" ({cols}) VALUES ({marks})', values
        )

    def rows(self, table):
        cur = self.con.execute(f'SELECT * FROM "proj.ds.{table}" ORDER BY rowid')
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]


class _Bucket:
    def __init__(self):
        self.iam_configuration = SimpleNamespace(
            uniform_bucket_level_access_enabled=True,
            public_access_prevention="enforced",
        )

    def exists(self):
        return True

    def reload(self):
        pass


class _Storage:
    def bucket(self, _name):
        return _Bucket()


@pytest.fixture(autouse=True)
def _bigquery_sdk_stub(monkeypatch):
    """Bound queries need the SDK's two symbols; give the store a stand-in, not the package."""

    class _Param:
        def __init__(self, name, _kind, value):
            self.name, self.value = name, value

    class _JobConfig:
        def __init__(self, query_parameters):
            self.query_parameters = query_parameters

    monkeypatch.setitem(
        sys.modules,
        "google.cloud",
        SimpleNamespace(
            bigquery=SimpleNamespace(
                ScalarQueryParameter=_Param, QueryJobConfig=_JobConfig
            )
        ),
    )


def _bq_store(client, *, ensure=True, flush_every=100):
    return BigQueryStore(
        "proj",
        "ds",
        photo_bucket="photos",
        flush_every=flush_every,
        client=client,
        storage_client=_Storage(),
        ensure=ensure,
    )


class _Harness:
    def __init__(self, kind, tmp_path):
        self.kind = kind
        if kind == "sqlite":
            self.client = None
            self.store = SQLiteStore(tmp_path / "store.db")
        else:
            self.client = _SqlBQ()
            self.store = _bq_store(self.client)

    def label(self, liked, value, profile_id, *, source="manual", run_id="r1"):
        self.store.add_label(
            run_id, "hinge", liked, [value] * 4, source=source, profile_id=profile_id
        )


@pytest.fixture(params=["sqlite", "bigquery"])
def h(request, tmp_path):
    harness = _Harness(request.param, tmp_path)
    yield harness
    harness.store.close()


def _legacy_zero_card_row(h, profile_id):
    """A zero-card row as an earlier build stored it (no API writes one any more), written
    behind the API before the store's first feature read, dated 2026-01-01."""
    row = {
        "run_id": "r1",
        "app": "hinge",
        "profile_id": profile_id,
        "feature_version": V,
        "card_count": 0,
        "frames_sha256": _sha("zero"),
        "source": "live",
    }
    if h.kind == "sqlite":
        h.store.con.execute(
            "INSERT INTO photo_features (run_id, app, profile_id, created_at,"
            " feature_version, embedding, card_count, frames_sha256, source)"
            " VALUES (:run_id,:app,:profile_id,:created,:feature_version,'[]',:card_count,"
            ":frames_sha256,:source)",
            {**row, "created": datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()},
        )
        h.store.con.commit()
    else:
        h.client.insert(
            "photo_features",
            {**row, "embedding": [], "created_at": "2026-01-01T00:00:00+00:00"},
        )


# ------------------------------------------------------------------ protocol / surface parity
def test_both_stores_conform_to_the_protocol_and_match_its_signatures(tmp_path):
    sqlite_store = SQLiteStore(tmp_path / "s.db")
    bq_store = _bq_store(_SqlBQ())
    try:
        assert isinstance(sqlite_store, Store) and isinstance(bq_store, Store)
    finally:
        sqlite_store.close()
        bq_store.close()
    for name in NEW_METHODS:
        protocol = inspect.signature(getattr(Store, name))
        for cls in (SQLiteStore, BigQueryStore):
            actual = inspect.signature(getattr(cls, name))
            assert list(actual.parameters) == list(protocol.parameters), (cls, name)
            for pname, param in actual.parameters.items():
                assert param.kind == protocol.parameters[pname].kind, (cls, name, pname)
                assert param.default == protocol.parameters[pname].default
    # add_photo_features is keyword-only: a positional card_count/embedding swap is impossible.
    params = list(inspect.signature(Store.add_photo_features).parameters.values())[1:]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params)


def test_photo_features_is_a_new_bigquery_table_with_no_migration_line():
    assert _TABLES["photo_features"] == (
        "run_id STRING, app STRING, profile_id STRING, created_at TIMESTAMP, "
        "feature_version STRING, embedding ARRAY<FLOAT64>, card_count INT64, "
        "frames_sha256 STRING, source STRING"
    )
    assert not [m for m in _MIGRATIONS if "photo_features" in m]


def test_a_project_without_the_table_gets_it_created_and_nothing_else():
    class _DdlBQ:
        def __init__(self):
            self.scripts = []

        def query(self, sql, job_config=None):
            if "INFORMATION_SCHEMA.COLUMNS" in sql:
                return _Job(
                    [
                        {"table_name": table, "column_name": col}
                        for table in _TABLES
                        if table != "photo_features"
                        for col in _columns(table)
                    ]
                )
            self.scripts.append(sql)
            return _Job([])

    client = _DdlBQ()
    _bq_store(client)
    script = "\n".join(client.scripts)
    assert (
        f"CREATE TABLE IF NOT EXISTS `proj.ds.photo_features` ({_TABLES['photo_features']});"
        in script
    )
    assert script.count("CREATE TABLE") == 1 and "ALTER TABLE" not in script


def test_an_existing_sqlite_file_gains_the_table_on_open(tmp_path):
    path = tmp_path / "old.db"
    SQLiteStore(path).close()
    con = sqlite3.connect(path)
    con.execute("DROP TABLE photo_features")
    con.commit()
    con.close()
    store = SQLiteStore(path)
    try:
        assert store.add_photo_features(**_features()) is True
    finally:
        store.close()


# ------------------------------------------------------------------ writes + validation
@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"embedding": [0.1] * (WHOLE_PHOTO_DIMS - 1)}, "dimensions"),
        ({"embedding": [*_vec(0.1)[:-1], float("nan")]}, "finite"),
        ({"embedding": [*_vec(0.1)[:-1], True]}, "finite"),
        ({"embedding": "0.1"}, "number sequence"),
        ({"embedding": None, "card_count": 2}, "number sequence"),
        # zero photo cards is an extraction failure, never a row (RT-2)
        ({"embedding": None, "card_count": 0}, "positive integer"),
        ({"card_count": 0}, "positive integer"),
        ({"card_count": -1}, "card_count"),
        ({"card_count": True}, "card_count"),
        ({"card_count": 2.0}, "card_count"),
        ({"frames_sha256": "A" * 64}, "frames_sha256"),
        ({"frames_sha256": "a" * 63}, "frames_sha256"),
        ({"source": "manual"}, "source"),
        ({"feature_version": "whole_photo_v2"}, "feature_version"),
        ({"profile_id": ""}, "profile_id"),
        ({"profile_id": "  "}, "profile_id"),
        ({"app": None}, "app"),
        ({"run_id": None}, "run_id"),
    ],
)
def test_a_malformed_row_is_refused_and_nothing_is_written(h, overrides, needle):
    with pytest.raises(ValueError, match=needle):
        h.store.add_photo_features(**_features(**overrides))
    h.label(True, 0.1, "p1")
    assert h.store.load_label_feature_rows(V)[0][3:] == (None, False)


def test_validation_normalises_numpy_like_numbers_to_floats():
    class _Real(float):
        pass

    row = validate_photo_features(
        **_features(embedding=[_Real(0.25)] * WHOLE_PHOTO_DIMS)
    )
    assert {type(x) for x in row["embedding"]} == {float}


def test_both_sources_are_accepted_and_true_is_returned(h):
    assert h.store.add_photo_features(**_features(source="live")) is True
    assert h.store.add_photo_features(**_features(source="backfill")) is True


def test_an_unknown_feature_version_is_refused_on_every_read(h):
    for method in NEW_METHODS[1:]:
        with pytest.raises(ValueError, match="feature_version"):
            getattr(h.store, method)("whole_photo_v2")


# ------------------------------------------------------------------ the label join
def test_feature_rows_are_exactly_the_label_rows_plus_the_photo_columns(h):
    h.label(True, 0.1, "p1")
    h.label(False, 0.2, "p2")
    h.label(False, 0.3, "p3", source="auto_review_random")
    h.label(True, 0.4, "")  # a label with no profile identity can never match
    _legacy_zero_card_row(h, "p2")
    h.store.add_photo_features(**_features(profile_id="p1", embedding=_vec(0.7)))
    rows = h.store.load_label_feature_rows(V)
    assert [r[:3] for r in rows] == h.store.load_label_rows()
    assert [(r[3] is not None and r[3][0], r[4]) for r in rows] == [
        (0.7, True),
        (False, True),  # a stored zero-card row: a row, but no vector
        (False, False),
        (False, False),
    ]
    assert rows[0][3] == _vec(0.7)
    assert h.store.photo_feature_coverage(V) == {
        "labels": 4,
        "with_features": 2,
        "missing": 2,
        "zero_cards": 1,
    }


def test_the_latest_row_for_a_profile_wins(h):
    h.label(True, 0.1, "p1")
    _legacy_zero_card_row(h, "p1")
    assert h.store.load_label_feature_rows(V)[0][3:] == (None, True)
    h.store.add_photo_features(**_features(embedding=_vec(0.1)))
    assert h.store.load_label_feature_rows(V)[0][3] == _vec(0.1)
    h.store.add_photo_features(**_features(embedding=_vec(0.9), source="backfill"))
    assert h.store.load_label_feature_rows(V)[0][3] == _vec(0.9)


def test_a_stored_zero_card_row_is_relisted_for_the_backfill_until_a_vector_wins(h):
    """RT-2: a zero-card row is not a usable feature, so the work list offers its label again
    (the backfill's newer row then wins) instead of the loader's NOT READY reason pointing at a
    backfill with nothing to do."""
    h.label(True, 0.1, "p1")
    h.label(False, 0.2, "p2")
    _legacy_zero_card_row(h, "p1")
    h.store.add_photo_features(**_features(profile_id="p2"))
    assert [m["profile_id"] for m in h.store.labels_missing_photo_features(V)] == ["p1"]
    assert h.store.photo_feature_coverage(V)["zero_cards"] == 1
    h.store.add_photo_features(**_features(profile_id="p1", source="backfill"))
    assert h.store.labels_missing_photo_features(V) == []
    assert h.store.photo_feature_coverage(V)["zero_cards"] == 0


def test_a_zero_card_row_is_refused_by_both_write_paths(h):
    for best_effort in (False, True):
        with pytest.raises(ValueError, match="positive integer"):
            h.store.add_photo_features(
                **_features(embedding=None, card_count=0), best_effort=best_effort
            )
    h.label(True, 0.1, "p1")
    assert h.store.load_label_feature_rows(V)[0][3:] == (None, False)


def test_an_identity_less_label_never_matches_even_a_row_written_behind_the_api(h):
    h.label(True, 0.1, "")
    row = validate_photo_features(**_features())
    row["profile_id"] = ""
    if h.kind == "sqlite":
        h.store.con.execute(
            "INSERT INTO photo_features (run_id, app, profile_id, created_at,"
            " feature_version, embedding, card_count, frames_sha256, source)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                row["run_id"],
                row["app"],
                "",
                1.0,
                V,
                json.dumps(row["embedding"]),
                row["card_count"],
                row["frames_sha256"],
                row["source"],
            ),
        )
        h.store.con.commit()
    else:
        h.client.insert(
            "photo_features", {**row, "created_at": "2026-01-01T00:00:00+00:00"}
        )
    assert h.store.load_label_feature_rows(V)[0][3:] == (None, False)


def test_bigquery_rows_with_one_timestamp_resolve_to_the_last_written(monkeypatch):
    monkeypatch.setattr(bq_module, "_now", lambda: "2026-01-01T00:00:00+00:00")
    store = _bq_store(_SqlBQ())
    store.add_label("r1", "hinge", True, [0.1] * 4, profile_id="p1")
    store.add_photo_features(**_features(embedding=_vec(0.1)))
    store.add_photo_features(**_features(embedding=_vec(0.2)))
    assert store.load_label_feature_rows(V)[0][3] == _vec(0.2)
    store.close()


def test_a_row_for_another_app_does_not_match(h):
    h.label(True, 0.1, "p1")
    h.store.add_photo_features(**_features(app="bumble"))
    assert h.store.load_label_feature_rows(V)[0][4] is False


def test_a_retracted_label_is_hidden_from_feature_rows_like_from_label_rows(h):
    h.label(True, 0.1, "p1")
    h.label(False, 0.2, "p2")
    if h.kind == "sqlite":
        created = h.store.con.execute(
            "SELECT created_at FROM labels WHERE profile_id='p1'"
        ).fetchone()[0]
        h.store.con.execute(
            "INSERT INTO label_retractions VALUES ('c','r1','hinge','manual','p1',?,0,'f',"
            "'x','e',0)",
            (created,),
        )
        h.store.con.commit()
    else:
        h.store.flush()
        created = h.client.con.execute(
            "SELECT created_at FROM \"proj.ds.labels\" WHERE profile_id='p1'"
        ).fetchone()[0]
        h.client.con.execute(
            'INSERT INTO "proj.ds.label_retractions" (correction_id, run_id, app, source,'
            " profile_id, label_created_at) VALUES ('c','r1','hinge','manual','p1',?)",
            (created,),
        )
        h.store._labels_cache = None  # what append_label_retraction does
    rows = h.store.load_label_feature_rows(V)
    assert (
        [r[:3] for r in rows]
        == h.store.load_label_rows()
        == [(False, [0.2] * 4, "manual")]
    )
    assert [m["profile_id"] for m in h.store.labels_missing_photo_features(V)] == ["p2"]


def test_a_corrupt_stored_row_raises_instead_of_being_guessed(h):
    h.label(True, 0.1, "p1")
    if h.kind == "sqlite":
        h.store.con.execute(
            "INSERT INTO photo_features (run_id, app, profile_id, created_at,"
            " feature_version, embedding, card_count, frames_sha256, source)"
            " VALUES ('r1','hinge','p1',1,?,'[]',4,?,'live')",
            (V, _sha("x")),
        )
        h.store.con.commit()
    else:
        h.client.insert(
            "photo_features",
            {
                "run_id": "r1",
                "app": "hinge",
                "profile_id": "p1",
                "created_at": "2026-01-01T00:00:00+00:00",
                "feature_version": V,
                "embedding": [],
                "card_count": 4,
                "frames_sha256": _sha("x"),
                "source": "live",
            },
        )
    with pytest.raises(ValueError, match="corrupt"):
        h.store.load_label_feature_rows(V)


# ------------------------------------------------------------------ backfill work list
def test_missing_list_names_each_label_lacking_a_row_once(h):
    h.label(True, 0.1, "p1")
    h.label(False, 0.2, "p2", source="auto_review_near_cut", run_id="r2")
    h.label(True, 0.3, "p2", source="auto_review_near_cut", run_id="r2")  # same key
    h.label(False, 0.4, "p3")
    h.label(True, 0.5, "")
    h.label(True, 0.6, "")
    h.store.add_photo_features(**_features(profile_id="p3"))
    missing = h.store.labels_missing_photo_features(V)
    assert [(m["run_id"], m["profile_id"], m["source"]) for m in missing] == [
        ("r1", "p1", "manual"),
        ("r2", "p2", "auto_review_near_cut"),
        ("r1", "", "manual"),
        ("r1", "", "manual"),
    ]
    for entry in missing:
        assert set(entry) == {
            "run_id",
            "app",
            "profile_id",
            "source",
            "frame_uris",
            "frame_sha256s",
        }
        assert entry["app"] == "hinge"
    if h.kind == "sqlite":
        # This backend archives no object-store frames, so it can never feed a backfill.
        assert all(m["frame_uris"] == m["frame_sha256s"] == [] for m in missing)


def test_sqlite_auto_pass_blobs_are_not_offered_as_frames(tmp_path):
    store = SQLiteStore(tmp_path / "s.db")
    try:
        assert store.record_auto_pass(
            run_id="r1",
            app="hinge",
            profile_id="p1",
            score=0.2,
            threshold=0.5,
            embedding=[0.1] * 4,
            photos=[b"\x89PNG\r\n\x1a\none", b"\x89PNG\r\n\x1a\ntwo"],
            random_sample=True,
        )
        store.record_auto_pass_review(profile_id="p1", would_like=True)
        [entry] = store.labels_missing_photo_features(V)
        assert entry["source"] == "auto_review_random"
        assert entry["frame_uris"] == entry["frame_sha256s"] == []
    finally:
        store.close()


def _photo_row(table, profile_id, index, *, run_id="r1", created="2026-01-01T00:00:00"):
    return (
        table,
        {
            "run_id": run_id,
            "app": "hinge",
            "profile_id": profile_id,
            "created_at": created + "+00:00",
            "photo_index": index,
            "gcs_uri": f"gs://photos/{profile_id}/{created}/{index}.png",
            "sha256": _sha(f"{profile_id}-{index}"),
            "byte_size": 10,
            "content_type": "image/png",
        },
    )


def test_bigquery_frames_come_from_the_right_archive_in_photo_index_order():
    client = _SqlBQ()
    store = _bq_store(client)
    store.add_label("r1", "hinge", True, [0.1] * 4, profile_id="m1")
    store.add_label(
        "r2", "hinge", False, [0.2] * 4, source="auto_review_random", profile_id="a1"
    )
    for index in (2, 0, 1):
        client.insert(*_photo_row("profile_photos", "m1", index))
        client.insert(*_photo_row("auto_pass_photos", "a1", index, run_id="r2"))
    # The same profile_id archived under a different run is not this label's frames.
    client.insert(*_photo_row("profile_photos", "m1", 0, run_id="other"))
    by_id = {m["profile_id"]: m for m in store.labels_missing_photo_features(V)}
    assert by_id["m1"]["frame_uris"] == [
        f"gs://photos/m1/2026-01-01T00:00:00/{i}.png" for i in range(3)
    ]
    assert by_id["m1"]["frame_sha256s"] == [_sha(f"m1-{i}") for i in range(3)]
    assert by_id["a1"]["frame_sha256s"] == [_sha(f"a1-{i}") for i in range(3)]
    # A pruned AUTO pass has no frames left.
    client.con.execute('DELETE FROM "proj.ds.auto_pass_photos"')
    by_id = {m["profile_id"]: m for m in store.labels_missing_photo_features(V)}
    assert by_id["a1"]["frame_uris"] == by_id["a1"]["frame_sha256s"] == []
    assert len(by_id["m1"]["frame_uris"]) == 3
    store.close()


def test_bigquery_reads_only_the_archives_a_missing_label_needs():
    client = _SqlBQ()
    store = _bq_store(client)
    store.add_label("r1", "hinge", True, [0.1] * 4, profile_id="m1")
    client.queries.clear()
    store.labels_missing_photo_features(V)
    assert any(".profile_photos`" in q for q in client.queries)
    assert not any(".auto_pass_photos`" in q for q in client.queries)
    store.add_photo_features(**_features(profile_id="m1"))
    client.queries.clear()
    assert store.labels_missing_photo_features(V) == []
    assert not any("_photos`" in q for q in client.queries)
    store.close()


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([], ([], [])),
        ([(1.0, 1, "gs://b/1", _sha("1"))], ([], [])),  # index set is not 0..n-1
        ([(1.0, 0, "gs://b/0", _sha("0")), (1.0, 2, "gs://b/2", _sha("2"))], ([], [])),
        ([(1.0, 0, "", _sha("0"))], ([], [])),  # no object URI (a SQLite BLOB row)
        ([(1.0, 0, "gs://b/0", "short")], ([], [])),
        ([(1.0, "0", "gs://b/0", _sha("0"))], ([], [])),
        # one index carrying two different photos
        ([(1.0, 0, "gs://b/0", _sha("0")), (1.0, 0, "gs://b/0", _sha("x"))], ([], [])),
        # an exact duplicate row is the same photo
        (
            [(1.0, 0, "gs://b/0", _sha("0")), (1.0, 0, "gs://b/0", _sha("0"))],
            (["gs://b/0"], [_sha("0")]),
        ),
        # two archived sets of the SAME frames: the latest set's URIs
        (
            [(2.0, 0, "gs://new/0", _sha("0")), (1.0, 0, "gs://old/0", _sha("0"))],
            (["gs://new/0"], [_sha("0")]),
        ),
        # two archived sets that disagree: no frames rather than a guess
        (
            [(2.0, 0, "gs://new/0", _sha("0")), (1.0, 0, "gs://old/0", _sha("y"))],
            ([], []),
        ),
    ],
)
def test_frame_resolution_never_returns_a_partial_or_guessed_set(rows, expected):
    assert resolve_label_frames(rows) == expected


# ------------------------------------------------------------------ BigQuery specifics
def test_bigquery_buffers_and_flushes_rows():
    client = _SqlBQ()
    store = _bq_store(client, flush_every=100)
    store.add_photo_features(**_features(profile_id="p1"))
    store.add_photo_features(**_features(profile_id="p2", embedding=_vec(0.2)))
    assert [r["profile_id"] for r in store._buf["photo_features"]] == ["p1", "p2"]
    assert client.rows("photo_features") == []
    store.flush()
    stored = client.rows("photo_features")
    assert [
        (r["profile_id"], json.loads(r["embedding"])[0], r["card_count"])
        for r in stored
    ] == [
        ("p1", 0.5, 3),
        ("p2", 0.2, 3),
    ]
    assert {r["feature_version"] for r in stored} == {V}
    assert {r["source"] for r in stored} == {"live"}
    assert all(isinstance(r["created_at"], float) for r in stored)
    store.close()


class _RejectingSqlBQ(_SqlBQ):
    """Inserts every table except photo_features, which BigQuery rejects or raises on."""

    def __init__(self, *, raise_exc=None):
        super().__init__()
        self.raise_exc = raise_exc
        self.photo_attempts = 0

    def insert_rows_json(self, table_id, rows, row_ids=None):
        if table_id.endswith(".photo_features"):
            self.photo_attempts += 1
            if self.raise_exc is not None:
                raise self.raise_exc
            return [{"index": 0, "errors": [{"reason": "invalid"}]}]
        return super().insert_rows_json(table_id, rows, row_ids=row_ids)


def test_bigquery_best_effort_row_is_inserted_at_once_and_never_buffered():
    client = _SqlBQ()
    store = _bq_store(client, flush_every=100)
    store.add_label("r1", "hinge", True, [0.1] * 4, profile_id="p1")
    assert store.load_label_feature_rows(V)[0][4] is False  # fills the feature map
    assert store.add_photo_features(**_features(), best_effort=True) is True
    assert store._buf["photo_features"] == []
    assert [r["profile_id"] for r in client.rows("photo_features")] == ["p1"]
    assert store.load_label_feature_rows(V)[0][3] == _vec(0.5)  # the map saw it
    assert "photo_features=1" in store.saved_summary()
    store.close()


@pytest.mark.parametrize(
    "raise_exc",
    [None, NotFound("404 Not found: Table proj:ds.photo_features")],
    ids=["row-errors", "raised"],
)
def test_bigquery_rejected_best_effort_row_never_poisons_a_later_flush(
    raise_exc, capsys
):
    """RT-1: in face_crop_v1 mode nothing reads the row, so its rejection must not stay
    buffered to fail every later flush() and the shutdown close()."""
    client = _RejectingSqlBQ(raise_exc=raise_exc)
    store = _bq_store(client, flush_every=1)  # the shipped config.yaml value
    store.add_label("r1", "hinge", True, [0.1] * 4, profile_id="p1")
    assert store.add_photo_features(**_features(), best_effort=True) is False
    assert "WARNING" in capsys.readouterr().out
    assert store._buf["photo_features"] == []
    for _ in range(3):
        store.flush()  # nothing left to resend
    store.add_label("r1", "hinge", False, [0.2] * 4, profile_id="p2")
    store.close()
    assert client.photo_attempts == 1
    assert [r["profile_id"] for r in client.rows("labels")] == ["p1", "p2"]
    assert store.dropped_rows() == {}


def test_bigquery_required_row_still_fails_the_flush_it_belongs_to():
    """Whole-photo mode keeps the buffered, flush-bounded write: a rejection is loud there."""
    client = _RejectingSqlBQ()
    store = _bq_store(client, flush_every=100)
    assert store.add_photo_features(**_features()) is True
    with pytest.raises(RuntimeError, match="photo_features"):
        store.flush()


def test_sqlite_best_effort_row_reports_a_database_error_as_false(h, capsys):
    if h.kind != "sqlite":
        pytest.skip("SQLite only")
    h.store.con.execute("DROP TABLE photo_features")
    assert h.store.add_photo_features(**_features(), best_effort=True) is False
    assert "WARNING" in capsys.readouterr().out
    with pytest.raises(sqlite3.Error):
        h.store.add_photo_features(**_features())


def test_bigquery_feature_map_is_read_once_and_kept_coherent_by_writes():
    client = _SqlBQ()
    store = _bq_store(client)
    store.add_label("r1", "hinge", True, [0.1] * 4, profile_id="p1")
    store.add_label("r1", "hinge", False, [0.2] * 4, profile_id="p2")
    assert [r[4] for r in store.load_label_feature_rows(V)] == [False, False]
    client.queries.clear()
    store.add_photo_features(**_features(profile_id="p2", embedding=_vec(0.3)))
    store.add_label("r1", "hinge", True, [0.4] * 4, profile_id="p3")
    store.add_photo_features(
        **_features(profile_id="p3", embedding=_vec(0.6)), best_effort=True
    )
    rows = store.load_label_feature_rows(V)
    assert (
        client.queries == []
    )  # served from the caches add_label/add_photo_features keep
    assert [r[:3] for r in rows] == store.load_label_rows()
    assert [(r[3] is not None and r[3][0], r[4]) for r in rows] == [
        (False, False),
        (0.3, True),
        (0.6, True),
    ]
    store.close()


def test_bigquery_table_rows_resolve_by_created_at_not_by_read_order():
    client = _SqlBQ()
    store = _bq_store(client)
    store.add_label("r1", "hinge", True, [0.1] * 4, profile_id="p1")
    for created, value in (
        ("2026-01-03", 0.9),
        ("2026-01-01", 0.1),
        ("2026-01-02", 0.5),
    ):
        row = validate_photo_features(**_features(embedding=_vec(value)))
        client.insert(
            "photo_features", {**row, "created_at": f"{created}T00:00:00+00:00"}
        )
    assert store.load_label_feature_rows(V)[0][3] == _vec(0.9)
    store.close()


def test_bigquery_v1_reads_never_touch_photo_features():
    client = _SqlBQ()
    store = _bq_store(client)
    store.add_label("r1", "hinge", True, [0.1] * 4, profile_id="p1")
    store.add_photo_features(**_features())
    store.flush()
    client.queries.clear()
    store.load_label_rows()
    store.load_labels()
    assert client.queries and not any("photo_features" in q for q in client.queries)
    # load_label_rows' own SELECT, unchanged: no identity columns ride along in v1.
    assert all(
        q.startswith("SELECT l.liked, l.embedding, l.source FROM")
        for q in client.queries
    )
    store.close()


def test_bigquery_cache_filled_by_load_label_rows_is_refilled_with_identities():
    client = _SqlBQ()
    store = _bq_store(client)
    store.add_label("r1", "hinge", True, [0.1] * 4, profile_id="p1")
    store.add_label("r1", "hinge", False, [0.2] * 4, profile_id="p2")
    store.flush()
    store.add_label("r1", "hinge", True, [0.3] * 4, profile_id="p3")  # still buffered
    store.add_photo_features(**_features(profile_id="p3"))
    v1_rows = store.load_label_rows()  # fills the cache WITHOUT identities
    rows = store.load_label_feature_rows(V)
    assert [r[:3] for r in rows] == v1_rows == store.load_label_rows()
    assert [r[4] for r in rows] == [False, False, True]
    # After the refill the identities ride along with every append, so no query repeats.
    client.queries.clear()
    store.add_label("r1", "hinge", False, [0.4] * 4, profile_id="p1")
    store.add_photo_features(**_features(profile_id="p1", embedding=_vec(0.8)))
    rows = store.load_label_feature_rows(V)
    assert client.queries == []
    assert [(r[0], r[3] is not None, r[4]) for r in rows] == [
        (True, True, True),
        (False, False, False),
        (True, True, True),
        (False, True, True),
    ]
    store.close()


def _flushed_bq_with_labels(labels):
    client = _SqlBQ()
    store = _bq_store(client)
    for liked, value, profile_id in labels:
        store.add_label("r1", "hinge", liked, [value] * 4, profile_id=profile_id)
    store.flush()
    return client, store


def test_bigquery_identities_keep_the_order_load_label_rows_already_served():
    client, store = _flushed_bq_with_labels(
        [(True, 0.1, "p1"), (False, 0.2, "p2"), (True, 0.3, "p3")]
    )
    store.add_photo_features(**_features(profile_id="p2", embedding=_vec(0.2)))
    v1_rows = store.load_label_rows()
    client.reverse_reads_containing = "l.app, l.profile_id"  # the identity read
    rows = store.load_label_feature_rows(V)
    assert [r[:3] for r in rows] == v1_rows == store.load_label_rows()
    assert [(r[1][0], r[4]) for r in rows] == [(0.1, False), (0.2, True), (0.3, False)]
    assert rows[1][3] == _vec(0.2)
    store.close()


def test_bigquery_identical_labels_under_two_identities_keep_the_cache_order():
    client, store = _flushed_bq_with_labels(
        [(True, 0.1, "p1"), (False, 0.5, "p3"), (True, 0.1, "p2")]
    )
    store.add_photo_features(**_features(profile_id="p2", embedding=_vec(0.2)))
    v1_rows = store.load_label_rows()
    cache_before = store._labels_cache
    client.reverse_reads_containing = "l.app, l.profile_id"
    rows = store.load_label_feature_rows(V)
    assert store._labels_cache is cache_before  # aligned, not refilled
    assert [r[:3] for r in rows] == v1_rows
    # The two identical rows are interchangeable: exactly one of them carries p2's vector.
    assert sorted((r[1][0], r[4]) for r in rows) == [
        (0.1, False),
        (0.1, True),
        (0.5, False),
    ]
    store.close()


def test_bigquery_cached_row_missing_from_the_read_refills_from_the_read():
    client, store = _flushed_bq_with_labels([(True, 0.1, "p1"), (False, 0.2, "p2")])
    store.load_label_rows()
    client.con.execute("DELETE FROM \"proj.ds.labels\" WHERE profile_id = 'p1'")
    rows = store.load_label_feature_rows(V)
    assert rows == [(False, [0.2] * 4, "manual", None, False)]
    assert store.load_label_rows() == [(False, [0.2] * 4, "manual")]
    store.close()


def test_bigquery_rows_written_elsewhere_since_the_fill_are_ignored_like_v1():
    client, store = _flushed_bq_with_labels([(True, 0.1, "p1")])
    v1_rows = store.load_label_rows()
    client.insert(
        "labels",
        {
            "run_id": "r9",
            "app": "hinge",
            "profile_id": "p9",
            "created_at": "2026-01-01T00:00:00+00:00",
            "liked": False,
            "source": "manual",
            "embedding": [0.9] * 4,
        },
    )
    rows = store.load_label_feature_rows(V)
    assert [r[:3] for r in rows] == v1_rows == store.load_label_rows()
    store.close()


def test_bigquery_invalidated_cache_never_reuses_stale_identities():
    client = _SqlBQ()
    store = _bq_store(client)
    store.add_label("r1", "hinge", True, [0.1] * 4, profile_id="p1")
    store.add_photo_features(**_features(profile_id="p1"))
    assert store.load_label_feature_rows(V)[0][4] is True
    store.flush()
    # Invalidate the way a retraction/undo does, then let load_label_rows refill it (a new
    # list with no identities) with a DIFFERENT label set of the same length.
    client.con.execute('DELETE FROM "proj.ds.labels"')
    client.insert(
        "labels",
        {
            "run_id": "r1",
            "app": "hinge",
            "profile_id": "p9",
            "created_at": "2026-01-01T00:00:00+00:00",
            "liked": False,
            "source": "manual",
            "embedding": [0.9] * 4,
        },
    )
    store._labels_cache = None
    assert store.load_label_rows() == [(False, [0.9] * 4, "manual")]
    rows = store.load_label_feature_rows(V)
    assert rows == [(False, [0.9] * 4, "manual", None, False)]
    store.close()


def test_bigquery_read_only_store_reads_a_missing_table_as_no_rows():
    client = _SqlBQ()
    client.missing_tables = {"photo_features", "auto_pass_photos"}
    store = _bq_store(client, ensure=False)
    client.insert(
        "labels",
        {
            "run_id": "r1",
            "app": "hinge",
            "profile_id": "a1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "liked": True,
            "source": "auto_review_random",
            "embedding": [0.1] * 4,
        },
    )
    assert store.photo_feature_coverage(V) == {
        "labels": 1,
        "with_features": 0,
        "missing": 1,
        "zero_cards": 0,
    }
    [entry] = store.labels_missing_photo_features(V)
    assert entry["frame_uris"] == []
    store.close()


def test_bigquery_writing_store_does_not_hide_a_missing_table():
    client = _SqlBQ()
    store = _bq_store(client, ensure=True)
    client.missing_tables = {"photo_features"}
    with pytest.raises(NotFound):
        store.load_label_feature_rows(V)
    store.close()


def test_bigquery_refuses_a_late_write_after_close():
    store = _bq_store(_SqlBQ())
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.add_photo_features(**_features())


def test_missing_table_accommodation_names_only_photo_features():
    exc = NotFound("Not found: Table proj:ds.photo_features")
    assert bq_module._missing_optional_photo_features(exc, ensure=False)
    assert not bq_module._missing_optional_photo_features(exc, ensure=True)
    assert not bq_module._missing_optional_photo_features(
        NotFound("Not found: Table proj:ds.labels"), ensure=False
    )
