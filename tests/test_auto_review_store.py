"""AUTO pass archive + owner review, in BOTH store backends.

AUTO archives every profile the model passes; the owner later answers "would have liked" or
"correct pass", and every answer becomes a training label. The behaviours that matter and
that these tests pin down:

* an archived pass is NOT a label, a decision, or a profiles/profile_photos row;
* an answer is final (a second tap must never become a second label);
* the label and its review row land together (SQLite: one transaction; BigQuery: label first,
  and a retry after a partial failure never mints a second label);
* only passes whose screenshots are all stored are ever offered for review;
* ``load_label_rows`` serves the same rows as ``load_labels`` plus each source, from the same
  cache, so a retrain after every label stays coherent.

The BigQuery half runs the store's REAL SQL against SQLite (``_SqlBQ`` below translates nothing
but ``RAND()``/``COUNTIF``), so a typo in a generated query, a wrong join, or a dropped
predicate fails here instead of in production. Tests that mutate-sensitive guards say which
guard they pin.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import re
import sqlite3
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import operation_love.ranker.bigquery_store as bq_module
from operation_love.ranker import Store
from operation_love.ranker.bigquery_store import _TABLES, BigQueryStore
from operation_love.ranker.store import (
    AUTO_REVIEW_KINDS,
    SQLiteStore,
    auto_review_label_source,
)

PNG = b"\x89PNG\r\n\x1a\n"

NEW_METHODS = (
    "record_auto_pass",
    "next_auto_pass_for_review",
    "auto_pass_photo",
    "record_auto_pass_review",
    "auto_review_stats",
    "undo_latest_auto_review",
    "prune_auto_passes",
    "load_label_rows",
)


def _png(tag: str) -> bytes:
    return PNG + tag.encode()


def _columns(table: str) -> list[str]:
    return [c.strip().split()[0] for c in _TABLES[table].split(",")]


# --------------------------------------------------------------------------------------------
# A BigQuery test double that EXECUTES the store's SQL (on SQLite) and models GCS.
# --------------------------------------------------------------------------------------------
class NotFound(
    Exception
):  # the store classifies by type NAME, like the real SDK's NotFound
    pass


class _CountIf:
    def __init__(self):
        self.n = 0

    def step(self, value):
        if value:
            self.n += 1

    def finalize(self):
        return self.n


_TIMESTAMP_COLUMNS = {
    table: {
        c.strip().split()[0]
        for c in spec.split(",")
        if c.strip().split()[1].upper() == "TIMESTAMP"
    }
    for table, spec in _TABLES.items()
}
_BOOL_COLUMNS = {"liked", "random_sample", "would_like", "capture_truncated"}


class _SqlBQ:
    """Runs generated SELECTs against in-memory SQLite tables named like BigQuery's ids."""

    def __init__(self):
        self.con = sqlite3.connect(":memory:", check_same_thread=False)
        self.con.create_function("RAND", 0, random.random)
        self.con.create_function(
            "STARTS_WITH", 2, lambda s, p: None if s is None else s.startswith(p)
        )
        self.con.create_aggregate("COUNTIF", 1, _CountIf)
        for table in _TABLES:
            self.con.execute(
                f'CREATE TABLE "proj.ds.{table}" ({", ".join(_columns(table))})'
            )
        self.queries: list[tuple[str, object]] = []
        self.dml_log: list[str] = []
        self.dml_refusals: set[str] = set()
        self.insert_log: list[tuple[str, list[dict]]] = []
        self.insert_failures: dict[str, int] = {}
        self.missing_tables: set[str] = set()
        # BigQuery's streaming buffer is not instantly visible to queries: rows inserted into
        # a "lagged" table are accepted but only readable after settle().
        self.lagged: set[str] = set()
        self._unsettled: list[tuple[str, dict]] = []

    # ---- reads
    def query(self, sql, job_config=None):
        self.queries.append((sql, job_config))
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
        statement = sql.lstrip().upper()
        assert statement.startswith(("SELECT", "DELETE")), sql
        # A bound TIMESTAMP arrives as a datetime; the tables here store epoch floats.
        params = {
            p.name: p.value.timestamp() if isinstance(p.value, datetime) else p.value
            for p in (getattr(job_config, "query_parameters", None) or [])
        }
        cur = self.con.execute(sql, params)
        if statement.startswith("DELETE"):
            # Real BigQuery reports the affected rows on the job; a refusal to touch rows in
            # the streaming buffer is modelled by `dml_refusals` (affected rows == 0).
            self.dml_log.append(sql)
            affected = cur.rowcount
            for table in self.dml_refusals:
                if f".{table}`" in sql:
                    self.con.rollback()
                    affected = 0
            return _DmlJob(affected)
        names = [d[0] for d in cur.description]
        rows = []
        for raw in cur.fetchall():
            row = dict(zip(names, raw, strict=True))
            for key, value in row.items():
                if key == "embedding" and isinstance(value, str):
                    row[key] = json.loads(value)
                elif key in {"created_at", "reviewed_at"} and isinstance(value, float):
                    row[key] = datetime.fromtimestamp(value, tz=timezone.utc)
                elif key in _BOOL_COLUMNS and value is not None:
                    row[key] = bool(value)
            rows.append(row)
        return _Job(rows)

    # ---- writes
    def insert_rows_json(self, table_id, rows, row_ids=None):
        table = table_id.rsplit(".", 1)[1]
        self.insert_log.append((table, [dict(r) for r in rows]))
        if self.insert_failures.get(table, 0) > 0:
            self.insert_failures[table] -= 1
            return [{"index": 0, "errors": [{"reason": "invalid"}]}]
        for row in rows:
            if table in self.lagged:
                self._unsettled.append((table, dict(row)))
            else:
                self.insert(table, row)
        return []

    def settle(self):
        for table, row in self._unsettled:
            self.insert(table, row)
        self._unsettled.clear()

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

    def count(self, table, where="1=1"):
        return self.con.execute(
            f'SELECT COUNT(*) FROM "proj.ds.{table}" WHERE {where}'
        ).fetchone()[0]

    def rows(self, table):
        cur = self.con.execute(f'SELECT * FROM "proj.ds.{table}" ORDER BY rowid')
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]


class _Job:
    def __init__(self, rows):
        self._rows = rows

    def result(self):
        return self._rows


class _DmlJob(_Job):
    def __init__(self, affected):
        super().__init__([])
        self.num_dml_affected_rows = affected


class _Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    def upload_from_string(self, data, content_type=None):
        if self.bucket.fail_upload(self.name):
            raise RuntimeError("simulated GCS upload failure")
        self.bucket.objects[self.name] = (bytes(data), content_type)
        self.bucket.upload_log.append(self.name)

    def download_as_bytes(self):
        if self.name not in self.bucket.objects:
            raise NotFound(f"No such object: {self.name}")
        return self.bucket.objects[self.name][0]

    def delete(self):
        self.bucket.objects.pop(self.name, None)
        self.bucket.deleted.append(self.name)


class _Iam:
    uniform_bucket_level_access_enabled = True
    public_access_prevention = "enforced"


class _Bucket:
    def __init__(self, name):
        self.name = name
        self.objects: dict[str, tuple[bytes, str | None]] = {}
        self.upload_log: list[str] = []
        self.deleted: list[str] = []
        self.iam_configuration = _Iam()
        self.fail_upload = lambda name: False

    def exists(self):
        return True

    def reload(self):
        pass

    def blob(self, name):
        return _Blob(self, name)


class _Storage:
    def __init__(self):
        self.buckets: dict[str, _Bucket] = {}

    def bucket(self, name):
        return self.buckets.setdefault(name, _Bucket(name))


@pytest.fixture(autouse=True)
def _bigquery_sdk_stub(monkeypatch):
    """The test_bigquery_store.py convention: bound queries need the SDK's two symbols; CI
    installs no bigquery extra, so give the store a stand-in instead of the real package."""

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
    monkeypatch.setattr(bq_module, "_UPLOAD_BACKOFF_S", 0.0)


def _bq_store(client, storage, *, ensure=True, flush_every=100):
    return BigQueryStore(
        "proj",
        "ds",
        photo_bucket="photos",
        flush_every=flush_every,
        client=client,
        storage_client=storage,
        ensure=ensure,
    )


# --------------------------------------------------------------------------------------------
# Dual-backend harness: the behaviours both stores must share.
# --------------------------------------------------------------------------------------------
class _Harness:
    def __init__(self, kind, tmp_path):
        self.kind = kind
        if kind == "sqlite":
            self.store = SQLiteStore(tmp_path / "store.db")
            self.client = self.storage = None
        else:
            self.client, self.storage = _SqlBQ(), _Storage()
            self.store = _bq_store(self.client, self.storage)

    def close(self):
        self.store.close()

    def count(self, table, where="1=1"):
        if self.kind == "sqlite":
            return self.store.con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {where}"
            ).fetchone()[0]
        self.store.flush()
        return self.client.count(table, where)

    def labels(self):
        """Durable label rows as dicts (source/profile_id/liked/embedding...)."""
        if self.kind == "sqlite":
            cur = self.store.con.execute(
                "SELECT run_id, app, liked, source, embedding, photo_count, profile_id FROM labels ORDER BY id"
            )
            names = [d[0] for d in cur.description]
            rows = [dict(zip(names, r, strict=True)) for r in cur.fetchall()]
            for row in rows:
                row["liked"] = bool(row["liked"])
                row["embedding"] = json.loads(row["embedding"])
            return rows
        self.store.flush()
        rows = self.client.rows("labels")
        for row in rows:
            row["liked"] = bool(row["liked"])
            row["embedding"] = json.loads(row["embedding"])
        return rows

    def reviews(self):
        if self.kind == "sqlite":
            cur = self.store.con.execute(
                "SELECT profile_id, app, reviewed_at, would_like, kind FROM auto_pass_reviews ORDER BY id"
            )
        else:
            self.store.flush()
            cur = self.client.con.execute(
                'SELECT profile_id, app, reviewed_at, would_like, kind FROM "proj.ds.auto_pass_reviews" ORDER BY rowid'
            )
        names = [d[0] for d in cur.description]
        out = [dict(zip(names, r, strict=True)) for r in cur.fetchall()]
        for row in out:
            row["would_like"] = bool(row["would_like"])
        return out

    def set_pass_time(self, profile_id, epoch):
        """Move an archived pass (manifest AND photo rows) to an explicit time."""
        if self.kind == "sqlite":
            for table in ("auto_passes", "auto_pass_photos"):
                self.store.con.execute(
                    f"UPDATE {table} SET created_at=? WHERE profile_id=?",
                    (float(epoch), profile_id),
                )
            self.store.con.commit()
        else:
            self.store.flush()
            for table in ("auto_passes", "auto_pass_photos"):
                self.client.con.execute(
                    f'UPDATE "proj.ds.{table}" SET created_at=? WHERE profile_id=?',
                    (float(epoch), profile_id),
                )

    def objects(self):
        """Names of the GCS objects the BigQuery double holds (empty for SQLite)."""
        if self.kind == "sqlite":
            return set()
        return set(self.storage.bucket("photos").objects)

    def seed_review(self, profile_id, app, would_like, kind, reviewed_at):
        """Insert a review row at an explicit time (window tests need deterministic order)."""
        if self.kind == "sqlite":
            self.store.con.execute(
                "INSERT INTO auto_pass_reviews (profile_id, app, reviewed_at, would_like, kind)"
                " VALUES (?,?,?,?,?)",
                (profile_id, app, float(reviewed_at), int(would_like), kind),
            )
            self.store.con.commit()
        else:
            self.client.insert(
                "auto_pass_reviews",
                {
                    "profile_id": profile_id,
                    "app": app,
                    "reviewed_at": float(reviewed_at),
                    "would_like": would_like,
                    "kind": kind,
                },
            )


@pytest.fixture(params=["sqlite", "bigquery"])
def h(request, tmp_path):
    harness = _Harness(request.param, tmp_path)
    yield harness
    try:
        harness.close()
    except Exception:  # noqa: BLE001 - a test may leave a deliberately failing buffer
        pass


def _archive(h, pid, **overrides):
    """record_auto_pass with sensible defaults; `overrides` replace any keyword verbatim
    (so a test can pass a deliberately malformed one). `photos` may be a count or a list."""
    kwargs = {
        "run_id": "run-1",
        "app": "hinge",
        "profile_id": pid,
        "score": 0.40,
        "threshold": 0.50,
        "embedding": [0.25, -0.5, float(len(pid))],
        "photos": 2,
        "profile_name": overrides.pop("name", ""),
        "random_sample": False,
    }
    kwargs.update(overrides)
    if type(kwargs["photos"]) is int:
        kwargs["photos"] = [_png(f"{pid}-{i}") for i in range(kwargs["photos"])]
    return h.store.record_auto_pass(**kwargs)


# ------------------------------------------------------------------ protocol / surface parity
def test_both_stores_expose_the_same_keyword_only_surface_as_the_protocol():
    for name in NEW_METHODS:
        protocol = inspect.signature(getattr(Store, name))
        for cls in (SQLiteStore, BigQueryStore):
            actual = inspect.signature(getattr(cls, name))
            assert list(actual.parameters) == list(protocol.parameters), (cls, name)
            for pname, param in actual.parameters.items():
                if pname == "self":
                    continue
                assert param.kind == protocol.parameters[pname].kind, (cls, name, pname)
                assert param.default == protocol.parameters[pname].default, (
                    cls,
                    name,
                    pname,
                )
    # Everything but load_label_rows is keyword-only (a positional score/threshold swap is
    # exactly the bug this surface must make impossible).
    for name in NEW_METHODS[:-1]:
        params = list(
            inspect.signature(getattr(SQLiteStore, name)).parameters.values()
        )[1:]
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params), name


def test_review_label_sources_share_the_prefix_the_ranker_keys_on():
    assert auto_review_label_source("random") == "auto_review_random"
    assert auto_review_label_source("near_cut") == "auto_review_near_cut"
    assert AUTO_REVIEW_KINDS == ("random", "near_cut")
    with pytest.raises(ValueError):
        auto_review_label_source("whatever")


# ------------------------------------------------------------------ record_auto_pass
def test_record_auto_pass_archives_the_pass_and_writes_no_label_or_decision(h):
    assert _archive(h, "p1", score=0.41, threshold=0.5, name="Ada", random_sample=True)

    assert h.count("auto_passes" if h.kind == "sqlite" else "auto_passes") == 1
    assert h.count("auto_pass_photos") == 2
    # An unanswered pass is not training data and not an action record.
    assert h.count("labels") == 0
    assert h.count("decisions") == 0
    pending = h.store.auto_review_stats()
    assert (pending["pending_random"], pending["pending_near_cut"]) == (1, 0)


@pytest.mark.parametrize(
    "override",
    [
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": "0.4"},
        {"score": True},
        {"threshold": float("nan")},
        {"threshold": None},
        {"embedding": []},
        {"embedding": [0.1, float("nan")]},
        {"embedding": "abc"},
        {"embedding": None},
        {"random_sample": 1},
        {"random_sample": "true"},
        {"random_sample": None},
        {"profile_id": ""},
        {"profile_id": "   "},
        {"app": ""},
        {"run_id": None},
    ],
    ids=lambda o: "-".join(f"{k}={v!r}" for k, v in o.items()),
)
def test_record_auto_pass_rejects_malformed_arguments_and_writes_nothing(h, override):
    with pytest.raises(ValueError):
        _archive(h, "p1", **override)
    assert h.count("auto_passes") == 0
    assert h.count("auto_pass_photos") == 0
    if h.kind == "bigquery":
        assert h.storage.bucket("photos").upload_log == []  # refused BEFORE any upload


def test_record_auto_pass_accepts_numpy_style_scalars(h):
    np = pytest.importorskip("numpy")
    ok = h.store.record_auto_pass(
        run_id="r",
        app="hinge",
        profile_id="np",
        score=np.float32(0.4),
        threshold=np.float64(0.5),
        embedding=np.asarray([0.1, 0.2], dtype=np.float32),
        photos=[_png("np")],
        random_sample=False,
    )
    assert ok is True


@pytest.mark.parametrize(
    "photos",
    [[], None, [b""], [_png("a"), b""], [_png("a"), "not-bytes"], 3.5],
    ids=["empty", "none", "empty-bytes", "second-empty", "str-photo", "not-a-list"],
)
def test_record_auto_pass_without_usable_photos_returns_false_loudly(h, photos, capsys):
    assert _archive(h, "p1", photos=photos) is False
    assert "Auto pass archive warning" in capsys.readouterr().out
    # No manifest: a pass the owner cannot see cannot be judged.
    assert h.count("auto_passes") == 0
    assert h.count("auto_pass_photos") == 0


def test_second_archive_of_the_same_pass_is_refused_and_keeps_the_first(h, capsys):
    assert _archive(h, "p1", score=0.41) is True
    assert _archive(h, "p1", score=0.10, photos=1) is False
    assert "already archived" in capsys.readouterr().out
    assert h.count("auto_passes") == 1
    assert h.count("auto_pass_photos") == 2
    assert h.store.next_auto_pass_for_review(kind="near_cut")["score"] == pytest.approx(
        0.41
    )


# ------------------------------------------------------------------ next_auto_pass_for_review
def test_next_auto_pass_returns_the_documented_shape(h):
    _archive(
        h,
        "p1",
        score=0.41,
        threshold=0.52,
        name="Ada",
        app="bumble",
        run_id="run-9",
        photos=3,
    )
    found = h.store.next_auto_pass_for_review(kind="near_cut")
    assert set(found) == {
        "profile_id",
        "app",
        "run_id",
        "created_at",
        "score",
        "threshold",
        "photo_count",
        "profile_name",
        "kind",
    }
    assert found["profile_id"] == "p1"
    assert (found["app"], found["run_id"], found["kind"]) == (
        "bumble",
        "run-9",
        "near_cut",
    )
    assert found["score"] == pytest.approx(0.41)
    assert found["threshold"] == pytest.approx(0.52)
    assert found["photo_count"] == 3
    assert found["profile_name"] == "Ada"
    assert type(found["created_at"]) is float
    assert abs(found["created_at"] - datetime.now(timezone.utc).timestamp()) < 60


def test_next_auto_pass_validates_kind_and_app(h):
    for bad in ("", "RANDOM", "both", None, 3):
        with pytest.raises(ValueError):
            h.store.next_auto_pass_for_review(kind=bad)
    with pytest.raises(ValueError):
        h.store.next_auto_pass_for_review(kind="random", app="")
    assert h.store.next_auto_pass_for_review(kind="random") is None
    assert h.store.next_auto_pass_for_review(kind="near_cut") is None


def test_random_pool_only_offers_random_unreviewed_passes_and_is_uniform(h):
    for pid in ("r1", "r2", "r3"):
        _archive(h, pid, random_sample=True)
    _archive(h, "n1", random_sample=False)

    seen = {
        h.store.next_auto_pass_for_review(kind="random")["profile_id"]
        for _ in range(150)
    }
    assert seen == {
        "r1",
        "r2",
        "r3",
    }  # never the near-cut pass; every random one reachable

    h.store.record_auto_pass_review(profile_id="r2", would_like=False)
    seen = {
        h.store.next_auto_pass_for_review(kind="random")["profile_id"]
        for _ in range(150)
    }
    assert seen == {"r1", "r3"}  # a reviewed pass is never offered again


def test_near_cut_pool_offers_the_pass_closest_below_its_own_threshold_first(h):
    # Chosen so neither the highest score (c) nor the lowest threshold (b) is the answer:
    # the order is by (score - threshold), each pass against ITS OWN threshold.
    _archive(h, "a", score=0.45, threshold=0.50)  # -0.05
    _archive(h, "b", score=0.10, threshold=0.12)  # -0.02  <- closest
    _archive(h, "c", score=0.49, threshold=0.60)  # -0.11
    _archive(
        h, "rand", score=0.499, threshold=0.5, random_sample=True
    )  # never near_cut

    order = []
    for _ in range(3):
        nxt = h.store.next_auto_pass_for_review(kind="near_cut")
        order.append(nxt["profile_id"])
        h.store.record_auto_pass_review(profile_id=nxt["profile_id"], would_like=False)
    assert order == ["b", "a", "c"]
    assert h.store.next_auto_pass_for_review(kind="near_cut") is None


def test_next_auto_pass_app_filter(h):
    _archive(h, "hin", app="hinge", random_sample=True)
    _archive(h, "bum", app="bumble", random_sample=True)
    for _ in range(40):
        assert (
            h.store.next_auto_pass_for_review(kind="random", app="bumble")["profile_id"]
            == "bum"
        )
        assert (
            h.store.next_auto_pass_for_review(kind="random", app="hinge")["profile_id"]
            == "hin"
        )
    assert h.store.next_auto_pass_for_review(kind="random", app="tinder") is None


# ------------------------------------------------------------------ auto_pass_photo
def test_auto_pass_photo_round_trips_the_exact_bytes_the_model_saw(h):
    _archive(h, "p1", photos=3)
    for i in range(3):
        assert h.store.auto_pass_photo(profile_id="p1", index=i) == _png(f"p1-{i}")
    assert h.store.auto_pass_photo(profile_id="p1", index=3) is None
    assert h.store.auto_pass_photo(profile_id="nobody", index=0) is None


@pytest.mark.parametrize("index", [-1, "0", 1.0, True, None])
def test_auto_pass_photo_rejects_a_bad_index(h, index):
    with pytest.raises(ValueError):
        h.store.auto_pass_photo(profile_id="p1", index=index)


# ------------------------------------------------------------------ record_auto_pass_review
@pytest.mark.parametrize(
    ("random_sample", "source", "kind"),
    [
        (True, "auto_review_random", "random"),
        (False, "auto_review_near_cut", "near_cut"),
    ],
)
def test_review_writes_a_training_label_and_a_review_row(
    h, random_sample, source, kind
):
    emb = [0.125, -0.75, 3.5]
    _archive(
        h,
        "p1",
        random_sample=random_sample,
        embedding=emb,
        name="Ada",
        photos=3,
        run_id="run-7",
        app="bumble",
    )

    result = h.store.record_auto_pass_review(profile_id="p1", would_like=True)

    assert result == {"kind": kind, "label_source": source}
    (label,) = h.labels()
    assert (label["run_id"], label["app"], label["profile_id"]) == (
        "run-7",
        "bumble",
        "p1",
    )
    assert label["liked"] is True
    assert label["source"] == source
    assert label["embedding"] == emb  # the stored embedding, not a recomputation
    assert label["photo_count"] == 3
    (review,) = h.reviews()
    assert (
        review["profile_id"],
        review["app"],
        review["would_like"],
        review["kind"],
    ) == (
        "p1",
        "bumble",
        True,
        kind,
    )
    # "Correct pass" is a label too, with liked False.
    _archive(h, "p2", random_sample=random_sample)
    h.store.record_auto_pass_review(profile_id="p2", would_like=False)
    assert [lbl["liked"] for lbl in h.labels()] == [True, False]


def test_review_is_final_a_second_answer_never_becomes_a_second_label(h):
    _archive(h, "p1")
    h.store.record_auto_pass_review(profile_id="p1", would_like=False)
    for answer in (False, True):
        with pytest.raises(ValueError, match="already reviewed"):
            h.store.record_auto_pass_review(profile_id="p1", would_like=answer)
    assert len(h.labels()) == 1
    assert len(h.reviews()) == 1
    assert h.labels()[0]["liked"] is False


def test_review_of_an_unknown_pass_is_a_value_error_and_writes_nothing(h):
    with pytest.raises(ValueError, match="no archived auto pass"):
        h.store.record_auto_pass_review(profile_id="ghost", would_like=True)
    assert h.labels() == []
    assert h.reviews() == []


@pytest.mark.parametrize("answer", [1, 0, "yes", None])
def test_review_requires_a_real_bool_answer(h, answer):
    _archive(h, "p1")
    with pytest.raises(ValueError):
        h.store.record_auto_pass_review(profile_id="p1", would_like=answer)
    assert h.labels() == []
    # ... and the pass is still reviewable afterwards.
    h.store.record_auto_pass_review(profile_id="p1", would_like=True)


def test_review_does_not_touch_decisions_or_the_pass_archive(h):
    _archive(h, "p1", photos=2)
    h.store.record_auto_pass_review(profile_id="p1", would_like=True)
    assert h.count("decisions") == 0
    assert h.count("auto_passes") == 1
    assert h.count("auto_pass_photos") == 2
    assert h.store.auto_pass_photo(profile_id="p1", index=1) == _png(
        "p1-1"
    )  # still viewable


# ------------------------------------------------------------------ load_label_rows
def test_load_label_rows_matches_load_labels_and_carries_each_source(h):
    h.store.add_label(
        "run", "hinge", True, [1.0, 2.0], source="manual", profile_id="m1"
    )
    h.store.add_label("run", "hinge", False, [3.0], source="auto", profile_id="a1")
    _archive(h, "p1", embedding=[0.5, 0.25], random_sample=True)
    h.store.record_auto_pass_review(profile_id="p1", would_like=True)

    rows = h.store.load_label_rows()
    assert rows == [
        (True, [1.0, 2.0], "manual"),
        (False, [3.0], "auto"),
        (True, [0.5, 0.25], "auto_review_random"),
    ]
    assert h.store.load_labels() == [(liked, emb) for liked, emb, _ in rows]


def test_load_label_rows_maps_a_legacy_null_source_to_empty_string(h):
    if h.kind == "sqlite":
        h.store.con.execute(
            "INSERT INTO labels (run_id, app, created_at, liked, source, embedding, photo_count, profile_id)"
            " VALUES ('r','hinge',1.0,1,NULL,'[1.0]',0,'old')"
        )
        h.store.con.commit()
    else:
        h.client.insert(
            "labels",
            {
                "run_id": "r",
                "app": "hinge",
                "profile_id": "old",
                "created_at": 1.0,
                "liked": True,
                "source": None,
                "embedding": [1.0],
                "photo_count": 0,
                "profile_name": "",
            },
        )
    assert h.store.load_label_rows() == [(True, [1.0], "")]
    assert h.store.load_labels() == [(True, [1.0])]


def test_load_label_rows_hides_a_retracted_label_exactly_like_load_labels(h):
    h.store.add_label("run", "hinge", True, [1.0], source="manual", profile_id="keep")
    h.store.add_label("run", "hinge", True, [2.0], source="manual", profile_id="gone")
    if h.kind == "sqlite":
        created = h.store.con.execute(
            "SELECT created_at FROM labels WHERE profile_id='gone'"
        ).fetchone()[0]
        h.store.con.execute(
            "INSERT INTO label_retractions VALUES ('c1','run','hinge','manual','gone',?,?,'fp','r','e',1.0)",
            (created, created),
        )
        h.store.con.commit()
    else:
        h.store.flush()
        created = h.client.con.execute(
            "SELECT created_at FROM \"proj.ds.labels\" WHERE profile_id='gone'"
        ).fetchone()[0]
        h.client.insert(
            "label_retractions",
            {
                "correction_id": "c1",
                "run_id": "run",
                "app": "hinge",
                "source": "manual",
                "profile_id": "gone",
                "label_created_at": created,
                "decision_created_at": created,
                "decision_fingerprint": "fp",
                "reason": "r",
                "evidence_ref": "e",
                "created_at": 1.0,
            },
        )
        h.store._labels_cache = (
            None  # the cache predates the tombstone; reload from the table
        )
    assert h.store.load_label_rows() == [(True, [1.0], "manual")]
    assert h.store.load_labels() == [(True, [1.0])]


# ------------------------------------------------------------------ auto_review_stats
def test_stats_are_all_zero_on_an_empty_store(h):
    assert h.store.auto_review_stats() == {
        "pending_random": 0,
        "pending_near_cut": 0,
        "random_reviewed": 0,
        "random_would_like": 0,
        "near_cut_reviewed": 0,
        "near_cut_would_like": 0,
        "auto_likes": 0,
        "auto_passes": 0,
        "auto_model_passes": 0,
    }


def test_stats_count_pending_reviewed_and_auto_decisions_since_the_first_pass(h):
    # Decisions BEFORE the first archived pass must not count (the archive did not exist yet,
    # so those passes could never be reviewed): they are the "since" half of the contract.
    h.store.record_decision(
        "old", "hinge", "like", 0.9, source="auto", created_at=1000.0
    )
    h.store.record_decision(
        "old", "hinge", "dislike", 0.1, source="auto", created_at=1000.0
    )
    _archive(h, "r1", random_sample=True)
    _archive(h, "r2", random_sample=True)
    _archive(h, "r3", random_sample=True)
    _archive(h, "n1")
    _archive(h, "n2")
    _archive(h, "n3")
    _archive(h, "b1", app="bumble", random_sample=True)
    _archive(h, "b2", app="bumble")
    for _ in range(3):
        h.store.record_decision("run", "hinge", "like", 0.9, source="auto")
    for _ in range(5):
        h.store.record_decision("run", "hinge", "dislike", 0.1, source="auto")
    h.store.record_decision("run", "bumble", "like", 0.9, source="auto")
    h.store.record_decision("run", "bumble", "dislike", 0.1, source="auto")
    h.store.record_decision("run", "hinge", "like", 0.9, source="manual")  # not AUTO
    h.store.record_decision("run", "hinge", "dislike", 0.1, source="manual")

    for pid, answer in (
        ("r1", True),
        ("r2", False),
        ("r3", False),
        ("n1", True),
        ("n2", False),
    ):
        h.store.record_auto_pass_review(profile_id=pid, would_like=answer)

    assert (
        h.store.auto_review_stats()
        == {
            "pending_random": 1,  # b1
            "pending_near_cut": 2,  # n3, b2
            "random_reviewed": 3,
            "random_would_like": 1,
            "near_cut_reviewed": 2,
            "near_cut_would_like": 1,
            "auto_likes": 4,
            "auto_passes": 6,
            "auto_model_passes": 0,  # these decision rows carry no lineage id: none entered the archive
        }
    )
    assert h.store.auto_review_stats(app="hinge") == {
        "pending_random": 0,
        "pending_near_cut": 1,
        "random_reviewed": 3,
        "random_would_like": 1,
        "near_cut_reviewed": 2,
        "near_cut_would_like": 1,
        "auto_likes": 3,
        "auto_passes": 5,
        "auto_model_passes": 0,
    }
    assert h.store.auto_review_stats(app="bumble") == {
        "pending_random": 1,
        "pending_near_cut": 1,
        "random_reviewed": 0,
        "random_would_like": 0,
        "near_cut_reviewed": 0,
        "near_cut_would_like": 0,
        "auto_likes": 1,
        "auto_passes": 1,
        "auto_model_passes": 0,
    }


def test_stats_window_opens_at_the_first_archived_passes_own_decision(h):
    # The worker records a pass's decision row BEFORE archiving it.  Counting from the archive
    # row's timestamp therefore dropped the first pass's own decision (3 passes read back as 2),
    # which biased pass_share and so the live recall correction.
    h.store.record_decision(
        "run",
        "hinge",
        "dislike",
        0.1,
        source="auto",
        profile_id="unarchived",
        created_at=500.0,
    )  # an older, never-archived pass: before the window
    for i, pid in enumerate(("p1", "p2", "p3")):
        h.store.record_decision(
            "run",
            "hinge",
            "dislike",
            0.1,
            source="auto",
            profile_id=pid,
            created_at=1000.0 + i,
        )
        _archive(h, pid, random_sample=bool(i % 2))

    stats = h.store.auto_review_stats()
    assert stats["auto_passes"] == 3
    assert h.store.auto_review_stats(app="bumble")["auto_passes"] == 0


def test_stats_model_passes_exclude_likes_the_session_demoted_to_a_pass(h):
    """`auto_passes` counts every AUTO 'dislike' decision, including a like the contextual
    policy or the like-ratio ceiling demoted to a pass.  Those are never archived, so the random
    pool that measures the false-pass rate cannot have sampled them, yet they were counted in
    the share of passes that measurement is multiplied by.  `auto_model_passes` is the part that
    entered the archive.  Mutating the join out (counting every dislike) makes it 5, not 3."""
    for i, pid in enumerate(
        ("m1", "m2", "m3")
    ):  # genuine model passes: decision + archive
        h.store.record_decision(
            "run",
            "hinge",
            "dislike",
            0.1,
            source="auto",
            profile_id=pid,
            created_at=1000.0 + i,
        )
        _archive(h, pid, random_sample=bool(i % 2))
    for i, pid in enumerate(
        ("d1", "d2")
    ):  # demoted likes: a 'dislike' row, never archived
        h.store.record_decision(
            "run",
            "hinge",
            "dislike",
            0.7,
            source="auto",
            profile_id=pid,
            created_at=1010.0 + i,
        )
    h.store.record_decision(
        "run", "hinge", "like", 0.9, source="auto", profile_id="l1", created_at=1020.0
    )
    h.store.record_decision(  # another app's archived pass
        "run",
        "bumble",
        "dislike",
        0.1,
        source="auto",
        profile_id="b1",
        created_at=1030.0,
    )
    _archive(h, "b1", app="bumble")
    # A manual decision with an archived pass's id must not count either.
    h.store.record_decision(
        "run",
        "hinge",
        "dislike",
        0.1,
        source="manual",
        profile_id="m1",
        created_at=1040.0,
    )

    stats = h.store.auto_review_stats()
    assert (stats["auto_likes"], stats["auto_passes"], stats["auto_model_passes"]) == (
        1,
        6,
        4,
    )
    hinge = h.store.auto_review_stats(app="hinge")
    assert (hinge["auto_passes"], hinge["auto_model_passes"]) == (5, 3)
    bumble = h.store.auto_review_stats(app="bumble")
    assert (bumble["auto_passes"], bumble["auto_model_passes"]) == (1, 1)


def test_stats_model_passes_are_not_multiplied_by_a_duplicated_archive_row(h):
    """BigQuery's streaming dedup is best effort, so an archive row can exist twice.  A join to
    the archive would count its decision twice and push `auto_model_passes` above `auto_passes`,
    which the ranker's live-review parser rightly rejects as a store fault."""
    if h.kind != "bigquery":
        pytest.skip("SQLite's unique index makes a duplicate archive row impossible")
    h.store.record_decision(
        "run",
        "hinge",
        "dislike",
        0.1,
        source="auto",
        profile_id="m1",
        created_at=1000.0,
    )
    _archive(h, "m1")
    h.store.flush()
    row = h.client.rows("auto_passes")[0]
    h.client.insert("auto_passes", {**row, "embedding": json.loads(row["embedding"])})
    assert (
        h.client.count("auto_passes") == 2
    )  # precondition: the duplicate is really there

    stats = h.store.auto_review_stats()
    assert (stats["auto_passes"], stats["auto_model_passes"]) == (1, 1)


def test_stats_have_no_auto_decision_counts_until_something_is_archived(h):
    h.store.record_decision("run", "hinge", "like", 0.9, source="auto")
    h.store.record_decision("run", "hinge", "dislike", 0.1, source="auto")
    stats = h.store.auto_review_stats()
    assert (stats["auto_likes"], stats["auto_passes"], stats["auto_model_passes"]) == (
        0,
        0,
        0,
    )  # 0/0 when none archived


def test_random_review_counts_cover_only_the_most_recent_window(h):
    for i in range(5):
        _archive(h, f"r{i}", random_sample=True)
    # r0..r4 answered in that order at t=100..104: would_like = True, True, False, True, False.
    for i, answer in enumerate((True, True, False, True, False)):
        h.seed_review(f"r{i}", "hinge", answer, "random", 100.0 + i)
    # near-cut reviews are cumulative whatever the window.
    _archive(h, "n0")
    _archive(h, "n1")
    h.seed_review("n0", "hinge", True, "near_cut", 1.0)
    h.seed_review("n1", "hinge", False, "near_cut", 2.0)

    full = h.store.auto_review_stats()
    assert (full["random_reviewed"], full["random_would_like"]) == (5, 3)
    last3 = h.store.auto_review_stats(window=3)  # r2, r3, r4 = F, T, F
    assert (last3["random_reviewed"], last3["random_would_like"]) == (3, 1)
    last1 = h.store.auto_review_stats(window=1)  # r4 = F
    assert (last1["random_reviewed"], last1["random_would_like"]) == (1, 0)
    for stats in (full, last3, last1):
        assert (stats["near_cut_reviewed"], stats["near_cut_would_like"]) == (2, 1)
        assert stats["pending_random"] == 0 and stats["pending_near_cut"] == 0


@pytest.mark.parametrize("window", [0, -1, "5", None, 2.0, True])
def test_stats_reject_a_bad_window(h, window):
    with pytest.raises(ValueError):
        h.store.auto_review_stats(window=window)


def test_stats_reject_a_bad_app(h):
    with pytest.raises(ValueError):
        h.store.auto_review_stats(app="")


# ------------------------------------------------------------------ undo of a review answer
def _answer(h, pid, answer, **archive):
    _archive(h, pid, **archive)
    return h.store.record_auto_pass_review(profile_id=pid, would_like=answer)


def test_undo_with_nothing_answered_does_nothing(h):
    assert h.store.undo_latest_auto_review() is None
    _archive(h, "p1")  # an archived but unanswered pass is not an answer
    assert h.store.undo_latest_auto_review() is None
    assert h.count("auto_passes") == 1


def test_undo_deletes_the_newest_answers_label_and_review_row_and_reopens_the_pass(h):
    h.store.add_label(
        "run-0", "hinge", True, [0.9, 0.8], source="manual", profile_id="m0"
    )
    _answer(h, "p1", True, random_sample=True, name="Ada")
    _answer(h, "p2", False, random_sample=False, name="Bea")
    before = h.store.auto_review_stats()
    assert (before["near_cut_reviewed"], before["pending_near_cut"]) == (1, 0)

    undone = h.store.undo_latest_auto_review()

    assert undone == {
        "profile_id": "p2",
        "profile_name": "Bea",
        "kind": "near_cut",
        "would_like": False,
    }
    # Only p2's answer went: its label, its review row.  The earlier answer and the manual label
    # are untouched, and the pass can be answered again.
    assert [(r["profile_id"], r["source"]) for r in h.labels()] == [
        ("m0", "manual"),
        ("p1", "auto_review_random"),
    ]
    assert [r["profile_id"] for r in h.reviews()] == ["p1"]
    after = h.store.auto_review_stats()
    assert (after["near_cut_reviewed"], after["pending_near_cut"]) == (0, 1)
    assert after["random_reviewed"] == before["random_reviewed"] == 1
    again = h.store.record_auto_pass_review(profile_id="p2", would_like=True)
    assert again["label_source"] == "auto_review_near_cut"
    assert [
        (r["profile_id"], r["liked"]) for r in h.labels() if r["profile_id"] == "p2"
    ] == [("p2", True)]  # the corrected answer, exactly one label for the pass

    # The training set the ranker reads follows: no stale label for the undone answer.
    assert [row[2] for row in h.store.load_label_rows()] == [
        "manual",
        "auto_review_random",
        "auto_review_near_cut",
    ]


def test_undo_walks_back_one_answer_at_a_time_and_never_touches_manual_labels(h):
    h.store.add_label(
        "run-0", "hinge", True, [0.9, 0.8], source="manual", profile_id="m0"
    )
    for pid in ("a", "b", "c"):
        _answer(h, pid, pid == "b", random_sample=True)
    seen = [h.store.undo_latest_auto_review()["profile_id"] for _ in range(3)]
    assert seen == ["c", "b", "a"]
    assert h.store.undo_latest_auto_review() is None
    assert [r["profile_id"] for r in h.labels()] == ["m0"]
    assert h.reviews() == []
    assert h.store.auto_review_stats()["pending_random"] == 3


def test_removing_the_latest_label_refuses_when_it_is_a_review_answer(h):
    """The newest label is a review answer: 'Remove last saved label' must not reach past it
    and delete an older, unrelated swipe while the mistaken answer stays in the training set."""
    h.store.add_label(
        "run-0",
        "hinge",
        True,
        [0.9, 0.8],
        source="manual",
        profile_id="old",
        profile_name="Ada",
    )
    _answer(h, "p1", True, random_sample=True)
    snapshot = h.store.load_label_rows()
    assert [row[2] for row in snapshot] == ["manual", "auto_review_random"]

    with pytest.raises(ValueError, match="Undo last answer"):
        h.store.remove_latest_training_label()

    assert h.store.load_label_rows() == snapshot  # nothing was removed
    assert [r["profile_id"] for r in h.reviews()] == ["p1"]
    # After the answer is undone the swipe IS the latest label again, and is removable.
    assert h.store.undo_latest_auto_review()["profile_id"] == "p1"
    assert h.store.remove_latest_training_label() == {
        "profile_name": "Ada",
        "profile_id": "old",
    }
    assert h.store.load_label_rows() == []


def test_removing_the_latest_label_still_removes_a_swipe_newer_than_the_answers(h):
    _answer(h, "p1", False, random_sample=True)
    h.store.add_label(
        "run-1",
        "hinge",
        False,
        [0.1, 0.2],
        source="manual",
        profile_id="new",
        profile_name="Cy",
    )
    assert h.store.remove_latest_training_label() == {
        "profile_name": "Cy",
        "profile_id": "new",
    }
    assert [row[2] for row in h.store.load_label_rows()] == ["auto_review_random"]


# ------------------------------------------------------------------ retention
def _archive_aged(h, pids_ages_days, now=2_000_000_000.0):
    """Archive passes, then place each ``age`` days before ``now`` (oldest listed first)."""
    for pid, age in pids_ages_days:
        _archive(h, pid, photos=2)
        h.set_pass_time(pid, now - age * 86400.0)
    return now


def _pass_ids(h):
    if h.kind == "sqlite":
        rows = h.store.con.execute("SELECT profile_id FROM auto_passes").fetchall()
    else:
        h.store.flush()
        rows = h.client.con.execute(
            'SELECT profile_id FROM "proj.ds.auto_passes"'
        ).fetchall()
    return {r[0] for r in rows}


def test_prune_by_age_removes_the_stale_passes_and_their_photos_and_nothing_else(h):
    now = _archive_aged(h, [("old1", 90), ("old2", 40), ("new1", 5), ("new2", 1)])
    assert len(h.objects()) == (8 if h.kind == "bigquery" else 0)

    removed = h.store.prune_auto_passes(max_age_days=30.0, now=now)

    assert removed == 2
    assert _pass_ids(h) == {"new1", "new2"}
    assert h.count("auto_pass_photos", "profile_id IN ('old1','old2')") == 0
    assert h.count("auto_pass_photos", "profile_id IN ('new1','new2')") == 4
    assert h.store.auto_pass_photo(profile_id="old1", index=0) is None
    assert h.store.auto_pass_photo(profile_id="new1", index=0) is not None
    if h.kind == "bigquery":  # the stored objects go too, or they would leak forever
        assert len(h.objects()) == 4
        assert all("/new" in name for name in h.objects())
    assert h.store.prune_auto_passes(max_age_days=30.0, now=now) == 0  # idempotent


def test_prune_by_count_keeps_the_newest_passes(h):
    now = _archive_aged(h, [("a", 5), ("b", 4), ("c", 3), ("d", 2), ("e", 1)])
    assert h.store.prune_auto_passes(max_passes=2, now=now) == 3
    assert _pass_ids(h) == {"d", "e"}
    assert h.store.prune_auto_passes(max_passes=2, now=now) == 0


def test_prune_applies_both_bounds_and_zero_or_none_means_unbounded(h):
    now = _archive_aged(h, [("a", 100), ("b", 3), ("c", 2), ("d", 1)])
    assert h.store.prune_auto_passes(now=now) == 0
    assert h.store.prune_auto_passes(max_passes=0, max_age_days=0, now=now) == 0
    assert h.store.prune_auto_passes(max_passes=None, max_age_days=None) == 0
    assert _pass_ids(h) == {"a", "b", "c", "d"}
    # age drops "a"; count then drops the oldest of what is left ("b").
    assert h.store.prune_auto_passes(max_passes=2, max_age_days=30, now=now) == 2
    assert _pass_ids(h) == {"c", "d"}


def test_prune_never_touches_the_owners_answers(h):
    """An answer is a label and a review row; the screenshots and embedding of the stranger it
    was about are what retention removes.  The live false-pass measurement (reviews) and the
    training set (labels) must survive the pass they came from."""
    now = _archive_aged(h, [("a", 90), ("b", 80), ("c", 1)])
    h.store.record_auto_pass_review(profile_id="a", would_like=True)
    h.store.record_auto_pass_review(profile_id="b", would_like=False)
    labels, reviews = h.labels(), h.reviews()
    assert len(labels) == 2 and len(reviews) == 2

    assert h.store.prune_auto_passes(max_age_days=30.0, now=now) == 2

    assert h.labels() == labels and h.reviews() == reviews
    assert h.store.load_label_rows()[0][2] == "auto_review_near_cut"
    # An answered-then-pruned pass can no longer be answered, undone cleanly, or shown.
    with pytest.raises(ValueError, match="no archived auto pass"):
        h.store.record_auto_pass_review(profile_id="a", would_like=False)
    assert h.store.next_auto_pass_for_review(kind="near_cut")["profile_id"] == "c"
    undone = h.store.undo_latest_auto_review()
    assert undone["profile_id"] == "b" and undone["profile_name"] == ""
    assert [r["profile_id"] for r in h.labels()] == ["a"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_passes": -1},
        {"max_passes": True},
        {"max_passes": 2.0},
        {"max_passes": "5"},
        {"max_age_days": -1},
        {"max_age_days": True},
        {"max_age_days": float("nan")},
        {"max_age_days": float("inf")},
        {"max_age_days": "30"},
    ],
)
def test_prune_rejects_nonsense_bounds_instead_of_keeping_everything(h, kwargs):
    _archive(h, "p1")
    with pytest.raises(ValueError):
        h.store.prune_auto_passes(**kwargs)
    assert _pass_ids(h) == {"p1"}


def test_prune_leaves_other_apps_passes_to_the_same_rules(h):
    now = _archive_aged(h, [("h1", 90)])
    _archive(h, "b1", app="bumble")
    h.set_pass_time("b1", now - 90 * 86400.0)
    assert h.store.prune_auto_passes(max_age_days=30, now=now) == 2
    assert _pass_ids(h) == set()


# ------------------------------------------------------------------ advisory opener cleanup counts
def test_a_review_answer_is_not_like_evidence_for_the_run_it_came_from(h):
    """record_auto_pass_review writes the label under the pass's own AUTO run_id.  The opener
    cleanup planner refuses any run with effective Like evidence, and one tap on 'would have
    liked' must not turn a run that never landed a Like into one that did."""
    _archive(h, "p1", run_id="run-A", random_sample=True)
    _archive(h, "p2", run_id="run-A", random_sample=False)
    h.store.record_auto_pass_review(profile_id="p1", would_like=True)
    h.store.record_auto_pass_review(profile_id="p2", would_like=False)
    h.store.flush()

    rows = h.store.advisory_opener_run_rows("run-A", "hinge")
    assert rows["effective_counts"]["like_labels"] == 0
    assert rows["effective_counts"]["pass_labels"] == 0
    # The raw preference rows are still reported: the labels exist, they just are not swipes.
    assert rows["preference_counts"]["labels"] == 2

    # A genuine manual Like in that run still counts (the exclusion is by source, not blanket).
    h.store.add_label(
        "run-A", "hinge", True, [1.0, 2.0], source="manual", profile_id="m1"
    )
    h.store.add_label("run-A", "hinge", False, [1.0, 2.0], source="", profile_id="m2")
    h.store.flush()  # BigQuery buffers labels; the audit reads what is stored
    rows = h.store.advisory_opener_run_rows("run-A", "hinge")
    assert rows["effective_counts"]["like_labels"] == 1
    assert rows["effective_counts"]["pass_labels"] == 1


# ============================================================================================
# SQLite specifics
# ============================================================================================
class _HookConnection:
    """Forwards to a real connection; `before(sql)` / `after(sql)` may inject faults."""

    def __init__(self, con, before=None, after=None):
        self._con, self._before, self._after = con, before, after

    def execute(self, sql, *args, **kwargs):
        normal = " ".join(str(sql).split())
        if self._before:
            self._before(normal)
        result = self._con.execute(sql, *args, **kwargs)
        if self._after:
            self._after(normal)
        return result

    def __getattr__(self, name):
        return getattr(self._con, name)


def _sqlite(tmp_path):
    return SQLiteStore(tmp_path / "store.db")


def _counts(store):
    return {
        table: store.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "labels",
            "training_label_names",
            "auto_passes",
            "auto_pass_photos",
            "auto_pass_reviews",
        )
    }


def test_sqlite_pass_photos_are_stored_as_blobs_with_their_hashes(tmp_path):
    store = _sqlite(tmp_path)
    try:
        photo = _png("one")
        assert store.record_auto_pass(
            run_id="r",
            app="hinge",
            profile_id="p1",
            score=0.4,
            threshold=0.5,
            embedding=[0.5, 0.25],
            photos=[photo],
            profile_name="Ada",
            random_sample=True,
            capture_truncated=True,
        )
        manifest = store.con.execute(
            "SELECT run_id, app, profile_id, score, threshold, embedding, photo_count,"
            " profile_name, random_sample, capture_truncated FROM auto_passes"
        ).fetchone()
        assert manifest == ("r", "hinge", "p1", 0.4, 0.5, "[0.5, 0.25]", 1, "Ada", 1, 1)
        row = store.con.execute(
            "SELECT run_id, app, profile_id, photo_index, sha256, byte_size, content_type, photo"
            " FROM auto_pass_photos"
        ).fetchone()
        assert row == (
            "r",
            "hinge",
            "p1",
            0,
            hashlib.sha256(photo).hexdigest(),
            len(photo),
            "image/png",
            photo,
        )
    finally:
        store.close()


def test_sqlite_archive_is_one_transaction_a_failing_photo_leaves_no_manifest(tmp_path):
    store = _sqlite(tmp_path)
    try:
        inserted = []

        def fail_second_photo(sql):
            if sql.startswith("INSERT INTO auto_pass_photos"):
                inserted.append(sql)
                if len(inserted) == 2:
                    raise sqlite3.OperationalError("disk I/O error")

        store.con = _HookConnection(store.con, before=fail_second_photo)
        with pytest.raises(sqlite3.OperationalError):
            store.record_auto_pass(
                run_id="r",
                app="hinge",
                profile_id="p1",
                score=0.4,
                threshold=0.5,
                embedding=[0.1],
                photos=[_png("a"), _png("b"), _png("c")],
                random_sample=False,
            )
        assert _counts(store)["auto_passes"] == 0
        assert _counts(store)["auto_pass_photos"] == 0
        # The connection is usable again: nothing is left half-open.
        store.con = store.con._con
        assert store.record_auto_pass(
            run_id="r",
            app="hinge",
            profile_id="p1",
            score=0.4,
            threshold=0.5,
            embedding=[0.1],
            photos=[_png("a")],
            random_sample=False,
        )
    finally:
        store.close()


def test_sqlite_a_conflicting_photo_row_rolls_the_manifest_back_and_returns_false(
    tmp_path, capsys
):
    """The manifest INSERT succeeds first; the photo conflict must take it down with it."""
    store = _sqlite(tmp_path)
    try:
        store.con.execute(
            "INSERT INTO auto_pass_photos (run_id, app, profile_id, created_at, photo_index,"
            " gcs_uri, sha256, byte_size, content_type, photo)"
            " VALUES ('r','hinge','p1',1.0,0,'','x',1,'image/png',x'00')"
        )
        store.con.commit()
        ok = store.record_auto_pass(
            run_id="r",
            app="hinge",
            profile_id="p1",
            score=0.4,
            threshold=0.5,
            embedding=[0.1],
            photos=[_png("a")],
            random_sample=False,
        )
        assert ok is False
        assert "already archived" in capsys.readouterr().out
        assert _counts(store)["auto_passes"] == 0  # no manifest without its photos
        assert store.con.in_transaction is False  # nothing left half-open
    finally:
        store.close()


def test_sqlite_review_is_one_transaction_a_failing_review_row_leaves_no_label(
    tmp_path,
):
    store = _sqlite(tmp_path)
    try:
        store.record_auto_pass(
            run_id="r",
            app="hinge",
            profile_id="p1",
            score=0.4,
            threshold=0.5,
            embedding=[0.1, 0.2],
            photos=[_png("a")],
            profile_name="Ada",
            random_sample=False,
        )

        def fail_review_insert(sql):
            if sql.startswith("INSERT INTO auto_pass_reviews"):
                raise sqlite3.OperationalError("disk I/O error")

        real = store.con
        store.con = _HookConnection(real, before=fail_review_insert)
        with pytest.raises(sqlite3.OperationalError):
            store.record_auto_pass_review(profile_id="p1", would_like=True)
        # The label (and its display-name row) were inserted BEFORE the failing statement; the
        # rollback must have taken them with it. A label without its review would leave the
        # pass pending and a retry would mint a second label.
        counts = _counts(store)
        assert counts["labels"] == 0 and counts["training_label_names"] == 0
        assert counts["auto_pass_reviews"] == 0

        store.con = real
        assert store.record_auto_pass_review(profile_id="p1", would_like=True) == {
            "kind": "near_cut",
            "label_source": "auto_review_near_cut",
        }
        counts = _counts(store)
        assert counts["labels"] == 1 and counts["auto_pass_reviews"] == 1
    finally:
        store.close()


def test_sqlite_a_label_failure_leaves_no_review_row_either(tmp_path):
    store = _sqlite(tmp_path)
    try:
        store.record_auto_pass(
            run_id="r",
            app="hinge",
            profile_id="p1",
            score=0.4,
            threshold=0.5,
            embedding=[0.1],
            photos=[_png("a")],
            random_sample=False,
        )

        def fail_label_name(sql):
            if sql.startswith("INSERT INTO training_label_names"):
                raise sqlite3.OperationalError("disk I/O error")

        real = store.con
        store.con = _HookConnection(real, before=fail_label_name)
        with pytest.raises(sqlite3.OperationalError):
            store.record_auto_pass_review(profile_id="p1", would_like=False)
        counts = _counts(store)
        assert counts["labels"] == 0 and counts["auto_pass_reviews"] == 0
    finally:
        store.close()


def test_sqlite_a_concurrent_answer_from_another_process_wins_cleanly(tmp_path):
    """The UNIQUE index is the authority across processes, and the loser's label rolls back."""
    store = _sqlite(tmp_path)
    try:
        store.record_auto_pass(
            run_id="r",
            app="hinge",
            profile_id="p1",
            score=0.4,
            threshold=0.5,
            embedding=[0.1],
            photos=[_png("a")],
            random_sample=True,
        )
        fired = []

        def other_process_answers_after_the_check(sql):
            if sql.startswith("SELECT 1 FROM auto_pass_reviews") and not fired:
                fired.append(True)
                other = sqlite3.connect(tmp_path / "store.db")
                other.execute(
                    "INSERT INTO auto_pass_reviews (profile_id, app, reviewed_at, would_like, kind)"
                    " VALUES ('p1','hinge',1.0,1,'random')"
                )
                other.commit()
                other.close()

        real = store.con
        store.con = _HookConnection(real, after=other_process_answers_after_the_check)
        with pytest.raises(ValueError, match="already reviewed"):
            store.record_auto_pass_review(profile_id="p1", would_like=False)
        store.con = real
        counts = _counts(store)
        assert counts["labels"] == 0  # the loser did not leave a label behind
        assert counts["auto_pass_reviews"] == 1  # the winner's row only
    finally:
        store.close()


def test_sqlite_unique_indexes_pin_one_archive_and_one_answer_per_pass(tmp_path):
    store = _sqlite(tmp_path)
    try:
        store.record_auto_pass(
            run_id="r",
            app="hinge",
            profile_id="p1",
            score=0.4,
            threshold=0.5,
            embedding=[0.1],
            photos=[_png("a")],
            random_sample=True,
        )
        with pytest.raises(sqlite3.IntegrityError):
            store.con.execute(
                "INSERT INTO auto_passes (run_id, app, profile_id, created_at, score, threshold,"
                " embedding, photo_count, random_sample, capture_truncated)"
                " VALUES ('r2','hinge','p1',1,0,0,'[]',1,1,0)"
            )
        store.con.rollback()
        store.record_auto_pass_review(profile_id="p1", would_like=True)
        with pytest.raises(sqlite3.IntegrityError):
            store.con.execute(
                "INSERT INTO auto_pass_reviews (profile_id, app, reviewed_at, would_like, kind)"
                " VALUES ('p1','hinge',2.0,0,'random')"
            )
    finally:
        store.close()


def test_sqlite_ambiguous_profile_id_across_apps_is_refused(tmp_path):
    store = _sqlite(tmp_path)
    try:
        for app in ("hinge", "bumble"):  # same uuid in two apps cannot be told apart
            store.record_auto_pass(
                run_id="r",
                app=app,
                profile_id="dup",
                score=0.4,
                threshold=0.5,
                embedding=[0.1],
                photos=[_png(app)],
                random_sample=True,
            )
        with pytest.raises(ValueError, match="refusing to guess"):
            store.record_auto_pass_review(profile_id="dup", would_like=True)
        assert _counts(store)["labels"] == 0
    finally:
        store.close()


def test_sqlite_new_methods_read_and_write_only_under_the_store_lock(tmp_path):
    store = _sqlite(tmp_path)
    try:
        unlocked = []

        def watch(sql):
            if not store._lock.locked():
                unlocked.append(sql)

        store.con = _HookConnection(store.con, before=watch)
        store.record_auto_pass(
            run_id="r",
            app="hinge",
            profile_id="p1",
            score=0.4,
            threshold=0.5,
            embedding=[0.1],
            photos=[_png("a")],
            random_sample=True,
        )
        store.next_auto_pass_for_review(kind="random")
        store.next_auto_pass_for_review(kind="near_cut", app="hinge")
        store.auto_pass_photo(profile_id="p1", index=0)
        store.auto_review_stats()
        store.auto_review_stats(app="hinge", window=5)
        store.record_auto_pass_review(profile_id="p1", would_like=True)
        store.load_label_rows()
        assert unlocked == []
    finally:
        store.close()


def test_sqlite_existing_database_without_the_new_tables_gains_them(tmp_path):
    db = tmp_path / "store.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE labels (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, app TEXT,"
        " created_at REAL, liked INTEGER, source TEXT, embedding TEXT, photo_count INTEGER)"
    )
    con.execute(
        "INSERT INTO labels (run_id, app, created_at, liked, source, embedding, photo_count)"
        " VALUES ('r','hinge',1.0,1,NULL,'[1.0]',0)"
    )
    con.commit()
    con.close()

    store = SQLiteStore(db)
    try:
        tables = {
            r[0]
            for r in store.con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"auto_passes", "auto_pass_photos", "auto_pass_reviews"} <= tables
        # And the legacy row (NULL source) is served as "" by the new reader.
        assert store.load_label_rows() == [(True, [1.0], "")]
        # Re-opening is idempotent.
    finally:
        store.close()
    SQLiteStore(db).close()


# ============================================================================================
# BigQuery specifics
# ============================================================================================
class _DdlBQ:
    """Answers the INFORMATION_SCHEMA discovery read and records the DDL script."""

    def __init__(self, existing_tables=()):
        self.existing = set(existing_tables)
        self.scripts: list[str] = []

    def query(self, sql, job_config=None):
        if "INFORMATION_SCHEMA.COLUMNS" in sql:
            return _Job(
                [
                    {"table_name": table, "column_name": col}
                    for table in self.existing
                    for col in _columns(table)
                ]
            )
        self.scripts.append(sql)
        return _Job([])

    def insert_rows_json(self, *_a, **_k):  # pragma: no cover - never reached
        return []


def test_new_tables_are_declared_with_the_spec_columns_and_types():
    assert _TABLES["auto_passes"] == (
        "run_id STRING, app STRING, profile_id STRING, created_at TIMESTAMP, score FLOAT64, "
        "threshold FLOAT64, embedding ARRAY<FLOAT64>, photo_count INT64, profile_name STRING, "
        "random_sample BOOL, capture_truncated BOOL"
    )
    # "same columns as profile_photos": the shared uploader's rows are inserted verbatim.
    assert _TABLES["auto_pass_photos"] == _TABLES["profile_photos"]
    assert _TABLES["auto_pass_reviews"] == (
        "profile_id STRING, app STRING, reviewed_at TIMESTAMP, would_like BOOL, kind STRING"
    )
    # No migration line is needed (and none must exist): they are whole new tables.
    assert not [m for m in bq_module._MIGRATIONS if "auto_pass" in m]


def test_a_project_without_the_new_tables_gets_them_created_and_nothing_else():
    old = [t for t in _TABLES if not t.startswith("auto_pass")]
    client = _DdlBQ(existing_tables=old)
    _bq_store(client, _Storage(), ensure=True)
    script = "\n".join(client.scripts)
    for table in ("auto_passes", "auto_pass_photos", "auto_pass_reviews"):
        assert (
            f"CREATE TABLE IF NOT EXISTS `proj.ds.{table}` ({_TABLES[table]});"
            in script
        )
    assert script.count("CREATE TABLE") == 3
    assert "ALTER TABLE" not in script


def test_a_current_project_runs_no_ddl_at_startup():
    client = _DdlBQ(existing_tables=_TABLES)
    _bq_store(client, _Storage(), ensure=True)
    assert client.scripts == []


@pytest.fixture
def bq():
    client, storage = _SqlBQ(), _Storage()
    store = _bq_store(client, storage)
    yield SimpleNamespace(
        client=client, storage=storage, bucket=storage.bucket("photos"), store=store
    )


def _bq_archive(env, pid, **kwargs):
    kwargs.setdefault("score", 0.4)
    kwargs.setdefault("threshold", 0.5)
    kwargs.setdefault("embedding", [0.1, 0.2])
    kwargs.setdefault("random_sample", False)
    photos = kwargs.pop("photos", 2)
    return env.store.record_auto_pass(
        run_id=kwargs.pop("run_id", "run-1"),
        app=kwargs.pop("app", "hinge"),
        profile_id=pid,
        photos=[_png(f"{pid}-{i}") for i in range(photos)]
        if isinstance(photos, int)
        else photos,
        **kwargs,
    )


def test_bq_archive_uploads_under_the_auto_pass_prefix_and_buffers_both_tables(bq):
    assert _bq_archive(
        bq,
        "p1",
        run_id="run-9",
        app="bumble",
        profile_name="Ada",
        random_sample=True,
        capture_truncated=True,
    )

    names = sorted(bq.bucket.objects)
    assert len(names) == 2
    pattern = re.compile(r"^auto_passes/bumble/run-9/p1/0([01])-[0-9a-f]{16}\.png$")
    assert all(pattern.match(n) for n in names), names
    (manifest,) = bq.store._buf["auto_passes"]
    assert set(manifest) == set(_columns("auto_passes"))
    assert manifest["photo_count"] == 2 and manifest["random_sample"] is True
    assert manifest["capture_truncated"] is True and manifest["profile_name"] == "Ada"
    assert manifest["embedding"] == [0.1, 0.2]
    photos = bq.store._buf["auto_pass_photos"]
    assert [p["photo_index"] for p in photos] == [0, 1]
    for row in photos:
        assert set(row) == set(_columns("profile_photos"))
        data = bq.bucket.objects[row["gcs_uri"].removeprefix("gs://photos/")][0]
        assert row["sha256"] == hashlib.sha256(data).hexdigest() and row[
            "byte_size"
        ] == len(data)
        assert row["content_type"] == "image/png"
    # The training archive and the label/decision streams are untouched.
    for table in ("profiles", "profile_photos", "labels", "decisions"):
        assert bq.store._buf[table] == []

    bq.store.flush()
    assert (
        bq.client.count("auto_passes") == 1 and bq.client.count("auto_pass_photos") == 2
    )
    assert "auto_passes=1" in bq.store.saved_summary()
    assert "auto_pass_photos=2" in bq.store.saved_summary()


def test_bq_record_profile_still_uploads_under_the_profiles_prefix(bq):
    assert bq.store.record_profile(
        "run-1", "hinge", "tp", True, photos=[_png("t")], photo_count=1
    )
    assert [n.split("/")[0] for n in bq.bucket.objects] == ["profiles"]


def test_bq_unverified_store_refuses_the_upload_exactly_like_record_profile():
    client, storage = _SqlBQ(), _Storage()
    store = _bq_store(client, storage, ensure=False)
    with pytest.raises(RuntimeError, match="ensure=False"):
        store.record_auto_pass(
            run_id="r",
            app="hinge",
            profile_id="p1",
            score=0.4,
            threshold=0.5,
            embedding=[0.1],
            photos=[_png("a")],
            random_sample=False,
        )
    assert storage.bucket("photos").upload_log == []
    assert store._buf["auto_passes"] == [] and store._buf["auto_pass_photos"] == []


def test_bq_a_failed_upload_deletes_the_partial_set_and_writes_no_manifest(bq, capsys):
    third = {"n": 0}

    def fail_third(name):
        third["n"] += 1
        return "02-" in name

    bq.bucket.fail_upload = fail_third
    assert _bq_archive(bq, "p1", photos=4) is False
    out = capsys.readouterr().out
    assert "archived 2/4 photos" in out
    assert bq.bucket.objects == {}  # both uploaded blobs were deleted again
    assert len(bq.bucket.deleted) == 2
    assert (
        bq.store._buf["auto_passes"] == [] and bq.store._buf["auto_pass_photos"] == []
    )
    # Not remembered as archived: a later retry may succeed.
    bq.bucket.fail_upload = lambda name: False
    assert _bq_archive(bq, "p1", photos=4) is True


def test_bq_archive_after_close_is_refused_not_silently_lost(bq):
    bq.store.close()
    with pytest.raises(RuntimeError, match="closed"):
        _bq_archive(bq, "p1")
    assert bq.bucket.upload_log == []


def test_bq_incomplete_photo_set_is_never_offered_or_counted(bq):
    """The guard behind `ph.stored >= p.photo_count`: no screenshots, no review."""
    _bq_archive(bq, "whole", random_sample=True, photos=2)
    _bq_archive(bq, "torn", random_sample=True, photos=3)
    bq.store.flush()
    bq.client.con.execute(
        "DELETE FROM \"proj.ds.auto_pass_photos\" WHERE profile_id='torn' AND photo_index=2"
    )  # a dropped insert / failed flush: the manifest promises 3, only 2 exist
    bq.client.con.execute(
        'INSERT INTO "proj.ds.auto_passes" (run_id, app, profile_id, created_at, score, threshold,'
        " embedding, photo_count, profile_name, random_sample, capture_truncated)"
        " VALUES ('r','hinge','orphan',1.0,0.1,0.5,'[0.1]',2,'',1,0)"  # no photo rows at all
    )

    seen = {
        bq.store.next_auto_pass_for_review(kind="random")["profile_id"]
        for _ in range(100)
    }
    assert seen == {"whole"}
    assert bq.store.auto_review_stats()["pending_random"] == 1


def test_bq_reads_flush_this_processes_buffered_archive_first(bq):
    store = bq.store  # flush_every=100: nothing reaches BigQuery on its own
    _bq_archive(bq, "p1", random_sample=True)
    assert bq.client.count("auto_passes") == 0
    assert store.next_auto_pass_for_review(kind="random")["profile_id"] == "p1"
    assert (
        bq.client.count("auto_passes") == 1 and bq.client.count("auto_pass_photos") == 2
    )


def test_bq_a_hub_store_without_bucket_verification_can_review_an_earlier_runs_pass():
    client, storage = _SqlBQ(), _Storage()
    runner = _bq_store(client, storage, ensure=True)
    _bq_archive(SimpleNamespace(store=runner), "p1", random_sample=True)
    runner.close()

    hub = _bq_store(
        client, storage, ensure=False
    )  # read-mostly: bucket privacy not verified
    found = hub.next_auto_pass_for_review(kind="random")
    assert found["profile_id"] == "p1"
    assert hub.auto_pass_photo(profile_id="p1", index=0) == _png(
        "p1-0"
    )  # reads need no verification
    assert hub.record_auto_pass_review(profile_id="p1", would_like=True) == {
        "kind": "random",
        "label_source": "auto_review_random",
    }
    hub.close()
    assert client.count("labels") == 1 and client.count("auto_pass_reviews") == 1
    # ... and a THIRD store (a fresh process) sees it as answered via BigQuery alone.
    third = _bq_store(client, storage, ensure=False)
    with pytest.raises(ValueError, match="already reviewed"):
        third.record_auto_pass_review(profile_id="p1", would_like=False)
    assert client.count("labels") == 1


def test_bq_an_answer_is_final_even_before_bigquery_makes_the_review_row_visible(bq):
    """Streaming inserts lag: the process must remember what it already answered itself."""
    _bq_archive(bq, "p1", random_sample=True)
    bq.store.flush()
    bq.client.lagged = {"auto_pass_reviews", "labels"}
    bq.store.record_auto_pass_review(profile_id="p1", would_like=True)
    assert bq.client.count("auto_pass_reviews") == 0  # accepted, not yet queryable
    with pytest.raises(ValueError, match="already reviewed"):
        bq.store.record_auto_pass_review(profile_id="p1", would_like=False)
    bq.client.settle()
    assert bq.client.count("labels") == 1 and bq.client.count("auto_pass_reviews") == 1


def test_bq_missing_tables_read_as_empty_only_for_a_read_only_store():
    client, storage = _SqlBQ(), _Storage()
    client.missing_tables = {"auto_passes", "auto_pass_photos", "auto_pass_reviews"}
    reader = _bq_store(client, storage, ensure=False)
    assert reader.next_auto_pass_for_review(kind="random") is None
    assert reader.auto_pass_photo(profile_id="p", index=0) is None
    assert reader.auto_review_stats()["pending_random"] == 0
    with pytest.raises(ValueError, match="no archived auto pass"):
        reader.record_auto_pass_review(profile_id="p", would_like=True)

    verified = _bq_store(
        client, storage, ensure=True
    )  # tables must exist here: a real error
    with pytest.raises(NotFound):
        verified.next_auto_pass_for_review(kind="random")
    with pytest.raises(NotFound):
        verified.auto_review_stats()


def test_bq_every_auto_review_query_binds_its_values(bq):
    _bq_archive(bq, "p1", app="hinge", random_sample=True)
    bq.client.queries.clear()
    bq.store.next_auto_pass_for_review(kind="random", app="hinge")
    bq.store.auto_pass_photo(profile_id="p1", index=0)
    bq.store.auto_review_stats(app="hinge", window=7)
    bq.store.record_auto_pass_review(profile_id="p1", would_like=True)

    assert bq.client.queries
    for sql, config in bq.client.queries:
        bound = {p.name: p.value for p in (config.query_parameters if config else [])}
        for value in ("hinge", "p1", 7):
            assert f"'{value}'" not in sql and f'"{value}"' not in sql, sql
        for name in re.findall(r"@(\w+)", sql):
            assert name in bound, (name, sql)
    assert ("review_window", 7) in [
        (p.name, p.value)
        for sql, config in bq.client.queries
        if "LIMIT @review_window" in sql
        for p in config.query_parameters
    ]


# ---- photo reads
def test_bq_photo_read_checks_integrity_and_stays_inside_the_private_bucket(bq):
    _bq_archive(bq, "p1")
    bq.store.flush()
    name = next(iter(bq.bucket.objects))
    assert (
        bq.store.auto_pass_photo(profile_id="p1", index=0) == bq.bucket.objects[name][0]
    )

    # Corrupted object: raise rather than show a picture the sha does not vouch for.
    bq.bucket.objects[name] = (b"tampered", "image/png")
    index = int(re.search(r"/0(\d)-", name).group(1))
    with pytest.raises(RuntimeError, match="sha256"):
        bq.store.auto_pass_photo(profile_id="p1", index=index)

    # A manifest pointing outside the configured bucket is never fetched.
    bq.client.con.execute(
        "UPDATE \"proj.ds.auto_pass_photos\" SET gcs_uri='gs://elsewhere/x.png' WHERE photo_index=0"
    )
    with pytest.raises(RuntimeError, match="outside the configured private bucket"):
        bq.store.auto_pass_photo(profile_id="p1", index=0)


def test_bq_photo_read_of_an_unflushed_archive_and_of_a_vanished_object(bq, capsys):
    _bq_archive(bq, "p1")
    assert bq.client.count("auto_pass_photos") == 0  # still buffered
    assert bq.store.auto_pass_photo(profile_id="p1", index=1) == _png("p1-1")
    for name in list(bq.bucket.objects):
        del bq.bucket.objects[name]
    assert bq.store.auto_pass_photo(profile_id="p1", index=1) is None
    assert "is missing" in capsys.readouterr().out


# ---- ordering, retry and idempotency of the answer
def _writes(client, table):
    return [rows for t, rows in client.insert_log if t == table]


def test_bq_the_label_is_written_before_the_review_row(bq):
    _bq_archive(bq, "p1", random_sample=True)
    bq.store.flush()
    bq.client.insert_log.clear()

    bq.store.record_auto_pass_review(profile_id="p1", would_like=True)

    tables = [t for t, _ in bq.client.insert_log]
    assert tables == ["labels", "auto_pass_reviews"]  # label durable first
    (label,) = _writes(bq.client, "labels")[0]
    assert label["source"] == "auto_review_random" and label["profile_id"] == "p1"
    (review,) = _writes(bq.client, "auto_pass_reviews")[0]
    assert set(review) == set(_columns("auto_pass_reviews"))
    assert review["would_like"] is True and review["kind"] == "random"
    assert isinstance(review["reviewed_at"], str) and review["reviewed_at"].endswith(
        "+00:00"
    )


def test_bq_a_failed_label_write_leaves_no_review_row_and_the_retry_adds_no_second_label(
    bq,
):
    _bq_archive(bq, "p1")
    bq.store.flush()
    bq.client.insert_failures["labels"] = 1

    with pytest.raises(RuntimeError, match="insert errors"):
        bq.store.record_auto_pass_review(profile_id="p1", would_like=True)
    assert bq.store._buf["auto_pass_reviews"] == []  # no review without a durable label
    assert bq.client.count("auto_pass_reviews") == 0
    assert (
        len(bq.store._buf["labels"]) == 1
    )  # the label is queued, awaiting a clean flush

    assert (
        bq.store.record_auto_pass_review(profile_id="p1", would_like=True)["kind"]
        == "near_cut"
    )
    assert (
        bq.client.count("labels") == 1
    )  # exactly one: the retry found the queued label
    assert bq.client.count("auto_pass_reviews") == 1


def test_bq_a_crash_between_label_and_review_is_repaired_by_the_next_process(
    bq, capsys
):
    _bq_archive(bq, "p1", random_sample=True)
    bq.store.flush()
    bq.client.insert_failures["auto_pass_reviews"] = (
        99  # the review row never lands ...
    )
    result = bq.store.record_auto_pass_review(profile_id="p1", would_like=True)
    assert result["kind"] == "random"
    assert "queued but not yet written" in capsys.readouterr().out  # loud, not silent
    assert bq.client.count("labels") == 1 and bq.client.count("auto_pass_reviews") == 0
    # ... and the process dies (its buffer is gone). A fresh process:
    fresh = _bq_store(bq.client, bq.storage, ensure=False)
    assert (
        fresh.next_auto_pass_for_review(kind="random")["profile_id"] == "p1"
    )  # pending again

    # The owner answers again -- and changes their mind. The label that training already used
    # wins; the review row follows it; no second label is ever written.
    bq.client.insert_failures["auto_pass_reviews"] = 0
    fresh.record_auto_pass_review(profile_id="p1", would_like=False)
    out = capsys.readouterr().out
    assert "already has a auto_review_random label" in out and "NOT applied" in out
    assert bq.client.count("labels") == 1
    (label,) = [r for r in bq.client.rows("labels")]
    assert bool(label["liked"]) is True
    (review,) = bq.client.rows("auto_pass_reviews")
    assert bool(review["would_like"]) is True  # agrees with the label training used
    with pytest.raises(ValueError, match="already reviewed"):
        fresh.record_auto_pass_review(profile_id="p1", would_like=False)


def test_bq_a_review_whose_row_flush_failed_is_re_sent_not_refused(bq):
    _bq_archive(bq, "p1")
    bq.store.flush()
    bq.client.insert_failures["auto_pass_reviews"] = 1
    bq.store.record_auto_pass_review(
        profile_id="p1", would_like=True
    )  # queued, printed
    assert bq.client.count("auto_pass_reviews") == 0
    with pytest.raises(
        ValueError, match="already reviewed"
    ):  # no second label meanwhile
        bq.store.record_auto_pass_review(profile_id="p1", would_like=True)
    bq.store.flush()  # the queued row lands exactly once
    assert bq.client.count("auto_pass_reviews") == 1 and bq.client.count("labels") == 1


def test_bq_reviewing_after_close_is_refused(bq):
    _bq_archive(bq, "p1")
    bq.store.flush()
    bq.store.close()
    with pytest.raises(RuntimeError, match="closed"):
        bq.store.record_auto_pass_review(profile_id="p1", would_like=True)


def test_bq_ambiguous_profile_id_across_runs_is_refused(bq):
    _bq_archive(bq, "dup", run_id="run-a")
    bq.store.flush()
    bq.client.insert(
        "auto_passes",
        {
            "run_id": "run-b",
            "app": "hinge",
            "profile_id": "dup",
            "created_at": 2.0,
            "score": 0.1,
            "threshold": 0.5,
            "embedding": [0.1],
            "photo_count": 1,
            "profile_name": "",
            "random_sample": True,
            "capture_truncated": False,
        },
    )
    with pytest.raises(ValueError, match="refusing to guess"):
        bq.store.record_auto_pass_review(profile_id="dup", would_like=True)
    assert bq.client.count("labels") == 0


# ---- label cache coherence
def _label_loads(bq):
    """The queries that (re)build the label cache -- not the per-review label lookup."""
    return [
        q
        for q, _ in bq.client.queries
        if q.startswith("SELECT l.liked, l.embedding, l.source")
    ]


def test_bq_load_label_rows_is_served_from_the_same_cache_as_load_labels(bq):
    bq.store.add_label("run", "hinge", True, [1.0], source="manual", profile_id="m")
    bq.store.flush()
    bq.store._labels_cache = None
    bq.client.queries.clear()

    assert bq.store.load_label_rows() == [(True, [1.0], "manual")]
    assert bq.store.load_labels() == [(True, [1.0])]
    assert len(_label_loads(bq)) == 1

    # A label added AFTER the cache filled lands in it with its source -- retrain-after-every-
    # label depends on this, and neither reader may re-query to see it.
    _bq_archive(bq, "p1", embedding=[0.5, 0.25], random_sample=True)
    bq.store.record_auto_pass_review(profile_id="p1", would_like=True)
    bq.store.add_label("run", "hinge", False, [9.0])  # default source
    assert bq.store.load_label_rows() == [
        (True, [1.0], "manual"),
        (True, [0.5, 0.25], "auto_review_random"),
        (False, [9.0], "manual"),
    ]
    assert bq.store.load_labels() == [
        (True, [1.0]),
        (True, [0.5, 0.25]),
        (False, [9.0]),
    ]
    assert len(_label_loads(bq)) == 1


def test_bq_load_label_rows_folds_in_buffered_labels_and_returns_copies(bq):
    bq.store.add_label(
        "run", "hinge", True, [1.0], source="manual"
    )  # buffered, not flushed
    rows = bq.store.load_label_rows()
    assert rows == [(True, [1.0], "manual")]
    rows[0][1].append(99.0)  # a caller mutating its copy must not poison the cache
    assert bq.store.load_label_rows() == [(True, [1.0], "manual")]


def test_bq_removing_the_latest_label_invalidates_the_row_cache(bq):
    bq.store.add_label("run", "hinge", True, [1.0], source="manual")
    bq.store.load_label_rows()
    assert bq.store._labels_cache is not None
    bq.store._labels_cache = (
        None  # remove_latest_training_label's contract (DML is not modelled here)
    )
    assert bq.store.load_label_rows() == [
        (True, [1.0], "manual")
    ]  # rebuilt, triples intact


def test_bq_dropped_review_rows_surface_in_dropped_rows(bq):
    """The new tables ride the generic dropped/written accounting (no allow-list to extend)."""
    _bq_archive(bq, "p1")
    bq.store.flush()
    bq.client.insert_failures["auto_pass_reviews"] = 999
    bq.store.record_auto_pass_review(profile_id="p1", would_like=True)
    # record_auto_pass_review's own flush was attempt 1; the row is dropped on attempt 5.
    for _ in range(bq_module._MAX_INSERT_ATTEMPTS - 1):
        with pytest.raises(RuntimeError):
            bq.store.flush()
    assert bq.store.dropped_rows() == {"auto_pass_reviews": 1}
    assert "auto_pass_reviews=1" in bq.store.saved_summary()
