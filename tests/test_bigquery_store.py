"""BigQueryStore tests with a fake client — no google SDK or network required."""
import hashlib
import inspect
import math
import re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from operation_love.costing import Usage
from operation_love.ranker import Store
from operation_love.ranker.bigquery_store import BigQueryStore
from operation_love.ranker.store import local_midnight_epoch


class _FakeJob:
    def __init__(self, rows=None):
        self._rows = rows or []

    def result(self):
        return self._rows


class _FakeBQ:
    """Minimal stand-in for google.cloud.bigquery.Client."""

    def __init__(self, label_rows=None, insert_errors=None):
        self.queries = []
        self.job_configs = []          # parallel to `queries`, index-aligned
        self.inserted: dict[str, list[dict]] = {}
        self.row_ids: dict[str, list] = {}
        self._label_rows = label_rows or []
        self._insert_errors = insert_errors or []

    def query(self, sql, job_config=None):
        self.queries.append(sql)
        self.job_configs.append(job_config)
        if sql.strip().upper().startswith("SELECT"):
            return _FakeJob(self._label_rows)
        return _FakeJob()  # DDL

    def insert_rows_json(self, table_id, rows, row_ids=None):
        # Real BigQuery semantics with skip_invalid_rows left unset (the default,
        # False, and what BigQueryStore actually sends): if any row is invalid the
        # WHOLE request fails and NOTHING is written, even though `errors` names only
        # the offending row(s). Model that here rather than the (wrong) "rows not
        # named in errors were accepted" behaviour.
        if self._insert_errors:
            return self._insert_errors
        self.inserted.setdefault(table_id, []).extend(rows)
        self.row_ids.setdefault(table_id, []).extend(row_ids or [])
        return []


class _FakeBlob:
    def __init__(self, name, fail=False):
        self.name = name
        self.data = b""
        self.content_type = ""
        self.public = False
        self.fail = fail
        self.deleted = False

    def upload_from_string(self, data, content_type=None):
        if self.fail:
            raise RuntimeError("simulated GCS upload failure")
        self.data = data
        self.content_type = content_type or ""

    def delete(self):
        self.deleted = True

    def make_public(self):
        self.public = True


class _FakeIam:
    def __init__(self):
        self.uniform_bucket_level_access_enabled = False
        self.public_access_prevention = "inherited"


class _FakeBucket:
    def __init__(self, name, exists=True, fail_uploads=False):
        self.name = name
        self._exists = exists
        self.fail_uploads = fail_uploads
        self.blobs: dict[str, _FakeBlob] = {}
        self.iam_configuration = _FakeIam()
        self.patched = False

    def exists(self):
        return self._exists

    def blob(self, name):
        self.blobs.setdefault(name, _FakeBlob(name, fail=self.fail_uploads))
        return self.blobs[name]

    def patch(self):
        self.patched = True

    def reload(self):
        pass


class _FakeStorage:
    def __init__(self, bucket_exists=True, fail_uploads=False):
        self.buckets: dict[str, _FakeBucket] = {}
        self.created = []
        self._bucket_exists = bucket_exists
        self._fail_uploads = fail_uploads

    def bucket(self, name):
        self.buckets.setdefault(
            name, _FakeBucket(name, exists=self._bucket_exists, fail_uploads=self._fail_uploads))
        return self.buckets[name]

    def create_bucket(self, bucket, location=None):
        bucket._exists = True
        self.buckets[bucket.name] = bucket
        self.created.append((bucket.name, location))
        return bucket


def _store(client, flush_every=25):
    return BigQueryStore("proj", "ds", photo_bucket="photos", flush_every=flush_every,
                         client=client, storage_client=_FakeStorage(), ensure=False)


@pytest.mark.parametrize("value", [True, math.nan, math.inf, -math.inf, 1e300, 10 ** 10_000,
                                   "2026-01-01T00:00:00"],
                         ids=["bool", "nan", "inf", "negative-inf", "out-of-range-float",
                              "huge-int", "naive-iso"])
def test_bigquery_action_writes_reject_invalid_timestamp_values_without_overflow(value):
    store = _store(_FakeBQ(), flush_every=100)
    with pytest.raises(ValueError, match="timestamp"):
        store.record_decision("r", "hinge", "like", 1.0, created_at=value)
    with pytest.raises(ValueError, match="timestamp"):
        store.record_opener(
            "r", "hinge", "model", "opener", "reference",
            decision_created_at=value,
        )
    assert store._buf["decisions"] == []
    assert store._buf["openers"] == []


def test_bigquery_action_timestamp_normalizes_timezone_aware_iso_text():
    store = _store(_FakeBQ(), flush_every=100)
    store.record_decision("r", "hinge", "like", 1.0,
                          created_at="2026-01-01T00:00:00Z")
    assert store._buf["decisions"][0]["created_at"] == "2026-01-01T00:00:00+00:00"


def _schema_rows_without(*missing: tuple[str, str]) -> list[dict[str, str]]:
    """INFORMATION_SCHEMA rows for the current schema minus selected legacy columns."""
    from operation_love.ranker.bigquery_store import _TABLES

    excluded = set(missing)
    return [
        {"table_name": table, "column_name": declaration.strip().split()[0]}
        for table, columns in _TABLES.items()
        for declaration in columns.split(",")
        if (table, declaration.strip().split()[0]) not in excluded
    ]


class _SchemaAwareBQ(_FakeBQ):
    """Small schema-state fake used to verify repeated startup, not BigQuery itself."""

    def __init__(self, *, dataset_exists=True, missing=()):
        super().__init__()
        self.dataset_exists = dataset_exists
        rows = _schema_rows_without(*missing) if dataset_exists else []
        self.schema: dict[str, set[str]] = {}
        for row in rows:
            self.schema.setdefault(row["table_name"], set()).add(row["column_name"])

    def query(self, sql, job_config=None):
        self.queries.append(sql)
        self.job_configs.append(job_config)
        if "INFORMATION_SCHEMA.COLUMNS" in sql:
            if not self.dataset_exists:
                class NotFound(Exception):
                    pass
                raise NotFound("Not found: Dataset proj:ds")
            rows = [{"table_name": table, "column_name": column}
                    for table, columns in self.schema.items() for column in columns]
            return _FakeJob(rows)
        if "CREATE SCHEMA IF NOT EXISTS" in sql:
            self.dataset_exists = True
            return _FakeJob()
        for match in re.finditer(
                r"CREATE TABLE IF NOT EXISTS `[^`]+\.(?P<table>[a-z_]+)` "
                r"\((?P<columns>[^;]+)\);", sql):
            self.schema.setdefault(match["table"], set()).update(
                declaration.strip().split()[0]
                for declaration in match["columns"].split(",")
            )
        for match in re.finditer(
                r"ALTER TABLE `[^`]+\.(?P<table>[a-z_]+)`\n(?P<clauses>[^;]+);", sql):
            self.schema.setdefault(match["table"], set()).update(
                re.findall(r"ADD COLUMN IF NOT EXISTS ([A-Za-z_][A-Za-z0-9_]*)", match["clauses"])
            )
        return _FakeJob()


def test_bigquery_observe_release_summary_is_bound_and_outcome_specific(monkeypatch):
    """The offline AUTO gate may read aggregate counts, never rows or interpolated run IDs."""
    client = _FakeBQ(label_rows=[{
        "pass_labels": 1, "like_labels": 1,
        "pass_decisions": 1, "like_decisions": 1,
        "successful_hinge_openers": 1,
    }])
    store = _store(client)

    class _Param:
        def __init__(self, name, _kind, value):
            self.name, self.value = name, value

    class _Job:
        def __init__(self, query_parameters):
            self.query_parameters = query_parameters

    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "google.cloud", SimpleNamespace(
        bigquery=SimpleNamespace(ScalarQueryParameter=_Param, QueryJobConfig=_Job)))

    assert store.observe_release_persistence_summary("run'unsafe", "hinge") == {
        "manual_pass_labels": 1,
        "manual_like_labels": 1,
        "manual_pass_decisions": 1,
        "manual_like_decisions": 1,
        "successful_hinge_openers": 1,
    }
    sql = client.queries[-1]
    assert "run'unsafe" not in sql
    assert "liked=FALSE" in sql and "liked=TRUE" in sql
    assert "decision='dislike'" in sql and "decision='like'" in sql
    assert "opener_rejections" not in sql and "spend" not in sql
    assert [(p.name, p.value) for p in client.job_configs[-1].query_parameters] == [
        ("run_id", "run'unsafe"), ("app", "hinge")]


def test_read_only_summary_treats_only_missing_opener_retractions_as_empty(monkeypatch):
    import sys
    from types import SimpleNamespace

    class _Param:
        def __init__(self, name, _kind, value): self.name, self.value = name, value

    class _JobConfig:
        def __init__(self, query_parameters): self.query_parameters = query_parameters

    class NotFound(Exception):
        pass

    class _MissingOptional(_FakeBQ):
        def query(self, sql, job_config=None):
            self.queries.append(sql)
            self.job_configs.append(job_config)
            if "opener_retractions" in sql:
                raise NotFound("Not found: proj.ds.opener_retractions")
            return _FakeJob([{
                "pass_labels": 0, "like_labels": 0, "pass_decisions": 0,
                "like_decisions": 0, "successful_hinge_openers": 1,
            }])

    monkeypatch.setitem(sys.modules, "google.cloud", SimpleNamespace(
        bigquery=SimpleNamespace(ScalarQueryParameter=_Param, QueryJobConfig=_JobConfig)))
    store = _store(_MissingOptional())
    assert store.observe_release_persistence_summary("run", "hinge")["successful_hinge_openers"] == 1
    assert "opener_retractions" not in store.client.queries[-1]

    class _MissingRequired(_MissingOptional):
        def query(self, sql, job_config=None):
            raise NotFound("Not found: proj.ds.labels")

    with pytest.raises(NotFound):
        _store(_MissingRequired()).observe_release_persistence_summary("run", "hinge")


def test_bigquery_two_opener_tombstones_are_per_row_idempotent(monkeypatch):
    import sys
    from types import SimpleNamespace
    from operation_love.ranker.retractions import canonical_sha

    class _Param:
        def __init__(self, name, _kind, value): self.name, self.value = name, value
    class _JobConfig:
        def __init__(self, query_parameters): self.query_parameters = query_parameters
    monkeypatch.setitem(sys.modules, "google.cloud", SimpleNamespace(
        bigquery=SimpleNamespace(ScalarQueryParameter=_Param, QueryJobConfig=_JobConfig)))

    stamp_a = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stamp_b = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    sources = {stamp_a: ("gemini-x", "first"), stamp_b: ("gemini-y", "second")}
    class _RowsBQ(_FakeBQ):
        def __init__(self):
            super().__init__()
            self.tombstones = []
        def query(self, sql, job_config=None):
            params = {item.name: item.value for item in (job_config.query_parameters or [])}
            if "FROM `proj.ds.openers`" in sql:
                model, opener = sources[params["opener_created_at"]]
                return _FakeJob([{"created_at": params["opener_created_at"], "model": model, "opener": opener}])
            if "FROM `proj.ds.opener_retractions`" in sql:
                rows = self.tombstones
                if "correction_id" in params:
                    rows = [row for row in rows if row["correction_id"] == params["correction_id"]]
                if "opener_created_at" in params:
                    rows = [row for row in rows if row["opener_created_at"] == params["opener_created_at"].isoformat()]
                return _FakeJob(rows)
            return _FakeJob()
        def insert_rows_json(self, _table, rows, row_ids=None):
            self.tombstones.extend(rows)
            return []

    client = _RowsBQ()
    store = _store(client)
    for stamp, (model, opener) in sources.items():
        row = {"correction_id": "same-plan", "run_id": "run", "app": "hinge",
               "opener_created_at": stamp.isoformat(), "model": model,
               "opener_fingerprint": canonical_sha({"run_id": "run", "app": "hinge",
                   "created_at": stamp.isoformat(), "model": model, "opener": opener}),
               "reason": "unacted", "evidence_ref": "debug"}
        assert store.append_opener_retraction(row) is True
    assert len(client.tombstones) == 2
    assert store.append_opener_retraction(row) is False


def test_bigquery_retraction_rows_normalize_timestamps_and_append_is_bound_idempotent(monkeypatch):
    import sys
    from types import SimpleNamespace

    class _Param:
        def __init__(self, name, _kind, value):
            self.name, self.value = name, value

    class _Job:
        def __init__(self, query_parameters):
            self.query_parameters = query_parameters

    monkeypatch.setitem(sys.modules, "google.cloud", SimpleNamespace(
        bigquery=SimpleNamespace(ScalarQueryParameter=_Param, QueryJobConfig=_Job)))

    class _RowsBQ(_FakeBQ):
        def query(self, sql, job_config=None):
            self.queries.append(sql)
            self.job_configs.append(job_config)
            if "FROM `proj.ds.profiles`" in sql:
                return _FakeJob([{"profile_id": "profile"}])
            if "FROM `proj.ds.labels`" in sql:
                return _FakeJob([{"profile_id": "profile", "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
                                  "liked": False}])
            if "FROM `proj.ds.decisions`" in sql:
                return _FakeJob([{"profile_id": "profile",
                                  "created_at": datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                                  "decision": "dislike", "score": 0.0}])
            return _FakeJob([])

    client = _RowsBQ()
    store = _store(client)
    rows = store.retraction_run_rows("run'unsafe", "hinge", "external_ai_review")
    assert rows["labels"][0]["created_at"] == "2026-01-01T00:00:00+00:00"
    assert rows["decisions"][0]["profile_id"] == "profile"
    assert "run'unsafe" not in client.queries[-1]
    assert [(p.name, p.value) for p in client.job_configs[-1].query_parameters] == [
        ("run_id", "run'unsafe"), ("app", "hinge"), ("source", "external_ai_review")]
    row = {"correction_id": "c", "run_id": "run'unsafe", "app": "hinge", "source": "external_ai_review",
           "profile_id": "profile", "label_created_at": "2026-01-01T00:00:00+00:00",
           "decision_created_at": "2026-01-01T00:00:01+00:00", "decision_fingerprint": "fp",
           "reason": "false", "evidence_ref": "debug#225", "created_at": "2026-01-01T00:01:00+00:00"}
    assert store.append_label_retraction(row) is True
    assert store.append_label_retraction(row) is False
    target_job = next(config for sql, config in zip(
                      client.queries, client.job_configs, strict=True)
                      if "label_created_at=@label_created_at" in sql)
    assert [(p.name, p.value) for p in target_job.query_parameters][-2:] == [
        ("label_created_at", datetime(2026, 1, 1, tzinfo=timezone.utc)),
        ("decision_created_at", datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)),
    ]
    inserted = client.inserted["proj.ds.label_retractions"]
    assert inserted[0]["correction_id"] == "c"
    assert client.row_ids["proj.ds.label_retractions"] == ["c"]


def _declared_columns(table):
    """The column NAMES _TABLES declares for `table`, in declaration order.

    Every type in this schema is a single token (STRING / INT64 / BOOL / TIMESTAMP /
    FLOAT64) or ARRAY<FLOAT64>, none of which contain a comma, so splitting the column
    spec on ',' and taking each declaration's first whitespace-separated token is exact.
    (It would not be if a nested STRUCT<a INT64, b INT64> ever appeared here.)"""
    from operation_love.ranker.bigquery_store import _TABLES

    return [col.strip().split()[0] for col in _TABLES[table].split(",")]


def _declared_tables():
    """Every table name _TABLES declares — i.e. exactly the keys _ensure_tables formats the
    `{table}` placeholders in _MIGRATIONS against."""
    from operation_love.ranker.bigquery_store import _TABLES

    return list(_TABLES)


def test_bigquery_store_conforms_to_store_protocol():
    assert isinstance(_store(_FakeBQ()), Store)


def test_ensure_false_skips_schema_discovery_and_mutation():
    client = _FakeBQ()

    _store(client)

    assert client.queries == []


def test_load_labels_parses_and_counts():
    client = _FakeBQ(label_rows=[{"liked": True, "embedding": [0.1, 0.2]},
                                 {"liked": False, "embedding": [0.3]}])
    s = _store(client)
    labels = s.load_labels()
    assert labels == [(True, [0.1, 0.2]), (False, [0.3])]


def test_buffer_flushes_at_threshold():
    client = _FakeBQ()
    s = _store(client, flush_every=2)
    s.add_label("r", "bumble", True, [0.1])
    assert "proj.ds.labels" not in client.inserted          # buffered, not yet sent
    s.add_label("r", "bumble", False, [0.2])
    assert len(client.inserted["proj.ds.labels"]) == 2       # flushed at threshold


def test_add_label_includes_profile_id():
    client = _FakeBQ()
    s = _store(client, flush_every=1)
    s.add_label("r", "bumble", True, [0.1], profile_id="profile-1")

    assert client.inserted["proj.ds.labels"][0]["profile_id"] == "profile-1"


def test_ensure_tables_runs_decision_source_migration():
    client = _FakeBQ(label_rows=_schema_rows_without(
        ("labels", "profile_id"), ("decisions", "source")))
    BigQueryStore("proj", "ds", photo_bucket="photos", client=client,
                  storage_client=_FakeStorage(), ensure=True)
    ddl = "\n".join(client.queries)
    assert "ALTER TABLE `proj.ds.labels`\nADD COLUMN IF NOT EXISTS profile_id STRING" in ddl
    assert "ALTER TABLE `proj.ds.decisions`\nADD COLUMN IF NOT EXISTS source STRING" in ddl


def test_ensure_tables_fresh_project_creates_current_schema_once_without_alters():
    client = _SchemaAwareBQ(dataset_exists=False)

    BigQueryStore("proj", "ds", photo_bucket="photos", client=client,
                  storage_client=_FakeStorage(), ensure=True)
    first_start = "\n".join(client.queries)
    assert "CREATE SCHEMA IF NOT EXISTS `proj.ds`" in first_start
    for table in _declared_tables():
        assert f"CREATE TABLE IF NOT EXISTS `proj.ds.{table}`" in first_start
    assert "ALTER TABLE" not in first_start

    before = len(client.queries)
    BigQueryStore("proj", "ds", photo_bucket="photos", client=client,
                  storage_client=_FakeStorage(), ensure=True)
    second_start = "\n".join(client.queries[before:])
    assert "INFORMATION_SCHEMA.COLUMNS" in second_start
    assert "CREATE SCHEMA" not in second_start
    assert "CREATE TABLE" not in second_start
    assert "ALTER TABLE" not in second_start


def test_ensure_tables_groups_all_missing_opener_columns_into_one_update():
    from operation_love.ranker.bigquery_store import _MIGRATIONS, _migration_parts

    opener_columns = [column for statement in _MIGRATIONS
                      for table, column, _kind in [_migration_parts(statement)]
                      if table == "openers"]
    assert len(opener_columns) > 5  # reproduces the reported per-table quota failure
    client = _SchemaAwareBQ(missing=[("openers", column) for column in opener_columns])

    BigQueryStore("proj", "ds", photo_bucket="photos", client=client,
                  storage_client=_FakeStorage(), ensure=True)
    mutation = "\n".join(query for query in client.queries if "ALTER TABLE" in query)
    assert mutation.count("ALTER TABLE `proj.ds.openers`") == 1
    assert mutation.count("ADD COLUMN IF NOT EXISTS") == len(opener_columns)
    for column in opener_columns:
        assert f"ADD COLUMN IF NOT EXISTS {column} " in mutation

    before = len(client.queries)
    BigQueryStore("proj", "ds", photo_bucket="photos", client=client,
                  storage_client=_FakeStorage(), ensure=True)
    second_start = "\n".join(client.queries[before:])
    assert "CREATE TABLE" not in second_start
    assert "ALTER TABLE" not in second_start


def test_ensure_tables_retries_transient_table_update_quota(monkeypatch):
    class _RateLimitedJob:
        def result(self):
            raise RuntimeError(
                "Exceeded rate limits: too many table update operations for this table")

    class _RateLimitedOnceBQ(_FakeBQ):
        def __init__(self):
            super().__init__(label_rows=_schema_rows_without(("openers", "angle")))
            self.update_attempts = 0

        def query(self, sql, job_config=None):
            if "ALTER TABLE" not in sql:
                return super().query(sql, job_config=job_config)
            self.queries.append(sql)
            self.job_configs.append(job_config)
            self.update_attempts += 1
            return _RateLimitedJob() if self.update_attempts == 1 else _FakeJob()

    sleeps = []
    monkeypatch.setattr("operation_love.ranker.bigquery_store.time.sleep", sleeps.append)
    client = _RateLimitedOnceBQ()

    BigQueryStore("proj", "ds", photo_bucket="photos", client=client,
                  storage_client=_FakeStorage(), ensure=True)

    assert client.update_attempts == 2
    assert sleeps == [1.0]


def test_record_decision_includes_source():
    client = _FakeBQ()
    s = _store(client, flush_every=1)
    s.record_decision("r", "bumble", "like", 0.91, source="manual")

    row = client.inserted["proj.ds.decisions"][0]
    assert row["source"] == "manual"
    assert row["decision"] == "like" and row["score"] == 0.91


def test_record_opener_buffers_and_flushes_the_expected_row_including_angle():
    referenced = "Prompt card: two truths and a lie about 30 countries, cilantro, and a pop star"
    angle = "guess which of her three claims is the lie and commit to it"
    client = _FakeBQ()
    s = _store(client, flush_every=2)

    s.record_opener("r", "hinge", "gemini-x", "The cilantro one is the lie, I can feel it.",
                    referenced, angle)
    assert "proj.ds.openers" not in client.inserted          # buffered, not yet sent
    s.record_opener("r", "hinge", "gemini-x", "second opener", "second referenced", "tease")

    row = client.inserted["proj.ds.openers"][0]              # flushed at threshold
    assert row["run_id"] == "r" and row["app"] == "hinge" and row["model"] == "gemini-x"
    assert row["opener"] == "The cilantro one is the lie, I can feel it."
    assert row["referenced"] == referenced
    # `angle` is the model's own free-text words for what the opener is DOING. It is stored
    # ALONGSIDE (never folded into) `referenced` -- what the opener is reacting to -- and
    # `opener` -- the text actually sent -- because they answer three different questions.
    # Only this column makes "which opener shapes correlate with matches" an answerable query,
    # and it is telemetry only: nothing reads it back at runtime.
    assert row["angle"] == angle
    assert row["angle"] != row["referenced"] and row["angle"] != row["opener"]


def test_record_opener_defaults_angle_to_empty_string_and_never_omits_the_field():
    """A caller that predates `angle` (or a model response that carried none) must still write
    a COMPLETE row: the key is always present holding "", never absent. An absent key inserts
    as NULL, which is indistinguishable from the rows written before the column existed --
    "this run produced no angle" and "this row predates angle" are different facts and must
    stay separable in the system of record."""
    client = _FakeBQ()
    s = _store(client, flush_every=1)

    s.record_opener("r", "hinge", "gemini-x", "Based on that ridgeline I am going to guess Norway.",
                    "Photo of her on a ridge with mountains behind her")

    row = client.inserted["proj.ds.openers"][0]
    assert "angle" in row
    assert row["angle"] == ""
    # Same rule for item_description, and for exactly the same reason: absent-vs-"" is the
    # difference between "this row predates the column" and "this generation produced none".
    assert "item_description" in row
    assert row["item_description"] == ""


def test_record_opener_writes_item_description_alongside_referenced_and_angle():
    """ops/OPENER-REDESIGN.md 5.7: `item_description` describes the ITEM the model picked,
    `referenced` is the DETAIL the opener reacts to, and the doc is explicit that neither
    replaces the other -- dropping `referenced` would blank the telemetry column and take the
    redundancy monitor (3.7) dark. So the row must carry all three as separate values."""
    client = _FakeBQ()
    s = _store(client, flush_every=1)

    s.record_opener("r", "hinge", "gemini-x", "The cilantro one is the lie, I can feel it.",
                    "her two truths and a lie, the cilantro claim",
                    "calling the lie and committing to it",
                    "a written prompt card, two truths and a lie")

    row = client.inserted["proj.ds.openers"][0]
    assert row["item_description"] == "a written prompt card, two truths and a lie"
    assert row["referenced"] == "her two truths and a lie, the cilantro claim"
    assert row["angle"] == "calling the lie and committing to it"
    # Three columns answering three different questions -- what the item IS, what the opener
    # is reacting to, and what the opener is doing. Any two of them being equal would mean the
    # positional call site had bound the wrong argument.
    assert len({row["item_description"], row["referenced"], row["angle"]}) == 3


def test_record_opener_writes_exactly_the_columns_the_openers_table_declares():
    """Structural guard against the two ways this table silently drifts: writing a field the
    table does not declare (BigQuery rejects the insert as "no such field" and, with
    skip_invalid_rows unset, loses the WHOLE batch with it), or declaring a column nothing ever
    populates. Key ORDER is deliberately not pinned: rows go out as JSON objects, so their key
    order carries no meaning to BigQuery and pinning it would assert something untrue."""
    client = _FakeBQ()
    s = _store(client, flush_every=1)

    s.record_opener("r", "hinge", "gemini-x", "hi", "her ridgeline photo", "guess")

    assert "angle" in _declared_columns("openers")           # the column exists to be written
    assert set(client.inserted["proj.ds.openers"][0]) == set(_declared_columns("openers"))


def test_auto_opener_evidence_uploads_private_png_and_buffers_queryable_linkage():
    client = _FakeBQ()
    storage = _FakeStorage()
    store = BigQueryStore(
        "proj", "ds", photo_bucket="photos", flush_every=1,
        client=client, storage_client=storage, ensure=True)
    frame = b"\x89PNG\r\n\x1a\npre-send"
    opener = "Quiet trail first, then coffee?"
    evidence_id = hashlib.sha256(frame + b"\0" + opener.encode()).hexdigest()

    assert store.record_opener_send_evidence(
        "r", "hinge", opener, profile_id="profile-1", decision_source="auto",
        decision_created_at=123.0, model_item_index=3,
        evidence={"frame": frame, "evidence_id": evidence_id}) is True

    row = client.inserted["proj.ds.opener_send_evidence"][0]
    assert row["opener"] == opener
    assert row["evidence_id"] == evidence_id
    assert row["profile_id"] == "profile-1"
    assert row["model_item_index"] == 3
    assert row["outcome"] == "like_landed"
    assert row["decision_created_at"] == "1970-01-01T00:02:03+00:00"
    assert row["gcs_uri"].startswith(
        "gs://photos/opener-evidence/hinge/r/profile-1/")
    object_name = row["gcs_uri"].removeprefix("gs://photos/")
    blob = storage.buckets["photos"].blobs[object_name]
    assert blob.data == frame and blob.content_type == "image/png"
    assert set(row) == set(_declared_columns("opener_send_evidence"))


def test_auto_opener_evidence_upload_failure_does_not_create_a_dangling_bq_row():
    client = _FakeBQ()
    storage = _FakeStorage(fail_uploads=True)
    store = BigQueryStore(
        "proj", "ds", photo_bucket="photos", flush_every=1,
        client=client, storage_client=storage, ensure=True)

    assert store.record_opener_send_evidence(
        "r", "hinge", "opener", profile_id="p",
        evidence={"frame": b"\x89PNG\r\n\x1a\nframe"}) is False
    assert "proj.ds.opener_send_evidence" not in client.inserted


def test_record_opener_signature_stays_positional_compatible_with_the_store_protocol():
    """`angle` and `item_description` are TRAILING with "" defaults in both the Store Protocol
    and this backend, so every existing five-positional-argument caller (and every test double
    implementing the Protocol) keeps working untouched. A backend that inserted either
    parameter anywhere earlier would silently bind it to `opener`/`referenced` at those call
    sites and write garbage to the system of record rather than failing loudly.

    ORDER between the two trailing parameters is pinned, not incidental: opener/service.py
    passes both POSITIONALLY (`record_opener(..., referenced, angle, item_description)`), so
    swapping them here would file every angle under item_description and vice versa, in the
    system of record, with no error anywhere."""
    params = inspect.signature(BigQueryStore.record_opener).parameters
    assert list(params) == ["self", "run_id", "app", "model", "opener", "referenced", "angle",
                            "item_description", "profile_id", "decision", "decision_source",
                            "decision_created_at", "model_item_index"]
    assert params["angle"].default == ""
    assert params["item_description"].default == ""
    for name in ("profile_id", "decision", "decision_source", "decision_created_at",
                 "model_item_index"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert list(params) == list(inspect.signature(Store.record_opener).parameters)


def test_ensure_tables_declares_angle_on_the_openers_create_table():
    """The CREATE half of the two-part column rollout: this is what reaches a NEW/empty
    project, where the table does not exist yet. It is NOT what fixes the live project -- see
    test_ensure_tables_runs_openers_angle_migration for that half. Both are required."""
    client = _FakeBQ()
    BigQueryStore("proj", "ds", photo_bucket="photos", client=client,
                  storage_client=_FakeStorage(), ensure=True)

    ddl = "\n".join(client.queries)
    # Match the openers CREATE statement specifically, not the whole script: shared column
    # names like "run_id STRING" appear in every table, so a substring test against the full
    # script would pass even if the openers column list were missing them entirely.
    create = next(line for line in ddl.splitlines()
                  if line.startswith("CREATE TABLE IF NOT EXISTS `proj.ds.openers`"))
    for col in ("run_id STRING", "app STRING", "created_at TIMESTAMP", "model STRING",
                "opener STRING", "referenced STRING", "angle STRING",
                "item_description STRING"):
        assert col in create


def test_ensure_tables_runs_openers_angle_migration():
    """The production-critical half. The live `openers` table ALREADY EXISTS and already holds
    real rows from live runs, so CREATE TABLE IF NOT EXISTS is a silent no-op against it: the
    ALTER below is the ONLY thing that ever puts `angle` on the production table. Without it,
    the first insert carrying the field is rejected as "no such field: angle" in production --
    taking the whole batch with it, since skip_invalid_rows is unset -- while every local test
    (which always creates the table fresh) keeps passing. Pin BOTH halves: that the migration
    is declared, and that it actually reaches the script _ensure_tables submits."""
    from operation_love.ranker.bigquery_store import _MIGRATIONS

    assert "ALTER TABLE `{openers}` ADD COLUMN IF NOT EXISTS angle STRING;" in _MIGRATIONS

    client = _FakeBQ(label_rows=_schema_rows_without(("openers", "angle")))
    BigQueryStore("proj", "ds", photo_bucket="photos", client=client,
                  storage_client=_FakeStorage(), ensure=True)
    ddl = "\n".join(client.queries)
    assert "ALTER TABLE `proj.ds.openers`\nADD COLUMN IF NOT EXISTS angle STRING;" in ddl
    assert "CREATE TABLE IF NOT EXISTS `proj.ds.openers`" not in ddl


def test_ensure_tables_runs_openers_item_description_migration():
    """The same production-critical half for `item_description` (ops/OPENER-REDESIGN.md 5.7).
    Pinned SEPARATELY from angle's migration rather than folded into it: the live `openers`
    table holds real rows, so CREATE TABLE IF NOT EXISTS never touches it, and a column that
    exists only in _TABLES is a column that does not exist in production. The first insert
    carrying it would then be rejected as "no such field: item_description" and take the whole
    batch with it (skip_invalid_rows is unset), while every local test -- which always creates
    the table fresh -- keeps passing."""
    from operation_love.ranker.bigquery_store import _MIGRATIONS

    assert ("ALTER TABLE `{openers}` ADD COLUMN IF NOT EXISTS item_description STRING;"
            in _MIGRATIONS)

    client = _FakeBQ(label_rows=_schema_rows_without(("openers", "item_description")))
    BigQueryStore("proj", "ds", photo_bucket="photos", client=client,
                  storage_client=_FakeStorage(), ensure=True)
    ddl = "\n".join(client.queries)
    assert ("ALTER TABLE `proj.ds.openers`\nADD COLUMN IF NOT EXISTS item_description STRING;"
            in ddl)
    assert "CREATE TABLE IF NOT EXISTS `proj.ds.openers`" not in ddl


def test_ensure_tables_creates_opener_rejections_table():
    """opener_rejections persists every REJECTED opener attempt (OpenerParseError), not just
    the successes the `openers` table holds -- see ranker/bigquery_store.py's _TABLES entry
    and opener/service.py's OpenerParseError handling."""
    client = _FakeBQ()
    BigQueryStore("proj", "ds", photo_bucket="photos", client=client,
                  storage_client=_FakeStorage(), ensure=True)
    ddl = "\n".join(client.queries)
    assert "CREATE TABLE IF NOT EXISTS `proj.ds.opener_rejections`" in ddl
    for col in ("run_id STRING", "app STRING", "created_at TIMESTAMP", "model STRING",
                "attempt INT64", "reason_code STRING", "reason STRING", "raw_opener STRING"):
        assert col in ddl


def test_record_opener_rejection_buffers_and_flushes_the_expected_row():
    client = _FakeBQ()
    s = _store(client, flush_every=1)

    s.record_opener_rejection("r", "hinge", "gemini-x", 2, "scaffolding",
                              "Gemini's opener contained scaffolding text", "Here's: hi")

    row = client.inserted["proj.ds.opener_rejections"][0]
    assert row["run_id"] == "r" and row["app"] == "hinge" and row["model"] == "gemini-x"
    assert row["attempt"] == 2 and row["reason_code"] == "scaffolding"
    assert row["reason"] == "Gemini's opener contained scaffolding text"
    assert row["raw_opener"] == "Here's: hi"


def test_record_opener_rejection_accepts_none_raw_opener_and_reason_code():
    """no_text/max_tokens rejections carry no candidate opener text at all -- raw_opener (and,
    for an external caller that doesn't classify a failure, reason_code) may be None; this
    must serialize fine as a NULLable column, not raise."""
    client = _FakeBQ()
    s = _store(client, flush_every=1)

    s.record_opener_rejection("r", "hinge", "gemini-x", 1, None, "no text content", None)

    row = client.inserted["proj.ds.opener_rejections"][0]
    assert row["reason_code"] is None and row["raw_opener"] is None


def test_count_today_counts_the_requested_decision_source():
    client = _FakeBQ(label_rows=[{"c": 7}])
    s = _store(client)

    assert s.count_today("bumble") == 7
    assert "AND source=@source" in client.queries[-1]
    assert {p.name: p.value for p in client.job_configs[-1].query_parameters}["source"] == "auto"
    assert s.count_today("bumble", source="manual") == 7
    assert {p.name: p.value for p in client.job_configs[-1].query_parameters}["source"] == "manual"
    with pytest.raises(ValueError, match="source"):
        s.count_today("bumble", source="automation")


def test_record_profile_uploads_photos_and_manifest_rows():
    client = _FakeBQ()
    storage = _FakeStorage()
    s = BigQueryStore("proj", "ds", photo_bucket="photos", flush_every=100,
                      client=client, storage_client=storage, ensure=True)
    png = b"\x89PNG\r\n\x1a\nprofile"
    jpg = b"\xff\xd8\xffprofile"

    s.record_profile("r", "bumble", "profile-1", True, photos=[png, jpg], photo_count=2)
    s.flush()

    profile = client.inserted["proj.ds.profiles"][0]
    assert profile["profile_id"] == "profile-1"
    assert profile["liked"] is True
    assert "bio" not in profile and "prompts" not in profile

    photo_rows = client.inserted["proj.ds.profile_photos"]
    assert [r["photo_index"] for r in photo_rows] == [0, 1]
    assert all(r["profile_id"] == "profile-1" for r in photo_rows)
    assert photo_rows[0]["gcs_uri"].startswith("gs://photos/profiles/bumble/r/profile-1/")
    assert photo_rows[0]["content_type"] == "image/png"
    assert photo_rows[1]["content_type"] == "image/jpeg"

    blobs = storage.bucket("photos").blobs
    assert len(blobs) == 2
    assert all(not blob.public for blob in blobs.values())


def test_ensure_false_store_refuses_profile_photos_without_touching_bucket():
    client = _FakeBQ()
    storage = _FakeStorage()
    store = BigQueryStore(
        "proj", "ds", photo_bucket="photos", client=client,
        storage_client=storage, ensure=False,
    )

    with pytest.raises(RuntimeError, match="bucket privacy policy was not verified"):
        store.record_profile(
            "r", "hinge", "private-profile", True,
            photos=[b"\x89PNG\r\n\x1a\nprivate"], photo_count=1,
        )

    assert storage.bucket("photos").blobs == {}
    assert client.inserted == {}


def test_record_profile_rejects_empty_photo_set():
    client = _FakeBQ()
    s = _store(client, flush_every=1)

    ok = s.record_profile("r", "bumble", "profile-empty", False, photos=[], photo_count=0)
    s.flush()

    assert ok is False
    assert "proj.ds.profiles" not in client.inserted
    assert "proj.ds.profile_photos" not in client.inserted


def test_record_profile_returns_false_and_records_nothing_when_all_uploads_fail():
    client = _FakeBQ()
    storage = _FakeStorage(fail_uploads=True)
    s = BigQueryStore("proj", "ds", photo_bucket="photos", flush_every=1,
                      client=client, storage_client=storage, ensure=True)

    ok = s.record_profile("r", "bumble", "profile-x", True, photos=[b"\x89PNG\r\n\x1a\nx"], photo_count=1)
    s.flush()

    assert ok is False                                   # signals the worker to skip the label
    assert "proj.ds.profiles" not in client.inserted     # no orphan manifest row...
    assert "proj.ds.profile_photos" not in client.inserted   # ...and no photo rows


def test_record_profile_rolls_back_partial_upload_failure():
    client = _FakeBQ()
    storage = _FakeStorage()
    bucket = storage.bucket("photos")
    s = BigQueryStore("proj", "ds", photo_bucket="photos", flush_every=100,
                      client=client, storage_client=storage, ensure=True)

    # Make only the 2nd photo's blob fail (object path carries the photo index "/01-").
    real_blob = bucket.blob
    def flaky_blob(name):
        b = real_blob(name)
        b.fail = "/01-" in name
        return b
    bucket.blob = flaky_blob

    ok = s.record_profile("r", "bumble", "profile-y", True, photos=[b"one", b"two", b"three"], photo_count=3)
    s.flush()

    assert ok is False
    assert "proj.ds.profiles" not in client.inserted
    assert "proj.ds.profile_photos" not in client.inserted
    assert next(blob for name, blob in bucket.blobs.items() if "/00-" in name).deleted is True


def test_record_profile_persists_capture_truncated():
    """capture_truncated marks a label as coming from an INCOMPLETE profile read (the
    driver hit its scroll ceiling before reaching the true bottom) -- this is the
    column that lets a later BigQuery query separate truncated-read labels out, e.g.
    to check whether they're noisier than complete reads."""
    client = _FakeBQ()
    s = BigQueryStore(
        "proj", "ds", photo_bucket="photos", flush_every=100,
        client=client, storage_client=_FakeStorage(), ensure=True,
    )
    s.record_profile("r", "hinge", "profile-2", True, photos=[b"\x89PNG\r\n\x1a\nx"],
                     photo_count=1, capture_truncated=True)
    s.flush()

    profile = client.inserted["proj.ds.profiles"][0]
    assert profile["capture_truncated"] is True


def test_record_profile_defaults_capture_truncated_to_false():
    client = _FakeBQ()
    s = BigQueryStore(
        "proj", "ds", photo_bucket="photos", flush_every=100,
        client=client, storage_client=_FakeStorage(), ensure=True,
    )
    s.record_profile("r", "hinge", "profile-3", True, photos=[b"\x89PNG\r\n\x1a\nx"], photo_count=1)
    s.flush()

    profile = client.inserted["proj.ds.profiles"][0]
    assert profile["capture_truncated"] is False


def test_ensure_tables_runs_capture_truncated_migration():
    client = _FakeBQ(label_rows=_schema_rows_without(("profiles", "capture_truncated")))
    BigQueryStore("proj", "ds", photo_bucket="photos", client=client,
                  storage_client=_FakeStorage(), ensure=True)
    ddl = "\n".join(client.queries)
    assert "ALTER TABLE `proj.ds.profiles`\nADD COLUMN IF NOT EXISTS capture_truncated BOOL" in ddl


def test_create_bucket_enforces_private_access():
    client = _FakeBQ()
    storage = _FakeStorage(bucket_exists=False)
    BigQueryStore("proj", "ds", photo_bucket="newbucket", client=client,
                  storage_client=storage, ensure=True)

    bucket = storage.buckets["newbucket"]
    assert ("newbucket", "US") in storage.created
    assert bucket.iam_configuration.uniform_bucket_level_access_enabled is True
    assert bucket.iam_configuration.public_access_prevention == "enforced"
    assert bucket.patched is True


def test_existing_bucket_gets_hardened():
    # An ALREADY-EXISTING bucket (created before the hardening, e.g. fine-grained ACLs)
    # must still be locked down on startup — not just freshly created ones.
    client = _FakeBQ()
    storage = _FakeStorage(bucket_exists=True)        # bucket already exists, unhardened defaults
    BigQueryStore("proj", "ds", photo_bucket="existing", client=client,
                  storage_client=storage, ensure=True)

    bucket = storage.buckets["existing"]
    assert ("existing", "US") not in storage.created   # not re-created
    assert bucket.iam_configuration.uniform_bucket_level_access_enabled is True
    assert bucket.iam_configuration.public_access_prevention == "enforced"
    assert bucket.patched is True


def test_already_hardened_bucket_is_not_repatched():
    # Idempotent: a bucket already uniform + enforced shouldn't be patched again every startup.
    client = _FakeBQ()
    storage = _FakeStorage(bucket_exists=True)
    storage.bucket("locked").iam_configuration.uniform_bucket_level_access_enabled = True
    storage.bucket("locked").iam_configuration.public_access_prevention = "enforced"
    BigQueryStore("proj", "ds", photo_bucket="locked", client=client,
                  storage_client=storage, ensure=True)
    assert storage.buckets["locked"].patched is False   # no needless write


@pytest.mark.parametrize("operation", ["reload", "patch"])
def test_existing_bucket_hardening_failure_aborts_before_store_is_usable(operation):
    client = _FakeBQ()
    storage = _FakeStorage(bucket_exists=True)
    bucket = storage.bucket("private-required")

    def refused():
        raise PermissionError(f"{operation} denied")

    setattr(bucket, operation, refused)
    with pytest.raises(RuntimeError, match="Refusing BigQuery photo storage") as caught:
        BigQueryStore("proj", "ds", photo_bucket="private-required", client=client,
                      storage_client=storage, ensure=True)

    assert isinstance(caught.value.__cause__, PermissionError)
    assert bucket.blobs == {}
    assert client.queries == []


def test_bucket_hardening_reloads_and_refuses_if_settings_did_not_persist():
    client = _FakeBQ()
    storage = _FakeStorage(bucket_exists=True)
    bucket = storage.bucket("nonpersistent-policy")
    reloads = 0

    def reload_without_persisting_patch():
        nonlocal reloads
        reloads += 1
        if bucket.patched:
            bucket.iam_configuration.uniform_bucket_level_access_enabled = False
            bucket.iam_configuration.public_access_prevention = "inherited"

    bucket.reload = reload_without_persisting_patch
    with pytest.raises(RuntimeError, match="Refusing BigQuery photo storage") as caught:
        BigQueryStore("proj", "ds", photo_bucket="nonpersistent-policy", client=client,
                      storage_client=storage, ensure=True)

    assert reloads == 2
    assert "did not retain" in str(caught.value.__cause__)
    assert bucket.blobs == {}


def test_load_labels_includes_buffered_labels_after_initial_load():
    client = _FakeBQ(label_rows=[{"liked": False, "embedding": [0.0]}])
    s = _store(client, flush_every=100)
    assert s.load_labels() == [(False, [0.0])]

    s.add_label("r", "bumble", True, [0.1, 0.2])

    assert s.load_labels() == [(False, [0.0]), (True, [0.1, 0.2])]
    assert len([q for q in client.queries if q.strip().upper().startswith("SELECT")]) == 1


def test_load_labels_includes_unflushed_labels_before_initial_load():
    client = _FakeBQ(label_rows=[{"liked": False, "embedding": [0.0]}])
    s = _store(client, flush_every=100)
    s.add_label("r", "bumble", True, [0.1])

    assert s.load_labels() == [(False, [0.0]), (True, [0.1])]


def test_flush_on_close():
    client = _FakeBQ()
    s = _store(client, flush_every=100)
    s.record_spend("r", "gemini-test-model", Usage(input_tokens=10, output_tokens=5), 0.0001)
    assert "proj.ds.spend" not in client.inserted
    s.close()
    assert len(client.inserted["proj.ds.spend"]) == 1
    row = client.inserted["proj.ds.spend"][0]
    assert row["input_tokens"] == 10 and row["cost_usd"] == 0.0001


def test_record_spend_stores_none_cost_as_null_not_zero():
    # cost=None means "unknown/unpriceable" (e.g. no budget.pricing entry) -- must be
    # stored as NULL, distinct from a genuinely free $0.00 call.
    client = _FakeBQ()
    s = _store(client, flush_every=100)
    s.record_spend("r", "gemini-unknown", Usage(input_tokens=10, output_tokens=5), None)
    s.close()
    row = client.inserted["proj.ds.spend"][0]
    assert row["cost_usd"] is None


def test_saved_summary_tallies_confirmed_inserts():
    client = _FakeBQ()
    s = _store(client, flush_every=100)
    s.add_label("r", "bumble", True, [0.1])
    s.add_label("r", "bumble", False, [0.2])
    s.record_decision("r", "bumble", "like", 1.0)
    # before flush: buffered, nothing confirmed yet -> shutdown would show it pending
    summary = s.saved_summary()
    assert "nothing new" in summary and "pending=3" in summary
    s.flush()
    summary = s.saved_summary()
    assert "labels=2" in summary and "decisions=1" in summary
    assert "pending" not in summary            # buffers drained -> everything landed


def test_insert_errors_raise():
    client = _FakeBQ(insert_errors=[{"index": 0, "errors": ["bad row"]}])
    s = _store(client, flush_every=1)
    try:
        s.add_label("r", "bumble", True, [0.1])
    except RuntimeError as e:
        assert "BigQuery insert errors" in str(e)
    else:
        raise AssertionError("expected RuntimeError on insert errors")

    # Real BigQuery rejected the whole request -- nothing was actually written, so
    # nothing may be credited as written, and the row must stay buffered (not lost).
    assert s._written["labels"] == 0
    assert "proj.ds.labels" not in client.inserted
    assert len(s._buf["labels"]) == 1


def test_flush_partial_failure_attempts_all_tables():
    """When one table fails during flush(), the other tables are still attempted."""
    class _PartialFailBQ(_FakeBQ):
        def __init__(self, fail_table_suffix):
            super().__init__()
            self.fail_suffix = fail_table_suffix

        def insert_rows_json(self, table_id, rows, row_ids=None):
            if table_id.endswith(self.fail_suffix):
                return [{"index": 0, "errors": ["injected failure"]}]
            self.inserted.setdefault(table_id, []).extend(rows)
            return []

    client = _PartialFailBQ(fail_table_suffix="labels")
    s = _store(client, flush_every=100)
    s.add_label("r", "bumble", True, [0.1])          # goes to 'labels' (will fail)
    s.record_decision("r", "bumble", "like", 1.0)    # goes to 'decisions' (should succeed)

    try:
        s.flush()
    except RuntimeError as e:
        err_msg = str(e)
        assert "labels" in err_msg                    # the failing table is named
    else:
        raise AssertionError("expected RuntimeError from partial flush failure")

    # Despite 'labels' failing, 'decisions' must still have been attempted and inserted.
    decisions_keys = [k for k in client.inserted if k.endswith("decisions")]
    assert decisions_keys, "decisions table was never attempted after labels table failed"
    assert len(client.inserted[decisions_keys[0]]) == 1


def test_requires_project_id():
    try:
        BigQueryStore("", "ds", photo_bucket="photos", client=_FakeBQ(),
                      storage_client=_FakeStorage(), ensure=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError without project_id")


def test_requires_photo_bucket():
    try:
        BigQueryStore("proj", "ds", client=_FakeBQ(), storage_client=_FakeStorage(), ensure=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError without photo_bucket")


@pytest.mark.parametrize(("field", "value"), [
    ("project_id", None),
    ("project_id", True),
    ("project_id", "project.name"),
    ("dataset", "dataset name"),
    ("dataset", "dataset.name"),
    ("location", "US' OR 1=1"),
    ("location", 123),
    ("photo_bucket", None),
    ("photo_bucket", "   "),
    ("photo_bucket", " bucket "),
])
def test_constructor_strictly_validates_cloud_identifiers_and_bucket(field, value):
    kwargs = {
        "project_id": "valid-project",
        "dataset": "valid_dataset",
        "location": "northamerica-northeast1",
        "photo_bucket": "valid-bucket",
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        BigQueryStore(
            **kwargs, client=_FakeBQ(), storage_client=_FakeStorage(), ensure=False)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "10"])
def test_constructor_requires_exact_positive_integer_flush_every(value):
    with pytest.raises(ValueError, match="flush_every"):
        BigQueryStore(
            "proj", "ds", photo_bucket="photos", flush_every=value,
            client=_FakeBQ(), storage_client=_FakeStorage(), ensure=False,
        )


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_constructor_requires_exact_boolean_ensure(value):
    with pytest.raises(ValueError, match="ensure"):
        BigQueryStore(
            "proj", "ds", photo_bucket="photos", client=_FakeBQ(),
            storage_client=_FakeStorage(), ensure=value,
        )


def test_constructor_closes_owned_bigquery_client_if_storage_client_creation_fails(
        monkeypatch):
    import sys

    class OwnedBigQueryClient:
        closed = False

        def close(self):
            self.closed = True

    owned = OwnedBigQueryClient()

    class BrokenStorageClient:
        def __init__(self, **_kwargs):
            raise RuntimeError("storage credentials failed")

    monkeypatch.setitem(sys.modules, "google.cloud", SimpleNamespace(
        bigquery=SimpleNamespace(Client=lambda **_kwargs: owned),
        storage=SimpleNamespace(Client=BrokenStorageClient),
    ))

    with pytest.raises(RuntimeError, match="storage credentials failed"):
        BigQueryStore("proj", "ds", photo_bucket="photos", ensure=False)

    assert owned.closed is True


def test_make_store_does_not_coerce_invalid_flush_every_before_constructor(monkeypatch):
    import operation_love.ranker as ranker
    import operation_love.ranker.bigquery_store as bigquery_module

    captured = {}

    def fake_store(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(bigquery_module, "BigQueryStore", fake_store)
    cfg = SimpleNamespace(
        storage=SimpleNamespace(
            backend="bigquery",
            bigquery={"project_id": "proj", "photo_bucket": "photos", "flush_every": "9"},
        ),
    )

    ranker.make_store(cfg)

    assert captured["flush_every"] == "9"


def test_load_labels_ordered_queries_in_created_at_order():
    client = _FakeBQ(label_rows=[{"liked": True, "embedding": [0.1]}])
    s = _store(client)

    assert s.load_labels_ordered() == [(True, [0.1])]
    assert "ORDER BY created_at" in client.queries[-1]


def test_count_today_parameterizes_local_midnight_matching_sqlite():
    """count_today's day boundary must be the SAME instant SQLiteStore derives from
    local_midnight_epoch() -- not a UTC-day truncation done in SQL."""
    client = _FakeBQ(label_rows=[{"c": 0}])
    s = _store(client)
    s.count_today("bumble")

    sql = client.queries[-1]
    assert "@day_start" in sql
    assert "app=@app" in sql
    assert "TIMESTAMP_TRUNC" not in sql

    params = {param.name: param for param in client.job_configs[-1].query_parameters}
    param = params["day_start"]
    expected = datetime.fromtimestamp(local_midnight_epoch(), tz=timezone.utc)
    assert abs((param.value - expected).total_seconds()) < 2   # same boundary as SQLite
    assert params["app"].value == "bumble"


def test_count_today_binds_app_names_without_mutating_them():
    client = _FakeBQ(label_rows=[{"c": 0}])
    s = _store(client)

    s.count_today("o'brien")

    assert "o'brien" not in client.queries[-1]
    params = {param.name: param for param in client.job_configs[-1].query_parameters}
    assert params["app"].value == "o'brien"


def test_spend_today_parameterizes_local_midnight_and_sums_cost():
    client = _FakeBQ(label_rows=[{"total": 3.5}])
    s = _store(client)

    assert s.spend_today() == 3.5

    sql = client.queries[-1]
    assert "@day_start" in sql
    assert "TIMESTAMP_TRUNC" not in sql

    param = client.job_configs[-1].query_parameters[0]
    assert param.name == "day_start"
    expected = datetime.fromtimestamp(local_midnight_epoch(), tz=timezone.utc)
    assert abs((param.value - expected).total_seconds()) < 2


def test_spend_today_propagates_query_failure_instead_of_disabling_daily_cap():
    class FailedJob:
        def result(self):
            raise PermissionError("spend ledger unreadable")

    client = _FakeBQ()
    client.query = lambda *_args, **_kwargs: FailedJob()
    store = _store(client)

    with pytest.raises(PermissionError, match="spend ledger unreadable"):
        store.spend_today()


class _PoisonRowBQ(_FakeBQ):
    """Models REAL BigQuery insertAll semantics with skip_invalid_rows left unset (the
    default -- and what BigQueryStore actually sends): if the current call's `rows`
    contains a row matching `bad_embedding`, the WHOLE request is rejected and NOTHING
    is written -- not even the other, valid rows in that same batch. `errors` names
    only the poison row's index within THAT call.

    The poison row is matched by content, not position: a row's index shifts across
    retries as other rows are appended to / evicted from the buffer, so matching by
    content is what makes this fake keep "permanently" rejecting the same logical row
    call after call, the way a genuinely malformed row would in reality."""

    def __init__(self, bad_embedding):
        super().__init__()
        self.bad_embedding = bad_embedding
        self.calls: list[tuple[str, list[dict], list]] = []

    def insert_rows_json(self, table_id, rows, row_ids=None):
        self.calls.append((table_id, list(rows), list(row_ids or [])))
        bad_idx = next((i for i, r in enumerate(rows) if r.get("embedding") == self.bad_embedding), None)
        if bad_idx is None:
            self.inserted.setdefault(table_id, []).extend(rows)
            self.row_ids.setdefault(table_id, []).extend(row_ids or [])
            return []
        return [{"index": bad_idx, "errors": ["boom"]}]        # whole request rejected


def test_flush_partial_failure_keeps_whole_batch_buffered_not_just_the_named_row():
    """Real BigQuery (skip_invalid_rows unset) fails the WHOLE request when any row is
    invalid -- rows NOT named in `errors` are not written either. So a rejected flush
    must keep the entire batch buffered (not evict the rows absent from `errors`, the
    old -- wrong -- assumption), and `_written` must not credit rows that were never
    actually persisted."""
    client = _PoisonRowBQ(bad_embedding=[0.1])
    s = _store(client, flush_every=100)
    s.add_label("r", "bumble", True, [0.0])     # good
    s.add_label("r", "bumble", False, [0.1])    # poison -- BigQuery keeps rejecting this one
    s.add_label("r", "bumble", True, [0.2])     # good

    try:
        s.flush()
        raise AssertionError("expected RuntimeError on partial failure")
    except RuntimeError:
        pass

    # Nothing was actually written -- the whole request failed -- so nothing may be
    # credited as written, no row is evicted, and no row may be BOTH counted as
    # written AND absent from the store.
    assert s._written["labels"] == 0
    assert len(s._buf["labels"]) == 3
    assert "proj.ds.labels" not in client.inserted
    for row in s._buf["labels"]:
        assert row not in client.inserted.get("proj.ds.labels", [])

    # Remove the poison row by hand (simulating the bad data getting fixed/discarded)
    # and retry: now the whole batch is valid, and it lands together, using the SAME
    # row_ids as the first attempt (content-derived, stable) so a real BigQuery's
    # best-effort dedup would not double-insert anything it happened to have ingested.
    s._buf["labels"] = [r for r in s._buf["labels"] if r["embedding"] != [0.1]]
    s.flush()

    assert s._written["labels"] == 2
    assert len(client.inserted["proj.ds.labels"]) == 2
    assert s._buf["labels"] == []

    _, first_rows, first_ids = client.calls[0]
    assert len(first_ids) == len(set(first_ids))            # distinct ids within a batch
    _, second_rows, second_ids = client.calls[1]
    good_id_first_attempt = first_ids[[r["embedding"] for r in first_rows].index([0.0])]
    good_id_retry = second_ids[[r["embedding"] for r in second_rows].index([0.0])]
    assert good_id_first_attempt == good_id_retry            # same content -> same id on retry


def test_flush_drops_permanently_invalid_row_after_bound_and_keeps_good_rows():
    """A row BigQuery keeps rejecting forever must not block its table's buffer (and
    every good row queued behind it) forever. After the bounded number of consecutive
    failed attempts, the store must drop ONLY that row -- loudly, and never crediting
    it as written, since it never was -- while the surviving good rows remain buffered
    and land on the very next attempt."""
    from operation_love.ranker.bigquery_store import _MAX_INSERT_ATTEMPTS

    client = _PoisonRowBQ(bad_embedding=[0.1])
    s = _store(client, flush_every=100)
    s.add_label("r", "bumble", True, [0.0])     # good
    s.add_label("r", "bumble", False, [0.1])    # permanently poison
    s.add_label("r", "bumble", True, [0.2])     # good

    for _ in range(_MAX_INSERT_ATTEMPTS):
        try:
            s.flush()
        except RuntimeError:
            pass

    # The poison row is gone from the buffer (dropped, not retried forever) but was
    # NEVER counted as written -- it was never accepted by BigQuery, so it is neither
    # "written" nor silently lost without a trace: saved_summary() must surface it.
    assert [r["embedding"] for r in s._buf["labels"]] == [[0.0], [0.2]]
    assert s._written["labels"] == 0
    assert s._dropped["labels"] == 1
    assert "DROPPED" in s.saved_summary()
    assert "labels=1" in s.saved_summary()

    # With the poison row gone, the surviving good rows now flush successfully -- no
    # data lost among them.
    s.flush()
    assert s._written["labels"] == 2
    assert [r["embedding"] for r in client.inserted["proj.ds.labels"]] == [[0.0], [0.2]]
    assert s._buf["labels"] == []


def test_flush_partial_failure_attempts_survive_interleaved_new_rows():
    """A row's failed-attempt count is tracked by its stable content-derived row_id,
    not by its position in the buffer, so it must keep accumulating correctly even
    as new (unrelated, good) rows are appended between retries -- which shifts every
    row's index."""
    from operation_love.ranker.bigquery_store import _MAX_INSERT_ATTEMPTS

    client = _PoisonRowBQ(bad_embedding=[0.1])
    s = _store(client, flush_every=100)
    s.add_label("r", "bumble", False, [0.1])    # poison, starts at index 0

    for i in range(_MAX_INSERT_ATTEMPTS - 1):
        try:
            s.flush()
        except RuntimeError:
            pass
        s.add_label("r", "bumble", True, [float(i)])   # shifts the poison row's index

    assert s._dropped["labels"] == 0             # not yet at the bound

    try:
        s.flush()
    except RuntimeError:
        pass

    assert s._dropped["labels"] == 1
    assert all(r["embedding"] != [0.1] for r in s._buf["labels"])
