"""BigQueryStore tests with a fake client — no google SDK or network required."""
import inspect
from datetime import datetime, timezone

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
    client = _FakeBQ()
    BigQueryStore("proj", "ds", photo_bucket="photos", client=client,
                  storage_client=_FakeStorage(), ensure=True)
    ddl = "\n".join(client.queries)
    assert "ALTER TABLE `proj.ds.labels` ADD COLUMN IF NOT EXISTS profile_id STRING" in ddl
    assert "ALTER TABLE `proj.ds.decisions` ADD COLUMN IF NOT EXISTS source STRING" in ddl


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
                            "item_description"]
    assert params["angle"].default == ""
    assert params["item_description"].default == ""
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

    client = _FakeBQ()
    BigQueryStore("proj", "ds", photo_bucket="photos", client=client,
                  storage_client=_FakeStorage(), ensure=True)
    ddl = "\n".join(client.queries)
    assert "ALTER TABLE `proj.ds.openers` ADD COLUMN IF NOT EXISTS angle STRING;" in ddl

    # Ordering matters inside the single multi-statement script: an ALTER submitted before its
    # table's CREATE would fail against a brand-new project.
    assert (ddl.index("CREATE TABLE IF NOT EXISTS `proj.ds.openers`")
            < ddl.index("ALTER TABLE `proj.ds.openers`"))

    # And every declared migration reaches the script, not just this one -- so a future column
    # added to _MIGRATIONS can't be left unapplied by an _ensure_tables that stopped looping.
    tids = {name: f"proj.ds.{name}" for name in _declared_tables()}
    for stmt in _MIGRATIONS:
        assert stmt.format(**tids) in ddl


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

    client = _FakeBQ()
    BigQueryStore("proj", "ds", photo_bucket="photos", client=client,
                  storage_client=_FakeStorage(), ensure=True)
    ddl = "\n".join(client.queries)
    assert ("ALTER TABLE `proj.ds.openers` ADD COLUMN IF NOT EXISTS item_description STRING;"
            in ddl)
    assert (ddl.index("CREATE TABLE IF NOT EXISTS `proj.ds.openers`")
            < ddl.index("ADD COLUMN IF NOT EXISTS item_description"))


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


def test_count_today_counts_only_auto_decisions():
    client = _FakeBQ(label_rows=[{"c": 7}])
    s = _store(client)

    assert s.count_today("bumble") == 7
    assert "AND source='auto'" in client.queries[-1]


def test_record_profile_uploads_photos_and_manifest_rows():
    client = _FakeBQ()
    storage = _FakeStorage()
    s = BigQueryStore("proj", "ds", photo_bucket="photos", flush_every=100,
                      client=client, storage_client=storage, ensure=False)
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
                      client=client, storage_client=storage, ensure=False)

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
                      client=client, storage_client=storage, ensure=False)

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
    s = _store(client, flush_every=100)
    s.record_profile("r", "hinge", "profile-2", True, photos=[b"\x89PNG\r\n\x1a\nx"],
                     photo_count=1, capture_truncated=True)
    s.flush()

    profile = client.inserted["proj.ds.profiles"][0]
    assert profile["capture_truncated"] is True


def test_record_profile_defaults_capture_truncated_to_false():
    client = _FakeBQ()
    s = _store(client, flush_every=100)
    s.record_profile("r", "hinge", "profile-3", True, photos=[b"\x89PNG\r\n\x1a\nx"], photo_count=1)
    s.flush()

    profile = client.inserted["proj.ds.profiles"][0]
    assert profile["capture_truncated"] is False


def test_ensure_tables_runs_capture_truncated_migration():
    client = _FakeBQ()
    BigQueryStore("proj", "ds", photo_bucket="photos", client=client,
                  storage_client=_FakeStorage(), ensure=True)
    ddl = "\n".join(client.queries)
    assert "ALTER TABLE `proj.ds.profiles` ADD COLUMN IF NOT EXISTS capture_truncated BOOL" in ddl


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
    assert "TIMESTAMP_TRUNC" not in sql

    param = client.job_configs[-1].query_parameters[0]
    assert param.name == "day_start"
    expected = datetime.fromtimestamp(local_midnight_epoch(), tz=timezone.utc)
    assert abs((param.value - expected).total_seconds()) < 2   # same boundary as SQLite


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
