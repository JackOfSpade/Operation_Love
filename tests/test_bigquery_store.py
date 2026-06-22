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
    s.record_spend("r", "claude-opus-4-8", Usage(input_tokens=10, output_tokens=5), 0.0001)
    assert "proj.ds.spend" not in client.inserted
    s.close()
    assert len(client.inserted["proj.ds.spend"]) == 1
    row = client.inserted["proj.ds.spend"][0]
    assert row["input_tokens"] == 10 and row["cost_usd"] == 0.0001


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
