"""tools/backfill_photo_features.py: the whole-photo backfill, on fakes only.

Fakes stand in for the store (its missing-row list follows what has been written, so resumption
is real), Cloud Storage (bytes by object name) and the embedder (a deterministic stand-in for
``Embedder.embed_whole_photos``).  No model, no network, no frame ever touches disk.
"""

from __future__ import annotations

import hashlib
import threading
import time
from types import SimpleNamespace

import pytest

from tools import backfill_photo_features as bf

VERSION = "whole_photo_v1"
DIMS = 768
BUCKET = "test-photo-bucket"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class NotFound(Exception):
    """Named like google.api_core.exceptions.NotFound, which is all the tool looks at."""


class FakeBlob:
    def __init__(self, objects: dict, name: str):
        self._objects, self.name = objects, name

    def download_as_bytes(self, timeout=None):
        if self.name not in self._objects:
            raise NotFound(self.name)
        return self._objects[self.name]


class FakeBucket:
    def __init__(self, objects: dict):
        self._objects = objects

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self._objects, name)


class FakeStorageClient:
    def __init__(self, objects: dict):
        self.objects = objects
        self.buckets: list[str] = []

    def bucket(self, name: str) -> FakeBucket:
        self.buckets.append(name)
        return FakeBucket(self.objects)


class FakeStore:
    """The slice of the Store protocol the backfill uses, with a real missing-row list."""

    def __init__(self, labels: list[dict], objects: dict, *, refuse=()):
        self.labels = labels
        self.rows: dict[tuple[str, str], dict] = {}
        self.writes: list[dict] = []
        self.flushes = 0
        self.closed = False
        self.refuse = set(refuse)
        self.dropped = 0
        self.photo_bucket_name = BUCKET
        self.storage_client = FakeStorageClient(objects)

    def labels_missing_photo_features(self, feature_version):
        assert feature_version == VERSION
        # Like the real stores: no row, or a zero-card row an earlier build stored.
        return [
            dict(label)
            for label in self.labels
            if self.rows.get((label["app"], label["profile_id"]), {}).get(
                "card_count", 0
            )
            < 1
        ]

    def photo_feature_coverage(self, feature_version):
        assert feature_version == VERSION
        with_rows = [
            label
            for label in self.labels
            if (label["app"], label["profile_id"]) in self.rows
        ]
        zero = sum(
            1
            for label in with_rows
            if self.rows[(label["app"], label["profile_id"])]["card_count"] == 0
        )
        return {
            "labels": len(self.labels),
            "with_features": len(with_rows),
            "missing": len(self.labels) - len(with_rows),
            "zero_cards": zero,
        }

    def add_photo_features(self, **row):
        if row["profile_id"] in self.refuse:
            return False
        self.writes.append(row)
        self.rows[(row["app"], row["profile_id"])] = row
        return True

    def flush(self):
        self.flushes += 1

    def dropped_rows(self):
        return {"photo_features": self.dropped} if self.dropped else {}

    def close(self):
        self.closed = True


class FakeEmbedder:
    """Deterministic stand-in: card_count = number of frames whose first byte is b"P"."""

    def __init__(self):
        self.calls = 0
        self.warmups = 0
        self._lock = threading.Lock()

    def warmup(self):
        self.warmups += 1

    def embed_whole_photos(self, frames):
        from operation_love.vision.photo_cards import NoPhotoCards

        with self._lock:
            self.calls += 1
        cards = sum(1 for f in frames if f[:1] == b"P")
        if cards == 0:  # the live contract: zero cards is an extraction failure
            raise NoPhotoCards(f"{len(frames)} frame(s) hold no photo card")
        joined = "".join(_sha(f) for f in frames)
        return SimpleNamespace(
            vector=[1.0 / (DIMS**0.5)] * DIMS,
            card_count=cards,
            frames_sha256=hashlib.sha256(joined.encode("ascii")).hexdigest(),
            feature_version=VERSION,
        )


def _label(n: int, frames: list[bytes], objects: dict, *, source="manual") -> dict:
    uris, shas = [], []
    for index, data in enumerate(frames):
        name = f"profiles/p{n}/{index}.png"
        objects[name] = data
        uris.append(f"gs://{BUCKET}/{name}")
        shas.append(_sha(data))
    return {
        "run_id": f"run{n}",
        "app": "hinge",
        "profile_id": f"p{n}",
        "source": source,
        "frame_uris": uris,
        "frame_sha256s": shas,
    }


def _world(count: int = 4):
    objects: dict = {}
    labels = [
        _label(n, [b"P-photo-%d-a" % n, b"T-text-%d" % n, b"P-photo-%d-b" % n], objects)
        for n in range(count)
    ]
    return FakeStore(labels, objects), objects


def _run(store, embedder=None, **kwargs):
    lines: list[str] = []
    report = bf.backfill(
        store,
        embedder or FakeEmbedder(),
        bf.FrameSource(store).frames,
        feature_version=VERSION,
        dims=DIMS,
        log=lines.append,
        **kwargs,
    )
    return report, lines


def test_frames_digest_is_sha256_of_the_concatenated_hex_digests():
    a, b = b"frame-a", b"frame-b"
    expected = hashlib.sha256(
        (hashlib.sha256(a).hexdigest() + hashlib.sha256(b).hexdigest()).encode()
    ).hexdigest()
    assert bf.frames_digest([_sha(a), _sha(b)]) == expected
    # Order matters: the vector is a function of the frames IN ORDER.
    assert bf.frames_digest([_sha(b), _sha(a)]) != expected


def test_frames_digest_matches_the_cards_module_formula():
    from operation_love.vision.photo_cards import frames_sha256

    frames = [b"one", b"two", b"three"]
    assert bf.frames_digest([_sha(f) for f in frames]) == frames_sha256(frames)


def test_live_contract_constants():
    from operation_love.vision.photo_cards import (
        WHOLE_PHOTO_DIMS,
        WHOLE_PHOTO_FEATURE_VERSION,
    )

    assert (WHOLE_PHOTO_FEATURE_VERSION, WHOLE_PHOTO_DIMS) == (VERSION, DIMS)


def test_writes_every_missing_row_with_source_backfill():
    store, _ = _world(3)
    embedder = FakeEmbedder()
    report, lines = _run(store, embedder)
    assert report.ok and report.computed == report.written == 3
    assert embedder.warmups == 1 and embedder.calls == 3
    assert {w["profile_id"] for w in store.writes} == {"p0", "p1", "p2"}
    for w in store.writes:
        assert w["source"] == "backfill"
        assert w["feature_version"] == VERSION
        assert w["card_count"] == 2 and len(w["embedding"]) == DIMS
        label = next(
            lab for lab in store.labels if lab["profile_id"] == w["profile_id"]
        )
        assert w["frames_sha256"] == bf.frames_digest(label["frame_sha256s"])
        assert w["run_id"] == label["run_id"] and w["app"] == label["app"]
    assert store.flushes == 1
    assert report.card_counts == {2: 3}
    assert report.coverage_before["missing"] == 3
    assert report.coverage_after == {
        "labels": 3,
        "with_features": 3,
        "missing": 0,
        "zero_cards": 0,
    }
    text = bf.format_report(report)
    assert "2 cards x3" in text and "Backfill complete." in text


def test_dry_run_computes_but_writes_and_flushes_nothing():
    store, _ = _world(3)
    embedder = FakeEmbedder()
    report, lines = _run(store, embedder, dry_run=True)
    assert report.ok and report.computed == 3 and report.written == 0
    assert embedder.calls == 3
    assert store.writes == [] and store.rows == {} and store.flushes == 0
    assert report.coverage_after == report.coverage_before
    assert "DRY RUN" in lines[0]
    assert "wrote nothing (dry run)" in bf.format_report(report)


def test_skips_labels_that_already_have_a_row_and_recomputes_a_zero_card_one():
    store, _ = _world(4)
    store.rows[("hinge", "p1")] = {"card_count": 3}
    store.rows[("hinge", "p3")] = {"card_count": 0}  # stored by an earlier build
    embedder = FakeEmbedder()
    report, _ = _run(store, embedder)
    assert report.listed == 3 and embedder.calls == 3
    assert sorted(w["profile_id"] for w in store.writes) == ["p0", "p2", "p3"]
    assert report.coverage_before["with_features"] == 2
    assert report.coverage_before["zero_cards"] == 1
    assert report.coverage_after["zero_cards"] == 0


def test_resumes_after_a_limited_run_and_is_idempotent_when_done():
    store, _ = _world(5)
    first, _ = _run(store, limit=2)
    assert first.ok and first.written == 2 and first.listed == 5
    second, _ = _run(store)
    assert second.ok and second.listed == 3 and second.written == 3
    assert len({w["profile_id"] for w in store.writes}) == 5  # never a profile twice
    embedder = FakeEmbedder()
    third, lines = _run(store, embedder)
    assert third.ok and third.listed == 0 and third.written == 0
    assert embedder.calls == 0 and embedder.warmups == 0  # nothing to do: no model load
    assert len(store.writes) == 5


def test_refuses_a_label_whose_frame_does_not_match_its_stored_sha256():
    store, objects = _world(3)
    objects["profiles/p1/2.png"] = b"P-tampered"
    report, lines = _run(store)
    assert not report.ok
    assert [
        w["profile_id"] for w in sorted(store.writes, key=lambda w: w["profile_id"])
    ] == [
        "p0",
        "p2",
    ]
    [(name, category, reason)] = report.failures
    assert "hinge/p1" in name and category == bf.SHA_MISMATCH
    assert "frame 2" in reason
    assert any("REFUSED hinge/p1" in line for line in lines)
    assert "Backfill INCOMPLETE" in bf.format_report(report)


def test_a_label_with_no_frames_is_refused_never_written_as_zero_cards():
    store, _ = _world(1)
    store.labels.append(
        {
            "run_id": "r",
            "app": "hinge",
            "profile_id": "pruned",
            "source": "auto_review_random",
            "frame_uris": [],
            "frame_sha256s": [],
        }
    )
    report, _ = _run(store)
    assert [w["profile_id"] for w in store.writes] == ["p0"]
    assert [c for _, c, _ in report.failures] == [bf.NO_FRAMES]
    assert "NOT READY" in bf.format_report(report)


def test_frames_with_no_photo_card_are_refused_never_written_as_zero_cards():
    """RT-2: zero photo cards from a Hinge capture is an extraction failure with its own
    category, never a recorded zero-card row."""
    objects: dict = {}
    store = FakeStore(
        [
            _label(0, [b"T-text", b"T-vitals"], objects),
            _label(1, [b"P-photo", b"T-text"], objects),
        ],
        objects,
    )
    report, lines = _run(store)
    assert not report.ok
    assert [w["profile_id"] for w in store.writes] == ["p1"]
    [(name, category, reason)] = report.failures
    assert "hinge/p0" in name and category == bf.NO_PHOTO_CARDS
    assert "no photo card" in reason
    assert any("REFUSED hinge/p0" in line for line in lines)
    text = bf.format_report(report)
    assert "no photo card found x1" in text and "NOT READY" in text
    assert report.card_counts == {1: 1}


@pytest.mark.parametrize(
    ("change", "category"),
    [
        ({"frames_sha256": "0" * 64}, bf.DIGEST_MISMATCH),
        ({"feature_version": "whole_photo_v0"}, bf.BAD_EMBEDDING),
        ({"vector": [0.1] * 10}, bf.BAD_EMBEDDING),
        ({"vector": None}, bf.BAD_EMBEDDING),
        ({"vector": [float("nan")] * DIMS}, bf.BAD_EMBEDDING),
        ({"card_count": 0}, bf.BAD_EMBEDDING),
    ],
    ids=["digest", "version", "dims", "no-vector", "nan", "zero-count"],
)
def test_a_malformed_embedding_result_refuses_the_label(change, category):
    class Bent(FakeEmbedder):
        def embed_whole_photos(self, frames):
            result = super().embed_whole_photos(frames)
            for key, value in change.items():
                setattr(result, key, value)
            return result

    store, _ = _world(1)
    report, _ = _run(store, Bent())
    assert store.writes == []
    assert [c for _, c, _ in report.failures] == [category]


def test_an_embedder_error_fails_that_label_and_the_run_exit_status():
    class Flaky(FakeEmbedder):
        def embed_whole_photos(self, frames):
            if b"P-photo-1-a" in frames:
                raise RuntimeError("CLIP exploded")
            return super().embed_whole_photos(frames)

    store, _ = _world(3)
    report, _ = _run(store, Flaky())
    assert not report.ok and report.written == 2
    [(name, category, reason)] = report.failures
    assert category == bf.ERROR and "CLIP exploded" in reason


def test_a_store_refusal_or_dropped_row_is_a_failure():
    store, _ = _world(2)
    store.refuse = {"p0"}
    report, _ = _run(store)
    assert [c for _, c, _ in report.failures] == [bf.STORE_REFUSED]
    store2, _ = _world(1)
    store2.dropped = 1
    report2, _ = _run(store2)
    assert report2.written == 1 and not report2.ok
    assert "DATA LOSS" in bf.format_report(report2)


def test_frame_source_refuses_uris_outside_the_configured_bucket_and_missing_objects():
    store, objects = _world(2)
    store.labels[0]["frame_uris"][0] = "gs://someone-elses-bucket/x.png"
    del objects["profiles/p1/1.png"]
    report, _ = _run(store)
    assert store.writes == []
    assert sorted(c for _, c, _ in report.failures) == sorted(
        [bf.OUTSIDE_BUCKET, bf.MISSING_OBJECT]
    )
    assert store.storage_client.buckets == [BUCKET]


def test_frame_source_refuses_a_manifest_it_cannot_verify():
    source = bf.FrameSource(
        None, storage_client=FakeStorageClient({}), bucket_name=BUCKET
    )
    good_uri = f"gs://{BUCKET}/a.png"
    for uris, shas in (
        ([good_uri, good_uri], ["a" * 64]),  # count mismatch
        ([good_uri], ["not-a-sha"]),
        ([""], ["a" * 64]),
    ):
        with pytest.raises(bf.LabelRefused) as exc:
            source.frames({"frame_uris": uris, "frame_sha256s": shas})
        assert exc.value.category == bf.MANIFEST


def test_never_more_than_workers_labels_in_flight():
    store, _ = _world(8)
    inner = bf.FrameSource(store).frames
    active, peak = [0], [0]
    lock = threading.Lock()

    def tracked(label):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        try:
            time.sleep(0.01)
            return inner(label)
        finally:
            with lock:
                active[0] -= 1

    report = bf.backfill(
        store,
        FakeEmbedder(),
        tracked,
        feature_version=VERSION,
        dims=DIMS,
        workers=2,
        log=lambda _line: None,
    )
    assert report.ok and report.written == 8
    assert 1 <= peak[0] <= 2


def test_invalid_controls_are_rejected():
    store, _ = _world(1)
    with pytest.raises(ValueError):
        _run(store, workers=0)
    with pytest.raises(ValueError):
        _run(store, workers=bf.MAX_WORKERS + 1)
    with pytest.raises(ValueError):
        _run(store, limit=0)


def _patch_cli(monkeypatch, store, embedder):
    from operation_love import config as cfg_mod
    from operation_love import ranker
    from operation_love.vision import embed

    built = []

    def make_store(cfg, ensure=True):
        built.append(ensure)
        return store

    monkeypatch.setattr(cfg_mod, "load", lambda path: SimpleNamespace(path=path))
    monkeypatch.setattr(ranker, "make_store", make_store)
    monkeypatch.setattr(embed, "Embedder", lambda: embedder)
    return built


def test_cli_writes_with_ensure_true_and_exits_zero(monkeypatch, capsys):
    store, _ = _world(2)
    built = _patch_cli(monkeypatch, store, FakeEmbedder())
    assert bf.main(["--config", "x.yaml", "--workers", "1"]) == 0
    assert built == [True] and store.closed and len(store.writes) == 2
    assert "Backfill complete." in capsys.readouterr().out


def test_cli_dry_run_opens_the_store_read_only(monkeypatch, capsys):
    store, _ = _world(2)
    built = _patch_cli(monkeypatch, store, FakeEmbedder())
    assert bf.main(["--config", "x.yaml", "--dry-run", "--limit", "1"]) == 0
    assert built == [False] and store.closed and store.writes == []
    out = capsys.readouterr().out
    assert "Processed 1: computed 1, wrote nothing (dry run)" in out


def test_cli_exits_nonzero_when_any_label_failed(monkeypatch, capsys):
    store, objects = _world(2)
    objects["profiles/p0/0.png"] = b"P-corrupt"
    _patch_cli(monkeypatch, store, FakeEmbedder())
    assert bf.main(["--config", "x.yaml"]) == 1
    assert store.closed and [w["profile_id"] for w in store.writes] == ["p1"]
    assert "frame sha256 mismatch x1" in capsys.readouterr().out


def test_cli_rejects_too_many_workers(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        bf.main(["--workers", str(bf.MAX_WORKERS + 1)])
    assert exc.value.code == 2


def test_accepts_the_live_whole_photo_embedding_type():
    from operation_love.vision.embed import WholePhotoEmbedding

    class Live(FakeEmbedder):
        def embed_whole_photos(self, frames):
            r = super().embed_whole_photos(frames)
            return WholePhotoEmbedding(
                vector=r.vector, card_count=r.card_count, frames_sha256=r.frames_sha256
            )

    store, _ = _world(2)
    report, _ = _run(store, Live())
    assert report.ok and report.written == 2
    assert {w["feature_version"] for w in store.writes} == {VERSION}


def test_every_write_is_best_effort_so_a_rejected_row_poisons_nothing():
    store, _ = _world(2)
    _run(store)
    assert {w["best_effort"] for w in store.writes} == {True}


def test_band_cards_are_counted_and_printed():
    from operation_love.vision import photo_cards as pc

    class Banded(FakeEmbedder):
        def embed_whole_photos(self, frames):
            result = super().embed_whole_photos(frames)
            result.band_cards = (pc.BAND_TEXT, pc.BAND_CONTIGUOUS, pc.BAND_BOTH)
            return result

    store, _ = _world(2)
    report, _ = _run(store, Banded(), dry_run=True)
    assert report.band_cards == {
        pc.BAND_TEXT: 2,
        pc.BAND_CONTIGUOUS: 2,
        pc.BAND_BOTH: 2,
    }
    assert (
        "band cards (white share 0.5..0.8): 6 seen; kept by edge media only 0, by "
        "contiguous content only 2, by both 2; dropped as text 2"
    ) in bf.format_report(report)
    plain, _ = _run(_world(1)[0], dry_run=True)
    assert "band cards" not in bf.format_report(plain)


def test_a_failed_final_flush_is_reported_not_raised():
    store, _ = _world(2)

    def flush():
        raise RuntimeError("BigQuery flush failed for 1 table(s)")

    store.flush = flush
    report, lines = _run(store)
    assert report.written == 2 and not report.ok
    assert "flush failed" in report.flush_error
    text = bf.format_report(report)
    assert "FLUSH FAILED" in text and "Backfill INCOMPLETE" in text
    assert "after:" in text  # the coverage summary still printed


def test_cli_prints_the_summary_and_exits_nonzero_when_close_fails(monkeypatch, capsys):
    store, _ = _world(1)

    def close():
        raise RuntimeError("close failed")

    store.close = close
    _patch_cli(monkeypatch, store, FakeEmbedder())
    assert bf.main(["--config", "x.yaml"]) == 1
    out = capsys.readouterr().out
    assert "Processed 1: computed 1, wrote 1" in out
    assert "Closing the store FAILED: RuntimeError: close failed" in out
