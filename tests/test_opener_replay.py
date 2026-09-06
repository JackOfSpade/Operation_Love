"""operation_love/opener/replay_corpus.py (the on-disk format + writer) and
tools/opener_replay.py (the CLI harness that replays it through the current prompt).

No network, ever: every CLI test injects a fake GeminiTransport exactly as
tests/test_gemini_opener.py and tests/test_gemini_model_probe.py do. NEVER make a real API call
from a test.
"""
from __future__ import annotations

import json
import time

import pytest
import yaml

from operation_love.opener import replay_corpus as rc
from operation_love.ranker.store import SQLiteStore
from tools import opener_replay as m

# ---------------------------------------------------------------------------------------
# local helpers (no conftest.py in this repo)
# ---------------------------------------------------------------------------------------


def _success(opener="Nice hike, where was that?", *, index=1,
             angle="asking about the trail", item_description="a photo on a ridge"):
    return (200, {
        "candidates": [{"content": {"parts": [{"text": json.dumps({
            "item_index": index, "referenced": "ridge photo", "angle": angle,
            "item_description": item_description, "opener": opener,
        })}]}}],
        "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 5},
    })


class _ScriptedTransport:
    """Replays a fixed list of (code, body) responses in call order; records every call."""

    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, payload, headers, timeout, *, method="POST"):
        self.calls.append({"method": method, "url": url, "payload": payload})
        if not self.responses:
            raise AssertionError("no more scripted transport responses")
        return self.responses.pop(0)


def _forbidden_transport(*_a, **_k):
    raise AssertionError("transport must not be called in dry-run mode")


class _ForbiddenStore:
    """Raises the instant any store method is touched -- used to prove dry run writes nothing."""

    def record_opener(self, *a, **kw):
        raise AssertionError("store.record_opener must not be called in dry-run mode")

    def record_opener_rejection(self, *a, **kw):
        raise AssertionError("store.record_opener_rejection must not be called in dry-run mode")


def _config_path(tmp_path, models=("gemini-x",), style="Be warm, specific, and brief."):
    path = tmp_path / "config.yaml"
    models_yaml = ", ".join(f'"{model}"' for model in models)
    path.write_text(f'opener:\n  models: [{models_yaml}]\n  style: "{style}"\n')
    return path


def _write_capture(root, *, name="Alex", items=(b"item-1-bytes",), context=(),
                   truncated=False, prompt_sha256="captured-era", captured_at=None):
    result = rc.write_replay_capture(root, items=items, name=name, context=context,
                                     truncated=truncated, prompt_sha256=prompt_sha256,
                                     captured_at=captured_at)
    assert result.ok, result.error
    return result


# ---------------------------------------------------------------------------------------
# replay_corpus.py -- manifest round-trip
# ---------------------------------------------------------------------------------------

def test_write_then_load_roundtrips_every_field(tmp_path):
    written = _write_capture(
        tmp_path, name="Jamie", items=(b"item-one", b"item-two"),
        context=(b"context-one",), truncated=True, prompt_sha256="abc123", captured_at=1700.5)

    capture = rc.load_replay_capture(tmp_path, written.replay_id)

    assert capture.replay_id == written.replay_id
    assert capture.name == "Jamie"
    assert capture.items == (b"item-one", b"item-two")
    assert capture.context == (b"context-one",)
    assert capture.truncated is True
    assert capture.prompt_sha256 == "abc123"
    assert capture.captured_at == 1700.5


def test_manifest_json_has_exactly_the_documented_shape(tmp_path):
    written = _write_capture(tmp_path, name="Sam", items=(b"i1", b"i2"), context=(b"c1",),
                             truncated=False, prompt_sha256="era-1")
    manifest_path = written.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["format_version"] == 1
    assert manifest["replay_id"] == written.replay_id
    assert manifest["name"] == "Sam"
    assert manifest["truncated"] is False
    assert manifest["prompt_sha256"] == "era-1"
    assert manifest["items"] == ["item_001.bin", "item_002.bin"]
    assert manifest["context"] == ["context_001.bin"]
    assert isinstance(manifest["captured_at"], float)


def test_write_replay_capture_is_idempotent_by_content(tmp_path):
    first = _write_capture(tmp_path, name="Robin", items=(b"same-bytes",))
    second = _write_capture(tmp_path, name="Robin", items=(b"same-bytes",))

    assert first.replay_id == second.replay_id
    assert first.path == second.path
    assert len(list(tmp_path.iterdir())) == 1  # no duplicate directory was created


# ---------------------------------------------------------------------------------------
# replay_corpus.py -- discovery, listing, ordering
# ---------------------------------------------------------------------------------------

def test_list_replay_ids_orders_by_captured_at_then_replay_id(tmp_path):
    late = _write_capture(tmp_path, name="Late", items=(b"late-bytes",), captured_at=300.0)
    early = _write_capture(tmp_path, name="Early", items=(b"early-bytes",), captured_at=100.0)
    middle = _write_capture(tmp_path, name="Mid", items=(b"mid-bytes",), captured_at=200.0)

    ordered = rc.list_replay_ids(tmp_path)

    assert ordered == [early.replay_id, middle.replay_id, late.replay_id]


def test_list_replay_ids_skips_directories_with_no_manifest(tmp_path):
    _write_capture(tmp_path, items=(b"real-capture",))
    stray_dir = tmp_path / "not-a-capture"
    stray_dir.mkdir()
    (stray_dir / "readme.txt").write_text("not a manifest")
    (tmp_path / "stray-file.json").write_text("{}")

    ids = rc.list_replay_ids(tmp_path)

    assert len(ids) == 1


def test_list_replay_ids_on_missing_root_returns_empty(tmp_path):
    assert rc.list_replay_ids(tmp_path / "does-not-exist") == []


def test_load_replay_corpus_replay_ids_selects_subset_and_order(tmp_path):
    a = _write_capture(tmp_path, name="A", items=(b"a-bytes",), captured_at=1.0)
    b = _write_capture(tmp_path, name="B", items=(b"b-bytes",), captured_at=2.0)
    c = _write_capture(tmp_path, name="C", items=(b"c-bytes",), captured_at=3.0)

    # Deliberately reversed and a subset, to prove --replay-ids controls BOTH which captures
    # are used and the order they come back in, independent of captured_at.
    chosen = [c.replay_id, a.replay_id]
    captures = rc.load_replay_corpus(tmp_path, replay_ids=chosen)

    assert [cap.replay_id for cap in captures] == chosen
    assert b.replay_id not in [cap.replay_id for cap in captures]


def test_load_replay_corpus_limit_applies_after_ordering(tmp_path):
    a = _write_capture(tmp_path, name="A", items=(b"a-bytes",), captured_at=1.0)
    _write_capture(tmp_path, name="B", items=(b"b-bytes",), captured_at=2.0)
    c = _write_capture(tmp_path, name="C", items=(b"c-bytes",), captured_at=3.0)

    # explicit order reversed relative to captured_at; limit=1 must keep only the FIRST of
    # THIS order (c), not the earliest by captured_at (a).
    captures = rc.load_replay_corpus(tmp_path, replay_ids=[c.replay_id, a.replay_id], limit=1)

    assert [cap.replay_id for cap in captures] == [c.replay_id]


# ---------------------------------------------------------------------------------------
# replay_corpus.py -- content-derived id, validation, safe-to-fail writer
# ---------------------------------------------------------------------------------------

def test_compute_replay_id_changes_with_content():
    base = rc.compute_replay_id(name="Alex", items=(b"photo-a",), context=(), truncated=False)
    different_name = rc.compute_replay_id(name="Sam", items=(b"photo-a",), context=(),
                                          truncated=False)
    different_photo = rc.compute_replay_id(name="Alex", items=(b"photo-b",), context=(),
                                           truncated=False)
    different_truncated = rc.compute_replay_id(name="Alex", items=(b"photo-a",), context=(),
                                               truncated=True)
    same_again = rc.compute_replay_id(name="Alex", items=(b"photo-a",), context=(),
                                      truncated=False)

    assert base == same_again
    assert len({base, different_name, different_photo, different_truncated}) == 4


def test_write_replay_capture_rejects_zero_items(tmp_path):
    result = rc.write_replay_capture(tmp_path, items=(), name="Nobody")

    assert result.ok is False
    assert "numbered item" in result.error
    assert list(tmp_path.iterdir()) == []  # nothing was left on disk


def test_write_replay_capture_never_raises_on_a_disk_failure(tmp_path, monkeypatch):
    """The writer's core safety property: a real OSError from the filesystem must come back
    as ReplayWriteResult(ok=False), never propagate."""
    def _boom(self, *_a, **_k):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(rc.Path, "write_bytes", _boom)

    result = rc.write_replay_capture(tmp_path, items=(b"item-bytes",), name="Casey")

    assert result.ok is False
    assert "disk full" in result.error


def test_load_replay_capture_raises_on_missing_replay_id(tmp_path):
    with pytest.raises(FileNotFoundError):
        rc.load_replay_capture(tmp_path, "does-not-exist")


def test_load_replay_capture_raises_when_manifest_replay_id_disagrees_with_directory(tmp_path):
    written = _write_capture(tmp_path, items=(b"item-bytes",))
    manifest_path = written.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["replay_id"] = "some-other-id"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="does not match"):
        rc.load_replay_capture(tmp_path, written.replay_id)


# ---------------------------------------------------------------------------------------
# tools/opener_replay.py -- dry run: no network, no store write
# ---------------------------------------------------------------------------------------

def test_main_dry_run_makes_no_network_call_and_no_store_write(tmp_path, capsys):
    corpus_dir = tmp_path / "corpus"
    _write_capture(corpus_dir, name="Alex", items=(b"item-one", b"item-two"),
                   context=(b"context-one",))
    config_path = _config_path(tmp_path)
    db_path = tmp_path / "store.db"

    rc_code = m.main(
        ["--corpus-dir", str(corpus_dir), "--config", str(config_path), "--db", str(db_path)],
        transport=_forbidden_transport, store=_ForbiddenStore())

    assert rc_code == 0
    assert not db_path.exists()   # the store file was never even opened
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "item_count=2" in out
    assert "context_count=1" in out


def test_main_dry_run_json_report_has_counts_and_models(tmp_path, capsys):
    corpus_dir = tmp_path / "corpus"
    written = _write_capture(corpus_dir, name="Robin", items=(b"only-item",))
    config_path = _config_path(tmp_path, models=("gemini-a", "gemini-b"))

    rc_code = m.main(
        ["--corpus-dir", str(corpus_dir), "--config", str(config_path), "--json"],
        transport=_forbidden_transport, store=_ForbiddenStore())

    assert rc_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "dry_run"
    assert report["models"] == ["gemini-a", "gemini-b"]
    assert len(report["captures"]) == 1
    entry = report["captures"][0]
    assert entry["replay_id"] == written.replay_id
    assert entry["item_count"] == 1
    assert entry["context_count"] == 0
    assert isinstance(entry["estimated_request_bytes"], int)
    assert entry["estimated_request_bytes"] > 0
    assert entry["name_present"] is True
    assert "Robin" not in json.dumps(entry)  # her literal name is never echoed into the report


def test_main_dry_run_reports_nothing_to_do_on_an_empty_corpus(tmp_path, capsys):
    config_path = _config_path(tmp_path)
    rc_code = m.main(["--corpus-dir", str(tmp_path / "empty"), "--config", str(config_path)],
                    transport=_forbidden_transport, store=_ForbiddenStore())
    assert rc_code == 0
    assert "Nothing to do" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------
# tools/opener_replay.py -- --live: confirmation gate, api key requirement
# ---------------------------------------------------------------------------------------

def test_main_live_requires_confirmation_and_does_not_call_transport_when_declined(tmp_path):
    corpus_dir = tmp_path / "corpus"
    _write_capture(corpus_dir, items=(b"item-bytes",))
    config_path = _config_path(tmp_path)
    transport = _ScriptedTransport([])

    rc_code = m.main(
        ["--corpus-dir", str(corpus_dir), "--config", str(config_path), "--live"],
        transport=transport, env={"GEMINI_API_KEY": "k"}, confirm=lambda _p: False,
        store=_ForbiddenStore())

    assert rc_code != 0
    assert transport.calls == []


def test_main_live_requires_api_key(tmp_path):
    corpus_dir = tmp_path / "corpus"
    _write_capture(corpus_dir, items=(b"item-bytes",))
    config_path = _config_path(tmp_path)
    # A transport that WOULD succeed if it were ever called -- a stronger canary than one that
    # merely raises: it proves the missing-key guard is what stops the call, not an unrelated
    # zero-successes fallback that would also make rc_code non-zero on its own.
    transport = _ScriptedTransport([_success()])

    rc_code = m.main(
        ["--corpus-dir", str(corpus_dir), "--config", str(config_path), "--live", "--yes"],
        transport=transport, env={}, store=_ForbiddenStore())

    assert rc_code != 0
    assert transport.calls == []


# ---------------------------------------------------------------------------------------
# tools/opener_replay.py -- --live: the replay marker, the CURRENT prompt_sha256, real store
# ---------------------------------------------------------------------------------------

def test_main_live_writes_replay_marker_under_the_current_prompt_era(tmp_path):
    corpus_dir = tmp_path / "corpus"
    _write_capture(corpus_dir, name="Alex", items=(b"item-bytes",),
                  prompt_sha256="some-stale-captured-era")
    config_path = _config_path(tmp_path, style="Current live style guide text.")
    db_path = tmp_path / "store.db"
    transport = _ScriptedTransport([_success("Great trail, where was that?")])

    rc_code = m.main(
        ["--corpus-dir", str(corpus_dir), "--config", str(config_path), "--db", str(db_path),
         "--live", "--yes", "--run-id", "test-run-1"],
        transport=transport, env={"GEMINI_API_KEY": "k"})

    assert rc_code == 0
    assert len(transport.calls) == 1

    store = SQLiteStore(db_path)
    try:
        rows = store.con.execute(
            "SELECT run_id, decision, prompt_sha256, opener, model_item_index FROM openers"
        ).fetchall()
    finally:
        store.close()

    assert len(rows) == 1
    run_id, decision, prompt_sha256, opener_text, item_index = rows[0]
    assert run_id == "test-run-1"
    assert decision == m.DECISION_REPLAY
    # The CURRENT config's era, never the stale one recorded at capture time.
    from operation_love.opener.opener import prompt_stamp
    cfg = yaml.safe_load(config_path.read_text())
    assert prompt_sha256 == prompt_stamp(cfg["opener"]["style"])
    assert prompt_sha256 != "some-stale-captured-era"
    assert opener_text == "Great trail, where was that?"
    assert item_index == 1


def test_main_live_replay_marker_is_never_the_sent_decision_value(tmp_path):
    """decision_bucket() in tools/opener_corpus_report.py treats decision == "like" as a real
    landed send; the whole point of DECISION_REPLAY is that it can never collide with that."""
    assert m.DECISION_REPLAY != "like"
    assert m.DECISION_REPLAY not in ("like", "dislike", "never_sent", "")


# ---------------------------------------------------------------------------------------
# tools/opener_replay.py -- --purge: remove captures from the corpus, a wholly separate action
# from replaying. Dry run is the default (deletes nothing); --delete performs the removal.
# ---------------------------------------------------------------------------------------

def test_purge_dry_run_whole_corpus_deletes_nothing(tmp_path, capsys):
    corpus_dir = tmp_path / "corpus"
    a = _write_capture(corpus_dir, name="A", items=(b"a-bytes",), captured_at=100.0)
    b = _write_capture(corpus_dir, name="B", items=(b"b-bytes",), captured_at=200.0)

    rc_code = m.main(["--corpus-dir", str(corpus_dir), "--purge"],
                     transport=_forbidden_transport, store=_ForbiddenStore())

    assert rc_code == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "would remove 2 capture(s)" in out
    assert "nothing was deleted" in out
    # THE CORE GUARANTEE: nothing on disk was touched.
    assert (corpus_dir / a.replay_id).is_dir()
    assert (corpus_dir / b.replay_id).is_dir()


def test_purge_delete_whole_corpus_removes_everything_and_prints_what_it_removed(
        tmp_path, capsys):
    corpus_dir = tmp_path / "corpus"
    a = _write_capture(corpus_dir, name="A", items=(b"a-bytes",), captured_at=100.0)
    b = _write_capture(corpus_dir, name="B", items=(b"b-bytes",), captured_at=200.0)

    rc_code = m.main(["--corpus-dir", str(corpus_dir), "--purge", "--delete"],
                     transport=_forbidden_transport, store=_ForbiddenStore())

    assert rc_code == 0
    out = capsys.readouterr().out
    assert "removed 2 capture(s)" in out
    assert a.replay_id in out
    assert b.replay_id in out
    assert not corpus_dir.exists()


def test_purge_dry_run_older_than_days_reports_only_the_old_one(tmp_path, capsys):
    corpus_dir = tmp_path / "corpus"
    now = time.time()
    old = _write_capture(corpus_dir, name="Old", items=(b"old-bytes",),
                         captured_at=now - 30 * 86400)
    new = _write_capture(corpus_dir, name="New", items=(b"new-bytes",),
                         captured_at=now - 1 * 86400)

    rc_code = m.main(
        ["--corpus-dir", str(corpus_dir), "--purge", "--purge-older-than-days", "10"],
        transport=_forbidden_transport, store=_ForbiddenStore())

    assert rc_code == 0
    out = capsys.readouterr().out
    assert "would remove 1 capture(s)" in out
    assert old.replay_id in out
    assert new.replay_id not in out
    # THE CORE GUARANTEE: even the capture that WOULD be removed is left alone in dry-run mode.
    assert (corpus_dir / old.replay_id).is_dir()
    assert (corpus_dir / new.replay_id).is_dir()


def test_purge_delete_older_than_days_removes_only_the_old_one(tmp_path, capsys):
    corpus_dir = tmp_path / "corpus"
    now = time.time()
    old = _write_capture(corpus_dir, name="Old", items=(b"old-bytes",),
                         captured_at=now - 30 * 86400)
    new = _write_capture(corpus_dir, name="New", items=(b"new-bytes",),
                         captured_at=now - 1 * 86400)

    rc_code = m.main(
        ["--corpus-dir", str(corpus_dir), "--purge", "--purge-older-than-days", "10", "--delete"],
        transport=_forbidden_transport, store=_ForbiddenStore())

    assert rc_code == 0
    out = capsys.readouterr().out
    assert "removed 1 capture(s)" in out
    assert old.replay_id in out
    assert not (corpus_dir / old.replay_id).exists()
    assert (corpus_dir / new.replay_id).is_dir()


def test_purge_on_empty_corpus_reports_nothing_matched_and_succeeds(tmp_path, capsys):
    rc_code = m.main(["--corpus-dir", str(tmp_path / "empty"), "--purge"],
                     transport=_forbidden_transport, store=_ForbiddenStore())
    assert rc_code == 0
    out = capsys.readouterr().out
    assert "nothing matched" in out


def test_purge_json_mode_reports_structured_removed_list_and_stays_a_dry_run(tmp_path, capsys):
    corpus_dir = tmp_path / "corpus"
    a = _write_capture(corpus_dir, name="A", items=(b"a-bytes",), captured_at=100.0)

    rc_code = m.main(["--corpus-dir", str(corpus_dir), "--purge", "--json"],
                     transport=_forbidden_transport, store=_ForbiddenStore())

    assert rc_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "dry_run"
    assert report["purge_older_than_days"] is None
    assert len(report["removed"]) == 1
    assert report["removed"][0]["replay_id"] == a.replay_id
    assert (corpus_dir / a.replay_id).is_dir()   # still a dry run -- nothing deleted


def test_purge_json_delete_mode_reports_delete_and_the_age_bound(tmp_path, capsys):
    corpus_dir = tmp_path / "corpus"
    now = time.time()
    old = _write_capture(corpus_dir, name="Old", items=(b"old-bytes",),
                         captured_at=now - 30 * 86400)

    rc_code = m.main(
        ["--corpus-dir", str(corpus_dir), "--purge", "--purge-older-than-days", "10",
         "--delete", "--json"],
        transport=_forbidden_transport, store=_ForbiddenStore())

    assert rc_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "delete"
    assert report["purge_older_than_days"] == 10.0
    assert report["removed"][0]["replay_id"] == old.replay_id
    assert report["removed"][0]["reason"] == "max_age_days"
    assert not (corpus_dir / old.replay_id).exists()


def test_purge_never_reads_config_or_touches_the_transport_or_store(tmp_path):
    # --purge is dispatched before any of the replay-specific setup (config.yaml, transport,
    # store) -- prove it by pointing --config at a file that does not exist at all: a replay run
    # would fail loudly on that, --purge must not even look at it.
    corpus_dir = tmp_path / "corpus"
    _write_capture(corpus_dir, items=(b"item-bytes",))

    rc_code = m.main(
        ["--corpus-dir", str(corpus_dir), "--config", str(tmp_path / "nope.yaml"), "--purge"],
        transport=_forbidden_transport, store=_ForbiddenStore())

    assert rc_code == 0


def test_purge_age_based_reports_an_error_for_a_negative_age(tmp_path, capsys):
    corpus_dir = tmp_path / "corpus"
    _write_capture(corpus_dir, items=(b"item-bytes",))

    rc_code = m.main(
        ["--corpus-dir", str(corpus_dir), "--purge", "--purge-older-than-days", "-1"],
        transport=_forbidden_transport, store=_ForbiddenStore())

    assert rc_code != 0
    err = capsys.readouterr().err
    assert "ERROR" in err


def test_main_replay_ids_flag_gives_a_deterministic_subset_and_order(tmp_path, capsys):
    corpus_dir = tmp_path / "corpus"
    a = _write_capture(corpus_dir, name="A", items=(b"a-bytes",), captured_at=1.0)
    _write_capture(corpus_dir, name="B", items=(b"b-bytes",), captured_at=2.0)
    c = _write_capture(corpus_dir, name="C", items=(b"c-bytes",), captured_at=3.0)
    config_path = _config_path(tmp_path)

    chosen = f"{c.replay_id},{a.replay_id}"
    first_run = m.main(
        ["--corpus-dir", str(corpus_dir), "--config", str(config_path), "--json",
         "--replay-ids", chosen],
        transport=_forbidden_transport, store=_ForbiddenStore())
    first_report = json.loads(capsys.readouterr().out)

    second_run = m.main(
        ["--corpus-dir", str(corpus_dir), "--config", str(config_path), "--json",
         "--replay-ids", chosen],
        transport=_forbidden_transport, store=_ForbiddenStore())
    second_report = json.loads(capsys.readouterr().out)

    assert first_run == 0 and second_run == 0
    first_ids = [entry["replay_id"] for entry in first_report["captures"]]
    second_ids = [entry["replay_id"] for entry in second_report["captures"]]
    assert first_ids == [c.replay_id, a.replay_id]
    assert first_ids == second_ids
