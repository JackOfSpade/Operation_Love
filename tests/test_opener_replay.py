"""operation_love/opener/replay_corpus.py (the on-disk format + writer) and
tools/opener_replay.py (the CLI harness that replays it through the current prompt).

No network, ever: every CLI test injects a fake GeminiTransport exactly as
tests/test_gemini_opener.py and tests/test_gemini_model_probe.py do. NEVER make a real API call
from a test.
"""
from __future__ import annotations

import base64
import ctypes
import json
import os
import shutil
import struct
import threading
import time
from pathlib import Path

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
    assert [p.name for p in tmp_path.iterdir() if not p.name.startswith(".lock-")] == [
        first.replay_id
    ]


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

    real_open = rc.os.open

    def _disk_full(path, flags, *args, **kwargs):
        if str(path).startswith("item_"):
            raise OSError("disk full (simulated)")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(rc.os, "open", _disk_full)

    result = rc.write_replay_capture(tmp_path, items=(b"item-bytes",), name="Casey")

    assert result.ok is False
    assert "disk full" in result.error


def test_interrupted_staged_write_leaves_no_canonical_partial_and_retry_succeeds(tmp_path, monkeypatch):
    """A failed crop write cannot poison its deterministic replay id forever."""
    original = rc._write_new_capture_file
    failed = False

    def fail_first_crop(directory_fd, name, data):
        nonlocal failed
        if name.startswith("item_") and not failed:
            failed = True
            original(directory_fd, name, data)
            raise OSError("simulated crash after partial crop")
        return original(directory_fd, name, data)

    monkeypatch.setattr(rc, "_write_new_capture_file", fail_first_crop)
    first = rc.write_replay_capture(tmp_path, name="Casey", items=(b"same",))
    assert not first.ok
    replay_id = rc.compute_replay_id(name="Casey", items=(b"same",))
    assert not (tmp_path / replay_id).exists()

    monkeypatch.setattr(rc, "_write_new_capture_file", original)
    second = rc.write_replay_capture(tmp_path, name="Casey", items=(b"same",))
    assert second.ok, second.error
    assert rc.load_replay_capture(tmp_path, replay_id).items == (b"same",)


def test_legacy_canonical_partial_crop_is_quarantined_and_the_same_write_recovers(tmp_path):
    """Exact pre-transaction repro: crop present, manifest absent, then retry same content."""
    first = _write_capture(tmp_path, name="Casey", items=(b"same",))
    (first.path / "manifest.json").unlink()

    recovered = rc.write_replay_capture(tmp_path, name="Casey", items=(b"same",))

    assert recovered.ok, recovered.error
    assert rc.load_replay_capture(tmp_path, first.replay_id).items == (b"same",)
    quarantines = [p for p in tmp_path.iterdir() if p.name.startswith(".incomplete-")]
    assert len(quarantines) == 1
    assert (quarantines[0] / "item_001.bin").read_bytes() == b"same"


def test_two_writers_cannot_quarantine_a_completed_legacy_recovery_winner(tmp_path, monkeypatch):
    """The legacy observation and its rename share the per-id OS lock."""
    first = _write_capture(tmp_path, name="Casey", items=(b"same",))
    (first.path / "manifest.json").unlink()
    original = rc._quarantine_incomplete_capture
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def pause_first(root_fd, replay_id):
        calls.append(replay_id)
        original(root_fd, replay_id)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)

    monkeypatch.setattr(rc, "_quarantine_incomplete_capture", pause_first)
    results = []
    a = threading.Thread(target=lambda: results.append(
        rc.write_replay_capture(tmp_path, name="Casey", items=(b"same",))))
    b = threading.Thread(target=lambda: results.append(
        rc.write_replay_capture(tmp_path, name="Casey", items=(b"same",))))
    a.start()
    assert entered.wait(5)
    b.start()
    release.set()
    a.join(5)
    b.join(5)

    assert all(result.ok for result in results)
    assert calls == [first.replay_id]
    assert rc.load_replay_capture(tmp_path, first.replay_id).items == (b"same",)
    assert all(not (path / "manifest.json").exists()
               for path in tmp_path.iterdir() if path.name.startswith(".incomplete-"))


def test_idempotent_writer_fails_closed_on_tampered_completed_crop(tmp_path):
    written = _write_capture(tmp_path, name="Casey", items=(b"same",))
    (written.path / "item_001.bin").write_bytes(b"tampered")

    result = rc.write_replay_capture(tmp_path, name="Casey", items=(b"same",))

    assert not result.ok
    assert "canonical" in result.error


def test_unverified_manifest_absent_canonical_directory_is_never_quarantined(tmp_path):
    replay_id = rc.compute_replay_id(name="Casey", items=(b"same",))
    partial = tmp_path / replay_id
    partial.mkdir()
    (partial / "item_001.bin").write_bytes(b"different")

    result = rc.write_replay_capture(tmp_path, name="Casey", items=(b"same",))

    assert not result.ok
    assert partial.is_dir()
    assert (partial / "item_001.bin").read_bytes() == b"different"


def test_publish_collision_only_accepts_a_valid_completed_winner(tmp_path, monkeypatch):
    """A competing canonical directory is never removed or replaced by our staged writer."""
    original = rc._publish_staging_capture
    replay_id = rc.compute_replay_id(name="Casey", items=(b"same",))
    winner = rc.write_replay_capture(tmp_path, name="Casey", items=(b"same",))
    assert winner.ok

    def collide(root_fd, staging, target):
        assert target == replay_id
        raise FileExistsError(target)

    monkeypatch.setattr(rc, "_publish_staging_capture", collide)
    result = rc.write_replay_capture(tmp_path, name="Casey", items=(b"same",))
    assert result.ok, result.error
    assert rc.load_replay_capture(tmp_path, replay_id).items == (b"same",)
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".write-")]
    monkeypatch.setattr(rc, "_publish_staging_capture", original)


def test_writer_refuses_a_precreated_capture_symlink(tmp_path):
    """A predictable content id must not let a local attacker redirect a future write."""
    outside = tmp_path / "outside"
    outside.mkdir()
    replay_id = rc.compute_replay_id(name="Alex", items=(b"item",))
    os.symlink(outside, tmp_path / replay_id)

    result = rc.write_replay_capture(tmp_path, name="Alex", items=(b"item",))

    assert result.ok is False
    assert list(outside.iterdir()) == []


def test_writer_refuses_a_symlinked_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root_link = tmp_path / "corpus"
    os.symlink(outside, root_link)

    result = rc.write_replay_capture(root_link, name="Alex", items=(b"item",))

    assert result.ok is False
    assert list(outside.iterdir()) == []
    assert rc.list_replay_ids(root_link) == []
    with pytest.raises(FileNotFoundError):
        rc.load_replay_capture(root_link, "0" * 64)
    assert rc.prune_replay_corpus(root_link, max_captures=1).ok is True
    assert list(outside.iterdir()) == []


def test_every_root_path_component_is_no_follow_for_write_read_prune_and_purge(tmp_path, capsys):
    """O_NOFOLLOW on only the final root component would let this escape via ``hop``."""
    outside = tmp_path / "outside"
    outside.mkdir()
    container = tmp_path / "container"
    container.mkdir()
    os.symlink(outside, container / "hop")
    root = container / "hop" / "corpus"

    result = rc.write_replay_capture(root, name="Alex", items=(b"item",))

    assert result.ok is False
    assert list(outside.iterdir()) == []
    assert rc.list_replay_ids(root) == []
    with pytest.raises(FileNotFoundError):
        rc.load_replay_capture(root, "0" * 64)
    assert rc.prune_replay_corpus(root, max_captures=1).ok is True
    assert m.main(["--corpus-dir", str(root), "--purge"]) == 0
    assert "nothing matched" in capsys.readouterr().out
    assert list(outside.iterdir()) == []


def test_writer_rejects_root_component_replaced_by_symlink_during_creation(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "new-corpus"
    real_mkdir = rc.os.mkdir
    swapped = False

    def create_then_swap(path, *args, **kwargs):
        nonlocal swapped
        result = real_mkdir(path, *args, **kwargs)
        if path == "new-corpus" and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            root.rmdir()
            os.symlink(outside, root)
        return result

    monkeypatch.setattr(rc.os, "mkdir", create_then_swap)
    result = rc.write_replay_capture(root, name="Alex", items=(b"item",))

    assert result.ok is False
    assert list(outside.iterdir()) == []


def test_writer_stays_with_open_root_after_root_path_swap(tmp_path, monkeypatch):
    root = tmp_path / "corpus"
    root.mkdir()
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    hidden = tmp_path / "hidden"
    real_mkdir = rc.os.mkdir
    swapped = False

    def swap_before_capture_create(path, *args, **kwargs):
        nonlocal swapped
        if not swapped and kwargs.get("dir_fd") is not None:
            swapped = True
            root.rename(hidden)
            replacement.rename(root)
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(rc.os, "mkdir", swap_before_capture_create)
    result = rc.write_replay_capture(root, name="Alex", items=(b"item",))

    assert result.ok
    assert (hidden / result.replay_id / "manifest.json").is_file()
    assert list(root.iterdir()) == []


def test_load_replay_capture_raises_on_missing_replay_id(tmp_path):
    with pytest.raises(FileNotFoundError):
        rc.load_replay_capture(tmp_path, "0" * 64)


def test_load_replay_capture_raises_when_manifest_replay_id_disagrees_with_directory(tmp_path):
    written = _write_capture(tmp_path, items=(b"item-bytes",))
    manifest_path = written.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["replay_id"] = "some-other-id"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="does not match"):
        rc.load_replay_capture(tmp_path, written.replay_id)


@pytest.mark.parametrize("field", ["items", "context"])
@pytest.mark.parametrize("unsafe_name", ["../outside.bin", "/tmp/outside.bin", "..\\outside.bin"])
def test_load_replay_capture_rejects_manifest_paths_outside_its_directory(
        tmp_path, field, unsafe_name):
    written = _write_capture(tmp_path, items=(b"safe-item",), context=(b"safe-context",))
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"must never be read")
    manifest_path = written.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = [unsafe_name]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="flat filename"):
        rc.load_replay_capture(tmp_path, written.replay_id)


def test_load_replay_capture_rejects_symlinked_capture_and_item_file(tmp_path):
    outside_root = tmp_path / "outside"
    outside = _write_capture(outside_root, items=(b"outside-item",))
    linked_capture = tmp_path / outside.replay_id
    os.symlink(outside.path, linked_capture)

    assert outside.replay_id not in rc.list_replay_ids(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        rc.load_replay_capture(tmp_path, outside.replay_id)

    written = _write_capture(tmp_path, items=(b"safe-item",))
    item_path = next(written.path.glob("item_*"))
    item_path.unlink()
    os.symlink(outside.path / "item_001.bin", item_path)
    with pytest.raises(ValueError, match="missing or unsafe"):
        rc.load_replay_capture(tmp_path, written.replay_id)


@pytest.mark.parametrize("target", ["manifest.json", "item_001.bin"])
def test_load_replay_capture_rejects_hard_linked_manifest_and_crops(tmp_path, target):
    written = _write_capture(tmp_path, items=(b"safe-item",))
    source = written.path / target
    os.link(source, tmp_path / f"linked-{target}")

    with pytest.raises(ValueError, match="singly-linked regular file"):
        rc.load_replay_capture(tmp_path, written.replay_id)


@pytest.mark.parametrize("target", ["manifest.json", "item_001.bin"])
def test_load_replay_capture_rejects_fifo_manifest_and_crops_without_blocking(tmp_path, target):
    written = _write_capture(tmp_path, items=(b"safe-item",))
    path = written.path / target
    path.unlink()
    os.mkfifo(path)

    with pytest.raises(ValueError, match="singly-linked regular file"):
        rc.load_replay_capture(tmp_path, written.replay_id)


def test_load_replay_capture_keeps_reading_the_open_root_after_a_root_path_swap(tmp_path, monkeypatch):
    root = tmp_path / "corpus"
    written = _write_capture(root, items=(b"original-item",))
    replacement = tmp_path / "replacement"
    replacement_capture = replacement / written.replay_id
    replacement_capture.mkdir(parents=True)
    (replacement_capture / "manifest.json").write_bytes(
        (written.path / "manifest.json").read_bytes())
    (replacement_capture / "item_001.bin").write_bytes(b"replacement-item")
    moved = tmp_path / "original-root-now-hidden"
    real_open_capture = rc._open_capture_directory

    def swap_root_then_open(root_fd, replay_id):
        root.rename(moved)
        replacement.rename(root)
        return real_open_capture(root_fd, replay_id)

    monkeypatch.setattr(rc, "_open_capture_directory", swap_root_then_open)
    capture = rc.load_replay_capture(root, written.replay_id)

    assert capture.items == (b"original-item",)


def test_load_replay_capture_rejects_noncanonical_replay_id_before_accessing_root(tmp_path):
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        rc.load_replay_capture(tmp_path, "../not-a-capture")


def test_windows_native_relative_name_uses_utf16_bytes_and_refuses_wraparound():
    """A counted NT name must not turn a long manifest filename into its short prefix."""
    calls = []

    class Ntdll:
        @staticmethod
        def NtCreateFile(handle_ptr, _access, attrs_ptr, *_rest):
            calls.append(attrs_ptr._obj.ObjectName.contents)
            ctypes.cast(handle_ptr, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(7)
            return 0

    native = object.__new__(rc._NativeWindowsCorpusFilesystem)
    native._ctypes = ctypes
    native._ntdll = Ntdll()
    native._validate = lambda *_args, **_kwargs: None

    # One astral code point occupies two UTF-16 code units, not one Python code point.
    assert native._open_relative(1, "😀", directory=False) == 7
    assert calls[0].Length == 4
    assert calls[0].MaximumLength == 6

    calls.clear()
    # Without the bound, this would wrap Length to the byte length of item_001.bin and open it.
    with pytest.raises(ValueError, match="too long"):
        native._open_relative(1, "item_001.bin" + ("x" * 32768), directory=False)
    assert calls == []


def _directory_information_record(name: str, *, next_offset: int = 0,
                                  filename_length: int | None = None) -> bytes:
    encoded = name.encode("utf-16-le")
    length = len(encoded) if filename_length is None else filename_length
    return (struct.pack("<II", next_offset, 0) + (b"\0" * 48)
            + struct.pack("<II", 0, length) + encoded)


def test_windows_native_directory_query_grows_for_overflow_and_zero_information():
    record = _directory_information_record("capture")

    class Ntdll:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.sizes = []

        def NtQueryDirectoryFile(self, _directory, _event, _apc, _context, iosb_ptr,
                                 buffer, length, *_rest):
            status, payload = next(self.responses)
            self.sizes.append(length)
            iosb = ctypes.cast(iosb_ptr, ctypes.POINTER(rc._WinIoStatusBlock)).contents
            iosb.Information = len(payload)
            buffer[:len(payload)] = payload
            return status

    for first_status in (rc._NativeWindowsCorpusFilesystem._STATUS_BUFFER_OVERFLOW, 0):
        ntdll = Ntdll([(first_status, b""), (0, record),
                       (rc._NativeWindowsCorpusFilesystem._STATUS_NO_MORE_FILES, b"")])
        native = object.__new__(rc._NativeWindowsCorpusFilesystem)
        native._ctypes = ctypes
        native._ntdll = ntdll

        assert native.listdir(7) == ["capture"]
        assert ntdll.sizes[:2] == [65536, 131072]


@pytest.mark.parametrize("record", [
    _directory_information_record("x", next_offset=2),
    _directory_information_record("x", filename_length=4)[:-1],
])
def test_windows_native_directory_parser_rejects_malformed_record_bounds(record):
    with pytest.raises(ValueError, match="FILE_DIRECTORY_INFORMATION"):
        rc._NativeWindowsCorpusFilesystem._parse_directory_information(record)


def test_delete_replay_captures_skips_a_replaced_nondirectory_and_continues(tmp_path, monkeypatch):
    root = tmp_path / "corpus"
    replaced = _write_capture(root, name="replaced", items=(b"one",))
    survivor = _write_capture(root, name="survivor", items=(b"two",))
    real_open = rc._open_capture_directory

    def replaced_before_open(root_fd, replay_id):
        if replay_id == replaced.replay_id:
            raise ValueError("capture was replaced by a file")
        return real_open(root_fd, replay_id)

    monkeypatch.setattr(rc, "_open_capture_directory", replaced_before_open)

    assert rc.delete_replay_captures(root, [replaced.replay_id, survivor.replay_id]) == (
        survivor.replay_id,)
    assert replaced.path.is_dir()
    assert not survivor.path.exists()


def test_delete_replay_captures_never_deletes_a_manifestless_same_id_replacement(tmp_path):
    root = tmp_path / "corpus"
    written = _write_capture(root, name="Casey", items=(b"same",))
    (written.path / "manifest.json").unlink()

    assert rc.delete_replay_captures(root, [written.replay_id]) == ()
    assert written.path.is_dir()
    assert (written.path / "item_001.bin").read_bytes() == b"same"


def test_delete_replay_captures_preserves_a_replacement_after_validation(tmp_path, monkeypatch):
    root = tmp_path / "corpus"
    written = _write_capture(root, name="Casey", items=(b"same",))
    real_rename = rc.os.rename
    replaced = False

    def replace_before_quarantine(source, target, *args, **kwargs):
        nonlocal replaced
        if source == written.replay_id and not replaced:
            replaced = True
            shutil.rmtree(written.path)
            written.path.mkdir()
            (written.path / "keep.txt").write_text("replacement", encoding="utf-8")
        return real_rename(source, target, *args, **kwargs)

    monkeypatch.setattr(rc.os, "rename", replace_before_quarantine)
    real_identity = rc._capture_identity
    identity_calls = 0

    def create_new_destination_on_mismatch(directory_fd):
        nonlocal identity_calls
        identity_calls += 1
        if identity_calls == 2:
            written.path.mkdir()
            (written.path / "newer.txt").write_text("new destination", encoding="utf-8")
        return real_identity(directory_fd)

    monkeypatch.setattr(rc, "_capture_identity", create_new_destination_on_mismatch)
    assert rc.delete_replay_captures(root, [written.replay_id]) == ()
    assert (written.path / "newer.txt").read_text(encoding="utf-8") == "new destination"
    quarantines = [path for path in root.iterdir() if path.name.startswith(".prune-")]
    assert len(quarantines) == 1
    assert (quarantines[0] / "keep.txt").read_text(encoding="utf-8") == "replacement"


class _FakeWindowsCorpusFilesystem:
    """Deterministic handle model for the Windows adapter seam.

    It intentionally has no pathname operation after ``open_root``: handles carry an opaque
    root token, so a test can model a root path being replaced and prove the corpus code still
    uses the original handle.  Native Windows coverage belongs to CI on Windows; this lets the
    normal Linux suite exercise the same adapter/race decisions.
    """

    def __init__(self):
        self.captures = {}
        self.reparse_captures = set()
        self.reparse_files = set()
        self.hardlinked_files = set()
        self.closed = []
        self.root_token = object()
        self.remove_roots = []
        self.swap_before_remove = set()
        self.locks = {}

    def open_root(self, root):
        return Path(root), ("root", self.root_token)

    def ensure_root(self, root):
        return self.open_root(root)

    def close(self, handle):
        self.closed.append(handle)
        if isinstance(handle, tuple) and handle[0] == "lock":
            handle[2].release()

    def lock_capture(self, root, replay_id):
        assert root == ("root", self.root_token)
        lock = self.locks.setdefault(replay_id, threading.Lock())
        lock.acquire()
        return ("lock", replay_id, lock)

    def listdir(self, root):
        if root == ("root", self.root_token):
            return list(self.captures)
        assert root[0] == "capture" and root[2] is self.root_token
        return list(self.captures[root[1]])

    def open_capture(self, root, replay_id):
        assert root == ("root", self.root_token)
        if replay_id in self.reparse_captures or replay_id not in self.captures:
            raise OSError("reparse point or missing capture")
        return ("capture", replay_id, self.root_token)

    def mkdir_capture(self, root, replay_id):
        assert root == ("root", self.root_token)
        if replay_id in self.captures:
            raise FileExistsError(replay_id)
        self.captures[replay_id] = {}

    def manifest_exists(self, directory):
        return "manifest.json" in self.captures[directory[1]]

    def read_regular(self, directory, name):
        key = (directory[1], name)
        if key in self.reparse_files or key in self.hardlinked_files:
            raise OSError("unsafe reparse point or hard link")
        try:
            return self.captures[directory[1]][name]
        except KeyError as exc:
            raise OSError("missing file") from exc

    def write_new(self, directory, name, data):
        files = self.captures[directory[1]]
        if name in files:
            raise FileExistsError(name)
        files[name] = data

    def replace(self, directory, source, target):
        files = self.captures[directory[1]]
        files[target] = files.pop(source)

    def publish_capture(self, root, staging, replay_id):
        assert root == ("root", self.root_token)
        if replay_id in self.captures:
            raise FileExistsError(replay_id)
        self.captures[replay_id] = self.captures.pop(staging)

    def remove_capture(self, root, replay_id):
        self.remove_roots.append(root)
        assert root == ("root", self.root_token)
        if replay_id in self.reparse_captures or replay_id in self.swap_before_remove:
            raise OSError("capture replaced by a reparse point")
        del self.captures[replay_id]

    def capture_identity(self, directory):
        return id(self.captures[directory[1]])

    def remove_open_capture(self, directory, expected_identity):
        replay_id = directory[1]
        self.remove_roots.append(("root", directory[2]))
        if replay_id in self.reparse_captures or replay_id in self.swap_before_remove:
            return False
        if replay_id not in self.captures or id(self.captures[replay_id]) != expected_identity:
            return False
        del self.captures[replay_id]
        return True


def test_windows_handle_adapter_supports_write_list_load_prune_and_purge(tmp_path, monkeypatch):
    windows = _FakeWindowsCorpusFilesystem()
    monkeypatch.setattr(rc, "_WINDOWS_FILESYSTEM", windows)
    root = tmp_path / "corpus"
    old = _write_capture(root, name="old", items=(b"old-item",), captured_at=1.0)
    fresh = _write_capture(root, name="fresh", items=(b"fresh-item",), captured_at=time.time())

    assert rc.list_replay_ids(root) == [old.replay_id, fresh.replay_id]
    assert rc.load_replay_capture(root, fresh.replay_id).items == (b"fresh-item",)
    pruned = rc.prune_replay_corpus(root, max_captures=1)
    assert [entry.replay_id for entry in pruned.removed] == [old.replay_id]
    assert rc.delete_replay_captures(root, [fresh.replay_id]) == (fresh.replay_id,)
    assert windows.captures == {}


def test_windows_handle_adapter_recovers_the_legacy_manifest_absent_partial(tmp_path, monkeypatch):
    windows = _FakeWindowsCorpusFilesystem()
    monkeypatch.setattr(rc, "_WINDOWS_FILESYSTEM", windows)
    root = tmp_path / "corpus"
    first = _write_capture(root, name="Casey", items=(b"same",))
    del windows.captures[first.replay_id]["manifest.json"]

    recovered = rc.write_replay_capture(root, name="Casey", items=(b"same",))

    assert recovered.ok, recovered.error
    assert rc.load_replay_capture(root, first.replay_id).items == (b"same",)
    assert any(name.startswith(".incomplete-") for name in windows.captures)


def test_windows_handle_adapter_fails_closed_on_tampered_completed_winner(tmp_path, monkeypatch):
    windows = _FakeWindowsCorpusFilesystem()
    monkeypatch.setattr(rc, "_WINDOWS_FILESYSTEM", windows)
    root = tmp_path / "corpus"
    written = _write_capture(root, name="Casey", items=(b"same",))
    windows.captures[written.replay_id]["item_001.bin"] = b"tampered"

    result = rc.write_replay_capture(root, name="Casey", items=(b"same",))

    assert not result.ok
    assert "canonical" in result.error


def test_windows_handle_adapter_purge_keeps_manifestless_same_id_replacement(tmp_path, monkeypatch):
    windows = _FakeWindowsCorpusFilesystem()
    monkeypatch.setattr(rc, "_WINDOWS_FILESYSTEM", windows)
    root = tmp_path / "corpus"
    written = _write_capture(root, name="Casey", items=(b"same",))
    del windows.captures[written.replay_id]["manifest.json"]

    assert rc.delete_replay_captures(root, [written.replay_id]) == ()
    assert written.replay_id in windows.captures


def test_windows_handle_adapter_rejects_reparse_hardlink_and_replacement_races(tmp_path, monkeypatch):
    windows = _FakeWindowsCorpusFilesystem()
    monkeypatch.setattr(rc, "_WINDOWS_FILESYSTEM", windows)
    root = tmp_path / "corpus"
    written = _write_capture(root, name="safe", items=(b"safe-item",), captured_at=1.0)

    # A capture junction is neither listed nor loadable; a file reparse point/hardlink is never
    # read even though the manifest still names it.
    windows.reparse_captures.add(written.replay_id)
    assert rc.list_replay_ids(root) == []
    with pytest.raises(FileNotFoundError):
        rc.load_replay_capture(root, written.replay_id)
    windows.reparse_captures.clear()
    windows.hardlinked_files.add((written.replay_id, "item_001.bin"))
    with pytest.raises(ValueError, match="missing or unsafe"):
        rc.load_replay_capture(root, written.replay_id)
    windows.hardlinked_files.clear()

    # Model a race after prune discovery.  ``remove_capture`` sees the original opaque root
    # handle (not a re-opened root pathname) and refuses the newly reparse-pointed target.
    windows.swap_before_remove.add(written.replay_id)
    result = rc.prune_replay_corpus(root, max_captures=0, max_age_days=1)
    assert result.removed == ()
    assert windows.remove_roots == [("root", windows.root_token)]


def test_main_live_rejects_a_traversal_manifest_before_provider_transport(tmp_path, capsys):
    corpus_dir = tmp_path / "corpus"
    written = _write_capture(corpus_dir, items=(b"safe-item",))
    (tmp_path / "private.bin").write_bytes(b"must never reach Gemini")
    manifest_path = written.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["items"] = ["../../private.bin"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    transport = _ScriptedTransport([_success()])

    result = m.main(
        ["--live", "--corpus-dir", str(corpus_dir), "--config", str(_config_path(tmp_path)),
         "--replay-ids", written.replay_id],
        transport=transport, env={"GEMINI_API_KEY": "k"}, confirm=lambda _prompt: True)

    assert result == 1
    assert transport.calls == []
    assert "flat filename" in capsys.readouterr().err


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
    assert "sent_context_count=0" in out


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
    assert entry["sent_context_count"] == 0
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


def test_main_live_omits_retained_context_crops_from_the_gemini_request(tmp_path):
    """Context stays available in replay storage, but is never an unselectable visual premise."""
    corpus_dir = tmp_path / "corpus"
    _write_capture(corpus_dir, items=(b"selected-item",), context=(b"forensic-context",))
    config_path = _config_path(tmp_path)
    transport = _ScriptedTransport([_success()])

    store = SQLiteStore(tmp_path / "db.sqlite")
    try:
        rc_code = m.main(
            ["--corpus-dir", str(corpus_dir), "--config", str(config_path), "--live", "--yes"],
            transport=transport, env={"GEMINI_API_KEY": "k"}, store=store)
    finally:
        store.close()

    assert rc_code == 0
    parts = transport.calls[0]["payload"]["contents"][0]["parts"]
    wire = json.dumps(parts)
    assert base64.standard_b64encode(b"selected-item").decode("ascii") in wire
    assert base64.standard_b64encode(b"forensic-context").decode("ascii") not in wire
    assert not any("CONTEXT" in part.get("text", "") for part in parts)
    assert len([part for part in parts if "inlineData" in part]) == 1


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
    # Purge removes only verified capture children; it deliberately preserves the root rather
    # than recursively deleting an arbitrary path supplied on the command line.
    assert corpus_dir.is_dir()
    assert all(path.name.startswith(".lock-") for path in corpus_dir.iterdir())


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
