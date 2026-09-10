"""operation_love/opener/replay_corpus.py's prune_replay_corpus: bounded retention for the
replay corpus (ops/OPENER-REDESIGN.md 5.2/5.7), added alongside enabling the corpus for real.

This corpus stores REAL PEOPLE'S PHOTOS locally -- see replay_corpus.py's own module docstring
PRIVACY section. Every test here operates against a fresh tmp_path, never data/.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

import pytest

from operation_love.opener import replay_corpus as rc


# ---------------------------------------------------------------------------------------
# local helpers (no conftest.py in this repo, matching tests/test_opener_replay.py)
# ---------------------------------------------------------------------------------------

def _capture(root, name, captured_at, *, extra_bytes=b""):
    """Write one real capture (via the real writer, so its manifest/id shape is exactly what
    write_replay_capture produces) and return its replay_id."""
    result = rc.write_replay_capture(
        root, items=[f"item-{name}".encode() + extra_bytes], name=name,
        captured_at=captured_at)
    assert result.ok, result.error
    return result.replay_id


def _manual_capture_dir(root, replay_id, *, manifest_text=None, manifest_obj=None,
                        with_manifest=True):
    """Hand-build a capture-shaped directory under root, bypassing write_replay_capture, for
    the corrupt/missing-manifest edge cases the writer itself would never produce."""
    directory = os.path.join(str(root), replay_id)
    os.makedirs(directory, exist_ok=True)
    if with_manifest:
        path = os.path.join(directory, "manifest.json")
        if manifest_text is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(manifest_text)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(manifest_obj, f)
    return directory


_HEX64_A = "a" * 64
_HEX64_B = "b" * 64
_HEX64_C = "c" * 64


# ---------------------------------------------------------------------------------------
# basic contract: unlimited, empty/missing root
# ---------------------------------------------------------------------------------------

def test_both_bounds_zero_is_a_documented_no_op(tmp_path):
    now = time.time()
    for i in range(5):
        _capture(tmp_path, f"n{i}", now - i * 86400)

    result = rc.prune_replay_corpus(tmp_path, max_captures=0, max_age_days=0)

    assert result.ok is True
    assert result.removed == ()
    assert result.kept == 5
    assert len(rc.list_replay_ids(tmp_path)) == 5


def test_nonexistent_root_is_a_no_op_not_an_error(tmp_path):
    missing = tmp_path / "does_not_exist_yet"
    result = rc.prune_replay_corpus(missing, max_captures=1, max_age_days=1)
    assert result.ok is True
    assert result.removed == ()
    assert result.kept == 0


# ---------------------------------------------------------------------------------------
# pruning by count, oldest-first
# ---------------------------------------------------------------------------------------

def test_prune_by_count_removes_oldest_first(tmp_path):
    now = time.time()
    ids_oldest_to_newest = [_capture(tmp_path, f"n{i}", now - (10 - i) * 86400)
                            for i in range(5)]

    result = rc.prune_replay_corpus(tmp_path, max_captures=3)

    assert result.ok is True
    assert result.kept == 3
    removed_ids = {r.replay_id for r in result.removed}
    assert removed_ids == set(ids_oldest_to_newest[:2])   # the two OLDEST, not any others
    assert all(r.reason == rc.PRUNE_REASON_MAX_CAPTURES for r in result.removed)
    survivors = set(rc.list_replay_ids(tmp_path))
    assert survivors == set(ids_oldest_to_newest[2:])


def test_prune_by_count_is_a_no_op_when_under_the_cap(tmp_path):
    now = time.time()
    for i in range(3):
        _capture(tmp_path, f"n{i}", now - i * 86400)

    result = rc.prune_replay_corpus(tmp_path, max_captures=10)

    assert result.removed == ()
    assert result.kept == 3
    assert len(rc.list_replay_ids(tmp_path)) == 3


# ---------------------------------------------------------------------------------------
# pruning by age
# ---------------------------------------------------------------------------------------

def test_prune_by_age_removes_only_captures_older_than_the_cutoff(tmp_path):
    now = time.time()
    old_id = _capture(tmp_path, "old", now - 200 * 86400)
    borderline_id = _capture(tmp_path, "borderline", now - 89 * 86400)
    new_id = _capture(tmp_path, "new", now - 1 * 86400)

    result = rc.prune_replay_corpus(tmp_path, max_age_days=90)

    assert result.ok is True
    removed_ids = {r.replay_id for r in result.removed}
    assert removed_ids == {old_id}
    assert result.removed[0].reason == rc.PRUNE_REASON_MAX_AGE_DAYS
    survivors = set(rc.list_replay_ids(tmp_path))
    assert survivors == {borderline_id, new_id}


def test_prune_by_age_is_a_no_op_when_nothing_is_old_enough(tmp_path):
    now = time.time()
    for i in range(3):
        _capture(tmp_path, f"n{i}", now - i * 86400)

    result = rc.prune_replay_corpus(tmp_path, max_age_days=365)

    assert result.removed == ()
    assert result.kept == 3


# ---------------------------------------------------------------------------------------
# both bounds together
# ---------------------------------------------------------------------------------------

def test_prune_both_bounds_together_age_first_then_count(tmp_path):
    """Age removal runs first; count removal then trims whatever survives it. With 2 captures
    older than the age cutoff and 3 within it, and max_captures=2, the two old ones must be
    removed for AGE (not count), and exactly one more (the older of the two survivors) removed
    for COUNT, leaving the newest 2."""
    now = time.time()
    ids = [
        _capture(tmp_path, "ancient1", now - 400 * 86400),
        _capture(tmp_path, "ancient2", now - 300 * 86400),
        _capture(tmp_path, "recent1", now - 3 * 86400),
        _capture(tmp_path, "recent2", now - 2 * 86400),
        _capture(tmp_path, "recent3", now - 1 * 86400),
    ]

    result = rc.prune_replay_corpus(tmp_path, max_captures=2, max_age_days=90)

    reasons = {r.replay_id: r.reason for r in result.removed}
    assert reasons[ids[0]] == rc.PRUNE_REASON_MAX_AGE_DAYS
    assert reasons[ids[1]] == rc.PRUNE_REASON_MAX_AGE_DAYS
    assert reasons[ids[2]] == rc.PRUNE_REASON_MAX_CAPTURES   # oldest of the 3 survivors
    assert set(reasons) == {ids[0], ids[1], ids[2]}
    assert result.kept == 2
    assert set(rc.list_replay_ids(tmp_path)) == {ids[3], ids[4]}


# ---------------------------------------------------------------------------------------
# a manifest that fails to parse: DELIBERATE DECISION -- kept, never deleted, never counted
# toward either bound (see prune_replay_corpus's own docstring for the full justification).
# ---------------------------------------------------------------------------------------

def test_corrupt_manifest_is_kept_not_deleted_and_not_counted(tmp_path):
    # Comfortably inside the max_age_days=1 cutoff used below, with margin for test runtime.
    good_id = _capture(tmp_path, "good", time.time() - 3600)
    corrupt_dir = _manual_capture_dir(tmp_path, _HEX64_A, manifest_text="{not valid json at all")

    # Force pruning as aggressively as possible on every axis a corrupt entry might otherwise
    # be swept up by: a tiny count cap AND a tiny age cutoff.
    result = rc.prune_replay_corpus(tmp_path, max_captures=0, max_age_days=1)

    assert result.ok is True
    assert result.skipped_unparseable == 1
    assert result.kept == 1                      # only "good" counted; corrupt is invisible
    assert all(r.replay_id != _HEX64_A for r in result.removed)
    assert os.path.isdir(corrupt_dir)             # never deleted
    assert good_id in rc.list_replay_ids(tmp_path)


def test_manifest_that_is_valid_json_but_not_an_object_is_treated_the_same_as_corrupt(tmp_path):
    _manual_capture_dir(tmp_path, _HEX64_B, manifest_obj=["not", "an", "object"])

    result = rc.prune_replay_corpus(tmp_path, max_captures=0, max_age_days=1)

    assert result.ok is True
    assert result.skipped_unparseable == 1
    assert result.kept == 0
    assert result.removed == ()


def _manual_ageless_capture(root, replay_id):
    """A manifest that PARSES (unlike the corrupt-manifest tests above) but whose captured_at
    is not a usable number -- e.g. hand-edited or written by a future format variant."""
    directory = _manual_capture_dir(
        root, replay_id,
        manifest_obj={"format_version": 1, "replay_id": replay_id, "captured_at": "not-a-number",
                      "name": "", "truncated": False, "items": ["item_001.png"], "context": []})
    with open(os.path.join(directory, "item_001.png"), "wb") as f:
        f.write(b"x")
    return directory


def test_manifest_with_unusable_captured_at_is_never_removed_for_age(tmp_path):
    """A parseable manifest with no usable captured_at IS a candidate (unlike a corrupt one),
    but its unknown age must never be compared against an age cutoff -- it sorts as +inf,
    matching list_replay_ids' own tie-break convention, so it can never be judged "too old"."""
    _manual_ageless_capture(tmp_path, _HEX64_C)
    old_id = _capture(tmp_path, "old", time.time() - 400 * 86400)

    result = rc.prune_replay_corpus(tmp_path, max_age_days=1)

    assert {r.replay_id for r in result.removed} == {old_id}
    assert _HEX64_C in rc.list_replay_ids(tmp_path)


def test_manifest_with_unusable_captured_at_sorts_last_for_count_based_removal(tmp_path):
    """Oldest-first count trimming must never preferentially punish an unknown-age capture --
    it sorts as "newest" (+inf), so it is the LAST candidate ever removed. With 3 candidates
    (ageless, a real older one, a real newer one) and max_captures=2, the real OLDER one must
    be the one evicted, never the ageless one."""
    _manual_ageless_capture(tmp_path, _HEX64_C)
    older_id = _capture(tmp_path, "older", time.time() - 10 * 86400)
    newer_id = _capture(tmp_path, "newer", time.time() - 1 * 86400)

    result = rc.prune_replay_corpus(tmp_path, max_captures=2)

    assert {r.replay_id for r in result.removed} == {older_id}
    survivors = set(rc.list_replay_ids(tmp_path))
    assert survivors == {_HEX64_C, newer_id}


# ---------------------------------------------------------------------------------------
# a capture directory with NO manifest at all (write_replay_capture's own documented residual
# of a crash mid-write) -- matches list_replay_ids: not a (complete) capture, never touched.
# ---------------------------------------------------------------------------------------

def test_partial_write_with_no_manifest_is_untouched(tmp_path):
    # Comfortably inside the max_age_days=1 cutoff used below, with margin for test runtime.
    good_id = _capture(tmp_path, "good", time.time() - 3600)
    partial_dir = _manual_capture_dir(tmp_path, _HEX64_A, with_manifest=False)
    with open(os.path.join(partial_dir, "item_001.png"), "wb") as f:
        f.write(b"orphaned crop, no manifest.json")

    result = rc.prune_replay_corpus(tmp_path, max_captures=0, max_age_days=1)

    assert result.ok is True
    assert result.skipped_unparseable == 0    # not even attempted -- no manifest.json at all
    assert result.kept == 1
    assert os.path.isdir(partial_dir)
    assert good_id in rc.list_replay_ids(tmp_path)


# ---------------------------------------------------------------------------------------
# DELETE-PATH SAFETY -- the highest-risk part of this function. Each test below attempts one
# specific escape and proves it is refused, not merely "not currently exploited".
# ---------------------------------------------------------------------------------------

def test_directory_name_not_shaped_like_a_replay_id_is_never_touched(tmp_path):
    """A directory that is not named like anything compute_replay_id could have produced is
    never ours to manage, no matter how old or how validly-formed its own manifest is."""
    weird_dir = _manual_capture_dir(
        tmp_path, "not-a-valid-replay-id",
        manifest_obj={"captured_at": 1.0, "replay_id": "not-a-valid-replay-id",
                      "items": ["item_001.png"], "context": []})
    with open(os.path.join(weird_dir, "item_001.png"), "wb") as f:
        f.write(b"x")

    result = rc.prune_replay_corpus(tmp_path, max_captures=0, max_age_days=1)

    assert result.ok is True
    assert result.skipped_unparseable == 0     # never even attempted -- wrong shape entirely
    assert result.kept == 0                    # never counted as a capture at all
    assert result.removed == ()
    assert os.path.isdir(weird_dir)


@pytest.mark.parametrize("bad_name", [
    "AAAA0000000000000000000000000000000000000000000000000000000000",  # uppercase hex
    "a" * 63,                                    # one short
    "a" * 65,                                    # one long
    "a" * 63 + "g",                               # non-hex trailing char
    "../evil",
    "a/b",
    "..",
    ".",
    "",
])
def test_is_safe_capture_id_rejects_every_non_replay_id_shape(bad_name):
    assert rc._is_safe_capture_id(bad_name) is False


def test_is_safe_capture_id_accepts_a_real_compute_replay_id_output():
    real_id = rc.compute_replay_id(name="x", items=[b"y"])
    assert rc._is_safe_capture_id(real_id) is True


def test_symlink_escape_is_refused_and_the_outside_target_survives(tmp_path):
    """A symlink living INSIDE root, named like a valid replay_id, pointing to a directory
    OUTSIDE root: must never be followed, never counted as a kept/removed capture, and the
    outside directory it points to must be completely untouched -- even under a max_age_days
    aggressive enough that a genuine capture with this timestamp would be removed."""
    outside = tmp_path.parent / (tmp_path.name + "_outside")
    outside.mkdir()
    outside_target = outside / "secret_photos"
    outside_target.mkdir()
    with open(outside_target / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"captured_at": time.time() - 400 * 86400, "replay_id": _HEX64_A,
                  "items": ["item_001.png"], "context": []}, f)
    with open(outside_target / "item_001.png", "wb") as f:
        f.write(b"real person's photo")

    root = tmp_path / "corpus"
    root.mkdir()
    # Comfortably inside the max_age_days=1 cutoff used below, with margin for test runtime.
    real_id = _capture(root, "real", time.time() - 3600)
    link_path = root / _HEX64_A
    os.symlink(str(outside_target), str(link_path))

    result = rc.prune_replay_corpus(root, max_captures=0, max_age_days=1)

    assert result.ok is True
    # The symlink must never even become a counted candidate -- proves the discovery-time
    # is_symlink() guard fired, not merely that deletion happened to fail downstream.
    assert result.kept == 1
    assert all(r.replay_id != _HEX64_A for r in result.removed)
    assert link_path.is_symlink()                       # the symlink itself: untouched
    assert outside_target.is_dir()                       # its target: fully intact
    assert (outside_target / "item_001.png").read_bytes() == b"real person's photo"
    assert real_id in rc.list_replay_ids(root)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_prune_keeps_a_capture_with_an_unsafe_manifest_link(tmp_path, link_kind):
    root = tmp_path / "corpus"
    root.mkdir()
    capture_id = _capture(root, "unsafe-manifest", time.time() - 400 * 86400)
    capture_dir = root / capture_id
    manifest = capture_dir / "manifest.json"
    if link_kind == "symlink":
        outside = tmp_path / "outside-manifest.json"
        outside.write_bytes(manifest.read_bytes())
        manifest.unlink()
        os.symlink(outside, manifest)
    else:
        os.link(manifest, tmp_path / "manifest-hardlink-proof.json")

    result = rc.prune_replay_corpus(root, max_age_days=1)

    assert result.ok is True
    assert result.removed == ()
    assert result.skipped_unparseable == 1
    assert capture_dir.is_dir()


def test_prune_never_deletes_anything_outside_root_even_with_many_old_captures(tmp_path):
    """End-to-end sanity: a sibling directory that merely happens to sit next to the corpus
    root is never touched by an aggressive prune, regardless of what is inside root."""
    root = tmp_path / "corpus"
    root.mkdir()
    sibling = tmp_path / "sibling_untouched"
    sibling.mkdir()
    (sibling / "keepme.txt").write_text("do not delete", encoding="utf-8")
    now = time.time()
    for i in range(10):
        _capture(root, f"n{i}", now - (i + 1) * 400 * 86400)   # all very old

    result = rc.prune_replay_corpus(root, max_captures=1, max_age_days=1)

    assert result.ok is True
    assert sibling.is_dir()
    assert (sibling / "keepme.txt").read_text(encoding="utf-8") == "do not delete"


def test_prune_preserves_a_replacement_and_newer_destination_after_validation(tmp_path, monkeypatch):
    root = tmp_path / "corpus"
    root.mkdir()
    capture_id = _capture(root, "old", time.time() - 400 * 86400)
    capture_dir = root / capture_id
    real_rename = rc.os.rename
    replaced = False

    def replace_before_quarantine(source, target, *args, **kwargs):
        nonlocal replaced
        if source == capture_id and not replaced:
            replaced = True
            shutil.rmtree(capture_dir)
            capture_dir.mkdir()
            (capture_dir / "replacement.txt").write_text("keep", encoding="utf-8")
        return real_rename(source, target, *args, **kwargs)

    monkeypatch.setattr(rc.os, "rename", replace_before_quarantine)
    real_identity = rc._capture_identity
    identity_calls = 0

    def create_new_destination_on_mismatch(directory_fd):
        nonlocal identity_calls
        identity_calls += 1
        if identity_calls == 2:
            capture_dir.mkdir()
            (capture_dir / "newer.txt").write_text("new", encoding="utf-8")
        return real_identity(directory_fd)

    monkeypatch.setattr(rc, "_capture_identity", create_new_destination_on_mismatch)
    result = rc.prune_replay_corpus(root, max_age_days=1)

    assert result.removed == ()
    assert (capture_dir / "newer.txt").read_text(encoding="utf-8") == "new"
    quarantines = [path for path in root.iterdir() if path.name.startswith(".prune-")]
    assert len(quarantines) == 1
    assert (quarantines[0] / "replacement.txt").read_text(encoding="utf-8") == "keep"


def test_fd_relative_delete_stays_with_open_root_after_root_path_swap(tmp_path):
    """The final rename/delete is bound to the root FD, not its replaceable pathname."""
    root = tmp_path / "corpus"
    capture_id = _capture(root, "old", time.time() - 400 * 86400)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    replacement_capture = replacement / capture_id
    replacement_capture.mkdir()
    (replacement_capture / "keep.txt").write_text("outside replacement", encoding="utf-8")
    root_path, root_fd = rc._open_corpus_root(root)
    hidden = tmp_path / "hidden-original"
    root.rename(hidden)
    replacement.rename(root)
    try:
        rc._remove_capture_directory(root_fd, capture_id)
    finally:
        os.close(root_fd)

    assert not (hidden / capture_id).exists()
    assert (root / capture_id / "keep.txt").read_text(encoding="utf-8") == "outside replacement"


def test_windows_adapter_prune_fails_closed_when_target_changes_after_discovery(tmp_path, monkeypatch):
    """Exercise the injectable Windows branch without requiring a Windows CI runner.

    The fake exposes only opaque handles.  Its delete operation represents the target turning
    into a junction after the handle-relative re-check; prune must leave it alone and must never
    turn that operation into a pathname-based retry.
    """
    capture_id = _HEX64_A

    class WindowsRaceFilesystem:
        def __init__(self):
            self.root_handle = object()
            self.deleted_with = []
            self.manifest = json.dumps({"replay_id": capture_id, "captured_at": 1.0}).encode()

        def open_root(self, root):
            return Path(root), self.root_handle

        def close(self, _handle):
            pass

        def listdir(self, handle):
            assert handle is self.root_handle
            return [capture_id]

        def open_capture(self, root, replay_id):
            assert root is self.root_handle
            assert replay_id == capture_id
            return object()

        def manifest_exists(self, _capture):
            return True

        def read_regular(self, _capture, name):
            assert name == "manifest.json"
            return self.manifest

        def remove_capture(self, root, replay_id):
            self.deleted_with.append((root, replay_id))
            raise OSError("target was replaced by a junction")

        def capture_identity(self, _capture):
            return 1

        def remove_open_capture(self, _capture, expected_identity):
            assert expected_identity == 1
            self.deleted_with.append((self.root_handle, capture_id))
            raise OSError("target was replaced by a junction")

    root = tmp_path / "corpus"
    root.mkdir()
    windows = WindowsRaceFilesystem()
    monkeypatch.setattr(rc, "_WINDOWS_FILESYSTEM", windows)

    result = rc.prune_replay_corpus(root, max_age_days=1)

    assert result.ok is True
    assert result.removed == ()
    assert windows.deleted_with == [(windows.root_handle, capture_id)]


# ---------------------------------------------------------------------------------------
# SAFE TO FAIL: never raises, for any reason.
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad_max_captures", [-1, True, 1.5, "5", None, math.nan])
def test_prune_never_raises_on_a_bad_max_captures(tmp_path, bad_max_captures):
    result = rc.prune_replay_corpus(tmp_path, max_captures=bad_max_captures)
    assert result.ok is False
    assert result.error


@pytest.mark.parametrize("bad_max_age_days", [-1, True, "5", None, math.nan, math.inf, -math.inf])
def test_prune_never_raises_on_a_bad_max_age_days(tmp_path, bad_max_age_days):
    result = rc.prune_replay_corpus(tmp_path, max_age_days=bad_max_age_days)
    assert result.ok is False
    assert result.error


def test_prune_never_raises_on_an_unexpected_internal_error(tmp_path, monkeypatch):
    """A failure that has nothing to do with argument validation (a permissions error, a
    filesystem oddity, ...) must still come back as PruneResult(ok=False, ...) rather than
    propagate -- the same SAFE TO FAIL contract write_replay_capture already guarantees."""
    def _boom(*_a, **_kw):
        raise RuntimeError("simulated unexpected failure")

    monkeypatch.setattr(rc, "Path", _boom)

    result = rc.prune_replay_corpus(str(tmp_path), max_captures=1)

    assert result.ok is False
    assert "simulated unexpected failure" in result.error


def test_prune_zero_max_captures_means_unlimited_not_zero_captures(tmp_path):
    """0 must not be misread as "keep zero captures" -- it is the documented unlimited
    sentinel. A real capture must survive a max_captures=0 prune."""
    now = time.time()
    for i in range(5):
        _capture(tmp_path, f"n{i}", now - i * 86400)

    result = rc.prune_replay_corpus(tmp_path, max_captures=0)

    assert result.removed == ()
    assert result.kept == 5


# ---------------------------------------------------------------------------------------
# PruneResult carries enough detail to log without a second filesystem read.
# ---------------------------------------------------------------------------------------

def test_prune_result_removed_entries_carry_loggable_detail(tmp_path):
    now = time.time()
    old_id = _capture(tmp_path, "old", now - 10 * 86400)
    _capture(tmp_path, "new", now - 1 * 86400)

    result = rc.prune_replay_corpus(tmp_path, max_captures=1)

    assert len(result.removed) == 1
    entry = result.removed[0]
    assert entry.replay_id == old_id
    assert entry.captured_at == pytest.approx(now - 10 * 86400, abs=1.0)
    assert entry.reason == rc.PRUNE_REASON_MAX_CAPTURES
