"""Focused tests for privacy-sensitive artifact filesystem primitives."""
from __future__ import annotations

import os
import stat

import pytest

from operation_love.private_files import (
    UnsafePrivatePathError,
    append_private_text,
    atomic_write_private_bytes,
    ensure_private_dir,
    load_private_dotenv,
    open_private_rw,
    tighten_private_file,
    write_private_bytes,
)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_private_directory_and_created_files_use_owner_only_modes(tmp_path):
    broad_parent = tmp_path / "shared"
    broad_parent.mkdir(mode=0o755)
    broad_parent.chmod(0o755)
    leaf = ensure_private_dir(broad_parent / "artifacts")

    binary = leaf / "capture.png"
    log = leaf / "manifest.jsonl"
    write_private_bytes(binary, b"image", parent=leaf)
    append_private_text(log, "one\n", parent=leaf)
    append_private_text(log, "two\n", parent=leaf)

    assert stat.S_IMODE(broad_parent.stat().st_mode) == 0o755
    assert stat.S_IMODE(leaf.stat().st_mode) == 0o700
    assert stat.S_IMODE(binary.stat().st_mode) == 0o600
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    assert log.read_text() == "one\ntwo\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_existing_directory_and_regular_file_are_tightened(tmp_path):
    leaf = tmp_path / "artifacts"
    leaf.mkdir(mode=0o755)
    artifact = leaf / "manifest.json"
    artifact.write_text("{}")
    leaf.chmod(0o755)
    artifact.chmod(0o644)

    ensure_private_dir(leaf)
    assert tighten_private_file(artifact, parent=leaf) is True

    assert stat.S_IMODE(leaf.stat().st_mode) == 0o700
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_private_rw_open_tightens_existing_file_and_supports_descriptor_lifetime(tmp_path):
    leaf = ensure_private_dir(tmp_path / "locks")
    lock_path = leaf / "android.lock"
    lock_path.write_text("old")
    lock_path.chmod(0o644)

    fd = open_private_rw(lock_path, parent=leaf)
    try:
        os.lseek(fd, 0, os.SEEK_END)
        os.write(fd, b"-new")
    finally:
        os.close(fd)

    assert lock_path.read_text() == "old-new"
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX hard-link semantics")
def test_private_rw_open_refuses_hardlink_without_mutating_shared_target(tmp_path):
    leaf = ensure_private_dir(tmp_path / "locks")
    outside = tmp_path / "unrelated.txt"
    outside.write_text("untouched")
    outside.chmod(0o644)
    managed = leaf / "android.lock"
    os.link(outside, managed)

    with pytest.raises(UnsafePrivatePathError):
        open_private_rw(managed, parent=leaf)

    assert outside.read_text() == "untouched"
    assert stat.S_IMODE(outside.stat().st_mode) == 0o644


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission and link semantics")
def test_private_dotenv_tightens_regular_file_and_refuses_link_leaves(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPLOVE_DOTENV_TEST=loaded\n")
    dotenv.chmod(0o644)
    monkeypatch.delenv("OPLOVE_DOTENV_TEST", raising=False)

    assert load_private_dotenv(dotenv) is True
    assert os.environ["OPLOVE_DOTENV_TEST"] == "loaded"
    assert stat.S_IMODE(dotenv.stat().st_mode) == 0o600

    outside = tmp_path / "outside.env"
    outside.write_text("OPLOVE_DOTENV_TEST=outside\n")
    outside.chmod(0o644)
    linked = tmp_path / "linked.env"
    linked.symlink_to(outside)
    with pytest.raises(UnsafePrivatePathError):
        load_private_dotenv(linked)
    assert outside.read_text() == "OPLOVE_DOTENV_TEST=outside\n"
    assert stat.S_IMODE(outside.stat().st_mode) == 0o644

    hardlinked = tmp_path / "hardlinked.env"
    os.link(outside, hardlinked)
    with pytest.raises(UnsafePrivatePathError):
        load_private_dotenv(hardlinked)
    assert outside.read_text() == "OPLOVE_DOTENV_TEST=outside\n"
    assert stat.S_IMODE(outside.stat().st_mode) == 0o644


def test_exclusive_private_write_refuses_collision_without_changing_content(tmp_path):
    leaf = ensure_private_dir(tmp_path / "artifacts")
    artifact = leaf / "capture.png"
    write_private_bytes(artifact, b"first", parent=leaf)

    with pytest.raises(FileExistsError):
        write_private_bytes(artifact, b"second", parent=leaf)

    assert artifact.read_bytes() == b"first"


@pytest.mark.skipif(os.name != "posix", reason="symlink semantics")
def test_directory_and_file_helpers_reject_symlinks_without_touching_targets(tmp_path):
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir(mode=0o755)
    outside_dir.chmod(0o755)
    directory_link = tmp_path / "artifacts"
    directory_link.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(UnsafePrivatePathError):
        ensure_private_dir(directory_link)
    assert stat.S_IMODE(outside_dir.stat().st_mode) == 0o755

    leaf = ensure_private_dir(tmp_path / "safe")
    outside_file = tmp_path / "outside.json"
    outside_file.write_text("untouched")
    outside_file.chmod(0o644)
    file_link = leaf / "manifest.json"
    file_link.symlink_to(outside_file)

    with pytest.raises(UnsafePrivatePathError):
        tighten_private_file(file_link, parent=leaf)
    with pytest.raises(UnsafePrivatePathError):
        append_private_text(file_link, "bad", parent=leaf)
    with pytest.raises(UnsafePrivatePathError):
        open_private_rw(file_link, parent=leaf)

    assert outside_file.read_text() == "untouched"
    assert stat.S_IMODE(outside_file.stat().st_mode) == 0o644


@pytest.mark.skipif(os.name != "posix", reason="symlink semantics")
def test_atomic_private_replace_preserves_mode_and_refuses_symlink_destination(tmp_path):
    leaf = ensure_private_dir(tmp_path / "artifacts")
    artifact = leaf / "manifest.json"
    write_private_bytes(artifact, b"old", parent=leaf)
    artifact.chmod(0o644)

    atomic_write_private_bytes(artifact, b"new", parent=leaf)

    assert artifact.read_bytes() == b"new"
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
    assert {path.name for path in leaf.iterdir()} == {"manifest.json"}

    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside")
    artifact.unlink()
    artifact.symlink_to(outside)
    with pytest.raises(UnsafePrivatePathError):
        atomic_write_private_bytes(artifact, b"bad", parent=leaf)
    assert outside.read_bytes() == b"outside"
