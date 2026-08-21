"""Small, cross-platform primitives for privacy-sensitive local artifacts.

Only the requested leaf directory is chmod'd; parent directories are created when needed but
their existing permissions are never changed. File operations reject a symlink at the managed
leaf and use ``O_NOFOLLOW`` where the host provides it. Callers decide whether an error is fatal
(configuration and release tools) or best-effort (debug logging).
"""
from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


class UnsafePrivatePathError(OSError):
    """A managed private path is a symlink or has an unexpected filesystem type."""


def _lstat(path: Path):
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _chmod_nofollow(path: Path, mode: int) -> None:
    """chmod one already-validated leaf without intentionally following a symlink."""
    try:
        os.chmod(path, mode, follow_symlinks=False)
    except (NotImplementedError, TypeError):
        # Windows filesystems do not expose POSIX modes in the same way and some Python/OS
        # combinations cannot chmod a link itself. The immediately preceding lstat in every
        # public caller has already rejected a symlink; this fallback retains useful behavior
        # on those platforms.
        os.chmod(path, mode)


def _require_directory(path: Path) -> None:
    info = _lstat(path)
    if info is None or not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise UnsafePrivatePathError(f"private directory is not a real directory: {path}")


def _require_regular(path: Path, *, missing_ok: bool = False) -> bool:
    info = _lstat(path)
    if info is None:
        if missing_ok:
            return False
        raise FileNotFoundError(path)
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
            or getattr(info, "st_nlink", 1) != 1):
        raise UnsafePrivatePathError(f"private file is not a regular file: {path}")
    return True


def _require_direct_parent(path: Path, parent: Path | None) -> Path:
    expected = Path(parent) if parent is not None else path.parent
    if path.parent.absolute() != expected.absolute():
        raise UnsafePrivatePathError(f"private file is outside its managed directory: {path}")
    _require_directory(expected)
    return expected


def ensure_private_dir(path: str | Path, *, exist_ok: bool = True) -> Path:
    """Create/tighten one leaf directory to 0700 without chmod'ing any ancestor.

    ``exist_ok=False`` preserves fresh-output-directory semantics without briefly exposing a
    newly created directory at the process umask's default mode.
    """
    directory = Path(path)
    info = _lstat(directory)
    if info is not None and (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)):
        raise UnsafePrivatePathError(f"private directory is not a real directory: {directory}")
    if info is not None and not exist_ok:
        raise FileExistsError(directory)
    directory.mkdir(mode=PRIVATE_DIR_MODE, parents=True, exist_ok=exist_ok)
    _require_directory(directory)
    _chmod_nofollow(directory, PRIVATE_DIR_MODE)
    return directory


def tighten_private_file(path: str | Path, *, parent: str | Path | None = None,
                         missing_ok: bool = False) -> bool:
    """Tighten an existing regular leaf to 0600; never chmod through a symlink."""
    target = Path(path)
    _require_direct_parent(target, Path(parent) if parent is not None else None)
    if not _require_regular(target, missing_ok=missing_ok):
        return False
    _chmod_nofollow(target, PRIVATE_FILE_MODE)
    return True


def _private_open_flags(flags: int) -> int:
    return (flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_BINARY", 0))


def _verify_private_fd(fd: int, path: Path) -> None:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_nlink", 1) != 1:
        raise UnsafePrivatePathError(f"private file is not a regular file: {path}")
    try:
        os.fchmod(fd, PRIVATE_FILE_MODE)
    except (AttributeError, NotImplementedError):
        # Windows has no meaningful POSIX group/other mode bits; leaf validation and its ACLs
        # remain authoritative there.
        pass


def write_private_bytes(path: str | Path, data: bytes, *, parent: str | Path | None = None) -> None:
    """Create one new 0600 regular file, refusing collisions and symlink leaves."""
    target = Path(path)
    _require_direct_parent(target, Path(parent) if parent is not None else None)
    fd = os.open(target, _private_open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
                 PRIVATE_FILE_MODE)
    try:
        _verify_private_fd(fd, target)
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(data)
    finally:
        if fd >= 0:
            os.close(fd)


def open_private_rw(path: str | Path, *, parent: str | Path | None = None) -> int:
    """Open/create one regular leaf read-write at 0600, returning an owned file descriptor.

    This is intended for lock files whose descriptor must stay open for the lock's lifetime.
    The caller must close the returned descriptor.
    """
    target = Path(path)
    _require_direct_parent(target, Path(parent) if parent is not None else None)
    existing = _lstat(target)
    if existing is not None and (stat.S_ISLNK(existing.st_mode)
                                 or not stat.S_ISREG(existing.st_mode)
                                 or getattr(existing, "st_nlink", 1) != 1):
        raise UnsafePrivatePathError(f"private file is not a regular file: {target}")
    fd = os.open(target, _private_open_flags(os.O_RDWR | os.O_CREAT), PRIVATE_FILE_MODE)
    try:
        _verify_private_fd(fd, target)
    except BaseException:
        os.close(fd)
        raise
    return fd


def load_private_dotenv(path: str | Path) -> bool:
    """Load one project-local ``.env`` only after securing its leaf file.

    A missing file is intentionally a no-op: credentials may come from the real
    environment.  An existing file is opened by descriptor with ``O_NOFOLLOW``
    where available, verified as a singly linked regular file, and tightened to
    owner-only mode *before* python-dotenv parses it.  This keeps an accidental
    group/world-readable file repairable while refusing a symlink or hard link
    that could make a credential load escape the intended project directory.
    """
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "python-dotenv is not installed, so the project's local .env cannot be loaded. "
            "python-dotenv is a required dependency; install it with `pip install "
            "python-dotenv` or reinstall the project (`pip install -e .`)."
        ) from exc

    target = Path(path)
    _require_direct_parent(target, None)
    if not _require_regular(target, missing_ok=True):
        return False
    fd = os.open(target, _private_open_flags(os.O_RDONLY))
    try:
        _verify_private_fd(fd, target)
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            fd = -1
            return bool(load_dotenv(stream=stream))
    finally:
        if fd >= 0:
            os.close(fd)


def append_private_text(path: str | Path, text: str, *, parent: str | Path | None = None) -> None:
    """Append text to a regular 0600 file without following a symlink leaf."""
    target = Path(path)
    _require_direct_parent(target, Path(parent) if parent is not None else None)
    existing = _lstat(target)
    if existing is not None and (stat.S_ISLNK(existing.st_mode)
                                 or not stat.S_ISREG(existing.st_mode)
                                 or getattr(existing, "st_nlink", 1) != 1):
        raise UnsafePrivatePathError(f"private file is not a regular file: {target}")
    fd = os.open(target, _private_open_flags(os.O_WRONLY | os.O_APPEND | os.O_CREAT),
                 PRIVATE_FILE_MODE)
    try:
        _verify_private_fd(fd, target)
        with os.fdopen(fd, "a", encoding="utf-8") as stream:
            fd = -1
            stream.write(text)
    finally:
        if fd >= 0:
            os.close(fd)


def atomic_write_private_bytes(path: str | Path, data: bytes, *,
                               parent: str | Path | None = None) -> None:
    """Atomically replace a regular artifact with a same-directory 0600 temporary file."""
    target = Path(path)
    directory = _require_direct_parent(target, Path(parent) if parent is not None else None)
    _require_regular(target, missing_ok=True)  # explicitly reject an existing symlink/special file
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=directory)
    temporary = Path(temporary_name)
    try:
        _verify_private_fd(fd, temporary)
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        tighten_private_file(target, parent=directory)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_private_text(path: str | Path, text: str, *,
                              parent: str | Path | None = None) -> None:
    atomic_write_private_bytes(path, text.encode("utf-8"), parent=parent)
