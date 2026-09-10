"""On-disk REPLAY CORPUS format for opener item-crop requests (ops/OPENER-REDESIGN.md 5.2/5.7).

THE PROBLEM THIS EXISTS TO FIX. The numbered item crops actually sent to Gemini
(opener.opener.ItemRequest -- her name as text, the numbered item crops in order, and the
truncation flag) are not persisted anywhere today. Unnumbered context crops are retained beside
them as forensic/research evidence, but current Gemini generation and replay deliberately omit
them from the request.
``data/hinge_debug/<run_id>/`` holds navigation and verification screenshots
(``capture_before``, ``verify_sheet_item``, ``navigate_to_item`` and so on), not the request
INPUTS. So no historical draft can ever be re-run through a revised prompt: every prompt
revision needs a fresh live batch to measure, which is exactly why no prompt change has ever
been measured before/after (see tools/opener_corpus_report.py's own WHY THIS EXISTS section
for the eight prompt-only fixes that shipped with no outcome measurement at all). This module
is the missing persistence layer: a captured request, written once, can be replayed through as
many future prompt revisions as needed with no phone and no owner time.

THE FORMAT. One directory per captured request, named after its own ``replay_id``, holding the
crops as plain image files plus one ``manifest.json``:

    <root>/<replay_id>/
        manifest.json
        item_001.png            (or .jpg / .webp / .bin -- see _guess_extension)
        item_002.jpg
        ...
        context_001.png
        ...

``manifest.json`` carries exactly the fields needed to reconstruct the current numbered-item
request, plus retained unnumbered forensic context and bookkeeping, and NOTHING else (see PRIVACY
below for why "nothing else" is a deliberate constraint, not laziness):

    {
      "format_version": 1,
      "replay_id": "<sha256 hex digest, content-derived -- see compute_replay_id>",
      "captured_at": 1757000000.0,           # epoch seconds, wall clock at write time
      "name": "Alex",                        # her name as text ("" if not read -- see
                                              # opener.ItemRequest.name / opener._NAME_UNAVAILABLE)
      "truncated": false,                    # opener.ItemRequest.truncated
      "prompt_sha256": "<hex digest or null>",  # the prompt ERA current at CAPTURE time (see
                                              # opener.prompt_stamp) -- historical metadata only;
                                              # a replay tool re-generating through the CURRENT
                                              # prompt computes its own era stamp and does not
                                              # read this field to build a request
      "items": ["item_001.png", "item_002.jpg"],   # ordered filenames; items[k-1] is item k
      "context": ["context_001.png"]                # retained forensic tier, never replayed
    }

Deliberately NOT captured: the style guide text itself (the replay tool always re-generates
through whatever config.yaml currently configures, never the style that was live at capture --
storing it here would invite a replay tool to accidentally use a stale copy instead of the
current one), her profile bio/prompt text (``Profile.text_blob()`` is always empty on Hinge,
the only live driver as of this writing -- see opener.opener.GeminiOpener.generate()'s own
comment: "profile.text_blob(), empty on Hinge but potentially populated by a future browser
driver"), and any run/profile identifier beyond the content-derived ``replay_id`` itself (no
``run_id``, no ``profile_id`` -- see PRIVACY below).

PRIVACY. This corpus is REAL PEOPLE'S PHOTOS -- the exact crops a live Hinge capture sent to a
third-party model. It is LOCAL ONLY BY DEFAULT: this module never uploads, never calls out to
BigQuery or any network endpoint, and never imports anything that could (no
``operation_love.ranker.bigquery_store``, no HTTP client). ``DEFAULT_CORPUS_DIR`` sits under
``data/``, which project-wide ``.gitignore`` already excludes wholesale (see
``data/hinge_debug`` for the existing precedent this format follows). RETENTION is BOUNDED, from
the very first capture, by ``prune_replay_corpus`` (below): it removes captures oldest-first once
their count exceeds a configured ``max_captures`` and/or their age exceeds a configured
``max_age_days`` (0 means unlimited for either -- see that function's own docstring for the full
contract, including its delete-path safety guarantees and its documented handling of a manifest
that fails to parse). The two bounds are owner-tunable via ``opener.replay_corpus_max_captures``
/ ``opener.replay_corpus_max_age_days`` in ``config.yaml``; ``OpenerService`` calls the prune
after every successful capture (see ``OpenerService._capture_replay_corpus``), so the corpus can
never grow unbounded even for a fully-enabled, long-running deployment.

WHAT THIS MODULE DELIBERATELY DOES NOT DO. It has NO dependency on ``operation_love.opener.
opener`` or ``operation_love.opener.service`` -- not even to import ``ItemRequest`` -- and no
dependency on any store or provider client. It is pure, generic I/O over plain values (bytes,
str, bool, float), by design: the live opener path (wired up from
``OpenerService._capture_replay_corpus``, see that method's own docstring) calls the writer with
the four fields ``opener.opener.ItemRequest`` already carries (``items``, ``name``, ``context``,
``truncated``) plus its own already-computed ``prompt_sha256``, without this module ever needing
to know what an ``ItemRequest`` is. Of those fields, only ``items``/``name``/``truncated`` are
model-visible request inputs; ``context`` is the retained forensic tier. ``tools/opener_replay.py``
is the one place that bridges the numbered tier back to ``ItemRequest`` for an actual replay; it
preserves the context tier on disk but intentionally does not pass it to Gemini.

SAFE TO FAIL. ``write_replay_capture`` never raises -- every exception (a full disk, a
permissions error, a malformed argument) is caught and reported through its return value
instead (see ``ReplayWriteResult``), so a caller on the opener/decision/send/refusal path (this
project's own hard rule: telemetry must be best-effort and must never raise into that path) can
call it unconditionally with no try/except of its own. The READ side (``load_replay_capture``,
``load_replay_corpus``) is the opposite on purpose: a tool reading the corpus back wants to know
loudly about a missing file or a corrupt manifest, not have it silently skipped, so those DO
raise.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import time
import uuid
from ctypes import (POINTER, Structure, byref, c_byte, c_long, c_ulong, c_ushort, c_void_p,
                    c_wchar_p, create_string_buffer, pointer)
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

# Where the corpus lives when a caller doesn't override it -- matches this project's existing
# data/hinge_debug convention (tools/opener_corpus_report.py's DEFAULT_DEBUG_DIR), and, like
# it, sits under data/ which project-wide .gitignore already excludes wholesale.
DEFAULT_CORPUS_DIR = "data/opener_replay_corpus"

_MANIFEST_NAME = "manifest.json"
# Bumped only if the on-disk shape changes in a way a reader must know about (a field renamed
# or reinterpreted, not merely a field added -- an added optional field is forward-compatible
# and does not need a bump, since every reader here uses .get() with a default).
_FORMAT_VERSION = 1

# Every ``replay_id`` this module has ever produced is a SHA-256 hex digest (see
# ``compute_replay_id``): exactly 64 lowercase hex characters, nothing else. ``prune_replay_corpus``
# uses this as its FIRST delete-path safety gate -- a directory name is never even considered as
# a removal candidate unless it matches this shape, which by construction can never contain a
# path separator (``/`` or, on a platform where ``os.sep``/``os.altsep`` differ, either of those),
# a ``.``/``..`` traversal segment, or anything else that could steer a filesystem join outside
# ``root``. See ``_is_safe_capture_id`` for the (redundant, deliberately so) second check applied
# again immediately before any actual deletion.
_REPLAY_ID_RE = re.compile(r"[0-9a-f]{64}")


def _is_safe_capture_id(name: object) -> bool:
    """Whether ``name`` is safe to treat as a ``replay_id`` path component under the corpus
    root -- i.e. it is EXACTLY the shape ``compute_replay_id`` produces and nothing more. Used
    both when discovering prune candidates and, redundantly, a second time immediately before
    ``prune_replay_corpus`` actually deletes anything, so a bug in one call site can never be the
    only thing standing between a bad value and a real ``shutil.rmtree``.
    """
    return isinstance(name, str) and bool(_REPLAY_ID_RE.fullmatch(name))


_DIR_OPEN_FLAGS = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                   | getattr(os, "O_NOFOLLOW", 0))
_FILE_OPEN_FLAGS = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
_HAS_DIR_FD = os.open in os.supports_dir_fd


class _WindowsCorpusFilesystem(Protocol):
    """Small seam around Windows handles, deliberately injectable by tests.

    ``dir_fd`` is not implemented by CPython on Windows.  Do not replace this with
    ``Path`` operations there: a check followed by a pathname operation can be redirected by a
    junction/reparse-point swap.  The native implementation below opens every child relative to
    an already-open directory handle (``NtCreateFile``'s ``RootDirectory``) and inspects the
    resulting handle before using it.
    """
    def open_root(self, root: str | Path) -> tuple[Path, object]: ...
    def ensure_root(self, root: str | Path) -> tuple[Path, object]: ...
    def close(self, handle: object) -> None: ...
    def listdir(self, directory: object) -> list[str]: ...
    def open_capture(self, root: object, replay_id: str) -> object: ...
    def mkdir_capture(self, root: object, replay_id: str) -> None: ...
    def manifest_exists(self, directory: object) -> bool: ...
    def read_regular(self, directory: object, name: str) -> bytes: ...
    def write_new(self, directory: object, name: str, data: bytes) -> None: ...
    def replace(self, directory: object, source: str, target: str) -> None: ...
    def publish_capture(self, root: object, staging: str, replay_id: str) -> None: ...
    def remove_capture(self, root: object, replay_id: str) -> None: ...
    def capture_identity(self, directory: object) -> object: ...
    def remove_open_capture(self, directory: object, expected_identity: object) -> bool: ...
    def lock_capture(self, root: object, replay_id: str) -> object: ...


class _WinByHandleFileInformation(Structure):
    _fields_ = [
        ("attributes", c_ulong), ("creation_low", c_ulong), ("creation_high", c_ulong),
        ("access_low", c_ulong), ("access_high", c_ulong), ("write_low", c_ulong),
        ("write_high", c_ulong), ("volume", c_ulong), ("size_high", c_ulong),
        ("size_low", c_ulong), ("links", c_ulong), ("index_high", c_ulong),
        ("index_low", c_ulong),
    ]


class _WinUnicodeString(Structure):
    _fields_ = [("Length", c_ushort), ("MaximumLength", c_ushort), ("Buffer", c_void_p)]


class _WinObjectAttributes(Structure):
    _fields_ = [("Length", c_ulong), ("RootDirectory", c_void_p),
                ("ObjectName", POINTER(_WinUnicodeString)), ("Attributes", c_ulong),
                ("SecurityDescriptor", c_void_p), ("SecurityQualityOfService", c_void_p)]


class _WinIoStatusBlock(Structure):
    # IO_STATUS_BLOCK starts with an NTSTATUS (a signed 32-bit LONG), followed by a
    # pointer-sized ULONG_PTR.  Do not model Status as a pointer: its offset happens to be
    # the same, but the documented ABI is important when reading Information below.
    _fields_ = [("Status", c_long), ("Information", c_void_p)]


class _WinOverlapped(Structure):
    _fields_ = [("Internal", c_void_p), ("InternalHigh", c_void_p),
                ("Offset", c_ulong), ("OffsetHigh", c_ulong), ("hEvent", c_void_p)]


class _NativeWindowsCorpusFilesystem:
    """Windows implementation using handle-relative NT file operations.

    ``CreateFileW`` is used for the initial configured root.  Once that root handle exists,
    every child open/create/rename/delete is an ``NtCreateFile`` operation rooted at that handle.
    ``FILE_OPEN_REPARSE_POINT`` means a junction/symlink is opened as the link itself and then
    rejected from its handle metadata; it is never traversed.  This is intentionally a private
    adapter: its compact API also lets the Linux suite simulate Windows races without pretending
    that POSIX descriptors behave like Windows handles.
    """
    _FILE_ATTRIBUTE_DIRECTORY = 0x10
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x400
    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _DELETE = 0x00010000
    _SYNCHRONIZE = 0x00100000
    _FILE_READ_ATTRIBUTES = 0x80
    _FILE_LIST_DIRECTORY = 0x1
    _FILE_SHARE_ALL = 0x7
    _OPEN_EXISTING = 3
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _INVALID_HANDLE_VALUE = c_void_p(-1).value
    _FILE_OPEN = 1
    _FILE_CREATE = 2
    _FILE_DIRECTORY_FILE = 0x1
    _FILE_NON_DIRECTORY_FILE = 0x40
    _FILE_SYNCHRONOUS_IO_NONALERT = 0x20
    _FILE_OPEN_FOR_BACKUP_INTENT = 0x4000
    _FILE_OPEN_REPARSE_POINT = 0x200000
    _FILE_DIRECTORY_INFORMATION = 1
    _FILE_RENAME_INFORMATION = 10
    _FILE_DISPOSITION_INFORMATION = 13
    _STATUS_NO_MORE_FILES = 0x80000006
    _STATUS_BUFFER_OVERFLOW = 0x80000005
    _STATUS_OBJECT_NAME_NOT_FOUND = 0xC0000034
    _STATUS_OBJECT_NAME_COLLISION = 0xC0000035
    _STATUS_OBJECT_PATH_NOT_FOUND = 0xC000003A
    _DIRECTORY_QUERY_INITIAL_BUFFER = 65536
    _DIRECTORY_QUERY_MAX_BUFFER = 1024 * 1024

    def __init__(self) -> None:
        # Kept lazy so importing this module on Unix never attempts to load WinDLL.
        import ctypes
        self._ctypes = ctypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._ntdll = ctypes.WinDLL("ntdll")
        self._kernel32.CreateFileW.argtypes = [c_wchar_p, c_ulong, c_ulong, c_void_p, c_ulong,
                                               c_ulong, c_void_p]
        self._kernel32.CreateFileW.restype = c_void_p
        self._kernel32.GetFileInformationByHandle.argtypes = [c_void_p, POINTER(_WinByHandleFileInformation)]
        self._kernel32.GetFileInformationByHandle.restype = c_byte
        self._kernel32.CloseHandle.argtypes = [c_void_p]
        self._kernel32.CloseHandle.restype = c_byte
        self._kernel32.ReadFile.argtypes = [c_void_p, c_void_p, c_ulong, POINTER(c_ulong), c_void_p]
        self._kernel32.ReadFile.restype = c_byte
        self._kernel32.WriteFile.argtypes = [c_void_p, c_void_p, c_ulong, POINTER(c_ulong), c_void_p]
        self._kernel32.WriteFile.restype = c_byte
        self._kernel32.LockFileEx.argtypes = [c_void_p, c_ulong, c_ulong, c_ulong, c_ulong,
                                              POINTER(_WinOverlapped)]
        self._kernel32.LockFileEx.restype = c_byte
        self._ntdll.NtCreateFile.argtypes = [POINTER(c_void_p), c_ulong, POINTER(_WinObjectAttributes),
                                             POINTER(_WinIoStatusBlock), c_void_p, c_ulong, c_ulong,
                                             c_ulong, c_ulong, c_void_p, c_ulong]
        self._ntdll.NtCreateFile.restype = c_ulong
        self._ntdll.NtQueryDirectoryFile.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p,
                                                      POINTER(_WinIoStatusBlock), c_void_p, c_ulong,
                                                      c_ulong, c_byte, c_void_p, c_byte]
        self._ntdll.NtQueryDirectoryFile.restype = c_ulong
        self._ntdll.NtSetInformationFile.argtypes = [c_void_p, POINTER(_WinIoStatusBlock), c_void_p,
                                                      c_ulong, c_ulong]
        self._ntdll.NtSetInformationFile.restype = c_ulong

    @staticmethod
    def _failed(status: int) -> bool:
        return bool(status & 0x80000000)

    def _error(self, message: str) -> OSError:
        return OSError(message)

    def _info(self, handle: object):
        info = _WinByHandleFileInformation()
        if not self._kernel32.GetFileInformationByHandle(c_void_p(handle), byref(info)):
            raise self._error("GetFileInformationByHandle failed")
        return info

    def _validate(self, handle: object, *, directory: bool, regular: bool = False) -> None:
        info = self._info(handle)
        if info.attributes & self._FILE_ATTRIBUTE_REPARSE_POINT:
            raise self._error("reparse point is unsafe")
        if directory != bool(info.attributes & self._FILE_ATTRIBUTE_DIRECTORY):
            raise self._error("wrong file type")
        if regular and info.links != 1:
            raise self._error("hard-linked file is unsafe")

    @staticmethod
    def _unicode_name(name: str) -> tuple[bytes, int]:
        """Encode an NT counted string without silently wrapping its USHORT lengths."""
        if "\x00" in name:
            raise ValueError("Windows relative object names cannot contain NUL")
        encoded = name.encode("utf-16-le")
        # MaximumLength includes the terminating WCHAR in create_unicode_buffer().  Both
        # UNICODE_STRING members are USHORT, so leave room for that terminator rather than
        # allowing ctypes' c_ushort conversion to turn a long untrusted name into a prefix.
        if len(encoded) > 65532:
            raise ValueError("Windows relative object name is too long")
        return encoded, len(encoded) + 2

    def _open_relative(self, parent: object, name: str, *, directory: bool,
                       disposition: int = _FILE_OPEN, access: int | None = None) -> object:
        # Keep the backing unicode buffer alive across NtCreateFile.
        encoded, maximum_length = self._unicode_name(name)
        text = self._ctypes.create_unicode_buffer(name)
        us = _WinUnicodeString(len(encoded), maximum_length,
                                  self._ctypes.cast(text, c_void_p))
        attrs = _WinObjectAttributes(self._ctypes.sizeof(_WinObjectAttributes), c_void_p(parent),
                                        pointer(us), 0, None, None)
        iosb = _WinIoStatusBlock()
        handle = c_void_p()
        options = (self._FILE_DIRECTORY_FILE if directory else self._FILE_NON_DIRECTORY_FILE)
        options |= (self._FILE_SYNCHRONOUS_IO_NONALERT | self._FILE_OPEN_FOR_BACKUP_INTENT
                    | self._FILE_OPEN_REPARSE_POINT)
        desired = access if access is not None else (self._GENERIC_READ | self._FILE_READ_ATTRIBUTES | self._SYNCHRONIZE)
        status = self._ntdll.NtCreateFile(byref(handle), c_ulong(desired), byref(attrs), byref(iosb),
                                          None, 0, self._FILE_SHARE_ALL, disposition, options,
                                          None, 0)
        if self._failed(status):
            if disposition == self._FILE_CREATE and status == self._STATUS_OBJECT_NAME_COLLISION:
                raise FileExistsError(name)
            if status in {self._STATUS_OBJECT_NAME_NOT_FOUND, self._STATUS_OBJECT_PATH_NOT_FOUND}:
                raise FileNotFoundError(name)
            raise self._error("NtCreateFile failed")
        try:
            self._validate(handle.value, directory=directory, regular=not directory)
            return handle.value
        except Exception:
            self.close(handle.value)
            raise

    def _open_anchor(self, path: Path) -> object:
        """Open the drive/UNC anchor itself without traversing a reparse point."""
        handle = self._kernel32.CreateFileW(path.anchor, self._GENERIC_READ | self._FILE_LIST_DIRECTORY
                                            | self._FILE_READ_ATTRIBUTES | self._SYNCHRONIZE,
                                            self._FILE_SHARE_ALL, None, self._OPEN_EXISTING,
                                            self._FILE_FLAG_BACKUP_SEMANTICS | self._FILE_FLAG_OPEN_REPARSE_POINT,
                                            None)
        if handle == self._INVALID_HANDLE_VALUE:
            raise FileNotFoundError(f"replay corpus root is missing or unsafe: {path}")
        try:
            self._validate(handle, directory=True)
            return handle
        except Exception as exc:
            self.close(handle)
            raise FileNotFoundError(f"replay corpus root is missing or unsafe: {path}") from exc

    def _walk_root(self, root: str | Path, *, create: bool) -> tuple[Path, object]:
        """Traverse every root component by handle, never through a reparse point."""
        path = Path(root).absolute()
        if not path.anchor:
            raise ValueError(f"replay corpus root is not an absolute Windows path: {path}")
        handle = self._open_anchor(path)
        try:
            # ``parts[0]`` is the drive or UNC share anchor; all remaining components are opened
            # relative to its already-validated handle.
            for component in path.parts[1:]:
                try:
                    child = self._open_relative(handle, component, directory=True)
                except FileNotFoundError:
                    if not create:
                        raise
                    try:
                        self.mkdir_capture(handle, component)
                    except FileExistsError:
                        # As on POSIX, a concurrent creator is harmless only if the
                        # handle-relative, no-reparse open below accepts its result.
                        pass
                    child = self._open_relative(handle, component, directory=True)
                self.close(handle)
                handle = child
            return path, handle
        except Exception:
            self.close(handle)
            raise

    def open_root(self, root: str | Path) -> tuple[Path, object]:
        return self._walk_root(root, create=False)

    def ensure_root(self, root: str | Path) -> tuple[Path, object]:
        return self._walk_root(root, create=True)

    def close(self, handle: object) -> None:
        if handle and not self._kernel32.CloseHandle(c_void_p(handle)):
            raise self._error("CloseHandle failed")

    def open_capture(self, root: object, replay_id: str) -> object:
        return self._open_relative(root, replay_id, directory=True)

    def mkdir_capture(self, root: object, replay_id: str) -> None:
        handle = self._open_relative(root, replay_id, directory=True, disposition=self._FILE_CREATE,
                                     access=self._GENERIC_WRITE | self._FILE_READ_ATTRIBUTES | self._SYNCHRONIZE)
        self.close(handle)

    def manifest_exists(self, directory: object) -> bool:
        try:
            handle = self._open_relative(directory, _MANIFEST_NAME, directory=False)
        except FileNotFoundError:
            return False
        self.close(handle)
        return True

    def _read(self, handle: object) -> bytes:
        chunks = []
        while True:
            buffer = create_string_buffer(65536)
            count = c_ulong()
            if not self._kernel32.ReadFile(c_void_p(handle), buffer, len(buffer), byref(count), None):
                raise self._error("ReadFile failed")
            if not count.value:
                return b"".join(chunks)
            chunks.append(buffer.raw[:count.value])

    def read_regular(self, directory: object, name: str) -> bytes:
        handle = self._open_relative(directory, name, directory=False)
        try:
            return self._read(handle)
        finally:
            self.close(handle)

    def write_new(self, directory: object, name: str, data: bytes) -> None:
        handle = self._open_relative(directory, name, directory=False, disposition=self._FILE_CREATE,
                                     access=self._GENERIC_WRITE | self._FILE_READ_ATTRIBUTES | self._SYNCHRONIZE)
        try:
            view = memoryview(data)
            while view:
                chunk = view[:65536]
                buffer = create_string_buffer(chunk.tobytes())
                count = c_ulong()
                if not self._kernel32.WriteFile(c_void_p(handle), buffer, len(chunk), byref(count), None):
                    raise self._error("WriteFile failed")
                if not count.value:
                    raise self._error("WriteFile made no progress")
                view = view[count.value:]
        finally:
            self.close(handle)

    @classmethod
    def _parse_directory_information(cls, raw: bytes) -> list[str]:
        """Parse only complete, aligned FILE_DIRECTORY_INFORMATION records."""
        result: list[str] = []
        offset = 0
        limit = len(raw)
        while offset < limit:
            if limit - offset < 64:
                raise ValueError("truncated FILE_DIRECTORY_INFORMATION header")
            next_offset = int.from_bytes(raw[offset:offset + 4], "little")
            length = int.from_bytes(raw[offset + 60:offset + 64], "little")
            if length % 2:
                raise ValueError("odd FILE_DIRECTORY_INFORMATION filename length")
            name_end = offset + 64 + length
            if name_end > limit:
                raise ValueError("truncated FILE_DIRECTORY_INFORMATION filename")
            if next_offset:
                # Records are LONGLONG-aligned; the next record must start after this one and
                # still be wholly inside the byte count NtQueryDirectoryFile reported.
                if (next_offset % 8 or next_offset < 64 + length
                        or offset + next_offset > limit):
                    raise ValueError("invalid FILE_DIRECTORY_INFORMATION NextEntryOffset")
            name = raw[offset + 64:name_end].decode("utf-16-le")
            if name not in {".", ".."}:
                result.append(name)
            if not next_offset:
                return result
            offset += next_offset
        raise ValueError("FILE_DIRECTORY_INFORMATION lacks a terminating record")

    def listdir(self, directory: object) -> list[str]:
        # FILE_DIRECTORY_INFORMATION: NextEntryOffset (u32), FileIndex (u32), six 64-bit fields,
        # attributes (u32), FileNameLength (u32), then UTF-16LE filename.  Querying by HANDLE is
        # essential: FindFirstFileW would reopen a raceable pathname.
        result: list[str] = []
        restart = True
        buffer_size = self._DIRECTORY_QUERY_INITIAL_BUFFER
        while True:
            buffer = create_string_buffer(buffer_size)
            iosb = _WinIoStatusBlock()
            status = self._ntdll.NtQueryDirectoryFile(c_void_p(directory), None, None, None, byref(iosb),
                                                      buffer, len(buffer), self._FILE_DIRECTORY_INFORMATION,
                                                      False, None, bool(restart))
            if status == self._STATUS_NO_MORE_FILES:
                return result
            written = int(iosb.Information or 0)
            # The API documents both STATUS_BUFFER_OVERFLOW and STATUS_SUCCESS with zero
            # Information for an entry that does not fit.  Restart from a clean result after
            # growing so an entry cannot be lost or duplicated.
            if status == self._STATUS_BUFFER_OVERFLOW or (not self._failed(status) and not written):
                if buffer_size >= self._DIRECTORY_QUERY_MAX_BUFFER:
                    raise self._error("NtQueryDirectoryFile entry exceeds maximum buffer")
                buffer_size = min(buffer_size * 2, self._DIRECTORY_QUERY_MAX_BUFFER)
                result.clear()
                restart = True
                continue
            if self._failed(status):
                raise self._error("NtQueryDirectoryFile failed")
            if written > len(buffer):
                raise self._error("NtQueryDirectoryFile reported an oversized buffer")
            result.extend(self._parse_directory_information(buffer.raw[:written]))
            restart = False

    def replace(self, directory: object, source: str, target: str) -> None:
        handle = self._open_relative(directory, source, directory=False,
                                     access=self._DELETE | self._FILE_READ_ATTRIBUTES | self._SYNCHRONIZE)
        try:
            # FILE_RENAME_INFORMATION has native pointer alignment before RootDirectory.
            # The trailing WCHAR[] begins after FileNameLength, not at sizeof(struct): x64 has
            # four bytes of *trailing* alignment padding in FILE_RENAME_INFORMATION.
            prefix = 20 if self._ctypes.sizeof(c_void_p) == 8 else 12
            encoded = target.encode("utf-16-le")
            buffer = create_string_buffer(prefix + len(encoded))
            buffer[0] = b"\x01"  # ReplaceIfExists
            self._ctypes.memmove(self._ctypes.addressof(buffer) + (8 if prefix == 20 else 4),
                                 byref(c_void_p(directory)), self._ctypes.sizeof(c_void_p))
            self._ctypes.memmove(self._ctypes.addressof(buffer) + (16 if prefix == 20 else 8),
                                 byref(c_ulong(len(encoded))), 4)
            self._ctypes.memmove(self._ctypes.addressof(buffer) + prefix, encoded, len(encoded))
            iosb = _WinIoStatusBlock()
            status = self._ntdll.NtSetInformationFile(c_void_p(handle), byref(iosb), buffer,
                                                       len(buffer), self._FILE_RENAME_INFORMATION)
            if self._failed(status):
                raise self._error("NtSetInformationFile rename failed")
        finally:
            self.close(handle)

    def publish_capture(self, root: object, staging: str, replay_id: str) -> None:
        """Atomically publish a completed staging directory, never replacing a winner."""
        handle = self._open_relative(root, staging, directory=True,
                                     access=self._DELETE | self._FILE_READ_ATTRIBUTES | self._SYNCHRONIZE)
        try:
            prefix = 20 if self._ctypes.sizeof(c_void_p) == 8 else 12
            encoded = replay_id.encode("utf-16-le")
            buffer = create_string_buffer(prefix + len(encoded))
            # ReplaceIfExists stays false: a competing completed capture must win intact.
            self._ctypes.memmove(self._ctypes.addressof(buffer) + (8 if prefix == 20 else 4),
                                 byref(c_void_p(root)), self._ctypes.sizeof(c_void_p))
            self._ctypes.memmove(self._ctypes.addressof(buffer) + (16 if prefix == 20 else 8),
                                 byref(c_ulong(len(encoded))), 4)
            self._ctypes.memmove(self._ctypes.addressof(buffer) + prefix, encoded, len(encoded))
            iosb = _WinIoStatusBlock()
            status = self._ntdll.NtSetInformationFile(c_void_p(handle), byref(iosb), buffer,
                                                       len(buffer), self._FILE_RENAME_INFORMATION)
            if status == self._STATUS_OBJECT_NAME_COLLISION:
                raise FileExistsError(replay_id)
            if self._failed(status):
                raise self._error("NtSetInformationFile capture publish failed")
        finally:
            self.close(handle)

    def _delete_handle(self, handle: object) -> None:
        value = c_byte(1)
        iosb = _WinIoStatusBlock()
        status = self._ntdll.NtSetInformationFile(c_void_p(handle), byref(iosb), byref(value), 1,
                                                   self._FILE_DISPOSITION_INFORMATION)
        if self._failed(status):
            raise self._error("NtSetInformationFile delete failed")

    def _delete_tree(self, directory: object) -> None:
        for name in self.listdir(directory):
            # The open itself rejects junctions/symlinks.  A malicious hard-linked crop is also
            # rejected by _open_relative; leaving it makes the parent deletion fail closed.
            try:
                child = self._open_relative(directory, name, directory=True,
                                            access=self._DELETE | self._FILE_LIST_DIRECTORY
                                            | self._FILE_READ_ATTRIBUTES | self._SYNCHRONIZE)
            except OSError:
                child = self._open_relative(directory, name, directory=False,
                                            access=self._DELETE | self._FILE_READ_ATTRIBUTES | self._SYNCHRONIZE)
            try:
                if self._info(child).attributes & self._FILE_ATTRIBUTE_DIRECTORY:
                    self._delete_tree(child)
                self._delete_handle(child)
            finally:
                self.close(child)

    def remove_capture(self, root: object, replay_id: str) -> None:
        handle = self._open_relative(root, replay_id, directory=True,
                                     access=self._DELETE | self._FILE_LIST_DIRECTORY
                                     | self._FILE_READ_ATTRIBUTES | self._SYNCHRONIZE)
        try:
            self._delete_tree(handle)
            self._delete_handle(handle)
        finally:
            self.close(handle)

    def capture_identity(self, directory: object) -> object:
        info = self._info(directory)
        return (info.volume, info.index_high, info.index_low)

    def remove_open_capture(self, directory: object, expected_identity: object) -> bool:
        """Delete exactly this already-open directory, never a later name replacement."""
        if self.capture_identity(directory) != expected_identity:
            return False
        self._delete_tree(directory)
        self._delete_handle(directory)
        return True

    def lock_capture(self, root: object, replay_id: str) -> object:
        """Take an OS-owned exclusive lock for one canonical id.

        Closing the returned handle releases the lock even if its owning process dies.  The
        harmless lock file remains under the corpus root and is ignored by discovery/pruning;
        unlike a mkdir sentinel it can never strand future recovery after a crash.
        """
        handle = self._open_relative(
            root, f".lock-{replay_id}", directory=False, disposition=3,
            access=(self._GENERIC_READ | self._GENERIC_WRITE | self._FILE_READ_ATTRIBUTES
                    | self._SYNCHRONIZE))
        overlapped = _WinOverlapped()
        if not self._kernel32.LockFileEx(c_void_p(handle), 0x2, 0, 0xffffffff, 0xffffffff,
                                         byref(overlapped)):
            self.close(handle)
            raise self._error("LockFileEx failed")
        return handle


# Tests replace this object with a deterministic fake.  Keeping selection in one predicate makes
# it possible to exercise Windows rejection/race paths on Linux without monkeypatching ``os.name``.
_WINDOWS_FILESYSTEM: _WindowsCorpusFilesystem | None = None


def _windows_filesystem() -> _WindowsCorpusFilesystem | None:
    global _WINDOWS_FILESYSTEM
    if _WINDOWS_FILESYSTEM is not None:
        return _WINDOWS_FILESYSTEM
    if os.name != "nt":
        return None
    _WINDOWS_FILESYSTEM = _NativeWindowsCorpusFilesystem()
    return _WINDOWS_FILESYSTEM


def _require_safe_open_support() -> None:
    """Fail closed where the platform cannot provide race-free no-follow descriptor opens."""
    if _windows_filesystem() is not None:
        return
    if not hasattr(os, "O_NOFOLLOW") or not _HAS_DIR_FD:
        raise ValueError(
            "replay corpus safe reads require directory-relative O_NOFOLLOW support on this platform")


def _open_corpus_root(root: str | Path) -> tuple[Path, object]:
    """Open a stable corpus-root directory FD, resolving its configured path only once."""
    windows = _windows_filesystem()
    if windows is not None:
        return windows.open_root(root)
    _require_safe_open_support()
    return _walk_posix_corpus_root(root, create=False)


def _walk_posix_corpus_root(root: str | Path, *, create: bool) -> tuple[Path, int]:
    """Open every lexical root component relative to a trusted directory FD.

    ``O_NOFOLLOW`` protects only the *last* pathname component.  Walking one component at a
    time is therefore required to prevent ``root/a`` from escaping through a symlink at
    ``root``.  When ``create`` is true, the same no-follow walk creates missing components for
    the writer without reopening an attacker-controlled pathname.
    """
    _require_safe_open_support()
    root_path = Path(root).absolute()
    anchor_fd = os.open("/", _DIR_OPEN_FLAGS)
    fd = anchor_fd
    try:
        for component in root_path.parts[1:]:
            try:
                child_fd = os.open(component, _DIR_OPEN_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, dir_fd=fd)
                except FileExistsError:
                    # A concurrent creator won; the no-follow open below still validates what
                    # it created rather than trusting the raced pathname.
                    pass
                child_fd = os.open(component, _DIR_OPEN_FLAGS, dir_fd=fd)
            if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
                os.close(child_fd)
                raise ValueError(f"replay corpus root is not a directory: {root_path}")
            os.close(fd)
            fd = child_fd
        return root_path, fd
    except OSError as exc:
        if fd >= 0:
            os.close(fd)
        raise FileNotFoundError(f"replay corpus root is missing or unsafe: {root_path}") from exc
    except Exception:
        if fd >= 0:
            os.close(fd)
        raise


def _ensure_corpus_root(root: str | Path) -> tuple[Path, object]:
    windows = _windows_filesystem()
    if windows is not None:
        return windows.ensure_root(root)
    return _walk_posix_corpus_root(root, create=True)


def _open_capture_directory(root_fd: object, replay_id: object) -> object:
    """Open one canonical capture directory relative to an already-open corpus root."""
    if not _is_safe_capture_id(replay_id):
        raise ValueError("replay_id must be exactly 64 lowercase hexadecimal characters")
    windows = _windows_filesystem()
    if windows is not None:
        try:
            return windows.open_capture(root_fd, replay_id)
        except OSError as exc:
            raise FileNotFoundError(f"replay capture {replay_id!r} is missing or unsafe") from exc
    try:
        if stat.S_ISLNK(os.stat(replay_id, dir_fd=root_fd, follow_symlinks=False).st_mode):
            raise ValueError(f"replay capture {replay_id!r} is a symlink and is unsafe")
        fd = os.open(replay_id, _DIR_OPEN_FLAGS, dir_fd=root_fd)
    except ValueError:
        raise
    except OSError as exc:
        raise FileNotFoundError(f"replay capture {replay_id!r} is missing or unsafe") from exc
    if not stat.S_ISDIR(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError(f"replay capture {replay_id!r} is not a directory")
    return fd


def _safe_capture_filename(name: object, *, field: str) -> str:
    """Validate a manifest filename before using it with a directory file descriptor."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"replay corpus manifest {field} filename must be nonempty text")
    candidate = Path(name)
    # Check both separator families even on Unix.  Tests intentionally exercise the Windows
    # adapter on Unix, and an NT relative object name treats ``\\`` (and drive/stream ``:``)
    # differently from a POSIX filename.
    if (candidate.name != name or name in {".", ".."}
            or any(marker in name for marker in ("/", "\\", ":"))):
        raise ValueError(f"replay corpus manifest {field} filename is not a flat filename")
    return name


def _read_regular_capture_file(directory_fd: object, name: object, *, field: str) -> bytes:
    """Read one flat, singly-linked regular file without blocking or following links."""
    filename = _safe_capture_filename(name, field=field)
    windows = _windows_filesystem()
    if windows is not None:
        try:
            return windows.read_regular(directory_fd, filename)
        except OSError as exc:
            raise ValueError(f"replay corpus {field} file {filename!r} is missing or unsafe") from exc
    try:
        fd = os.open(filename, _FILE_OPEN_FLAGS, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError(f"replay corpus {field} file {filename!r} is missing or unsafe") from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(
                f"replay corpus {field} file {filename!r} is not a singly-linked regular file")
        stream = os.fdopen(fd, "rb")
        fd = -1  # ownership transferred to the context manager
        with stream:
            return stream.read()
    finally:
        if fd >= 0:
            os.close(fd)


def _load_manifest(directory_fd: object, directory: Path) -> dict:
    """Read a manifest through the same no-follow regular-file boundary as crops."""
    try:
        manifest = json.loads(_read_regular_capture_file(
            directory_fd, _MANIFEST_NAME, field="manifest").decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"replay corpus manifest at {directory / _MANIFEST_NAME} is not UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"replay corpus manifest at {directory / _MANIFEST_NAME} is invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"replay corpus manifest at {directory / _MANIFEST_NAME} is not a JSON object")
    return manifest


def _manifest_exists(directory_fd: object) -> bool:
    """Whether a manifest entry exists at all, without following it."""
    windows = _windows_filesystem()
    if windows is not None:
        return windows.manifest_exists(directory_fd)
    try:
        os.stat(_MANIFEST_NAME, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _capture_identity(directory_fd: object) -> object:
    windows = _windows_filesystem()
    if windows is not None:
        return windows.capture_identity(directory_fd)
    metadata = os.fstat(directory_fd)
    return (metadata.st_dev, metadata.st_ino)


def _remove_capture_directory(root_fd: object, replay_id: str, *,
                              expected_identity: object | None = None,
                              open_capture: object | None = None) -> bool:
    """Delete a capture only after an identity-bound private quarantine move.

    The rename binds the deletion to the already-open corpus, so a concurrent replacement of
    either the configured root path or ``replay_id`` cannot redirect ``rmtree`` elsewhere.
    ``shutil.rmtree(..., dir_fd=...)`` is descriptor-relative on platforms that expose the
    descriptor primitives used by this module.
    """
    windows = _windows_filesystem()
    if windows is not None:
        if open_capture is not None and expected_identity is not None:
            return windows.remove_open_capture(open_capture, expected_identity)
        windows.remove_capture(root_fd, replay_id)
        return True
    quarantine = f".prune-{replay_id}-{uuid.uuid4().hex}"
    os.rename(replay_id, quarantine, src_dir_fd=root_fd, dst_dir_fd=root_fd)
    if expected_identity is not None:
        quarantine_fd = -1
        try:
            quarantine_fd = os.open(quarantine, _DIR_OPEN_FLAGS, dir_fd=root_fd)
            if _capture_identity(quarantine_fd) != expected_identity:
                # The name changed after validation.  Do not recursively remove somebody
                # else's replacement, and do not use a normal rename to restore it: POSIX
                # rename would replace a newer canonical destination created in this window.
                # Keeping the quarantined object is the only portable fail-closed outcome.
                return False
        finally:
            if quarantine_fd >= 0:
                os.close(quarantine_fd)
    shutil.rmtree(quarantine, dir_fd=root_fd)
    return True


def _close_directory(handle: object) -> None:
    windows = _windows_filesystem()
    if windows is not None:
        windows.close(handle)
    else:
        os.close(handle)


def _list_directory(handle: object) -> list[str]:
    windows = _windows_filesystem()
    return windows.listdir(handle) if windows is not None else os.listdir(handle)


def _mkdir_capture(root_fd: object, replay_id: str) -> None:
    windows = _windows_filesystem()
    if windows is not None:
        windows.mkdir_capture(root_fd, replay_id)
    else:
        os.mkdir(replay_id, dir_fd=root_fd)


def _write_new_capture_file(directory_fd: object, name: str, data: bytes) -> None:
    windows = _windows_filesystem()
    if windows is not None:
        windows.write_new(directory_fd, name, data)
        return
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory_fd)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


def _replace_capture_file(directory_fd: object, source: str, target: str) -> None:
    windows = _windows_filesystem()
    if windows is not None:
        windows.replace(directory_fd, source, target)
    else:
        os.replace(source, target, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)


def _open_staging_directory(root_fd: object, staging: str) -> object:
    """Open our private staging child without treating its random name as a replay id."""
    windows = _windows_filesystem()
    if windows is not None:
        return windows.open_capture(root_fd, staging)
    return os.open(staging, _DIR_OPEN_FLAGS, dir_fd=root_fd)


def _publish_staging_capture(root_fd: object, staging: str, replay_id: str) -> None:
    """Publish a complete staging directory without replacing an existing capture."""
    windows = _windows_filesystem()
    if windows is not None:
        windows.publish_capture(root_fd, staging, replay_id)
    else:
        os.rename(staging, replay_id, src_dir_fd=root_fd, dst_dir_fd=root_fd)


def _quarantine_incomplete_capture(root_fd: object, replay_id: str) -> None:
    """Move a verified manifest-absent legacy capture aside without deleting it."""
    quarantine = f".incomplete-{replay_id}-{uuid.uuid4().hex}"
    windows = _windows_filesystem()
    if windows is not None:
        windows.publish_capture(root_fd, replay_id, quarantine)
    else:
        os.rename(replay_id, quarantine, src_dir_fd=root_fd, dst_dir_fd=root_fd)


def _lock_capture(root_fd: object, replay_id: str) -> object:
    """Serialize all cooperating writers for one content-addressed destination.

    Legacy recovery has to inspect a manifest-absent canonical directory and then rename it.
    An OS lock spans that inspection through publication, so a second writer cannot act on a
    stale "no manifest" observation and quarantine a completed winner.  The lock is released
    by close/process exit; its ignored sidecar file carries no capture data.
    """
    windows = _windows_filesystem()
    if windows is not None:
        return windows.lock_capture(root_fd, replay_id)
    import fcntl
    name = f".lock-{replay_id}"
    fd = os.open(name, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600,
                 dir_fd=root_fd)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("replay corpus lock is not a singly-linked regular file")
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd
    except Exception:
        os.close(fd)
        raise


def _validate_completed_capture(directory_fd: object, directory: Path, *, replay_id: str,
                                name: str, items: Sequence[bytes], context: Sequence[bytes],
                                truncated: bool) -> None:
    """Prove a canonical winner is exactly the request this writer wants to persist.

    A parseable manifest alone is not an idempotence proof: a corrupted or manually-created
    directory can retain the right name while naming different crops.  Read every declared crop
    through the no-follow boundary and compare both the canonical manifest fields and bytes.
    """
    manifest = _load_manifest(directory_fd, directory)
    expected_items = [f"item_{position:03d}{_guess_extension(image)}"
                      for position, image in enumerate(items, start=1)]
    expected_context = [f"context_{position:03d}{_guess_extension(image)}"
                        for position, image in enumerate(context, start=1)]
    if (manifest.get("format_version") != _FORMAT_VERSION
            or manifest.get("replay_id") != replay_id
            or manifest.get("name") != name
            or type(manifest.get("truncated")) is not bool
            or manifest["truncated"] != truncated
            or manifest.get("items") != expected_items
            or manifest.get("context") != expected_context):
        raise ValueError("existing replay capture does not match the requested canonical content")
    expected_names = set(expected_items + expected_context + [_MANIFEST_NAME])
    if set(_list_directory(directory_fd)) != expected_names:
        raise ValueError("existing replay capture contains unexpected files")
    actual_items = tuple(_read_regular_capture_file(directory_fd, filename, field="items")
                         for filename in expected_items)
    actual_context = tuple(_read_regular_capture_file(directory_fd, filename, field="context")
                           for filename in expected_context)
    if actual_items != tuple(items) or actual_context != tuple(context):
        raise ValueError("existing replay capture crop bytes do not match its canonical id")


def _recoverable_incomplete_capture(directory_fd: object, *, items: Sequence[bytes],
                                    context: Sequence[bytes]) -> bool:
    """True only for an old direct-write partial whose present crops exactly match this request.

    Current writers publish only complete staging directories, so a canonical directory with no
    manifest is necessarily pre-transactional/legacy state.  Do not erase it: retain it under a
    private quarantine name, and only after proving every present entry is an expected,
    singly-linked regular crop with byte-exact expected content.  This refuses arbitrary local
    directories and a different request's collision rather than treating missing metadata as
    permission to move somebody else's files.
    """
    expected = {
        f"item_{position:03d}{_guess_extension(image)}": image
        for position, image in enumerate(items, start=1)
    }
    expected.update({
        f"context_{position:03d}{_guess_extension(image)}": image
        for position, image in enumerate(context, start=1)
    })
    try:
        names = set(_list_directory(directory_fd))
        if not names.issubset(expected):
            return False
        return all(_read_regular_capture_file(directory_fd, name, field="incomplete") == data
                   for name, data in expected.items() if name in names)
    except (OSError, ValueError):
        return False


def _validate_capture_identity(directory_fd: object, directory: Path, replay_id: str) -> None:
    """Validate a discovered capture before a destructive operation.

    This is deliberately stronger than discovery's "parseable manifest" threshold: purge must
    never turn a newly-created partial or a same-name replacement into a deletion target.
    """
    manifest = _load_manifest(directory_fd, directory)
    name = manifest.get("name")
    items = manifest.get("items")
    context = manifest.get("context")
    truncated = manifest.get("truncated")
    if (manifest.get("format_version") != _FORMAT_VERSION or manifest.get("replay_id") != replay_id
            or not isinstance(name, str) or not isinstance(items, list) or not items
            or not isinstance(context, list) or type(truncated) is not bool
            or len(set(items + context)) != len(items) + len(context)):
        raise ValueError("replay capture is not a complete canonical capture")
    item_bytes = tuple(_read_regular_capture_file(directory_fd, filename, field="items")
                       for filename in items)
    context_bytes = tuple(_read_regular_capture_file(directory_fd, filename, field="context")
                          for filename in context)
    if compute_replay_id(name=name, items=item_bytes, context=context_bytes,
                         truncated=truncated) != replay_id:
        raise ValueError("replay capture contents do not match its canonical id")


def delete_replay_captures(root: str | Path, replay_ids: Sequence[str]) -> tuple[str, ...]:
    """Remove selected canonical captures using one stable root descriptor.

    This is intentionally narrow: callers must discover the ids separately, and malformed or
    concurrently replaced entries are left alone.  It is used by the CLI's whole-corpus purge
    rather than allowing that command to recursively delete an unverified root pathname.
    """
    root_path, root_fd = _open_corpus_root(root)
    try:
        removed = []
        for replay_id in replay_ids:
            if not _is_safe_capture_id(replay_id):
                continue
            lock_fd: object | None = None
            try:
                lock_fd = _lock_capture(root_fd, replay_id)
                capture_fd = _open_capture_directory(root_fd, replay_id)
                try:
                    _validate_capture_identity(capture_fd, root_path / replay_id, replay_id)
                    identity = _capture_identity(capture_fd)
                    if not _remove_capture_directory(root_fd, replay_id,
                                                     expected_identity=identity,
                                                     open_capture=capture_fd):
                        continue
                finally:
                    _close_directory(capture_fd)
            except (OSError, ValueError):
                continue
            finally:
                if lock_fd is not None:
                    _close_directory(lock_fd)
            removed.append(replay_id)
        return tuple(removed)
    finally:
        _close_directory(root_fd)


def _guess_extension(data: bytes) -> str:
    """A human-readable file extension for ``data``'s image format, purely for on-disk
    friendliness (so a directory listing itself hints at what a file is). Never load-bearing --
    every reader here reads bytes off the filename recorded in the manifest, not off the
    extension, so a format this function doesn't recognize still round-trips correctly under
    the generic ".bin" fallback."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    return ".bin"


def compute_replay_id(*, name: str, items: Sequence[bytes], context: Sequence[bytes] = (),
                      truncated: bool = False) -> str:
    """The STABLE identifier for one captured request: a SHA-256 digest of its own content
    (her name, the truncation flag, and every item/context crop's own digest, in order), never
    of wall-clock time or a random value.

    STABLE means two things, both deliberate: (1) writing the exact same capture twice (a
    caller retrying after a transient failure, say) lands in the exact same directory rather
    than creating a duplicate copy of someone's photos on disk, and (2) two independent replay
    runs over "the same set" of captures can refer to entries by this id and know they mean the
    same underlying request -- see tools/opener_replay.py's --replay-ids selection.

    Per-image content is folded in via each image's OWN sha256 digest rather than the raw bytes
    directly, so this function never has to hold two full copies of a multi-megabyte crop in
    memory at once (the image's digest, then this function's own running hash) -- only one.
    """
    hasher = hashlib.sha256()
    hasher.update(str(name or "").strip().encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(b"1" if truncated else b"0")
    for image in items:
        hasher.update(b"\x00")
        hasher.update(hashlib.sha256(bytes(image)).digest())
    hasher.update(b"\x00\x00")  # a fixed double-NUL boundary between the items tier and the
                                # context tier, so an items-heavy capture and a context-heavy
                                # one can never hash the same way merely by shifting crops
                                # across that boundary.
    for image in context:
        hasher.update(b"\x00")
        hasher.update(hashlib.sha256(bytes(image)).digest())
    return hasher.hexdigest()


@dataclass(frozen=True)
class ReplayWriteResult:
    """The outcome of one ``write_replay_capture`` call. Always returned, never raised (see
    this module's docstring's SAFE TO FAIL section) -- a caller checks ``ok`` and moves on
    either way."""
    ok: bool
    replay_id: str | None = None
    path: Path | None = None
    error: str | None = None


def write_replay_capture(root: str | Path, *, items: Sequence[bytes], name: str = "",
                         context: Sequence[bytes] = (), truncated: bool = False,
                         prompt_sha256: str | None = None,
                         captured_at: float | None = None) -> ReplayWriteResult:
    """Write one captured item-crop request to the corpus under ``root``. PURE I/O: this
    function has no dependency on ``operation_love.opener`` or any service internals -- it
    takes the numbered request fields (``items``, ``name``, ``truncated``) plus the separately
    retained forensic ``context`` crops as plain bytes/str/bool, plus the caller's own
    already-computed ``prompt_sha256`` (this module never computes one itself -- see the module
    docstring).

    WIRED INTO THE OPENER PATH from ``operation_love.opener.service.OpenerService.
    _capture_replay_corpus``, which calls this once per profile with the model-visible
    ``ItemRequest`` fields it just built plus its retained forensic context, right before making
    the actual provider call (see that method's own docstring). It remains a plain, standalone
    API otherwise -- a test or a manual script can still call it directly with no service
    involved.

    NEVER RAISES (see this module's docstring's SAFE TO FAIL section) -- every failure, from a
    bad argument to a full disk, comes back as ``ReplayWriteResult(ok=False, error=...)``, so a
    caller on the opener/decision/send/refusal path can call this unconditionally with no
    try/except of its own, exactly as this project's telemetry rule requires.

    IDEMPOTENT BY CONTENT: ``replay_id`` is derived purely from ``name``/``items``/``context``/
    ``truncated`` (see ``compute_replay_id``), so writing the identical capture twice is a
    cheap no-op the second time (detected via the existing directory already having a
    ``manifest.json``) rather than a duplicate copy of someone's photos on disk.

    ATOMICITY, to the extent a plain filesystem allows it: crops and the manifest are first
    written inside a private random sibling directory. The completed directory is then atomically
    renamed into the content-derived ``replay_id`` only after its manifest is in place. A crash
    or exception can therefore leave only a private staging orphan; it can never leave partial
    crops occupying the deterministic id and poisoning a later retry. Discovery ignores staging
    names, and a later write neither reads nor deletes them, so a concurrent writer's incomplete
    work is never mistaken for ours.
    """
    try:
        items_t = tuple(bytes(image) for image in items)
        context_t = tuple(bytes(image) for image in context)
        name_s = str(name or "").strip()
        truncated_b = bool(truncated)
        if not items_t:
            raise ValueError(
                "write_replay_capture requires at least one numbered item crop; a request "
                "with none is not something a replay could ever reconstruct meaningfully")
        for position, image in enumerate(items_t + context_t):
            if not image:
                raise ValueError(f"replay capture image at position {position} is empty")

        replay_id = compute_replay_id(name=name_s, items=items_t, context=context_t,
                                      truncated=truncated_b)
        # Create/open the root once, then do every untrusted child operation relative to that
        # descriptor.  In particular, never let a pre-created ``<id>`` symlink turn a capture
        # into a write outside the corpus.
        directory, root_fd = _ensure_corpus_root(root)
        lock_fd: object | None = None
        try:
            lock_fd = _lock_capture(root_fd, replay_id)
            # Versions before the staging transaction wrote crops directly under replay_id.
            # Recover that one legacy failure shape without trusting/deleting arbitrary content:
            # a manifest-bearing directory is the normal idempotent winner; a manifest-absent
            # directory is moved aside only when every present crop byte-exactly belongs to THIS
            # request.  New writers cannot create this state.
            try:
                existing_fd = _open_capture_directory(root_fd, replay_id)
            except FileNotFoundError:
                existing_fd = None
            if existing_fd is not None:
                try:
                    if _manifest_exists(existing_fd):
                        _validate_completed_capture(
                            existing_fd, directory / replay_id, replay_id=replay_id,
                            name=name_s, items=items_t, context=context_t,
                            truncated=truncated_b)
                        return ReplayWriteResult(ok=True, replay_id=replay_id,
                                                 path=directory / replay_id)
                    if not _recoverable_incomplete_capture(
                            existing_fd, items=items_t, context=context_t):
                        raise ValueError(
                            "existing manifest-absent replay directory is not a verified partial "
                            "for this request")
                finally:
                    _close_directory(existing_fd)
                _quarantine_incomplete_capture(root_fd, replay_id)

            # Never write crops in the content-addressed destination itself.  If an I/O error
            # or process crash happens before its manifest, retrying the same request must not
            # inherit an unverified partial crop (nor race another writer's partial directory).
            # A private, random sibling is invisible to discovery/pruning and is atomically
            # renamed into the canonical id only after the manifest is safely in place.
            staging = f".write-{uuid.uuid4().hex}"
            _mkdir_capture(root_fd, staging)
            published = False
            capture_fd = _open_staging_directory(root_fd, staging)
            try:
                item_files = []
                for position, image in enumerate(items_t, start=1):
                    filename = f"item_{position:03d}{_guess_extension(image)}"
                    _write_new_capture_file(capture_fd, filename, image)
                    item_files.append(filename)

                context_files = []
                for position, image in enumerate(context_t, start=1):
                    filename = f"context_{position:03d}{_guess_extension(image)}"
                    _write_new_capture_file(capture_fd, filename, image)
                    context_files.append(filename)

                manifest = {
                    "format_version": _FORMAT_VERSION,
                    "replay_id": replay_id,
                    "captured_at": float(captured_at) if captured_at is not None else time.time(),
                    "name": name_s,
                    "truncated": truncated_b,
                    "prompt_sha256": str(prompt_sha256) if prompt_sha256 else None,
                    "items": item_files,
                    "context": context_files,
                }
                tmp_name = _MANIFEST_NAME + ".tmp"
                _write_new_capture_file(capture_fd, tmp_name,
                                        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
                _replace_capture_file(capture_fd, tmp_name, _MANIFEST_NAME)
            finally:
                _close_directory(capture_fd)
            try:
                _publish_staging_capture(root_fd, staging, replay_id)
                published = True
                return ReplayWriteResult(ok=True, replay_id=replay_id,
                                         path=directory / replay_id)
            except OSError as publish_exc:
                # A concurrent writer (or an already-complete prior request) won.  Its
                # manifest is the only acceptable proof of idempotent success; never replace,
                # remove, or reuse an existing content-id directory.
                try:
                    winner_fd = _open_capture_directory(root_fd, replay_id)
                    try:
                        _load_manifest(winner_fd, directory / replay_id)
                        _validate_completed_capture(
                            winner_fd, directory / replay_id, replay_id=replay_id,
                            name=name_s, items=items_t, context=context_t,
                            truncated=truncated_b)
                    finally:
                        _close_directory(winner_fd)
                except (FileNotFoundError, ValueError, OSError):
                    # POSIX reports a nonempty-directory collision as ENOTEMPTY rather than
                    # EEXIST.  Do not mistake any other publish failure for a winner unless its
                    # complete manifest crossed the same safe boundary as a normal read.
                    raise publish_exc
                return ReplayWriteResult(ok=True, replay_id=replay_id,
                                         path=directory / replay_id)
            finally:
                if not published:
                    try:
                        _remove_capture_directory(root_fd, staging)
                    except Exception:
                        # A crash-safe orphaned private staging directory is preferable to
                        # risking any operation on the canonical id or a competing writer.
                        pass
        finally:
            if lock_fd is not None:
                _close_directory(lock_fd)
            _close_directory(root_fd)
    except Exception as exc:  # noqa: BLE001 -- see docstring: this function must never raise
        return ReplayWriteResult(ok=False, error=f"{type(exc).__name__}: {exc}")


@dataclass(frozen=True)
class ReplayCapture:
    """One captured request, read back off disk -- the read-side mirror of the four fields
    ``write_replay_capture`` took in, plus the bookkeeping fields every manifest carries.

    ``items``/``context`` are populated (actual crop bytes, in order) rather than left as
    filenames. A caller can reconstruct the numbered Gemini request without a second filesystem
    pass; the context bytes remain available only for forensic inspection and research.
    """
    replay_id: str
    path: Path
    name: str
    items: tuple[bytes, ...]
    context: tuple[bytes, ...]
    truncated: bool
    prompt_sha256: str | None
    captured_at: float | None


def list_replay_ids(root: str | Path) -> list[str]:
    """Every replay_id under ``root`` with a readable, parseable manifest, in a DETERMINISTIC
    default order: ascending ``captured_at`` (a capture made earlier sorts first), tie-broken by
    ``replay_id`` itself so two entries captured in the same wall-clock instant still sort the
    same way on every call. A directory that is not a capture at all (no manifest.json, or a
    manifest.json that fails to parse) is silently skipped rather than raising -- discovery must
    survive a stray file or an in-progress write sitting under the same root; see
    ``load_replay_capture`` for the loud, raising counterpart used once a specific id is chosen.

    Returns ``[]`` for a root that does not exist yet -- an empty corpus, not an error.
    """
    try:
        root_path, root_fd = _open_corpus_root(root)
    except FileNotFoundError:
        return []
    entries: list[tuple[float, str]] = []
    try:
        for name in _list_directory(root_fd):
            if not _is_safe_capture_id(name):
                continue
            try:
                capture_fd = _open_capture_directory(root_fd, name)
                try:
                    manifest = _load_manifest(capture_fd, root_path / name)
                finally:
                    _close_directory(capture_fd)
            except (OSError, ValueError):
                continue
            captured_at = manifest.get("captured_at")
            sort_key = float(captured_at) if isinstance(captured_at, (int, float)) else float("inf")
            entries.append((sort_key, name))
    finally:
        _close_directory(root_fd)
    entries.sort(key=lambda pair: (pair[0], pair[1]))
    return [replay_id for _sort_key, replay_id in entries]


def load_replay_capture(root: str | Path, replay_id: str) -> ReplayCapture:
    """Read one capture back off disk.

    Raises ``FileNotFoundError`` or ``ValueError`` for a missing/unsafe directory, a malformed
    manifest, or an unsafe/missing crop.  In particular, invalid JSON is normalized to
    ``ValueError`` so callers get one clear malformed-corpus contract rather than needing to
    know the JSON parser's exception hierarchy.
    """
    root_path, root_fd = _open_corpus_root(root)
    directory = root_path / replay_id
    manifest_path = directory / _MANIFEST_NAME
    try:
        capture_fd = _open_capture_directory(root_fd, replay_id)
        try:
            manifest = _load_manifest(capture_fd, directory)
            manifest_replay_id = manifest.get("replay_id")
            if manifest_replay_id != replay_id:
                raise ValueError(
                    f"replay corpus manifest at {manifest_path} declares replay_id "
                    f"{manifest_replay_id!r}, which does not match its own directory name {replay_id!r}")

            item_files = manifest.get("items")
            if not isinstance(item_files, list) or not item_files:
                raise ValueError(
                    f"replay corpus manifest at {manifest_path} has no numbered items")
            context_files = manifest.get("context")
            if context_files is None:
                context_files = []
            if not isinstance(context_files, list):
                raise ValueError(
                    f"replay corpus manifest at {manifest_path} has a non-list 'context' field")

            items = tuple(_read_regular_capture_file(capture_fd, fname, field="items")
                          for fname in item_files)
            context = tuple(_read_regular_capture_file(capture_fd, fname, field="context")
                            for fname in context_files)
            captured_at = manifest.get("captured_at")
            prompt_sha256 = manifest.get("prompt_sha256")
            return ReplayCapture(
                replay_id=replay_id,
                path=directory,
                name=str(manifest.get("name") or ""),
                items=items,
                context=context,
                truncated=bool(manifest.get("truncated", False)),
                prompt_sha256=str(prompt_sha256) if prompt_sha256 else None,
                captured_at=float(captured_at) if isinstance(captured_at, (int, float)) else None,
            )
        finally:
            _close_directory(capture_fd)
    finally:
        _close_directory(root_fd)


def load_replay_corpus(root: str | Path, *, replay_ids: Sequence[str] | None = None,
                       limit: int | None = None) -> list[ReplayCapture]:
    """Load a whole selection from the corpus, in order.

    ``replay_ids``, when given, is BOTH the subset AND the exact order -- passing the same list
    twice (e.g. across two prompt-revision replay runs meant to be compared against each other)
    guarantees the same captures in the same order both times, which ``list_replay_ids``'s own
    default (captured_at, replay_id) ordering also guarantees on its own but a caller may still
    want to pin explicitly. When omitted, defaults to every id ``list_replay_ids`` finds under
    ``root``, in its deterministic order.

    ``limit``, when given, keeps only the first ``limit`` entries of whichever ordering was
    already chosen above -- applied AFTER subset/order selection, never before, so
    ``replay_ids=[...]`` plus ``limit=5`` means "the first 5 of exactly this list", not
    "5 arbitrary ids from this list".
    """
    ids = list(replay_ids) if replay_ids is not None else list_replay_ids(root)
    if limit is not None:
        ids = ids[:limit]
    return [load_replay_capture(root, replay_id) for replay_id in ids]


@dataclass(frozen=True)
class PruneRemoval:
    """One capture ``prune_replay_corpus`` actually removed -- enough to log without a second
    filesystem read (the directory is already gone by the time a caller sees this)."""
    replay_id: str
    captured_at: float | None
    reason: str   # PRUNE_REASON_MAX_CAPTURES or PRUNE_REASON_MAX_AGE_DAYS


PRUNE_REASON_MAX_CAPTURES = "max_captures"
PRUNE_REASON_MAX_AGE_DAYS = "max_age_days"


@dataclass(frozen=True)
class PruneResult:
    """The outcome of one ``prune_replay_corpus`` call. Always returned, never raised (see that
    function's own SAFE TO FAIL section) -- a caller checks ``ok`` and moves on either way,
    exactly like ``ReplayWriteResult``."""
    ok: bool
    removed: tuple[PruneRemoval, ...] = ()
    kept: int = 0
    skipped_unparseable: int = 0
    error: str | None = None


def prune_replay_corpus(root: str | Path, *, max_captures: int = 0,
                        max_age_days: float = 0) -> PruneResult:
    """Bound the corpus under ``root`` by deleting whole capture directories, OLDEST-FIRST by
    the manifest's own ``captured_at``, until it satisfies both ``max_captures`` (an absolute
    cap on how many captures may exist) and ``max_age_days`` (no surviving capture is older than
    this many days). **0 MEANS UNLIMITED for either bound** -- ``max_captures=0`` never removes
    anything for being over-count, ``max_age_days=0`` never removes anything for being too old,
    and ``prune_replay_corpus(root)`` with both left at their defaults is a documented no-op.

    ORDER OF OPERATIONS: age-based removal is applied FIRST, then count-based removal is applied
    to whatever survives it. This is deliberate, not incidental -- it means the two bounds
    compose the way an operator would expect ("nothing older than N days, and no more than M
    captures even within that window") rather than either bound alone deciding survivors
    independently of the other. A capture matches at most one reason in the returned
    ``PruneResult.removed`` (age-removed captures are excluded from the count-removal pool by
    construction, so the two groups can never overlap).

    A CORRUPT OR UNREADABLE MANIFEST IS A DELIBERATE, DOCUMENTED "KEEP", NEVER A DELETE. A
    directory under ``root`` whose ``manifest.json`` is missing, unreadable, not valid JSON, or
    not a JSON object is treated EXACTLY the way ``list_replay_ids`` already treats it: not a
    (complete) capture at all. This function therefore never counts it toward ``max_captures``,
    never compares it against ``max_age_days``, and never deletes it -- it is silently excluded
    from every decision this function makes, the same way it is silently excluded from every
    listing ``list_replay_ids`` produces. Three reasons this is the right default, not merely the
    easy one: (1) this function's own delete-path safety mandate is to make it impossible to
    remove the wrong thing, and a directory whose manifest cannot even be parsed is one this
    function cannot verify is genuinely a spent, replaceable capture rather than, say, a
    mid-external-copy directory or hand-edited evidence -- guessing wrong in the "prunable"
    direction is irreversible, guessing wrong in the "keep" direction only costs disk; (2) reader
    and pruner sharing the exact same definition of "what counts as a capture" means a tool built
    against ``list_replay_ids`` can never see a directory that ``prune_replay_corpus`` silently
    disagreed with it about; (3) ``write_replay_capture``'s own atomicity guarantee (the manifest
    is written LAST, via an atomic replace, only after every crop file already landed) means a
    present-but-corrupt ``manifest.json`` can only happen from something OUTSIDE this module
    entirely -- external disk corruption or manual tampering -- which is rare enough that an
    unbounded-but-rare residual is an acceptable, explicitly accepted risk (the same shape of
    trade-off ``write_replay_capture`` itself already documents for orphaned crop files after a
    crash). ``PruneResult.skipped_unparseable`` counts these so the residual is at least visible
    to a caller who logs it, even though it is never acted on.

    A capture whose manifest DOES parse but omits/mistypes ``captured_at`` (unusual -- every
    manifest this module itself writes always sets it via ``time.time()``) is handled
    differently, on purpose: it is still a candidate (a parseable, well-formed capture), but with
    an unknown age. It sorts as `+inf` for ordering, MATCHING ``list_replay_ids``'s own
    ``captured_at`` tie-break convention exactly, which has two effects: it is NEVER removed for
    age (an unknown age can never be compared against a cutoff), and it is the LAST candidate
    ever removed for count (oldest-first ordering means "sorts newest" removes last), so this
    function never preferentially punishes a capture merely because its age cannot be determined.

    DELETE-PATH SAFETY, the highest-risk part of this function's contract: it must be impossible
    for this function to delete anything outside ``root``, or anything a caller did not mean for
    it to manage. Every one of these must hold before a single ``shutil.rmtree`` call is made for
    a given candidate:

      * ``root`` itself is resolved once, up front (``Path(root).resolve()``), and every later
        comparison is against that resolved path, never the original possibly-relative one;
      * the candidate is an IMMEDIATE child of ``root`` discovered via ``root.iterdir()`` --
        never a name built from string concatenation or user input;
      * the candidate's directory NAME must independently match ``_is_safe_capture_id`` (exactly
        64 lowercase hex characters -- the shape ``compute_replay_id`` produces and the ONLY
        shape this module has ever written to disk), checked once at discovery and AGAIN,
        redundantly, immediately before deletion;
      * the candidate must not be a symlink (``Path.is_symlink()``) -- checked at discovery, so a
        symlink is never even parsed as a capture, let alone considered for removal; this module
        never follows a symlink to decide what to delete;
      * immediately before deletion, the candidate is resolved again and its resolved path must
        (a) be a direct child of resolved ``root`` (``resolved.parent == root`` and
        ``resolved.name`` equal to the id being removed) AND (b) satisfy
        ``resolved.is_relative_to(root)`` -- belt-and-suspenders confirmation that nothing in
        between discovery and deletion (nor any symlink component this function did not
        anticipate) could have moved the real target outside ``root``.

    Every one of the checks above is a `continue`, never a raise: a candidate that fails ANY of
    them is simply left alone (kept), and the loop moves on to the next one. See
    tests/test_replay_corpus_prune.py for a test that attempts each escape explicitly and proves
    it is refused rather than merely "not currently exploited".

    NEVER RAISES (same SAFE TO FAIL contract as ``write_replay_capture``): the ENTIRE body,
    including argument validation and the directory walk itself, runs inside one try/except, so
    a bad argument, a permissions error on ``root``, a full disk, or any other unexpected failure
    comes back as ``PruneResult(ok=False, error=...)`` rather than propagating -- a caller on the
    opener/decision/send/refusal path (this project's own hard rule: telemetry must be
    best-effort and must never raise into that path) can call this unconditionally with no
    try/except of its own, exactly like ``write_replay_capture``. A per-directory deletion
    failure (e.g. one directory becomes unwritable mid-run) is handled the same way one level
    down: that single candidate is skipped (left in place, NOT counted as removed) and every
    other candidate is still attempted -- one bad directory can never abort the whole prune.
    """
    try:
        if isinstance(max_captures, bool) or not isinstance(max_captures, int) or max_captures < 0:
            raise ValueError(
                "prune_replay_corpus max_captures must be a non-negative integer "
                f"(0 = unlimited), got {max_captures!r}")
        if (isinstance(max_age_days, bool) or not isinstance(max_age_days, (int, float))
                or max_age_days < 0 or max_age_days != max_age_days  # NaN check, no math import
                or max_age_days in (float("inf"), float("-inf"))):
            raise ValueError(
                "prune_replay_corpus max_age_days must be a non-negative, finite number of days "
                f"(0 = unlimited), got {max_age_days!r}")

        candidates: list[tuple[str, float | None]] = []
        skipped_unparseable = 0
        try:
            root_path, root_fd = _open_corpus_root(root)
        except FileNotFoundError:
            # An empty/nonexistent corpus is nothing to prune, not an error -- matches
            # list_replay_ids's own "[] for a root that does not exist yet" contract.
            return PruneResult(ok=True)
        try:
            for name in _list_directory(root_fd):
                if not _is_safe_capture_id(name):
                    # Not shaped like anything this module has ever written; not ours to manage.
                    continue
                try:
                    capture_fd = _open_capture_directory(root_fd, name)
                except (FileNotFoundError, ValueError):
                    # Includes a symlinked capture directory: never follow it or count it.
                    continue
                try:
                    if not _manifest_exists(capture_fd):
                        # An incomplete writer directory has no manifest at all and is not a
                        # malformed capture; preserve the established no-count/no-delete policy.
                        continue
                    try:
                        manifest = _load_manifest(capture_fd, root_path / name)
                    except ValueError:
                        # A missing, malformed, symlinked, hard-linked, or special manifest is
                        # unparseable evidence: keep the directory rather than deleting it.
                        skipped_unparseable += 1
                        continue
                finally:
                    _close_directory(capture_fd)
                captured_at = manifest.get("captured_at")
                captured_at_f = (float(captured_at)
                                 if isinstance(captured_at, (int, float)) else None)
                candidates.append((name, captured_at_f))
            # Keep this descriptor open through deletion.  Re-opening ``root_path`` here would
            # reintroduce the exact root-swap race the discovery pass avoided.
            candidates.sort(key=lambda pair: (pair[1] if pair[1] is not None else float("inf"),
                                              pair[0]))
            age_ids = {cid for cid, cat in candidates
                       if max_age_days > 0 and cat is not None
                       and cat < time.time() - (max_age_days * 86400.0)}
            remaining = [pair for pair in candidates if pair[0] not in age_ids]
            count_ids = ({cid for cid, _cat in remaining[:len(remaining) - max_captures]}
                         if max_captures > 0 and len(remaining) > max_captures else set())
            removal_plan = [
                (cid, cat, PRUNE_REASON_MAX_AGE_DAYS if cid in age_ids
                 else PRUNE_REASON_MAX_CAPTURES)
                for cid, cat in candidates if cid in age_ids or cid in count_ids]

            removed: list[PruneRemoval] = []
            for cid, cat, reason in removal_plan:
                if not _is_safe_capture_id(cid):
                    continue
                try:
                    # Re-open verifies it is still a real, no-follow directory immediately
                    # before the descriptor-relative rename/delete.
                    check_fd = _open_capture_directory(root_fd, cid)
                    try:
                        # Match discovery's documented threshold (a parseable manifest) while
                        # binding the following deletion to this exact opened object.  The
                        # dry-run age preview intentionally mirrors manifests only.
                        _load_manifest(check_fd, root_path / cid)
                        identity = _capture_identity(check_fd)
                        if not _remove_capture_directory(root_fd, cid,
                                                         expected_identity=identity,
                                                         open_capture=check_fd):
                            continue
                    finally:
                        _close_directory(check_fd)
                except Exception:  # noqa: BLE001 -- one bad directory must never abort prune
                    continue
                removed.append(PruneRemoval(replay_id=cid, captured_at=cat, reason=reason))
            return PruneResult(ok=True, removed=tuple(removed),
                               kept=len(candidates) - len(removed),
                               skipped_unparseable=skipped_unparseable)
        finally:
            _close_directory(root_fd)
    except Exception as exc:  # noqa: BLE001 -- see docstring: this function must never raise
        return PruneResult(ok=False, error=f"{type(exc).__name__}: {exc}")


# The `decision` column marker stamped on every openers row written by an OFFLINE REPLAY
# (tools/opener_replay.py) rather than captured from a live run.
#
# IT LIVES HERE, NOT IN THE CLI, for an import-path reason worth stating: `operation_love` is an
# installed package and importable from anywhere, while `tools/` is only importable when the repo
# root happens to be on sys.path. A consumer that did `from tools.opener_replay import ...` broke
# `python tools/opener_corpus_report.py` outright (ModuleNotFoundError: No module named 'tools'),
# because a direct script run puts tools/ on sys.path rather than the repo root. Both the writer
# (tools/opener_replay.py) and the reader (tools/opener_corpus_report.py) import it from here so
# the value is spelled exactly once and neither depends on the other.
#
# It is deliberately neither "like" (a landed send) nor "dislike"/"never_sent" (a real human or
# AUTO decision not to send), so a synthetic row can never be miscounted as either. A raw query
# can always separate replay rows from real ones with `WHERE decision = 'synthetic_replay'`.
DECISION_REPLAY = "synthetic_replay"
