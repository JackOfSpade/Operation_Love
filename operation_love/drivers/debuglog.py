"""Host-side debug log for a driver (Hinge phone, Bumble browser).

When an auto-mode run misbehaves, this reconstructs what the screen showed and what we did.
Each session writes a per-run folder under the configured debug dir containing `actions.jsonl`
(one record per capture / like / dislike / error) plus optional before/after screenshots, with
a rotating cap so it never fills the disk. Hinge logs before/after shots per action; Bumble
(a watchable browser) logs a text action trail plus a screenshot only on failure. Enabled via
`apps.<app>.debug_log`.

All methods are best-effort and never raise — debug logging must not break a live run.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import deque
from datetime import datetime
from pathlib import Path


class DebugLog:
    def __init__(self, base_dir: str, *, keep_shots: int = 400, run_id: str | None = None):
        stamp = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.dir = Path(base_dir) / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self._log = self.dir / "actions.jsonl"
        # A worker can restart a driver while retaining its run id. actions.jsonl
        # already appends in that case, so continue the screenshot sequence too:
        # restarting at zero would overwrite the first run's evidence while old
        # records still referenced those names.
        self._n = _highest_shot_sequence(self.dir)
        self._shots: deque[tuple[tuple[str, str], Path]] = deque()
        # (label, sha256(frame bytes)) -> filename, for shots currently alive in `_shots`
        # (rotating, normal shots only -- see _save_shot). The LABEL is part of the key on
        # purpose. Deduping on the digest alone also collapses ACROSS action types, and the
        # filename bakes in the label of whichever action wrote the bytes first -- so an
        # `observe_decision` record could end up pointing at `00007_observe_waiting_before.png`.
        # The image is right and nothing is lost, but the folder is meant to be readable by
        # eye, and a decision frame filed under a waiting name reads as a filing bug at exactly
        # the moment someone is trying to reconstruct what happened. Keying on the label keeps
        # every record's screenshot named after its own action while still collapsing the case
        # that actually causes the bloat: the same action repeating on an unchanged screen.
        self._shot_hashes: dict[tuple[str, str], str] = {}
        # Scoped to THIS pool deliberately: rotate=False error shots are kept forever already,
        # so nothing referencing one can ever go stale, and never populating/consulting this
        # map for them keeps that path exactly as simple and unconditional as it was before
        # dedup existed (see _save_shot's `if not rotate` branch, which never touches this dict
        # in either direction).
        self._keep = max(1, int(keep_shots))
        # A restart retains the run directory and JSONL. Rebuild the live normal-shot ring before
        # accepting any more frames, otherwise every restarted DebugLog gets a fresh cap and one
        # long run can grow without bound. The rare keep_before screenshot is named explicitly;
        # error records already identify their permanently kept ``screenshot``. Any malformed or
        # missing record is handled as an ordinary numbered shot, so recovery remains best-effort
        # and never blocks a driver.
        self._restore_shot_state()
        # Observe suggestions can publish from their provider thread while the device thread is
        # waiting for a manual decision. Release-evidence facts share this log, so guard the
        # filename counter, rotating-shot index and JSONL append as one operation.
        self._lock = threading.RLock()

    def _save_shot(self, label: str, frame: bytes | None, *, rotate: bool = True) -> str | None:
        if not frame:
            return None
        key = None
        if rotate:
            # The observe_waiting heartbeat (_note_observe_waiting, hinge.py) hands over the
            # frame its verdict was computed from -- no extra screencap is taken -- but it
            # still arrives here roughly every 15s of human deliberation, and consecutive
            # no_change notices carry the same bytes, because reason="no_change" is precisely
            # the claim that the screen has NOT moved. An audited 15-minute run
            # (data/hinge_debug/run_20260810_203956) wrote ~30MB of screenshots, and every
            # no_change poll during a single 3-minute decision (12 of them) was a byte-for-byte
            # duplicate of the one before it. That churn was also what drove genuinely
            # informative older frames out of the `keep_shots` ring before a developer ever saw
            # them. Hash the bytes instead of holding frames in memory to compare (a long run
            # can have hundreds of ~1MB frames alive across its lifetime) and, if this exact
            # frame is already saved under this same label, point the record at that file
            # instead of writing a second identical copy.
            key = (label, hashlib.sha256(frame).hexdigest())
            existing = self._shot_hashes.get(key)
            if existing is not None:
                return existing
        self._n += 1
        name = f"{self._n:05d}_{label}.png"
        try:
            (self.dir / name).write_bytes(frame)
        except Exception:  # noqa: BLE001 — best-effort; logging must not break the run
            return None
        if not rotate:
            return name                                    # error shots are kept forever (never rotated)
        self._shot_hashes[key] = name
        self._shots.append((key, self.dir / name))
        self._trim_normal_shots()
        return name

    def _restore_shot_state(self) -> None:
        """Rebuild the normal-shot ring from a prior instance of this run.

        The directory is authoritative for liveness: old JSONL records may legitimately point
        at files already removed by rotation. New keep_before records name that exception, while
        error logs retain their established ``screenshot`` convention; any other legacy numbered
        PNG is treated as normal so the cap can still be recovered rather than silently abandoned.
        """
        protected = _protected_shot_names(self._log)
        seen: set[str] = set()
        for path in sorted(self.dir.glob("*.png"), key=_shot_sort_key):
            if path.name in protected or path.name in seen or _shot_sequence(path) is None:
                continue
            seen.add(path.name)
            key = _shot_key(path)
            if key is None:
                # An unreadable file cannot be a valid dedup target, but it still counts toward
                # the cap and can be cleaned up like every other ordinary screenshot.
                key = ("", path.name)
            else:
                self._shot_hashes[key] = path.name
            self._shots.append((key, path))
        self._trim_normal_shots()

    def _trim_normal_shots(self) -> None:
        """Enforce the run-wide cap without touching protected evidence."""
        while len(self._shots) > self._keep:
            old_key, old_path = self._shots.popleft()
            # Drop the hash entry together with the file it names. Without this, the NEXT
            # identical frame would keep resolving to a filename that no longer exists on disk.
            # Old JSONL references are intentionally left as historical records, exactly as
            # they were before restart recovery existed.
            if self._shot_hashes.get(old_key) == old_path.name:
                del self._shot_hashes[old_key]
            try:
                old_path.unlink()
            except Exception:  # noqa: BLE001 — best-effort cleanup must never break logging
                pass

    def _write(self, record: dict) -> None:
        try:
            with self._log.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def action(self, name: str, *, before: bytes | None = None,
               after: bytes | None = None, anchor: bytes | None = None,
               keep_before: bool = False, **fields) -> None:
        """Record an action and its optional frames.

        ``keep_before`` is for a rare recoverable refusal whose raw frame is the evidence needed
        to audit that recovery. Ordinary action screenshots remain in the bounded rotating set.
        """
        with self._lock:
            rec = {"ts": datetime.now().isoformat(timespec="seconds"), "action": name, **fields}
            b = self._save_shot(f"{name}_before", before, rotate=not keep_before)
            a = self._save_shot(f"{name}_after", after)
            anchor_name = self._save_shot(f"{name}_anchor", anchor)
            if b:
                rec["before"] = b
            if a:
                rec["after"] = a
            if anchor_name:
                rec["anchor"] = anchor_name
            if b and keep_before:
                # The only exceptional action-shot policy. Normal before/after/anchor files are
                # the recovery default, so recording each one would bloat a long-lived JSONL.
                rec["kept_before"] = b
            self._write(rec)

    def error(self, name: str, frame: bytes | None, exc: BaseException) -> None:
        with self._lock:
            rec = {"ts": datetime.now().isoformat(timespec="seconds"), "action": name,
                   "error": f"{type(exc).__name__}: {exc}"}
            shot = self._save_shot(f"{name}_error", frame, rotate=False)   # kept forever (never rotated)
            if shot:
                rec["screenshot"] = shot
            self._write(rec)


HingeDebugLog = DebugLog          # backward-compat alias (older imports / tests)


def _highest_shot_sequence(directory: Path) -> int:
    """Return the highest numeric prefix used by a debug screenshot filename.

    Ignore unrelated PNGs and malformed names in a user-managed debug directory;
    only this module's ``00001_label.png`` convention reserves a sequence number.
    """
    highest = 0
    for path in directory.glob("*.png"):
        prefix, separator, _ = path.name.partition("_")
        if separator and prefix.isdecimal():
            highest = max(highest, int(prefix))
    return highest


def _shot_sequence(path: Path) -> int | None:
    """This module's numeric screenshot prefix, or None for unrelated PNGs."""
    prefix, separator, _ = path.name.partition("_")
    return int(prefix) if separator and prefix.isdecimal() else None


def _shot_sort_key(path: Path) -> tuple[int, str]:
    return (_shot_sequence(path) or 0, path.name)


def _shot_key(path: Path) -> tuple[str, str] | None:
    """Dedup key for a persisted normal shot, or None if it cannot be read."""
    try:
        _prefix, _separator, label = path.name.partition("_")
        return (label.removesuffix(".png"), hashlib.sha256(path.read_bytes()).hexdigest())
    except Exception:  # noqa: BLE001 — restart recovery must remain best-effort
        return None


def _protected_shot_names(log_path: Path) -> set[str]:
    """Names that restart recovery must never put in the rotating pool."""
    protected: set[str] = set()
    try:
        lines = log_path.open()
    except Exception:  # noqa: BLE001 — absent/unreadable log is a recoverable empty history
        return protected
    try:
        with lines:
            for line in lines:
                try:
                    record = json.loads(line)
                except Exception:  # noqa: BLE001 — retain usable history around a malformed line
                    continue
                kept_before = record.get("kept_before")
                if isinstance(kept_before, str):
                    protected.add(kept_before)
                # Error records have always used this shape and were never rotating.
                screenshot = record.get("screenshot")
                if isinstance(screenshot, str) and "error" in record:
                    protected.add(screenshot)
    except Exception:  # noqa: BLE001 — a mid-read I/O failure keeps usable prior history
        pass
    return protected
