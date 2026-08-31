"""Host-side debug log shared by device and browser drivers.

When an auto-mode run misbehaves, this reconstructs what the screen showed and what we did.
Each session writes a per-run folder under the configured debug dir containing `actions.jsonl`
(one record per capture / like / dislike / error) plus optional before/after screenshots, with
bounded normal and retained-evidence pools so it cannot grow without limit. Drivers may log
before/after shots per action or
a text trail plus failure screenshots. Enabled via
`apps.<app>.debug_log`.

All methods are best-effort and never raise — debug logging must not break a live run.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

from ..private_files import (
    append_private_text,
    ensure_private_dir,
    tighten_private_file,
    write_private_bytes,
)


_MAX_KEEP_SHOTS = 2_000
_MAX_RETAINED_SHOTS = 100


class DebugLog:
    def __init__(self, base_dir: str, *, keep_shots: int = 400, run_id: str | None = None,
                 keep_runs: int | None = None, protect_runs=()):
        if type(keep_shots) is not int or not 1 <= keep_shots <= _MAX_KEEP_SHOTS:
            raise ValueError(
                f"keep_shots must be an integer from 1 to {_MAX_KEEP_SHOTS} "
                f"(got {keep_shots!r})")
        if keep_runs is not None and (type(keep_runs) is not int or keep_runs < 1):
            raise ValueError(f"keep_runs must be a positive integer or None (got {keep_runs!r})")
        if run_id is not None and (
                not isinstance(run_id, str) or not run_id or run_id in {".", ".."}
                or "/" in run_id or "\\" in run_id or Path(run_id).name != run_id):
            raise ValueError("debug run_id must be one non-dot path component")
        stamp = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        base_path = Path(base_dir)
        if base_path.name in {"", ".", ".."}:
            raise ValueError("debug base_dir must name a dedicated leaf directory")
        base = ensure_private_dir(base_path)
        self.dir = ensure_private_dir(base / stamp)
        self._log = self.dir / "actions.jsonl"
        # A restarted run can inherit files created under an older/default umask. Tighten the
        # fixed log entry before reading it; a symlink is unsafe and disables this optional
        # logger through open_debug_log's best-effort construction guard.
        tighten_private_file(self._log, parent=self.dir, missing_ok=True)
        # A worker can restart a driver while retaining its run id. actions.jsonl
        # already appends in that case, so continue the screenshot sequence too:
        # restarting at zero would overwrite the first run's evidence while old
        # records still referenced those names.
        if keep_runs is not None:
            self._trim_old_runs(base, keep_runs, frozenset(protect_runs or ()))
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
        # Retained error/recovery evidence has its own bounded pool and is never a normal-shot
        # dedup source. This preserves recent incident evidence without allowing repeated
        # failures to grow a run forever.
        self._keep = keep_shots
        self._retained_shots: deque[Path] = deque()
        # A restart retains the run directory and JSONL. Rebuild the live normal-shot ring before
        # accepting any more frames, otherwise every restarted DebugLog gets a fresh cap and one
        # long run can grow without bound. The rare keep_before screenshot is named explicitly;
        # error records already identify their retained ``screenshot``. Any malformed or
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
            write_private_bytes(self.dir / name, frame, parent=self.dir)
        except Exception:  # noqa: BLE001 — best-effort; logging must not break the run
            return None
        if not rotate:
            self._retained_shots.append(self.dir / name)
            self._trim_retained_shots()
            return name
        self._shot_hashes[key] = name
        self._shots.append((key, self.dir / name))
        self._trim_normal_shots()
        return name

    def _restore_shot_state(self) -> None:
        """Rebuild the normal-shot ring from a prior instance of this run.

        The directory is authoritative for liveness: old JSONL records may legitimately point
        at files already removed by rotation. New keep_before records name that exception, while
        error logs retain their established ``screenshot`` convention in a separate bounded
        retained-evidence pool; any other legacy numbered PNG is treated as normal so the cap
        can still be recovered rather than silently abandoned.
        """
        retained_names = _retained_shot_names(self._log)
        retained_set = set(retained_names)
        retained_seen: set[str] = set()
        for name in retained_names:
            if name in retained_seen:
                continue
            retained_seen.add(name)
            path = self.dir / name
            if _shot_sequence(path) is None:
                continue
            try:
                tighten_private_file(path, parent=self.dir)
            except Exception:  # noqa: BLE001 — restart recovery remains best-effort
                continue
            self._retained_shots.append(path)
        self._trim_retained_shots()
        seen: set[str] = set()
        for path in sorted(self.dir.glob("*.png"), key=_shot_sort_key):
            if _shot_sequence(path) is None:
                continue
            try:
                # Numeric PNGs are this logger's managed namespace. Never read/chmod through a
                # planted symlink; a bad entry is simply not a restart dedup/rotation candidate.
                tighten_private_file(path, parent=self.dir)
            except Exception:  # noqa: BLE001 — restart recovery remains best-effort
                continue
            if path.name in retained_set or path.name in seen:
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

    def _trim_old_runs(self, base: Path, keep_runs: int,
                       protect_runs: frozenset[str] = frozenset()) -> None:
        """Delete whole run directories beyond the newest ``keep_runs``.

        The per-run caps above bound ONE run's screenshots; nothing bounded the number of runs,
        so ``data/hinge_debug`` reached 12GB across 274 runs by 2026-08-28. This is the missing
        half of that policy.

        Deletion is irreversible, so the candidate rule is deliberately narrow. A directory is
        only ever a candidate when ALL of these hold:

          * it is an immediate subdirectory of this logger's own base directory;
          * it is a real directory, not a symlink (never follow one out of the tree);
          * it contains an ``actions.jsonl``, i.e. it is demonstrably a run THIS logger wrote --
            an unrelated folder someone parked in here is never touched;
          * it is not the run currently being written;
          * it is not NAMED IN ``protect_runs``.

        That last rule is not decoration.  Run directories are cited from outside themselves --
        ``config.yaml``'s ``observe_release_evidence.production_run_reference`` is a literal
        ``data/hinge_debug/<id>`` path, ``ops/release/<id>/`` holds the signed artifacts for the
        run that gated a release, and tests cite run ids as the provenance of their fixtures.
        Age is a terrible proxy for value there: the release-evidence run is by definition an old
        one.  Deleting it would break the evidence chain behind a shipped gate, so the ids are
        listed explicitly in config rather than inferred.

        ``keep_runs`` counts *all* normal run directories, including the active one.  Survivors
        are therefore the active run plus the newest ``keep_runs - 1`` prior runs by modification
        time, with every protected run retained in addition.  Everything here is best-effort: a
        failure to prune must never break a run that is otherwise fine.
        """
        try:
            current = self.dir.resolve()
            candidates = []
            for entry in base.iterdir():
                if entry.is_symlink() or not entry.is_dir():
                    continue
                if not (entry / "actions.jsonl").exists():
                    continue          # not one of ours; leave it alone
                if entry.resolve() == current:
                    continue          # never the run being written right now
                if entry.name in protect_runs:
                    continue          # cited as evidence somewhere outside this directory
                candidates.append(entry)
            # ``self.dir`` is an unconditionally retained normal run, so it consumes one of the
            # configured run-directory slots even when it has not logged its first action yet.
            prior_runs_to_keep = keep_runs - 1
            if len(candidates) <= prior_runs_to_keep:
                return
            candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            doomed = candidates[prior_runs_to_keep:]
        except Exception:  # noqa: BLE001 — retention must never break logging
            return
        removed = 0
        for entry in doomed:
            try:
                shutil.rmtree(entry)
                removed += 1
            except Exception:  # noqa: BLE001 — best-effort, per directory
                continue
        if removed:
            # Say it out loud: silent deletion of an operator's diagnostics is exactly the
            # thing they would not think to look for when evidence turns out to be missing.
            print(f"Debug log: pruned {removed} run directory(ies) from {base}, "
                  f"keeping {keep_runs} run directory(ies), including the active run.")

    def _trim_retained_shots(self) -> None:
        """Keep only the newest bounded error/recovery screenshots across restarts."""
        while len(self._retained_shots) > _MAX_RETAINED_SHOTS:
            old_path = self._retained_shots.popleft()
            try:
                old_path.unlink()
            except Exception:  # noqa: BLE001 — evidence rotation is best-effort
                pass

    def _trim_normal_shots(self) -> None:
        """Enforce the run-wide normal-shot cap without touching retained evidence."""
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
            append_private_text(
                self._log, json.dumps(record) + "\n", parent=self.dir)
        except Exception:  # noqa: BLE001
            pass

    def action(self, name: str, *, before: bytes | None = None,
               after: bytes | None = None, anchor: bytes | None = None,
               keep_before: bool = False, keep_after: bool = False, **fields) -> None:
        """Record an action and its optional frames.

        ``keep_before`` is for a rare recoverable refusal whose raw frame is the evidence needed
        to audit that recovery. Ordinary action screenshots remain in the bounded rotating set.
        """
        with self._lock:
            # Reserved audit identity wins even if a direct caller passes colliding **fields.
            rec = {**fields, "ts": datetime.now().isoformat(timespec="seconds"), "action": name}
            b = self._save_shot(f"{name}_before", before, rotate=not keep_before)
            a = self._save_shot(f"{name}_after", after, rotate=not keep_after)
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
            if a and keep_after:
                rec["kept_after"] = a
            self._write(rec)

    def error(self, name: str, frame: bytes | None, exc: BaseException) -> None:
        with self._lock:
            rec = {"ts": datetime.now().isoformat(timespec="seconds"), "action": name,
                   "error": f"{type(exc).__name__}: {exc}"}
            shot = self._save_shot(f"{name}_error", frame, rotate=False)
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


def _retained_shot_names(log_path: Path) -> list[str]:
    """Retained screenshot names in JSONL chronology for restart cap enforcement."""
    retained: list[str] = []
    try:
        lines = log_path.open()
    except Exception:  # noqa: BLE001 — absent/unreadable log is a recoverable empty history
        return retained
    try:
        with lines:
            for line in lines:
                try:
                    record = json.loads(line)
                except Exception:  # noqa: BLE001 — retain usable history around a malformed line
                    continue
                kept_before = record.get("kept_before")
                if isinstance(kept_before, str):
                    retained.append(kept_before)
                kept_after = record.get("kept_after")
                if isinstance(kept_after, str):
                    retained.append(kept_after)
                # Error records use this established shape.
                screenshot = record.get("screenshot")
                if isinstance(screenshot, str) and "error" in record:
                    retained.append(screenshot)
    except Exception:  # noqa: BLE001 — a mid-read I/O failure keeps usable prior history
        pass
    return retained
