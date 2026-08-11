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
from collections import deque
from datetime import datetime
from pathlib import Path


class DebugLog:
    def __init__(self, base_dir: str, *, keep_shots: int = 400, run_id: str | None = None):
        stamp = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.dir = Path(base_dir) / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self._log = self.dir / "actions.jsonl"
        self._n = 0
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
        # Scoped to THIS pool deliberately: rotate=False
        # error shots are kept forever already, so nothing referencing one can ever go stale,
        # and never populating/consulting this map for them keeps that path exactly as simple
        # and unconditional as it was before dedup existed (see _save_shot's `if not rotate`
        # branch, which never touches this dict in either direction).
        self._shot_hashes: dict[str, str] = {}
        self._keep = max(1, int(keep_shots))

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
        while len(self._shots) > self._keep:               # rotation: drop the oldest NORMAL shots
            old_key, old_path = self._shots.popleft()
            # Drop the hash entry together with the file it names. Without this, the NEXT
            # identical frame would keep resolving to a filename that no longer exists on disk
            # -- turning "dedup points at a live file" into "dedup points at nothing" the moment
            # rotation runs, rather than only much later. This does not make stale references
            # impossible: a JSONL record written earlier for `old_path`, before it aged out of
            # the ring just now, is left exactly as stale as any pre-dedup rotated-away record
            # has always been -- rotation has never gone back and rewritten old actions.jsonl
            # lines, dedup or not. Clearing the entry here only stops the problem from
            # compounding forward: a repeat of that same screen AFTER this rotation is simply
            # treated as new and saved fresh, same as any other frame that's never been seen.
            if self._shot_hashes.get(old_key) == old_path.name:
                del self._shot_hashes[old_key]
            try:
                old_path.unlink()
            except Exception:  # noqa: BLE001
                pass
        return name

    def _write(self, record: dict) -> None:
        try:
            with self._log.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def action(self, name: str, *, before: bytes | None = None,
               after: bytes | None = None, **fields) -> None:
        rec = {"ts": datetime.now().isoformat(timespec="seconds"), "action": name, **fields}
        b = self._save_shot(f"{name}_before", before)
        a = self._save_shot(f"{name}_after", after)
        if b:
            rec["before"] = b
        if a:
            rec["after"] = a
        self._write(rec)

    def error(self, name: str, frame: bytes | None, exc: BaseException) -> None:
        rec = {"ts": datetime.now().isoformat(timespec="seconds"), "action": name,
               "error": f"{type(exc).__name__}: {exc}"}
        shot = self._save_shot(f"{name}_error", frame, rotate=False)   # kept forever (never rotated)
        if shot:
            rec["screenshot"] = shot
        self._write(rec)


HingeDebugLog = DebugLog          # backward-compat alias (older imports / tests)
