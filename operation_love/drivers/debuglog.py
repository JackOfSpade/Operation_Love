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
        self._shots: deque[Path] = deque()
        self._keep = max(1, int(keep_shots))

    def _save_shot(self, label: str, frame: bytes | None, *, rotate: bool = True) -> str | None:
        if not frame:
            return None
        self._n += 1
        name = f"{self._n:05d}_{label}.png"
        try:
            (self.dir / name).write_bytes(frame)
        except Exception:  # noqa: BLE001 — best-effort; logging must not break the run
            return None
        if not rotate:
            return name                                    # error shots are kept forever (never rotated)
        self._shots.append(self.dir / name)
        while len(self._shots) > self._keep:               # rotation: drop the oldest NORMAL shots
            old = self._shots.popleft()
            try:
                old.unlink()
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
