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
import time
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

# How long a COMPLETELY EMPTY run directory must have sat untouched before retention may
# reclaim it (see _trim_old_runs' "second rule").  The directory is created by __init__ before
# the run logs anything, so an empty one that is only seconds old is very likely a logger that
# is alive right now and simply has not written its first action yet -- deleting it would break
# that process's later appends.  An hour is far longer than the gap between a DebugLog being
# constructed and its first record (milliseconds in practice: the driver logs its first capture
# as soon as it has a frame), and far shorter than the multi-day lifetime of the leak this
# bounds, so the exact value is not delicate.  It is deliberately NOT derived from keep_runs:
# this is a staleness threshold about a concurrent writer, not a retention depth.
_EMPTY_RUN_RECLAIM_AGE_S = 3600.0


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
        """Delete whole run directories beyond the newest ``keep_runs``, and reclaim empty ones.

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

        The ``actions.jsonl`` rule has a second, narrower sibling, added 2026-09-16.  It exists
        because that rule leaks: ``__init__`` creates the run directory before the run logs
        anything, so a session that starts and dies (or is stopped) without logging a single
        action leaves a directory that can NEVER become a candidate, because it will never
        contain an ``actions.jsonl``.  Measured on the live tree that day, ``data/hinge_debug``
        held 106 immediate subdirectories: 61 real runs (50 normal -- exactly ``keep_runs`` --
        plus 11 protected, i.e. pruning of real runs works precisely as designed) and 45 holding
        NOTHING AT ALL, 0 bytes each, all named with the ``run_%Y%m%d_%H%M%S`` fallback stamp
        from __init__.  The newest was 2026-09-11, so it is an ongoing leak, not a relic of an
        old naming era.  So retention may ALSO reclaim a directory that is completely empty,
        subject to every narrowing condition above (immediate subdirectory, real directory,
        never the active run, never a protected name) plus one more:

          * its mtime is at least ``_EMPTY_RUN_RECLAIM_AGE_S`` old.

        That age check is the concurrency guard, and it is the reason this rule is not simply
        "empty means junk".  A directory that is empty RIGHT NOW may be one another process
        created moments ago and has not written to yet; deleting it would break that logger's
        later appends.  The active-run check already covers *this* process (including a restart
        that re-enters an old, never-written run id, whose directory is both empty and old), but
        two drivers can share a debug dir, so the age threshold covers the others.

        An empty directory is a safe addition to a deletion rule precisely because it holds no
        evidence BY CONSTRUCTION -- which matters in a module that has already destroyed
        irreplaceable data once (see the ``LOST:`` entries in config.yaml's debug_protect_runs).
        Two further deliberate choices follow from that: the reclaim uses ``rmdir`` and not
        ``rmtree``, so if the directory stopped being empty between the check and the delete the
        operation FAILS instead of destroying whatever just arrived; and empty directories are
        reclaimed OUTSIDE the ``keep_runs`` slicing below, because an empty directory holds no
        diagnostics and must never consume a retention slot that a real run could have used.

        ``keep_runs`` counts *all* normal run directories, including the active one.  Survivors
        are therefore the active run plus the newest ``keep_runs - 1`` prior runs by modification
        time, with every protected run retained in addition.  Everything here is best-effort: a
        failure to prune must never break a run that is otherwise fine.

        The printed message reports THREE disjoint buckets of survivors, and every directory this
        pass looked at falls in exactly one of them (or in none, if it is not a run directory at
        all): normal runs -- which always includes the active run, see ``protected_kept`` below --
        protected runs, and empty run directories this pass left in place.  Each count is derived
        from what the loop below observed and what the deletions below actually achieved, never
        from a re-scan, so a concurrent writer cannot make the sentence describe a directory state
        this pass never had.  Deliberately NOT counted: foreign directories with contents and
        symlinks, because they are not this logger's to report on -- so a fully paranoid operator
        reconciling against ``ls`` should expect the three counts to sum to the subdirectories
        that belong to the debug log, not to every entry in the base directory.
        """
        try:
            current = self.dir.resolve()
            now = time.time()
            candidates = []
            reclaimable_empty = []
            # Empty run directories the age guard declined (see the message comment below): they
            # are neither pruned nor reclaimed, but they ARE still on disk, so a message that
            # omitted them would under-count the survivors an operator can see with ls.
            fresh_empty = 0
            protected_kept = 0
            for entry in base.iterdir():
                if entry.is_symlink() or not entry.is_dir():
                    continue
                # The active-run and protected-name checks now run BEFORE the actions.jsonl
                # check, because the empty-directory rule below needs both of them. Neither
                # reorders a decision: a protected or active directory was never a candidate
                # under the old order either, it simply fell out one test earlier or later.
                if entry.resolve() == current:
                    # Never the run being written right now -- and, because this test runs FIRST,
                    # the active run is counted by the "1 +" below and by nothing else.  That
                    # matters when the active run's own id is ALSO listed in protect_runs (a
                    # legitimate combination: config.yaml protects ids precisely so a future run
                    # reusing one is never pruned).  The normal clause owns it, on purpose: the
                    # active run is retained because it is active, which is true whether or not
                    # anybody named it, and the message's normal clause is the one that says
                    # "including the active run" out loud.  ``protected_kept`` therefore means
                    # "protected runs OTHER than the active one", so the two counts stay disjoint
                    # and a protected active run is reported exactly once.
                    continue
                if entry.name in protect_runs:
                    protected_kept += 1
                    continue          # cited as evidence somewhere outside this directory
                if not (entry / "actions.jsonl").exists():
                    # Not one of ours -- UNLESS it holds nothing at all and has gone stale, in
                    # which case it is one of ours that never got to log (see the docstring).
                    # A foreign folder with contents still falls through here untouched.
                    if _is_reclaimable_empty_run(entry, now):
                        reclaimable_empty.append(entry)
                    elif _is_empty_run_too_fresh_to_reclaim(entry, now):
                        # Reporting only -- this branch never deletes anything.  It exists so the
                        # message can mention a directory the age guard spared, which would
                        # otherwise be in no count at all despite still being on disk.  Asking a
                        # second question re-reads the directory, so in principle the two answers
                        # can straddle a concurrent write; that is harmless, because the second
                        # question is only ever asked when the first said "no" (a directory can
                        # never land in both buckets) and both fail closed toward saying nothing.
                        fresh_empty += 1
                    continue
                candidates.append(entry)
            # ``self.dir`` is an unconditionally retained normal run, so it consumes one of the
            # configured run-directory slots even when it has not logged its first action yet.
            prior_runs_to_keep = keep_runs - 1
            doomed = []
            if len(candidates) > prior_runs_to_keep:
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
        reclaimed = 0
        for entry in reclaimable_empty:
            try:
                # rmdir, NOT rmtree: this is the last line of the concurrency guard. If another
                # process wrote into the directory between the emptiness check above and this
                # call, rmdir raises ENOTEMPTY and we leave the new evidence alone, where an
                # rmtree would have destroyed it without ever noticing.
                entry.rmdir()
                reclaimed += 1
            except Exception:  # noqa: BLE001 — best-effort, per directory
                continue
        if removed or reclaimed:
            # Say it out loud: silent deletion of an operator's diagnostics is exactly the
            # thing they would not think to look for when evidence turns out to be missing.
            #
            # Report the OUTCOME, not the configured cap. Until 2026-09-16 this line said
            # "keeping {keep_runs} run directory(ies)" -- i.e. "keeping 50" -- while 61 real run
            # directories in fact remained (50 normal + 11 protected), plus 45 leaked empty ones
            # it had never even considered. Describing the setting as though it were the result
            # is the same defect commit 64e5d6b6 ("Stop the durable records from asserting what
            # they never knew") removed from the durable records. The counts below are derived
            # from what this pass actually saw and actually deleted -- ``candidates`` are the
            # normal prior runs it enumerated, ``removed`` the ones whose rmtree really
            # succeeded, +1 for the active run -- rather than from a re-scan, so a concurrent
            # writer cannot make the sentence describe a directory state this pass never had.
            did = []
            if removed:
                did.append(f"pruned {removed} run directory(ies)")
            if reclaimed:
                did.append(f"reclaimed {reclaimed} empty run directory(ies)")
            normal_remaining = 1 + len(candidates) - removed
            protected_note = (f", plus {protected_kept} protected run directory(ies) retained "
                              f"regardless of age" if protected_kept else "")
            # The third clause, added later the same day after review. ``normal_remaining``
            # counts only the active run plus directories that HAVE an actions.jsonl, and
            # ``protected_kept`` only names from config -- so an EMPTY directory the age guard
            # declined to reclaim was in no clause at all, while still sitting on disk for the
            # operator to find. That is the same class of small untruth as reporting the cap,
            # one notch down, in a sentence whose whole point is to report the OUTCOME.
            #
            # Of the two fixes offered, this counts the stragglers rather than narrowing the
            # sentence to "run directory(ies) WITH RECORDED ACTIONS remain". The narrowed
            # wording would have been true but unusable: an operator verifies a claim like this
            # by listing the directory, and no ls tells you which subdirectories contain an
            # actions.jsonl. Counting keeps the sentence reconcilable against what they can see.
            #
            # Two things land here, and both are directories THIS pass observed and left behind:
            # the empty ones spared by the _EMPTY_RUN_RECLAIM_AGE_S concurrency guard, and the
            # ones it did mean to reclaim where the rmdir failed (a late write that made the
            # directory non-empty between check and delete, or a permission error). The second
            # is rare, but excluding it would re-open the hole this clause exists to close.
            empty_kept = fresh_empty + (len(reclaimable_empty) - reclaimed)
            empty_note = (f", plus {empty_kept} empty run directory(ies) left in place"
                          if empty_kept else "")
            print(f"Debug log: {' and '.join(did)} from {base}; "
                  f"{normal_remaining} run directory(ies) remain, including the active run"
                  f"{protected_note}{empty_note}.")

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
            line = json.dumps(record) + "\n"
        except Exception:  # noqa: BLE001 — an unserialisable field must not break the run
            # Serialisation is pulled out of the append's try so the retry below can reuse the
            # line, so it needs a guard of its own -- this is where an unserialisable **fields
            # value dies, and it must still not reach the caller. Not hypothetical: the
            # `NAV_CHAIN_BROKEN` path in hinge.py (~11045) documents raw PNG bytes reaching
            # `action(**fields)` and names json.dumps here as what would swallow the WHOLE
            # record; that call site routes the frame through keep_before instead.
            return
        try:
            append_private_text(self._log, line, parent=self.dir)
            return
        except Exception:  # noqa: BLE001
            pass
        # Self-heal once, then give up quietly (2026-09-16).
        #
        # The empty-run reclaim above deletes a run directory whose logger has been silent past
        # _EMPTY_RUN_RECLAIM_AGE_S. Its docstring argues that window never opens because "the
        # driver logs its first capture as soon as it has a frame" -- true of the Hinge driver,
        # but NOT of a driver that is constructed and then blocked: a paywalled deck, a wedged
        # app, a run waiting on a human. Such a logger writes nothing for an hour, another
        # process reclaims its (genuinely empty, genuinely lossless) directory, and from then on
        # every append lands in a directory that no longer exists. Because this method is
        # best-effort by contract, the victim would lose every SUBSEQUENT record in silence --
        # no exception, no message, just a run with no debug trail.
        #
        # The fix is here rather than in the deletion rule on purpose. Widening the rule (never
        # reclaim) restores the 45-directory leak; narrowing it (longer age) only moves the
        # window. Nothing was lost when the directory went -- it was empty by construction -- so
        # only FUTURE evidence is at risk, and the writer is the one component that knows it is
        # still alive. ensure_private_dir is the module's existing 0700 leaf-directory helper
        # (parents included, symlink at the leaf refused), so recovery cannot silently re-create
        # the run directory with looser permissions than the original, and the retry goes back
        # through append_private_text, so every symlink/hard-link check still applies.
        #
        # One attempt only: if the retry fails too, the cause is not a missing directory and
        # retrying harder inside a live run buys nothing. Screenshots are not retried here
        # because they do not need to be -- _save_shot runs before _write in action(), so the
        # frames of the record that heals the directory are lost, but every later action's
        # frames write normally into the restored directory.
        try:
            ensure_private_dir(self.dir)
            append_private_text(self._log, line, parent=self.dir)
        except Exception:  # noqa: BLE001 — a second failure must still never reach the caller
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


def _empty_run_age_s(entry: Path, now: float) -> float | None:
    """Seconds since ``entry`` was touched, or None when it is not an empty directory at all.

    Retention asks two questions about a directory with no ``actions.jsonl``: may it be
    reclaimed, and -- if not -- is it nevertheless an empty run directory this pass is leaving
    behind and therefore owes the operator a mention. Both turn on the SAME notion of "empty",
    so that notion is defined once, here. Two regexes for one permitted opener move disagreeing
    about their noun sets (2026-09-14) is the local precedent for why the shared half of a pair
    of predicates gets hoisted instead of written out twice.

    Fails CLOSED on anything unexpected: a directory we cannot list or stat is not proof of
    emptiness, so it is neither a deletion candidate nor something we claim to have seen.
    """
    try:
        if any(entry.iterdir()):
            return None
        return now - entry.stat().st_mtime
    except Exception:  # noqa: BLE001 — unreadable is never a licence to delete, or to claim
        return None


def _is_reclaimable_empty_run(entry: Path, now: float) -> bool:
    """True when ``entry`` holds nothing at all AND has sat untouched long enough to be dead.

    Both halves are load-bearing (see DebugLog._trim_old_runs): emptiness is what makes deleting
    it lossless, and age is what makes it safe against a logger that was constructed seconds ago
    and has not written its first record yet.

    Fails CLOSED on anything unexpected (via ``_empty_run_age_s``). A directory we cannot list
    or stat is not proof of emptiness, so it is simply not a deletion candidate -- the same
    posture every other narrowing condition in the candidate rule takes.
    """
    age_s = _empty_run_age_s(entry, now)
    return age_s is not None and age_s >= _EMPTY_RUN_RECLAIM_AGE_S


def _is_empty_run_too_fresh_to_reclaim(entry: Path, now: float) -> bool:
    """True for an empty run directory the age guard spares. REPORTING ONLY -- never deletes.

    The exact complement of ``_is_reclaimable_empty_run`` over the empty-directory domain, which
    is why both are built on ``_empty_run_age_s``: if the two ever disagreed about what "empty"
    means, a directory could be counted in both buckets of the retention message or in neither.
    """
    age_s = _empty_run_age_s(entry, now)
    return age_s is not None and age_s < _EMPTY_RUN_RECLAIM_AGE_S


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
