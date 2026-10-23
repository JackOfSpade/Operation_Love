"""Offline verifier for the Hinge supervised-Training AUTO release gate.

This tool never constructs ADB, a driver, Worker, or opener client.  It verifies local evidence
from an already-completed supervised Training run on the EXACT calibrated build (no
``version_compatibility_review`` active) and emits the ``apps.hinge.training_release_evidence``
mapping that config validation accepts as Hinge AUTO's release gate.  It is a third, separately
named path: the production-OBSERVE verifiers (``tools.hinge_observe_release`` and
``tools.hinge_observe_ai_release``) keep their meaning and are not reused here.

What the evidence proves: within one profile capture each, a human-approved Like and a
human-approved Pass went through the Hub checkpoint, landed on a re-verified screen, and were
persisted as manual store rows, on the calibrated build/device/frame.  What it does not prove is
stated in the artifact as ``not_exercised`` rather than claimed: AUTO's keyboard-up pre-send
verification, AUTO's deck pass, and AUTO's like landing without Training's successor proof.

Control facts come from the run's own debug rows and retained frames (evidence ids are
recomputed from the kept PNGs); persistence comes from one aggregate, read-only store query.
The artifact holds digests only: no opener text, profile ids, names or tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from operation_love import config as cfg_mod
from operation_love.private_files import atomic_write_private_bytes, ensure_private_dir
from operation_love.ranker import make_store
from tools._debug_frames import debug_frame_path

ARTIFACT_NAME = "hinge_training_release.json"
KIND = "hinge_supervised_training_release"
NOT_EXERCISED = [
    "auto_keyboard_up_presend_verification",
    "auto_deck_pass",
    "auto_like_landing_without_training_successor_proof",
]
_ARTIFACT_KEYS = {
    "schema_version",
    "kind",
    "completed",
    "human_ground_truth",
    "source",
    "calibration_calibrated_at",
    "calibration_sha256",
    "device",
    "hinge_version_name",
    "frame_size_px",
    "production_run_reference",
    "production_run_id",
    "debug_actions_sha256",
    "chain_evidence_sha256",
    "store_persistence_evidence_sha256",
    "not_exercised",
    "verified_at",
}
_PASTE_KEYS = {
    "schema_version",
    "calibration_calibrated_at",
    "calibration_sha256",
    "device",
    "hinge_version_name",
    "frame_size_px",
    "production_run_reference",
    "production_run_id",
    "debug_actions_sha256",
    "verification_file",
    "verification_sha256",
    "verified_at",
}
_STORE_KEYS = (
    "like_labels",
    "like_decisions",
    "like_openers",
    "like_send_evidence",
    "pass_labels",
    "pass_decisions",
)
_ENVIRONMENT_KEYS = {
    "ts",
    "action",
    "session_mode",
    "hinge_version_name",
    "frame_size_px",
    "device",
    "calibration_sha256",
    "compatible_version",
    "targeting_available",
}
# The driver's training_release_fact whitelist, plus DebugLog's own ts/action.
_FACT_KEYS = {
    "training_release_checkpoint_published": {
        "ts",
        "action",
        "evidence_id",
        "profile_token_sha256",
        "approval_token_sha256",
    },
    "training_release_decision_claimed": {
        "ts",
        "action",
        "command",
        "evidence_id",
        "idempotency_token_sha256",
        "waited_s",
    },
    "training_release_action_persisted": {
        "ts",
        "action",
        "decision",
        "evidence_id",
        "send_evidence_id",
        "profile_id",
    },
}
_ADVANCE_PROOFS = {"identity", "name", "same_name_successor"}
# Rows after which the current profile's control facts can no longer license a decision.
_PROFILE_RESET_ACTIONS = {
    "capture_aborted",
    "training_cancelled",
    "training_dislike_unverified",
    "like_rejected",
    "training_session_environment",
}
_SHARED_PREFIX = (
    "like_anchor",
    "auto_opener_pre_send",
    "training_release_checkpoint_published",
    "training_release_decision_claimed",
)
_LIKE_ORDER = _SHARED_PREFIX + (
    "auto_opener_resumed_send",
    "like_attempt",
    "training_advance_probe",
    "like",
    "training_release_action_persisted",
)
_PASS_ORDER = _SHARED_PREFIX + (
    "training_advance_probe",
    "training_dislike",
    "training_release_action_persisted",
)
_STEP_ACTIONS = set(_LIKE_ORDER) | set(_PASS_ORDER)
# A terminal row that does not fit ends the profile; any other misfit only voids the sheet.
_TERMINAL_ACTIONS = {"like", "training_dislike", "training_release_action_persisted"}


class ReleaseEvidenceRefused(RuntimeError):
    """The supplied local evidence cannot safely unlock Hinge AUTO."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value) -> str:
    return _sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    )


_debug_frame_path = debug_frame_path


def _is_int(value) -> bool:
    return type(value) is int


def _same_int(a, b) -> bool:
    return _is_int(a) and _is_int(b) and a == b


def _is_hex64(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _nonempty_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _read_rows(run_dir: Path) -> tuple[bytes, list[dict]]:
    actions = run_dir / "actions.jsonl"
    try:
        raw = actions.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseEvidenceRefused(
            f"could not read Training debug actions {actions}: {type(exc).__name__}"
        ) from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ReleaseEvidenceRefused("Training debug actions are empty or malformed")
    return raw, rows


def _verify_run_binding(rows: list[dict], run_dir: Path, run_id: str) -> None:
    if run_dir.name != run_id:
        raise ReleaseEvidenceRefused(
            "debug-run directory name must exactly equal --run-id"
        )
    binding = rows[0]
    if (
        set(binding) != {"ts", "action", "run_id", "app"}
        or binding.get("action") != "observe_release_run_binding"
        or binding.get("run_id") != run_id
        or binding.get("app") != "hinge"
        or not _nonempty_text(binding.get("ts"))
    ):
        raise ReleaseEvidenceRefused(
            "debug run lacks the required first production Worker run binding; it may be a "
            "legacy timestamp-named or unrelated actions.jsonl"
        )


def _environment_refusal(row: dict, calibration: dict) -> str | None:
    """Why one ``training_session_environment`` row does not bind the calibration, or None."""
    if set(row) != _ENVIRONMENT_KEYS:
        return "it does not carry the exact environment fields"
    if row.get("session_mode") != "training":
        return f"session_mode is {row.get('session_mode')!r}, not 'training'"
    for key in ("hinge_version_name", "frame_size_px", "device"):
        if row.get(key) != calibration.get(key):
            return (
                f"live {key} {row.get(key)!r} != calibration {calibration.get(key)!r}"
            )
    if row.get("calibration_sha256") != _canonical_sha256(calibration):
        return (
            "the session's calibration digest does not match the configured calibration"
        )
    if row.get("compatible_version") is not None:
        return "a version-compatibility review was active; only exact-build calibration counts"
    if row.get("targeting_available") is not True:
        return "targeting was not available for the session"
    return None


def _verify_environment(rows: list[dict], calibration: dict) -> None:
    if len(rows) < 2 or rows[1].get("action") != "training_session_environment":
        raise ReleaseEvidenceRefused(
            "debug run lacks the training_session_environment row directly after its run "
            "binding, so its live build, frame, device and calibration are unknown"
        )
    for index, row in enumerate(rows):
        if row.get("action") != "training_session_environment":
            continue
        reason = _environment_refusal(row, calibration)
        if reason is not None:
            raise ReleaseEvidenceRefused(
                f"debug row {index} training_session_environment does not bind the "
                f"calibration: {reason}"
            )


class _ChainScanner:
    """Find the first complete Training LIKE chain and PASS chain, each inside one capture.

    A ``capture`` row opens a profile.  The post-heart ``verify_sheet_identity`` +
    ``verify_sheet_item`` pair licenses one ``like_anchor``; every later step must then arrive
    in order and agree with the anchor, the approved pre-send snapshot and each other.  A step
    that is out of order or fails its checks voids the sheet (a new anchor may restart it), a
    terminal row that does not fit ends the profile, and a reset row ends it outright.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.like: dict | None = None
        self.pass_: dict | None = None
        # In-order steps refused because a frame they cite is gone; the retained pool rotates.
        self.missing_frame_drops = 0
        self._frame_missing = False
        self._end_profile()

    # --- state ------------------------------------------------------------------------
    def _end_profile(self) -> None:
        self.capture: tuple[int, dict] | None = None
        self._void_sheet()

    def _void_sheet(self) -> None:
        self.identity: tuple[int, dict] | None = None
        self.item: tuple[int, dict] | None = None
        # The verify pair that licensed the current anchor, then the anchor and later steps.
        self.pair: list[tuple[int, dict]] = []
        self.steps: list[tuple[int, dict]] = []

    def _row(self, action: str) -> dict:
        return next(row for _index, row in self.steps if row["action"] == action)

    def _order(self) -> tuple[str, ...]:
        if len(self.steps) >= len(_SHARED_PREFIX):
            command = self._row("training_release_decision_claimed")["command"]
            return _LIKE_ORDER if command == "like" else _PASS_ORDER
        return _SHARED_PREFIX

    def _frame(self, name) -> bytes | None:
        path = _debug_frame_path(self.run_dir, name)
        try:
            frame = None if path is None else path.read_bytes()
        except OSError:
            frame = None
        self._frame_missing = self._frame_missing or frame is None
        return frame

    # --- feed -------------------------------------------------------------------------
    def feed(self, index: int, row: dict) -> None:
        action = row.get("action")
        if action == "capture":
            self.capture = (index, row)
            self._void_sheet()
            return
        if action in _PROFILE_RESET_ACTIONS or row.get("error") is not None:
            self._end_profile()
            return
        if self.capture is None:
            return
        if action == "verify_sheet_identity":
            ok = row.get("outcome") == "identity_match" and _is_int(row.get("item"))
            self.identity = (index, row) if ok else None
            self.item = None
            return
        if action == "verify_sheet_item":
            ok = (
                self.identity is not None
                and row.get("outcome") == "verify_match"
                and _same_int(row.get("item"), self.identity[1].get("item"))
            )
            self.item = (index, row) if ok else None
            return
        if action == "like_anchor":
            self._anchor(index, row)
            return
        if action not in _STEP_ACTIONS:
            return
        if action == "training_advance_probe" and row.get("outcome") != "accepted":
            return  # a retry poll is diagnostics; only the accepted probe is a step
        order = self._order()
        self._frame_missing = False
        fits = (
            bool(self.steps)
            and len(self.steps) < len(order)
            and order[len(self.steps)] == action
            and self._check(action, row)
        )
        if not fits:
            if self._frame_missing:
                self.missing_frame_drops += 1
            if action in _TERMINAL_ACTIONS:
                self._end_profile()
            else:
                self._void_sheet()
            return
        self.steps.append((index, row))
        # Re-read the order: the claimed command chooses the LIKE or PASS continuation.
        if len(self.steps) == len(self._order()):
            self._complete()

    def _anchor(self, index: int, row: dict) -> None:
        # The anchor's own frame is not required: the row carries no frame digest and the frame
        # sits in the rotating ring.  The sheet is proven by the verify pair plus the
        # digest-bound pre-send kept frame.
        item = row.get("model_item_index")
        ok = (
            row.get("verified") is True
            and _is_int(item)
            and self.identity is not None
            and self.item is not None
            and _same_int(self.item[1].get("item"), item)
        )
        pair = [self.identity, self.item]
        self._void_sheet()
        if ok:
            self.pair = pair
            self.steps = [(index, row)]

    def _complete(self) -> None:
        events = [self.capture, *self.pair, *self.steps]
        persisted = self.steps[-1][1]
        chain = {
            "events": [
                {
                    "action": row["action"],
                    "index": index,
                    "row_sha256": _canonical_sha256(row),
                }
                for index, row in events
            ],
            "profile_id": persisted["profile_id"],
            "send_evidence_id": persisted["send_evidence_id"],
        }
        if self._row("training_release_decision_claimed")["command"] == "like":
            self.like = self.like or chain
        else:
            self.pass_ = self.pass_ or chain
        self._end_profile()

    # --- per-step checks ----------------------------------------------------------------
    def _check(self, action: str, row: dict) -> bool:
        anchor = self.steps[0][1]
        item = anchor["model_item_index"]
        if action == "auto_opener_pre_send":
            return self._check_pre_send(row, item)
        pre = self._row("auto_opener_pre_send")
        pre_id = pre["evidence_id"]
        if action in _FACT_KEYS and set(row) != _FACT_KEYS[action]:
            return False
        if action == "training_release_checkpoint_published":
            return (
                row.get("evidence_id") == pre_id
                and _is_hex64(row.get("profile_token_sha256"))
                and _is_hex64(row.get("approval_token_sha256"))
            )
        if action == "training_release_decision_claimed":
            waited = row.get("waited_s")
            return (
                row.get("command") in {"like", "dislike"}
                and row.get("evidence_id") == pre_id
                and _is_hex64(row.get("idempotency_token_sha256"))
                and isinstance(waited, (int, float))
                and not isinstance(waited, bool)
                and math.isfinite(waited)
                and waited >= 0
            )
        if action == "auto_opener_resumed_send":
            return self._check_resumed(row, item, pre)
        if action == "training_advance_probe":
            return (
                row.get("outcome") == "accepted"
                and row.get("stable") is True
                and row.get("names_agree") is True
                and row.get("kept_before") != row.get("kept_after")
                and self._frame(row.get("kept_before")) is not None
                and self._frame(row.get("kept_after")) is not None
            )
        if action in {"like_attempt", "like"}:
            resumed = self._row("auto_opener_resumed_send")
            return (
                row.get("verified") is True
                and _same_int(row.get("model_item_index"), item)
                and _same_int(row.get("opener_chars"), pre["opener_chars"])
                and row.get("pre_send_evidence_id") == pre_id
                and row.get("resumed_send_evidence_id") == resumed["evidence_id"]
            )
        if action == "training_dislike":
            return (
                row.get("verified") is True
                and _same_int(row.get("model_item_index"), item)
                and row.get("advance_proof") in _ADVANCE_PROOFS
                and row.get("pre_send_evidence_id") == pre_id
            )
        if action == "training_release_action_persisted":
            command = self._row("training_release_decision_claimed")["command"]
            send_id = (
                self._row("auto_opener_resumed_send")["evidence_id"]
                if command == "like"
                else None
            )
            return (
                row.get("decision") == command
                and row.get("evidence_id") == pre_id
                and row.get("send_evidence_id") == send_id
                and _nonempty_text(row.get("profile_id"))
            )
        raise AssertionError(f"unhandled Training release step {action}")

    def _check_pre_send(self, row: dict, item: int) -> bool:
        opener = row.get("opener")
        if (
            row.get("session_mode") != "training"
            or not _same_int(row.get("model_item_index"), item)
            or not isinstance(opener, str)
            or not opener
            or not _same_int(row.get("opener_chars"), len(opener))
        ):
            return False
        opener_bytes = opener.encode("utf-8")
        frame = self._frame(row.get("kept_before"))
        return (
            frame is not None
            and row.get("opener_sha256") == _sha256(opener_bytes)
            and row.get("frame_sha256") == _sha256(frame)
            and row.get("evidence_id") == _sha256(frame + b"\0" + opener_bytes)
        )

    def _check_resumed(self, row: dict, item: int, pre: dict) -> bool:
        frame = self._frame(row.get("kept_before"))
        return (
            frame is not None
            and row.get("session_mode") == "training"
            and _same_int(row.get("model_item_index"), item)
            and row.get("approval_evidence_id") == pre["evidence_id"]
            and row.get("opener_sha256") == pre["opener_sha256"]
            and row.get("frame_sha256") == _sha256(frame)
            # The resumed frame is bound to the APPROVED opener, not to the row's own copy.
            and row.get("evidence_id")
            == _sha256(frame + b"\0" + pre["opener"].encode("utf-8"))
        )


def _missing_frame_note(drops: int) -> str:
    if not drops:
        return ""
    return (
        f"; {drops} otherwise in-order chain step(s) were dropped because a kept frame they "
        "cite is missing or unreadable (the driver's retained frame pool rotates, so a Like "
        "or Pass far back in a long session loses its frames)"
    )


def _verify_debug_run(run_dir: Path, run_id: str, calibration: dict) -> dict:
    """Return the debug digest and both verified chains, or refuse."""
    raw, rows = _read_rows(run_dir)
    _verify_run_binding(rows, run_dir, run_id)
    _verify_environment(rows, calibration)
    scanner = _ChainScanner(run_dir)
    for index, row in enumerate(rows):
        scanner.feed(index, row)
    missing = _missing_frame_note(scanner.missing_frame_drops)
    if scanner.like is None:
        raise ReleaseEvidenceRefused(
            "debug run lacks a complete Training LIKE chain inside one capture (verified "
            "sheet, anchor, approved pre-send snapshot, Hub checkpoint publication and claimed "
            "like, re-verified resumed send, like attempt, accepted successor proof with both "
            f"frames, verified like, persisted action){missing}"
        )
    if scanner.pass_ is None:
        raise ReleaseEvidenceRefused(
            "debug run lacks a complete Training PASS chain inside one capture (verified "
            "sheet, anchor, approved pre-send snapshot, Hub checkpoint publication and claimed "
            "dislike, accepted successor proof with both frames, verified training_dislike, "
            f"persisted action){missing}"
        )
    if scanner.like["profile_id"] == scanner.pass_["profile_id"]:
        raise ReleaseEvidenceRefused(
            "the persisted Like and Pass name the same profile id; one card cannot be both"
        )
    return {
        "debug_actions_sha256": _sha256(raw),
        "like": scanner.like,
        "pass": scanner.pass_,
    }


def _verify_store_persistence(
    store,
    run_id: str,
    *,
    like_profile_id: str,
    pass_profile_id: str,
    like_send_evidence_id: str,
) -> str:
    """Read aggregate persistence proof through the active store, never raw profile data."""
    try:
        summary = store.training_release_persistence_summary(
            run_id,
            "hinge",
            like_profile_id=like_profile_id,
            pass_profile_id=pass_profile_id,
            like_send_evidence_id=like_send_evidence_id,
        )
    except Exception as exc:  # noqa: BLE001 -- store backend errors must fail the release closed
        raise ReleaseEvidenceRefused(
            f"could not query active store's read-only release summary: {type(exc).__name__}"
        ) from exc
    if (
        not isinstance(summary, dict)
        or set(summary) != set(_STORE_KEYS)
        or any(type(summary.get(key)) is not int for key in _STORE_KEYS)
    ):
        raise ReleaseEvidenceRefused(
            "active store returned malformed Training release-persistence counts"
        )
    missing = [key for key in _STORE_KEYS if summary[key] < 1]
    if missing:
        raise ReleaseEvidenceRefused(
            "active store lacks the persisted manual Like and Pass the debug chains name for "
            f"this run_id ({', '.join(missing)} missing)"
        )
    return _canonical_sha256(summary)


def _verify_config(cfg) -> dict:
    """Return the calibration a release may bind, or refuse."""
    try:
        cfg_mod.validate(cfg)
    except ValueError as exc:
        raise ReleaseEvidenceRefused(f"config does not validate: {exc}") from exc
    if "hinge" not in cfg.enabled_apps:
        raise ReleaseEvidenceRefused("Hinge must be an enabled app")
    app_cfg = (cfg.apps or {}).get("hinge", {}) or {}
    if app_cfg.get("mode", cfg.mode) != "training":
        raise ReleaseEvidenceRefused(
            "Training release verification requires config mode training, never auto"
        )
    if (
        cfg.hinge_compatible_version is not None
        or "version_compatibility_review" in app_cfg
    ):
        raise ReleaseEvidenceRefused(
            "a version_compatibility_review is configured; a Training release binds only an "
            "exact-build calibration, so remove the review and record the run on the "
            "calibrated build"
        )
    calibration = app_cfg.get("targeting_calibration")
    if not isinstance(calibration, dict):
        raise ReleaseEvidenceRefused(
            "apps.hinge.targeting_calibration is required before Training release verification"
        )
    return calibration


def verify(
    *, cfg, run_dir: Path, run_id: str, out_dir: Path, store=None
) -> tuple[dict, dict]:
    """Verify existing evidence only and return ``(artifact, paste_mapping)``."""
    calibration = _verify_config(cfg)
    if not run_id.strip():
        raise ReleaseEvidenceRefused("--run-id must be the production Worker run id")
    debug = _verify_debug_run(run_dir, run_id, calibration)
    owned_store = store is None
    store = make_store(cfg, ensure=False) if store is None else store
    try:
        persistence_sha = _verify_store_persistence(
            store,
            run_id,
            like_profile_id=debug["like"]["profile_id"],
            pass_profile_id=debug["pass"]["profile_id"],
            like_send_evidence_id=debug["like"]["send_evidence_id"],
        )
    finally:
        if owned_store:
            try:
                store.close()
            except Exception:  # noqa: BLE001 -- verification has already made the safe decision
                pass
    root = Path.cwd().resolve()
    try:
        run_ref = str(run_dir.resolve().relative_to(root))
        out_dir.resolve().relative_to(root)
    except ValueError as exc:
        raise ReleaseEvidenceRefused(
            "--debug-run and --out must be inside the repository"
        ) from exc
    verified_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "schema_version": 1,
        "kind": KIND,
        "completed": True,
        "human_ground_truth": True,
        "source": "manual",
        "calibration_calibrated_at": calibration["calibrated_at"],
        "calibration_sha256": _canonical_sha256(calibration),
        "device": calibration["device"],
        "hinge_version_name": calibration["hinge_version_name"],
        "frame_size_px": calibration["frame_size_px"],
        "production_run_reference": run_ref,
        "production_run_id": run_id,
        "debug_actions_sha256": debug["debug_actions_sha256"],
        "chain_evidence_sha256": _canonical_sha256(
            {"like": debug["like"]["events"], "pass": debug["pass"]["events"]}
        ),
        "store_persistence_evidence_sha256": persistence_sha,
        "not_exercised": list(NOT_EXERCISED),
        "verified_at": verified_at,
    }
    if set(artifact) != _ARTIFACT_KEYS:
        raise AssertionError("training release artifact schema drift")
    try:
        ensure_private_dir(out_dir, exist_ok=False)
    except FileExistsError as exc:
        raise ReleaseEvidenceRefused("--out must be a fresh directory") from exc
    artifact_path = out_dir / ARTIFACT_NAME
    artifact_bytes = (
        json.dumps(artifact, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    atomic_write_private_bytes(artifact_path, artifact_bytes, parent=out_dir)
    artifact_ref = str(artifact_path.resolve().relative_to(root))
    paste = {
        "schema_version": 1,
        "calibration_calibrated_at": artifact["calibration_calibrated_at"],
        "calibration_sha256": artifact["calibration_sha256"],
        "device": artifact["device"],
        "hinge_version_name": artifact["hinge_version_name"],
        "frame_size_px": artifact["frame_size_px"],
        "production_run_reference": run_ref,
        "production_run_id": run_id,
        "debug_actions_sha256": artifact["debug_actions_sha256"],
        "verification_file": artifact_ref,
        "verification_sha256": _sha256(artifact_bytes),
        "verified_at": verified_at,
    }
    if set(paste) != _PASTE_KEYS:
        raise AssertionError("training release mapping schema drift")
    return artifact, paste


def _install_reminder(app_cfg: dict) -> str:
    """What installing the printed mapping in place of the configured gates must keep."""
    reminder = (
        "AUTO accepts exactly one release gate: remove apps.hinge.observe_release_evidence "
        "and apps.hinge.ai_reviewed_observe_release_evidence when installing this mapping. "
        "Before removing either, list its production_run_id in apps.hinge.debug_protect_runs "
        "so the run behind its signed ops/release/<run>/ artifacts survives debug retention "
        "(the driver derives that protection from observe_release_evidence, so removing the "
        "gate drops it)."
    )
    protected = set(app_cfg.get("debug_protect_runs") or ())
    unlisted = []
    for key in ("observe_release_evidence", "ai_reviewed_observe_release_evidence"):
        evidence = app_cfg.get(key)
        run = evidence.get("production_run_id") if isinstance(evidence, dict) else None
        if isinstance(run, str) and run and run not in protected:
            unlisted.append(f"{run} ({key})")
    if unlisted:
        reminder += f" Not listed yet: {', '.join(unlisted)}."
    return reminder


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tools.hinge_training_release")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--debug-run", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = cfg_mod.load(args.config)
    out = Path(args.out)
    root = Path.cwd().resolve()
    try:
        if out.is_absolute():
            raise ReleaseEvidenceRefused(
                "--out must be a repo-relative fresh directory"
            )
        try:
            (root / out).resolve().relative_to(root)
        except ValueError as exc:
            raise ReleaseEvidenceRefused(
                "--out must stay inside the repository"
            ) from exc
        _artifact, paste = verify(
            cfg=cfg, run_dir=Path(args.debug_run), run_id=args.run_id, out_dir=out
        )
    except ReleaseEvidenceRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(1)
    print(
        yaml.safe_dump(
            {"apps": {"hinge": {"training_release_evidence": paste}}}, sort_keys=False
        )
    )
    print(
        _install_reminder((cfg.apps or {}).get("hinge", {}) or {}),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
