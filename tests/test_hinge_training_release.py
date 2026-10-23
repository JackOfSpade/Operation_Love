"""Training release gate: a verified supervised Training run unlocks AUTO, nothing weaker does."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from operation_love import config as config_mod
from tools import hinge_training_release as release

TS = "2026-10-23T03:00:00"
LIKE_OPENER = "You picked the one table with the ocean behind it"
PASS_OPENER = "That phone case is doing a lot of talking"
_DELETE = object()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _png(seed: int) -> bytes:
    pixels = np.zeros((6, 4, 3), dtype=np.uint8)
    pixels[:, :, 0] = seed % 256
    pixels[0, 0] = ((seed * 37) % 256, (seed * 11) % 256, 7)
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    return encoded.tobytes()


def _calibration():
    return {
        "schema_version": 3,
        "device": "pixel",
        "hinge_version_name": "10.9.0",
        "frame_size_px": [1080, 2400],
        "composer_layout_id": "hinge_inline_v1",
        "item_selection_policy_id": "hinge_photos_only_v2",
        "identity_match_max_dist": 1.0,
        "inline_item_max_dist": 2.0,
        "calibrated_at": "2026-10-23T00:00:00Z",
        "identity_band": [0.0, 0.0, 1.0, 0.1],
        "content_band": [0.1, 0.9],
    }


def _cfg(calibration, *, mode="training"):
    return SimpleNamespace(
        enabled_apps=["hinge"],
        mode=mode,
        hinge_compatible_version=None,
        db_file=Path("store.db"),
        storage=SimpleNamespace(backend="sqlite", bigquery={}),
        apps={"hinge": {"mode": mode, "targeting_calibration": calibration}},
    )


def _environment(calibration, **overrides):
    row = {
        "session_mode": "training",
        "hinge_version_name": calibration["hinge_version_name"],
        "frame_size_px": list(calibration["frame_size_px"]),
        "device": calibration["device"],
        "calibration_sha256": release._canonical_sha256(calibration),
        "compatible_version": None,
        "targeting_available": True,
        "ts": TS,
        "action": "training_session_environment",
    }
    row.update(overrides)
    return row


class _Run:
    """A synthetic Training debug run with real PNG frames, addressable by step key."""

    def __init__(self, root: Path, run_id: str, calibration: dict):
        self.run_id = run_id
        self.calibration = calibration
        self.dir = root / "data" / "hinge_debug" / run_id
        self.dir.mkdir(parents=True)
        self.rows: list[dict] = []
        self.ix: dict[str, dict] = {}
        self._frames = 0

    def frame(self) -> str:
        self._frames += 1
        name = f"{self._frames:05d}_frame.png"
        (self.dir / name).write_bytes(_png(self._frames))
        return name

    def bytes_of(self, name: str) -> bytes:
        return (self.dir / name).read_bytes()

    def add(self, key: str | None, row: dict) -> dict:
        self.rows.append(row)
        if key is not None:
            self.ix[key] = row
        return row

    def pos(self, key: str) -> int:
        return next(i for i, row in enumerate(self.rows) if row is self.ix[key])

    def remove(self, key: str) -> None:
        del self.rows[self.pos(key)]

    def insert_after(self, key: str, row: dict) -> None:
        self.rows.insert(self.pos(key) + 1, row)

    def swap(self, first: str, second: str) -> None:
        a, b = self.pos(first), self.pos(second)
        self.rows[a], self.rows[b] = self.rows[b], self.rows[a]

    def write(self) -> None:
        (self.dir / "actions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows)
        )

    def verify_debug(self):
        self.write()
        return release._verify_debug_run(self.dir, self.run_id, self.calibration)

    def _verify_pair(self, prefix: str | None, item: int) -> None:
        self.add(
            f"{prefix}.identity" if prefix else None,
            {
                "item": item,
                "outcome": "identity_match",
                "distance": 0.0,
                "bound": 1.8,
                "ts": TS,
                "action": "verify_sheet_identity",
                "before": self.frame(),
                "after": self.frame(),
            },
        )
        self.add(
            f"{prefix}.item" if prefix else None,
            {
                "item": item,
                "outcome": "verify_match",
                "nearest": item,
                "ts": TS,
                "action": "verify_sheet_item",
                "before": self.frame(),
                "after": self.frame(),
            },
        )

    def chain(self, kind: str, *, item: int, opener: str, profile_id: str) -> None:
        p = kind
        command = "like" if kind == "like" else "dislike"
        self.add(
            f"{p}.capture",
            {
                "before": self.frame(),
                "photos": 18,
                "capture_truncated": False,
                "items": 6,
                "profile_name": "Synthetic",
                "ts": TS,
                "action": "capture",
            },
        )
        self._verify_pair(p, item)
        self.add(
            f"{p}.anchor",
            {
                "item_index": None,
                "model_item_index": item,
                "verified": True,
                "ts": TS,
                "action": "like_anchor",
                "before": self.frame(),
            },
        )
        # Focused-field and post-type re-verification, as the driver logs them.
        self._verify_pair(None, item)
        self._verify_pair(None, item)
        opener_bytes = opener.encode("utf-8")
        pre_frame = self.frame()
        pre_bytes = self.bytes_of(pre_frame)
        pre_id = _sha(pre_bytes + b"\0" + opener_bytes)
        self.add(
            f"{p}.pre_send",
            {
                "opener": opener,
                "opener_chars": len(opener),
                "opener_sha256": _sha(opener_bytes),
                "frame_sha256": _sha(pre_bytes),
                "evidence_id": pre_id,
                "item_index": None,
                "model_item_index": item,
                "session_mode": "training",
                "draft_sha256": "d" * 64,
                "draft_rect": [95, 1124, 985, 1302],
                "prompt_sha256": "e" * 64,
                "model": "gemini-x",
                "index_space": "model_items",
                "ts": TS,
                "action": "auto_opener_pre_send",
                "before": pre_frame,
                "kept_before": pre_frame,
            },
        )
        self.add(
            f"{p}.published",
            {
                "evidence_id": pre_id,
                "profile_token_sha256": "1" * 64,
                "approval_token_sha256": "2" * 64,
                "ts": TS,
                "action": "training_release_checkpoint_published",
            },
        )
        self.add(
            f"{p}.claimed",
            {
                "command": command,
                "evidence_id": pre_id,
                "idempotency_token_sha256": "3" * 64,
                "waited_s": 12.5,
                "ts": TS,
                "action": "training_release_decision_claimed",
            },
        )
        self._verify_pair(None, item)
        send_id = None
        if kind == "like":
            resumed_frame = self.frame()
            resumed_bytes = self.bytes_of(resumed_frame)
            send_id = _sha(resumed_bytes + b"\0" + opener_bytes)
            self.add(
                "like.resumed",
                {
                    "opener": opener,
                    "opener_chars": len(opener),
                    "opener_sha256": _sha(opener_bytes),
                    "frame_sha256": _sha(resumed_bytes),
                    "evidence_id": send_id,
                    "approval_evidence_id": pre_id,
                    "item_index": None,
                    "model_item_index": item,
                    "session_mode": "training",
                    "prompt_sha256": "e" * 64,
                    "ts": TS,
                    "action": "auto_opener_resumed_send",
                    "before": resumed_frame,
                    "kept_before": resumed_frame,
                },
            )
        self.add(
            None,
            {
                "kind": "tap",
                "session_mode": "training",
                "ts": TS,
                "action": "device_input",
            },
        )
        attempt = {
            "heart": [939, 1344],
            "opener_chars": len(opener),
            "item_index": None,
            "model_item_index": item,
            "verified": True,
            "pre_send_evidence_id": pre_id,
            "resumed_send_evidence_id": send_id,
        }
        if kind == "like":
            self.add(
                "like.attempt",
                {
                    **attempt,
                    "ts": TS,
                    "action": "like_attempt",
                    "before": self.frame(),
                    "after": self.frame(),
                },
            )
        self.add(
            f"{p}.retry",
            {
                "attempt": 1,
                "outcome": "retry",
                "first": {"deck_ready": False},
                "second": None,
                "stable": None,
                "names_agree": None,
                "same_name_successor_enabled": kind == "pass",
                "ts": TS,
                "action": "training_advance_probe",
            },
        )
        probe_before, probe_after = self.frame(), self.frame()
        self.add(
            f"{p}.probe",
            {
                "attempt": 2,
                "outcome": "accepted",
                "first": {"deck_ready": True},
                "second": {"deck_ready": True},
                "stable": True,
                "names_agree": True,
                "same_name_successor_enabled": kind == "pass",
                "ts": TS,
                "action": "training_advance_probe",
                "before": probe_before,
                "after": probe_after,
                "kept_before": probe_before,
                "kept_after": probe_after,
            },
        )
        if kind == "like":
            self.add(
                "like.like",
                {
                    **attempt,
                    "rose_modal": False,
                    "ts": TS,
                    "action": "like",
                    "before": self.frame(),
                    "after": self.frame(),
                },
            )
        else:
            self.add(
                "pass.dislike",
                {
                    "x": [116, 2062],
                    "model_item_index": item,
                    "verified": True,
                    "advance_proof": "name",
                    "pre_send_evidence_id": pre_id,
                    "ts": TS,
                    "action": "training_dislike",
                    "before": self.frame(),
                    "after": self.frame(),
                },
            )
        self.add(
            f"{p}.persisted",
            {
                "decision": command,
                "evidence_id": pre_id,
                "send_evidence_id": send_id,
                "profile_id": profile_id,
                "ts": TS,
                "action": "training_release_action_persisted",
            },
        )


def _valid_run(root: Path, run_id="run-1", *, pass_first=False) -> _Run:
    run = _Run(root, run_id, _calibration())
    run.add(
        "binding",
        {
            "ts": TS,
            "action": "observe_release_run_binding",
            "run_id": run_id,
            "app": "hinge",
        },
    )
    run.add("env", _environment(run.calibration))
    chains = [
        ("like", 3, LIKE_OPENER, "profile-like"),
        ("pass", 6, PASS_OPENER, "profile-pass"),
    ]
    for kind, item, opener, profile_id in reversed(chains) if pass_first else chains:
        run.chain(kind, item=item, opener=opener, profile_id=profile_id)
    run.write()
    return run


class _Store:
    def __init__(self, **overrides):
        self.calls = []
        self.summary = {key: 1 for key in release._STORE_KEYS}
        self.summary.update(overrides)

    def training_release_persistence_summary(self, run_id, app, **kwargs):
        self.calls.append((run_id, app, kwargs))
        return dict(self.summary)


@pytest.fixture
def validated(monkeypatch, tmp_path):
    """A repo rooted at tmp_path whose config validation itself is not under test."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod, "validate", lambda cfg: None)
    monkeypatch.setattr(config_mod, "hinge_targeting_unavailable_reason", lambda: None)
    return tmp_path


def _verify(run: _Run, out: Path, *, cfg=None, store=None):
    cfg = _cfg(run.calibration) if cfg is None else cfg
    return release.verify(
        cfg=cfg,
        run_dir=run.dir,
        run_id=run.run_id,
        out_dir=out,
        store=_Store() if store is None else store,
    )


def _released(root: Path):
    run = _valid_run(root)
    artifact, paste = _verify(run, root / "ops" / "release" / "training")
    cfg = _cfg(run.calibration, mode="auto")
    cfg.apps["hinge"]["training_release_evidence"] = paste
    return run, artifact, paste, cfg


# --- verifier: accepted evidence ------------------------------------------------------------


def test_verifier_emits_exact_training_release_mapping_that_config_accepts(validated):
    run = _valid_run(validated)
    store = _Store()
    artifact, paste = _verify(
        run, validated / "ops" / "release" / "training", store=store
    )

    assert set(artifact) == release._ARTIFACT_KEYS
    assert artifact["kind"] == "hinge_supervised_training_release"
    assert artifact["completed"] is True and artifact["human_ground_truth"] is True
    assert artifact["source"] == "manual"
    assert artifact["not_exercised"] == release.NOT_EXERCISED
    assert set(paste) == release._PASTE_KEYS
    assert paste["debug_actions_sha256"] == _sha(
        (run.dir / "actions.jsonl").read_bytes()
    )
    assert store.calls == [
        (
            "run-1",
            "hinge",
            {
                "like_profile_id": "profile-like",
                "pass_profile_id": "profile-pass",
                "like_send_evidence_id": run.ix["like.resumed"]["evidence_id"],
            },
        )
    ]
    artifact_path = validated / paste["verification_file"]
    content = artifact_path.read_bytes()
    assert _sha(content) == paste["verification_sha256"]
    # Digests only: no opener text, profile ids, or raw evidence ids leave the debug run.
    for secret in (LIKE_OPENER, PASS_OPENER, "profile-like", "profile-pass"):
        assert secret.encode() not in content
    if os.name == "posix":
        assert stat.S_IMODE(artifact_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(artifact_path.stat().st_mode) == 0o600

    cfg = _cfg(run.calibration, mode="auto")
    cfg.apps["hinge"]["training_release_evidence"] = paste
    config_mod._validate_hinge_auto_release_gate(cfg)


def test_verifier_and_config_share_one_artifact_contract():
    assert release._ARTIFACT_KEYS == config_mod._TRAINING_RELEASE_ARTIFACT_KEYS
    assert release._PASTE_KEYS == config_mod._TRAINING_RELEASE_EVIDENCE_KEYS
    assert release.NOT_EXERCISED == config_mod._TRAINING_RELEASE_NOT_EXERCISED
    assert release.KIND == config_mod._TRAINING_RELEASE_KIND


def test_pass_may_precede_like_and_unrelated_rows_do_not_break_a_chain(tmp_path):
    run = _valid_run(tmp_path, pass_first=True)
    run.insert_after(
        "like.attempt",
        {"ts": TS, "action": "message_warning", "outcome": "send_anyway"},
    )
    debug = run.verify_debug()
    assert debug["like"]["profile_id"] == "profile-like"
    assert debug["pass"]["profile_id"] == "profile-pass"
    assert [event["action"] for event in debug["like"]["events"]] == [
        "capture",
        "verify_sheet_identity",
        "verify_sheet_item",
        *release._LIKE_ORDER,
    ]
    assert [event["action"] for event in debug["pass"]["events"]] == [
        "capture",
        "verify_sheet_identity",
        "verify_sheet_item",
        *release._PASS_ORDER,
    ]


def test_a_reopened_sheet_restarts_from_its_new_anchor(tmp_path):
    """An earlier verified sheet on the same card does not void a later complete chain."""
    run = _valid_run(tmp_path)
    earlier = [
        {
            "item": 2,
            "outcome": "identity_match",
            "ts": TS,
            "action": "verify_sheet_identity",
        },
        {"item": 2, "outcome": "verify_match", "ts": TS, "action": "verify_sheet_item"},
        {
            "item_index": None,
            "model_item_index": 2,
            "verified": True,
            "ts": TS,
            "action": "like_anchor",
            "before": run.frame(),
        },
        {"ts": TS, "action": "like_dismissed"},
    ]
    for row in reversed(earlier):
        run.insert_after("like.capture", row)
    debug = run.verify_debug()
    anchor = next(e for e in debug["like"]["events"] if e["action"] == "like_anchor")
    assert anchor["index"] == run.pos("like.anchor")


def test_only_frames_cited_by_the_chosen_chains_must_exist(tmp_path):
    run = _valid_run(tmp_path)
    for key in ("like.capture", "like.like", "pass.dislike"):
        for frame in ("before", "after"):
            name = run.ix[key].get(frame)
            if name:
                (run.dir / name).unlink()
    run.verify_debug()


# --- verifier: every chain requirement is load-bearing ----------------------------------------

LIKE_STEPS = [
    "like.identity",
    "like.item",
    "like.anchor",
    "like.pre_send",
    "like.published",
    "like.claimed",
    "like.resumed",
    "like.attempt",
    "like.probe",
    "like.like",
    "like.persisted",
]
PASS_STEPS = [
    "pass.identity",
    "pass.item",
    "pass.anchor",
    "pass.pre_send",
    "pass.published",
    "pass.claimed",
    "pass.probe",
    "pass.dislike",
    "pass.persisted",
]


def _chain_needle(key: str) -> str:
    return "LIKE chain" if key.startswith("like.") else "PASS chain"


@pytest.mark.parametrize("key", LIKE_STEPS + PASS_STEPS)
def test_every_chain_step_is_required(tmp_path, key):
    run = _valid_run(tmp_path)
    run.remove(key)
    with pytest.raises(release.ReleaseEvidenceRefused, match=_chain_needle(key)):
        run.verify_debug()


@pytest.mark.parametrize(
    "first,second",
    [
        ("like.anchor", "like.pre_send"),
        ("like.pre_send", "like.published"),
        ("like.published", "like.claimed"),
        ("like.claimed", "like.resumed"),
        ("like.resumed", "like.attempt"),
        ("like.attempt", "like.probe"),
        ("like.probe", "like.like"),
        ("like.like", "like.persisted"),
        ("pass.published", "pass.claimed"),
        ("pass.claimed", "pass.probe"),
        ("pass.probe", "pass.dislike"),
        ("pass.dislike", "pass.persisted"),
    ],
)
def test_reordered_chain_steps_are_refused(tmp_path, first, second):
    run = _valid_run(tmp_path)
    run.swap(first, second)
    with pytest.raises(release.ReleaseEvidenceRefused, match=_chain_needle(first)):
        run.verify_debug()


@pytest.mark.parametrize(
    "key,field,value",
    [
        ("like.identity", "outcome", "identity_mismatch"),
        ("like.identity", "outcome", "identity_unknown"),
        ("like.item", "outcome", "verify_mismatch"),
        ("like.item", "item", 4),
        ("like.anchor", "verified", False),
        ("like.anchor", "model_item_index", "3"),
        ("like.pre_send", "session_mode", "auto"),
        ("like.pre_send", "model_item_index", 4),
        ("like.pre_send", "opener_chars", len(LIKE_OPENER) + 1),
        ("like.pre_send", "opener", ""),
        ("like.pre_send", "opener_sha256", "0" * 64),
        ("like.pre_send", "frame_sha256", "0" * 64),
        ("like.pre_send", "evidence_id", "0" * 64),
        ("like.pre_send", "kept_before", _DELETE),
        ("like.published", "evidence_id", "0" * 64),
        ("like.published", "evidence_id", None),
        ("like.published", "approval_token_sha256", "raw-approval-token"),
        ("like.published", "opener", LIKE_OPENER),
        ("like.claimed", "evidence_id", "0" * 64),
        ("like.claimed", "command", "dislike"),
        ("like.claimed", "command", "superlike"),
        ("like.claimed", "idempotency_token_sha256", "raw-token"),
        ("like.claimed", "waited_s", -1.0),
        ("like.claimed", "waited_s", True),
        ("like.resumed", "approval_evidence_id", "0" * 64),
        ("like.resumed", "opener_sha256", "0" * 64),
        ("like.resumed", "evidence_id", "0" * 64),
        ("like.resumed", "frame_sha256", "0" * 64),
        ("like.resumed", "model_item_index", 4),
        ("like.resumed", "session_mode", "auto"),
        ("like.resumed", "kept_before", _DELETE),
        ("like.attempt", "pre_send_evidence_id", "0" * 64),
        ("like.attempt", "resumed_send_evidence_id", "0" * 64),
        ("like.attempt", "verified", False),
        ("like.attempt", "opener_chars", 0),
        ("like.probe", "stable", False),
        ("like.probe", "names_agree", False),
        ("like.probe", "kept_after", _DELETE),
        ("like.probe", "kept_before", _DELETE),
        ("like.like", "verified", False),
        ("like.like", "model_item_index", 4),
        ("like.like", "pre_send_evidence_id", "0" * 64),
        ("like.like", "resumed_send_evidence_id", "0" * 64),
        ("like.persisted", "decision", "dislike"),
        ("like.persisted", "evidence_id", "0" * 64),
        ("like.persisted", "send_evidence_id", None),
        ("like.persisted", "profile_id", ""),
        ("like.persisted", "profile_name", "Synthetic"),
        ("pass.claimed", "command", "like"),
        ("pass.probe", "kept_after", _DELETE),
        ("pass.dislike", "verified", False),
        ("pass.dislike", "advance_proof", "unproven"),
        ("pass.dislike", "pre_send_evidence_id", "0" * 64),
        ("pass.dislike", "model_item_index", 3),
        ("pass.persisted", "decision", "like"),
        ("pass.persisted", "evidence_id", "0" * 64),
        ("pass.persisted", "send_evidence_id", "0" * 64),
        ("pass.persisted", "profile_id", None),
    ],
)
def test_every_chain_field_is_load_bearing(tmp_path, key, field, value):
    run = _valid_run(tmp_path)
    if value is _DELETE:
        del run.ix[key][field]
    else:
        run.ix[key][field] = value
    with pytest.raises(release.ReleaseEvidenceRefused, match=_chain_needle(key)):
        run.verify_debug()


@pytest.mark.parametrize("key", ["like.pre_send", "like.resumed", "pass.pre_send"])
def test_a_tampered_kept_snapshot_is_refused(tmp_path, key):
    """Evidence ids are recomputed from the kept PNG, so its pixels are bound."""
    run = _valid_run(tmp_path)
    (run.dir / run.ix[key]["kept_before"]).write_bytes(_png(250))
    with pytest.raises(release.ReleaseEvidenceRefused, match=_chain_needle(key)):
        run.verify_debug()


@pytest.mark.parametrize("key", ["like.pre_send", "like.resumed", "pass.pre_send"])
def test_a_replaced_kept_frame_with_a_rewritten_frame_digest_is_refused(tmp_path, key):
    """frame_sha256 is the row's own claim; the evidence id still binds frame and opener."""
    run = _valid_run(tmp_path)
    replacement = _png(250)
    (run.dir / run.ix[key]["kept_before"]).write_bytes(replacement)
    run.ix[key]["frame_sha256"] = _sha(replacement)
    with pytest.raises(release.ReleaseEvidenceRefused, match=_chain_needle(key)):
        run.verify_debug()


@pytest.mark.parametrize("key", ["like.pre_send", "like.resumed", "pass.pre_send"])
def test_a_forged_but_self_consistent_evidence_id_is_refused(tmp_path, key):
    """Every row may agree on an id; it must still be the digest of the kept frame + opener."""
    run = _valid_run(tmp_path)
    original = run.ix[key]["evidence_id"]
    for row in run.rows:
        for field, value in list(row.items()):
            if value == original:
                row[field] = "f" * 64
    with pytest.raises(release.ReleaseEvidenceRefused, match=_chain_needle(key)):
        run.verify_debug()


def test_the_resumed_frame_is_bound_to_the_approved_opener(tmp_path):
    """A resumed row carrying its own, different opener cannot re-bind the send."""
    run = _valid_run(tmp_path)
    resumed = run.ix["like.resumed"]
    edited = LIKE_OPENER + " (edited)"
    frame = run.bytes_of(resumed["kept_before"])
    forged = _sha(frame + b"\0" + edited.encode())
    for row in run.rows:
        for field, value in list(row.items()):
            if value == resumed["evidence_id"]:
                row[field] = forged
    resumed["opener"] = edited
    resumed["opener_chars"] = len(edited)
    with pytest.raises(release.ReleaseEvidenceRefused, match="LIKE chain"):
        run.verify_debug()


@pytest.mark.parametrize(
    "key,field",
    [
        ("like.pre_send", "kept_before"),
        ("like.resumed", "kept_before"),
        ("like.probe", "kept_before"),
        ("like.probe", "kept_after"),
        ("pass.pre_send", "kept_before"),
        ("pass.probe", "kept_before"),
        ("pass.probe", "kept_after"),
    ],
)
def test_a_missing_cited_frame_is_refused(tmp_path, key, field):
    run = _valid_run(tmp_path)
    (run.dir / run.ix[key][field]).unlink()
    with pytest.raises(
        release.ReleaseEvidenceRefused, match=_chain_needle(key)
    ) as refused:
        run.verify_debug()
    # The refusal says a frame was missing, not only that no chain was found.
    assert "1 otherwise in-order chain step(s) were dropped" in str(refused.value)
    assert "kept frame" in str(refused.value)


def test_a_complete_chain_without_a_kept_frame_problem_names_no_missing_frame(tmp_path):
    run = _valid_run(tmp_path)
    run.remove("like.persisted")
    with pytest.raises(release.ReleaseEvidenceRefused, match="LIKE chain") as refused:
        run.verify_debug()
    assert "kept frame" not in str(refused.value)


def test_a_rotated_away_anchor_frame_does_not_void_the_chain(tmp_path):
    """like_anchor's frame sits in the rotating ring and carries no digest, so it is not proof.

    Every kept frame is still on disk; only the two anchors' ordinary frames are gone.
    """
    run = _valid_run(tmp_path)
    for key in ("like.anchor", "pass.anchor"):
        (run.dir / run.ix[key]["before"]).unlink()
    debug = run.verify_debug()
    assert debug["like"]["profile_id"] == "profile-like"
    assert debug["pass"]["profile_id"] == "profile-pass"


@pytest.mark.parametrize("key", ["like.probe", "pass.probe"])
def test_a_successor_proof_needs_two_distinct_kept_frames(tmp_path, key):
    run = _valid_run(tmp_path)
    run.ix[key]["kept_after"] = run.ix[key]["kept_before"]
    with pytest.raises(release.ReleaseEvidenceRefused, match=_chain_needle(key)):
        run.verify_debug()


def test_a_kept_frame_outside_the_run_is_refused(tmp_path):
    run = _valid_run(tmp_path)
    outside = tmp_path / "outside.png"
    name = run.ix["like.pre_send"]["kept_before"]
    outside.write_bytes(run.bytes_of(name))
    run.ix["like.pre_send"]["kept_before"] = "../../../outside.png"
    with pytest.raises(release.ReleaseEvidenceRefused, match="LIKE chain"):
        run.verify_debug()


@pytest.mark.parametrize(
    "row",
    [
        {"ts": TS, "action": "capture", "before": "x.png"},
        {"ts": TS, "action": "capture_aborted", "frames": 3},
        {"ts": TS, "action": "training_cancelled", "reason": "stop_requested"},
        {"ts": TS, "action": "training_dislike_unverified", "error": "X: y"},
        {"ts": TS, "action": "like_rejected", "rejection": "deck_blocked"},
        {"ts": TS, "action": "unexpected", "error": "HingeActionError: stuck"},
        _environment(_calibration()),
    ],
    ids=[
        "capture",
        "capture_aborted",
        "training_cancelled",
        "dislike_unverified",
        "like_rejected",
        "error_row",
        "new_session",
    ],
)
@pytest.mark.parametrize("after", ["like.anchor", "like.claimed", "like.like"])
def test_an_interruption_inside_the_chain_resets_it(tmp_path, row, after):
    run = _valid_run(tmp_path)
    run.insert_after(after, dict(row))
    with pytest.raises(release.ReleaseEvidenceRefused, match="LIKE chain"):
        run.verify_debug()


def test_a_second_anchor_mid_chain_voids_the_earlier_snapshot(tmp_path):
    run = _valid_run(tmp_path)
    run.insert_after(
        "like.pre_send",
        {
            "item_index": None,
            "model_item_index": 3,
            "verified": True,
            "ts": TS,
            "action": "like_anchor",
            "before": run.ix["like.anchor"]["before"],
        },
    )
    with pytest.raises(release.ReleaseEvidenceRefused, match="LIKE chain"):
        run.verify_debug()


def test_a_like_and_pass_on_one_profile_id_are_refused(tmp_path):
    run = _valid_run(tmp_path)
    run.ix["pass.persisted"]["profile_id"] = "profile-like"
    with pytest.raises(release.ReleaseEvidenceRefused, match="same profile id"):
        run.verify_debug()


# --- verifier: run binding and session environment --------------------------------------------


def test_directory_must_be_named_for_the_run(tmp_path):
    run = _valid_run(tmp_path)
    with pytest.raises(release.ReleaseEvidenceRefused, match="directory name"):
        release._verify_debug_run(run.dir, "other-run", run.calibration)


@pytest.mark.parametrize(
    "field,value",
    [("run_id", "other-run"), ("app", "bumble"), ("extra", 1), ("ts", "")],
)
def test_first_row_must_be_the_exact_run_binding(tmp_path, field, value):
    run = _valid_run(tmp_path)
    run.ix["binding"][field] = value
    with pytest.raises(release.ReleaseEvidenceRefused, match="first production"):
        run.verify_debug()


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_mode", "auto"),
        ("session_mode", "observe"),
        ("hinge_version_name", "10.8.0"),
        ("hinge_version_name", None),
        ("frame_size_px", [1080, 2340]),
        ("frame_size_px", None),
        ("device", "another-pixel"),
        ("calibration_sha256", "0" * 64),
        ("compatible_version", "10.9.0"),
        ("targeting_available", False),
        ("opener", "leaked"),
    ],
)
def test_the_session_environment_must_bind_the_exact_calibration(
    tmp_path, field, value
):
    run = _valid_run(tmp_path)
    run.ix["env"][field] = value
    with pytest.raises(release.ReleaseEvidenceRefused, match="does not bind"):
        run.verify_debug()


def test_the_environment_row_must_directly_follow_the_binding(tmp_path):
    run = _valid_run(tmp_path)
    env = run.ix["env"]
    run.remove("env")
    with pytest.raises(release.ReleaseEvidenceRefused, match="directly after"):
        run.verify_debug()
    run.insert_after("like.capture", env)
    with pytest.raises(release.ReleaseEvidenceRefused, match="directly after"):
        run.verify_debug()


def test_a_later_environment_row_that_does_not_bind_refuses_the_whole_run(tmp_path):
    run = _valid_run(tmp_path)
    run.add(None, _environment(run.calibration, hinge_version_name="10.10.0"))
    with pytest.raises(release.ReleaseEvidenceRefused, match="does not bind"):
        run.verify_debug()


def test_cli_prints_the_mapping_and_refuses_an_out_outside_the_repo(
    validated, monkeypatch, capsys
):
    import yaml

    run = _valid_run(validated)
    monkeypatch.setattr(config_mod, "load", lambda _path: _cfg(run.calibration))
    monkeypatch.setattr(release, "make_store", lambda _cfg, ensure=True: _Store())
    args = ["--debug-run", "data/hinge_debug/run-1", "--run-id", "run-1", "--out"]
    for out in (str(validated / "absolute"), "../outside"):
        with pytest.raises(SystemExit):
            release.main([*args, out])
        assert "REFUSED" in capsys.readouterr().err

    release.main([*args, "ops/release/run-1/training-release"])
    printed = capsys.readouterr()
    paste = yaml.safe_load(printed.out)["apps"]["hinge"]["training_release_evidence"]
    assert set(paste) == release._PASTE_KEYS
    assert paste["verification_file"] == (
        "ops/release/run-1/training-release/hinge_training_release.json"
    )
    assert "exactly one release gate" in printed.err
    assert "apps.hinge.debug_protect_runs" in printed.err


def test_the_install_reminder_names_each_release_run_a_removed_gate_leaves_unlisted():
    """Removing a gate removes the retention protection the driver derives from it."""
    reminder = release._install_reminder(
        {
            "debug_protect_runs": ["listed_run_01"],
            "observe_release_evidence": {"production_run_id": "observe_run_01"},
            "ai_reviewed_observe_release_evidence": {
                "production_run_id": "listed_run_01"
            },
        }
    )
    assert "observe_run_01 (observe_release_evidence)" in reminder
    assert "listed_run_01" not in reminder
    for app_cfg in ({}, {"observe_release_evidence": "pending"}):
        assert "Not listed yet" not in release._install_reminder(app_cfg)


# --- verifier: config preconditions -------------------------------------------------------------


def test_config_that_does_not_validate_is_refused(validated, monkeypatch):
    run = _valid_run(validated)

    def refuse(cfg):
        raise ValueError("Config: synthetic refusal")

    monkeypatch.setattr(config_mod, "validate", refuse)
    with pytest.raises(release.ReleaseEvidenceRefused, match="synthetic refusal"):
        _verify(run, validated / "out")


def test_auto_mode_at_verify_time_is_refused(validated):
    run = _valid_run(validated)
    with pytest.raises(release.ReleaseEvidenceRefused, match="mode training"):
        _verify(run, validated / "out", cfg=_cfg(run.calibration, mode="auto"))


def test_a_version_compatibility_review_is_refused(validated):
    run = _valid_run(validated)
    cfg = _cfg(run.calibration)
    cfg.apps["hinge"]["version_compatibility_review"] = {"receipt": "x"}
    with pytest.raises(release.ReleaseEvidenceRefused, match="exact-build"):
        _verify(run, validated / "out", cfg=cfg)
    del cfg.apps["hinge"]["version_compatibility_review"]
    cfg.hinge_compatible_version = "10.9.0"
    with pytest.raises(release.ReleaseEvidenceRefused, match="exact-build"):
        _verify(run, validated / "out", cfg=cfg)


def test_a_missing_calibration_is_refused(validated):
    run = _valid_run(validated)
    cfg = _cfg(run.calibration)
    del cfg.apps["hinge"]["targeting_calibration"]
    with pytest.raises(release.ReleaseEvidenceRefused, match="targeting_calibration"):
        _verify(run, validated / "out", cfg=cfg)


def test_out_directory_must_be_fresh(validated):
    run = _valid_run(validated)
    (validated / "out").mkdir()
    with pytest.raises(release.ReleaseEvidenceRefused, match="fresh"):
        _verify(run, validated / "out")


# --- verifier: store persistence ------------------------------------------------------------------


@pytest.mark.parametrize("key", release._STORE_KEYS)
def test_each_store_count_is_required(validated, key):
    run = _valid_run(validated)
    with pytest.raises(release.ReleaseEvidenceRefused, match=key):
        _verify(run, validated / "out", store=_Store(**{key: 0}))
    assert not (validated / "out").exists()


@pytest.mark.parametrize(
    "summary",
    [
        {key: 1 for key in release._STORE_KEYS if key != "like_openers"},
        {**{key: 1 for key in release._STORE_KEYS}, "successful_hinge_openers": 1},
        {**{key: 1 for key in release._STORE_KEYS}, "like_labels": True},
        {**{key: 1 for key in release._STORE_KEYS}, "like_labels": 1.0},
        [1, 1, 1, 1, 1, 1],
    ],
    ids=["missing_key", "extra_key", "bool_count", "float_count", "not_a_dict"],
)
def test_a_malformed_store_summary_is_refused(validated, summary):
    run = _valid_run(validated)

    class _Malformed:
        def training_release_persistence_summary(self, *_args, **_kwargs):
            return summary

    with pytest.raises(release.ReleaseEvidenceRefused, match="malformed"):
        _verify(run, validated / "out", store=_Malformed())


def test_a_store_error_is_refused(validated):
    run = _valid_run(validated)

    class _Broken:
        def training_release_persistence_summary(self, *_args, **_kwargs):
            raise RuntimeError("backend down")

    with pytest.raises(release.ReleaseEvidenceRefused, match="RuntimeError"):
        _verify(run, validated / "out", store=_Broken())


# --- config gate --------------------------------------------------------------------------------


def _rewrite_artifact(root: Path, paste: dict, **changes) -> None:
    path = root / paste["verification_file"]
    artifact = json.loads(path.read_bytes())
    artifact.update(changes)
    content = json.dumps(artifact, indent=2, sort_keys=True).encode() + b"\n"
    path.write_bytes(content)
    paste["verification_sha256"] = _sha(content)


@pytest.mark.parametrize(
    "others",
    [
        ("observe_release_evidence",),
        ("ai_reviewed_observe_release_evidence",),
        ("observe_release_evidence", "ai_reviewed_observe_release_evidence"),
    ],
)
def test_config_requires_exactly_one_of_three_release_gates(validated, others):
    _run, _artifact, _paste, cfg = _released(validated)
    for key in others:
        cfg.apps["hinge"][key] = {"not": "this one"}
    with pytest.raises(ValueError, match="exactly one release gate") as excinfo:
        config_mod._validate_hinge_auto_release_gate(cfg)
    for name in (
        "observe_release_evidence",
        "ai_reviewed_observe_release_evidence",
        "training_release_evidence",
    ):
        assert name in str(excinfo.value)


def test_config_without_any_gate_points_at_the_training_release_tool(validated):
    cfg = _cfg(_calibration(), mode="auto")
    with pytest.raises(ValueError, match="observe_release_evidence") as excinfo:
        config_mod._validate_hinge_auto_release_gate(cfg)
    assert "python -m tools.hinge_training_release" in str(excinfo.value)
    assert "training_release_evidence" in str(excinfo.value)


@pytest.mark.parametrize(
    "gate", ["observe_release_evidence", "ai_reviewed_observe_release_evidence"]
)
def test_an_observe_gate_refusal_points_at_the_training_release_tool(validated, gate):
    """Each OBSERVE gate keeps its own refusal; config adds the one step that can still run."""
    cfg = _cfg(_calibration(), mode="auto")
    cfg.apps["hinge"][gate] = {"schema_version": 1}
    with pytest.raises(ValueError) as excinfo:
        config_mod._validate_hinge_auto_release_gate(cfg)
    assert (
        "python -m tools.hinge_training_release and install the "
        f"apps.hinge.training_release_evidence mapping it prints in place of apps.hinge.{gate}"
    ) in str(excinfo.value)


def test_config_rehashes_the_retained_debug_log(validated):
    run, _artifact, _paste, cfg = _released(validated)
    actions = run.dir / "actions.jsonl"
    original = actions.read_bytes()
    actions.write_bytes(original + b'{"action": "appended"}\n')
    with pytest.raises(ValueError, match="does not match the retained run"):
        config_mod._validate_hinge_auto_release_gate(cfg)
    actions.unlink()
    with pytest.raises(ValueError, match="actions.jsonl is unreadable"):
        config_mod._validate_hinge_auto_release_gate(cfg)
    actions.write_bytes(original)
    config_mod._validate_hinge_auto_release_gate(cfg)


def test_config_refuses_a_mapping_digest_that_disagrees_with_the_artifact(validated):
    _run, _artifact, paste, cfg = _released(validated)
    paste["debug_actions_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="disagrees on debug_actions_sha256"):
        config_mod._validate_hinge_auto_release_gate(cfg)


def test_config_refuses_a_tampered_artifact(validated):
    _run, _artifact, paste, cfg = _released(validated)
    path = validated / paste["verification_file"]
    path.write_bytes(path.read_bytes().replace(b'"manual"', b'"automation"'))
    with pytest.raises(ValueError, match="verification_sha256 does not match"):
        config_mod._validate_hinge_auto_release_gate(cfg)


@pytest.mark.parametrize(
    "changes,needle",
    [
        ({"not_exercised": release.NOT_EXERCISED[:2]}, "does not exercise"),
        ({"not_exercised": []}, "does not exercise"),
        ({"not_exercised": list(reversed(release.NOT_EXERCISED))}, "does not exercise"),
        ({"human_ground_truth": False}, "manual human ground truth"),
        ({"source": "external_ai_review"}, "manual human ground truth"),
        ({"kind": "hinge_production_observe_release"}, "schema/kind"),
        ({"schema_version": True}, "schema/kind"),
        ({"completed": False}, "incomplete"),
        ({"chain_evidence_sha256": "not-a-digest"}, "not a SHA-256"),
        ({"store_persistence_evidence_sha256": "F" * 64}, "not a SHA-256"),
        ({"verified_at": "2000-01-01T00:00:00+00:00"}, "disagrees on verified_at"),
        ({"device": "another-pixel"}, "disagrees on device"),
        ({"opener": "leaked"}, "invalid exact schema"),
    ],
)
def test_config_refuses_an_artifact_that_claims_something_else(
    validated, changes, needle
):
    _run, _artifact, paste, cfg = _released(validated)
    _rewrite_artifact(validated, paste, **changes)
    with pytest.raises(ValueError, match=needle):
        config_mod._validate_hinge_auto_release_gate(cfg)


@pytest.mark.parametrize(
    "field,value,needle",
    [
        ("calibration_calibrated_at", "2026-01-01", "does not bind"),
        ("calibration_sha256", "0" * 64, "does not bind"),
        ("device", "another-pixel", "does not bind"),
        ("hinge_version_name", "10.8.0", "does not bind"),
        ("frame_size_px", [1080, 2340], "does not bind"),
        ("schema_version", True, "integer 1"),
        ("production_run_id", "other-run", "does not bind production_run_id"),
        ("production_run_reference", "/abs/run-1", "repo-relative"),
        ("production_run_reference", "../escape/run-1", "escapes the repository"),
        ("verification_file", "../elsewhere.json", "escapes the repository"),
        ("debug_actions_sha256", "short", "not a SHA-256"),
        ("verified_at", " ", "nonempty text"),
    ],
)
def test_config_refuses_a_mapping_that_does_not_bind(validated, field, value, needle):
    _run, _artifact, paste, cfg = _released(validated)
    paste[field] = value
    with pytest.raises(ValueError, match=needle):
        config_mod._validate_hinge_auto_release_gate(cfg)


def test_config_refuses_a_changed_calibration_and_unknown_mapping_keys(validated):
    _run, _artifact, paste, cfg = _released(validated)
    cfg.apps["hinge"]["targeting_calibration"]["calibrated_at"] = "2026-10-24"
    with pytest.raises(ValueError, match="does not bind this calibration"):
        config_mod._validate_hinge_auto_release_gate(cfg)
    cfg.apps["hinge"]["targeting_calibration"]["calibrated_at"] = _calibration()[
        "calibrated_at"
    ]
    paste["acceptance"] = "I_ACCEPT"
    with pytest.raises(ValueError, match="must carry exactly"):
        config_mod._validate_hinge_auto_release_gate(cfg)


def test_config_requires_a_still_photo_licence(validated, monkeypatch):
    _run, _artifact, _paste, cfg = _released(validated)
    monkeypatch.setattr(
        config_mod, "hinge_targeting_unavailable_reason", lambda: "no licence"
    )
    with pytest.raises(ValueError, match="no licence"):
        config_mod._validate_hinge_auto_release_gate(cfg)


def test_the_training_validator_is_only_reached_in_auto(validated):
    _run, _artifact, paste, cfg = _released(validated)
    paste["verification_sha256"] = "0" * 64
    cfg.apps["hinge"]["mode"] = "training"
    cfg.mode = "training"
    config_mod._validate_hinge_auto_release_gate(cfg)


# --- end to end through the real config validator ---------------------------------------------


def test_round_trip_through_real_config_validation(monkeypatch, tmp_path):
    """A real Training config verifies, and its mapping licenses AUTO under full validate()."""
    from operation_love import config as c
    from tests.test_config import _assumption_config, _calibration as cfg_calibration
    from tests.test_config import _load

    monkeypatch.chdir(tmp_path)
    calibration = cfg_calibration()
    training = _load(
        _assumption_config(mode="training", targeting_calibration=calibration)
    )
    run = _Run(tmp_path, "run-1", training.apps["hinge"]["targeting_calibration"])
    run.add(
        "binding",
        {
            "ts": TS,
            "action": "observe_release_run_binding",
            "run_id": "run-1",
            "app": "hinge",
        },
    )
    run.add("env", _environment(run.calibration))
    run.chain("like", item=3, opener=LIKE_OPENER, profile_id="profile-like")
    run.chain("pass", item=6, opener=PASS_OPENER, profile_id="profile-pass")
    run.write()

    _artifact, paste = release.verify(
        cfg=training,
        run_dir=run.dir,
        run_id="run-1",
        out_dir=Path("ops/release/run-1/training-release"),
        store=_Store(),
    )

    auto = _assumption_config(
        mode="auto", targeting_calibration=calibration, training_release_evidence=paste
    )
    c.validate(_load(auto))
    auto["apps"]["hinge"]["observe_release_evidence"] = {"stale": "observe gate"}
    with pytest.raises(ValueError, match="exactly one release gate"):
        c.validate(_load(auto))
