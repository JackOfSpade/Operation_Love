"""Non-manual OBSERVE release artifacts are distinct, bound, and independently reviewed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from operation_love import config as config_mod
from tools import hinge_observe_ai_release as release


def _canon(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _calibration():
    return {"schema_version": 3, "device": "pixel", "hinge_version_name": "9.134.0",
            "frame_size_px": [1080, 2400], "composer_layout_id": "hinge_inline_v1",
            "item_selection_policy_id": "hinge_photos_only_v1", "identity_match_max_dist": 1.0,
            "inline_item_max_dist": 2.0, "calibrated_at": "2026-08-14T00:00:00Z",
            "identity_band": [0.0, 0.0, 1.0, 0.1], "content_band": [0.1, 0.9]}


def _cfg(calibration, mode="observe"):
    return SimpleNamespace(enabled_apps=["hinge"], mode=mode, db_file=Path("store.db"),
                           storage=SimpleNamespace(backend="sqlite", bigquery={}),
                           apps={"hinge": {"mode": mode, "targeting_calibration": calibration}})


def _controller_cfg(calibration):
    cfg = _cfg(calibration)
    cfg.apps["hinge"].update({"observe_evidence_source": "external_ai_review",
                               "ai_reviewed_observe_controller": {
                                   "schema_version": 1, "source": "external_ai_review",
                                   "acceptance": release.ACCEPTANCE,
                                   "executor": {"model": "gpt", "id": "driver", "version": "v1",
                                                "process": "controller"}}})
    return cfg


def _inputs(tmp_path: Path, *, source="automation"):
    run_id = "ai-run"
    debug = tmp_path / "data" / "hinge_debug" / run_id
    debug.mkdir(parents=True)
    shots = {name: f"shot-{name}".encode() for name in
             ("pass", "anchor", "open", "attempt", "like", "landed")}
    for name, content in shots.items():
        (debug / f"{name}.png").write_bytes(content)
    rows = [
        {"ts": "now", "action": "observe_release_run_binding", "run_id": run_id, "app": "hinge"},
        {"action": "observe_decision", "decision": "pass", "reviewed": True, "before": "pass.png"},
        {"action": "observe_release_hub_pre_tap_published"},
        {"action": "observe_like_anchor", "before": "anchor.png"},
        {"action": "observe_release_post_tap_item_verified"},
        {"action": "observe_reviewed_open", "verified": True, "model_item_index": 3,
         "before": "open.png"},
        {"action": "observe_reviewed_like_attempt", "verified": True, "model_item_index": 3,
         "opener_chars": 12, "before": "attempt.png"},
        {"action": "observe_decision", "decision": "like", "reviewed": True,
         "model_item_index": 3, "opener_chars": 12, "after": "like.png"},
        {"action": "observe_reviewed_like", "verified": True, "model_item_index": 3,
         "opener_chars": 12, "after": "landed.png"},
    ]
    raw = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    (debug / "actions.jsonl").write_bytes(raw)
    events = []
    for name, index in (("pass", 1), ("hub_pre_tap_published", 2), ("like_anchor", 3),
                        ("post_tap_item_verified", 4), ("reviewed_open", 5),
                        ("reviewed_like_attempt", 6), ("like", 7), ("reviewed_like", 8)):
        events.append({"name": name, "index": index, "row_sha256": _canon(rows[index])})
    provenance = {"schema_version": 1, "kind": "hinge_ai_observe_action_provenance", "completed": True,
                  "human_ground_truth": False, "source": source,
                  "production_run_reference": str(debug.relative_to(tmp_path)), "production_run_id": run_id,
                  "debug_actions_sha256": hashlib.sha256(raw).hexdigest(), "device": "pixel",
                  "hinge_version_name": "9.134.0", "frame_size_px": [1080, 2400],
                  "executor": {"model": "gpt-5.6-sol", "id": "executor", "version": "v1", "process": "driver"},
                  "events": events}
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance))

    class Store:
        def ai_observe_release_persistence_summary(self, actual_run, app, actual_source):
            assert (actual_run, app, actual_source) == (run_id, "hinge", source)
            return {"ai_pass_labels": 1, "ai_like_labels": 1, "ai_pass_decisions": 1,
                    "ai_like_decisions": 1, "successful_hinge_openers": 1}
    return debug, run_id, provenance_path, Store(), provenance


def _review(tmp_path, debug, run_id, provenance):
    return release.review(run_dir=debug, run_id=run_id, provenance_path=provenance,
                          out_dir=tmp_path / "review", reviewer={"model": "gpt-5.6-terra",
                          "id": "reviewer", "version": "v1", "process": "offline-review"})


def test_ai_release_requires_complete_bound_non_manual_review_then_config_accepts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    debug, run_id, provenance, store, _ = _inputs(tmp_path)
    _review(tmp_path, debug, run_id, provenance)
    cfg = _cfg(_calibration())
    artifact, paste = release.verify(cfg=cfg, run_dir=debug, run_id=run_id, provenance_path=provenance,
                                     review_path=tmp_path / "review" / "hinge_ai_observe_independent_review.json",
                                     out_dir=tmp_path / "release", acceptance=release.ACCEPTANCE, store=store)
    assert artifact["human_ground_truth"] is False
    assert artifact["source"] == "automation"
    cfg.mode = cfg.apps["hinge"]["mode"] = "auto"
    cfg.apps["hinge"]["ai_reviewed_observe_release_evidence"] = paste
    config_mod._validate_hinge_auto_release_gate(cfg)


def test_provenance_compiler_derives_only_ordered_runtime_facts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    debug, run_id, _provenance, _store, _ = _inputs(tmp_path)
    cfg = _controller_cfg(_calibration())
    artifact = release.record_provenance(
        cfg=cfg, run_dir=debug, run_id=run_id, out_dir=tmp_path / "compiled",
        source="external_ai_review", acceptance=release.ACCEPTANCE,
        executor={"model": "gpt", "id": "driver", "version": "v1", "process": "controller"})
    assert artifact["human_ground_truth"] is False
    assert [event["name"] for event in artifact["events"]] == [name for name, _ in release.REQUIRED_EVENTS]
    assert (tmp_path / "compiled" / "hinge_ai_observe_action_provenance.json").is_file()


@pytest.mark.parametrize("mutation, match", [
    ("manual_pass", "ordered runtime fact for pass"),
    ("no_landing", "ordered runtime fact for reviewed_like"),
    ("mismatched_target", "one exact target item"),
    ("empty_opener", "nonempty sent opener"),
])
def test_bridge_release_refuses_manual_or_unbound_send_traces(monkeypatch, tmp_path, mutation, match):
    """A passive like_sending row cannot substitute for bridge-owned send proof."""
    monkeypatch.chdir(tmp_path)
    debug, run_id, _provenance_path, _store, _provenance = _inputs(tmp_path)
    rows = [json.loads(line) for line in (debug / "actions.jsonl").read_text().splitlines()]
    if mutation == "manual_pass":
        rows[1]["reviewed"] = False
    elif mutation == "no_landing":
        rows.pop()
        rows.append({"action": "observe_waiting", "reason": "like_sending", "before": "attempt.png"})
    elif mutation == "mismatched_target":
        rows[6]["model_item_index"] = 1
    else:
        rows[6]["opener_chars"] = 0
    raw = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    (debug / "actions.jsonl").write_bytes(raw)
    with pytest.raises(release.AIReleaseEvidenceRefused, match=match):
        release.record_provenance(
            cfg=_controller_cfg(_calibration()), run_dir=debug, run_id=run_id,
            out_dir=tmp_path / "compiled", source="external_ai_review", acceptance=release.ACCEPTANCE,
            executor={"model": "gpt", "id": "driver", "version": "v1", "process": "controller"})


@pytest.mark.parametrize("mutation, match", [
    ("human", "human_ground_truth=false"),
    ("wrong_hash", "row hash"),
    ("incomplete", "completed"),
])
def test_ai_release_refuses_forged_or_incomplete_action_provenance(monkeypatch, tmp_path, mutation, match):
    monkeypatch.chdir(tmp_path)
    debug, run_id, provenance_path, _store, provenance = _inputs(tmp_path)
    if mutation == "human":
        provenance["human_ground_truth"] = True
    elif mutation == "wrong_hash":
        provenance["events"][0]["row_sha256"] = "0" * 64
    else:
        provenance["completed"] = False
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(release.AIReleaseEvidenceRefused, match=match):
        _review(tmp_path, debug, run_id, provenance_path)


def test_ai_release_refuses_same_reviewer_executor_identity_and_process(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    debug, run_id, provenance, _store, _ = _inputs(tmp_path)
    with pytest.raises(release.AIReleaseEvidenceRefused, match="must not share"):
        release.review(run_dir=debug, run_id=run_id, provenance_path=provenance, out_dir=tmp_path / "review",
                       reviewer={"model": "gpt", "id": "executor", "version": "v1", "process": "driver"})


def test_ai_release_refuses_manual_store_rows_and_mismatched_review(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    debug, run_id, provenance, store, _ = _inputs(tmp_path)
    _review(tmp_path, debug, run_id, provenance)

    class ManualOnlyStore:
        def ai_observe_release_persistence_summary(self, *_):
            return {"ai_pass_labels": 0, "ai_like_labels": 0, "ai_pass_decisions": 0,
                    "ai_like_decisions": 0, "successful_hinge_openers": 1}
    with pytest.raises(release.AIReleaseEvidenceRefused, match="complete non-manual"):
        release.verify(cfg=_cfg(_calibration()), run_dir=debug, run_id=run_id, provenance_path=provenance,
                       review_path=tmp_path / "review" / "hinge_ai_observe_independent_review.json",
                       out_dir=tmp_path / "release-a", acceptance=release.ACCEPTANCE, store=ManualOnlyStore())
    review_path = tmp_path / "review" / "hinge_ai_observe_independent_review.json"
    review = json.loads(review_path.read_text())
    review["production_run_id"] = "other"
    review_path.write_text(json.dumps(review))
    with pytest.raises(release.AIReleaseEvidenceRefused, match="exact production run"):
        release.verify(cfg=_cfg(_calibration()), run_dir=debug, run_id=run_id, provenance_path=provenance,
                       review_path=review_path, out_dir=tmp_path / "release-b", acceptance=release.ACCEPTANCE, store=store)


def test_ai_release_refuses_provenance_from_another_device_or_build(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    debug, run_id, provenance_path, store, provenance = _inputs(tmp_path)
    _review(tmp_path, debug, run_id, provenance_path)
    provenance["hinge_version_name"] = "9.999.0"
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(release.AIReleaseEvidenceRefused, match="device/build/framebuffer"):
        release.verify(cfg=_cfg(_calibration()), run_dir=debug, run_id=run_id, provenance_path=provenance_path,
                       review_path=tmp_path / "review" / "hinge_ai_observe_independent_review.json",
                       out_dir=tmp_path / "release", acceptance=release.ACCEPTANCE, store=store)


def test_config_rejects_cross_mode_and_missing_explicit_acceptance(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    debug, run_id, provenance, store, _ = _inputs(tmp_path)
    _review(tmp_path, debug, run_id, provenance)
    cfg = _cfg(_calibration())
    _artifact, paste = release.verify(cfg=cfg, run_dir=debug, run_id=run_id, provenance_path=provenance,
                                      review_path=tmp_path / "review" / "hinge_ai_observe_independent_review.json",
                                      out_dir=tmp_path / "release", acceptance=release.ACCEPTANCE, store=store)
    cfg.mode = cfg.apps["hinge"]["mode"] = "auto"
    cfg.apps["hinge"]["ai_reviewed_observe_release_evidence"] = paste
    cfg.apps["hinge"]["observe_release_evidence"] = {"not": "manual"}
    with pytest.raises(ValueError, match="exactly one release gate"):
        config_mod._validate_hinge_auto_release_gate(cfg)
    del cfg.apps["hinge"]["observe_release_evidence"]
    paste["acceptance"] = "yes"
    with pytest.raises(ValueError, match="exact explicit acceptance"):
        config_mod._validate_hinge_auto_release_gate(cfg)
