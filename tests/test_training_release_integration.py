"""The Training release seam: what the REAL driver and Worker write, the REAL verifier accepts.

The verifier (tools/hinge_training_release.py) and the rows it parses (hinge.py, worker.py) were
written by different hands against one contract.  This drives a supervised Training LIKE and a
supervised Training PASS through the real Worker, the real TrainingActionBridge, the real
HingeDriver.open_session/like() code paths and the real DebugLog, then runs the verifier over the
directory that run left behind.

Only the phone and its vision are faked: the ADB link, the touch transport, the screenshot source,
and the detector/verifier functions that read pixels (composer and glyph locators, the profile
identity and item comparisons, the post-action deck-advance proof).  The profile READ is replaced
too (``next_profile``), because a real capture needs a real phone; it writes the ``capture`` row
and holds the item index exactly as the real capture leaves them.  Everything that emits a row
the verifier reads -- the verify pair, ``like_anchor``, the pre-send and resumed-send snapshots,
the three ``training_release_*`` facts, ``like_attempt``, the accepted advance probe, ``like`` and
``training_dislike`` -- runs unmodified.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import operation_love.worker as worker_module
from operation_love import config as config_mod
from operation_love.drivers import hinge
from operation_love.drivers.hinge import HingeDriver
from operation_love.perception.capture import Profile
from operation_love.ranker.store import SQLiteStore
from operation_love.training_actions import TrainingActionBridge
from operation_love.worker import Worker
from tests.test_training_worker import _Decider, _Opener, _Pacing, _Store
from tools import hinge_training_release as release

RUN_ID = "5eedc0ffee01"
_TIMEOUT_S = 15.0
_FAKE_IDENTITY = SimpleNamespace(known=True, fingerprint=(10, 20, 30), grid=(64, 16))
_COMPOSER = SimpleNamespace(
    comment_rect=SimpleNamespace(x0=95, y0=1124, x1=985, y1=1302, center=(540, 1213)),
    confirm_point=(690, 1390),
)
_PASS_X = (80, 80)
_OPENER = _Opener([]).pick.text


def _calibration() -> dict:
    return {
        "schema_version": 3,
        "hinge_version_name": "10.1.0",
        "frame_size_px": [1080, 2400],
        "composer_layout_id": "hinge_inline_v1",
        "item_selection_policy_id": "hinge_photos_only_v2",
        "identity_match_max_dist": 2.0,
        "inline_item_max_dist": 10.0,
        "device": "pixel",
        "calibrated_at": "2026-10-23T00:00:00Z",
        "identity_band": list(hinge.HINGE_SPEC.identity_band),
        "content_band": list(hinge.HINGE_SPEC.content_band),
    }


def _png(label: str) -> bytes:
    """A full-frame PNG that differs only by its label, drawn ABOVE the comment field.

    The comment field itself always shows the same draft, as an unedited one would, so the
    Like resume's draft proof sees the approved draft on every frame.
    """
    image = np.full((2400, 1080), 255, dtype=np.uint8)
    image[300:1090, 95:985] = 128
    cv2.putText(image, label, (103, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    cv2.putText(image, _OPENER, (103, 1184), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


class _Phone:
    """The fake ADB link: the live build and frame size, the foreground app, recorded input."""

    def __init__(self, version_name: str):
        self.version_name = version_name
        self.inputs: list[tuple] = []
        self._debug_after = _png("debug after")

    def devices(self):
        return ["pixel"]

    def screen_size(self):
        return (1080, 2400)

    def shell(self, command="", **_kwargs):
        if command.startswith("dumpsys package"):
            return f"    versionName={self.version_name}\n"
        return ""

    def foreground_package(self):
        return "co.hinge.app"

    def screencap(self):
        return self._debug_after

    def text(self, value):
        self.inputs.append(("text", value))

    def keyevent(self, keycode):
        self.inputs.append(("key", keycode))


class _Touch:
    def __init__(self, phone):
        self.phone = phone

    def open(self):
        pass

    def close(self):
        pass

    def tap(self, x, y):
        self.phone.inputs.append(("tap", (x, y)))


class _ReleaseDriver(HingeDriver):
    """The real HingeDriver with only the profile READ (and its paywall OCR) replaced."""

    def __init__(self, cfg, names):
        super().__init__(cfg)
        self._names = list(names)
        self._served = 0

    def blocked_reason(self):
        return None

    def out_of_profiles(self):
        return self._served >= len(self._names)

    def next_profile(self, *, should_stop=None):
        name = self._names[self._served]
        self._served += 1
        photo = _png(f"capture {name}")
        # What a real enumeration read leaves behind for like(): the held item index.
        self._current_item_index = SimpleNamespace(identity=_FAKE_IDENTITY)
        self._dbg.action(
            "capture",
            before=photo,
            photos=1,
            items=2,
            profile_name=name,
        )
        return Profile(photos=[photo], name=name, items=(b"item one", b"item two"))


def _install_fake_phone(monkeypatch, phone, *, pass_covered_on=()):
    """Fake the phone; ``pass_covered_on`` names the 1-based pass-glyph searches the IME hides."""
    screens = iter(range(1, 10_000))
    searches = iter(range(1, 10_000))
    monkeypatch.setattr(hinge, "Adb", lambda *_a, **_k: phone)
    monkeypatch.setattr(hinge, "UhidTouch", _Touch)
    monkeypatch.setattr(hinge, "PersistentUhidTouch", _Touch)
    monkeypatch.setattr(
        "operation_love.platforms.unavailable_reason", lambda *_a, **_k: None
    )
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(worker_module, "notify_training_decision_ready", lambda: True)
    # Pixel readers. Each answers what the real detector answers on the screen it stands for.
    monkeypatch.setattr(hinge, "locate_inline_composer", lambda *_a, **_k: _COMPOSER)
    monkeypatch.setattr(
        hinge,
        "_match_glyph",
        lambda *_a, **_k: [] if next(searches) in pass_covered_on else [_PASS_X],
    )
    monkeypatch.setattr(
        hinge,
        "compare_profile_identity",
        lambda *_a, **_k: SimpleNamespace(
            state="identity_match",
            distance=0.4,
            match_max=2.0,
            mismatched=False,
            unknown=False,
            matched=True,
            reason="",
        ),
    )
    monkeypatch.setattr(
        hinge,
        "verify_sheet_item",
        lambda _sheet, _payload, item, **_k: SimpleNamespace(
            state="verify_match",
            nearest_index=item,
            distance=1.5,
            bound=10.0,
            preview=SimpleNamespace(y0=300, y1=1090, x0=95, x1=985),
            matched=True,
            comparisons=(),
            reason="",
        ),
    )
    return lambda: _png(f"screen {next(screens)}")


def _stub_driver_vision(monkeypatch, driver, next_screen):
    monkeypatch.setattr(driver, "_screencap", lambda **_k: next_screen())
    monkeypatch.setattr(driver, "_verifiable_payload", lambda _item: object())
    monkeypatch.setattr(driver, "_confirm_payload_profile", lambda _item: None)
    monkeypatch.setattr(driver, "_navigate_to_model_item", lambda _item, **_k: (10, 10))
    monkeypatch.setattr(driver, "_await_sheet_open", lambda **_k: _COMPOSER)
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame: _COMPOSER)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_k: True)
    monkeypatch.setattr(driver, "_handle_rose_upsell", lambda: False)
    monkeypatch.setattr(driver, "_verify_like_landed", lambda _before, **_k: None)
    # Two settled frames that each prove a different card, so the REAL probe accepts them.
    monkeypatch.setattr(
        driver,
        "_training_dislike_surface_proof",
        lambda _frame, _item, *, diagnostics=None, **_k: ("identity", None),
    )
    monkeypatch.setattr(driver, "_changed", lambda _a, _b: False)


def _wait_until(predicate):
    deadline = time.monotonic() + _TIMEOUT_S
    pause = threading.Event()
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        pause.wait(0.01)
    pytest.fail("timed out waiting for the Training worker")


def _card(bridge, *, other_than=None):
    snapshot = bridge.snapshot(run_id=RUN_ID, app="hinge")
    for card in snapshot["checkpoints"]:
        if card["profile_token"] != other_than:
            return card
    return None


def _submit(bridge, card, command):
    ok, result, code = bridge.submit(
        {
            "command": command,
            "run_id": card["run_id"],
            "app": card["app"],
            "profile_token": card["profile_token"],
            "approval_token": card["approval_token"],
            "idempotency_token": f"{command}-{card['profile_token']}",
        }
    )
    assert (ok, code, result["status"]) == (True, 202, "queued")


def _supervised_run(
    tmp_path,
    monkeypatch,
    *,
    store=None,
    commands=("like", "dislike"),
    pass_covered_on=(),
):
    """One real Training session: open, then one reviewed decision per command, in order."""
    monkeypatch.chdir(tmp_path)
    calibration = _calibration()
    phone = _Phone(calibration["hinge_version_name"])
    next_screen = _install_fake_phone(
        monkeypatch, phone, pass_covered_on=pass_covered_on
    )
    cfg = SimpleNamespace(
        mode="training",
        apps={
            "hinge": {
                "serial": "pixel",
                "debug_log": True,
                "debug_dir": str(Path("data") / "hinge_debug"),
                "observe_touch_watch": False,
                "targeting_calibration": calibration,
            }
        },
        hinge_compatible_version=None,
    )
    driver = _ReleaseDriver(cfg, [f"Profile {n}" for n in range(len(commands))])
    _stub_driver_vision(monkeypatch, driver, next_screen)
    events: list = []
    store = store if store is not None else _Store(events)
    bridge = TrainingActionBridge()
    worker = Worker(
        "hinge",
        driver,
        _Decider(events),
        _Opener(events),
        store,
        RUN_ID,
        _Pacing(),
        threading.Event(),
        mode="training",
        training_action_bridge=bridge,
    )
    worker.start()
    previous = None
    try:
        for command in commands:
            card = _wait_until(
                lambda previous=previous: _card(bridge, other_than=previous)
            )
            _submit(bridge, card, command)
            previous = card["profile_token"]
        worker.join(_TIMEOUT_S)
    finally:
        worker.stop_event.set()
        worker.join(_TIMEOUT_S)
    assert not worker.is_alive()
    results = bridge.snapshot(run_id=RUN_ID, app="hinge")["results"]
    assert [(r["command"], r["status"]) for r in results] == [
        (command, "completed") for command in commands
    ]
    run_dir = tmp_path / "data" / "hinge_debug" / RUN_ID
    rows = [
        json.loads(line)
        for line in (run_dir / "actions.jsonl").read_text().splitlines()
    ]
    return SimpleNamespace(
        calibration=calibration,
        run_dir=run_dir,
        rows=rows,
        store=store,
        phone=phone,
    )


def _row(rows, action, nth=0):
    return [row for row in rows if row["action"] == action][nth]


class _SummaryStore:
    def __init__(self):
        self.calls = []

    def training_release_persistence_summary(self, run_id, app, **kwargs):
        self.calls.append((run_id, app, kwargs))
        return {key: 1 for key in release._STORE_KEYS}


# --- the seam itself ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "commands",
    [("like", "dislike"), ("dislike", "like")],
    ids=["like-first", "pass-first"],
)
def test_a_real_supervised_like_and_pass_satisfy_the_release_verifier(
    tmp_path, monkeypatch, commands
):
    run = _supervised_run(tmp_path, monkeypatch, commands=commands)

    # The rows the verifier names were written by the real code under test, in its order.
    assert [row["action"] for row in run.rows[:2]] == [
        "observe_release_run_binding",
        "training_session_environment",
    ]
    debug = release._verify_debug_run(run.run_dir, RUN_ID, run.calibration)

    # The chains it chose are the ones this run produced, bound to the store's own profile ids.
    profile_of = {liked: profile_id for profile_id, liked, *_ in run.store.profiles}
    like_profile, pass_profile = profile_of[True], profile_of[False]
    assert debug["like"]["profile_id"] == like_profile
    assert debug["pass"]["profile_id"] == pass_profile
    resumed = _row(run.rows, "auto_opener_resumed_send")
    assert debug["like"]["send_evidence_id"] == resumed["evidence_id"]
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
    assert (
        debug["debug_actions_sha256"]
        == hashlib.sha256((run.run_dir / "actions.jsonl").read_bytes()).hexdigest()
    )


def test_the_real_run_mints_a_mapping_the_auto_gate_accepts(tmp_path, monkeypatch):
    run = _supervised_run(tmp_path, monkeypatch)
    # Config validation itself is covered by tests/test_hinge_training_release.py.
    monkeypatch.setattr(config_mod, "validate", lambda _cfg: None)
    monkeypatch.setattr(config_mod, "hinge_targeting_unavailable_reason", lambda: None)
    training_cfg = SimpleNamespace(
        enabled_apps=["hinge"],
        mode="training",
        hinge_compatible_version=None,
        apps={"hinge": {"mode": "training", "targeting_calibration": run.calibration}},
    )
    store = _SummaryStore()

    artifact, paste = release.verify(
        cfg=training_cfg,
        run_dir=Path("data") / "hinge_debug" / RUN_ID,
        run_id=RUN_ID,
        out_dir=Path("ops") / "release" / RUN_ID / "training-release",
        store=store,
    )

    like_profile, pass_profile = (profile_id for profile_id, *_ in run.store.profiles)
    assert store.calls == [
        (
            RUN_ID,
            "hinge",
            {
                "like_profile_id": like_profile,
                "pass_profile_id": pass_profile,
                "like_send_evidence_id": _row(run.rows, "auto_opener_resumed_send")[
                    "evidence_id"
                ],
            },
        )
    ]
    assert artifact["production_run_reference"] == f"data/hinge_debug/{RUN_ID}"
    content = (tmp_path / paste["verification_file"]).read_bytes()
    for secret in (_OPENER, like_profile, pass_profile, "Profile 0"):
        assert secret.encode() not in content

    auto_cfg = SimpleNamespace(
        enabled_apps=["hinge"],
        mode="auto",
        hinge_compatible_version=None,
        apps={
            "hinge": {
                "mode": "auto",
                "targeting_calibration": run.calibration,
                "training_release_evidence": paste,
            }
        },
    )
    config_mod._validate_hinge_auto_release_gate(auto_cfg)


def test_the_worker_writes_exactly_the_rows_the_store_summary_counts(
    tmp_path, monkeypatch
):
    """The verifier queries by the profile ids the debug chain names; the store must hold them."""
    store = SQLiteStore(tmp_path / "store.db")
    try:
        run = _supervised_run(tmp_path, monkeypatch, store=store)
        debug = release._verify_debug_run(run.run_dir, RUN_ID, run.calibration)
        summary = store.training_release_persistence_summary(
            RUN_ID,
            "hinge",
            like_profile_id=debug["like"]["profile_id"],
            pass_profile_id=debug["pass"]["profile_id"],
            like_send_evidence_id=debug["like"]["send_evidence_id"],
        )
    finally:
        store.close()
    assert summary["like_labels"] == 1
    assert summary["like_decisions"] == 1
    assert summary["pass_labels"] == 1
    assert summary["pass_decisions"] == 1
    # The test opener fake writes no `openers` row, and SQLite has no send-evidence table.
    assert summary["like_openers"] == 0
    assert summary["like_send_evidence"] == 0


@pytest.mark.parametrize(
    ("covered_search", "decision_row"),
    [
        # Pass-glyph searches in order: Like pre-send (1), Like resume (2), Pass pre-send (3),
        # Pass resume (4).  Covering a RESUME search is what a reviewer re-focusing the field does.
        (2, "auto_opener_resumed_send"),
        (4, "training_dislike"),
    ],
    ids=["like-resume", "pass-resume"],
)
def test_a_recovered_reviewer_keyboard_still_yields_release_evidence(
    tmp_path, monkeypatch, covered_search, decision_row
):
    run = _supervised_run(tmp_path, monkeypatch, pass_covered_on=(covered_search,))

    recovery = _row(run.rows, "training_checkpoint_keyboard_recovery")
    assert [kind for kind, _value in run.phone.inputs].count("key") == 3
    debug = release._verify_debug_run(run.run_dir, RUN_ID, run.calibration)
    assert debug["like"] is not None and debug["pass"] is not None
    if decision_row == "auto_opener_resumed_send":
        # The resumed send is bound to the RECOVERED frame, which the verifier re-hashed.
        resumed = _row(run.rows, decision_row)
        assert (run.run_dir / resumed["kept_before"]).read_bytes() == (
            run.run_dir / recovery["after"]
        ).read_bytes()


def test_a_like_alone_is_not_a_release(tmp_path, monkeypatch):
    run = _supervised_run(tmp_path, monkeypatch, commands=("like",))
    with pytest.raises(release.ReleaseEvidenceRefused, match="PASS chain"):
        release._verify_debug_run(run.run_dir, RUN_ID, run.calibration)
