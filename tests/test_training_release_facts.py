"""The supervised-Training release facts: driver rows and the Worker calls that feed them.

A Training run is release evidence only if its debug log names the build, frame, device and
calibration it ran against, and records each human decision from checkpoint publication through
durable persistence.  These rows carry digests and ids only.  Recording them is best effort on
the Worker side: a missing or refused fact only makes a later release verification refuse, it
never stops or fails the Training run that produced it.
"""

from __future__ import annotations

import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

import operation_love.worker as worker_module
from operation_love import config as config_module
from operation_love.drivers import hinge
from operation_love.drivers.debuglog import DebugLog
from operation_love.drivers.hinge import AndroidDriver, HingeDriver
from operation_love.worker import Worker
from tests.test_hinge_auto_presend_snapshot import _driver
from tests.test_training_worker import (
    _TIMEOUT_S,
    _Decider,
    _Opener,
    _Pacing,
    _RecordingBridge,
    _SilentBridge,
    _Store,
    _TrainingDriver,
    _checkpoint,
    _submit,
    _wait_until,
)

_SHA_A = "a" * 64
_SHA_B = "0123456789abcdef" * 4
_SHA_C = "f" * 64


@pytest.fixture(autouse=True)
def _quiet_host(monkeypatch):
    """No real host notification and no real sleeping in any test here."""
    monkeypatch.setattr(worker_module, "notify_training_decision_ready", lambda: True)
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bare_driver(**app_cfg):
    class _Cfg:
        apps = {"hinge": {"serial": "pixel", "halt_on_error": False, **app_cfg}}

    return HingeDriver(_Cfg())


def _rows(debug: DebugLog) -> list[dict]:
    log = debug.dir / "actions.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def _without_ts(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "ts"}


def _valid_fields(fact: str) -> dict:
    return {
        "checkpoint_published": {
            "evidence_id": "pre-send-evidence",
            "profile_token_sha256": _SHA_A,
            "approval_token_sha256": _SHA_B,
        },
        "decision_claimed": {
            "command": "like",
            "evidence_id": "pre-send-evidence",
            "idempotency_token_sha256": _SHA_C,
            "waited_s": 4.0,
        },
        "action_persisted": {
            "decision": "like",
            "evidence_id": "pre-send-evidence",
            "send_evidence_id": "resumed-send-evidence",
            "profile_id": "store-profile-id",
        },
    }[fact]


# --- AndroidDriver.training_release_fact ---------------------------------------------------


def test_each_fact_logs_exactly_its_whitelisted_fields_without_device_io(tmp_path):
    driver = _bare_driver()
    assert driver._adb is None  # transport-free: nothing here may screencap or shell
    driver._dbg = DebugLog(str(tmp_path / "debug"), run_id="facts")

    driver.training_release_fact(
        "checkpoint_published",
        evidence_id=None,
        profile_token_sha256=_SHA_A,
        approval_token_sha256=_SHA_B,
    )
    driver.training_release_fact(
        "decision_claimed",
        command="dislike",
        evidence_id="pre-send-evidence",
        idempotency_token_sha256=_SHA_C,
        waited_s=12.349,
    )
    driver.training_release_fact(
        "action_persisted",
        decision="dislike",
        evidence_id="pre-send-evidence",
        send_evidence_id=None,
        profile_id="store-profile-id",
    )

    assert [_without_ts(row) for row in _rows(driver._dbg)] == [
        {
            "action": "training_release_checkpoint_published",
            "evidence_id": None,
            "profile_token_sha256": _SHA_A,
            "approval_token_sha256": _SHA_B,
        },
        {
            "action": "training_release_decision_claimed",
            "command": "dislike",
            "evidence_id": "pre-send-evidence",
            "idempotency_token_sha256": _SHA_C,
            "waited_s": 12.3,
        },
        {
            "action": "training_release_action_persisted",
            "decision": "dislike",
            "evidence_id": "pre-send-evidence",
            "send_evidence_id": None,
            "profile_id": "store-profile-id",
        },
    ]
    assert list(driver._dbg.dir.glob("*.png")) == []


def test_waited_seconds_are_logged_as_a_float_rounded_to_a_tenth(tmp_path):
    driver = _bare_driver()
    driver._dbg = DebugLog(str(tmp_path / "debug"), run_id="facts")
    for waited in (3, 0.04, 7.25):
        driver.training_release_fact(
            "decision_claimed",
            **{**_valid_fields("decision_claimed"), "waited_s": waited},
        )
    logged = [row["waited_s"] for row in _rows(driver._dbg)]
    assert logged == [3.0, 0.0, 7.2]
    assert all(type(value) is float for value in logged)


@pytest.mark.parametrize(
    "fact",
    [
        "",
        "hub_pre_tap_published",
        "training_release_checkpoint_published",
        "Checkpoint_Published",
        "action_landed",
    ],
)
def test_an_unknown_fact_is_refused_and_logs_nothing(tmp_path, fact):
    driver = _bare_driver()
    driver._dbg = DebugLog(str(tmp_path / "debug"), run_id="facts")
    with pytest.raises(ValueError, match="unknown training release fact"):
        driver.training_release_fact(fact, **_valid_fields("checkpoint_published"))
    assert _rows(driver._dbg) == []


@pytest.mark.parametrize(
    ("fact", "change"),
    [
        ("checkpoint_published", {"drop": "evidence_id"}),
        ("checkpoint_published", {"add": ("profile_token", "raw-token")}),
        ("decision_claimed", {"drop": "waited_s"}),
        ("decision_claimed", {"add": ("idempotency_token", "raw-token")}),
        ("action_persisted", {"drop": "send_evidence_id"}),
        ("action_persisted", {"add": ("profile_name", "Ari")}),
        ("action_persisted", {"add": ("opener", "A precise typed opener")}),
    ],
)
def test_a_missing_or_extra_field_is_refused_and_logs_nothing(tmp_path, fact, change):
    driver = _bare_driver()
    driver._dbg = DebugLog(str(tmp_path / "debug"), run_id="facts")
    fields = _valid_fields(fact)
    if "drop" in change:
        del fields[change["drop"]]
    else:
        name, value = change["add"]
        fields[name] = value
    with pytest.raises(ValueError, match="must carry exactly"):
        driver.training_release_fact(fact, **fields)
    assert _rows(driver._dbg) == []


@pytest.mark.parametrize(
    ("fact", "field", "value"),
    [
        ("checkpoint_published", "evidence_id", 7),
        ("checkpoint_published", "profile_token_sha256", "A" * 64),
        ("checkpoint_published", "profile_token_sha256", "a" * 63),
        ("checkpoint_published", "approval_token_sha256", None),
        ("checkpoint_published", "approval_token_sha256", "raw-approval-token"),
        ("decision_claimed", "command", "superlike"),
        ("decision_claimed", "command", "Like"),
        ("decision_claimed", "command", None),
        ("decision_claimed", "idempotency_token_sha256", None),
        ("decision_claimed", "waited_s", -0.1),
        ("decision_claimed", "waited_s", float("nan")),
        ("decision_claimed", "waited_s", float("inf")),
        ("decision_claimed", "waited_s", True),
        ("decision_claimed", "waited_s", "4.0"),
        ("decision_claimed", "waited_s", None),
        ("action_persisted", "decision", "pass"),
        ("action_persisted", "evidence_id", b"pre-send-evidence"),
        ("action_persisted", "send_evidence_id", 5),
        ("action_persisted", "profile_id", ""),
        ("action_persisted", "profile_id", None),
    ],
)
def test_a_malformed_value_is_refused_and_logs_nothing(tmp_path, fact, field, value):
    driver = _bare_driver()
    driver._dbg = DebugLog(str(tmp_path / "debug"), run_id="facts")
    with pytest.raises(ValueError, match=f"invalid {field!r}"):
        driver.training_release_fact(fact, **{**_valid_fields(fact), field: value})
    assert _rows(driver._dbg) == []


def test_validation_does_not_depend_on_an_open_debug_log():
    driver = _bare_driver()
    assert driver._dbg is None
    driver.training_release_fact(
        "action_persisted", **_valid_fields("action_persisted")
    )
    with pytest.raises(ValueError):
        driver.training_release_fact("unknown")
    with pytest.raises(ValueError):
        driver.training_release_fact(
            "action_persisted", **{**_valid_fields("action_persisted"), "decision": "x"}
        )


# --- canonical calibration digest ------------------------------------------------------------


def _calibration(**overrides) -> dict:
    calibration = {
        "schema_version": 3,
        "hinge_version_name": "10.1.0",
        "frame_size_px": [1080, 2400],
        "composer_layout_id": "hinge_inline_v1",
        "item_selection_policy_id": "hinge_photos_only_v2",
        "identity_match_max_dist": 2.0,
        "inline_item_max_dist": 10.0,
        "device": "pixel",
        "calibrated_at": "2026-08-12",
        "identity_band": list(hinge.HINGE_SPEC.identity_band),
        "content_band": list(hinge.HINGE_SPEC.content_band),
    }
    calibration.update(overrides)
    return calibration


def test_the_calibration_digest_is_configs_canonical_sha256():
    calibration = _calibration(note="café — measured")
    reordered = dict(reversed(list(calibration.items())))
    expected = hashlib.sha256(
        json.dumps(
            calibration, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()

    assert config_module._canonical_sha256(calibration) == expected
    assert hinge._configured_calibration_sha256(calibration) == expected
    assert hinge._configured_calibration_sha256(reordered) == expected
    assert (
        hinge._configured_calibration_sha256(_calibration(calibrated_at="2026-08-13"))
        != expected
    )


@pytest.mark.parametrize("raw", [None, [], "calibration", {"bad": {1, 2}}])
def test_an_absent_or_unserializable_calibration_has_no_digest(raw):
    assert hinge._configured_calibration_sha256(raw) is None


# --- training_session_environment ----------------------------------------------------------


class _SessionAdb:
    def __init__(self, version_name):
        self.version_name = version_name

    def devices(self):
        return ["pixel"]

    def screen_size(self):
        return (1080, 2400)

    def shell(self, command="", **_):
        if command.startswith("dumpsys package"):
            return f"    versionName={self.version_name}\n"
        return ""


class _FakeTouch:
    def __init__(self, adb):
        self.adb = adb

    def open(self):
        pass

    def close(self):
        pass


def _open_session(
    tmp_path,
    monkeypatch,
    *,
    training=True,
    live_version="10.1.0",
    calibration=None,
    compatible_version=None,
    run_id="training-run-01",
):
    monkeypatch.setattr(hinge, "Adb", lambda *_a, **_k: _SessionAdb(live_version))
    monkeypatch.setattr(hinge, "UhidTouch", _FakeTouch)
    monkeypatch.setattr(hinge, "PersistentUhidTouch", _FakeTouch)
    monkeypatch.setattr(
        "operation_love.platforms.unavailable_reason", lambda *_a, **_k: None
    )
    debug_dir = tmp_path / "hinge_debug"
    app_cfg = {
        "serial": "pixel",
        "debug_log": True,
        "debug_dir": str(debug_dir),
        "observe_touch_watch": False,
    }
    if calibration is not None:
        app_cfg["targeting_calibration"] = calibration
    cfg = SimpleNamespace(
        mode="training",
        apps={"hinge": app_cfg},
        hinge_compatible_version=compatible_version,
    )
    driver = HingeDriver(cfg)
    driver.bind_debug_run(run_id)
    if training:
        driver.begin_training_session()
    driver.open_session()
    return driver, _rows(driver._dbg)


def test_open_session_writes_the_environment_right_after_the_run_binding(
    tmp_path, monkeypatch
):
    calibration = _calibration()
    driver, rows = _open_session(tmp_path, monkeypatch, calibration=calibration)

    assert [row["action"] for row in rows] == [
        "observe_release_run_binding",
        "training_session_environment",
    ]
    # The binding row stays row 0 with exactly its historical fields.
    assert _without_ts(rows[0]) == {
        "action": "observe_release_run_binding",
        "run_id": "training-run-01",
        "app": "hinge",
    }
    assert _without_ts(rows[1]) == {
        "action": "training_session_environment",
        "session_mode": "training",
        "hinge_version_name": "10.1.0",
        "frame_size_px": [1080, 2400],
        "device": "pixel",
        "calibration_sha256": config_module._canonical_sha256(calibration),
        "compatible_version": None,
        "targeting_available": True,
    }
    assert driver.targeting_calibration is not None


def test_the_environment_reports_a_live_build_the_calibration_does_not_license(
    tmp_path, monkeypatch
):
    calibration = _calibration()
    driver, rows = _open_session(
        tmp_path, monkeypatch, calibration=calibration, live_version="10.2.0"
    )

    environment = rows[1]
    assert environment["action"] == "training_session_environment"
    assert environment["hinge_version_name"] == "10.2.0"
    assert environment["targeting_available"] is False
    # Still the digest of the mapping the operator configured, not of a refused parse.
    assert environment["calibration_sha256"] == config_module._canonical_sha256(
        calibration
    )
    assert driver.targeting_calibration is None


def test_the_environment_names_a_compatibility_review_build(tmp_path, monkeypatch):
    _driver_obj, rows = _open_session(
        tmp_path,
        monkeypatch,
        calibration=_calibration(),
        live_version="10.2.0",
        compatible_version="10.2.0",
    )
    environment = rows[1]
    assert environment["hinge_version_name"] == "10.2.0"
    assert environment["compatible_version"] == "10.2.0"
    assert environment["targeting_available"] is True


def test_the_environment_without_a_calibration_or_training_session(
    tmp_path, monkeypatch
):
    _driver_obj, rows = _open_session(tmp_path, monkeypatch, training=False)
    assert _without_ts(rows[1]) == {
        "action": "training_session_environment",
        "session_mode": "observe",
        # No calibration means no binding read, so the live build is not known here.
        "hinge_version_name": None,
        "frame_size_px": None,
        "device": "pixel",
        "calibration_sha256": None,
        "compatible_version": None,
        "targeting_available": False,
    }


# --- retained successor proof and protected release runs ------------------------------------


def test_the_accepted_advance_probe_retains_both_settled_frames(tmp_path, monkeypatch):
    driver, _adb = _driver([])
    driver._dbg = DebugLog(str(tmp_path), run_id="advance-probe-pair")
    frames = iter((b"FIRST_SETTLED", b"SECOND_SETTLED"))
    monkeypatch.setattr(driver, "_screencap", lambda **_kw: next(frames))
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        driver,
        "_training_dislike_surface_proof",
        lambda _frame, _item, diagnostics=None: ("identity", None),
    )
    monkeypatch.setattr(driver, "_changed", lambda _a, _b: False)

    assert driver._verify_training_deck_advanced(1) == "identity"

    probe = next(
        row for row in _rows(driver._dbg) if row["action"] == "training_advance_probe"
    )
    assert probe["outcome"] == "accepted"
    assert probe["kept_before"] == probe["before"]
    assert probe["kept_after"] == probe["after"]
    assert (driver._dbg.dir / probe["kept_before"]).read_bytes() == b"FIRST_SETTLED"
    assert (driver._dbg.dir / probe["kept_after"]).read_bytes() == b"SECOND_SETTLED"


def test_both_release_evidence_runs_are_protected_from_retention():
    driver = _bare_driver(
        debug_protect_runs=["fixture_run_01"],
        observe_release_evidence={"production_run_id": "observe_run_01"},
        training_release_evidence={"production_run_id": "training_run_01"},
    )
    assert driver.debug_protect_runs == frozenset(
        {"fixture_run_01", "observe_run_01", "training_run_01"}
    )
    assert _bare_driver(
        training_release_evidence={"production_run_id": "training_run_01"}
    ).debug_protect_runs == frozenset({"training_run_01"})


@pytest.mark.parametrize(
    "evidence", [None, {}, {"production_run_id": ""}, {"production_run_id": 7}]
)
def test_an_absent_or_malformed_training_release_run_protects_nothing(evidence):
    driver = _bare_driver(training_release_evidence=evidence)
    assert driver.debug_protect_runs == frozenset()


# --- Worker: the three facts across one reviewed decision -----------------------------------


class _RecordingDebug:
    def __init__(self, events):
        self.events = events

    def action(self, name, **fields):
        self.events.append(("fact", name, fields))


class _FactDriver(_TrainingDriver):
    """The one-card Training seam with the REAL driver fact whitelist bound to it."""

    training_release_fact = AndroidDriver.training_release_fact

    def __init__(self, events, **kwargs):
        super().__init__(events, **kwargs)
        self._dbg = _RecordingDebug(events)


class _RaisingFactDriver(_TrainingDriver):
    def __init__(self, events, error, **kwargs):
        super().__init__(events, **kwargs)
        self.error = error

    def training_release_fact(self, fact, **fields):
        self.events.append(("fact_attempt", fact))
        raise self.error


class _NonCallableFactDriver(_TrainingDriver):
    training_release_fact = "not a method"


def _fact_worker(driver, events, *, bridge=None, store=None):
    store = store or _Store(events)
    bridge = bridge or _RecordingBridge(events)
    worker = Worker(
        "hinge",
        driver,
        _Decider(events),
        _Opener(events),
        store,
        "training-run",
        _Pacing(),
        threading.Event(),
        mode="training",
        training_action_bridge=bridge,
    )
    return worker, store, bridge


def _facts(events):
    return [event for event in events if event[0] == "fact"]


@pytest.mark.parametrize("command", ["like", "dislike"])
def test_a_reviewed_decision_emits_the_three_release_facts_in_order(command):
    events = []
    driver = _FactDriver(events)
    worker, store, bridge = _fact_worker(driver, events)
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _submit(bridge, card, command)
    worker.join(_TIMEOUT_S)
    assert not worker.is_alive()

    facts = _facts(events)
    assert [name for _kind, name, _fields in facts] == [
        "training_release_checkpoint_published",
        "training_release_decision_claimed",
        "training_release_action_persisted",
    ]
    published, claimed, persisted = (fields for _kind, _name, fields in facts)
    # The driver's checkpoint callback passes evidence id "verified-composer"; the bridge copies
    # it onto the published card, and every fact names that same card.
    assert published == {
        "evidence_id": "verified-composer",
        "profile_token_sha256": _sha(card["profile_token"]),
        "approval_token_sha256": _sha(card["approval_token"]),
    }
    assert set(claimed) == {
        "command",
        "evidence_id",
        "idempotency_token_sha256",
        "waited_s",
    }
    assert claimed["command"] == command
    assert claimed["evidence_id"] == "verified-composer"
    assert claimed["idempotency_token_sha256"] == _sha(f"{command}-request")
    assert type(claimed["waited_s"]) is float and claimed["waited_s"] >= 0.0
    assert persisted == {
        "decision": command,
        "evidence_id": "verified-composer",
        "send_evidence_id": ("verified-landed-composer" if command == "like" else None),
        "profile_id": store.profiles[0][0],
    }

    # Published before the human answered, claimed before the device acted on that answer,
    # persisted only once the durable flush returned and before the Hub heard "completed".
    assert events.index(facts[1]) < events.index(("driver_choice", command))
    assert events.index(("flush",)) < events.index(facts[2])
    assert events.index(facts[2]) < events.index(("hub_complete", "completed"))

    # Digests and ids only: no raw token, opener text or profile name in any fact.
    logged = json.dumps([fields for _kind, _name, fields in facts])
    for secret in (
        card["profile_token"],
        card["approval_token"],
        f"{command}-request",
        "A precise typed opener",
        "Ari",
    ):
        assert secret not in logged


def test_a_checkpoint_closed_without_an_answer_is_published_but_never_claimed():
    events = []
    driver = _FactDriver(events)
    worker, store, _bridge = _fact_worker(driver, events, bridge=_SilentBridge(events))
    worker.run()

    assert [name for _kind, name, _fields in _facts(events)] == [
        "training_release_checkpoint_published"
    ]
    assert store.decisions == store.labels == []
    assert worker._training_published_evidence_id is None


def test_a_failed_flush_is_never_recorded_as_persisted():
    events = []
    driver = _FactDriver(events)
    store = _Store(events, flush_error=RuntimeError("durable flush failed"))
    worker, _store, bridge = _fact_worker(driver, events, store=store)
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _submit(bridge, card, "like")
    worker.join(_TIMEOUT_S)
    assert not worker.is_alive()

    assert [name for _kind, name, _fields in _facts(events)] == [
        "training_release_checkpoint_published",
        "training_release_decision_claimed",
    ]
    result = bridge.snapshot(run_id="training-run", app="hinge")["results"][-1]
    assert result["status"] == "failed"


@pytest.mark.parametrize("command", ["like", "dislike"])
@pytest.mark.parametrize(
    "make_driver",
    [
        pytest.param(
            lambda events: _RaisingFactDriver(events, RuntimeError("debug log gone")),
            id="raises",
        ),
        pytest.param(
            lambda events: _RaisingFactDriver(events, ValueError("refused field")),
            id="refuses",
        ),
        pytest.param(_NonCallableFactDriver, id="not-callable"),
        pytest.param(_TrainingDriver, id="missing"),
    ],
)
def test_a_broken_or_missing_fact_recorder_never_changes_the_training_outcome(
    make_driver, command
):
    events = []
    driver = make_driver(events)
    worker, store, bridge = _fact_worker(driver, events)
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _submit(bridge, card, command)
    worker.join(_TIMEOUT_S)
    assert not worker.is_alive()

    assert not worker.stop_event.is_set()
    assert store.decisions[0][0:3] == (
        command,
        1.0 if command == "like" else 0.0,
        "manual",
    )
    assert store.labels[0][0] is (command == "like")
    result = bridge.snapshot(run_id="training-run", app="hinge")["results"][-1]
    assert (result["command"], result["status"]) == (command, "completed")
    if isinstance(driver, _RaisingFactDriver):
        assert [event for event in events if event[0] == "fact_attempt"] == [
            ("fact_attempt", "checkpoint_published"),
            ("fact_attempt", "decision_claimed"),
            ("fact_attempt", "action_persisted"),
        ]


def test_the_worker_fact_helper_swallows_a_failing_attribute_lookup():
    class _Exploding:
        @property
        def training_release_fact(self):
            raise RuntimeError("driver state is gone")

    holder = SimpleNamespace(driver=_Exploding())
    assert (
        Worker._training_release_fact(holder, "action_persisted", decision="like")
        is None
    )
    holder.driver = object()
    assert (
        Worker._training_release_fact(holder, "action_persisted", decision="like")
        is None
    )
