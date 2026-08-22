"""The video false-accept bound campaign never touches the phone and never labels itself.

Two properties carry most of the weight here.  The first is structural: this harness must be
incapable of putting anything onto the touchscreen, so the module namespace and its source are
checked for every transport/injection symbol rather than trusted to a comment.  The second is
the reason the design exists at all (ops/STILL-PHOTO-DISCRIMINATOR.md section 2 objection 1):
the OWNER's typed label is the only ground truth, the mute matcher is recorded beside it as
observation, and every refusal branch that keeps a weak corpus from becoming a shipped bound is
exercised.  No test here opens a device, runs adb, or sleeps.
"""
from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from operation_love import targeting_policy
from tools import hinge_video_bound as bound

_W, _H = 16, 32
_BAND = (0.125, 0.875)          # rows 4..28 of a 32-row frame, HINGE_SPEC's real band
_FRAME_TIMES = (0.0, 4.0, 8.0, 12.0)


# =====================================================================================
# tiny synthetic frames + campaigns
# =====================================================================================

def _png(fill: int = 40, *, marks=()) -> bytes:
    canvas = np.full((_H, _W, 3), fill, np.uint8)
    for row, col, value in marks:
        canvas[row, col] = value
    ok, buffer = cv2.imencode(".png", canvas)
    assert ok
    return buffer.tobytes()


def _still_frames(n: int = 4, fill: int = 40) -> list[bytes]:
    """One identical byte-for-byte frame repeated: a still photo under C2."""
    frame = _png(fill)
    return [frame] * n


def _moving_frames(n: int = 4) -> list[bytes]:
    """Every frame different inside the content band: a video emitting frames."""
    return [_png(40, marks=[(12, 7, (index * 17) % 255)]) for index in range(n)]


def _partly_still_frames() -> list[bytes]:
    """Two identical frames, then motion: a video that briefly held one frame."""
    held = _png(40, marks=[(12, 7, 3)])
    return [held, held, _png(40, marks=[(12, 7, 90)]), _png(40, marks=[(12, 7, 200)])]


def _card(label: str, frames: list[bytes], *, times=None, prompted=None, **extra) -> dict:
    times = list(times or _FRAME_TIMES[:len(frames)])
    return {"label": label, "payloads": frames, "times": times,
            "prompted": times[-1] + 0.5 if prompted is None else prompted, "extra": extra}


def _write_campaign(root: Path, cards: list[dict], **manifest_overrides) -> Path:
    campaign = root / "ops" / "calibration" / "videobound_test"
    (campaign / "cards").mkdir(parents=True, exist_ok=True)
    records = []
    for ordinal, card in enumerate(cards, start=1):
        card_id = f"card_{ordinal:04d}"
        card_dir = campaign / "cards" / card_id
        card_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for index, payload in enumerate(card["payloads"]):
            name = f"frame_{index:03d}.png"
            (card_dir / name).write_bytes(payload)
            frames.append({"path": f"cards/{card_id}/{name}",
                           "sha256": hashlib.sha256(payload).hexdigest(),
                           "t": float(card["times"][index])})
        record = {"card_id": card_id, "label": card["label"], "frames": frames,
                  "settled": True, "settle_reads": 2, "planned_frames": len(frames),
                  "planned_window_s": 12.0,
                  "burst_completed_t": float(card["times"][-1]),
                  "label_prompted_t": float(card["prompted"]),
                  "mute_matcher_observations": [{"screened": False, "score": None,
                                                 "observational_only": True}
                                                for _ in frames]}
        record.update(card["extra"])
        records.append(record)
    manifest = {
        "schema_version": 1, "kind": "hinge_still_photo_bound_campaign", "tool_version": "1",
        "ground_truth_channel": bound.GROUND_TRUTH_CHANNEL, "human_ground_truth": True,
        "mute_matcher_is_observational_not_ground_truth": True, "completed": True,
        "ended": "profiles_reached", "device": "PIXEL7A", "hinge_version_name": "10.0.1",
        "frame_size_px": [_W, _H], "content_band": [_BAND[0], _BAND[1]],
        "config_sha256": None, "captured_at": "2026-08-21T00:00:00+00:00", "cards": records,
    }
    manifest.update(manifest_overrides)
    (campaign / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return campaign


def _passing_cards(*, videos: int | None = None, photos: int | None = None) -> list[dict]:
    videos = bound.MIN_VIDEO_CARDS if videos is None else videos
    photos = bound.MIN_PHOTO_CARDS if photos is None else photos
    return ([_card("video", _moving_frames()) for _ in range(videos)]
            + [_card("photo", _still_frames()) for _ in range(photos)])


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """Every path check in the tool is repo-relative, so run the tests inside a fake repo."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ops" / "calibration").mkdir(parents=True)
    return tmp_path


def _config(root: Path, body: dict | None = None) -> Path:
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(body if body is not None else
                                   {"apps": {"hinge": {"serial": "PIXEL7A"}}}))
    return path


# =====================================================================================
# structural: this tool cannot inject input
# =====================================================================================

_FORBIDDEN_NAMES = {"UhidTouch", "Adb", "HingeDriver", "AndroidDriver", "TouchTransport",
                    "input_tap", "motionevent", "HINGE_SPEC", "navigate_to_item"}
_FORBIDDEN_SOURCE = ("input tap", "input swipe", "input keyevent", "input text", "sendevent",
                     "motionevent", "UhidTouch", "uhid", "TouchTransport", "dislike",
                     "send_like", "swipe")


def test_module_namespace_holds_no_input_or_gesture_transport():
    assert not set(vars(bound)) & _FORBIDDEN_NAMES
    for name, value in vars(bound).items():
        if isinstance(value, types.ModuleType):
            assert not value.__name__.startswith("operation_love.drivers"), name
        if isinstance(value, type):
            assert "Driver" not in value.__name__, name
            assert "Touch" not in value.__name__, name


def test_module_source_contains_no_injection_primitive():
    source = Path(bound.__file__).read_text()
    for token in _FORBIDDEN_SOURCE:
        assert token not in source, token
    # The complete device vocabulary: read a frame, list devices, read the build string.
    assert "exec-out" in source and "screencap" in source
    assert source.count('"shell"') == 1


def test_policy_thresholds_come_from_targeting_policy():
    for name, value in (
            ("STILL_PHOTO_BOUND_MIN_VIDEO_CARDS", bound.MIN_VIDEO_CARDS),
            ("STILL_PHOTO_BOUND_MIN_PHOTO_CARDS", bound.MIN_PHOTO_CARDS),
            ("STILL_PHOTO_BOUND_MAX_PHOTO_FALSE_REFUSAL_FRAC",
             bound.MAX_PHOTO_FALSE_REFUSAL_FRAC),
            ("STILL_PHOTO_BOUND_REQUIRED_VIDEO_ACCEPTS", bound.REQUIRED_VIDEO_ACCEPTS),
            ("STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL", bound.GROUND_TRUTH_CHANNEL),
            ("STILL_PHOTO_DWELL_WINDOW_SAFETY_FACTOR", bound.DWELL_WINDOW_SAFETY_FACTOR)):
        if hasattr(targeting_policy, name):
            assert getattr(targeting_policy, name) == value, name
    assert bound._POLICY_FALLBACKS == ()


# =====================================================================================
# gitignored output
# =====================================================================================

def test_out_dir_outside_calibration_root_is_refused(repo):
    with pytest.raises(bound.VideoBoundRefused, match="gitignored"):
        bound._private_out_dir(str(repo / "public"), prefix="videobound")
    assert not (repo / "public").exists()


def test_out_dir_git_does_not_ignore_is_refused(repo, monkeypatch):
    monkeypatch.setattr(bound, "_ignored_by_git", lambda path: False)
    target = repo / "ops" / "calibration" / "videobound_x"
    with pytest.raises(bound.VideoBoundRefused, match="git does not ignore"):
        bound._private_out_dir(str(target), prefix="videobound")
    assert not target.exists()


def test_out_dir_refuses_a_non_empty_directory(repo, monkeypatch):
    monkeypatch.setattr(bound, "_ignored_by_git", lambda path: True)
    target = repo / "ops" / "calibration" / "videobound_x"
    target.mkdir()
    (target / "stale.png").write_bytes(b"x")
    with pytest.raises(bound.VideoBoundRefused, match="not empty"):
        bound._private_out_dir(str(target), prefix="videobound")


def test_default_out_dir_is_created_private_under_the_root(repo, monkeypatch):
    monkeypatch.setattr(bound, "_ignored_by_git", lambda path: True)
    out = bound._private_out_dir(None, prefix="videobound")
    assert out.parent == (repo / "ops" / "calibration").resolve()
    assert out.is_dir()


# =====================================================================================
# settle detection
# =====================================================================================

def test_settle_returns_the_first_repeated_whole_frame():
    frames = iter([b"a", b"b", b"c", b"c", b"d"])
    naps: list[float] = []
    frame, settled, reads = bound.settle(lambda: next(frames), sleep_fn=naps.append,
                                         rnd=_Rnd())
    assert (frame, settled, reads) == (b"c", True, 4)
    assert len(naps) == 3 and all(0.30 <= nap <= 0.90 for nap in naps)


def test_settle_reports_unsettled_rather_than_failing_on_a_playing_card():
    counter = iter(range(1000))
    frame, settled, reads = bound.settle(
        lambda: str(next(counter)).encode(), sleep_fn=lambda _s: None, rnd=_Rnd(), max_reads=5)
    assert settled is False and reads == 5 and frame == b"4"


# =====================================================================================
# hazard-randomized burst plan (auto-mode owner rule)
# =====================================================================================

class _Rnd:
    """A deterministic stand-in for random.Random that still returns varying draws."""

    def __init__(self, seed: int = 7):
        self._values = iter([((seed * (index + 3) * 37) % 97) / 97.0
                             for index in range(4096)])

    def random(self) -> float:
        return next(self._values)


def test_burst_plan_stays_inside_the_hazard_spans():
    for seed in range(1, 12):
        plan = bound.plan_burst(_Rnd(seed))
        assert bound._BURST_FRAMES_SPAN[0] <= plan.frames <= bound._BURST_FRAMES_SPAN[1]
        assert bound._BURST_WINDOW_S_SPAN[0] <= plan.window_s <= bound._BURST_WINDOW_S_SPAN[1]
        assert len(plan.gaps_s) == plan.frames - 1
        assert plan.gaps_s and min(plan.gaps_s) > 0
        assert plan.window_s == pytest.approx(sum(plan.gaps_s))


def test_burst_plan_is_not_a_fixed_constant_across_draws():
    plans = [bound.plan_burst(_Rnd(seed)) for seed in range(1, 25)]
    assert len({round(plan.window_s, 6) for plan in plans}) > 1
    assert len({plan.frames for plan in plans}) > 1
    assert len({plan.gaps_s for plan in plans}) > 1


def test_hazard_value_spans_its_range_and_never_leaves_it():
    draws = [bound._hazard_value(8.0, 15.0, _Rnd(seed)) for seed in range(1, 60)]
    assert all(8.0 <= draw <= 15.0 for draw in draws)
    assert max(draws) - min(draws) > 0.5


# =====================================================================================
# burst recording and the label prompt
# =====================================================================================

class _Clock:
    def __init__(self, step: float = 0.5):
        self.now, self.step = 0.0, step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def test_record_burst_stamps_monotonic_times_for_every_planned_frame():
    plan = bound.BurstPlan(frames=4, window_s=9.0, gaps_s=(3.0, 3.0, 3.0))
    frames = iter([b"0", b"1", b"2", b"3"])
    clock = _Clock()
    naps: list[float] = []
    burst = bound.record_burst(lambda: next(frames), plan, sleep_fn=naps.append, clock=clock)
    assert [payload for payload, _t in burst] == [b"0", b"1", b"2", b"3"]
    stamps = [stamp for _payload, stamp in burst]
    assert stamps == sorted(stamps)
    assert len(naps) == 3 and all(nap >= 0 for nap in naps)


@pytest.mark.parametrize("typed", ["video", "photo", "unsure", "skip", "done"])
def test_prompt_label_accepts_each_label(typed):
    assert bound.prompt_label(lambda _p: f"  {typed.upper()} ", card_no=1,
                              print_fn=lambda *_a: None) == typed


def test_prompt_label_reprompts_until_a_real_label_is_typed():
    answers = iter(["", "maybe", "vidoe", "photo"])
    lines: list[str] = []
    assert bound.prompt_label(lambda _p: next(answers), card_no=3,
                              print_fn=lambda text: lines.append(text)) == "photo"
    assert sum(1 for line in lines if "not a label" in line) == 3


def test_prompt_label_treats_eof_as_done():
    def _eof(_prompt):
        raise EOFError

    assert bound.prompt_label(_eof, card_no=1, print_fn=lambda *_a: None) == "done"


# =====================================================================================
# capture: the burst closes before the prompt, and the matcher never labels
# =====================================================================================

def _run_capture(repo, monkeypatch, *, answers, frames, profiles=2):
    monkeypatch.setattr(bound, "_ignored_by_git", lambda path: True)
    monkeypatch.setattr(bound, "_device_version_name", lambda *a, **k: "10.0.1")
    out = bound._private_out_dir(str(repo / "ops" / "calibration" / "cap"), prefix="videobound")
    supply = iter(frames)
    typed = iter(answers)
    return out, bound.run_capture(
        out_dir=out, profiles=profiles, serial="PIXEL7A", adb_path="adb", band=_BAND,
        package="co.hinge.app", config_sha256="cfg", capture_fn=lambda: next(supply),
        input_fn=lambda _p: next(typed), print_fn=lambda *_a: None,
        sleep_fn=lambda _s: None, clock=_Clock(), rnd=_Rnd())


def test_capture_labels_strictly_after_the_burst_and_persists_every_frame(repo, monkeypatch):
    settle_pair = [_png(40), _png(40)]
    burst = _moving_frames(20)
    out, manifest = _run_capture(
        repo, monkeypatch, answers=["", "video", "done"],
        frames=settle_pair + burst, profiles=3)
    assert manifest["completed"] is True and manifest["ended"] == "done_at_ready_prompt"
    assert [card["label"] for card in manifest["cards"]] == ["video"]
    card = manifest["cards"][0]
    assert card["label_prompted_t"] >= card["frames"][-1]["t"]
    assert card["label_prompted_t"] >= card["burst_completed_t"]
    for frame in card["frames"]:
        payload = (out / frame["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == frame["sha256"]
    assert manifest["ground_truth_channel"] == bound.GROUND_TRUTH_CHANNEL
    assert manifest["human_ground_truth"] is True
    assert manifest["mute_matcher_is_observational_not_ground_truth"] is True
    assert len(card["mute_matcher_observations"]) == len(card["frames"])
    assert all(observation["observational_only"] is True
               for observation in card["mute_matcher_observations"])
    # The matcher's own verdict appears nowhere in the label.
    assert set(json.dumps(card["mute_matcher_observations"])) and card["label"] == "video"


def test_capture_records_done_at_the_label_prompt_as_a_skipped_card(repo, monkeypatch):
    _out, manifest = _run_capture(repo, monkeypatch, answers=["", "done"],
                                  frames=[_png(40), _png(40)] + _moving_frames(20), profiles=4)
    assert manifest["ended"] == "done_at_label_prompt"
    assert [card["label"] for card in manifest["cards"]] == ["skip"]


def test_capture_marks_a_never_settling_card_and_still_records_it(repo, monkeypatch):
    monkeypatch.setattr(bound, "_SETTLE_MAX_READS", 4)
    _out, manifest = _run_capture(repo, monkeypatch, answers=["", "video"],
                                  frames=_moving_frames(64), profiles=1)
    assert manifest["cards"][0]["settled"] is False


def test_capture_eof_at_the_ready_prompt_writes_an_incomplete_manifest(repo, monkeypatch):
    def _eof(_prompt):
        raise EOFError

    monkeypatch.setattr(bound, "_ignored_by_git", lambda path: True)
    monkeypatch.setattr(bound, "_device_version_name", lambda *a, **k: "10.0.1")
    out = bound._private_out_dir(str(repo / "ops" / "calibration" / "cap"), prefix="videobound")
    manifest = bound.run_capture(
        out_dir=out, profiles=2, serial="PIXEL7A", adb_path="adb", band=_BAND,
        package="co.hinge.app", config_sha256=None, capture_fn=lambda: _png(40),
        input_fn=_eof, print_fn=lambda *_a: None, sleep_fn=lambda _s: None,
        clock=_Clock(), rnd=_Rnd())
    assert manifest["completed"] is False and manifest["cards"] == []
    with pytest.raises(bound.VideoBoundRefused, match="not completed"):
        bound.measure(out)


def test_capture_keeps_what_it_recorded_when_the_device_drops(repo, monkeypatch):
    """An hour of owner labeling survives a mid-run failure, marked incomplete and re-raised."""
    monkeypatch.setattr(bound, "_ignored_by_git", lambda path: True)
    monkeypatch.setattr(bound, "_device_version_name", lambda *a, **k: "10.0.1")
    out = bound._private_out_dir(str(repo / "ops" / "calibration" / "cap"), prefix="videobound")
    supply = iter([_png(40), _png(40)] + _moving_frames(20))

    def _capture():
        try:
            return next(supply)
        except StopIteration:
            raise bound.VideoBoundRefused("screencap did not return a PNG frame") from None

    typed = iter(["", "video", "", "photo"])
    with pytest.raises(bound.VideoBoundRefused, match="screencap"):
        bound.run_capture(
            out_dir=out, profiles=4, serial="PIXEL7A", adb_path="adb", band=_BAND,
            package="co.hinge.app", config_sha256=None, capture_fn=_capture,
            input_fn=lambda _p: next(typed), print_fn=lambda *_a: None,
            sleep_fn=lambda _s: None, clock=_Clock(), rnd=_Rnd())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["completed"] is False
    assert manifest["ended"] == "aborted_VideoBoundRefused"
    assert [card["label"] for card in manifest["cards"]] == ["video"]


# =====================================================================================
# measure: per-card statistics
# =====================================================================================

def test_measure_computes_the_per_card_longest_byte_exact_run(repo):
    cards = _passing_cards(videos=bound.MIN_VIDEO_CARDS - 1, photos=bound.MIN_PHOTO_CARDS)
    cards.append(_card("video", _partly_still_frames()))
    result = bound.measure(_write_campaign(repo, cards))
    held = next(stat for stat in result.stats if stat.label == "video"
                and stat.longest_exact_run_s > 0)
    assert held.longest_exact_run_s == pytest.approx(4.0)
    assert held.longest_exact_run_frames == 2
    assert result.max_video_exact_run_s == pytest.approx(4.0)
    photo = next(stat for stat in result.stats if stat.label == "photo")
    assert photo.all_pairs_exact is True
    assert photo.longest_exact_run_s == pytest.approx(12.0)


def test_measure_ignores_a_one_pixel_change_outside_the_content_band(repo):
    outside = [_png(40), _png(40, marks=[(1, 1, 255)]), _png(40), _png(40)]
    cards = _passing_cards(photos=bound.MIN_PHOTO_CARDS - 1)
    cards.append(_card("photo", outside))
    result = bound.measure(_write_campaign(repo, cards))
    assert result.photo_false_refusals == 0
    assert result.photo_cards == bound.MIN_PHOTO_CARDS


def test_measure_counts_a_one_pixel_change_inside_the_content_band(repo):
    inside = [_png(40), _png(40, marks=[(12, 7, 255)]), _png(40), _png(40)]
    cards = _passing_cards(photos=bound.MIN_PHOTO_CARDS - 1)
    cards.append(_card("photo", inside))
    result = bound.measure(_write_campaign(repo, cards))
    assert result.photo_false_refusals == 1


def test_measure_excludes_unsure_and_skip_from_both_denominators(repo):
    cards = _passing_cards()
    cards += [_card("unsure", _moving_frames()) for _ in range(3)]
    cards += [_card("skip", _still_frames()) for _ in range(2)]
    result = bound.measure(_write_campaign(repo, cards))
    assert (result.video_cards, result.photo_cards) == (bound.MIN_VIDEO_CARDS,
                                                        bound.MIN_PHOTO_CARDS)
    assert (result.unsure_cards, result.skipped_cards) == (3, 2)


def test_measure_passes_a_complete_campaign_and_reports_the_window(repo):
    result = bound.measure(_write_campaign(repo, _passing_cards()))
    assert result.video_accepts == bound.REQUIRED_VIDEO_ACCEPTS == 0
    assert result.photo_false_refusals == 0
    assert result.max_video_exact_run_s == 0.0
    assert result.dwell_min_window_s == pytest.approx(12.0)
    assert result.dwell_min_frames == 4
    lines: list[str] = []
    bound.print_bound_report(result, print_fn=lines.append)
    assert any("owner_tap_to_play_v1" in line for line in lines)


# =====================================================================================
# measure: every refusal branch
# =====================================================================================

def test_measure_refuses_too_few_video_cards(repo):
    campaign = _write_campaign(repo, _passing_cards(videos=bound.MIN_VIDEO_CARDS - 1))
    with pytest.raises(bound.VideoBoundRefused, match="owner-labeled video cards"):
        bound.measure(campaign)


def test_measure_refuses_too_few_photo_cards(repo):
    campaign = _write_campaign(repo, _passing_cards(photos=bound.MIN_PHOTO_CARDS - 1))
    with pytest.raises(bound.VideoBoundRefused, match="owner-labeled still-photo cards"):
        bound.measure(campaign)


def test_unsure_cards_cannot_make_up_a_short_video_denominator(repo):
    cards = _passing_cards(videos=bound.MIN_VIDEO_CARDS - 4)
    cards += [_card("unsure", _moving_frames()) for _ in range(8)]
    with pytest.raises(bound.VideoBoundRefused, match="owner-labeled video cards"):
        bound.measure(_write_campaign(repo, cards))


def test_measure_refuses_a_single_video_accept(repo):
    cards = _passing_cards(videos=bound.MIN_VIDEO_CARDS - 1)
    cards.append(_card("video", _still_frames()))     # held one frame for the whole burst
    with pytest.raises(bound.VideoBoundRefused, match="would have been ACCEPTED"):
        bound.measure(_write_campaign(repo, cards))


def test_measure_refuses_a_window_the_campaign_never_observed(repo):
    long_times = (0.0, 4.5, 9.0, 13.5)
    cards = _passing_cards(videos=bound.MIN_VIDEO_CARDS - 1)
    cards.append(_card("video", _partly_still_frames(), times=long_times))
    with pytest.raises(bound.VideoBoundRefused, match="exceeds the"):
        bound.measure(_write_campaign(repo, cards))


def test_measure_refuses_too_many_photo_false_refusals(repo):
    blip = [_png(40), _png(40, marks=[(12, 7, 255)]), _png(40), _png(40)]
    ceiling = int(bound.MAX_PHOTO_FALSE_REFUSAL_FRAC * bound.MIN_PHOTO_CARDS)
    cards = _passing_cards(photos=bound.MIN_PHOTO_CARDS - ceiling - 1)
    cards += [_card("photo", blip) for _ in range(ceiling + 1)]
    with pytest.raises(bound.VideoBoundRefused, match="falsely refused"):
        bound.measure(_write_campaign(repo, cards))


def test_measure_refuses_a_frame_that_does_not_match_its_manifest_digest(repo):
    campaign = _write_campaign(repo, _passing_cards())
    tampered = campaign / "cards" / "card_0001" / "frame_001.png"
    tampered.write_bytes(_png(200))
    with pytest.raises(bound.VideoBoundRefused, match="does not match its manifest sha256"):
        bound.measure(campaign)


def test_measure_refuses_a_missing_frame(repo):
    campaign = _write_campaign(repo, _passing_cards())
    (campaign / "cards" / "card_0001" / "frame_002.png").unlink()
    with pytest.raises(bound.VideoBoundRefused, match="is missing"):
        bound.measure(campaign)


def test_measure_refuses_a_label_typed_before_the_burst_closed(repo):
    cards = _passing_cards()
    cards[0] = _card("video", _moving_frames(), prompted=5.0)
    with pytest.raises(bound.VideoBoundRefused, match="labeled before its dwell burst closed"):
        bound.measure(_write_campaign(repo, cards))


def test_measure_refuses_a_foreign_ground_truth_channel(repo):
    campaign = _write_campaign(repo, _passing_cards(),
                               ground_truth_channel="video_mute_v1")
    with pytest.raises(bound.VideoBoundRefused, match="ground_truth_channel"):
        bound.measure(campaign)


def test_measure_refuses_an_artifact_that_drops_human_ground_truth(repo):
    campaign = _write_campaign(repo, _passing_cards(), human_ground_truth=False)
    with pytest.raises(bound.VideoBoundRefused, match="human ground truth"):
        bound.measure(campaign)


def test_measure_refuses_a_campaign_captured_against_other_config_bytes(repo):
    campaign = _write_campaign(repo, _passing_cards(), config_sha256="0" * 64)
    config = _config(repo)
    with pytest.raises(bound.VideoBoundRefused, match="different config bytes"):
        bound.measure(campaign, config_path=str(config))


def test_measure_writes_no_artifact_when_it_refuses(repo):
    campaign = _write_campaign(repo, _passing_cards(videos=1))
    with pytest.raises(bound.VideoBoundRefused):
        bound.measure(campaign)
    assert not (campaign / "bound.json").exists()


def test_campaign_directory_outside_the_private_root_is_refused(repo):
    with pytest.raises(bound.VideoBoundRefused, match="private"):
        bound._existing_campaign_dir(str(repo / "elsewhere"))


# =====================================================================================
# emit: artifact schema, self digest, paste block
# =====================================================================================

def _emit(repo, monkeypatch, cards=None, *, live_version=None):
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: [])
    monkeypatch.setattr(bound, "_device_version_name",
                        lambda *a, **k: live_version)
    campaign = _write_campaign(repo, cards if cards is not None else _passing_cards())
    config = _config(repo)
    return campaign, config, bound.emit(campaign, config_path=str(config))


def test_emit_writes_the_exact_artifact_schema(repo, monkeypatch):
    campaign, config, (artifact, _paste) = _emit(repo, monkeypatch)
    assert set(artifact) == bound.BOUND_ARTIFACT_KEYS
    assert artifact["schema_version"] == 1
    assert artifact["ground_truth_channel"] == "owner_tap_to_play_v1"
    assert artifact["human_ground_truth"] is True
    assert artifact["device"] == "PIXEL7A"
    assert artifact["hinge_version_name"] == "10.0.1"
    assert artifact["frame_size_px"] == [_W, _H]
    assert artifact["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert artifact["video_cards"] == bound.MIN_VIDEO_CARDS
    assert artifact["video_accepts"] == 0
    assert artifact["photo_cards"] == bound.MIN_PHOTO_CARDS
    assert artifact["photo_false_refusals"] == 0
    assert artifact["max_video_exact_run_s"] == 0.0
    assert set(artifact["dwell"]) == {"min_frames", "min_window_s"}
    assert all(set(card) == bound.BOUND_CARD_KEYS for card in artifact["cards"])
    assert all(set(frame) == {"path", "sha256", "t"}
               for card in artifact["cards"] for frame in card["frames"])
    on_disk = json.loads((campaign / "bound.json").read_text())
    assert on_disk == artifact


def test_emit_artifact_self_digest_verifies_and_detects_tampering(repo, monkeypatch):
    _campaign, _config, (artifact, _paste) = _emit(repo, monkeypatch)
    assert bound.verify_artifact_digest(artifact) is True
    tampered = dict(artifact)
    tampered["video_accepts"] = 0 if artifact["video_accepts"] else 1
    assert bound.verify_artifact_digest(tampered) is False
    short = dict(artifact)
    short.pop("dwell")
    assert bound.verify_artifact_digest(short) is False


def test_emit_paste_block_has_exactly_the_config_key_set(repo, monkeypatch):
    campaign, _config, (artifact, paste) = _emit(repo, monkeypatch)
    assert tuple(paste) == bound.PASTE_KEYS
    assert tuple(paste) == (
        "artifact_path", "artifact_sha256", "ground_truth_channel", "video_cards",
        "video_accepts", "photo_cards", "photo_false_refusals", "max_video_exact_run_s",
        "captured_at", "device", "hinge_version_name")
    payload = (campaign / "bound.json").read_bytes()
    assert paste["artifact_sha256"] == hashlib.sha256(payload).hexdigest()
    assert paste["artifact_path"] == "ops/calibration/videobound_test/bound.json"
    assert not Path(paste["artifact_path"]).is_absolute()
    for key in ("ground_truth_channel", "video_cards", "video_accepts", "photo_cards",
                "photo_false_refusals", "max_video_exact_run_s", "captured_at", "device",
                "hinge_version_name"):
        assert paste[key] == artifact[key]
    block = yaml.safe_load(yaml.safe_dump(
        {"apps": {"hinge": {"still_photo_bound_evidence": paste}}}, sort_keys=False))
    assert block["apps"]["hinge"]["still_photo_bound_evidence"] == paste


def test_emit_prefers_the_live_build_string_over_the_manifest(repo, monkeypatch):
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: ["PIXEL7A"])
    monkeypatch.setattr(bound, "_device_version_name", lambda *a, **k: "10.2.0")
    campaign = _write_campaign(repo, _passing_cards())
    artifact, paste = bound.emit(campaign, config_path=str(_config(repo)))
    assert artifact["hinge_version_name"] == "10.2.0"
    assert paste["hinge_version_name"] == "10.2.0"


def test_emit_ignores_a_build_string_read_from_a_different_phone(repo, monkeypatch):
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: ["OTHER"])
    monkeypatch.setattr(bound, "_device_version_name", lambda *a, **k: "11.9.9")
    campaign = _write_campaign(repo, _passing_cards())
    config = _config(repo, {"apps": {"hinge": {"serial": "OTHER"}}})
    artifact, _paste = bound.emit(campaign, config_path=str(config))
    assert artifact["hinge_version_name"] == "10.0.1"


def test_emit_refuses_without_any_build_string(repo, monkeypatch):
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: [])
    monkeypatch.setattr(bound, "_device_version_name", lambda *a, **k: None)
    campaign = _write_campaign(repo, _passing_cards(), hinge_version_name=None)
    with pytest.raises(bound.VideoBoundRefused, match="versionName"):
        bound.emit(campaign, config_path=str(_config(repo)))


def test_emit_refuses_and_writes_nothing_when_the_bound_fails(repo, monkeypatch):
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: [])
    monkeypatch.setattr(bound, "_device_version_name", lambda *a, **k: "10.0.1")
    cards = _passing_cards(videos=bound.MIN_VIDEO_CARDS - 1)
    cards.append(_card("video", _still_frames()))
    campaign = _write_campaign(repo, cards)
    with pytest.raises(bound.VideoBoundRefused, match="would have been ACCEPTED"):
        bound.emit(campaign, config_path=str(_config(repo)))
    assert not (campaign / "bound.json").exists()


def test_emitted_paste_block_and_artifact_satisfy_config_validation(repo, monkeypatch):
    """The block this tool prints is exactly what operation_love/config.py accepts.

    Two independently written key sets have to agree for numbering readiness to install at all,
    so bind them here rather than trusting that both files were edited the same day.
    """
    from operation_love import config as config_mod

    validate = getattr(config_mod, "_validate_hinge_still_photo_bound_evidence", None)
    clear = getattr(targeting_policy, "clear_installed_still_photo_bound", None)
    if validate is None or clear is None:
        pytest.skip("config-side still-photo bound validation is not present in this tree")
    _campaign, _config, (artifact, paste) = _emit(repo, monkeypatch)
    cfg = types.SimpleNamespace(
        enabled_apps=["hinge"],
        apps={"hinge": {"serial": artifact["device"], "still_photo_bound_evidence": paste}})
    try:
        validate(cfg)
        assert targeting_policy.hinge_targeting_unavailable_reason() is None
    finally:
        clear()
    assert targeting_policy.hinge_targeting_unavailable_reason() is not None


# =====================================================================================
# hold-test falsifier
# =====================================================================================

def _hold(repo, monkeypatch, frames, *, seconds=4.0):
    monkeypatch.setattr(bound, "_ignored_by_git", lambda path: True)
    out = bound._private_out_dir(str(repo / "ops" / "calibration" / "hold"),
                                 prefix="videobound_hold")
    supply = iter(frames)
    lines: list[str] = []
    report = bound.run_hold_test(
        out_dir=out, seconds=seconds, interval=0.5, band=_BAND, serial="PIXEL7A",
        adb_path="adb", capture_fn=lambda: next(supply), print_fn=lines.append,
        sleep_fn=lambda _s: None, clock=_Clock())
    return out, report, lines


def test_hold_test_reports_no_exact_pair_for_a_playing_video(repo, monkeypatch):
    out, report, lines = _hold(repo, monkeypatch, _moving_frames(8))
    assert report["frame_count"] == 3 and report["pair_count"] == 2
    assert report["band_exact_pairs"] == 0
    assert report["band_longest_exact_run_s"] == 0.0
    assert any("did\nnot fire" in line or "not fire" in line for line in lines)
    persisted = json.loads((out / "holdtest.json").read_text())
    assert persisted == report
    for frame in report["frames"]:
        payload = (out / frame["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == frame["sha256"]


def test_hold_test_prints_the_design_consequence_when_the_falsifier_fires(repo, monkeypatch):
    _out, report, lines = _hold(repo, monkeypatch, [_png(40)] * 8)
    assert report["band_exact_pairs"] == 2
    assert report["band_longest_exact_run_s"] == pytest.approx(3.0)
    joined = "\n".join(lines)
    assert "VERDICT" in joined
    assert "must grow" in joined and "abandoned" in joined
    assert "\U0001f7e2" not in joined and "\U0001f534" not in joined


def test_hold_test_row_blocks_separate_moving_chrome_from_a_held_card(repo, monkeypatch):
    # Only frame row 5 (a chrome strip) changes; the card rows below never move at all.
    frames = [_png(40, marks=[(5, col, 200) for col in range(_W)][:index + 1])
              for index in range(1, 9)]
    _out, report, _lines = _hold(repo, monkeypatch, frames)
    assert report["band_exact_pairs"] == 0
    blocks = report["changed_row_blocks"]
    assert blocks and all(block["rows"][1] - block["rows"][0] <= 2 for block in blocks)
    # The still card region shows up as never-changed rows, which the whole-band line hides.
    assert report["unchanged_row_count"] >= report["band_rows"] - 2
    span = report["longest_unchanged_row_span"]
    assert span is not None and span[1] - span[0] >= 10


def test_hold_test_refuses_a_degenerate_window(repo, monkeypatch):
    monkeypatch.setattr(bound, "_ignored_by_git", lambda path: True)
    out = bound._private_out_dir(str(repo / "ops" / "calibration" / "hold"),
                                 prefix="videobound_hold")
    with pytest.raises(bound.VideoBoundRefused, match="positive"):
        bound.run_hold_test(out_dir=out, seconds=0.0, interval=0.5, band=_BAND,
                            serial="PIXEL7A", adb_path="adb", capture_fn=lambda: _png(40),
                            print_fn=lambda *_a: None, sleep_fn=lambda _s: None,
                            clock=_Clock())


# =====================================================================================
# config reading and device resolution (no adb is ever run)
# =====================================================================================

def test_effective_content_band_prefers_the_apps_hinge_override():
    assert bound._effective_content_band({"content_band": [0.2, 0.8]}, None) == (0.2, 0.8)
    assert bound._effective_content_band({}, "0.1,0.9") == (0.1, 0.9)


def test_effective_content_band_falls_back_to_the_shipped_spec():
    assert bound._effective_content_band({}, None) == _BAND


def test_effective_content_band_refuses_a_degenerate_band():
    with pytest.raises(bound.VideoBoundRefused, match="ordered fractions"):
        bound._effective_content_band({}, "0.9,0.1")


def test_resolve_serial_refuses_without_a_configured_serial(monkeypatch):
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: ["A", "B"])
    with pytest.raises(bound.VideoBoundRefused, match="apps.hinge.serial is not configured"):
        bound._resolve_serial({}, None)


def test_resolve_serial_refuses_multiple_ready_devices_without_an_override(monkeypatch):
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: ["A", "B"])
    with pytest.raises(bound.VideoBoundRefused, match="multiple ADB devices"):
        bound._resolve_serial({"serial": "A"}, None)
    assert bound._resolve_serial({"serial": "A"}, "B") == ("B", "adb")


def test_resolve_serial_refuses_a_serial_that_is_not_ready(monkeypatch):
    monkeypatch.setattr(bound, "_ready_devices", lambda adb_path: ["A"])
    with pytest.raises(bound.VideoBoundRefused, match="not among the ready"):
        bound._resolve_serial({"serial": "Z"}, None)


def test_device_version_name_refuses_a_bogus_package():
    with pytest.raises(bound.VideoBoundRefused, match="Android package id"):
        bound._device_version_name("A", "adb", "co.hinge.app; rm -rf /")
