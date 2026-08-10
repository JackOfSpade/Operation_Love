"""Unit tests for the pure autonomous-session interaction policy."""
import random

import pytest

from operation_love.interaction import AutoSessionPolicy
from operation_love.perception.capture import Profile
from operation_love.ranker.decider import Decision


def _profile(*, photos=2, frames=2, scrolls=1, bio=""):
    return Profile(
        photos=[b"x"] * photos,
        bio=bio,
        meta={"capture_frames": frames, "read_scrolls": scrolls},
    )


def test_read_steps_are_bounded_and_not_a_fixed_scroll_macro():
    policy = AutoSessionPolicy(rng=random.Random(7), local_hour=lambda: 14)
    steps = [policy.read_step(i, captured_frames=i + 1) for i in range(7)]

    assert all(0.42 <= s.fraction <= 0.60 for s in steps)
    assert all(0.42 <= s.x_frac <= 0.58 for s in steps)
    assert all(0.55 <= s.dwell_s <= 3.50 for s in steps)
    assert len({round(s.fraction, 5) for s in steps}) > 1
    assert len({round(s.x_frac, 5) for s in steps}) > 1
    assert len({round(s.dwell_s, 5) for s in steps}) > 1


def test_read_step_is_reproducible_with_an_injected_seed():
    a = AutoSessionPolicy(rng=random.Random(23), local_hour=lambda: 12)
    b = AutoSessionPolicy(rng=random.Random(23), local_hour=lambda: 12)
    assert [a.read_step(i, captured_frames=3) for i in range(4)] == [
        b.read_step(i, captured_frames=3) for i in range(4)
    ]


def test_post_action_delay_uses_profile_context_time_and_landed_history():
    simple = _profile(photos=1, frames=1, scrolls=0)
    rich = _profile(photos=7, frames=7, scrolls=6, bio="a detailed profile " * 20)
    early = AutoSessionPolicy(rng=random.Random(4), local_hour=lambda: 14)
    late = AutoSessionPolicy(rng=random.Random(4), local_hour=lambda: 2)
    for _ in range(18):
        late.record_landed_action("like")

    # Same seed means the difference is context rather than an accidental draw.
    assert late.post_action_delay_s("like", rich, 0.51) > early.post_action_delay_s("like", simple, 0.90)


def test_post_action_delay_has_session_correlated_variation_and_validates_scale():
    policy = AutoSessionPolicy(rng=random.Random(44), local_hour=lambda: 16)
    p = _profile()
    delays = [policy.post_action_delay_s("dislike", p, 0.5) for _ in range(8)]
    assert all(0.35 <= d <= 45.0 for d in delays)
    assert len({round(d, 5) for d in delays}) > 1
    with pytest.raises(ValueError, match="positive"):
        policy.post_action_delay_s("like", p, 0.5, scale=-1)


def test_post_action_delay_rejects_zero_scale_too():
    """A `scale <= 0` guard, not just `scale < 0`: worker.py's documented no-pacing
    bypass is skipping the call to this method entirely (see _pace's `swipe_delay_s
    == 0` early return), never calling in with scale=0. If this method were ever
    reached with an exact-zero scale anyway, every modifier collapses and it would
    silently return a bare `0.0` wait -- an instant, zero-jitter action -- instead of
    failing loudly the way the existing negative-scale case already does."""
    policy = AutoSessionPolicy(rng=random.Random(1), local_hour=lambda: 12)
    p = _profile()
    with pytest.raises(ValueError, match="positive"):
        policy.post_action_delay_s("like", p, 0.5, scale=0)


@pytest.mark.parametrize("bad_value", [
    "not-a-number",     # non-numeric string
    None,                # explicit None (present but unset -- distinct from key absent)
    -7,                  # negative
    float("inf"),        # huge / unbounded
    float("nan"),        # not-a-number float
    object(),             # arbitrary non-numeric type
])
def test_post_action_delay_tolerates_malformed_capture_frames_metadata(bad_value):
    """Confirmed-live bug from the audit: a non-numeric meta['capture_frames'] raised a
    bare ValueError out of _profile_complexity, and worker.py's auto loop has no
    try/except around apply_decision()/post_action_delay_s() -- it would propagate to
    the loop's blanket exception handler and HALT THE WHOLE RUN over pacing metadata,
    not a real navigation/screen error. Every one of these malformed shapes must now
    degrade to sane, bounded pacing instead of raising."""
    p = Profile(photos=[b"x"] * 3, bio="", meta={"capture_frames": bad_value})
    policy = AutoSessionPolicy(rng=random.Random(5), local_hour=lambda: 12)
    delay = policy.post_action_delay_s("like", p, 0.6)   # must not raise
    assert 0.35 <= delay <= 45.0


@pytest.mark.parametrize("bad_value", ["nope", None, -3, float("inf"), float("nan")])
def test_post_action_delay_tolerates_malformed_read_scrolls_metadata(bad_value):
    """Same guard, the other malformed pacing field -- meta['read_scrolls']."""
    p = Profile(photos=[b"x"] * 2, bio="", meta={"capture_frames": 2, "read_scrolls": bad_value})
    policy = AutoSessionPolicy(rng=random.Random(6), local_hour=lambda: 12)
    delay = policy.post_action_delay_s("dislike", p, 0.4)   # must not raise
    assert 0.35 <= delay <= 45.0


def test_safe_meta_int_falls_back_on_malformed_values_and_clamps_negative():
    """Direct unit coverage of the actual coercion helper (_profile_complexity's fix),
    not just the fact that the larger call doesn't raise -- pins the exact fallback
    value for each malformed shape."""
    meta_int = AutoSessionPolicy._safe_meta_int
    assert meta_int({"k": "garbage"}, "k", 9) == 9      # non-numeric string -> default
    assert meta_int({"k": None}, "k", 9) == 9            # explicit None -> default
    assert meta_int({"k": float("inf")}, "k", 9) == 9    # OverflowError -> default
    assert meta_int({"k": float("nan")}, "k", 9) == 9    # ValueError -> default
    assert meta_int({"k": -5}, "k", 9) == 0               # negative -> clamped to 0, not default
    assert meta_int({"k": "4"}, "k", 9) == 4              # numeric string still parses normally
    assert meta_int({}, "k", 9) == 9                      # key absent -> default
    assert meta_int({"k": 0}, "k", 9) == 9                # explicit 0 -> "not provided", matches
    # the pre-existing `or default` semantics this replaced.


def test_contextual_threshold_only_demotes_borderline_likes():
    policy = AutoSessionPolicy(rng=random.Random(9), local_hour=lambda: 2)
    p = _profile(photos=1, frames=1, scrolls=0)
    for _ in range(20):
        policy.record_landed_action("like")

    borderline = Decision("like", 0.501, [1.0], "ranker")
    result = policy.apply_decision(borderline, p)
    assert result.demoted is True
    assert result.decision.decision == "dislike"
    assert result.decision.score == borderline.score
    assert result.effective_threshold >= 0.5

    # A policy never turns a classifier-dislike into a like, even if its score
    # happens to be near the configured boundary.
    dislike = Decision("dislike", 0.499, [1.0], "ranker")
    assert policy.apply_decision(dislike, p).decision is dislike


def test_borderline_like_has_no_single_replacement_cutoff_across_sessions():
    p = _profile(photos=3, frames=3, scrolls=2)
    decisions = []
    thresholds = []
    for seed in range(250):
        policy = AutoSessionPolicy(rng=random.Random(seed), local_hour=lambda: 18)
        result = policy.apply_decision(Decision("like", 0.508, [1.0], "ranker"), p)
        decisions.append(result.decision.decision)
        thresholds.append(round(result.effective_threshold, 5))

    assert set(decisions) == {"like", "dislike"}
    assert len(set(thresholds)) > 100


def test_confident_likes_and_non_ranker_terminal_decisions_keep_semantics():
    policy = AutoSessionPolicy(rng=random.Random(3), local_hour=lambda: 13)
    p = _profile(photos=5, frames=5, scrolls=4)
    confident = Decision("like", 0.9, [1.0], "ranker")
    assert policy.apply_decision(confident, p).decision is confident
    for terminal in (Decision("defer", 0.0, [], "cold_start"), Decision("no_face", 0.0, [], "ranker")):
        result = policy.apply_decision(terminal, p)
        assert result.decision is terminal
        assert result.effective_threshold is None


def test_zero_threshold_lift_is_exact_legacy_decision_boundary():
    policy = AutoSessionPolicy(rng=random.Random(1), local_hour=lambda: 1,
                               max_threshold_lift=0.0)
    d = Decision("like", 0.5, [1.0], "ranker")
    result = policy.apply_decision(d, _profile())
    assert result.decision is d
    assert result.effective_threshold == 0.5


def test_record_landed_action_tracks_streak_only_after_real_actions():
    policy = AutoSessionPolicy(rng=random.Random(2), local_hour=lambda: 12)
    policy.record_landed_action("like")
    policy.record_landed_action("like")
    policy.record_landed_action("dislike")
    assert (policy.actions, policy.likes, policy.action_streak) == (3, 2, 1)
    with pytest.raises(ValueError, match="landed action"):
        policy.record_landed_action("defer")
