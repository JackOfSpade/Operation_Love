"""Human-like timing (log-normal delays) — no network, pure math."""
import math
import random

import pytest

from operation_love.human import human_cooldown, human_delay


def test_cooldown_never_below_anchor():
    random.seed(1)
    anchor = 5.0
    for _ in range(5000):
        assert human_cooldown(anchor) >= anchor          # backoffs only ever wait longer


def test_delay_spreads_both_sides_of_anchor():
    random.seed(2)
    anchor = 3.5
    samples = [human_delay(anchor) for _ in range(5000)]
    assert all(s > 0 for s in samples)                   # never negative
    assert any(s < anchor for s in samples)              # sometimes quicker
    assert any(s > anchor for s in samples)              # sometimes slower
    mean = sum(samples) / len(samples)
    assert anchor * 0.8 < mean < anchor * 1.3            # centered near the anchor


def test_zero_anchor_is_zero():
    assert human_delay(0.0) == 0.0
    assert human_cooldown(0.0) == 0.0


def test_sigma_zero_returns_anchor_exactly():
    assert human_delay(4.0, sigma=0.0) == 4.0
    assert human_cooldown(4.0, sigma=0.0) == 4.0


@pytest.mark.parametrize("function", [human_delay, human_cooldown])
@pytest.mark.parametrize("seconds", [
    True, -1, math.nan, math.inf, pytest.param(10 ** 10_000, id="huge_int"), "1",
])
def test_delay_functions_reject_invalid_anchors(function, seconds):
    with pytest.raises(ValueError, match="seconds"):
        function(seconds)


@pytest.mark.parametrize("function", [human_delay, human_cooldown])
@pytest.mark.parametrize("sigma", [
    True, -0.1, math.nan, math.inf, pytest.param(10 ** 10_000, id="huge_int"), "0.2",
])
def test_delay_functions_reject_invalid_sigma(function, sigma):
    with pytest.raises(ValueError, match="sigma"):
        function(1.0, sigma=sigma)
