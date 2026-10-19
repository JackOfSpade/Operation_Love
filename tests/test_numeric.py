"""Boundary contracts shared by runtime status and decision limits."""

from fractions import Fraction

import pytest

from operation_love.numeric import finite_real_as_float


@pytest.mark.parametrize(
    "value",
    [True, False, None, "1", 1j, float("nan"), float("inf"), -float("inf"), 10**1000],
)
def test_invalid_or_unrepresentable_numbers_are_rejected(value):
    assert finite_real_as_float(value) is None


@pytest.mark.parametrize(
    "value, expected", [(0, 0.0), (-2, -2.0), (1.5, 1.5), (Fraction(1, 4), 0.25)]
)
def test_real_numbers_are_normalized_without_range_policy(value, expected):
    assert finite_real_as_float(value) == expected
