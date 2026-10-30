import pytest

from operation_love.typography import format_duration


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf")])
def test_unavailable_duration_never_breaks_diagnostics(value):
    assert format_duration(value) == "unknown duration"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(45, "45s"), (97, "1m37s"), (3723, "1h02m03s"), (-1, "0s")],
)
def test_duration_keeps_compact_finite_format(seconds, expected):
    assert format_duration(seconds) == expected
