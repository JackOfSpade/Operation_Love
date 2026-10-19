"""Shared normalization for finite real-valued configuration and runtime data."""

from __future__ import annotations

import math
from numbers import Real


def finite_real_as_float(value: object) -> float | None:
    """Return a finite float, rejecting booleans, non-reals and conversion overflow.

    Callers retain their own range rules and error messages. Numeric strings are
    deliberately rejected rather than silently changing the public input contract.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None
