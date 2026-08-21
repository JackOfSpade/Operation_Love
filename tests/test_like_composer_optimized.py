"""The composer detector's fail-closed contract survives optimized Python."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pytest

from operation_love.drivers import like_composer


def _blank_frame() -> bytes:
    ok, encoded = cv2.imencode(".png", np.full((2400, 1080), 249, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_empty_internal_candidate_set_raises_the_public_detection_error(monkeypatch):
    monkeypatch.setattr(like_composer, "_confirm_matches", lambda *_args, **_kwargs: [])

    with pytest.raises(
            like_composer.ComposerDetectionError,
            match="no actionable confirmation candidate"):
        like_composer.locate_inline_composer(
            _blank_frame(), np.ones((2, 2), dtype=np.uint8))


def test_empty_internal_candidate_set_still_raises_composer_error_under_python_optimized():
    script = textwrap.dedent(
        """
        import cv2
        import numpy as np
        from operation_love.drivers import like_composer

        canvas = np.full((2400, 1080), 249, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", canvas)
        if not ok:
            raise RuntimeError("could not encode test frame")
        like_composer._confirm_matches = lambda *_args, **_kwargs: []
        try:
            like_composer.locate_inline_composer(
                encoded.tobytes(), np.ones((2, 2), dtype=np.uint8))
        except like_composer.ComposerDetectionError as exc:
            if "no actionable confirmation candidate" not in str(exc):
                raise
        else:
            raise RuntimeError("optimized detector accepted an empty candidate set")
        """
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
