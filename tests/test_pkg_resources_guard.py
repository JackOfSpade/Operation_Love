"""Regression guard: pkg_resources must stay importable for pyiqa's clipiqa metric.

Why this exists: pyiqa's "clipiqa" arch (the quality_filter.metric configured in
config.yaml) imports the legacy `openai-clip` PyPI package (import name `clip`) via
pyiqa/archs/clip_imports.py. clip/clip.py does `from pkg_resources import packaging`
at *module import time* -- not lazily. operation_love/vision/quality.py's
QualityFilter is deliberately fail-loud (warmup()/keep() re-raise rather than degrade
silently -- see quality.py's docstrings), so if pkg_resources ever becomes
unimportable in an installed environment, that chain aborts every real run.

pyproject.toml's `ml` extra pins `setuptools<82` -- the verified boundary where
setuptools stopped shipping pkg_resources (still present through 81.0.0, permanently
gone starting 82.0.0) -- specifically to keep this import alive. This test reproduces
the actual failing import (`import clip`) so it breaks loudly, with an actionable
message, the moment that protection stops working (a resolver quirk, a relaxed or
removed pin, setuptools dropping pkg_resources even earlier than 82, etc.).

This is deliberately not a duplicate of tests/test_warnings.py: that test only asserts
the warning *filter list* is configured; it never imports clip or pkg_resources, so it
would stay green even if pkg_resources vanished entirely. This test actually exercises
the import chain that breaks in that scenario.

Fast and offline: importing `clip` is pure Python (it only touches the network inside
clip.load(), which this test never calls).
"""
from __future__ import annotations

import importlib

import pytest


def test_pkg_resources_stays_importable_for_legacy_clip():
    try:
        importlib.import_module("clip")
    except ModuleNotFoundError as exc:
        if exc.name == "clip":
            pytest.skip(
                "openai-clip (the `clip` package) is not installed in this environment "
                "-- the `ml` extra isn't present, so this guard doesn't apply here."
            )
        pytest.fail(
            "`import clip` failed because "
            f"{exc}. "
            "pyiqa's clipiqa metric (config.yaml quality_filter.metric) needs the legacy "
            "`clip` package, which needs pkg_resources at import time -- and "
            "operation_love/vision/quality.py's QualityFilter is fail-loud, so this "
            "means every real run using the quality filter will now abort. "
            "This is exactly what pyproject.toml's `setuptools<82` pin (in the `ml` "
            "extra) exists to prevent -- reinstall with `pip install -e '.[ml,...]'` so "
            "the pin applies, or if setuptools has changed its pkg_resources removal "
            "point, re-verify the boundary and update that pin (and its comment) "
            "accordingly."
        )
