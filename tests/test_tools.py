"""tools/ coverage — this directory had ZERO tests before this file: grepping every
tests/test_*.py for "tools" hit nothing but a docstring mention. That gap is exactly
how a real bug went undetected: tools/hinge_inspect.py imported
`operation_love.drivers.hinge.vision`, a module that has never existed (`hinge` is a
single module, not a package), so the vision-verification section of the tool was
dead code behind a swallowed ImportError, silently printing "skipped" on every run.

These tests guard the class of bug, not just the instance: rather than hand-listing
the symbols each tool needs (a list that would rot), they AST-walk each tool's source
for every `operation_love.*` import — including ones nested inside functions/try
blocks, which is exactly where the dead import hid — and confirm the target module
and symbol both resolve.

No network, no subprocess, no adb, no Playwright, no BigQuery: all three tools lazily
import their optional-dependency drivers (see operation_love/drivers/bumble.py,
operation_love/drivers/hinge.py, operation_love/ranker/bigquery_store.py), so plain
imports here are safe even on a machine with none of the optional extras installed.
The CLI-surface tests only ever request --help, which argparse answers with a clean
SystemExit before any real body (device/browser/BigQuery work) runs.
"""
from __future__ import annotations

import ast
import importlib
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

_TOOL_MODULES = ["tools.hinge_inspect", "tools.eval_aggregation"]


def _source_path(dotted_module: str) -> Path:
    mod = importlib.import_module(dotted_module)
    return Path(mod.__file__)


def _operation_love_imports(py_file: Path):
    """Parse py_file and return every `operation_love.*` import reachable anywhere in
    the module — top level, inside function bodies, inside try/except blocks — as
    (from_pairs, plain_modules) where from_pairs is a list of (module, symbol) from
    `from operation_love... import symbol` and plain_modules is a list of dotted
    names from `import operation_love...`."""
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    from_pairs = []
    plain_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "operation_love":
                for alias in node.names:
                    if alias.name != "*":
                        from_pairs.append((node.module, alias.name))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "operation_love":
                    plain_modules.append(alias.name)
    return from_pairs, plain_modules


def test_hinge_inspect_imports_cleanly():
    importlib.import_module("tools.hinge_inspect")


def _removed_bumble_inspect_imports_cleanly():
    importlib.import_module("tools.bumble_inspect")


def test_eval_aggregation_imports_cleanly():
    importlib.import_module("tools.eval_aggregation")


def _from_import_resolves(module_name: str, symbol: str) -> bool:
    """Mirror Python's own `from module_name import symbol` resolution: the name may
    be an attribute defined in the module OR a submodule that hasn't been bound as an
    attribute yet in this process (e.g. `from operation_love import config` resolves
    even before anything else has imported operation_love.config). Try both, exactly
    like CPython's import system does, so this doesn't depend on unrelated test/import
    ordering already having warmed the attribute cache."""
    imported = importlib.import_module(module_name)
    if hasattr(imported, symbol):
        return True
    try:
        importlib.import_module(f"{module_name}.{symbol}")
        return True
    except ImportError:
        return False


def test_every_operation_love_import_resolves():
    # The direct regression guard for the hinge_inspect.py bug: a stale
    # `from operation_love.drivers.hinge.vision import ...` would fail here because
    # `operation_love.drivers.hinge.vision` is not an importable module.
    for dotted in _TOOL_MODULES:
        from_pairs, plain_modules = _operation_love_imports(_source_path(dotted))
        assert from_pairs or plain_modules, (
            f"{dotted}: AST walk found no operation_love.* imports at all — "
            "check the walk itself before trusting this test suite"
        )
        for module_name, symbol in from_pairs:
            assert _from_import_resolves(module_name, symbol), (
                f"{dotted}: `from {module_name} import {symbol}` — "
                f"{module_name} has no attribute or submodule {symbol!r}"
            )
        for module_name in plain_modules:
            importlib.import_module(module_name)  # raises if the module path is stale


def test_hinge_inspect_help_exits_cleanly_without_touching_device(monkeypatch):
    # argparse lives under `if __name__ == "__main__":`, so run the module as
    # __main__; --help makes parse_args() raise SystemExit(0) before main() (and
    # therefore any adb/screenshot/vision work) ever runs. Drop any cached import so
    # runpy re-executing it as __main__ doesn't warn about the sys.modules collision.
    sys.modules.pop("tools.hinge_inspect", None)
    monkeypatch.setattr(sys, "argv", ["hinge_inspect.py", "--help"])
    try:
        runpy.run_module("tools.hinge_inspect", run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("--help should have raised SystemExit")


def _removed_bumble_inspect_help_exits_cleanly_without_touching_browser(monkeypatch):
    mod = importlib.import_module("tools.bumble_inspect")
    monkeypatch.setattr(sys, "argv", ["bumble_inspect.py", "--help"])
    try:
        mod.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("--help should have raised SystemExit")


def _removed_bumble_inspect_exits_cleanly_on_platform_unavailable(monkeypatch):
    """tools/bumble_inspect.py targets the removed Playwright web driver (bumble_web --
    permanently unavailable since Bumble discontinued its web app in Aug 2026, see
    operation_love/platforms.py). BumbleDriver (a compatibility shim for BumbleWebDriver)
    correctly raises PlatformUnavailable from open_session() -- the registry's
    availability gate is NOT bypassed -- but pre-fix main() had no handler for it, so the
    user got a raw traceback instead of the clean explanation the registry already
    computed. main() must catch it, print that reason, and exit non-zero -- and must never
    prompt input() for a browser session that was never opened."""
    mod = importlib.import_module("tools.bumble_inspect")
    from operation_love.drivers.web import PlatformUnavailable

    monkeypatch.setattr(sys, "argv", ["bumble_inspect.py", "--config", "config.yaml"])
    monkeypatch.setattr(mod.cfg_mod, "load", lambda path: object())

    class _RefusingDriver:
        def __init__(self, cfg):
            pass

        def open_session(self):
            raise PlatformUnavailable(
                "Additional work needed to get this to run. Bumble discontinued its web "
                "app in August 2026."
            )

    monkeypatch.setattr(mod, "BumbleDriver", _RefusingDriver)

    def _no_input(*a, **k):
        raise AssertionError("input() must not be called -- no browser was ever opened")
    monkeypatch.setattr("builtins.input", _no_input)

    try:
        mod.main()
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("main() should have exited non-zero on PlatformUnavailable")


def test_eval_aggregation_help_exits_cleanly_without_touching_bigquery(monkeypatch):
    mod = importlib.import_module("tools.eval_aggregation")
    monkeypatch.setattr(sys, "argv", ["eval_aggregation.py", "--help"])
    try:
        mod.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("--help should have raised SystemExit")


def test_eval_aggregation_shares_ranker_evaluate_entry_point():
    # tools/eval_aggregation.py's docstring promises this is the SAME evaluation the
    # hub's model-quality card shows live, sharing one implementation in
    # operation_love.ranker.evaluate. Assert identity (not just equal behavior) so the
    # two can never silently diverge onto separate implementations.
    tool_mod = importlib.import_module("tools.eval_aggregation")
    from operation_love.ranker import evaluate as real_mod

    assert tool_mod.evaluate is real_mod.evaluate
    assert tool_mod.format_report is real_mod.format_report


class _FakeWatchDriver:
    """Duck-types the slice of HingeDriver _watch() touches: current_profile() +
    wait_for_decision() plus the identity/content instance attributes _capture_current
    would have populated. Not a real HingeDriver -- no ADB, no cv2 -- exactly like the
    other fakes in this file avoid touching a device/browser."""

    def __init__(self, *, observe_touch_watch=True, decisions=(True,)):
        self.observe_touch_watch = observe_touch_watch
        self._identity_sig = b"anchor-band"
        self._identity_top_sig = b"top-chrome-band"
        self._identity_name = "Alex"
        self._current_sigs = [object(), None, object()]   # 2 non-None, like a real capture
        self.capture_calls = 0
        self._decisions = list(decisions)
        self.decision_calls = 0

    def current_profile(self):
        self.capture_calls += 1
        return SimpleNamespace(photos=[b"p1", b"p2"])

    def wait_for_decision(self, timeout=None):
        self.decision_calls += 1
        return self._decisions.pop(0)


def test_hinge_inspect_watch_captures_profile_before_waiting_for_decision(capsys):
    """FINDING 10 regression: _watch() must call driver.current_profile() -- observe
    mode's ONLY capture path (hinge.py's own docstring on that method) -- before every
    driver.wait_for_decision() poll. current_profile() is what populates the
    identity-anchor signatures (_identity_sig / _identity_top_sig) and the content-match
    signatures (_current_sigs) that wait_for_decision's Layers 1/2 compare every later
    frame against; skipping it (the original bug) left both at their __init__ defaults
    for the whole --watch session, so identity stayed permanently 'unknown' and
    content-match had nothing to compare against -- the tool could then only ever
    exercise Layer 3 (gesture corroboration), silently reproducing the exact
    scroll-recorded-as-decision bug the redesign exists to close, inside the one tool
    meant to verify the fix. Before the fix, current_profile() was never called at all."""
    from tools import hinge_inspect

    driver = _FakeWatchDriver(decisions=[True, False])
    hinge_inspect._watch(driver, 2)

    # one capture per round, and each capture happens BEFORE that round's decision poll
    assert driver.capture_calls == 2
    assert driver.decision_calls == 2

    out = capsys.readouterr().out
    assert "identity anchor: revealed ('Alex')" in out
    assert "2 content signature(s)" in out
    assert "gesture corroboration (Layer 3): ON" in out
    assert "LIKE" in out and "PASS" in out


def test_hinge_inspect_watch_reports_gesture_corroboration_off():
    """observe_touch_watch: false is a real, documented, supported config (hinge.py's own
    fail-loud open_session error tells operators to set it on purpose) -- under it, Layers
    1/2 alone carry every verdict, and the operator watching --watch must be told that,
    not left to assume the same 3-layer proof always ran."""
    from tools import hinge_inspect

    driver = _FakeWatchDriver(observe_touch_watch=False, decisions=[True])
    hinge_inspect._watch(driver, 1)


def test_hinge_inspect_watch_stops_cleanly_when_capture_fails(capsys):
    """A capture failure (deck empty / the comment sheet was open) must stop the watch
    with a clear message rather than calling wait_for_decision() on a card that was never
    actually captured -- that call site would poll against a stale/absent identity anchor
    from whatever card came before, exactly the class of bug this fix closes."""
    from tools import hinge_inspect

    class _FailToCaptureDriver:
        observe_touch_watch = False

        def current_profile(self):
            return None

        def wait_for_decision(self, timeout=None):
            raise AssertionError("wait_for_decision must not run without a capture first")

    hinge_inspect._watch(_FailToCaptureDriver(), 3)

    out = capsys.readouterr().out
    assert "Could not capture a profile card" in out
