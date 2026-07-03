"""operation_love.__main__ — CLI dispatch (no test file exercised this before).

Each command lazily imports its target inside main(), so patching the target
module's attribute before invoking main() (via sys.argv) intercepts the real
call without doing real work (no browser, no ADB, no network).
"""
import sys

from operation_love import __main__ as main_mod


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["operation_love", *argv])
    main_mod.main()


def test_stats_command_dispatches_to_stats_show(monkeypatch):
    calls = []
    monkeypatch.setattr("operation_love.stats.show", lambda config: calls.append(config))
    _run(monkeypatch, ["stats", "--config", "x.yaml"])
    assert calls == ["x.yaml"]


def test_bugreport_command_dispatches_and_prints(monkeypatch, capsys):
    monkeypatch.setattr("operation_love.bugreport.build_report",
                        lambda state, config_path: f"REPORT for {config_path}")
    _run(monkeypatch, ["bugreport", "--config", "y.yaml"])
    assert "REPORT for y.yaml" in capsys.readouterr().out


def test_hub_command_serves_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "operation_love.hub.serve",
        lambda config, port, open_browser: calls.append((config, port, open_browser)))
    _run(monkeypatch, ["hub", "--config", "z.yaml", "--port", "9999"])
    assert calls == [("z.yaml", 9999, True)]


def test_hub_no_browser_flag_is_threaded_through(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "operation_love.hub.serve",
        lambda config, port, open_browser: calls.append(open_browser))
    _run(monkeypatch, ["hub", "--no-browser"])
    assert calls == [False]


def test_hub_make_launchers_flag_skips_serve(monkeypatch):
    served = []
    made = []
    monkeypatch.setattr("operation_love.hub.serve", lambda *a, **k: served.append(True))
    monkeypatch.setattr("operation_love.hub.make_launchers", lambda config: made.append(config))
    _run(monkeypatch, ["hub", "--make-launchers", "--config", "w.yaml"])
    assert made == ["w.yaml"] and served == []


def test_no_command_defaults_to_run(monkeypatch):
    calls = []
    monkeypatch.setattr("operation_love.supervisor.run", lambda config: calls.append(config))
    _run(monkeypatch, ["--config", "v.yaml"])
    assert calls == ["v.yaml"]
