"""Hub HTTP endpoints — offline smoke test (no real run started).

Spins the stdlib server in a thread and hits /, /api/config, /api/status,
/api/stop, and a 404. We never POST /api/start (that would launch a real run /
browser); start/stop wiring is covered by HubState's logic.
"""
import json
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from operation_love.hub import HubState, _Handler, _MAC_UPDATE_RUN, _PAGE, _bind

NODE_BIN = shutil.which("node")


def _extract_js_function(js: str, name: str) -> str:
    """Pull a top-level `function name(...) { ... }` out of the hub page's <script>, by
    brace-matching from the definition to its close. Evaluating the REAL body with node
    (rather than asserting a substring) means a comment-only revert of the guarded logic
    can't fool the test — see the audit note this addresses."""
    m = re.search(rf"(?:async\s+)?function\s+{re.escape(name)}\s*\([^)]*\)\s*{{", js)
    assert m, f"function {name} not found in hub.html"
    start = m.end() - 1                      # index of the opening '{'
    depth = 0
    for i in range(start, len(js)):
        if js[i] == "{":
            depth += 1
        elif js[i] == "}":
            depth -= 1
            if depth == 0:
                return js[m.start():i + 1]
    raise AssertionError(f"unbalanced braces extracting {name}")


def _run_node(script: str):
    assert NODE_BIN, "node not available"
    r = subprocess.run([NODE_BIN, "-e", script], capture_output=True, text=True, timeout=10)
    assert r.returncode == 0, f"node script failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
    return json.loads(r.stdout)


def _join_hub_watch_threads(timeout=3.0):
    """Wait for hub.py's `hub-tab-stale-watch` daemon threads to exit.

    /api/hub/open spawns one, and its loop only notices it should stop on its next wake —
    i.e. after `_BROWSER_STALE_CHECK_S`. A test that returns without waiting leaks a live
    thread that keeps sleeping and scheduler-contends with everything that runs after it,
    which is what made the full suite non-deterministic (individual files passed alone,
    the whole run failed in a different file each time). Pair this with monkeypatching
    `_BROWSER_STALE_CHECK_S` down so the wait is instant.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(t.name == "hub-tab-stale-watch" and t.is_alive()
                   for t in threading.enumerate()):
            return
        time.sleep(0.01)
    raise AssertionError("hub-tab-stale-watch thread outlived the test")


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=5) as r:
        return r.status, r.read().decode()


def _post(base, path, body=b"{}"):
    req = urllib.request.Request(base + path, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def test_hub_endpoints():
    _Handler.state = HubState("config.yaml")
    httpd = _bind("127.0.0.1", 8799)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        code, html = _get(base, "/")
        assert code == 200 and "Operation" in html and "/api/status" in html
        assert "model quality" in html
        assert "saving data" in html

        code, raw = _get(base, "/api/config")
        cfg = json.loads(raw)
        assert "mode" in cfg or "error" in cfg          # config.yaml loads from repo root

        code, raw = _get(base, "/api/status")
        snap = json.loads(raw)
        assert snap["running"] is False and snap["status"] is None

        code, body = _post(base, "/api/stop")
        assert body["ok"] is True

        try:
            _get(base, "/nope")
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()   # release the listening socket too; shutdown() alone leaves it open
        _join_hub_watch_threads()


def test_hub_tab_close_shuts_server_after_last_client(monkeypatch):
    # Collapse the reload grace so the test asserts the shutdown CONTRACT, not the constant's
    # value — otherwise retuning the grace silently breaks a test that isn't about timing.
    # Patch on hub.server directly: operation_love.hub (the package __init__) re-exports these
    # names as its OWN module-level attributes, a separate object from hub.server's globals —
    # patching the re-export wouldn't touch what _schedule_shutdown_if_tab_stayed_closed and
    # _start_stale_browser_watch actually read.
    from operation_love.hub import server as hub_server
    monkeypatch.setattr(hub_server, "_BROWSER_SHUTDOWN_GRACE_S", 0.01)
    monkeypatch.setattr(hub_server, "_BROWSER_STALE_CHECK_S", 0.01)   # so the watch thread can't outlive us
    _Handler.state = HubState("config.yaml")
    httpd = _bind("127.0.0.1", 8799)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        body = json.dumps({"id": "tab-1"}).encode()
        code, opened = _post(base, "/api/hub/open", body)
        assert code == 200 and opened["ok"] is True

        code, closed = _post(base, "/api/hub/closed", body)
        assert code == 200 and closed["ok"] is True

        t.join(timeout=4)
        assert t.is_alive() is False
    finally:
        httpd.shutdown()
        httpd.server_close()   # release the listening socket too; shutdown() alone leaves it open
        _join_hub_watch_threads()


def test_hub_tab_stale_heartbeat_shutdown_when_close_beacon_is_missing(monkeypatch):
    from operation_love.hub import server as hub_server, state as hub_state
    monkeypatch.setattr(hub_state, "_BROWSER_CLIENT_STALE_S", 0.05)
    monkeypatch.setattr(hub_server, "_BROWSER_STALE_CHECK_S", 0.01)
    monkeypatch.setattr(hub_server, "_BROWSER_SHUTDOWN_GRACE_S", 0.01)

    _Handler.state = HubState("config.yaml")
    httpd = _bind("127.0.0.1", 8799)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        body = json.dumps({"id": "tab-1"}).encode()
        code, opened = _post(base, "/api/hub/open", body)
        assert code == 200 and opened["ok"] is True

        t.join(timeout=2)
        assert t.is_alive() is False
    finally:
        httpd.shutdown()
        httpd.server_close()   # release the listening socket too; shutdown() alone leaves it open
        _join_hub_watch_threads()


def test_hubstate_double_start_blocked():
    st = HubState("config.yaml")
    # fake a live run so the second start is rejected without launching anything
    st._thread = threading.Thread(target=lambda: __import__("time").sleep(0.3))
    st._thread.start()
    ok, msg = st.start()
    assert ok is False and "active" in msg
    st._thread.join()


def test_wait_for_run_returns_true_when_thread_finishes_in_time():
    st = HubState("config.yaml")
    st._thread = threading.Thread(target=lambda: time.sleep(0.05))
    st._thread.start()
    assert st.wait_for_run(timeout=5) is True
    assert st._thread.is_alive() is False


def test_wait_for_run_returns_false_and_does_not_block_past_timeout():
    st = HubState("config.yaml")
    stuck = threading.Event()
    st._thread = threading.Thread(target=stuck.wait, daemon=True)   # never finishes on its own
    st._thread.start()
    start = time.monotonic()
    assert st.wait_for_run(timeout=0.1) is False
    assert time.monotonic() - start < 2.0       # returned promptly, did not hang
    assert st._thread.is_alive() is True
    stuck.set()
    st._thread.join(timeout=2)


def test_serve_shutdown_warns_and_exits_when_run_does_not_finish_in_time(monkeypatch):
    # A fake httpd (no real socket/serve_forever/shutdown handshake to fight) so this
    # test exercises serve()'s own try/except/finally control flow in isolation.
    import operation_love.hub.server as hub_server

    monkeypatch.setattr(hub_server, "_SHUTDOWN_SAVE_TIMEOUT_S", 0.05)

    stuck = threading.Event()
    state = HubState("config.yaml")
    state._thread = threading.Thread(target=stuck.wait, daemon=True)
    state._thread.start()
    monkeypatch.setattr(hub_server, "HubState", lambda config_path: state)

    class _FakeHttpd:
        server_address = ("127.0.0.1", 8799)

        def serve_forever(self):
            raise KeyboardInterrupt

        def shutdown(self):
            pass

    monkeypatch.setattr(hub_server, "_bind", lambda host, port: _FakeHttpd())

    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a))))
    try:
        hub_server.serve("config.yaml", open_browser=False)
    finally:
        stuck.set()
        state._thread.join(timeout=2)

    assert any("did not finish saving" in line for line in printed)


def test_hubstate_browser_client_lifecycle(monkeypatch):
    import operation_love.hub as hub
    now = 1000.0
    monkeypatch.setattr(hub.time, "time", lambda: now)

    st = HubState("config.yaml")
    st.browser_client_opened("a")
    st.browser_client_opened("b")
    assert st.has_browser_clients() is True

    assert st.browser_client_closed("a") is False
    assert st.has_browser_clients() is True

    assert st.browser_client_closed("b") is True
    assert st.has_browser_clients() is False
    assert st.browser_client_closed("b") is False

    # A late /api/hub/open from the just-closed page must not resurrect it.
    st.browser_client_opened("b")
    assert st.has_browser_clients() is False

    # The closed-id guard is only for late requests; it must not grow into a
    # permanent tombstone set over a long hub session.
    now += hub._CLOSED_BROWSER_CLIENT_TTL_S + 1
    st.browser_client_opened("b")
    assert st.has_browser_clients() is True
    assert st.browser_client_closed("b") is True

    st.browser_client_opened("c")
    assert st.has_browser_clients() is True
    assert st.browser_client_closed("c") is True


def test_hubstate_browser_heartbeat_prevents_stale_expiry(monkeypatch):
    import operation_love.hub as hub
    from operation_love.hub import state as hub_state
    now = 1000.0
    monkeypatch.setattr(hub.time, "time", lambda: now)
    monkeypatch.setattr(hub_state, "_BROWSER_CLIENT_STALE_S", 10.0)

    st = HubState("config.yaml")
    assert st.browser_client_opened("a") is True
    now += 9.0
    assert st.browser_client_ping("a") is False
    now += 9.0
    assert st.expire_stale_browser_clients() is False
    assert st.has_browser_clients() is True

    now += 11.0
    assert st.expire_stale_browser_clients() is True
    assert st.has_browser_clients() is False


def test_resolve_run_cap_override_semantics():
    from operation_love.supervisor import _resolve_run_cap
    assert _resolve_run_cap(30, None) == 30      # no override -> config value
    assert _resolve_run_cap(30, 0) is None       # 0 -> unlimited (no per-run cap)
    assert _resolve_run_cap(30, 8) == 8          # N -> cap this run at N
    assert _resolve_run_cap(None, 8) == 8        # config had no cap, override still applies


def test_hubstate_forwards_max_per_run(monkeypatch):
    import operation_love.hub as hub
    seen = {}
    done = threading.Event()

    def fake_run(config_path, **kw):
        seen.update(kw)
        done.set()

    monkeypatch.setattr(hub.supervisor, "run", fake_run)
    st = HubState("config.yaml")
    # hinge, not bumble: bumble is an Android target that starts out uncalibrated
    # (platforms.py) and HubState.start() now rejects an unrunnable selection up front.
    ok, _ = st.start(mode="auto", apps=["hinge"], max_per_run=8)
    assert ok is True
    assert done.wait(timeout=5)
    st._thread.join(timeout=5)
    assert seen["mode"] == "auto" and seen["max_per_run"] == 8


def test_hub_card_shows_single_accuracy_metric():
    # One metric (ROC-AUC as accuracy %); the old PR-AUC/ROC-AUC/Brier/diminishing lines are gone.
    assert "e.roc_auc[0]*100" in _PAGE                # accuracy = ROC-AUC x 100
    assert "ranks a like above a pass" in _PAGE
    assert "Brier score" not in _PAGE
    assert "diminishing returns" not in _PAGE
    assert "PR-AUC" not in _PAGE


def test_hub_page_reports_browser_tab_lifecycle():
    assert "/api/hub/open" in _PAGE
    assert "/api/hub/ping" in _PAGE
    assert "/api/hub/closed" in _PAGE
    assert "pagehide" in _PAGE
    assert "pageshow" in _PAGE
    assert "sendBeacon" in _PAGE
    assert "if(sent) return" in _PAGE
    assert "!event.persisted" in _PAGE


def test_committed_mac_launcher_matches_template():
    expected = _MAC_UPDATE_RUN.replace("__EXTRAS__", "ml,bq,bumble,hinge")
    assert Path("Operation Love.command").read_text() == expected


def test_launcher_default_extras_cover_every_deployable_optional_dependency():
    # Regression for the bug fixed in commit 640406b1: make_launchers()'s default `extras`
    # silently dropped `hinge` (opencv never installed -> Hinge's vision degraded to
    # fixed-coordinate taps with no warning). The committed launcher was hand-patched to
    # include it, but the GENERATOR's default argument was never fixed, so regenerating the
    # launcher would silently reintroduce the exact same gap. Guard the default itself
    # (not a generated file, which is platform-dependent) against every extra pyproject.toml
    # declares that the SHIPPED APP needs -- `dev` (pytest) is excluded on purpose: it's for
    # running this repo's test suite, not for using the app the launcher installs.
    import inspect
    import tomllib

    from operation_love.hub.launchers import make_launchers

    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    declared = set(pyproject["project"]["optional-dependencies"])
    deployable = declared - {"dev"}
    assert deployable, "sanity: pyproject.toml declares no deployable optional-dependencies"

    default_extras = inspect.signature(make_launchers).parameters["extras"].default
    have = set(default_extras.split(","))

    missing = deployable - have
    assert not missing, (
        f"make_launchers()'s default extras={default_extras!r} is missing {sorted(missing)} "
        "from pyproject.toml [project.optional-dependencies] -- a regenerated launcher "
        "would silently skip installing them."
    )


def test_hub_card_shows_label_gated_refresh_progress():
    # The model-quality card shows observe-mode "since/every till next refresh" progress.
    assert "till next refresh" in _PAGE
    assert "r.mode !== 'observe'" in _PAGE          # only while a live observe run feeds labels
    assert "r.since == null" in _PAGE
    assert "${progress}/${every}" in _PAGE


def test_attach_refresh_counts_down_to_next_recompute():
    class LiveObserve:
        running = True
        mode = "observe"

    # 2 of 5 new labels collected since the last compute -> 3 swipes remaining.
    r = HubState._attach_refresh({"status": "ok"}, every=5, live=42, base=40, status=LiveObserve())
    assert r["refresh"] == {"every": 5, "since": 2, "remaining": 3, "live": True, "mode": "observe"}

    # Right after a recompute (since == every) it rolls to a full cycle, never shows 0/5.
    edge = HubState._attach_refresh({"status": "ok"}, every=5, live=45, base=40, status=LiveObserve())
    assert edge["refresh"]["since"] == 5 and edge["refresh"]["remaining"] == 5


def test_hub_layout_drops_app_cards_and_reorders_controls():
    # App status cards removed; controls card sits between the live log and the bug report.
    assert 'id="apps"' not in _PAGE
    assert "renderApps" not in _PAGE
    assert _PAGE.index('id="logs"') < _PAGE.index('id="start"') < _PAGE.index('id="bugdesc"')


def test_hub_ranker_card_simplified_to_budget():
    # The top status card shows only the budget now (no ranker/progress-bar/labels line).
    assert 'id="barfill"' not in _PAGE
    assert 'id="ranker"' not in _PAGE
    assert 'id="budget"' in _PAGE


def test_hub_renders_accuracy_trajectory_chart():
    assert "accSvg(e.trajectory)" in _PAGE
    assert "accuracy over labels" in _PAGE


def test_attach_refresh_inactive_without_live_run():
    # No run at all -> countdown is inactive (card hides it); cached dict is never mutated.
    payload = {"status": "ok"}
    r = HubState._attach_refresh(payload, every=5, live=None, base=None, status=None)
    assert r["refresh"]["live"] is False and r["refresh"]["remaining"] is None
    assert "refresh" not in payload


def test_eval_snapshot_holds_full_progress_until_background_refresh_finishes(monkeypatch):
    import operation_love.hub as hub
    import operation_love.ranker as ranker
    import operation_love.ranker.evaluate as eval_mod

    class LiveObserve:
        running = True
        mode = "observe"
        labels = 45

    started = threading.Event()
    release = threading.Event()
    store_calls = []

    class Store:
        def load_labels(self):
            store_calls.append("load")
            started.set()
            assert release.wait(timeout=5)
            return [object()] * 45

        def close(self):
            pass

    def fake_make_store(cfg, ensure=True):
        assert ensure is False
        return Store()

    monkeypatch.setattr(hub.cfg_mod, "load", lambda path: object())
    monkeypatch.setattr(ranker, "make_store", fake_make_store)
    monkeypatch.setattr(eval_mod, "evaluate",
                        lambda samples: {"status": "ok", "marker": "new", "labels": len(samples)})

    st = HubState("config.yaml")
    with st._lock:
        st._eval = {"status": "ok", "marker": "old"}
        st._eval_at = time.time()
        st._eval_labels = 40
        st._status = LiveObserve()

    first = st.eval_snapshot(every=5)
    assert first["marker"] == "old"
    assert first["refresh"]["since"] == 5
    assert first["refresh"]["remaining"] == 5
    assert started.wait(timeout=5)

    second = st.eval_snapshot(every=5)
    assert second["marker"] == "old"
    assert len(store_calls) == 1

    release.set()
    deadline = time.time() + 5
    while time.time() < deadline:
        with st._lock:
            marker = st._eval.get("marker") if st._eval else None
            refreshing = st._eval_refreshing
        if marker == "new" and not refreshing:
            break
        time.sleep(0.01)
    else:
        assert False, "background eval did not finish"

    third = st.eval_snapshot(every=5)
    assert third["marker"] == "new"
    assert third["refresh"]["since"] == 0
    assert third["refresh"]["remaining"] == 5
    assert len(store_calls) == 1


def test_eval_snapshot_reads_live_store_while_running(monkeypatch):
    # Freshness fix: while a run is live, the card must read the worker's in-memory store
    # (every swipe, incl. unflushed) — NOT re-query BigQuery (committed-only, lagging).
    import operation_love.hub as hub
    import operation_love.ranker as ranker
    import operation_love.ranker.evaluate as eval_mod

    class LiveObserve:
        running = True
        mode = "observe"
        labels = 30

    live_calls = []

    class LiveStore:
        def load_labels(self):
            live_calls.append("live")
            return [object()] * 30           # the live in-memory set (all swipes)

    class Fresh:
        def load_labels(self):
            return [object()] * 5            # committed-only (would lag) — must NOT be used here
        def load_labels_ordered(self):
            return [object()] * 5
        def close(self):
            pass

    monkeypatch.setattr(hub.cfg_mod, "load", lambda p: object())
    monkeypatch.setattr(ranker, "make_store", lambda cfg, ensure=True: Fresh())
    monkeypatch.setattr(eval_mod, "evaluate", lambda s: {
        "status": "ok", "labels": len(s), "identities": len(s), "base_rate": 0.4,
        "pr_auc": [0.7, 0.05], "roc_auc": [0.8, 0.03], "brier": [0.2, 0.0]})
    monkeypatch.setattr(eval_mod, "quality_trajectory", lambda ordered, step=5: [])

    st = HubState("config.yaml")
    st._thread = threading.Thread(target=lambda: time.sleep(0.5))   # fake a live run
    st._thread.start()
    with st._lock:
        st._status = LiveObserve()
        st._live_store = LiveStore()
    res = st.eval_snapshot(every=5)
    st._thread.join()
    assert res["labels"] == 30 and live_calls == ["live"]   # used the live store, not a fresh BQ read


def test_eval_snapshot_falls_back_when_live_store_read_fails(monkeypatch):
    # Run-end race: if the live store is closed mid-read, the card must fall back to a
    # fresh committed read (not error out for a cycle).
    import operation_love.hub as hub
    import operation_love.ranker as ranker
    import operation_love.ranker.evaluate as eval_mod

    class LiveObserve:
        running = True
        mode = "observe"
        labels = 12

    class DeadStore:
        def load_labels(self):
            raise RuntimeError("Cannot operate on a closed database")

    fresh_used = []

    class Fresh:
        def load_labels(self):
            fresh_used.append(1)
            return [object()] * 12
        def load_labels_ordered(self):
            return []
        def close(self):
            pass

    monkeypatch.setattr(hub.cfg_mod, "load", lambda p: object())
    monkeypatch.setattr(ranker, "make_store", lambda cfg, ensure=True: Fresh())
    monkeypatch.setattr(eval_mod, "evaluate", lambda s: {
        "status": "ok", "labels": len(s), "identities": len(s), "base_rate": 0.4,
        "pr_auc": [0.7, 0.05], "roc_auc": [0.8, 0.03], "brier": [0.2, 0.0]})
    monkeypatch.setattr(eval_mod, "quality_trajectory", lambda ordered, step=5, **k: [])

    st = HubState("config.yaml")
    st._thread = threading.Thread(target=lambda: time.sleep(0.3))
    st._thread.start()
    with st._lock:
        st._status = LiveObserve()
        st._live_store = DeadStore()
    res = st.eval_snapshot(every=5)
    st._thread.join()
    assert res["status"] == "ok" and res["labels"] == 12   # recovered via the fresh read
    assert fresh_used                                       # the fallback store was used


def test_eval_snapshot_returns_error_dict_when_compute_raises(monkeypatch):
    # If config load / make_store / evaluate blow up, the card gets a well-formed error
    # dict (status=error, a message, identities None) — with the refresh block still attached.
    import operation_love.hub as hub

    def boom(path):
        raise RuntimeError("config blew up")

    monkeypatch.setattr(hub.cfg_mod, "load", boom)

    st = HubState("config.yaml")
    res = st.eval_snapshot(every=5)
    assert res["status"] == "error"
    assert res["message"]
    assert "config blew up" in res["message"]
    assert res["identities"] is None
    assert "refresh" in res


def test_hub_max_per_run_invalid_input_delegates_to_config():
    # Clearing/breaking the "max profiles this run" box sends null ("use config"), not
    # the distinct explicit-unlimited override 0. The shipped config is itself uncapped,
    # but this distinction matters if an operator later configures a standing ceiling.
    #
    # Evaluated for real with node (not a substring check): an earlier substring-only test
    # passed after the input handling was gutted as long as the phrase survived in a comment.
    # Extracting computeMaxPerRun() lets the test assert the actual result.
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fn = _extract_js_function(_PAGE, "computeMaxPerRun")
    # (mode, unlimitedChecked, rawValue)
    cases = [
        ("observe", False, "8"),    # not auto mode -> always null, regardless of the box
        ("auto", True, "8"),        # unlimited checkbox -> 0, regardless of the box
        ("auto", True, ""),
        ("auto", False, ""),        # empty box -> delegate to config
        ("auto", False, "abc"),     # non-numeric -> delegate to config
        ("auto", False, "0"),       # typing 0 directly delegates; checkbox owns explicit 0
        ("auto", False, "-3"),      # negative -> delegate to config
        ("auto", False, "3.7"),     # parseInt truncates
        ("auto", False, "8"),       # a genuine positive cap is honored
    ]
    expected = [None, 0, 0, None, None, None, None, 3, 8]
    script = (
        fn + "\n"
        "const cases = " + json.dumps(cases) + ";\n"
        "console.log(JSON.stringify(cases.map(c => computeMaxPerRun(c[0], c[1], c[2]))));\n"
    )
    assert _run_node(script) == expected


def test_hub_page_rereigsters_on_tab_wake_events():
    # Chrome throttles a hidden tab's setInterval ping to ~once/minute; the page must also
    # re-register the instant the tab visibly wakes (these fire un-throttled), same as pageshow.
    #
    # Evaluated for real with node: a mutation audit proved the old substring-only version of
    # this test still passed after both callback BODIES were gutted to no-ops, since the
    # "visibilitychange"/"addEventListener('focus'" substrings still appeared in the wiring
    # line. Extracting the callbacks as named functions lets the test call them and assert on
    # whether they actually re-registered.
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    assert "document.addEventListener('visibilitychange', _onHubVisibilityWake)" in _PAGE
    assert "window.addEventListener('focus', _onHubFocusWake)" in _PAGE
    vis_fn = _extract_js_function(_PAGE, "_onHubVisibilityWake")
    focus_fn = _extract_js_function(_PAGE, "_onHubFocusWake")
    script = (
        "let calls = [];\n"
        "function hubLifecycle(path){ calls.push(path); }\n"
        "let document = { hidden: true };\n"
        + vis_fn + "\n" + focus_fn + "\n"
        "_onHubVisibilityWake();\n"                 # hidden -> must NOT re-register
        "const afterHidden = calls.slice();\n"
        "document.hidden = false;\n"
        "_onHubVisibilityWake();\n"                 # became visible -> must re-register
        "const afterVisible = calls.slice();\n"
        "document.hidden = true;\n"                 # focus must fire regardless of hidden state
        "_onHubFocusWake();\n"
        "const afterFocus = calls.slice();\n"
        "console.log(JSON.stringify({afterHidden, afterVisible, afterFocus}));\n"
    )
    result = _run_node(script)
    assert result["afterHidden"] == []
    assert result["afterVisible"] == ["/api/hub/open"]
    assert result["afterFocus"] == ["/api/hub/open", "/api/hub/open"]


def test_browser_stale_window_clears_throttled_worst_case():
    import operation_love.hub as hub
    # Chrome's intensive timer throttling aligns a hidden tab's setInterval to ~once/minute;
    # the stale window must clear that with real margin so a throttled-but-alive tab is never
    # mistaken for closed and used to stop a live run. Assert the INTENT (comfortably above a
    # ~60s worst-case throttled ping) rather than pinning the exact constant.
    assert hub._BROWSER_CLIENT_STALE_S >= 90.0


def test_hub_dead_min_labels_key_removed():
    # min_labels was carried by config_defaults()/the JS default but nothing ever read it.
    assert "min_labels" not in _PAGE
    assert "min_labels" not in HubState("config.yaml").config_defaults()


def test_hubstate_start_rejects_empty_apps_list():
    # chosenApps() sends [] when every app checkbox is unchecked. Silently falling back to
    # config.enabled_apps (supervisor's behavior for a falsy override) would start the apps
    # the user just deselected — refuse instead, with a clear reason.
    st = HubState("config.yaml")
    ok, msg = st.start(apps=[])
    assert ok is False
    assert "no apps selected" in msg
    assert st.is_running() is False


def test_hubstate_start_rejects_unavailable_platform_with_registry_reason_verbatim(monkeypatch):
    # This is the mechanism behind "select Web-based, press Start, see 'Additional work
    # needed to get this to run...'" -- no frontend special-casing, the registry's reason
    # flows straight through to the existing {ok, msg} shape the hub already renders into
    # #hint. Also asserts no run/thread/driver is ever touched.
    import operation_love.hub as hub
    from operation_love import platforms

    called = []
    monkeypatch.setattr(hub.supervisor, "run", lambda config_path, **kw: called.append(kw))

    st = HubState("config.yaml")
    ok, msg = st.start(apps=["bumble_web"])
    assert ok is False
    assert msg == platforms.unavailable_reason("bumble_web")
    assert st.is_running() is False
    assert called == []                     # never reached supervisor.run


def test_hubstate_start_rejects_uncalibrated_android_platform(monkeypatch):
    import operation_love.hub as hub
    from operation_love import platforms

    monkeypatch.setattr(hub.supervisor, "run", lambda config_path, **kw: None)
    st = HubState("config.yaml")
    ok, msg = st.start(apps=["bumble"])
    assert ok is False
    assert msg == platforms.unavailable_reason("bumble")
    assert st.is_running() is False


def test_hubstate_start_rejects_two_android_platforms_together(monkeypatch):
    import operation_love.hub as hub

    monkeypatch.setattr(hub.supervisor, "run", lambda config_path, **kw: None)
    st = HubState("config.yaml")
    ok, msg = st.start(apps=["hinge", "bumble"])
    assert ok is False
    assert st.is_running() is False


def test_hubstate_start_apps_none_is_not_rejected(monkeypatch):
    # apps=None means "no override" (unrelated to the empty-list user-error case above) and
    # must still fall through to supervisor.run/config as before.
    import operation_love.hub as hub
    seen = {}
    done = threading.Event()

    def fake_run(config_path, **kw):
        seen.update(kw)
        done.set()

    monkeypatch.setattr(hub.supervisor, "run", fake_run)
    st = HubState("config.yaml")
    ok, msg = st.start(apps=None)
    assert ok is True
    assert done.wait(timeout=5)
    st._thread.join(timeout=5)
    assert seen["enabled_apps"] is None


def test_hubstate_retains_final_status_after_run_thread_exits(monkeypatch):
    import operation_love.hub as hub
    from operation_love.status import RunStatus

    def fake_run(config_path, **kwargs):
        status = RunStatus("finished", ["hinge"], min_labels=40, mode="auto")
        status.set_app("hinge", mode="auto", state="error", error="unrecognized screen")
        status.set_global(running=False, phase="stopped")
        kwargs["on_status"](status)

    monkeypatch.setattr(hub.supervisor, "run", fake_run)
    st = HubState("config.yaml")
    ok, _ = st.start(mode="auto", apps=["hinge"])
    assert ok is True
    st._thread.join(timeout=5)

    snap = st.snapshot()
    assert snap["running"] is False               # HubState follows Thread.is_alive()
    assert snap["status"] is not None             # final RunStatus is still available
    assert snap["status"]["apps"]["hinge"]["state"] == "error"


def test_hubstate_surfaces_systemexit_from_supervisor(monkeypatch):
    # supervisor.run raises SystemExit (a BaseException, not Exception) for a fatal startup
    # failure (e.g. missing cloud deps). It must surface in self._error like any other
    # failure, not be silently swallowed by threading's default SystemExit handling.
    import operation_love.hub as hub

    def boom(config_path, **kw):
        raise SystemExit("Storage.backend=bigquery but cloud storage dependencies are missing")

    monkeypatch.setattr(hub.supervisor, "run", boom)
    st = HubState("config.yaml")
    ok, _ = st.start()
    assert ok is True
    st._thread.join(timeout=5)
    assert st._error is not None
    assert st._error.startswith("SystemExit:")
    assert "missing" in st._error


def test_wait_for_run_honors_timeout():
    # Ctrl-C quit gives the run a bounded wait so a wedged worker can't hang quit forever.
    st = HubState("config.yaml")
    still_running = threading.Event()

    def slow():
        still_running.wait(timeout=5)

    st._thread = threading.Thread(target=slow, daemon=True)
    st._thread.start()
    t0 = time.time()
    st.wait_for_run(timeout=0.2)
    elapsed = time.time() - t0
    assert elapsed < 2.0             # returned promptly, not blocked for the full 5s
    assert st._thread.is_alive() is True   # timed out, didn't actually finish
    still_running.set()
    st._thread.join(timeout=5)


def test_swipe_banner_gates_on_per_app_mode_not_global_mode():
    # X4: config.yaml documents apps.<app>.mode overriding the global run mode, and
    # worker.py publishes that per-app mode via status.set_app(app, mode=...). The banner
    # is the ONLY cue in Hinge observe mode (no on-phone overlay), so it must key off each
    # app's OWN mode — not snap.status.mode (the global value), which would hide the
    # banner for an observe-mode app running under a global auto mode (or vice versa).
    #
    # Evaluated for real with node: a mutation audit proved the old substring-only version of
    # this test still passed after the gating was reverted to key off the global s.mode (the
    # exact pre-fix bug), because the reverted code still contained "observeApps" and an
    # "a.mode === 'observe'" comparison somewhere reachable. Extracting selectObserveApps() as
    # a pure function lets the test build a status where the global mode and the per-app modes
    # DISAGREE, and assert on which apps actually get selected.
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fn = _extract_js_function(_PAGE, "selectObserveApps")

    # Global mode says 'auto', but bumble is individually in 'observe' -> must still show for
    # bumble only (not all-or-nothing on the global mode).
    status_a = {
        "mode": "auto",
        "apps": {
            "bumble": {"mode": "observe", "state": "waiting"},
            "hinge": {"mode": "auto", "state": "acting"},
        },
    }
    script_a = (
        fn + "\n"
        "const status = " + json.dumps(status_a) + ";\n"
        "console.log(JSON.stringify(selectObserveApps(status).map(a => a.state)));\n"
    )
    assert _run_node(script_a) == ["waiting"]

    # Inverse: global mode says 'observe', but every app is individually in 'auto' -> must
    # select none. A global-mode-gated implementation would wrongly show the banner here.
    status_b = {"mode": "observe", "apps": {"bumble": {"mode": "auto", "state": "waiting"}}}
    script_b = (
        fn + "\n"
        "const status = " + json.dumps(status_b) + ";\n"
        "console.log(JSON.stringify(selectObserveApps(status).map(a => a.state)));\n"
    )
    assert _run_node(script_b) == []


def _observe_status_script(snap: dict) -> str:
    """Run the real observe banner renderer with the minimal DOM it needs."""
    fns = (_extract_js_function(_PAGE, "escHtml") + "\n"
           + _extract_js_function(_PAGE, "selectObserveApps") + "\n"
           + _extract_js_function(_PAGE, "renderSwipe"))
    return (
        "let el = {style:{display:''}, innerHTML:''};\n"
        "function $(sel){ return sel === '#swipebanner' ? el : null; }\n"
        + fns + "\n"
        "renderSwipe(" + json.dumps(snap) + ");\n"
        "console.log(JSON.stringify({display: el.style.display, html: el.innerHTML}));\n"
    )


def test_observe_banner_uses_click_language_and_replaces_it_with_hinge_opener():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    waiting = {
        "running": True,
        "status": {"apps": {"hinge": {"app": "hinge", "mode": "observe", "state": "waiting"}}},
    }
    ordinary = _run_node(_observe_status_script(waiting))
    assert ordinary["display"] == "block"
    assert "click pass X or heart" in ordinary["html"]
    assert "swipe" not in ordinary["html"].lower()

    # Hinge uses the physically labelled X/heart controls; other platforms may not.
    other_app = {
        "running": True,
        "status": {"apps": {"bumble": {"app": "bumble", "mode": "observe", "state": "waiting"}}},
    }
    generic = _run_node(_observe_status_script(other_app))
    assert "use the app's pass or like control" in generic["html"]
    assert "click pass X or heart" not in generic["html"]
    assert "swipe" not in generic["html"].lower()

    # Text comes from an AI response and must remain text, never markup in the hub.
    with_sheet = {
        "running": True,
        "status": {"apps": {"hinge": {
            "app": "hinge", "mode": "observe", "state": "waiting_for_send",
            "opener_suggestion": 'I like <your prompt> & "this"',
        }}},
    }
    suggestion = _run_node(_observe_status_script(with_sheet))
    assert "type this in Hinge, then tap Send Like" in suggestion["html"]
    assert "I like &lt;your prompt&gt; &amp; &quot;this&quot;" in suggestion["html"]
    assert "click pass X or heart" not in suggestion["html"]
    assert "<your prompt>" not in suggestion["html"]

    # An unavailable opener provider must not expose the internal waiting_for_send state
    # or leave the person without a next action; they can write their own opener instead.
    no_suggestion = {
        "running": True,
        "status": {"apps": {"hinge": {
            "app": "hinge", "mode": "observe", "state": "waiting_for_send",
            "opener_suggestion": None,
        }}},
    }
    fallback = _run_node(_observe_status_script(no_suggestion))
    assert "No suggestion available" in fallback["html"]
    assert "type your own opener, then tap Send Like" in fallback["html"]
    assert "waiting_for_send" not in fallback["html"]

    # The card uses innerHTML, so app names are escaped on non-opener branches too.
    capturing = {
        "running": True,
        "status": {"apps": {"hostile": {
            "app": '<img src=x onerror="alert(1)">', "mode": "observe", "state": "capturing",
        }}},
    }
    escaped_app = _run_node(_observe_status_script(capturing))
    assert "<img" not in escaped_app["html"] and "&lt;img" in escaped_app["html"]


def test_observe_banner_shows_wait_cue_while_a_suggestion_is_being_generated():
    """B: while the worker is blocked inside the (up to opener.request_timeout_s) advisory
    maybe_opener() call, the hub must show SOMETHING instead of the stale 'click pass X or
    heart' GO banner from before the heart tap -- otherwise the operator has no way to tell
    anything is happening. OWNER UI RULE: GO/WAIT cues use only 🟢/🔴 circles, so this must
    render as the same WAIT (🔴) style as capturing/acting/starting, not a new indicator."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    suggesting = {
        "running": True,
        "status": {"apps": {"hinge": {"app": "hinge", "mode": "observe", "state": "suggesting"}}},
    }
    out = _run_node(_observe_status_script(suggesting))
    assert out["display"] == "block"
    assert "🔴" in out["html"]                     # WAIT cue -- the owner's circle-only rule
    assert "🟢" not in out["html"]                  # not the GO cue this state replaces
    assert "💬" in out["html"]                      # stays visually tied to the suggestion feature
    assert "click pass X or heart" not in out["html"]   # the stale GO instruction must be gone
    assert "hinge" in out["html"]

    # The card uses innerHTML -- the app name must still be escaped on this branch too.
    hostile = {
        "running": True,
        "status": {"apps": {"hostile": {
            "app": '<img src=x onerror="alert(1)">', "mode": "observe", "state": "suggesting",
        }}},
    }
    escaped = _run_node(_observe_status_script(hostile))
    assert "<img" not in escaped["html"] and "&lt;img" in escaped["html"]


def test_select_auto_apps_filters_to_auto_mode_only():
    # selectAutoApps is renderAutoStatus's pure gating helper (mirrors selectObserveApps
    # above for the auto-mode banner): an app's OWN mode decides whether it's included,
    # not the global run mode, since apps.<app>.mode can override the run's mode per app.
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fn = _extract_js_function(_PAGE, "selectAutoApps")
    status = {
        "mode": "observe",
        "apps": {
            "bumble": {"mode": "auto", "state": "error", "app": "bumble"},
            "hinge": {"mode": "observe", "state": "waiting", "app": "hinge"},
        },
    }
    script = (
        fn + "\n"
        "const status = " + json.dumps(status) + ";\n"
        "console.log(JSON.stringify(selectAutoApps(status).map(a => a.app)));\n"
    )
    assert _run_node(script) == ["bumble"]

    # Inverse: every app individually in 'observe' -> select none, even under a global
    # 'auto' run mode. A global-mode-gated implementation would wrongly include hinge here.
    status_b = {"mode": "auto", "apps": {"hinge": {"mode": "observe", "state": "waiting", "app": "hinge"}}}
    script_b = (
        fn + "\n"
        "const status = " + json.dumps(status_b) + ";\n"
        "console.log(JSON.stringify(selectAutoApps(status).map(a => a.app)));\n"
    )
    assert _run_node(script_b) == []


def _autostatus_script(snap: dict) -> str:
    """renderAutoStatus (unlike the pure selector functions above) touches the DOM via
    `$('#autobanner')` -- shim just enough of that (a single fake element keyed by
    selector) for node to run the REAL function body, same approach the tab-wake test
    uses for `document`."""
    fns = (_extract_js_function(_PAGE, "escHtml") + "\n"
           + _extract_js_function(_PAGE, "selectAutoApps") + "\n"
           + _extract_js_function(_PAGE, "renderAutoStatus"))
    return (
        "let el = {style:{display:''}, innerHTML:''};\n"
        "function $(sel){ return sel === '#autobanner' ? el : null; }\n"
        + fns + "\n"
        "renderAutoStatus(" + json.dumps(snap) + ");\n"
        "console.log(JSON.stringify({display: el.style.display, html: el.innerHTML}));\n"
    )


def test_render_auto_status_shows_error_box_with_worker_message():
    # This is the hub-visible half of the supervisor fix: an auto-mode app that halted via
    # worker.py's HALT-on-unexpected path (unrecognized screen / any other exception) must
    # show that reason here, using the app's own `.error` message -- not just a raw log tail.
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {
        "running": True,
        "status": {
            "apps": {
                "hinge": {"app": "hinge", "mode": "auto", "state": "error",
                          "error": "UnlocatedControlError: unrecognized screen", "swipes_run": 3},
            },
        },
    }
    result = _run_node(_autostatus_script(snap))
    assert result["display"] == "block"
    assert "hinge" in result["html"]
    assert "UnlocatedControlError: unrecognized screen" in result["html"]


def test_render_auto_status_escapes_error_and_app_before_using_inner_html():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {
        "running": False,
        "status": {
            "apps": {
                "hostile": {"app": '<img src=x onerror="alert(1)">', "mode": "auto",
                            "state": "error", "error": "bad <script>alert(2)</script>"},
            },
        },
    }
    result = _run_node(_autostatus_script(snap))
    assert "<img" not in result["html"] and "<script" not in result["html"]
    assert "&lt;img" in result["html"] and "&lt;script&gt;" in result["html"]


def test_render_auto_status_shows_cold_start_defer_message():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {
        "running": True,
        "status": {
            "apps": {
                "bumble": {"app": "bumble", "mode": "auto", "state": "stopped",
                           "last_decision": "defer", "swipes_run": 0},
            },
        },
    }
    result = _run_node(_autostatus_script(snap))
    assert result["display"] == "block"
    assert "cold-start" in result["html"]


def test_render_auto_status_shows_opener_exhaustion_stop_reason():
    # WS-opener-reason: an auto-mode app halted because OpenerService ran out of opener
    # capacity (budget/credit/provider failure) used to render IDENTICALLY to a plain
    # operator-clicked Stop -- both were just state='stopped' with no reason field. The
    # banner must show the specific cause (AppStatus.stop_reason), the same way it already
    # shows `.error` for the exception path above.
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {
        "running": True,
        "status": {
            "apps": {
                "hinge": {"app": "hinge", "mode": "auto", "state": "stopped",
                          "stop_reason": "run budget reached", "swipes_run": 4},
            },
        },
    }
    result = _run_node(_autostatus_script(snap))
    assert result["display"] == "block"
    assert "run budget reached" in result["html"]
    assert "opener capacity exhausted" in result["html"]
    # The reason takes over the box's sub-line instead of the ordinary swipe count.
    assert "4 swipes this run" not in result["html"]


def test_render_auto_status_escapes_stop_reason_before_using_inner_html():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {
        "running": False,
        "status": {
            "apps": {
                "hinge": {"app": "hinge", "mode": "auto", "state": "stopped",
                          "stop_reason": "bad <script>alert(3)</script>"},
            },
        },
    }
    result = _run_node(_autostatus_script(snap))
    assert "<script" not in result["html"]
    assert "&lt;script&gt;" in result["html"]


def test_render_auto_status_bare_stop_still_shown_without_a_reason():
    # A plain operator-clicked Stop (no OpenerService involvement) must keep rendering
    # exactly as before -- stop_reason absent, not an empty string standing in for one.
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {
        "running": True,
        "status": {
            "apps": {"bumble": {"app": "bumble", "mode": "auto", "state": "stopped"}},
        },
    }
    result = _run_node(_autostatus_script(snap))
    assert result["display"] == "block"
    assert "bumble: stopped</div>" in result["html"]


def test_render_auto_status_hides_without_auto_apps_but_survives_run_completion():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    # No auto-mode apps at all (only observe) -> hide.
    snap_no_auto = {
        "running": True,
        "status": {"apps": {"hinge": {"app": "hinge", "mode": "observe", "state": "waiting"}}},
    }
    result_a = _run_node(_autostatus_script(snap_no_auto))
    assert result_a["display"] == "none"

    # A completed auto run must retain its terminal explanation. HubState.running follows
    # Thread.is_alive(), so this is the state a person returning after the run sees.
    snap_not_running = {
        "running": False,
        "status": {"apps": {"hinge": {"app": "hinge", "mode": "auto", "state": "error",
                                           "error": "unrecognized screen"}}},
    }
    result_b = _run_node(_autostatus_script(snap_not_running))
    assert result_b["display"] == "block"
    assert "unrecognized screen" in result_b["html"]


def test_render_auto_status_does_not_claim_running_after_the_run_ended():
    """The banner deliberately outlives the run (see the test above), which makes the
    fall-through branch's old assumption -- "not a known terminal state => the worker is
    live" -- wrong once `running` is false. A run whose shutdown never reached its final
    status.set_app leaves a non-terminal state like 'capturing' behind; pre-fix that
    rendered a GREEN "hinge: running" box while the run pill said stopped -- directly
    contradictory, and it hid the fact that the run ended abnormally."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {
        "running": False,
        "status": {
            "apps": {
                "hinge": {"app": "hinge", "mode": "auto", "state": "capturing", "swipes_run": 7},
            },
        },
    }
    result = _run_node(_autostatus_script(snap))
    assert result["display"] == "block"
    assert "running" not in result["html"]        # never claim a live worker after the run ended
    assert "ended" in result["html"]              # ...say the run ended instead
    assert "capturing" in result["html"]          # ...and surface the abnormal last state
    assert "7 swipes this run" in result["html"]

    # Same non-terminal state while the run IS live still reads as running.
    live = {"running": True, "status": {"apps": {"hinge": dict(snap["status"]["apps"]["hinge"])}}}
    assert "hinge: running" in _run_node(_autostatus_script(live))["html"]


def test_tick_drives_real_auto_status_renderer_and_page_owns_banner_element():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    assert re.search(r'<div\s+id="autobanner"(?:\s|>)', _PAGE)
    tick_fn = _extract_js_function(_PAGE, "tick")
    script = (
        "const calls = [];\n"
        "async function getJSON(){ return {running:false,status:{apps:{}}}; }\n"
        "function renderGlobal(){ calls.push('global'); }\n"
        "function renderSwipe(){ calls.push('swipe'); }\n"
        "function renderAutoStatus(){ calls.push('auto'); }\n"
        + tick_fn + "\n"
        "tick().then(() => console.log(JSON.stringify(calls)));\n"
    )
    assert _run_node(script) == ["global", "swipe", "auto"]


def test_hub_auto_volume_control_defaults_to_unlimited():
    tag = re.search(r'<input\b[^>]*\bid="unlimited"[^>]*>', _PAGE)
    assert tag, "unlimited control missing"
    assert re.search(r'\bchecked(?:\s|=|>)', tag.group(0)), (
        "the shipped config is uncapped, so a normal hub start must not silently add a cap"
    )


def test_hub_start_handler_posts_explicit_unlimited_override():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    assert re.search(r"\$\('#start'\)\.onclick\s*=\s*startRunFromControls", _PAGE)
    fns = (_extract_js_function(_PAGE, "escHtml") + "\n"
           + _extract_js_function(_PAGE, "computeMaxPerRun") + "\n"
           + _extract_js_function(_PAGE, "startRunFromControls"))
    script = (
        "const els = {hint:{textContent:''}, mode:{value:'auto'}, unlimited:{checked:true}, "
        "maxrun:{value:'8'}, platnote:{innerHTML:''}};\n"
        "function $(sel){ return els[sel.slice(1)]; }\n"
        "function chosenApps(){ return ['hinge']; }\n"
        "let posted = null;\n"
        "async function postJSON(path, body){ posted={path,body}; return {ok:true,msg:'started'}; }\n"
        "function tick(){}\n"
        + fns + "\n"
        "startRunFromControls().then(() => console.log(JSON.stringify(posted)));\n"
    )
    assert _run_node(script) == {
        "path": "/api/start",
        "body": {"mode": "auto", "apps": ["hinge"], "max_per_run": 0},
    }


def test_eval_snapshot_cold_start_single_flights_concurrent_pollers(monkeypatch):
    # H-06: the page polls /api/eval every 5s and ThreadingHTTPServer gives each request
    # its own thread. On a cold cache (_eval is None), concurrent pollers must share ONE
    # underlying computation instead of each running a full grouped CV.
    import operation_love.hub as hub
    import operation_love.ranker as ranker
    import operation_love.ranker.evaluate as eval_mod

    started = threading.Event()
    release = threading.Event()
    compute_calls = []

    class Store:
        def load_labels(self):
            compute_calls.append(1)
            started.set()
            assert release.wait(timeout=5)
            return [object()] * 10

        def close(self):
            pass

    monkeypatch.setattr(hub.cfg_mod, "load", lambda path: object())
    monkeypatch.setattr(ranker, "make_store", lambda cfg, ensure=True: Store())
    monkeypatch.setattr(eval_mod, "evaluate",
                        lambda samples: {"status": "ok", "labels": len(samples),
                                          "identities": len(samples)})

    st = HubState("config.yaml")
    results = []
    errors = []

    def caller():
        try:
            results.append(st.eval_snapshot(every=5))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=caller) for _ in range(5)]
    for t in threads:
        t.start()
    assert started.wait(timeout=5)     # one caller is now inside the (single) computation
    time.sleep(0.05)                   # let the other pollers reach the cold path as waiters
    release.set()
    for t in threads:
        t.join(timeout=5)

    assert not errors
    assert len(results) == 5
    assert len(compute_calls) == 1     # only ONE actual computation ran despite 5 pollers
    assert all(r["status"] == "ok" and r["labels"] == 10 for r in results)


def test_bugreport_uses_hub_config_path_not_default():
    # A hub started with --config other.yaml must produce a report describing THAT config,
    # not silently default to config.yaml.
    _Handler.state = HubState("definitely-not-a-real-config.yaml")
    httpd = _bind("127.0.0.1", 8799)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        code, md = _get(base, "/api/bugreport")
        assert code == 200
        assert "definitely-not-a-real-config.yaml" in md
    finally:
        httpd.shutdown()
        httpd.server_close()   # release the listening socket too; shutdown() alone leaves it open
        _join_hub_watch_threads()


# --- platform picker (two-tier: kind -> app) ----------------------------------
# Bumble discontinued its web app in Aug 2026, so "which dating app" and "how we drive
# it" no longer line up one to one: Bumble became a second ANDROID target beside Hinge,
# and "web" is a transport with no live target. The picker models that as kind -> app.
# These evaluate the REAL functions with node rather than substring-matching the page,
# for the same reason the computeMaxPerRun test does.

_PICKER_CFG = {
    "kinds": [
        {
            "kind": "android",
            "label": "App-based",
            "platforms": [
                {"app": "hinge", "label": "Hinge", "available": True, "reason": None},
                {"app": "bumble", "label": "Bumble", "available": False, "reason": "not calibrated"},
            ],
        },
        {
            "kind": "web",
            "label": "Web-based",
            "platforms": [
                {"app": "bumble_web", "label": "Bumble (web)", "available": False,
                 "reason": "Additional work needed to get this to run."},
            ],
        },
    ],
    "selected": {"kind": "android", "app": "hinge"},
}


def _picker_script(body: str, *names) -> str:
    return "\n".join(_extract_js_function(_PAGE, n) for n in names) + "\n" + body


def test_picker_expands_only_a_kind_with_more_than_one_target():
    # App-based has two apps so it must expand to a choice; Web-based has exactly one
    # (unavailable) target, so expanding it would be a pointless one-item list -- the
    # kind button IS the selection there.
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    script = _picker_script(
        "const cfg = " + json.dumps(_PICKER_CFG) + ";\n"
        "console.log(JSON.stringify(["
        "  shouldExpand(kindEntry(cfg, 'android')),"
        "  shouldExpand(kindEntry(cfg, 'web')),"
        "  shouldExpand(kindEntry(cfg, 'nonesuch')),"
        "]));\n",
        "kindEntry", "shouldExpand",
    )
    assert _run_node(script) == [True, False, False]


def test_picker_defaults_to_an_available_target_within_a_kind():
    # Ordering must not decide this: Hinge is available and Bumble is not, so clicking
    # "App-based" has to land on Hinge even though both are listed.
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    reversed_cfg = {"kinds": [{
        "kind": "android", "label": "App-based",
        "platforms": list(reversed(_PICKER_CFG["kinds"][0]["platforms"])),
    }]}
    script = _picker_script(
        "const a = " + json.dumps(_PICKER_CFG) + ";\n"
        "const b = " + json.dumps(reversed_cfg) + ";\n"
        "console.log(JSON.stringify(["
        "  defaultAppForKind(kindEntry(a, 'android')),"
        "  defaultAppForKind(kindEntry(b, 'android')),"
        "  defaultAppForKind(kindEntry(a, 'web')),"   # none available -> first anyway
        "  defaultAppForKind(null),"
        "]));\n",
        "kindEntry", "defaultAppForKind",
    )
    assert _run_node(script) == ["hinge", "hinge", "bumble_web", None]


def test_picker_initial_selection_honours_server_then_falls_back_to_runnable():
    # The server's `selected` comes from config's enabled_apps and wins when it still
    # names a registered platform. When it names something the registry dropped, we must
    # land on a kind that actually has a runnable target rather than on a dead one --
    # otherwise a stale config silently parks the hub on a platform that cannot start.
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    stale = dict(_PICKER_CFG, selected={"kind": "android", "app": "tinder"})
    web_first = {"kinds": list(reversed(_PICKER_CFG["kinds"])), "selected": None}
    script = _picker_script(
        "const cases = " + json.dumps([_PICKER_CFG, stale, web_first, {"kinds": []}]) + ";\n"
        "console.log(JSON.stringify(cases.map(initialSelection)));\n",
        "kindEntry", "defaultAppForKind", "initialSelection",
    )
    assert _run_node(script) == [
        {"kind": "android", "app": "hinge"},   # server's choice honoured
        {"kind": "android", "app": "hinge"},   # unknown app -> first kind with a runnable target
        {"kind": "android", "app": "hinge"},   # web listed first, but nothing there can run
        {"kind": None, "app": None},           # empty registry -> select nothing
    ]


def test_picker_selection_is_single_not_multi():
    # Both dating apps now live on the one physical Pixel, and Android foregrounds a
    # single app -- screencap captures whatever is on top and the virtual touchscreen
    # delivers to whatever holds focus -- so the picker must never be able to ask for
    # two at once. chosenApps() is what the Start button posts.
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    script = (
        _extract_js_function(_PAGE, "chosenApps") + "\n"
        "let sel = {kind:'android', app:'hinge'};\n"
        "const one = chosenApps();\n"
        "sel = {kind:null, app:null};\n"
        "console.log(JSON.stringify([one, chosenApps()]));\n"
    )
    assert _run_node(script) == [["hinge"], []]
