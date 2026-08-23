"""Hub HTTP endpoints — offline smoke test (no real run started).

Spins the stdlib server in a thread and hits /, /api/config, /api/status,
/api/stop, and a 404. We never POST /api/start (that would launch a real run /
browser); start/stop wiring is covered by HubState's logic.
"""
import errno
import http.client
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from operation_love.hub import HubState, _Handler, _MAC_UPDATE_RUN, _PAGE, _bind

NODE_BIN = shutil.which("node")

# Liveness bound, not a performance bound: it exists only so a genuine hang fails a test
# instead of hanging the whole suite forever. Widened 2026-08-22 when `python -m pytest` moved
# to one worker per core (pyproject.toml addopts `-n auto --dist loadgroup`), which measured a
# ~15x slowdown (0.33s idle vs 5.06s under load) on tests/test_concurrency.py's positive
# liveness waits of the same shape. Nothing about the property under test (did the background
# thread reach the expected state / finish?) depends on the exact number, so widening it loses
# nothing.
_LIVENESS_TIMEOUT_S = 15.0


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
    r = subprocess.run(
        [NODE_BIN, "-e", script], capture_output=True, text=True, timeout=10, check=False)
    assert r.returncode == 0, f"node script failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
    return json.loads(r.stdout)


def _join_hub_watch_threads(timeout=_LIVENESS_TIMEOUT_S):
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


def _get(base, path, *, timeout=5):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return r.status, r.read().decode()


def _csrf_token(base):
    _, page = _get(base, "/")
    match = re.search(r'<meta name="operation-love-csrf" content="([^"]+)">', page)
    assert match, "served hub page did not contain a CSRF token"
    return match.group(1)


def _post(base, path, body=b"{}"):
    payload = json.loads(body)
    payload["_csrf_token"] = _csrf_token(base)
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


@pytest.fixture
def secured_stop_hub():
    class StopSpy:
        def __init__(self):
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            return True, "stopping"

    state = StopSpy()
    _Handler.state = state
    httpd = _bind("127.0.0.1", 8799)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", httpd, state
    finally:
        httpd.shutdown()
        httpd.server_close()
        _join_hub_watch_threads()


def _raw_post(base, path, body, headers):
    request = urllib.request.Request(
        base + path, data=body, method="POST", headers=headers)
    return urllib.request.urlopen(request, timeout=5)


def test_hub_tokens_are_unique_per_server_and_only_rendered_into_served_page():
    first = _bind("127.0.0.1", 8799)
    second = _bind("127.0.0.1", 8799)
    thread = threading.Thread(target=first.serve_forever, daemon=True)
    thread.start()
    try:
        assert first.csrf_token != second.csrf_token
        assert len(first.csrf_token) >= 32
        base = f"http://127.0.0.1:{first.server_address[1]}"
        _, page = _get(base, "/")
        assert "__OPERATION_LOVE_CSRF_TOKEN__" not in page
        assert f'content="{first.csrf_token}"' in page
        assert second.csrf_token not in page
    finally:
        first.shutdown()
        first.server_close()
        second.server_close()


def test_bind_rejects_non_loopback_host():
    with pytest.raises(ValueError, match="localhost or 127\\.0\\.0\\.1"):
        _bind("0.0.0.0", 8799)


@pytest.mark.parametrize("host", [" localhost ", "127.0.0.1 ", "LOCALHOST"])
def test_bind_rejects_noncanonical_loopback_host_spelling(host):
    with pytest.raises(ValueError, match="localhost or 127\\.0\\.0\\.1"):
        _bind(host, 8799)


@pytest.mark.parametrize("port", [True, False, 0, -1, 65536, "8799", 1.5])
def test_bind_rejects_invalid_ports_without_calling_socket_layer(monkeypatch, port):
    from operation_love.hub import server as hub_server

    calls = []
    monkeypatch.setattr(
        hub_server, "_HubHTTPServer", lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(ValueError, match="1 to 65535"):
        hub_server._bind("127.0.0.1", port)
    assert calls == []


def test_bind_retries_only_address_in_use_and_never_scans_past_65535(monkeypatch):
    from operation_love.hub import server as hub_server

    calls = []

    def occupied(address, handler):
        del handler
        calls.append(address)
        raise OSError(errno.EADDRINUSE, "already in use")

    monkeypatch.setattr(hub_server, "_HubHTTPServer", occupied)
    with pytest.raises(SystemExit, match=r"65535\.\.65535"):
        hub_server._bind("127.0.0.1", 65535)
    assert calls == [("127.0.0.1", 65535)]


def test_bind_reraises_non_contention_socket_errors_immediately(monkeypatch):
    from operation_love.hub import server as hub_server

    calls = []

    def denied(address, handler):
        del handler
        calls.append(address)
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(hub_server, "_HubHTTPServer", denied)
    with pytest.raises(OSError) as exc_info:
        hub_server._bind("127.0.0.1", 8799)
    assert exc_info.value.errno == errno.EACCES
    assert calls == [("127.0.0.1", 8799)]


@pytest.mark.parametrize("origin", [
    "https://attacker.example",
    "null",
])
def test_api_stop_rejects_hostile_origin_without_stopping(secured_stop_hub, origin):
    base, _httpd, state = secured_stop_hub
    body = json.dumps({"_csrf_token": _csrf_token(base)}).encode()
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _raw_post(base, "/api/stop", body, {
            "Content-Type": "application/json",
            "Origin": origin,
        })
    assert exc_info.value.code == 403
    assert state.stop_calls == 0


def test_api_stop_rejects_a_different_loopback_origin(secured_stop_hub):
    base, httpd, state = secured_stop_hub
    origin = f"http://localhost:{httpd.server_address[1]}"
    body = json.dumps({"_csrf_token": _csrf_token(base)}).encode()
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _raw_post(base, "/api/stop", body, {
            "Content-Type": "application/json",
            "Origin": origin,
        })
    assert exc_info.value.code == 403
    assert state.stop_calls == 0


@pytest.mark.parametrize(("content_type", "body_kind"), [
    ("text/plain", "json"),
    ("application/x-www-form-urlencoded", "empty"),
])
def test_api_stop_rejects_non_json_and_empty_form_posts(
        secured_stop_hub, content_type, body_kind):
    base, _httpd, state = secured_stop_hub
    body = (
        json.dumps({"_csrf_token": _csrf_token(base)}).encode()
        if body_kind == "json" else b""
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _raw_post(base, "/api/stop", body, {
            "Content-Type": content_type,
            "Origin": base,
        })
    assert exc_info.value.code == 415
    assert state.stop_calls == 0


def test_api_stop_rejects_duplicate_content_length_before_parsing(secured_stop_hub):
    base, httpd, state = secured_stop_hub
    body = json.dumps({"_csrf_token": _csrf_token(base)}).encode()
    connection = http.client.HTTPConnection(
        "127.0.0.1", httpd.server_address[1], timeout=5)
    try:
        connection.putrequest("POST", "/api/stop")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(len(body)))
        connection.putheader("Content-Length", str(len(body)))
        connection.putheader("Origin", base)
        connection.endheaders(body)
        response = connection.getresponse()
        payload = json.loads(response.read())
    finally:
        connection.close()

    assert response.status == 400
    assert "Content-Length" in payload["msg"]
    assert state.stop_calls == 0


@pytest.mark.parametrize("token", [None, "not-the-server-token"])
def test_api_stop_rejects_missing_or_wrong_csrf_token(secured_stop_hub, token):
    base, _httpd, state = secured_stop_hub
    payload = {} if token is None else {"_csrf_token": token}
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _raw_post(base, "/api/stop", json.dumps(payload).encode(), {
            "Content-Type": "application/json",
            "Origin": base,
        })
    assert exc_info.value.code == 403
    assert state.stop_calls == 0


@pytest.mark.parametrize("host", ["attacker.example", "127.0.0.1:1"])
def test_hub_rejects_invalid_host_before_serving_token(secured_stop_hub, host):
    base, _httpd, state = secured_stop_hub
    request = urllib.request.Request(base + "/", headers={"Host": host})
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(request, timeout=5)
    assert exc_info.value.code == 403
    assert state.stop_calls == 0


def test_api_stop_accepts_valid_same_origin_json_and_csrf_token(secured_stop_hub):
    base, _httpd, state = secured_stop_hub
    body = json.dumps({"_csrf_token": _csrf_token(base)}).encode()
    with _raw_post(base, "/api/stop", body, {
        "Content-Type": "application/json; charset=utf-8",
        "Origin": base,
    }) as response:
        assert response.status == 200
        assert json.loads(response.read())["ok"] is True
    assert state.stop_calls == 1


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
        assert "mode" in cfg                             # config.yaml loads from repo root
        hinge = next(p for kind in cfg["kinds"] for p in kind["platforms"]
                     if p["app"] == "hinge")
        assert hinge["modes"] == {"observe": True, "auto": False}
        assert hinge["mode_reasons"]["observe"] is None
        assert hinge["mode_reasons"]["auto"].startswith("Hinge Auto is blocked")

        code, raw = _get(base, "/api/status")
        snap = json.loads(raw)
        assert snap["running"] is False and snap["status"] is None

        # No run is active in this offline smoke test -- HubState.stop() must refuse
        # honestly (ok=False, 409) rather than the old unconditional "stopping" success.
        try:
            _post(base, "/api/stop")
            assert False, "expected 409 (no run active)"
        except urllib.error.HTTPError as e:
            assert e.code == 409
            body = json.loads(e.read())
            assert body["ok"] is False
            assert "no run is active" in body["msg"]

        try:
            _get(base, "/nope")
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()   # release the listening socket too; shutdown() alone leaves it open
        _join_hub_watch_threads()


def test_every_hub_response_has_anti_framing_and_content_hardening_headers():
    _Handler.state = HubState("config.yaml")
    httpd = _bind("127.0.0.1", 8799)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        for path in ("/", "/api/status", "/favicon.ico", "/not-found"):
            try:
                response = urllib.request.urlopen(base + path, timeout=5)
            except urllib.error.HTTPError as exc:
                response = exc
            try:
                assert response.headers["Content-Security-Policy"] == "frame-ancestors 'none'"
                assert response.headers["X-Frame-Options"] == "DENY"
                assert response.headers["X-Content-Type-Options"] == "nosniff"
                assert response.headers["Referrer-Policy"] == "no-referrer"
            finally:
                response.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        _join_hub_watch_threads()


def test_api_stop_returns_200_with_ok_true_when_a_run_is_active():
    # Mirrors /api/start's existing status-code convention (200 on success, 409 on refusal)
    # -- the handler must apply the same rule to HubState.stop()'s new (ok, msg) contract.
    _Handler.state = HubState("config.yaml")
    _Handler.state._stop = threading.Event()
    _Handler.state._thread = threading.Thread(target=lambda: time.sleep(0.3), daemon=True)
    _Handler.state._thread.start()
    httpd = _bind("127.0.0.1", 8799)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        code, body = _post(base, "/api/stop")
        assert code == 200 and body["ok"] is True
        assert _Handler.state._stop.is_set() is True
    finally:
        httpd.shutdown()
        httpd.server_close()
        _Handler.state._thread.join(timeout=_LIVENESS_TIMEOUT_S)
        _join_hub_watch_threads()


def test_api_status_snapshot_carries_stop_kind_alongside_stop_reason():
    """The real /api/status HTTP endpoint -- not just RunStatus/HubState in isolation -- must
    actually serialize AppStatus.stop_kind through the handler's own json.dumps, alongside
    stop_reason. stop_kind disambiguates stop_reason's SOURCE now that two different worker.py
    code paths populate it: OpenerService exhaustion (the original, sole source) and the
    blocked-deck check (Hinge's out-of-free-likes Hinge+ paywall) -- see status.py's
    AppStatus.stop_kind docstring. A
    consumer that only ever saw stop_reason (older hub.html, an external tool reading this
    endpoint) would misreport a blocked deck as an opener/quota problem without this field
    actually reaching the wire."""
    from operation_love.status import RunStatus

    _Handler.state = HubState("config.yaml")
    status = RunStatus("r1", ["hinge"], min_labels=1, mode="observe")
    status.set_app(
        "hinge", mode="observe", state="blocked",
        stop_reason="Hinge is out of free likes for today — the Hinge+ upgrade screen is up",
        stop_kind="deck_blocked")
    with _Handler.state._lock:
        _Handler.state._status = status

    httpd = _bind("127.0.0.1", 8799)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        code, raw = _get(base, "/api/status")
        app = json.loads(raw)["status"]["apps"]["hinge"]
        assert app["stop_kind"] == "deck_blocked"
        assert app["stop_reason"] == (
            "Hinge is out of free likes for today — the Hinge+ upgrade screen is up")
    finally:
        httpd.shutdown()
        httpd.server_close()
        _join_hub_watch_threads()


def test_api_status_snapshot_stop_kind_defaults_to_none_for_ordinary_stops():
    """A plain operator-clicked Stop (or any other terminal path that never populates
    stop_reason) must leave stop_kind at AppStatus's own default of None in the served JSON --
    never a stray truthy placeholder that would make a consumer of this endpoint (the bug
    report's _app_diagnostics_md, or a future hub-UI branch) render an explanation for a stop
    that was never actually disambiguated. Two apps in one snapshot: "hinge" never even calls
    set_app with a stop_reason/stop_kind (the AppStatus dataclass default, exercised by a run
    that hasn't stopped at all yet); "bumble" reaches an explicit state="stopped" the same way
    a manual Stop click does, again with no stop_reason/stop_kind -- both must serialize the
    same None, not two different "absent" shapes."""
    from operation_love.status import RunStatus

    _Handler.state = HubState("config.yaml")
    status = RunStatus("r1", ["hinge", "bumble"], min_labels=1, mode="observe")
    status.set_app("bumble", mode="observe", state="stopped")
    with _Handler.state._lock:
        _Handler.state._status = status

    httpd = _bind("127.0.0.1", 8799)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        code, raw = _get(base, "/api/status")
        apps = json.loads(raw)["status"]["apps"]
        assert apps["hinge"]["stop_kind"] is None
        assert apps["hinge"]["stop_reason"] is None
        assert apps["bumble"]["stop_kind"] is None
        assert apps["bumble"]["stop_reason"] is None
    finally:
        httpd.shutdown()
        httpd.server_close()
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

        t.join(timeout=_LIVENESS_TIMEOUT_S)
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

        t.join(timeout=_LIVENESS_TIMEOUT_S)
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


# --- audit fix: Stop must be honest -- both about whether it did anything, and (on the
# terminal/hub live-log panel, since bugreport.install_log_capture tees stdout) that a swipe
# made after this point won't be recorded. -------------------------------------------------

def test_hubstate_stop_without_active_run_returns_false_with_reason():
    # Pre-fix this always returned (True, "stopping") even with nothing to stop -- a stray
    # /api/stop (double-click, stale tab) looked like it had worked.
    st = HubState("config.yaml")
    ok, msg = st.stop()
    assert ok is False
    assert msg == "no run is active"


def _live_hubstate():
    """A HubState that looks like it has a run in progress: stop() gates on LIVENESS, not on
    the mere existence of an Event (self._stop is set by start() and never reset, so a gate on
    the object alone would keep reporting success long after a run ended)."""
    st = HubState("config.yaml")
    st._stop = threading.Event()          # stand-in for what start() would have set up
    done = threading.Event()
    st._thread = threading.Thread(target=done.wait, daemon=True)
    st._thread.start()
    return st, done


def test_hubstate_stop_after_a_run_already_finished_reports_no_active_run():
    """The Event outlives the run: without a liveness check, a stray /api/stop against a hub
    whose run ended (deck exhausted, rate limit, an error halt) claimed a stop had worked."""
    st, done = _live_hubstate()
    done.set()
    st._thread.join(timeout=_LIVENESS_TIMEOUT_S)

    ok, msg = st.stop()
    assert ok is False and msg == "no run is active"
    assert st._stop.is_set() is False     # and it did not quietly set a dead run's event


def test_hubstate_stop_sets_the_event_and_prints_an_operator_line(capsys):
    st, done = _live_hubstate()
    ok, msg = st.stop()
    done.set()
    assert ok is True and msg == "stopping"
    assert st._stop.is_set() is True
    out = capsys.readouterr().out
    # Every OTHER shutdown trigger (tab-close, SIGINT, startup-abort) already prints a line;
    # Stop via the hub button was the one silent path, on both the terminal and the hub's
    # own live-log panel (which reads the same tee'd stdout).
    assert "Hub: stop requested" in out
    assert "will NOT be recorded" in out


def test_hubstate_snapshot_forwards_the_stopping_field_from_run_status():
    # HubState.snapshot() must not need a redundant field of its own -- RunStatus.snapshot()
    # (embedded verbatim as snap["status"]) already carries `stopping`, so nothing extra is
    # needed here as long as nothing strips it back out.
    from operation_love.status import RunStatus

    st = HubState("config.yaml")
    status = RunStatus("r1", ["hinge"], min_labels=1, mode="observe")
    status.set_global(stopping=True, phase="stopping")
    with st._lock:
        st._status = status
    snap = st.snapshot()
    assert snap["status"]["stopping"] is True
    assert snap["status"]["phase"] == "stopping"


def test_snapshot_reports_stopping_for_a_stop_pressed_during_startup():
    """RunStatus.stopping is written by supervisor.run()'s shutdown finally, which a Stop
    pressed DURING STARTUP never reaches (_abort_startup handles that path). The page would
    then show running=true / stopping=false with the Stop button still live for the whole
    model warmup -- the exact "did my click register?" gap this state exists to close."""
    from operation_love.status import RunStatus

    st, done = _live_hubstate()
    status = RunStatus("r1", ["hinge"], min_labels=1, mode="observe")
    status.set_global(phase="loading ML models")     # still starting up; stopping stays False
    with st._lock:
        st._status = status
    assert st.snapshot()["status"]["stopping"] is False

    st.stop()
    assert st.snapshot()["status"]["stopping"] is True
    done.set()


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
    # `stuck` is never set before this assertion runs, so there is no alternate bounded
    # path to race against here (unlike the `elapsed < ...` check a few tests down, which
    # is deliberately compared against a specific in-test duration and must stay tight):
    # if wait_for_run() ignored its own `timeout=0.1` and blocked for real, this would just
    # hang forever. So this is pure hang detection and safe to widen like any other liveness
    # bound.
    assert time.monotonic() - start < _LIVENESS_TIMEOUT_S   # returned promptly, did not hang
    assert st._thread.is_alive() is True
    stuck.set()
    st._thread.join(timeout=_LIVENESS_TIMEOUT_S)


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
        state._thread.join(timeout=_LIVENESS_TIMEOUT_S)

    assert any("did not finish saving" in line for line in printed)


def test_hubstate_browser_client_lifecycle(monkeypatch):
    import operation_love.hub as hub
    now = 1000.0
    monkeypatch.setattr(hub.time, "monotonic", lambda: now)

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
    monkeypatch.setattr(hub.time, "monotonic", lambda: now)
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


def test_hubstate_browser_liveness_ignores_wall_clock_jumps(monkeypatch):
    from operation_love.hub import state as hub_state

    monotonic_now = 1000.0
    wall_now = 1000.0
    monkeypatch.setattr(hub_state.time, "monotonic", lambda: monotonic_now)
    monkeypatch.setattr(hub_state.time, "time", lambda: wall_now)

    st = HubState("config.yaml")
    assert st.browser_client_opened("a") is True
    wall_now += 10_000_000

    assert st.expire_stale_browser_clients() is False
    assert st.has_browser_clients() is True


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

    # This test isolates argument forwarding from the shipped config's deliberate Hinge
    # Auto release gate, which has its own start-rejection coverage below.
    monkeypatch.setattr(hub.supervisor, "load_effective_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(hub.supervisor, "run", fake_run)
    st = HubState("config.yaml")
    # hinge, not bumble: bumble is an Android target that starts out uncalibrated
    # (platforms.py) and HubState.start() now rejects an unrunnable selection up front.
    ok, _ = st.start(mode="auto", apps=["hinge"], max_per_run=8)
    assert ok is True
    assert done.wait(timeout=_LIVENESS_TIMEOUT_S)
    st._thread.join(timeout=_LIVENESS_TIMEOUT_S)
    assert seen["mode"] == "auto" and seen["max_per_run"] == 8


@pytest.mark.parametrize("mode", ["observe", "auto"])
def test_hubstate_timed_stop_stops_each_run_mode_and_reports_countdown(monkeypatch, mode):
    """The hub's timer is mode-agnostic: it stops supervisor's shared Event, not a worker.

    This uses a real one-second local timer rather than calling stop() directly so the test
    covers scheduling, expiry, and the exact Event supervisor receives.
    """
    import operation_love.hub as hub

    entered = threading.Event()
    finished = threading.Event()

    def fake_run(config_path, **kw):
        entered.set()
        assert kw["stop_event"].wait(timeout=_LIVENESS_TIMEOUT_S)
        finished.set()

    # Exercise Timer/Event behavior in both modes without coupling it to release evidence.
    monkeypatch.setattr(hub.supervisor, "load_effective_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(hub.supervisor, "run", fake_run)
    st = HubState("config.yaml")
    ok, _ = st.start(mode=mode, apps=["hinge"], stop_after_seconds=1)
    assert ok is True
    assert entered.wait(timeout=_LIVENESS_TIMEOUT_S)

    timed_stop = st.snapshot()["timed_stop"]
    assert timed_stop is not None
    assert timed_stop["duration_seconds"] == 1
    assert 0 <= timed_stop["remaining_seconds"] <= 1

    assert finished.wait(timeout=_LIVENESS_TIMEOUT_S)
    st._thread.join(timeout=_LIVENESS_TIMEOUT_S)
    assert st.snapshot()["timed_stop"] is None


def test_hubstate_manual_stop_cancels_the_timed_stop(monkeypatch):
    import operation_love.hub as hub

    entered = threading.Event()
    released = threading.Event()

    def fake_run(config_path, **kw):
        entered.set()
        assert kw["stop_event"].wait(timeout=_LIVENESS_TIMEOUT_S)
        released.set()

    monkeypatch.setattr(hub.supervisor, "run", fake_run)
    st = HubState("config.yaml")
    ok, _ = st.start(mode="observe", apps=["hinge"], stop_after_seconds=10)
    assert ok is True and entered.wait(timeout=_LIVENESS_TIMEOUT_S)
    assert st.stop()[0] is True
    assert st.snapshot()["timed_stop"] is None
    assert released.wait(timeout=_LIVENESS_TIMEOUT_S)
    st._thread.join(timeout=_LIVENESS_TIMEOUT_S)


def test_hubstate_stale_timed_stop_cannot_stop_a_newer_generation():
    st = HubState("config.yaml")
    old_stop = threading.Event()
    new_stop = threading.Event()
    keep_alive = threading.Event()
    st._thread = threading.Thread(target=keep_alive.wait, daemon=True)
    st._thread.start()
    with st._lock:
        st._run_generation = 2
        st._stop = new_stop

    # Simulate a callback from a cancelled timer belonging to run generation 1.  It must
    # not set the Event owned by generation 2, even though a run is currently live.
    st._timed_stop_elapsed(1, old_stop)
    assert old_stop.is_set() is False
    assert new_stop.is_set() is False

    keep_alive.set()
    st._thread.join(timeout=_LIVENESS_TIMEOUT_S)


@pytest.mark.parametrize("value", [True, False, -1, 0.0, -0.0, 1.5, "30", "1.0", []])
def test_hub_timed_stop_validation_rejects_non_integer_or_negative_values(value):
    from operation_love.hub.state import validate_stop_after_seconds

    ok, normalized, error = validate_stop_after_seconds(value)
    assert ok is False
    assert normalized is None
    assert error


@pytest.mark.parametrize("value, expected", [(None, None), (0, None), (1, 1), (90, 90)])
def test_hub_timed_stop_validation_accepts_unlimited_or_positive_seconds(value, expected):
    from operation_love.hub.state import validate_stop_after_seconds

    assert validate_stop_after_seconds(value) == (True, expected, None)


@pytest.mark.parametrize("value", [True, False, -1, 1.5, "8", {}, [], 1_000_001])
def test_hub_max_per_run_validation_rejects_malformed_or_excessive_values(value):
    from operation_love.hub.state import validate_max_per_run

    ok, normalized, error = validate_max_per_run(value)
    assert ok is False
    assert normalized is None
    assert error


@pytest.mark.parametrize("value", [None, 0, 1, 8, 1_000_000])
def test_hub_max_per_run_validation_preserves_null_unlimited_and_positive_caps(value):
    from operation_love.hub.state import validate_max_per_run

    assert validate_max_per_run(value) == (True, value, None)


def test_api_start_rejects_invalid_timed_stop_before_launching_run(monkeypatch):
    import operation_love.hub as hub

    launched = []
    monkeypatch.setattr(hub.supervisor, "run", lambda *a, **k: launched.append((a, k)))
    _Handler.state = HubState("config.yaml")
    httpd = _bind("127.0.0.1", 8799)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        req = urllib.request.Request(
            f"{base}/api/start",
            data=json.dumps({"mode": "observe", "apps": ["hinge"],
                             "stop_after_seconds": "not-a-number",
                             "_csrf_token": _csrf_token(base)}).encode(),
            method="POST", headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 400
        assert json.loads(exc_info.value.read())["ok"] is False
        assert launched == []
    finally:
        httpd.shutdown()
        httpd.server_close()
        _join_hub_watch_threads()


def test_api_start_rejects_unknown_control_field_before_launching_run(monkeypatch):
    import operation_love.hub as hub

    launched = []
    monkeypatch.setattr(hub.supervisor, "run", lambda *a, **k: launched.append((a, k)))
    _Handler.state = HubState("config.yaml")
    httpd = _bind("127.0.0.1", 8799)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        request = urllib.request.Request(
            f"{base}/api/start",
            data=json.dumps({
                "mode": "observe", "apps": ["hinge"], "max_per_ru": 8,
                "_csrf_token": _csrf_token(base),
            }).encode(),
            method="POST", headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request, timeout=5)
        assert exc_info.value.code == 400
        assert "max_per_ru" in json.loads(exc_info.value.read())["msg"]
        assert launched == []
        assert _Handler.state._thread is None
        assert _Handler.state._timed_stop_timer is None
    finally:
        httpd.shutdown()
        httpd.server_close()
        _join_hub_watch_threads()


@pytest.mark.parametrize("field,value", [
    ("max_per_run", True),
    ("max_per_run", "8"),
    ("max_per_run", -1),
    ("max_per_run", {}),
    ("apps", "hinge"),
    ("apps", ["hinge", "hinge"]),
    ("apps", [1]),
])
def test_api_start_rejects_malformed_controls_before_launching_run(
        monkeypatch, field, value):
    import operation_love.hub as hub

    launched = []
    monkeypatch.setattr(hub.supervisor, "run", lambda *a, **k: launched.append((a, k)))
    _Handler.state = HubState("config.yaml")
    httpd = _bind("127.0.0.1", 8799)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        payload = {"mode": "observe", "apps": ["hinge"], "max_per_run": None,
                   "_csrf_token": _csrf_token(base)}
        payload[field] = value
        request = urllib.request.Request(
            f"{base}/api/start", data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request, timeout=5)
        assert exc_info.value.code == 400
        assert json.loads(exc_info.value.read())["ok"] is False
        assert launched == []
        assert _Handler.state._thread is None
        assert _Handler.state._timed_stop_timer is None
    finally:
        httpd.shutdown()
        httpd.server_close()
        _join_hub_watch_threads()


@pytest.mark.parametrize("payload", [b"not-json", b"[]", b"null"])
def test_api_start_rejects_non_object_json_before_launching_run(monkeypatch, payload):
    import operation_love.hub as hub

    launched = []
    monkeypatch.setattr(hub.supervisor, "run", lambda *a, **k: launched.append((a, k)))
    _Handler.state = HubState("config.yaml")
    httpd = _bind("127.0.0.1", 8799)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_address[1]}/api/start", data=payload,
            method="POST", headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 400
        assert launched == []
    finally:
        httpd.shutdown()
        httpd.server_close()
        _join_hub_watch_threads()


@pytest.mark.parametrize("path", [
    "/api/observe/action", "/api/hub/open", "/api/hub/ping", "/api/hub/closed",
])
@pytest.mark.parametrize("payload", [b"not-json", b"[]", b"null"])
def test_api_post_endpoints_reject_non_object_json_without_crashing(path, payload):
    """POST handlers all read JSON fields, so their shared HTTP boundary must reject
    malformed/non-object JSON before any handler reaches ``body.get(...)``."""
    _Handler.state = HubState("config.yaml")
    httpd = _bind("127.0.0.1", 8799)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_address[1]}{path}", data=payload,
            method="POST", headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 400
        response = json.loads(exc_info.value.read())
        assert response["ok"] is False
        assert "JSON object" in response["msg"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        _join_hub_watch_threads()


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


def test_hub_page_adds_csrf_token_to_fetch_and_close_beacon_json():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    post_fn = _extract_js_function(_PAGE, "postJSON")
    lifecycle_fn = _extract_js_function(_PAGE, "hubLifecycle")
    script = (
        "const hubCsrfToken = 'server-secret';\n"
        "const hubClientId = 'client-1';\n"
        "let calls = [];\n"
        "class Blob { constructor(parts, options){ this.body=parts.join(''); "
        "this.type=options.type; } }\n"
        "const navigator = {sendBeacon(path, blob){ calls.push({kind:'beacon', path, "
        "body:JSON.parse(blob.body), type:blob.type}); return true; }};\n"
        "async function fetch(path, options){ calls.push({kind:'fetch', path, "
        "body:JSON.parse(options.body), type:options.headers['Content-Type']}); "
        "return {json:async()=>({ok:true})}; }\n"
        + post_fn + "\n" + lifecycle_fn + "\n"
        "postJSON('/api/stop', {}).then(() => {\n"
        "  hubLifecycle('/api/hub/closed');\n"
        "  console.log(JSON.stringify(calls));\n"
        "});\n"
    )
    assert _run_node(script) == [
        {
            "kind": "fetch",
            "path": "/api/stop",
            "body": {"_csrf_token": "server-secret"},
            "type": "application/json",
        },
        {
            "kind": "beacon",
            "path": "/api/hub/closed",
            "body": {"id": "client-1", "_csrf_token": "server-secret"},
            "type": "application/json",
        },
    ]


def test_committed_mac_launcher_matches_template():
    expected = _MAC_UPDATE_RUN.replace("__EXTRAS__", "ml,bq,hinge")
    assert Path("Operation Love.command").read_text() == expected


def test_launcher_default_extras_cover_runtime_dependencies_but_not_reference_web_tools():
    # Regression for the bug fixed in commit 640406b1: make_launchers()'s default `extras`
    # silently dropped `hinge` (opencv never installed -> Hinge's vision degraded to
    # fixed-coordinate taps with no warning). The committed launcher was hand-patched to
    # include it, but the GENERATOR's default argument was never fixed, so regenerating the
    # launcher would silently reintroduce the exact same gap. Guard the default itself
    # (not a generated file, which is platform-dependent) against every extra pyproject.toml
    # declares that the SHIPPED APP needs. `dev` is for this repo's tests, while `web` is a
    # reference-only Playwright base with no live target; neither belongs in the launcher.
    import inspect
    import tomllib

    from operation_love.hub.launchers import make_launchers

    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    declared = set(pyproject["project"]["optional-dependencies"])
    runtime = declared - {"dev", "web"}
    assert runtime, "sanity: pyproject.toml declares no runtime optional dependencies"

    default_extras = inspect.signature(make_launchers).parameters["extras"].default
    have = set(default_extras.split(","))

    assert have == runtime
    assert have.isdisjoint({"dev", "web"})


def _launcher_project(tmp_path: Path) -> Path:
    (tmp_path / "config.yaml").write_text("mode: observe\n")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "launcher-test"\nversion = "1"\n'
        '[project.optional-dependencies]\nhinge = []\n')
    return tmp_path / "config.yaml"


@pytest.mark.skipif(os.name == "nt", reason="POSIX launcher modes")
@pytest.mark.parametrize(("platform", "name", "mode"), [
    ("darwin", "Operation Love.command", 0o755),
    ("linux", "operation-love.sh", 0o755),
    ("win32", "Operation Love.bat", 0o644),
])
def test_launcher_is_atomically_written_with_exact_mode(
        tmp_path, monkeypatch, platform, name, mode):
    from operation_love.hub.launchers import make_launchers

    config = _launcher_project(tmp_path)
    destination = tmp_path / name
    destination.write_text("old partial launcher")
    destination.chmod(0o666)
    monkeypatch.setattr(sys, "platform", platform)

    make_launchers(str(config), extras="hinge")

    assert "old partial launcher" not in destination.read_text()
    assert ".[hinge]" in destination.read_text()
    assert stat.S_IMODE(destination.stat().st_mode) == mode
    assert not list(tmp_path.glob(f".{name}.*"))


@pytest.mark.skipif(os.name == "nt", reason="requires symlink support")
def test_launcher_refuses_symlink_without_mutating_its_target(tmp_path, monkeypatch):
    from operation_love.hub.launchers import make_launchers

    config = _launcher_project(tmp_path)
    target = tmp_path / "unrelated.txt"
    target.write_text("do not overwrite")
    destination = tmp_path / "Operation Love.command"
    destination.symlink_to(target)
    monkeypatch.setattr(sys, "platform", "darwin")

    with pytest.raises(OSError, match="not a regular file"):
        make_launchers(str(config), extras="hinge")

    assert destination.is_symlink()
    assert target.read_text() == "do not overwrite"


def test_launcher_rejects_undeclared_or_shell_syntax_extras(tmp_path, monkeypatch):
    from operation_love.hub.launchers import make_launchers

    config = _launcher_project(tmp_path)
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ValueError, match="comma-separated"):
        make_launchers(str(config), extras="hinge,$(touch owned)")
    with pytest.raises(ValueError, match="unknown optional-dependency"):
        make_launchers(str(config), extras="hinge,bq")
    assert not (tmp_path / "Operation Love.command").exists()


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


def test_hub_model_quality_card_shows_training_record_balance():
    assert "training record" in _PAGE
    assert "% like /" in _PAGE and "% dislike" in _PAGE
    assert "const dislikePct = 100 - likePct" in _PAGE
    assert "const training = e.training || e" in _PAGE


def test_cached_eval_uses_live_training_mix_without_waiting_for_next_cv_refresh():
    class LiveObserve:
        running = True
        mode = "observe"
        labels = 3

    class LiveStore:
        def load_labels(self):
            return [(True, []), (False, []), (True, [])]

    st = HubState("config.yaml")
    with st._lock:
        st._eval = {"status": "insufficient_data", "labels": 2, "likes": 1, "passes": 1}
        st._eval_at = time.time()
        st._eval_labels = 0
        st._status = LiveObserve()
        st._live_store = LiveStore()

    # A running thread is how HubState distinguishes a live worker from a retained status.
    st._thread = threading.Thread(target=lambda: time.sleep(0.05))
    st._thread.start()
    result = st.eval_snapshot(every=15)
    st._thread.join()

    assert result["labels"] == 2              # CV itself remains cached until its cadence
    assert result["training"] == {"labels": 3, "likes": 2, "passes": 1}


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
            assert release.wait(timeout=_LIVENESS_TIMEOUT_S)
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
    assert started.wait(timeout=_LIVENESS_TIMEOUT_S)

    second = st.eval_snapshot(every=5)
    assert second["marker"] == "old"
    assert len(store_calls) == 1

    release.set()
    # Same liveness-not-performance bound as `_LIVENESS_TIMEOUT_S`'s own comment: this poll is
    # waiting for the background refresh thread to publish, and the only failure it is built to
    # catch is "it never does". The literal 5s it used to carry was the last gate of this shape
    # left after the 2026-08-22 sweep, and two of its 5s siblings had already been seen failing
    # live under the one-worker-per-core default.
    deadline = time.time() + _LIVENESS_TIMEOUT_S
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


# --- audit fix: an honest "stopping" tail, driven by status.stopping ------------------------

def test_stop_button_state_reflects_stopping_and_disables_a_second_press():
    # Once status.stopping is true the button must relabel AND disable, in the same call --
    # a second press is impossible-by-design, not merely a no-op HubState.stop() tolerates.
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fn = _extract_js_function(_PAGE, "stopButtonState")
    cases = [(True, False), (True, True), (False, False), (False, True)]
    script = (
        fn + "\n"
        "const cases = " + json.dumps(cases) + ";\n"
        "console.log(JSON.stringify(cases.map(c => stopButtonState(c[0], c[1]))));\n"
    )
    assert _run_node(script) == [
        {"label": "■ Stop", "disabled": False},        # running, not stopping -> normal
        {"label": "■ Stopping…", "disabled": True},    # running AND stopping -> the shutdown tail
        {"label": "■ Stop", "disabled": True},          # not running -> disabled regardless
        {"label": "■ Stopping…", "disabled": True},    # not running but stopping (brief overlap)
    ]


def test_should_clear_hint_only_fires_on_the_running_to_stopped_transition():
    # A blanket "clear whenever !running" would also wipe a legitimate persistent message
    # like 'not started' (set while running is ALREADY false) on the very next poll.
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fn = _extract_js_function(_PAGE, "shouldClearHint")
    # (running, wasRunning)
    cases = [(True, True), (False, True), (False, False), (True, False)]
    script = (
        fn + "\n"
        "const cases = " + json.dumps(cases) + ";\n"
        "console.log(JSON.stringify(cases.map(c => shouldClearHint(c[0], c[1]))));\n"
    )
    assert _run_node(script) == [
        False,  # still running -> nothing to clear yet
        True,   # WAS running, just stopped -> the real transition -> clear
        False,  # already not running (e.g. a failed Start) -> leave it alone
        False,  # just started running -> nothing to clear
    ]


def _render_global_script(calls: list) -> str:
    """Run the real renderGlobal against a minimal DOM, once per entry in `calls`, in the
    SAME node process/script and in order -- renderGlobal is stateful across polls (the
    module-level `_wasRunning` shouldClearHint reads), so a fresh eval per call would miss
    exactly the transition this is testing."""
    fns = (_extract_js_function(_PAGE, "stopButtonState") + "\n"
           + _extract_js_function(_PAGE, "shouldClearHint") + "\n"
           + _extract_js_function(_PAGE, "formatTimedStopRemaining") + "\n"
           + _extract_js_function(_PAGE, "renderGlobal"))
    return (
        "let _wasRunning = false;\n"
        "let els = {runpill:{textContent:'',className:''}, start:{disabled:false}, "
        "stop:{disabled:false,textContent:''}, hint:{textContent:''}, budget:{textContent:''}, "
        "err:{textContent:''}, stopafter:{disabled:false}, timerhint:{textContent:''}};\n"
        "function $(sel){ return els[sel.slice(1)]; }\n"
        + fns + "\n"
        "const calls = " + json.dumps(calls) + ";\n"
        "const results = [];\n"
        "for (const c of calls) {\n"
        "  if (c.presetHint != null) els.hint.textContent = c.presetHint;\n"
        "  renderGlobal(c.snap);\n"
        "  results.push({hint: els.hint.textContent, pill: els.runpill.textContent, "
        "stopDisabled: els.stop.disabled, stopLabel: els.stop.textContent, "
        "timerDisabled: els.stopafter.disabled, timerHint: els.timerhint.textContent});\n"
        "}\n"
        "console.log(JSON.stringify(results));\n"
    )


def test_render_global_clears_a_stuck_stopping_hint_only_on_the_running_transition():
    # The exact audit bug: a Stop pressed during startup jumps straight from a starting-
    # phase to phase='stopped' (supervisor.py's _abort_startup), with no 'saving data'
    # phase ever published in between -- so the old code, which only cleared the literal
    # string 'saving data…', left 'stopping…' stuck next to a 'stopped' pill forever.
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    calls = [
        {"presetHint": "stopping…", "snap": {"running": True, "status": {"phase": "starting"}}},
        {"snap": {"running": False, "status": None}},   # startup-abort: straight to stopped
    ]
    results = _run_node(_render_global_script(calls))
    assert results[0]["hint"] == "stopping…"    # still running -> not cleared yet
    assert results[1]["hint"] == ""              # the transition -> cleared (was stuck before)

    # A FAILED Start (running already false, no transition ever happens) must not have its
    # explanation wiped out from under it.
    calls_b = [{"presetHint": "not started", "snap": {"running": False, "status": None}}]
    results_b = _run_node(_render_global_script(calls_b))
    assert results_b[0]["hint"] == "not started"


def test_render_global_shows_stopping_on_the_pill_and_disables_the_stop_button():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    calls = [{"snap": {"running": True, "status": {
        "phase": "stopping", "stopping": True, "budget_spent": 0}}}]
    results = _run_node(_render_global_script(calls))
    assert results[0]["pill"] == "stopping"
    assert results[0]["stopDisabled"] is True
    assert results[0]["stopLabel"] == "■ Stopping…"


def test_render_global_shows_server_timer_countdown_and_locks_the_choice_while_running():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    calls = [
        {"snap": {"running": True, "status": {"phase": "live", "budget_spent": 0},
                  "timed_stop": {"duration_seconds": 1800, "remaining_seconds": 65}}},
        {"snap": {"running": False, "status": None, "timed_stop": None}},
    ]
    results = _run_node(_render_global_script(calls))
    assert results[0]["timerDisabled"] is True
    assert results[0]["timerHint"] == "stops in 1m 05s"
    assert results[1]["timerDisabled"] is False
    assert results[1]["timerHint"] == ""


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
    # chosenApps() sends [] when every app checkbox is unchecked. The shared supervisor gate
    # rejects it too; the Hub should retain its clearer selection-specific reason.
    st = HubState("config.yaml")
    ok, msg = st.start(apps=[])
    assert ok is False
    assert "no apps selected" in msg
    assert st.is_running() is False


@pytest.mark.parametrize(("kwargs", "needle"), [
    ({"max_per_run": True}, "max_per_run"),
    ({"max_per_run": "8"}, "max_per_run"),
    ({"max_per_run": -1}, "max_per_run"),
    ({"apps": "hinge"}, "apps must"),
    ({"apps": ["hinge", "hinge"]}, "duplicate"),
    ({"apps": [1]}, "app ids"),
])
def test_hubstate_rejects_malformed_controls_before_background_allocation(
        monkeypatch, kwargs, needle):
    import operation_love.hub as hub

    run_calls = []
    monkeypatch.setattr(hub.supervisor, "run", lambda *a, **k: run_calls.append((a, k)))
    st = HubState("config.yaml")
    ok, msg = st.start(stop_after_seconds=60, **kwargs)

    assert ok is False
    assert needle in msg
    assert run_calls == []
    assert st._thread is None
    assert st._stop is None
    assert st._timed_stop_timer is None


def test_hubstate_timer_start_failure_returns_clean_refusal_without_worker(
        monkeypatch):
    from operation_love.hub import state as hub_state

    run_calls = []

    class BrokenTimer:
        instances = []

        def __init__(self, *_args, **_kwargs):
            self.daemon = False
            self.cancelled = False
            self.instances.append(self)

        def start(self):
            raise RuntimeError("timer thread quota exhausted")

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(hub_state.threading, "Timer", BrokenTimer)
    monkeypatch.setattr(
        hub_state.supervisor, "run", lambda *args, **kwargs: run_calls.append((args, kwargs)))
    st = HubState("config.yaml")
    previous_thread = threading.Thread()
    previous_stop = threading.Event()
    previous_status = object()
    previous_store = object()
    previous_service = object()
    st._thread = previous_thread
    st._stop = previous_stop
    st._status = previous_status
    st._error = "previous completed-run error"
    st._live_store = previous_store
    st._opener_service = previous_service
    st._completed_openers = [{"opener": "previous evidence"}]

    ok, msg = st.start(stop_after_seconds=60)

    assert ok is False
    assert "could not start timed-stop timer" in msg
    assert run_calls == []
    assert st._thread is previous_thread
    assert st._stop is previous_stop
    assert st._status is previous_status
    assert st._error == "previous completed-run error"
    assert st._live_store is previous_store
    assert st._opener_service is previous_service
    assert st._timed_stop_timer is None
    assert st._timed_stop_deadline is None
    assert BrokenTimer.instances[0].cancelled is True
    assert st._completed_openers == [{"opener": "previous evidence"}]


def test_hubstate_worker_start_failure_cancels_started_timer_and_resets_state(
        monkeypatch):
    from operation_love.hub import state as hub_state

    run_calls = []

    class RecordingTimer:
        instances = []

        def __init__(self, *_args, **_kwargs):
            self.daemon = False
            self.started = False
            self.cancelled = False
            self.instances.append(self)

        def start(self):
            self.started = True

        def cancel(self):
            self.cancelled = True

    class BrokenRunThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("run thread quota exhausted")

    monkeypatch.setattr(hub_state.threading, "Timer", RecordingTimer)
    monkeypatch.setattr(hub_state.threading, "Thread", BrokenRunThread)
    monkeypatch.setattr(
        hub_state.supervisor, "run", lambda *args, **kwargs: run_calls.append((args, kwargs)))
    st = HubState("config.yaml")

    ok, msg = st.start(stop_after_seconds=60)

    assert ok is False
    assert "could not start run thread" in msg
    assert run_calls == []
    assert st._thread is None
    assert st._stop is None
    assert st._status is None
    assert st._timed_stop_timer is None
    assert st._timed_stop_deadline is None
    timer = RecordingTimer.instances[0]
    assert timer.started is True
    assert timer.cancelled is True


@pytest.mark.parametrize("mode", ["observe", "auto"])
def test_hubstate_start_rejects_uncalibrated_bumble_in_both_modes(mode):
    st = HubState("config.yaml")
    ok, msg = st.start(mode=mode, apps=["bumble"])
    assert ok is False
    assert "not calibrated" in msg
    assert st.is_running() is False


def test_hub_config_combines_registry_and_config_level_mode_readiness():
    defaults = HubState("config.yaml").config_defaults()
    app_platforms = [platform for kind in defaults["kinds"]
                     if kind["kind"] == "android" for platform in kind["platforms"]]
    modes = {platform["app"]: platform["modes"] for platform in app_platforms}
    reasons = {platform["app"]: platform["mode_reasons"] for platform in app_platforms}
    pending = {platform["app"]: platform for platform in defaults["pending_platforms"]}

    # Hinge implements both modes, but the shipped config intentionally has no current
    # production-Observe release artifact, so the operational picker must not offer Auto.
    assert modes == {"hinge": {"observe": True, "auto": False}}
    assert reasons["hinge"]["observe"] is None
    assert reasons["hinge"]["auto"].startswith("Hinge Auto is blocked")
    assert set(pending) == {"bumble"}
    assert "not calibrated" in pending["bumble"]["reason"]


def test_hubstate_start_rejects_two_android_platforms_together(monkeypatch):
    import operation_love.hub as hub

    monkeypatch.setattr(hub.supervisor, "run", lambda config_path, **kw: None)
    st = HubState("config.yaml")
    ok, msg = st.start(apps=["hinge", "bumble"])
    assert ok is False
    assert st.is_running() is False


def test_hubstate_rejects_config_blocked_hinge_auto_before_background_work(monkeypatch):
    """Hub and direct supervisor starts return the same release-gate message, and the Hub
    rejects before allocating any run thread or timed-stop timer."""
    import operation_love.hub as hub

    with pytest.raises(ValueError) as direct:
        hub.supervisor.run("config.yaml", mode="auto", enabled_apps=["hinge"])

    run_calls = []
    monkeypatch.setattr(
        hub.supervisor, "run", lambda *args, **kwargs: run_calls.append((args, kwargs)))
    st = HubState("config.yaml")
    ok, msg = st.start(
        mode="auto", apps=["hinge"], max_per_run=8, stop_after_seconds=60)

    assert ok is False
    assert msg == str(direct.value)
    assert run_calls == []
    assert st._thread is None
    assert st._stop is None
    assert st._timed_stop_timer is None
    assert st._timed_stop_deadline is None


def test_hubstate_apps_none_validates_the_effective_file_config(tmp_path):
    """No app override means config.yaml still receives the synchronous full validation."""
    cfg_path = tmp_path / "config.yaml"
    cfg_text = re.sub(
        r"(?m)^mode: observe\b", "mode: auto", Path("config.yaml").read_text(), count=1)
    cfg_path.write_text(cfg_text)

    st = HubState(str(cfg_path))
    ok, msg = st.start(apps=None, stop_after_seconds=60)

    assert ok is False
    assert msg.startswith("Hinge Auto is blocked")
    assert st._thread is None
    assert st._timed_stop_timer is None


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
    assert done.wait(timeout=_LIVENESS_TIMEOUT_S)
    st._thread.join(timeout=_LIVENESS_TIMEOUT_S)
    assert seen["enabled_apps"] is None


def test_hubstate_retains_final_status_after_run_thread_exits(monkeypatch):
    import operation_love.hub as hub
    from operation_love.status import RunStatus

    def fake_run(config_path, **kwargs):
        status = RunStatus("finished", ["hinge"], min_labels=40, mode="auto")
        status.set_app("hinge", mode="auto", state="error", error="unrecognized screen")
        status.set_global(running=False, phase="stopped")
        kwargs["on_status"](status)

    # This is final-status retention, not release-readiness coverage.
    monkeypatch.setattr(hub.supervisor, "load_effective_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(hub.supervisor, "run", fake_run)
    st = HubState("config.yaml")
    ok, _ = st.start(mode="auto", apps=["hinge"])
    assert ok is True
    st._thread.join(timeout=_LIVENESS_TIMEOUT_S)

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
    st._thread.join(timeout=_LIVENESS_TIMEOUT_S)
    assert st._error is not None
    assert st._error.startswith("SystemExit:")
    assert "missing" in st._error


def test_hubstate_recent_openers_returns_empty_list_with_no_active_run():
    """No run has ever started -- there is no OpenerService to read from. The bug report
    calls this unconditionally, so it must degrade to a plain [] rather than raising
    (AttributeError on self._opener_service being None, or similar)."""
    st = HubState("config.yaml")
    assert st.recent_openers() == []


def test_hubstate_recent_openers_forwards_to_the_captured_opener_service():
    """Once a run has captured an OpenerService (via on_opener_service -> _capture_opener_service),
    recent_openers() must hand back exactly what that service's own recent_openers_snapshot()
    returns -- this is the real data route the bug report's 'Recent openers' section relies on."""
    class FakeOpenerService:
        def recent_openers_snapshot(self):
            return [{"ts": "t0", "app": "hinge", "model": "gemini-2.5-flash",
                     "advisory": False, "index": 0, "referenced": "the beach photo",
                     "opener": "love the beach shot"}]

    st = HubState("config.yaml")
    with st._lock:
        st._opener_service = FakeOpenerService()
    assert st.recent_openers() == FakeOpenerService().recent_openers_snapshot()


def test_hubstate_recent_openers_swallows_a_raising_service():
    """A service mid-teardown (run just ended, supervisor is tearing down objects) could have
    its recent_openers_snapshot() raise instead of returning cleanly -- e.g. a lock object
    already released/replaced. recent_openers() must never propagate that into the bug report;
    it degrades to []."""
    class ExplodingOpenerService:
        def recent_openers_snapshot(self):
            raise RuntimeError("torn down mid-read")

    st = HubState("config.yaml")
    with st._lock:
        st._opener_service = ExplodingOpenerService()
    assert st.recent_openers() == []


def test_hubstate_freezes_opener_telemetry_when_a_run_finishes(monkeypatch):
    """The report remains diagnostic after shutdown without retaining the service itself.

    This is the exact lifecycle that used to lose the ring buffers: HubState's target finally
    cleared _opener_service after supervisor.run returned, so the terminal status could say
    ``openers=1`` while Recent openers incorrectly said none.  The fake keeps nested data too,
    proving the retained value is a detached copy rather than a reference into the service.
    """
    import operation_love.hub as hub

    class FakeOpenerService:
        def __init__(self):
            self.openers = [{"ts": "t1", "app": "hinge", "opener": "hello",
                             "nested": ["marker"]}]
            self.rejections = [{"ts": "t2", "app": "hinge", "reason_code": "bad_json"}]

        def recent_openers_snapshot(self):
            return list(self.openers)

        def recent_rejections_snapshot(self):
            return list(self.rejections)

    service = FakeOpenerService()

    def completed_run(_config_path, *, on_opener_service=None, **_kwargs):
        on_opener_service(service)

    monkeypatch.setattr(hub.supervisor, "run", completed_run)
    st = HubState("config.yaml")
    ok, _ = st.start()
    assert ok is True
    assert st.wait_for_run(timeout=5) is True

    with st._lock:
        assert st._opener_service is None
    assert st.recent_openers() == [{"ts": "t1", "app": "hinge", "opener": "hello",
                                    "nested": ["marker"]}]
    assert st.recent_opener_rejections() == [
        {"ts": "t2", "app": "hinge", "reason_code": "bad_json"}]
    service.openers[0]["nested"].append("changed after shutdown")
    assert st.recent_openers()[0]["nested"] == ["marker"]


def test_wait_for_run_honors_timeout():
    # Ctrl-C quit gives the run a bounded wait so a wedged worker can't hang quit forever.
    st = HubState("config.yaml")
    still_running = threading.Event()

    def slow():
        # This is only a failsafe against the background thread hanging forever if the test
        # never reaches `still_running.set()` below; the actual wait it simulates ends there,
        # long before this elapses even at _LIVENESS_TIMEOUT_S, so widening it only makes the
        # `elapsed < 2.0` discriminator below MORE reliable (a bigger gap to the buggy-path
        # duration it is proving wait_for_run() did NOT fall through to), never less.
        still_running.wait(timeout=_LIVENESS_TIMEOUT_S)

    st._thread = threading.Thread(target=slow, daemon=True)
    st._thread.start()
    t0 = time.time()
    st.wait_for_run(timeout=0.2)
    elapsed = time.time() - t0
    # Deliberately NOT widened to _LIVENESS_TIMEOUT_S (2026-08-22 pass): this bound is not a
    # liveness gate, it is the whole assertion -- it exists to prove wait_for_run() actually
    # honored its own `timeout=0.2` instead of falling through to the still_running thread's
    # own (now widened) internal wait above. Raising THIS number past that wait's duration
    # would let that exact bug pass silently, so it has to stay well under it regardless of
    # machine load.
    assert elapsed < 2.0             # returned promptly, not blocked for the full wait above
    assert st._thread.is_alive() is True   # timed out, didn't actually finish
    still_running.set()
    st._thread.join(timeout=_LIVENESS_TIMEOUT_S)


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


def test_observe_banner_uses_explicit_pass_or_like_language_and_replaces_it_with_hinge_opener():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    waiting = {
        "running": True,
        "status": {"apps": {"hinge": {"app": "hinge", "mode": "observe", "state": "waiting"}}},
    }
    ordinary = _run_node(_observe_status_script(waiting))
    assert ordinary["display"] == "block"
    assert "tap X to pass, or tap a heart to like" in ordinary["html"]
    assert "swipe" not in ordinary["html"].lower()

    # Hinge uses the physically labelled X/heart controls; other platforms may not.
    other_app = {
        "running": True,
        "status": {"apps": {"bumble": {"app": "bumble", "mode": "observe", "state": "waiting"}}},
    }
    generic = _run_node(_observe_status_script(other_app))
    assert "use the app's pass or like control" in generic["html"]
    assert "tap X to pass" not in generic["html"]
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
    assert "then tap Send Like / Send Priority Like in Hinge" in suggestion["html"]
    assert "I like &lt;your prompt&gt; &amp; &quot;this&quot;" in suggestion["html"]
    assert "tap X to pass" not in suggestion["html"]
    assert "<your prompt>" not in suggestion["html"]

    # A malformed/legacy status without an item must still make the conditional role clear;
    # it must never render the grammatical but misleading "if you choose to like, use, then".
    pre_tap_without_item = {
        "running": True,
        "status": {"apps": {"hinge": {
            "app": "hinge", "mode": "observe", "state": "waiting",
            "opener_suggestion": "A safe fallback",
        }}},
    }
    missing_item = _run_node(_observe_status_script(pre_tap_without_item))
    assert "Optional:" in missing_item["html"]
    assert "if you choose to like, type exactly this" in missing_item["html"]
    assert "if you choose to like, use, then" not in missing_item["html"]

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
    assert "type your own opener, then tap Send Like / Send Priority Like" in fallback["html"]
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


def test_observe_banner_opener_row_is_the_auto_mode_canary():
    """OWNER REQUIREMENT: the observe banner's suggested text must let the operator see
    EXACTLY where the opener starts and stops, because that same OpenerResult.opener string
    is what auto mode types verbatim (driver.like() -> adb.text(), see worker.py). If a model
    wraps the real opener in scaffolding ("Sure! Here's a great opener: ...") that scaffolding
    must show up in the banner too, byte for byte -- and it must be visually unmistakable from
    the hub's own chrome, or the operator has no way to tell "the model's words" from "our
    instructions" and the canary is defeated. This pins the opener onto its own delimited
    block containing nothing but the escaped opener, structurally separate from the label and
    chrome rows -- not merely present as a substring somewhere in the banner."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    scaffolded = 'Sure! Here\'s a great opener: "Nice antlers."'
    snap = {
        "running": True,
        "status": {"apps": {"hinge": {
            "app": "hinge", "mode": "observe", "state": "waiting_for_send",
            "opener_suggestion": scaffolded,
        }}},
    }
    html = _run_node(_observe_status_script(snap))["html"]

    escaped_opener = (
        scaffolded.replace("&", "&amp;").replace("<", "&lt;")
                  .replace(">", "&gt;").replace('"', "&quot;")
    )
    # The whole point of the canary: the scaffolding words survive into the banner
    # byte-for-byte, exactly once, so the operator sees the same wrapper auto mode would type
    # rather than a hub-cleaned version of it.
    assert escaped_opener in html
    assert html.count(escaped_opener) == 1

    # Structural check, not substring-only -- a future revert to box()'s inline "title + sub
    # flowed on one run-on line" rendering must fail this test even though the opener text
    # would still appear somewhere in the markup. Label -> opener block -> chrome block must
    # be three separate elements in this order, and the opener's own element must contain
    # NOTHING else: no hub instruction text bleeding into the same row the operator is meant
    # to read as "type exactly this."
    m = re.search(
        r'type exactly this</div>\s*<div[^>]*>(.*?)</div>\s*<div[^>]*>(.*?)</div>',
        html, re.S,
    )
    assert m, f"expected label -> opener block -> chrome block structure, got:\n{html}"
    opener_row, chrome_row = m.group(1), m.group(2)
    assert opener_row == escaped_opener
    assert "then tap Send Like / Send Priority Like in Hinge" in chrome_row
    assert "then tap Send Like / Send Priority Like in Hinge" not in opener_row


def test_observe_banner_opener_row_escapes_html_metacharacters_without_altering_content():
    """The opener's delimited row still goes through innerHTML (the text comes straight from
    an AI response), so it must stay injection-safe -- but escaping must be the ONLY thing
    that happens to it. The canary (see test above) only holds if what's shown, once
    unescaped by the browser, is identical to what auto mode types; if escaping ever dropped,
    reordered, or added characters beyond turning &<>" into entities, the operator would be
    proofreading a string auto mode never actually sends."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    raw = 'Nice <ears> & "antlers", right?'
    snap = {
        "running": True,
        "status": {"apps": {"hinge": {
            "app": "hinge", "mode": "observe", "state": "waiting_for_send",
            "opener_suggestion": raw,
        }}},
    }
    html = _run_node(_observe_status_script(snap))["html"]
    assert "<ears>" not in html
    assert "Nice &lt;ears&gt; &amp; &quot;antlers&quot;, right?" in html


def test_observe_banner_shows_the_selected_item_after_a_heart_is_open():
    """Once the sheet is open, direct copy identifies the selected item and keeps the opener
    in its own canary row. The conditional pre-tap wording belongs only to `waiting`.

    The canary rule still binds: the instruction is hub chrome and must stay OUT of the opener's
    own row, which is asserted structurally here rather than by looking for the text anywhere in
    the box."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {
        "running": True,
        "status": {"apps": {"hinge": {
            "app": "hinge", "mode": "observe", "state": "waiting_for_send",
            "opener_suggestion": "Based on that ridgeline I'm going to guess Norway",
            "opener_item": 3, "opener_media_ordinal": 3,
            "opener_item_description": "the ridgeline photo",
            "opener_referenced": "the mountain behind her",
        }}},
    }
    html = _run_node(_observe_status_script(snap))["html"]

    assert "selected media item 3 — the ridgeline photo" in html
    assert "the ridgeline photo" in html
    m = re.search(r'type exactly this</div>\s*<div[^>]*>(.*?)</div>\s*<div[^>]*>(.*?)</div>',
                  html, re.S)
    assert m, f"expected heading -> opener block -> chrome block structure, got:\n{html}"
    opener_row, chrome_row = m.group(1), m.group(2)
    assert opener_row == "Based on that ridgeline I'm going to guess Norway"
    assert "item 3" not in opener_row           # the instruction is chrome, never inline with it
    assert "the mountain behind her" in chrome_row


def test_observe_banner_keeps_the_pass_option_visible_while_a_suggestion_is_up():
    """THE INVERSION MOVED THE SUGGESTION INTO THE DECISION WINDOW, AND THE DECISION IS STILL
    THE HUMAN'S. Until doc 5.9 a suggestion could only be published AFTER the heart tap
    ('waiting_for_send'), so it never competed with the 🟢 GO cue that says it is the operator's
    turn and that PASSING is one of the two things they may do. It is published before the tap
    now, and the opener branch runs ahead of the 'waiting' branch, so the whole decision window
    read as "like item 3 ... then tap Send Like / Send Priority Like in Hinge": no circle (against the owner's
    GO/WAIT convention), no X, and an instruction about a sheet that is not open yet -- in the
    one mode whose entire output is the owner's own like/pass labels.

    The canary rule is unaffected and is re-checked here rather than assumed: the pass cue is
    chrome and must stay out of the opener's own row."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    app = {"app": "hinge", "mode": "observe",
           "opener_suggestion": "Based on that ridgeline I'm going to guess Norway",
           "opener_item": 3, "opener_media_ordinal": 3,
           "opener_item_description": "the ridgeline photo"}

    deciding = {"running": True,
                "status": {"apps": {"hinge": dict(app, state="waiting")}}}
    html = _run_node(_observe_status_script(deciding))["html"]
    assert "Optional:" in html   # the suggestion is not a verdict
    assert "if you choose to like, use media item 3 — the ridgeline photo" in html
    assert "🟢" in html                                 # ...and it is still their turn
    assert "tap X to pass, or tap the heart on media item 3 to like" in html
    assert "then tap Send Like / Send Priority Like in Hinge" not in html   # no sheet is open yet
    m = re.search(r'then type exactly this</div>\s*<div[^>]*>(.*?)</div>\s*<div[^>]*>(.*?)</div>',
                  html, re.S)
    assert m, f"expected heading -> opener block -> chrome block structure, got:\n{html}"
    assert m.group(1) == "Based on that ridgeline I'm going to guess Norway"
    assert "tap X to pass, or tap the heart on media item 3 to like" in m.group(2)

    # Once the sheet IS open the choice has been made, so the cue becomes the send instruction
    # and the pass wording goes away -- offering "pass X" under an open comment sheet would be
    # advice about a control the operator is no longer looking at.
    sending = {"running": True,
               "status": {"apps": {"hinge": dict(app, state="waiting_for_send")}}}
    html2 = _run_node(_observe_status_script(sending))["html"]
    assert "then tap Send Like / Send Priority Like in Hinge" in html2
    assert "tap X to pass" not in html2

    # Non-Hinge apps keep the generic control wording they already had in the plain GO cue.
    other = {"running": True, "status": {"apps": {"bumble": dict(
        app, app="bumble", state="waiting")}}}
    html3 = _run_node(_observe_status_script(other))["html"]
    assert "use the app's pass or like control" in html3
    assert "tap X to pass" not in html3


def test_observe_banner_uses_the_media_ordinal_not_the_heart_ordinal():
    """Written prompts must not make the fifth media card appear as an eighth item."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {"running": True, "status": {"apps": {"hinge": {
        "app": "hinge", "mode": "observe", "state": "waiting",
        "opener_suggestion": "Based on that ridgeline I'm going to guess Norway",
        "opener_item": 5, "opener_media_ordinal": 5,
        "opener_item_description": "the ridgeline photo",
    }}}}
    html = _run_node(_observe_status_script(snap))["html"]

    assert "use media item 5 — the ridgeline photo" in html
    assert "tap the heart on media item 5" in html
    assert "item 8" not in html
    assert "media item 8" not in html


def test_observe_banner_replaces_the_opener_with_a_warning_on_a_mismatch():
    """DOC 5.9's mismatch surface, and the property that makes it worth having: the opener is
    REPLACED, not annotated. Text left on screen beside a caveat is text that gets typed anyway,
    which is exactly the wrong-item comment the inversion would otherwise reintroduce. WAIT (🔴)
    styling per the owner's circle-only rule, because the one thing not to do here is copy
    something -- and there is deliberately nothing to copy."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {
        "running": True,
        "status": {"apps": {"hinge": {
            "app": "hinge", "mode": "observe", "state": "waiting_for_send",
            "opener_suggestion": None, "opener_item": 3,
            "opener_warning": "you opened item 5, but this was written about item 3",
        }}},
    }
    out = _run_node(_observe_status_script(snap))
    assert out["display"] == "block"
    assert "no suggestion to type" in out["html"]
    assert "you opened item 5" in out["html"]
    assert "🔴" in out["html"] and "🟢" not in out["html"]
    assert "type exactly this" not in out["html"]
    # ...and it still says what to do instead. This box replaces the OPENER BLOCK, not the
    # banner, so it is the only thing on screen: without a next action a card whose suggestion
    # failed states a problem and stops, and a session where they all fail (dead quota, a driver
    # that cannot enumerate) would never show the operator a cue at all.
    assert "type your own opener, then tap Send Like / Send Priority Like" in out["html"]

    # Before the tap the cue is the DECISION, not the send -- same wording the suggestion box
    # uses in that window, and the 🟢 that says it is their turn.
    deciding = {"running": True, "status": {"apps": {"hinge": dict(
        snap["status"]["apps"]["hinge"], state="waiting")}}}
    out_deciding = _run_node(_observe_status_script(deciding))
    assert "🟢 your call: tap X to pass, or tap a heart to like" in out_deciding["html"]
    assert "you opened item 5" in out_deciding["html"]
    assert "type exactly this" not in out_deciding["html"]

    # A warning is a per-card WARNING, not a stop: while the run is shutting down the "do not
    # swipe" box still takes precedence, because a decision about to be discarded is the more
    # urgent thing to say.
    stopping = {"running": True, "status": {"stopping": True, "apps": {"hinge": dict(
        snap["status"]["apps"]["hinge"])}}}
    out2 = _run_node(_observe_status_script(stopping))
    assert "stopping — do not swipe" in out2["html"]
    assert "you opened item 5" not in out2["html"]

    # And the warning text is escaped like everything else that reaches innerHTML.
    hostile = {"running": True, "status": {"apps": {"hinge": {
        "app": "hinge", "mode": "observe", "state": "waiting_for_send",
        "opener_warning": '<img src=x onerror="alert(1)">'}}}}
    out3 = _run_node(_observe_status_script(hostile))
    assert "<img" not in out3["html"] and "&lt;img" in out3["html"]


def test_observe_banner_keeps_the_go_cue_when_hinge_targeting_calibration_is_absent():
    """A missing calibration is a RUN-LEVEL, currently intentional condition — the decision
    window must stay a 🟢 GO box, never become a warning.

    2026-08-21, second bug report about this same banner in one day: rendering the condition
    through the WAIT-styled warning box put an amber 🔴 "setup required" notice over EVERY
    decision window of the run. In the owner's circle convention amber+🔴 means "hands off",
    so the operator sat out a 2m24s window that was waiting on THEM, then stopped the run.
    The cue must be byte-identical to the plain no-warning waiting state's GO box; the setup
    situation is context and lives in the box's fine-print sub row.
    """
    if NODE_BIN is None:
        pytest.skip("node not available on this machine")
    # The REAL driver string (hinge.py's items-unavailable reason), internal consequence
    # clause included — the trim below must be proven against what production actually sends.
    driver_warning = (
        "apps.hinge.targeting_calibration is unavailable (not configured in config.yaml), "
        "so a model-selected item could not be verified or targeted and no numbered item "
        "list would have a usable consumer")
    # The worker publishes the next step because only its process knows which still-photo
    # licence is installed; this is the real pre-licence string (targeting_policy's
    # TARGETING_SETUP_NEXT_STEP_BLOCKED), so the rendering is proven against what ships.
    next_step_blocked = (
        "per ops/RUNBOOK.md section 2, positive still-photo proof must exist before a fresh "
        "calibration can enable suggestions")
    snap = {
        "running": True,
        "status": {"apps": {"hinge": {
            "app": "hinge", "mode": "observe", "state": "waiting",
            "opener_warning": driver_warning,
            "targeting_setup_next_step": next_step_blocked,
        }}},
    }
    out = _run_node(_observe_status_script(snap))
    html = out["html"]
    assert out["display"] == "block"
    # The decision cue is the box's full-size title, identical to the plain waiting state,
    # and the setup note is the fine-print <span> that follows it.
    assert "🟢 hinge: make your choice: tap X to pass, or tap a heart to like <span" in html
    # One circle only, and the GO box style — not the amber WAIT style the warning box uses.
    assert "🔴" not in html
    assert "background:#123a23" in html and "background:#3a2f12" not in html
    # The old shape is gone: no "setup required" demand (config validation currently refuses
    # the very setup it demanded) and no warning-box title.
    assert "targeted opener setup required" not in html
    assert "no suggestion to type" not in html
    # The fine print keeps the diagnosable core of the driver's reason but drops the internal
    # consequence clause — that sentence is for logs and bug reports, not the operator.
    assert "no targeted suggestion this run" in html
    assert "not configured in config.yaml" in html
    assert "usable consumer" not in html
    assert "manual pass/like labels still work" in html.lower()
    # The next step is whatever the worker published, verbatim — never a sentence this page
    # decided for itself. Hardcoding it here is what produced the 2026-08-22 report: the page
    # kept demanding still-photo proof after the owner's acceptance had already licensed
    # numbering, so the only remaining step (capture the calibration) was never named.
    assert "ops/RUNBOOK.md" in html and next_step_blocked in html
    # GO cue first, context after — the inverse of the reported screenshot.
    assert html.index("🟢") < html.index("still-photo proof")

    # Sheet open: still a GO box (type and send), with the same fine-print context. The
    # trim also covers the driver's other phrasing (targeted_suggestion_blocker's).
    sending = {"running": True, "status": {"apps": {"hinge": dict(
        snap["status"]["apps"]["hinge"], state="waiting_for_send",
        opener_warning=("targeted suggestion is unavailable because "
                        "apps.hinge.targeting_calibration is unavailable "
                        "(not configured in config.yaml); no opener text is offered"))}}}
    sending_html = _run_node(_observe_status_script(sending))["html"]
    assert "type your own opener, then tap Send Like / Send Priority Like" in sending_html
    assert "background:#123a23" in sending_html and "🔴" not in sending_html
    assert "targeted suggestion is unavailable because" not in sending_html
    assert "no opener text is offered" not in sending_html
    assert "not configured in config.yaml" in sending_html
    assert "click pass X or heart" not in sending_html

    # Any other state falls through to that state's own cue: a stale run-level note must not
    # override "reading profile" with a GO instruction (the warning branch used to intercept
    # every state). And the fall-through must not loosen doc 5.9's nothing-copyable pin: even
    # with an opener_suggestion impossibly present beside the warning, no copyable text may
    # render — the state box wins, exactly as for the warning-only shape.
    capturing = {"running": True, "status": {"apps": {"hinge": dict(
        snap["status"]["apps"]["hinge"], state="capturing",
        opener_suggestion="Based on that ridgeline I'm going to guess Norway")}}}
    capturing_html = _run_node(_observe_status_script(capturing))["html"]
    assert "🔴 wait — reading hinge profile…" in capturing_html
    assert "make your choice" not in capturing_html
    assert "still-photo proof" not in capturing_html
    assert "Norway" not in capturing_html and "type exactly this" not in capturing_html

    # Shutdown still outranks the GO cue: a decision made now would be discarded.
    stopping = {"running": True, "status": {"stopping": True, "apps": {"hinge": dict(
        snap["status"]["apps"]["hinge"])}}}
    stopping_html = _run_node(_observe_status_script(stopping))["html"]
    assert "stopping — do not swipe" in stopping_html
    assert "make your choice" not in stopping_html

    # The run-level reason is operator data, not markup: it is escaped like every other
    # dynamic value that reaches innerHTML.
    hostile = {"running": True, "status": {"apps": {"hinge": dict(
        snap["status"]["apps"]["hinge"],
        opener_warning='targeting_calibration <img src=x onerror="alert(1)">')}}}
    hostile_html = _run_node(_observe_status_script(hostile))["html"]
    assert "<img" not in hostile_html and "&lt;img" in hostile_html

    # THE BUG OF 2026-08-22. Once a still-photo licence is installed the worker publishes the
    # OTHER next step, and the fine print must follow it there. The page must not keep naming a
    # prerequisite the operator already satisfied, and must name the one step that would restore
    # suggestions — otherwise a solvable run-level gate reads as an upstream block, which is
    # exactly why the calibration went uncaptured for a day.
    next_step_calibrate = (
        "still-photo numbering is licensed, so the one remaining step is the per-device "
        "targeting calibration campaign in ops/RUNBOOK.md section 2: capture both splits, "
        "measure them, and paste the emitted apps.hinge.targeting_calibration block into "
        "config.yaml")
    licensed = {"running": True, "status": {"apps": {"hinge": dict(
        snap["status"]["apps"]["hinge"],
        targeting_setup_next_step=next_step_calibrate)}}}
    licensed_html = _run_node(_observe_status_script(licensed))["html"]
    # No &<>"' in the real string, so escaping is the identity here and a raw match is exact.
    assert next_step_calibrate in licensed_html
    assert "still-photo proof must exist" not in licensed_html
    # Still the same single 🟢 GO cue: which next step is named never restyles the decision window.
    assert "🟢 hinge: make your choice: tap X to pass, or tap a heart to like <span" in licensed_html
    assert "🔴" not in licensed_html
    assert "background:#123a23" in licensed_html and "background:#3a2f12" not in licensed_html

    # With nothing published (an older worker, or a poll that raced the run's start) the note
    # ends after "manual pass/like labels still work" rather than asserting a prerequisite this
    # page cannot verify. Guessing is what went stale; silence cannot.
    unpublished = {"running": True, "status": {"apps": {"hinge": {
        "app": "hinge", "mode": "observe", "state": "waiting",
        "opener_warning": driver_warning}}}}
    unpublished_html = _run_node(_observe_status_script(unpublished))["html"]
    assert "no targeted suggestion this run" in unpublished_html
    assert "manual pass/like labels still work" in unpublished_html.lower()
    assert "still-photo proof" not in unpublished_html
    assert "ops/RUNBOOK.md" not in unpublished_html
    assert "🟢 hinge: make your choice: tap X to pass, or tap a heart to like <span" in unpublished_html


def test_observe_banner_offers_no_text_to_type_when_a_warning_and_an_opener_arrive_together():
    """THE CANARY RULE AND THE MISMATCH RULE MEET HERE, and the hub is the last place either can
    be enforced. worker.py publishes opener_warning or opener_suggestion and never both (see
    _ObserveSuggestion._display, which RETURNS at the mismatch), so this shape should be
    unreachable -- which is exactly why the hub's own precedence has to be pinned rather than
    assumed. A stale poll, a reordered publish, or a future producer that annotates instead of
    replacing would otherwise put a wrong-item opener back on screen next to a caveat, and text
    on screen beside a caveat is text that gets typed anyway (doc 5.9).

    "Offers nothing to type" is asserted on the OPENER STRING ITSELF, not on the absence of a
    label: the failure this guards against is the operator copying those exact words."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {
        "running": True,
        "status": {"apps": {"hinge": {
            "app": "hinge", "mode": "observe", "state": "waiting_for_send",
            "opener_suggestion": "Based on that ridgeline I'm going to guess Norway",
            "opener_referenced": "the mountain behind her",
            "opener_item": 3, "opener_item_description": "the ridgeline photo",
            "opener_warning": "you opened item 5, but this suggestion was written about item 3",
        }}},
    }
    html = _run_node(_observe_status_script(snap))["html"]

    assert "you opened item 5" in html
    assert "no suggestion to type" in html
    # Nothing copyable survives: not the opener, not the "type exactly this" instruction that
    # would tell the operator there is something to copy, and not the referenced caption that
    # only makes sense under a suggestion.
    assert "ridgeline I'm going to guess Norway" not in html
    assert "type exactly this" not in html
    assert "the mountain behind her" not in html
    assert "🔴" in html and "🟢" not in html


def test_observe_banner_keeps_the_go_cue_while_the_suggestion_is_still_being_written():
    """DOC 5.9's timing rule at the surface: READY is published immediately and the suggestion
    fills in behind it, so `opener_pending` is a NOTE beside a live GO cue rather than a WAIT
    state that holds the operator up. The old blocking "suggesting" state was correct when the
    call sat between the heart tap and the suggestion; it would be a lie now."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    pending = {
        "running": True,
        "status": {"apps": {"hinge": {"app": "hinge", "mode": "observe", "state": "waiting",
                                      "opener_pending": True}}},
    }
    out = _run_node(_observe_status_script(pending))
    assert "🟢" in out["html"]                        # still GO: they may act right now
    assert "tap X to pass, or tap a heart to like" in out["html"]
    assert "wait a moment for a suggestion" in out["html"]

    settled = {
        "running": True,
        "status": {"apps": {"hinge": {"app": "hinge", "mode": "observe", "state": "waiting",
                                      "opener_pending": False}}},
    }
    out2 = _run_node(_observe_status_script(settled))
    assert "make your choice: tap X to pass, or tap a heart to like" in out2["html"]
    assert "wait a moment for a suggestion" not in out2["html"]

    # The race doc 5.9 names: the human taps FASTER than the model answers, so the sheet is open
    # with nothing to type yet. That must read as "still writing", not as "there is none" -- the
    # operator's next move differs (wait a beat, versus write your own).
    tapped_first = {
        "running": True,
        "status": {"apps": {"hinge": {"app": "hinge", "mode": "observe",
                                      "state": "waiting_for_send", "opener_pending": True}}},
    }
    out3 = _run_node(_observe_status_script(tapped_first))
    assert "still writing a suggestion" in out3["html"]
    assert "No suggestion available" not in out3["html"]


def test_observe_banner_shows_wait_cue_while_a_suggestion_is_being_generated():
    """The legacy blocking "suggesting" state. NOTHING PUBLISHES IT since doc 5.9's inversion
    (the suggestion is generated on its own thread while the operator is already free to act --
    see the opener_pending test above), but it stays a legal AppStatus.state that an older
    snapshot can carry, so the branch that renders it stays pinned rather than silently rotting.

    OWNER UI RULE: GO/WAIT cues use only 🟢/🔴 circles, so this must render as the same WAIT (🔴)
    style as capturing/acting/starting, not a new indicator."""
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


def test_observe_banner_replaces_every_go_cue_with_a_stop_box_while_stopping():
    """Audit fix: worker.py discards any decision recorded once stop_event lands (see the
    stop_event re-checks around the observe loop's decision point), so telling the operator
    to act -- plain 'waiting', the no-suggestion 'waiting_for_send' fallback, or a live
    opener suggestion -- would be actively misleading during the shutdown tail
    (status.stopping, set for the whole join+save window -- see status.py's docstring).
    Same WAIT (🔴) style as the existing suggesting/capturing cues, never 🟢, per the
    owner's circle-only status convention."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    waiting_while_stopping = {
        "running": True,
        "status": {"stopping": True,
                   "apps": {"hinge": {"app": "hinge", "mode": "observe", "state": "waiting"}}},
    }
    out = _run_node(_observe_status_script(waiting_while_stopping))
    assert out["display"] == "block"
    assert "stopping — do not swipe" in out["html"]
    assert "this decision will not be recorded" in out["html"]
    assert "click pass X or heart" not in out["html"]
    assert "🔴" in out["html"] and "🟢" not in out["html"]

    # The live opener-suggestion GO box must be overridden too, and the suggestion text
    # itself suppressed (there is nothing useful to type into a decision that gets thrown
    # away).
    opener_while_stopping = {
        "running": True,
        "status": {"stopping": True, "apps": {"hinge": {
            "app": "hinge", "mode": "observe", "state": "waiting_for_send",
            "opener_suggestion": "love the beach shot",
        }}},
    }
    out2 = _run_node(_observe_status_script(opener_while_stopping))
    assert "stopping — do not swipe" in out2["html"]
    assert "love the beach shot" not in out2["html"]

    # Regression guard: while NOT stopping, the ordinary GO cue is untouched.
    not_stopping = {
        "running": True,
        "status": {"stopping": False,
                   "apps": {"hinge": {"app": "hinge", "mode": "observe", "state": "waiting"}}},
    }
    out3 = _run_node(_observe_status_script(not_stopping))
    assert "tap X to pass, or tap a heart to like" in out3["html"]
    assert "stopping — do not swipe" not in out3["html"]


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
    #
    # stop_kind="opener" is required here since the 2026-08-11 blocked-deck addition gave
    # stop_reason a SECOND possible cause (worker.py's blocked-deck check) with its own
    # stop_kind: hub.html now branches the "opener capacity exhausted" wording specifically on
    # stop_kind==='opener' rather than on stop_reason's mere presence (see hub.html's comment
    # right above that branch) -- a real run always sets both together (worker.py's four
    # opener-triggered stop sites), so omitting it here would test a shape no live run ever
    # actually produces.
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {
        "running": True,
        "status": {
            "apps": {
                "hinge": {"app": "hinge", "mode": "auto", "state": "stopped",
                          "stop_reason": "run budget reached", "stop_kind": "opener",
                          "swipes_run": 4},
            },
        },
    }
    result = _run_node(_autostatus_script(snap))
    assert result["display"] == "block"
    assert "run budget reached" in result["html"]
    assert "opener capacity exhausted" in result["html"]
    # The reason takes over the box's sub-line instead of the ordinary swipe count.
    assert "4 swipes this run" not in result["html"]


def test_render_auto_status_reads_a_targeting_stop_as_one_not_as_opener_capacity():
    """ops/OPENER-REDESIGN.md 5.6's hard stop, at the surface the operator reads. The bot
    reached the like, could not put it on the item the model chose, and put it nowhere. That
    used to publish stop_kind="opener" and therefore rendered under the branch above titled
    "opener capacity exhausted" -- a confidently wrong label (nothing was exhausted; the opener
    is fine) on the one stop that exists to prove we never comment on the wrong item, with the
    truth demoted to the sub-line. It now has its own stop_kind and its own title.

    Two properties beyond the wording. It must NOT render as an error box: worker.py catches
    ItemTargetingError precisely so a rule being obeyed does not look like a crash, and a red
    banner here would undo that one layer up. And the reason -- which leads with INTENDED and
    ACTUAL (Worker._targeting_stop_reason) -- has to survive into the box, because those two
    numbers are the whole diagnosis."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    reason = ("the like was NOT sent: reaching the item the opener was written about failed. "
              "Intended: item 4 (model_items). Actual: item 6 (model_items).")
    snap = {
        "running": True,
        "status": {"apps": {"hinge": {"app": "hinge", "mode": "auto", "state": "stopped",
                                      "stop_reason": reason, "stop_kind": "targeting",
                                      "swipes_run": 2}}},
    }
    result = _run_node(_autostatus_script(snap))
    assert result["display"] == "block"
    html = result["html"]

    assert "could not like the item the opener was written about" in html
    assert "opener capacity exhausted" not in html
    assert "Intended: item 4" in html and "Actual: item 6" in html
    # Not the error box: 'idle' styling (the same neutral box every other non-crash stop uses),
    # never 'err'. Asserted on the styles themselves so a future re-colour has to come through
    # this test and worker.py's reasoning for catching the exception at all.
    assert "#22222b" in html                      # css.idle background
    assert "#3a1414" not in html                  # css.err background
    assert "🔴" not in html                        # the owner's WAIT/error circle stays for crashes

    # And the branch it was split OUT of is untouched: a genuine capacity stop still reads as
    # one, so the split gained a true label rather than trading one wrong label for another.
    capacity = {
        "running": True,
        "status": {"apps": {"hinge": {"app": "hinge", "mode": "auto", "state": "stopped",
                                      "stop_reason": "run budget reached", "stop_kind": "opener",
                                      "swipes_run": 2}}},
    }
    other = _run_node(_autostatus_script(capacity))["html"]
    assert "opener capacity exhausted" in other
    assert "could not like the item" not in other


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


def test_render_auto_status_shows_stopping_instead_of_running_during_the_shutdown_tail():
    """Sibling of the test above: the run is STILL alive (thread not yet exited, so
    snap.running is True) but status.stopping is True -- supervisor.run()'s shutdown tail
    covers every non-terminal per-app state this fall-through renders (acting/capturing/
    scoring/saving), and a live worker in that window is about to have its current decision
    discarded (worker.py re-checks stop_event and drops it), so a green "running" box would
    tell the operator the opposite of what's actually happening."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {
        "running": True,
        "status": {"stopping": True, "apps": {
            "hinge": {"app": "hinge", "mode": "auto", "state": "acting", "swipes_run": 5},
        }},
    }
    result = _run_node(_autostatus_script(snap))
    assert result["display"] == "block"
    assert "stopping — finishing the current profile" in result["html"]
    assert "hinge: running" not in result["html"]

    # Regression guard: same non-terminal state, NOT stopping, still reads as running.
    not_stopping = {
        "running": True,
        "status": {"stopping": False, "apps": {
            "hinge": {"app": "hinge", "mode": "auto", "state": "acting", "swipes_run": 5},
        }},
    }
    assert "hinge: running" in _run_node(_autostatus_script(not_stopping))["html"]


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


def test_hub_timed_stop_is_visible_in_both_modes_and_defaults_to_unlimited():
    tag = re.search(r'<select\b[^>]*\bid="stopafter"[^>]*>(.*?)</select>', _PAGE, re.S)
    assert tag, "timed stop control missing"
    assert re.search(r'<option\s+value="0"\s+selected>Unlimited</option>', tag.group(0))
    assert [int(v) for v in re.findall(r'<option\s+value="(\d+)"', tag.group(1))] == [
        0, 900, 1800, 3600, 7200,
    ]
    # The duration is a run-wide safety limit, including an Observe run waiting for the
    # operator; unlike the auto-only max-profile control, it must never be mode-gated.
    assert "$('#stopafter').disabled = !!running" in _PAGE
    assert "$('#mode').value === 'auto'" not in tag.group(0)


def test_hub_defaults_to_observe_even_when_config_defaults_to_auto():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fn = _extract_js_function(_PAGE, "initialHubMode")
    assert _run_node(fn + "\nconsole.log(JSON.stringify(initialHubMode()));\n") == "observe"
    assert "$('#mode').value = initialHubMode()" in _PAGE
    assert "$('#mode').value = cfg.mode" not in _PAGE


def test_hub_start_handler_posts_explicit_unlimited_override():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    assert re.search(r"\$\('#start'\)\.onclick\s*=\s*startRunFromControls", _PAGE)
    fns = (_extract_js_function(_PAGE, "escHtml") + "\n"
           + _extract_js_function(_PAGE, "computeMaxPerRun") + "\n"
           + _extract_js_function(_PAGE, "stopAfterSeconds") + "\n"
           + _extract_js_function(_PAGE, "startRunFromControls"))
    script = (
        "const els = {hint:{textContent:''}, mode:{value:'auto'}, unlimited:{checked:true}, "
        "maxrun:{value:'8'}, stopafter:{value:'0'}, platnote:{innerHTML:''}};\n"
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
        "body": {"mode": "auto", "apps": ["hinge"], "max_per_run": 0,
                 "stop_after_seconds": 0},
    }


def test_hub_start_posts_a_timed_stop_for_observe_runs_too():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fns = (_extract_js_function(_PAGE, "escHtml") + "\n"
           + _extract_js_function(_PAGE, "computeMaxPerRun") + "\n"
           + _extract_js_function(_PAGE, "stopAfterSeconds") + "\n"
           + _extract_js_function(_PAGE, "startRunFromControls"))
    script = (
        "const els = {hint:{textContent:''}, mode:{value:'observe'}, unlimited:{checked:true}, "
        "maxrun:{value:'8'}, stopafter:{value:'1800'}, platnote:{innerHTML:''}};\n"
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
        "body": {"mode": "observe", "apps": ["hinge"], "max_per_run": None,
                 "stop_after_seconds": 1800},
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
            assert release.wait(timeout=_LIVENESS_TIMEOUT_S)
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
    assert started.wait(timeout=_LIVENESS_TIMEOUT_S)  # one caller is inside the computation
    time.sleep(0.05)                   # let the other pollers reach the cold path as waiters
    release.set()
    for t in threads:
        t.join(timeout=_LIVENESS_TIMEOUT_S)

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
        # First bug-report generation cold-imports optional ML diagnostics. Keep ordinary Hub
        # endpoint tests on the 5s default, but do not make this integration assertion depend on
        # whether those imports fit inside that same interactive-response budget under load.
        code, md = _get(base, "/api/bugreport", timeout=15)
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
                {"app": "future_web", "label": "Future web target", "available": False,
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
    assert _run_node(script) == ["hinge", "hinge", "future_web", None]


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


def test_pending_platform_note_uses_and_escapes_server_reason_with_safe_fallback():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    cases = [
        {"label": "Bumble", "reason": "Both Observe and Auto are not calibrated."},
        {"label": "Hostile", "reason": '<img src=x onerror="alert(1)">'},
        {"label": "No reason"},
    ]
    script = _picker_script(
        "console.log(JSON.stringify(" + json.dumps(cases)
        + ".map(pendingPlatformNoteHtml)));\n",
        "escHtml", "pendingPlatformNoteHtml",
    )
    notes = _run_node(script)

    assert "Both Observe and Auto are not calibrated." in notes[0]
    assert "<img" not in notes[1]
    assert "&lt;img" in notes[1] and "&quot;alert(1)&quot;" in notes[1]
    assert "This platform is not available yet." in notes[2]


def test_mode_picker_exposes_escaped_config_reason_through_accessible_hint():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    mode_tag = re.search(r'<select\b[^>]*\bid="mode"[^>]*>', _PAGE)
    hint_tag = re.search(r'<span\b[^>]*\bid="modehint"[^>]*>', _PAGE)
    assert mode_tag and 'aria-describedby="modehint"' in mode_tag.group(0)
    assert hint_tag and 'role="status"' in hint_tag.group(0)
    assert 'aria-live="polite"' in hint_tag.group(0)

    hostile_reason = '<img src=x onerror="alert(1)"> release gate blocked'
    platform = {
        "app": "hinge",
        "modes": {"observe": True, "auto": False},
        "mode_reasons": {"observe": None, "auto": hostile_reason},
    }
    script = _picker_script(
        "const cfg={kinds:[{kind:'android',platforms:[" + json.dumps(platform) + "]}]};\n"
        "let sel={kind:'android',app:'hinge'};\n"
        "const options=[{value:'observe'},{value:'auto'}];\n"
        "let selected='auto';\n"
        "const select={options,selectedIndex:1,title:''};\n"
        "Object.defineProperty(select,'value',{get(){return selected;},set(v){selected=v;this.selectedIndex=options.findIndex(o=>o.value===v);}});\n"
        "const hint={innerHTML:'',textContent:''};\n"
        "function $(selector){return selector==='#mode'?select:hint;}\n"
        "syncModeAvailability();\n"
        "console.log(JSON.stringify({value:select.value,disabled:options[1].disabled,"
        "hidden:options[1].hidden,optionTitle:options[1].title,selectTitle:select.title,"
        "hint:hint.innerHTML}));\n",
        "kindEntry", "escHtml", "modeUnavailableReason", "syncModeAvailability",
    )
    result = _run_node(script)

    assert result["value"] == "observe"
    assert result["disabled"] is True and result["hidden"] is True
    assert result["optionTitle"] == hostile_reason
    assert result["selectTitle"].startswith("Auto unavailable:")
    assert "<img" not in result["hint"]
    assert "&lt;img" in result["hint"] and "&quot;alert(1)&quot;" in result["hint"]


def test_observe_banner_shows_an_unmeasured_licence_as_context_never_as_a_wait_cue():
    """Owner decision 2026-08-21: numbering may ship on an UNMEASURED centered-autoplay
    assumption, and when it does every run must SAY so.

    Numbering looks identical whether a measured held-out bound or the owner's assumption
    licensed it, so words are the only place the difference can survive to an operator. But it
    is context about a decision already taken, not something they must stop for -- so it obeys
    the same rule as the targeting-setup note above: the decision window keeps its 🟢 GO box,
    its GO colours and its single circle, and the provenance is neutral fine print beneath.
    """
    if NODE_BIN is None:
        pytest.skip("node not available on this machine")
    from operation_love import targeting_policy as tp

    notice = tp.STILL_PHOTO_ASSUMPTION_OPERATOR_NOTICE
    waiting = {"running": True, "status": {"apps": {"hinge": {
        "app": "hinge", "mode": "observe", "state": "waiting",
        "targeting_licence_notice": notice}}}}
    html = _run_node(_observe_status_script(waiting))["html"]

    # The decision cue is byte-identical to an ordinary waiting run's.
    assert "🟢 hinge: make your choice: tap X to pass, or tap a heart to like" in html
    # The provenance is stated plainly, and leads with what is missing.
    assert "UNMEASURED assumption (centered autoplay)" in html
    assert "no video false-accept rate has been measured" in html
    # GO styling only: no amber WAIT box, no second circle, no "hands off" cue.
    assert "background:#123a23" in html and "background:#3a2f12" not in html
    assert "🔴" not in html
    assert html.count("🟢") == 1
    # GO cue first, context after -- the inverse of the banner bug reported twice on 2026-08-21.
    assert html.index("🟢") < html.index("UNMEASURED")

    # An ordinary licensed-by-measurement run publishes no notice, so nothing is rendered:
    # a confession printed on every run regardless is how operators learn to stop reading.
    measured = {"running": True, "status": {"apps": {"hinge": {
        "app": "hinge", "mode": "observe", "state": "waiting"}}}}
    measured_html = _run_node(_observe_status_script(measured))["html"]
    assert "UNMEASURED" not in measured_html
    assert "false-accept" not in measured_html

    # It rides along with the copyable-opener box too -- that box is the one the operator reads
    # most, and it must not become the one run state where the provenance disappears.
    with_opener = {"running": True, "status": {"apps": {"hinge": dict(
        waiting["status"]["apps"]["hinge"], state="waiting_for_send",
        opener_suggestion="That ridgeline looks like the Lofoten traverse")}}}
    opener_html = _run_node(_observe_status_script(with_opener))["html"]
    assert "That ridgeline looks like the Lofoten traverse" in opener_html
    assert "no video false-accept rate has been measured" in opener_html
    # And it must not be welded onto the opener line itself (owner rule: the string shown for
    # typing is byte-identical to what auto would type, so hub chrome never shares its row).
    # The provenance follows the whole opener box, in its own neutral row.
    assert opener_html.index("Lofoten traverse") < opener_html.index("false-accept")
    assert "color:#9a9aa2" in opener_html.split("Lofoten traverse")[1]

    # Same escaping rule as every other dynamic value that reaches innerHTML.
    hostile = {"running": True, "status": {"apps": {"hinge": dict(
        waiting["status"]["apps"]["hinge"],
        targeting_licence_notice='<img src=x onerror="alert(1)">')}}}
    hostile_html = _run_node(_observe_status_script(hostile))["html"]
    assert "<img" not in hostile_html and "&lt;img" in hostile_html
