"""Hub HTTP endpoints — offline smoke test (no real run started).

Spins the stdlib server in a thread and hits /, /api/config, /api/status,
/api/stop, and a 404. We never POST /api/start (that would launch a real run /
browser); start/stop wiring is covered by HubState's logic.
"""
import errno
import http.client
import io
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
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


def _notification_window_literal(permission) -> str:
    """The `window` a node harness needs to run the real permission reader.

    `None` means the Notification API is missing entirely (older WebViews, or the hub opened
    over a non-secure origin) -- a different case from a permission that exists and is 'denied',
    and the one `trainingAlertPermission` reports as ''.
    """
    if permission is None:
        return "const window={};\n"
    return "const window={Notification:{permission:" + json.dumps(permission) + "}};\n"


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


@pytest.mark.parametrize("token", [
    None,
    "not-the-server-token",
    # hmac.compare_digest raises TypeError on a str operand holding any non-ASCII character.
    # Nothing in do_POST catches it, so the refusal below used to be no response at all (the
    # caller saw RemoteDisconnected) plus a traceback teed into the operator's live log and every
    # bug report taken afterwards.
    "é" * 43,
    # json.loads hands back a lone surrogate for a "\\ud800" escape, which a plain .encode()
    # would then raise UnicodeEncodeError on -- the same failure one layer down.
    "\ud800",
])
def test_api_stop_rejects_missing_or_wrong_csrf_token(secured_stop_hub, token):
    base, _httpd, state = secured_stop_hub
    payload = {} if token is None else {"_csrf_token": token}
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _raw_post(base, "/api/stop", json.dumps(payload).encode(), {
            "Content-Type": "application/json",
            "Origin": base,
        })
    assert exc_info.value.code == 403
    assert json.loads(exc_info.value.read()) == {"ok": False, "msg": "invalid CSRF token"}
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
        assert hinge["modes"] == {"training": True, "auto": False}
        assert hinge["mode_reasons"]["training"] is None
        assert "does not bind this calibration" in hinge["mode_reasons"]["auto"]
        assert set(hinge["mode_reasons"]) == {"training", "auto"}

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


def test_training_profile_image_endpoint_serves_one_bound_png():
    image = b"\x89PNG\r\n\x1a\nreview-frame"

    class _ImageState:
        def training_profile_review_image(self, **binding):
            assert binding == {
                "run_id": "run-1", "app": "hinge",
                "profile_token": "profile-1", "index": 2,
            }
            return image

    _Handler.state = _ImageState()
    httpd = _bind("127.0.0.1", 8799)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        path = ("/api/training/image?run_id=run-1&app=hinge"
                "&profile_token=profile-1&index=2")
        with urllib.request.urlopen(base + path, timeout=5) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/png"
            assert response.headers["Cache-Control"] == "no-store"
            assert response.read() == image
    finally:
        httpd.shutdown()
        httpd.server_close()
        _join_hub_watch_threads()


def test_training_image_response_never_caches_sensitive_profile_pixels():
    """Exercise the response writer without a loopback socket (sandbox-safe regression)."""
    handler = object.__new__(_Handler)
    headers = {}
    handler.send_response = lambda code: headers.update(status=code)
    handler.send_header = lambda name, value: headers.__setitem__(name, value)
    handler.end_headers = lambda: None
    handler.wfile = io.BytesIO()

    handler._training_image(b"\x89PNG\r\n\x1a\nreview-frame")

    assert headers == {
        "status": 200,
        "Content-Type": "image/png",
        "Content-Length": "20",
        "Cache-Control": "no-store",
    }
    assert handler.wfile.getvalue().startswith(b"\x89PNG")


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
    status = RunStatus("r1", ["hinge"], min_labels=1, mode="training")
    status.set_app(
        "hinge", mode="training", state="blocked",
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
    status = RunStatus("r1", ["hinge", "bumble"], min_labels=1, mode="training")
    status.set_app("bumble", mode="training", state="stopped")
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


def test_hub_tab_close_keeps_waiting_training_approval_alive_until_reopened(monkeypatch,
                                                                              capsys):
    """A preview-tab unload must not cancel the verified Training composer checkpoint."""
    from operation_love.hub import server as hub_server
    from operation_love.status import RunStatus

    monkeypatch.setattr(hub_server, "_BROWSER_SHUTDOWN_GRACE_S", 0.01)
    monkeypatch.setattr(hub_server, "_BROWSER_STALE_CHECK_S", 0.01)

    stop = threading.Event()
    state = HubState("config.yaml")
    status = RunStatus("training-run", ["hinge"], min_labels=0, mode="training")
    status.set_app("hinge", mode="training", state="waiting_approval")
    state._status = status
    state._stop = stop
    state._thread = threading.Thread(target=stop.wait, name="waiting-training-run")
    state._thread.start()
    _Handler.state = state
    httpd = _bind("127.0.0.1", 8799)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        first = json.dumps({"id": "preview-tab"}).encode()
        code, opened = _post(base, "/api/hub/open", first)
        assert code == 200 and opened["ok"] is True
        code, closed = _post(base, "/api/hub/closed", first)
        assert code == 200 and closed["ok"] is True

        # Let the delayed tab-close callback run. The server and worker must both remain live,
        # so a user can reopen the exact checkpoint instead of losing the typed composer.
        time.sleep(0.08)
        assert t.is_alive() is True
        assert state.is_running() is True
        assert stop.is_set() is False
        assert "keeping the run and verified checkpoint alive" in capsys.readouterr().out

        # A new preview/browser tab gets a normal live Hub. Once the approval boundary is gone,
        # the established last-tab-close policy still stops the live run and closes the server.
        reopened = json.dumps({"id": "reopened-tab"}).encode()
        code, opened = _post(base, "/api/hub/open", reopened)
        assert code == 200 and opened["ok"] is True
        status.set_app("hinge", state="scoring")
        code, closed = _post(base, "/api/hub/closed", reopened)
        assert code == 200 and closed["ok"] is True

        t.join(timeout=_LIVENESS_TIMEOUT_S)
        assert t.is_alive() is False
        assert stop.is_set() is True
    finally:
        stop.set()
        state.wait_for_run(timeout=_LIVENESS_TIMEOUT_S)
        httpd.shutdown()
        httpd.server_close()
        _join_hub_watch_threads()


def _tiny_review_png() -> bytes:
    """A 1x1 PNG that satisfies TrainingActionBridge.publish_checkpoint's strict frame
    validation (training_actions._raster_mime/_valid_png_idat reject a bare magic number)."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x00")) + chunk(b"IEND", b""))


@pytest.mark.parametrize("phase", ["ready", "queued", "executing"])
def test_browser_close_preserves_live_checkpoint_before_worker_reports_waiting_approval(phase):
    """A bridge checkpoint keeps the hub alive through every decision phase.

    publish_checkpoint() runs on the training worker's own thread and precedes its status
    update to waiting_approval by however long that thread takes to report back -- see the
    comment on browser_close_shutdown_disposition.  The bridge, rather than that later status
    snapshot, must also protect a choice that is queued or being applied/recorded.
    """
    class _Worker:
        run_id = "training-run"
        app = "hinge"
        mode = "training"
        training_action_supported = True

        def __init__(self):
            self.stop_event = threading.Event()

    class _Pick:
        text = "The typed opener"
        referenced = "mountain photo"
        index = 2
        item_description = "mountain photo"

    st = HubState("config.yaml")
    st._stop = threading.Event()
    st._thread = threading.Thread(target=st._stop.wait, name="fake-training-run", daemon=True)
    st._thread.start()
    try:
        worker = _Worker()
        st._training_actions.register(worker)
        card = st._training_actions.publish_checkpoint(worker, _tiny_review_png(), _Pick())
        if phase != "ready":
            ok, result, code = st._training_actions.submit({
                "command": "like", "run_id": card["run_id"], "app": card["app"],
                "profile_token": card["profile_token"],
                "approval_token": card["approval_token"], "idempotency_token": "test-action",
            })
            assert (ok, result["status"], code) == (True, "queued", 202)
        if phase == "executing":
            action = st._training_actions.wait_for_action(
                worker, card["profile_token"], worker.stop_event)
            assert action is not None and action["status"] == "executing"

        # The worker has not yet reported waiting_approval, so the status-snapshot branch cannot
        # protect this card.  Every live card -- untouched, queued, or being applied -- must keep
        # the server and run alive.
        assert st._status is None
        assert st._training_actions.has_live_checkpoint() is True
        assert st.browser_close_shutdown_disposition() == "preserve_training_approval"
    finally:
        st._stop.set()
        st._thread.join(timeout=_LIVENESS_TIMEOUT_S)


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


def test_hubstate_refuses_to_start_while_training_data_is_being_mutated(monkeypatch):
    import operation_love.ranker as ranker

    entered = threading.Event()
    release = threading.Event()

    class _Store:
        def remove_latest_training_label(self):
            entered.set()
            assert release.wait(timeout=_LIVENESS_TIMEOUT_S)
            return {"profile_name": "Taylor"}

        def close(self):
            pass

    monkeypatch.setattr(ranker, "make_store", lambda cfg: _Store())
    monkeypatch.setattr("operation_love.hub.state.cfg_mod.load", lambda path: object())
    st = HubState("config.yaml")
    result = []
    mutation = threading.Thread(
        target=lambda: result.append(st._training_store_mutation("remove_latest_training_label")),
        daemon=True,
    )
    mutation.start()
    assert entered.wait(timeout=_LIVENESS_TIMEOUT_S)

    ok, message = st.start(mode="training", apps=["hinge"])

    assert ok is False
    assert "training data is being updated" in message
    release.set()
    mutation.join(timeout=_LIVENESS_TIMEOUT_S)
    assert result == [(True, {"profile_name": "Taylor"})]
    assert st._training_data_mutating is False


def test_hubstate_snapshot_forwards_the_stopping_field_from_run_status():
    # HubState.snapshot() must not need a redundant field of its own -- RunStatus.snapshot()
    # (embedded verbatim as snap["status"]) already carries `stopping`, so nothing extra is
    # needed here as long as nothing strips it back out.
    from operation_love.status import RunStatus

    st = HubState("config.yaml")
    status = RunStatus("r1", ["hinge"], min_labels=1, mode="training")
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
    status = RunStatus("r1", ["hinge"], min_labels=1, mode="training")
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

        def __init__(self):
            self.closed = False

        def serve_forever(self):
            raise KeyboardInterrupt

        def shutdown(self):
            pass

        def server_close(self):
            self.closed = True

    httpd = _FakeHttpd()
    monkeypatch.setattr(hub_server, "_bind", lambda host, port: httpd)

    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a))))
    try:
        hub_server.serve("config.yaml", open_browser=False)
    finally:
        stuck.set()
        state._thread.join(timeout=_LIVENESS_TIMEOUT_S)

    assert any("did not finish saving" in line for line in printed)
    assert httpd.closed is True


def _shutdown_cfg(scroll_captures: int):
    """Minimal stand-in carrying only the fields the two shutdown bounds actually read.

    Opener enabled at the LARGEST legal request timeout so the worker-join term is at its
    ceiling too: the invariant has to hold for the worst legal run, not the shipped one.
    """
    import operation_love.config as config_module
    from types import SimpleNamespace

    return SimpleNamespace(
        enabled_apps=["hinge"],
        apps={"hinge": {"scroll_captures": scroll_captures}},
        opener=SimpleNamespace(enabled=True,
                               request_timeout_s=config_module._MAX_REQUEST_TIMEOUT_S))


def test_hub_archive_ceiling_covers_the_supervisors_join_for_every_legal_capture_budget():
    """The Hub's archive extension is composed per-run, and must never invert the invariant.

    The join a healthy worker can legitimately use is three terms, not one: an in-flight opener
    request, the profile archive that store.close() then blocks on anyway, and the margin the
    worker gets afterwards to write that already-landed action's rows.

    The trap this pins is that the archive term is NOT a fixed number: _archive_join_grace_s
    reads it off the run's own scroll_captures, which config validation admits all the way to
    _MAX_ANDROID_SCROLL_CAPTURES -- while the Training review ceiling is only
    MAX_PROFILE_REVIEW_FRAMES. A Hub bound frozen at the review ceiling INVERTS for every legal
    run above it (auto with scroll_captures=24 already qualifies): the Hub abandons the process
    while the supervisor's join is still legitimately running, so flush()/close() never happen
    and the buffered rows of an irreversible landed action are lost. Comparing the grace alone
    against the outer bound -- as this test previously did, at a single 12-capture config --
    cannot see that class at all, because both sides move.
    """
    import operation_love.config as config_module
    import operation_love.hub.server as hub_server
    import operation_love.supervisor as supervisor_module
    from operation_love.training_actions import MAX_PROFILE_REVIEW_FRAMES

    # The FLAT budget carries the opener-side join on its own: a worker riding out one
    # in-flight opener request registers no store write, so the conditional extension below
    # cannot rescue it -- the probe would (correctly) answer "no archive" and the Hub would
    # abandon a healthy join.
    assert hub_server._SHUTDOWN_SAVE_TIMEOUT_S >= (
        config_module._MAX_REQUEST_TIMEOUT_S
        + supervisor_module._WORKER_JOIN_TIMEOUT_MARGIN_S)

    budgets = (1, MAX_PROFILE_REVIEW_FRAMES, 24, config_module._MAX_ANDROID_SCROLL_CAPTURES)
    assert config_module._MAX_ANDROID_SCROLL_CAPTURES > MAX_PROFILE_REVIEW_FRAMES  # the gap
    for captures in budgets:
        cfg = _shutdown_cfg(captures)
        supervisor_join = (supervisor_module._worker_join_timeout_s(cfg)
                           + supervisor_module._archive_join_grace_s(cfg)
                           + supervisor_module._WORKER_JOIN_TIMEOUT_MARGIN_S)
        assert hub_server._run_archive_ceiling_s(cfg) >= supervisor_join, captures
    # The archive term is load-bearing: a ceiling that silently dropped back to the opener-only
    # derivation would be far below the 100-capture run's own grace.
    biggest = _shutdown_cfg(config_module._MAX_ANDROID_SCROLL_CAPTURES)
    assert hub_server._run_archive_ceiling_s(biggest) > (
        config_module._MAX_REQUEST_TIMEOUT_S
        + 2 * supervisor_module._WORKER_JOIN_TIMEOUT_MARGIN_S
        + hub_server._SHUTDOWN_SAVE_HEADROOM_S)
    # ...and the ceiling is genuinely per-run, not a constant wearing a function's clothes.
    assert (hub_server._run_archive_ceiling_s(biggest)
            > hub_server._run_archive_ceiling_s(_shutdown_cfg(1)))
    # No cfg in reach (nothing bound yet) still yields a usable flat bound, never a crash.
    assert hub_server._run_archive_ceiling_s(None) == (
        hub_server._SHUTDOWN_ARCHIVE_CEILING_FALLBACK_S)
    assert hub_server._run_archive_ceiling_s(object()) == (
        hub_server._SHUTDOWN_ARCHIVE_CEILING_FALLBACK_S)


def test_hub_shutdown_stops_at_the_flat_budget_when_nothing_is_archiving(monkeypatch):
    """Finding B/C: the conditional extension must cost an idle shutdown nothing.

    A wedged worker with no registered write in flight has no data to protect, so the Hub owes
    it only the flat prompt-exit budget -- that is what keeps "close the tab -> hub exits ->
    launcher closes the Terminal tab" feeling like quitting.
    """
    import operation_love.hub.server as hub_server

    monkeypatch.setattr(hub_server, "_SHUTDOWN_SAVE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(hub_server, "_SHUTDOWN_ARCHIVE_POLL_S", 0.02)
    # Deliberately enormous next to the flat budget: if the extension were unconditional, this
    # test would visibly hang rather than fail by a hair.
    monkeypatch.setattr(hub_server, "_run_archive_ceiling_s", lambda cfg: 30.0)

    stuck = threading.Event()
    st = HubState("config.yaml")
    st._thread = threading.Thread(target=stuck.wait, daemon=True)
    st._thread.start()
    assert st.live_store() is None          # nothing archiving: the probe's False answer

    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a))))
    try:
        t0 = time.monotonic()
        assert hub_server._wait_for_run_shutdown(st) is False
        elapsed = time.monotonic() - t0
    finally:
        stuck.set()
        st._thread.join(timeout=_LIVENESS_TIMEOUT_S)
    # Not a liveness bound: this IS the assertion. It has to stay well under the 30.0s ceiling
    # above, since falling through to that ceiling is exactly the regression being excluded.
    assert elapsed < 5.0
    # And the operator is never told an archive is holding shutdown open when none is: that
    # notice announces a wait this path did not take.
    assert not any("still archiving" in line for line in printed)


class _ArchiveProbe:
    """The duck-typed seam supervisor._store_archive_in_flight reads (bigquery_store's).

    Reports an archive in flight for the first ``in_flight_for`` probes, then drained.
    """

    def __init__(self, in_flight_for: int):
        self._in_flight_for = in_flight_for
        self.probes = 0

    def archive_writes_in_flight(self) -> bool:
        self.probes += 1
        return self.probes <= self._in_flight_for


def test_hub_shutdown_extends_past_the_flat_budget_while_an_archive_is_writing(monkeypatch):
    """Finding A: an already-landed decision's archive outlives the flat prompt-exit budget.

    Without the extension the Hub returns False here and serve()'s finally exits the process
    while the supervisor's own join is still legitimately running, so store.flush()/close()
    never write that action's decision/opener/label rows.
    """
    import operation_love.hub.server as hub_server

    monkeypatch.setattr(hub_server, "_SHUTDOWN_SAVE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(hub_server, "_SHUTDOWN_ARCHIVE_POLL_S", 0.02)
    monkeypatch.setattr(hub_server, "_run_archive_ceiling_s", lambda cfg: _LIVENESS_TIMEOUT_S)

    saving = threading.Event()
    st = HubState("config.yaml")
    # Still saving when the flat budget expires, then finishes -- the healthy shutdown the flat
    # bound alone would have truncated.
    st._thread = threading.Thread(target=lambda: saving.wait(_LIVENESS_TIMEOUT_S), daemon=True)
    st._thread.start()
    st._live_store = _ArchiveProbe(in_flight_for=10**6)   # never drains on its own
    releaser = threading.Timer(0.3, saving.set)
    releaser.daemon = True          # a test helper must never outlive the test run
    releaser.start()

    try:
        assert hub_server._wait_for_run_shutdown(st) is True
    finally:
        saving.set()
        st._thread.join(timeout=_LIVENESS_TIMEOUT_S)
    # More than the single pre-loop probe: it kept slicing rather than sampling once.
    assert st._live_store.probes >= 2


def test_hub_shutdown_ends_the_archive_extension_as_soon_as_the_write_drains(monkeypatch):
    """The extension is the archive's budget, and no other wedge may inherit it.

    Re-probing every slice (not sampling once before the loop) is what keeps a worker that is
    genuinely stuck from spending the whole per-run ceiling after the write it was protecting
    has already finished.
    """
    import operation_love.hub.server as hub_server

    monkeypatch.setattr(hub_server, "_SHUTDOWN_SAVE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(hub_server, "_SHUTDOWN_ARCHIVE_POLL_S", 0.02)
    monkeypatch.setattr(hub_server, "_run_archive_ceiling_s", lambda cfg: _LIVENESS_TIMEOUT_S)

    stuck = threading.Event()
    st = HubState("config.yaml")
    st._thread = threading.Thread(target=stuck.wait, daemon=True)   # never finishes
    st._thread.start()
    st._live_store = _ArchiveProbe(in_flight_for=2)

    try:
        t0 = time.monotonic()
        assert hub_server._wait_for_run_shutdown(st) is False
        elapsed = time.monotonic() - t0
    finally:
        stuck.set()
        st._thread.join(timeout=_LIVENESS_TIMEOUT_S)
    assert st._live_store.probes >= 2          # it did extend past the flat budget…
    # …and then stopped at the drain. Not a liveness bound: this IS the assertion, and it must
    # stay far below the _LIVENESS_TIMEOUT_S ceiling above, which is what burning the whole
    # extension on a drained archive would cost.
    assert elapsed < 5.0


def test_eval_refresh_start_failure_clears_its_inflight_guard(monkeypatch):
    import operation_love.hub.state as hub_state

    class _StartFails:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def start(self):
            raise RuntimeError("thread resources exhausted")

    monkeypatch.setattr(hub_state.threading, "Thread", _StartFails)
    st = HubState("config.yaml")

    st._start_eval_refresh(every=5)

    assert st._eval_refreshing is False


def test_serve_browser_timer_is_daemon_and_socket_is_closed(monkeypatch):
    import operation_love.hub.server as hub_server

    class _FakeHttpd:
        server_address = ("127.0.0.1", 8799)

        def __init__(self):
            self.closed = False

        def serve_forever(self):
            return None

        def shutdown(self):
            pass

        def server_close(self):
            self.closed = True

    timers = []

    class _FakeTimer:
        def __init__(self, delay, target):
            self.delay = delay
            self.target = target
            self.daemon = False
            self.started = False
            timers.append(self)

        def start(self):
            self.started = True

    httpd = _FakeHttpd()
    monkeypatch.setattr(hub_server, "HubState", lambda config_path: HubState(config_path))
    monkeypatch.setattr(hub_server, "_bind", lambda host, port: httpd)
    monkeypatch.setattr(hub_server.threading, "Timer", _FakeTimer)

    hub_server.serve("config.yaml", open_browser=True)

    assert len(timers) == 1
    assert timers[0].delay == 0.6 and timers[0].daemon and timers[0].started
    assert httpd.closed is True


def test_serve_closes_socket_when_optional_browser_timer_cannot_start(monkeypatch, capsys):
    import operation_love.hub.server as hub_server

    class _FakeHttpd:
        server_address = ("127.0.0.1", 8799)

        def __init__(self):
            self.served = self.closed = False

        def serve_forever(self):
            self.served = True

        def shutdown(self):
            pass

        def server_close(self):
            self.closed = True

    class _StartFails:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.daemon = False

        def start(self):
            raise RuntimeError("thread resources exhausted")

    httpd = _FakeHttpd()
    monkeypatch.setattr(hub_server, "HubState", lambda config_path: HubState(config_path))
    monkeypatch.setattr(hub_server, "_bind", lambda host, port: httpd)
    monkeypatch.setattr(hub_server.threading, "Timer", _StartFails)

    hub_server.serve("config.yaml", open_browser=True)

    assert httpd.served and httpd.closed
    assert "could not open browser automatically" in capsys.readouterr().out


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


@pytest.mark.parametrize("client_id", [None, "", "   ", [], {}, 1, "x" * 129])
def test_hubstate_rejects_invalid_browser_client_ids_without_touching_liveness_state(client_id):
    st = HubState("config.yaml")

    assert st.browser_client_opened(client_id) is False
    assert st.browser_client_closed(client_id) is False
    assert st.has_browser_clients() is False


def test_browser_liveness_watch_can_be_retried_after_thread_start_failure():
    st = HubState("config.yaml")
    assert st.browser_client_opened("client-a") is True
    assert st.browser_stale_watch_active() is True

    st.browser_stale_watch_start_failed()

    assert st.browser_stale_watch_active() is False
    assert st.browser_client_ping("client-a") is True


def test_json_parser_rejects_duplicate_control_fields():
    from operation_love.hub.server import _json_object_without_duplicate_keys

    with pytest.raises(ValueError, match="duplicate JSON field: mode"):
        json.loads(
            '{"mode":"training","mode":"auto"}',
            object_pairs_hook=_json_object_without_duplicate_keys,
        )


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


@pytest.mark.parametrize(("requested", "effective"), [
    ("training", "auto"),
    ("auto", "training"),
])
def test_hub_explicit_mode_never_silently_yields_to_per_app_autonomy(
        monkeypatch, requested, effective):
    """Per-app config precedence is valid for CLI, not an explicit Hub control click."""
    import operation_love.hub as hub
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        mode=requested, enabled_apps=["hinge"], apps={"hinge": {"mode": effective}})
    launched = []
    monkeypatch.setattr(hub.supervisor, "load_effective_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(hub.supervisor, "run", lambda *args, **kwargs: launched.append(True))
    st = HubState("config.yaml")

    ok, message = st.start(mode=requested, apps=["hinge"])

    assert ok is False
    assert f"hinge={effective}" in message
    assert launched == []
    assert st._thread is None


@pytest.mark.parametrize("mode", ["training", "auto"])
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
    ok, _ = st.start(mode="training", apps=["hinge"], stop_after_seconds=10)
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
            data=json.dumps({"mode": "training", "apps": ["hinge"],
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
                "mode": "training", "apps": ["hinge"], "max_per_ru": 8,
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
        payload = {"mode": "training", "apps": ["hinge"], "max_per_run": None,
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
    "/api/training/action", "/api/hub/open", "/api/hub/ping",
    "/api/hub/closed", "/api/training/alert",
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


def test_training_alert_endpoint_records_only_bounded_browser_telemetry():
    class AlertSpy:
        def __init__(self):
            self.rows = []

        def record_training_browser_notification(self, body):
            self.rows.append(body)
            return True, "recorded"

    state = AlertSpy()
    _Handler.state = state
    httpd = _bind("127.0.0.1", 8799)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        body = json.dumps({
            "run_id": "run-1", "app": "hinge", "profile_token": "profile-1",
            "notification": "requested",
        }).encode()
        code, result = _post(base, "/api/training/alert", body)
        assert code == 200 and result == {"ok": True, "msg": "recorded"}
        assert state.rows == [{
            "run_id": "run-1", "app": "hinge", "profile_token": "profile-1",
            "notification": "requested",
        }]
    finally:
        httpd.shutdown()
        httpd.server_close()


def _alert_telemetry_worker():
    class _Worker:
        run_id = "alert-run"
        app = "hinge"
        mode = "training"
        training_action_supported = True

        def __init__(self):
            self.stop_event = threading.Event()

    class _Pick:
        text = "The typed opener"
        referenced = "mountain photo"
        index = 2
        item_description = "mountain photo"

    return _Worker(), _Pick()


def test_hub_state_binds_browser_alert_telemetry_to_the_live_checkpoint():
    """The endpoint test above swaps HubState out, so it can see none of this validation.

    HubState.record_training_browser_notification is the only writer of this telemetry, and it
    must accept a row ONLY for the checkpoint that is live right now and ONLY from the fixed
    outcome vocabulary -- otherwise any local POST could write rows naming a run/profile that is
    not on screen. Both refusal branches were previously unreachable from any test: replacing the
    whole method with ``return True, "recorded"`` left the suite green.
    """
    from operation_love import notifications

    worker, pick = _alert_telemetry_worker()
    st = HubState("config.yaml")
    st._training_actions.register(worker)
    card = st._training_actions.publish_checkpoint(worker, _tiny_review_png(), pick)
    live = {"run_id": card["run_id"], "app": card["app"],
            "profile_token": card["profile_token"]}

    before = len(notifications.recent_training_alerts())
    assert st.record_training_browser_notification(
        {**live, "notification": "requested"}) == (True, "recorded")
    rows = notifications.recent_training_alerts()
    assert len(rows) == before + 1
    assert rows[-1]["channel"] == "hub-browser"
    assert rows[-1]["notification"] == "requested"

    for wrong in ({"profile_token": "a-retired-profile"}, {"run_id": "some-other-run"},
                  {"app": "bumble"}, {"profile_token": None}, {"run_id": {"nested": "object"}},
                  {"app": None}):
        assert st.record_training_browser_notification(
            {**live, **wrong, "notification": "requested"}) == (
                False, "no matching live training checkpoint")

    for outcome in ("sent", "", None, {"notification": "requested"}, "REQUESTED"):
        assert st.record_training_browser_notification(
            {**live, "notification": outcome}) == (
                False, "invalid browser notification outcome")

    # A refused row must not reach the diagnostic ring at all.
    assert len(notifications.recent_training_alerts()) == before + 1


def test_training_alert_endpoint_refuses_a_retired_checkpoint_with_409():
    """Pins hub/server.py's ``200 if ok else 409`` on the real state object."""
    worker, pick = _alert_telemetry_worker()
    st = HubState("config.yaml")
    st._training_actions.register(worker)
    card = st._training_actions.publish_checkpoint(worker, _tiny_review_png(), pick)

    _Handler.state = st
    httpd = _bind("127.0.0.1", 8799)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        live = {"run_id": card["run_id"], "app": card["app"],
                "profile_token": card["profile_token"], "notification": "requested"}
        code, result = _post(base, "/api/training/alert", json.dumps(live).encode())
        assert (code, result) == (200, {"ok": True, "msg": "recorded"})

        stale = {**live, "profile_token": "a-retired-profile"}
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post(base, "/api/training/alert", json.dumps(stale).encode())
        assert exc_info.value.code == 409
        assert json.loads(exc_info.value.read()) == {
            "ok": False, "msg": "no matching live training checkpoint"}
    finally:
        httpd.shutdown()
        httpd.server_close()
        _join_hub_watch_threads()


def test_hub_card_prioritizes_false_dislike_readiness_instead_of_accuracy():
    assert "base-model safety check" in _PAGE
    assert "Accepted profiles kept" in _PAGE
    assert "false dislikes" in _PAGE
    assert "TARGET_RECALL = 0.95" in _PAGE
    assert "TARGET_IDENTITIES = 500" in _PAGE
    assert "not an AUTO release approval" in _PAGE
    assert "e.roc_auc[0]*100" not in _PAGE
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
    expected = _MAC_UPDATE_RUN.replace("__EXTRAS__", "ml,bq,hinge").replace("__CONFIG_ARG__", "")
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
    (tmp_path / "config.yaml").write_text("mode: training\n")
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


@pytest.mark.parametrize(("platform", "name", "expected"), [
    ("darwin", "Operation Love.command", '"$PY" -m operation_love hub\n'),
    ("linux", "operation-love.sh", 'exec "$PY" -m operation_love hub\n'),
    ("win32", "Operation Love.bat", '"%PY%" -m operation_love hub\n'),
])
def test_launcher_default_config_preserves_existing_hub_command(
        tmp_path, monkeypatch, platform, name, expected):
    from operation_love.hub.launchers import make_launchers

    monkeypatch.setattr(sys, "platform", platform)
    make_launchers(str(_launcher_project(tmp_path)), extras="hinge")

    launcher = (tmp_path / name).read_text()
    assert expected in launcher
    assert "--config" not in launcher


@pytest.mark.parametrize(("platform", "name", "config_name", "expected"), [
    ("darwin", "Operation Love.command", "--evening.yaml", "hub --config=--evening.yaml"),
    ("linux", "operation-love.sh", "--evening.yaml", "hub --config=--evening.yaml"),
    ("win32", "Operation Love.bat", "--evening.yaml", 'hub --config="--evening.yaml"'),
    ("darwin", "Operation Love.command", "evening plans' %!^&.yaml",
     "hub --config='evening plans'\"'\"' %!^&.yaml'"),
    ("linux", "operation-love.sh", "evening plans' %!^&.yaml",
     "hub --config='evening plans'\"'\"' %!^&.yaml'"),
    ("win32", "Operation Love.bat", "evening plans' %!^&.yaml",
     'hub --config="evening plans\' %%!^&.yaml"'),
])
def test_launcher_forwards_and_quotes_custom_config_filename(
        tmp_path, monkeypatch, platform, name, config_name, expected):
    from operation_love.hub.launchers import make_launchers

    config = _launcher_project(tmp_path).with_name(config_name)
    config.write_text("mode: training\n")
    monkeypatch.setattr(sys, "platform", platform)

    make_launchers(str(config), extras="hinge")

    launcher = (tmp_path / name).read_text()
    assert expected in launcher
    if platform == "win32":
        assert "setlocal DisableDelayedExpansion\n" in launcher


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
    # The readiness card says when its metrics will next update, without an unexplained ratio.
    # It must read HubState._attach_refresh's own `remaining` (see the edge-case test below)
    # rather than re-deriving it from `since`/`every` a second, independently-drifting way.
    assert "Updates after ${remaining} more training label" in _PAGE
    assert "r.mode !== 'training'" in _PAGE          # only while a live training run feeds labels
    assert "r.remaining == null" in _PAGE
    assert "Number(r.remaining)" in _PAGE


def test_attach_refresh_counts_down_to_next_recompute():
    class LiveTraining:
        running = True
        mode = "training"

    # 2 of 5 new labels collected since the last compute -> 3 swipes remaining.
    r = HubState._attach_refresh({"status": "ok"}, every=5, live=42, base=40, status=LiveTraining())
    assert r["refresh"] == {"every": 5, "since": 2, "remaining": 3, "live": True, "mode": "training"}

    # Right after a recompute (since == every) it rolls to a full cycle, never shows 0/5.
    edge = HubState._attach_refresh({"status": "ok"}, every=5, live=45, base=40, status=LiveTraining())
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


def test_hub_removes_accuracy_trajectory_chart():
    assert "accSvg(e.trajectory)" not in _PAGE
    assert "accuracy over labels" not in _PAGE


def test_hub_readiness_card_collapses_confusion_matrix_details():
    assert '<details class="readiness-details">' in _PAGE
    assert "Model likes" in _PAGE and "Model dislikes" in _PAGE
    assert "true_accepted" in _PAGE and "false_dislikes" in _PAGE


def test_cached_eval_uses_live_training_mix_without_waiting_for_next_cv_refresh():
    class LiveTraining:
        running = True
        mode = "training"
        labels = 3

    class LiveStore:
        def load_labels(self):
            return [(True, []), (False, []), (True, [])]

    st = HubState("config.yaml")
    with st._lock:
        st._eval = {"status": "insufficient_data", "labels": 2, "likes": 1, "passes": 1}
        st._eval_at = time.time()
        st._eval_labels = 0
        st._status = LiveTraining()
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

    class LiveTraining:
        running = True
        mode = "training"
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
                        lambda samples, **kwargs: {"status": "ok", "marker": "new", "labels": len(samples)})

    st = HubState("config.yaml")
    with st._lock:
        st._eval = {"status": "ok", "marker": "old"}
        st._eval_at = time.time()
        st._eval_labels = 40
        st._status = LiveTraining()

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

    class LiveTraining:
        running = True
        mode = "training"
        labels = 30

    live_calls = []

    class LiveStore:
        def load_labels(self):
            live_calls.append("live")
            return [object()] * 30           # the live in-memory set (all swipes)

    class Fresh:
        def load_labels(self):
            return [object()] * 5            # committed-only (would lag) — must NOT be used here
        def close(self):
            pass

    monkeypatch.setattr(hub.cfg_mod, "load", lambda p: object())
    monkeypatch.setattr(ranker, "make_store", lambda cfg, ensure=True: Fresh())
    monkeypatch.setattr(eval_mod, "evaluate", lambda s, **kwargs: {
        "status": "ok", "labels": len(s), "identities": len(s), "base_rate": 0.4,
        "pr_auc": [0.7, 0.05], "roc_auc": [0.8, 0.03], "brier": [0.2, 0.0]})

    st = HubState("config.yaml")
    st._thread = threading.Thread(target=lambda: time.sleep(0.5))   # fake a live run
    st._thread.start()
    with st._lock:
        st._status = LiveTraining()
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

    class LiveTraining:
        running = True
        mode = "training"
        labels = 12

    class DeadStore:
        def load_labels(self):
            raise RuntimeError("Cannot operate on a closed database")

    fresh_used = []

    class Fresh:
        def load_labels(self):
            fresh_used.append(1)
            return [object()] * 12
        def close(self):
            pass

    monkeypatch.setattr(hub.cfg_mod, "load", lambda p: object())
    monkeypatch.setattr(ranker, "make_store", lambda cfg, ensure=True: Fresh())
    monkeypatch.setattr(eval_mod, "evaluate", lambda s, **kwargs: {
        "status": "ok", "labels": len(s), "identities": len(s), "base_rate": 0.4,
        "pr_auc": [0.7, 0.05], "roc_auc": [0.8, 0.03], "brier": [0.2, 0.0]})

    st = HubState("config.yaml")
    st._thread = threading.Thread(target=lambda: time.sleep(0.3))
    st._thread.start()
    with st._lock:
        st._status = LiveTraining()
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
    # the distinct explicit-unlimited override 0. The shipped config is currently uncapped,
    # but this distinction preserves any deliberate future configured ceiling.
    #
    # Evaluated for real with node (not a substring check): an earlier substring-only test
    # passed after the input handling was gutted as long as the phrase survived in a comment.
    # Extracting computeMaxPerRun() lets the test assert the actual result.
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fn = _extract_js_function(_PAGE, "computeMaxPerRun")
    # (mode, unlimitedChecked, rawValue)
    cases = [
        ("training", False, "8"),   # Training uses the same run-cap controls as Auto.
        ("auto", True, "8"),        # unlimited checkbox -> 0, regardless of the box
        ("auto", True, ""),
        ("auto", False, ""),        # empty box -> delegate to config
        ("auto", False, "abc"),     # non-numeric -> delegate to config
        ("auto", False, "0"),       # typing 0 directly delegates; checkbox owns explicit 0
        ("auto", False, "-3"),      # negative -> delegate to config
        ("auto", False, "3.7"),     # never silently truncate a requested cap
        ("auto", False, "8profiles"),
        ("auto", False, str(2 ** 53)),
        ("auto", False, "8"),       # a genuine positive cap is honored
    ]
    expected = [8, 0, 0, None, None, None, None, None, None, None, 8]
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


def _render_global_script(calls: list, alert_permission: str = "granted") -> str:
    """Run the real renderGlobal against a minimal DOM, once per entry in `calls`, in the
    SAME node process/script and in order -- renderGlobal is stateful across polls (the
    module-level `_wasRunning` shouldClearHint reads), so a fresh eval per call would miss
    exactly the transition this is testing."""
    fns = (_extract_js_function(_PAGE, "stopButtonState") + "\n"
           + _extract_js_function(_PAGE, "shouldClearHint") + "\n"
           + _extract_js_function(_PAGE, "formatTimedStopRemaining") + "\n"
           # renderGlobal re-derives #alerthint on the running transition, so the REAL hint
           # chain is wired here rather than stubbed: a stub would let the wording drift back
           # to naming a control the operator cannot reach.
           + _extract_js_function(_PAGE, "trainingAlertPermission") + "\n"
           + _extract_js_function(_PAGE, "trainingAlertHintText") + "\n"
           + _extract_js_function(_PAGE, "syncTrainingAlertHint") + "\n"
           + _extract_js_function(_PAGE, "renderGlobal"))
    return (
        "let _wasRunning = false, _trainingDataMutationBusy = false, _configReady = true;\n"
        + _notification_window_literal(alert_permission) +
        "let els = {runpill:{textContent:'',className:''}, start:{disabled:false}, "
        "stop:{disabled:false,textContent:''}, hint:{textContent:''}, budget:{textContent:''}, "
        "err:{textContent:''}, stopafter:{disabled:false}, timerhint:{textContent:''}, "
        "alerthint:{textContent:''}, removeLatestTraining:{disabled:false}};\n"
        "function $(sel){ return els[sel.slice(1)]; }\n"
        + fns + "\n"
        "const calls = " + json.dumps(calls) + ";\n"
        "const results = [];\n"
        "for (const c of calls) {\n"
        "  if (c.presetHint != null) els.hint.textContent = c.presetHint;\n"
        "  els.alerthint.textContent = '';\n"
        "  renderGlobal(c.snap);\n"
        "  results.push({hint: els.hint.textContent, pill: els.runpill.textContent, "
        "stopDisabled: els.stop.disabled, stopLabel: els.stop.textContent, "
        "timerDisabled: els.stopafter.disabled, timerHint: els.timerhint.textContent, "
        "alertHint: els.alerthint.textContent});\n"
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


def _hub_wake_script(body: str) -> str:
    """The three wake callbacks over a stub lifecycle + stub `tick`, run for real under node.

    `tick` is counted rather than stubbed away: re-registering the tab and re-polling status are
    two different obligations of the same wake, and only counting both can tell them apart.
    """
    return (
        "let calls = [], ticks = 0;\n"
        "function hubLifecycle(path){ calls.push(path); }\n"
        "function tick(){ ticks += 1; }\n"
        "let document = { hidden: true };\n"
        + _extract_js_function(_PAGE, "hubWake") + "\n"
        + _extract_js_function(_PAGE, "_onHubVisibilityWake") + "\n"
        + _extract_js_function(_PAGE, "_onHubFocusWake") + "\n"
        + body
    )


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
    script = _hub_wake_script(
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


def test_every_tab_wake_also_repolls_status_not_just_the_lifecycle_registration():
    """The audited gap: a hidden tab's 1s status poll is throttled exactly like its ping, so a
    checkpoint raised while the operator was elsewhere could sit un-rendered for up to a minute
    after they came back -- the macOS chime beat the browser banner by 41s in the audited run,
    and on that host the native API is sound-only, so the banner is the only text channel.

    All three wake paths must re-poll, not just re-register: `pageshow` is wired to the SAME
    named callback, so asserting on the wiring plus these two proves the third. A visibility
    event on a still-hidden tab must stay silent on both counts -- a poll there would neither
    help (the operator is not looking) nor be honest about what this fix can do.
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    assert "window.addEventListener('pageshow', hubWake)" in _PAGE
    script = _hub_wake_script(
        "_onHubVisibilityWake();\n"                 # still hidden -> no poll either
        "const hiddenTicks = ticks;\n"
        "document.hidden = false;\n"
        "_onHubVisibilityWake();\n"                 # woke visible -> poll now, not in ~60s
        "const visibleTicks = ticks;\n"
        "_onHubFocusWake();\n"
        "hubWake();\n"                              # the pageshow/bfcache leg
        "console.log(JSON.stringify({hiddenTicks, visibleTicks, ticks, calls: calls.length}));\n"
    )
    assert _run_node(script) == {
        "hiddenTicks": 0, "visibleTicks": 1, "ticks": 3, "calls": 3,
    }


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


@pytest.mark.parametrize("mode", ["training", "auto"])
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

    # Training and AUTO are the only modes exposed by the Hub.
    assert modes == {"hinge": {"training": True, "auto": False}}
    assert reasons["hinge"]["training"] is None
    assert "does not bind this calibration" in reasons["hinge"]["auto"]
    assert set(reasons["hinge"]) == {"training", "auto"}
    assert set(pending) == {"bumble"}
    assert "not calibrated" in pending["bumble"]["reason"]


def test_hub_config_defaults_uses_one_validating_startup_gate_per_mode(monkeypatch):
    """A config-level AUTO refusal must not fall through to a redundant registry probe."""
    from operation_love.hub import state as hub_state

    original = hub_state.platforms.unavailable_reason
    auto_reasons = []

    def record_reason(app, mode=None):
        reason = original(app, mode)
        if app == "hinge" and mode == "auto":
            auto_reasons.append(reason)
        return reason

    monkeypatch.setattr(hub_state.platforms, "unavailable_reason", record_reason)
    defaults = HubState("config.yaml").config_defaults()

    hinge = next(p for kind in defaults["kinds"] for p in kind["platforms"]
                 if p["app"] == "hinge")
    assert hinge["modes"]["auto"] is False
    assert "does not bind this calibration" in hinge["mode_reasons"]["auto"]
    assert auto_reasons == []


def test_hub_config_defaults_marks_per_app_mode_override_as_unavailable(monkeypatch):
    """The picker must not advertise Auto when Start will reject an app's Training override."""
    from types import SimpleNamespace
    from operation_love.hub import state as hub_state

    platform = SimpleNamespace(app="hinge", label="Hinge", kind="android",
                               available=True, reason=None)
    cfg = SimpleNamespace(enabled_apps=["hinge"], mode="auto",
                          storage=SimpleNamespace(backend="sqlite"))
    effective = SimpleNamespace(mode="auto", apps={"hinge": {"mode": "training"}})
    monkeypatch.setattr(hub_state.cfg_mod, "load", lambda _path: cfg)
    monkeypatch.setattr(hub_state.supervisor, "load_effective_config",
                        lambda *args, **kwargs: effective)
    monkeypatch.setattr(hub_state.platforms, "kinds", lambda: ("android",))
    monkeypatch.setattr(hub_state.platforms, "for_kind", lambda _kind: (platform,))
    monkeypatch.setattr(hub_state.platforms, "all_platforms", lambda: (platform,))
    monkeypatch.setattr(hub_state.platforms, "get", lambda _app: platform)

    defaults = HubState("config.yaml").config_defaults()
    hinge = defaults["kinds"][0]["platforms"][0]
    assert hinge["modes"] == {"training": True, "auto": False}
    assert "per-app mode override" in hinge["mode_reasons"]["auto"]


def test_hubstate_start_rejects_two_android_platforms_together(monkeypatch):
    import operation_love.hub as hub

    monkeypatch.setattr(hub.supervisor, "run", lambda config_path, **kw: None)
    st = HubState("config.yaml")
    ok, msg = st.start(apps=["hinge", "bumble"])
    assert ok is False
    assert st.is_running() is False


def test_hubstate_rejects_config_blocked_hinge_auto_before_background_work(monkeypatch, tmp_path):
    """Hub and direct supervisor starts return the same release-gate message, and the Hub
    rejects before allocating any run thread or timed-stop timer."""
    import operation_love.hub as hub

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(re.sub(
        r"(?m)^    observe_release_evidence:\n(?:^      [^\n]*\n)+", "",
        Path("config.yaml").read_text(), count=1))
    with pytest.raises(ValueError) as direct:
        hub.supervisor.run(str(cfg_path), mode="auto", enabled_apps=["hinge"])

    run_calls = []
    monkeypatch.setattr(
        hub.supervisor, "run", lambda *args, **kwargs: run_calls.append((args, kwargs)))
    st = HubState(str(cfg_path))
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
        # Keep this line-oriented: DOTALL would let ``.*`` swallow the subsequent storage
        # section and turn this release-gate test into an unrelated storage-config failure.
        r"(?m)^    observe_release_evidence:\n(?:^      [^\n]*\n)+", "",
        Path("config.yaml").read_text(), count=1)
    # The shipped file intentionally starts in Training while its 10.3.0 production-observe
    # release is pending (config.yaml's targeting_calibration pins hinge_version_name 10.3.0;
    # this comment said 10.1.0 long after that moved on). Exercise this test's AUTO release gate
    # explicitly.
    cfg_text = cfg_text.replace("mode: training", "mode: auto", 1)
    cfg_path.write_text(cfg_text)

    st = HubState(str(cfg_path))
    ok, msg = st.start(apps=None, stop_after_seconds=60)

    assert ok is False
    assert msg.startswith("Config: Hinge AUTO is blocked")
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
                     "index": 0, "referenced": "the beach photo",
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


def _runstatus_script(snap: dict) -> str:
    """Run the one durable run-status banner against a minimal DOM.

    The Hub must never reconstruct one app's state in two different banners: a training
    run has its separate actionable review panel, but exactly one non-action status surface.
    """
    fns = (_extract_js_function(_PAGE, "escHtml") + "\n"
           + _extract_js_function(_PAGE, "selectRunApps") + "\n"
           + _extract_js_function(_PAGE, "renderRunStatus"))
    return (
        "let el = {style:{display:''}, innerHTML:''};\n"
        "function $(sel){ return sel === '#runbanner' ? el : null; }\n"
        + fns + "\n"
        "renderRunStatus(" + json.dumps(snap) + ");\n"
        "console.log(JSON.stringify({display: el.style.display, html: el.innerHTML}));\n"
    )


def test_run_status_renders_live_training_once_with_checkpoint_detail_and_swipes():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {"running": True, "status": {"mode": "auto", "apps": {
        "hinge": {"app": "hinge", "mode": "training", "state": "scoring",
                  "swipes_run": 2, "detail": "verifying photo item 2 of 3"},
    }}}
    result = _run_node(_runstatus_script(snap))
    assert result["display"] == "block"
    assert "preparing the training checkpoint" in result["html"]
    assert "verifying photo item 2 of 3" in result["html"]
    assert "2 swipes this run" in result["html"]
    assert result["html"].count("hinge") == 1
    assert "tap X" not in result["html"] and "Send Like" not in result["html"]


@pytest.mark.parametrize(("state", "expected"), [
    ("scoring", "preparing the training checkpoint"),
    ("waiting_approval", "waiting for your Hub decision"),
    ("acting", "carrying out your hinge decision"),
    ("starting", "starting hinge"),
])
def test_run_status_maps_live_training_producer_states_to_one_clear_cue(state, expected):
    """These are the states the training worker actually publishes."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {"running": True, "status": {"apps": {
        "hinge": {"app": "hinge", "mode": "training", "state": state, "swipes_run": 1},
    }}}
    html = _run_node(_runstatus_script(snap))["html"]
    assert expected in html
    assert html.count("hinge") == 1


def test_run_status_tells_the_two_scoring_publishes_apart():
    """`scoring` is published for two different things and must not be titled as one.

    The worker republishes state="scoring" AFTER a decision is durably saved, purely to stop
    claiming a device action is in flight during the pacing / session-break idle window. Titling
    that "preparing the training checkpoint" promises the operator a review that is not coming,
    on a profile that is already finished.
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    prep = {"running": True, "status": {"apps": {
        "hinge": {"app": "hinge", "mode": "training", "state": "scoring",
                  "detail": "verifying photo item 2 of 3"},
    }}}
    prep_html = _run_node(_runstatus_script(prep))["html"]
    assert "preparing the training checkpoint" in prep_html
    assert "decision saved" not in prep_html

    saved = {"running": True, "status": {"apps": {
        "hinge": {"app": "hinge", "mode": "training", "state": "scoring",
                  "detail": "Like recorded and saved; pacing before the next profile"},
    }}}
    saved_html = _run_node(_runstatus_script(saved))["html"]
    # The title is the discriminator: the worker's detail already contains the word "pacing",
    # so asserting on that alone could not tell the title apart from its own fine print.
    assert "decision saved — pacing before the next profile" in saved_html
    assert "preparing the training checkpoint" not in saved_html
    # Detail survives as fine print, and this stays a red-circle WAIT box -- a finished profile
    # in its pacing window is still not a moment to act, so it must not restyle toward the GO cue.
    assert "Like recorded and saved; pacing before the next profile" in saved_html
    assert "🔴" in saved_html
    assert "background:#3a2f12" in saved_html          # css.wait
    assert "background:#123a23" not in saved_html      # never css.go


def test_run_status_explains_when_an_in_flight_training_decision_is_counted():
    """The phone may advance before confirmation/persistence completes.

    Keep the durable count and the in-flight action visibly distinct so the operator does not
    mistake the current profile for a stale status from the prior one.
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {"running": True, "status": {"apps": {
        "hinge": {"app": "hinge", "mode": "training", "state": "acting",
                  "swipes_run": 2},
    }}}

    html = _run_node(_runstatus_script(snap))["html"]
    assert "carrying out your hinge decision" in html
    assert "completed-swipe count updates after the action is confirmed and saved" in html
    assert "2 swipes this run" in html
    assert "recording your hinge decision" not in html


def test_run_status_shows_landed_training_persistence_detail_in_the_acting_banner():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {"running": True, "status": {"apps": {
        "hinge": {"app": "hinge", "mode": "training", "state": "acting",
                  "detail": ("Like landed in Hinge; archive complete—flushing the label "
                             "and evidence to storage")},
    }}}

    html = _run_node(_runstatus_script(snap))["html"]
    assert "decision landed — saving training data" in html
    assert "carrying out your hinge decision" not in html
    assert "Like landed in Hinge; archive complete—flushing the label and evidence to storage" in html
    assert "completed-swipe count updates after the action is confirmed and saved" not in html


def test_run_status_escapes_live_training_app():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    hostile_app = '<img src=x onerror="alert(1)">'
    app_snap = {"running": True, "status": {"apps": {
        "hostile": {"app": hostile_app, "mode": "training", "state": "scoring"},
    }}}
    app_html = _run_node(_runstatus_script(app_snap))["html"]
    assert "<img" not in app_html and "&lt;img" in app_html

def test_run_status_hides_without_a_supported_app():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {"running": True, "status": {"apps": {
        "hinge": {"app": "hinge", "mode": "unsupported", "state": "waiting"},
    }}}
    assert _run_node(_runstatus_script(snap))["display"] == "none"


def test_run_status_retains_training_terminal_reason_after_the_run_ends():
    """A completed training run has no second banner to fall back to.

    The review panel is intentionally transient, but the durable run banner must keep a
    terminal error/reason visible after HubState's thread has finished.
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {
        "running": False,
        "status": {"apps": {"hinge": {
            "app": "hinge", "mode": "training", "state": "stopped",
            "stop_reason": "Training requires a complete generated opener",
            "swipes_run": 3,
        }}},
    }
    result = _run_node(_runstatus_script(snap))
    assert result["display"] == "block"
    assert "Training requires a complete generated opener" in result["html"]


def test_run_status_suppresses_training_decision_cues_while_stopping():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    snap = {
        "running": True,
        "status": {"stopping": True, "apps": {"hinge": {
            "app": "hinge", "mode": "training", "state": "waiting", "swipes_run": 1,
        }}},
    }
    html = _run_node(_runstatus_script(snap))["html"]
    assert "stopping" in html
    assert "preparing the checkpoint" not in html
    assert "running" not in html


def test_select_run_apps_filters_to_each_supported_mode_once():
    # An app's OWN mode decides whether it belongs in the one run banner, not the global
    # run mode, since apps.<app>.mode can override it.
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fn = _extract_js_function(_PAGE, "selectRunApps")
    status = {
        "mode": "training",
        "apps": {
            "bumble": {"mode": "auto", "state": "error", "app": "bumble"},
            "training": {"mode": "training", "state": "waiting_training_decision", "app": "training"},
            "retired": {"mode": "unsupported", "state": "waiting", "app": "retired"},
        },
    }
    script = (
        fn + "\n"
        "const status = " + json.dumps(status) + ";\n"
        "console.log(JSON.stringify(selectRunApps(status).map(a => a.app)));\n"
    )
    assert _run_node(script) == ["bumble", "training"]

    # Retired modes stay out even if the global mode would otherwise look runnable.
    status_b = {"mode": "auto", "apps": {"hinge": {"mode": "unsupported", "state": "waiting", "app": "hinge"}}}
    script_b = (
        fn + "\n"
        "const status = " + json.dumps(status_b) + ";\n"
        "console.log(JSON.stringify(selectRunApps(status).map(a => a.app)));\n"
    )
    assert _run_node(script_b) == []


def test_render_run_status_shows_auto_error_box_with_worker_message():
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
    result = _run_node(_runstatus_script(snap))
    assert result["display"] == "block"
    assert "hinge" in result["html"]
    assert "UnlocatedControlError: unrecognized screen" in result["html"]


def test_render_run_status_escapes_auto_error_and_app_before_using_inner_html():
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
    result = _run_node(_runstatus_script(snap))
    assert "<img" not in result["html"] and "<script" not in result["html"]
    assert "&lt;img" in result["html"] and "&lt;script&gt;" in result["html"]


def test_render_run_status_shows_auto_cold_start_defer_message():
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
    result = _run_node(_runstatus_script(snap))
    assert result["display"] == "block"
    assert "cold-start" in result["html"]


def test_render_run_status_shows_auto_opener_exhaustion_stop_reason():
    # WS-opener-reason: an auto-mode app halted because OpenerService ran out of opener
    # capacity (budget/credit/provider failure) used to render IDENTICALLY to a plain
    # operator-clicked Stop -- both were just state='stopped' with no reason field. The
    # banner must show the specific cause (AppStatus.stop_reason), the same way it already
    # shows `.error` for the exception path above.
    #
    # stop_kind="opener" is required here since the 2026-08-11 blocked-deck addition gave
    # stop_reason a SECOND possible cause (worker.py's blocked-deck check) with its own
    # stop_kind: hub.html now branches the safe-opener wording specifically on
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
    result = _run_node(_runstatus_script(snap))
    assert result["display"] == "block"
    assert "run budget reached" in result["html"]
    assert "could not prepare a safe opener" in result["html"]
    # The reason takes over the box's sub-line instead of the ordinary swipe count.
    assert "4 swipes this run" not in result["html"]


def test_render_run_status_reads_an_auto_targeting_stop_as_one_not_as_opener_capacity():
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
    result = _run_node(_runstatus_script(snap))
    assert result["display"] == "block"
    html = result["html"]

    assert "could not attach the generated opener to its selected item" in html
    assert "could not prepare a safe opener" not in html
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
    other = _run_node(_runstatus_script(capacity))["html"]
    assert "could not prepare a safe opener" in other
    assert "could not attach the generated opener" not in other


def test_render_run_status_names_pre_opener_targeting_calibration_stop():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    reason = ("Live Hinge 10.1.0 does not match the installed 10.0.1 calibration. "
              "No opener, action, or label was produced.")
    snap = {
        "running": True,
        "status": {"apps": {"hinge": {
            "app": "hinge", "mode": "training", "state": "stopped",
            "stop_reason": reason, "stop_kind": "targeting_calibration", "swipes_run": 0,
        }}},
    }

    html = _run_node(_runstatus_script(snap))["html"]
    assert "targeting calibration must be renewed" in html
    assert reason in html
    assert "could not prepare a safe opener" not in html
    assert "could not attach the generated opener" not in html


def test_render_run_status_headlines_a_failed_cold_relaunch_not_a_blocked_deck():
    """A latched blocked REASON is not proof that the DECK is what is blocked.

    The worker has exactly one channel for a sentence coming out of a capture that returned
    None, and a failed cold-relaunch recovery latches through it too, so the Hub used to
    headline "stopped -- deck blocked" for a run where nothing was blocking the deck: the
    operator was sent to look for a paywall that was never there. Owner rule: the hub must
    clearly show why a run stopped.
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    reason = ("Hinge was relaunched cold and the deck could not be proven re-entered; "
              "no action or label was recorded.")
    snap = {
        "running": True,
        "status": {"apps": {"hinge": {
            "app": "hinge", "mode": "auto", "state": "blocked",
            "stop_reason": reason, "stop_kind": "cold_relaunch_recovery", "swipes_run": 3,
        }}},
    }

    html = _run_node(_runstatus_script(snap))["html"]
    assert "could not recover after the app relaunched" in html
    assert "deck blocked" not in html
    assert reason in html                          # the driver's own sentence stays underneath
    assert "🔴" not in html                         # a graceful stop, not a crash cue

    # The branch it was split OUT of is untouched: a genuine paywall still reads as one, and so
    # does a blocked stop from a driver that never classified its reason at all.
    for unclassified in ({"stop_kind": "deck_blocked"}, {}):
        other = _run_node(_runstatus_script({
            "running": True,
            "status": {"apps": {"hinge": {
                "app": "hinge", "mode": "auto", "state": "blocked",
                "stop_reason": "Hinge is showing its out-of-likes upgrade screen.",
                **unclassified,
            }}},
        }))["html"]
        assert "deck blocked" in other
        assert "could not recover after the app relaunched" not in other


def test_render_run_status_escapes_auto_stop_reason_before_using_inner_html():
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
    result = _run_node(_runstatus_script(snap))
    assert "<script" not in result["html"]
    assert "&lt;script&gt;" in result["html"]


def test_render_run_status_shows_a_bare_auto_stop_without_a_reason():
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
    result = _run_node(_runstatus_script(snap))
    assert result["display"] == "block"
    assert "bumble: stopped</div>" in result["html"]


def test_render_run_status_survives_auto_run_completion():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    # A completed auto run must retain its terminal explanation. HubState.running follows
    # Thread.is_alive(), so this is the state a person returning after the run sees.
    snap_not_running = {
        "running": False,
        "status": {"apps": {"hinge": {"app": "hinge", "mode": "auto", "state": "error",
                                           "error": "unrecognized screen"}}},
    }
    result_b = _run_node(_runstatus_script(snap_not_running))
    assert result_b["display"] == "block"
    assert "unrecognized screen" in result_b["html"]


def test_render_run_status_does_not_claim_auto_running_after_the_run_ended():
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
    result = _run_node(_runstatus_script(snap))
    assert result["display"] == "block"
    assert "running" not in result["html"]        # never claim a live worker after the run ended
    assert "ended" in result["html"]              # ...say the run ended instead
    assert "capturing" in result["html"]          # ...and surface the abnormal last state
    assert "7 swipes this run" in result["html"]

    # Same non-terminal state while the run IS live still reads as running.
    live = {"running": True, "status": {"apps": {"hinge": dict(snap["status"]["apps"]["hinge"])}}}
    assert "hinge: running" in _run_node(_runstatus_script(live))["html"]


def test_render_run_status_shows_auto_stopping_instead_of_running_during_the_shutdown_tail():
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
    result = _run_node(_runstatus_script(snap))
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
    assert "hinge: running" in _run_node(_runstatus_script(not_stopping))["html"]


def test_tick_drives_one_real_run_status_banner_and_the_separate_training_checkpoint_renderer():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    assert len(re.findall(r'<div\s+id="runbanner"(?:\s|>)', _PAGE)) == 1
    assert "id=\"trainingbanner\"" not in _PAGE
    assert "id=\"autobanner\"" not in _PAGE
    assert re.search(r'<aside\s+id="trainingpanel"(?:\s|>)', _PAGE)
    tick_fn = _extract_js_function(_PAGE, "tick")
    script = (
        "const calls = []; let _statusRequest = 0;\n"
        "async function getJSON(){ return {running:false,status:{apps:{}}}; }\n"
        "function renderGlobal(){ calls.push('global'); }\n"
        "function renderRunStatus(){ calls.push('run-status'); }\n"
        "function tickTrainingCheckpoint(){ calls.push('training'); }\n"
        "function clearHubConnectionLost(){}\n"
        "function renderHubConnectionLost(){}\n"
        + tick_fn + "\n"
        "tick().then(() => console.log(JSON.stringify(calls)));\n"
    )
    assert _run_node(script) == ["global", "run-status", "training"]


def test_tick_discards_an_out_of_order_status_response():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    tick_fn = _extract_js_function(_PAGE, "tick")
    script = (
        "const renders=[]; const resolvers=[]; let _statusRequest=0;\n"
        "function getJSON(){return new Promise(resolve=>resolvers.push(resolve));}\n"
        "function renderGlobal(s){renders.push(['global',s.version]);}\n"
        "function renderRunStatus(s){renders.push(['run',s.version]);}\n"
        "function tickTrainingCheckpoint(s){renders.push(['training',s.version]);}\n"
        "function clearHubConnectionLost(){}\n"
        "function renderHubConnectionLost(){}\n"
        + tick_fn + "\n"
        "(async()=>{const first=tick(), second=tick(); resolvers[1]({version:'new'}); await second; "
        "resolvers[0]({version:'old'}); await first; console.log(JSON.stringify(renders));})();\n"
    )
    assert _run_node(script) == [
        ["global", "new"], ["run", "new"], ["training", "new"],
    ]


def _training_panel_script(checkpoint, alert_permission="granted", running=False):
    """Run the real renderer against the smallest DOM surface it needs.

    `alert_permission` / `running` drive the "Allow alerts" affordance only. Their defaults
    (a granted permission, no run in progress) are the combination that renders nothing extra,
    so every longstanding markup test below keeps asserting on exactly the card it always did.
    """
    return (
        "const panel={style:{display:''},innerHTML:''};\n"
        "const buttons={};\n"
        "const layout={active:false,classList:{toggle(_name,value){layout.active=!!value;}}};\n"
        "const document={querySelector:(selector)=>selector==='.hub-layout'?layout:null};\n"
        "function $(selector){ return selector==='#trainingpanel' ? panel : buttons[selector]; }\n"
        "let _trainingCheckpoint=null, _trainingActionBusy=false, _trainingBusyKey='', _trainingBusyRequest=0, _trainingImageKey='', _trainingImageIndex=0, _trainingImageFailedIndex=null,_trainingImageRenderGeneration=0,_trainingImageFailedGeneration=null; const _trainingIdempotency=new Map();\n"
        # The permission gate the "Allow alerts" fine print is derived from. `window` is absent
        # in node, and _wasRunning is renderGlobal's cross-poll run flag -- both are read for
        # real here rather than stubbed, so the affordance's two preconditions are exercised.
        + _notification_window_literal(alert_permission) +
        "let _wasRunning=" + ("true" if running else "false") + ";\n"
        # The real state synchronizer closes a modal when the checkpoint changes.  This
        # renderer-only harness has no persistent modal, so its inert stand-in keeps these
        # longstanding markup tests deliberately scoped to the panel itself.
        "function setTrainingImageZoom(){}\n"
        + _extract_js_function(_PAGE, "trainingAlertPermission") + "\n"
        + _extract_js_function(_PAGE, "shouldOfferTrainingAlertGesture") + "\n"
        + _extract_js_function(_PAGE, "trainingAlertGestureOffered") + "\n"
        + _extract_js_function(_PAGE, "escHtml") + "\n"
        + _extract_js_function(_PAGE, "safeCheckpointImageDataUrl") + "\n"
        + _extract_js_function(_PAGE, "trainingCheckpointKey") + "\n"
        + _extract_js_function(_PAGE, "resetTrainingIdempotencyIfCardChanged") + "\n"
        + _extract_js_function(_PAGE, "resetTrainingBusyIfCardChanged") + "\n"
        + _extract_js_function(_PAGE, "trainingActionBusyFor") + "\n"
        + _extract_js_function(_PAGE, "checkpointReviewImages") + "\n"
        + _extract_js_function(_PAGE, "syncTrainingImageState") + "\n"
        + _extract_js_function(_PAGE, "renderTrainingCheckpoint") + "\n"
        + "renderTrainingCheckpoint(" + json.dumps(checkpoint) + ");\n"
        + "console.log(JSON.stringify({display:panel.style.display,html:panel.innerHTML,active:layout.active}));\n"
    )


def test_training_panel_is_a_separate_responsive_right_column():
    assert 'class="hub-layout"' in _PAGE
    assert '.hub-layout { max-width:680px; margin:0 auto; }' in _PAGE
    assert '.hub-layout.training-active { max-width:1120px; display:grid;' in _PAGE
    assert 'grid-template-columns:minmax(0,680px) minmax(320px,420px)' in _PAGE
    assert '@media (max-width:900px)' in _PAGE
    assert re.search(r'<aside\s+id="trainingpanel"[^>]*aria-live="polite"', _PAGE)
    mode = re.search(r'<select\b[^>]*\bid="mode"[^>]*>(.*?)</select>', _PAGE, re.S)
    assert mode and 'value="training"' in mode.group(1) and 'auto_testing' not in mode.group(1)


def test_training_panel_shows_full_opener_target_image_and_escapes_text():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    checkpoint = {
        "run_id": "run-1", "app": "hinge", "profile_token": "profile-1",
        "approval_token": "approval-1", "phase": "waiting_training_decision", "pending": True,
        "action": "ready", "item": 3, "item_description": '<img src=x onerror="boom">',
        "referenced": "your <great> travel photo", "opener": "Line one\nLine <two>",
        "image_data_url": "data:image/png;base64,AA==",
    }
    result = _run_node(_training_panel_script(checkpoint))
    assert result["display"] == "block"
    assert result["active"] is True
    assert 'src="data:image/png;base64,AA=="' in result["html"]
    assert "Line one\nLine &lt;two&gt;" in result["html"]
    assert "&lt;img" in result["html"] and '<img src=x' not in result["html"]
    assert "your &lt;great&gt; travel photo" in result["html"]
    assert "already written on this target" in result["html"]
    assert '>Like</button>' in result["html"]
    assert '>Dislike</button>' in result["html"]


def _training_checkpoint_with(**extra):
    return {
        "run_id": "run-1", "app": "hinge", "profile_token": "profile-1",
        "approval_token": "approval-1", "phase": "waiting_training_decision", "pending": True,
        "action": "ready", "item": 3, "item_description": "the surfing photo",
        "opener": "Complete opener", "image_data_url": "data:image/png;base64,AA==",
        **extra,
    }


def test_training_panel_prints_the_media_ordinal_beside_the_item_description():
    """The reviewer is holding the phone and can only count what Hinge drew, so the target line
    also names the target's position over photos and videos. Fine print on the existing line:
    it is not a control, not a decision gate, and adds no chrome of its own."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")

    result = _run_node(_training_panel_script(
        _training_checkpoint_with(item_media_ordinal=2)))

    html = result["html"]
    assert "item 3 — the surfing photo · photo/video 2 counting from the top" in html
    # It rides the existing target `.meta` line -- no new element, no new class, no styling.
    assert 'target · item 3 — the surfing photo · photo/video 2' in html
    assert html.count('class="meta"') == _run_node(_training_panel_script(
        _training_checkpoint_with()))["html"].count('class="meta"')


@pytest.mark.parametrize("extra", [
    {},                              # the field the worker omits when nothing could be counted
    {"item_media_ordinal": None},
    {"item_media_ordinal": 0},
    {"item_media_ordinal": -1},
    {"item_media_ordinal": "2"},     # a string is never rendered as a count
    # A float is deliberately NOT in this list: JSON and JavaScript cannot tell 2.0 from 2, so
    # `Number.isInteger` is not the layer that can reject one. `publish_checkpoint` is, and
    # tests/test_training_actions.py holds it to that -- a non-`int` never reaches the wire.
])
def test_training_panel_without_a_media_ordinal_renders_exactly_as_it_did_before(extra):
    """An absent or unusable hint must leave the card byte-identical to today's, including the
    GO/WAIT wording and the decision controls: it is an affordance, never a capability."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")

    result = _run_node(_training_panel_script(_training_checkpoint_with(**extra)))

    assert "photo/video" not in result["html"]
    assert "target · item 3 — the surfing photo</div>" in result["html"]
    assert result["html"] == _run_node(
        _training_panel_script(_training_checkpoint_with()))["html"]


def test_training_panel_embeds_vertical_navigation_in_order_and_starts_on_target():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    checkpoint = {
        "run_id": "run-1", "app": "hinge", "profile_token": "profile-1",
        "approval_token": "approval-1", "phase": "waiting_training_decision", "pending": True,
        "action": "ready", "item": 3, "opener": "Complete opener",
        "image_data_url": "data:image/png;base64,AA==", "profile_image_count": 3,
    }

    result = _run_node(_training_panel_script(checkpoint))
    html = result["html"]
    assert 'id="trainingimageup"' in html and 'scroll up' in html
    assert 'id="trainingimagedown"' in html and 'scroll down' in html
    assert "target view · 4/4" in html
    assert re.search(r'id="trainingimagedown"[^>]* disabled', html)
    assert not re.search(r'id="trainingimageup"[^>]* disabled', html)


def test_hidden_training_image_error_overrides_its_normal_flow_display_rule():
    """An empty, hidden error must not expand the image wrap and displace its controls."""
    normal_flow = re.search(
        r"\.training-image-wrap\s+\.training-image-error\s*\{([^}]*)\}", _PAGE)
    hidden = re.search(r"\.training-image-error\[hidden\]\s*\{([^}]*)\}", _PAGE)

    assert normal_flow and re.search(r"\bdisplay\s*:\s*block\b", normal_flow.group(1))
    assert hidden and re.search(r"\bdisplay\s*:\s*none\s*!important\b", hidden.group(1))
    assert _PAGE.count('class="training-image-error" hidden') == 2


def test_training_image_navigation_uses_top_to_bottom_profile_snapshot_order():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    functions = "\n".join((
        _extract_js_function(_PAGE, "safeCheckpointImageDataUrl"),
        _extract_js_function(_PAGE, "trainingCheckpointKey"),
        _extract_js_function(_PAGE, "checkpointReviewImages"),
        _extract_js_function(_PAGE, "showTrainingImage"),
        _extract_js_function(_PAGE, "moveTrainingImage"),
    ))
    script = (
        "const elements={}; function $(selector){return elements[selector]||(elements[selector]={setAttribute(){}});}\n"
        "let _trainingImageIndex=3,_trainingImageFailedIndex=null,_trainingImageRenderGeneration=0,_trainingImageFailedGeneration=null,_trainingImageZoomed=false; const _trainingCheckpoint={run_id:'run-1',app:'hinge',"
        "profile_token:'profile-1',approval_token:'approval-1',profile_image_count:3,"
        "image_data_url:'data:image/png;base64,AA=='};\n"
        + functions + "\nmoveTrainingImage(-1);\n"
        "const first={index:_trainingImageIndex,src:elements['#trainingimage'].src,"
        "label:elements['#trainingimageposition'].textContent};\n"
        "moveTrainingImage(-1);\n"
        "console.log(JSON.stringify({first,second:{index:_trainingImageIndex,"
        "src:elements['#trainingimage'].src,label:elements['#trainingimageposition'].textContent}}));\n"
    )
    result = _run_node(script)
    assert result["first"]["index"] == 2
    assert "index=2" in result["first"]["src"]
    assert result["first"]["label"] == "profile snapshot 3 of 3 · 3/4"
    assert result["second"]["index"] == 1
    assert "index=1" in result["second"]["src"]


def test_training_preview_up_click_and_failed_or_missing_snapshot_are_recoverable():
    """The actual rendered Up control must not let a broken review frame take down Hub.

    This covers both the normal click binding (rather than a direct helper call) and the two
    defensive states that can occur when a local response/image no longer matches the card.
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    functions = "\n".join((
        _extract_js_function(_PAGE, "escHtml"),
        _extract_js_function(_PAGE, "safeCheckpointImageDataUrl"),
        _extract_js_function(_PAGE, "trainingCheckpointKey"),
        _extract_js_function(_PAGE, "resetTrainingIdempotencyIfCardChanged"),
        _extract_js_function(_PAGE, "resetTrainingBusyIfCardChanged"),
        _extract_js_function(_PAGE, "trainingActionBusyFor"),
        _extract_js_function(_PAGE, "checkpointReviewImages"),
        _extract_js_function(_PAGE, "syncTrainingImageState"),
        _extract_js_function(_PAGE, "setTrainingImageNavigationDisabled"),
        _extract_js_function(_PAGE, "reportTrainingImageLoadError"),
        _extract_js_function(_PAGE, "showTrainingImage"),
        _extract_js_function(_PAGE, "moveTrainingImage"),
        _extract_js_function(_PAGE, "renderTrainingCheckpoint"),
    ))
    checkpoint = {
        "run_id": "run-1", "app": "hinge", "profile_token": "profile-1",
        "approval_token": "approval-1", "phase": "waiting_training_decision",
        "pending": True, "action": "ready", "item": 3, "opener": "Complete opener",
        "image_data_url": "data:image/png;base64,AA==", "profile_image_count": 3,
    }
    script = (
        "function node(){return {hidden:false,disabled:false,src:'',alt:'',textContent:'',attrs:{},"
        "setAttribute(k,v){this.attrs[k]=String(v);},removeAttribute(k){delete this.attrs[k];if(k==='src')this.src='';}}}\n"
        "const elements={}; for(const id of ['#trainingimagezoomed','#trainingimagezoomposition',"
        "'#trainingimagezoomerror','#trainingimagezoomup','#trainingimagezoomdown'])elements[id]=node();\n"
        "const panel={style:{display:''},_html:'',get innerHTML(){return this._html;},set innerHTML(value){"
        "this._html=value; for(const id of ['#trainingimage','#trainingimageup','#trainingimagedown',"
        "'#trainingimagezoom','#trainingimageerror','#traininglike','#trainingdislike'])"
        "if(value.includes('id=\\\"'+id.slice(1)+'\\\"'))elements[id]=node();}}; elements['#trainingpanel']=panel;\n"
        "const layout={classList:{toggle(){}}}; const document={querySelector(s){return s==='.hub-layout'?layout:null;}};"
        "function $(selector){return elements[selector]||null;} function setTrainingImageZoom(){}\n"
        "let _trainingCheckpoint=null,_trainingActionBusy=false,_trainingBusyKey='',_trainingBusyRequest=0,"
        "_trainingImageKey='',_trainingImageIndex=0,_trainingImageFailedIndex=null,_trainingImageRenderGeneration=0,_trainingImageFailedGeneration=null,_trainingImageZoomed=false;"
        "const _trainingIdempotency=new Map();\n"
        + functions + "\n"
        "renderTrainingCheckpoint(" + json.dumps(checkpoint) + "); elements['#trainingimageup'].onclick();"
        "const afterUp={index:_trainingImageIndex,src:elements['#trainingimage'].src};"
        "elements['#trainingimage'].onerror(); const afterError={index:_trainingImageIndex,"
        "up:elements['#trainingimageup'].disabled,down:elements['#trainingimagedown'].disabled,"
        "like:elements['#traininglike'].disabled,dislike:elements['#trainingdislike'].disabled,"
        "message:elements['#trainingimageerror'].textContent};\n"
        "checkpointReviewImages=()=>[]; _trainingImageFailedIndex=null; _trainingImageRenderGeneration=0; _trainingImageFailedGeneration=null; renderTrainingCheckpoint(" + json.dumps(checkpoint) + ");"
        "console.log(JSON.stringify({afterUp,afterError,fallback:panel.innerHTML}));\n"
    )
    result = _run_node(script)

    assert result["afterUp"]["index"] == 2
    assert "index=2" in result["afterUp"]["src"]
    assert result["afterError"]["index"] == 2
    assert result["afterError"]["up"] is True and result["afterError"]["down"] is True
    assert result["afterError"]["like"] is True and result["afterError"]["dislike"] is True
    assert "could not be loaded" in result["afterError"]["message"]
    assert "Profile review image was unavailable" in result["fallback"]
    assert 'id="trainingimage"' not in result["fallback"]


def test_training_image_load_error_clears_once_the_same_frame_loads_successfully():
    """A TRANSIENT /api/training/image failure must not become a permanent lockout.

    Every poll tick rebuilds #trainingimage with the same src, so the browser keeps retrying
    on its own; once one of those retries actually succeeds, onload must undo exactly what
    onerror latched -- without ever flipping Like/Dislike on directly (that stays owned by
    the normal ready/opener/action-boundary gate re-run inside renderTrainingCheckpoint).
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    functions = "\n".join((
        _extract_js_function(_PAGE, "escHtml"),
        _extract_js_function(_PAGE, "safeCheckpointImageDataUrl"),
        _extract_js_function(_PAGE, "trainingCheckpointKey"),
        _extract_js_function(_PAGE, "resetTrainingIdempotencyIfCardChanged"),
        _extract_js_function(_PAGE, "resetTrainingBusyIfCardChanged"),
        _extract_js_function(_PAGE, "trainingActionBusyFor"),
        _extract_js_function(_PAGE, "checkpointReviewImages"),
        _extract_js_function(_PAGE, "syncTrainingImageState"),
        _extract_js_function(_PAGE, "setTrainingImageNavigationDisabled"),
        _extract_js_function(_PAGE, "reportTrainingImageLoadError"),
        _extract_js_function(_PAGE, "clearTrainingImageLoadError"),
        _extract_js_function(_PAGE, "showTrainingImage"),
        _extract_js_function(_PAGE, "moveTrainingImage"),
        _extract_js_function(_PAGE, "renderTrainingCheckpoint"),
    ))
    checkpoint = {
        "run_id": "run-1", "app": "hinge", "profile_token": "profile-1",
        "approval_token": "approval-1", "phase": "waiting_training_decision",
        "pending": True, "action": "ready", "item": 3, "opener": "Complete opener",
        "image_data_url": "data:image/png;base64,AA==", "profile_image_count": 3,
    }
    script = (
        "function node(){return {hidden:false,disabled:false,src:'',alt:'',textContent:'',attrs:{},"
        "setAttribute(k,v){this.attrs[k]=String(v);},removeAttribute(k){delete this.attrs[k];if(k==='src')this.src='';}}}\n"
        "const elements={}; for(const id of ['#trainingimagezoomed','#trainingimagezoomposition',"
        "'#trainingimagezoomerror','#trainingimagezoomup','#trainingimagezoomdown'])elements[id]=node();\n"
        "const panel={style:{display:''},_html:'',get innerHTML(){return this._html;},set innerHTML(value){"
        "this._html=value; for(const id of ['#trainingimage','#trainingimageup','#trainingimagedown',"
        "'#trainingimagezoom','#trainingimageerror','#traininglike','#trainingdislike'])"
        "if(value.includes('id=\\\"'+id.slice(1)+'\\\"'))elements[id]=node();}}; elements['#trainingpanel']=panel;\n"
        "const layout={classList:{toggle(){}}}; const document={querySelector(s){return s==='.hub-layout'?layout:null;}};"
        "function $(selector){return elements[selector]||null;} function setTrainingImageZoom(){}\n"
        "let _trainingCheckpoint=null,_trainingActionBusy=false,_trainingBusyKey='',_trainingBusyRequest=0,"
        "_trainingImageKey='',_trainingImageIndex=0,_trainingImageFailedIndex=null,_trainingImageRenderGeneration=0,_trainingImageFailedGeneration=null,_trainingImageZoomed=false;"
        "const _trainingIdempotency=new Map();\n"
        + functions + "\n"
        "renderTrainingCheckpoint(" + json.dumps(checkpoint) + ");\n"
        "const beforeError={like:elements['#traininglike'].disabled,dislike:elements['#trainingdislike'].disabled,"
        "image:elements['#trainingimage']};\n"
        "const failingImage=elements['#trainingimage']; failingImage.onerror();\n"
        "const afterError={like:elements['#traininglike'].disabled,dislike:elements['#trainingdislike'].disabled,"
        "hidden:failingImage.hidden,up:elements['#trainingimageup'].disabled,"
        "zoom:elements['#trainingimagezoom'].disabled,message:elements['#trainingimageerror'].textContent};\n"
        # The identical element that reported onerror is what a later successful retry of the
        # same <img src> fires onload on -- no navigation and no fresh render happened between.
        "failingImage.onload();\n"
        "const recoveredImage=elements['#trainingimage'];\n"
        "const afterRecovery={like:elements['#traininglike'].disabled,dislike:elements['#trainingdislike'].disabled,"
        "hidden:recoveredImage.hidden,up:elements['#trainingimageup'].disabled,"
        "zoom:elements['#trainingimagezoom'].disabled,sameNode:recoveredImage===failingImage,"
        "html:panel.innerHTML};\n"
        "console.log(JSON.stringify({beforeError:{like:beforeError.like,dislike:beforeError.dislike},"
        "afterError,afterRecovery}));\n"
    )
    result = _run_node(script)

    assert result["beforeError"]["like"] is False and result["beforeError"]["dislike"] is False
    assert result["afterError"]["like"] is True and result["afterError"]["dislike"] is True
    assert result["afterError"]["hidden"] is True and result["afterError"]["up"] is True
    assert result["afterError"]["zoom"] is True
    assert "could not be loaded" in result["afterError"]["message"]
    # Recovery must go through a full re-render (a new #trainingimage node), so Like/Dislike
    # come back through the same canDecide gate a first render would have used -- never a
    # direct flip of the disabled property on the old, now-discarded element.
    assert result["afterRecovery"]["sameNode"] is False
    assert result["afterRecovery"]["like"] is False and result["afterRecovery"]["dislike"] is False
    assert result["afterRecovery"]["hidden"] is False and result["afterRecovery"]["up"] is False
    assert result["afterRecovery"]["zoom"] is False
    assert "Profile review image was unavailable" not in result["afterRecovery"]["html"]


def test_a_superseded_training_image_load_cannot_clear_a_live_failure_latch():
    """renderTrainingCheckpoint rebuilds the panel's innerHTML, so each render creates a NEW
    <img>. The previous one is detached but its in-flight request still fires onload/onerror
    at whatever moment the network returns. A stale success arriving after a fresh render must
    NOT unlock the checkpoint the live frame legitimately locked -- that would re-arm Like on a
    snapshot the reviewer cannot actually see, which is the exact failure the latch exists to
    prevent. Symmetrically, a stale FAILURE must not lock a frame that loaded fine."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    functions = "\n".join((
        _extract_js_function(_PAGE, "escHtml"),
        _extract_js_function(_PAGE, "safeCheckpointImageDataUrl"),
        _extract_js_function(_PAGE, "trainingCheckpointKey"),
        _extract_js_function(_PAGE, "resetTrainingIdempotencyIfCardChanged"),
        _extract_js_function(_PAGE, "resetTrainingBusyIfCardChanged"),
        _extract_js_function(_PAGE, "trainingActionBusyFor"),
        _extract_js_function(_PAGE, "checkpointReviewImages"),
        _extract_js_function(_PAGE, "syncTrainingImageState"),
        _extract_js_function(_PAGE, "setTrainingImageNavigationDisabled"),
        _extract_js_function(_PAGE, "reportTrainingImageLoadError"),
        _extract_js_function(_PAGE, "clearTrainingImageLoadError"),
        _extract_js_function(_PAGE, "showTrainingImage"),
        _extract_js_function(_PAGE, "moveTrainingImage"),
        _extract_js_function(_PAGE, "renderTrainingCheckpoint"),
    ))
    checkpoint = {
        "run_id": "run-1", "app": "hinge", "profile_token": "profile-1",
        "approval_token": "approval-1", "phase": "waiting_training_decision",
        "pending": True, "action": "ready", "item": 3, "opener": "Complete opener",
        "image_data_url": "data:image/png;base64,AA==", "profile_image_count": 3,
    }
    script = (
        "function node(){return {hidden:false,disabled:false,src:'',alt:'',textContent:'',attrs:{},"
        "setAttribute(k,v){this.attrs[k]=String(v);},removeAttribute(k){delete this.attrs[k];if(k==='src')this.src='';}}}\n"
        "const elements={}; for(const id of ['#trainingimagezoomed','#trainingimagezoomposition',"
        "'#trainingimagezoomerror','#trainingimagezoomup','#trainingimagezoomdown'])elements[id]=node();\n"
        "const panel={style:{display:''},_html:'',get innerHTML(){return this._html;},set innerHTML(value){"
        "this._html=value; for(const id of ['#trainingimage','#trainingimageup','#trainingimagedown',"
        "'#trainingimagezoom','#trainingimageerror','#traininglike','#trainingdislike'])"
        "if(value.includes('id=\\\"'+id.slice(1)+'\\\"'))elements[id]=node();}}; elements['#trainingpanel']=panel;\n"
        "const layout={classList:{toggle(){}}}; const document={querySelector(s){return s==='.hub-layout'?layout:null;}};"
        "function $(selector){return elements[selector]||null;} function setTrainingImageZoom(){}\n"
        "let _trainingCheckpoint=null,_trainingActionBusy=false,_trainingBusyKey='',_trainingBusyRequest=0,"
        "_trainingImageKey='',_trainingImageIndex=0,_trainingImageFailedIndex=null,"
        "_trainingImageRenderGeneration=0,_trainingImageFailedGeneration=null,_trainingImageZoomed=false;"
        "const _trainingIdempotency=new Map();\n"
        + functions + "\n"
        "renderTrainingCheckpoint(" + json.dumps(checkpoint) + ");\n"
        # Keep a handle on generation 1's <img>, then let an ordinary poll tick re-render.
        "const staleImage=elements['#trainingimage'];\n"
        "renderTrainingCheckpoint(" + json.dumps(checkpoint) + ");\n"
        "const liveImage=elements['#trainingimage'];\n"
        # The frame on screen NOW fails. That is the lockout the reviewer must see.
        "liveImage.onerror();\n"
        "const locked={like:elements['#traininglike'].disabled,"
        "dislike:elements['#trainingdislike'].disabled,"
        "message:elements['#trainingimageerror'].textContent};\n"
        # Generation 1's request finally succeeds, long after its element was discarded.
        "staleImage.onload();\n"
        "const afterStaleLoad={like:elements['#traininglike'].disabled,"
        "dislike:elements['#trainingdislike'].disabled,"
        "message:elements['#trainingimageerror'].textContent,"
        "sameNode:elements['#trainingimage']===liveImage};\n"
        # The live frame's OWN load succeeds -- its stamp matches, so this is the one event
        # allowed to clear the lockout. It re-renders, so a third generation's <img> appears.
        "liveImage.onload();\n"
        "const healthyImage=elements['#trainingimage'];\n"
        "const recovered={like:elements['#traininglike'].disabled,"
        "dislike:elements['#trainingdislike'].disabled,"
        "sameNode:healthyImage===liveImage};\n"
        # Mirror case: generation 2's element now errors, after generation 3 replaced it. A
        # superseded FAILURE must not lock the frame the reviewer is actually looking at.
        "liveImage.onerror();\n"
        "const afterStaleError={like:elements['#traininglike'].disabled,"
        "dislike:elements['#trainingdislike'].disabled,"
        "sameNode:elements['#trainingimage']===healthyImage};\n"
        "console.log(JSON.stringify({locked,afterStaleLoad,recovered,afterStaleError}));\n"
    )
    result = _run_node(script)

    assert result["locked"]["like"] is True and result["locked"]["dislike"] is True
    assert "could not be loaded" in result["locked"]["message"]
    # The superseded load changed nothing: still locked, and no re-render was triggered.
    assert result["afterStaleLoad"]["like"] is True
    assert result["afterStaleLoad"]["dislike"] is True
    assert "could not be loaded" in result["afterStaleLoad"]["message"]
    assert result["afterStaleLoad"]["sameNode"] is True
    # The live element's own load is the one event that may unlock, and it re-renders.
    assert result["recovered"]["like"] is False and result["recovered"]["dislike"] is False
    assert result["recovered"]["sameNode"] is False
    # A superseded onerror likewise cannot lock the frame the reviewer is actually looking at.
    assert result["afterStaleError"]["like"] is False
    assert result["afterStaleError"]["dislike"] is False
    assert result["afterStaleError"]["sameNode"] is True


def test_training_image_zoom_keeps_the_modal_open_and_in_sync_while_navigating():
    """The review modal is outside the polling-replaced panel, so navigation must update
    both copies of the snapshot without closing the modal or changing its profile binding."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    functions = "\n".join((
        _extract_js_function(_PAGE, "safeCheckpointImageDataUrl"),
        _extract_js_function(_PAGE, "trainingCheckpointKey"),
        _extract_js_function(_PAGE, "checkpointReviewImages"),
        _extract_js_function(_PAGE, "setTrainingImageZoom"),
        _extract_js_function(_PAGE, "toggleTrainingImageZoom"),
        _extract_js_function(_PAGE, "onTrainingImageZoomKeydown"),
        _extract_js_function(_PAGE, "showTrainingImage"),
        _extract_js_function(_PAGE, "moveTrainingImage"),
    ))
    script = (
        "function node(){return {hidden:false,disabled:false,src:'',alt:'',title:'',"
        "textContent:'',attrs:{},focused:0,setAttribute(k,v){this.attrs[k]=String(v);},"
        "focus(){this.focused+=1;}}}\n"
        "const elements={}; for (const id of ['#trainingimagezoomoverlay','#trainingimagezoom',"
        "'#trainingimagezoomclose','#trainingimage','#trainingimageposition','#trainingimageup',"
        "'#trainingimagedown','#trainingimagezoomed','#trainingimagezoomposition',"
        "'#trainingimagezoomup','#trainingimagezoomdown']) elements[id]=node();\n"
        "const classes=new Set(); const document={activeElement:elements['#trainingimagezoom'],"
        "body:{classList:{toggle(name,on){if(on)classes.add(name);else classes.delete(name);}}},"
        "contains(){return true;}}; function $(selector){return elements[selector]||null;}\n"
        "let _trainingImageIndex=3,_trainingImageKey='',_trainingImageFailedIndex=null,_trainingImageRenderGeneration=0,_trainingImageFailedGeneration=null,_trainingImageZoomed=false,"
        "_trainingImageZoomRestoreFocus=null; const _trainingCheckpoint={run_id:'run-1',app:'hinge',"
        "profile_token:'profile-1',approval_token:'approval-1',profile_image_count:3,"
        "image_data_url:'data:image/png;base64,AA=='};\n"
        + functions + "\n"
        "toggleTrainingImageZoom(); moveTrainingImage(-1);\n"
        "const afterButtonNav={open:!elements['#trainingimagezoomoverlay'].hidden,"
        "bodyOpen:classes.has('training-image-zoom-open'),index:_trainingImageIndex,"
        "preview:elements['#trainingimage'].src,zoomed:elements['#trainingimagezoomed'].src,"
        "position:elements['#trainingimagezoomposition'].textContent,"
        "upDisabled:elements['#trainingimagezoomup'].disabled,"
        "downDisabled:elements['#trainingimagezoomdown'].disabled};\n"
        "let prevented=0; onTrainingImageZoomKeydown({key:'ArrowUp',preventDefault(){prevented+=1;}});\n"
        "const afterKeyNav={open:!elements['#trainingimagezoomoverlay'].hidden,index:_trainingImageIndex,"
        "preview:elements['#trainingimage'].src,zoomed:elements['#trainingimagezoomed'].src};\n"
        "onTrainingImageZoomKeydown({key:'Escape',preventDefault(){prevented+=1;}});\n"
        "console.log(JSON.stringify({afterButtonNav,afterKeyNav,closed:elements['#trainingimagezoomoverlay'].hidden,"
        "bodyOpen:classes.has('training-image-zoom-open'),pressed:elements['#trainingimagezoom'].attrs['aria-pressed'],"
        "prevented,restoredFocus:elements['#trainingimagezoom'].focused}));\n"
    )
    result = _run_node(script)

    assert result["afterButtonNav"]["open"] is True
    assert result["afterButtonNav"]["bodyOpen"] is True
    assert result["afterButtonNav"]["index"] == 2
    assert result["afterButtonNav"]["preview"] == result["afterButtonNav"]["zoomed"]
    assert result["afterButtonNav"]["position"] == "profile snapshot 3 of 3 · 3/4"
    assert result["afterButtonNav"]["upDisabled"] is False
    assert result["afterButtonNav"]["downDisabled"] is False
    assert "index=2" in result["afterButtonNav"]["preview"]
    assert result["afterKeyNav"]["open"] is True
    assert result["afterKeyNav"]["index"] == 1
    assert result["afterKeyNav"]["preview"] == result["afterKeyNav"]["zoomed"]
    assert "index=1" in result["afterKeyNav"]["preview"]
    assert result["closed"] is True and result["bodyOpen"] is False
    assert result["pressed"] == "false" and result["prevented"] == 2
    assert result["restoredFocus"] == 1


def test_training_zoom_trigger_and_persistent_modal_are_wired_for_click_and_reset():
    assert re.search(r'<div\s+id="trainingimagezoomoverlay"[^>]*role="dialog"[^>]*aria-modal="true"', _PAGE)
    assert 'id="trainingimagezoomup"' in _PAGE and 'id="trainingimagezoomdown"' in _PAGE
    assert re.search(r"imageZoomButton\.onclick\s*=\s*\(\)\s*=>\s*toggleTrainingImageZoom\(\)", _PAGE)
    # Every path that retires/replaces the card must use the same close routine, so the
    # background's inert state and focus restoration are never left behind.
    for name in ("syncTrainingImageState", "renderTrainingCheckpoint",
                 "renderTrainingFeedback", "submitTrainingAction"):
        assert "setTrainingImageZoom(false)" in _extract_js_function(_PAGE, name)


def test_training_zoom_makes_background_inert_and_traps_tab_until_closed():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    functions = "\n".join((
        _extract_js_function(_PAGE, "setTrainingImageZoom"),
        _extract_js_function(_PAGE, "onTrainingImageZoomKeydown"),
    ))
    script = (
        "let document; function node(){return {hidden:false,disabled:false,attrs:{},focused:0,"
        "setAttribute(k,v){this.attrs[k]=String(v);},getAttribute(k){return this.attrs[k]||'';},"
        "focus(){this.focused+=1;document.activeElement=this;}}}\n"
        "const elements={}; for(const id of ['#trainingimagezoomoverlay','#trainingimagezoom',"
        "'#trainingimagezoomclose','#trainingimagezoomup','#trainingimagezoomdown'])elements[id]=node();\n"
        "const bodyClasses=new Set(),layout={inert:false}; document={activeElement:elements['#trainingimagezoom'],"
        "body:{classList:{toggle(name,on){if(on)bodyClasses.add(name);else bodyClasses.delete(name);}}},"
        "querySelector(selector){return selector==='.hub-layout'?layout:null;},contains(){return true;}};\n"
        "function $(selector){return elements[selector]||null;} let _trainingImageZoomed=false,"
        "_trainingImageZoomRestoreFocus=null;\n"
        + functions + "\nsetTrainingImageZoom(true); const opened={inert:layout.inert,"
        "visible:!elements['#trainingimagezoomoverlay'].hidden,focused:document.activeElement===elements['#trainingimagezoomclose']};\n"
        "let prevented=0; onTrainingImageZoomKeydown({key:'Tab',shiftKey:false,preventDefault(){prevented+=1;}});"
        "const forward=document.activeElement===elements['#trainingimagezoomup'];\n"
        "onTrainingImageZoomKeydown({key:'Tab',shiftKey:false,preventDefault(){prevented+=1;}});"
        "const last=document.activeElement===elements['#trainingimagezoomdown'];\n"
        "onTrainingImageZoomKeydown({key:'Tab',shiftKey:true,preventDefault(){prevented+=1;}});"
        "const backward=document.activeElement===elements['#trainingimagezoomup'];\n"
        "onTrainingImageZoomKeydown({key:'Escape',preventDefault(){prevented+=1;}});\n"
        "console.log(JSON.stringify({opened,forward,last,backward,prevented,closed:elements['#trainingimagezoomoverlay'].hidden,"
        "inert:layout.inert,bodyOpen:bodyClasses.has('training-image-zoom-open'),"
        "focusRestored:document.activeElement===elements['#trainingimagezoom']}));\n"
    )
    result = _run_node(script)

    assert result["opened"] == {"inert": True, "visible": True, "focused": True}
    assert result["forward"] is True and result["last"] is True and result["backward"] is True
    assert result["prevented"] == 4
    assert result["closed"] is True and result["inert"] is False and result["bodyOpen"] is False
    assert result["focusRestored"] is True


def test_replacing_a_zoomed_checkpoint_moves_focus_to_its_new_preview_trigger():
    """Polling replaces the review-card markup while the persistent overlay is open.

    Closing the overlay first nominally restores focus to the old trigger, but that node is
    about to be discarded.  The new card must explicitly take focus after its trigger exists.
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    old = {
        "run_id": "run-1", "app": "hinge", "profile_token": "old-profile",
        "approval_token": "old-approval", "phase": "waiting_training_decision",
        "pending": True, "action": "ready", "item": 1, "opener": "Old opener",
        "image_data_url": "data:image/png;base64,AA==", "profile_image_count": 1,
    }
    new = {**old, "profile_token": "new-profile", "approval_token": "new-approval",
           "item": 2, "opener": "New opener"}
    functions = "\n".join((
        _extract_js_function(_PAGE, "escHtml"),
        _extract_js_function(_PAGE, "safeCheckpointImageDataUrl"),
        _extract_js_function(_PAGE, "trainingCheckpointKey"),
        _extract_js_function(_PAGE, "resetTrainingIdempotencyIfCardChanged"),
        _extract_js_function(_PAGE, "resetTrainingBusyIfCardChanged"),
        _extract_js_function(_PAGE, "trainingActionBusyFor"),
        _extract_js_function(_PAGE, "checkpointReviewImages"),
        _extract_js_function(_PAGE, "syncTrainingImageState"),
        _extract_js_function(_PAGE, "setTrainingImageZoom"),
        _extract_js_function(_PAGE, "renderTrainingCheckpoint"),
    ))
    script = (
        "let document; function node(name){return {name,hidden:false,disabled:false,attrs:{},"
        "focused:0,connected:true,setAttribute(k,v){this.attrs[k]=String(v);},"
        "getAttribute(k){return this.attrs[k]||'';},focus(){this.focused+=1;document.activeElement=this;}}}\n"
        "const oldTrigger=node('old'); const elements={'#trainingimagezoom':oldTrigger};\n"
        "for(const id of ['#trainingimagezoomoverlay','#trainingimagezoomclose','#trainingimagezoomed',"
        "'#trainingimagezoomposition','#trainingimagezoomup','#trainingimagezoomdown'])elements[id]=node(id);\n"
        "const layout={inert:false,classList:{toggle(){}}}; const bodyClasses=new Set();\n"
        "const panel={style:{display:''},_html:'',get innerHTML(){return this._html;},set innerHTML(value){"
        "this._html=value; if(value.includes('id=\\\"trainingimagezoom\\\"')){oldTrigger.connected=false;"
        "elements['#trainingimagezoom']=node('new');}}}; elements['#trainingpanel']=panel;\n"
        "document={activeElement:elements['#trainingimagezoomclose'],body:{classList:{toggle(name,on){"
        "if(on)bodyClasses.add(name);else bodyClasses.delete(name);}}},querySelector(selector){"
        "return selector==='.hub-layout'?layout:null;},contains(item){return !!item.connected;}};\n"
        "function $(selector){return elements[selector]||null;}\n"
        "let _trainingCheckpoint=" + json.dumps(old) + ",_trainingActionBusy=false,_trainingBusyKey='',"
        "_trainingBusyRequest=0,_trainingImageKey=" + json.dumps(json.dumps([
            old["run_id"], old["app"], old["profile_token"], old["approval_token"]])) + ","
        "_trainingImageIndex=1,_trainingImageFailedIndex=null,_trainingImageRenderGeneration=0,_trainingImageFailedGeneration=null,_trainingImageZoomed=true,_trainingImageZoomRestoreFocus=oldTrigger;"
        "const _trainingIdempotency=new Map();\n"
        + functions + "\nrenderTrainingCheckpoint(" + json.dumps(new) + ");\n"
        "const replacement=elements['#trainingimagezoom']; console.log(JSON.stringify({"
        "focusOnReplacement:document.activeElement===replacement,replacementFocused:replacement.focused,"
        "oldFocused:oldTrigger.focused,oldConnected:oldTrigger.connected,overlayHidden:elements['#trainingimagezoomoverlay'].hidden,"
        "zoomed:_trainingImageZoomed,inert:layout.inert,bodyLocked:bodyClasses.has('training-image-zoom-open')}));\n"
    )
    result = _run_node(script)

    assert result == {
        "focusOnReplacement": True, "replacementFocused": 1,
        "oldFocused": 1, "oldConnected": False, "overlayHidden": True,
        "zoomed": False, "inert": False, "bodyLocked": False,
    }


def test_training_panel_hides_and_restores_the_original_single_column_layout():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    result = _run_node(_training_panel_script(None))
    assert result == {"display": "none", "html": "", "active": False}


def test_actionable_training_checkpoint_requests_one_silent_browser_notification():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    functions = "\n".join((
        _extract_js_function(_PAGE, "trainingCheckpointKey"),
        _extract_js_function(_PAGE, "trainingAlertPermission"),
        _extract_js_function(_PAGE, "trainingAlertHintText"),
        _extract_js_function(_PAGE, "syncTrainingAlertHint"),
        _extract_js_function(_PAGE, "reportTrainingBrowserNotification"),
        _extract_js_function(_PAGE, "notifyTrainingCheckpoint"),
    ))
    script = (
        "const hint={textContent:''},posts=[],notices=[]; let focused=0;\n"
        "class BrowserNotification { constructor(title,options){this.title=title;this.options=options;"
        "this.closed=0;notices.push(this);} close(){this.closed+=1;} }\n"
        "BrowserNotification.permission='granted'; const window={Notification:BrowserNotification,"
        "focus(){focused+=1;}}; function $(selector){return selector==='#alerthint'?hint:null;}\n"
        "function postJSON(path,body){posts.push({path,body});return Promise.resolve({ok:true});}\n"
        "let _trainingBrowserAlertedKey='',_trainingBrowserPermissionPendingKey='',_wasRunning=true;\n"
        + functions + "\n"
        "const card={run_id:'run-1',app:'hinge',profile_token:'profile-1',"
        "approval_token:'approval-1',pending:true,phase:'waiting_training_decision',action:'ready'};\n"
        "notifyTrainingCheckpoint(card);notifyTrainingCheckpoint(card);notices[0].onclick();\n"
        "setImmediate(()=>console.log(JSON.stringify({count:notices.length,options:notices[0].options,"
        "post:posts[0],focused,closed:notices[0].closed,hint:hint.textContent})));\n"
    )
    result = _run_node(script)
    assert result["count"] == 1
    assert result["options"]["silent"] is True
    assert result["options"]["requireInteraction"] is True
    assert result["post"] == {
        "path": "/api/training/alert",
        "body": {"run_id": "run-1", "app": "hinge", "profile_token": "profile-1",
                 "notification": "requested"},
    }
    assert result["focused"] == result["closed"] == 1
    assert "browser banners allowed" in result["hint"]


def test_training_start_requests_browser_permission_inside_click_handler():
    assert "prepareTrainingBrowserNotifications()" in _extract_js_function(
        _PAGE, "startRunFromControls")
    assert re.search(r'id="alerthint"[^>]*role="status"', _PAGE)


def _alert_hint_script(cases) -> str:
    """The real hint text, derived for each (permission, running) pair under node."""
    return (
        _extract_js_function(_PAGE, "trainingAlertHintText") + "\n"
        "const cases=" + json.dumps(cases) + ";\n"
        "console.log(JSON.stringify(cases.map(c => trainingAlertHintText(c[0], c[1]))));\n"
    )


def test_alert_hint_names_a_gesture_the_operator_can_actually_press():
    """The audited defect: permission='default' reached mid-run could never be armed again.

    requestPermission is only honoured from a real user gesture, and the page's original only
    gesture was the Start click -- which renderGlobal disables for the whole of a live run. So
    the hint told the operator to press a greyed-out button while every checkpoint posted
    notification=permission-default and returned before constructing a banner. Guidance derives
    from the precondition that is actually blocking, so the wording has to flip with the run:
    Start while Start is clickable, the decision card's control once it is not.
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")

    hints = _run_node(_alert_hint_script([
        ["default", False], ["default", True],
        ["granted", True], ["denied", True], ["", True],
    ]))
    idle_default, live_default, granted, denied, unsupported = hints

    assert "Start Training" in idle_default and "Allow alerts" not in idle_default
    # The live-run case must NOT send them to Start: it is disabled, and saying so is the point.
    assert "Allow alerts" in live_default and "Start Training" not in live_default
    assert "Start is disabled" in live_default
    assert "browser banners allowed" in granted and "Allow alerts" not in granted
    assert "browser banners blocked" in denied and "Allow alerts" not in denied
    assert "Browser notifications are unavailable" in unsupported
    # Every branch still says the macOS host sound is a separate channel -- on the audited host
    # the deprecated native API is sound-only, so the two are never the same alert.
    assert all(re.search(r"macOS|Mac sound", hint) for hint in hints)


def test_alert_hint_is_rewritten_on_the_running_transition_and_only_then():
    """renderGlobal owns `running`, so it is the only place that can notice the flip.

    #alerthint is an aria-live region: re-deriving it on every poll would make a screen reader
    re-announce identical text once a second, so the rewrite is gated on the transition. The
    harness blanks the element before each poll, so a poll that did not re-derive reads ''.
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")

    results = _run_node(_render_global_script([
        {"snap": {"running": False, "status": None}},        # no change from the initial state
        {"snap": {"running": True, "status": {"phase": "live", "budget_spent": 0}}},
        {"snap": {"running": True, "status": {"phase": "live", "budget_spent": 0}}},
        {"snap": {"running": False, "status": None}},
    ], alert_permission="default"))

    assert results[0]["alertHint"] == ""                     # nothing flipped -> no rewrite
    assert "Allow alerts" in results[1]["alertHint"]         # Start just became unpressable
    assert results[2]["alertHint"] == ""                     # still live -> no aria-live churn
    assert "Start Training" in results[3]["alertHint"]       # Start is pressable again


def test_live_run_at_default_permission_offers_an_enabled_allow_alerts_control():
    """The escape hatch itself: a real, clickable gesture inside the decision panel.

    Enabled is the whole point -- a disabled affordance is the bug being fixed -- and it must
    stay fine print: no primary/danger styling, below the decision row rather than ahead of it,
    and it must not introduce a GO/WAIT cue of its own (owner rule: the decision window stays
    the green GO box, run-level context is fine print).
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")

    html = _run_node(_training_panel_script(
        _training_checkpoint_with(), alert_permission="default", running=True))["html"]

    assert 'id="trainingallowalerts"' in html
    assert not re.search(r'id="trainingallowalerts"[^>]* disabled', html)
    assert ">Allow alerts</button>" in html
    # Fine print, not a decision: plain button inside a `.meta` row, after Like/Dislike.
    assert 'id="trainingalertgesture"' in html and 'class="meta" id="trainingalertgesture"' in html
    assert not re.search(r'id="trainingallowalerts"[^>]*class="(primary|danger)"', html)
    assert html.index('id="trainingallowalerts"') > html.index('id="trainingdislike"')
    # It adds no GO/WAIT cue and does not restyle one (hands are banned outright; circles belong
    # to the run banner, not to this affordance).
    assert "🟢" not in html and "🔴" not in html and "👍" not in html and "👎" not in html
    # The decision controls themselves are untouched by its presence.
    assert ">Like</button>" in html and ">Dislike</button>" in html
    assert not re.search(r'id="traininglike"[^>]* disabled', html)


@pytest.mark.parametrize("permission,running", [
    ("granted", True),      # already armed -- nothing to ask for
    ("denied", True),       # a denied permission can never be re-asked from a page
    ("default", False),     # Start is right there and enabled; it is the gesture
    (None, True),           # no Notification API at all
])
def test_allow_alerts_control_is_absent_whenever_start_or_the_answer_already_settles_it(
        permission, running):
    """Both halves of the precondition are load-bearing, so neither alone may render it."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")

    html = _run_node(_training_panel_script(
        _training_checkpoint_with(), alert_permission=permission, running=running))["html"]

    assert "trainingallowalerts" not in html and "trainingalertgesture" not in html


# The literal seam the optional affordance is spliced into: the controls row's closing tag, the
# newline, the template's six-space indentation, and the action-hint row that follows. Written
# out here as ONE constant on purpose -- see the test below for what a symmetric comparison
# cannot see.
_TRAINING_CARD_SEAM = (
    '</div>\n       <div class="meta" id="trainingactionhint" style="margin-top:8px">')


def test_card_without_the_allow_alerts_control_is_byte_identical_to_the_plain_card():
    """It is an affordance, never a capability: absent, the card must render exactly as before.

    THE SUBTRACTION ALONE PROVES NOTHING ABOUT THE SEAM (2026-09-15). Both renders come from the
    same template, so a review that deleted whitespace from the shared trailing fragment changed
    BOTH identically: the subtraction still matched and this test stayed green, blind to the very
    whitespace bug it was written after. Pin the BOUNDARY as a literal against the render that
    has no affordance in it, then require that same literal to survive in the with-control render
    once the block is lifted out -- a shared-fragment edit now fails the first assertion, and a
    splice that eats or adds a byte at the join fails the second.
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")

    with_control = _run_node(_training_panel_script(
        _training_checkpoint_with(), alert_permission="default", running=True))["html"]
    without = _run_node(_training_panel_script(_training_checkpoint_with()))["html"]

    assert without.count(_TRAINING_CARD_SEAM) == 1
    block = re.search(
        r'\n       <div class="meta" id="trainingalertgesture".*?\n       </div>',
        with_control, re.S)
    assert block, "the affordance should render as one self-contained fine-print row"
    lifted = with_control.replace(block.group(0), "", 1)
    assert lifted.count(_TRAINING_CARD_SEAM) == 1
    assert lifted == without


def test_allow_alerts_click_requests_permission_and_buys_exactly_one_repaint():
    """The click must reach requestPermission synchronously (a gesture survives nothing else),
    and the answer landing must free ONE repaint -- but a DISMISSAL must free none.

    trainingPollRenderFingerprint deliberately ignores permission -- putting it in there would
    re-fetch both no-store review PNGs on every poll. But a waiting card is byte-identical poll
    after poll, so without clearing the cached signature the panel would go on offering a
    control whose precondition is gone. Clearing it on the ANSWER (not on the click) means the
    repaint happens once, after the operator has actually chosen.

    The dismissal leg (2026-09-15) is the half that was missing: closing the browser's prompt
    without answering resolves it with 'default' still in place. Nothing about the card changed
    and the control must stay offered, so clearing the signature there bought a full checkpoint
    repaint -- which DOES re-create the `<img src="/api/training/review.png...">` elements and
    re-fetch both no-store PNGs -- every single time an operator dismissed.
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    functions = "\n".join((
        _extract_js_function(_PAGE, "escHtml"),
        _extract_js_function(_PAGE, "safeCheckpointImageDataUrl"),
        _extract_js_function(_PAGE, "trainingCheckpointKey"),
        _extract_js_function(_PAGE, "resetTrainingIdempotencyIfCardChanged"),
        _extract_js_function(_PAGE, "resetTrainingBusyIfCardChanged"),
        _extract_js_function(_PAGE, "trainingActionBusyFor"),
        _extract_js_function(_PAGE, "checkpointReviewImages"),
        _extract_js_function(_PAGE, "syncTrainingImageState"),
        _extract_js_function(_PAGE, "trainingAlertPermission"),
        _extract_js_function(_PAGE, "trainingAlertHintText"),
        _extract_js_function(_PAGE, "shouldOfferTrainingAlertGesture"),
        _extract_js_function(_PAGE, "trainingAlertGestureOffered"),
        _extract_js_function(_PAGE, "syncTrainingAlertHint"),
        _extract_js_function(_PAGE, "prepareTrainingBrowserNotifications"),
        _extract_js_function(_PAGE, "renderTrainingCheckpoint"),
    ))
    script = (
        "function node(){return {hidden:false,disabled:false,src:'',alt:'',textContent:'',"
        "setAttribute(){},removeAttribute(){}};}\n"
        "const elements={'#alerthint':node()};\n"
        "const panel={style:{display:''},_html:'',get innerHTML(){return this._html;},"
        "set innerHTML(value){this._html=value; for(const id of ['#trainingallowalerts',"
        "'#traininglike','#trainingdislike','#trainingimage','#trainingimagezoom'])"
        "if(value.includes('id=\\\"'+id.slice(1)+'\\\"'))elements[id]=node();}};\n"
        "elements['#trainingpanel']=panel;\n"
        "const layout={classList:{toggle(){}}};\n"
        "const document={querySelector(s){return s==='.hub-layout'?layout:null;}};\n"
        "function $(selector){return elements[selector]||null;} function setTrainingImageZoom(){}\n"
        "let asked=0,resolvePermission=null;\n"
        "const window={Notification:{permission:'default',requestPermission(){asked+=1;"
        "return new Promise(r=>{resolvePermission=r;});}}};\n"
        "let _trainingCheckpoint=null,_trainingActionBusy=false,_trainingBusyKey='',"
        "_trainingBusyRequest=0,_trainingImageKey='',_trainingImageIndex=0,"
        "_trainingImageFailedIndex=null,_trainingImageRenderGeneration=0,"
        "_trainingImageFailedGeneration=null,_wasRunning=true,"
        "_trainingPollRenderSignature='a-waiting-card';\n"
        "const _trainingIdempotency=new Map();\n"
        + functions + "\n"
        "renderTrainingCheckpoint(" + json.dumps(_training_checkpoint_with()) + ");\n"
        "const rendered=!!elements['#trainingallowalerts'];\n"
        "elements['#trainingallowalerts'].onclick();\n"
        "const onClick={asked,signature:_trainingPollRenderSignature};\n"
        # Leg 1: the operator DISMISSES. The prompt resolves with 'default' left in place.
        "resolvePermission('default');\n"
        "setImmediate(()=>{const dismissed={asked,signature:_trainingPollRenderSignature,"
        "hint:elements['#alerthint'].textContent};\n"
        # Leg 2: the control is still there, so they click again and this time allow.
        "elements['#trainingallowalerts'].onclick();\n"
        "window.Notification.permission='granted'; resolvePermission('granted');\n"
        "setImmediate(()=>console.log(JSON.stringify({rendered,onClick,dismissed,asked,"
        "signature:_trainingPollRenderSignature,hint:elements['#alerthint'].textContent})));});\n"
    )
    result = _run_node(script)

    assert result["rendered"] is True
    # Synchronous inside the click: asked already incremented before any promise settled.
    assert result["onClick"]["asked"] == 1
    # The click alone must NOT invalidate -- the operator may still be staring at the prompt.
    assert result["onClick"]["signature"] == "a-waiting-card"
    # A DISMISSAL settles the promise without moving the permission, so it buys no repaint at
    # all: the cached signature survives and the hint still names the Allow alerts gesture.
    assert result["dismissed"]["asked"] == 1
    assert result["dismissed"]["signature"] == "a-waiting-card"
    assert "Allow alerts button" in result["dismissed"]["hint"]
    # The real ANSWER does, exactly once, and the hint follows the new permission.
    assert result["asked"] == 2
    assert result["signature"] == ""
    assert "browser banners allowed" in result["hint"]


def test_checkpoint_retries_after_an_open_notification_permission_prompt_resolves():
    """Driven through the POLL entry point, which is the only thing that runs in production.

    notifyTrainingCheckpoint keeps a card eligible while its permission prompt is still open, but
    nothing in the render fingerprint changes when the operator finally clicks Allow. Calling
    notifyTrainingCheckpoint directly here (the earlier shape of this test) proved nothing about
    that retry: the poll path short-circuits on an unchanged card, so the banner was never
    delivered for the card that was on screen at the moment of the grant. The renders counter
    below pins the other half -- the retry must NOT cost a repaint of the no-store review PNGs.
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    functions = "\n".join((
        _extract_js_function(_PAGE, "trainingCheckpointKey"),
        _extract_js_function(_PAGE, "trainingPollRenderFingerprint"),
        _extract_js_function(_PAGE, "renderTrainingCheckpointFromPoll"),
        _extract_js_function(_PAGE, "trainingAlertPermission"),
        _extract_js_function(_PAGE, "trainingAlertHintText"),
        _extract_js_function(_PAGE, "syncTrainingAlertHint"),
        _extract_js_function(_PAGE, "reportTrainingBrowserNotification"),
        _extract_js_function(_PAGE, "notifyTrainingCheckpoint"),
    ))
    script = (
        "const hint={textContent:''},posts=[],notices=[]; class BrowserNotification {"
        "constructor(){notices.push(this);} } BrowserNotification.permission='default';"
        "const window={Notification:BrowserNotification,focus(){}}; function $(selector){"
        "return selector==='#alerthint'?hint:null;} function postJSON(path,body){posts.push({path,body});"
        "return Promise.resolve({ok:true});} let _trainingBrowserAlertedKey='',"
        "_trainingBrowserPermissionPendingKey='',_trainingPollRenderSignature='',renders=0,_wasRunning=true;"
        "function renderTrainingCheckpoint(){renders+=1;}\n" + functions + "\n"
        "const card={run_id:'run-1',app:'hinge',profile_token:'profile-1',approval_token:'approval-1',"
        "pending:true,phase:'waiting_training_decision',action:'ready'};"
        "renderTrainingCheckpointFromPoll(card);renderTrainingCheckpointFromPoll({...card});"
        "const pending={notices:notices.length,posts:posts.map(x=>x.body.notification),renders};"
        "BrowserNotification.permission='granted';renderTrainingCheckpointFromPoll({...card});"
        "setImmediate(()=>console.log(JSON.stringify({pending,notices:notices.length,"
        "posts:posts.map(x=>x.body.notification),renders})));"
    )
    result = _run_node(script)
    assert result == {
        "pending": {"notices": 0, "posts": ["permission-default"], "renders": 1},
        "notices": 1,
        "posts": ["permission-default", "requested"],
        "renders": 1,
    }


def test_training_panel_disables_both_decisions_when_review_data_is_incomplete():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    checkpoint = {
        "run_id": "run-1", "app": "hinge", "profile_token": "profile-1",
        "approval_token": "approval-1", "phase": "waiting_training_decision", "pending": True,
        "action": "ready", "item": None, "opener": None, "image_data_url": "",
    }
    result = _run_node(_training_panel_script(checkpoint))
    assert "Review data is incomplete" in result["html"]
    assert 'id="traininglike" class="primary" aria-label="Like — send opener" disabled' in result["html"]
    assert 'id="trainingdislike" class="danger" aria-label="Dislike — pass this profile" disabled' in result["html"]


def test_training_panel_keeps_queued_decision_visible_but_not_actionable():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    checkpoint = {
        "run_id": "run-1", "app": "hinge", "profile_token": "profile-1",
        "approval_token": "approval-1", "phase": "waiting_training_decision", "pending": False,
        "action": "queued", "command": "like", "opener": "typed opener",
        "image_data_url": "data:image/png;base64,AA==",
    }
    result = _run_node(_training_panel_script(checkpoint))
    assert "Like is queued; waiting for the worker to claim it." in result["html"]
    assert 'id="traininglike" class="primary" aria-label="Like — send opener" disabled' in result["html"]
    assert 'id="trainingdislike" class="danger" aria-label="Dislike — pass this profile" disabled' in result["html"]


def test_training_feedback_surfaces_terminal_status_and_reason():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    script = (
        "const panel={style:{display:''},innerHTML:''};\n"
        "const layout={active:true,classList:{toggle(_name,value){layout.active=!!value;}}};\n"
        "const document={querySelector:(selector)=>selector==='.hub-layout'?layout:null};\n"
        "function $(selector){ return selector==='#trainingpanel' ? panel : null; }\n"
        "let _trainingCheckpoint={run_id:'r',app:'hinge',profile_token:'p',approval_token:'a'}, _trainingActionBusy=false, _trainingBusyKey='', _trainingBusyRequest=0; const _trainingIdempotency=new Map();\n"
        + _extract_js_function(_PAGE, "escHtml") + "\n"
        + _extract_js_function(_PAGE, "trainingCheckpointKey") + "\n"
        + _extract_js_function(_PAGE, "resetTrainingIdempotencyIfCardChanged") + "\n"
        + _extract_js_function(_PAGE, "resetTrainingBusyIfCardChanged") + "\n"
        + _extract_js_function(_PAGE, "renderTrainingFeedback") + "\n"
        + "renderTrainingFeedback({command:'dislike',status:'failed',reason:'device verification failed'});\n"
        + "console.log(JSON.stringify({html:panel.innerHTML,display:panel.style.display,active:layout.active}));\n"
    )
    result = _run_node(script)
    assert "decision failed" in result["html"]
    assert "device verification failed" in result["html"]
    assert result["display"] == "block" and result["active"] is False


def test_stopping_immediately_replaces_actionable_checkpoint_with_non_actionable_feedback():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    script = (
        "let _trainingRequest=0, seen=null;\n"
        "function selectRunApps(status){return Object.values(status.apps).filter(a=>a.mode==='training'||a.mode==='auto');}\n"
        "function renderTrainingFeedback(result,message){seen={result,message};}\n"
        "function renderTrainingCheckpoint(){throw new Error('must not render an actionable card while stopping');}\n"
        + _extract_js_function(_PAGE, "tickTrainingCheckpoint") + "\n"
        + "tickTrainingCheckpoint({running:true,status:{run_id:'run-1',stopping:true,apps:{hinge:{app:'hinge',mode:'training'}}}}).then(()=>console.log(JSON.stringify(seen)));\n"
    )
    result = _run_node(script)
    assert result["result"] is None
    assert "no new choice is available" in result["message"]
    assert "finishing and being recorded" in result["message"]


def test_training_panel_accepts_only_png_image_urls():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fn = _extract_js_function(_PAGE, "safeCheckpointImageDataUrl")
    script = (fn + "\nconsole.log(JSON.stringify(["
              "safeCheckpointImageDataUrl('data:image/png;base64,AA=='),"
              "safeCheckpointImageDataUrl('data:image/jpeg;base64,AA=='),"
              "safeCheckpointImageDataUrl('data:image/svg+xml;base64,PHN2Zz4='),"
              "safeCheckpointImageDataUrl('https://example.test/a.png')" "]));\n")
    assert _run_node(script) == ["data:image/png;base64,AA==", "", "", ""]


def test_training_choice_posts_the_bound_checkpoint_and_idempotency_token():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fns = (_extract_js_function(_PAGE, "safeCheckpointImageDataUrl") + "\n"
           + _extract_js_function(_PAGE, "trainingCheckpointKey") + "\n"
           + _extract_js_function(_PAGE, "trainingActionBusyFor") + "\n"
           + _extract_js_function(_PAGE, "trainingActionToken") + "\n"
           + _extract_js_function(_PAGE, "submitTrainingAction"))
    script = (
        "const hubClientId='hub-client';\n"
        "const window={crypto:{randomUUID:()=> 'nonce'}}; const crypto=window.crypto;\n"
        "const _trainingIdempotency=new Map(); let _trainingActionBusy=false, _trainingBusyKey='', _trainingBusyRequest=0, _trainingImageFailedIndex=null,_trainingImageRenderGeneration=0,_trainingImageFailedGeneration=null;\n"
        "let _trainingCheckpoint={run_id:'run-1',app:'hinge',profile_token:'profile-1',approval_token:'approval-1',opener:'full opener',image_data_url:'data:image/png;base64,AA=='};\n"
        "let posted=null, renders=[], ticks=0;\n"
        "function renderTrainingCheckpoint(checkpoint,message){renders.push([checkpoint,message]);}\n"
        "async function postJSON(path,body){posted={path,body};return {ok:true};}\n"
        "function tick(){ticks+=1;}\n"
        + fns + "\nsubmitTrainingAction('like').then(()=>console.log(JSON.stringify({posted,renders,ticks})));\n"
    )
    result = _run_node(script)
    assert result["posted"]["path"] == "/api/training/action"
    assert result["posted"]["body"] == {
        "command": "like", "run_id": "run-1", "app": "hinge",
        "profile_token": "profile-1", "approval_token": "approval-1",
        "idempotency_token": "hub-client-nonce",
    }
    assert result["ticks"] == 1


def test_training_choice_does_not_post_when_the_review_data_is_missing():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fns = (_extract_js_function(_PAGE, "safeCheckpointImageDataUrl") + "\n"
           + _extract_js_function(_PAGE, "trainingCheckpointKey") + "\n"
           + _extract_js_function(_PAGE, "trainingActionBusyFor") + "\n"
           + _extract_js_function(_PAGE, "trainingActionToken") + "\n"
           + _extract_js_function(_PAGE, "submitTrainingAction"))
    script = (
        "const hubClientId='hub-client';\n"
        "const window={crypto:{randomUUID:()=> 'nonce'}}; const crypto=window.crypto;\n"
        "const _trainingIdempotency=new Map(); let _trainingActionBusy=false, _trainingBusyKey='', _trainingBusyRequest=0, _trainingImageFailedIndex=null,_trainingImageRenderGeneration=0,_trainingImageFailedGeneration=null;\n"
        "let _trainingCheckpoint={run_id:'run-1',app:'hinge',profile_token:'profile-1',approval_token:'approval-1',opener:'',image_data_url:''};\n"
        "let posted=null, renders=[], ticks=0;\n"
        "function renderTrainingCheckpoint(checkpoint,message){renders.push([checkpoint,message]);}\n"
        "async function postJSON(path,body){posted={path,body};return {ok:true};}\n"
        "function tick(){ticks+=1;}\n"
        + fns + "\nsubmitTrainingAction('like').then(()=>console.log(JSON.stringify({posted,renders,ticks})));\n"
    )
    assert _run_node(script) == {"posted": None, "renders": [], "ticks": 0}


def test_training_action_failure_reenables_the_live_card_before_the_next_poll():
    """A dropped local POST must not leave Like/Dislike disabled until a later poll succeeds."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fns = (_extract_js_function(_PAGE, "safeCheckpointImageDataUrl") + "\n"
           + _extract_js_function(_PAGE, "trainingCheckpointKey") + "\n"
           + _extract_js_function(_PAGE, "trainingActionBusyFor") + "\n"
           + _extract_js_function(_PAGE, "trainingActionToken") + "\n"
           + _extract_js_function(_PAGE, "submitTrainingAction"))
    script = (
        "const hubClientId='hub-client';\n"
        "const window={crypto:{randomUUID:()=> 'nonce'}}; const crypto=window.crypto;\n"
        "const _trainingIdempotency=new Map(); let _trainingActionBusy=false, _trainingBusyKey='', _trainingBusyRequest=0, _trainingImageFailedIndex=null,_trainingImageRenderGeneration=0,_trainingImageFailedGeneration=null;\n"
        "let _trainingCheckpoint={run_id:'run-1',app:'hinge',profile_token:'profile-1',approval_token:'approval-1',opener:'full opener',image_data_url:'data:image/png;base64,AA=='};\n"
        "let renders=[], ticks=0;\n"
        "function renderTrainingCheckpoint(checkpoint,message){renders.push({message,busy:_trainingActionBusy,checkpoint});}\n"
        "async function postJSON(){return {ok:false,result:{reason:'stale approval'}};}\n"
        "function tick(){ticks+=1;}\n"
        + fns + "\nsubmitTrainingAction('dislike').then(()=>console.log(JSON.stringify({renders,ticks,busy:_trainingActionBusy})));\n"
    )
    result = _run_node(script)
    assert result["busy"] is False
    assert result["ticks"] == 1
    assert result["renders"][-1]["message"] == "stale approval"
    assert result["renders"][-1]["busy"] is False


def test_old_hung_training_post_cannot_change_new_card_busy_state():
    """A delayed request from a stopped run must not strand or unlock a new card."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fns = (_extract_js_function(_PAGE, "safeCheckpointImageDataUrl") + "\n"
           + _extract_js_function(_PAGE, "trainingCheckpointKey") + "\n"
           + _extract_js_function(_PAGE, "resetTrainingBusyIfCardChanged") + "\n"
           + _extract_js_function(_PAGE, "trainingActionBusyFor") + "\n"
           + _extract_js_function(_PAGE, "trainingActionToken") + "\n"
           + _extract_js_function(_PAGE, "submitTrainingAction"))
    script = (
        "const hubClientId='hub-client';\n"
        "const window={crypto:{randomUUID:()=> 'nonce'}}; const crypto=window.crypto;\n"
        "const _trainingIdempotency=new Map(); let _trainingActionBusy=false, _trainingBusyKey='', _trainingBusyRequest=0, _trainingImageFailedIndex=null,_trainingImageRenderGeneration=0,_trainingImageFailedGeneration=null;\n"
        "const oldCard={run_id:'old',app:'hinge',profile_token:'p1',approval_token:'a1',opener:'full opener',image_data_url:'data:image/png;base64,AA=='};\n"
        "const newCard={run_id:'new',app:'hinge',profile_token:'p2',approval_token:'a2',opener:'full opener',image_data_url:'data:image/png;base64,AA=='};\n"
        "let _trainingCheckpoint=oldCard, resolves=[];\n"
        "function renderTrainingCheckpoint(){} function tick(){}\n"
        "function postJSON(){return new Promise(resolve=>resolves.push(resolve));}\n"
        + fns + "\n"
        "(async()=>{const first=submitTrainingAction('like'); await Promise.resolve();\n"
        "resetTrainingBusyIfCardChanged(oldCard,newCard); _trainingCheckpoint=newCard;\n"
        "const second=submitTrainingAction('dislike'); await Promise.resolve();\n"
        "resolves[0]({ok:true}); await first; const afterOld={busy:_trainingActionBusy,key:_trainingBusyKey};\n"
        "resolves[1]({ok:true}); await second;\n"
        "console.log(JSON.stringify({afterOld,afterNew:{busy:_trainingActionBusy,key:_trainingBusyKey}}));})();\n"
    )
    result = _run_node(script)
    assert result["afterOld"]["busy"] is True
    assert result["afterOld"]["key"] == '["new","hinge","p2","a2"]'
    assert result["afterNew"] == {"busy": False, "key": ""}


def test_training_idempotency_tokens_are_discarded_when_the_card_changes():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fns = (_extract_js_function(_PAGE, "trainingCheckpointKey") + "\n"
           + _extract_js_function(_PAGE, "resetTrainingIdempotencyIfCardChanged"))
    script = (
        "const _trainingIdempotency=new Map([['old-action','old-token']]);\n"
        "const oldCard={run_id:'r',app:'hinge',profile_token:'old',approval_token:'a'};\n"
        "const sameCard={run_id:'r',app:'hinge',profile_token:'old',approval_token:'a'};\n"
        "const newCard={run_id:'r',app:'hinge',profile_token:'new',approval_token:'b'};\n"
        + fns + "\nresetTrainingIdempotencyIfCardChanged(oldCard,sameCard);\n"
        "const same=_trainingIdempotency.size;\n"
        "resetTrainingIdempotencyIfCardChanged(oldCard,newCard);\n"
        "console.log(JSON.stringify({same,changed:_trainingIdempotency.size}));\n"
    )
    assert _run_node(script) == {"same": 1, "changed": 0}


def test_hub_auto_volume_control_defaults_to_config_cap():
    tag = re.search(r'<input\b[^>]*\bid="unlimited"[^>]*>', _PAGE)
    assert tag, "unlimited control missing"
    assert not re.search(r'\bchecked(?:\s|=|>)', tag.group(0)), (
        "a normal Hub start must delegate to config instead of explicitly clearing its cap"
    )
    box = re.search(r'<input\b[^>]*\bid="maxrun"[^>]*>', _PAGE)
    assert box and re.search(r'\bvalue=""', box.group(0)), (
        "the normal per-run input must be blank so computeMaxPerRun posts null"
    )


def test_hub_timed_stop_is_visible_in_both_modes_and_defaults_to_unlimited():
    tag = re.search(r'<select\b[^>]*\bid="stopafter"[^>]*>(.*?)</select>', _PAGE, re.S)
    assert tag, "timed stop control missing"
    assert re.search(r'<option\s+value="0"\s+selected>Unlimited</option>', tag.group(0))
    assert [int(v) for v in re.findall(r'<option\s+value="(\d+)"', tag.group(1))] == [
        0, 900, 1800, 3600, 7200,
    ]
    # The duration is a run-wide safety limit, including a Training run waiting for the
    # operator; unlike the auto-only max-profile control, it must never be mode-gated.
    assert "$('#stopafter').disabled = !!running" in _PAGE
    assert "$('#mode').value === 'auto'" not in tag.group(0)


def test_hub_defaults_to_training_even_when_config_defaults_to_auto():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fn = _extract_js_function(_PAGE, "initialHubMode")
    assert _run_node(fn + "\nconsole.log(JSON.stringify(initialHubMode()));\n") == "training"
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


def test_hub_start_handler_delegates_blank_default_cap_to_config():
    """A blank Hub cap delegates to the config (uncapped in the shipped config)."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fns = (_extract_js_function(_PAGE, "escHtml") + "\n"
           + _extract_js_function(_PAGE, "computeMaxPerRun") + "\n"
           + _extract_js_function(_PAGE, "stopAfterSeconds") + "\n"
           + _extract_js_function(_PAGE, "startRunFromControls"))
    script = (
        "const els = {hint:{textContent:''}, mode:{value:'auto'}, unlimited:{checked:false}, "
        "maxrun:{value:''}, stopafter:{value:'0'}, platnote:{innerHTML:''}};\n"
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
        "body": {"mode": "auto", "apps": ["hinge"], "max_per_run": None,
                 "stop_after_seconds": 0},
    }


def test_hub_start_posts_a_timed_stop_for_training_runs_too():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fns = (_extract_js_function(_PAGE, "escHtml") + "\n"
           + _extract_js_function(_PAGE, "computeMaxPerRun") + "\n"
           + _extract_js_function(_PAGE, "stopAfterSeconds") + "\n"
           + _extract_js_function(_PAGE, "startRunFromControls"))
    script = (
        "const els = {hint:{textContent:''}, mode:{value:'training'}, unlimited:{checked:true}, "
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
        "body": {"mode": "training", "apps": ["hinge"], "max_per_run": 0,
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
                        lambda samples, **kwargs: {"status": "ok", "labels": len(samples),
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


def test_bug_report_generation_reports_failure_instead_of_an_unhandled_rejection():
    """getReport() was the lone fetch site in this file with no error path -- a hub that is
    gone/unreachable at the exact moment a bug report is wanted must not leave a dead button
    and an unhandled promise rejection behind it; it must say so and hand back null."""
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    fn = _extract_js_function(_PAGE, "getReport")
    script = (
        "const hint={textContent:''}; const desc={value:''};\n"
        "function $(selector){return selector==='#bughint'?hint:selector==='#bugdesc'?desc:null;}\n"
        "async function fetch(){throw new TypeError('Failed to fetch');}\n"
        + fn + "\n"
        "getReport().then(md => console.log(JSON.stringify({md, hint: hint.textContent})));\n"
    )
    result = _run_node(script)
    assert result["md"] is None
    assert "could not contact the local hub" in result["hint"]


def test_unchanged_checkpoint_poll_does_not_rebuild_no_store_review_images():
    """Only a real card/action transition may replace the mounted review image.

    Driven through the FULL per-poll prologue on purpose: tick() calls clearHubConnectionLost()
    on every successful /api/status before tickTrainingCheckpoint, so a version of this test that
    calls renderTrainingCheckpointFromPoll alone passes vacuously against a build that wipes the
    fingerprint once a second from there (found 2026-09-04).
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    functions = "\n".join((
        _extract_js_function(_PAGE, "trainingCheckpointKey"),
        _extract_js_function(_PAGE, "trainingPollRenderFingerprint"),
        _extract_js_function(_PAGE, "renderTrainingCheckpointFromPoll"),
        _extract_js_function(_PAGE, "clearHubConnectionLost"),
    ))
    script = (
        "let _trainingPollRenderSignature='',renders=[];"
        "const err={textContent:''};function $(selector){return selector==='#err'?err:null;}"
        "function renderTrainingCheckpoint(card){renders.push(card.action);}\n"
        + functions + "\n"
        "const card={run_id:'run',app:'hinge',profile_token:'profile',approval_token:'approval',"
        "phase:'waiting_training_decision',action:'ready',pending:true,profile_image_count:12};"
        "const poll=next=>{clearHubConnectionLost();renderTrainingCheckpointFromPoll(next);};"
        "poll(card);poll({...card});poll({...card});"
        "poll({...card,phase:'executing_training_decision',"
        "action:'executing',pending:false,command:'like'});"
        "poll({...card,phase:'executing_training_decision',"
        "action:'executing',pending:false,command:'like'});"
        "console.log(JSON.stringify(renders));"
    )
    assert _run_node(script) == ["ready", "executing"]


def test_unchanged_polls_let_the_image_retry_timer_actually_fire():
    """The 2000ms image retry must survive a 1s poll cadence (found 2026-09-04).

    renderTrainingCheckpoint re-applies a latched image failure onto the frame it just rebuilt,
    and that re-application clears and re-arms scheduleTrainingImageRetry's timer. While the poll
    path repainted every second the 2000ms retry was therefore rescheduled before it could ever
    run, so a transient /api/training/image failure still locked Like/Dislike for the rest of the
    profile -- the exact failure the retry was added to end.
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    functions = "\n".join((
        _extract_js_function(_PAGE, "trainingCheckpointKey"),
        _extract_js_function(_PAGE, "trainingPollRenderFingerprint"),
        _extract_js_function(_PAGE, "renderTrainingCheckpointFromPoll"),
        _extract_js_function(_PAGE, "clearHubConnectionLost"),
        _extract_js_function(_PAGE, "scheduleTrainingImageRetry"),
    ))
    script = (
        # virtual clock: the real 2000ms retry against a real 900ms poll cadence, no sleeping
        "let now=0,seq=0,timers=[];"
        "globalThis.setTimeout=(fn,ms)=>{const id=++seq;timers.push({id,at:now+ms,fn});return id;};"
        "globalThis.clearTimeout=id=>{timers=timers.filter(t=>t.id!==id);};"
        "function advance(ms){const target=now+ms;for(;;){"
        "const due=timers.filter(t=>t.at<=target).sort((a,b)=>a.at-b.at)[0];if(!due)break;"
        "timers=timers.filter(t=>t!==due);now=due.at;due.fn();}now=target;}\n"
        "let _trainingPollRenderSignature='',_trainingImageRetryTimer=null,_trainingCheckpoint=null,"
        "_trainingImageFailedIndex=0,_trainingImageFailedGeneration=7;const renderedAt=[];"
        "const err={textContent:''};function $(selector){return selector==='#err'?err:null;}"
        # stand-in for the real renderer's tail: it re-applies the latched failure, which re-arms
        "function renderTrainingCheckpoint(card){renderedAt.push(now);_trainingCheckpoint=card;"
        "if(_trainingImageFailedIndex!=null)"
        "scheduleTrainingImageRetry(_trainingImageFailedGeneration,_trainingImageFailedIndex);}\n"
        + functions + "\n"
        "const card={run_id:'run',app:'hinge',profile_token:'profile',approval_token:'approval',"
        "phase:'waiting_training_decision',action:'ready',pending:true,profile_image_count:2};"
        "const poll=next=>{clearHubConnectionLost();renderTrainingCheckpointFromPoll(next);};"
        "poll(card);"
        "advance(900);poll({...card});"
        "advance(900);poll({...card});"
        "advance(900);poll({...card});"
        "console.log(JSON.stringify(renderedAt));"
    )
    # t=0 the first paint arms the retry for t=2000; the 900/1800/2700 polls dedupe, so the
    # retry runs on time.  A repaint-per-poll build renders at 0/900/1800/2700 and never at 2000.
    assert _run_node(script) == [0, 2000]


def test_hub_disconnect_replaces_stale_in_progress_state_and_disables_actions():
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    functions = "\n".join((
        _extract_js_function(_PAGE, "escHtml"),
        _extract_js_function(_PAGE, "renderHubConnectionLost"),
        _extract_js_function(_PAGE, "clearHubConnectionLost"),
    ))
    script = (
        "function element(){return {textContent:'',innerHTML:'',style:{},disabled:false};}"
        "const elements={};for(const id of ['#err','#runbanner','#start','#stop','#traininglike',"
        "'#trainingdislike','#trainingactionhint'])elements[id]=element();"
        "function $(selector){return elements[selector]||null;}\n" + functions + "\n"
        "renderHubConnectionLost();const lost={error:elements['#err'].textContent,"
        "banner:elements['#runbanner'].innerHTML,start:elements['#start'].disabled,"
        "stop:elements['#stop'].disabled,like:elements['#traininglike'].disabled,"
        "dislike:elements['#trainingdislike'].disabled,hint:elements['#trainingactionhint'].textContent};"
        "clearHubConnectionLost();console.log(JSON.stringify({lost,cleared:elements['#err'].textContent}));"
    )
    result = _run_node(script)
    assert "Local hub disconnected" in result["lost"]["error"]
    assert "last displayed run and decision state is stale" in result["lost"]["banner"]
    assert result["lost"]["start"] is result["lost"]["stop"] is True
    assert result["lost"]["like"] is result["lost"]["dislike"] is True
    assert "do not repeat a physical action" in result["lost"]["hint"]
    assert result["cleared"] == ""


def test_a_failed_checkpoint_refresh_does_not_pin_its_error_on_a_live_card():
    """The sibling of the /api/status lockout, on /api/training/checkpoint (found 2026-09-04).

    tickTrainingCheckpoint's catch and submitTrainingAction's retry both paint a message straight
    through renderTrainingCheckpoint, bypassing the poll fingerprint that still matches the card
    underneath. A checkpoint waiting on the operator never changes, so the next identical poll
    short-circuits and "Could not refresh decision status" stays on a card whose endpoint has
    already recovered. The message repaint has to invalidate the signature -- once, so the
    following identical polls still dedupe and the review PNGs stay mounted.
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    functions = "\n".join((
        _extract_js_function(_PAGE, "escHtml"),
        _extract_js_function(_PAGE, "safeCheckpointImageDataUrl"),
        _extract_js_function(_PAGE, "trainingCheckpointKey"),
        _extract_js_function(_PAGE, "trainingPollRenderFingerprint"),
        _extract_js_function(_PAGE, "resetTrainingIdempotencyIfCardChanged"),
        _extract_js_function(_PAGE, "resetTrainingBusyIfCardChanged"),
        _extract_js_function(_PAGE, "trainingActionBusyFor"),
        _extract_js_function(_PAGE, "checkpointReviewImages"),
        _extract_js_function(_PAGE, "syncTrainingImageState"),
        _extract_js_function(_PAGE, "renderTrainingCheckpoint"),
        _extract_js_function(_PAGE, "renderTrainingCheckpointFromPoll"),
        _extract_js_function(_PAGE, "clearHubConnectionLost"),
    ))
    script = (
        "const panel={style:{display:''},innerHTML:''};"
        "const layout={classList:{toggle(){}}};"
        "const document={querySelector:()=>layout};"
        "function $(selector){return selector==='#trainingpanel'?panel:null;}"
        "function setTrainingImageZoom(){}"
        "let _trainingCheckpoint=null,_trainingActionBusy=false,_trainingBusyKey='',"
        "_trainingBusyRequest=0,_trainingImageKey='',_trainingImageIndex=0,"
        "_trainingImageFailedIndex=null,_trainingImageRenderGeneration=0,"
        "_trainingImageFailedGeneration=null,_trainingPollRenderSignature='';"
        "const _trainingIdempotency=new Map();\n" + functions + "\n"
        "const hint=()=>{const m=/id=\"trainingactionhint\"[^>]*>([^<]*)</.exec(panel.innerHTML);"
        "return m?m[1]:null;};"
        "const card={run_id:'run',app:'hinge',profile_token:'profile',approval_token:'approval',"
        "phase:'waiting_training_decision',action:'ready',pending:true,profile_image_count:2,"
        "item:3,opener:'a written opener',image_data_url:'data:image/png;base64,AAAA'};"
        "const poll=next=>{clearHubConnectionLost();renderTrainingCheckpointFromPoll(next);};"
        "poll(card);const healthy=hint();"
        # the /api/training/checkpoint fetch fails: the catch paints directly, not via the poll
        "renderTrainingCheckpoint(_trainingCheckpoint,"
        "'Could not refresh decision status. Retry your choice or Stop the run.');"
        "const failed=hint();"
        "poll({...card});const recovered=hint();"       # endpoint recovered; card never changed
        "panel.innerHTML='SENTINEL';poll({...card});"    # and the dedupe must resume immediately
        "console.log(JSON.stringify({healthy,failed,recovered,"
        "deduped:panel.innerHTML==='SENTINEL'}));"
    )
    result = _run_node(script)
    assert result["healthy"] == ""
    assert "Could not refresh decision status" in result["failed"]
    assert result["recovered"] == "", "the recovered poll must repaint and clear the stale error"
    assert result["deduped"] is True, "one repaint, not a repaint on every later poll"


def test_a_transient_disconnect_does_not_leave_the_decision_buttons_dead():
    """Reconnecting must give the operator their Like/Dislike back (found 2026-09-04).

    `renderHubConnectionLost()` disables the two Training buttons directly, and the ONLY code
    that rebuilds them is `renderTrainingCheckpoint` -- which the poll entry point skips whenever
    the checkpoint fingerprint is unchanged. A checkpoint waiting on the operator does not change,
    so one failed `/api/status` fetch used to leave the controls dead, and the hint reading "Local
    hub disconnected", with the run still live. Only a page reload recovered it.
    """
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    functions = "\n".join((
        _extract_js_function(_PAGE, "escHtml"),
        _extract_js_function(_PAGE, "trainingCheckpointKey"),
        _extract_js_function(_PAGE, "trainingPollRenderFingerprint"),
        _extract_js_function(_PAGE, "renderTrainingCheckpointFromPoll"),
        _extract_js_function(_PAGE, "renderHubConnectionLost"),
        _extract_js_function(_PAGE, "clearHubConnectionLost"),
    ))
    script = (
        "let _trainingPollRenderSignature='',renders=0;"
        "function element(){return {textContent:'',innerHTML:'',style:{},disabled:false};}"
        "const elements={};for(const id of ['#err','#runbanner','#start','#stop','#traininglike',"
        "'#trainingdislike','#trainingactionhint'])elements[id]=element();"
        "function $(selector){return elements[selector]||null;}"
        # stand-in for the real renderer: it is what re-enables the buttons
        "function renderTrainingCheckpoint(card){renders++;"
        "elements['#traininglike'].disabled=false;elements['#trainingdislike'].disabled=false;"
        "elements['#trainingactionhint'].textContent='ready';}\n" + functions + "\n"
        "const card={run_id:'run',app:'hinge',profile_token:'profile',approval_token:'approval',"
        "phase:'waiting_training_decision',action:'ready',pending:true,profile_image_count:12};"
        # a normal poll paints the card, then one fetch fails, then the SAME card comes back
        "renderTrainingCheckpointFromPoll(card);"
        "renderHubConnectionLost();"
        "const lost={like:elements['#traininglike'].disabled};"
        "clearHubConnectionLost();"
        "renderTrainingCheckpointFromPoll({...card});"
        "console.log(JSON.stringify({lost,renders,"
        "like:elements['#traininglike'].disabled,dislike:elements['#trainingdislike'].disabled,"
        "hint:elements['#trainingactionhint'].textContent}));"
    )
    result = _run_node(script)
    assert result["lost"]["like"] is True, "the disconnect must still disable the decision"
    assert result["renders"] == 2, "the reconnecting poll has to repaint, not short-circuit"
    assert result["like"] is False and result["dislike"] is False
    assert result["hint"] == "ready"


def test_bug_report_buttons_never_write_or_download_a_failed_report():
    # Both callers must check getReport()'s null before touching the clipboard or a Blob. This
    # is a source-order check (not a node run) because both handlers are anonymous arrow
    # functions assigned directly to an onclick property, matching this file's established
    # pattern for asserting other anonymous click-wiring (e.g. imageZoomButton.onclick above).
    copy_block = re.search(r"\$\('#bugcopy'\)\.onclick = async \(\) => \{(.*?)\n\};",
                            _PAGE, re.S)
    dl_block = re.search(r"\$\('#bugdl'\)\.onclick = async \(\) => \{(.*?)\n\};", _PAGE, re.S)
    assert copy_block and dl_block
    for block, guarded_call in ((copy_block, "navigator.clipboard.writeText"),
                                 (dl_block, "document.createElement")):
        body = block.group(1)
        get_at = body.index("await getReport()")
        guard_at = body.index("if (md == null) return;")
        call_at = body.index(guarded_call)
        assert get_at < guard_at < call_at


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
        # endpoint tests on the 5s default.  With xdist's full worker fan-out, optional ML
        # imports can contend for CPU/disk long enough to exceed the normal liveness allowance;
        # this one integration check therefore allows a documented one-minute cold start while
        # retaining the endpoint's exact success/content assertions below.
        code, md = _get(base, "/api/bugreport", timeout=60)
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
        {"label": "Bumble", "reason": "Both Training and Auto are not calibrated."},
        {"label": "Hostile", "reason": '<img src=x onerror="alert(1)">'},
        {"label": "No reason"},
    ]
    script = _picker_script(
        "console.log(JSON.stringify(" + json.dumps(cases)
        + ".map(pendingPlatformNoteHtml)));\n",
        "escHtml", "pendingPlatformNoteHtml",
    )
    notes = _run_node(script)

    assert "Both Training and Auto are not calibrated." in notes[0]
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
        "modes": {"training": True, "auto": False},
        "mode_reasons": {"training": None, "auto": hostile_reason},
    }
    script = _picker_script(
        "const cfg={kinds:[{kind:'android',platforms:[" + json.dumps(platform) + "]}]};\n"
        "let sel={kind:'android',app:'hinge'};\n"
        "const options=[{value:'training'},{value:'auto'}];\n"
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

    assert result["value"] == "training"
    assert result["disabled"] is True and result["hidden"] is True
    assert result["optionTitle"] == hostile_reason
    assert result["selectTitle"].startswith("Auto unavailable:")
    assert "<img" not in result["hint"]
    assert "&lt;img" in result["hint"] and "&quot;alert(1)&quot;" in result["hint"]
