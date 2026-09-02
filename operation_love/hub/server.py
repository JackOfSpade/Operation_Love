"""Hub HTTP server — stdlib http.server request handling and process lifecycle.

The browser is only the face: start/stop POST to this local server, which does
the real work via HubState. UI choice (this being a plain http.server page) has
no bearing on what the backend can do.
"""
from __future__ import annotations

import errno
import hmac
import html
import json
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .page import _PAGE
from .state import (HubState, validate_apps, validate_max_per_run,
                    validate_browser_client_id, validate_stop_after_seconds)

_BROWSER_SHUTDOWN_GRACE_S = 1.5
_BROWSER_STALE_CHECK_S = 5.0
# Hub requests only carry a few IDs and small action records.  Bound the body before reading it
# so a malformed/local client cannot make a ThreadingHTTPServer worker buffer an unbounded body.
_MAX_REQUEST_BODY_BYTES = 1_048_576
_CSRF_TOKEN_FIELD = "_csrf_token"
_CSRF_TOKEN_PLACEHOLDER = "__OPERATION_LOVE_CSRF_TOKEN__"
_POST_FIELDS = {
    "/api/start": {"mode", "apps", "max_per_run", "stop_after_seconds"},
    "/api/stop": set(),
    "/api/training/action": {
        "command", "run_id", "app", "profile_token", "approval_token", "idempotency_token",
    },
    "/api/training/remove-latest": set(),
    "/api/hub/open": {"id"},
    "/api/hub/ping": {"id"},
    "/api/hub/closed": {"id"},
}
# Ctrl-C's grace period for an active run to flush/save before the process exits
# anyway. Generous enough to cover the run's own bounded shutdown (supervisor.py
# joins each of up to 2 app workers for up to 30s each) plus real save time, while
# still guaranteeing Ctrl-C is never fully unresponsive if something is wedged
# (e.g. a hung network call inside store.flush(), which has no timeout of its own).
_SHUTDOWN_SAVE_TIMEOUT_S = 90.0


class _HubHTTPServer(ThreadingHTTPServer):
    csrf_token: str


def _json_object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """Build a JSON object while rejecting ambiguous duplicate keys.

    The control API has a strict one-value schema. Keeping a single interpretation at the
    parser boundary prevents different layers from acting on different copies of a key.
    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


class _Handler(BaseHTTPRequestHandler):
    state: HubState | None = None           # set by serve()

    def end_headers(self) -> None:
        """Attach browser hardening to every response, including empty/error responses.

        Keep CSP deliberately limited to framing: the self-contained Hub currently uses an
        inline script/style, and adding a default-src/script-src policy without nonces would
        disable its own controls. ``frame-ancestors`` alone closes clickjacking without that
        compatibility break; X-Frame-Options covers older browsers.
        """
        self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def _send(self, code: int, body, ctype: str) -> None:
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")   # always serve the live build (no stale UI)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj), "application/json")

    def _training_image(self, data: bytes) -> None:
        """Serve one capability-bound review PNG without repeating it in JSON.

        The pixels are sensitive profile-review material, so never retain them in a browser
        cache even though the capability token itself is bound to one checkpoint.
        """
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _request_authority(self) -> tuple[str, int] | None:
        """Return the normalized loopback Host authority, or ``None`` if invalid.

        The hub is deliberately local-only.  Checking the HTTP Host header as well as
        binding to loopback prevents a public hostname that resolves to 127.0.0.1 from
        using the browser as a DNS-rebinding bridge into the control API.
        """
        values = self.headers.get_all("Host", [])
        if len(values) != 1:
            return None
        port = int(self.server.server_address[1])
        authorities = {
            f"127.0.0.1:{port}": ("127.0.0.1", port),
            f"localhost:{port}": ("localhost", port),
            f"[::1]:{port}": ("::1", port),
        }
        if port == 80:
            authorities.update({
                "127.0.0.1": ("127.0.0.1", port),
                "localhost": ("localhost", port),
                "[::1]": ("::1", port),
            })
        return authorities.get(values[0].strip().lower())

    def _require_local_host(self) -> tuple[str, int] | None:
        authority = self._request_authority()
        if authority is None:
            self._json({"ok": False, "msg": "invalid Host for local hub"}, 403)
        return authority

    def _origin_matches(self, authority: tuple[str, int]) -> bool:
        """Accept an absent Origin for authenticated non-browser clients.

        Browsers send Origin on these POSTs.  If present it must identify the exact
        authority in Host: even two loopback aliases (localhost and 127.0.0.1) are
        different web origins.
        """
        values = self.headers.get_all("Origin", [])
        if not values:
            return True
        if len(values) != 1:
            return False
        try:
            origin = urlsplit(values[0])
            origin_port = origin.port
        except ValueError:
            return False
        if (
            origin.scheme.lower() != "http"
            or origin.username is not None
            or origin.password is not None
            or origin.query
            or origin.fragment
            or origin.path
            or origin.hostname is None
        ):
            return False
        if origin_port is None:
            origin_port = 80
        return (origin.hostname.lower(), origin_port) == authority

    def _render_page(self) -> str:
        token = html.escape(str(getattr(self.server, "csrf_token", "")), quote=True)
        if _PAGE.count(_CSRF_TOKEN_PLACEHOLDER) != 1:
            raise RuntimeError("hub page must contain exactly one CSRF token placeholder")
        return _PAGE.replace(_CSRF_TOKEN_PLACEHOLDER, token)

    def do_GET(self) -> None:
        if self._require_local_host() is None:
            return
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, self._render_page(), "text/html; charset=utf-8")
        elif path == "/favicon.ico":
            self.send_response(204)             # no icon; avoids a console 404
            self.end_headers()
        elif path == "/api/status":
            self._json(self.state.snapshot())
        elif path == "/api/training/checkpoint":
            from urllib.parse import parse_qs, urlparse
            query = parse_qs(urlparse(self.path).query)
            self._json(self.state.training_action_snapshot(
                run_id=query.get("run_id", [None])[0], app=query.get("app", [None])[0]))
        elif path == "/api/training/image":
            from urllib.parse import parse_qs, urlparse
            query = parse_qs(urlparse(self.path).query)
            try:
                index = int(query.get("index", [""])[0])
            except (TypeError, ValueError):
                index = -1
            data = self.state.training_profile_review_image(
                run_id=query.get("run_id", [""])[0],
                app=query.get("app", [""])[0],
                profile_token=query.get("profile_token", [""])[0], index=index)
            if data is None:
                self._json({"error": "training profile image not found"}, 404)
            else:
                self._training_image(data)
        elif path == "/api/config":
            self._json(self.state.config_defaults())
        elif path == "/api/eval":
            self._json(self.state.eval_snapshot())
        elif path == "/api/logs":
            from ..bugreport import recent_logs   # tee'd stdout/stderr ring (install_log_capture)
            self._json({"lines": recent_logs(300)})
        elif path == "/api/bugreport":
            from ..bugreport import build_report
            desc = ""
            if "?" in self.path:
                from urllib.parse import parse_qs, urlparse
                desc = (parse_qs(urlparse(self.path).query).get("desc", [""])[0])
            md = build_report(self.state, description=desc, config_path=self.state.config_path)
            self._send(200, md, "text/markdown; charset=utf-8")
        else:
            self._json({"error": "not found"}, 404)

    def _schedule_shutdown_if_tab_stayed_closed(self) -> None:
        state = self.state
        server = self.server

        def _target():
            time.sleep(_BROWSER_SHUTDOWN_GRACE_S)  # reload/new tab gets a moment to register
            disposition = state.browser_close_shutdown_disposition() if state else "shutdown"
            if disposition == "client_reopened":
                return
            if disposition == "preserve_training_approval":
                host, port = server.server_address[:2]
                print("Hub: browser hub tab closed while a Training approval is waiting; "
                      "keeping the run and verified checkpoint alive. Reopen "
                      f"http://{host}:{port}/ to decide, or press Stop deliberately.")
                return
            if state and state.is_running():
                print("Hub: browser hub tab closed; stopping active run before shutdown.")
                state.stop()
                # Bounded the same way as the Ctrl-C path (_SHUTDOWN_SAVE_TIMEOUT_S): a wedged
                # worker must not block this thread forever, or server.shutdown() below is
                # never reached and the process lives on invisibly after the tab is gone —
                # defeating the "close tab -> hub exits -> launcher closes the Terminal tab" UX.
                state.wait_for_run(timeout=_SHUTDOWN_SAVE_TIMEOUT_S)
            else:
                print("Hub: browser hub tab closed; shutting down.")
            server.shutdown()

        threading.Thread(target=_target, name="hub-tab-close-shutdown", daemon=True).start()

    def _start_stale_browser_watch(self) -> None:
        state = self.state

        def _target():
            while state and state.browser_stale_watch_active():
                time.sleep(_BROWSER_STALE_CHECK_S)
                if state.expire_stale_browser_clients():
                    self._schedule_shutdown_if_tab_stayed_closed()
                    return

        try:
            threading.Thread(target=_target, name="hub-tab-stale-watch", daemon=True).start()
        except RuntimeError as exc:
            # browser_client_opened() claimed the one active watcher before this allocation.
            # Release that claim so a later heartbeat can retry rather than making stale-tab
            # shutdown silently unavailable for the rest of this hub process.
            if state:
                state.browser_stale_watch_start_failed()
            print(f"Hub: could not start browser liveness watch ({exc})")

    def do_POST(self) -> None:
        authority = self._require_local_host()
        if authority is None:
            return
        if not self._origin_matches(authority):
            self._json({"ok": False, "msg": "Origin does not match the local hub"}, 403)
            return
        content_types = self.headers.get_all("Content-Type", [])
        if len(content_types) != 1 or self.headers.get_content_type() != "application/json":
            self._json({"ok": False, "msg": "Content-Type must be application/json"}, 415)
            return
        content_lengths = self.headers.get_all("Content-Length", [])
        if len(content_lengths) > 1:
            self._json({"ok": False, "msg": "Content-Length must not be duplicated"}, 400)
            return
        try:
            length = int(content_lengths[0] or 0) if content_lengths else 0
        except ValueError:
            self._json({"ok": False, "msg": "Content-Length must be an integer"}, 400)
            return
        if length < 0 or length > _MAX_REQUEST_BODY_BYTES:
            self._json({"ok": False, "msg": "request body is too large"}, 413)
            return
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw, object_pairs_hook=_json_object_without_duplicate_keys) if raw else {}
        except ValueError:
            body = None
        if not isinstance(body, dict):
            self._json({"ok": False, "msg": "request body must be a JSON object"}, 400)
            return
        supplied_token = body.pop(_CSRF_TOKEN_FIELD, None)
        expected_token = getattr(self.server, "csrf_token", "")
        if (
            not isinstance(supplied_token, str)
            or not expected_token
            or not hmac.compare_digest(supplied_token, expected_token)
        ):
            self._json({"ok": False, "msg": "invalid CSRF token"}, 403)
            return
        allowed_fields = _POST_FIELDS.get(self.path)
        if allowed_fields is not None:
            unknown_fields = set(body) - allowed_fields
            if unknown_fields:
                self._json({
                    "ok": False,
                    "msg": f"unknown request field(s): {sorted(unknown_fields)}",
                }, 400)
                return
        if self.path in {"/api/hub/open", "/api/hub/ping", "/api/hub/closed"}:
            valid_client, client_id, client_error = validate_browser_client_id(body.get("id"))
            if not valid_client or client_id is None:
                self._json({"ok": False, "msg": client_error or "invalid browser client id"}, 400)
                return
            body["id"] = client_id
        if self.path == "/api/start":
            valid_cap, mpr, cap_error = validate_max_per_run(body.get("max_per_run"))
            if not valid_cap:
                self._json({"ok": False, "msg": cap_error}, 400)
                return
            valid_timeout, stop_after_seconds, timeout_error = validate_stop_after_seconds(
                body.get("stop_after_seconds"))
            if not valid_timeout:
                self._json({"ok": False, "msg": timeout_error}, 400)
                return
            valid_apps, apps, apps_error = validate_apps(body.get("apps"))
            if not valid_apps:
                self._json({"ok": False, "msg": apps_error}, 400)
                return
            ok, msg = self.state.start(body.get("mode"), apps, mpr,
                                       stop_after_seconds)
            self._json({"ok": ok, "msg": msg}, 200 if ok else 409)
        elif self.path == "/api/stop":
            # HubState.stop() now returns (False, "no run is active") instead of an
            # unconditional (True, "stopping") -- mirror /api/start's existing convention
            # (200 on success, 409 on a refusal) instead of a bare 200 that told the caller
            # a stop it never performed had "worked".
            ok, msg = self.state.stop()
            self._json({"ok": ok, "msg": msg}, 200 if ok else 409)
        elif self.path == "/api/training/action":
            ok, result, code = self.state.submit_training_action(body)
            self._json({"ok": ok, "result": result}, code)
        elif self.path == "/api/training/remove-latest":
            ok, result = self.state.remove_latest_training_label()
            self._json({"ok": ok, "result": result}, 200 if ok else 409)
        elif self.path == "/api/hub/open":
            start_watch = self.state.browser_client_opened(body.get("id"))
            self._json({"ok": True})
            if start_watch:
                self._start_stale_browser_watch()
        elif self.path == "/api/hub/ping":
            start_watch = self.state.browser_client_ping(body.get("id"))
            self._json({"ok": True})
            if start_watch:
                self._start_stale_browser_watch()
        elif self.path == "/api/hub/closed":
            should_shutdown = self.state.browser_client_closed(body.get("id"))
            self._json({"ok": True})
            if should_shutdown:
                self._schedule_shutdown_if_tab_stayed_closed()
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, *_):              # silence default stderr request logging
        pass


def _bind(host: str, port: int) -> _HubHTTPServer:
    if not isinstance(host, str) or host not in {"127.0.0.1", "localhost"}:
        raise ValueError("Hub host must be localhost or 127.0.0.1")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("Hub port must be an integer from 1 to 65535")
    last_port = min(port + 199, 65535)
    for p in range(port, last_port + 1):       # find a free port near the default
        try:
            httpd = _HubHTTPServer((host, p), _Handler)
            httpd.csrf_token = secrets.token_urlsafe(32)
            return httpd
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE or getattr(exc, "winerror", None) == 10048:
                continue
            raise
    raise SystemExit(f"Hub: no free port in {port}..{last_port}")


def serve(config_path: str = "config.yaml", host: str = "127.0.0.1",
          port: int = 8765, open_browser: bool = True) -> None:
    from .._warnings import configure_warnings
    from ..bugreport import install_log_capture
    configure_warnings()
    install_log_capture()                   # capture logs so bug reports include them
    _Handler.state = HubState(config_path)
    httpd = _bind(host, port)
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"Hub: Operation Love control hub → {url}   (Ctrl-C to quit)")
    try:
        if open_browser:
            browser_timer = threading.Timer(0.6, lambda: webbrowser.open(url))
            # The timer is only a convenience after the server is already live. It must not keep a
            # process alive after an immediate shutdown or a failed serve_forever startup.
            browser_timer.daemon = True
            try:
                browser_timer.start()
            except RuntimeError as exc:
                # A thread-resource failure must not leak the live listener merely because an
                # optional browser tab could not be scheduled. The hub remains usable at its
                # printed URL, and the surrounding finally releases the socket if it exits.
                print(f"Hub: could not open browser automatically ({exc}); use {url}")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nHub: shutting down…")
    finally:
        if _Handler.state:
            _Handler.state.stop()
            # Let the active run flush/save before the process exits — but bounded,
            # so Ctrl-C can never hang forever if something in the run is wedged.
            if not _Handler.state.wait_for_run(timeout=_SHUTDOWN_SAVE_TIMEOUT_S):
                print(f"Hub: run did not finish saving within "
                      f"{_SHUTDOWN_SAVE_TIMEOUT_S:.0f}s; exiting anyway "
                      "(data may not be fully flushed).")
        try:
            httpd.shutdown()
        finally:
            # shutdown() stops serve_forever but deliberately leaves the listening socket open.
            # Always release it so a stopped in-process hub can rebind its port immediately.
            httpd.server_close()
