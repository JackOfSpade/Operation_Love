"""Hub HTTP server — stdlib http.server request handling and process lifecycle.

The browser is only the face: start/stop POST to this local server, which does
the real work via HubState. UI choice (this being a plain http.server page) has
no bearing on what the backend can do.
"""
from __future__ import annotations

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .page import _PAGE
from .state import HubState

_BROWSER_SHUTDOWN_GRACE_S = 1.5
_BROWSER_STALE_CHECK_S = 5.0


class _Handler(BaseHTTPRequestHandler):
    state: HubState | None = None           # set by serve()

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

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, _PAGE, "text/html; charset=utf-8")
        elif path == "/favicon.ico":
            self.send_response(204)             # no icon; avoids a console 404
            self.end_headers()
        elif path == "/api/status":
            self._json(self.state.snapshot())
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
            md = build_report(self.state, description=desc)
            self._send(200, md, "text/markdown; charset=utf-8")
        else:
            self._json({"error": "not found"}, 404)

    def _schedule_shutdown_if_tab_stayed_closed(self) -> None:
        state = self.state
        server = self.server

        def _target():
            time.sleep(_BROWSER_SHUTDOWN_GRACE_S)  # reload/new tab gets a moment to register
            if state and state.has_browser_clients():
                return
            if state and state.is_running():
                print("Hub: browser hub tab closed; stopping active run before shutdown.")
                state.stop()
                state.wait_for_run()
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

        threading.Thread(target=_target, name="hub-tab-stale-watch", daemon=True).start()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            body = {}
        if self.path == "/api/start":
            mpr = body.get("max_per_run")           # auto-mode per-run cap (0 = unlimited)
            try:
                mpr = int(mpr) if mpr is not None else None
            except (TypeError, ValueError):
                mpr = None
            ok, msg = self.state.start(body.get("mode"), body.get("apps"), mpr)
            self._json({"ok": ok, "msg": msg}, 200 if ok else 409)
        elif self.path == "/api/stop":
            ok, msg = self.state.stop()
            self._json({"ok": ok, "msg": msg})
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


def _bind(host: str, port: int) -> ThreadingHTTPServer:
    for p in range(port, port + 200):       # find a free port near the default
        try:
            return ThreadingHTTPServer((host, p), _Handler)
        except OSError:
            continue
    raise SystemExit(f"Hub: no free port in {port}..{port + 199}")


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
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nHub: shutting down…")
    finally:
        if _Handler.state:
            _Handler.state.stop()
            _Handler.state.wait_for_run()    # let the active run flush/save before the process exits
        httpd.shutdown()
