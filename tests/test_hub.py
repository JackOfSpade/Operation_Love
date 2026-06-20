"""Hub HTTP endpoints — offline smoke test (no real run started).

Spins the stdlib server in a thread and hits /, /api/config, /api/status,
/api/stop, and a 404. We never POST /api/start (that would launch a real run /
browser); start/stop wiring is covered by HubState's logic.
"""
import json
import threading
import urllib.error
import urllib.request

from operation_love.hub import HubState, _Handler, _bind


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


def test_hubstate_double_start_blocked():
    st = HubState("config.yaml")
    # fake a live run so the second start is rejected without launching anything
    st._thread = threading.Thread(target=lambda: __import__("time").sleep(0.3))
    st._thread.start()
    ok, msg = st.start()
    assert ok is False and "active" in msg
    st._thread.join()


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    sys.exit(1 if failed else 0)
