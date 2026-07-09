"""Hub HTTP endpoints — offline smoke test (no real run started).

Spins the stdlib server in a thread and hits /, /api/config, /api/status,
/api/stop, and a 404. We never POST /api/start (that would launch a real run /
browser); start/stop wiring is covered by HubState's logic.
"""
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from operation_love.hub import HubState, _Handler, _MAC_UPDATE_RUN, _PAGE, _bind


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


def test_hub_tab_close_shuts_server_after_last_client():
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
    ok, _ = st.start(mode="auto", apps=["bumble"], max_per_run=8)
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
    expected = _MAC_UPDATE_RUN.replace("__EXTRAS__", "ml,bq,bumble")
    assert Path("Operation Love.command").read_text() == expected


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
