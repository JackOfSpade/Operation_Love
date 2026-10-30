"""Each listener owns its run, and startup failures always release the listener."""

import http.client
import json
import threading
from types import SimpleNamespace

import pytest

from operation_love.hub import server as hub_server


def test_bound_hubs_keep_their_own_state(monkeypatch):
    servers = []
    threads = []
    try:
        for name in ("first", "second"):
            state = SimpleNamespace(snapshot=lambda name=name: {"name": name})
            monkeypatch.setattr(hub_server._Handler, "state", state)
            server = hub_server._bind("127.0.0.1", 8799)
            servers.append(server)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            threads.append(thread)
            thread.start()

        for server, name in zip(servers, ("first", "second"), strict=True):
            connection = http.client.HTTPConnection(*server.server_address, timeout=5)
            try:
                connection.request("GET", "/api/status")
                response = connection.getresponse()
                assert response.status == 200
                assert json.loads(response.read()) == {"name": name}
            finally:
                connection.close()
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)


def test_serve_releases_listener_when_state_initialization_fails(monkeypatch):
    closed = []
    server = SimpleNamespace(server_close=lambda: closed.append(True))
    monkeypatch.setattr(hub_server, "_bind", lambda *args, **kwargs: server)

    def failed_state(config_path):
        raise RuntimeError("state unavailable")

    monkeypatch.setattr(hub_server, "HubState", failed_state)
    with pytest.raises(RuntimeError, match="state unavailable"):
        hub_server.serve(open_browser=False)
    assert closed == [True]


def test_bind_releases_listener_when_token_initialization_fails(monkeypatch):
    closed = []
    server = SimpleNamespace(server_close=lambda: closed.append(True))
    monkeypatch.setattr(hub_server, "_HubHTTPServer", lambda *args: server)

    def unavailable_token(size):
        raise OSError("random source unavailable")

    monkeypatch.setattr(hub_server.secrets, "token_urlsafe", unavailable_token)
    with pytest.raises(OSError, match="random source unavailable"):
        hub_server._bind("127.0.0.1", 8799)
    assert closed == [True]


def test_serve_releases_listener_even_when_run_cleanup_fails(monkeypatch):
    closed = []
    server = SimpleNamespace(
        server_address=("127.0.0.1", 8799),
        serve_forever=lambda: None,
        server_close=lambda: closed.append(True),
    )
    monkeypatch.setattr(hub_server, "_bind", lambda *args, **kwargs: server)

    def failed_stop():
        raise RuntimeError("stop unavailable")

    monkeypatch.setattr(
        hub_server, "HubState", lambda path: SimpleNamespace(stop=failed_stop)
    )
    with pytest.raises(RuntimeError, match="stop unavailable"):
        hub_server.serve(open_browser=False)
    assert closed == [True]


def test_serve_stops_only_the_state_it_created(monkeypatch):
    stopped = []
    state = SimpleNamespace(
        stop=lambda: stopped.append("own"), wait_for_run=lambda timeout: True
    )
    unrelated = SimpleNamespace(stop=lambda: stopped.append("unrelated"))
    monkeypatch.setattr(hub_server._Handler, "state", unrelated)
    server = SimpleNamespace(
        server_address=("127.0.0.1", 8799),
        serve_forever=lambda: None,
        server_close=lambda: None,
    )
    monkeypatch.setattr(hub_server, "_bind", lambda *args, **kwargs: server)
    monkeypatch.setattr(hub_server, "HubState", lambda path: state)

    hub_server.serve(open_browser=False)
    assert server.state is state
    assert hub_server._Handler.state is unrelated
    assert stopped == ["own"]


@pytest.mark.parametrize("failure", ["serve", "stop"])
def test_serve_cancels_browser_open_before_failed_server_cleanup(monkeypatch, failure):
    events = []

    class Timer:
        def __init__(self, delay, target):
            self.daemon = False

        def start(self):
            events.append("scheduled")

        def cancel(self):
            events.append("cancelled")

    def serve_forever():
        if failure == "serve":
            raise RuntimeError("server failed")

    def stop():
        events.append("stopped")
        if failure == "stop":
            raise RuntimeError("cleanup failed")

    server = SimpleNamespace(
        server_address=("127.0.0.1", 8799),
        serve_forever=serve_forever,
        server_close=lambda: events.append("closed"),
    )
    state = SimpleNamespace(stop=stop, wait_for_run=lambda timeout: True)
    monkeypatch.setattr(hub_server, "_bind", lambda *args, **kwargs: server)
    monkeypatch.setattr(hub_server, "HubState", lambda path: state)
    monkeypatch.setattr(hub_server.threading, "Timer", Timer)

    with pytest.raises(RuntimeError):
        hub_server.serve(open_browser=True)
    assert events == ["scheduled", "cancelled", "stopped", "closed"]
