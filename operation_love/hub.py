"""Local control hub — double-click to open, no terminal needed.

Starts a localhost HTTP server and opens your browser to a control panel that
shows live status for every app and starts/stops runs (observe/auto). Pure
stdlib (http.server), so there's no extra dependency. The run executes in a
background thread in THIS process; the page polls /api/status.

    python -m operation_love hub                  # open the hub
    python -m operation_love hub --make-launchers # write a double-click launcher

The browser is only the face: start/stop POST to this local server, which does
the real work (launch the Bumble browser, and — once an AVD exists — boot the
Hinge emulator). UI choice has no bearing on what the backend can do.
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config as cfg_mod
from . import supervisor


class HubState:
    """Owns the (at most one) active run and exposes a JSON-able snapshot."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._status = None                 # RunStatus, captured via on_status
        self._error: str | None = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, mode: str | None = None, apps=None) -> tuple[bool, str]:
        with self._lock:
            if self.is_running():
                return False, "a run is already active"
            self._stop = threading.Event()
            self._status = None
            self._error = None
            stop = self._stop

            def _capture(st):
                with self._lock:
                    self._status = st

            def _target():
                try:
                    supervisor.run(self.config_path, stop_event=stop, on_status=_capture,
                                   mode=mode, enabled_apps=apps)
                except Exception as exc:  # noqa: BLE001
                    with self._lock:
                        self._error = f"{type(exc).__name__}: {exc}"

            self._thread = threading.Thread(target=_target, name="hub-run", daemon=True)
            self._thread.start()
            return True, "started"

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            if self._stop:
                self._stop.set()
        return True, "stopping"

    def snapshot(self) -> dict:
        with self._lock:
            running = self.is_running()
            status = self._status
            error = self._error
        return {
            "running": running,
            "error": error,
            "status": status.snapshot() if status else None,
        }

    def config_defaults(self) -> dict:
        try:
            cfg = cfg_mod.load(self.config_path)
            return {
                "mode": cfg.mode,
                "enabled_apps": cfg.enabled_apps,
                "all_apps": list(cfg.apps.keys()) or cfg.enabled_apps,
                "min_labels": cfg.ranker.min_labels_to_engage,
                "backend": cfg.storage.backend,
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}


class _Handler(BaseHTTPRequestHandler):
    state: HubState | None = None           # set by serve()

    def _send(self, code: int, body, ctype: str) -> None:
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj), "application/json")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, _PAGE, "text/html; charset=utf-8")
        elif path == "/api/status":
            self._json(self.state.snapshot())
        elif path == "/api/config":
            self._json(self.state.config_defaults())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            body = {}
        if self.path == "/api/start":
            ok, msg = self.state.start(body.get("mode"), body.get("apps"))
            self._json({"ok": ok, "msg": msg}, 200 if ok else 409)
        elif self.path == "/api/stop":
            ok, msg = self.state.stop()
            self._json({"ok": ok, "msg": msg})
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, *_):              # silence default stderr request logging
        pass


def _bind(host: str, port: int) -> ThreadingHTTPServer:
    for p in range(port, port + 20):        # find a free port near the default
        try:
            return ThreadingHTTPServer((host, p), _Handler)
        except OSError:
            continue
    raise SystemExit(f"[hub] no free port in {port}..{port + 19}")


def serve(config_path: str = "config.yaml", host: str = "127.0.0.1",
          port: int = 8765, open_browser: bool = True) -> None:
    _Handler.state = HubState(config_path)
    httpd = _bind(host, port)
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"[hub] Operation Love control hub → {url}   (Ctrl-C to quit)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[hub] shutting down…")
    finally:
        if _Handler.state:
            _Handler.state.stop()
        httpd.shutdown()


def make_launchers(config_path: str = "config.yaml") -> None:
    """Write a double-click launcher for THIS OS that opens the hub.

    There's no single cross-OS double-click file, so we emit the right stub for
    the current platform (all call the same `python -m operation_love hub`):
      macOS  -> "Operation Love.command"   (double-click; opens Terminal + browser)
      Windows-> "Operation Love.bat"        (uses pythonw -> no console window)
      Linux  -> "operation-love.desktop"    (Exec entry; Terminal=false)
    """
    import stat
    import sys
    from pathlib import Path

    py = sys.executable
    proj = Path(config_path).resolve().parent
    plat = sys.platform

    if plat == "darwin":
        f = proj / "Operation Love.command"
        f.write_text(f'#!/bin/bash\ncd "{proj}"\nexec "{py}" -m operation_love hub\n')
        f.chmod(f.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    elif plat.startswith("win"):
        pyw = Path(py).with_name("pythonw.exe")
        launcher = str(pyw) if pyw.exists() else py
        f = proj / "Operation Love.bat"
        f.write_text(f'@echo off\r\ncd /d "{proj}"\r\nstart "" "{launcher}" -m operation_love hub\r\n')
    else:  # linux / *bsd
        f = proj / "operation-love.desktop"
        f.write_text(
            "[Desktop Entry]\nType=Application\nName=Operation Love\n"
            f'Exec="{py}" -m operation_love hub\nPath={proj}\nTerminal=false\nCategories=Utility;\n')
        f.chmod(f.stat().st_mode | stat.S_IEXEC)

    print(f"[hub] wrote launcher: {f}")
    print("      double-click it to open the hub — no terminal needed.")


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Operation Love — hub</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#0f0f14; color:#e8e8ea;
         font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }
  .wrap { max-width:680px; margin:0 auto; padding:22px 18px 60px; }
  h1 { font-size:18px; margin:0 0 2px; letter-spacing:.3px; }
  .sub { opacity:.55; font-size:12px; margin-bottom:18px; }
  .card { background:#17171d; border:1px solid rgba(255,255,255,.09);
          border-radius:14px; padding:14px 16px; margin-bottom:14px; }
  .row { display:flex; justify-content:space-between; align-items:center; gap:10px; }
  .muted { opacity:.6; } .b { font-weight:600; }
  .pill { font-size:11px; padding:2px 9px; border-radius:999px; }
  .run { background:rgba(57,217,138,.2); color:#39d98a; }
  .stop { background:rgba(255,255,255,.12); color:#c9c9cf; }
  .bar { height:6px; background:rgba(255,255,255,.12); border-radius:4px; margin:8px 0 4px; overflow:hidden; }
  .bar > div { height:100%; border-radius:4px; transition:width .3s; }
  .apps { display:grid; gap:10px; }
  .app { background:#1c1c23; border:1px solid rgba(255,255,255,.07);
         border-radius:11px; padding:11px 13px; }
  .app .nm { text-transform:capitalize; font-weight:600; }
  .dec-like { color:#39d98a; } .dec-pass { color:#ff6b6b; } .dec-other { color:#c9c9cf; }
  .controls { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
  select, button { font:inherit; border-radius:9px; border:1px solid rgba(255,255,255,.14);
         background:#22222b; color:#e8e8ea; padding:8px 12px; }
  button { cursor:pointer; }
  button.primary { background:#2f6df0; border-color:#2f6df0; }
  button.danger { background:#b3322f; border-color:#b3322f; }
  button:disabled { opacity:.4; cursor:not-allowed; }
  label.chk { display:inline-flex; align-items:center; gap:6px; padding:6px 10px;
         background:#22222b; border:1px solid rgba(255,255,255,.14); border-radius:9px; cursor:pointer; }
  .err { color:#ff8585; font-size:12px; margin-top:8px; }
  .meta { font-size:12px; opacity:.7; }
</style></head>
<body><div class="wrap">
  <div class="row"><h1>Operation&nbsp;Love</h1><span id="runpill" class="pill stop">stopped</span></div>
  <div class="sub" id="sub">control hub · live status</div>

  <div class="card">
    <div class="row"><span class="muted">ranker</span><span id="ranker" class="b">—</span></div>
    <div class="bar"><div id="barfill" style="width:0%;background:#f0b429"></div></div>
    <div class="row meta"><span id="labels">labels —</span><span id="budget">budget —</span></div>
  </div>

  <div class="apps" id="apps"></div>

  <div class="card">
    <div class="controls">
      <span class="muted">mode</span>
      <select id="mode"><option value="observe">observe (you swipe)</option>
        <option value="auto">auto (bot swipes)</option></select>
      <span id="appchecks"></span>
    </div>
    <div class="controls" style="margin-top:12px">
      <button id="start" class="primary">▶ Start</button>
      <button id="stop" class="danger" disabled>■ Stop</button>
      <span class="meta" id="hint"></span>
    </div>
    <div class="err" id="err"></div>
  </div>
</div>
<script>
const $ = s => document.querySelector(s);
async function getJSON(u){ const r = await fetch(u); return r.json(); }
async function postJSON(u,b){ const r = await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})}); return r.json(); }

let cfg = {all_apps:['bumble'], enabled_apps:['bumble'], mode:'observe', min_labels:40};
function decClass(d){ return d==='like'?'dec-like':(d==='pass'||d==='dislike')?'dec-pass':'dec-other'; }

function renderApps(snap){
  const apps = (snap && snap.status && snap.status.apps) || {};
  const known = cfg.all_apps.length ? cfg.all_apps : Object.keys(apps);
  $('#apps').innerHTML = known.map(name => {
    const a = apps[name];
    if(!a) return `<div class="app"><div class="row"><span class="nm">${name}</span>`
                 + `<span class="muted">idle</span></div></div>`;
    const dec = a.last_decision ? a.last_decision.toUpperCase() : '—';
    const score = (a.last_score==null)?'':' · '+Number(a.last_score).toFixed(2);
    return `<div class="app"><div class="row"><span class="nm">${name}</span>`
         + `<span class="muted">${a.mode||''} · ${a.state||'—'}</span></div>`
         + `<div class="row meta"><span>last <b class="${decClass(a.last_decision)}">${dec}</b>${score}</span>`
         + `<span>swipes this run <b>${a.swipes_run||0}</b></span></div></div>`;
  }).join('');
}

function renderGlobal(snap){
  const running = snap && snap.running;
  $('#runpill').textContent = running ? 'running' : 'stopped';
  $('#runpill').className = 'pill ' + (running ? 'run' : 'stop');
  $('#start').disabled = running; $('#stop').disabled = !running;
  const s = snap && snap.status;
  if(s){
    const ready = s.ranker_ready;
    $('#ranker').textContent = (ready?'ready':'defer') + (s.mode?(' · '+s.mode):'');
    $('#ranker').style.color = ready ? '#39d98a' : '#f0b429';
    const pct = s.min_labels ? Math.min(100, Math.round(100*s.labels/s.min_labels)) : 100;
    $('#barfill').style.width = pct+'%';
    $('#barfill').style.background = ready ? '#39d98a' : '#f0b429';
    $('#labels').textContent = `labels ${s.labels} / ${s.min_labels}` + (ready?'':` (need ${s.labels_needed})`);
    const cap = s.budget_cap!=null ? ' / $'+Number(s.budget_cap).toFixed(2) : '';
    $('#budget').textContent = `budget $${Number(s.budget_spent).toFixed(2)}${cap}`;
  } else {
    $('#ranker').textContent='—'; $('#labels').textContent=`labels — / ${cfg.min_labels}`;
    $('#budget').textContent='budget —'; $('#barfill').style.width='0%';
  }
  $('#err').textContent = (snap && snap.error) ? ('error: '+snap.error) : '';
}

async function tick(){ try { const snap = await getJSON('/api/status'); renderGlobal(snap); renderApps(snap); } catch(e){} }

function appChecks(){
  $('#appchecks').innerHTML = cfg.all_apps.map(a =>
    `<label class="chk"><input type="checkbox" value="${a}" ${cfg.enabled_apps.includes(a)?'checked':''}>${a}</label>`
  ).join('');
}
function chosenApps(){ return [...document.querySelectorAll('#appchecks input:checked')].map(i=>i.value); }

$('#start').onclick = async () => {
  $('#hint').textContent='starting…';
  const r = await postJSON('/api/start', {mode: $('#mode').value, apps: chosenApps()});
  $('#hint').textContent = r.ok ? '' : (r.msg||'could not start');
  tick();
};
$('#stop').onclick = async () => { $('#hint').textContent='stopping…'; await postJSON('/api/stop',{}); tick(); };

(async () => {
  const c = await getJSON('/api/config');
  if(!c.error){ cfg = Object.assign(cfg, c); $('#mode').value = cfg.mode; $('#sub').textContent = `control hub · storage: ${cfg.backend}`; }
  appChecks(); tick(); setInterval(tick, 1000);
})();
</script>
</body></html>
"""
