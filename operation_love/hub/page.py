"""The hub's single-page HTML/CSS/JS control panel, served as a raw string.

Kept separate from the request-handling and state/process-lifecycle code so
the frontend can be edited without touching either.
"""
from __future__ import annotations

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
  <div id="swipebanner" style="display:none"></div>

  <div class="card">
    <div class="row"><span class="muted">budget</span><span id="budget" class="b">—</span></div>
  </div>

  <div class="card">
    <div class="row"><span class="muted">model quality · leakage-free CV</span><span class="meta" id="evalhint"></span></div>
    <div class="meta" id="evalbody" style="margin-top:6px">evaluating…</div>
  </div>

  <div class="card">
    <div class="row"><span class="muted">live log</span><span class="meta" id="loghint"></span></div>
    <pre id="logs" style="max-height:320px;overflow:auto;background:#0c0c10;border:1px solid rgba(255,255,255,.09);border-radius:9px;padding:10px;margin-top:8px;font:11px/1.5 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap;color:#cfcfd6">waiting for output…</pre>
  </div>

  <div class="card">
    <div class="controls">
      <span class="muted">mode</span>
      <select id="mode"><option value="observe">observe (you swipe)</option>
        <option value="auto">auto (bot swipes)</option></select>
      <span id="appchecks"></span>
    </div>
    <div class="controls" id="maxrow" style="margin-top:10px;display:none">
      <span class="muted">max profiles this run</span>
      <input type="text" inputmode="numeric" id="maxrun" value="8"
        style="width:72px;text-align:center;background:#22222b;color:#e8e8ea;border:1px solid rgba(255,255,255,.14);border-radius:7px;padding:5px 7px;font:inherit">
      <label class="chk"><input type="checkbox" id="unlimited"> unlimited</label>
      <span class="meta">(the per-day cap still applies)</span>
    </div>
    <div class="controls" style="margin-top:12px">
      <button id="start" class="primary">▶ Start</button>
      <button id="stop" class="danger" disabled>■ Stop</button>
      <span class="meta" id="hint"></span>
    </div>
    <div class="err" id="err"></div>
  </div>

  <div class="card">
    <div class="row"><span class="muted">report a bug</span><span class="meta" id="bughint"></span></div>
    <textarea id="bugdesc" rows="2" placeholder="What went wrong? (optional)"
      style="width:100%;margin:8px 0;background:#22222b;color:#e8e8ea;border:1px solid rgba(255,255,255,.14);border-radius:9px;padding:8px;font:inherit;resize:vertical"></textarea>
    <div class="controls">
      <button id="bugcopy">📋 Copy report</button>
      <button id="bugdl">⬇ Download .md</button>
    </div>
    <pre id="bugout" style="display:none;max-height:260px;overflow:auto;background:#0c0c10;border:1px solid rgba(255,255,255,.09);border-radius:9px;padding:10px;margin-top:10px;font-size:11px;white-space:pre-wrap"></pre>
  </div>
</div>
<script>
const $ = s => document.querySelector(s);
async function getJSON(u){ const r = await fetch(u); return r.json(); }
async function postJSON(u,b){ const r = await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})}); return r.json(); }

let cfg = {all_apps:['bumble'], enabled_apps:['bumble'], mode:'observe', min_labels:40};
const hubClientId = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : (Date.now() + '-' + Math.random());
function hubLifecycle(path){
  const body = JSON.stringify({id: hubClientId});
  if(path === '/api/hub/closed' && navigator.sendBeacon){
    const sent = navigator.sendBeacon(path, new Blob([body], {type:'application/json'}));
    if(sent) return;
  }
  fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body, keepalive: path === '/api/hub/closed'}).catch(() => {});
}
hubLifecycle('/api/hub/open');
setInterval(() => hubLifecycle('/api/hub/ping'), 5000);
window.addEventListener('pageshow', () => hubLifecycle('/api/hub/open'));
window.addEventListener('pagehide', (event) => {
  if (!event.persisted) hubLifecycle('/api/hub/closed');
});

function renderGlobal(snap){
  const running = snap && snap.running;
  const s = snap && snap.status;
  const phase = s && s.phase;
  $('#runpill').textContent = running ? ((phase && phase!=='live') ? phase : 'running') : 'stopped';
  $('#runpill').className = 'pill ' + (running ? 'run' : 'stop');
  $('#start').disabled = running; $('#stop').disabled = !running;
  if(phase === 'saving data') $('#hint').textContent = 'saving data…';
  else if(!running && $('#hint').textContent === 'saving data…') $('#hint').textContent = '';
  if(s){
    const cap = s.budget_cap!=null ? ' / $'+Number(s.budget_cap).toFixed(2) : '';
    $('#budget').textContent = `$${Number(s.budget_spent).toFixed(2)}${cap}`;
  } else {
    $('#budget').textContent='—';
  }
  $('#err').textContent = (snap && snap.error) ? ('error: '+snap.error) : '';
}

// Observe mode has NO on-phone cue (an overlay would corrupt the screencaps), so this big
// banner tells you when to SWIPE vs WAIT per app, driven by the worker's per-app state.
function renderSwipe(snap){
  const el = $('#swipebanner'); if(!el) return;
  const s = snap && snap.status;
  if(!snap || !snap.running || !s || s.mode !== 'observe' || !s.apps){ el.style.display='none'; return; }
  const css = {go:'background:#123a23;border:1px solid #39d98a;color:#39d98a',
               wait:'background:#3a2f12;border:1px solid #f0b429;color:#f0b429',
               idle:'background:#22222b;border:1px solid rgba(255,255,255,.14);color:#9a9aa2'};
  const box = (k,t,sub) => `<div style="${css[k]};border-radius:10px;padding:12px 14px;margin-top:8px;font-weight:700;font-size:18px">${t}${sub?` <span style="font-weight:400;font-size:12px;opacity:.8">${sub}</span>`:''}</div>`;
  // Same per-app worker states drive this for EVERY app, so Bumble and Hinge show
  // identical SWIPE/WAIT guidance. WAIT covers both phases you must not swipe in:
  // reading the card (capturing) and learning the swipe you just made (acting).
  el.innerHTML = Object.values(s.apps).map(a => {
    const st = a.state || '';
    const last = a.last_decision ? ('last '+a.last_decision) : '';
    if(st==='waiting')   return box('go',   `🟢 SWIPE ${a.app} now`, 'like or pass');
    if(st==='capturing') return box('wait', `🔴 wait — reading ${a.app} profile…`);
    if(st==='acting')    return box('wait', `🔴 wait — processing your ${a.app} swipe…`);
    if(st==='starting')  return box('wait', `🔴 wait — starting ${a.app}…`);
    if(st==='out_of_profiles') return box('idle', `${a.app}: no more profiles`, last);
    return box('idle', `${a.app}: ${st||'…'}`, last);
  }).join('');
  el.style.display = 'block';
}

async function tick(){ try { const snap = await getJSON('/api/status'); renderGlobal(snap); renderSwipe(snap); } catch(e){} }

function renderEval(e){
  if(!e) return;
  const esc = s => String(s).replace(/[&<>"]/g, ch => (
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]
  ));
  // Single metric "accuracy" = ROC-AUC as a % = P(model ranks a profile you'd LIKE above one
  // you'd PASS). 50% = coin-flip, 100% = perfect; base-rate-independent, so it tracks the model
  // not your pickiness. The chart is the same metric over labels with a dashed 50% chance line.
  const accSvg = traj => {
    if(!Array.isArray(traj)) return '';
    const pts = traj.filter(p => p && p.roc_auc != null && isFinite(p.roc_auc));
    if(pts.length < 2) return '';
    const W=600, H=150, L=8, Rr=8, T=16, B=22;
    const xs = pts.map(p => p.labels);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const vv = pts.map(p => p.roc_auc).concat([0.5]);   // keep the chance line inside the y-range
    let lo = Math.min(...vv), hi = Math.max(...vv);
    const padv = (hi-lo)*0.18 || 0.05; lo = Math.max(0, lo-padv); hi = Math.min(1, hi+padv);
    const X = v => L + (xmax===xmin ? 0.5 : (v-xmin)/(xmax-xmin)) * (W-L-Rr);
    const Y = v => T + (1-(v-lo)/((hi-lo)||1)) * (H-T-B);
    const line = pts.map(p => `${X(p.labels).toFixed(1)},${Y(p.roc_auc).toFixed(1)}`).join(' ');
    const chanceY = Y(0.5).toFixed(1);
    const txt = (x,y,t,anchor,col) => `<text x="${x}" y="${y}" fill="${col||'#6a6a72'}" font-size="10"${anchor?` text-anchor="${anchor}"`:''}>${t}</text>`;
    const svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet" style="display:block;background:#0c0c10;border:1px solid rgba(255,255,255,.09);border-radius:9px">`
      + `<line x1="${L}" y1="${chanceY}" x2="${W-Rr}" y2="${chanceY}" stroke="#6a6a72" stroke-dasharray="3 3" opacity="0.5"/>`
      + `<polyline fill="none" stroke="#5ab0ff" stroke-width="1.8" points="${line}"/>`
      + txt(L, T-5, (hi*100).toFixed(3)+'%') + txt(L, H-7, (lo*100).toFixed(3)+'%')
      + txt(L+22, Number(chanceY)-3, 'random')
      + txt(L+22, H-7, xmin+' labels') + txt(W-Rr, H-7, xmax+'', 'end')
      + `</svg>`;
    return `<div style="margin-top:8px"><div class="meta" style="margin-bottom:3px">accuracy over labels</div>`
      + svg + `</div>`;
  };
  // Mini-goal: refresh is gated on new labels (every N), not the clock — so show progress
  // toward the next recompute. Only while a live observe run is feeding labels (auto mode
  // never adds training labels).
  const refreshHtml = () => {
    const r = e.refresh;
    if(!r || !r.live || r.mode !== 'observe' || r.since == null) return '';
    const every = Math.max(1, Math.floor(Number(r.every) || 1));
    const progress = Math.min(every, Math.max(0, Math.floor(Number(r.since) || 0)));
    return `<div class="muted" style="margin-top:4px">↻ ${progress}/${every} till next refresh</div>`;
  };
  if(e.status !== 'ok'){
    $('#evalhint').textContent = (e.identities!=null) ? `${e.identities} identities` : '';
    $('#evalbody').style.color = '#9a9aa2';
    $('#evalbody').innerHTML = `<div>${esc(e.message || 'evaluating…')}</div>` + refreshHtml();
    return;
  }
  // Accuracy % = ROC-AUC × 100 = P(ranks a profile you'd LIKE above one you'd PASS).
  // Grade the PESSIMISTIC band edge (mean-std) so a wide band can't read green.
  const G='#39d98a', A='#f0b429', R='#ff6b6b';
  const acc = e.roc_auc[0]*100, band = e.roc_auc[1]*100, lo = acc - band;
  const col = lo>=80 ? G : (lo>=70 ? A : R);
  $('#evalhint').textContent = `${e.folds}-fold · ${e.identities} identities`;
  $('#evalbody').style.color = '#e8e8ea';
  $('#evalbody').innerHTML =
    `<div>accuracy <b style="color:${col};font-size:18px">${acc.toFixed(3)}%</b>`
    + ` <span class="muted">±${band.toFixed(3)}% · ranks a like above a pass (50% = random)</span></div>`
    + refreshHtml() + accSvg(e.trajectory);
}
async function tickEval(){ try { renderEval(await getJSON('/api/eval')); } catch(e){} }

function _nearBottom(el){ return el.scrollHeight - el.scrollTop - el.clientHeight < 40; }
async function tickLogs(){
  try {
    const r = await getJSON('/api/logs'); const el = $('#logs');
    const stick = _nearBottom(el);                 // only auto-scroll if user is already at the bottom
    el.textContent = (r.lines||[]).join('\\n');
    $('#loghint').textContent = (r.lines||[]).length + ' lines';
    if(stick) el.scrollTop = el.scrollHeight;
  } catch(e){}
}

function appChecks(){
  $('#appchecks').innerHTML = cfg.all_apps.map(a =>
    `<label class="chk"><input type="checkbox" value="${a}" ${cfg.enabled_apps.includes(a)?'checked':''}>${a}</label>`
  ).join('');
}
function chosenApps(){ return [...document.querySelectorAll('#appchecks input:checked')].map(i=>i.value); }

function syncModeUI(){                         // the cap only applies in auto mode
  $('#maxrow').style.display = $('#mode').value === 'auto' ? '' : 'none';
  const off = $('#unlimited').checked;         // unlimited on -> show ∞, grey out + disable the number box
  const box = $('#maxrun');
  if (off) {
    if (box.value !== '∞') box.dataset.prev = box.value;   // remember the number to restore later
    box.value = '∞';
  } else if (box.value === '∞') {
    box.value = box.dataset.prev || '8';
  }
  box.disabled = off;
  box.style.opacity = off ? '.45' : '1';
  box.style.cursor = off ? 'not-allowed' : 'text';
}
$('#mode').onchange = syncModeUI;
$('#unlimited').onchange = syncModeUI;
$('#start').onclick = async () => {
  $('#hint').textContent='starting…';
  const mode = $('#mode').value;
  let max_per_run = null;                       // null = use config; 0 = unlimited; N = cap
  if (mode === 'auto') max_per_run = $('#unlimited').checked ? 0 : (parseInt($('#maxrun').value,10) || 0);
  const r = await postJSON('/api/start', {mode, apps: chosenApps(), max_per_run});
  $('#hint').textContent = r.ok ? '' : (r.msg||'could not start');
  tick();
};
$('#stop').onclick = async () => { $('#hint').textContent='stopping…'; await postJSON('/api/stop',{}); tick(); };

async function getReport(){          // generate fresh each time (Copy/Download do this implicitly)
  $('#bughint').textContent = 'generating…';
  const r = await fetch('/api/bugreport?desc=' + encodeURIComponent($('#bugdesc').value || ''));
  const md = await r.text();
  $('#bughint').textContent = md.length + ' chars';
  return md;
}
$('#bugcopy').onclick = async () => {
  const md = await getReport();
  try { await navigator.clipboard.writeText(md); $('#bughint').textContent = 'copied ✓'; }
  catch(e){ const out = $('#bugout'); out.textContent = md; out.style.display = 'block';
            $('#bughint').textContent = 'copy blocked — select the text below'; }
};
$('#bugdl').onclick = async () => {
  const md = await getReport();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([md], {type:'text/markdown'}));
  a.download = 'operation-love-bug-report.md'; a.click(); URL.revokeObjectURL(a.href);
  $('#bughint').textContent = 'downloaded ✓';
};

(async () => {
  const c = await getJSON('/api/config');
  if(!c.error){ cfg = Object.assign(cfg, c); $('#mode').value = cfg.mode; $('#sub').textContent = `control hub · storage: ${cfg.backend}`; }
  syncModeUI();
  appChecks();
  tick(); setInterval(tick, 1000);
  tickEval(); setInterval(tickEval, 5000);    // model quality: poll often for live label progress; the CV itself only recomputes every N new labels (server-gated)
  tickLogs(); setInterval(tickLogs, 1500);    // live log: tail the captured stdout/stderr
})();
</script>
</body></html>
"""
