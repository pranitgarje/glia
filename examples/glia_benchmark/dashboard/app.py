"""
dashboard/app.py
────────────────
Live benchmarking dashboard — Flask + Server-Sent Events.

Subscribes to the Redis pub/sub channel where the benchmark runner
publishes telemetry packets and streams them to the browser via SSE.
Also serves the static HTML dashboard page.

Run:
    pip install flask redis
    python dashboard/app.py
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any, Dict

from flask import Flask, Response, render_template_string, jsonify
import redis

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
CHANNEL   = "glia:benchmark:events"
PORT      = int(os.environ.get("PORT", 5050))

app = Flask(__name__)

# Thread-safe in-memory state for live aggregation
_state_lock = Lock()
_state: Dict[str, Any] = {
    "hit_count":         0,
    "miss_count":        0,
    "invalidations":     0,
    "mutations":         0,
    "latencies_glia":    defaultdict(list),   # db_type → [ms]
    "latencies_std":     defaultdict(list),
    "tokens_glia":       defaultdict(int),
    "tokens_std":        defaultdict(int),
    "inv_latencies":     [],
    "timeline":          deque(maxlen=120),   # last 120 events for sparklines
    "last_report":       None,
}

# ─────────────────────────────────────────────────────────────────────────────
# State updater
# ─────────────────────────────────────────────────────────────────────────────

def _update_state(msg: Dict) -> None:
    t = msg.get("type", "")
    with _state_lock:
        if t == "cache_hit":
            _state["hit_count"] += 1
            _state["timeline"].append({"t": msg.get("ts"), "ev": "hit"})

        elif t == "cache_miss":
            _state["miss_count"] += 1
            _state["timeline"].append({"t": msg.get("ts"), "ev": "miss"})

        elif t == "invalidation":
            _state["invalidations"] += 1
            lat = msg.get("latency_ms")
            if lat is not None:
                _state["inv_latencies"].append(lat)
            _state["timeline"].append({"t": msg.get("ts"), "ev": "inv",
                                        "lat": lat})

        elif t == "simulator_cycle":
            _state["mutations"] += msg.get("mutations", 0)

        elif t == "progress":
            agg = msg.get("aggregate", {})
            for db in ("relational", "graph", "vector"):
                glia_lat = (agg.get("glia", {}).get(db, {})
                               .get("latency", {}).get("mean", 0))
                std_lat  = (agg.get("standard", {}).get(db, {})
                               .get("latency", {}).get("mean", 0))
                glia_tok = (agg.get("glia", {}).get(db, {})
                               .get("total_tokens", 0))
                std_tok  = (agg.get("standard", {}).get(db, {})
                               .get("total_tokens", 0))
                if glia_lat: _state["latencies_glia"][db].append(glia_lat)
                if std_lat:  _state["latencies_std"][db].append(std_lat)
                _state["tokens_glia"][db] = glia_tok
                _state["tokens_std"][db]  = std_tok

        elif t == "final_report":
            _state["last_report"] = msg


def _snapshot() -> Dict:
    with _state_lock:
        hits   = _state["hit_count"]
        misses = _state["miss_count"]
        total  = hits + misses
        inv_l  = list(_state["inv_latencies"])

        def _mean(lst): return sum(lst) / len(lst) if lst else 0
        def _p(lst, pct):
            if not lst: return 0
            s = sorted(lst); return s[int(len(s) * pct)]

        return {
            "hits":            hits,
            "misses":          misses,
            "hit_rate":        hits / max(total, 1),
            "invalidations":   _state["invalidations"],
            "mutations":       _state["mutations"],
            "inv_lat_mean":    _mean(inv_l),
            "inv_lat_p95":     _p(inv_l, 0.95),
            "latencies_glia":  {k: _mean(v) for k, v in _state["latencies_glia"].items()},
            "latencies_std":   {k: _mean(v) for k, v in _state["latencies_std"].items()},
            "tokens_glia":     dict(_state["tokens_glia"]),
            "tokens_std":      dict(_state["tokens_std"]),
            "timeline":        list(_state["timeline"])[-30:],
            "has_report":      _state["last_report"] is not None,
        }

# ─────────────────────────────────────────────────────────────────────────────
# SSE endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/stream")
def stream():
    def generate():
        r = redis.from_url(REDIS_URL)
        ps = r.pubsub()
        ps.subscribe(CHANNEL)
        try:
            for raw_msg in ps.listen():
                if raw_msg["type"] != "message":
                    continue
                try:
                    data = json.loads(raw_msg["data"])
                    _update_state(data)
                    snap = _snapshot()
                    yield f"data: {json.dumps(snap)}\n\n"
                except Exception:
                    pass
        except GeneratorExit:
            ps.unsubscribe()
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/snapshot")
def api_snapshot():
    return jsonify(_snapshot())


@app.route("/api/report")
def api_report():
    with _state_lock:
        return jsonify(_state.get("last_report") or {})


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard HTML (inlined for single-file deployment)
# ─────────────────────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Glia Benchmark — Live Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;700&display=swap');

  :root {
    --bg:        #0a0c0f;
    --bg2:       #111318;
    --bg3:       #1a1d24;
    --border:    #252830;
    --glia:      #00e5a0;
    --std:       #ff6b6b;
    --inv:       #ffd93d;
    --hit:       #06d6a0;
    --miss:      #ef476f;
    --graph:     #118ab2;
    --vec:       #a855f7;
    --rel:       #f97316;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    min-height: 100vh;
  }

  /* ── Header ─── */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--bg2);
  }
  .logo {
    font-family: var(--mono);
    font-size: 18px;
    font-weight: 600;
    letter-spacing: -0.5px;
    color: var(--glia);
  }
  .logo span { color: var(--text); opacity: 0.5; }
  .status-pill {
    display: flex; align-items: center; gap: 8px;
    font-family: var(--mono); font-size: 12px;
    color: var(--muted);
  }
  .pulse {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--glia);
    box-shadow: 0 0 0 0 rgba(0,229,160,0.4);
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(0,229,160,0.4); }
    70%  { box-shadow: 0 0 0 8px rgba(0,229,160,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,229,160,0); }
  }

  /* ── Grid ─── */
  .grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    grid-template-rows: auto;
    gap: 12px;
    padding: 16px 20px;
    max-width: 1600px;
    margin: 0 auto;
  }

  /* ── Card ─── */
  .card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
  }
  .card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 2px;
  }
  .card.glia::before   { background: var(--glia); }
  .card.std::before    { background: var(--std); }
  .card.inv::before    { background: var(--inv); }
  .card.neutral::before { background: var(--border); }

  .card-label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
  }
  .card-value {
    font-family: var(--mono);
    font-size: 36px;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 4px;
  }
  .card-sub {
    font-size: 12px;
    color: var(--muted);
  }
  .card-value.green { color: var(--glia); }
  .card-value.red   { color: var(--std); }
  .card-value.yellow{ color: var(--inv); }

  /* ── Span cards ─── */
  .span2 { grid-column: span 2; }
  .span4 { grid-column: span 4; }

  /* ── Bar chart ─── */
  .bar-group { margin-top: 12px; }
  .bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }
  .bar-label {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    width: 70px;
    flex-shrink: 0;
  }
  .bar-track {
    flex: 1;
    height: 20px;
    background: var(--bg3);
    border-radius: 3px;
    overflow: hidden;
    position: relative;
  }
  .bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
    position: relative;
  }
  .bar-val {
    position: absolute;
    right: 6px;
    top: 50%;
    transform: translateY(-50%);
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    color: rgba(255,255,255,0.9);
  }

  /* ── Timeline ─── */
  .timeline { display: flex; gap: 3px; align-items: flex-end; height: 48px; margin-top: 12px; }
  .tl-bar {
    flex: 1;
    border-radius: 2px 2px 0 0;
    transition: height 0.3s ease;
    min-height: 4px;
  }
  .tl-bar.hit  { background: var(--hit); }
  .tl-bar.miss { background: var(--miss); }
  .tl-bar.inv  { background: var(--inv); }

  /* ── Comparison table ─── */
  table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 12px; }
  th {
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    color: var(--muted);
    font-weight: 400;
    letter-spacing: 0.5px;
    font-size: 10px;
    text-transform: uppercase;
  }
  td { padding: 8px 10px; border-bottom: 1px solid #1c1f26; }
  tr:last-child td { border-bottom: none; }
  .db-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  .db-rel { background: rgba(249,115,22,0.15); color: var(--rel); }
  .db-gph { background: rgba(17,138,178,0.15); color: var(--graph); }
  .db-vec { background: rgba(168,85,247,0.15); color: var(--vec); }

  .delta-pos { color: var(--glia); }
  .delta-neg { color: var(--std); }

  /* ── Hit rate donut ─── */
  .donut-wrap { display: flex; align-items: center; gap: 20px; margin-top: 12px; }
  .donut-svg  { flex-shrink: 0; }
  .donut-legend { display: flex; flex-direction: column; gap: 8px; }
  .legend-row { display: flex; align-items: center; gap: 8px; font-size: 12px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; }

  /* ── Footer ─── */
  footer {
    text-align: center;
    padding: 16px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    border-top: 1px solid var(--border);
    margin-top: 8px;
  }
</style>
</head>
<body>

<header>
  <div class="logo">glia<span>/</span>benchmark</div>
  <div class="status-pill">
    <div class="pulse" id="pulse"></div>
    <span id="status-text">connecting …</span>
  </div>
</header>

<div class="grid">

  <!-- KPI cards row 1 -->
  <div class="card glia">
    <div class="card-label">Cache Hits</div>
    <div class="card-value green" id="kpi-hits">0</div>
    <div class="card-sub">resolved without LLM call</div>
  </div>

  <div class="card std">
    <div class="card-label">Cache Misses</div>
    <div class="card-value red" id="kpi-misses">0</div>
    <div class="card-sub">full retrieval + LLM</div>
  </div>

  <div class="card inv">
    <div class="card-label">Invalidations</div>
    <div class="card-value yellow" id="kpi-inv">0</div>
    <div class="card-sub" id="kpi-inv-lat">— ms avg latency</div>
  </div>

  <div class="card neutral">
    <div class="card-label">DB Mutations (Simulator)</div>
    <div class="card-value" id="kpi-mutations">0</div>
    <div class="card-sub">across all three DB types</div>
  </div>

  <!-- Hit rate donut + timeline  -->
  <div class="card glia span2">
    <div class="card-label">Semantic Cache Hit Rate</div>
    <div class="donut-wrap">
      <svg class="donut-svg" width="80" height="80" viewBox="0 0 80 80">
        <circle id="donut-bg"   cx="40" cy="40" r="30" fill="none" stroke="#252830" stroke-width="10"/>
        <circle id="donut-fill" cx="40" cy="40" r="30" fill="none" stroke="#00e5a0"
                stroke-width="10" stroke-dasharray="0 188.5"
                stroke-linecap="round" transform="rotate(-90 40 40)"
                style="transition: stroke-dasharray 0.6s ease"/>
        <text x="40" y="44" text-anchor="middle" fill="#e2e8f0"
              font-family="IBM Plex Mono,monospace" font-size="13" font-weight="600"
              id="donut-label">0%</text>
      </svg>
      <div class="donut-legend">
        <div class="legend-row"><div class="legend-dot" style="background:var(--hit)"></div>Hit (cached)</div>
        <div class="legend-row"><div class="legend-dot" style="background:var(--miss)"></div>Miss (DB + LLM)</div>
        <div class="legend-row"><div class="legend-dot" style="background:var(--inv)"></div>Invalidated</div>
      </div>
    </div>
  </div>

  <div class="card neutral span2">
    <div class="card-label">Event Timeline (last 30)</div>
    <div class="timeline" id="timeline"></div>
    <div style="display:flex;gap:16px;margin-top:8px">
      <span style="font-size:11px;color:var(--hit)">■ hit</span>
      <span style="font-size:11px;color:var(--miss)">■ miss</span>
      <span style="font-size:11px;color:var(--inv)">■ invalidation</span>
    </div>
  </div>

  <!-- Latency comparison bars -->
  <div class="card neutral span2">
    <div class="card-label">Avg End-to-End Latency — Standard vs Glia (ms)</div>
    <div class="bar-group" id="latency-bars">
      <div class="bar-row">
        <div class="bar-label">RELATIONAL</div>
        <div style="flex:1">
          <div style="display:flex;gap:4px;margin-bottom:4px">
            <div class="bar-track" style="height:14px"><div class="bar-fill" id="lat-std-rel" style="background:var(--std);width:0%"><span class="bar-val" id="lv-std-rel">0ms</span></div></div>
          </div>
          <div class="bar-track" style="height:14px"><div class="bar-fill" id="lat-glia-rel" style="background:var(--glia);width:0%"><span class="bar-val" id="lv-glia-rel">0ms</span></div></div>
        </div>
      </div>
      <div class="bar-row">
        <div class="bar-label">GRAPH</div>
        <div style="flex:1">
          <div style="display:flex;gap:4px;margin-bottom:4px">
            <div class="bar-track" style="height:14px"><div class="bar-fill" id="lat-std-gph" style="background:var(--std);width:0%"><span class="bar-val" id="lv-std-gph">0ms</span></div></div>
          </div>
          <div class="bar-track" style="height:14px"><div class="bar-fill" id="lat-glia-gph" style="background:var(--glia);width:0%"><span class="bar-val" id="lv-glia-gph">0ms</span></div></div>
        </div>
      </div>
      <div class="bar-row">
        <div class="bar-label">VECTOR</div>
        <div style="flex:1">
          <div style="display:flex;gap:4px;margin-bottom:4px">
            <div class="bar-track" style="height:14px"><div class="bar-fill" id="lat-std-vec" style="background:var(--std);width:0%"><span class="bar-val" id="lv-std-vec">0ms</span></div></div>
          </div>
          <div class="bar-track" style="height:14px"><div class="bar-fill" id="lat-glia-vec" style="background:var(--glia);width:0%"><span class="bar-val" id="lv-glia-vec">0ms</span></div></div>
        </div>
      </div>
      <div style="display:flex;gap:16px;margin-top:4px">
        <span style="font-size:11px;color:var(--std)">■ Standard</span>
        <span style="font-size:11px;color:var(--glia)">■ Glia</span>
      </div>
    </div>
  </div>

  <!-- Token consumption bars -->
  <div class="card neutral span2">
    <div class="card-label">Token Consumption — Standard vs Glia</div>
    <div class="bar-group">
      <div class="bar-row">
        <div class="bar-label">RELATIONAL</div>
        <div style="flex:1">
          <div style="margin-bottom:4px"><div class="bar-track" style="height:14px"><div class="bar-fill" id="tok-std-rel" style="background:var(--std);width:0%"><span class="bar-val" id="tv-std-rel">0</span></div></div></div>
          <div class="bar-track" style="height:14px"><div class="bar-fill" id="tok-glia-rel" style="background:var(--glia);width:0%"><span class="bar-val" id="tv-glia-rel">0</span></div></div>
        </div>
      </div>
      <div class="bar-row">
        <div class="bar-label">GRAPH</div>
        <div style="flex:1">
          <div style="margin-bottom:4px"><div class="bar-track" style="height:14px"><div class="bar-fill" id="tok-std-gph" style="background:var(--std);width:0%"><span class="bar-val" id="tv-std-gph">0</span></div></div></div>
          <div class="bar-track" style="height:14px"><div class="bar-fill" id="tok-glia-gph" style="background:var(--glia);width:0%"><span class="bar-val" id="tv-glia-gph">0</span></div></div>
        </div>
      </div>
      <div class="bar-row">
        <div class="bar-label">VECTOR</div>
        <div style="flex:1">
          <div style="margin-bottom:4px"><div class="bar-track" style="height:14px"><div class="bar-fill" id="tok-std-vec" style="background:var(--std);width:0%"><span class="bar-val" id="tv-std-vec">0</span></div></div></div>
          <div class="bar-track" style="height:14px"><div class="bar-fill" id="tok-glia-vec" style="background:var(--glia);width:0%"><span class="bar-val" id="tv-glia-vec">0</span></div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Comparison table -->
  <div class="card neutral span4">
    <div class="card-label">Per-DB Comparative Summary</div>
    <table id="comparison-table" style="margin-top:12px">
      <thead>
        <tr>
          <th>Database Type</th>
          <th>Std Latency (mean)</th>
          <th>Glia Latency (mean)</th>
          <th>Latency Δ</th>
          <th>Std Tokens</th>
          <th>Glia Tokens</th>
          <th>Token Savings</th>
          <th>Hit Rate</th>
        </tr>
      </thead>
      <tbody id="table-body">
        <tr><td colspan="8" style="text-align:center;color:var(--muted);padding:20px">Waiting for benchmark data …</td></tr>
      </tbody>
    </table>
  </div>

</div>

<footer>glia benchmark dashboard · live via SSE · redis pub/sub · no polling</footer>

<script>
const $ = id => document.getElementById(id);
const fmt = (n, d=1) => n == null ? '—' : n.toFixed(d);
const fmtK = n => n > 1000 ? (n/1000).toFixed(1)+'K' : String(Math.round(n));

let latMax = { std: { rel:1, gph:1, vec:1 }, glia: { rel:1, gph:1, vec:1 } };
let tokMax = { std: { rel:1, gph:1, vec:1 }, glia: { rel:1, gph:1, vec:1 } };

function setBar(id, pct, val) {
  const el = $(id);
  if (!el) return;
  el.style.width = Math.min(pct, 100) + '%';
  const lbl = el.querySelector('.bar-val');
  if (lbl) lbl.textContent = val;
}

function updateDonut(rate) {
  const circ = 2 * Math.PI * 30; // 188.5
  const fill = circ * rate;
  $('donut-fill').setAttribute('stroke-dasharray', fill + ' ' + circ);
  $('donut-label').textContent = Math.round(rate * 100) + '%';
}

function updateTimeline(tl) {
  const el = $('timeline');
  el.innerHTML = '';
  tl.forEach(ev => {
    const b = document.createElement('div');
    b.className = 'tl-bar ' + (ev.ev || 'miss');
    b.style.height = (ev.ev === 'inv' ? 48 : ev.ev === 'hit' ? 32 : 20) + 'px';
    el.appendChild(b);
  });
}

function updateTable(snap) {
  const dbs = [
    { key:'relational', label:'Relational (PostgreSQL)', cls:'db-rel' },
    { key:'graph',      label:'Graph (Neo4j)',           cls:'db-gph' },
    { key:'vector',     label:'Vector (Qdrant)',         cls:'db-vec' },
  ];
  const rows = dbs.map(db => {
    const stdL  = snap.latencies_std[db.key]  || 0;
    const gliaL = snap.latencies_glia[db.key] || 0;
    const stdT  = snap.tokens_std[db.key]     || 0;
    const gliaT = snap.tokens_glia[db.key]    || 0;
    const deltaL = gliaL - stdL;
    const deltaT = stdT - gliaT;
    const hit   = (snap.hit_rate || 0) * 100;
    const dCls  = deltaL < 0 ? 'delta-pos' : 'delta-neg';
    return `<tr>
      <td><span class="db-badge ${db.cls}">${db.label}</span></td>
      <td>${fmt(stdL)} ms</td>
      <td>${fmt(gliaL)} ms</td>
      <td class="${dCls}">${deltaL >= 0 ? '+' : ''}${fmt(deltaL)} ms</td>
      <td>${fmtK(stdT)}</td>
      <td>${fmtK(gliaT)}</td>
      <td class="delta-pos">${deltaT >= 0 ? '+' : ''}${fmtK(deltaT)}</td>
      <td>${fmt(hit, 1)}%</td>
    </tr>`;
  }).join('');
  $('table-body').innerHTML = rows || '<tr><td colspan="8" style="color:var(--muted);text-align:center">Waiting …</td></tr>';
}

function render(snap) {
  // KPIs
  $('kpi-hits').textContent     = snap.hits.toLocaleString();
  $('kpi-misses').textContent   = snap.misses.toLocaleString();
  $('kpi-inv').textContent      = snap.invalidations.toLocaleString();
  $('kpi-mutations').textContent = snap.mutations.toLocaleString();
  $('kpi-inv-lat').textContent  = fmt(snap.inv_lat_mean) + ' ms avg · p95 ' + fmt(snap.inv_lat_p95) + ' ms';

  // Donut
  updateDonut(snap.hit_rate || 0);

  // Timeline
  updateTimeline(snap.timeline || []);

  // Latency bars
  const dbs = ['rel','gph','vec'];
  const dbKeys = { rel:'relational', gph:'graph', vec:'vector' };
  dbs.forEach(d => {
    const stdL  = snap.latencies_std[dbKeys[d]]  || 0;
    const gliaL = snap.latencies_glia[dbKeys[d]] || 0;
    latMax.std[d]  = Math.max(latMax.std[d],  stdL  || 1);
    latMax.glia[d] = Math.max(latMax.glia[d], gliaL || 1);
    const maxL = Math.max(latMax.std[d], latMax.glia[d], 1);
    setBar('lat-std-'  + d, stdL  / maxL * 100, fmt(stdL)  + 'ms');
    setBar('lat-glia-' + d, gliaL / maxL * 100, fmt(gliaL) + 'ms');

    const stdT  = snap.tokens_std[dbKeys[d]]  || 0;
    const gliaT = snap.tokens_glia[dbKeys[d]] || 0;
    const maxT = Math.max(stdT, gliaT, 1);
    setBar('tok-std-'  + d, stdT  / maxT * 100, fmtK(stdT));
    setBar('tok-glia-' + d, gliaT / maxT * 100, fmtK(gliaT));
  });

  // Table
  updateTable(snap);
}

// ── SSE connection ───────────────────────────────────────────────────────────
let es;
function connect() {
  es = new EventSource('/stream');
  es.onopen    = () => {
    $('status-text').textContent = 'live';
    $('pulse').style.background = 'var(--glia)';
  };
  es.onerror   = () => {
    $('status-text').textContent = 'reconnecting …';
    $('pulse').style.background = 'var(--std)';
    setTimeout(connect, 3000);
    es.close();
  };
  es.onmessage = e => {
    try { render(JSON.parse(e.data)); } catch {}
  };
}
connect();

// Poll for snapshot every 5s as fallback
setInterval(async () => {
  if (es && es.readyState === 1) return; // SSE active — skip
  const r = await fetch('/api/snapshot').catch(() => null);
  if (r && r.ok) render(await r.json());
}, 5000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
