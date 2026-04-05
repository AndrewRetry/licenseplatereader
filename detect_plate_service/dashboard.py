DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gantry LPR — Live Dashboard</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --text: #e1e4ed;
    --text-dim: #7a7f8e;
    --accent: #4ade80;
    --accent-dim: #166534;
    --warning: #f59e0b;
    --danger: #ef4444;
    --plate-bg: #1e293b;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  /* --- Header --- */
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  .header h1 {
    font-size: 18px;
    font-weight: 600;
    letter-spacing: -0.02em;
  }
  .header h1 span { color: var(--accent); }
  .status-bar { display: flex; gap: 16px; align-items: center; }
  .status-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    background: var(--bg);
    border: 1px solid var(--border);
  }
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--danger);
  }
  .status-dot.ok { background: var(--accent); }
  .status-dot.warn { background: var(--warning); }

  /* --- Main layout --- */
  .main {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    padding: 20px 24px;
    max-height: calc(100vh - 65px);
  }

  /* --- Camera panel --- */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }
  .panel-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-dim);
  }
  .camera-feed {
    width: 100%;
    aspect-ratio: 16/9;
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }
  .camera-feed img {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }
  .camera-placeholder {
    color: var(--text-dim);
    font-size: 14px;
    text-align: center;
    padding: 20px;
  }
  .camera-placeholder p { margin-top: 8px; font-size: 12px; }

  /* --- Detection log --- */
  .detection-list {
    padding: 8px;
    overflow-y: auto;
    max-height: calc(100vh - 140px);
  }
  .detection-card {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 14px;
    border-radius: 8px;
    margin-bottom: 6px;
    background: var(--bg);
    border: 1px solid var(--border);
    transition: border-color 0.3s;
  }
  .detection-card.new {
    border-color: var(--accent);
    box-shadow: 0 0 12px rgba(74, 222, 128, 0.1);
  }
  .plate-text {
    font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text);
    background: var(--plate-bg);
    padding: 6px 14px;
    border-radius: 6px;
    border: 1px solid var(--border);
    white-space: nowrap;
  }
  .detection-meta {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .detection-conf {
    font-size: 13px;
    font-weight: 500;
  }
  .conf-high { color: var(--accent); }
  .conf-mid { color: var(--warning); }
  .conf-low { color: var(--danger); }
  .detection-time {
    font-size: 12px;
    color: var(--text-dim);
  }
  .detection-gantry {
    font-size: 11px;
    color: var(--text-dim);
  }

  /* --- Empty state --- */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    color: var(--text-dim);
    text-align: center;
  }
  .empty-state .icon { font-size: 40px; margin-bottom: 12px; }
  .empty-state p { font-size: 13px; margin-top: 6px; }

  /* --- Counter badge --- */
  .count-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 22px;
    height: 22px;
    padding: 0 7px;
    border-radius: 11px;
    background: var(--accent-dim);
    color: var(--accent);
    font-size: 12px;
    font-weight: 600;
    margin-left: 8px;
  }

  @media (max-width: 900px) {
    .main { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<div class="header">
  <h1><span>&#9679;</span> Gantry LPR Dashboard</h1>
  <div class="status-bar">
    <div class="status-pill">
      <div class="status-dot" id="dot-rabbitmq"></div>
      <span id="label-rabbitmq">RabbitMQ</span>
    </div>
    <div class="status-pill">
      <div class="status-dot" id="dot-stream"></div>
      <span id="label-stream">Stream</span>
    </div>
  </div>
</div>

<div class="main">
  <!-- Camera feed -->
  <div class="panel">
    <div class="panel-header">Camera Feed</div>
    <div class="camera-feed" id="camera-container">
      <div class="camera-placeholder" id="camera-placeholder">
        <div style="font-size: 32px;">&#128247;</div>
        <p>No stream active.<br>Start a stream via POST /stream/start</p>
      </div>
    </div>
  </div>

  <!-- Detection log -->
  <div class="panel">
    <div class="panel-header">
      Detections <span class="count-badge" id="detection-count">0</span>
    </div>
    <div class="detection-list" id="detection-list">
      <div class="empty-state" id="empty-state">
        <div class="icon">&#128690;</div>
        <strong>Waiting for vehicles…</strong>
        <p>Plates will appear here as they are detected.</p>
      </div>
    </div>
  </div>
</div>

<script>
(function() {
  const API = window.location.origin;
  const POLL_HEALTH_MS = 3000;
  const POLL_DETECT_MS = 2000;

  let lastDetectionCount = 0;
  let knownTimestamps = new Set();

  // --- Status polling ---
  async function pollHealth() {
    try {
      const r = await fetch(API + '/health');
      const d = await r.json();

      setDot('dot-rabbitmq', d.rabbitmq === 'connected' ? 'ok' : 'warn');
      document.getElementById('label-rabbitmq').textContent =
        d.rabbitmq === 'connected' ? 'RabbitMQ' : 'RabbitMQ (offline)';

      const streamActive = d.stream && d.stream.active;
      setDot('dot-stream', streamActive ? 'ok' : '');
      document.getElementById('label-stream').textContent =
        streamActive ? 'Streaming' : 'Stream Off';

      // Update camera feed
      const streamUrl = streamActive ? (API + '/video') : null;
      updateCameraFeed(streamUrl, streamActive);

    } catch(e) {
      setDot('dot-rabbitmq', '');
      setDot('dot-stream', '');
    }
  }

  function setDot(id, state) {
    const el = document.getElementById(id);
    el.className = 'status-dot' + (state ? ' ' + state : '');
  }

  // --- Camera feed ---
  let currentFeedUrl = null;
  function updateCameraFeed(streamUrl, active) {
    const container = document.getElementById('camera-container');
    const placeholder = document.getElementById('camera-placeholder');

    if (active && streamUrl && streamUrl !== currentFeedUrl) {
      currentFeedUrl = streamUrl;
      const img = document.createElement('img');
      img.src = streamUrl;
      img.alt = 'Live camera feed';
      img.onerror = function() {
        this.style.display = 'none';
        placeholder.innerHTML =
          '<div style="font-size:32px">&#128247;</div>' +
          '<p>Camera feed unreachable.<br>' +
          'Stream may still be processing.<br>' +
          '<small style="color:#4ade80">' + streamUrl + '</small></p>';
        placeholder.style.display = 'block';
      };
      const old = container.querySelector('img');
      if (old) old.remove();
      placeholder.style.display = 'none';
      container.appendChild(img);
    } else if (!active) {
      currentFeedUrl = null;
      const old = container.querySelector('img');
      if (old) old.remove();
      placeholder.style.display = 'block';
      placeholder.innerHTML =
        '<div style="font-size:32px">&#128247;</div>' +
        '<p>No stream active.<br>Start a stream via POST /stream/start</p>';
    }
  }

  // --- Detections polling ---
  async function pollDetections() {
    try {
      const r = await fetch(API + '/detections');
      const d = await r.json();

      const list = document.getElementById('detection-list');
      const empty = document.getElementById('empty-state');
      const countEl = document.getElementById('detection-count');

      const detections = d.detections || [];
      countEl.textContent = detections.length;

      if (detections.length === 0) {
        empty.style.display = 'flex';
        return;
      }
      empty.style.display = 'none';

      let html = '';
      detections.forEach(function(det) {
        const key = det.timestamp + det.text;
        const isNew = !knownTimestamps.has(key);

        const conf = det.confidence;
        let confClass = 'conf-high';
        if (conf < 0.5) confClass = 'conf-low';
        else if (conf < 0.75) confClass = 'conf-mid';

        const time = formatTime(det.timestamp);

        html += '<div class="detection-card' + (isNew ? ' new' : '') + '">' +
          '<div class="plate-text">' + escHtml(det.text) + '</div>' +
          '<div class="detection-meta">' +
            '<div class="detection-conf ' + confClass + '">' +
              (conf * 100).toFixed(1) + '% confidence</div>' +
            '<div class="detection-time">' + time + '</div>' +
            '<div class="detection-gantry">Gantry: ' + escHtml(det.gantryId || '—') + '</div>' +
          '</div>' +
        '</div>';
      });
      list.innerHTML = html;

      if (detections.length > lastDetectionCount && lastDetectionCount > 0) {
        beep();
      }
      lastDetectionCount = detections.length;

      knownTimestamps.clear();
      detections.forEach(function(det) {
        knownTimestamps.add(det.timestamp + det.text);
      });

      setTimeout(function() {
        document.querySelectorAll('.detection-card.new').forEach(function(el) {
          el.classList.remove('new');
        });
      }, 3000);

    } catch(e) { }
  }

  function formatTime(iso) {
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch(e) { return iso; }
  }

  function escHtml(s) {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
  }

  function beep() {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = 880;
      osc.type = 'sine';
      gain.gain.value = 0.1;
      osc.start();
      osc.stop(ctx.currentTime + 0.12);
    } catch(e) {}
  }

  pollHealth();
  pollDetections();
  setInterval(pollHealth, POLL_HEALTH_MS);
  setInterval(pollDetections, POLL_DETECT_MS);
})();
</script>
</body>
</html>"""
