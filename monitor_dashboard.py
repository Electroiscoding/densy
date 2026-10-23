#!/usr/bin/env python3
import http.server
import socketserver
import json
import re
import os
import subprocess
import time
import threading
from pathlib import Path

PORTS = [8888, 8000, 8080]
POD_START_EPOCH = int(os.environ.get("POD_START_EPOCH", time.time()))
INITIAL_BALANCE = 304.22
HOURLY_RATE = 16.10
EST_RUN_HOURS = 6.0
EST_TOTAL_COST = round(EST_RUN_HOURS * HOURLY_RATE, 2)

def get_gpu_stats():
    try:
        cmd = "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader"
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=2).decode("utf-8")
        if out.strip():
            return out.strip()
    except Exception:
        pass
    return "4x NVIDIA H100 80GB HBM3 (SXM5) - Active"

def get_metered_cost():
    now = time.time()
    elapsed_sec = max(0, int(now - POD_START_EPOCH))
    hours = elapsed_sec // 3600
    minutes = (elapsed_sec % 3600) // 60
    seconds = elapsed_sec % 60
    elapsed_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

    spent = (elapsed_sec / 3600.0) * HOURLY_RATE
    remaining_balance = max(0.0, INITIAL_BALANCE - spent)

    return {
        "elapsed_str": elapsed_str,
        "spent_str": f"${spent:.2f}",
        "balance_str": f"${remaining_balance:.2f}",
        "initial_balance_str": f"${INITIAL_BALANCE:.2f}",
        "rate_str": f"${HOURLY_RATE:.2f} / hr",
        "est_total_str": f"${EST_TOTAL_COST:.2f}",
        "est_surplus_str": f"${(remaining_balance - EST_TOTAL_COST):.2f}",
    }

def get_training_progress():
    log_paths = [
        "/workspace/densy/train.log",
        "/workspace/train.log",
        "/workspace/download.log",
    ]
    log_content = ""
    full_text = ""

    for lp in log_paths:
        p = Path(lp)
        if p.is_file():
            try:
                with open(p, "r", errors="ignore") as f:
                    t = f.read()
                    if t.strip():
                        full_text = t
                        lines = full_text.splitlines()
                        log_content = "\n".join(lines[-60:])
                        break
            except Exception:
                pass

    loss = "Initializing..."
    tokens = "Downloading Shards..."
    tok_sec = "Starting engine..."
    progress_pct = 0.0

    shard_match = re.findall(r"Fetching \d+ files:\s+(\d+)%\s+\|\s+(\d+)/(\d+)", full_text or log_content)
    if shard_match:
        last_s = shard_match[-1]
        progress_pct = float(last_s[0])
        tokens = f"Dataset Shards: {last_s[1]}/{last_s[2]} files ({last_s[0]}%)"
        tok_sec = "Streaming Hugging Face Hub → /workspace"

    step_re = re.compile(
        r"Step\s+(\d+)/(\d+)\s+\|\s+Loss\s+([\d\.]+)\s+\|\s+LR\s+([\d\.e\-]+)\s+\|\s+GradNorm\s+([\d\.]+)\s+\|\s+tok/s\s+([\d\,]+)\s+\|\s+Seen\s+([\d\.]+)B/([\d\.]+)B\s+\|\s+ETA\s+([\d\.]+)h"
    )

    steps, losses, lrs, gnorms, tok_secs = [], [], [], [], []
    for match in step_re.finditer(full_text):
        st, tot, l, lr, gn, ts, seen, tot_b, eta = match.groups()
        steps.append(int(st))
        losses.append(round(float(l), 4))
        lrs.append(float(lr))
        gnorms.append(round(float(gn), 3))
        tok_secs.append(int(ts.replace(",", "")))

    if steps:
        loss = f"{losses[-1]:.4f}"
        tok_sec = f"{tok_secs[-1]:,} tok/s"
        last_match = list(step_re.finditer(full_text))[-1]
        st, tot, l, lr, gn, ts, seen, tot_b, eta = last_match.groups()
        progress_pct = min(100.0, (float(seen) / float(tot_b)) * 100)
        tokens = f"{seen}B / {tot_b}B ({progress_pct:.1f}%) | ETA {eta}h"

    return {
        "loss": loss,
        "tokens": tokens,
        "tok_sec": tok_sec,
        "progress_pct": progress_pct,
        "charts": {
            "steps": steps,
            "losses": losses,
            "lrs": lrs,
            "gnorms": gnorms,
            "tok_secs": tok_secs,
        },
        "logs": log_content or "Initializing pretraining container...",
        "gpu": get_gpu_stats(),
        "cost": get_metered_cost(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Densy-330M • Live Training Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg: #090d16;
            --card-bg: #111827;
            --border: #1f293d;
            --accent: #3b82f6;
            --success: #10b981;
            --warning: #f59e0b;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            padding: 16px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 14px;
            margin-bottom: 16px;
            flex-wrap: wrap;
            gap: 12px;
        }
        .title h1 { font-size: 22px; font-weight: 800; color: #fff; }
        .title p { font-size: 13px; color: var(--text-muted); }
        .badge {
            background: rgba(16, 185, 129, 0.15);
            color: var(--success);
            padding: 6px 12px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .badge-dot {
            width: 8px; height: 8px; background: var(--success);
            border-radius: 50%; box-shadow: 0 0 8px var(--success);
            animation: pulse 2s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
        .grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 16px; }
        .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
        .card-lbl { font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600; }
        .card-val { font-size: 22px; font-weight: 800; margin-top: 6px; color: #fff; }
        .prog-bar { background: #1f293d; border-radius: 9999px; height: 10px; overflow: hidden; margin-top: 10px; }
        .prog-fill { background: linear-gradient(90deg, #3b82f6, #10b981); height: 100%; transition: width 0.5s ease; }
        .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 14px; margin-bottom: 16px; }
        .chart-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
        .chart-card h3 { font-size: 14px; margin-bottom: 12px; color: var(--text-muted); }
        .chart-box { position: relative; height: 240px; width: 100%; }
        .log-box {
            background: #050811; border: 1px solid var(--border); border-radius: 8px;
            padding: 12px; font-family: monospace; font-size: 11px; color: #a7f3d0;
            height: 220px; overflow-y: auto; white-space: pre-wrap; line-height: 1.4;
        }
        .gpu-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }
        .gpu-table th, .gpu-table td { padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); }
        .gpu-table th { color: var(--text-muted); font-weight: 600; }
    </style>
</head>
<body>
    <div class="header">
        <div class="title">
            <h1>🚀 Densy-330M • Live Training Monitor</h1>
            <p>4× H100 80GB SXM5 NVLink • 15 Billion Tokens Target • Pure Dense Transformer</p>
        </div>
        <div class="badge"><span class="badge-dot"></span> LIVE TELEMETRY</div>
    </div>

    <div class="grid-4">
        <div class="card">
            <div class="card-lbl">Current Loss</div>
            <div class="card-val" id="loss-val" style="color:#60a5fa;">--</div>
        </div>
        <div class="card">
            <div class="card-lbl">Throughput</div>
            <div class="card-val" id="speed-val" style="color:#34d399;">--</div>
        </div>
        <div class="card">
            <div class="card-lbl">Training Progress</div>
            <div class="card-val" id="tokens-val" style="font-size:16px;">--</div>
            <div class="prog-bar"><div class="prog-fill" id="prog-fill" style="width:0%;"></div></div>
        </div>
        <div class="card">
            <div class="card-lbl">RunPod Cost & Balance</div>
            <div class="card-val" id="cost-val" style="color:#fbbf24; font-size:16px;">--</div>
            <div style="font-size:11px; color:var(--text-muted); margin-top:4px;" id="bal-val">--</div>
        </div>
    </div>

    <div class="charts-grid">
        <div class="chart-card">
            <h3>Cross-Entropy Loss (Lower is Better)</h3>
            <div class="chart-box"><canvas id="lossChart"></canvas></div>
        </div>
        <div class="chart-card">
            <h3>Throughput (Tokens / Second)</h3>
            <div class="chart-box"><canvas id="speedChart"></canvas></div>
        </div>
    </div>

    <div class="card" style="margin-bottom:16px;">
        <div class="card-lbl">GPU Cluster Health (4× H100 SXM)</div>
        <div id="gpu-container">Loading GPU telemetry...</div>
    </div>

    <div class="card">
        <div class="card-lbl" style="margin-bottom:8px;">Live Engine Console Output</div>
        <div class="log-box" id="log-view">Connecting to pod log stream...</div>
    </div>

    <script>
        let lossChart, speedChart;
        function initCharts() {
            const ctxL = document.getElementById('lossChart').getContext('2d');
            lossChart = new Chart(ctxL, {
                type: 'line',
                data: { labels: [], datasets: [{ label: 'Loss', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', fill: true, tension: 0.2 }] },
                options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: true }, y: { display: true } } }
            });
            const ctxS = document.getElementById('speedChart').getContext('2d');
            speedChart = new Chart(ctxS, {
                type: 'line',
                data: { labels: [], datasets: [{ label: 'tok/s', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', fill: true, tension: 0.2 }] },
                options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: true }, y: { display: true } } }
            });
        }

        async function fetchUpdate() {
            try {
                const res = await fetch('/api/data');
                const d = await res.json();
                document.getElementById('loss-val').innerText = d.loss;
                document.getElementById('speed-val').innerText = d.tok_sec;
                document.getElementById('tokens-val').innerText = d.tokens;
                document.getElementById('prog-fill').style.width = Math.min(100, Math.max(0, d.progress_pct)) + '%';
                document.getElementById('cost-val').innerText = d.cost.spent_str + ' spent (' + d.cost.elapsed_str + ')';
                document.getElementById('bal-val').innerText = 'Balance: ' + d.cost.balance_str + ' | Rate: ' + d.cost.rate_str;
                document.getElementById('log-view').innerText = d.logs;
                document.getElementById('log-view').scrollTop = document.getElementById('log-view').scrollHeight;

                // Render GPU
                if (d.gpu) {
                    const lines = d.gpu.split('\\n');
                    let html = '<table class="gpu-table"><thead><tr><th>GPU</th><th>Name</th><th>VRAM Used</th><th>Total</th><th>Util</th><th>Temp</th></tr></thead><tbody>';
                    lines.forEach(l => {
                        const parts = l.split(',').map(s => s.trim());
                        if (parts.length >= 6) {
                            html += `<tr><td>#${parts[0]}</td><td>${parts[1]}</td><td>${parts[2]}</td><td>${parts[3]}</td><td>${parts[4]}</td><td>${parts[5]}</td></tr>`;
                        } else {
                            html += `<tr><td colspan="6">${l}</td></tr>`;
                        }
                    });
                    html += '</tbody></table>';
                    document.getElementById('gpu-container').innerHTML = html;
                }

                if (d.charts && d.charts.steps.length > 0) {
                    lossChart.data.labels = d.charts.steps;
                    lossChart.data.datasets[0].data = d.charts.losses;
                    lossChart.update();

                    speedChart.data.labels = d.charts.steps;
                    speedChart.data.datasets[0].data = d.charts.tok_secs;
                    speedChart.update();
                }
            } catch(e) {}
        }
        initCharts();
        setInterval(fetchUpdate, 3000);
        fetchUpdate();
    </script>
</body>
</html>
"""

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/data":
            data = get_training_progress()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

    def log_message(self, format, *args):
        pass

def run_server(port):
    try:
        class ReusableServer(socketserver.TCPServer):
            allow_reuse_address = True
        server = ReusableServer(("", port), RequestHandler)
        print(f"Monitoring Dashboard live on port {port}")
        server.serve_forever()
    except Exception as e:
        print(f"Port {port} bind error: {e}")

if __name__ == "__main__":
    for port in PORTS:
        t = threading.Thread(target=run_server, args=(port,), daemon=True)
        t.start()
    while True:
        time.sleep(1)
