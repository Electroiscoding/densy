#!/usr/bin/env python3
import http.server
import socketserver
import json
import re
import os
import glob
import subprocess
import time
import threading
from pathlib import Path

PORTS = [8888, 8000, 8080]
HOURLY_RATE = float(os.environ.get("HOURLY_RATE", "13.56"))
INITIAL_BALANCE = float(os.environ.get("INITIAL_BALANCE", "304.22"))
HF_REPO_ID = "Phase-Technologies/densy-330m-pretrain"
STATE_FILE = Path("/workspace/checkpoints/.hf_uploaded.json")

def get_train_elapsed_sec():
    try:
        out = subprocess.check_output(
            "ps -eo etimes,cmd | grep run_pretrain.py | grep -v grep | head -n 1",
            shell=True,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        if out:
            return int(out.split()[0])
    except Exception:
        pass
    return 0

def get_live_gpu_data():
    gpus = []
    total_power = 0.0
    total_mem_used = 0.0
    total_mem_cap = 0.0
    avg_util = 0.0

    try:
        cmd = "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=2).decode("utf-8")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 8:
                idx = parts[0]
                name = parts[1]
                mem_used = float(parts[2])
                mem_total = float(parts[3])
                util = float(parts[4])
                temp = float(parts[5])
                power = float(parts[6])
                p_limit = float(parts[7])

                total_power += power
                total_mem_used += mem_used
                total_mem_cap += mem_total
                avg_util += util

                gpus.append({
                    "index": idx,
                    "name": name,
                    "mem_used_mb": mem_used,
                    "mem_total_mb": mem_total,
                    "mem_pct": round((mem_used / max(mem_total, 1)) * 100, 1),
                    "util_pct": util,
                    "temp_c": temp,
                    "power_w": power,
                    "power_limit_w": p_limit,
                })
        if gpus:
            avg_util = round(avg_util / len(gpus), 1)
    except Exception:
        pass

    return {
        "gpus": gpus,
        "total_power_w": round(total_power, 1),
        "total_mem_used_gb": round(total_mem_used / 1024, 1),
        "total_mem_cap_gb": round(total_mem_cap / 1024, 1),
        "avg_util_pct": avg_util,
    }

def get_checkpoints_info():
    uploaded_set = set()
    if STATE_FILE.exists():
        try:
            uploaded_set = set(json.loads(STATE_FILE.read_text()))
        except Exception:
            pass

    ckpts = []
    ckpt_dir = Path("/workspace/checkpoints/densy_330m")
    if ckpt_dir.exists():
        for p in sorted(ckpt_dir.iterdir(), key=os.path.getmtime, reverse=True):
            if p.is_dir():
                try:
                    size_mb = sum(f.stat().st_size for f in p.glob("**/*") if f.is_file()) / (1024 * 1024)
                    mtime = time.strftime("%H:%M:%S", time.localtime(p.stat().st_mtime))
                    is_hf = p.name in uploaded_set
                    ckpts.append({
                        "name": p.name,
                        "size_mb": round(size_mb, 1),
                        "time": mtime,
                        "hf_synced": is_hf
                    })
                except Exception:
                    pass
    return ckpts[:10], len(uploaded_set)

def get_live_telemetry():
    log_paths = [
        "/workspace/densy/train.log",
        "/workspace/train.log",
    ]
    full_text = ""
    log_content = ""

    for lp in log_paths:
        p = Path(lp)
        if p.is_file():
            try:
                with open(p, "r", errors="ignore") as f:
                    t = f.read()
                    if t.strip():
                        full_text = t
                        lines = full_text.splitlines()
                        log_content = "\n".join(lines[-80:])
                        break
            except Exception:
                pass

    step_re = re.compile(
        r"Step\s+(\d+)/(\d+)\s+\|\s+Loss\s+([\d\.]+)\s+\|\s+LR\s+([\d\.e\-]+)\s+\|\s+GradNorm\s+([\d\.]+)\s+\|\s+tok/s\s+([\d\,]+)\s+\|\s+Seen\s+([\d\.]+)B/([\d\.]+)B\s+\|\s+ETA\s+([\d\.]+)h(?:\s+\|\s+Cost\s+\$([\d\.]+))?"
    )

    steps = []
    losses = []
    lrs = []
    gnorms = []
    tok_secs = []

    curr_step = 0
    total_steps = 3576
    curr_loss = 0.0
    curr_lr = 0.0
    curr_gnorm = 0.0
    curr_tok_s = 0
    seen_b = 0.0
    target_b = 15.0
    eta_h = 0.0
    logged_cost = None

    for m in step_re.finditer(full_text):
        st, tot, l, lr, gn, ts, seen, tot_b, eta, cost = m.groups()
        s_int = int(st)
        steps.append(s_int)
        losses.append(round(float(l), 4))
        lrs.append(float(lr))
        gnorms.append(round(float(gn), 3))
        tok_secs.append(int(ts.replace(",", "")))

        curr_step = s_int
        total_steps = int(tot)
        curr_loss = float(l)
        curr_lr = float(lr)
        curr_gnorm = float(gn)
        curr_tok_s = int(ts.replace(",", ""))
        seen_b = float(seen)
        target_b = float(tot_b)
        eta_h = float(eta)
        if cost:
            logged_cost = float(cost)

    elapsed_sec = get_train_elapsed_sec()
    hours = elapsed_sec // 3600
    minutes = (elapsed_sec % 3600) // 60
    seconds = elapsed_sec % 60
    elapsed_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

    if logged_cost is not None and logged_cost > 0:
        spent = logged_cost
    else:
        spent = (elapsed_sec / 3600.0) * HOURLY_RATE

    remaining_balance = max(0.0, INITIAL_BALANCE - spent)

    if curr_tok_s > 0 and seen_b < target_b:
        rem_tokens = (target_b - seen_b) * 1e9
        dyn_eta_sec = rem_tokens / curr_tok_s
        dyn_eta_h = dyn_eta_sec / 3600.0
        est_total_cost = spent + (dyn_eta_h * HOURLY_RATE)
        eta_str = f"{int(dyn_eta_h):02d}h {int((dyn_eta_sec % 3600)//60):02d}m"
    else:
        dyn_eta_h = eta_h
        est_total_cost = spent + (dyn_eta_h * HOURLY_RATE)
        eta_str = f"{dyn_eta_h:.1f}h"

    progress_pct = round((seen_b / max(target_b, 1.0)) * 100, 2)
    gpu_data = get_live_gpu_data()
    checkpoints, hf_sync_count = get_checkpoints_info()

    return {
        "status": "TRAINING" if gpu_data["avg_util_pct"] > 10 or curr_tok_s > 0 else "IDLE",
        "curr_step": curr_step,
        "total_steps": total_steps,
        "loss": round(curr_loss, 4),
        "lr": f"{curr_lr:.2e}",
        "gnorm": round(curr_gnorm, 3),
        "tok_sec": curr_tok_s,
        "seen_b": seen_b,
        "target_b": target_b,
        "progress_pct": progress_pct,
        "eta_str": eta_str,
        "elapsed_str": elapsed_str,
        "spent_str": f"${spent:.2f}",
        "balance_str": f"${remaining_balance:.2f}",
        "rate_str": f"${HOURLY_RATE:.2f}/hr",
        "est_total_cost_str": f"${est_total_cost:.2f}",
        "est_surplus_str": f"${(remaining_balance - (dyn_eta_h * HOURLY_RATE)):.2f}",
        "hf_repo_id": HF_REPO_ID,
        "hf_sync_count": hf_sync_count,
        "charts": {
            "steps": steps,
            "losses": losses,
            "lrs": lrs,
            "gnorms": gnorms,
            "tok_secs": tok_secs,
        },
        "gpus": gpu_data,
        "checkpoints": checkpoints,
        "logs": log_content or "Streaming training engine console output...",
        "timestamp": time.strftime("%H:%M:%S UTC", time.gmtime()),
    }

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
    <title>Densy-330M • Real-Time Engine Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg: #07090e;
            --panel: #0e131f;
            --panel-border: #1a2333;
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.25);
            --green: #10b981;
            --yellow: #f59e0b;
            --red: #ef4444;
            --text-1: #f8fafc;
            --text-2: #94a3b8;
            --text-3: #64748b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: var(--bg);
            color: var(--text-1);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
            padding: 16px;
            max-width: 1440px;
            margin: 0 auto;
            -webkit-font-smoothing: antialiased;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--panel-border);
            margin-bottom: 16px;
        }
        .header h1 {
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .header p { font-size: 13px; color: var(--text-2); margin-top: 3px; }
        .live-tag {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: var(--green);
            font-size: 12px;
            font-weight: 700;
            padding: 5px 12px;
            border-radius: 9999px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .live-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--green); box-shadow: 0 0 10px var(--green);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.3; transform: scale(0.85); } }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }
        .stat-card {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            position: relative;
            overflow: hidden;
        }
        .stat-card::after {
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent), transparent);
            opacity: 0.4;
        }
        .stat-title { font-size: 11px; text-transform: uppercase; font-weight: 700; color: var(--text-3); letter-spacing: 0.05em; }
        .stat-value { font-size: 26px; font-weight: 800; margin: 8px 0 4px; letter-spacing: -0.02em; }
        .stat-sub { font-size: 12px; color: var(--text-2); }
        .prog-bar {
            height: 8px; background: #162032; border-radius: 9999px; overflow: hidden; margin-top: 8px;
        }
        .prog-fill { height: 100%; background: linear-gradient(90deg, #38bdf8, #10b981); width: 0%; transition: width 0.3s ease; }

        .hf-banner {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-left: 4px solid #f59e0b;
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        .hf-banner a {
            color: #38bdf8;
            text-decoration: none;
            font-weight: 700;
        }
        .hf-banner a:hover {
            text-decoration: underline;
        }

        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 14px;
            margin-bottom: 16px;
        }
        @media(max-width: 600px) {
            .charts-grid { grid-template-columns: 1fr; }
        }
        .chart-card {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 16px;
        }
        .chart-header {
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;
        }
        .chart-header h3 { font-size: 13px; font-weight: 700; color: var(--text-2); text-transform: uppercase; }
        .chart-box { position: relative; height: 220px; width: 100%; }

        .cluster-panel {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
        }
        .cluster-head {
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;
        }
        .cluster-head h3 { font-size: 14px; font-weight: 700; text-transform: uppercase; color: var(--text-2); }
        .cluster-metrics { display: flex; gap: 16px; font-size: 12px; font-weight: 600; color: var(--text-2); }
        .gpu-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px;
        }
        .gpu-item {
            background: #090e18; border: 1px solid #1a2538; border-radius: 8px; padding: 12px;
        }
        .gpu-item-head {
            display: flex; justify-content: space-between; align-items: center; font-size: 12px; font-weight: 700; margin-bottom: 8px;
        }
        .gpu-meter {
            height: 6px; background: #1a2538; border-radius: 9999px; overflow: hidden; margin-bottom: 6px;
        }
        .gpu-meter-fill { height: 100%; background: #38bdf8; transition: width 0.3s; }
        .gpu-meta {
            display: flex; justify-content: space-between; font-size: 11px; color: var(--text-3);
        }

        .log-panel {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 16px;
        }
        .log-head {
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;
        }
        .log-head h3 { font-size: 13px; font-weight: 700; text-transform: uppercase; color: var(--text-2); }
        .log-box {
            background: #04070d;
            border: 1px solid #141c2c;
            border-radius: 8px;
            padding: 12px;
            font-family: ui-monospace, "Cascadia Code", "Source Code Pro", Menlo, monospace;
            font-size: 11.5px;
            color: #6ee7b7;
            height: 240px;
            overflow-y: auto;
            white-space: pre-wrap;
            line-height: 1.45;
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>⚡ Densy-330M Real-Time Cluster Telemetry</h1>
            <p id="sub-header">4× H100 80GB SXM5 NVLink • Pure Dense Causal Transformer (24L, 768d, 12H)</p>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
            <div class="live-tag"><span class="live-dot"></span> <span id="status-text">LIVE (1s)</span></div>
            <div style="font-size:12px; color:var(--text-3); font-family:monospace;" id="clock-text">--:--:-- UTC</div>
        </div>
    </div>

    <div class="hf-banner">
        <div>
            <span style="font-size:14px; font-weight:700; color:#fbbf24;">🤗 Hugging Face Hub Auto-Sync: </span>
            <a id="hf-link" href="https://huggingface.co/Phase-Technologies/densy-330m-pretrain" target="_blank">Phase-Technologies/densy-330m-pretrain</a>
        </div>
        <div style="font-size:12px; color:var(--text-2);">
            Sync Status: <b style="color:#10b981;">Active Daemon (Every 10s)</b> • Uploaded: <b id="hf-synced-count" style="color:#38bdf8;">0 ckpts</b>
        </div>
    </div>

    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-title">Cross-Entropy Loss</div>
            <div class="stat-value" id="loss-val" style="color:#38bdf8;">--</div>
            <div class="stat-sub" id="loss-sub">Step: -- / --</div>
        </div>

        <div class="stat-card">
            <div class="stat-title">Throughput</div>
            <div class="stat-value" id="speed-val" style="color:#10b981;">--</div>
            <div class="stat-sub" id="speed-sub">Per GPU: -- tok/s</div>
        </div>

        <div class="stat-card">
            <div class="stat-title">Dataset Progress (15B Target)</div>
            <div class="stat-value" id="progress-val" style="font-size:20px; color:#f8fafc;">--</div>
            <div class="prog-bar"><div class="prog-fill" id="prog-fill"></div></div>
            <div class="stat-sub" style="margin-top:6px;" id="progress-sub">ETA: Calculating...</div>
        </div>

        <div class="stat-card">
            <div class="stat-title">Metered Cost & RunPod Balance</div>
            <div class="stat-value" id="cost-val" style="color:#f59e0b; font-size:22px;">--</div>
            <div class="stat-sub" id="cost-sub">Balance: -- | Rate: $13.56/hr</div>
        </div>
    </div>

    <div class="charts-grid">
        <div class="chart-card">
            <div class="chart-header">
                <h3>Loss Convergence Curve (Step-by-Step)</h3>
                <span id="chart-loss-min" style="font-size:11px; color:#38bdf8; font-weight:700;"></span>
            </div>
            <div class="chart-box"><canvas id="lossChart"></canvas></div>
        </div>

        <div class="chart-card">
            <div class="chart-header">
                <h3>Compute Throughput (Tokens / Sec)</h3>
                <span id="chart-speed-max" style="font-size:11px; color:#10b981; font-weight:700;"></span>
            </div>
            <div class="chart-box"><canvas id="speedChart"></canvas></div>
        </div>
    </div>

    <div class="cluster-panel">
        <div class="cluster-head">
            <h3>GPU Cluster Hardware (4× H100 SXM5 NVLink)</h3>
            <div class="cluster-metrics" id="cluster-summary">
                <span>Avg Core: <b id="gpu-avg-util" style="color:#38bdf8;">--%</b></span>
                <span>Total Power: <b id="gpu-total-power" style="color:#f59e0b;">-- W</b></span>
                <span>VRAM: <b id="gpu-total-vram" style="color:#10b981;">-- GB</b></span>
            </div>
        </div>
        <div class="gpu-grid" id="gpu-cards">
            <!-- Dynamic GPUs -->
        </div>
    </div>

    <div class="log-panel">
        <div class="log-head">
            <h3>Live Distributed Engine Log Stream</h3>
            <div style="font-size:11px; color:var(--text-3);">Auto-refreshing every 1s</div>
        </div>
        <div class="log-box" id="log-console">Connecting to live process...</div>
    </div>

    <script>
        let lossChart, speedChart;
        function initCharts() {
            const optBase = {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b', maxTicksLimit: 10 } },
                    y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b' } }
                }
            };

            const ctxL = document.getElementById('lossChart').getContext('2d');
            lossChart = new Chart(ctxL, {
                type: 'line',
                data: { labels: [], datasets: [{ data: [], borderColor: '#38bdf8', borderWidth: 2, pointRadius: 2, pointHoverRadius: 4, fill: true, backgroundColor: 'rgba(56, 189, 248, 0.08)', tension: 0.15 }] },
                options: optBase
            });

            const ctxS = document.getElementById('speedChart').getContext('2d');
            speedChart = new Chart(ctxS, {
                type: 'line',
                data: { labels: [], datasets: [{ data: [], borderColor: '#10b981', borderWidth: 2, pointRadius: 2, pointHoverRadius: 4, fill: true, backgroundColor: 'rgba(16, 185, 129, 0.08)', tension: 0.15 }] },
                options: optBase
            });
        }

        async function updateDashboard() {
            try {
                const res = await fetch('/api/data?t=' + Date.now());
                const d = await res.json();

                document.getElementById('clock-text').innerText = d.timestamp;
                document.getElementById('status-text').innerText = d.status + ' (1s)';

                document.getElementById('loss-val').innerText = d.loss > 0 ? d.loss.toFixed(4) : '--';
                document.getElementById('loss-sub').innerText = `Step: ${d.curr_step} / ${d.total_steps} (LR: ${d.lr})`;

                const tokPerSecStr = d.tok_sec > 0 ? Number(d.tok_sec).toLocaleString() + ' tok/s' : '--';
                document.getElementById('speed-val').innerText = tokPerSecStr;
                const perGpu = d.tok_sec > 0 ? Math.round(d.tok_sec / 4).toLocaleString() : '--';
                document.getElementById('speed-sub').innerText = `Per GPU: ${perGpu} tok/s | Norm: ${d.gnorm}`;

                document.getElementById('progress-val').innerText = `${d.seen_b.toFixed(2)}B / ${d.target_b.toFixed(1)}B (${d.progress_pct}%)`;
                document.getElementById('prog-fill').style.width = Math.min(100, Math.max(0, d.progress_pct)) + '%';
                document.getElementById('progress-sub').innerText = `ETA: ${d.eta_str} remaining`;

                document.getElementById('cost-val').innerText = `${d.spent_str} spent (${d.elapsed_str})`;
                document.getElementById('cost-sub').innerText = `Balance: ${d.balance_str} | Projected Total: ${d.est_total_cost_str}`;

                document.getElementById('hf-synced-count').innerText = `${d.hf_sync_count} checkpoints synced`;

                // Render GPUs
                if (d.gpus && d.gpus.gpus && d.gpus.gpus.length > 0) {
                    document.getElementById('gpu-avg-util').innerText = d.gpus.avg_util_pct + '%';
                    document.getElementById('gpu-total-power').innerText = d.gpus.total_power_w + ' W';
                    document.getElementById('gpu-total-vram').innerText = `${d.gpus.total_mem_used_gb} / ${d.gpus.total_mem_cap_gb} GB`;

                    let gpuHtml = '';
                    d.gpus.gpus.forEach(g => {
                        gpuHtml += `
                        <div class="gpu-item">
                            <div class="gpu-item-head">
                                <span>#${g.index} ${g.name}</span>
                                <span style="color:${g.util_pct > 80 ? '#10b981' : '#38bdf8'}">${g.util_pct}% Util</span>
                            </div>
                            <div class="gpu-meter"><div class="gpu-meter-fill" style="width:${g.mem_pct}%"></div></div>
                            <div class="gpu-meta">
                                <span>VRAM: ${(g.mem_used_mb/1024).toFixed(1)} / ${(g.mem_total_mb/1024).toFixed(1)} GB (${g.mem_pct}%)</span>
                                <span>${g.power_w}W • ${g.temp_c}°C</span>
                            </div>
                        </div>`;
                    });
                    document.getElementById('gpu-cards').innerHTML = gpuHtml;
                }

                // Render Logs
                const logEl = document.getElementById('log-console');
                const shouldScroll = (logEl.scrollTop + logEl.clientHeight + 50) >= logEl.scrollHeight;
                logEl.innerText = d.logs;
                if (shouldScroll) {
                    logEl.scrollTop = logEl.scrollHeight;
                }

                // Update charts if data points changed
                if (d.charts && d.charts.steps.length > 0) {
                    if (lossChart.data.labels.length !== d.charts.steps.length) {
                        lossChart.data.labels = d.charts.steps;
                        lossChart.data.datasets[0].data = d.charts.losses;
                        lossChart.update();

                        speedChart.data.labels = d.charts.steps;
                        speedChart.data.datasets[0].data = d.charts.tok_secs;
                        speedChart.update();

                        const minLoss = Math.min(...d.charts.losses);
                        const maxSpeed = Math.max(...d.charts.tok_secs);
                        document.getElementById('chart-loss-min').innerText = `Best: ${minLoss.toFixed(4)}`;
                        document.getElementById('chart-speed-max').innerText = `Peak: ${maxSpeed.toLocaleString()} tok/s`;
                    }
                }
            } catch(e) {}
        }

        initCharts();
        setInterval(updateDashboard, 1000);
        updateDashboard();
    </script>
</body>
</html>
"""

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/data"):
            data = get_live_telemetry()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
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
