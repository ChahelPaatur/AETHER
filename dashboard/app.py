"""AETHER v3 — Retro Terminal Web Dashboard.

Usage:
    python -m dashboard.app          # standalone on port 5000
    python main.py --mode dashboard  # integrated via main.py
"""
import glob
import json
import os
import sys
import threading
import time

from flask import Flask, jsonify, request, send_file, render_template_string

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# ── Shared state ──────────────────────────────────────────────────────
_start_time = time.time()
_latest_result = {
    "objective": "", "status": "IDLE", "output": "",
    "faults": 0, "actions": 0,
}
_mission_log = []
_agent_lock = threading.Lock()

# Lazy-init: discovery + tools built once on first objective
_agent_ctx = {"ready": False}

# ── OLED result renderer + idle animation ────────────────────────────
_oled_renderer = None
_oled_lock = threading.Lock()
_idle_stop = threading.Event()
_idle_thread = None


def _get_oled_renderer():
    """Lazy-init a sim-mode OLEDTool for rendering results to PNG."""
    global _oled_renderer
    if _oled_renderer is not None:
        return _oled_renderer
    try:
        from aether.core.tool_builder import OLEDTool
        _oled_renderer = OLEDTool(force_sim=True)
        return _oled_renderer
    except Exception:
        return None


def _dashboard_idle_loop(stop_event):
    """Bouncing ^^ chevron eyes with smile — runs until stopped."""
    oled = _get_oled_renderer()
    if oled is None:
        return
    Image = oled._Image
    ImageDraw = oled._ImageDraw
    if Image is None or ImageDraw is None:
        return
    positions = [0, -2, -4, -2, 0, 2, 4, 2]
    i = 0
    while not stop_event.is_set():
        offset = positions[i % len(positions)]
        image = Image.new('1', (128, 64), 0)
        draw = ImageDraw.Draw(image)

        ey = 28 + offset
        lx, rx = 32, 96
        # ^ chevron eyes
        draw.line([(lx - 8, ey + 4), (lx, ey - 6), (lx + 8, ey + 4)], fill=1, width=2)
        draw.line([(rx - 8, ey + 4), (rx, ey - 6), (rx + 8, ey + 4)], fill=1, width=2)
        # Smile mouth
        draw.arc([44, 48, 84, 62], 0, 180, fill=1, width=2)

        with _oled_lock:
            oled._push(image, tag="idle")
        if stop_event.wait(0.15):
            break
        i += 1


def _start_idle():
    """Start the idle animation thread."""
    global _idle_thread, _idle_stop
    _stop_idle()  # stop any existing
    _idle_stop = threading.Event()
    _idle_thread = threading.Thread(target=_dashboard_idle_loop,
                                    args=(_idle_stop,), daemon=True)
    _idle_thread.start()


def _stop_idle():
    """Stop the idle animation thread."""
    global _idle_thread
    _idle_stop.set()
    if _idle_thread is not None:
        _idle_thread.join(timeout=0.5)
        _idle_thread = None


def _render_result_to_oled(objective: str, status: str, output: str):
    """Render the objective result onto the OLED sim PNG so dashboard shows it."""
    with _oled_lock:
        oled = _get_oled_renderer()
        if oled is None:
            return
        import re
        # Try to extract a key numeric value from the output
        # finger_count
        m = re.search(r"finger_count['\"]?\s*:\s*(\d+)", output)
        if m:
            oled.show_value(label="Fingers", value=m.group(1))
            return
        # cpu_temp
        m = re.search(r"cpu_temp_c['\"]?\s*:\s*([\d.]+)", output)
        if m:
            oled.show_value(label="CPU Temp", value=m.group(1), unit="\u00b0C")
            return
        # cpu_percent
        m = re.search(r"cpu_percent['\"]?\s*:\s*([\d.]+)", output)
        if m:
            oled.show_value(label="CPU", value=m.group(1), unit="%")
            return
        # ram_percent
        m = re.search(r"ram_percent['\"]?\s*:\s*([\d.]+)", output)
        if m:
            oled.show_value(label="RAM", value=m.group(1), unit="%")
            return
        # brightness
        m = re.search(r"brightness['\"]?\s*:\s*([\d.]+)", output)
        if m:
            val = float(m.group(1))
            oled.show_value(label="Bright", value=f"{val*100:.0f}", unit="%")
            return
        # battery
        m = re.search(r"battery['\"]?\s*:\s*([\d.]+)", output)
        if m:
            val = float(m.group(1))
            oled.show_value(label="Battery", value=f"{val*100:.0f}", unit="%")
            return

        # If it's a short text result, display it
        # Find the last [OK] line with arrow for the meaningful result
        for line in reversed(output.split("\n")):
            if "[OK]" in line and "\u2192" in line:
                part = line.split("\u2192")[-1].strip()
                # Skip sim-mode dict junk
                if "mode" in part and "sim" in part:
                    continue
                if len(part) < 80:
                    oled.display_text(part[:60])
                    return

        # Fall back to status + objective on screen
        if status == "SUCCESS":
            oled.draw_face("happy")
        elif status == "ERROR" or status == "TIMEOUT":
            oled.draw_face("alert")
        else:
            oled.display_text(f"{status}\n{objective[:40]}")




def _ensure_agent():
    """One-time init of AETHER agent pipeline (thread-safe)."""
    if _agent_ctx["ready"]:
        return
    from aether.core.tool_registry import ToolRegistry, register_built_tools
    from aether.core.tool_discovery import ToolDiscovery
    from aether.core.llm_planner import LLMPlanner, build_tool_descriptions
    from aether.agents.correction_agent import CorrectionAgent
    from aether.core.goal_parser import GoalParser
    from aether.core.navigation_engine import NavigationEngine
    from aether.core.tool_builder import ToolBuilder

    discovery = ToolDiscovery()
    discovery.discover()

    registry = ToolRegistry()
    builder = ToolBuilder(discovery.manifest)
    built_tools = builder.build_all()
    nav_engine = NavigationEngine(discovery.manifest, tools=built_tools)

    manifest = discovery.manifest
    manifest["tool_descriptions"] = build_tool_descriptions(registry)
    manifest["navigation_actions"] = nav_engine.available_actions()
    manifest["navigation_level"] = nav_engine.level
    manifest["available_tools"] = sorted(
        set(manifest.get("available_tools", []) + nav_engine.available_actions()))

    register_built_tools(registry, built_tools, nav_engine, manifest=manifest)

    llm_planner = LLMPlanner()
    corrector = CorrectionAgent(available_tools=registry.available_tools())
    available = registry.available_tools()

    # Load domain definitions
    defs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "context", "aether_definitions.txt")
    if os.path.exists(defs_path):
        with open(defs_path) as f:
            defs_text = f.read()
        summarizer = registry.get("summarize_text")
        if summarizer and hasattr(summarizer, "set_system_context"):
            summarizer.set_system_context(defs_text)

    _agent_ctx.update({
        "ready": True,
        "discovery": discovery,
        "registry": registry,
        "built_tools": built_tools,
        "llm_planner": llm_planner,
        "corrector": corrector,
        "goal_parser": GoalParser(),
        "manifest": manifest,
        "available": available,
    })


# ── Routes ────────────────────────────────────────────────────────────
_idle_started = False


@app.before_request
def _auto_start_idle():
    global _idle_started
    if not _idle_started:
        _idle_started = True
        _start_idle()


@app.route("/")
def index():
    return render_template_string(_DASHBOARD_HTML)


@app.route("/api/status")
def api_status():
    _ensure_agent()
    ctx = _agent_ctx
    manifest = ctx.get("manifest", {})
    hw = manifest.get("hardware", {})

    cpu_temp = None
    try:
        cpu_temp = int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except Exception:
        pass

    cpu_pct = ram_pct = 0.0
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.1)
        ram_pct = psutil.virtual_memory().percent
    except Exception:
        pass

    return jsonify({
        "platform": manifest.get("platform", "unknown"),
        "version": "3.2.6",
        "capability_score": manifest.get("capability_score", 0),
        "cpu_temp": cpu_temp,
        "ram_percent": round(ram_pct, 1),
        "cpu_percent": round(cpu_pct, 1),
        "camera_available": hw.get("camera", {}).get("available", False),
        "oled_available": hw.get("oled", {}).get("available", False),
        "uptime_seconds": round(time.time() - _start_time, 1),
        **_latest_result,
    })


@app.route("/api/metrics")
def api_metrics():
    cpu_temp = None
    try:
        cpu_temp = int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except Exception:
        pass

    cpu_pct = ram_pct = disk_free = 0.0
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.1)
        ram_pct = psutil.virtual_memory().percent
        disk_free = round(psutil.disk_usage("/").free / 1e9, 1)
    except Exception:
        pass

    return jsonify({
        "cpu_percent": round(cpu_pct, 1),
        "ram_percent": round(ram_pct, 1),
        "cpu_temp": cpu_temp,
        "uptime": round(time.time() - _start_time, 1),
        "disk_free_gb": disk_free,
    })


@app.route("/api/objective", methods=["POST"])
def api_objective():
    data = request.get_json(silent=True) or {}
    objective = data.get("objective", "").strip()
    if not objective:
        return jsonify({"error": "empty objective"}), 400

    _latest_result.update({
        "status": "RUNNING", "objective": objective,
        "output": "", "faults": 0, "actions": 0,
    })

    # Stop idle animation when objective starts
    _stop_idle()

    def _execute():
        with _agent_lock:
            oled = _get_oled_renderer()

            # Companion: thinking face while planning
            if oled:
                with _oled_lock:
                    oled.draw_face("thinking")

            try:
                import subprocess

                # Companion: blink before execution starts
                if oled:
                    time.sleep(0.6)
                    oled.animate_blink(1)

                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                result = subprocess.run(
                    [sys.executable, "main.py", "--mode", "agent",
                     "--no-install", "--no-update", "--once", objective],
                    capture_output=True, text=True, timeout=120,
                    cwd=project_root,
                    env={**os.environ, "AETHER_NO_DISPLAY": "1"},
                )

                # Companion: speaking face while parsing
                if oled:
                    oled.draw_face("speaking")

                output = result.stdout + result.stderr
                # Parse status from output
                if "STATUS:   SUCCESS" in output:
                    status = "SUCCESS"
                elif "STATUS:   DEGRADED" in output:
                    status = "DEGRADED"
                else:
                    status = "ERROR"
                faults = output.count("!! FAULT:")
                actions = 0
                for line in output.split("\n"):
                    if "Actions:" in line:
                        try:
                            actions = int(line.split(":")[1].strip())
                        except (ValueError, IndexError):
                            pass
                out_text = output[-500:]

                # Companion: reaction face
                if oled:
                    if faults == 0:
                        oled.draw_face("happy")
                    else:
                        oled.draw_face("alert")
                    time.sleep(1.5)

                # Render the actual result onto OLED sim PNG (holds until next objective)
                _render_result_to_oled(objective, status, output)

                _latest_result.update({
                    "status": status,
                    "output": out_text,
                    "faults": faults,
                    "actions": actions,
                    "objective": objective,
                })
                _mission_log.insert(0, {
                    "obj": objective, "status": status,
                    "faults": faults, "actions": actions,
                    "time": time.strftime("%H:%M:%S"),
                })
            except subprocess.TimeoutExpired:
                if oled:
                    oled.draw_face("alert")
                    time.sleep(1.5)
                _render_result_to_oled(objective, "TIMEOUT", "")
                _latest_result.update({
                    "status": "TIMEOUT", "output": "Objective timed out after 120s",
                    "faults": 1, "actions": 0, "objective": objective,
                })
                _mission_log.insert(0, {
                    "obj": objective, "status": "TIMEOUT",
                    "faults": 1, "actions": 0,
                    "time": time.strftime("%H:%M:%S"),
                })
            except Exception as e:
                if oled:
                    oled.draw_face("alert")
                    time.sleep(1.5)
                _render_result_to_oled(objective, "ERROR", str(e))
                _latest_result.update({
                    "status": "ERROR", "output": str(e),
                    "faults": 1, "actions": 0, "objective": objective,
                })
                _mission_log.insert(0, {
                    "obj": objective, "status": "ERROR",
                    "faults": 1, "actions": 0,
                    "time": time.strftime("%H:%M:%S"),
                })
            # Cap log
            while len(_mission_log) > 20:
                _mission_log.pop()

            # Hold result on screen for 5 seconds, then restart idle
            time.sleep(5)
            _start_idle()

    threading.Thread(target=_execute, daemon=True).start()
    return jsonify({"status": "queued", "objective": objective})


@app.route("/api/camera")
def api_camera():
    patterns = [
        "logs/captures/frame_*.jpg",
        "logs/captures/*.jpg",
        "logs/captures/*.png",
    ]
    for pat in patterns:
        files = sorted(glob.glob(pat))
        if files:
            mime = "image/png" if files[-1].endswith(".png") else "image/jpeg"
            return send_file(os.path.abspath(files[-1]), mimetype=mime)
    return "", 404


@app.route("/api/oled_sim")
def api_oled_sim():
    """Serve the most recently modified OLED simulation PNG."""
    files = glob.glob("logs/oled_sim/*.png")
    if files:
        newest = max(files, key=os.path.getmtime)
        return send_file(os.path.abspath(newest), mimetype="image/png")
    return "", 404


@app.route("/api/memory")
def api_memory():
    try:
        from aether.core.memory import PersistentMemory
        mem = PersistentMemory()
        entries = mem.recent(10)
        return jsonify(entries)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/oled", methods=["POST"])
def api_oled():
    data = request.get_json(silent=True) or {}
    action = data.get("action", "")
    params = data.get("params", {})

    if not action:
        return jsonify({"error": "missing action"}), 400

    try:
        _ensure_agent()
        oled = _agent_ctx.get("built_tools", {}).get("oled")
        if not oled:
            return jsonify({"error": "OLED not available"}), 404

        fn = getattr(oled, action, None)
        if fn is None or not callable(fn):
            return jsonify({"error": f"unknown action: {action}"}), 400

        result = fn(**params)
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status/stream")
def api_status_stream():
    """Server-Sent Events endpoint for live result updates."""
    def generate():
        while True:
            data = json.dumps({
                **_latest_result,
                "log": _mission_log[:10],
                "uptime": round(time.time() - _start_time, 1),
            })
            yield f"data: {data}\n\n"
            time.sleep(1)
    return app.response_class(generate(), mimetype="text/event-stream")


# ── Dashboard HTML ────────────────────────────────────────────────────

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AETHER v3 — DASHBOARD</title>
<style>
:root {
  --bg: #0a0000;
  --bg-panel: #110000;
  --border: #8b0000;
  --accent: #ff2200;
  --accent-dim: #cc1100;
  --text: #ffffff;
  --text-dim: #ff9999;
  --success: #ff4400;
  --fail: #660000;
  --font: 'Courier New', Courier, monospace;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 13px;
  line-height: 1.5;
  min-height: 100vh;
  overflow-x: hidden;
}
/* CRT flicker */
body::after {
  content: '';
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  pointer-events: none; z-index: 9999;
  background: repeating-linear-gradient(
    0deg, rgba(0,0,0,0.03) 0px, rgba(0,0,0,0.03) 1px, transparent 1px, transparent 2px
  );
  animation: flicker 0.15s infinite alternate;
}
@keyframes flicker {
  0% { opacity: 0.97; }
  100% { opacity: 1; }
}
@keyframes blink {
  0%, 49% { opacity: 1; }
  50%, 100% { opacity: 0; }
}
.container {
  max-width: 900px;
  margin: 0 auto;
  padding: 16px;
}
/* HEADER */
.header {
  text-align: center;
  margin-bottom: 20px;
}
.header pre {
  color: var(--accent);
  font-size: 11px;
  line-height: 1.2;
  display: inline-block;
  text-align: left;
}
.header .subtitle {
  color: var(--text-dim);
  font-size: 11px;
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-top: 4px;
}
.header .version-line {
  color: var(--accent);
  font-size: 12px;
  margin-top: 2px;
}
.cursor {
  animation: blink 1s step-end infinite;
  color: var(--accent);
}
/* PANELS */
.panel {
  border: 1px solid var(--border);
  background: var(--bg-panel);
  margin-bottom: 12px;
}
.panel-title {
  background: var(--border);
  color: var(--text);
  padding: 3px 10px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1px;
}
.panel-body {
  padding: 10px 12px;
}
/* STATUS GRID */
.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 6px;
}
.stat-item {
  display: flex;
  justify-content: space-between;
  padding: 2px 0;
}
.stat-label { color: var(--text-dim); text-transform: uppercase; font-size: 11px; }
.stat-value { color: var(--text); font-weight: bold; }
.cap-bar {
  display: inline-block;
  width: 60px;
  height: 10px;
  background: var(--fail);
  position: relative;
  vertical-align: middle;
  margin-right: 6px;
}
.cap-bar-fill {
  height: 100%;
  background: var(--accent);
  transition: width 0.5s;
}
/* CAMERA */
.camera-wrapper {
  position: relative;
  text-align: center;
  min-height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.camera-wrapper img {
  max-width: 100%;
  max-height: 300px;
  image-rendering: pixelated;
}
.camera-wrapper .scanlines {
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  pointer-events: none;
  background: repeating-linear-gradient(
    0deg, rgba(0,0,0,0.15) 0px, rgba(0,0,0,0.15) 1px, transparent 1px, transparent 3px
  );
}
.camera-placeholder {
  color: var(--text-dim);
  font-size: 12px;
  text-transform: uppercase;
}
/* OBJECTIVE INPUT */
.obj-form {
  display: flex;
  gap: 8px;
}
.obj-input {
  flex: 1;
  background: #000;
  border: 1px solid var(--border);
  color: #00ff00;
  font-family: var(--font);
  font-size: 13px;
  padding: 8px 10px;
  outline: none;
  caret-color: #00ff00;
}
.obj-input:focus { border-color: var(--accent); }
.obj-input::placeholder { color: #336633; }
.btn {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  color: var(--accent);
  font-family: var(--font);
  font-size: 12px;
  padding: 8px 16px;
  cursor: pointer;
  text-transform: uppercase;
  letter-spacing: 1px;
  transition: all 0.15s;
}
.btn:hover {
  background: var(--accent);
  color: #000;
}
.btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
/* LAST RESULT */
.result-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 4px;
  margin-bottom: 8px;
}
.result-output {
  color: var(--text-dim);
  font-size: 12px;
  max-height: 120px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-word;
  margin-top: 6px;
  padding: 6px;
  background: rgba(0,0,0,0.5);
  border: 1px solid var(--fail);
}
/* MISSION LOG */
.log-list {
  max-height: 200px;
  overflow-y: auto;
}
.log-entry {
  display: flex;
  gap: 10px;
  padding: 3px 0;
  font-size: 12px;
  border-bottom: 1px solid rgba(139,0,0,0.3);
}
.log-tag {
  font-weight: bold;
  min-width: 40px;
}
.log-tag.ok { color: var(--success); }
.log-tag.err { color: var(--fail); }
.log-obj { flex: 1; color: var(--text-dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.log-status { min-width: 70px; text-align: right; }
.log-time { color: var(--text-dim); min-width: 60px; text-align: right; font-size: 11px; }
/* OLED CONTROLS */
.oled-btns {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 8px;
}
.oled-text-form {
  display: flex;
  gap: 6px;
}
.oled-input {
  flex: 1;
  background: #000;
  border: 1px solid var(--border);
  color: var(--text);
  font-family: var(--font);
  font-size: 12px;
  padding: 6px 8px;
  outline: none;
}
/* OLED PREVIEW */
.oled-preview-wrapper {
  text-align: center;
  min-height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.oled-preview-wrapper img {
  image-rendering: pixelated;
  image-rendering: crisp-edges;
  width: 256px;
  height: 128px;
  border: 1px solid var(--border);
  background: #000;
}
.oled-no-preview {
  color: var(--text-dim);
  font-size: 12px;
  text-transform: uppercase;
}
/* MOBILE */
@media (max-width: 600px) {
  .container { padding: 8px; }
  .header pre { font-size: 8px; }
  .status-grid { grid-template-columns: 1fr; }
  .result-grid { grid-template-columns: 1fr; }
  .obj-form { flex-direction: column; }
  .oled-btns { justify-content: center; }
}
</style>
</head>
<body>
<div class="container">

<!-- HEADER -->
<div class="header">
<pre>
    &#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;
   &#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;
  &#9608;&#9608;&#9608;&#9608;&#9608;&#9556;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9559;&#9608;&#9608;&#9608;&#9608;&#9608;
  &#9608;&#9608;&#9608;&#9608;&#9608;&#9553;  &#9604;&#9608;&#9608;&#9608;&#9608;&#9604;   &#9604;&#9608;&#9608;&#9608;&#9608;&#9604;  &#9553;&#9608;&#9608;&#9608;&#9608;&#9608;
  &#9608;&#9608;&#9608;&#9608;&#9608;&#9553; &#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;  &#9553;&#9608;&#9608;&#9608;&#9608;&#9608;
  &#9608;&#9608;&#9608;&#9608;&#9608;&#9553;  &#9612; AETHER  v3 &#9616;  &#9553;&#9608;&#9608;&#9608;&#9608;&#9608;
  &#9608;&#9608;&#9608;&#9608;&#9608;&#9562;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9565;&#9608;&#9608;&#9608;&#9608;&#9608;
   &#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;&#9608;
</pre>
<div class="subtitle">ADAPTIVE EMBODIED TASK HIERARCHY FOR EXECUTABLE ROBOTICS</div>
<div class="version-line">DRL-FIRST HYBRID FDIR &middot; v3.0 &nbsp; <span class="cursor">&#9679; ONLINE_</span></div>
</div>

<!-- SYSTEM STATUS -->
<div class="panel">
  <div class="panel-title">&#9484;&#9472; SYSTEM STATUS &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9488;</div>
  <div class="panel-body">
    <div class="status-grid">
      <div class="stat-item"><span class="stat-label">CPU TEMP</span><span class="stat-value" id="s-temp">--</span></div>
      <div class="stat-item"><span class="stat-label">CPU</span><span class="stat-value" id="s-cpu">--%</span></div>
      <div class="stat-item"><span class="stat-label">RAM</span><span class="stat-value" id="s-ram">--%</span></div>
      <div class="stat-item"><span class="stat-label">UPTIME</span><span class="stat-value" id="s-uptime">00:00:00</span></div>
      <div class="stat-item"><span class="stat-label">PLATFORM</span><span class="stat-value" id="s-platform">--</span></div>
      <div class="stat-item">
        <span class="stat-label">CAP SCORE</span>
        <span class="stat-value">
          <span class="cap-bar"><span class="cap-bar-fill" id="s-capbar"></span></span>
          <span id="s-capscore">--</span>
        </span>
      </div>
      <div class="stat-item"><span class="stat-label">CAMERA</span><span class="stat-value" id="s-camera">--</span></div>
      <div class="stat-item"><span class="stat-label">OLED</span><span class="stat-value" id="s-oled">--</span></div>
    </div>
  </div>
</div>

<!-- CAMERA FEED -->
<div class="panel">
  <div class="panel-title">&#9484;&#9472; CAMERA FEED &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9488;</div>
  <div class="panel-body">
    <div class="camera-wrapper">
      <img id="cam-img" style="display:none" alt="camera feed">
      <span id="cam-placeholder" class="camera-placeholder">[ NO FEED ]</span>
      <div class="scanlines"></div>
    </div>
  </div>
</div>

<!-- OBJECTIVE INPUT -->
<div class="panel">
  <div class="panel-title">&#9484;&#9472; OBJECTIVE INPUT &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9488;</div>
  <div class="panel-body">
    <div class="obj-form">
      <input type="text" class="obj-input" id="obj-input" placeholder="> enter objective..." autocomplete="off">
      <button class="btn" id="obj-btn" onclick="submitObjective()">EXECUTE</button>
    </div>
  </div>
</div>

<!-- LAST RESULT -->
<div class="panel">
  <div class="panel-title">&#9484;&#9472; LAST RESULT &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9488;</div>
  <div class="panel-body">
    <div class="result-grid">
      <div class="stat-item"><span class="stat-label">STATUS</span><span class="stat-value" id="r-status">IDLE</span></div>
      <div class="stat-item"><span class="stat-label">FAULTS</span><span class="stat-value" id="r-faults">0</span></div>
      <div class="stat-item"><span class="stat-label">ACTIONS</span><span class="stat-value" id="r-actions">0</span></div>
    </div>
    <div style="text-align:center; margin-top:10px;">
      <div style="color:var(--text-dim); font-size:11px; text-transform:uppercase; margin-bottom:4px;">OLED OUTPUT:</div>
      <img id="r-oled" src="/api/oled_sim"
           style="image-rendering:pixelated; image-rendering:crisp-edges;
                  width:256px; height:128px; border:1px solid var(--border);
                  background:#000; display:block; margin:0 auto;"
           onerror="this.style.display='none'">
    </div>
  </div>
</div>

<!-- MISSION LOG -->
<div class="panel">
  <div class="panel-title">&#9484;&#9472; MISSION LOG &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9488;</div>
  <div class="panel-body">
    <div class="log-list" id="log-list">
      <div style="color:var(--text-dim); font-size:12px;">No missions yet.</div>
    </div>
  </div>
</div>

<!-- OLED CONTROLS -->
<div class="panel">
  <div class="panel-title">&#9484;&#9472; OLED CONTROLS &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9488;</div>
  <div class="panel-body">
    <div class="oled-btns">
      <button class="btn" onclick="oledFace('happy')">HAPPY</button>
      <button class="btn" onclick="oledFace('thinking')">THINK</button>
      <button class="btn" onclick="oledFace('alert')">ALERT</button>
      <button class="btn" onclick="oledFace('sleeping')">SLEEP</button>
      <button class="btn" onclick="oledFace('neutral')">IDLE</button>
    </div>
    <div class="oled-text-form">
      <input type="text" class="oled-input" id="oled-text" placeholder="display text...">
      <button class="btn" onclick="oledText()">SEND</button>
    </div>
  </div>
</div>

<!-- OLED PREVIEW -->
<div class="panel">
  <div class="panel-title">&#9484;&#9472; OLED PREVIEW &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9488;</div>
  <div class="panel-body">
    <div class="oled-preview-wrapper">
      <img id="oled-img" style="display:none" alt="OLED display">
      <span id="oled-placeholder" class="oled-no-preview">[ NO DISPLAY ]</span>
    </div>
  </div>
</div>

</div><!-- container -->

<script>
// ── Uptime counter ──
let uptimeBase = 0;
let uptimeStart = Date.now();
function fmtUptime(s) {
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), ss = Math.floor(s%60);
  return String(h).padStart(2,'0')+':'+String(m).padStart(2,'0')+':'+String(ss).padStart(2,'0');
}
setInterval(() => {
  const elapsed = uptimeBase + (Date.now() - uptimeStart) / 1000;
  document.getElementById('s-uptime').textContent = fmtUptime(elapsed);
}, 1000);

// ── Metrics polling (3s) ──
async function fetchMetrics() {
  try {
    const r = await fetch('/api/metrics');
    const d = await r.json();
    document.getElementById('s-cpu').textContent = d.cpu_percent + '%';
    document.getElementById('s-ram').textContent = d.ram_percent + '%';
    document.getElementById('s-temp').textContent = d.cpu_temp != null ? d.cpu_temp.toFixed(1)+'°C' : 'N/A';
    uptimeBase = d.uptime;
    uptimeStart = Date.now();
  } catch(e) {}
}
setInterval(fetchMetrics, 3000);

// ── Live status via SSE (with polling fallback) ──
function updateResult(d) {
  document.getElementById('r-status').textContent = d.status || 'IDLE';
  document.getElementById('r-faults').textContent = d.faults || 0;
  document.getElementById('r-actions').textContent = d.actions || 0;
  if (d.status !== 'RUNNING') {
    document.getElementById('obj-btn').disabled = false;
    document.getElementById('obj-btn').textContent = 'EXECUTE';
  }
  // Update log from SSE data
  if (d.log && Array.isArray(d.log) && d.log.length > 0) {
    renderLog(d.log);
  }
}
function renderLog(entries) {
  const list = document.getElementById('log-list');
  list.innerHTML = entries.map(e => {
    const obj = e.obj || e.objective || '?';
    const st = e.status || e.outcome || '?';
    const isOk = st === 'SUCCESS';
    const tag = isOk ? 'OK' : 'ERR';
    const cls = isOk ? 'ok' : 'err';
    const t = e.time || '';
    return '<div class="log-entry">' +
      '<span class="log-tag '+cls+'">['+tag+']</span>' +
      '<span class="log-obj">'+escHtml(obj)+'</span>' +
      '<span class="log-status">'+st+'</span>' +
      '<span class="log-time">'+t+'</span></div>';
  }).join('');
}
// Try SSE first, fall back to polling
let sseActive = false;
try {
  const evtSrc = new EventSource('/api/status/stream');
  evtSrc.onmessage = function(e) {
    sseActive = true;
    try { updateResult(JSON.parse(e.data)); } catch(err) {}
  };
  evtSrc.onerror = function() { sseActive = false; };
} catch(e) {}

// Polling fallback for status (only if SSE fails)
async function fetchStatus() {
  if (sseActive) return;
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    document.getElementById('s-platform').textContent = (d.platform||'').toUpperCase();
    document.getElementById('s-capscore').textContent = d.capability_score;
    document.getElementById('s-capbar').style.width = d.capability_score + '%';
    document.getElementById('s-camera').textContent = d.camera_available ? 'OK' : 'N/A';
    document.getElementById('s-oled').textContent = d.oled_available ? 'OK' : 'SIM';
    updateResult(d);
  } catch(e) {}
}
setInterval(fetchStatus, 3000);

// Status panel (platform, cap score, etc.) — always poll since SSE doesn't include these
async function fetchStatusPanel() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    document.getElementById('s-platform').textContent = (d.platform||'').toUpperCase();
    document.getElementById('s-capscore').textContent = d.capability_score;
    document.getElementById('s-capbar').style.width = d.capability_score + '%';
    document.getElementById('s-camera').textContent = d.camera_available ? 'OK' : 'N/A';
    document.getElementById('s-oled').textContent = d.oled_available ? 'OK' : 'SIM';
  } catch(e) {}
}
fetchStatusPanel();
setInterval(fetchStatusPanel, 10000);

// ── Camera refresh (5s) ──
async function fetchCamera() {
  try {
    const r = await fetch('/api/camera');
    if (r.ok) {
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const img = document.getElementById('cam-img');
      img.src = url;
      img.style.display = 'block';
      document.getElementById('cam-placeholder').style.display = 'none';
    }
  } catch(e) {}
}
setInterval(fetchCamera, 5000);

// ── OLED image refresh (2s, cache-bust) ──
setInterval(() => {
  const ts = Date.now();
  const rOled = document.getElementById('r-oled');
  if (rOled) { rOled.src = '/api/oled_sim?' + ts; rOled.style.display = 'block'; }
  const pOled = document.getElementById('oled-img');
  if (pOled) { pOled.src = '/api/oled_sim?' + ts; pOled.style.display = 'block';
    document.getElementById('oled-placeholder').style.display = 'none'; }
}, 2000);

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// ── Objective submit ──
function submitObjective() {
  const inp = document.getElementById('obj-input');
  const obj = inp.value.trim();
  if (!obj) return;
  document.getElementById('obj-btn').disabled = true;
  document.getElementById('obj-btn').textContent = 'RUNNING...';
  document.getElementById('r-status').textContent = 'RUNNING';
  fetch('/api/objective', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({objective: obj})
  }).then(r => r.json()).then(d => {
    inp.value = '';
  }).catch(e => {
    document.getElementById('obj-btn').disabled = false;
    document.getElementById('obj-btn').textContent = 'EXECUTE';
  });
}
document.getElementById('obj-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') submitObjective();
});

// ── OLED controls ──
function oledFace(expr) {
  fetch('/api/oled', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action: 'draw_face', params: {expression: expr}})
  });
}
function oledText() {
  const text = document.getElementById('oled-text').value.trim();
  if (!text) return;
  fetch('/api/oled', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action: 'display_text', params: {text: text}})
  });
  document.getElementById('oled-text').value = '';
}

// ── Initial fetches ──
fetchMetrics();
fetchStatus();
fetchCamera();
</script>
</body>
</html>
"""


# ── Standalone entry point ────────────────────────────────────────────
if __name__ == "__main__":
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "localhost"

    # Start idle animation on launch
    _start_idle()

    print()
    print("  ┌─[ AETHER DASHBOARD ]─────────────────────┐")
    print(f"  │  http://{local_ip}:5000                    │")
    print(f"  │  http://localhost:5000                    │")
    print("  └───────────────────────────────────────────┘")
    print()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
