"""
TaskScheduler: continuous and time-based objective execution.

Supports four scheduling modes:
  run_for(minutes, objective)           — repeat for M minutes
  run_until(time_str, objective)        — repeat until HH:MM
  run_every(interval_seconds, objective)— repeat indefinitely at interval
  monitor(objective, alert_interval)    — run continuously, summarize every N min

All modes log each execution to logs/scheduled_tasks.json and handle
Ctrl+C gracefully by saving a final summary before exit.
"""
import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional


_LOG_PATH = os.path.join("logs", "scheduled_tasks.json")


def _format_remaining(seconds: float) -> str:
    """Format seconds as M:SS string."""
    if seconds <= 0:
        return "0:00"
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def _humanize_result(result: Dict) -> str:
    """Convert tool result to human-readable text.

    Handles visual_scan and describe_scene outputs specially —
    extracts meaningful fields instead of dumping raw vectors.

    If *result* is an execution summary with ``step_results``, formats
    each raw step output instead of the top-level summary.
    """
    # Prefer raw step outputs from execution summary
    step_results = result.get("step_results")
    if step_results and isinstance(step_results, list):
        parts = []
        for sr in step_results:
            if isinstance(sr, dict):
                parts.append(_humanize_dict(sr))
            elif isinstance(sr, str) and sr:
                parts.append(sr[:200])
        if parts:
            text = " | ".join(parts)
            if len(text) > 500:
                text = text[:497] + "..."
            return text

    raw = result.get("result")
    if raw is None:
        return ""
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, dict):
        text = _humanize_dict(raw)
    else:
        text = str(raw)
    # Truncate for log readability
    if len(text) > 500:
        text = text[:497] + "..."
    return text


def _humanize_dict(d: Dict) -> str:
    """Convert a result dict to readable text, handling known tool formats."""
    # Unwrap tool_builder _ok() envelope
    if "success" in d and "result" in d and isinstance(d["result"], dict):
        d = d["result"]

    # describe_scene output
    if "description" in d:
        parts = [d["description"]]
        if "object_count" in d:
            parts.append(f"Objects: {d['object_count']}")
        if "unique_classes" in d:
            parts.append(f"Classes: {d['unique_classes']}")
        return " | ".join(parts)

    # visual_scan output
    if "brightness" in d or "edges" in d or "contours" in d:
        parts = []
        for key in ("brightness", "edges", "contours", "motion",
                    "dominant_color", "scene_type"):
            if key in d:
                val = d[key]
                if isinstance(val, float):
                    parts.append(f"{key}={val:.2f}")
                else:
                    parts.append(f"{key}={val}")
        return "Visual: " + ", ".join(parts) if parts else str(d)

    # Raw battery dict (psutil-style: percent, plugged_in, seconds_left)
    if "percent" in d and "plugged_in" in d:
        batt = d["percent"]
        label = "charging" if d.get("plugged_in") else "discharging"
        secs = d.get("seconds_left") or d.get("secsleft") or 0
        time_str = ""
        if secs and isinstance(secs, (int, float)) and secs > 0:
            h = int(secs) // 3600
            m = (int(secs) % 3600) // 60
            time_str = f" {h}h{m}m remaining"
        batt_fmt = f"{batt:.0f}" if isinstance(batt, float) else str(batt)
        return f"Battery: {batt_fmt}% ({label}){time_str}"

    # YOLO detections
    if "detections" in d and isinstance(d["detections"], list):
        dets = d["detections"]
        if not dets:
            return "No objects detected"
        names = [det.get("class_name", "?") for det in dets[:5]]
        return f"Detected {len(dets)} object(s): {', '.join(names)}"

    # Generic: format key=value pairs, skip raw arrays
    parts = []
    for k, v in d.items():
        if isinstance(v, (list, dict)) and len(str(v)) > 80:
            continue  # skip large nested structures
        if isinstance(v, float):
            parts.append(f"{k}={v:.3f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else str(d)


def _try_parse_dict_string(s: str) -> Optional[Dict]:
    """Try to parse a stringified dict like "{'percent': 85, ...}".

    Handles both Python repr (single quotes) and JSON (double quotes).
    Returns the parsed dict or None.
    """
    s = s.strip()
    if not s.startswith("{"):
        return None
    # Try JSON first
    try:
        import json as _json
        return _json.loads(s)
    except (ValueError, TypeError):
        pass
    # Try Python literal eval (handles single-quoted dicts, True/False/None)
    try:
        import ast
        result = ast.literal_eval(s)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    return None


def format_for_log(result: Dict) -> str:
    """Convert a tool execution result to a single human-readable log line.

    Handles common tool outputs (system_metrics, battery, cpu, visual_scan,
    describe_scene) and formats them as timestamped text.

    If the result is an execution summary (contains ``step_results``),
    extracts and formats the raw tool outputs from each step instead of
    formatting the summary itself.

    Example output::

        2026-04-01 14:32:00 | CPU: 23% | Battery: 87% (charging) | RAM: 45%
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # If this is an execution summary with raw step outputs, format those
    # instead of the top-level status dict.
    step_results = result.get("step_results")
    if step_results and isinstance(step_results, list):
        parts = [ts]
        for sr in step_results:
            if isinstance(sr, dict):
                formatted = _format_single_result(sr)
                if formatted:
                    parts.extend(formatted)
            elif isinstance(sr, str) and sr:
                parts.append(sr[:200])
        if len(parts) > 1:
            return " | ".join(parts)
        # Fall through if step_results were empty/unformattable

    parts = [ts]

    raw = result.get("result")
    if raw is None:
        raw = result

    # If raw is a string, try to parse as a dict (str(prev_output) from executor)
    if isinstance(raw, str):
        parsed = _try_parse_dict_string(raw)
        if parsed is not None:
            formatted = _format_single_result(parsed)
            if formatted:
                return " | ".join([ts] + formatted)
        return f"{ts} | {raw[:200]}"

    if not isinstance(raw, dict):
        return f"{ts} | {str(raw)[:200]}"

    formatted = _format_single_result(raw)
    if formatted:
        parts.extend(formatted)

    return " | ".join(parts)


def _format_single_result(d: Dict) -> List[str]:
    """Extract human-readable metric parts from a single tool output dict.

    Returns a list of formatted strings (e.g. ``["CPU: 23%", "Battery: 87%"]``)
    without a timestamp prefix.  Returns an empty list if nothing matched.

    Handles three wrapper layers that tool results may arrive in:
      1. ``{"success": True, "result": {...}}``  (tool_builder _ok() wrapper)
      2. ``{"cpu": {...}, "battery": {...}, ...}`` (system_metrics composite)
      3. ``{"percent": 85, "plugged_in": False}``  (raw psutil dict)
    """
    # Unwrap tool_builder _ok() envelope: {"success": ..., "result": {...}}
    if "success" in d and "result" in d and isinstance(d["result"], dict):
        d = d["result"]

    # Unwrap system_metrics composite: {"cpu": {...}, "battery": {...}, ...}
    # Detect by checking for sub-dicts named cpu/battery/ram/disk
    _composite_keys = {"cpu", "battery", "ram", "disk"}
    if _composite_keys & set(d.keys()):
        parts: List[str] = []
        # CPU sub-dict
        cpu = d.get("cpu")
        if isinstance(cpu, dict) and "percent" in cpu:
            parts.append(f"CPU: {cpu['percent']}%")
        elif isinstance(cpu, (int, float)):
            parts.append(f"CPU: {cpu:.0f}%")
        # RAM sub-dict
        ram = d.get("ram")
        if isinstance(ram, dict):
            pct = ram.get("percent") or ram.get("ram_percent")
            if pct is not None:
                parts.append(f"RAM: {pct}%")
        # Battery sub-dict
        batt = d.get("battery")
        if isinstance(batt, dict) and "percent" in batt:
            pct = batt["percent"]
            charging = batt.get("plugged_in", False)
            label = "charging" if charging else "discharging"
            secs = batt.get("seconds_left") or batt.get("secsleft") or 0
            time_str = ""
            if secs and isinstance(secs, (int, float)) and secs > 0:
                h = int(secs) // 3600
                m = (int(secs) % 3600) // 60
                time_str = f" {h}h{m}m remaining"
            batt_fmt = f"{pct:.0f}" if isinstance(pct, float) else str(pct)
            parts.append(f"Battery: {batt_fmt}% ({label}){time_str}")
        # Disk sub-dict
        disk = d.get("disk")
        if isinstance(disk, dict) and "percent" in disk:
            parts.append(f"Disk: {disk['percent']}%")
        if parts:
            return parts
        # Fall through if sub-dicts didn't contain expected keys

    parts: List[str] = []

    # CPU
    if "cpu_percent" in d:
        parts.append(f"CPU: {d['cpu_percent']:.0f}%"
                     if isinstance(d['cpu_percent'], float)
                     else f"CPU: {d['cpu_percent']}%")

    # Battery — handle both normalized ("battery_percent") and raw
    # psutil-style ("percent" + "plugged_in" + "seconds_left") dicts.
    if "battery_percent" in d:
        batt = d["battery_percent"]
        charging = d.get("plugged_in", d.get("charging", False))
        label = "charging" if charging else "discharging"
        parts.append(f"Battery: {batt:.0f}% ({label})"
                     if isinstance(batt, float)
                     else f"Battery: {batt}% ({label})")
    elif "percent" in d and "plugged_in" in d:
        batt = d["percent"]
        charging = d.get("plugged_in", False)
        label = "charging" if charging else "discharging"
        secs = d.get("seconds_left") or d.get("secsleft") or 0
        time_str = ""
        if secs and isinstance(secs, (int, float)) and secs > 0:
            h = int(secs) // 3600
            m = (int(secs) % 3600) // 60
            time_str = f" {h}h{m}m remaining"
        batt_fmt = (f"{batt:.0f}" if isinstance(batt, float)
                    else str(batt))
        parts.append(f"Battery: {batt_fmt}% ({label}){time_str}")

    # RAM
    if "ram_percent" in d:
        parts.append(f"RAM: {d['ram_percent']:.0f}%"
                     if isinstance(d['ram_percent'], float)
                     else f"RAM: {d['ram_percent']}%")

    # Disk
    if "disk_percent" in d:
        parts.append(f"Disk: {d['disk_percent']:.0f}%"
                     if isinstance(d['disk_percent'], float)
                     else f"Disk: {d['disk_percent']}%")

    # Temperature
    if "temperature" in d:
        parts.append(f"Temp: {d['temperature']}°C")

    # Visual scan
    if "brightness" in d:
        parts.append(f"Brightness: {d['brightness']:.2f}"
                     if isinstance(d['brightness'], float)
                     else f"Brightness: {d['brightness']}")
    if "contours" in d:
        parts.append(f"Contours: {d['contours']}")
    if "motion" in d:
        parts.append(f"Motion: {'yes' if d['motion'] else 'no'}")

    # describe_scene
    if "description" in d:
        parts.append(d["description"][:120])

    # Detections
    if "detections" in d and isinstance(d["detections"], list):
        names = [det.get("class_name", "?") for det in d["detections"][:5]]
        parts.append(f"Detected: {', '.join(names)}")

    # Status (only if nothing else matched — avoids logging execution summaries)
    if "status" in d and not parts:
        parts.append(f"Status: {d['status']}")

    # Fallback: if no known keys matched, dump key=value
    if not parts:
        for k, v in d.items():
            if isinstance(v, (list, dict)):
                continue
            if isinstance(v, float):
                parts.append(f"{k}: {v:.2f}")
            else:
                parts.append(f"{k}: {v}")
            if len(parts) > 7:
                break

    return parts


def _clean_objective_for_planner(objective: str) -> str:
    """Strip write/save/log/append instructions from the objective.

    The scheduler handles all file I/O, so the planner should only
    receive data-retrieval instructions.

    Example::

        "check system health and log CPU and battery to status.txt"
        → "check system health, get CPU percent and battery status"
    """
    cleaned = objective

    # Remove phrases like "save to file.txt", "log to monitoring.log",
    # "write results to output.txt", "append to log.txt"
    cleaned = re.sub(
        r',?\s*(?:and\s+)?(?:save|write|log|append|record|store|output)'
        r'\s+(?:(?:the\s+)?(?:results?|data|output|metrics)\s+)?'
        r'(?:to|into|in)\s+\S+',
        '', cleaned, flags=re.I)

    # Remove standalone "and log X" / "and save X" at end
    cleaned = re.sub(
        r',?\s*(?:and\s+)?(?:save|write|log|append|record)\s+'
        r'(?:all\s+|the\s+)?(?:results?|data|output|metrics|it|them|everything)'
        r'(?:\s+to\s+\S+)?',
        '', cleaned, flags=re.I)

    # Remove "save to monitoring_log.txt" style standalone phrases
    cleaned = re.sub(
        r',?\s*(?:save|write|log|append)\s+to\s+\S+',
        '', cleaned, flags=re.I)

    # Replace "log X" at start/middle with "get X"
    cleaned = re.sub(
        r'\blog\s+(CPU|battery|RAM|disk|sensor|system|telemetry)',
        r'get \1', cleaned, flags=re.I)

    # Clean up trailing commas, extra spaces, leading/trailing punctuation
    cleaned = re.sub(r'[,;.\s]+$', '', cleaned)
    cleaned = re.sub(r'^[,;.\s]+', '', cleaned)
    cleaned = re.sub(r'\s*,\s*,\s*', ', ', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()

    return cleaned if cleaned else objective


class TaskScheduler:
    """Time-based repeating objective executor.

    Parameters
    ----------
    execute_fn : callable
        ``execute_fn(objective: str) -> dict`` — runs one objective cycle
        and returns a result dict with at least ``status`` and ``faults``.
    """

    def __init__(self, execute_fn: Callable[[str], Dict]):
        self._execute = execute_fn
        self._log: List[Dict] = []
        self._running = False
        self._iteration_counter = 0
        self._session_log: Optional[str] = None

    def session_log_file(self, objective: str) -> str:
        """Determine a consistent log filename for this scheduled session.

        Uses the objective text to derive a stable filename so that
        every run within the same session appends to the same file.
        """
        if self._session_log:
            return self._session_log
        # Derive filename from objective: lowercase, replace spaces, truncate
        slug = re.sub(r'[^a-z0-9]+', '_', objective.lower()).strip('_')[:40]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_log = os.path.join("logs", f"scheduled_{ts}.log")
        return self._session_log

    # ── Public scheduling methods ─────────────────────────────────────

    def run_for(self, minutes: float, objective: str,
                interval: float = 30.0) -> Dict:
        """Execute *objective* every *interval* seconds for *minutes* minutes."""
        deadline = time.time() + minutes * 60
        total_runs = max(1, int((minutes * 60) / interval))
        return self._loop(objective, interval, deadline=deadline,
                          label=f"for {minutes:.0f}min",
                          estimated_runs=total_runs)

    def run_until(self, time_str: str, objective: str,
                  interval: float = 60.0) -> Dict:
        """Execute *objective* every *interval* seconds until clock reaches *time_str*.

        *time_str* is ``HH:MM`` (24-hour).  If that time has already passed
        today, it is interpreted as tomorrow.
        """
        target = self._parse_time(time_str)
        deadline = target.timestamp()
        remaining_s = max(0, deadline - time.time())
        total_runs = max(1, int(remaining_s / interval))
        return self._loop(objective, interval, deadline=deadline,
                          label=f"until {time_str}",
                          estimated_runs=total_runs)

    def run_every(self, interval_seconds: float, objective: str) -> Dict:
        """Execute *objective* every *interval_seconds* indefinitely (Ctrl+C to stop)."""
        return self._loop(objective, interval_seconds, deadline=None,
                          label=f"every {interval_seconds:.0f}s",
                          estimated_runs=None)

    def monitor(self, objective: str,
                alert_interval_minutes: float = 5.0) -> Dict:
        """Run *objective* continuously, printing a summary every *alert_interval_minutes*."""
        interval = alert_interval_minutes * 60
        return self._loop(objective, interval, deadline=None,
                          label=f"monitor (summary every {alert_interval_minutes:.0f}min)",
                          estimated_runs=None)

    # ── Schedule string parser ────────────────────────────────────────

    @staticmethod
    def parse_schedule(schedule: str) -> Optional[Dict]:
        """Parse a ``--schedule`` string into scheduling parameters.

        Accepted formats::

            "every 30s: <objective>"
            "every 2min: <objective>"
            "for 30min: <objective>"
            "for 5min every 30s: <objective>"
            "for 1h: <objective>"
            "until 22:00: <objective>"
            "until 22:00 every 1min: <objective>"
            "monitor 5min: <objective>"

        Returns dict with keys: mode, objective, and mode-specific params.
        Returns None if the string doesn't match any pattern.
        """
        s = schedule.strip()

        # "for Nmin every Xs: objective" (compound: duration + interval)
        m = re.match(
            r'for\s+(\d+(?:\.\d+)?)\s*(m|min|minutes?|h|hours?)'
            r'\s*,?\s*every\s+(\d+(?:\.\d+)?)\s*(s|sec|seconds?|m|min|minutes?)'
            r'\s*:\s*(.+)',
            s, re.I)
        if m:
            dur = float(m.group(1))
            dur_unit = m.group(2).lower()
            if dur_unit.startswith("h"):
                dur *= 60
            intv = float(m.group(3))
            intv_unit = m.group(4).lower()
            if intv_unit.startswith("m"):
                intv *= 60
            return {"mode": "for", "minutes": dur, "interval": intv,
                    "objective": m.group(5).strip()}

        # "until HH:MM every Ns/Nmin: objective" (compound)
        m = re.match(
            r'until\s+(\d{1,2}:\d{2})'
            r'\s*,?\s*every\s+(\d+(?:\.\d+)?)\s*(s|sec|seconds?|m|min|minutes?)'
            r'\s*:\s*(.+)',
            s, re.I)
        if m:
            intv = float(m.group(2))
            intv_unit = m.group(3).lower()
            if intv_unit.startswith("m"):
                intv *= 60
            return {"mode": "until", "time": m.group(1).strip(),
                    "interval": intv, "objective": m.group(4).strip()}

        # "every Ns" / "every Nmin"
        m = re.match(
            r'every\s+(\d+(?:\.\d+)?)\s*(s|sec|seconds?|m|min|minutes?)\s*:\s*(.+)',
            s, re.I)
        if m:
            val = float(m.group(1))
            unit = m.group(2).lower()
            if unit.startswith("m"):
                val *= 60
            return {"mode": "every", "interval": val,
                    "objective": m.group(3).strip()}

        # "for Nmin" / "for Nh"
        m = re.match(
            r'for\s+(\d+(?:\.\d+)?)\s*(m|min|minutes?|h|hours?)\s*:\s*(.+)',
            s, re.I)
        if m:
            val = float(m.group(1))
            unit = m.group(2).lower()
            if unit.startswith("h"):
                val *= 60
            return {"mode": "for", "minutes": val,
                    "objective": m.group(3).strip()}

        # "until HH:MM"
        m = re.match(
            r'until\s+(\d{1,2}:\d{2})\s*:\s*(.+)', s, re.I)
        if m:
            return {"mode": "until", "time": m.group(1).strip(),
                    "objective": m.group(2).strip()}

        # "monitor Nmin"
        m = re.match(
            r'monitor\s+(\d+(?:\.\d+)?)\s*(m|min|minutes?)\s*:\s*(.+)',
            s, re.I)
        if m:
            val = float(m.group(1))
            return {"mode": "monitor", "alert_interval": val,
                    "objective": m.group(3).strip()}

        # ── Natural language fallback ─────────────────────────────────
        # Extract time specs from anywhere in the sentence:
        #   "monitor the camera for 5 minutes, detect people every 30 seconds"
        # becomes: duration=5min, interval=30s, objective=rest of text
        return TaskScheduler._parse_natural_language(s)

    @staticmethod
    def _parse_natural_language(s: str) -> Optional[Dict]:
        """Extract time specs embedded in natural language sentences.

        Scans for patterns like 'for N minutes', 'every N seconds',
        'until HH:MM' anywhere in the string.  Removes the matched
        time phrases and uses the remainder as the objective.
        """
        duration_min: Optional[float] = None
        interval_s: Optional[float] = None
        until_time: Optional[str] = None
        remainder = s

        # Duration: "for N minutes/hours/min/h"
        dur_pat = re.compile(
            r'\bfor\s+(\d+(?:\.\d+)?)\s*'
            r'(minutes?|mins?|hours?|hrs?|h|m|seconds?|secs?|s)\b',
            re.I)
        dur_m = dur_pat.search(remainder)
        if dur_m:
            val = float(dur_m.group(1))
            unit = dur_m.group(2).lower()
            if unit.startswith("h"):
                val *= 60
            elif unit.startswith("s"):
                val /= 60
            duration_min = val
            remainder = remainder[:dur_m.start()] + remainder[dur_m.end():]

        # Interval: "every N seconds/minutes/min/s"
        intv_pat = re.compile(
            r'\bevery\s+(\d+(?:\.\d+)?)\s*'
            r'(seconds?|secs?|s|minutes?|mins?|m)\b',
            re.I)
        intv_m = intv_pat.search(remainder)
        if intv_m:
            val = float(intv_m.group(1))
            unit = intv_m.group(2).lower()
            if unit.startswith("m"):
                val *= 60
            interval_s = val
            remainder = remainder[:intv_m.start()] + remainder[intv_m.end():]

        # Until: "until HH:MM"
        until_pat = re.compile(r'\buntil\s+(\d{1,2}:\d{2})\b', re.I)
        until_m = until_pat.search(remainder)
        if until_m:
            until_time = until_m.group(1)
            remainder = remainder[:until_m.start()] + remainder[until_m.end():]

        # Nothing found — give up
        if duration_min is None and interval_s is None and until_time is None:
            return None

        # Clean up objective: strip leading/trailing punctuation and whitespace
        objective = re.sub(r'[,;.\s]+$', '', remainder)
        objective = re.sub(r'^[,;.\s]+', '', objective)
        # Collapse multiple spaces/commas from removed phrases
        objective = re.sub(r'\s*,\s*,\s*', ', ', objective)
        objective = re.sub(r'\s{2,}', ' ', objective).strip()

        if not objective:
            objective = "monitor"

        # Determine mode
        if until_time:
            return {"mode": "until", "time": until_time,
                    "interval": interval_s or 60.0,
                    "objective": objective}
        if duration_min is not None:
            return {"mode": "for", "minutes": duration_min,
                    "interval": interval_s or 30.0,
                    "objective": objective}
        if interval_s is not None:
            return {"mode": "every", "interval": interval_s,
                    "objective": objective}

        return None

    def dispatch(self, schedule: str) -> Dict:
        """Parse a schedule string and run the appropriate mode."""
        parsed = self.parse_schedule(schedule)
        if parsed is None:
            print(f"  [SCHED] ERROR: Could not parse schedule: {schedule}")
            print(f"  [SCHED] Formats: \"every 30s: ...\", \"for 5min: ...\", "
                  f"\"until 22:00: ...\", \"monitor 5min: ...\"")
            return {"error": f"Could not parse schedule: {schedule}"}

        mode = parsed["mode"]
        obj = parsed["objective"]

        if mode == "every":
            return self.run_every(parsed["interval"], obj)
        elif mode == "for":
            interval = parsed.get("interval", 30.0)
            return self.run_for(parsed["minutes"], obj, interval=interval)
        elif mode == "until":
            interval = parsed.get("interval", 60.0)
            return self.run_until(parsed["time"], obj, interval=interval)
        elif mode == "monitor":
            return self.monitor(obj, parsed["alert_interval"])
        return {"error": f"Unknown mode: {mode}"}

    # ── Core loop ─────────────────────────────────────────────────────

    def _loop(self, objective: str, interval: float,
              deadline: Optional[float], label: str,
              estimated_runs: Optional[int] = None) -> Dict:
        """Central scheduling loop.  Runs objective at *interval* until
        *deadline* (epoch seconds) or indefinitely if *deadline* is None.
        """
        self._running = True
        self._log.clear()
        start = time.time()
        iteration = 0
        last_result: Optional[Dict] = None

        # Clean the objective: strip write/save/log instructions so the
        # planner only does data retrieval.  Scheduler handles all logging.
        clean_obj = _clean_objective_for_planner(objective)

        # Ensure session log file is determined
        if not self._session_log:
            self.session_log_file(objective)

        # Header
        est_str = f" | Estimated runs: {estimated_runs}" if estimated_runs else ""
        print(f"\n  [Scheduler] Starting: {label}")
        print(f"  [Scheduler] Objective: {objective}")
        if clean_obj != objective:
            print(f"  [Scheduler] Planner objective: {clean_obj}")
        print(f"  [Scheduler] Log file: {self._session_log}")
        if deadline:
            total_s = max(0, deadline - start)
            print(f"  [Scheduler] Duration: {_format_remaining(total_s)} "
                  f"| Interval: {interval:.0f}s{est_str}")
        else:
            print(f"  [Scheduler] Interval: {interval:.0f}s | Until Ctrl+C")
        print()

        try:
            while self._running:
                # Check deadline
                now = time.time()
                if deadline and now >= deadline:
                    print(f"\n  [Scheduler] Deadline reached.")
                    break

                iteration += 1
                self._iteration_counter = iteration
                iter_start = now

                # Progress header with Run X/N format
                if estimated_runs:
                    remaining_s = max(0, deadline - now) if deadline else 0
                    print(f"  [Scheduler] Run {iteration}/{estimated_runs} "
                          f"| Time remaining: {_format_remaining(remaining_s)}")
                else:
                    elapsed_total = now - start
                    print(f"  [Scheduler] Run {iteration} "
                          f"| Elapsed: {_format_remaining(elapsed_total)}")

                # Execute cleaned objective (no write/save/log instructions)
                try:
                    result = self._execute(clean_obj)
                    last_result = result
                except Exception as e:
                    result = {"status": "ERROR", "faults": 0,
                              "error": str(e)}

                elapsed_iter = time.time() - iter_start

                # Format result for logging
                human_result = _humanize_result(result)

                # Log entry
                entry = {
                    "iteration": iteration,
                    "timestamp": datetime.now().isoformat(),
                    "elapsed_s": round(elapsed_iter, 2),
                    "status": result.get("status", "UNKNOWN"),
                    "faults": result.get("faults", 0),
                    "tool_chain": result.get("tool_chain", []),
                    "result_text": human_result,
                }
                self._log.append(entry)

                # Auto-append formatted result to session log file
                log_line = format_for_log(result)
                if self._session_log:
                    try:
                        os.makedirs(os.path.dirname(self._session_log) or ".",
                                    exist_ok=True)
                        with open(self._session_log, "a") as lf:
                            lf.write(log_line + "\n")
                    except OSError:
                        pass

                # Status line
                status = result.get("status", "?")
                faults = result.get("faults", 0)
                summary_line = f"  [Scheduler] Result: {status}"
                if faults:
                    summary_line += f" ({faults} fault(s))"
                summary_line += f" | {elapsed_iter:.1f}s"
                if log_line:
                    preview = log_line[:80]
                    if len(log_line) > 80:
                        preview += "..."
                    summary_line += f"\n              {preview}"
                print(summary_line)

                # Sleep until next interval (subtract execution time)
                sleep_time = max(0, interval - (time.time() - iter_start))
                if sleep_time > 0:
                    if deadline:
                        sleep_time = min(sleep_time,
                                         max(0, deadline - time.time()))
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n  [Scheduler] Stopped by user at run #{iteration}.")

        # Final summary
        total_elapsed = time.time() - start
        summary = self._build_summary(objective, label, total_elapsed,
                                      iteration, last_result)
        self._save_log(summary)
        self._print_summary(summary)
        return summary

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_time(time_str: str) -> datetime:
        """Parse HH:MM into a datetime.  If already past, use tomorrow."""
        parts = time_str.strip().split(":")
        hour, minute = int(parts[0]), int(parts[1])
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target

    def _build_summary(self, objective: str, label: str,
                       total_elapsed: float, iterations: int,
                       last_result: Optional[Dict]) -> Dict:
        successes = sum(1 for e in self._log if e["status"] == "SUCCESS")
        total_faults = sum(e.get("faults", 0) for e in self._log)
        return {
            "schedule": label,
            "objective": objective,
            "total_elapsed_s": round(total_elapsed, 2),
            "total_elapsed_min": round(total_elapsed / 60, 1),
            "iterations": iterations,
            "successes": successes,
            "degraded": iterations - successes,
            "total_faults": total_faults,
            "last_status": last_result.get("status") if last_result else None,
            "log": self._log,
        }

    def _save_log(self, summary: Dict) -> None:
        """Append summary (with full log) to logs/scheduled_tasks.json."""
        os.makedirs(os.path.dirname(_LOG_PATH) or ".", exist_ok=True)
        existing = []
        if os.path.exists(_LOG_PATH):
            try:
                with open(_LOG_PATH) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = []
        entry = {
            "completed_at": datetime.now().isoformat(),
            "schedule": summary["schedule"],
            "objective": summary["objective"],
            "total_elapsed_s": summary["total_elapsed_s"],
            "total_elapsed_min": summary["total_elapsed_min"],
            "iterations": summary["iterations"],
            "successes": summary["successes"],
            "degraded": summary["degraded"],
            "total_faults": summary["total_faults"],
            "runs": [
                {
                    "iteration": e["iteration"],
                    "timestamp": e["timestamp"],
                    "elapsed_s": e["elapsed_s"],
                    "status": e["status"],
                    "faults": e["faults"],
                    "tools": e["tool_chain"],
                    "result": e.get("result_text", ""),
                }
                for e in summary["log"]
            ],
        }
        existing.append(entry)
        with open(_LOG_PATH, "w") as f:
            json.dump(existing, f, indent=2)

    @staticmethod
    def _print_summary(summary: Dict) -> None:
        mins = summary["total_elapsed_min"]
        print(f"\n  {'='*50}")
        print(f"  Scheduled Task Complete")
        print(f"  {'='*50}")
        print(f"  Schedule:     {summary['schedule']}")
        print(f"  Objective:    {summary['objective'][:60]}")
        print(f"  Duration:     {mins:.1f} minutes "
              f"({summary['total_elapsed_s']:.0f}s)")
        print(f"  Iterations:   {summary['iterations']}")
        print(f"  Successes:    {summary['successes']}")
        print(f"  Degraded:     {summary['degraded']}")
        print(f"  Total faults: {summary['total_faults']}")
        print(f"  Log saved:    {_LOG_PATH}")
        print(f"  {'='*50}")
