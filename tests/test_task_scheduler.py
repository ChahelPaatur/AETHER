"""Tests for aether.core.task_scheduler.TaskScheduler."""
import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aether.core.task_scheduler import (
    TaskScheduler, _humanize_result, _format_remaining, format_for_log,
    _clean_objective_for_planner, _format_single_result,
    _try_parse_dict_string,
)


def _mock_execute(objective: str) -> dict:
    """Instant mock executor."""
    return {"status": "SUCCESS", "faults": 0, "tool_chain": ["mock"],
            "result": f"ran: {objective}"}


class TestFormatRemaining(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(_format_remaining(0), "0:00")

    def test_seconds(self):
        self.assertEqual(_format_remaining(65), "1:05")

    def test_large(self):
        self.assertEqual(_format_remaining(3661), "61:01")


class TestHumanizeResult(unittest.TestCase):
    def test_string_result(self):
        r = {"result": "hello world"}
        self.assertEqual(_humanize_result(r), "hello world")

    def test_describe_scene(self):
        r = {"result": {"description": "I can see: 2 cats",
                         "object_count": 2, "unique_classes": 1}}
        text = _humanize_result(r)
        self.assertIn("I can see: 2 cats", text)
        self.assertIn("Objects: 2", text)

    def test_visual_scan(self):
        r = {"result": {"brightness": 0.72, "edges": 150,
                         "contours": 12, "motion": False}}
        text = _humanize_result(r)
        self.assertIn("Visual:", text)
        self.assertIn("brightness=0.72", text)

    def test_detections(self):
        r = {"result": {"detections": [
            {"class_name": "person"}, {"class_name": "car"}]}}
        text = _humanize_result(r)
        self.assertIn("person", text)
        self.assertIn("car", text)

    def test_raw_battery_dict(self):
        r = {"result": {"percent": 87, "plugged_in": False,
                         "seconds_left": 16740}}
        text = _humanize_result(r)
        self.assertIn("Battery: 87%", text)
        self.assertIn("discharging", text)
        self.assertIn("4h39m remaining", text)

    def test_none_result(self):
        self.assertEqual(_humanize_result({}), "")


class TestParseSchedule(unittest.TestCase):
    """Test schedule string parsing."""

    def test_every_seconds(self):
        r = TaskScheduler.parse_schedule("every 30s: scan environment")
        self.assertEqual(r["mode"], "every")
        self.assertAlmostEqual(r["interval"], 30.0)
        self.assertEqual(r["objective"], "scan environment")

    def test_every_minutes(self):
        r = TaskScheduler.parse_schedule("every 2min: check status")
        self.assertEqual(r["mode"], "every")
        self.assertAlmostEqual(r["interval"], 120.0)

    def test_for_minutes(self):
        r = TaskScheduler.parse_schedule("for 30min: monitor camera")
        self.assertEqual(r["mode"], "for")
        self.assertAlmostEqual(r["minutes"], 30.0)
        self.assertEqual(r["objective"], "monitor camera")

    def test_for_hours(self):
        r = TaskScheduler.parse_schedule("for 2h: record data")
        self.assertEqual(r["mode"], "for")
        self.assertAlmostEqual(r["minutes"], 120.0)

    def test_for_with_interval(self):
        r = TaskScheduler.parse_schedule(
            "for 5min every 30s: scan environment and log detections")
        self.assertEqual(r["mode"], "for")
        self.assertAlmostEqual(r["minutes"], 5.0)
        self.assertAlmostEqual(r["interval"], 30.0)
        self.assertIn("scan environment", r["objective"])

    def test_for_with_interval_comma(self):
        r = TaskScheduler.parse_schedule(
            "for 10min, every 1min: check battery")
        self.assertEqual(r["mode"], "for")
        self.assertAlmostEqual(r["minutes"], 10.0)
        self.assertAlmostEqual(r["interval"], 60.0)

    def test_until(self):
        r = TaskScheduler.parse_schedule("until 22:00: log telemetry")
        self.assertEqual(r["mode"], "until")
        self.assertEqual(r["time"], "22:00")
        self.assertEqual(r["objective"], "log telemetry")

    def test_until_with_interval(self):
        r = TaskScheduler.parse_schedule(
            "until 22:00 every 1min: record flight telemetry")
        self.assertEqual(r["mode"], "until")
        self.assertEqual(r["time"], "22:00")
        self.assertAlmostEqual(r["interval"], 60.0)
        self.assertIn("flight telemetry", r["objective"])

    def test_monitor(self):
        r = TaskScheduler.parse_schedule("monitor 5min: watch environment")
        self.assertEqual(r["mode"], "monitor")
        self.assertAlmostEqual(r["alert_interval"], 5.0)
        self.assertEqual(r["objective"], "watch environment")

    def test_invalid_returns_none(self):
        r = TaskScheduler.parse_schedule("just do something")
        self.assertIsNone(r)


class TestNaturalLanguageParsing(unittest.TestCase):
    """Test natural language time extraction from full sentences."""

    def test_full_sentence_for_and_every(self):
        r = TaskScheduler.parse_schedule(
            "monitor the camera for 5 minutes, detect any people "
            "every 30 seconds, save to monitoring_log.txt")
        self.assertIsNotNone(r)
        self.assertEqual(r["mode"], "for")
        self.assertAlmostEqual(r["minutes"], 5.0)
        self.assertAlmostEqual(r["interval"], 30.0)
        self.assertIn("monitor the camera", r["objective"])
        self.assertIn("save to monitoring_log.txt", r["objective"])
        # Time specs should be stripped from objective
        self.assertNotIn("for 5 minutes", r["objective"])
        self.assertNotIn("every 30 seconds", r["objective"])

    def test_sentence_with_every_only(self):
        r = TaskScheduler.parse_schedule(
            "check battery status every 60 seconds")
        self.assertIsNotNone(r)
        self.assertEqual(r["mode"], "every")
        self.assertAlmostEqual(r["interval"], 60.0)
        self.assertIn("check battery status", r["objective"])

    def test_sentence_with_for_only(self):
        r = TaskScheduler.parse_schedule(
            "scan environment for 10 minutes")
        self.assertIsNotNone(r)
        self.assertEqual(r["mode"], "for")
        self.assertAlmostEqual(r["minutes"], 10.0)
        self.assertIn("scan environment", r["objective"])

    def test_sentence_with_until(self):
        r = TaskScheduler.parse_schedule(
            "record flight telemetry until 22:00 every 1 min")
        self.assertIsNotNone(r)
        self.assertEqual(r["mode"], "until")
        self.assertEqual(r["time"], "22:00")
        self.assertAlmostEqual(r["interval"], 60.0)
        self.assertIn("record flight telemetry", r["objective"])

    def test_sentence_hours(self):
        r = TaskScheduler.parse_schedule(
            "log sensor data for 2 hours")
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r["minutes"], 120.0)


class TestRunFor(unittest.TestCase):
    """Test run_for executes within time bounds."""

    def test_run_for_short(self):
        scheduler = TaskScheduler(_mock_execute)
        result = scheduler.run_for(0.05, "test objective", interval=1.0)
        self.assertGreaterEqual(result["iterations"], 1)
        self.assertEqual(result["objective"], "test objective")
        self.assertIn("schedule", result)
        self.assertGreaterEqual(result["successes"], 1)

    def test_logs_have_result_text(self):
        scheduler = TaskScheduler(_mock_execute)
        result = scheduler.run_for(0.03, "log test", interval=0.5)
        self.assertGreater(len(result["log"]), 0)
        for entry in result["log"]:
            self.assertIn("iteration", entry)
            self.assertIn("status", entry)
            self.assertIn("timestamp", entry)
            self.assertIn("result_text", entry)

    def test_run_for_with_custom_interval(self):
        calls = []
        def _counting(obj):
            calls.append(time.time())
            return {"status": "SUCCESS", "faults": 0, "tool_chain": []}
        scheduler = TaskScheduler(_counting)
        scheduler.run_for(0.05, "fast", interval=0.5)
        self.assertGreaterEqual(len(calls), 1)


class TestRunEvery(unittest.TestCase):
    """Test run_every via dispatch."""

    def test_dispatch_every(self):
        calls = []

        def _counting_execute(obj):
            calls.append(obj)
            if len(calls) >= 3:
                raise KeyboardInterrupt
            return {"status": "SUCCESS", "faults": 0, "tool_chain": []}

        scheduler = TaskScheduler(_counting_execute)
        result = scheduler.run_every(0.1, "repeat task")
        self.assertGreaterEqual(result["iterations"], 2)


class TestDispatch(unittest.TestCase):
    """Test dispatch routes correctly."""

    def test_dispatch_for(self):
        scheduler = TaskScheduler(_mock_execute)
        result = scheduler.dispatch("for 0.03min: quick test")
        self.assertIn("iterations", result)

    def test_dispatch_for_with_interval(self):
        scheduler = TaskScheduler(_mock_execute)
        result = scheduler.dispatch(
            "for 0.05min every 1s: interval test")
        self.assertIn("iterations", result)
        self.assertGreaterEqual(result["iterations"], 1)

    def test_dispatch_invalid(self):
        scheduler = TaskScheduler(_mock_execute)
        result = scheduler.dispatch("nonsense string")
        self.assertIn("error", result)


class TestLogSaving(unittest.TestCase):
    """Test that logs are saved to disk."""

    def test_log_file_created(self):
        log_path = os.path.join("logs", "scheduled_tasks.json")
        scheduler = TaskScheduler(_mock_execute)
        scheduler.run_for(0.02, "log save test", interval=0.5)
        self.assertTrue(os.path.exists(log_path))
        with open(log_path) as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        last = data[-1]
        self.assertIn("runs", last)
        self.assertIn("objective", last)
        # Each run entry should have human-readable result
        if last["runs"]:
            self.assertIn("result", last["runs"][0])


class TestFormatForLog(unittest.TestCase):
    """Test format_for_log produces human-readable lines."""

    def test_cpu_result(self):
        r = {"result": {"cpu_percent": 23.0}}
        line = format_for_log(r)
        self.assertIn("CPU: 23%", line)
        # Should start with timestamp
        self.assertRegex(line, r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')

    def test_battery_result(self):
        r = {"result": {"battery_percent": 87.0, "plugged_in": True}}
        line = format_for_log(r)
        self.assertIn("Battery: 87%", line)
        self.assertIn("charging", line)

    def test_combined_metrics(self):
        r = {"result": {"cpu_percent": 12.0, "ram_percent": 45.0,
                         "battery_percent": 92.0, "plugged_in": False}}
        line = format_for_log(r)
        self.assertIn("CPU: 12%", line)
        self.assertIn("RAM: 45%", line)
        self.assertIn("discharging", line)

    def test_describe_scene(self):
        r = {"result": {"description": "I can see: 2 cats, 1 dog"}}
        line = format_for_log(r)
        self.assertIn("I can see: 2 cats", line)

    def test_string_result(self):
        r = {"result": "simple string output"}
        line = format_for_log(r)
        self.assertIn("simple string output", line)

    def test_raw_battery_dict(self):
        """Raw psutil-style battery dict with percent/plugged_in/seconds_left."""
        r = {"result": {"percent": 87, "plugged_in": False,
                         "seconds_left": 16740}}
        line = format_for_log(r)
        self.assertIn("Battery: 87%", line)
        self.assertIn("discharging", line)
        self.assertIn("4h39m remaining", line)

    def test_raw_battery_charging_no_seconds(self):
        r = {"result": {"percent": 95, "plugged_in": True,
                         "seconds_left": 0}}
        line = format_for_log(r)
        self.assertIn("Battery: 95%", line)
        self.assertIn("charging", line)
        self.assertNotIn("remaining", line)

    def test_status_key_in_result(self):
        r = {"result": {"status": "healthy"}}
        line = format_for_log(r)
        self.assertIn("Status: healthy", line)

    def test_visual_scan(self):
        r = {"result": {"brightness": 0.72, "contours": 12,
                         "motion": True}}
        line = format_for_log(r)
        self.assertIn("Brightness: 0.72", line)
        self.assertIn("Motion: yes", line)


class TestFormatForLogExecutionSummary(unittest.TestCase):
    """Test format_for_log with execution summary dicts containing step_results."""

    def test_execution_summary_uses_step_results(self):
        """Execution summary with step_results should format raw outputs."""
        r = {
            "status": "SUCCESS",
            "faults": 0,
            "tool_chain": ["system_metrics"],
            "step_results": [
                {"percent": 87, "plugged_in": False, "seconds_left": 16740}
            ],
        }
        line = format_for_log(r)
        self.assertIn("Battery: 87%", line)
        self.assertIn("discharging", line)
        self.assertIn("4h39m remaining", line)
        # Should NOT just show "Status: SUCCESS"
        self.assertNotIn("Status: SUCCESS", line)

    def test_multiple_step_results(self):
        """Multiple step outputs are merged into one line."""
        r = {
            "status": "SUCCESS",
            "faults": 0,
            "step_results": [
                {"cpu_percent": 42.0},
                {"percent": 91, "plugged_in": True, "seconds_left": 0},
            ],
        }
        line = format_for_log(r)
        self.assertIn("CPU: 42%", line)
        self.assertIn("Battery: 91%", line)
        self.assertIn("charging", line)

    def test_empty_step_results_falls_through(self):
        """Empty step_results should fall back to normal formatting."""
        r = {"result": "fallback text", "step_results": []}
        line = format_for_log(r)
        self.assertIn("fallback text", line)

    def test_humanize_result_uses_step_results(self):
        """_humanize_result should also prefer step_results."""
        r = {
            "status": "SUCCESS",
            "result": "some stringified output",
            "step_results": [
                {"percent": 50, "plugged_in": True, "seconds_left": 3600}
            ],
        }
        text = _humanize_result(r)
        self.assertIn("Battery: 50%", text)
        self.assertIn("1h0m remaining", text)


class TestFormatSingleResult(unittest.TestCase):
    """Test _format_single_result helper."""

    def test_cpu(self):
        parts = _format_single_result({"cpu_percent": 55.0})
        self.assertIn("CPU: 55%", parts)

    def test_raw_battery(self):
        parts = _format_single_result(
            {"percent": 87, "plugged_in": False, "seconds_left": 16740})
        joined = " | ".join(parts)
        self.assertIn("Battery: 87%", joined)
        self.assertIn("4h39m remaining", joined)

    def test_empty_dict(self):
        parts = _format_single_result({})
        self.assertEqual(parts, [])

    def test_unwraps_ok_envelope(self):
        """_ok() wrapper: {"success": True, "result": {...}}."""
        parts = _format_single_result({
            "success": True,
            "result": {"percent": 85, "plugged_in": False,
                       "seconds_left": 15840},
            "error": ""
        })
        joined = " | ".join(parts)
        self.assertIn("Battery: 85%", joined)
        self.assertIn("discharging", joined)

    def test_system_metrics_composite(self):
        """system_metrics returns {cpu: {...}, battery: {...}, ...}."""
        parts = _format_single_result({
            "cpu": {"percent": 23.0},
            "ram": {"percent": 82.0},
            "battery": {"percent": 87, "plugged_in": False,
                        "seconds_left": 16740},
            "disk": {"percent": 45},
        })
        joined = " | ".join(parts)
        self.assertIn("CPU: 23", joined)
        self.assertIn("RAM: 82", joined)
        self.assertIn("Battery: 87%", joined)
        self.assertIn("Disk: 45%", joined)


class TestTryParseDictString(unittest.TestCase):
    """Test _try_parse_dict_string for stringified dicts."""

    def test_python_repr(self):
        s = "{'percent': 85, 'plugged_in': False, 'seconds_left': 15840}"
        d = _try_parse_dict_string(s)
        self.assertIsNotNone(d)
        self.assertEqual(d["percent"], 85)
        self.assertFalse(d["plugged_in"])

    def test_json_string(self):
        s = '{"percent": 85, "plugged_in": false}'
        d = _try_parse_dict_string(s)
        self.assertIsNotNone(d)
        self.assertEqual(d["percent"], 85)

    def test_non_dict_returns_none(self):
        self.assertIsNone(_try_parse_dict_string("hello world"))
        self.assertIsNone(_try_parse_dict_string("[1, 2, 3]"))

    def test_format_for_log_parses_stringified_battery(self):
        """format_for_log should parse str(dict) in result field."""
        r = {
            "status": "SUCCESS",
            "faults": 0,
            "result": "{'percent': 85, 'plugged_in': False, "
                      "'seconds_left': 15840}",
        }
        line = format_for_log(r)
        self.assertIn("Battery: 85%", line)
        self.assertIn("discharging", line)
        self.assertNotIn("'percent'", line)


class TestSessionLogFile(unittest.TestCase):
    """Test session log file creation."""

    def test_session_log_consistent(self):
        scheduler = TaskScheduler(_mock_execute)
        f1 = scheduler.session_log_file("monitor camera")
        f2 = scheduler.session_log_file("different objective")
        # Should return the same file once set
        self.assertEqual(f1, f2)

    def test_session_log_path(self):
        scheduler = TaskScheduler(_mock_execute)
        path = scheduler.session_log_file("test task")
        self.assertTrue(path.startswith("logs/"))
        self.assertTrue(path.endswith(".log"))

    def test_session_log_written(self):
        scheduler = TaskScheduler(_mock_execute)
        scheduler.run_for(0.03, "session log test", interval=0.5)
        self.assertIsNotNone(scheduler._session_log)
        self.assertTrue(os.path.exists(scheduler._session_log))
        with open(scheduler._session_log) as f:
            content = f.read()
        # Should have at least one timestamped line
        self.assertRegex(content, r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')


class TestCleanObjectiveForPlanner(unittest.TestCase):
    """Test stripping write/save/log phrases from objectives."""

    def test_save_to_file(self):
        obj = "check system health and save to status.txt"
        cleaned = _clean_objective_for_planner(obj)
        self.assertNotIn("save", cleaned.lower())
        self.assertNotIn("status.txt", cleaned)
        self.assertIn("check system health", cleaned)

    def test_log_to_file(self):
        obj = "check CPU and battery, log to monitoring_log.txt"
        cleaned = _clean_objective_for_planner(obj)
        self.assertNotIn("monitoring_log", cleaned)
        self.assertIn("check CPU and battery", cleaned)

    def test_write_results_to(self):
        obj = "scan environment, write results to scan_output.txt"
        cleaned = _clean_objective_for_planner(obj)
        self.assertNotIn("write", cleaned.lower())
        self.assertNotIn("scan_output", cleaned)
        self.assertIn("scan environment", cleaned)

    def test_append_to(self):
        obj = "get system metrics, append to log.txt"
        cleaned = _clean_objective_for_planner(obj)
        self.assertNotIn("append", cleaned.lower())
        self.assertIn("get system metrics", cleaned)

    def test_log_keyword_replaced(self):
        obj = "log CPU and battery status"
        cleaned = _clean_objective_for_planner(obj)
        self.assertIn("get CPU", cleaned)

    def test_no_change_when_clean(self):
        obj = "check system health and get CPU percent"
        cleaned = _clean_objective_for_planner(obj)
        self.assertEqual(cleaned, obj)

    def test_save_all_results(self):
        obj = "scan environment and save all results"
        cleaned = _clean_objective_for_planner(obj)
        self.assertNotIn("save", cleaned.lower())
        self.assertIn("scan environment", cleaned)

    def test_complex_sentence(self):
        obj = ("monitor the camera, detect any people "
               "every 30 seconds, save to monitoring_log.txt")
        # Note: time specs are stripped separately by parse_schedule,
        # this function only strips write/save/log phrases
        cleaned = _clean_objective_for_planner(obj)
        self.assertNotIn("monitoring_log", cleaned)
        self.assertIn("monitor the camera", cleaned)


class TestSchedulerUsesCleanObjective(unittest.TestCase):
    """Verify the scheduler passes cleaned objective to execute_fn."""

    def test_execute_receives_clean_objective(self):
        received = []

        def _capture(obj):
            received.append(obj)
            return {"status": "SUCCESS", "faults": 0, "tool_chain": []}

        scheduler = TaskScheduler(_capture)
        scheduler.run_for(0.03, "check health and save to log.txt",
                          interval=0.5)
        self.assertGreater(len(received), 0)
        # The execute function should NOT receive "save to log.txt"
        for obj in received:
            self.assertNotIn("save to log.txt", obj)
            self.assertIn("check health", obj)


if __name__ == "__main__":
    unittest.main()
