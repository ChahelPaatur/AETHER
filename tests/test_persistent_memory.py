"""Tests for aether.core.memory.PersistentMemory."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aether.core.memory import PersistentMemory, _extract_keywords


class TestPersistentMemoryBasics(unittest.TestCase):
    """Test basic record/load/save operations."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._path = os.path.join(self._tmpdir, "test_memory.json")
        self.mem = PersistentMemory(path=self._path)

    def tearDown(self):
        if os.path.exists(self._path):
            os.remove(self._path)
        os.rmdir(self._tmpdir)

    def test_starts_empty(self):
        self.assertEqual(self.mem.count, 0)

    def test_record_and_count(self):
        self.mem.record("check CPU", "SUCCESS", ["system_metrics"], faults=0)
        self.assertEqual(self.mem.count, 1)

    def test_record_persists_to_disk(self):
        self.mem.record("test persist", "SUCCESS", ["web_search"])
        self.assertTrue(os.path.exists(self._path))
        with open(self._path) as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["objective"], "test persist")

    def test_loads_on_startup(self):
        self.mem.record("obj1", "SUCCESS", ["tool_a"])
        self.mem.record("obj2", "DEGRADED", ["tool_b"], faults=1)

        # Create new instance — should load from disk
        mem2 = PersistentMemory(path=self._path)
        self.assertEqual(mem2.count, 2)

    def test_recent(self):
        for i in range(10):
            self.mem.record(f"obj_{i}", "SUCCESS", ["t"])
        recent = self.mem.recent(3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[-1]["objective"], "obj_9")

    def test_max_entries_eviction(self):
        for i in range(120):
            self.mem.record(f"obj_{i}", "SUCCESS", ["t"])
        self.assertLessEqual(self.mem.count, 100)
        # Oldest should be gone
        objs = [e["objective"] for e in self.mem.recent(100)]
        self.assertNotIn("obj_0", objs)
        self.assertIn("obj_119", objs)

    def test_entry_has_required_fields(self):
        self.mem.record("test", "SUCCESS", ["tool_a"], faults=2, duration=1.5)
        entry = self.mem.recent(1)[0]
        self.assertIn("timestamp", entry)
        self.assertIn("datetime", entry)
        self.assertIn("session_id", entry)
        self.assertIn("objective", entry)
        self.assertIn("outcome", entry)
        self.assertIn("tool_chain", entry)
        self.assertIn("faults", entry)
        self.assertIn("duration_s", entry)
        self.assertIn("platform", entry)
        self.assertEqual(entry["faults"], 2)
        self.assertEqual(entry["duration_s"], 1.5)


class TestPersistentMemorySearch(unittest.TestCase):
    """Test keyword-based search."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._path = os.path.join(self._tmpdir, "test_memory.json")
        self.mem = PersistentMemory(path=self._path)
        self.mem.record("check system health and CPU", "SUCCESS",
                        ["system_metrics", "get_cpu_percent"])
        self.mem.record("scan environment for obstacles", "SUCCESS",
                        ["visual_scan", "describe_scene"])
        self.mem.record("check battery status", "DEGRADED",
                        ["get_battery"], faults=1)
        self.mem.record("monitor CPU temperature", "SUCCESS",
                        ["get_cpu_temp"])

    def tearDown(self):
        if os.path.exists(self._path):
            os.remove(self._path)
        os.rmdir(self._tmpdir)

    def test_search_finds_similar(self):
        results = self.mem.search("check CPU percent")
        self.assertGreater(len(results), 0)
        objectives = [r["objective"] for r in results]
        self.assertTrue(any("CPU" in o for o in objectives))

    def test_search_no_results(self):
        results = self.mem.search("deploy kubernetes cluster")
        self.assertEqual(len(results), 0)

    def test_search_respects_limit(self):
        results = self.mem.search("check", limit=2)
        self.assertLessEqual(len(results), 2)


class TestPlanningHints(unittest.TestCase):
    """Test planning hint generation."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._path = os.path.join(self._tmpdir, "test_memory.json")
        self.mem = PersistentMemory(path=self._path)

    def tearDown(self):
        if os.path.exists(self._path):
            os.remove(self._path)
        os.rmdir(self._tmpdir)

    def test_hints_with_success(self):
        self.mem.record("check battery status", "SUCCESS",
                        ["get_battery"])
        hints = self.mem.planning_hints("check battery level")
        self.assertIsNotNone(hints)
        self.assertIn("get_battery", hints)
        self.assertIn("succeeded", hints)

    def test_hints_with_failure(self):
        self.mem.record("check battery status", "DEGRADED",
                        ["system_metrics", "get_battery"], faults=1)
        hints = self.mem.planning_hints("check battery percent")
        self.assertIsNotNone(hints)
        self.assertIn("failures", hints.lower())

    def test_no_hints_for_unrelated(self):
        self.mem.record("scan environment", "SUCCESS", ["visual_scan"])
        hints = self.mem.planning_hints("deploy firmware update")
        self.assertIsNone(hints)


class TestFormatSummary(unittest.TestCase):
    """Test human-readable summary formatting."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._path = os.path.join(self._tmpdir, "test_memory.json")
        self.mem = PersistentMemory(path=self._path)

    def tearDown(self):
        if os.path.exists(self._path):
            os.remove(self._path)
        os.rmdir(self._tmpdir)

    def test_empty_summary(self):
        summary = self.mem.format_summary()
        self.assertIn("No past experience", summary)

    def test_summary_with_data(self):
        self.mem.record("check CPU", "SUCCESS", ["get_cpu_percent"],
                        duration=1.2)
        self.mem.record("check battery", "DEGRADED",
                        ["get_battery"], faults=1, duration=2.0)
        summary = self.mem.format_summary()
        self.assertIn("Total entries:", summary)
        self.assertIn("Success rate:", summary)
        self.assertIn("Top tools:", summary)
        self.assertIn("get_cpu_percent", summary)
        self.assertIn("get_battery", summary)


class TestExtractKeywords(unittest.TestCase):
    """Test keyword extraction."""

    def test_basic(self):
        kws = _extract_keywords("check system health and CPU status")
        self.assertIn("system", kws)
        self.assertIn("health", kws)
        self.assertIn("cpu", kws)
        self.assertIn("status", kws)
        # Stop words should be excluded
        self.assertNotIn("and", kws)
        self.assertNotIn("the", kws)

    def test_short_words_excluded(self):
        kws = _extract_keywords("do it on a go")
        self.assertEqual(kws, [])

    def test_empty(self):
        self.assertEqual(_extract_keywords(""), [])


class TestPrintStartup(unittest.TestCase):
    """Test startup message."""

    def test_startup_with_entries(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "test.json")
        mem = PersistentMemory(path=path)
        mem.record("test", "SUCCESS", ["t"])
        # Just verify it doesn't crash
        mem.print_startup()
        os.remove(path)
        os.rmdir(tmpdir)

    def test_startup_empty(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "empty.json")
        mem = PersistentMemory(path=path)
        mem.print_startup()
        os.rmdir(tmpdir)


if __name__ == "__main__":
    unittest.main()
