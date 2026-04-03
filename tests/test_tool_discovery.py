"""Tests for ToolDiscovery: platform detection, manifest structure."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aether.core.tool_discovery import ToolDiscovery


@pytest.fixture(scope="module")
def manifest():
    d = ToolDiscovery()
    d.discover()
    return d.manifest


class TestToolDiscovery:
    def test_platform_detected(self, manifest):
        assert "platform" in manifest
        assert manifest["platform"] != ""

    def test_camera_found_or_missing(self, manifest):
        hw = manifest.get("hardware", {})
        cam = hw.get("camera", {})
        # camera key must exist with an 'available' field
        assert "available" in cam

    def test_manifest_has_capability_score(self, manifest):
        assert "capability_score" in manifest
        score = manifest["capability_score"]
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100

    def test_available_tools_non_empty(self, manifest):
        tools = manifest.get("available_tools", [])
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_missing_capabilities_is_list(self, manifest):
        missing = manifest.get("missing_capabilities", [])
        assert isinstance(missing, list)
        # On a dev machine at least some capabilities will be missing
        # (e.g., GPIO on macOS), so the list should be non-empty
        assert len(missing) > 0
