"""Tests for ToolBuilder: tool construction from manifest."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aether.core.tool_discovery import ToolDiscovery
from aether.core.tool_builder import ToolBuilder


@pytest.fixture(scope="module")
def setup():
    d = ToolDiscovery()
    d.discover()
    manifest = d.manifest
    builder = ToolBuilder(manifest)
    tools = builder.build_all()
    return manifest, tools


class TestToolBuilder:
    def test_build_all_returns_dict(self, setup):
        _, tools = setup
        assert isinstance(tools, dict)
        assert len(tools) > 0

    def test_system_tool_has_cpu_method(self, setup):
        _, tools = setup
        sys_tool = tools.get("system")
        assert sys_tool is not None
        assert hasattr(sys_tool, "get_cpu_percent")

    def test_camera_tool_exists_if_available(self, setup):
        manifest, tools = setup
        cam_available = manifest.get("hardware", {}).get(
            "camera", {}).get("available", False)
        if cam_available:
            assert "camera" in tools
        # If camera not available, just pass — nothing to assert

    def test_all_tools_callable(self, setup):
        _, tools = setup
        for name, tool in tools.items():
            # Every tool should have at least one public callable method
            methods = [m for m in dir(tool)
                       if not m.startswith("_") and callable(getattr(tool, m))]
            assert len(methods) > 0, f"Tool '{name}' has no public methods"

    def test_system_tool_no_exception(self, setup):
        _, tools = setup
        sys_tool = tools.get("system")
        if sys_tool is None:
            pytest.skip("No system tool built")
        # Should not raise
        result = sys_tool.get_cpu_percent()
        assert isinstance(result, dict)
        assert "success" in result
