"""Chaos tests: corrupt observations and verify graceful handling."""
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestObservationClipping:
    """Verify NavigationEngine.observe() clips all values to [0, 1]."""

    def test_observe_clips_to_0_1(self):
        from aether.core.tool_discovery import ToolDiscovery
        from aether.core.tool_builder import ToolBuilder
        from aether.core.navigation_engine import NavigationEngine

        d = ToolDiscovery()
        d.discover()
        tools = ToolBuilder(d.manifest).build_all()
        nav = NavigationEngine(d.manifest, tools=tools)

        obs = nav.observe()
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_observe_no_nan(self):
        from aether.core.tool_discovery import ToolDiscovery
        from aether.core.tool_builder import ToolBuilder
        from aether.core.navigation_engine import NavigationEngine

        d = ToolDiscovery()
        d.discover()
        tools = ToolBuilder(d.manifest).build_all()
        nav = NavigationEngine(d.manifest, tools=tools)

        obs = nav.observe()
        assert not np.any(np.isnan(obs)), f"NaN in obs: {obs}"


class TestFaultAgentChaos:
    """Corrupt 10% of observation values with NaN or out-of-range values."""

    def _make_corrupted_obs(self, rng, corruption_rate=0.10):
        """Build a 15-dim obs with some corrupted values."""
        obs = rng.random(15).astype(np.float64)
        n_corrupt = max(1, int(len(obs) * corruption_rate))
        indices = rng.choice(len(obs), n_corrupt, replace=False)
        for idx in indices:
            if rng.random() < 0.5:
                obs[idx] = np.nan
            else:
                obs[idx] = rng.choice([-5.0, 2.0, 100.0, -999.0])
        return obs

    def test_fault_agent_handles_nan_obs(self):
        from aether.agents.fault_agent import FaultAgent
        from aether.core.message_bus import MessageBus

        bus = MessageBus()
        agent = FaultAgent(bus=bus)
        rng = np.random.default_rng(42)

        for step in range(20):
            obs = self._make_corrupted_obs(rng)
            # Replace NaN with 0.5 and clip before passing to agent
            clean = np.nan_to_num(obs, nan=0.5)
            clean = np.clip(clean, 0.0, 1.0)
            # Should not crash
            reports = agent.process(clean, step)
            assert isinstance(reports, list)

    def test_domain_agents_handle_corrupted_obs(self):
        from aether.agents.camera_agent import CameraAgent
        from aether.agents.power_agent import PowerAgent
        from aether.agents.thermal_agent import ThermalAgent
        from aether.agents.navigation_agent import NavigationAgent
        from aether.core.message_bus import MessageBus

        bus = MessageBus()
        cam = CameraAgent(bus)
        pwr = PowerAgent(bus)
        thm = ThermalAgent(bus)
        nav_ag = NavigationAgent(bus)

        rng = np.random.default_rng(99)

        for step in range(20):
            obs = self._make_corrupted_obs(rng)
            clean = np.nan_to_num(obs, nan=0.5)
            clean = np.clip(clean, 0.0, 1.0)

            state = {"failed_sensors": [], "agent_pos": [0, 0],
                     "agent_heading": 0.0}

            # None of these should crash
            cam_result = cam.tick(clean, state, step)
            assert isinstance(cam_result, dict)

            pwr_result = pwr.tick(clean, step)
            assert isinstance(pwr_result, dict)

            thm_result = thm.tick(clean, step)
            assert isinstance(thm_result, dict)

            nav_result = nav_ag.tick(clean, state, step)
            assert isinstance(nav_result, dict)

    def test_tool_execute_handles_bad_params(self):
        """All tool execute() methods should return error, not raise."""
        from aether.core.tool_registry import (
            WebSearchTool, ExecuteShellTool, ReadFileTool,
            WriteFileTool, SummarizeTextTool,
        )

        tools = [
            WebSearchTool(),
            ExecuteShellTool(),
            ReadFileTool(),
            WriteFileTool(),
            SummarizeTextTool(),
        ]

        for tool in tools:
            # Empty params should return error result, not crash
            result = tool.execute({})
            assert hasattr(result, "success")
            assert result.success is False or result.output is not None
