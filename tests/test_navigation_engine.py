"""Tests for NavigationEngine: level detection, observation vector."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aether.core.tool_discovery import ToolDiscovery
from aether.core.tool_builder import ToolBuilder
from aether.core.navigation_engine import NavigationEngine


@pytest.fixture(scope="module")
def nav():
    d = ToolDiscovery()
    d.discover()
    manifest = d.manifest
    tools = ToolBuilder(manifest).build_all()
    return NavigationEngine(manifest, tools=tools)


class TestNavigationEngine:
    def test_level_in_range(self, nav):
        assert nav.level in (1, 2, 3)

    def test_available_actions_non_empty(self, nav):
        actions = nav.available_actions()
        assert isinstance(actions, list)
        assert len(actions) > 0

    def test_observe_returns_length_15(self, nav):
        obs = nav.observe()
        assert isinstance(obs, np.ndarray)
        assert len(obs) == 15

    def test_observe_values_clipped_0_1(self, nav):
        obs = nav.observe()
        assert np.all(obs >= 0.0), f"Values below 0: {obs[obs < 0]}"
        assert np.all(obs <= 1.0), f"Values above 1: {obs[obs > 1]}"

    def test_repeated_observe_no_crash(self, nav):
        for _ in range(10):
            obs = nav.observe()
            assert len(obs) == 15
