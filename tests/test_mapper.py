"""Tests for EnvironmentMapper: grid building, flow computation, save."""
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aether.core.mapper import EnvironmentMapper, GRID_SIZE


@pytest.fixture
def mock_camera():
    """Camera that returns incrementally different frames."""
    cam = MagicMock()
    _counter = [0]

    def _capture():
        _counter[0] += 1
        frame = np.full((64, 64, 3), _counter[0] * 30, dtype=np.uint8)
        # Add a moving "object" in a different position each frame
        y = min(60, _counter[0] * 8)
        frame[y:y + 4, 10:20, :] = 255
        return frame

    cam.capture_frame = _capture
    return cam


@pytest.fixture
def noop_motor():
    return MagicMock()


class TestEnvironmentMapper:
    def test_run_returns_grid(self, mock_camera, noop_motor, tmp_path):
        mapper = EnvironmentMapper(
            mock_camera, noop_motor, configs_dir=str(tmp_path))
        result = mapper.run()
        assert "grid" in result
        assert result["grid"].shape == (GRID_SIZE, GRID_SIZE)

    def test_saves_npy_file(self, mock_camera, noop_motor, tmp_path):
        mapper = EnvironmentMapper(
            mock_camera, noop_motor, configs_dir=str(tmp_path))
        result = mapper.run()
        assert os.path.exists(result["npy_path"])
        loaded = np.load(result["npy_path"])
        assert loaded.shape == (GRID_SIZE, GRID_SIZE)

    def test_saves_png_file(self, mock_camera, noop_motor, tmp_path):
        mapper = EnvironmentMapper(
            mock_camera, noop_motor, configs_dir=str(tmp_path))
        result = mapper.run()
        assert os.path.exists(result["png_path"])

    def test_waypoints_count(self, mock_camera, noop_motor, tmp_path):
        mapper = EnvironmentMapper(
            mock_camera, noop_motor, configs_dir=str(tmp_path))
        result = mapper.run()
        # 1 initial + 7 pattern steps = 8 waypoints
        assert result["waypoints"] == 8

    def test_motor_called_for_each_action(self, mock_camera, noop_motor, tmp_path):
        mapper = EnvironmentMapper(
            mock_camera, noop_motor, configs_dir=str(tmp_path))
        mapper.run()
        assert noop_motor.call_count == 7  # 7 actions in pattern

    def test_grid_has_occupied_cells(self, mock_camera, noop_motor, tmp_path):
        mapper = EnvironmentMapper(
            mock_camera, noop_motor, configs_dir=str(tmp_path))
        result = mapper.run()
        # With different frames, some cells should be marked occupied
        assert result["occupied_cells"] > 0

    def test_resize_utility(self):
        arr = np.ones((50, 50), dtype=np.float32)
        resized = EnvironmentMapper._resize(arr, 100, 100)
        assert resized.shape == (100, 100)
        assert np.all(resized == 1.0)

    def test_handles_camera_failure(self, noop_motor, tmp_path):
        cam = MagicMock()
        cam.capture_frame.return_value = None
        mapper = EnvironmentMapper(cam, noop_motor, configs_dir=str(tmp_path))
        # Should not crash
        result = mapper.run()
        assert result["waypoints"] == 0
        assert result["occupied_cells"] == 0
