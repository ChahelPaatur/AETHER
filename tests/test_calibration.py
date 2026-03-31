"""Tests for CalibrationWizard — core/calibration.py."""
import json
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aether.core.calibration import (
    CalibrationWizard,
    load_calibration,
    _expected_direction,
    GROUND_VEHICLE,
    AERIAL_VEHICLE,
    ROBOTIC_ARM,
    AQUATIC_VEHICLE,
    CUSTOM,
)


@pytest.fixture
def tmp_configs(tmp_path):
    """Create a temporary configs directory."""
    configs = tmp_path / "configs"
    configs.mkdir()
    return str(configs)


@pytest.fixture
def gpio_manifest():
    """Manifest with GPIO hardware detected."""
    return {
        "hardware": {
            "camera": {"available": True, "type": "cv2"},
            "gpio": {"available": True, "type": "RPi.GPIO",
                     "pins": [17, 18, 22, 23]},
            "mavlink": {"available": False},
        },
        "software": {},
        "network": {},
    }


@pytest.fixture
def mavlink_manifest():
    """Manifest with MAVLink hardware."""
    return {
        "hardware": {
            "camera": {"available": True},
            "gpio": {"available": False},
            "mavlink": {"available": True, "motor_count": 4},
        },
    }


@pytest.fixture
def empty_manifest():
    """Manifest with no actuators."""
    return {
        "hardware": {
            "camera": {"available": True},
            "gpio": {"available": False},
            "mavlink": {"available": False},
        },
    }


class TestCalibrationWizardInit:
    def test_detects_gpio_channels(self, gpio_manifest, tmp_configs):
        wiz = CalibrationWizard(gpio_manifest, configs_dir=tmp_configs)
        wiz._detect_channels()
        assert len(wiz.motor_channels) == 4

    def test_detects_mavlink_channels(self, mavlink_manifest, tmp_configs):
        wiz = CalibrationWizard(mavlink_manifest, configs_dir=tmp_configs)
        wiz._detect_channels()
        assert len(wiz.motor_channels) == 4

    def test_no_channels_without_hardware(self, empty_manifest, tmp_configs):
        wiz = CalibrationWizard(empty_manifest, configs_dir=tmp_configs)
        wiz._detect_channels()
        assert len(wiz.motor_channels) == 0


class TestNeedsCalibration:
    def test_needs_calibration_new_hardware(self, gpio_manifest, tmp_configs):
        wiz = CalibrationWizard(gpio_manifest, configs_dir=tmp_configs)
        assert wiz.needs_calibration() is True

    def test_no_calibration_without_actuators(self, empty_manifest, tmp_configs):
        wiz = CalibrationWizard(empty_manifest, configs_dir=tmp_configs)
        assert wiz.needs_calibration() is False

    def test_force_calibration(self, empty_manifest, tmp_configs):
        wiz = CalibrationWizard(empty_manifest, configs_dir=tmp_configs)
        assert wiz.needs_calibration(force=True) is True

    def test_skip_if_profile_exists(self, gpio_manifest, tmp_configs):
        # Create a fake calibration profile
        profile = {"calibrated": True, "robot_name": "test"}
        path = os.path.join(tmp_configs, "calibrated_test_20260330.json")
        with open(path, "w") as f:
            json.dump(profile, f)

        wiz = CalibrationWizard(gpio_manifest, configs_dir=tmp_configs)
        assert wiz.needs_calibration() is False


class TestAutoModeFullRun:
    def test_auto_ground_vehicle(self, gpio_manifest, tmp_configs):
        """Auto mode should complete without user input."""
        wiz = CalibrationWizard(gpio_manifest, configs_dir=tmp_configs,
                                auto_mode=True)
        wiz._detect_channels()
        wiz.robot_type = GROUND_VEHICLE
        wiz.robot_name = "test_rover"
        wiz.profile["drive_type"] = "differential"
        wiz._phase2_motor_mapping()

        assert len(wiz.motor_map) == 4
        for ch, mapping in wiz.motor_map.items():
            assert "function" in mapping

    def test_auto_arm(self, gpio_manifest, tmp_configs):
        wiz = CalibrationWizard(gpio_manifest, configs_dir=tmp_configs,
                                auto_mode=True)
        wiz._detect_channels()
        wiz.robot_type = ROBOTIC_ARM
        wiz.profile["joint_ranges"] = {
            f"joint_{i+1}": {"min": -90, "max": 90}
            for i in range(4)
        }
        wiz._phase2_motor_mapping()

        assert len(wiz.motor_map) == 4
        assert len(wiz.components) == 4

    def test_auto_aerial(self, mavlink_manifest, tmp_configs):
        wiz = CalibrationWizard(mavlink_manifest, configs_dir=tmp_configs,
                                auto_mode=True)
        wiz._detect_channels()
        wiz.robot_type = AERIAL_VEHICLE
        wiz._phase2_motor_mapping()

        assert len(wiz.motor_map) == 4


class TestSafetyLimits:
    def test_auto_generic_safety(self, gpio_manifest, tmp_configs):
        wiz = CalibrationWizard(gpio_manifest, configs_dir=tmp_configs,
                                auto_mode=True)
        wiz.robot_type = GROUND_VEHICLE
        wiz.motor_map = {
            "channel_1": {"function": "left_wheel", "direction": "forward"},
            "channel_2": {"function": "right_wheel", "direction": "forward"},
        }
        wiz._phase4_safety_limits()

        assert "left_wheel" in wiz.safety_limits
        assert wiz.safety_limits["left_wheel"]["max_speed"] == 0.7

    def test_auto_aerial_safety(self, mavlink_manifest, tmp_configs):
        wiz = CalibrationWizard(mavlink_manifest, configs_dir=tmp_configs,
                                auto_mode=True)
        wiz.robot_type = AERIAL_VEHICLE
        wiz._phase4_safety_limits()

        assert "flight" in wiz.safety_limits
        assert wiz.safety_limits["flight"]["failsafe"] == "return_to_launch"

    def test_auto_arm_safety(self, gpio_manifest, tmp_configs):
        wiz = CalibrationWizard(gpio_manifest, configs_dir=tmp_configs,
                                auto_mode=True)
        wiz.robot_type = ROBOTIC_ARM
        wiz.motor_map = {
            "channel_1": {"function": "base_rotation",
                          "min_angle": -90, "max_angle": 90},
        }
        wiz._phase4_safety_limits()

        assert "base_rotation" in wiz.safety_limits
        assert wiz.safety_limits["base_rotation"]["hard_stop"] is True


class TestSaveAndLoad:
    def test_save_profile(self, gpio_manifest, tmp_configs):
        wiz = CalibrationWizard(gpio_manifest, configs_dir=tmp_configs,
                                auto_mode=True)
        wiz.robot_type = GROUND_VEHICLE
        wiz.robot_name = "test_rover"
        wiz.motor_map = {
            "channel_1": {"function": "left_wheel", "direction": "forward"},
        }
        wiz.safety_limits = {"left_wheel": {"max_speed": 0.7}}

        profile = wiz._save_profile()

        assert profile["calibrated"] is True
        assert profile["robot_name"] == "test_rover"
        assert profile["robot_type"] == GROUND_VEHICLE
        assert "left_wheel" in profile["components"]

        # Verify file was written
        files = os.listdir(tmp_configs)
        cal_files = [f for f in files if f.startswith("calibrated_")]
        assert len(cal_files) == 1

    def test_load_calibration(self, tmp_configs):
        profile = {
            "calibrated": True,
            "robot_name": "loader_test",
            "robot_type": GROUND_VEHICLE,
        }
        path = os.path.join(tmp_configs,
                            "calibrated_loader_test_20260330_120000.json")
        with open(path, "w") as f:
            json.dump(profile, f)

        loaded = load_calibration(tmp_configs)
        assert loaded is not None
        assert loaded["robot_name"] == "loader_test"

    def test_load_returns_none_if_empty(self, tmp_configs):
        loaded = load_calibration(tmp_configs)
        assert loaded is None


class TestActionGeneration:
    def test_ground_vehicle_actions(self):
        profile = {
            "robot_type": GROUND_VEHICLE,
            "components": {"left_wheel": {"channel": 1}},
            "safety_limits": {"left_wheel": {"max_speed": 0.7}},
        }
        actions = CalibrationWizard.generate_actions(profile)
        names = [a["name"] for a in actions]

        assert "move_forward" in names
        assert "turn_left" in names
        assert "navigate_to_color" in names
        assert "stop" in names

    def test_arm_actions(self):
        profile = {
            "robot_type": ROBOTIC_ARM,
            "components": {
                "base_rotation": {"channel": 1, "type": "servo",
                                  "min": -90, "max": 90},
                "shoulder": {"channel": 2, "type": "servo",
                             "min": 0, "max": 180},
            },
            "safety_limits": {
                "base_rotation": {"max_speed_dps": 45},
                "shoulder": {"max_speed_dps": 30},
            },
        }
        actions = CalibrationWizard.generate_actions(profile)
        names = [a["name"] for a in actions]

        assert "move_joint_base_rotation" in names
        assert "move_joint_shoulder" in names
        assert "grip" in names
        assert "release" in names
        assert "point_to" in names
        assert "home_position" in names
        assert "stop" in names

    def test_aerial_actions(self):
        profile = {
            "robot_type": AERIAL_VEHICLE,
            "components": {},
            "safety_limits": {
                "flight": {"max_altitude_m": 10, "failsafe": "return_to_launch"},
            },
        }
        actions = CalibrationWizard.generate_actions(profile)
        names = [a["name"] for a in actions]

        assert "takeoff" in names
        assert "hover" in names
        assert "land" in names
        assert "return_to_launch" in names
        assert "stop" in names

    def test_custom_only_has_stop(self):
        profile = {"robot_type": CUSTOM, "components": {}, "safety_limits": {}}
        actions = CalibrationWizard.generate_actions(profile)
        names = [a["name"] for a in actions]
        assert names == ["stop"]


class TestRuntimeChecks:
    def test_check_movement_no_frames(self):
        result = CalibrationWizard.check_movement(
            "move_forward", {}, None, None)
        assert result["matches"] is True
        assert "no frames" in result["warning"]

    def test_expected_direction_mapping(self):
        assert _expected_direction("move_forward") == "forward"
        assert _expected_direction("turn_left") == "leftward"
        assert _expected_direction("turn_right") == "rightward"
        assert _expected_direction("stop") is None


class TestFullAutoRun:
    def test_camera_only_run(self, empty_manifest, tmp_configs):
        """No actuators should produce a camera-only profile."""
        wiz = CalibrationWizard(empty_manifest, configs_dir=tmp_configs,
                                auto_mode=True)
        profile = wiz.run()

        assert profile["robot_type"] == CUSTOM
        assert profile["robot_name"] == "camera_only"
        assert "visual_scan" in profile["capabilities"]
        assert profile["calibrated"] is True

    def test_auto_ground_full_run(self, gpio_manifest, tmp_configs):
        """Full auto run for ground vehicle."""
        wiz = CalibrationWizard(gpio_manifest, configs_dir=tmp_configs,
                                auto_mode=True)
        profile = wiz.run()

        assert profile["calibrated"] is True
        assert profile["robot_type"] == GROUND_VEHICLE
        assert len(profile["motor_map"]) == 4
        assert len(profile["capabilities"]) > 0
        assert "stop" in profile["capabilities"]

    def test_auto_aerial_full_run(self, mavlink_manifest, tmp_configs):
        """Full auto run for aerial vehicle."""
        wiz = CalibrationWizard(mavlink_manifest, configs_dir=tmp_configs,
                                auto_mode=True)
        profile = wiz.run()

        assert profile["calibrated"] is True
        assert profile["robot_type"] == AERIAL_VEHICLE
