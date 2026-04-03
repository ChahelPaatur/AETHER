"""Unit tests for CalibrationWizard: instantiation, run, profile fields."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aether.core.calibration import CalibrationWizard


@pytest.fixture
def minimal_manifest():
    return {
        "hardware": {
            "camera": {"available": True, "type": "cv2"},
            "gpio": {"available": False},
            "mavlink": {"available": False},
        },
        "software": {"ultralytics": False},
        "network": {"internet": False},
        "motor_controllers": [],
    }


class TestCalibrationUnit:
    def test_instantiates(self, minimal_manifest, tmp_path):
        wizard = CalibrationWizard(
            minimal_manifest,
            configs_dir=str(tmp_path),
            auto_mode=True,
        )
        assert wizard is not None
        assert wizard.auto_mode is True

    def test_generates_profile_on_run(self, minimal_manifest, tmp_path):
        wizard = CalibrationWizard(
            minimal_manifest,
            configs_dir=str(tmp_path),
            auto_mode=True,
        )
        profile = wizard.run()
        assert isinstance(profile, dict)
        assert profile.get("calibrated") is True

    def test_profile_has_required_fields(self, minimal_manifest, tmp_path):
        wizard = CalibrationWizard(
            minimal_manifest,
            configs_dir=str(tmp_path),
            auto_mode=True,
        )
        profile = wizard.run()
        required = [
            "robot_name", "robot_type", "calibrated",
            "calibration_date", "capabilities", "camera_verified",
        ]
        for field in required:
            assert field in profile, f"Missing field: {field}"
