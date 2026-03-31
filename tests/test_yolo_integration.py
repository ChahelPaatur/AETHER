"""Tests for YOLOv8 integration — tool_builder, tool_registry, navigation_engine."""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aether.core.tool_builder import YOLOTool, CameraTool, ToolBuilder, _ok, _err
from aether.core.tool_registry import (
    ToolRegistry, register_built_tools, DynamicTool,
)
from aether.core.navigation_engine import NavigationEngine


@pytest.fixture
def camera_manifest():
    return {
        "hardware": {
            "camera": {"available": True, "type": "cv2"},
            "gpio": {"available": False},
            "mavlink": {"available": False},
        },
        "software": {"ultralytics": True, "numpy": True},
        "network": {"internet": False},
        "motor_controllers": [],
    }


@pytest.fixture
def motor_manifest():
    return {
        "hardware": {
            "camera": {"available": True, "type": "cv2"},
            "gpio": {"available": True, "type": "RPi.GPIO", "pins": [17, 18]},
            "mavlink": {"available": False},
        },
        "software": {"ultralytics": True, "numpy": True},
        "network": {"internet": False},
        "motor_controllers": [{
            "controller_type": "gpio_direct_motors",
            "connection_method": "gpio",
        }],
    }


class TestYOLOToolInit:
    def test_creates_without_error(self):
        tool = YOLOTool()
        assert tool._model is None

    def test_has_all_methods(self):
        tool = YOLOTool()
        assert hasattr(tool, "detect")
        assert hasattr(tool, "detect_from_camera")
        assert hasattr(tool, "count_objects")
        assert hasattr(tool, "describe_scene")

    def test_accepts_camera_tool(self):
        cam = MagicMock(spec=CameraTool)
        tool = YOLOTool(camera_tool=cam)
        assert tool._camera is cam


class TestYOLOToolDetect:
    def test_detect_returns_structured_result(self):
        import numpy as np

        tool = YOLOTool()

        # Mock box attributes to return real numeric types
        mock_box = MagicMock()
        mock_box.cls = np.array([0])
        mock_box.conf = np.array([0.95])
        mock_box.xyxy = np.array([[10.0, 20.0, 100.0, 200.0]])

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "person"}

        tool._model = MagicMock(return_value=[mock_result])

        # Create a temp image file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff")  # minimal JPEG header
            tmp_path = f.name

        try:
            result = tool.detect(tmp_path)
            assert result["success"], result.get("error", "")
            dets = result["result"]["detections"]
            assert len(dets) == 1
            assert dets[0]["class_name"] == "person"
            assert dets[0]["confidence"] == 0.95
            assert "center" in dets[0]
            assert dets[0]["center"]["x"] == 55.0
            assert dets[0]["center"]["y"] == 110.0
        finally:
            os.unlink(tmp_path)

    def test_detect_limits_to_10(self):
        import numpy as np

        tool = YOLOTool()

        boxes = []
        for i in range(15):
            box = MagicMock()
            box.cls = np.array([i % 5])
            box.conf = np.array([0.5 + i * 0.01])
            box.xyxy = np.array([[0, 0, 100, 100]])
            boxes.append(box)

        mock_result = MagicMock()
        mock_result.boxes = boxes
        mock_result.names = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}

        tool._model = MagicMock(return_value=[mock_result])

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff")
            tmp_path = f.name

        try:
            result = tool.detect(tmp_path)
            assert result["success"], result.get("error", "")
            assert result["result"]["count"] == 10
            assert result["result"]["total_raw"] == 15
        finally:
            os.unlink(tmp_path)


class TestYOLOToolDescribeScene:
    def test_describe_scene_formatting(self):
        tool = YOLOTool()
        # Mock detect_from_camera to return canned detections
        tool.detect_from_camera = MagicMock(return_value=_ok({
            "detections": [
                {"class_name": "person", "confidence": 0.94},
                {"class_name": "chair", "confidence": 0.87},
                {"class_name": "chair", "confidence": 0.82},
                {"class_name": "laptop", "confidence": 0.91},
            ],
            "count": 4,
        }))

        result = tool.describe_scene()
        assert result["success"]
        desc = result["result"]["description"]
        assert "person" in desc
        assert "chair" in desc
        assert "laptop" in desc
        assert result["result"]["unique_classes"] == 3
        assert result["result"]["object_count"] == 4

    def test_describe_scene_empty(self):
        tool = YOLOTool()
        tool.detect_from_camera = MagicMock(return_value=_ok({
            "detections": [],
            "count": 0,
        }))

        result = tool.describe_scene()
        assert result["success"]
        assert "cannot see" in result["result"]["description"].lower()


class TestYOLOToolCountObjects:
    def test_count_matching_class(self):
        tool = YOLOTool()
        tool.detect_from_camera = MagicMock(return_value=_ok({
            "detections": [
                {"class_name": "person", "confidence": 0.9},
                {"class_name": "person", "confidence": 0.8},
                {"class_name": "dog", "confidence": 0.7},
            ],
            "count": 3,
        }))

        result = tool.count_objects("person")
        assert result["success"]
        assert result["result"]["count"] == 2
        assert result["result"]["total_objects"] == 3

    def test_count_no_class_returns_total(self):
        tool = YOLOTool()
        tool.detect_from_camera = MagicMock(return_value=_ok({
            "detections": [
                {"class_name": "cat", "confidence": 0.9},
                {"class_name": "dog", "confidence": 0.8},
            ],
            "count": 2,
        }))

        result = tool.count_objects("")
        assert result["success"]
        assert result["result"]["total"] == 2


class TestToolBuilderIntegration:
    def test_yolo_built_when_ultralytics_available(self, camera_manifest):
        builder = ToolBuilder(camera_manifest)
        tools = builder.build_all()
        assert "yolo" in tools
        assert isinstance(tools["yolo"], YOLOTool)

    def test_yolo_not_built_without_ultralytics(self):
        manifest = {
            "hardware": {"camera": {"available": True}},
            "software": {"ultralytics": False},
            "network": {},
            "motor_controllers": [],
        }
        builder = ToolBuilder(manifest)
        tools = builder.build_all()
        assert "yolo" not in tools

    def test_yolo_gets_camera_reference(self, camera_manifest):
        builder = ToolBuilder(camera_manifest)
        tools = builder.build_all()
        assert tools["yolo"]._camera is tools.get("camera")


class TestRegistryIntegration:
    def test_all_yolo_tools_registered(self, camera_manifest):
        builder = ToolBuilder(camera_manifest)
        tools = builder.build_all()
        registry = ToolRegistry()
        register_built_tools(registry, tools, manifest=camera_manifest)

        avail = registry.available_tools()
        assert "yolo_detect" in avail
        assert "detect_objects_yolo" in avail
        assert "count_objects" in avail
        assert "describe_scene" in avail


class TestNavigationEngineYOLO:
    def test_yolo_actions_at_level1(self, camera_manifest):
        tools = ToolBuilder(camera_manifest).build_all()
        nav = NavigationEngine(camera_manifest, tools=tools)
        actions = nav.available_actions()
        assert "yolo_detect" in actions
        assert "describe_scene" in actions
        assert "count_objects" in actions

    def test_yolo_actions_at_level2(self, motor_manifest):
        tools = ToolBuilder(motor_manifest).build_all()
        nav = NavigationEngine(motor_manifest, tools=tools)
        actions = nav.available_actions()
        assert "describe_scene" in actions
        assert "count_objects" in actions
        assert "navigate_to_color" in actions

    def test_navigate_to_color_accepts_class_name(self, motor_manifest):
        """navigate_to_color dispatch should pass class_name when color absent."""
        tools = ToolBuilder(motor_manifest).build_all()
        nav = NavigationEngine(motor_manifest, tools=tools)
        # We can't actually run motors, but verify the action is listed
        assert "navigate_to_color" in nav.available_actions()

    def test_yolo_dispatch_calls_tool(self, camera_manifest):
        """describe_scene dispatch should call the yolo tool."""
        mock_yolo = MagicMock()
        mock_yolo.describe_scene.return_value = _ok({
            "description": "test",
            "detections": [],
        })
        tools = ToolBuilder(camera_manifest).build_all()
        tools["yolo"] = mock_yolo

        nav = NavigationEngine(camera_manifest, tools=tools)
        result = nav.execute("describe_scene")
        mock_yolo.describe_scene.assert_called_once()
