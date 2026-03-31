"""
CapabilityLoader v3: loads robot configs, builds CapabilityGraph, includes CompatibilityChecker.
"""
import json
import os
from typing import Dict, List, Optional, Set


ABSTRACT_TO_HARDWARE = {
    "move_forward":     {"actuators": ["wheels", "thrusters", "legs"], "sensors": []},
    "move_backward":    {"actuators": ["wheels", "thrusters", "legs"], "sensors": []},
    "turn_left":        {"actuators": ["wheels", "thrusters", "legs"], "sensors": []},
    "turn_right":       {"actuators": ["wheels", "thrusters", "legs"], "sensors": []},
    "stop":             {"actuators": ["wheels", "thrusters", "legs"], "sensors": []},
    "emergency_stop":   {"actuators": ["wheels", "thrusters", "legs"], "sensors": []},
    "follow_target":    {"actuators": ["wheels", "thrusters"],         "sensors": ["camera"]},
    "avoid_obstacle":   {"actuators": ["wheels", "thrusters"],         "sensors": ["ultrasonic", "lidar", "camera"]},
    "scan_environment": {"actuators": [],                              "sensors": ["camera", "lidar", "ultrasonic"]},
    "report_state":     {"actuators": [],                              "sensors": []},
    "safe_mode":        {"actuators": ["wheels", "thrusters"],         "sensors": []},
    "recharge":         {"actuators": [],                              "sensors": []},
    "plan_path":        {"actuators": [],                              "sensors": ["camera", "lidar", "ultrasonic"]},
    "recalibrate_imu":  {"actuators": [],                              "sensors": ["imu"]},
    "reduce_speed":     {"actuators": ["wheels", "thrusters"],         "sensors": []},
    "diagnostic_scan":  {"actuators": [],                              "sensors": ["camera", "imu"]},
}


class CapabilityGraph:
    """Maps abstract actions to available hardware actions given a robot config."""

    def __init__(self, config: Dict):
        self.robot_name = config.get("name", "unknown")
        self.actuators: Set[str] = set(config.get("actuators", []))
        self.sensors: Set[str] = set(config.get("sensors", []))
        self.hardware_actions: Set[str] = set(config.get("actions", []))
        self.limits: Dict = config.get("limits", {})
        self.fault_thresholds: Dict = config.get("fault_thresholds", {})
        self._available: Dict[str, Dict] = {}
        self._build()

    def _build(self) -> None:
        for action, reqs in ABSTRACT_TO_HARDWARE.items():
            act_ok = (not reqs["actuators"]
                      or bool(self.actuators & set(reqs["actuators"])))
            sens_ok = (not reqs["sensors"]
                       or bool(self.sensors & set(reqs["sensors"])))
            if act_ok and sens_ok:
                self._available[action] = {
                    "actuators": list(self.actuators & set(reqs["actuators"])),
                    "sensors": list(self.sensors & set(reqs["sensors"])),
                }

    def can_do(self, action: str) -> bool:
        return action in self._available

    def available_actions(self) -> List[str]:
        return list(self._available.keys())

    def degraded_actions(self, failed_sensors: List[str] = None,
                          failed_actuators: List[str] = None) -> List[str]:
        """Return actions still available given current failures."""
        fs = set(failed_sensors or [])
        fa = set(failed_actuators or [])
        active_sensors = self.sensors - fs
        active_actuators = self.actuators - fa
        available = []
        for action, reqs in ABSTRACT_TO_HARDWARE.items():
            if action not in self._available:
                continue
            act_ok = (not reqs["actuators"]
                      or bool(active_actuators & set(reqs["actuators"])))
            sens_ok = (not reqs["sensors"]
                       or bool(active_sensors & set(reqs["sensors"])))
            if act_ok and sens_ok:
                available.append(action)
        return available or ["stop"]

    def get_limit(self, key: str, default=None):
        return self.limits.get(key, default)

    def get_threshold(self, key: str, default=None):
        return self.fault_thresholds.get(key, default)


class CompatibilityChecker:
    """Validates a task plan against loaded hardware capabilities."""

    def check(self, task: Dict, cap_graph: CapabilityGraph) -> Dict:
        required = set(task.get("subtasks", []))
        available = set(cap_graph.available_actions())
        missing = required - available
        return {
            "compatible": len(missing) == 0,
            "missing": list(missing),
            "available": list(available),
        }


class CapabilityLoader:
    """Loads robot configs from JSON files."""

    def load_from_file(self, path: str) -> CapabilityGraph:
        with open(path) as f:
            config = json.load(f)
        return CapabilityGraph(config)

    def load_from_dict(self, config: Dict) -> CapabilityGraph:
        return CapabilityGraph(config)

    def load_from_directory(self, directory: str) -> Dict[str, CapabilityGraph]:
        graphs = {}
        for fname in os.listdir(directory):
            if fname.endswith(".json"):
                graphs[fname[:-5]] = self.load_from_file(os.path.join(directory, fname))
        return graphs
