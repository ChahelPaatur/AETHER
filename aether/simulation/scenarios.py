"""
ScenarioGenerator v3: structured scenario definitions for AETHER experiments.
Supports deterministic, random, fault-heavy, and multi-task scenarios.
"""
import copy
from typing import Dict, List, Optional
import numpy as np


class ScenarioGenerator:
    """Generates scenario dicts consumed by SimulationEnvironment.reset()."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def random_scenario(self, difficulty: str = "medium") -> Dict:
        cfg = {"easy": (3, 0, 0.01), "medium": (7, 2, 0.02), "hard": (12, 4, 0.04)}
        n_obs, n_dyn, fault_prob = cfg.get(difficulty, cfg["medium"])

        obstacles = []
        for _ in range(n_obs):
            obstacles.append({
                "pos": [float(self.rng.uniform(20, 80)), float(self.rng.uniform(20, 80))],
                "radius": float(self.rng.uniform(1.5, 3.5)),
                "dynamic": False,
            })
        for _ in range(n_dyn):
            vel_angle = float(self.rng.uniform(0, 360))
            obstacles.append({
                "pos": [float(self.rng.uniform(20, 80)), float(self.rng.uniform(20, 80))],
                "radius": 2.0,
                "dynamic": True,
                "vel": [float(np.cos(np.radians(vel_angle))), float(np.sin(np.radians(vel_angle)))],
            })

        target = [float(self.rng.uniform(60, 95)), float(self.rng.uniform(60, 95))]
        faults = []
        fault_types = ["sensor_noise_imu", "battery_drain", "thermal_spike", "imu_drift"]
        for ft in fault_types:
            if self.rng.random() < fault_prob * 10:
                faults.append({
                    "type": ft,
                    "start_step": int(self.rng.integers(5, 80)),
                    "subsystem": ft.split("_")[0],
                })

        return {
            "name": f"random_{difficulty}",
            "environment": {"size": [100, 100], "n_obstacles": n_obs},
            "agent_start": [10.0, 10.0],
            "agent_heading": 45.0,
            "obstacles": obstacles,
            "task": {"type": "navigate", "target": target, "constraints": ["avoid_obstacles"]},
            "faults": faults,
            "robot_config": "rover_v1",
            "max_steps": 300,
        }

    def deterministic_scenario(self, scenario_id: int) -> Dict:
        scenarios = {
            0: {
                "name": "simple_navigate",
                "agent_start": [10.0, 10.0],
                "agent_heading": 45.0,
                "obstacles": [],
                "task": {"type": "navigate", "target": [80.0, 80.0], "constraints": []},
                "faults": [],
                "max_steps": 300,
            },
            1: {
                "name": "obstacle_field",
                "agent_start": [10.0, 10.0],
                "agent_heading": 45.0,
                "obstacles": [
                    {"pos": [30.0, 30.0], "radius": 3.0},
                    {"pos": [50.0, 50.0], "radius": 4.0},
                    {"pos": [65.0, 35.0], "radius": 2.5},
                    {"pos": [40.0, 65.0], "radius": 3.0},
                ],
                "task": {"type": "navigate", "target": [80.0, 80.0], "constraints": ["avoid_obstacles"]},
                "faults": [],
                "max_steps": 300,
            },
            2: {
                "name": "sensor_failure_imu",
                "agent_start": [10.0, 10.0],
                "agent_heading": 45.0,
                "obstacles": [{"pos": [40.0, 40.0], "radius": 3.0}],
                "task": {"type": "navigate", "target": [80.0, 80.0], "constraints": ["avoid_obstacles"]},
                "faults": [{"type": "imu_drift", "start_step": 15, "subsystem": "imu", "severity": 0.7}],
                "max_steps": 500,
            },
            3: {
                "name": "battery_critical",
                "agent_start": [10.0, 10.0],
                "agent_heading": 45.0,
                "obstacles": [],
                "task": {"type": "navigate", "target": [80.0, 80.0], "constraints": []},
                "faults": [{"type": "battery_drain", "start_step": 10, "subsystem": "battery", "severity": 0.9}],
                "max_steps": 400,
            },
            4: {
                "name": "compound_faults",
                "agent_start": [10.0, 10.0],
                "agent_heading": 45.0,
                "obstacles": [{"pos": [35.0, 35.0], "radius": 3.0}],
                "task": {"type": "navigate", "target": [80.0, 80.0], "constraints": ["avoid_obstacles"]},
                "faults": [
                    {"type": "imu_drift", "start_step": 20, "subsystem": "imu", "severity": 0.6},
                    {"type": "thermal_spike", "start_step": 40, "subsystem": "thermal", "severity": 0.7},
                ],
                "max_steps": 500,
            },
        }
        return scenarios.get(scenario_id % len(scenarios), scenarios[0])

    def fault_heavy_scenario(self) -> Dict:
        """Maximum fault pressure scenario for stress-testing FDIR."""
        return {
            "name": "fault_heavy",
            "agent_start": [10.0, 10.0],
            "agent_heading": 45.0,
            "obstacles": [
                {"pos": [25.0, 25.0], "radius": 3.0},
                {"pos": [50.0, 50.0], "radius": 4.0},
                {"pos": [70.0, 30.0], "radius": 2.0},
            ],
            "task": {"type": "navigate", "target": [80.0, 80.0], "constraints": ["avoid_obstacles"]},
            "faults": [
                {"type": "imu_drift", "start_step": 10, "subsystem": "imu", "severity": 0.8},
                {"type": "battery_drain", "start_step": 20, "subsystem": "battery", "severity": 0.7},
                {"type": "thermal_spike", "start_step": 35, "subsystem": "thermal", "severity": 0.75},
                {"type": "sensor_noise_imu", "start_step": 50, "subsystem": "imu", "severity": 0.6},
            ],
            "max_steps": 600,
        }

    def multi_task_scenario(self) -> Dict:
        """Multiple sequential waypoints."""
        return {
            "name": "multi_task",
            "agent_start": [5.0, 5.0],
            "agent_heading": 0.0,
            "obstacles": [{"pos": [30.0, 30.0], "radius": 2.5}],
            "task": {"type": "navigate", "target": [50.0, 50.0], "constraints": ["avoid_obstacles"]},
            "faults": [{"type": "intermittent", "start_step": 25, "subsystem": "imu"}],
            "max_steps": 400,
        }


NAMED_SCENARIOS = {
    "simple": ScenarioGenerator(42).deterministic_scenario(0),
    "obstacles": ScenarioGenerator(42).deterministic_scenario(1),
    "imu_fault": ScenarioGenerator(42).deterministic_scenario(2),
    "battery": ScenarioGenerator(42).deterministic_scenario(3),
    "compound": ScenarioGenerator(42).deterministic_scenario(4),
    "fault_heavy": ScenarioGenerator(42).fault_heavy_scenario(),
    "multi_task": ScenarioGenerator(42).multi_task_scenario(),
}


def get_scenario(name: str) -> Optional[Dict]:
    """Return a deep copy of the named scenario to prevent shared-state mutation."""
    base = NAMED_SCENARIOS.get(name)
    return copy.deepcopy(base) if base is not None else None


def list_scenarios() -> List[str]:
    return list(NAMED_SCENARIOS.keys())
