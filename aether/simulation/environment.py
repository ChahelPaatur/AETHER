"""
SimulationEnvironment v3: continuous 2D physics-lite world with 15-dim observation vector,
reward function, ASCII rendering, and partial observability.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np


OBS_LABELS = [
    "battery_level", "solar_power", "bus_voltage",       # EPS (0-2)
    "imu_x", "imu_y", "imu_z", "attitude_error",         # ADCS (3-6)
    "temperature_panel", "temperature_core",             # TCS (7-8)
    "obstacle_dist_front", "obstacle_dist_left",         # Proximity (9-11)
    "obstacle_dist_right",
    "target_dist", "target_bearing",                     # Target (12-13)
    "mission_progress",                                  # Meta (14)
]
OBS_DIM = 15


class SimulationEnvironment:
    """
    2D continuous environment with spacecraft-inspired subsystem telemetry.
    Observation: 15-dim normalized vector (all values in [0, 1]).
    """

    def __init__(self, width: float = 100.0, height: float = 100.0,
                 seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)
        self._sensor_range = 20.0
        self._fov = 120.0  # degrees, field-of-view cone

        # State
        self.agent_pos: np.ndarray = np.array([10.0, 10.0])
        self.agent_heading: float = 0.0
        self.obstacles: List[Dict] = []
        self.targets: List[Dict] = []
        self.step_count: int = 0

        # Subsystem state
        self._battery: float = 0.95
        self._solar: float = 0.3
        self._bus_voltage: float = 0.90
        self._imu: np.ndarray = np.zeros(3)
        self._attitude_error: float = 0.0
        self._temp_panel: float = 0.28
        self._temp_core: float = 0.42

        # Fault injection state (applied externally by FaultInjector)
        self.injected_noise: np.ndarray = np.zeros(OBS_DIM)
        self.failed_sensors: List[str] = []
        self.failed_actuators: List[str] = []

        self._initial_dist: float = 1.0
        self._episode_reward: float = 0.0
        self._max_steps: int = 500

    # ------------------------------------------------------------------ #
    #  Environment Interface                                               #
    # ------------------------------------------------------------------ #

    def reset(self, scenario: Optional[Dict] = None) -> np.ndarray:
        """Reset environment to scenario or defaults. Returns initial observation."""
        self.step_count = 0
        self._episode_reward = 0.0
        self.injected_noise = np.zeros(OBS_DIM)
        self.failed_sensors = []
        self.failed_actuators = []
        self._battery = 0.95
        self._solar = 0.3
        self._bus_voltage = 0.90
        self._imu = np.zeros(3)
        self._attitude_error = 0.0
        self._temp_panel = 0.28
        self._temp_core = 0.42

        if scenario:
            self._max_steps = scenario.get("max_steps", 500)
            env_cfg = scenario.get("environment", {})
            self.agent_pos = np.array(scenario.get("agent_start", [10.0, 10.0]), dtype=float)
            self.agent_heading = scenario.get("agent_heading", 0.0)
            self.obstacles = [{"pos": np.array(o["pos"], dtype=float),
                               "radius": o.get("radius", 2.0),
                               "dynamic": o.get("dynamic", False),
                               "vel": np.array(o.get("vel", [0, 0]), dtype=float)}
                              for o in scenario.get("obstacles", [])]
            task = scenario.get("task", {})
            target_pos = task.get("target", [80.0, 80.0])
            self.targets = [{"id": "target_0",
                             "pos": np.array(target_pos, dtype=float),
                             "type": "waypoint"}]
        else:
            self.agent_pos = np.array([10.0, 10.0])
            self.agent_heading = 45.0
            self.obstacles = []
            self.targets = [{"id": "target_0",
                             "pos": np.array([80.0, 80.0]),
                             "type": "waypoint"}]

        self._initial_dist = self._dist_to_target()
        return self.observe()

    def step(self, action: str, params: Optional[Dict] = None
             ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action; return (obs, reward, done, info)."""
        params = params or {}
        self.step_count += 1
        self._update_subsystems()
        self._update_dynamic_obstacles()

        reward = -0.5  # per-step cost
        collision = False
        success = False

        speed = min(params.get("speed", 1.0), 1.0)

        if "wheels" in self.failed_actuators or "thrusters" in self.failed_actuators:
            reward -= 0.2
        else:
            collision = self._apply_action(action, speed)
            if collision:
                reward -= 5.0

        if self._at_target():
            reward += 10.0
            success = True
        else:
            reward += 1.0 * self._proximity_bonus()

        if self.failed_sensors or self.failed_actuators:
            reward -= 2.0

        self._episode_reward += reward
        obs = self.observe()
        done = success or self.step_count >= self._max_steps
        return obs, reward, done, {"collision": collision, "success": success,
                                   "step": self.step_count}

    def observe(self) -> np.ndarray:
        """Return the 15-dim normalized observation vector."""
        # IMU: map [-0.15, 0.15] drift range → [0, 1], center (no drift) = 0.5
        imu_norm = np.clip((self._imu + 0.15) / 0.30, 0.0, 1.0)
        obs = np.array([
            self._battery,
            self._solar,
            self._bus_voltage,
            float(imu_norm[0]),
            float(imu_norm[1]),
            float(imu_norm[2]),
            self._attitude_error,
            self._temp_panel,
            self._temp_core,
            self._obstacle_dist(0.0),    # front
            self._obstacle_dist(90.0),   # left
            self._obstacle_dist(-90.0),  # right
            self._norm_target_dist(),
            self._norm_target_bearing(),
            self._mission_progress(),
        ], dtype=float)

        # Apply injected noise/faults
        obs = np.clip(obs + self.injected_noise, 0.0, 1.0)

        # Sensor dropout
        if "imu" in self.failed_sensors:
            obs[3:7] = 0.5  # zeroed IMU
        if "camera" in self.failed_sensors:
            obs[12:14] = 0.5  # no target info
        if "ultrasonic" in self.failed_sensors:
            obs[9:12] = 1.0  # no obstacle detection = assume clear (unsafe)
        if "temperature" in self.failed_sensors:
            obs[7:9] = 0.5

        return obs

    def inject_failure(self, fault_config: Dict) -> None:
        """Apply fault to environment subsystems."""
        ftype = fault_config.get("type", "")
        target = fault_config.get("target", "")
        amount = float(fault_config.get("amount", 0.1))

        if ftype == "sensor":
            if target not in self.failed_sensors:
                self.failed_sensors.append(target)
        elif ftype == "actuator":
            if target not in self.failed_actuators:
                self.failed_actuators.append(target)
        elif ftype == "noise":
            idx = fault_config.get("obs_index", 0)
            self.injected_noise[idx] += amount
        elif ftype == "battery_drain":
            self._battery = max(0.0, self._battery - amount)
        elif ftype == "thermal_spike":
            self._temp_core = min(1.0, self._temp_core + amount)
        elif ftype == "imu_drift":
            self._imu += amount
            self._imu = np.clip(self._imu, -1.0, 1.0)
            self._attitude_error = min(1.0, self._attitude_error + amount)
        elif ftype == "bus_voltage_drop":
            self._bus_voltage = max(0.0, self._bus_voltage - amount)

    def clear_failure(self, fault_type: str, target: str) -> None:
        if fault_type == "sensor" and target in self.failed_sensors:
            self.failed_sensors.remove(target)
        elif fault_type == "actuator" and target in self.failed_actuators:
            self.failed_actuators.remove(target)
        elif fault_type == "imu_drift":
            self._imu = np.zeros(3)
            self._attitude_error = 0.0

    def render(self, mode: str = "ascii") -> str:
        """ASCII art render of the environment."""
        W, H = 40, 20
        scale_x = W / self.width
        scale_y = H / self.height
        grid = [["." for _ in range(W)] for _ in range(H)]

        for obs in self.obstacles:
            cx = int(obs["pos"][0] * scale_x)
            cy = int(obs["pos"][1] * scale_y)
            if 0 <= cx < W and 0 <= H - cy - 1 < H:
                grid[H - cy - 1][cx] = "#"

        for tgt in self.targets:
            tx = int(tgt["pos"][0] * scale_x)
            ty = int(tgt["pos"][1] * scale_y)
            if 0 <= tx < W and 0 <= H - ty - 1 < H:
                grid[H - ty - 1][tx] = "T"

        ax = int(self.agent_pos[0] * scale_x)
        ay = int(self.agent_pos[1] * scale_y)
        if 0 <= ax < W and 0 <= H - ay - 1 < H:
            grid[H - ay - 1][ax] = "A"

        lines = [" " + "-" * W]
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append(" " + "-" * W)
        battery_bar = int(self._battery * 10)
        lines.append(f" Battery: [{'#'*battery_bar}{'.'*(10-battery_bar)}] "
                     f"Step: {self.step_count} | Pos: ({self.agent_pos[0]:.1f},{self.agent_pos[1]:.1f})")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _apply_action(self, action: str, speed: float) -> bool:
        rad = math.radians(self.agent_heading)
        if action in ("move_forward",):
            dx, dy = math.cos(rad) * speed, math.sin(rad) * speed
            new_pos = self.agent_pos + np.array([dx, dy])
            if self._check_collision(new_pos):
                return True
            self.agent_pos = np.clip(new_pos, 0, [self.width, self.height])
        elif action == "move_backward":
            dx, dy = -math.cos(rad) * speed * 0.5, -math.sin(rad) * speed * 0.5
            new_pos = self.agent_pos + np.array([dx, dy])
            if not self._check_collision(new_pos):
                self.agent_pos = np.clip(new_pos, 0, [self.width, self.height])
        elif action in ("turn_left",):
            self.agent_heading = (self.agent_heading + 30.0) % 360
        elif action in ("turn_right",):
            self.agent_heading = (self.agent_heading - 30.0) % 360
        elif action == "follow_target":
            tgt = self._nearest_target()
            if tgt is not None:
                direction = tgt["pos"] - self.agent_pos
                dist = np.linalg.norm(direction)
                if dist > 0.5:
                    step = min(speed, dist)
                    new_pos = self.agent_pos + (direction / dist) * step
                    if not self._check_collision(new_pos):
                        self.agent_pos = new_pos
                        self.agent_heading = math.degrees(math.atan2(direction[1], direction[0]))
                    else:
                        # Tangential avoidance
                        perp = np.array([-direction[1], direction[0]]) / dist
                        new_pos2 = self.agent_pos + perp * speed
                        if not self._check_collision(new_pos2):
                            self.agent_pos = np.clip(new_pos2, 0, [self.width, self.height])
                        else:
                            return True
        return False

    def _update_subsystems(self) -> None:
        self._battery = max(0.0, self._battery - 0.001)
        self._solar = 0.3 + 0.05 * math.sin(self.step_count * 0.1)
        self._battery = min(1.0, self._battery + self._solar * 0.0005)
        self._imu += self.rng.normal(0, 0.001, 3)
        self._imu = np.clip(self._imu, -1.0, 1.0)
        self._temp_core = min(1.0, self._temp_core + 0.0002)
        self._temp_panel = max(0.0, self._temp_panel + self.rng.normal(0, 0.001))

    def _update_dynamic_obstacles(self) -> None:
        for obs in self.obstacles:
            if obs["dynamic"]:
                obs["pos"] += obs["vel"] * 0.1
                # Bounce off walls
                for i in range(2):
                    if obs["pos"][i] < 0 or obs["pos"][i] > (self.width if i == 0 else self.height):
                        obs["vel"][i] *= -1

    def _check_collision(self, pos: np.ndarray) -> bool:
        for obs in self.obstacles:
            if np.linalg.norm(pos - obs["pos"]) < obs["radius"] + 1.0:
                return True
        return False

    def _obstacle_dist(self, rel_angle: float) -> float:
        """Normalized distance to nearest obstacle in direction (0=blocked, 1=clear)."""
        abs_angle = math.radians(self.agent_heading + rel_angle)
        direction = np.array([math.cos(abs_angle), math.sin(abs_angle)])
        min_dist = self._sensor_range
        for step in range(1, int(self._sensor_range) + 1):
            probe = self.agent_pos + direction * step
            if self._check_collision(probe):
                min_dist = float(step)
                break
        return min(1.0, min_dist / self._sensor_range)

    def _dist_to_target(self) -> float:
        tgt = self._nearest_target()
        if tgt is None:
            return 0.0
        return float(np.linalg.norm(self.agent_pos - tgt["pos"]))

    def _norm_target_dist(self) -> float:
        dist = self._dist_to_target()
        max_dist = math.sqrt(self.width**2 + self.height**2)
        return 1.0 - min(1.0, dist / max_dist)

    def _norm_target_bearing(self) -> float:
        tgt = self._nearest_target()
        if tgt is None:
            return 0.5
        dx, dy = tgt["pos"] - self.agent_pos
        bearing = math.degrees(math.atan2(dy, dx))
        diff = ((bearing - self.agent_heading) + 180) % 360 - 180
        return (diff + 180) / 360.0

    def _mission_progress(self) -> float:
        if self._initial_dist <= 0:
            return 1.0
        current = self._dist_to_target()
        return min(1.0, max(0.0, 1.0 - current / self._initial_dist))

    def _proximity_bonus(self) -> float:
        current = self._dist_to_target()
        if self._initial_dist <= 0:
            return 0.0
        return max(0.0, 1.0 - current / self._initial_dist)

    def _at_target(self, threshold: float = 3.0) -> bool:
        return self._dist_to_target() < threshold

    def _nearest_target(self) -> Optional[Dict]:
        if not self.targets:
            return None
        return min(self.targets, key=lambda t: np.linalg.norm(self.agent_pos - t["pos"]))

    def get_state_dict(self) -> Dict:
        """Return human-readable state dict (used by agents for decisions)."""
        obs = self.observe()
        return {
            "agent_pos": self.agent_pos.tolist(),
            "agent_heading": self.agent_heading,
            "step": self.step_count,
            "observation": obs.tolist(),
            "battery": float(obs[0]),
            "temperature_core": float(obs[8]),
            "at_target": self._at_target(),
            "target_info": {"pos": self._nearest_target()["pos"].tolist()} if self._nearest_target() else None,
            "obstacles_nearby": [o["pos"].tolist() for o in self.obstacles
                                 if np.linalg.norm(self.agent_pos - o["pos"]) < 10.0],
            "failed_sensors": list(self.failed_sensors),
            "failed_actuators": list(self.failed_actuators),
            "mission_progress": float(obs[14]),
        }
