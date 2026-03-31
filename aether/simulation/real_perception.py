"""
RealPerceptionAdapter: bridges real-world sensors (webcam + system metrics)
into AETHER's 15-dimensional observation vector, allowing the full agent
pipeline (FaultAgent, AdaptationAgent, Planner) to run on live hardware.

Observation vector mapping:
  [0]  battery_level       ← psutil battery percent / 100 (or 0.95 if no battery)
  [1]  solar_power         ← 0.3 default (no sensor)
  [2]  bus_voltage          ← 0.9 default (no sensor)
  [3]  imu_x               ← 0.5 default (healthy, no IMU)
  [4]  imu_y               ← 0.5 default
  [5]  imu_z               ← 0.5 default
  [6]  attitude_error       ← 0.5 default
  [7]  temperature_panel    ← CPU percent / 100 (proxy for thermal load)
  [8]  temperature_core     ← RAM percent / 100 (proxy for system load)
  [9]  obstacle_dist_front  ← 1.0 default (clear, no proximity sensor)
  [10] obstacle_dist_left   ← 1.0 default
  [11] obstacle_dist_right  ← 1.0 default
  [12] target_dist          ← 1.0 if motion detected, 0.5 if none
  [13] target_bearing       ← motion centroid X normalized 0-1 from frame center
  [14] mission_progress     ← step_count / max_steps
"""
import time
from typing import Dict, List, Optional

import numpy as np

from .environment import OBS_DIM


class RealPerceptionAdapter:
    """
    Drop-in replacement for SimulationEnvironment that reads from real
    hardware: webcam via cv2.VideoCapture and system metrics via psutil.

    Implements the same interface consumed by PerceptionAgent:
      observe() → np.ndarray (15-dim)
      get_state_dict() → Dict
      reset(scenario) → np.ndarray
    """

    def __init__(self, camera_index: int = 0, max_steps: int = 300):
        self._camera_index = camera_index
        self._max_steps = max_steps
        self._cap = None
        self._prev_gray = None
        self.step_count = 0

        # Motion detection state
        self._motion_detected = False
        self._motion_bearing = 0.5  # center of frame

        # System metrics cache
        self._battery = 0.95
        self._cpu_pct = 0.0
        self._ram_pct = 0.0

        # Compatibility fields (PerceptionAgent reads these)
        self.failed_sensors: List[str] = []
        self.failed_actuators: List[str] = []
        self.injected_noise: np.ndarray = np.zeros(OBS_DIM)
        self.agent_pos: np.ndarray = np.array([50.0, 50.0])
        self.agent_heading: float = 0.0
        self.obstacles: List[Dict] = []
        self.targets: List[Dict] = [
            {"id": "target_0", "pos": np.array([80.0, 80.0]), "type": "realworld"}
        ]

        self._psutil = self._try_import_psutil()
        self._cv2 = self._try_import_cv2()

    @staticmethod
    def _try_import_psutil():
        try:
            import psutil
            return psutil
        except ImportError:
            return None

    @staticmethod
    def _try_import_cv2():
        try:
            import cv2
            return cv2
        except ImportError:
            return None

    def reset(self, scenario: Optional[Dict] = None) -> np.ndarray:
        """Initialize sensors. Returns initial observation."""
        self.step_count = 0
        self._prev_gray = None
        self._motion_detected = False
        self._motion_bearing = 0.5
        self.failed_sensors = []
        self.failed_actuators = []
        self.injected_noise = np.zeros(OBS_DIM)

        if scenario:
            self._max_steps = scenario.get("max_steps", self._max_steps)

        # Open camera
        if self._cv2 is not None:
            if self._cap is not None:
                self._cap.release()
            self._cap = self._cv2.VideoCapture(self._camera_index)
            if not self._cap.isOpened():
                print("[RealPerception] WARNING: Camera not available")
                self.failed_sensors.append("camera")
                self._cap = None
        else:
            print("[RealPerception] WARNING: cv2 not installed — camera disabled")
            self.failed_sensors.append("camera")

        if self._psutil is None:
            print("[RealPerception] WARNING: psutil not installed — using default metrics")

        return self.observe()

    def step(self, action: str, params: Optional[Dict] = None):
        """Compatibility with environment step — in realworld mode we just observe."""
        self.step_count += 1
        obs = self.observe()
        done = self.step_count >= self._max_steps
        return obs, 0.0, done, {"step": self.step_count, "success": False}

    def observe(self) -> np.ndarray:
        """Read real sensors and return the 15-dim observation vector."""
        self._read_system_metrics()
        self._read_camera()

        obs = np.array([
            self._battery,           # [0] battery
            0.3,                     # [1] solar (no sensor → default)
            0.9,                     # [2] bus voltage (no sensor → default)
            0.5,                     # [3] imu_x (no IMU → healthy default)
            0.5,                     # [4] imu_y
            0.5,                     # [5] imu_z
            0.5,                     # [6] attitude_error (no IMU → default)
            self._cpu_pct,           # [7] CPU percent as thermal proxy
            self._ram_pct,           # [8] RAM percent as system load proxy
            1.0,                     # [9] obstacle_dist_front (no sensor → clear)
            1.0,                     # [10] obstacle_dist_left
            1.0,                     # [11] obstacle_dist_right
            1.0 if self._motion_detected else 0.5,  # [12] target_dist
            self._motion_bearing,    # [13] target_bearing
            min(self.step_count / max(self._max_steps, 1), 1.0),  # [14] progress
        ], dtype=float)

        # Apply any injected noise (from FaultInjector if attached)
        obs = np.clip(obs + self.injected_noise, 0.0, 1.0)

        # Sensor dropout compatibility
        if "camera" in self.failed_sensors:
            obs[12:14] = 0.5
        if "imu" in self.failed_sensors:
            obs[3:7] = 0.5

        return obs

    def get_state_dict(self) -> Dict:
        """Return structured state dict compatible with PerceptionAgent."""
        obs = self.observe()
        return {
            "agent_pos": self.agent_pos.tolist(),
            "agent_heading": self.agent_heading,
            "step": self.step_count,
            "observation": obs.tolist(),
            "battery": float(obs[0]),
            "temperature_core": float(obs[8]),
            "at_target": False,  # realworld: never auto-complete
            "target_info": {"pos": self.targets[0]["pos"].tolist()} if self.targets else None,
            "obstacles_nearby": [],
            "failed_sensors": list(self.failed_sensors),
            "failed_actuators": list(self.failed_actuators),
            "mission_progress": float(obs[14]),
            "motion_detected": self._motion_detected,
            "motion_bearing": self._motion_bearing,
            "cpu_percent": self._cpu_pct * 100,
            "ram_percent": self._ram_pct * 100,
        }

    def _read_system_metrics(self) -> None:
        """Read battery, CPU, and RAM via psutil."""
        if self._psutil is None:
            return

        # Battery
        batt = self._psutil.sensors_battery()
        if batt is not None:
            self._battery = batt.percent / 100.0
        else:
            self._battery = 0.95  # desktop without battery

        # CPU and RAM
        self._cpu_pct = self._psutil.cpu_percent(interval=None) / 100.0
        self._ram_pct = self._psutil.virtual_memory().percent / 100.0

    def _read_camera(self) -> None:
        """Read webcam frame, detect motion via frame differencing."""
        if self._cap is None or self._cv2 is None:
            self._motion_detected = False
            self._motion_bearing = 0.5
            return

        ret, frame = self._cap.read()
        if not ret:
            self._motion_detected = False
            self._motion_bearing = 0.5
            return

        cv2 = self._cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._motion_detected = False
            self._motion_bearing = 0.5
            return

        # Frame difference → threshold → find contours
        delta = cv2.absdiff(self._prev_gray, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        self._prev_gray = gray

        # Filter small contours (noise)
        min_area = frame.shape[0] * frame.shape[1] * 0.005  # 0.5% of frame
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if large_contours:
            self._motion_detected = True
            # Find centroid of largest contour
            biggest = max(large_contours, key=cv2.contourArea)
            M = cv2.moments(biggest)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                # Normalize X position to 0-1 (left=0, right=1)
                self._motion_bearing = cx / frame.shape[1]
            else:
                self._motion_bearing = 0.5
        else:
            self._motion_detected = False
            self._motion_bearing = 0.5

    def inject_failure(self, fault_config: Dict) -> None:
        """Apply fault to subsystems (compatibility with SimulationEnvironment)."""
        ftype = fault_config.get("type", "")
        target = fault_config.get("target", "")
        if ftype == "sensor":
            if target not in self.failed_sensors:
                self.failed_sensors.append(target)
        elif ftype == "actuator":
            if target not in self.failed_actuators:
                self.failed_actuators.append(target)

    def close(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def render(self) -> str:
        """Text rendering for realworld mode."""
        obs = self.observe()
        motion = "MOTION" if self._motion_detected else "static"
        return (
            f"[RealWorld] Step {self.step_count} | "
            f"Battery: {self._battery*100:.0f}% | "
            f"CPU: {self._cpu_pct*100:.0f}% | RAM: {self._ram_pct*100:.0f}% | "
            f"Camera: {motion} (bearing: {self._motion_bearing:.2f})"
        )
