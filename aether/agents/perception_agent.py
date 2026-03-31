"""
PerceptionAgent: reads from SimulationEnvironment, returns structured state with confidence.
Implements partial observability (sensor range, FOV, noise injection via FaultInjector).
"""
from typing import Dict, Optional
import numpy as np
import math

from ..core.message_bus import MessageBus
from ..faults.fault_injector import FaultInjector


class PerceptionAgent:
    """
    Observes the environment and produces a structured state dict
    with confidence scores per subsystem.
    """

    name = "perception_agent"
    SENSOR_RANGE = 20.0
    FOV = 120.0  # degrees

    def __init__(self, environment, bus: MessageBus,
                 fault_injector: Optional[FaultInjector] = None):
        self.env = environment
        self.bus = bus
        self.fault_injector = fault_injector
        self._step = 0

    def observe(self, step: int) -> Dict:
        """
        Read and process environment observation.
        Returns structured dict with raw observation vector + derived state + confidence.
        """
        self._step = step
        raw_obs = self.env.observe()

        # Apply fault injection to observation
        if self.fault_injector:
            corrupted_obs = self.fault_injector.tick(raw_obs, step)
        else:
            corrupted_obs = raw_obs

        state_dict = self.env.get_state_dict()
        confidence = self._compute_confidence(corrupted_obs, state_dict)

        structured = {
            "observation": corrupted_obs,
            "position": state_dict["agent_pos"],
            "heading": state_dict["agent_heading"],
            "obstacles": self._format_obstacles(state_dict, corrupted_obs),
            "target": self._format_target(state_dict, corrupted_obs),
            "subsystem_health": self._extract_subsystem_health(corrupted_obs),
            "noise_level": float(np.std(corrupted_obs - raw_obs)),
            "observation_confidence": confidence,
            "failed_sensors": state_dict["failed_sensors"],
            "failed_actuators": state_dict["failed_actuators"],
            "at_target": state_dict["at_target"],
            "mission_progress": state_dict["mission_progress"],
            "battery": float(corrupted_obs[0]),
            "step": step,
        }

        # Broadcast state update to all agents
        self.bus.publish("STATE_UPDATE", self.name, {
            "observation": corrupted_obs.tolist(),
            "step": step,
            **{k: v for k, v in structured.items() if k != "observation"},
        })

        return structured

    def _compute_confidence(self, obs: np.ndarray, state: Dict) -> float:
        """Estimate observation confidence based on sensor health."""
        failed = state.get("failed_sensors", [])
        noise = float(np.var(obs))
        base = 1.0 - len(failed) * 0.15 - noise * 0.3
        return float(np.clip(base, 0.1, 1.0))

    def _extract_subsystem_health(self, obs: np.ndarray) -> Dict[str, float]:
        """Map observation indices to subsystem health scores."""
        return {
            "battery": float(obs[0]),
            "solar": float(obs[1]),
            "bus_voltage": float(obs[2]),
            "imu": float(1.0 - obs[6]),  # attitude_error inverted → health
            "temperature_panel": float(1.0 - obs[7]),
            "temperature_core": float(1.0 - obs[8]),
            "obstacle_clearance_front": float(obs[9]),
            "obstacle_clearance_left": float(obs[10]),
            "obstacle_clearance_right": float(obs[11]),
        }

    def _format_obstacles(self, state: Dict, obs: np.ndarray) -> list:
        obstacles = state.get("obstacles_nearby", [])
        formatted = []
        for pos in obstacles:
            dist = math.sqrt((state["agent_pos"][0] - pos[0]) ** 2 +
                             (state["agent_pos"][1] - pos[1]) ** 2)
            formatted.append((*pos, dist))
        return formatted

    def _format_target(self, state: Dict, obs: np.ndarray) -> Optional[Dict]:
        tgt = state.get("target_info")
        if tgt is None:
            return None
        return {
            "pos": tgt["pos"],
            "dist": float(1.0 - obs[12]) * math.sqrt(100**2 + 100**2),
            "bearing": float(obs[13]) * 360 - 180,
            "visible": "camera" not in state.get("failed_sensors", []),
        }
