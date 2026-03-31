"""
RoverAdapter v3: translates abstract actions into rover hardware commands.
Implements HardwareAdapter interface with degradation simulation.
"""
from typing import Dict, Optional, Tuple
from .base_adapter import HardwareAdapter


class RoverAdapter(HardwareAdapter):
    """Full rover adapter with degradation simulation and actuator state tracking."""

    ROBOT_TYPE = "rover"

    def __init__(self, environment, max_speed: float = 1.0):
        self.env = environment
        self.max_speed = max_speed
        self._wheel_degradation: float = 0.0
        self._arm_degradation: float = 0.0
        self._actuator_state: Dict = {"wheels": "nominal", "arm": "nominal"}

    def execute(self, action: str, state: Dict) -> Tuple[Dict, bool]:
        """Execute abstract action → rover hardware command → env step."""
        params = self._translate(action, state)
        effective_action = self._map_to_env_action(action)
        obs, reward, done, info = self.env.step(effective_action, params)
        success = info.get("success", False) or not info.get("collision", False)
        result = {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info,
            "action_taken": effective_action,
            "state_dict": self.env.get_state_dict(),
        }
        return result, success

    def is_action_available(self, action: str) -> bool:
        if "wheels" in self.env.failed_actuators and action in (
                "move_forward", "move_backward", "turn_left", "turn_right", "follow_target"):
            return False
        return True

    def get_degradation_state(self) -> Dict:
        return {
            "wheels": self._wheel_degradation,
            "arm": self._arm_degradation,
        }

    def get_actuator_state(self) -> Dict:
        return dict(self._actuator_state)

    def simulate_degradation(self, degradation_level: float) -> None:
        self._wheel_degradation = min(1.0, degradation_level)
        if self._wheel_degradation > 0.8:
            self._actuator_state["wheels"] = "critical"
        elif self._wheel_degradation > 0.4:
            self._actuator_state["wheels"] = "degraded"

    def _translate(self, action: str, state: Dict) -> Dict:
        eff_speed = self.max_speed * (1.0 - self._wheel_degradation * 0.8)
        if action == "move_forward":
            return {"speed": eff_speed}
        elif action == "move_backward":
            return {"speed": eff_speed * 0.5}
        elif action in ("turn_left", "turn_right"):
            return {}
        elif action == "follow_target":
            return {"speed": eff_speed, "target": state.get("target_info")}
        elif action in ("stop", "emergency_stop"):
            return {"speed": 0}
        elif action == "scan":
            return {}
        elif action == "safe_mode":
            return {"speed": 0}
        return {}

    def _map_to_env_action(self, action: str) -> str:
        mapping = {
            "turn_left": "turn_left",
            "turn_right": "turn_right",
            "follow_target": "follow_target",
            "emergency_stop": "stop",
            "safe_mode": "stop",
            "scan": "scan_environment",
            "report_state": "scan_environment",
        }
        return mapping.get(action, action)
