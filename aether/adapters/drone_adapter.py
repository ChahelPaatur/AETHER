"""
DroneAdapter v3: stub adapter for drone hardware, same interface as RoverAdapter.
Raises NotImplementedError for unimplemented flight-specific actions.
"""
from typing import Dict, Tuple
from .base_adapter import HardwareAdapter


class DroneAdapter(HardwareAdapter):
    """Drone adapter stub — implements interface, flight controller integration pending."""

    ROBOT_TYPE = "drone"
    DEFAULT_ALTITUDE = 5.0

    def __init__(self, environment, max_speed: float = 2.0):
        self.env = environment
        self.max_speed = max_speed
        self.altitude = self.DEFAULT_ALTITUDE
        self._thruster_degradation: float = 0.0

    def execute(self, action: str, state: Dict) -> Tuple[Dict, bool]:
        params = self._translate(action, state)
        mapped = self._map_to_env_action(action)
        obs, reward, done, info = self.env.step(mapped, params)
        result = {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info,
            "action_taken": mapped,
            "state_dict": self.env.get_state_dict(),
        }
        return result, not info.get("collision", False)

    def is_action_available(self, action: str) -> bool:
        if "thrusters" in self.env.failed_actuators and action not in ("stop", "emergency_stop"):
            return False
        return True

    def get_degradation_state(self) -> Dict:
        return {"thrusters": self._thruster_degradation}

    def simulate_degradation(self, level: float) -> None:
        self._thruster_degradation = min(1.0, level)

    def _translate(self, action: str, state: Dict) -> Dict:
        eff_speed = self.max_speed * (1.0 - self._thruster_degradation * 0.9)
        if action == "move_forward":
            return {"speed": eff_speed, "altitude": self.altitude}
        elif action == "hover":
            return {"speed": 0, "altitude": self.altitude}
        elif action == "avoid_obstacle":
            return {"altitude": self.altitude + 3.0, "speed": eff_speed * 0.5}
        elif action in ("turn_left", "turn_right"):
            return {"altitude": self.altitude}
        return {"speed": 0, "altitude": self.altitude}

    def _map_to_env_action(self, action: str) -> str:
        mapping = {
            "hover": "stop",
            "emergency_stop": "stop",
            "safe_mode": "stop",
            "turn_left": "turn_left",
            "turn_right": "turn_right",
            "follow_target": "follow_target",
        }
        return mapping.get(action, action)
