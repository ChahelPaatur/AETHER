"""
CameraAgent: handles all vision tasks.
Reads camera sensor data (proximity + target indices) from the 15-dim obs vector,
processes the visual field, and publishes structured observations to the MessageBus.
"""
from typing import Dict, List, Optional
from ..core.message_bus import MessageBus, Message


# Observation vector indices
_PROX_FRONT = 9
_PROX_LEFT  = 10
_PROX_RIGHT = 11
_TGT_DIST   = 12
_TGT_BEAR   = 13

OBSTACLE_CRITICAL = 0.15   # proximity < this → imminent collision
OBSTACLE_WARNING  = 0.30   # proximity < this → caution zone


class CameraAgent:
    """
    Vision subsystem agent.
    Processes obstacle proximity and target tracking data each tick.
    Publishes CAMERA_UPDATE; responds to sensor fault signals to degrade gracefully.
    """

    name = "camera_agent"

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self._camera_failed: bool = False
        self._ultrasonic_failed: bool = False
        self._last_target: Optional[Dict] = None
        self._frame_count: int = 0

        bus.subscribe("FAULT_DETECTED", self._on_fault)

    def tick(self, obs, state: Dict, step: int) -> Dict:
        """
        Process visual field from observation vector.
        Returns structured visual state dict and publishes CAMERA_UPDATE.
        """
        import numpy as np
        obs = obs if hasattr(obs, '__len__') else list(obs)
        self._frame_count += 1

        failed_sensors = state.get("failed_sensors", [])
        self._camera_failed = "camera" in failed_sensors
        self._ultrasonic_failed = "ultrasonic" in failed_sensors

        # Obstacle map from proximity sensors
        prox_front = float(obs[_PROX_FRONT])
        prox_left  = float(obs[_PROX_LEFT])
        prox_right = float(obs[_PROX_RIGHT])

        obstacle_map = {
            "front": prox_front,
            "left":  prox_left,
            "right": prox_right,
        }
        min_clearance = min(prox_front, prox_left, prox_right)
        collision_imminent = (not self._ultrasonic_failed) and (min_clearance < OBSTACLE_CRITICAL)
        obstacle_warning   = (not self._ultrasonic_failed) and (min_clearance < OBSTACLE_WARNING)

        # Target tracking
        target_visible = not self._camera_failed
        target_dist_norm = float(obs[_TGT_DIST])
        target_bearing_norm = float(obs[_TGT_BEAR])
        # Denormalize bearing: [0,1] → [-180, 180]
        target_bearing_deg = target_bearing_norm * 360.0 - 180.0

        if target_visible:
            self._last_target = {
                "dist_norm":  target_dist_norm,
                "bearing_deg": target_bearing_deg,
            }

        # Visual clarity: degraded by sensor failure + proximity to obstacles
        proximity_noise = 1.0 - min_clearance
        visual_clarity = float(
            (1.0 if target_visible else 0.3) *
            (1.0 if not self._ultrasonic_failed else 0.5) *
            max(0.2, 1.0 - proximity_noise * 0.5)
        )

        result = {
            "target_visible":     target_visible,
            "target_dist_norm":   target_dist_norm,
            "target_bearing_deg": target_bearing_deg,
            "obstacle_map":       obstacle_map,
            "min_clearance":      min_clearance,
            "collision_imminent": collision_imminent,
            "obstacle_warning":   obstacle_warning,
            "visual_clarity":     visual_clarity,
            "camera_failed":      self._camera_failed,
            "ultrasonic_failed":  self._ultrasonic_failed,
            "step": step,
        }

        self.bus.publish("CAMERA_UPDATE", self.name, result)
        return result

    def _on_fault(self, msg: Message) -> None:
        fault_type = msg.data.get("fault_type", "")
        subsystem  = msg.data.get("subsystem", "")
        if fault_type == "SENSOR_FAILURE" and subsystem in ("camera", "ultrasonic", "sensor"):
            self._camera_failed = True

    def reset(self) -> None:
        self._camera_failed = False
        self._ultrasonic_failed = False
        self._last_target = None
        self._frame_count = 0
