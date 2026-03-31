"""
NavigationAgent: fuses IMU, camera, and proximity sensor data to maintain
an accurate position estimate and heading. Feeds corrected pose to
MovementAgent and PlannerAgent via the MessageBus.
"""
import math
from collections import deque
from typing import Dict, List, Optional
from ..core.message_bus import MessageBus, Message


# ADCS observation indices
_IMU_X         = 3
_IMU_Y         = 4
_IMU_Z         = 5
_ATTITUDE_ERR  = 6

IMU_HEALTHY_CENTER = 0.5   # normalized healthy = 0.5
IMU_DRIFT_THRESHOLD = 0.25  # deviation > this from center → unhealthy


class NavigationAgent:
    """
    Navigation subsystem agent.
    Computes IMU health, estimates pose confidence by fusing gyro and visual data,
    and publishes NAVIGATION_UPDATE each tick.
    Responds to IMU_DRIFT faults by flagging recalibration need.
    """

    name = "navigation_agent"

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self._imu_healthy: bool = True
        self._needs_recalibration: bool = False
        self._heading_estimate: float = 0.0
        self._position_estimate: Optional[List] = None
        self._imu_window: deque = deque(maxlen=5)
        self._last_camera: Optional[Dict] = None
        self._target_dist_history: deque = deque(maxlen=6)

        bus.subscribe("CAMERA_UPDATE",  self._on_camera_update)
        bus.subscribe("FAULT_DETECTED", self._on_fault)

    def tick(self, obs, state: Dict, step: int) -> Dict:
        """
        Fuse IMU and camera data. Publish NAVIGATION_UPDATE.
        Returns navigation state dict.
        """
        import numpy as np

        imu_x    = float(obs[_IMU_X])
        imu_y    = float(obs[_IMU_Y])
        imu_z    = float(obs[_IMU_Z])
        att_err  = float(obs[_ATTITUDE_ERR])

        # IMU health: deviation of each axis from healthy center
        imu_vals = [imu_x, imu_y, imu_z]
        max_dev  = max(abs(v - IMU_HEALTHY_CENTER) for v in imu_vals)
        imu_drift_magnitude = max_dev   # 0 = healthy, 0.5 = max drift

        self._imu_healthy = (
            imu_drift_magnitude < IMU_DRIFT_THRESHOLD
            and att_err < 0.75
            and "imu" not in state.get("failed_sensors", [])
        )

        # Sliding variance to detect intermittent IMU spikes
        self._imu_window.append(imu_drift_magnitude)
        imu_variance = float(np.var(list(self._imu_window))) if len(self._imu_window) > 1 else 0.0

        # Pose: use env ground-truth when available, flag confidence from IMU
        self._position_estimate = state.get("agent_pos", [0.0, 0.0])
        raw_heading = float(state.get("agent_heading", 0.0))

        # Heading fusion: if camera sees target, use bearing to refine heading
        heading_corrected = raw_heading
        if self._last_camera and self._last_camera.get("target_visible"):
            bearing_deg = self._last_camera.get("target_bearing_deg", 0.0)
            # Smooth blend: 80% gyro, 20% visual correction
            heading_corrected = raw_heading * 0.8 + (raw_heading + bearing_deg) * 0.2

        self._heading_estimate = heading_corrected % 360.0

        # Pose confidence: drops with IMU drift and attitude error
        pose_confidence = float(max(0.1, min(1.0,
            1.0
            - imu_drift_magnitude * 1.5
            - att_err * 0.5
            - imu_variance * 2.0
        )))

        # Penalize NavConf when target distance increases for 5+ consecutive steps
        target = state.get("target", {})
        target_dist = target.get("dist", 0) if isinstance(target, dict) else 0
        if target_dist > 0:
            self._target_dist_history.append(target_dist)
        if len(self._target_dist_history) >= 6:
            dists = list(self._target_dist_history)
            if all(dists[i] < dists[i + 1] for i in range(len(dists) - 1)):
                pose_confidence = max(0.1, pose_confidence - 0.4)

        # Dead-reckoning heading correction when IMU drift is significant.
        # Positive drift_x = agent drifting right → correct LEFT → add to heading
        # (env convention: higher heading = more counterclockwise = left).
        heading_correction_deg = 0.0
        if not self._imu_healthy and imu_drift_magnitude > 0.15:
            drift_x = imu_x - IMU_HEALTHY_CENTER   # positive = drifting right
            drift_y = imu_y - IMU_HEALTHY_CENTER
            # Gentle proportional correction, capped at ±12°
            heading_correction_deg = float(max(-12.0, min(12.0,
                (drift_x * 20.0 + drift_y * 10.0)
            )))

        result = {
            "position":              self._position_estimate,
            "heading_estimate":      round(self._heading_estimate, 2),
            "imu_drift_magnitude":   round(imu_drift_magnitude, 4),
            "imu_healthy":           self._imu_healthy,
            "imu_variance":          round(imu_variance, 6),
            "attitude_error":        round(att_err, 4),
            "pose_confidence":       round(pose_confidence, 4),
            "needs_recalibration":   self._needs_recalibration,
            "heading_correction_deg": round(heading_correction_deg, 2),
            "step":                  step,
        }

        priority = "HIGH" if not self._imu_healthy else "NORMAL"
        self.bus.publish("NAVIGATION_UPDATE", self.name, result, priority=priority)

        # Early fault signal when IMU drift is significant but not yet rule-threshold
        if imu_drift_magnitude > IMU_DRIFT_THRESHOLD * 0.8 and self._imu_healthy:
            self.bus.publish("FAULT_DETECTED", self.name, {
                "fault_type":         "IMU_DRIFT",
                "subsystem":          "imu",
                "severity":           round(imu_drift_magnitude / 0.5, 4),
                "confidence":         0.80,
                "detection_method":   "NAVIGATION_AGENT",
                "recommended_action": "recalibrate_imu",
                "timestep":           step,
            }, priority="HIGH")

        return result

    def _on_camera_update(self, msg: Message) -> None:
        self._last_camera = msg.data

    def _on_fault(self, msg: Message) -> None:
        fault_type = msg.data.get("fault_type", "")
        if fault_type == "IMU_DRIFT":
            self._needs_recalibration = True
            self._imu_healthy = False
        if msg.data.get("recovery_action") == "recalibrate_imu":
            self._needs_recalibration = False

    def reset(self) -> None:
        self._imu_healthy = True
        self._needs_recalibration = False
        self._heading_estimate = 0.0
        self._position_estimate = None
        self._imu_window.clear()
        self._last_camera = None
        self._target_dist_history.clear()
