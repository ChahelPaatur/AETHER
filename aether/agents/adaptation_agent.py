"""
AdaptationAgent: receives fault reports and selects recovery strategies.
Implements graceful degradation with ranked strategies updated by MemoryAgent.
"""
from collections import deque
from typing import Dict, List, Optional
from ..core.message_bus import MessageBus, Message


RECOVERY_STRATEGIES = {
    "POWER_CRITICAL": [
        "safe_mode", "reduce_speed", "stop",
    ],
    "THERMAL_ANOMALY": [
        "reduce_power", "safe_mode", "stop",
    ],
    "IMU_DRIFT": [
        "recalibrate_imu", "reduce_speed", "safe_mode",
    ],
    "SENSOR_FAILURE": [
        "switch_backup_sensor", "reduce_speed", "dead_reckoning", "safe_mode",
    ],
    "ACTUATOR_DEGRADATION": [
        "reduce_speed", "recalibrate_imu", "safe_mode", "stop",
    ],
    "INTERMITTENT_FAULT": [
        "diagnostic_scan", "recalibrate_imu", "safe_mode",
    ],
    "SAFE_MODE": [
        "safe_mode",
    ],
}


class AdaptationAgent:
    """
    Selects and applies recovery strategies for detected faults.
    Tracks adaptation latency and maintains ranked strategy lists.
    """

    name = "adaptation_agent"

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self._strategy_ranks: Dict[str, List[str]] = {
            k: list(v) for k, v in RECOVERY_STRATEGIES.items()
        }
        self._active_recovery: Optional[Dict] = None
        self._recovery_attempt: int = 0
        self._fault_detected_step: int = 0
        self._latency_log: List[int] = []
        self._in_safe_mode: bool = False
        self._step: int = 0

        # Stuck detection
        self._position_history: deque = deque(maxlen=10)
        self._stuck_escape_idx: int = 0
        self._stuck_escape_queue: List[str] = []  # queued escape actions

        # recalibrate_imu escalation tracking: fault_type → attempt count
        self._recalibrate_attempts: Dict[str, int] = {}

        bus.subscribe("FAULT_DETECTED", self._on_fault)
        bus.subscribe("ACTION_COMPLETE", self._on_action_complete)
        bus.subscribe("STATE_UPDATE", self._on_state_update)

    def update_strategy_ranks(self, fault_type: str, ranked: List[str]) -> None:
        """MemoryAgent calls this to update strategy priority."""
        if fault_type in self._strategy_ranks:
            self._strategy_ranks[fault_type] = ranked

    def _is_stuck(self) -> bool:
        """True if position has not meaningfully changed in the last 5 steps."""
        if len(self._position_history) < 5:
            return False
        recent = list(self._position_history)[-5:]
        ref = recent[0]
        return all(
            abs(p[0] - ref[0]) < 0.5 and abs(p[1] - ref[1]) < 0.5
            for p in recent[1:]
        )

    def adapt(self, fault_reports: List, step: int) -> Optional[str]:
        """
        Select recovery action for the highest-severity fault.
        Returns the recovery action string, or None if no fault or already recovering.
        Stuck detection takes priority: if position unchanged for 5+ steps, escalate
        to turn_left / turn_right to break out before resuming normal recovery.
        """
        self._step = step

        # Execute next action in stuck escape queue (drains over several steps)
        if self._stuck_escape_queue:
            action = self._stuck_escape_queue.pop(0)
            return action

        # Stuck detection: queue a directional escape sequence to break out of boundary
        if self._is_stuck():
            turns = ["turn_left", "turn_right"]
            turn = turns[self._stuck_escape_idx % len(turns)]
            self._stuck_escape_idx += 1
            # move_backward × 3 moves away from the wall, then turn, then move forward × 3
            self._stuck_escape_queue = [
                "move_backward", "move_backward", "move_backward",
                turn,
                "move_forward", "move_forward", "move_forward",
            ]
            escape = self._stuck_escape_queue.pop(0)
            self.bus.publish("REPLAN", self.name, {
                "fault_type": "STUCK",
                "recovery_action": escape,
                "severity": 0.5,
                "confidence": 1.0,
                "latency": 0,
                "step": step,
            }, priority="HIGH")
            return escape

        if not fault_reports:
            # No faults: clear active recovery if it was low-severity
            if (self._active_recovery
                    and step - self._fault_detected_step > 5
                    and not self._in_safe_mode):
                self._active_recovery = None
            return None

        # Sort by severity descending; ignore very low severity predictive warnings
        serious = [f for f in fault_reports if f.severity >= 0.4]
        if not serious:
            return None
        sorted_faults = sorted(serious, key=lambda f: f.severity, reverse=True)
        top_fault = sorted_faults[0]

        # Don't repeatedly trigger recovery for the same fault type in same step window
        if (self._active_recovery
                and self._active_recovery.get("fault_type") == top_fault.fault_type
                and step - self._fault_detected_step < 5
                and not self._active_recovery.get("escalate")):
            return None

        if self._active_recovery is None or top_fault.fault_type != self._active_recovery.get("fault_type"):
            self._active_recovery = {
                "fault_type": top_fault.fault_type,
                "attempt": 0,
                "detected_step": step,
            }
            self._recovery_attempt = 0
            self._fault_detected_step = step

        strategies = self._strategy_ranks.get(top_fault.fault_type, ["safe_mode"])

        attempt = self._active_recovery.get("attempt", 0)
        if attempt >= len(strategies):
            # All strategies exhausted → enter safe mode
            self._in_safe_mode = True
            strategy = "safe_mode"
        else:
            strategy = strategies[attempt]

        latency = step - self._fault_detected_step
        self._latency_log.append(latency)

        # Escalate recalibrate_imu to safe_mode after 3 failed attempts
        if strategy == "recalibrate_imu":
            key = top_fault.fault_type
            self._recalibrate_attempts[key] = self._recalibrate_attempts.get(key, 0) + 1
            if self._recalibrate_attempts[key] >= 3 and top_fault.severity > 0.90:
                strategy = "safe_mode"
                self._in_safe_mode = True

        # Publish replan signal
        self.bus.publish("REPLAN", self.name, {
            "fault_type": top_fault.fault_type,
            "recovery_action": strategy,
            "severity": top_fault.severity,
            "confidence": top_fault.confidence,
            "latency": latency,
            "step": step,
        }, priority="HIGH")

        return strategy

    def mark_recovery_success(self, fault_type: str) -> None:
        """Called when recovery was successful."""
        if self._active_recovery and self._active_recovery["fault_type"] == fault_type:
            latency = self._step - self._fault_detected_step
            self._latency_log.append(latency)
            self._active_recovery = None
            self._recovery_attempt = 0
            self._in_safe_mode = False
            self._recalibrate_attempts.pop(fault_type, None)
            self.bus.publish("RECOVERY_SUCCESS", self.name, {
                "fault_type": fault_type,
                "latency": latency,
                "step": self._step,
            })

    def mark_recovery_failure(self) -> None:
        """Called when a recovery strategy failed — escalate to next."""
        if self._active_recovery:
            self._active_recovery["attempt"] = self._active_recovery.get("attempt", 0) + 1
            self._recovery_attempt += 1

    @property
    def in_safe_mode(self) -> bool:
        return self._in_safe_mode

    @property
    def avg_latency(self) -> float:
        if not self._latency_log:
            return 0.0
        return sum(self._latency_log) / len(self._latency_log)

    def _on_fault(self, msg: Message) -> None:
        pass  # handled via direct adapt() calls

    def _on_action_complete(self, msg: Message) -> None:
        if msg.data.get("success") and self._active_recovery:
            self.mark_recovery_success(self._active_recovery["fault_type"])

    def _on_state_update(self, msg: Message) -> None:
        pos = msg.data.get("position")
        if pos is not None:
            self._position_history.append((float(pos[0]), float(pos[1])))

    def reset(self) -> None:
        self._active_recovery = None
        self._recovery_attempt = 0
        self._in_safe_mode = False
        self._latency_log.clear()
        self._position_history.clear()
        self._stuck_escape_idx = 0
        self._stuck_escape_queue.clear()
        self._recalibrate_attempts.clear()
