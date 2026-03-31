"""
MovementAgent: handles all locomotion.
Consumes plans from PlannerAgent and translates them into low-level movement
commands via the hardware adapter. Applies speed constraints from PowerAgent
and collision avoidance directives from CameraAgent.
"""
import math
from typing import Dict, List, Optional, Tuple
from ..core.message_bus import MessageBus, Message
from ..adapters.base_adapter import HardwareAdapter


class MovementAgent:
    """
    Locomotion subsystem agent.
    Executes the action selected by PlannerAgent, respects power and collision
    constraints, and publishes MOVEMENT_STATUS after each command.
    """

    name = "movement_agent"

    def __init__(self, adapter: HardwareAdapter, bus: MessageBus):
        self.adapter = adapter
        self.bus = bus
        self._speed_limit: float = 1.0       # reduced by PowerAgent when saving
        self._collision_imminent: bool = False
        self._last_pos: Optional[List] = None
        self._consecutive_failures: int = 0
        self._action_history: List[Dict] = []
        self._heading_correction: float = 0.0

        bus.subscribe("REPLAN",            self._on_replan)
        bus.subscribe("POWER_STATUS",      self._on_power_status)
        bus.subscribe("CAMERA_UPDATE",     self._on_camera_update)
        bus.subscribe("FAULT_DETECTED",    self._on_fault)
        bus.subscribe("NAVIGATION_UPDATE", self._on_navigation_update)

    def execute(self, action: str, state: Dict) -> Tuple[Dict, bool]:
        """
        Execute action via adapter with current speed / safety constraints.
        Returns (new_state, success).
        """
        # Safety override: if collision imminent and action is forward, force avoidance
        if self._collision_imminent and action == "move_forward":
            action = "turn_right"

        if not self.adapter.is_action_available(action):
            self.bus.publish("MOVEMENT_STATUS", self.name, {
                "action": action,
                "executed": "stop",
                "success": False,
                "reason": "action_unavailable",
                "step": state.get("step", 0),
            }, priority="HIGH")
            return state, False

        # Apply speed limit via state context (adapter uses params)
        state_with_speed = dict(state)
        state_with_speed["_speed_limit"] = self._speed_limit

        # Apply IMU drift heading correction for translational movement.
        # Temporarily offset env heading so the step moves in the corrected
        # direction, then restore original heading to prevent compounding.
        applied_correction = False
        original_heading = None
        if (self._heading_correction != 0.0
                and action in ("move_forward", "move_backward")
                and hasattr(self.adapter, "env")):
            original_heading = self.adapter.env.agent_heading
            self.adapter.env.agent_heading = (original_heading + self._heading_correction) % 360
            applied_correction = True

        result, success = self.adapter.execute(action, state_with_speed)

        if applied_correction and original_heading is not None:
            self.adapter.env.agent_heading = original_heading

        if success:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

        # Detect stalled movement
        new_state = result.get("state_dict", state)
        stalled = False
        if action == "move_forward" and success and self._last_pos:
            new_pos = new_state.get("agent_pos", [0, 0])
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(new_pos, self._last_pos)))
            if d < 0.01:
                stalled = True
                self.bus.publish("ANOMALY", self.name, {
                    "action": action,
                    "reason": "movement_stalled",
                    "step": state.get("step", 0),
                }, priority="HIGH")

        self._last_pos = new_state.get("agent_pos", [0, 0])

        status = {
            "action":     action,
            "executed":   result.get("action_taken", action),
            "success":    success and not stalled,
            "position":   self._last_pos,
            "heading":    new_state.get("agent_heading", 0.0),
            "stalled":    stalled,
            "step":       state.get("step", 0),
        }
        self.bus.publish("MOVEMENT_STATUS", self.name, status)
        self._action_history.append(status)

        return new_state, success

    def _on_replan(self, msg: Message) -> None:
        recovery = msg.data.get("recovery_action", "")
        if recovery in ("safe_mode", "stop", "reduce_speed"):
            self._speed_limit = 0.3 if recovery == "reduce_speed" else 0.0
        if recovery in ("safe_mode", "stop"):
            self._consecutive_failures = 0

    def _on_power_status(self, msg: Message) -> None:
        mode = msg.data.get("power_mode", "nominal")
        if mode == "critical":
            self._speed_limit = min(self._speed_limit, 0.3)
        elif mode == "saving":
            self._speed_limit = min(self._speed_limit, 0.7)
        else:
            # Restore only if not already throttled by replan
            self._speed_limit = max(self._speed_limit, 1.0)

    def _on_camera_update(self, msg: Message) -> None:
        self._collision_imminent = msg.data.get("collision_imminent", False)

    def _on_navigation_update(self, msg: Message) -> None:
        self._heading_correction = msg.data.get("heading_correction_deg", 0.0)

    def _on_fault(self, msg: Message) -> None:
        fault_type = msg.data.get("fault_type", "")
        if fault_type == "ACTUATOR_DEGRADATION":
            self._speed_limit = min(self._speed_limit, 0.5)

    def reset(self) -> None:
        self._speed_limit = 1.0
        self._collision_imminent = False
        self._last_pos = None
        self._consecutive_failures = 0
        self._action_history.clear()
        self._heading_correction = 0.0
