"""
ExecutionAgent: routes abstract actions through hardware adapter, detects anomalies.
"""
from typing import Dict, List, Optional, Tuple
from ..core.message_bus import MessageBus, Message
from ..adapters.base_adapter import HardwareAdapter


class ExecutionAgent:
    """
    Executes planned actions via the hardware adapter.
    Detects execution anomalies (commanded action ≠ state change).
    """

    name = "execution_agent"

    def __init__(self, adapter: HardwareAdapter, bus: MessageBus):
        self.adapter = adapter
        self.bus = bus
        self._last_state: Optional[Dict] = None
        self._execution_history: List[Dict] = []
        self._consecutive_failures: int = 0

        bus.subscribe("REPLAN", self._on_replan)

    def execute(self, action: str, state: Dict) -> Tuple[Dict, bool]:
        """
        Execute action. Returns (result, success).
        Publishes ACTION_COMPLETE and anomaly signals.
        """
        if not self.adapter.is_action_available(action):
            self.bus.publish("ANOMALY", self.name, {
                "action": action,
                "reason": "action_unavailable",
                "step": state.get("step", 0),
            }, priority="HIGH")
            return state, False

        result, success = self.adapter.execute(action, state)
        self._execution_history.append({
            "action": action,
            "success": success,
            "step": state.get("step", 0),
        })

        if not success:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                self.bus.publish("ANOMALY", self.name, {
                    "action": action,
                    "reason": "consecutive_failures",
                    "count": self._consecutive_failures,
                    "step": state.get("step", 0),
                }, priority="HIGH")
        else:
            self._consecutive_failures = 0

        self.bus.publish("ACTION_COMPLETE", self.name, {
            "action": action,
            "success": success,
            "step": state.get("step", 0),
        })

        # Detect anomaly: commanded move but no position change
        if self._last_state and action in ("move_forward",) and success:
            new_pos = result.get("state_dict", {}).get("agent_pos", [0, 0])
            old_pos = self._last_state.get("agent_pos", [0, 0])
            import math
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(new_pos, old_pos)))
            if d < 0.01:
                self.bus.publish("ANOMALY", self.name, {
                    "action": action,
                    "reason": "no_movement_detected",
                    "step": state.get("step", 0),
                }, priority="HIGH")

        new_state = result.get("state_dict", state)
        self._last_state = dict(new_state)
        return new_state, success

    def _on_replan(self, msg: Message) -> None:
        recovery_action = msg.data.get("recovery_action", "")
        if recovery_action in ("safe_mode", "stop"):
            self._consecutive_failures = 0

    def reset(self) -> None:
        self._last_state = None
        self._execution_history.clear()
        self._consecutive_failures = 0
