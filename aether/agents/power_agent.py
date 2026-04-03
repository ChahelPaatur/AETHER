"""
PowerAgent: monitors and manages the energy subsystem.
Tracks battery level, solar input, and bus voltage; triggers power-saving modes;
coordinates with FaultAgent when battery is critical.
"""
from typing import Dict, List
from ..core.message_bus import MessageBus, Message


# EPS observation indices
_BATTERY    = 0
_SOLAR      = 1
_BUS_VOLT   = 2

# Thresholds
BATTERY_CRITICAL = 0.15
BATTERY_SAVING   = 0.35
BUS_VOLT_LOW     = 0.65


class PowerAgent:
    """
    Energy subsystem agent.
    Classifies power mode each tick and publishes POWER_STATUS.
    Responds to fault signals that affect EPS (e.g. battery drain injections).
    """

    name = "power_agent"

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self._power_mode: str = "nominal"   # nominal | saving | critical
        self._history: List[Dict] = []
        self._step: int = 0

        bus.subscribe("FAULT_DETECTED", self._on_fault)

    def tick(self, obs, step: int) -> Dict:
        """
        Read EPS indices from observation, classify mode, publish POWER_STATUS.
        Returns power state dict.
        """
        self._step = step
        battery   = float(obs[_BATTERY])
        solar     = float(obs[_SOLAR])
        bus_volt  = float(obs[_BUS_VOLT])

        # Classify power mode
        if battery < BATTERY_CRITICAL or bus_volt < BUS_VOLT_LOW:
            mode = "critical"
        elif battery < BATTERY_SAVING:
            mode = "saving"
        else:
            mode = "nominal"

        mode_changed = mode != self._power_mode
        self._power_mode = mode

        # Estimate energy balance
        drain_rate = 0.001   # baseline per step
        net_balance = solar * 0.0005 - drain_rate
        steps_remaining = max(0, int(battery / drain_rate)) if net_balance < 0 else 9999

        result = {
            "battery":          battery,
            "solar":            solar,
            "bus_voltage":      bus_volt,
            "power_mode":       mode,
            "mode_changed":     mode_changed,
            "net_balance":      round(net_balance, 6),
            "steps_remaining":  steps_remaining,
            "step":             step,
        }

        priority = "HIGH" if mode == "critical" else "NORMAL"
        self.bus.publish("POWER_STATUS", self.name, result, priority=priority)

        if mode == "critical":
            self.bus.publish("FAULT_DETECTED", self.name, {
                "fault_type":         "POWER_CRITICAL",
                "subsystem":          "battery",
                "severity":           round(1.0 - battery / BATTERY_CRITICAL, 4),
                "confidence":         0.99,
                "detection_method":   "POWER_AGENT",
                "recommended_action": "safe_mode",
                "timestep":           step,
            }, priority="HIGH")

        self._history.append(result)
        if len(self._history) > 50:
            self._history.pop(0)

        return result

    def _on_fault(self, msg: Message) -> None:
        fault_type = msg.data.get("fault_type", "")
        if fault_type == "POWER_CRITICAL" and msg.sender != self.name:
            self._power_mode = "critical"

    def reset(self) -> None:
        self._power_mode = "nominal"
        self._history.clear()
        self._step = 0
