"""
ThermalAgent: monitors temperature across subsystems.
Activates cooling or heating responses and flags thermal anomalies to FaultAgent
early — before the rule-based threshold is crossed.
"""
from typing import Dict, List
from ..core.message_bus import MessageBus, Message


# TCS observation indices
_TEMP_PANEL = 7
_TEMP_CORE  = 8

# Thresholds
CORE_CRITICAL = 0.85
CORE_WARM     = 0.70
PANEL_COLD    = 0.10   # hypothermic risk (low-temp environment)

EARLY_WARNING_WINDOW = 5   # steps of trend to compute


class ThermalAgent:
    """
    Thermal subsystem agent.
    Classifies thermal mode each tick, trends temperature trajectory,
    and publishes THERMAL_STATUS. Raises early-warning fault signals
    when core temperature is trending toward threshold.
    """

    name = "thermal_agent"

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self._thermal_mode: str = "nominal"   # nominal | warm | critical
        self._core_history: List[float] = []
        self._step: int = 0

        bus.subscribe("FAULT_DETECTED", self._on_fault)

    def tick(self, obs, step: int) -> Dict:
        """
        Read TCS indices, classify thermal mode, publish THERMAL_STATUS.
        Returns thermal state dict.
        """
        self._step = step
        temp_panel = float(obs[_TEMP_PANEL])
        temp_core  = float(obs[_TEMP_CORE])

        # Maintain rolling temperature history for trend analysis
        self._core_history.append(temp_core)
        if len(self._core_history) > EARLY_WARNING_WINDOW:
            self._core_history.pop(0)

        # Trend: positive slope = heating, negative = cooling
        trend = 0.0
        if len(self._core_history) >= 3:
            n = len(self._core_history)
            trend = (self._core_history[-1] - self._core_history[0]) / max(1, n - 1)

        # Classify thermal mode
        if temp_core >= CORE_CRITICAL:
            mode = "critical"
        elif temp_core >= CORE_WARM:
            mode = "warm"
        else:
            mode = "nominal"

        mode_changed = mode != self._thermal_mode
        self._thermal_mode = mode

        # Cooling response recommendation
        cooling_active = temp_core >= CORE_WARM
        heating_active = temp_panel < PANEL_COLD

        # Early-warning: core is nominal but trending toward CORE_WARM
        early_warning = (
            mode == "nominal"
            and trend > 0.002
            and temp_core > CORE_WARM * 0.85
        )

        result = {
            "temp_panel":     temp_panel,
            "temp_core":      temp_core,
            "thermal_mode":   mode,
            "mode_changed":   mode_changed,
            "trend_per_step": round(trend, 6),
            "cooling_active": cooling_active,
            "heating_active": heating_active,
            "early_warning":  early_warning,
            "step":           step,
        }

        priority = "HIGH" if mode == "critical" else "NORMAL"
        self.bus.publish("THERMAL_STATUS", self.name, result, priority=priority)

        # Publish early-warning as predictive fault signal
        if early_warning or mode in ("warm", "critical"):
            severity = (temp_core - CORE_WARM) / (1.0 - CORE_WARM) if temp_core >= CORE_WARM else trend * 10.0
            self.bus.publish("FAULT_DETECTED", self.name, {
                "fault_type":         "THERMAL_ANOMALY",
                "subsystem":          "thermal",
                "severity":           round(min(1.0, max(0.1, severity)), 4),
                "confidence":         0.97 if mode == "critical" else 0.70,
                "detection_method":   "THERMAL_AGENT",
                "recommended_action": "reduce_power",
                "timestep":           step,
            }, priority=priority)

        return result

    def _on_fault(self, msg: Message) -> None:
        fault_type = msg.data.get("fault_type", "")
        if fault_type == "THERMAL_ANOMALY" and msg.sender != self.name:
            self._thermal_mode = "warm"

    def reset(self) -> None:
        self._thermal_mode = "nominal"
        self._core_history.clear()
        self._step = 0
