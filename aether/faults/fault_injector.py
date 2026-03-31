"""
FaultInjector v3: probabilistic, time-correlated fault injection with realistic fault models.
Faults persist and evolve — they don't just appear once per step.
"""
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ActiveFault:
    fault_type: str
    subsystem: str
    start_step: int
    severity: float
    config: Dict = field(default_factory=dict)


class BaseFaultModel:
    """Abstract base for time-correlated fault models."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.active: bool = False
        self.step: int = 0

    def tick(self) -> float:
        self.step += 1
        return 0.0

    def reset(self) -> None:
        self.active = False
        self.step = 0


class GaussianNoiseFault(BaseFaultModel):
    def __init__(self, rng, sigma: float = 0.1):
        super().__init__(rng)
        self.sigma = sigma

    def tick(self) -> float:
        return float(self.rng.normal(0, self.sigma))


class DropoutFault(BaseFaultModel):
    def __init__(self, rng, p_drop: float = 0.3):
        super().__init__(rng)
        self.p_drop = p_drop

    def tick(self) -> float:
        return -1.0 if self.rng.random() < self.p_drop else 0.0  # -1 = signal dropout


class DegradationFault(BaseFaultModel):
    def __init__(self, rng, rate: float = 0.005):
        super().__init__(rng)
        self.rate = rate
        self.accumulated = 0.0

    def tick(self) -> float:
        self.accumulated = min(1.0, self.accumulated + self.rate)
        return self.accumulated

    def reset(self) -> None:
        super().reset()
        self.accumulated = 0.0


class BatteryFault(BaseFaultModel):
    def __init__(self, rng, drain_rate: float = 0.02):
        super().__init__(rng)
        self.drain_rate = drain_rate
        self.accumulated = 0.0

    def tick(self) -> float:
        self.accumulated = min(1.0, self.accumulated + self.drain_rate)
        return -self.accumulated  # cumulative drain applied to observation

    def reset(self) -> None:
        super().reset()
        self.accumulated = 0.0


class ThermalFault(BaseFaultModel):
    def __init__(self, rng, delta: float = 0.03):
        super().__init__(rng)
        self.delta = delta
        self.accumulated = 0.0

    def tick(self) -> float:
        self.accumulated = min(1.0, self.accumulated + self.delta)
        return self.accumulated  # cumulative rise applied to observation

    def reset(self) -> None:
        super().reset()
        self.accumulated = 0.0


class DriftFault(BaseFaultModel):
    def __init__(self, rng, bias: float = 0.008):
        super().__init__(rng)
        self.bias = bias
        self.accumulated = 0.0

    def tick(self) -> float:
        noise = float(self.rng.normal(self.bias, self.bias * 0.1))
        self.accumulated = min(1.0, self.accumulated + abs(noise))
        return self.accumulated  # return cumulative drift, not per-step noise

    def reset(self) -> None:
        super().reset()
        self.accumulated = 0.0


class IntermittentFault(BaseFaultModel):
    def __init__(self, rng, on_prob: float = 0.1, off_prob: float = 0.4):
        super().__init__(rng)
        self.on_prob = on_prob
        self.off_prob = off_prob

    def tick(self) -> float:
        if self.active:
            if self.rng.random() < self.off_prob:
                self.active = False
            return 0.45 if self.active else 0.0
        else:
            if self.rng.random() < self.on_prob:
                self.active = True
            return 0.45 if self.active else 0.0


class FaultInjector:
    """
    Manages probabilistic, time-correlated fault injection.
    Faults affect specific indices of the 15-dim observation vector.
    """

    # Mapping: fault_type → (obs_indices_affected, model_class, default_params)
    FAULT_CATALOG = {
        "sensor_noise_imu":     ([3, 4, 5, 6], GaussianNoiseFault,  {"sigma": 0.20}),
        "sensor_dropout_camera":([12, 13],      DropoutFault,        {"p_drop": 0.35}),
        "actuator_degrade":     ([9, 10, 11],   BatteryFault,        {"drain_rate": 0.006}),
        "battery_drain":        ([0],           BatteryFault,        {"drain_rate": 0.030}),
        "thermal_spike":        ([7, 8],        ThermalFault,        {"delta": 0.050}),
        "imu_drift":            ([3, 4, 5, 6],  DriftFault,          {"bias": 0.012}),
        "intermittent":         ([3, 6, 12],    IntermittentFault,   {"on_prob": 0.15, "off_prob": 0.30}),
        "bus_voltage_drop":     ([2],           BatteryFault,        {"drain_rate": 0.005}),
    }

    def __init__(self, fault_probability: float = 0.02,
                 seed: Optional[int] = None):
        self.fault_probability = fault_probability
        self.rng = np.random.default_rng(seed)
        self._models: Dict[str, BaseFaultModel] = {}
        self._active_faults: Dict[str, ActiveFault] = {}
        self._scheduled: List[Dict] = []
        self._build_models()

    def _build_models(self) -> None:
        for name, (_, cls, params) in self.FAULT_CATALOG.items():
            self.rng, sub_rng = self.rng, np.random.default_rng(
                int(self.rng.integers(0, 2**31)))
            self._models[name] = cls(sub_rng, **params)

    def schedule(self, fault_type: str, start_step: int,
                 subsystem: str = "", severity: float = 0.5) -> None:
        """Schedule a deterministic fault injection."""
        self._scheduled.append({
            "fault_type": fault_type,
            "start_step": start_step,
            "subsystem": subsystem,
            "severity": severity,
        })

    def load_from_scenario(self, scenario: Dict) -> None:
        for fault in scenario.get("faults", []):
            self.schedule(
                fault_type=fault.get("type", "sensor_noise_imu"),
                start_step=fault.get("start_step", 0),
                subsystem=fault.get("subsystem", ""),
                severity=fault.get("severity", 0.5),
            )

    def tick(self, obs: np.ndarray, step: int) -> np.ndarray:
        """
        Apply active faults to observation vector.
        Also activates scheduled faults and probabilistically injects new ones.
        Returns corrupted observation.
        """
        obs = obs.copy()

        # Activate scheduled faults
        for s in self._scheduled:
            if s["start_step"] == step and s["fault_type"] not in self._active_faults:
                self._activate(s["fault_type"], step, s.get("severity", 0.5))

        # Probabilistic random fault injection
        if self.rng.random() < self.fault_probability:
            candidates = [k for k in self.FAULT_CATALOG if k not in self._active_faults]
            if candidates:
                chosen = candidates[int(self.rng.integers(0, len(candidates)))]
                self._activate(chosen, step, float(self.rng.uniform(0.3, 0.8)))

        # Apply all active fault models
        # Cumulative models (have 'accumulated' attr): severity was baked into
        # rate at activation — apply delta directly so all faults eventually
        # reach full magnitude (low severity = slow onset, not weak ceiling).
        # Non-cumulative models: severity scales per-step magnitude.
        for fault_name, fault_info in list(self._active_faults.items()):
            model = self._models.get(fault_name)
            if model is None:
                continue
            indices, _, _ = self.FAULT_CATALOG[fault_name]
            delta = model.tick()
            is_cumulative = hasattr(model, 'accumulated')
            severity_scale = fault_info.severity
            for idx in indices:
                if 0 <= idx < len(obs):
                    if delta == -1.0:  # dropout
                        obs[idx] = 0.5
                    elif is_cumulative:
                        obs[idx] = float(np.clip(obs[idx] + delta, 0.0, 1.0))
                    else:
                        obs[idx] = float(np.clip(obs[idx] + delta * severity_scale, 0.0, 1.0))

        return obs

    def _activate(self, fault_type: str, step: int, severity: float) -> None:
        if fault_type not in self.FAULT_CATALOG:
            return
        self._active_faults[fault_type] = ActiveFault(
            fault_type=fault_type,
            subsystem=self._fault_to_subsystem(fault_type),
            start_step=step,
            severity=severity,
        )
        model = self._models.get(fault_type)
        if model:
            model.reset()
            model.active = True
            # For cumulative models, severity scales the accumulation rate
            # (slow onset at low severity, but eventually reaches full magnitude)
            _, _, defaults = self.FAULT_CATALOG[fault_type]
            if hasattr(model, 'drain_rate') and 'drain_rate' in defaults:
                model.drain_rate = defaults['drain_rate'] * severity
            if hasattr(model, 'delta') and 'delta' in defaults:
                model.delta = defaults['delta'] * severity
            if hasattr(model, 'bias') and 'bias' in defaults:
                model.bias = defaults['bias'] * severity
            if hasattr(model, 'rate') and 'rate' in defaults:
                model.rate = defaults['rate'] * severity

    def _fault_to_subsystem(self, fault_type: str) -> str:
        mapping = {
            "sensor_noise_imu": "imu", "imu_drift": "imu",
            "sensor_dropout_camera": "camera",
            "actuator_degrade": "wheels",
            "battery_drain": "battery",
            "thermal_spike": "thermal",
            "intermittent": "imu",
            "bus_voltage_drop": "power",
        }
        return mapping.get(fault_type, "unknown")

    def get_active_faults(self) -> List[Dict]:
        return [
            {
                "fault_type": f.fault_type,
                "subsystem": f.subsystem,
                "start_step": f.start_step,
                "severity": f.severity,
            }
            for f in self._active_faults.values()
        ]

    # Recovery action → fault types to mitigate
    _RECOVERY_FAULT_MAP: Dict[str, List[str]] = {
        "recalibrate_imu":      ["imu_drift", "sensor_noise_imu"],
        "reduce_power":         ["thermal_spike"],
        "safe_mode":            ["battery_drain", "bus_voltage_drop"],
        "switch_backup_sensor": ["sensor_dropout_camera"],
        "reduce_speed":         ["battery_drain"],
    }

    def apply_recovery(self, recovery_action: str) -> bool:
        """
        Apply the effect of a recovery action to active faults.
        recalibrate_imu: fully resets drift accumulator to 0.0 (genuine correction window).
        Other actions: reduce accumulated and severity by 0.30.
        Returns True if any fault was mitigated.
        """
        targets = self._RECOVERY_FAULT_MAP.get(recovery_action, [])
        mitigated = False
        for fault_name in targets:
            if fault_name in self._active_faults:
                fault = self._active_faults[fault_name]
                model = self._models.get(fault_name)
                if recovery_action == "recalibrate_imu":
                    # Full reset: zero drift accumulator, halve rate
                    if model and hasattr(model, "accumulated"):
                        model.accumulated = 0.0
                    if hasattr(model, 'bias'):
                        model.bias *= 0.5
                    fault.severity = max(0.0, fault.severity * 0.5)
                else:
                    # Reduce accumulator directly (cumulative models)
                    if model and hasattr(model, "accumulated"):
                        model.accumulated = max(0.0, model.accumulated - 0.3)
                    fault.severity = max(0.0, fault.severity - 0.3)
                mitigated = True
        return mitigated

    def reset(self) -> None:
        self._active_faults = {}
        for model in self._models.values():
            model.reset()
        self._scheduled = []
