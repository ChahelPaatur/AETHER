"""
FaultDetector v3: rule-based detector used as the safety backup in the DRL-First FDIR.
Always-on deterministic floor beneath the neural network.
Combines absolute thresholds with rate-of-change detection for early fault identification.
"""
from collections import deque
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
import numpy as np


class FaultType(Enum):
    SENSOR_FAILURE = "SENSOR_FAILURE"
    ACTUATOR_DEGRADATION = "ACTUATOR_DEGRADATION"
    POWER_CRITICAL = "POWER_CRITICAL"
    THERMAL_ANOMALY = "THERMAL_ANOMALY"
    IMU_DRIFT = "IMU_DRIFT"
    INTERMITTENT_FAULT = "INTERMITTENT_FAULT"
    NO_FAULT = "NO_FAULT"


@dataclass
class FaultReport:
    fault_type: str
    subsystem: str
    severity: float
    confidence: float
    detection_method: str
    timestep: int
    recommended_action: str
    raw_data: Dict = None

    def to_dict(self) -> Dict:
        return {
            "fault_type": self.fault_type,
            "subsystem": self.subsystem,
            "severity": round(self.severity, 4),
            "confidence": round(self.confidence, 4),
            "detection_method": self.detection_method,
            "timestep": self.timestep,
            "recommended_action": self.recommended_action,
        }


class RuleBasedFaultDetector:
    """
    Classical threshold-based fault detector with rate-of-change detection.
    Operates on the 15-dim normalized observation vector.
    Absolute thresholds catch severe faults immediately.
    Rate-of-change detection catches gradual faults early by tracking
    per-step deltas that exceed natural environmental noise.
    """

    THRESHOLDS = {
        "battery_critical":   0.25,   # obs[0] < 0.25
        "bus_voltage_low":    0.65,   # obs[2] < 0.65
        "temperature_max":    0.85,   # obs[8] > 0.85
        "attitude_error_max": 0.75,   # obs[6] > 0.75
        "imu_drift_max":      0.70,   # max(obs[3:6]) > 0.70 (abs)
        "obstacle_critical":  0.10,   # any obs[9:12] < 0.10
    }

    # Rate-of-change thresholds: per-step delta above natural noise floor.
    # Natural rates: battery ~0.001/step drain, thermal ~0.0002/step rise.
    # Fault rates at severity 0.3: battery ~0.009/step, thermal ~0.015/step.
    # Thresholds set between natural and lowest fault rate.
    _ROC_BATTERY_THRESHOLD = 0.004   # per-step decline > natural 0.001
    _ROC_THERMAL_THRESHOLD = 0.003   # per-step rise > natural 0.0002
    _ROC_CONFIRM_STEPS = 3           # must exceed threshold for N consecutive steps

    RECOVERY_MAP = {
        "POWER_CRITICAL": "safe_mode",
        "THERMAL_ANOMALY": "reduce_power",
        "IMU_DRIFT": "recalibrate_imu",
        "SENSOR_FAILURE": "switch_backup_sensor",
        "ACTUATOR_DEGRADATION": "reduce_speed",
        "INTERMITTENT_FAULT": "diagnostic_scan",
    }

    # Camera dropout detection: both obs[12] and obs[13] snap to 0.5 (sentinel)
    _DROPOUT_TOLERANCE = 0.01
    _DROPOUT_WINDOW = 5
    _DROPOUT_MIN_HITS = 2

    # Actuator degradation: proximity sensors (obs[9:12]) drift upward consistently
    _ACTUATOR_ROC_THRESHOLD = 0.002  # per-step increase in mean proximity
    _ACTUATOR_ROC_CONFIRM = 4

    _ROC_BUS_THRESHOLD = 0.003  # per-step decline in bus voltage

    def __init__(self):
        self._prev_battery: float = -1.0
        self._prev_thermal: float = -1.0
        self._prev_bus: float = -1.0
        self._battery_roc_count: int = 0
        self._thermal_roc_count: int = 0
        self._bus_roc_count: int = 0
        self._battery_roc_fired: bool = False
        self._thermal_roc_fired: bool = False
        self._bus_roc_fired: bool = False
        # Camera dropout tracking
        self._dropout_hits: list = []
        self._camera_dropout_fired: bool = False
        # Actuator degradation tracking
        self._prev_prox_mean: float = -1.0
        self._actuator_roc_count: int = 0
        self._actuator_roc_fired: bool = False

    def check(self, obs: np.ndarray, step: int = 0) -> List[FaultReport]:
        """Check thresholds and rate-of-change, return list of detected faults."""
        faults: List[FaultReport] = []

        # EPS checks
        battery = float(obs[0])
        bus_v = float(obs[2])
        if battery < self.THRESHOLDS["battery_critical"]:
            faults.append(FaultReport(
                fault_type="POWER_CRITICAL",
                subsystem="battery",
                severity=1.0 - battery / self.THRESHOLDS["battery_critical"],
                confidence=0.99,
                detection_method="RULE_BASED",
                timestep=step,
                recommended_action="safe_mode",
            ))
        elif not self._battery_roc_fired and self._prev_battery > 0:
            # Rate-of-change: detect anomalous per-step decline
            delta = self._prev_battery - battery
            if delta > self._ROC_BATTERY_THRESHOLD:
                self._battery_roc_count += 1
            else:
                self._battery_roc_count = max(0, self._battery_roc_count - 1)
            if self._battery_roc_count >= self._ROC_CONFIRM_STEPS:
                self._battery_roc_fired = True
                faults.append(FaultReport(
                    fault_type="POWER_CRITICAL",
                    subsystem="battery",
                    severity=min(1.0, delta * 50),
                    confidence=0.82,
                    detection_method="RULE_BASED",
                    timestep=step,
                    recommended_action="safe_mode",
                ))
        self._prev_battery = battery

        if bus_v < self.THRESHOLDS["bus_voltage_low"]:
            faults.append(FaultReport(
                fault_type="POWER_CRITICAL",
                subsystem="bus",
                severity=1.0 - bus_v / self.THRESHOLDS["bus_voltage_low"],
                confidence=0.95,
                detection_method="RULE_BASED",
                timestep=step,
                recommended_action="safe_mode",
            ))
        elif not self._bus_roc_fired and self._prev_bus > 0:
            delta_bus = self._prev_bus - bus_v
            if delta_bus > self._ROC_BUS_THRESHOLD:
                self._bus_roc_count += 1
            else:
                self._bus_roc_count = max(0, self._bus_roc_count - 1)
            if self._bus_roc_count >= self._ROC_CONFIRM_STEPS:
                self._bus_roc_fired = True
                faults.append(FaultReport(
                    fault_type="POWER_CRITICAL",
                    subsystem="bus",
                    severity=min(1.0, delta_bus * 50),
                    confidence=0.80,
                    detection_method="RULE_BASED",
                    timestep=step,
                    recommended_action="safe_mode",
                ))
        self._prev_bus = bus_v

        # ADCS checks: IMU normalized so 0.5=healthy, deviation from 0.5 = drift
        imu_vals = obs[3:6]
        imu_max = float(np.max(np.abs(imu_vals - 0.5)))  # 0 = healthy, 0.5 = max drift
        attitude = float(obs[6])
        # imu_drift_max threshold is for the raw deviation (0.0–0.5 range)
        imu_threshold = 0.25  # deviation > 0.25 from center → fault
        if imu_max > imu_threshold or attitude > self.THRESHOLDS["attitude_error_max"]:
            severity = max(imu_max / 0.5, (attitude - 0.5) / 0.5 if attitude > 0.5 else 0.0)
            faults.append(FaultReport(
                fault_type="IMU_DRIFT",
                subsystem="imu",
                severity=min(1.0, severity),
                confidence=0.90,
                detection_method="RULE_BASED",
                timestep=step,
                recommended_action="recalibrate_imu",
            ))

        # TCS checks
        temp_core = float(obs[8])
        if temp_core > self.THRESHOLDS["temperature_max"]:
            faults.append(FaultReport(
                fault_type="THERMAL_ANOMALY",
                subsystem="thermal",
                severity=(temp_core - self.THRESHOLDS["temperature_max"]) / (1.0 - self.THRESHOLDS["temperature_max"]),
                confidence=0.97,
                detection_method="RULE_BASED",
                timestep=step,
                recommended_action="reduce_power",
            ))
        elif not self._thermal_roc_fired and self._prev_thermal > 0:
            # Rate-of-change: detect anomalous per-step rise
            delta = temp_core - self._prev_thermal
            if delta > self._ROC_THERMAL_THRESHOLD:
                self._thermal_roc_count += 1
            else:
                self._thermal_roc_count = max(0, self._thermal_roc_count - 1)
            if self._thermal_roc_count >= self._ROC_CONFIRM_STEPS:
                self._thermal_roc_fired = True
                faults.append(FaultReport(
                    fault_type="THERMAL_ANOMALY",
                    subsystem="thermal",
                    severity=min(1.0, delta * 30),
                    confidence=0.80,
                    detection_method="RULE_BASED",
                    timestep=step,
                    recommended_action="reduce_power",
                ))
        self._prev_thermal = temp_core

        # Proximity sensor dropout: only if we had close readings then sudden all-clear
        prox = obs[9:12]
        prev = getattr(self, "_prev_prox", None)
        if (prev is not None
                and float(np.min(prev)) < 0.3       # was close to obstacle
                and float(np.all(prox > 0.95))):    # now all clear = possible dropout
            faults.append(FaultReport(
                fault_type="SENSOR_FAILURE",
                subsystem="ultrasonic",
                severity=0.5,
                confidence=0.65,
                detection_method="RULE_BASED",
                timestep=step,
                recommended_action="switch_backup_sensor",
            ))
        self._prev_prox = prox.copy()

        # Camera dropout detection: obs[12] and obs[13] snap to 0.5 (sentinel value)
        # when camera DropoutFault fires. Track pattern over sliding window.
        target_dist = float(obs[12])
        target_bearing = float(obs[13])
        is_dropout = (abs(target_dist - 0.5) < self._DROPOUT_TOLERANCE
                      and abs(target_bearing - 0.5) < self._DROPOUT_TOLERANCE)
        self._dropout_hits.append(1 if is_dropout else 0)
        if len(self._dropout_hits) > self._DROPOUT_WINDOW:
            self._dropout_hits = self._dropout_hits[-self._DROPOUT_WINDOW:]
        if (not self._camera_dropout_fired
                and sum(self._dropout_hits) >= self._DROPOUT_MIN_HITS
                and not any(f.fault_type == "SENSOR_FAILURE" for f in faults)):
            self._camera_dropout_fired = True
            faults.append(FaultReport(
                fault_type="SENSOR_FAILURE",
                subsystem="camera",
                severity=0.6,
                confidence=0.85,
                detection_method="RULE_BASED",
                timestep=step,
                recommended_action="switch_backup_sensor",
            ))

        # Actuator degradation detection: proximity readings (obs[9:12]) decline
        # consistently as degraded motors fail to move the robot away from obstacles.
        prox_mean = float(np.mean(prox))
        if not self._actuator_roc_fired and self._prev_prox_mean > 0:
            delta_prox = self._prev_prox_mean - prox_mean  # positive = declining
            if delta_prox > self._ACTUATOR_ROC_THRESHOLD:
                self._actuator_roc_count += 1
            else:
                self._actuator_roc_count = max(0, self._actuator_roc_count - 1)
            if self._actuator_roc_count >= self._ACTUATOR_ROC_CONFIRM:
                self._actuator_roc_fired = True
                faults.append(FaultReport(
                    fault_type="ACTUATOR_DEGRADATION",
                    subsystem="wheels",
                    severity=min(1.0, delta_prox * 100),
                    confidence=0.78,
                    detection_method="RULE_BASED",
                    timestep=step,
                    recommended_action="reduce_speed",
                ))
        self._prev_prox_mean = prox_mean

        return faults

    def reset(self) -> None:
        """Reset per-episode state (ROC counters, previous values)."""
        self._prev_battery = -1.0
        self._prev_thermal = -1.0
        self._prev_bus = -1.0
        self._battery_roc_count = 0
        self._thermal_roc_count = 0
        self._bus_roc_count = 0
        self._battery_roc_fired = False
        self._thermal_roc_fired = False
        self._bus_roc_fired = False
        self._prev_prox = None
        self._dropout_hits = []
        self._camera_dropout_fired = False
        self._prev_prox_mean = -1.0
        self._actuator_roc_count = 0
        self._actuator_roc_fired = False

    def get_recommended_action(self, fault_type: str) -> str:
        return self.RECOVERY_MAP.get(fault_type, "stop")
