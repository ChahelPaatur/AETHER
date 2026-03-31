"""
MetricsTracker: tracks all FDIR and mission performance metrics.
Implements SFRI (Stability Fault Recovery Index) from the research framework.
"""
from typing import Dict, List, Optional
import math


class MetricsTracker:
    """
    Tracks MTTD, MTTR, detection rate, recovery rate, SFRI, and task metrics
    across a single episode or aggregated across many runs.
    """

    def __init__(self, max_steps: int = 200):
        self.max_steps = max_steps
        self._reset_episode()

    def _reset_episode(self) -> None:
        self.faults_injected: int = 0
        self.faults_detected: int = 0
        self.faults_recovered: int = 0
        self.false_positives: int = 0
        self.total_observations: int = 0

        self._detection_times: List[int] = []   # steps from injection to detection
        self._recovery_times: List[int] = []    # steps from detection to recovery
        self._adaptation_latencies: List[int] = []

        self.steps_to_completion: int = 0
        self.episode_reward: float = 0.0
        self.success: bool = False

        self._fault_log: List[Dict] = []
        self._active_faults: Dict[str, Dict] = {}  # subsystem → {injected_step, detected_step}

    def record_fault_injected(self, fault_type: str, subsystem: str, step: int) -> None:
        self.faults_injected += 1
        key = f"{fault_type}_{subsystem}_{step}"
        self._active_faults[key] = {
            "fault_type": fault_type,
            "subsystem": subsystem,
            "injected_step": step,
            "detected_step": None,
            "recovered_step": None,
        }

    def record_fault_detected(self, fault_type: str, subsystem: str, step: int,
                               method: str = "DRL") -> None:
        # Match to earliest open (undetected) injection of this type.
        # Repeat detections of an already-detected fault are silently ignored
        # (not counted as false positives) since the fault is still active.
        any_match = False
        for key, info in self._active_faults.items():
            if info["fault_type"] == fault_type:
                any_match = True
                if info["detected_step"] is None:
                    info["detected_step"] = step
                    delay = step - info["injected_step"]
                    self._detection_times.append(max(0, delay))
                    self.faults_detected += 1
                    return
        if not any_match:
            # No injection of this type at all → true false positive
            self.false_positives += 1

    def record_fault_recovered(self, fault_type: str, subsystem: str, step: int,
                                latency: int = 0) -> None:
        self._adaptation_latencies.append(latency)
        for key, info in self._active_faults.items():
            if (info["fault_type"] == fault_type
                    and info["detected_step"] is not None
                    and info["recovered_step"] is None):
                info["recovered_step"] = step
                self.faults_recovered += 1
                if info["detected_step"] is not None:
                    self._recovery_times.append(step - info["detected_step"])
                self._fault_log.append({
                    "fault_type": fault_type,
                    "subsystem": subsystem,
                    "recovered_step": step,
                    "latency": latency,
                })
                break

    def record_false_positive(self) -> None:
        self.false_positives += 1

    def record_step(self, reward: float) -> None:
        self.total_observations += 1
        self.episode_reward += reward

    def record_episode_end(self, success: bool, steps: int) -> None:
        self.success = success
        self.steps_to_completion = steps

    @property
    def MTTD(self) -> float:
        """Mean Time To Detect (steps)."""
        if not self._detection_times:
            return 0.0
        return sum(self._detection_times) / len(self._detection_times)

    @property
    def MTTR(self) -> float:
        """Mean Time To Recover (steps)."""
        if not self._recovery_times:
            return 0.0
        return sum(self._recovery_times) / len(self._recovery_times)

    @property
    def detection_rate(self) -> float:
        if self.faults_injected == 0:
            return 1.0
        return min(1.0, self.faults_detected / self.faults_injected)

    @property
    def recovery_rate(self) -> float:
        if self.faults_detected == 0:
            return 1.0
        return min(1.0, self.faults_recovered / self.faults_detected)

    @property
    def false_positive_rate(self) -> float:
        if self.total_observations == 0:
            return 0.0
        return self.false_positives / self.total_observations

    @property
    def adaptation_latency(self) -> float:
        if not self._adaptation_latencies:
            return 0.0
        return sum(self._adaptation_latencies) / len(self._adaptation_latencies)

    def compute_sfri(self) -> float:
        """
        Stability Fault Recovery Index — composite performance score.
        SFRI = 35*DR + 25*(1 - MTTR/max_steps) + 10*stability - 30*FPR
        Range: theoretically 0–70 (lower bound limited by false positive penalty).
        """
        dr = self.detection_rate * 100
        recovery_speed = (1.0 - min(1.0, self.MTTR / self.max_steps)) * 100
        stability = self.recovery_rate * 100
        fpr = self.false_positive_rate * 100
        return 35 * (dr / 100) + 25 * (recovery_speed / 100) + 10 * (stability / 100) - 30 * (fpr / 100)

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "steps_to_completion": self.steps_to_completion,
            "episode_reward": round(self.episode_reward, 3),
            "faults_injected": self.faults_injected,
            "faults_detected": self.faults_detected,
            "faults_recovered": self.faults_recovered,
            "false_positives": self.false_positives,
            "detection_rate": round(self.detection_rate, 4),
            "recovery_rate": round(self.recovery_rate, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "MTTD": round(self.MTTD, 2),
            "MTTR": round(self.MTTR, 2),
            "SFRI": round(self.compute_sfri(), 2),
            "adaptation_latency": round(self.adaptation_latency, 2),
        }

    def to_csv_row(self) -> List:
        d = self.to_dict()
        return [d[k] for k in sorted(d.keys())]

    @staticmethod
    def aggregate(runs: List[Dict]) -> Dict:
        """Compute mean ± std across multiple run dicts."""
        if not runs:
            return {}
        keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
        agg = {}
        for k in keys:
            vals = [r[k] for r in runs if k in r]
            n = len(vals)
            if n == 0:
                continue
            mean = sum(vals) / n
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
            agg[k] = {"mean": round(mean, 3), "std": round(std, 3)}
        return agg
