"""
FaultAgent: DRL-First Hybrid FDIR — the permanent fault detection backbone.
Pipeline: PPO Network → Predictive Analytics → Temporal Validation → Confidence Arbitration → Rule Safety Backup.
Monitors telemetry from all domain agents (CameraAgent, MovementAgent, PowerAgent,
ThermalAgent, NavigationAgent) simultaneously and can pause or redirect any domain
agent when a fault is detected in its subsystem.

Self-bootstrapping: auto-trains from scratch on first launch using the rule-based
detector as a teacher, then continuously updates via online learning across episodes.
"""
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..faults.fault_detector import RuleBasedFaultDetector, FaultReport
from ..core.message_bus import MessageBus, Message

OBS_DIM = 15
FAULT_CLASSES = [
    "NO_FAULT", "SENSOR_FAILURE", "ACTUATOR_DEGRADATION",
    "POWER_CRITICAL", "THERMAL_ANOMALY", "IMU_DRIFT",
    "INTERMITTENT_FAULT", "SAFE_MODE",
]
N_CLASSES = len(FAULT_CLASSES)
DRL_CONFIDENCE_THRESHOLD = 0.12
CRITICAL_SEVERITY_BYPASS  = 0.80

# Resolved at import time relative to project root
_PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_WEIGHTS_PATH = os.path.join(_PROJECT_ROOT, "weights", "fault_agent.npy")


# ------------------------------------------------------------------ #
#  PPO Network (numpy-only, 2-layer tanh feedforward)                 #
# ------------------------------------------------------------------ #

class PPONetwork:
    """
    Lightweight 2-layer feedforward network.
    Input: 15-dim normalized observation.
    Output: fault class probabilities + confidence score.
    Supports numpy-based backprop via train_step().
    """

    def __init__(self, input_dim: int = OBS_DIM, hidden_dim: int = 64,
                 output_dim: int = N_CLASSES, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = lambda d_in: np.sqrt(2.0 / d_in)
        self.W1 = rng.normal(0, scale(input_dim),  (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, scale(hidden_dim), (hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.W3 = rng.normal(0, scale(hidden_dim), (hidden_dim, output_dim))
        self.b3 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Returns (class_probabilities, confidence)."""
        h1 = np.tanh(x @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        probs = self._softmax(logits)
        return probs, float(np.max(probs))

    def train_step(self, x: np.ndarray, label_idx: int, lr: float = 0.001) -> float:
        """
        Single SGD step with cross-entropy loss and gradient clipping.
        Returns the scalar loss value.
        """
        x = x.astype(float)
        # Forward — cache intermediates
        h1_pre = x @ self.W1 + self.b1
        h1     = np.tanh(h1_pre)
        h2_pre = h1 @ self.W2 + self.b2
        h2     = np.tanh(h2_pre)
        logits = h2 @ self.W3 + self.b3
        probs  = self._softmax(logits)

        loss = -float(np.log(probs[label_idx] + 1e-8))

        # Backprop — softmax + cross-entropy combined gradient
        d3 = probs.copy();  d3[label_idx] -= 1.0

        dW3    = np.outer(h2, d3)
        db3    = d3.copy()
        dh2    = self.W3 @ d3
        dh2_p  = dh2 * (1.0 - h2 ** 2)

        dW2    = np.outer(h1, dh2_p)
        db2    = dh2_p.copy()
        dh1    = self.W2 @ dh2_p
        dh1_p  = dh1 * (1.0 - h1 ** 2)

        dW1    = np.outer(x, dh1_p)
        db1    = dh1_p.copy()

        # Gradient clipping (prevent explosion)
        clip = 1.0
        self.W3 -= lr * np.clip(dW3, -clip, clip)
        self.b3 -= lr * np.clip(db3, -clip, clip)
        self.W2 -= lr * np.clip(dW2, -clip, clip)
        self.b2 -= lr * np.clip(db2, -clip, clip)
        self.W1 -= lr * np.clip(dW1, -clip, clip)
        self.b1 -= lr * np.clip(db1, -clip, clip)
        return loss

    def load_weights(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path, allow_pickle=True).item()
            self.W1 = data["W1"]; self.b1 = data["b1"]
            self.W2 = data["W2"]; self.b2 = data["b2"]
            self.W3 = data["W3"]; self.b3 = data["b3"]
            return True
        except Exception:
            return False

    def save_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, {"W1": self.W1, "b1": self.b1,
                       "W2": self.W2, "b2": self.b2,
                       "W3": self.W3, "b3": self.b3})

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()


# ------------------------------------------------------------------ #
#  Predictive Analytics                                                #
# ------------------------------------------------------------------ #

class PredictiveAnalyticsModule:
    WINDOW = 5
    SUBSYSTEM_INDICES = {
        "eps": [0, 1, 2], "adcs": [3, 4, 5, 6],
        "tcs": [7, 8], "proximity": [9, 10, 11], "target": [12, 13],
    }

    def __init__(self):
        self._window: deque = deque(maxlen=self.WINDOW)

    def update(self, obs: np.ndarray) -> Dict[str, float]:
        self._window.append(obs.copy())
        if len(self._window) < 3:
            return {k: 0.0 for k in self.SUBSYSTEM_INDICES}
        history = np.array(self._window)
        indicators = {}
        for subsys, indices in self.SUBSYSTEM_INDICES.items():
            sub      = history[:, indices]
            variance = float(np.mean(np.var(sub, axis=0)))
            rate     = float(np.mean(np.abs(np.diff(sub, axis=0)))) if len(sub) > 1 else 0.0
            indicators[subsys] = float(np.clip(variance * 5.0 + rate * 3.0, 0.0, 1.0))
        return indicators

    def overall_indicator(self) -> float:
        if len(self._window) < 2:
            return 0.0
        history = np.array(self._window)
        return float(np.clip(np.mean(np.abs(np.diff(history, axis=0))) * 10.0, 0.0, 1.0))

    def reset(self) -> None:
        self._window.clear()


# ------------------------------------------------------------------ #
#  Temporal Validation                                                 #
# ------------------------------------------------------------------ #

class TemporalValidationFramework:
    def __init__(self):
        self._scores: Dict[str, List[float]] = {}
        self._confirmed: Dict[str, bool] = {}

    def update(self, fault_type: str, severity: float, drl_conf: float,
               predictive: float) -> Tuple[float, bool]:
        score = 0.4 * severity + 0.35 * drl_conf + 0.25 * predictive
        if fault_type not in self._scores:
            self._scores[fault_type] = []
        self._scores[fault_type] = (self._scores[fault_type] + [score])[-3:]
        if severity >= CRITICAL_SEVERITY_BYPASS:
            self._confirmed[fault_type] = True
            return score, True
        window = self._scores[fault_type]
        confirmed = len(window) >= 2 and sum(window) / len(window) > 0.35
        self._confirmed[fault_type] = confirmed
        return score, confirmed

    def reset_fault(self, fault_type: str) -> None:
        self._scores.pop(fault_type, None)
        self._confirmed.pop(fault_type, None)

    def reset(self) -> None:
        self._scores.clear()
        self._confirmed.clear()


# ------------------------------------------------------------------ #
#  FaultAgent                                                          #
# ------------------------------------------------------------------ #

class FaultAgent:
    """
    DRL-First Hybrid FDIR agent with self-bootstrapping and online learning.
    On first launch: trains from scratch using rule-based detector as teacher.
    Continuously updates weights from a replay buffer after every episode.
    """

    name = "fault_agent"
    _BOOTSTRAP_EPISODES = 200
    _BOOTSTRAP_STEPS    = 80     # steps per bootstrap episode
    _BOOTSTRAP_LR       = 0.005
    _ONLINE_LR          = 0.001
    _BATCH_SIZE         = 32
    _REPLAY_MAXLEN      = 1000
    _SAVE_EVERY         = 10     # save weights every N episodes

    def __init__(self, bus: MessageBus, weights_path: Optional[str] = None,
                 seed: int = 42, no_learning: bool = False):
        self.bus = bus
        self.network   = PPONetwork(seed=seed)
        self.predictive = PredictiveAnalyticsModule()
        self.temporal   = TemporalValidationFramework()
        self.rule_detector = RuleBasedFaultDetector()

        self._step = 0
        self._last_method = "DRL"
        self._detection_log: List[Dict] = []
        self._no_learning   = no_learning
        self._episode_count = 0
        self._replay_buffer: deque = deque(maxlen=self._REPLAY_MAXLEN)
        self._step_obs_buffer: List[Tuple[np.ndarray, int]] = []

        self._weights_path = weights_path or DEFAULT_WEIGHTS_PATH

        # Auto-bootstrap: load weights or train from scratch
        if not self.network.load_weights(self._weights_path):
            if not no_learning:
                self._bootstrap_train()
            # else: proceed with random weights for controlled experiments

        self._domain_state: dict = {}
        # Predictive suppression: tracks PREDICTIVE occurrences per fault type
        self._predictive_history: Dict[str, List[int]] = {}
        self._predictive_suppressed_until: Dict[str, int] = {}
        bus.subscribe("POWER_STATUS",      self._on_domain_update)
        bus.subscribe("THERMAL_STATUS",    self._on_domain_update)
        bus.subscribe("NAVIGATION_UPDATE", self._on_domain_update)
        bus.subscribe("CAMERA_UPDATE",     self._on_domain_update)
        bus.subscribe("MOVEMENT_STATUS",   self._on_domain_update)

    # ---------------------------------------------------------------- #
    #  Bootstrap training                                               #
    # ---------------------------------------------------------------- #

    def _bootstrap_train(self) -> None:
        """
        Self-supervised training using the rule-based detector as teacher.
        Runs _BOOTSTRAP_EPISODES episodes on fault_heavy scenario, collects
        (obs, label) pairs, and trains the PPO network via SGD.
        """
        # Lazy imports to avoid circular dependency
        from ..simulation.environment import SimulationEnvironment
        from ..faults.fault_injector import FaultInjector
        from ..simulation.scenarios import get_scenario

        print(f"[FaultAgent] No trained weights found. Starting bootstrap training "
              f"({self._BOOTSTRAP_EPISODES} episodes)...")

        env      = SimulationEnvironment(seed=0)
        injector = FaultInjector(fault_probability=0.05, seed=0)
        scenario = get_scenario("fault_heavy") or get_scenario("compound")

        bootstrap_buf: List[Tuple[np.ndarray, int]] = []
        loss_history: List[float] = []

        for ep in range(1, self._BOOTSTRAP_EPISODES + 1):
            raw_obs = env.reset(scenario)
            injector.reset()
            if scenario:
                injector.load_from_scenario(scenario)

            for step in range(self._BOOTSTRAP_STEPS):
                obs_c = injector.tick(raw_obs, step)
                label_idx = self._rule_label(obs_c, step)
                bootstrap_buf.append((obs_c.copy(), label_idx))

                # Advance environment (single-step, no planning)
                raw_obs, _, done, _ = env.step("follow_target")
                if done:
                    break

            # Mini-batch gradient update after each episode
            if len(bootstrap_buf) >= self._BATCH_SIZE:
                idxs  = np.random.default_rng(ep).choice(
                    len(bootstrap_buf), self._BATCH_SIZE, replace=False)
                ep_loss = 0.0
                for idx in idxs:
                    o, l = bootstrap_buf[idx]
                    ep_loss += self.network.train_step(o, l, lr=self._BOOTSTRAP_LR)
                loss_history.append(ep_loss / self._BATCH_SIZE)

            if ep % 20 == 0:
                recent_loss = (sum(loss_history[-5:]) / max(1, len(loss_history[-5:]))
                               if loss_history else 0.0)
                print(f"[FaultAgent] Bootstrap training... episode {ep:3d}/{self._BOOTSTRAP_EPISODES}"
                      f"  avg_loss={recent_loss:.4f}")

        # Save weights
        self.network.save_weights(self._weights_path)

        # Confidence calibration score
        calibration = self._calibrate(bootstrap_buf[-200:] if len(bootstrap_buf) >= 200
                                       else bootstrap_buf)
        random_baseline = 100.0 / N_CLASSES
        print(f"[FaultAgent] Bootstrap complete. "
              f"DRL confidence on teacher labels: {calibration:.1f}% "
              f"(random baseline: {random_baseline:.1f}%)")

    def _rule_label(self, obs: np.ndarray, step: int) -> int:
        """Highest-severity rule-based fault as integer class index."""
        faults = self.rule_detector.check(obs, step)
        if not faults:
            return 0  # NO_FAULT
        top = max(faults, key=lambda f: f.severity)
        try:
            return FAULT_CLASSES.index(top.fault_type)
        except ValueError:
            return 0

    def _calibrate(self, samples: List[Tuple[np.ndarray, int]]) -> float:
        """Mean probability the DRL assigns to the teacher's chosen class."""
        if not samples:
            return 0.0
        probs_on_label = []
        for obs, label_idx in samples:
            p, _ = self.network.forward(obs)
            probs_on_label.append(float(p[label_idx]))
        return float(np.mean(probs_on_label)) * 100.0

    # ---------------------------------------------------------------- #
    #  Online learning                                                   #
    # ---------------------------------------------------------------- #

    def finish_episode(self) -> None:
        """
        Called by TaskManager at end of each episode.
        Flushes step buffer to replay, runs one gradient update,
        saves weights every _SAVE_EVERY episodes.
        """
        if self._no_learning:
            self._step_obs_buffer.clear()
            return

        # Flush step buffer into replay buffer
        for entry in self._step_obs_buffer:
            self._replay_buffer.append(entry)
        self._step_obs_buffer.clear()

        self._episode_count += 1

        # Online gradient update
        if len(self._replay_buffer) >= self._BATCH_SIZE:
            self._online_update()

        # Periodic weight save
        if self._episode_count % self._SAVE_EVERY == 0:
            self.network.save_weights(self._weights_path)

    def _online_update(self) -> None:
        """Sample _BATCH_SIZE entries from replay buffer and run one SGD pass."""
        buf   = list(self._replay_buffer)
        idxs  = np.random.default_rng(self._episode_count).choice(
            len(buf), self._BATCH_SIZE, replace=False)
        for idx in idxs:
            obs, label_idx = buf[idx]
            self.network.train_step(obs, label_idx, lr=self._ONLINE_LR)

    # ---------------------------------------------------------------- #
    #  Inference pipeline                                               #
    # ---------------------------------------------------------------- #

    def process(self, obs: np.ndarray, step: int) -> List[FaultReport]:
        """Full FDIR pipeline. Returns list of confirmed fault reports."""
        self._step = step
        reports: List[FaultReport] = []

        # Stage 1: PPO Network
        probs, drl_confidence = self.network.forward(obs)
        fault_idx   = int(np.argmax(probs))
        fault_class = FAULT_CLASSES[fault_idx]
        drl_severity = float(probs[fault_idx])

        # Stage 2: Predictive analytics
        predictive_indicators = self.predictive.update(obs)
        predictive_score      = self.predictive.overall_indicator()

        # Stage 3: Temporal validation
        if fault_class != "NO_FAULT":
            temporal_score, confirmed = self.temporal.update(
                fault_class, drl_severity, drl_confidence, predictive_score)
        else:
            temporal_score, confirmed = 0.0, False

        # Stage 4: Confidence arbitration
        method = "DRL" if drl_confidence >= DRL_CONFIDENCE_THRESHOLD else "RULE_BASED"
        if method == "DRL" and confirmed and fault_class not in ("NO_FAULT", "SAFE_MODE"):
            reports.append(self._make_report(
                fault_class, drl_severity, drl_confidence, temporal_score, method, step))

        # Stage 5: Rule-based safety backup (always runs)
        rule_reports = self.rule_detector.check(obs, step)
        for rr in rule_reports:
            rr.detection_method = "RULE_BASED"
            if not any(r.fault_type == rr.fault_type for r in reports):
                reports.append(rr)

        # Stage 6: Predictive early-warning (with repetition suppression)
        for subsys, indicator in predictive_indicators.items():
            if indicator > 0.65:
                ft = self._subsys_to_fault_type(subsys)
                if not any(r.fault_type == ft for r in reports):
                    # Check if this fault type is currently suppressed
                    if ft in self._predictive_suppressed_until and step < self._predictive_suppressed_until[ft]:
                        continue

                    # Track predictive occurrence
                    if ft not in self._predictive_history:
                        self._predictive_history[ft] = []
                    self._predictive_history[ft].append(step)
                    # Keep only last 20 steps
                    self._predictive_history[ft] = [
                        s for s in self._predictive_history[ft] if step - s <= 20
                    ]

                    # Suppress if >5 PREDICTIVE in 20 steps without RULE_BASED confirmation
                    has_rule = any(
                        r.fault_type == ft and r.detection_method == "RULE_BASED"
                        for r in reports
                    )
                    if len(self._predictive_history[ft]) > 5 and not has_rule:
                        self._predictive_suppressed_until[ft] = step + 15
                        continue

                    reports.append(FaultReport(
                        fault_type=ft, subsystem=subsys,
                        severity=indicator * 0.6, confidence=indicator * 0.7,
                        detection_method="PREDICTIVE", timestep=step,
                        recommended_action=self.rule_detector.get_recommended_action(ft),
                    ))

        # Record (obs, rule-label) for online learning
        if not self._no_learning:
            rule_label = self._rule_label(obs, step)
            self._step_obs_buffer.append((obs.copy(), rule_label))

        self._last_method = method
        for r in reports:
            self._detection_log.append(r.to_dict())
            self.bus.publish("FAULT_DETECTED", self.name, r.to_dict(), priority="HIGH")

        return reports

    def _on_domain_update(self, msg: Message) -> None:
        self._domain_state[msg.type] = msg.data

    def _make_report(self, fault_class, severity, confidence,
                     temporal_score, method, step) -> FaultReport:
        subsystem_map = {
            "POWER_CRITICAL": "battery", "THERMAL_ANOMALY": "thermal",
            "IMU_DRIFT": "imu", "SENSOR_FAILURE": "sensor",
            "ACTUATOR_DEGRADATION": "wheels", "INTERMITTENT_FAULT": "imu",
            "SAFE_MODE": "system",
        }
        action_map = {
            "POWER_CRITICAL": "safe_mode", "THERMAL_ANOMALY": "reduce_power",
            "IMU_DRIFT": "recalibrate_imu", "SENSOR_FAILURE": "switch_backup_sensor",
            "ACTUATOR_DEGRADATION": "reduce_speed", "INTERMITTENT_FAULT": "diagnostic_scan",
            "SAFE_MODE": "safe_mode",
        }
        return FaultReport(
            fault_type=fault_class,
            subsystem=subsystem_map.get(fault_class, "unknown"),
            severity=round(severity, 4), confidence=round(confidence, 4),
            detection_method=method, timestep=step,
            recommended_action=action_map.get(fault_class, "stop"),
        )

    def _subsys_to_fault_type(self, subsys: str) -> str:
        return {"eps": "POWER_CRITICAL", "adcs": "IMU_DRIFT", "tcs": "THERMAL_ANOMALY",
                "proximity": "SENSOR_FAILURE", "target": "SENSOR_FAILURE"
                }.get(subsys, "SENSOR_FAILURE")

    def reset(self) -> None:
        self.predictive.reset()
        self.temporal.reset()
        self.rule_detector.reset()
        self._detection_log.clear()
        self._step_obs_buffer.clear()
        self._predictive_history.clear()
        self._predictive_suppressed_until.clear()

    @property
    def last_method(self) -> str:
        return self._last_method
