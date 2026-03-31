"""
MemoryAgent: logs all events, maintains experience bank, and updates planning weights.
Implements exponential moving average learning for strategy success rates.
"""
import glob
import json
import os
import time
from typing import Dict, List, Optional

from ..core.message_bus import MessageBus, Message


def cleanup_session_logs(log_dir: str, max_files: int = 50, keep: int = 10) -> int:
    """Delete old session files if count exceeds max_files, keeping the most recent."""
    pattern = os.path.join(log_dir, "session_*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if len(files) <= max_files:
        return 0
    to_delete = files[:-keep]
    for f in to_delete:
        try:
            os.remove(f)
        except OSError:
            pass
    return len(to_delete)


class MemoryAgent:
    """
    Persistent memory + learning agent.
    Tracks fault/recovery outcomes and updates strategy rankings for AdaptationAgent.
    """

    name = "memory_agent"
    EMA_ALPHA = 0.3  # exponential moving average smoothing

    # Conservative priors — real experience overrides quickly via EMA
    _COLD_START_PRIORS = [
        ("POWER_CRITICAL",      "safe_mode",             0.55),
        ("POWER_CRITICAL",      "reduce_speed",          0.50),
        ("THERMAL_ANOMALY",     "safe_mode",             0.55),
        ("THERMAL_ANOMALY",     "reduce_power",          0.52),
        ("IMU_DRIFT",           "recalibrate_imu",       0.55),
        ("IMU_DRIFT",           "reduce_speed",          0.50),
        ("ACTUATOR_DEGRADATION","reduce_speed",          0.55),
        ("SENSOR_FAILURE",      "switch_backup_sensor",  0.55),
        ("INTERMITTENT_FAULT",  "diagnostic_scan",       0.52),
    ]

    # Process-level session ID: all MemoryAgent instances in the same process
    # share one session file instead of creating one per episode.
    _process_session_id: Optional[str] = None

    def __init__(self, bus: MessageBus, session_log_dir: str = "logs"):
        self.bus = bus
        self.log_dir = session_log_dir
        os.makedirs(session_log_dir, exist_ok=True)

        self._experience_bank: List[Dict] = []
        self._strategy_scores: Dict[str, Dict[str, float]] = {}
        self._planning_weights: Dict = {
            "action_success_rate": {},
            "strategy_score": {},
            "fault_recovery_history": {},
        }
        self._event_log: List[Dict] = []
        self._episode_buffer: List[Dict] = []

        # Use a single session ID per process so 100+ episodes share one file
        # Include PID to prevent parallel workers from colliding on the same file
        if MemoryAgent._process_session_id is None:
            MemoryAgent._process_session_id = f"session_{os.getpid()}_{int(time.time())}"
            # Clean up old session files at first init
            deleted = cleanup_session_logs(session_log_dir)
            if deleted:
                import logging
                logging.getLogger(self.name).info(
                    f"Cleaned up {deleted} old session file(s)")
        self._session_id = MemoryAgent._process_session_id

        # Seed strategy scores so AdaptationAgent has reasonable priors from run 1
        self._inject_priors()

        bus.subscribe("FAULT_DETECTED", self._on_fault_detected)
        bus.subscribe("RECOVERY_SUCCESS", self._on_recovery_success)
        bus.subscribe("ACTION_COMPLETE", self._on_action_complete)
        bus.subscribe("STATE_UPDATE", self._on_state_update)

    def _inject_priors(self) -> None:
        """Seed strategy scores with conservative synthetic priors on cold start."""
        for fault_type, strategy, score in self._COLD_START_PRIORS:
            exp = {
                "fault_type": fault_type,
                "strategy_used": strategy,
                "outcome": "SUCCESS",
                "steps_to_recover": 0,
                "sfri_contribution": 0.0,
                "timestamp": time.time(),
                "synthetic": True,
            }
            self._experience_bank.append(exp)
            if fault_type not in self._strategy_scores:
                self._strategy_scores[fault_type] = {}
            # Set directly (bypass EMA) so real experience overrides from first run
            self._strategy_scores[fault_type][strategy] = score
            key = f"{fault_type}_{strategy}"
            self._planning_weights["strategy_score"][key] = score

    def record_episode_end(self, success: bool, steps: int, metrics: Dict) -> None:
        """Called at end of each episode to persist results."""
        entry = {
            "session_id": self._session_id,
            "timestamp": time.time(),
            "success": success,
            "steps": steps,
            **metrics,
            "experience_count": len(self._experience_bank),
        }
        self._save_log(entry)

    def record_experience(self, fault_type: str, strategy: str,
                          outcome: str, steps: int, sfri_delta: float = 0.0) -> None:
        exp = {
            "fault_type": fault_type,
            "strategy_used": strategy,
            "outcome": outcome,
            "steps_to_recover": steps,
            "sfri_contribution": round(sfri_delta, 3),
            "timestamp": time.time(),
        }
        self._experience_bank.append(exp)
        self._update_strategy_score(fault_type, strategy, outcome == "SUCCESS")

    def suggest_strategy(self, fault_type: str) -> List[str]:
        """Return ranked strategy list for a fault type, best first."""
        from ..agents.adaptation_agent import RECOVERY_STRATEGIES
        base = list(RECOVERY_STRATEGIES.get(fault_type, ["safe_mode"]))
        scores = self._strategy_scores.get(fault_type, {})
        if not scores:
            return base
        return sorted(base, key=lambda s: scores.get(s, 0.5), reverse=True)

    def get_planning_weights(self) -> Dict:
        return self._planning_weights

    def log_event(self, event_type: str, data: Dict) -> None:
        self._event_log.append({
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
        })

    def _update_strategy_score(self, fault_type: str, strategy: str, success: bool) -> None:
        if fault_type not in self._strategy_scores:
            self._strategy_scores[fault_type] = {}
        current = self._strategy_scores[fault_type].get(strategy, 0.5)
        new_score = self.EMA_ALPHA * float(success) + (1 - self.EMA_ALPHA) * current
        self._strategy_scores[fault_type][strategy] = round(new_score, 4)
        # Update planning weights
        key = f"{fault_type}_{strategy}"
        self._planning_weights["strategy_score"][key] = new_score
        self._planning_weights["fault_recovery_history"][fault_type] = (
            self._planning_weights["fault_recovery_history"].get(fault_type, 0) + (1 if success else 0)
        )

    def _on_fault_detected(self, msg: Message) -> None:
        self._event_log.append({"type": "FAULT_DETECTED", "data": msg.data,
                                 "step": msg.timestamp})

    def _on_recovery_success(self, msg: Message) -> None:
        self._event_log.append({"type": "RECOVERY_SUCCESS", "data": msg.data})

    def _on_action_complete(self, msg: Message) -> None:
        action = msg.data.get("action", "")
        success = msg.data.get("success", True)
        current = self._planning_weights["action_success_rate"].get(action, 0.5)
        updated = self.EMA_ALPHA * float(success) + (1 - self.EMA_ALPHA) * current
        self._planning_weights["action_success_rate"][action] = round(updated, 4)

    def _on_state_update(self, msg: Message) -> None:
        pass  # future: encode state for episodic memory

    def _save_log(self, entry: Dict) -> None:
        path = os.path.join(self.log_dir, f"{self._session_id}.json")
        existing = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = []  # corrupted file — start fresh
        existing.append(entry)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)

    def reset(self) -> None:
        self._event_log.clear()
        self._experience_bank.clear()
        # Keep process-level session ID — don't create a new file per reset
