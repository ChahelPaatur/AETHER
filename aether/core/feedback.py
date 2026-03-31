"""
Feedback Loop: evaluates current state vs goal after each step.
Determines success/failure and signals when re-planning is needed.
"""
from typing import Dict, List, Optional
import math
import logging

logger = logging.getLogger(__name__)


class FeedbackEvaluator:
    """
    Evaluates task progress and determines whether to continue, re-plan, or stop.
    """

    def __init__(self):
        self.total_steps = 0
        self.success_steps = 0
        self.failure_steps = 0
        self.replans = 0

    def evaluate(self, goal: Dict, state: Dict, faults: List) -> Dict:
        """
        Returns an evaluation dict with:
          - status: "success" | "failure" | "in_progress" | "replan"
          - reason: explanation
          - progress: 0.0–1.0
        """
        self.total_steps += 1
        task = goal.get("task", "navigate")

        # Task-specific success conditions
        if task in ("follow", "navigate"):
            if state.get("at_target"):
                self.success_steps += 1
                return {"status": "success", "reason": "reached_target", "progress": 1.0}

        if task == "scan":
            # Scan tasks succeed after a few scan actions
            if state.get("step", 0) > 5:
                self.success_steps += 1
                return {"status": "success", "reason": "scan_complete", "progress": 1.0}

        if task == "stop":
            self.success_steps += 1
            return {"status": "success", "reason": "stopped", "progress": 1.0}

        # Critical faults = failure
        if any(f.severity == "critical" for f in faults):
            self.failure_steps += 1
            return {
                "status": "failure",
                "reason": faults[0].description if faults else "critical_fault",
                "progress": self._estimate_progress(goal, state),
            }

        # Non-critical faults = trigger replan
        if faults:
            self.replans += 1
            return {
                "status": "replan",
                "reason": "; ".join(f.description for f in faults),
                "progress": self._estimate_progress(goal, state),
                "fault_actions": [f.suggested_action for f in faults],
            }

        return {
            "status": "in_progress",
            "reason": "executing",
            "progress": self._estimate_progress(goal, state),
        }

    def _estimate_progress(self, goal: Dict, state: Dict) -> float:
        target_info = state.get("target_info")
        if target_info is None:
            return 0.0
        ax, ay = state.get("agent_pos", (0, 0))
        tx, ty = target_info["pos"]
        current_dist = math.sqrt((ax - tx)**2 + (ay - ty)**2)
        # Rough estimate assuming max distance is ~28 (diagonal of 20x20 grid)
        max_dist = 28.0
        return max(0.0, min(1.0, 1.0 - current_dist / max_dist))

    def stats(self) -> Dict:
        return {
            "total_steps": self.total_steps,
            "success_steps": self.success_steps,
            "failure_steps": self.failure_steps,
            "replans": self.replans,
        }

    def reset(self) -> None:
        self.total_steps = 0
        self.success_steps = 0
        self.failure_steps = 0
        self.replans = 0
