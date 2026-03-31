"""
Memory Module: stores past failures and successful strategies for adaptation.
Simple key-value store with JSON persistence, designed for future vector/episodic extension.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional


class Memory:
    """
    Persistent memory for AETHER agent.
    Stores: past failures, successful strategies, task outcomes.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        self._store: Dict[str, List[Dict]] = {
            "failures": [],
            "successes": [],
            "strategies": {},
            "task_outcomes": [],
        }
        if persist_path and os.path.exists(persist_path):
            self._load()

    def record_failure(self, task: str, state: Dict, faults: List, step: int) -> None:
        entry = {
            "timestamp": time.time(),
            "task": task,
            "step": step,
            "faults": [{"type": f.fault_type.value, "description": f.description} for f in faults],
            "state_snapshot": {
                "agent_pos": state.get("agent_pos"),
                "failed_sensors": state.get("failed_sensors", []),
                "failed_actuators": state.get("failed_actuators", []),
            },
        }
        self._store["failures"].append(entry)
        self._maybe_save()

    def record_success(self, task: str, steps_taken: int, strategy: str) -> None:
        entry = {
            "timestamp": time.time(),
            "task": task,
            "steps_taken": steps_taken,
            "strategy": strategy,
        }
        self._store["successes"].append(entry)
        # Update strategy effectiveness
        if task not in self._store["strategies"]:
            self._store["strategies"][task] = {}
        strat_data = self._store["strategies"][task]
        strat_data[strategy] = strat_data.get(strategy, 0) + 1
        self._maybe_save()

    def record_outcome(self, task: str, scenario: str, status: str, steps: int, faults_encountered: int) -> None:
        self._store["task_outcomes"].append({
            "timestamp": time.time(),
            "task": task,
            "scenario": scenario,
            "status": status,
            "steps": steps,
            "faults_encountered": faults_encountered,
        })
        self._maybe_save()

    def best_strategy(self, task: str) -> Optional[str]:
        strats = self._store["strategies"].get(task, {})
        if not strats:
            return None
        return max(strats, key=strats.get)

    def recent_failures(self, task: Optional[str] = None, limit: int = 5) -> List[Dict]:
        failures = self._store["failures"]
        if task:
            failures = [f for f in failures if f["task"] == task]
        return failures[-limit:]

    def failure_count(self, task: Optional[str] = None) -> int:
        if task:
            return sum(1 for f in self._store["failures"] if f["task"] == task)
        return len(self._store["failures"])

    def success_rate(self, task: Optional[str] = None) -> float:
        outcomes = self._store["task_outcomes"]
        if task:
            outcomes = [o for o in outcomes if o["task"] == task]
        if not outcomes:
            return 0.0
        successes = sum(1 for o in outcomes if o["status"] == "success")
        return successes / len(outcomes)

    def summary(self) -> Dict:
        return {
            "total_failures": len(self._store["failures"]),
            "total_successes": len(self._store["successes"]),
            "total_outcomes": len(self._store["task_outcomes"]),
            "known_strategies": list(self._store["strategies"].keys()),
            "overall_success_rate": self.success_rate(),
        }

    def clear(self) -> None:
        self._store = {
            "failures": [],
            "successes": [],
            "strategies": {},
            "task_outcomes": [],
        }

    def _maybe_save(self) -> None:
        if self.persist_path:
            self._save()

    def _save(self) -> None:
        with open(self.persist_path, "w") as f:
            json.dump(self._store, f, indent=2)

    def _load(self) -> None:
        with open(self.persist_path, "r") as f:
            data = json.load(f)
        self._store.update(data)
