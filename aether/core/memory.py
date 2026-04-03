"""
Memory Module: stores past failures and successful strategies for adaptation.

Includes:
  - Memory: simulation FDIR memory (failures, strategies, task outcomes)
  - PersistentMemory: cross-session agent memory stored in ~/.aether/memory.json
"""
import json
import os
import platform
import re
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional


# ── Default persistent memory path ────────────────────────────────────
_DEFAULT_MEMORY_DIR = os.path.join(os.path.expanduser("~"), ".aether")
_DEFAULT_MEMORY_PATH = os.path.join(_DEFAULT_MEMORY_DIR, "memory.json")
_MAX_ENTRIES = 100


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


# ── PersistentMemory: cross-session agent memory ─────────────────────

class PersistentMemory:
    """Cross-session agent memory stored in ``~/.aether/memory.json``.

    Each entry records: timestamp, objective, outcome, tool_chain,
    faults, duration, platform, and a session_id.

    On startup, loads all past entries and prints a summary.
    Caps storage at *_MAX_ENTRIES* (oldest entries are evicted).
    """

    def __init__(self, path: str = _DEFAULT_MEMORY_PATH):
        self._path = path
        self._entries: List[Dict] = []
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._platform = platform.system()
        self._load()

    # ── Public API ─────────────────────────────────────────────────

    def record(self, objective: str, outcome: str, tool_chain: List[str],
               faults: int = 0, duration: float = 0.0) -> None:
        """Record a completed objective execution."""
        entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "session_id": self._session_id,
            "objective": objective,
            "outcome": outcome,
            "tool_chain": tool_chain,
            "faults": faults,
            "duration_s": round(duration, 2),
            "platform": self._platform,
        }
        self._entries.append(entry)
        # Evict oldest entries if over limit
        if len(self._entries) > _MAX_ENTRIES:
            self._entries = self._entries[-_MAX_ENTRIES:]
        self._save()

    def recent(self, n: int = 5) -> List[Dict]:
        """Return the last *n* entries."""
        return self._entries[-n:] if len(self._entries) > n else list(self._entries)

    def search(self, objective: str, limit: int = 3) -> List[Dict]:
        """Find past entries with similar objectives via keyword matching.

        Splits the objective into keywords and scores each entry by how
        many keywords appear in its objective text.  Returns the top
        *limit* matches sorted by relevance (highest first).
        """
        keywords = _extract_keywords(objective)
        if not keywords:
            return []

        scored = []
        for entry in self._entries:
            past_obj = entry.get("objective", "").lower()
            score = sum(1 for kw in keywords if kw in past_obj)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: (-x[0], -x[1].get("timestamp", 0)))
        return [entry for _, entry in scored[:limit]]

    def planning_hints(self, objective: str) -> Optional[str]:
        """Generate planning hints from past experience for an objective.

        Returns a string to prepend to the LLM planner context, or None
        if no relevant past experience exists.
        """
        similar = self.search(objective, limit=5)
        if not similar:
            return None

        successes = [e for e in similar if e["outcome"] == "SUCCESS"]
        failures = [e for e in similar if e["outcome"] != "SUCCESS"]

        parts = []
        if successes:
            best = successes[0]
            chain = " -> ".join(best["tool_chain"])
            parts.append(
                f"Similar past objective succeeded: \"{best['objective']}\"\n"
                f"  Tool chain used: {chain}")

        if failures:
            failed_chains = set()
            for f in failures:
                failed_chains.add(" -> ".join(f["tool_chain"]))
            parts.append(
                f"Previous failures ({len(failures)}): "
                f"avoid chains: {'; '.join(failed_chains)}")

        return "\n".join(parts) if parts else None

    def format_summary(self) -> str:
        """Format a human-readable summary of all past sessions."""
        if not self._entries:
            return "  No past experience recorded."

        sessions = set(e.get("session_id", "?") for e in self._entries)
        total = len(self._entries)
        successes = sum(1 for e in self._entries if e["outcome"] == "SUCCESS")
        degraded = sum(1 for e in self._entries if e["outcome"] == "DEGRADED")
        failed = total - successes - degraded
        rate = successes / total * 100 if total else 0

        # Most used tools
        tool_counts: Counter = Counter()
        for e in self._entries:
            for t in e.get("tool_chain", []):
                tool_counts[t] += 1
        top_tools = tool_counts.most_common(8)

        # Total faults
        total_faults = sum(e.get("faults", 0) for e in self._entries)

        # Most common objectives
        obj_counts: Counter = Counter()
        for e in self._entries:
            obj_counts[e.get("objective", "?")] += 1
        top_objectives = obj_counts.most_common(5)

        lines = [
            f"  {'='*50}",
            f"  AETHER Agent Memory Summary",
            f"  {'='*50}",
            f"  Total entries:   {total}",
            f"  Sessions:        {len(sessions)}",
            f"  Success rate:    {rate:.1f}% "
            f"({successes} ok, {degraded} degraded, {failed} failed)",
            f"  Total faults:    {total_faults}",
            f"",
            f"  Top tools:",
        ]
        for tool, count in top_tools:
            lines.append(f"    {tool:<25s} {count:>4d} uses")
        lines.append("")
        lines.append("  Most common objectives:")
        for obj, count in top_objectives:
            status_emoji = ""
            # Find latest outcome for this objective
            for e in reversed(self._entries):
                if e.get("objective") == obj:
                    status_emoji = ("ok" if e["outcome"] == "SUCCESS"
                                    else "FAIL")
                    break
            lines.append(f"    [{status_emoji:>4s}] {obj[:50]:<50s} x{count}")
        lines.append(f"  {'='*50}")

        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._entries)

    @property
    def session_count(self) -> int:
        return len(set(e.get("session_id", "?") for e in self._entries))

    # ── Persistence ───────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            if isinstance(data, list):
                self._entries = data[-_MAX_ENTRIES:]
        except (json.JSONDecodeError, OSError):
            self._entries = []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._entries, f, indent=2)

    def print_startup(self) -> None:
        """Print startup message with loaded memory stats."""
        if self._entries:
            sessions = self.session_count
            print(f"  [Memory] Loaded {len(self._entries)} past experiences "
                  f"across {sessions} session(s)")
        else:
            print("  [Memory] No past experiences — starting fresh")


def _extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from an objective string."""
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "is", "it", "my", "do", "get", "check", "run",
        "all", "this", "that", "with", "from", "what", "how", "can",
    }
    words = re.findall(r'[a-z]+', text.lower())
    return [w for w in words if w not in stop_words and len(w) > 2]
