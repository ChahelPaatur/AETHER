"""
GoalParser v3: parses natural language into structured task dicts.
Supports multi-constraint parsing, subtask generation, and TaskValidator.
"""
import re
from typing import Dict, List, Optional, Protocol


class GoalParserBackend(Protocol):
    def parse(self, text: str) -> Dict: ...


class RuleBasedParser:
    TASK_PATTERNS = [
        (r"\bfollow\b", "follow"),
        (r"\bnavigate\b|\bgo to\b|\btravel to\b|\breach\b|\bhead to\b", "navigate"),
        (r"\bscan\b|\bexplore\b|\bsurvey\b|\bmap\b", "scan"),
        (r"\bstop\b|\bhalt\b|\bfreeze\b", "stop"),
        (r"\bpatrol\b", "patrol"),
        (r"\bpick up\b|\bgrasp\b|\bcollect\b", "grasp"),
        (r"\bavoid\b", "avoid"),
    ]
    TARGET_PATTERNS = [
        (r"\bwaypoint\s+(\w+)\b", "waypoint"),
        (r"\b(person|human|man|woman)\b", "person"),
        (r"\b(car|vehicle|truck)\b", "vehicle"),
        (r"\b(object|item|package|box)\b", "object"),
        (r"\btarget\b|\bdestination\b", "target"),
        (r"\benvironment\b", "environment"),
    ]
    CONSTRAINT_PATTERNS = [
        (r"avoid(ing)?\s+obstacles?", "avoid_obstacles"),
        (r"avoid(ing)?\s+collision", "avoid_collision"),
        (r"follow(ing)?\s+\w+", "maintain_follow"),
        (r"safely?", "safety_first"),
        (r"quickly?|fast(ly)?", "speed_priority"),
    ]
    SUBTASK_MAP = {
        "navigate": ["scan_environment", "plan_path", "move_forward"],
        "follow": ["scan_environment", "follow_target"],
        "scan": ["scan_environment", "turn", "scan_environment"],
        "stop": ["stop"],
        "patrol": ["scan_environment", "move_forward", "turn"],
        "grasp": ["scan_environment", "move_forward", "stop"],
        "avoid": ["avoid_obstacle", "move_forward"],
    }
    PRIORITY_MAP = {"avoid_obstacles": "safety", "avoid_collision": "safety",
                    "safety_first": "safety", "speed_priority": "speed"}

    def parse(self, text: str) -> Dict:
        t = text.lower().strip()
        task = self._match_first(t, self.TASK_PATTERNS) or "navigate"
        target = self._extract_target(t)
        constraints = [c for p, c in self.CONSTRAINT_PATTERNS if re.search(p, t)]
        priority = next((self.PRIORITY_MAP[c] for c in constraints
                         if c in self.PRIORITY_MAP), "efficiency")
        subtasks = self.SUBTASK_MAP.get(task, ["scan_environment", "move_forward"])
        if "avoid_obstacles" in constraints and "avoid_obstacle" not in subtasks:
            subtasks = ["avoid_obstacle"] + subtasks
        return {"task": task, "target": target, "constraints": constraints,
                "priority": priority, "subtasks": subtasks, "timeout": 300, "raw": text}

    def _match_first(self, text: str, patterns: list) -> Optional[str]:
        for pattern, label in patterns:
            if re.search(pattern, text):
                return label
        return None

    def _extract_target(self, text: str) -> str:
        m = re.search(r"\bwaypoint\s+(\w+)", text)
        if m:
            return f"waypoint_{m.group(1)}"
        for p, label in self.TARGET_PATTERNS[1:]:
            if re.search(p, text):
                return label
        return "target"


class TaskValidator:
    """Validates that a parsed task is achievable given the capability graph."""

    def validate(self, task: Dict, available_actions: List[str]) -> Dict:
        required = set(task.get("subtasks", []))
        missing = required - set(available_actions)
        achievable = len(missing) == 0
        return {
            "achievable": achievable,
            "missing_actions": list(missing),
            "warnings": [f"Action '{a}' not available" for a in missing],
        }


class GoalParser:
    """Main goal parsing interface with pluggable backend."""

    def __init__(self, backend: Optional[GoalParserBackend] = None):
        self._backend = backend or RuleBasedParser()
        self.validator = TaskValidator()

    def parse(self, text: str) -> Dict:
        result = self._backend.parse(text)
        for key in ("task", "target", "constraints", "priority", "subtasks"):
            if key not in result:
                raise ValueError(f"GoalParser backend missing key: {key}")
        return result

    def parse_and_validate(self, text: str, available_actions: List[str]) -> Dict:
        goal = self.parse(text)
        validation = self.validator.validate(goal, available_actions)
        return {**goal, "validation": validation}
