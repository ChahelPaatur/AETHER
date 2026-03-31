"""
Hierarchical Planner v3: two-tier mission + execution planner with Behavior Tree executor.
Exposes select_action(state) interface for future RL backend integration.
"""
import math
from typing import Dict, List, Optional, Protocol


class PlannerBackend(Protocol):
    def plan(self, goal: Dict, state: Dict, available: List[str]) -> List[str]: ...


# ------------------------------------------------------------------ #
#  Behavior Tree Nodes                                                 #
# ------------------------------------------------------------------ #

class BTNode:
    def tick(self, state: Dict, available: List[str]) -> str:
        raise NotImplementedError


class Condition(BTNode):
    def __init__(self, check_fn, description: str = ""):
        self.check_fn = check_fn
        self.description = description

    def tick(self, state: Dict, available: List[str]) -> str:
        return "SUCCESS" if self.check_fn(state) else "FAILURE"


class Action(BTNode):
    def __init__(self, action: str):
        self.action = action

    def tick(self, state: Dict, available: List[str]) -> str:
        if self.action in available:
            return "RUNNING"
        return "FAILURE"

    def get_action(self) -> str:
        return self.action


class Sequence(BTNode):
    def __init__(self, children: List[BTNode]):
        self.children = children
        self._index = 0

    def tick(self, state: Dict, available: List[str]) -> str:
        for child in self.children[self._index:]:
            result = child.tick(state, available)
            if result == "FAILURE":
                self._index = 0
                return "FAILURE"
            if result == "RUNNING":
                return "RUNNING"
            self._index += 1
        self._index = 0
        return "SUCCESS"


class Selector(BTNode):
    def __init__(self, children: List[BTNode]):
        self.children = children

    def tick(self, state: Dict, available: List[str]) -> str:
        for child in self.children:
            result = child.tick(state, available)
            if result in ("SUCCESS", "RUNNING"):
                return result
        return "FAILURE"


# ------------------------------------------------------------------ #
#  Mission Planner (Tier 1)                                           #
# ------------------------------------------------------------------ #

class MissionPlanner:
    """Converts structured goal → sequence of subgoals with interruption handling."""

    TASK_SUBGOALS = {
        "navigate": ["scan", "approach", "follow", "arrive"],
        "follow":   ["scan", "follow", "maintain"],
        "scan":     ["scan", "turn", "scan"],
        "stop":     ["stop"],
        "patrol":   ["scan", "move", "turn", "scan"],
        "grasp":    ["scan", "approach", "stop"],
        "avoid":    ["avoid", "move"],
    }

    def __init__(self):
        self._goal_stack: List[str] = []
        self._weights: Dict[str, float] = {}

    def set_goal(self, goal: Dict) -> None:
        task = goal.get("task", "navigate")
        self._goal_stack = list(self.TASK_SUBGOALS.get(task, ["scan", "move"]))

    def next_subgoal(self, state: Dict) -> str:
        if not self._goal_stack:
            return "done"
        if state.get("at_target"):
            return "done"
        return self._goal_stack[0] if self._goal_stack else "done"

    def interrupt(self, reason: str) -> None:
        """Push an urgent subgoal to the front of the stack."""
        urgent_map = {
            "fault_detected": "recover",
            "obstacle": "avoid",
            "battery_low": "safe_mode",
        }
        if reason in urgent_map:
            self._goal_stack.insert(0, urgent_map[reason])


# ------------------------------------------------------------------ #
#  Execution Planner (Tier 2) — BT-based                             #
# ------------------------------------------------------------------ #

class ExecutionPlanner:
    """Converts each subgoal → sequence of primitive abstract actions via BT."""

    def __init__(self):
        self.state = "INIT"
        self._avoid_counter: int = 0
        self._recovery_action: Optional[str] = None
        self._low_navconf_steps: int = 0
        self._nav_recovery_queue: List[str] = []
        self._prev_position: Optional[List[float]] = None
        self._stuck_counter: int = 0
        self._degraded_resume_state: str = "APPROACH"
        self._wall_hug_counter: int = 0

    def reset(self) -> None:
        self.state = "INIT"
        self._avoid_counter = 0
        self._recovery_action = None
        self._low_navconf_steps = 0
        self._nav_recovery_queue = []
        self._prev_position = None
        self._stuck_counter = 0
        self._degraded_resume_state = "APPROACH"
        self._wall_hug_counter = 0

    def plan(self, goal: Dict, state: Dict, available: List[str]) -> List[str]:
        """Returns list of actions to execute this step."""
        # Recovery override
        if self._recovery_action and self._recovery_action in available:
            action = self._recovery_action
            self._recovery_action = None
            return [action]

        # State machine with BT-informed transitions
        self._transition(goal, state, available)
        return self._actions_for_state(goal, state, available)

    def set_recovery(self, action: str) -> None:
        self._recovery_action = action

    def _transition(self, goal: Dict, state: Dict, available: List[str]) -> None:
        failed_actuators = state.get("failed_actuators", [])
        failed_sensors = state.get("failed_sensors", [])
        obstacles = state.get("obstacles_nearby", [])

        if state.get("at_target"):
            self.state = "DONE"
            return
        if "wheels" in failed_actuators or "thrusters" in failed_actuators:
            self.state = "DEGRADED"
            return

        # ── Stuck detection: position change < 0.5 for 8+ steps ─────────
        pos = state.get("position", state.get("agent_pos"))
        if pos and self._prev_position:
            delta = (abs(float(pos[0]) - self._prev_position[0])
                     + abs(float(pos[1]) - self._prev_position[1]))
            if delta < 0.5:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0
        if pos:
            self._prev_position = [float(pos[0]), float(pos[1])]

        if (self._stuck_counter > 8
                and self.state == "APPROACH"
                and not self._nav_recovery_queue):
            turn = self._wall_escape_turn(pos)
            self._nav_recovery_queue = (
                ["stop"]
                + [turn] * 3              # ~135° rotation away from wall
                + ["move_forward"] * 10   # escape along new heading
            )
            self.state = "DEGRADED"
            self._degraded_resume_state = "SCAN"
            self._stuck_counter = 0
            return

        # ── Wall-hugging detection: near arena edge for 15+ steps ────────
        if pos:
            px, py = float(pos[0]), float(pos[1])
            near_wall = (px < 1.0 or px > 99.0 or py < 1.0 or py > 99.0)
        else:
            near_wall = False

        if near_wall:
            self._wall_hug_counter += 1
        else:
            self._wall_hug_counter = 0

        if (self._wall_hug_counter > 15
                and self.state == "APPROACH"
                and not self._nav_recovery_queue):
            target = state.get("target", {})
            target_pos = target.get("pos") if isinstance(target, dict) else None
            heading = state.get("heading", 0)
            self._nav_recovery_queue = self._compute_target_escape(pos, heading, target_pos)
            self.state = "DEGRADED"
            self._degraded_resume_state = "APPROACH"
            self._wall_hug_counter = 0
            return

        # ── NavConf degradation tracking ─────────────────────────────────
        nav_conf = state.get("nav_confidence", 1.0)
        if nav_conf < 0.15:
            self._low_navconf_steps += 1
        else:
            self._low_navconf_steps = 0

        # Escalate to DEGRADED re-orientation if NavConf critically low for 10+ steps
        if (self._low_navconf_steps >= 10
                and self.state == "APPROACH"
                and not self._nav_recovery_queue):
            self._nav_recovery_queue = [
                "stop", "scan_environment", "turn_left",
                "move_forward", "move_forward",
            ]
            self.state = "DEGRADED"
            self._degraded_resume_state = "APPROACH"
            self._low_navconf_steps = 0
            return

        if self.state == "INIT":
            self.state = "SCAN"
        elif self.state == "SCAN":
            if state.get("target_info") and "follow_target" in available:
                self.state = "FOLLOW"
            else:
                self.state = "APPROACH"
        elif self.state in ("APPROACH", "FOLLOW"):
            if obstacles and "avoid_obstacle" in available:
                self.state = "AVOID"
                self._avoid_counter = 0
        elif self.state == "AVOID":
            self._avoid_counter += 1
            if self._avoid_counter >= 3 and not obstacles:
                self.state = "FOLLOW" if (
                    state.get("target_info") and "follow_target" in available
                ) else "APPROACH"
            elif self._avoid_counter > 8:
                self.state = "APPROACH"
        elif self.state == "DEGRADED":
            # Actuator recovery
            if "wheels" not in failed_actuators and "thrusters" not in failed_actuators:
                if not self._nav_recovery_queue:
                    self.state = "SCAN"

    def _actions_for_state(self, goal: Dict, state: Dict, available: List[str]) -> List[str]:
        constraints = goal.get("constraints", [])
        failed_sensors = state.get("failed_sensors", [])
        obstacles = state.get("obstacles_nearby", [])

        # Safety override: obstacle with safety constraint
        if obstacles and "avoid_obstacles" in constraints and "avoid_obstacle" in available:
            return ["avoid_obstacle"]
        if obstacles and "avoid_obstacle" in available:
            return ["avoid_obstacle"]

        # Drain DEGRADED recovery queue (stuck escape or NavConf re-orientation)
        if self._nav_recovery_queue:
            action = self._nav_recovery_queue.pop(0)
            if not self._nav_recovery_queue:
                self.state = self._degraded_resume_state
            return [action] if action in available else ["stop"]

        state_map = {
            "INIT": ["scan_environment"],
            "SCAN": ["scan_environment"],
            "APPROACH": ["move_forward"],
            "FOLLOW": ["follow_target"],
            "AVOID": ["avoid_obstacle", "move_forward"],
            "DONE": ["stop"],
            "DEGRADED": ["stop"],
        }

        # Camera failure degradation
        if "camera" in failed_sensors and self.state in ("FOLLOW", "APPROACH"):
            return [a for a in ["move_forward", "turn_left"] if a in available] or ["stop"]

        desired = state_map.get(self.state, ["stop"])
        return [a for a in desired if a in available] or ["stop"]

    @staticmethod
    def _wall_escape_turn(pos) -> str:
        """Choose turn direction away from the nearest wall/corner."""
        if not pos:
            return "turn_left"
        x, y = float(pos[0]), float(pos[1])
        # Near low-coordinate edges (origin corner): turn_right moves away
        # Near high-coordinate edges: turn_left moves away
        if x + y < 40:
            return "turn_right"
        return "turn_left"

    @staticmethod
    def _compute_target_escape(pos, heading, target_pos) -> List[str]:
        """
        Compute escape sequence that turns the agent to face the target,
        then moves forward enough to clear the wall by 5+ units.
        """
        if not pos or not target_pos:
            # Fallback: generic 135° rotation + forward
            return ["stop"] + ["turn_right"] * 3 + ["move_forward"] * 10

        px, py = float(pos[0]), float(pos[1])
        tx, ty = float(target_pos[0]), float(target_pos[1])

        # Desired heading toward target (degrees, 0=east, CCW positive)
        desired_deg = math.degrees(math.atan2(ty - py, tx - px))
        current_deg = float(heading) % 360

        # Shortest angular difference, normalized to [-180, 180]
        diff = (desired_deg - current_deg + 180) % 360 - 180

        num_turns = max(1, round(abs(diff) / 30))  # env turns are 30° each
        turn_dir = "turn_left" if diff > 0 else "turn_right"

        return ["stop"] + [turn_dir] * num_turns + ["move_forward"] * 10

    @property
    def current_state(self) -> str:
        return self.state


# ------------------------------------------------------------------ #
#  Unified Planner Interface                                           #
# ------------------------------------------------------------------ #

class Planner:
    """Two-tier planner: MissionPlanner (Tier 1) + ExecutionPlanner (Tier 2)."""

    def __init__(self):
        self.mission = MissionPlanner()
        self.execution = ExecutionPlanner()
        self._goal: Optional[Dict] = None
        self._weights: Dict = {}

    def set_goal(self, goal: Dict) -> None:
        self._goal = goal
        self.mission.set_goal(goal)

    def plan(self, goal: Dict, state: Dict, available: List[str]) -> List[str]:
        if self._goal is None:
            self.set_goal(goal)
        return self.execution.plan(goal, state, available)

    def select_action(self, state: Dict) -> str:
        """RL-compatible interface: returns single action string."""
        goal = self._goal or {"task": "navigate", "constraints": [], "subtasks": []}
        actions = self.execution.plan(goal, state, list(state.get("available_actions", [])))
        return actions[0] if actions else "stop"

    def handle_replan(self, recovery_action: str) -> None:
        """Called by AdaptationAgent with recovery action."""
        self.execution.set_recovery(recovery_action)

    def reset(self) -> None:
        self.execution.reset()
        self._goal = None

    def update_weights(self, weights: Dict) -> None:
        self._weights = weights

    @property
    def planner_state(self) -> str:
        return self.execution.current_state
