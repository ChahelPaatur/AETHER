"""
PlannerAgent: wraps the Hierarchical Planner; handles REPLAN signals from AdaptationAgent.
"""
from typing import Dict, List, Optional
from ..core.planner import Planner
from ..core.message_bus import MessageBus, Message


class PlannerAgent:
    """Wrapper agent connecting the planner to the message bus."""

    name = "planner_agent"

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self.planner = Planner()
        self._current_goal: Optional[Dict] = None
        bus.subscribe("REPLAN", self._on_replan)

    def set_goal(self, goal: Dict) -> None:
        self._current_goal = goal
        self.planner.set_goal(goal)

    def plan(self, state: Dict, available_actions: List[str]) -> List[str]:
        if self._current_goal is None:
            return ["stop"]
        return self.planner.plan(self._current_goal, state, available_actions)

    def _on_replan(self, msg: Message) -> None:
        recovery = msg.data.get("recovery_action", "")
        if recovery:
            self.planner.handle_replan(recovery)

    def reset(self) -> None:
        self.planner.reset()
        self._current_goal = None

    @property
    def planner_state(self) -> str:
        return self.planner.planner_state
