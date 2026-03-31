"""
Action Abstraction Layer: hardware-independent action definitions.
Each AbstractAction defines what the action means, not how it's implemented.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ActionResult:
    action: str
    success: bool
    reason: str = ""
    state: Dict = field(default_factory=dict)


class AbstractAction(ABC):
    name: str = ""

    @abstractmethod
    def preconditions(self, state: Dict) -> bool:
        """Returns True if action can be executed given current state."""
        ...

    @abstractmethod
    def expected_effect(self, state: Dict) -> Dict:
        """Returns expected state changes after action."""
        ...

    def description(self) -> str:
        return f"Abstract action: {self.name}"


class MoveForward(AbstractAction):
    name = "move_forward"

    def preconditions(self, state: Dict) -> bool:
        return not state.get("obstacles_nearby") or len(state["obstacles_nearby"]) == 0

    def expected_effect(self, state: Dict) -> Dict:
        return {"agent_moved": True, "direction": "forward"}

    def description(self) -> str:
        return "Move the agent forward in its current heading direction"


class FollowTarget(AbstractAction):
    name = "follow_target"

    def preconditions(self, state: Dict) -> bool:
        return (
            state.get("target_detected", False)
            and state.get("target_info") is not None
            and "camera" not in state.get("failed_sensors", [])
        )

    def expected_effect(self, state: Dict) -> Dict:
        return {"closer_to_target": True, "following": True}

    def description(self) -> str:
        return "Move toward the nearest detected target"


class AvoidObstacle(AbstractAction):
    name = "avoid_obstacle"

    def preconditions(self, state: Dict) -> bool:
        return len(state.get("obstacles_nearby", [])) > 0

    def expected_effect(self, state: Dict) -> Dict:
        return {"obstacle_avoided": True, "heading_changed": True}

    def description(self) -> str:
        return "Steer away from detected obstacles"


class Stop(AbstractAction):
    name = "stop"

    def preconditions(self, state: Dict) -> bool:
        return True  # can always stop

    def expected_effect(self, state: Dict) -> Dict:
        return {"velocity": 0, "stopped": True}

    def description(self) -> str:
        return "Halt all movement immediately"


class ScanEnvironment(AbstractAction):
    name = "scan_environment"

    def preconditions(self, state: Dict) -> bool:
        sensors = state.get("failed_sensors", [])
        return "camera" not in sensors or "ultrasonic" not in sensors

    def expected_effect(self, state: Dict) -> Dict:
        return {"environment_mapped": True, "obstacles_known": True}

    def description(self) -> str:
        return "Use available sensors to observe and map surroundings"


class Turn(AbstractAction):
    name = "turn"

    def preconditions(self, state: Dict) -> bool:
        return "wheels" not in state.get("failed_actuators", [])

    def expected_effect(self, state: Dict) -> Dict:
        return {"heading_changed": True}

    def description(self) -> str:
        return "Rotate the agent in place"


ACTION_REGISTRY: Dict[str, AbstractAction] = {
    "move_forward": MoveForward(),
    "follow_target": FollowTarget(),
    "avoid_obstacle": AvoidObstacle(),
    "stop": Stop(),
    "scan_environment": ScanEnvironment(),
    "turn": Turn(),
}


def get_action(name: str) -> Optional[AbstractAction]:
    return ACTION_REGISTRY.get(name)


def check_preconditions(action_name: str, state: Dict) -> bool:
    action = get_action(action_name)
    if action is None:
        return False
    return action.preconditions(state)
