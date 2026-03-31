"""
Base hardware adapter ABC: uniform interface all robot adapters must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple


class HardwareAdapter(ABC):
    """Abstract base class for all robot hardware adapters."""

    @abstractmethod
    def execute(self, action: str, state: Dict) -> Tuple[Dict, bool]:
        """Execute action. Returns (result_dict, success)."""
        ...

    @abstractmethod
    def is_action_available(self, action: str) -> bool:
        """Check if an action is currently executable."""
        ...

    @abstractmethod
    def get_degradation_state(self) -> Dict:
        """Return current degradation levels per subsystem (0=healthy, 1=failed)."""
        ...

    def get_actuator_state(self) -> Dict:
        """Return current actuator states."""
        return {}

    def simulate_degradation(self, degradation_level: float) -> None:
        """Inject simulated degradation for testing (0=none, 1=full failure)."""
        pass
