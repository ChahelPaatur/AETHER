"""
Execution Layer: receives planned actions and dispatches them through hardware adapters.
Handles action sequencing, timing, and result collection.
"""
from typing import Dict, List, Optional, Protocol
import logging

logger = logging.getLogger(__name__)


class Adapter(Protocol):
    def execute(self, action: str, state: Dict) -> Dict:
        ...


class Executor:
    """
    Dispatches abstract actions to the bound hardware adapter.
    Tracks execution history and detects execution-level failures.
    """

    def __init__(self, adapter: Adapter):
        self.adapter = adapter
        self.history: List[Dict] = []

    def execute(self, actions: List[str], state: Dict) -> Dict:
        """
        Execute a list of planned actions sequentially.
        Returns the final state after executing all actions.
        """
        current_state = state
        for action in actions:
            result = self._execute_single(action, current_state)
            self.history.append(result)
            current_state = result

            # Stop executing if action failed (don't compound failures)
            if not result.get("success", True):
                logger.warning(f"Action '{action}' failed: {result.get('reason', 'unknown')}")
                break

            # Stop if at target (goal achieved mid-sequence)
            if result.get("at_target"):
                logger.info("Target reached, halting execution sequence.")
                break

        return current_state

    def _execute_single(self, action: str, state: Dict) -> Dict:
        try:
            result = self.adapter.execute(action, state)
            result["executed_action"] = action
            return result
        except Exception as e:
            logger.error(f"Execution error for action '{action}': {e}")
            return {
                **state,
                "executed_action": action,
                "success": False,
                "reason": f"execution_exception: {str(e)}",
            }

    def last_result(self) -> Optional[Dict]:
        return self.history[-1] if self.history else None

    def clear_history(self) -> None:
        self.history = []

    def swap_adapter(self, adapter: Adapter) -> None:
        """Hot-swap adapter (e.g., when robot type changes or adapter fails)."""
        self.adapter = adapter
