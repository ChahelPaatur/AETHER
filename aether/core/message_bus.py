"""
MessageBus: synchronous pub/sub system for inter-agent communication.
All agents communicate exclusively through this bus — no direct references.
"""
from typing import Callable, Dict, List, Any
from collections import defaultdict
import heapq


class Message:
    """Structured message passed between agents."""

    PRIORITY_HIGH = 0
    PRIORITY_NORMAL = 1
    PRIORITY_LOW = 2

    _PRIORITY_MAP = {"HIGH": 0, "NORMAL": 1, "LOW": 2}

    def __init__(
        self,
        type: str,
        sender: str,
        data: Dict,
        recipient: str = "broadcast",
        timestamp: int = 0,
        priority: str = "NORMAL",
    ):
        self.type = type
        self.sender = sender
        self.recipient = recipient
        self.data = data
        self.timestamp = timestamp
        self.priority = priority
        self._priority_val = self._PRIORITY_MAP.get(priority, 1)

    def __lt__(self, other: "Message") -> bool:
        return self._priority_val < other._priority_val

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "sender": self.sender,
            "recipient": self.recipient,
            "data": self.data,
            "timestamp": self.timestamp,
            "priority": self.priority,
        }


class MessageBus:
    """
    Central pub/sub message bus.
    HIGH-priority messages (faults, safety signals) are processed before NORMAL/LOW.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._queue: List[Message] = []
        self._step: int = 0
        self._history: List[Dict] = []

    def subscribe(self, topic: str, handler: Callable[[Message], None]) -> None:
        """Register a handler for a specific message type."""
        self._subscribers[topic].append(handler)

    def publish(self, topic: str, sender: str, data: Dict,
                recipient: str = "broadcast", priority: str = "NORMAL") -> None:
        """Enqueue a message for delivery."""
        msg = Message(
            type=topic,
            sender=sender,
            recipient=recipient,
            data=data,
            timestamp=self._step,
            priority=priority,
        )
        heapq.heappush(self._queue, msg)

    def broadcast(self, sender: str, data: Dict, priority: str = "NORMAL") -> None:
        """Publish to all subscribers regardless of topic."""
        self.publish("BROADCAST", sender, data, "broadcast", priority)

    def flush(self) -> int:
        """Process all queued messages in priority order. Returns count delivered."""
        count = 0
        while self._queue:
            msg = heapq.heappop(self._queue)
            self._history.append(msg.to_dict())
            handlers = self._subscribers.get(msg.type, [])
            if msg.recipient != "broadcast":
                handlers = [h for h in handlers if getattr(h, "__self__", None) is not None
                            and getattr(h.__self__, "name", "") == msg.recipient] or handlers
            for handler in handlers:
                handler(msg)
            count += 1
        return count

    def tick(self, step: int) -> None:
        """Advance the bus clock and flush all pending messages."""
        self._step = step
        self.flush()

    def clear(self) -> None:
        self._queue = []

    def history(self, last_n: int = 20) -> List[Dict]:
        return self._history[-last_n:]
