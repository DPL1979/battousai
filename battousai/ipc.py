"""
ipc.py — Battousai Inter-Process Communication
==========================================
Message-passing infrastructure for the Autonomous Intelligence Operating System.

Communication Primitives:
    Mailbox (async inbox)
        Every agent has an inbox — a FIFO queue of messages. Senders drop
        messages in; recipients pick them up on their next `think()` call.

    Request/Reply
        Callers attach a correlation_id to a message. The recipient echoes
        the same correlation_id in its response so the original caller can
        match the reply to its pending request.

    Broadcast
        A message sent to the special recipient BROADCAST_ALL is delivered
        to every registered agent's inbox.

    Bulletin Board (Pub/Sub)
        Agents publish values to named topics. Subscribers receive the latest
        value when they poll the board. Good for health-checks, system metrics,
        and shared observations.

Message Types (extensible enum):
    TASK, RESULT, STATUS, QUERY, REPLY, BROADCAST, HEARTBEAT, ERROR, CUSTOM
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Deque


# Sentinel value for broadcast recipients
BROADCAST_ALL = "__BROADCAST__"


class MessageType(Enum):
    TASK      = auto()   # Assign work to an agent
    RESULT    = auto()   # Return computation output
    STATUS    = auto()   # Inform about state change
    QUERY     = auto()   # Request information
    REPLY     = auto()   # Response to a QUERY
    BROADCAST = auto()   # System-wide announcement
    HEARTBEAT = auto()   # Periodic liveness signal
    ERROR     = auto()   # Signal an error condition
    CUSTOM    = auto()   # Application-defined type


@dataclass
class Message:
    """
    A single unit of inter-agent communication.

    Fields:
        sender_id      — agent_id of the sender ("kernel" for OS messages)
        recipient_id   — agent_id of the recipient, or BROADCAST_ALL
        message_type   — semantic classification
        payload        — arbitrary data (dict, str, int, etc.)
        timestamp      — system tick at which the message was created
        message_id     — globally unique identifier (auto-generated)
        correlation_id — set by the caller for request/reply matching;
                         the responder copies this into its REPLY message
        ttl            — ticks until message is discarded unread (0 = no expiry)
    """
    sender_id: str
    recipient_id: str
    message_type: MessageType
    payload: Any
    timestamp: int
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    correlation_id: Optional[str] = None
    ttl: int = 0  # 0 means no expiry

    def is_expired(self, current_tick: int) -> bool:
        return self.ttl > 0 and (current_tick - self.timestamp) >= self.ttl

    def __repr__(self) -> str:
        return (
            f"Message(id={self.message_id}, "
            f"{self.sender_id}->{self.recipient_id}, "
            f"type={self.message_type.name}, "
            f"tick={self.timestamp})"
        )


class Mailbox:
    """
    FIFO message queue for a single agent.

    Messages are placed here by the IPC manager and consumed by the
    agent during its `think()` call.
    """

    def __init__(self, agent_id: str, max_size: int = 128) -> None:
        self.agent_id = agent_id
        self.max_size = max_size
        self._queue: Deque[Message] = deque()
        self.total_received: int = 0

    def deliver(self, message: Message) -> bool:
        """
        Enqueue a message. Drops silently if the mailbox is full.
        Returns True if delivered, False if dropped.
        """
        if len(self._queue) >= self.max_size:
            return False
        self._queue.append(message)
        self.total_received += 1
        return True

    def receive(self, current_tick: int = 0) -> Optional[Message]:
        """
        Pop the oldest non-expired message from the queue.
        Expired messages are silently discarded.
        """
        while self._queue:
            msg = self._queue.popleft()
            if not msg.is_expired(current_tick):
                return msg
        return None

    def peek(self) -> Optional[Message]:
        """Look at the next message without removing it."""
        return self._queue[0] if self._queue else None

    def receive_all(self, current_tick: int = 0) -> List[Message]:
        """Drain all non-expired messages from the queue."""
        messages: List[Message] = []
        while self._queue:
            msg = self._queue.popleft()
            if not msg.is_expired(current_tick):
                messages.append(msg)
        return messages

    def size(self) -> int:
        return len(self._queue)

    def is_empty(self) -> bool:
        return len(self._queue) == 0


class BulletinBoard:
    """
    Shared pub/sub board for broadcasting named topics.

    Publishers write a topic → value pair. Subscribers read the latest
    value for any topic. Unlike mailboxes, the bulletin board is not
    consumed — it holds the most recent published value per topic.
    """

    def __init__(self) -> None:
        # topic → (value, publisher_id, timestamp)
        self._topics: Dict[str, tuple[Any, str, int]] = {}
        self._subscribers: Dict[str, List[str]] = {}   # topic → [agent_ids]
        self.publish_count: int = 0

    def publish(self, topic: str, value: Any, publisher_id: str, tick: int) -> None:
        """Publish a value to a topic. Overwrites any previous value."""
        self._topics[topic] = (value, publisher_id, tick)
        self.publish_count += 1

    def subscribe(self, topic: str, agent_id: str) -> None:
        """Register an agent as interested in a topic (informational only)."""
        self._subscribers.setdefault(topic, [])
        if agent_id not in self._subscribers[topic]:
            self._subscribers[topic].append(agent_id)

    def read(self, topic: str) -> Optional[Any]:
        """Get the latest published value for a topic, or None."""
        entry = self._topics.get(topic)
        return entry[0] if entry else None

    def read_full(self, topic: str) -> Optional[tuple]:
        """Return (value, publisher_id, tick) or None."""
        return self._topics.get(topic)

    def topics(self) -> List[str]:
        return list(self._topics.keys())

    def snapshot(self) -> Dict[str, Any]:
        return {topic: val for topic, (val, _, _) in self._topics.items()}


class IPCManager:
    """
    Central IPC manager for Battousai.

    Provides:
    - Agent mailbox registration and lookup
    - Message routing (unicast, broadcast)
    - The shared bulletin board
    - Statistics tracking
    """

    def __init__(self) -> None:
        self._mailboxes: Dict[str, Mailbox] = {}
        self.bulletin_board = BulletinBoard()
        self.total_sent: int = 0
        self.total_dropped: int = 0
        self._message_log: List[Message] = []
        self._log_limit: int = 5_000

    # ------------------------------------------------------------------
    # Mailbox management
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str, max_mailbox_size: int = 128) -> Mailbox:
        """Create and register a mailbox for a new agent."""
        mb = Mailbox(agent_id, max_mailbox_size)
        self._mailboxes[agent_id] = mb
        return mb

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent's mailbox when it terminates."""
        self._mailboxes.pop(agent_id, None)

    def get_mailbox(self, agent_id: str) -> Optional[Mailbox]:
        return self._mailboxes.get(agent_id)

    # ------------------------------------------------------------------
    # Message routing
    # ------------------------------------------------------------------

    def send(self, message: Message) -> bool:
        """
        Route a message to its recipient(s).

        If recipient_id is BROADCAST_ALL, delivers to all registered agents.
        Returns True if at least one delivery succeeded.
        """
        # Always log
        if len(self._message_log) < self._log_limit:
            self._message_log.append(message)
        self.total_sent += 1

        if message.recipient_id == BROADCAST_ALL:
            delivered = 0
            for agent_id, mb in self._mailboxes.items():
                if agent_id != message.sender_id:  # Don't echo to sender
                    if mb.deliver(message):
                        delivered += 1
                    else:
                        self.total_dropped += 1
            return delivered > 0

        mb = self._mailboxes.get(message.recipient_id)
        if mb is None:
            self.total_dropped += 1
            return False
        if mb.deliver(message):
            return True
        self.total_dropped += 1
        return False

    def create_message(
        self,
        sender_id: str,
        recipient_id: str,
        message_type: MessageType,
        payload: Any,
        timestamp: int,
        correlation_id: Optional[str] = None,
        ttl: int = 0,
    ) -> Message:
        """Factory helper — creates a Message and sends it immediately."""
        msg = Message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            timestamp=timestamp,
            correlation_id=correlation_id,
            ttl=ttl,
        )
        self.send(msg)
        return msg

    # ------------------------------------------------------------------
    # Bulletin board delegation
    # ------------------------------------------------------------------

    def publish(self, topic: str, value: Any, publisher_id: str, tick: int) -> None:
        self.bulletin_board.publish(topic, value, publisher_id, tick)

    def subscribe(self, topic: str, agent_id: str) -> None:
        self.bulletin_board.subscribe(topic, agent_id)

    def board_read(self, topic: str) -> Optional[Any]:
        return self.bulletin_board.read(topic)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "total_sent": self.total_sent,
            "total_dropped": self.total_dropped,
            "active_mailboxes": len(self._mailboxes),
            "bulletin_topics": len(self.bulletin_board.topics()),
            "mailbox_sizes": {aid: mb.size() for aid, mb in self._mailboxes.items()},
        }

    def message_log(self) -> List[Message]:
        return list(self._message_log)
