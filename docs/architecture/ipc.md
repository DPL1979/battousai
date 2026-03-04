# IPC — Inter-Process Communication

The `ipc.py` module provides all message-passing infrastructure for Battousai. Agents communicate exclusively through this layer — there is no shared mutable state between agents.

---

## Communication Primitives

| Primitive | Description |
|---|---|
| **Mailbox** | Each agent has an inbox — a FIFO queue. Senders drop messages in; recipients pick them up on their next `think()` call |
| **Broadcast** | A message sent to `BROADCAST_ALL` is delivered to every registered agent |
| **Request/Reply** | Callers attach a `correlation_id`; the recipient echoes it in the reply so the caller can match responses |
| **Bulletin Board** | Agents publish key-value pairs to named topics; subscribers read the latest value |

---

## `MessageType` Enum

```python
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
```

---

## `Message` Dataclass

```python
@dataclass
class Message:
    sender_id: str          # agent_id of the sender ("kernel" for OS messages)
    recipient_id: str       # agent_id of the recipient, or BROADCAST_ALL
    message_type: MessageType
    payload: Any            # arbitrary data (dict, str, int, etc.)
    timestamp: int          # system tick at which the message was created
    message_id: str         # globally unique identifier (auto-generated UUID4 short)
    correlation_id: Optional[str] = None  # for request/reply matching
    ttl: int = 0            # ticks until message is discarded (0 = no expiry)

    def is_expired(self, current_tick: int) -> bool: ...
```

---

## Mailboxes

Every agent gets a `Mailbox` when it is registered with the IPC manager:

```python
class Mailbox:
    def __init__(self, agent_id: str, max_size: int = 128) -> None

    def deliver(self, message: Message) -> bool    # enqueue; returns False if full
    def receive(self, current_tick: int = 0) -> Optional[Message]  # pop oldest
    def receive_all(self, current_tick: int = 0) -> List[Message]  # drain all
    def peek(self) -> Optional[Message]            # look without removing
    def size(self) -> int
    def is_empty(self) -> bool
```

Mailboxes have a maximum capacity of 128 messages by default. Messages are dropped silently if the mailbox is full (`total_dropped` is incremented on the `IPCManager`).

---

## Sending Messages

From within an agent, use the `send_message` convenience method:

```python
# Unicast to a specific agent
self.send_message(
    recipient_id="worker_0003",
    message_type=MessageType.TASK,
    payload={"description": "analyse data", "queries": ["q1", "q2"]},
)

# Broadcast to all agents
from battousai.ipc import BROADCAST_ALL
self.send_message(
    recipient_id=BROADCAST_ALL,
    message_type=MessageType.BROADCAST,
    payload="System alert: high agent count detected",
)
```

The sender does **not** receive their own broadcast.

---

## Reading Messages

Call `self.read_inbox()` inside `think()` to drain all pending messages:

```python
def think(self, tick: int) -> None:
    messages = self.read_inbox()
    for msg in messages:
        if msg.message_type == MessageType.TASK:
            task = msg.payload
            self.log(f"Received task: {task}")
        elif msg.message_type == MessageType.RESULT:
            self.log(f"Got result from {msg.sender_id}")
```

`read_inbox()` returns a `List[Message]`. Expired messages (TTL reached) are automatically discarded.

---

## Request/Reply with Correlation IDs

For patterns where agent A sends a query and needs to match the reply:

```python
import uuid

class QueryAgent(Agent):
    def __init__(self):
        super().__init__(name="Querier", priority=4)
        self._pending: dict = {}  # correlation_id → expected_sender

    def think(self, tick: int) -> None:
        from battousai.ipc import MessageType
        import uuid

        # Send a query with a correlation ID
        if tick == 1:
            cid = str(uuid.uuid4())[:8]
            self._pending[cid] = "oracle_0001"
            self.send_message(
                recipient_id="oracle_0001",
                message_type=MessageType.QUERY,
                payload={"question": "what is 6*7?"},
                correlation_id=cid,
            )

        # Match the reply
        for msg in self.read_inbox():
            if msg.message_type == MessageType.REPLY and msg.correlation_id in self._pending:
                self.log(f"Answer: {msg.payload}")
                del self._pending[msg.correlation_id]

        self.yield_cpu()


class OracleAgent(Agent):
    def think(self, tick: int) -> None:
        for msg in self.read_inbox():
            if msg.message_type == MessageType.QUERY:
                # Echo the correlation_id in the reply
                self.send_message(
                    recipient_id=msg.sender_id,
                    message_type=MessageType.REPLY,
                    payload={"answer": 42},
                    correlation_id=msg.correlation_id,
                )
        self.yield_cpu()
```

---

## Bulletin Board (Pub/Sub)

The `BulletinBoard` is a shared key-value store for broadcasting metrics, configuration, and discoveries:

```python
class BulletinBoard:
    def publish(self, topic: str, value: Any, publisher_id: str, tick: int) -> None
    def subscribe(self, topic: str, agent_id: str) -> None
    def read(self, topic: str) -> Optional[Any]         # latest value
    def read_full(self, topic: str) -> Optional[tuple]  # (value, publisher_id, tick)
    def topics(self) -> List[str]
    def snapshot(self) -> Dict[str, Any]
```

Unlike mailboxes, the bulletin board is not consumed — it retains the latest published value per topic indefinitely.

Publishing from an agent:

```python
# Inside think()
self.syscall("publish_topic", topic="system.health", value={
    "tick": tick,
    "agents_alive": len(self.list_agents()),
})
```

Subscribing (informational — updates routing metadata):

```python
self.syscall("subscribe", topic="system.health")
```

Reading a topic:

```python
# Direct kernel access (for external/test code)
value = kernel.ipc.board_read("system.health")
```

---

## `IPCManager` Reference

```python
class IPCManager:
    def __init__(self) -> None
    
    # Mailbox management
    def register_agent(self, agent_id: str, max_mailbox_size: int = 128) -> Mailbox
    def unregister_agent(self, agent_id: str) -> None
    def get_mailbox(self, agent_id: str) -> Optional[Mailbox]

    # Message routing
    def send(self, message: Message) -> bool
    def create_message(self, sender_id, recipient_id, message_type,
                       payload, timestamp, correlation_id=None, ttl=0) -> Message

    # Bulletin board
    def publish(self, topic, value, publisher_id, tick) -> None
    def subscribe(self, topic, agent_id) -> None
    def board_read(self, topic) -> Optional[Any]

    # Statistics
    def stats(self) -> Dict[str, Any]
    def message_log(self) -> List[Message]
    
    # Counters
    total_sent: int
    total_dropped: int
```

---

## Message TTL

Messages can expire if unread. Set `ttl` (in ticks) when sending:

```python
# This message expires after 5 ticks
msg = Message(
    sender_id="agent_a",
    recipient_id="agent_b",
    message_type=MessageType.STATUS,
    payload={"status": "processing"},
    timestamp=current_tick,
    ttl=5,
)
kernel.ipc.send(msg)
```

`msg.is_expired(current_tick)` returns `True` if `ttl > 0` and `current_tick - timestamp >= ttl`.

---

## Statistics

```python
stats = kernel.ipc.stats()
# {
#   "total_sent": 17,
#   "total_dropped": 0,
#   "active_mailboxes": 4,
#   "bulletin_topics": 1,
#   "mailbox_sizes": {"coordinator_0002": 0, "worker_0003": 0, ...},
# }
```

---

## Related Pages

- [Kernel](kernel.md) — how the kernel routes IPC syscalls
- [Agent API](../agents/api.md) — `send_message`, `read_inbox`, `syscall("publish_topic", ...)` wrappers
- [Demo Walkthrough](../getting-started/demo.md) — IPC in action during the research scenario
