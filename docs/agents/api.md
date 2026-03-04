# Agent API

The `Agent` base class in `agent.py` is the foundation of every process in Battousai. All custom agents subclass `Agent` and implement `think()`.

---

## Class Definition

```python
class Agent:
    def __init__(
        self,
        name: str,
        priority: int = 5,
        memory_allocation: int = 256,
        time_slice: int = 3,
    ) -> None
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Human-readable display name |
| `priority` | `int` | `5` | Scheduler priority 0 (highest) – 9 (lowest) |
| `memory_allocation` | `int` | `256` | Max memory keys for this agent |
| `time_slice` | `int` | `3` | Ticks per scheduling turn |

### Instance Attributes (set by kernel at spawn time)

| Attribute | Type | Description |
|---|---|---|
| `kernel` | `Optional[Kernel]` | Back-reference to the kernel |
| `agent_id` | `str` | Unique identifier (e.g. `"myagent_0003"`) |
| `name` | `str` | Human-readable display name |
| `priority` | `int` | Scheduler priority (0–9) |
| `memory_allocation` | `int` | Maximum memory keys |
| `time_slice` | `int` | Ticks per scheduling turn |

---

## Lifecycle Hooks

### `on_spawn()`

Called **once** immediately after the kernel registers the agent (before the first `think()` call). Override to initialise state:

```python
def on_spawn(self) -> None:
    self.log(f"{self.name} online!")
    self.mem_write("phase", "init")
    self.mem_write("results", [])
```

### `think(tick: int)`

**Must be implemented.** Called every tick the scheduler grants CPU time. This is the agent's "brain":

```python
def think(self, tick: int) -> None:
    messages = self.read_inbox()
    # ... process messages, make decisions, call tools ...
    self.yield_cpu()  # yield remaining time slice
```

!!! warning "Must call yield_cpu()"
    Always call `self.yield_cpu()` at the end of `think()`. Without it, the scheduler marks the agent as RUNNING until the end of the tick. Calling `yield_cpu()` lets the scheduler move on to the next agent immediately.

### `on_terminate()`

Called **once** before the kernel removes the agent. Override to clean up:

```python
def on_terminate(self) -> None:
    self.log(f"{self.name} shutting down cleanly.")
    final_count = self.mem_read("count") or 0
    self.log(f"Final count: {final_count}")
```

---

## Syscall Methods

All OS services go through these methods. They call `self.syscall(name, **kwargs)` internally and return `SyscallResult`.

### `syscall(name, **kwargs) → SyscallResult`

Low-level syscall interface. Prefer the named wrappers below, but use this for syscalls without a wrapper:

```python
result = self.syscall("publish_topic", topic="metrics", value={"tick": tick})
result = self.syscall("subscribe", topic="metrics")
```

### `SyscallResult`

```python
@dataclass
class SyscallResult:
    ok: bool
    value: Any = None
    error: Optional[str] = None

    def __bool__(self) -> bool: return self.ok
```

---

## Memory Methods

### `mem_write(key, value, memory_type=LONG_TERM, ttl=None) → SyscallResult`

Write a value to private memory:

```python
from battousai.memory import MemoryType

# Long-term (persists for the OS session)
self.mem_write("task", "research quantum computing")

# Short-term (expires after 10 ticks)
self.mem_write("scratch", intermediate_data, memory_type=MemoryType.SHORT_TERM, ttl=10)
```

### `mem_read(key) → Any`

Read a value from private memory. Returns `None` if the key doesn't exist or has expired:

```python
task = self.mem_read("task")
if task is None:
    self.log("No task set yet")
```

---

## Messaging Methods

### `send_message(recipient_id, message_type, payload, correlation_id=None) → SyscallResult`

Send a message to another agent:

```python
from battousai.ipc import MessageType, BROADCAST_ALL

# Unicast
self.send_message("worker_0003", MessageType.TASK, {"query": "quantum"})

# Broadcast
self.send_message(BROADCAST_ALL, MessageType.BROADCAST, "Alert: high load")

# With correlation ID for request/reply
import uuid
cid = str(uuid.uuid4())[:8]
self.send_message("oracle_0001", MessageType.QUERY, {"q": "status"}, correlation_id=cid)
```

### `read_inbox() → List[Message]`

Drain all pending messages from the mailbox:

```python
messages = self.read_inbox()
for msg in messages:
    if msg.message_type == MessageType.TASK:
        self.log(f"Got task from {msg.sender_id}")
```

---

## Tool Methods

### `use_tool(tool_name, **args) → SyscallResult`

Execute a registered tool:

```python
# Calculator
r = self.use_tool("calculator", expression="sqrt(144)")
if r.ok:
    self.log(f"Result: {r.value}")  # "12.0"

# Web search
r = self.use_tool("web_search", query="AI safety research")
if r.ok:
    results = r.value["results"]

# File operations via tool
r = self.use_tool("file_reader", path="/shared/data.txt")
```

---

## Filesystem Methods

### `write_file(path, data) → SyscallResult`

```python
self.write_file("/agents/self/notes.txt", "Important observation")
self.write_file("/shared/results/output.json", {"status": "done", "count": 42})
```

### `read_file(path) → SyscallResult`

```python
r = self.read_file("/shared/results/output.json")
if r.ok:
    data = r.value
```

---

## Agent Management Methods

### `spawn_child(agent_class, name, priority=5, **kwargs) → SyscallResult`

Spawn a child agent. Returns `SyscallResult` with `value=agent_id`:

```python
result = self.spawn_child(
    WorkerAgent,
    name="DataWorker",
    priority=4,
    subtask={"description": "process batch", "queries": ["q1", "q2"]},
)
if result.ok:
    child_id = result.value
    self.log(f"Spawned worker: {child_id}")
```

### `list_agents() → List[str]`

Return IDs of all currently living agents:

```python
agents = self.list_agents()
# ["coordinator_0002", "worker_0003", "worker_0004", "sysmonitor_0001"]

# Find the coordinator to report back to
coord_id = next((a for a in agents if "coordinator" in a), None)
```

### `get_status() → SyscallResult`

Return a full system metrics snapshot:

```python
status = self.get_status()
if status.ok:
    metrics = status.value
    agent_count = metrics["agent_count"]
    total_msgs = metrics["ipc_stats"]["total_sent"]
```

---

## CPU Scheduling Methods

### `yield_cpu() → None`

Voluntarily give up the remainder of this tick's CPU slice:

```python
def think(self, tick: int) -> None:
    # ... work ...
    self.yield_cpu()  # always call at end
```

---

## Logging

### `log(message, level=None) → None`

Log a message with the agent's ID as the source:

```python
from battousai.logger import LogLevel

self.log("Processing started")                          # INFO level
self.log("Detailed debug info", level=LogLevel.DEBUG)
self.log("Something wrong", level=LogLevel.WARN)
```

---

## Built-in Agent Types

### `CoordinatorAgent`

Decomposes high-level goals into subtasks, spawns workers, collects results, and synthesises a summary.

```python
from battousai.agent import CoordinatorAgent
coord_id = kernel.spawn_agent(CoordinatorAgent, name="Coordinator", priority=2)
```

Internal phases: `WAITING_TASK` → `DECOMPOSE` → `COLLECTING` → `DONE`

### `WorkerAgent`

Receives a task with a list of queries, runs `web_search` for each (one per tick), and reports results back to the coordinator.

```python
from battousai.agent import WorkerAgent
worker_id = kernel.spawn_agent(
    WorkerAgent,
    name="Worker-1",
    priority=4,
    subtask={
        "subtask_id": "st_1",
        "description": "Research AI safety",
        "queries": ["AI safety basics", "alignment problem"],
    },
)
```

### `MonitorAgent`

Samples system metrics every `_sample_interval` ticks (default 5) and publishes them to the `"system.health"` bulletin board topic. Logs an alert if agent count exceeds `_alert_threshold_agents` (default 20).

```python
from battousai.agent import MonitorAgent
monitor_id = kernel.spawn_agent(MonitorAgent, name="SysMonitor", priority=7)

# After the run, get the formatted report:
monitor = kernel._agents.get(monitor_id)
if monitor:
    print(monitor.get_report())
```

---

## Full Custom Agent Example

```python
from battousai.agent import Agent
from battousai.ipc import MessageType, BROADCAST_ALL
from battousai.memory import MemoryType

class ResearchAgent(Agent):
    def __init__(self, topic: str = "quantum computing"):
        super().__init__(
            name="Researcher",
            priority=4,
            memory_allocation=512,
            time_slice=3,
        )
        self._topic = topic
        self._phase = "INIT"

    def on_spawn(self) -> None:
        self.log(f"Online. Researching: {self._topic}")
        self.mem_write("topic", self._topic)
        self.mem_write("findings", [])
        self._phase = "SEARCHING"

    def think(self, tick: int) -> None:
        # Read inbox
        for msg in self.read_inbox():
            if msg.message_type == MessageType.TASK:
                new_topic = msg.payload.get("topic", self._topic)
                self.mem_write("topic", new_topic)
                self._phase = "SEARCHING"

        if self._phase == "SEARCHING":
            topic = self.mem_read("topic")
            result = self.use_tool("web_search", query=topic)
            if result.ok:
                findings = self.mem_read("findings") or []
                snippets = [r["snippet"] for r in result.value.get("results", [])]
                findings.extend(snippets)
                self.mem_write("findings", findings)

                # Write to filesystem
                self.write_file(
                    f"/agents/{self.agent_id}/workspace/findings.txt",
                    "\n".join(findings)
                )

                # Announce findings
                self.send_message(
                    BROADCAST_ALL,
                    MessageType.STATUS,
                    {"agent": self.agent_id, "findings_count": len(findings)},
                )
                self._phase = "DONE"

        self.yield_cpu()

    def on_terminate(self) -> None:
        findings = self.mem_read("findings") or []
        self.log(f"Terminating with {len(findings)} findings recorded.")
```

---

## Related Pages

- [LLM Integration](llm.md) — `LLMAgent` which uses LLM inference instead of hard-coded logic
- [Supervision Trees](supervision.md) — `SupervisorAgent` for fault-tolerant hierarchies
- [Custom Agents Guide](custom.md) — patterns for common agent designs
- [IPC](../architecture/ipc.md) — message types and mailbox mechanics
- [Memory](../architecture/memory.md) — private and shared memory
