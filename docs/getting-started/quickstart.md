# Quickstart

This guide gets you from zero to a running multi-agent system in under 5 minutes.

---

## 1. Boot the Kernel

The `Kernel` is the nucleus of Battousai. Create one, call `boot()`, and the OS initialises all subsystems:

```python
from battousai.kernel import Kernel

kernel = Kernel(max_ticks=50, debug=False)
kernel.boot()
```

```
[tick=0000] SYSTEM  kernel   Battousai v0.2.0 booting...
[tick=0000] SYSTEM  kernel   Filesystem initialised (/agents, /shared, /system/logs)
[tick=0000] SYSTEM  kernel   Memory manager online (global shared region created)
[tick=0000] SYSTEM  kernel   Tools registered: ['calculator', 'code_executor', 'file_reader', 'file_writer', 'web_search']
[tick=0000] SYSTEM  kernel   Boot sequence complete. Ready.
```

`max_ticks` controls how many iterations `kernel.run()` will execute. Set `debug=True` for verbose output.

---

## 2. Spawn Built-in Agents

Battousai ships with three ready-to-use agents: `CoordinatorAgent`, `WorkerAgent`, and `MonitorAgent`.

```python
from battousai.agent import CoordinatorAgent, MonitorAgent

# Spawn a monitor (low priority — observes passively)
monitor_id = kernel.spawn_agent(MonitorAgent, name="SysMonitor", priority=7)

# Spawn a coordinator (moderate-high priority — directs workers)
coord_id = kernel.spawn_agent(CoordinatorAgent, name="Coordinator", priority=2)

print(f"Monitor: {monitor_id}")    # sysmonitor_0001
print(f"Coordinator: {coord_id}")  # coordinator_0002
```

`spawn_agent` returns the agent's unique ID string (e.g. `"coordinator_0002"`). The kernel registers the agent with the scheduler, IPC manager, memory manager, and filesystem.

---

## 3. Send Messages

Agents communicate via typed messages. Send a task to the coordinator:

```python
from battousai.ipc import MessageType

# The kernel can inject messages directly via IPC
kernel.ipc.create_message(
    sender_id="kernel",
    recipient_id=coord_id,
    message_type=MessageType.TASK,
    payload={"task": "Research and summarize quantum computing"},
    timestamp=0,
)
```

The coordinator will receive this message on its first `think()` call during `kernel.run()`.

---

## 4. Run the Event Loop

```python
kernel.run()  # runs for max_ticks iterations
```

Or run for a specific number of ticks:

```python
kernel.run(ticks=10)
```

Each tick:
1. All READY agents execute their `think()` method once, in priority order
2. Memory garbage collection runs
3. Terminated agents are collected

---

## 5. Read the System Report

After the run, get a formatted summary of everything that happened:

```python
report = kernel.system_report()
print(report)
```

```
======================================================================
  Battousai SYSTEM REPORT  —  v0.2.0
======================================================================
  Total ticks run       : 50
  Agents spawned        : 4
  Agents killed         : 0
  Agents alive          : 4
  Syscalls dispatched   : 209

  IPC
    Messages sent       : 5
    Messages dropped    : 0
    Bulletin topics     : 1

  Scheduler
    READY               : 4
    Preemptions         : 0
    Voluntary yields    : 196
    Total scheduled     : 196

  Tools
    Total tool calls    : 5
    web_search          : 5 calls
...
```

---

## Writing a Custom Agent

Subclass `Agent` and implement `think(tick)`:

```python
from battousai.agent import Agent
from battousai.memory import MemoryType

class CounterAgent(Agent):
    def __init__(self):
        super().__init__(name="Counter", priority=5, memory_allocation=64)

    def on_spawn(self) -> None:
        """Called once when the kernel registers this agent."""
        self.log("CounterAgent online!")
        self.mem_write("count", 0)

    def think(self, tick: int) -> None:
        """Called every tick the scheduler grants CPU time."""
        # Read from private memory
        count = self.mem_read("count") or 0

        # Read pending messages
        messages = self.read_inbox()
        for msg in messages:
            self.log(f"Received: {msg.message_type.name} from {msg.sender_id}")

        # Use a tool
        result = self.use_tool("calculator", expression=f"{count} + 1")
        if result.ok:
            new_count = int(result.value)
            self.mem_write("count", new_count)
            self.log(f"Tick {tick}: count = {new_count}")

        # Write to the filesystem
        self.write_file(
            f"/agents/{self.agent_id}/workspace/state.txt",
            f"tick={tick}, count={count}"
        )

        # Yield remaining CPU slice
        self.yield_cpu()

    def on_terminate(self) -> None:
        """Called before the kernel removes this agent."""
        final = self.mem_read("count") or 0
        self.log(f"Shutting down. Final count: {final}")
```

Register and run:

```python
from battousai.kernel import Kernel

kernel = Kernel(max_ticks=10)
kernel.boot()
kernel.spawn_agent(CounterAgent, name="Counter", priority=5)
kernel.run()
```

---

## Sending Messages Between Agents

```python
from battousai.agent import Agent
from battousai.ipc import MessageType, BROADCAST_ALL

class SenderAgent(Agent):
    def __init__(self):
        super().__init__(name="Sender", priority=4)
        self._sent = False

    def think(self, tick: int) -> None:
        if not self._sent:
            # Unicast to a specific agent
            agents = self.list_agents()
            for aid in agents:
                if "receiver" in aid:
                    self.send_message(aid, MessageType.TASK, {"job": "process data"})
                    self.log(f"Sent task to {aid}")

            # Broadcast to all agents
            self.send_message(BROADCAST_ALL, MessageType.BROADCAST, "Hello everyone!")
            self._sent = True
        self.yield_cpu()


class ReceiverAgent(Agent):
    def __init__(self):
        super().__init__(name="Receiver", priority=5)

    def think(self, tick: int) -> None:
        messages = self.read_inbox()
        for msg in messages:
            self.log(f"Got {msg.message_type.name}: {msg.payload}")
        self.yield_cpu()
```

---

## Using Tools

```python
class ToolDemoAgent(Agent):
    def think(self, tick: int) -> None:
        if tick == 1:
            # Calculator
            r = self.use_tool("calculator", expression="sqrt(144)")
            self.log(f"sqrt(144) = {r.value}")  # 12.0

            # Web search (simulated)
            r = self.use_tool("web_search", query="quantum computing basics")
            if r.ok:
                results = r.value.get("results", [])
                self.log(f"Search returned {len(results)} result(s)")

            # Write a file
            self.write_file("/agents/self/notes.txt", "Important finding: ...")

            # Read it back
            r = self.read_file("/agents/self/notes.txt")
            self.log(f"File contents: {r.value}")

        self.yield_cpu()
```

---

## Next Steps

- [Demo Walkthrough](demo.md) — see the built-in research scenario in detail
- [Agent API](../agents/api.md) — full reference for all agent methods
- [Architecture Overview](../architecture/overview.md) — how all layers fit together
- [Custom Agents Guide](../agents/custom.md) — patterns for more complex agents
