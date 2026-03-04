# Kernel

The `Kernel` class in `kernel.py` is the central coordinator of Battousai. It owns all subsystems, manages the agent lifecycle, drives the event loop, and dispatches syscalls.

---

## Class Reference

```python
class Kernel:
    VERSION = "0.1.0"

    def __init__(self, max_ticks: int = 50, debug: bool = False) -> None
    def boot(self) -> None
    def spawn_agent(self, agent_class, name, priority=5, **kwargs) -> str
    def kill_agent(self, agent_id: str) -> bool
    def tick(self) -> None
    def run(self, ticks: Optional[int] = None) -> None
    def halt(self) -> None
    def system_report(self) -> str
```

---

## Boot Sequence

`kernel.boot()` initialises all subsystems in the correct dependency order and creates the standard directory tree:

```python
kernel = Kernel(max_ticks=50, debug=False)
kernel.boot()
```

Boot order:

```
1. Logger              — structured log output; min_level=DEBUG if debug else INFO
2. Filesystem          — inject logger reference; create /agents, /shared, /shared/results,
                         /system, /system/logs
3. Memory Manager      — create the global shared region (max_keys=1024)
4. IPC Manager         — online (stateless until agents register)
5. Scheduler           — online (empty ready queues)
6. Tool Manager        — inject filesystem reference
7. register_builtin_tools() — calculator, web_search, code_executor, file_reader, file_writer
```

After `boot()`, `kernel._booted = True`. Calling `spawn_agent` before `boot()` raises `KernelPanic`.

---

## Agent Lifecycle

### Spawning

```python
agent_id = kernel.spawn_agent(
    agent_class=MyAgent,
    name="MyAgent",
    priority=5,
    # extra kwargs forwarded to agent.__init__()
    custom_param="value",
)
```

Internally, `spawn_agent`:
1. Generates a unique `agent_id` in the format `{name}_{counter:04d}` (e.g. `"myagent_0003"`)
2. Instantiates `agent_class(name=name, priority=priority, **kwargs)`
3. Sets `agent.kernel = self` and `agent.agent_id = agent_id`
4. Registers with IPC manager (creates mailbox)
5. Allocates memory space (`agent.memory_allocation` keys)
6. Creates `/agents/{agent_id}/` and `/agents/{agent_id}/workspace/` directories
7. Adds process descriptor to scheduler with `agent.time_slice`
8. Calls `agent.on_spawn()`

### Killing

```python
ok = kernel.kill_agent("myagent_0003")
```

`kill_agent`:
1. Calls `agent.on_terminate()`
2. Marks process as `TERMINATED` in scheduler
3. Unregisters from IPC (destroys mailbox)
4. Deletes memory space
5. Removes from `kernel._agents`

### Collection

Agents that set their process state to `TERMINATED` (e.g. by calling `self.syscall("kill_agent", target_id=self.agent_id)`) are collected at the end of each tick automatically.

---

## The Tick Loop

`kernel.run()` calls `kernel.tick()` repeatedly:

```python
kernel.run()          # runs for self.max_ticks iterations
kernel.run(ticks=10)  # override for a specific number
kernel.halt()         # gracefully stop after current tick
```

Each `tick()` call:

```python
def tick(self) -> None:
    self._tick += 1
    # 1. Update tick counter in all subsystems (logger, filesystem, tools)

    # 2. Snapshot all READY agents at the START of this tick, in priority order
    #    (agents spawned mid-tick don't run until next tick)
    ready_this_tick = [proc.agent_id for prio in range(10) for proc in queues[prio]
                       if proc.state == READY]

    # 3. Run each ready agent exactly once
    for agent_id in ready_this_tick:
        proc.state = RUNNING
        agent._tick(current_tick)     # calls agent.think(tick)
        proc.state = READY            # unless agent yielded/terminated

    # 4. Collect terminated agents (remove from kernel, IPC, memory)

    # 5. Memory GC (evict expired SHORT_TERM entries)
```

!!! note "Agents spawned mid-tick"
    If a `think()` call spawns a new agent, that agent enters READY state but will **not** run until the *next* tick. The snapshot at the start of each tick prevents newly spawned agents from running in the same tick.

---

## Syscall Dispatch

Agents request OS services via `self.syscall(name, **kwargs)`, which routes through `kernel._dispatch_syscall`:

```python
result = self.syscall("access_tool", tool_name="calculator", args={"expression": "2+2"})
# result.ok == True
# result.value == "4"
```

The complete list of available syscalls:

| Syscall | Description |
|---|---|
| `spawn_agent` | Create and register a new agent; returns `agent_id` |
| `kill_agent` | Terminate an agent by ID |
| `send_message` | Post a message to another agent's mailbox |
| `read_inbox` | Drain all pending messages from the caller's mailbox |
| `read_memory` | Read a value from the caller's private memory |
| `write_memory` | Write a value to the caller's private memory |
| `access_tool` | Execute a registered tool |
| `list_agents` | Return IDs of all living agents |
| `get_status` | Return a full system-wide metrics snapshot |
| `yield_cpu` | Voluntarily give up the remainder of this tick's CPU slice |
| `write_file` | Write a file to the virtual filesystem |
| `read_file` | Read a file from the virtual filesystem |
| `list_dir` | List a directory in the virtual filesystem |
| `publish_topic` | Publish a value to the IPC bulletin board |
| `subscribe` | Subscribe to a bulletin board topic |

Agents call these indirectly through convenience wrappers on the `Agent` base class (`self.send_message(...)`, `self.mem_write(...)`, etc.). See the [Agent API](../agents/api.md) for the wrapper reference.

---

## `SyscallResult`

Every syscall returns a `SyscallResult`:

```python
@dataclass
class SyscallResult:
    ok: bool
    value: Any = None
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.ok
```

```python
result = self.use_tool("calculator", expression="1/0")
if result.ok:
    print(result.value)
else:
    print(f"Error: {result.error}")
```

---

## Metrics and Introspection

The kernel tracks key counters throughout its lifetime:

```python
kernel._tick          # current tick number
kernel._spawn_count   # total agents ever spawned
kernel._kill_count    # total agents ever killed
kernel._syscall_count # total syscalls dispatched
```

The `get_status` syscall (or `kernel.system_report()`) returns a comprehensive snapshot:

```python
# Via syscall from within an agent:
status = self.get_status()
metrics = status.value
# metrics["tick"], metrics["agent_count"], metrics["scheduler_stats"],
# metrics["ipc_stats"], metrics["memory_stats"], metrics["tool_stats"], ...

# Or directly for external code:
report_str = kernel.system_report()
print(report_str)
```

---

## `KernelPanic`

A `KernelPanic` exception is raised for unrecoverable OS errors:

```python
from battousai.kernel import KernelPanic

try:
    kernel.spawn_agent(MyAgent, "test")  # before boot()
except KernelPanic as e:
    print(e)  # Cannot spawn agents before boot()
```

---

## Full Example

```python
from battousai.kernel import Kernel
from battousai.agent import MonitorAgent, CoordinatorAgent
from battousai.ipc import MessageType

# Create kernel with 100 ticks and debug output
kernel = Kernel(max_ticks=100, debug=True)
kernel.boot()

# Spawn agents
monitor_id = kernel.spawn_agent(MonitorAgent, name="SysMonitor", priority=7)
coord_id = kernel.spawn_agent(CoordinatorAgent, name="Coordinator", priority=2)

# Seed initial task
kernel.ipc.create_message(
    sender_id="kernel",
    recipient_id=coord_id,
    message_type=MessageType.TASK,
    payload={"task": "Research AI safety"},
    timestamp=0,
)

# Run
kernel.run()

# Print report
print(kernel.system_report())
```

---

## Related Pages

- [Scheduler](scheduler.md) — how agents are prioritised and scheduled
- [IPC](ipc.md) — message routing between agents
- [Agent API](../agents/api.md) — syscall wrappers on the Agent base class
