# Scheduler

The `Scheduler` class in `scheduler.py` manages the execution order of agent processes in Battousai.

---

## Policy

**Priority-based preemptive scheduling with round-robin within the same priority band.**

- **Priority levels:** 0–9 (0 = highest, 9 = lowest)
- Real-time system agents: priority 0
- Normal worker agents: priority 4–5 (default 5)
- Background monitoring: priority 7–9
- **Time slicing:** each agent gets a configurable `time_slice` (default 3 ticks)
- **Round-robin:** agents at the same priority level rotate in FIFO order
- **Preemption:** a higher-priority READY agent arriving mid-slice immediately preempts the current agent

---

## Agent States

```python
class AgentState(Enum):
    READY      = auto()  # Eligible to run; waiting for CPU time
    RUNNING    = auto()  # Currently executing its think() method this tick
    WAITING    = auto()  # Blocked on a message/reply (will not be scheduled)
    BLOCKED    = auto()  # Blocked on a resource or I/O (future use)
    TERMINATED = auto()  # Execution complete; will be collected and removed
```

State transitions:

```
                  ┌─────────────────────────────┐
                  │                             │
spawn ──► READY ──┼──► RUNNING ──► READY        │ (normal tick)
                  │        │                    │
                  │        ├──► WAITING ─────────┘ (unblock_process)
                  │        │
                  │        └──► TERMINATED (kill_agent / self-kill)
                  │
                  └─► TERMINATED (kill_agent before first run)
```

---

## Class Reference

```python
class Scheduler:
    NUM_PRIORITIES = 10

    def __init__(self, default_time_slice: int = 3) -> None

    # Process registration
    def add_process(self, agent_id, name, priority, time_slice=None, spawn_tick=0) -> ProcessDescriptor
    def remove_process(self, agent_id: str) -> bool
    def get_process(self, agent_id: str) -> Optional[ProcessDescriptor]

    # State transitions
    def block_process(self, agent_id: str, wait_for: Optional[str] = None) -> None
    def unblock_process(self, agent_id: str) -> None
    def terminate_process(self, agent_id: str) -> None
    def reprioritize(self, agent_id: str, new_priority: int) -> None

    # Scheduling
    def tick(self, current_tick: int) -> Optional[ProcessDescriptor]

    # Introspection
    def ready_queue_snapshot(self) -> List[Tuple[int, List[str]]]
    def all_processes(self) -> Dict[str, ProcessDescriptor]
    def get_state(self, agent_id: str) -> Optional[AgentState]
    def stats(self) -> Dict[str, object]
```

---

## ProcessDescriptor

Every agent registered with the scheduler gets a `ProcessDescriptor`:

```python
@dataclass
class ProcessDescriptor:
    agent_id: str
    name: str
    priority: int           # 0 (highest) – 9 (lowest)
    state: AgentState = AgentState.READY
    time_slice: int = 3     # ticks per scheduling turn
    remaining_ticks: int    # ticks left in current slice
    ticks_run: int = 0      # total CPU ticks consumed
    spawn_tick: int = 0     # tick at which this process was created
    yield_requested: bool   # agent called yield_cpu()
    wait_for: Optional[str] # correlation_id if waiting for reply

    def yield_cpu(self) -> None   # signal voluntary yield
    def reset_slice(self) -> None # reset remaining_ticks = time_slice
```

---

## Priority Bands

| Priority | Typical use |
|---|---|
| 0 | OS-critical real-time agents (reserved) |
| 1–2 | Supervisors, coordinators |
| 3–4 | Workers with time-sensitive tasks |
| 5 | Normal worker agents (default) |
| 6 | Background data processors |
| 7–8 | Monitoring, health checking |
| 9 | Lowest-priority background tasks |

Example: with four agents at priorities 2, 4, 4, 7, the scheduler will always run the priority-2 agent first, then alternate between the two priority-4 agents (round-robin), then finally run the priority-7 agent.

---

## Round-Robin Within a Band

Agents at the same priority share CPU time round-robin. The scheduler maintains a `deque` per priority level:

```
Priority 4 queue: [Worker-A, Worker-B, Worker-C]
                        ↑ runs first
                   After Worker-A finishes its tick:
                   [Worker-B, Worker-C, Worker-A]
```

After each agent's turn, it is rotated to the back of its priority queue. This ensures fairness among equal-priority agents.

---

## Time Slicing

Each agent has a `time_slice` property (default 3). The kernel tracks `remaining_ticks` per tick:

```python
# In the kernel tick loop:
proc.remaining_ticks -= 1
```

When `remaining_ticks` reaches 0, the agent's slice is exhausted and it is preempted back to READY. Its slice is reset via `proc.reset_slice()`.

!!! note "Current tick behavior"
    In the current implementation, the kernel runs each READY agent **exactly once per tick** regardless of `time_slice`. The `time_slice` and `remaining_ticks` fields are maintained for future use where agents might run for multiple ticks before yielding.

---

## Voluntary Yield

An agent that finishes its work before its time slice expires should call `self.yield_cpu()`:

```python
def think(self, tick: int) -> None:
    # ... do work ...
    self.yield_cpu()  # signal: I'm done for this tick
```

This increments `scheduler.voluntary_yields` and rotates the agent to the back of its priority queue immediately.

---

## Preemption

If a higher-priority agent becomes READY mid-tick (e.g. spawned by another agent), it preempts the current agent:

```
Before: priority-5 agent is RUNNING
Event:  priority-2 agent spawned → enters READY state
Result: priority-5 agent is placed back at front of its queue (retains remaining slice)
        priority-2 agent runs next
```

Preemptions are tracked: `scheduler.preemptions`.

---

## Blocking and Unblocking

For future use (e.g. request/reply with correlation IDs), agents can be blocked:

```python
scheduler.block_process("worker_0001", wait_for="corr_id_abc")
# Process moves from READY to WAITING; removed from ready queue

scheduler.unblock_process("worker_0001")
# Process returns to READY; added back to priority queue
```

---

## Introspection

```python
# Get state of a specific agent
state = kernel.scheduler.get_state("worker_0001")
# AgentState.READY

# Snapshot of all ready queues
snapshot = kernel.scheduler.ready_queue_snapshot()
# [(2, ['coordinator_0002']), (4, ['worker_0003', 'worker_0004']), (7, ['sysmonitor_0001'])]

# Full stats dict
stats = kernel.scheduler.stats()
# {
#   "total_processes": 4,
#   "state_counts": {"READY": 4, "RUNNING": 0, "WAITING": 0, "BLOCKED": 0, "TERMINATED": 0},
#   "preemptions": 0,
#   "voluntary_yields": 196,
#   "total_scheduled": 200,
#   "ready_queue": [...],
# }
```

---

## Dynamic Reprioritization

An agent can change its own priority (or another agent's priority via a syscall):

```python
kernel.scheduler.reprioritize("worker_0003", new_priority=2)
```

The change takes effect on the next scheduling decision. The process is moved from its old queue to the new one.

---

## Related Pages

- [Kernel](kernel.md) — how the kernel drives the scheduler each tick
- [Agent API](../agents/api.md) — `yield_cpu()` and how agents interact with scheduling
