# Demo Walkthrough

Running `python -m battousai.main` boots the OS and executes a multi-agent research scenario from start to finish. This page walks through every tick.

---

## The Scenario

**Goal:** Research and summarise quantum computing using a team of autonomous agents.

The scenario involves three roles:

| Agent | Class | Priority | Role |
|---|---|---|---|
| `SysMonitor` | `MonitorAgent` | 7 (low) | Passively samples metrics and publishes to bulletin board |
| `Coordinator` | `CoordinatorAgent` | 2 (high) | Decomposes goal, spawns workers, synthesises results |
| `Worker-1` | `WorkerAgent` | 4 | Researches fundamentals and milestones |
| `Worker-2` | `WorkerAgent` | 4 | Researches applications and challenges |

Worker agents are spawned dynamically by the Coordinator — they don't exist at startup.

---

## Tick-by-Tick Breakdown

### Tick 0 — Boot and Setup

`main.py` runs before the event loop starts:

```python
kernel = Kernel(max_ticks=50)
kernel.boot()

monitor_id = kernel.spawn_agent(MonitorAgent, name="SysMonitor", priority=7)
coord_id = kernel.spawn_agent(CoordinatorAgent, name="Coordinator", priority=2)

# Seed the coordinator's inbox with the top-level task
kernel.ipc.create_message(
    sender_id="kernel",
    recipient_id=coord_id,
    message_type=MessageType.TASK,
    payload={"task": "Research and summarize quantum computing"},
    timestamp=0,
)
```

Expected output:

```
[tick=0000] SYSTEM  kernel               Battousai v0.2.0 booting...
[tick=0000] SYSTEM  kernel               Filesystem initialised (/agents, /shared, /system/logs)
[tick=0000] SYSTEM  kernel               Memory manager online (global shared region created)
[tick=0000] SYSTEM  kernel               Tools registered: ['calculator', 'code_executor', 'file_reader', 'file_writer', 'web_search']
[tick=0000] SYSTEM  kernel               Boot sequence complete. Ready.
[tick=0000] INFO    sysmonitor_0001      [SysMonitor] Monitoring system. Sampling every 5 ticks.
[tick=0000] INFO    coordinator_0002     [Coordinator] Online. Awaiting initial task assignment.
```

### Tick 1 — Coordinator Reads Task and Spawns Workers

The Coordinator runs first (priority 2 < 7). Its `think()` method:

1. Drains its inbox — finds the TASK message from the kernel
2. Stores the task to private memory: `self.mem_write("current_task", task)`
3. Calls `_decompose_and_spawn()` which spawns two WorkerAgents via `self.spawn_child(WorkerAgent, ...)`
4. Sends each worker a TASK message with their subtask payload
5. Transitions to `COLLECTING` phase

```
[tick=0001] INFO    coordinator_0002     [Coordinator] Received task: 'Research and summarize quantum computing'
[tick=0001] INFO    coordinator_0002     [Coordinator] Decomposing task into subtasks...
[tick=0001] SYSTEM  kernel               Spawned agent 'worker-1_0003' (class=WorkerAgent, priority=4)
[tick=0001] SYSTEM  kernel               Spawned agent 'worker-2_0004' (class=WorkerAgent, priority=4)
[tick=0001] INFO    coordinator_0002     [Coordinator] Collecting results from 2 workers...
```

Subtasks assigned:

- **Worker-1:** "Research fundamentals and technical landscape of quantum computing"
  - Queries: `["quantum computing basics", "quantum supremacy milestones", "quantum computing challenges"]`
- **Worker-2:** "Research applications and future outlook of quantum computing"
  - Queries: `["quantum computing applications", "quantum computing challenges"]`

### Tick 2 — Workers Execute Their First Query

Newly spawned workers run for the first time. Each worker:

1. Reads its inbox — finds the TASK message from the coordinator
2. Stores the subtask to memory
3. Executes the first query via `self.use_tool("web_search", query=...)`

```
[tick=0002] INFO    worker-1_0003        [Worker-1] Task received: Research fundamentals...
[tick=0002] INFO    worker-1_0003        [Worker-1] Searching: 'quantum computing basics'
[tick=0002] INFO    worker-2_0004        [Worker-2] Task received: Research applications...
[tick=0002] INFO    worker-2_0004        [Worker-2] Searching: 'quantum computing applications'
```

The `web_search` tool returns a structured dict with `results[].snippet`. Workers extract snippets and append them to `self._findings`.

### Tick 3 — Second Queries

Workers advance to their second queries (one query per tick):

```
[tick=0003] INFO    worker-1_0003        [Worker-1] Searching: 'quantum supremacy milestones'
[tick=0003] INFO    worker-2_0004        [Worker-2] Searching: 'quantum computing challenges'
```

### Tick 4 — Final Queries and Results Sent

Worker-1 runs its third query. Both workers finish all queries and send RESULT messages back to the Coordinator:

```
[tick=0004] INFO    worker-1_0003        [Worker-1] Searching: 'quantum computing challenges'
[tick=0004] INFO    worker-1_0003        [Worker-1] Results sent to coordinator_0002
[tick=0004] INFO    worker-2_0004        [Worker-2] Results sent to coordinator_0002
```

The result payload contains:
```python
{
    "subtask_description": "Research fundamentals...",
    "findings": [
        {"query": "quantum computing basics", "result": "...snippet..."},
        ...
    ],
    "worker_id": "worker-1_0003",
    "completed_at_tick": 4,
}
```

### Tick 5 — Monitor Samples Metrics

The Monitor runs every 5 ticks (`tick % 5 == 0`). It calls `self.get_status()` and publishes to the bulletin board topic `"system.health"`:

```
[tick=0005] INFO    sysmonitor_0001      [SysMonitor] Health tick=5 | agents=4 | msgs_sent=5
```

### Tick 6 — Coordinator Synthesises and Writes Summary

The Coordinator has collected results from both workers (detected in `think()` when `_pending_results` is fully populated). It:

1. Calls `_synthesise_results()`
2. Builds a formatted text report combining both workers' findings
3. Writes to `/shared/results/summary.txt` via `self.write_file(...)`

```
[tick=0006] INFO    coordinator_0002     [Coordinator] All results received. Synthesising summary...
[tick=0006] INFO    coordinator_0002     [Coordinator] Summary written to /shared/results/summary.txt
[tick=0006] INFO    coordinator_0002     [Coordinator] Task complete. Entering idle state.
```

### Ticks 7–50 — Idle

All agents are alive but idle. The Coordinator and Workers yield immediately each tick. The Monitor continues sampling every 5 ticks (ticks 10, 15, 20, 25, 30, 35, 40, 45, 50).

---

## End-of-Run Output

After all ticks complete, `main.py` prints:

1. The MonitorAgent's formatted metrics report
2. The full system report from `kernel.system_report()`
3. The contents of `/shared/results/summary.txt`
4. Wall-clock time and tick rate

```
=== MONITOR REPORT ===
  tick=0005 | agents=  4 | msgs=    5 | gc_runs=5
  tick=0010 | agents=  4 | msgs=    5 | gc_runs=10
  ...

======================================================================
  Battousai SYSTEM REPORT  —  v0.2.0
======================================================================
  Total ticks run       : 50
  Agents spawned        : 4
  Agents killed         : 0
  Agents alive          : 4
  Syscalls dispatched   : 209
  ...

  /shared/results/summary.txt:
------------------------------------------------------------
QUANTUM COMPUTING RESEARCH SUMMARY
Generated by Battousai Coordinator Agent at tick 6
Task: Research and summarize quantum computing
...
------------------------------------------------------------
```

---

## What the Demo Demonstrates

| Feature | Where it appears |
|---|---|
| Kernel boot sequence | Tick 0 output |
| Priority scheduling | Coordinator (2) runs before Workers (4) and Monitor (7) |
| Dynamic agent spawning | Workers spawned at tick 1 by Coordinator |
| IPC messaging | TASK messages tick 1, RESULT messages tick 4 |
| Tool usage | `web_search` at ticks 2–4 |
| Memory writes | Workers store task and findings; Coordinator stores summary flag |
| Filesystem writes | Summary written to `/shared/results/summary.txt` |
| Pub/sub bulletin board | Monitor publishes `system.health` at ticks 5, 10, 15... |
| System report | Full stats at end of run |

---

## Next Steps

- [Agent API](../agents/api.md) — understand every method agents can call
- [IPC](../architecture/ipc.md) — how messages flow between agents
- [Custom Agents](../agents/custom.md) — build your own agent for custom scenarios
