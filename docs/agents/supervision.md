# Supervision Trees

The `supervisor.py` module provides Erlang/OTP-style fault-tolerant agent hierarchies. A `SupervisorAgent` monitors its children and automatically restarts them on failure.

---

## Core Concepts

A supervisor is an agent whose job is to watch over other agents (its "children") and restart them if they crash. The restart behaviour is controlled by the **strategy** and the **restart type** of each child.

```
SupervisorAgent (strategy=ONE_FOR_ONE)
├── Worker-A  [PERMANENT]  → crashes → restarted immediately
├── Worker-B  [TRANSIENT]  → normal exit → not restarted
└── Worker-C  [TEMPORARY]  → crashes → never restarted
```

---

## `RestartStrategy` Enum

```python
class RestartStrategy(Enum):
    ONE_FOR_ONE  = auto()  # Restart only the crashed child
    ONE_FOR_ALL  = auto()  # Restart ALL children on any crash
    REST_FOR_ONE = auto()  # Restart crashed child + all children after it (pipelines)
```

| Strategy | Behavior | Use Case |
|---|---|---|
| `ONE_FOR_ONE` | Only the crashed child is restarted | Independent parallel workers |
| `ONE_FOR_ALL` | All children are terminated and restarted | Tightly coupled workers |
| `REST_FOR_ONE` | Crashed child and all children after it in spec order | Pipeline stages |

---

## `RestartType` Enum

Per-child policy for when to restart:

```python
class RestartType(Enum):
    PERMANENT  = auto()  # Always restart (regardless of how it exited)
    TRANSIENT  = auto()  # Only restart on abnormal exit (crash or kill)
    TEMPORARY  = auto()  # Never restart (one-shot workers)
```

---

## `ChildSpec` Dataclass

A blueprint describing how to spawn (and respawn) a child agent:

```python
@dataclass
class ChildSpec:
    agent_class: Type[Agent]       # The Agent subclass to instantiate
    name: str                      # Unique name within this supervisor
    priority: int = 5              # Scheduler priority
    restart_type: RestartType = RestartType.PERMANENT
    kwargs: Dict[str, Any] = field(default_factory=dict)  # Extra __init__ args
    shutdown_timeout: int = 3      # Ticks to wait for graceful shutdown
```

Example:
```python
ChildSpec(
    agent_class=WorkerAgent,
    name="DataWorker",
    priority=4,
    restart_type=RestartType.PERMANENT,
    kwargs={"subtask": {"description": "process batch", "queries": ["q1"]}},
    shutdown_timeout=5,
)
```

---

## `SupervisorAgent`

```python
class SupervisorAgent(Agent):
    def __init__(
        self,
        name: str = "Supervisor",
        priority: int = 2,
        strategy: RestartStrategy = RestartStrategy.ONE_FOR_ONE,
        children: Optional[List[ChildSpec]] = None,
        max_restarts: int = 5,
        window_ticks: int = 20,
    ) -> None
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `name` | `"Supervisor"` | Agent name |
| `priority` | `2` | Higher than workers so health checks run first |
| `strategy` | `ONE_FOR_ONE` | Restart strategy |
| `children` | `[]` | List of `ChildSpec` objects |
| `max_restarts` | `5` | Max restarts in `window_ticks` before escalation |
| `window_ticks` | `20` | Sliding window size for restart counting |

### Introspection Methods

```python
# Get status of all children
status = supervisor.child_status()
# {
#   "Worker-1": {"spec": "...", "agent_id": "worker_0003", "alive": True, "restart_type": "PERMANENT"},
#   "Worker-2": {"spec": "...", "agent_id": None, "alive": False, "restart_type": "TEMPORARY"},
# }

# Get restart history
history = supervisor.restart_history()
# [{"child_name": "Worker-1", "tick": 7}, {"child_name": "Worker-1", "tick": 15}]
```

---

## Restart Intensity

If a supervisor exceeds `max_restarts` within `window_ticks`, it **terminates itself** and escalates the fault to its own parent supervisor (if any):

```
max_restarts=3, window_ticks=10:
  Tick 5:  Worker-A crashes → restart #1
  Tick 7:  Worker-A crashes → restart #2
  Tick 9:  Worker-A crashes → restart #3 → INTENSITY EXCEEDED
  → Supervisor kills all children and terminates itself
  → Parent supervisor (if any) sees the supervisor crash and applies its strategy
```

```python
SupervisorAgent(
    strategy=RestartStrategy.ONE_FOR_ONE,
    max_restarts=5,     # allow up to 5 restarts
    window_ticks=10,    # within any 10-tick window
    children=[...],
)
```

---

## Simple Supervisor Example

```python
from battousai.kernel import Kernel
from battousai.supervisor import SupervisorAgent, ChildSpec, RestartStrategy, RestartType
from battousai.agent import WorkerAgent

kernel = Kernel(max_ticks=50)
kernel.boot()

# Spawn a supervisor with two workers
sup_id = kernel.spawn_agent(
    SupervisorAgent,
    name="WorkerPool",
    priority=2,
    strategy=RestartStrategy.ONE_FOR_ONE,
    children=[
        ChildSpec(WorkerAgent, name="Worker-1", priority=4,
                  restart_type=RestartType.PERMANENT),
        ChildSpec(WorkerAgent, name="Worker-2", priority=4,
                  restart_type=RestartType.PERMANENT),
    ],
    max_restarts=5,
    window_ticks=20,
)

kernel.run()
```

Expected output:
```
[tick=0001] INFO  workerpool_0001  [WorkerPool] SupervisorAgent online. Strategy=ONE_FOR_ONE, children=2, max_restarts=5/20t
[tick=0001] INFO  workerpool_0001  [WorkerPool] Spawning 2 initial children...
[tick=0001] INFO  workerpool_0001  [WorkerPool] Spawned child 'Worker-1' → worker-1_0002
[tick=0001] INFO  workerpool_0001  [WorkerPool] Spawned child 'Worker-2' → worker-2_0003
```

---

## `build_supervision_tree` — Multi-Level Hierarchies

For complex nested supervision trees, use the convenience factory:

```python
from battousai.supervisor import build_supervision_tree, SupervisorAgent, RestartStrategy
from battousai.agent import WorkerAgent, MonitorAgent

kernel.boot()

root_id = build_supervision_tree(kernel, {
    "name": "RootSupervisor",
    "priority": 1,
    "strategy": "ONE_FOR_ALL",   # or RestartStrategy.ONE_FOR_ALL
    "max_restarts": 3,
    "window_ticks": 15,
    "children": [
        {
            "name": "WorkerPool",
            "class": SupervisorAgent,
            "priority": 2,
            "strategy": "ONE_FOR_ONE",
            "children": [
                {"name": "Worker-1", "class": WorkerAgent, "priority": 4},
                {"name": "Worker-2", "class": WorkerAgent, "priority": 4},
                {"name": "Worker-3", "class": WorkerAgent, "priority": 4},
            ],
        },
        {
            "name": "SysMonitor",
            "class": MonitorAgent,
            "priority": 7,
            "restart_type": "PERMANENT",
        },
    ],
})
```

This creates the following tree:

```
RootSupervisor  (ONE_FOR_ALL)
├── WorkerPool  (ONE_FOR_ONE)
│   ├── Worker-1  [PERMANENT]
│   ├── Worker-2  [PERMANENT]
│   └── Worker-3  [PERMANENT]
└── SysMonitor  [PERMANENT]
```

`build_supervision_tree` returns the `agent_id` of the root supervisor.

---

## Supervisor with ChildSpec Objects

You can also use `ChildSpec` objects directly in the `children` list:

```python
from battousai.supervisor import SupervisorAgent, ChildSpec, RestartStrategy, RestartType
from battousai.agent import WorkerAgent, MonitorAgent

sup_id = kernel.spawn_agent(
    SupervisorAgent,
    name="MixedSupervisor",
    priority=2,
    strategy=RestartStrategy.REST_FOR_ONE,
    children=[
        ChildSpec(
            agent_class=WorkerAgent,
            name="DataIngestor",
            priority=3,
            restart_type=RestartType.PERMANENT,
            kwargs={"subtask": {"description": "ingest data", "queries": ["data sources"]}},
        ),
        ChildSpec(
            agent_class=WorkerAgent,
            name="DataProcessor",
            priority=4,
            restart_type=RestartType.TRANSIENT,  # only restart on crash
        ),
        ChildSpec(
            agent_class=MonitorAgent,
            name="ResultPublisher",
            priority=5,
            restart_type=RestartType.TEMPORARY,  # one-shot
        ),
    ],
)
```

With `REST_FOR_ONE`: if `DataIngestor` (index 0) crashes, all three are restarted; if `DataProcessor` (index 1) crashes, `DataProcessor` and `ResultPublisher` are restarted; if `ResultPublisher` (index 2) crashes, only it is restarted.

---

## `SupervisorTree` for Display

Visualise the hierarchy:

```python
from battousai.supervisor import SupervisorTree

tree = SupervisorTree(kernel=kernel)
tree.add_node(root_sup_id, label="RootSupervisor", parent=None)
tree.add_node(worker_pool_id, label="WorkerPool", parent=root_sup_id)
tree.add_node(worker_1_id, label="Worker-1 [PERMANENT]", parent=worker_pool_id)
tree.add_node(worker_2_id, label="Worker-2 [PERMANENT]", parent=worker_pool_id)

print(tree.render())
```

```
RootSupervisor [ONE_FOR_ALL, 2/2 alive]
    ├── WorkerPool [ONE_FOR_ONE, 2/2 alive]
    │       ├── Worker-1 [pid=worker-1_0002]
    │       └── Worker-2 [pid=worker-2_0003]
    └── SysMonitor [pid=sysmonitor_0004]
```

---

## Related Pages

- [Agent API](api.md) — the `Agent` base class that `SupervisorAgent` extends
- [Custom Agents](custom.md) — writing workers that survive supervision
- [Architecture Overview](../architecture/overview.md) — where supervision fits in the stack
