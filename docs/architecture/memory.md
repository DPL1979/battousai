# Memory Manager

The `memory.py` module manages memory spaces for agents in Battousai. Every agent gets isolated private storage; shared regions allow coordinated data exchange.

---

## Memory Architecture

```
MemoryManager
├── _agents: Dict[agent_id → AgentMemorySpace]
│   ├── coordinator_0002  (max 512 keys)
│   ├── worker_0003       (max 256 keys)
│   └── ...
└── _shared: Dict[name → SharedMemoryRegion]
    └── "global"  (max 1024 keys, open access)
```

- **Private memory** — isolated per agent; other agents cannot read it without explicit permission
- **Shared regions** — named namespaces any authorized agent can read/write
- **Kernel access** — the kernel always bypasses all access controls

---

## `MemoryType` Enum

```python
class MemoryType(Enum):
    SHORT_TERM = auto()   # Expires after ttl_ticks; for working memory
    LONG_TERM  = auto()   # Persists for the OS session
    SHARED     = auto()   # Stored in a named shared region
```

---

## Memory Errors

```python
class MemoryError(Exception): ...
class MemoryFullError(MemoryError): ...   # agent's allocation exhausted
class MemoryAccessError(MemoryError): ... # unauthorized cross-agent access
class MemoryKeyError(MemoryError): ...    # key not found or expired
```

---

## `MemoryEntry` Dataclass

```python
@dataclass
class MemoryEntry:
    key: str
    value: Any
    memory_type: MemoryType
    created_tick: int
    ttl_ticks: Optional[int] = None   # only for SHORT_TERM
    owner_agent_id: str = ""

    @property
    def expires_at(self) -> Optional[int]  # created_tick + ttl_ticks
    def is_expired(self, current_tick: int) -> bool
```

---

## Private Memory (Per-Agent)

### Writing

```python
# From within an agent's think() method:

# Long-term (persists forever)
self.mem_write("current_task", "Research quantum computing")
self.mem_write("findings", ["result 1", "result 2"])

# Short-term (expires after 10 ticks — working memory)
from battousai.memory import MemoryType
self.mem_write("scratch", intermediate_data, memory_type=MemoryType.SHORT_TERM, ttl=10)
```

The `mem_write` signature:

```python
def mem_write(
    self,
    key: str,
    value: Any,
    memory_type: MemoryType = MemoryType.LONG_TERM,
    ttl: Optional[int] = None,
) -> SyscallResult
```

### Reading

```python
# Returns the value or None if key doesn't exist / expired
task = self.mem_read("current_task")
if task is None:
    self.log("No task assigned yet")
else:
    self.log(f"Working on: {task}")
```

Reading an expired SHORT_TERM key returns `None` (the entry is deleted lazily).

### `AgentMemorySpace` Class

```python
class AgentMemorySpace:
    def __init__(self, agent_id: str, max_keys: int = 256) -> None

    def write(self, key, value, memory_type=LONG_TERM, current_tick=0, ttl_ticks=None) -> MemoryEntry
    def read(self, key: str, current_tick: int = 0) -> Any   # raises MemoryKeyError if missing
    def read_entry(self, key: str, current_tick: int = 0) -> MemoryEntry
    def exists(self, key: str, current_tick: int = 0) -> bool
    def delete(self, key: str) -> bool
    def gc(self, current_tick: int) -> List[str]   # evict expired; returns evicted keys
    def keys(self) -> List[str]
    def usage(self) -> Tuple[int, int]             # (used, max)
    def snapshot(self) -> Dict[str, Any]
```

---

## Shared Memory Regions

### Creating a Shared Region

```python
# Kernel creates the global region during boot:
kernel.memory.create_shared_region("global", max_keys=1024)

# Create a restricted region (only listed agents can access it):
kernel.memory.create_shared_region(
    "research_results",
    max_keys=256,
    authorized_agents=["coordinator_0002", "worker_0003"],
)
```

### Reading and Writing Shared Memory

Agents do not directly access `SharedMemoryRegion` — they go through the kernel's syscall. For direct use in kernel-level code:

```python
# Write to shared region
kernel.memory.shared_write(
    region_name="global",
    agent_id="coordinator_0002",
    key="synthesis_complete",
    value=True,
    current_tick=kernel._tick,
)

# Read from shared region
value = kernel.memory.shared_read("global", "coordinator_0002", "synthesis_complete")
```

### `SharedMemoryRegion` Class

```python
class SharedMemoryRegion:
    def __init__(self, name, max_keys=512, authorized_agents=None) -> None
    def write(self, agent_id, key, value, memory_type=SHARED,
              current_tick=0, ttl_ticks=None) -> MemoryEntry
    def read(self, agent_id, key, current_tick=0) -> Any
    def delete(self, agent_id, key) -> bool
    def gc(self, current_tick) -> List[str]
    def keys(self) -> List[str]
    def snapshot(self) -> Dict[str, Any]
```

Access control: if `authorized_agents` is non-empty, only listed agents can read or write. The `kernel` agent_id always bypasses this check.

---

## Memory Limits

Each agent has a `memory_allocation` (max keys). The default is 256, but agents can request more in `__init__`:

```python
class BigMemoryAgent(Agent):
    def __init__(self):
        super().__init__(name="BigMemory", priority=5, memory_allocation=1024)
```

If a write would exceed the allocation:

```python
# MemoryFullError is raised
# (caught by the kernel and returned as SyscallResult(ok=False))
result = self.mem_write("key_1001", "value")  # if at 1000/1000 capacity
if not result.ok:
    self.log(f"Memory full: {result.error}")
```

To avoid `MemoryFullError`, delete entries you no longer need:

```python
# Direct deletion isn't exposed as a syscall in the default setup,
# but you can overwrite keys (same key = update, not new allocation)
self.mem_write("scratch", None)  # overwrite with None to free the value
```

---

## Garbage Collection

The kernel calls `memory.gc_tick(current_tick)` once per tick. This scans all agent spaces and shared regions, evicting any `SHORT_TERM` entries whose TTL has expired:

```python
evictions = kernel.memory.gc_tick(kernel._tick)
# {"agent:worker_0003": ["scratch_key_1", "scratch_key_2"], ...}
```

Eviction is lazy on reads as well: reading an expired key removes it and raises `MemoryKeyError`.

---

## Statistics

```python
stats = kernel.memory.stats()
# {
#   "agents": {
#       "coordinator_0002": {"used": 5, "max": 512},
#       "worker_0003": {"used": 3, "max": 256},
#   },
#   "shared_regions": {
#       "global": {"keys": 12, "max": 1024}
#   },
#   "gc_runs": 50,
# }
```

---

## Example: Multi-Step Workflow Memory Pattern

```python
class WorkflowAgent(Agent):
    def __init__(self):
        super().__init__(name="Workflow", priority=4, memory_allocation=512)

    def on_spawn(self) -> None:
        self.mem_write("phase", "init")
        self.mem_write("step", 0)
        self.mem_write("results", [])

    def think(self, tick: int) -> None:
        phase = self.mem_read("phase")
        step = self.mem_read("step") or 0
        results = self.mem_read("results") or []

        if phase == "init":
            self.mem_write("phase", "collecting")
            self.mem_write("step", 0)

        elif phase == "collecting":
            # Use SHORT_TERM for intermediate work
            self.mem_write(
                f"temp_result_{step}",
                f"result at step {step}",
                memory_type=MemoryType.SHORT_TERM,
                ttl=5,
            )
            results.append(f"result_{step}")
            self.mem_write("results", results)
            self.mem_write("step", step + 1)

            if step >= 3:
                self.mem_write("phase", "done")

        elif phase == "done":
            self.log(f"All results: {results}")

        self.yield_cpu()
```

---

## Related Pages

- [Kernel](kernel.md) — how `write_memory` and `read_memory` syscalls are dispatched
- [Schemas](../security/schemas.md) — typed memory schemas for validated writes
- [Agent API](../agents/api.md) — `mem_write`, `mem_read` convenience wrappers
