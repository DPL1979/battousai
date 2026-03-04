# Capability-Based Security

The `capabilities.py` module enforces the principle of least privilege. Agents receive only the capabilities they need at spawn time; all other resource accesses are denied.

---

## Design Principles

- **No ambient authority** — without a capability token, an agent cannot access any resource
- **Unforgeable** — capability tokens are UUIDs managed exclusively by `CapabilityManager`; agents cannot create their own
- **Delegatable** — an agent holding a `delegatable=True` capability can grant a subset to child agents
- **Revocable** — the kernel or a supervisor can revoke capabilities at any tick
- **Audit logging** — every grant, revocation, delegation, and access check is logged

---

## `CapabilityType` Enum

```python
class CapabilityType(Enum):
    TOOL_USE     = auto()  # Use a tool by name or glob pattern
    FILE_READ    = auto()  # Read files matching a path glob
    FILE_WRITE   = auto()  # Write files matching a path glob
    MEMORY_READ  = auto()  # Read another agent's private memory
    MEMORY_WRITE = auto()  # Write to a named shared memory region
    SPAWN        = auto()  # Spawn agents of a given class (or "*")
    MESSAGE      = auto()  # Send IPC messages to matching agent_ids
    NETWORK      = auto()  # Communicate with remote nodes matching a pattern
    ADMIN        = auto()  # Unrestricted access — kernel agents only
```

---

## `Capability` Dataclass

```python
@dataclass
class Capability:
    cap_id: str                        # Globally unique UUID
    cap_type: CapabilityType
    resource_pattern: str              # Glob pattern for the resource(s) covered
    granted_to: str                    # Holder's agent_id
    granted_by: str                    # Grantor's agent_id ("kernel" for initial grants)
    created_at: int                    # Kernel tick at creation
    expires_at: Optional[int] = None   # Optional expiry tick
    delegatable: bool = False          # Can the holder further delegate?
    revoked: bool = False

    def is_active(self, current_tick: int) -> bool
    def covers(self, resource: str) -> bool   # uses fnmatch glob matching
```

The `resource_pattern` uses `fnmatch` glob syntax:
- `"*"` — matches anything
- `"web_search"` — exact tool name
- `"/shared/*"` — any file under /shared/
- `"/agents/{self}/**"` — agent's own directory (interpolated at grant time)

---

## `CapabilityManager`

The central authority for creating, checking, and revoking capabilities:

```python
from battousai.capabilities import CapabilityManager, CapabilityType

mgr = CapabilityManager()
```

### Creating Capabilities

```python
cap = mgr.create_capability(
    cap_type=CapabilityType.TOOL_USE,
    resource_pattern="web_search",
    agent_id="worker_0001",
    current_tick=1,
    granted_by="kernel",        # default
    delegatable=False,          # default
    expires_at=None,            # default — no expiry
)
print(cap)
# Capability(id=a1b2c3d4, TOOL_USE('web_search'), to='worker_0001', active)
```

Time-bounded capability (expires after 20 ticks):
```python
cap = mgr.create_capability(
    cap_type=CapabilityType.FILE_WRITE,
    resource_pattern="/shared/temp/*",
    agent_id="temp_worker_0005",
    current_tick=5,
    expires_at=25,   # valid from tick 5 to tick 24 inclusive
)
```

### Checking Capabilities

```python
# Boolean check (no exception)
allowed = mgr.check(
    agent_id="worker_0001",
    cap_type=CapabilityType.TOOL_USE,
    resource="web_search",
    current_tick=tick,
)
# True or False

# Raising check (raises CapabilityViolation on denial)
from battousai.capabilities import CapabilityViolation

try:
    mgr.require("worker_0001", CapabilityType.FILE_WRITE, "/shared/results/out.txt")
except CapabilityViolation as e:
    print(e)  # Agent 'worker_0001' lacks FILE_WRITE('/shared/results/out.txt') capability.
```

The `kernel` agent_id always passes any check.

### Revoking Capabilities

```python
# Revoke a specific capability by ID
mgr.revoke(cap.cap_id, current_tick=tick)

# Revoke all capabilities for an agent (e.g. on quarantine)
count = mgr.revoke_all("malicious_agent_0099", current_tick=tick)
print(f"Revoked {count} capabilities")
```

### Expiring Capabilities

Call once per tick to expire time-bounded capabilities:
```python
expired_ids = mgr.expire_caps(current_tick=tick)
if expired_ids:
    print(f"Expired: {expired_ids}")
```

### Delegating Capabilities

An agent with a `delegatable=True` capability can grant a copy to a child:

```python
# Coordinator has a delegatable FILE_WRITE cap
coordinator_cap = mgr.create_capability(
    cap_type=CapabilityType.FILE_WRITE,
    resource_pattern="/shared/*",
    agent_id="coordinator_0002",
    delegatable=True,   # can delegate
)

# Coordinator delegates to a worker
delegated_cap = mgr.delegate(
    grantor_id="coordinator_0002",
    cap_id=coordinator_cap.cap_id,
    target_agent_id="worker_0003",
    current_tick=tick,
    expires_at=tick + 10,  # worker cap expires sooner
    delegatable=False,     # worker cannot re-delegate
)
```

Delegated capabilities cannot outlive their parent.

---

## `@requires_capability` Decorator

Guard an agent method with a capability check:

```python
from battousai.capabilities import requires_capability, CapabilityType
from battousai.agent import Agent

class SecureWorkerAgent(Agent):
    @requires_capability(CapabilityType.TOOL_USE, "web_search")
    def do_research(self, topic: str) -> None:
        result = self.use_tool("web_search", query=topic)
        if result.ok:
            self.log(f"Found: {result.value['results'][0]['snippet'][:80]}")

    @requires_capability(CapabilityType.FILE_WRITE, "/shared/*")
    def publish_result(self, data: str) -> None:
        self.write_file("/shared/results/output.txt", data)

    def think(self, tick: int) -> None:
        if tick == 1:
            self.do_research("quantum computing")  # checked at call time
        if tick == 2:
            self.publish_result("my findings")    # checked at call time
        self.yield_cpu()
```

The decorator looks for a `CapabilityManager` on `self.kernel.capability_manager`. If no manager is found, the check is skipped (fail-open for tests).

---

## `SecurityPolicy`

Define default capabilities for agent classes:

```python
from battousai.capabilities import SecurityPolicy, CapabilityType

policy = SecurityPolicy(
    name="WorkerPolicy",
    class_policies={
        "WorkerAgent": [
            (CapabilityType.TOOL_USE,   "*",                    False),  # any tool
            (CapabilityType.MESSAGE,    "coordinator*",         False),  # message coordinators
            (CapabilityType.FILE_WRITE, "/agents/{self}/*",     False),  # own directory
            (CapabilityType.FILE_READ,  "/agents/{self}/*",     False),
            (CapabilityType.FILE_READ,  "/shared/*",            False),
        ],
    },
    default_caps=[
        (CapabilityType.MEMORY_READ, "global", False),  # all agents read global shared memory
    ],
)

# Apply policy when spawning an agent
policy.apply(
    manager=kernel.capability_manager,
    agent_id="worker_0003",
    class_name="WorkerAgent",
    current_tick=kernel._tick,
)
```

`{self}` in a resource pattern is automatically replaced with the `agent_id` at apply time.

---

## Default Policy (`DEFAULT_POLICY`)

Battousai ships with a built-in policy for the three standard agent types:

| Agent Class | Capabilities |
|---|---|
| `CoordinatorAgent` | SPAWN(*), MESSAGE(*), FILE_WRITE(/shared/*), FILE_READ(/shared/*), TOOL_USE(*), MEMORY_READ(*) |
| `WorkerAgent` | TOOL_USE(*), MESSAGE(coordinator*), FILE_WRITE(/agents/{self}/*), FILE_READ(/agents/{self}/*), FILE_READ(/shared/*) |
| `MonitorAgent` | MEMORY_READ(*), FILE_READ(*), MESSAGE(*) |
| All agents | MEMORY_READ(global) |

---

## Audit Log

Every capability event is recorded:

```python
# Get the full audit log
log = mgr.audit_log()
for entry in log:
    print(entry)
# AuditEntry(tick=1, agent='worker_0001', action=GRANT, TOOL_USE('web_search'), ALLOW)
# AuditEntry(tick=5, agent='worker_0001', action=CHECK_ALLOW, TOOL_USE('web_search'), ALLOW)
# AuditEntry(tick=8, agent='worker_0001', action=CHECK_DENY, FILE_WRITE('/etc/*'), DENY)

# Get log for a specific agent
agent_log = mgr.audit_log_for_agent("worker_0001")

# Stats
stats = mgr.stats()
# {
#   "registered_agents": 4,
#   "total_caps_issued": 12,
#   "active_caps": 10,
#   "audit_log_entries": 47,
#   "access_denials": 2,
# }
```

---

## `CapabilityViolation` Exception

```python
class CapabilityViolation(Exception):
    agent_id: str
    cap_type: CapabilityType
    resource: str
```

---

## Related Pages

- [Schemas](schemas.md) — typed memory schemas (complement to capability security)
- [Contracts](contracts.md) — behavioral contracts for runtime verification
- [Agent API](../agents/api.md) — how agents interact with OS resources
