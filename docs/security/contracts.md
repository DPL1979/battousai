# Formal Verification and Contracts

The `contracts.py` module adds Design-by-Contract (DbC) runtime verification to Battousai agents. Inspired by Eiffel's DbC methodology and Hoare logic, every `think()` call can be wrapped with preconditions, postconditions, and invariants.

---

## Safety Hierarchy

From weakest to strongest enforcement:

```
1. Precondition   → checked before think() begins
2. Postcondition  → checked after think() completes
3. Invariant      → checked every tick regardless of action
4. SafetyEnvelope → hard rate limits that BLOCK actions even without a contract
```

---

## Violation Policies

```python
POLICY_WARN  = "WARN"   # Log the violation; agent continues
POLICY_BLOCK = "BLOCK"  # Skip the offending action; agent continues
POLICY_KILL  = "KILL"   # Terminate the agent immediately
```

---

## Contract Conditions

### `Precondition`

Must be true **before** `think()` begins. A `BLOCK` policy causes the `think()` call to be skipped for this tick.

```python
from battousai.contracts import Precondition, POLICY_BLOCK, POLICY_KILL

Precondition(
    name="has_kernel",
    description="Agent must be attached to a kernel before thinking",
    check=lambda agent: agent.kernel is not None,
    on_violation=POLICY_BLOCK,
)

Precondition(
    name="has_budget",
    description="Agent must have a budget allocation",
    check=lambda agent: agent.mem_read("budget") is not None,
    on_violation=POLICY_BLOCK,
)
```

### `Postcondition`

Must be true **after** `think()` completes. Receives both the agent and the return value of `think()`:

```python
from battousai.contracts import Postcondition, POLICY_KILL

Postcondition(
    name="budget_non_negative",
    description="Budget must never go negative after any action",
    check=lambda agent, _result: (agent.mem_read("budget") or 0) >= 0,
    on_violation=POLICY_KILL,  # terminate immediately if violated
)
```

### `Invariant`

Must be true **every tick** regardless of what happened:

```python
from battousai.contracts import Invariant, POLICY_WARN

Invariant(
    name="audit_log_present",
    description="Audit log key must always be present",
    check=lambda agent: agent.mem_read("audit_log") is not None,
    on_violation=POLICY_WARN,
)
```

---

## `SafetyEnvelope`

Hard rate limits that block actions automatically, independent of any declared contract:

```python
from battousai.contracts import SafetyEnvelope

envelope = SafetyEnvelope(
    max_tool_calls_per_tick=3,      # max tool invocations per tick
    max_messages_per_tick=5,        # max messages sent per tick
    max_file_writes_per_tick=2,     # max file writes per tick
    max_memory_writes_per_tick=10,  # max memory writes per tick
    max_spawns_per_tick=1,          # max child agents spawned per tick
)
```

---

## `Contract`

A complete behavioral specification for an agent class:

```python
from battousai.contracts import (
    Contract, Precondition, Postcondition, Invariant, SafetyEnvelope,
    POLICY_BLOCK, POLICY_KILL, POLICY_WARN,
)

budget_contract = Contract(
    agent_class=FinancialAgent,
    preconditions=[
        Precondition(
            name="has_budget",
            description="Agent must have a budget allocation before acting",
            check=lambda agent: agent.mem_read("budget") is not None,
            on_violation=POLICY_BLOCK,
        ),
        Precondition(
            name="market_open",
            description="Can only trade when market is open",
            check=lambda agent: agent.mem_read("market_status") == "open",
            on_violation=POLICY_BLOCK,
        ),
    ],
    postconditions=[
        Postcondition(
            name="budget_non_negative",
            description="Budget must never go negative after an action",
            check=lambda agent, _result: (agent.mem_read("budget") or 0) >= 0,
            on_violation=POLICY_KILL,
        ),
    ],
    invariants=[
        Invariant(
            name="audit_log_present",
            description="Audit log key must always be present",
            check=lambda agent: agent.mem_read("audit_log") is not None,
            on_violation=POLICY_WARN,
        ),
    ],
    safety_envelope=SafetyEnvelope(
        max_tool_calls_per_tick=3,
        max_messages_per_tick=5,
        max_file_writes_per_tick=2,
    ),
)
```

---

## `ContractMonitor`

A built-in `Agent` subclass that runs as a background supervisor, checking all registered contracts every tick:

```python
from battousai.contracts import ContractMonitor

monitor_id = kernel.spawn_agent(
    ContractMonitor,
    name="ContractMonitor",
    priority=1,                  # high priority — checks before workers run
    contracts=[budget_contract],
)
```

The `ContractMonitor` sends violation reports via IPC messages. It reports violations to the kernel log and can optionally terminate violating agents based on the `on_violation` policy.

---

## `PropertyChecker`

Evaluate temporal logic properties over an agent's execution history:

```python
from battousai.contracts import PropertyChecker

checker = PropertyChecker(agent)

# always(P) — P must hold at every observed tick
always_ok = checker.always(lambda state: state.get("errors", 0) < 10)

# eventually(P) — P must hold at least once in the history
ever_completed = checker.eventually(lambda state: state.get("status") == "done")

# until(P, Q) — P holds continuously until Q becomes true
stayed_running = checker.until(
    lambda s: s.get("status") == "running",
    lambda s: s.get("status") == "done",
)

# never(P) — P never holds (safety property)
never_negative = checker.never(lambda state: (state.get("budget") or 0) < 0)
```

`PropertyChecker` tracks a sliding history of memory snapshots. Each property method returns `True` or `False`.

---

## Full Example: Monitored Financial Agent

```python
from battousai.kernel import Kernel
from battousai.agent import Agent
from battousai.contracts import (
    Contract, Precondition, Postcondition, Invariant, SafetyEnvelope,
    ContractMonitor, POLICY_BLOCK, POLICY_KILL, POLICY_WARN,
)

class FinancialAgent(Agent):
    def __init__(self):
        super().__init__(name="Financial", priority=4)

    def on_spawn(self) -> None:
        self.mem_write("budget", 1000.0)
        self.mem_write("audit_log", [])
        self.mem_write("market_status", "open")

    def think(self, tick: int) -> None:
        budget = self.mem_read("budget") or 0
        log = self.mem_read("audit_log") or []

        # Simulate spending
        if tick % 3 == 0:
            amount = 50.0
            if budget >= amount:
                self.mem_write("budget", budget - amount)
                log.append({"tick": tick, "action": "spend", "amount": amount})
                self.mem_write("audit_log", log)

        self.yield_cpu()


# Define the contract
financial_contract = Contract(
    agent_class=FinancialAgent,
    preconditions=[
        Precondition(
            name="has_budget",
            check=lambda a: a.mem_read("budget") is not None,
            description="Must have budget",
            on_violation=POLICY_BLOCK,
        ),
    ],
    postconditions=[
        Postcondition(
            name="budget_non_negative",
            check=lambda a, _: (a.mem_read("budget") or 0) >= 0,
            description="Budget stays non-negative",
            on_violation=POLICY_KILL,
        ),
    ],
    invariants=[
        Invariant(
            name="audit_log_exists",
            check=lambda a: a.mem_read("audit_log") is not None,
            description="Audit log always exists",
            on_violation=POLICY_WARN,
        ),
    ],
    safety_envelope=SafetyEnvelope(max_tool_calls_per_tick=2),
)

# Boot and run with contract monitoring
kernel = Kernel(max_ticks=30)
kernel.boot()

agent_id = kernel.spawn_agent(FinancialAgent, name="Financial", priority=4)

monitor_id = kernel.spawn_agent(
    ContractMonitor,
    name="ContractMonitor",
    priority=1,
    contracts=[financial_contract],
)

kernel.run()
```

---

## `ContractViolation` Exception

Raised internally when a contract is violated with `POLICY_KILL`:

```python
class ContractViolation(Exception):
    agent_id: str
    contract_name: str
    condition_name: str
    message: str
```

---

## Temporal Properties Quick Reference

| Property | Method | Meaning |
|---|---|---|
| `always(P)` | `checker.always(predicate)` | P holds at every observed tick |
| `eventually(P)` | `checker.eventually(predicate)` | P holds at least once |
| `until(P, Q)` | `checker.until(p, q)` | P holds until Q becomes true |
| `never(P)` | `checker.never(predicate)` | P never holds (safety) |

---

## Related Pages

- [Capabilities](capabilities.md) — access control (who can do what)
- [Schemas](schemas.md) — type-level constraints on memory values
- [Supervision Trees](../agents/supervision.md) — fault-tolerance via restart
- [Architecture Overview](../architecture/overview.md) — where contracts fit in the security layer
