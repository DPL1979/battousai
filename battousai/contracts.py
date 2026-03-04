"""
contracts.py — Formal Verification & Behavioral Contracts
============================================================
Agents declare behavioral contracts (preconditions, postconditions,
invariants) that are verified at runtime. This is the safety layer
of Battousai — it ensures agents behave according to their specification.

Design Rationale
----------------
Inspired by Eiffel's Design by Contract (DbC) and C. A. R. Hoare's
axiomatic semantics. The core idea: every agent is accompanied by a
formal specification of its intended behaviour. The ContractMonitor
enforces these specs at runtime, transforming informal expectations
into mechanically checked guarantees.

Why runtime verification?
    Formal static verification (model checking, theorem proving) is
    powerful but expensive and requires specialised tooling. Runtime
    verification is lightweight, incremental, and directly observable
    in a running system. It catches violations as they happen rather
    than relying on developers to predict all possible behaviours at
    design time.

The safety hierarchy (weakest to strongest):
    1. Precondition  — checked before a think() call begins
    2. Postcondition — checked after a think() call completes
    3. Invariant     — checked every tick regardless of action
    4. SafetyEnvelope — hard limits that BLOCK the offending action
                        even if a contract says nothing about it

The on_violation policy:
    "WARN"  — log the violation but allow the agent to continue
    "BLOCK" — prevent the violating action from taking effect
    "KILL"  — terminate the agent immediately

Inspired by Design by Contract (Eiffel) and Hoare logic.

Components:
    Contract           — a behavioral specification for an agent
    Precondition       — must be true before an action
    Postcondition      — must be true after an action
    Invariant          — must always be true during agent lifetime
    ContractMonitor    — runtime verification engine
    ContractViolation  — exception for broken contracts
    PropertyChecker    — temporal property verification (always, eventually, until)
    SafetyEnvelope     — hard limits that override agent behavior
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from battousai.agent import Agent


# ---------------------------------------------------------------------------
# Violation policies
# ---------------------------------------------------------------------------

POLICY_WARN  = "WARN"
POLICY_BLOCK = "BLOCK"
POLICY_KILL  = "KILL"

_VALID_POLICIES = {POLICY_WARN, POLICY_BLOCK, POLICY_KILL}


# ---------------------------------------------------------------------------
# Contract conditions
# ---------------------------------------------------------------------------

@dataclass
class Precondition:
    """
    A condition that must be true before an agent begins its think() call.

    The ``check`` callable receives the agent instance and returns bool.
    If the check fails and on_violation is BLOCK, the think() call is
    skipped for this tick. If KILL, the agent is terminated.

    Example::

        Precondition(
            name="has_kernel",
            description="Agent must be attached to a kernel before thinking",
            check=lambda agent: agent.kernel is not None,
            on_violation=POLICY_BLOCK,
        )
    """
    name: str
    description: str
    check: Callable[[Agent], bool]
    on_violation: str = POLICY_WARN

    def __post_init__(self) -> None:
        if self.on_violation not in _VALID_POLICIES:
            raise ValueError(f"Invalid policy {self.on_violation!r}. Must be one of {_VALID_POLICIES}")


@dataclass
class Postcondition:
    """
    A condition that must be true after an agent's think() call completes.

    The ``check`` callable receives the agent instance and returns bool.
    Post-conditions verify that the think() call left the agent in a
    consistent state (e.g., memory was updated, required messages were sent).

    Example::

        Postcondition(
            name="cpu_yielded",
            description="Idle agents must yield CPU each tick",
            check=lambda agent: True,  # placeholder; real check inspects scheduler
            on_violation=POLICY_WARN,
        )
    """
    name: str
    description: str
    check: Callable[[Agent], bool]
    on_violation: str = POLICY_WARN

    def __post_init__(self) -> None:
        if self.on_violation not in _VALID_POLICIES:
            raise ValueError(f"Invalid policy {self.on_violation!r}. Must be one of {_VALID_POLICIES}")


@dataclass
class Invariant:
    """
    A condition that must be true at all times during an agent's lifetime.

    Invariants are checked every tick by the ContractMonitor, independently
    of whether the agent was scheduled that tick. They represent properties
    of the agent's state that must never be violated.

    Example::

        Invariant(
            name="non_negative_memory",
            description="Agent memory allocation must always be positive",
            check=lambda agent: agent.memory_allocation > 0,
            on_violation=POLICY_KILL,
        )
    """
    name: str
    description: str
    check: Callable[[Agent], bool]
    on_violation: str = POLICY_WARN

    def __post_init__(self) -> None:
        if self.on_violation not in _VALID_POLICIES:
            raise ValueError(f"Invalid policy {self.on_violation!r}. Must be one of {_VALID_POLICIES}")


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

@dataclass
class Contract:
    """
    A complete behavioral specification for an agent class.

    A Contract bundles all three types of conditions (pre, post, invariant)
    together with metadata. One Contract object is shared across all instances
    of the specified agent class.

    Fields:
        name             — human-readable contract identifier
        agent_class_name — name of the Agent subclass this contract covers
        preconditions    — checked before each think() call
        postconditions   — checked after each think() call
        invariants       — checked every tick
        version          — contract version string for change tracking
    """
    name: str
    agent_class_name: str
    preconditions: List[Precondition] = field(default_factory=list)
    postconditions: List[Postcondition] = field(default_factory=list)
    invariants: List[Invariant] = field(default_factory=list)
    version: str = "1.0"
    description: str = ""

    def add_precondition(self, condition: Precondition) -> "Contract":
        """Builder-style method for chaining."""
        self.preconditions.append(condition)
        return self

    def add_postcondition(self, condition: Postcondition) -> "Contract":
        self.postconditions.append(condition)
        return self

    def add_invariant(self, condition: Invariant) -> "Contract":
        self.invariants.append(condition)
        return self

    def summary(self) -> str:
        return (
            f"Contract({self.name!r} for {self.agent_class_name!r} v{self.version}) "
            f"[{len(self.preconditions)} pre, {len(self.postconditions)} post, "
            f"{len(self.invariants)} inv]"
        )


# ---------------------------------------------------------------------------
# ContractViolation exception
# ---------------------------------------------------------------------------

class ContractViolation(Exception):
    """
    Raised when a contract condition is violated and the policy is BLOCK or KILL.

    Fields:
        agent_id       — the agent that violated the contract
        contract_name  — name of the violated contract
        condition_name — name of the specific condition that failed
        condition_type — "precondition", "postcondition", or "invariant"
        details        — human-readable description of the violation
        tick           — system tick at which the violation occurred
    """
    def __init__(
        self,
        agent_id: str,
        contract_name: str,
        condition_name: str,
        condition_type: str,
        details: str,
        tick: int,
    ) -> None:
        self.agent_id = agent_id
        self.contract_name = contract_name
        self.condition_name = condition_name
        self.condition_type = condition_type
        self.details = details
        self.tick = tick
        super().__init__(
            f"ContractViolation: agent={agent_id!r} contract={contract_name!r} "
            f"condition={condition_name!r} ({condition_type}) at tick={tick}: {details}"
        )


# ---------------------------------------------------------------------------
# ContractMonitor — the runtime verification engine
# ---------------------------------------------------------------------------

class ContractMonitor(Agent):
    """
    A special monitoring agent that enforces behavioral contracts at runtime.

    The ContractMonitor runs at the highest priority (0) and checks all
    registered agent contracts each tick. It maintains a violation log and
    enforces penalties (WARN/BLOCK/KILL) based on the contract's policy.

    Operation:
        Each tick, for every monitored agent:
            1. Check all invariants (regardless of scheduling)
            2. Check postconditions from the last tick (if agent ran)
            3. Pre-check preconditions for the upcoming tick

        Violations are logged to self.violation_log. BLOCK violations
        prevent the agent's think() call. KILL violations terminate the agent.

    Usage::

        monitor = ContractMonitor()
        kernel.spawn_agent(ContractMonitor, name="ContractMonitor", priority=0)
        monitor.register_contract(agent_id, contract)
    """

    def __init__(self, name: str = "ContractMonitor", priority: int = 0) -> None:
        super().__init__(name=name, priority=priority,
                         memory_allocation=1024, time_slice=5)
        # agent_id → Contract
        self._contracts: Dict[str, Contract] = {}
        # Violation log: list of dicts
        self.violation_log: List[Dict[str, Any]] = []
        # Agents blocked for this tick (set by pre-check phase)
        self._blocked_agents: set = set()
        # Track which agents ran last tick (for postcondition checking)
        self._agents_ran_last_tick: set = set()
        # Stats
        self._total_checks: int = 0
        self._total_violations: int = 0

    def register_contract(self, agent_id: str, contract: Contract) -> None:
        """Attach a contract to an agent. Replaces any existing contract."""
        self._contracts[agent_id] = contract

    def unregister_contract(self, agent_id: str) -> bool:
        """Remove a contract (e.g., when the agent terminates)."""
        return self._contracts.pop(agent_id, None) is not None

    def is_blocked(self, agent_id: str) -> bool:
        """Check if an agent is currently blocked by a precondition failure."""
        return agent_id in self._blocked_agents

    def clear_block(self, agent_id: str) -> None:
        """Remove an agent from the blocked set (called after its tick)."""
        self._blocked_agents.discard(agent_id)

    def think(self, tick: int) -> None:
        """
        Per-tick verification loop.

        Runs at the start of each tick (priority 0 = first to run).
        """
        if self.kernel is None:
            self.yield_cpu()
            return

        self._blocked_agents.clear()
        agents_this_tick: set = set()

        for agent_id, contract in list(self._contracts.items()):
            agent = self.kernel._agents.get(agent_id)
            if agent is None:
                continue

            # 1. Check invariants (every tick)
            for invariant in contract.invariants:
                self._check_condition(agent, contract, invariant, "invariant", tick)

            # 2. Check postconditions from last tick (if agent ran)
            if agent_id in self._agents_ran_last_tick:
                for postcond in contract.postconditions:
                    self._check_condition(agent, contract, postcond, "postcondition", tick)

            # 3. Check preconditions for this tick
            blocked_by_pre = False
            for precond in contract.preconditions:
                violation = self._check_condition(
                    agent, contract, precond, "precondition", tick
                )
                if violation and precond.on_violation == POLICY_BLOCK:
                    self._blocked_agents.add(agent_id)
                    blocked_by_pre = True
                    break

            if not blocked_by_pre:
                agents_this_tick.add(agent_id)

        self._agents_ran_last_tick = agents_this_tick
        self.mem_write("violation_count", len(self.violation_log))
        self.mem_write("monitored_agents", len(self._contracts))
        self.yield_cpu()

    def _check_condition(
        self,
        agent: Agent,
        contract: Contract,
        condition: Any,
        condition_type: str,
        tick: int,
    ) -> bool:
        """
        Evaluate a single condition.

        Returns True if a violation occurred, False if the check passed.
        Applies the on_violation policy (WARN/BLOCK/KILL).
        """
        self._total_checks += 1
        try:
            passed = condition.check(agent)
        except Exception as exc:
            # Condition itself raised — treat as violation
            passed = False
            details = f"Condition check raised exception: {exc}"
        else:
            details = f"{condition_type.capitalize()} '{condition.name}' failed"

        if passed:
            return False

        # Violation occurred
        self._total_violations += 1
        violation_record = {
            "tick": tick,
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "contract": contract.name,
            "condition_name": condition.name,
            "condition_type": condition_type,
            "policy": condition.on_violation,
            "details": details,
            "timestamp": time.time(),
        }
        self.violation_log.append(violation_record)

        if condition.on_violation == POLICY_WARN:
            if self.kernel:
                self.kernel.logger.system(
                    "ContractMonitor",
                    f"WARN violation: agent={agent.agent_id!r} "
                    f"condition={condition.name!r} ({condition_type})",
                )

        elif condition.on_violation == POLICY_KILL:
            if self.kernel:
                self.kernel.logger.system(
                    "ContractMonitor",
                    f"KILL: terminating agent={agent.agent_id!r} "
                    f"for {condition_type} violation: {condition.name!r}",
                )
                self.kernel.kill_agent(agent.agent_id)
            raise ContractViolation(
                agent.agent_id, contract.name, condition.name,
                condition_type, details, tick,
            )

        return True

    def get_violations_for(self, agent_id: str) -> List[Dict[str, Any]]:
        """Return all recorded violations for a specific agent."""
        return [v for v in self.violation_log if v["agent_id"] == agent_id]

    def violation_summary(self) -> str:
        """Format a human-readable violation summary."""
        if not self.violation_log:
            return "ContractMonitor: No violations recorded."
        lines = [f"=== ContractMonitor: {len(self.violation_log)} violations ==="]
        for v in self.violation_log[-50:]:
            lines.append(
                f"  tick={v['tick']:04d} | {v['policy']:<5} | "
                f"agent={v['agent_id']} | {v['condition_type']}: {v['condition_name']}"
            )
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        return {
            "monitored_agents": len(self._contracts),
            "total_checks": self._total_checks,
            "total_violations": self._total_violations,
            "blocked_agents": len(self._blocked_agents),
            "violation_log_size": len(self.violation_log),
        }


# ---------------------------------------------------------------------------
# PropertyChecker — temporal logic verification
# ---------------------------------------------------------------------------

class PropertyChecker:
    """
    Temporal logic property verification for agent behaviour.

    Supports four temporal operators:

    always(predicate)
        The predicate must be true on every tick we observe.
        Equivalent to □P in LTL (Linear Temporal Logic).

    eventually(predicate, within_ticks)
        The predicate must become true within the next N ticks.
        Equivalent to ◇P with a time bound.

    until(predicate_a, predicate_b)
        predicate_a must hold on every tick until predicate_b becomes true.
        Equivalent to A U B in LTL.

    never(predicate)
        The predicate must never be true.
        Equivalent to □¬P in LTL.

    Properties are registered by name and checked via check_all().
    Each property tracks its own state over time.
    """

    def __init__(self) -> None:
        # name → property descriptor
        self._properties: Dict[str, Dict[str, Any]] = {}
        self._results: Dict[str, List[bool]] = {}  # name → per-tick truth values
        self._violations: Dict[str, List[int]] = {}  # name → ticks of violation

    def always(
        self,
        name: str,
        predicate: Callable[[Agent], bool],
        description: str = "",
    ) -> "PropertyChecker":
        """Register an 'always' property."""
        self._properties[name] = {
            "type": "always",
            "predicate": predicate,
            "description": description,
        }
        self._results[name] = []
        self._violations[name] = []
        return self

    def eventually(
        self,
        name: str,
        predicate: Callable[[Agent], bool],
        within_ticks: int = 50,
        description: str = "",
    ) -> "PropertyChecker":
        """Register an 'eventually' property."""
        self._properties[name] = {
            "type": "eventually",
            "predicate": predicate,
            "within_ticks": within_ticks,
            "satisfied": False,
            "start_tick": None,
            "description": description,
        }
        self._results[name] = []
        self._violations[name] = []
        return self

    def until(
        self,
        name: str,
        predicate_a: Callable[[Agent], bool],
        predicate_b: Callable[[Agent], bool],
        description: str = "",
    ) -> "PropertyChecker":
        """Register an 'until' property: A must hold until B."""
        self._properties[name] = {
            "type": "until",
            "predicate_a": predicate_a,
            "predicate_b": predicate_b,
            "b_satisfied": False,
            "description": description,
        }
        self._results[name] = []
        self._violations[name] = []
        return self

    def never(
        self,
        name: str,
        predicate: Callable[[Agent], bool],
        description: str = "",
    ) -> "PropertyChecker":
        """Register a 'never' property."""
        self._properties[name] = {
            "type": "never",
            "predicate": predicate,
            "description": description,
        }
        self._results[name] = []
        self._violations[name] = []
        return self

    def check_all(self, agent: Agent, tick: int) -> Dict[str, bool]:
        """
        Evaluate all registered properties against ``agent`` at ``tick``.

        Returns a dict of property_name → satisfied (bool).
        """
        results: Dict[str, bool] = {}

        for name, prop in self._properties.items():
            satisfied = self._evaluate_property(name, prop, agent, tick)
            self._results[name].append(satisfied)
            if not satisfied:
                self._violations[name].append(tick)
            results[name] = satisfied

        return results

    def _evaluate_property(
        self,
        name: str,
        prop: Dict[str, Any],
        agent: Agent,
        tick: int,
    ) -> bool:
        """Evaluate a single property. Returns True if the property holds."""
        prop_type = prop["type"]

        try:
            if prop_type == "always":
                return bool(prop["predicate"](agent))

            elif prop_type == "never":
                return not bool(prop["predicate"](agent))

            elif prop_type == "eventually":
                if prop["satisfied"]:
                    return True
                if prop["start_tick"] is None:
                    prop["start_tick"] = tick
                if prop["predicate"](agent):
                    prop["satisfied"] = True
                    return True
                # Check if time window has expired
                elapsed = tick - prop["start_tick"]
                return elapsed <= prop["within_ticks"]

            elif prop_type == "until":
                if prop["b_satisfied"]:
                    return True  # B was achieved; property fulfilled
                b_holds = bool(prop["predicate_b"](agent))
                if b_holds:
                    prop["b_satisfied"] = True
                    return True
                # B hasn't happened yet — A must hold
                a_holds = bool(prop["predicate_a"](agent))
                return a_holds

        except Exception:
            return False  # Error in predicate = property fails

        return False

    def violations_for(self, name: str) -> List[int]:
        """Return the list of ticks where property ``name`` was violated."""
        return list(self._violations.get(name, []))

    def report(self) -> str:
        """Format a summary of all property check results."""
        lines = [f"=== PropertyChecker Report ({len(self._properties)} properties) ==="]
        for name, prop in self._properties.items():
            checks = len(self._results.get(name, []))
            viols = len(self._violations.get(name, []))
            status = "✓" if viols == 0 else "✗"
            lines.append(
                f"  {status} {name!r} ({prop['type']}) | "
                f"checks={checks} violations={viols}"
                + (f" | desc: {prop['description']}" if prop.get("description") else "")
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SafetyEnvelope — hard behavioural limits
# ---------------------------------------------------------------------------

@dataclass
class SafetyEnvelopeConfig:
    """
    Configuration for a SafetyEnvelope instance.

    All limits are per-agent per-tick unless otherwise noted.

    Fields:
        max_messages_per_tick    — cap on outbound IPC messages
        max_memory_writes_per_tick — cap on memory write syscalls
        max_tool_calls_per_tick  — cap on tool invocations
        max_spawn_per_tick       — cap on agent spawns (prevents fork bombs)
        forbidden_tools          — tools the agent must not invoke
        max_file_size            — max bytes for a single file write
        max_total_agents         — hard cap on total live agents (cluster-wide)
    """
    max_messages_per_tick: int = 20
    max_memory_writes_per_tick: int = 50
    max_tool_calls_per_tick: int = 10
    max_spawn_per_tick: int = 3
    forbidden_tools: List[str] = field(default_factory=list)
    max_file_size: int = 1_000_000  # 1 MB in characters
    max_total_agents: int = 100


# Default safety envelope with sensible production defaults
DEFAULT_SAFETY_ENVELOPE = SafetyEnvelopeConfig(
    max_messages_per_tick=10,
    max_memory_writes_per_tick=30,
    max_tool_calls_per_tick=5,
    max_spawn_per_tick=2,
    forbidden_tools=[],
    max_file_size=500_000,
    max_total_agents=50,
)


class SafetyEnvelope:
    """
    Hard behavioural limits that act as a last line of defence.

    Unlike contract conditions (which describe *intended* behaviour),
    the SafetyEnvelope imposes absolute upper bounds on resource consumption.
    These limits apply to ALL agents regardless of their contracts.

    When a limit is reached, the offending action is BLOCKED and a violation
    is logged. The agent's think() call continues — only the specific action
    is prevented.

    Instrumentation:
        SafetyEnvelope wraps the kernel's syscall dispatch to intercept
        resource-consuming syscalls before they execute.

    Integration with Kernel:
        Call install(kernel) to hook into the kernel's syscall path.
        This should be done once during boot, before any agents spawn.
    """

    def __init__(self, config: Optional[SafetyEnvelopeConfig] = None) -> None:
        self.config = config or SafetyEnvelopeConfig()
        # Per-agent per-tick counters: agent_id → {metric: count}
        self._tick_counters: Dict[str, Dict[str, int]] = {}
        self._current_tick: int = -1
        # Blocked action log
        self.blocked_log: List[Dict[str, Any]] = []

    def _reset_tick(self, tick: int) -> None:
        """Reset per-tick counters at the start of a new tick."""
        if tick != self._current_tick:
            self._current_tick = tick
            self._tick_counters.clear()

    def _counter(self, agent_id: str, metric: str) -> int:
        return self._tick_counters.get(agent_id, {}).get(metric, 0)

    def _increment(self, agent_id: str, metric: str) -> None:
        if agent_id not in self._tick_counters:
            self._tick_counters[agent_id] = {}
        self._tick_counters[agent_id][metric] = (
            self._tick_counters[agent_id].get(metric, 0) + 1
        )

    def _log_blocked(self, agent_id: str, syscall: str, reason: str, tick: int) -> None:
        self.blocked_log.append({
            "tick": tick,
            "agent_id": agent_id,
            "syscall": syscall,
            "reason": reason,
            "timestamp": time.time(),
        })

    def check_send_message(self, agent_id: str, tick: int) -> bool:
        """
        Check whether an agent may send a message this tick.

        Returns True (allowed) or False (blocked).
        """
        self._reset_tick(tick)
        count = self._counter(agent_id, "messages")
        if count >= self.config.max_messages_per_tick:
            self._log_blocked(
                agent_id, "send_message",
                f"Exceeded max_messages_per_tick ({self.config.max_messages_per_tick})",
                tick,
            )
            return False
        self._increment(agent_id, "messages")
        return True

    def check_write_memory(self, agent_id: str, tick: int) -> bool:
        """Check whether an agent may write to memory this tick."""
        self._reset_tick(tick)
        count = self._counter(agent_id, "mem_writes")
        if count >= self.config.max_memory_writes_per_tick:
            self._log_blocked(
                agent_id, "write_memory",
                f"Exceeded max_memory_writes_per_tick ({self.config.max_memory_writes_per_tick})",
                tick,
            )
            return False
        self._increment(agent_id, "mem_writes")
        return True

    def check_tool_call(self, agent_id: str, tool_name: str, tick: int) -> bool:
        """Check whether an agent may invoke a tool this tick."""
        self._reset_tick(tick)
        # Check forbidden tools
        if tool_name in self.config.forbidden_tools:
            self._log_blocked(
                agent_id, f"access_tool:{tool_name}",
                f"Tool '{tool_name}' is in the forbidden_tools list",
                tick,
            )
            return False
        # Check rate limit
        count = self._counter(agent_id, "tool_calls")
        if count >= self.config.max_tool_calls_per_tick:
            self._log_blocked(
                agent_id, "access_tool",
                f"Exceeded max_tool_calls_per_tick ({self.config.max_tool_calls_per_tick})",
                tick,
            )
            return False
        self._increment(agent_id, "tool_calls")
        return True

    def check_spawn_agent(self, agent_id: str, tick: int, total_agents: int) -> bool:
        """Check whether an agent may spawn a new child agent this tick."""
        self._reset_tick(tick)
        # Check global agent cap
        if total_agents >= self.config.max_total_agents:
            self._log_blocked(
                agent_id, "spawn_agent",
                f"Global agent cap reached ({self.config.max_total_agents})",
                tick,
            )
            return False
        # Check per-tick spawn limit
        count = self._counter(agent_id, "spawns")
        if count >= self.config.max_spawn_per_tick:
            self._log_blocked(
                agent_id, "spawn_agent",
                f"Exceeded max_spawn_per_tick ({self.config.max_spawn_per_tick})",
                tick,
            )
            return False
        self._increment(agent_id, "spawns")
        return True

    def check_file_write(self, agent_id: str, content_size: int, tick: int) -> bool:
        """Check whether a file write of given size is within the envelope."""
        self._reset_tick(tick)
        if content_size > self.config.max_file_size:
            self._log_blocked(
                agent_id, "file_write",
                f"File size {content_size} exceeds max_file_size ({self.config.max_file_size})",
                tick,
            )
            return False
        return True

    def blocked_summary(self) -> str:
        """Format a human-readable summary of all blocked actions."""
        if not self.blocked_log:
            return "SafetyEnvelope: No blocked actions recorded."
        lines = [f"=== SafetyEnvelope: {len(self.blocked_log)} blocked actions ==="]
        for entry in self.blocked_log[-30:]:
            lines.append(
                f"  tick={entry['tick']:04d} | agent={entry['agent_id']} | "
                f"syscall={entry['syscall']} | {entry['reason']}"
            )
        return "\n".join(lines)
