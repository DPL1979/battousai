"""
agent.py — Battousai Agent Runtime
================================
Base Agent class and built-in agent types for the Autonomous Intelligence
Operating System.

Every AI agent in Battousai inherits from `Agent`. Agents are autonomous
programs that run inside the OS — they have no direct access to the
file system, network, or tools outside the kernel's syscall interface.

Agent Lifecycle:
    1. `__init__()` — called by kernel.spawn_agent()
    2. `on_spawn()` — hook called once after the OS registers the agent
    3. `think(tick)` — called every tick the scheduler grants this agent CPU time
    4. `on_terminate()` — hook called before the agent is removed from the OS

Syscall Interface:
    Agents interact with OS services by calling `self.syscall(name, **kwargs)`.
    Available syscalls are defined in `kernel.py`. This is the ONLY way agents
    touch OS resources — there are no ambient globals.

Built-in Agent Types:
    CoordinatorAgent — Decomposes high-level goals into subtasks and
                        delegates them to worker agents via IPC.
    WorkerAgent      — Receives task messages, uses tools to execute the
                        task, and sends results back to the coordinator.
    MonitorAgent     — Passively observes system metrics each tick and
                        publishes them to the bulletin board.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from battousai.scheduler import AgentState

if TYPE_CHECKING:
    from battousai.kernel import Kernel


# ---------------------------------------------------------------------------
# Syscall result wrapper
# ---------------------------------------------------------------------------

@dataclass
class SyscallResult:
    """Wraps the outcome of a kernel syscall."""
    ok: bool
    value: Any = None
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    Base class for all Battousai agents.

    Subclasses must implement `think(tick)`. The think method is called
    each tick when the scheduler grants CPU time.

    Instance attributes set by the kernel at spawn time:
        kernel  — back-reference to the Kernel instance
        agent_id — unique string identifier
        name     — human-readable display name
        priority — scheduler priority (0=highest)
    """

    def __init__(
        self,
        name: str,
        priority: int = 5,
        memory_allocation: int = 256,
        time_slice: int = 3,
    ) -> None:
        # These are filled in by Kernel.spawn_agent()
        self.kernel: Optional["Kernel"] = None
        self.agent_id: str = ""
        self.name: str = name
        self.priority: int = max(0, min(9, priority))
        self.memory_allocation: int = memory_allocation
        self.time_slice: int = time_slice

        # Internal state
        self._spawn_tick: int = 0
        self._ticks_alive: int = 0
        self._terminated: bool = False

        # Subclass-specific initialisation goes in on_spawn()

    # ------------------------------------------------------------------
    # Lifecycle hooks (override in subclasses)
    # ------------------------------------------------------------------

    def on_spawn(self) -> None:
        """Called once after the kernel registers this agent. Override to init."""
        pass

    def on_terminate(self) -> None:
        """Called once before the kernel removes this agent. Override to clean up."""
        pass

    # ------------------------------------------------------------------
    # Core think loop (MUST be implemented by subclasses)
    # ------------------------------------------------------------------

    def think(self, tick: int) -> None:
        """
        Called each tick this agent is scheduled.

        This is the agent's 'brain'. It should:
        - Read incoming messages from the mailbox
        - Update internal memory/state
        - Make decisions
        - Send messages / invoke tools via syscalls
        - Yield CPU voluntarily if work is done for this tick
        """
        raise NotImplementedError(f"Agent {self.name!r} must implement think()")

    # ------------------------------------------------------------------
    # Syscall interface
    # ------------------------------------------------------------------

    def syscall(self, name: str, **kwargs: Any) -> SyscallResult:
        """
        Request an OS service from the kernel.

        Args:
            name   — syscall name (see Kernel.SYSCALLS)
            kwargs — arguments forwarded to the syscall handler

        Returns:
            SyscallResult with ok=True/False and value or error.
        """
        if self.kernel is None:
            return SyscallResult(ok=False, error="Agent not attached to a kernel")
        return self.kernel._dispatch_syscall(self.agent_id, name, **kwargs)

    # ------------------------------------------------------------------
    # Convenience helpers (thin wrappers around common syscalls)
    # ------------------------------------------------------------------

    def send_message(
        self,
        recipient_id: str,
        message_type,
        payload: Any,
        correlation_id: Optional[str] = None,
    ) -> SyscallResult:
        return self.syscall(
            "send_message",
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id,
        )

    def read_inbox(self) -> List[Any]:
        """Drain all pending messages from this agent's mailbox."""
        result = self.syscall("read_inbox")
        return result.value if result.ok else []

    def mem_write(self, key: str, value: Any, memory_type=None, ttl: Optional[int] = None) -> SyscallResult:
        from battousai.memory import MemoryType
        mt = memory_type or MemoryType.LONG_TERM
        return self.syscall("write_memory", key=key, value=value, memory_type=mt, ttl_ticks=ttl)

    def mem_read(self, key: str) -> Any:
        result = self.syscall("read_memory", key=key)
        return result.value if result.ok else None

    def use_tool(self, tool_name: str, **args: Any) -> SyscallResult:
        return self.syscall("access_tool", tool_name=tool_name, args=args)

    def log(self, message: str, level=None) -> None:
        from battousai.logger import LogLevel
        lvl = level or LogLevel.INFO
        if self.kernel:
            self.kernel.logger.log(lvl, self.agent_id, message)

    def yield_cpu(self) -> None:
        """Voluntarily give up the remainder of this tick's CPU slice."""
        self.syscall("yield_cpu")

    def spawn_child(self, agent_class, name: str, priority: int = 5, **kwargs: Any) -> SyscallResult:
        return self.syscall(
            "spawn_agent",
            agent_class=agent_class,
            agent_name=name,
            priority=priority,
            **kwargs,
        )

    def write_file(self, path: str, data: Any) -> SyscallResult:
        return self.syscall("write_file", path=path, data=data)

    def read_file(self, path: str) -> SyscallResult:
        return self.syscall("read_file", path=path)

    def get_status(self) -> SyscallResult:
        return self.syscall("get_status")

    def list_agents(self) -> List[str]:
        result = self.syscall("list_agents")
        return result.value if result.ok else []

    # ------------------------------------------------------------------
    # Internal (called by kernel)
    # ------------------------------------------------------------------

    def _tick(self, tick: int) -> None:
        """Internal tick driver — called by the kernel event loop."""
        self._ticks_alive += 1
        self.think(tick)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.agent_id!r}, name={self.name!r}, priority={self.priority})"


# ---------------------------------------------------------------------------
# CoordinatorAgent
# ---------------------------------------------------------------------------

class CoordinatorAgent(Agent):
    """
    Coordinator — decomposes high-level goals into subtasks and delegates
    them to WorkerAgents.
    """

    def __init__(self, name: str = "Coordinator", priority: int = 2) -> None:
        super().__init__(name=name, priority=priority, memory_allocation=512, time_slice=4)
        self._phase: str = "INIT"
        self._pending_results: Dict[str, Optional[str]] = {}
        self._task: Optional[str] = None
        self._workers_spawned: bool = False
        self._worker_ids: List[str] = []
        self._result_correlation: Dict[str, str] = {}
        self._summary_written: bool = False

    def on_spawn(self) -> None:
        self.log(f"[{self.name}] Online. Awaiting initial task assignment.")
        self._phase = "WAITING_TASK"

    def think(self, tick: int) -> None:
        from battousai.ipc import MessageType
        from battousai.memory import MemoryType

        messages = self.read_inbox()

        for msg in messages:
            if msg.message_type == MessageType.TASK and self._task is None:
                self._task = msg.payload.get("task", "") if isinstance(msg.payload, dict) else str(msg.payload)
                self.log(f"[{self.name}] Received task: '{self._task}'")
                self.mem_write("current_task", self._task)
                self._phase = "DECOMPOSE"

            elif msg.message_type == MessageType.RESULT:
                worker_id = msg.sender_id
                result_data = msg.payload
                self.log(f"[{self.name}] Received result from {worker_id}")
                self._pending_results[worker_id] = result_data
                self.mem_write(f"result_{worker_id}", result_data, memory_type=MemoryType.LONG_TERM)

        if self._phase == "DECOMPOSE" and not self._workers_spawned:
            self._decompose_and_spawn(tick)
        elif self._phase == "COLLECTING":
            all_received = (
                len(self._pending_results) == len(self._worker_ids)
                and all(v is not None for v in self._pending_results.values())
            )
            if all_received and not self._summary_written:
                self._synthesise_results(tick)

        self.yield_cpu()

    def _decompose_and_spawn(self, tick: int) -> None:
        from battousai.ipc import MessageType

        self.log(f"[{self.name}] Decomposing task into subtasks...")

        subtasks = [
            {
                "subtask_id": "subtask_1",
                "description": "Research fundamentals and technical landscape of quantum computing",
                "queries": ["quantum computing basics", "quantum supremacy milestones", "quantum computing challenges"],
            },
            {
                "subtask_id": "subtask_2",
                "description": "Research applications and future outlook of quantum computing",
                "queries": ["quantum computing applications", "quantum computing challenges"],
            },
        ]

        for i, subtask in enumerate(subtasks):
            result = self.spawn_child(WorkerAgent, name=f"Worker-{i+1}", priority=4, subtask=subtask)
            if result.ok:
                worker_id = result.value
                self._worker_ids.append(worker_id)
                self._pending_results[worker_id] = None
                self.send_message(recipient_id=worker_id, message_type=MessageType.TASK, payload=subtask)
                self.log(f"[{self.name}] Spawned {worker_id!r} for: {subtask['description']}")

        self._workers_spawned = True
        self._phase = "COLLECTING"
        self.log(f"[{self.name}] Collecting results from {len(self._worker_ids)} workers...")

    def _synthesise_results(self, tick: int) -> None:
        self.log(f"[{self.name}] All results received. Synthesising summary...")

        parts = [
            "QUANTUM COMPUTING RESEARCH SUMMARY",
            f"Generated by Battousai Coordinator Agent at tick {tick}",
            f"Task: {self._task}",
            "=" * 60,
            "",
        ]

        for i, (worker_id, result) in enumerate(self._pending_results.items(), 1):
            parts.append(f"--- Section {i}: Worker {worker_id} ---")
            if isinstance(result, dict):
                parts.append(f"Subtask: {result.get('subtask_description', 'N/A')}")
                parts.append("")
                findings = result.get("findings", [])
                for j, finding in enumerate(findings, 1):
                    parts.append(f"Finding {j}:")
                    parts.append(f"  Query: {finding.get('query', 'N/A')}")
                    parts.append(f"  Result: {finding.get('result', 'N/A')}")
                    parts.append("")
            else:
                parts.append(str(result))
                parts.append("")

        parts.extend(["=" * 60, "SYNTHESIS:", "Quantum computing represents a fundamental paradigm shift."])
        parts.append(f"Summary written at tick {tick} by {self.agent_id}")

        summary_text = "\n".join(parts)
        write_result = self.write_file("/shared/results/summary.txt", summary_text)
        if write_result.ok:
            self.log(f"[{self.name}] Summary written to /shared/results/summary.txt")
        else:
            self.log(f"[{self.name}] ERROR writing summary: {write_result.error}")

        self.mem_write("summary_complete", True)
        self._summary_written = True
        self._phase = "DONE"
        self.log(f"[{self.name}] Task complete. Entering idle state.")


# ---------------------------------------------------------------------------
# WorkerAgent
# ---------------------------------------------------------------------------

class WorkerAgent(Agent):
    """Worker — executes assigned subtasks using tools and reports results."""

    def __init__(self, name: str = "Worker", priority: int = 4, subtask: Optional[Dict] = None) -> None:
        super().__init__(name=name, priority=priority, memory_allocation=256, time_slice=3)
        self._subtask: Optional[Dict] = subtask
        self._phase: str = "INIT"
        self._findings: List[Dict] = []
        self._queries_done: int = 0
        self._result_sent: bool = False

    def on_spawn(self) -> None:
        self.log(f"[{self.name}] Online. Priority={self.priority}")
        self._phase = "WAITING_TASK"

    def think(self, tick: int) -> None:
        from battousai.ipc import MessageType

        messages = self.read_inbox()
        for msg in messages:
            if msg.message_type == MessageType.TASK:
                self._subtask = msg.payload if isinstance(msg.payload, dict) else {"description": str(msg.payload), "queries": [str(msg.payload)]}
                self.log(f"[{self.name}] Task received: {self._subtask.get('description', 'N/A')}")
                self.mem_write("assigned_subtask", self._subtask)
                self._phase = "EXECUTING"

        if self._phase == "EXECUTING" and self._subtask:
            queries = self._subtask.get("queries", [])
            if self._queries_done < len(queries):
                query = queries[self._queries_done]
                self.log(f"[{self.name}] Searching: '{query}'")
                result = self.use_tool("web_search", query=query)
                if result.ok:
                    search_data = result.value
                    snippets = [r["snippet"] for r in search_data.get("results", [])]
                    self._findings.append({"query": query, "result": " ".join(snippets)})
                    self.log(f"[{self.name}] Got result for '{query}'")
                else:
                    self.log(f"[{self.name}] Tool error: {result.error}")
                self._queries_done += 1
            elif not self._result_sent:
                self._phase = "REPORTING"
                result_payload = {
                    "subtask_description": self._subtask.get("description", ""),
                    "findings": self._findings,
                    "worker_id": self.agent_id,
                    "completed_at_tick": tick,
                }
                agents = self.list_agents()
                coordinator_id = next(
                    (aid for aid in agents if "coordinator" in aid.lower() or "coord" in aid.lower()),
                    None,
                )
                if coordinator_id:
                    self.send_message(recipient_id=coordinator_id, message_type=MessageType.RESULT, payload=result_payload)
                    self.log(f"[{self.name}] Results sent to {coordinator_id}")
                else:
                    self.log(f"[{self.name}] WARN: No coordinator found!")
                self.mem_write("task_complete", True)
                self._result_sent = True
                self._phase = "IDLE"

        self.yield_cpu()


# ---------------------------------------------------------------------------
# MonitorAgent
# ---------------------------------------------------------------------------

class MonitorAgent(Agent):
    """Monitor — observes system health metrics every tick."""

    def __init__(self, name: str = "Monitor", priority: int = 7) -> None:
        super().__init__(name=name, priority=priority, memory_allocation=512, time_slice=2)
        self._tick_samples: List[Dict] = []
        self._sample_interval: int = 5
        self._alert_threshold_agents: int = 20

    def on_spawn(self) -> None:
        self.log(f"[{self.name}] Monitoring system. Sampling every {self._sample_interval} ticks.")

    def think(self, tick: int) -> None:
        messages = self.read_inbox()
        for msg in messages:
            self.log(f"[{self.name}] Broadcast received: {msg.message_type.name}: {str(msg.payload)[:80]}")

        if tick % self._sample_interval == 0 and tick > 0:
            status = self.get_status()
            if status.ok:
                metrics = status.value
                sample = {
                    "tick": tick,
                    "agents_alive": metrics.get("agent_count", 0),
                    "messages_sent": metrics.get("ipc_stats", {}).get("total_sent", 0),
                    "memory_gc_runs": metrics.get("memory_stats", {}).get("gc_runs", 0),
                }
                self._tick_samples.append(sample)
                self.mem_write(f"sample_tick_{tick}", sample)
                self.syscall("publish_topic", topic="system.health", value=sample)
                self.log(f"[{self.name}] Health tick={tick} | agents={sample['agents_alive']} | msgs_sent={sample['messages_sent']}")

        self.yield_cpu()

    def get_report(self) -> str:
        """Produce a formatted metrics report from collected samples."""
        if not self._tick_samples:
            return "MonitorAgent: No samples collected yet."
        lines = ["=== MONITOR REPORT ==="]
        for s in self._tick_samples:
            lines.append(
                f"  tick={s['tick']:04d} | agents={s['agents_alive']:3d} | "
                f"msgs={s['messages_sent']:5d} | gc_runs={s['memory_gc_runs']}"
            )
        return "\n".join(lines)
