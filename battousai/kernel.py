"""
kernel.py — Battousai Kernel
=========================
Central coordinator of the Autonomous Intelligence Operating System.

The Kernel is the nucleus of Battousai. It:
    1. Boots the OS — initialises all subsystems in the correct order
    2. Manages the agent lifecycle — spawn, tick, terminate, collect
    3. Runs the event loop — drives the scheduler each tick
    4. Provides the syscall interface — the only way agents touch OS resources

Syscalls (agents call these via Agent.syscall(name, **kwargs)):
    spawn_agent    — Create and register a new agent
    kill_agent     — Terminate an agent by ID
    send_message   — Post a message to another agent's mailbox
    read_inbox     — Drain messages from the caller's own mailbox
    read_memory    — Read a value from the caller's private memory
    write_memory   — Write a value to the caller's private memory
    access_tool    — Execute a registered tool
    list_agents    — Return IDs of all living agents
    get_status     — Return a system-wide metrics snapshot
    yield_cpu      — Voluntarily give up CPU slice
    write_file     — Write a file to the virtual filesystem
    read_file      — Read a file from the virtual filesystem
    list_dir       — List a directory in the virtual filesystem
    publish_topic  — Publish a value to the IPC bulletin board
    subscribe      — Subscribe to a bulletin board topic

Boot Sequence:
    Logger → Filesystem → Memory Manager → IPC Manager →
    Tool Manager → Scheduler → (register tools) → (spawn initial agents) → Event Loop
"""

from __future__ import annotations

import traceback
import uuid
from typing import Any, Dict, List, Optional, Type

from battousai.agent import Agent, SyscallResult
from battousai.filesystem import VirtualFilesystem
from battousai.ipc import IPCManager, Message, MessageType, BROADCAST_ALL
from battousai.logger import Logger, LogLevel
from battousai.memory import MemoryManager, MemoryType
from battousai.scheduler import AgentState, Scheduler
from battousai.tools import ToolManager


class KernelPanic(Exception):
    """Unrecoverable OS error."""


class Kernel:
    """
    Battousai Kernel — boots, manages, and drives the multi-agent OS.

    Usage::

        kernel = Kernel(max_ticks=50)
        kernel.boot()
        kernel.run()
        report = kernel.system_report()
    """

    VERSION = "0.3.0"

    def __init__(self, max_ticks: int = 50, debug: bool = False) -> None:
        self.max_ticks = max_ticks
        self.debug = debug
        self._tick: int = 0
        self._booted: bool = False
        self._halted: bool = False
        self._agents: Dict[str, Agent] = {}        # agent_id → Agent instance
        self._agent_counter: int = 0               # monotonic ID counter

        # Subsystems (initialised during boot)
        self.logger: Logger = Logger(
            min_level=LogLevel.DEBUG if debug else LogLevel.INFO,
            console_output=True,
        )
        self.filesystem: VirtualFilesystem = VirtualFilesystem()
        self.memory: MemoryManager = MemoryManager()
        self.ipc: IPCManager = IPCManager()
        self.tools: ToolManager = ToolManager()
        self.scheduler: Scheduler = Scheduler()

        # Stats
        self._spawn_count: int = 0
        self._kill_count: int = 0
        self._syscall_count: int = 0

    # ------------------------------------------------------------------
    # Boot
    # ------------------------------------------------------------------

    def boot(self) -> None:
        """
        Boot the OS.

        Initialises all subsystems, wires inter-subsystem references,
        and creates the standard directory tree.
        """
        self.logger.system("kernel", f"Battousai v{self.VERSION} booting...")

        # Wire cross-subsystem references
        self.logger._inject_filesystem(self.filesystem)
        self.tools._inject_filesystem(self.filesystem)

        # Bootstrap the virtual filesystem directory tree
        self.filesystem._init_standard_dirs()
        self.logger.system("kernel", "Filesystem initialised (/agents, /shared, /system/logs)")

        # Create the shared memory region all agents can use
        self.memory.create_shared_region("global", max_keys=1024)
        self.logger.system("kernel", "Memory manager online (global shared region created)")

        self.logger.system("kernel", "IPC manager online")
        self.logger.system("kernel", "Scheduler online")

        from battousai.tools import register_builtin_tools
        register_builtin_tools(self.tools, self.filesystem)
        self.logger.system("kernel", f"Tools registered: {self.tools.list_tools()}")

        self._booted = True
        self.logger.system("kernel", "Boot sequence complete. Ready.")

    # ------------------------------------------------------------------
    # Agent Lifecycle
    # ------------------------------------------------------------------

    def spawn_agent(
        self,
        agent_class: Type[Agent],
        name: str,
        priority: int = 5,
        **kwargs: Any,
    ) -> str:
        """
        Instantiate and register a new agent.

        Returns the assigned agent_id.
        """
        if not self._booted:
            raise KernelPanic("Cannot spawn agents before boot()")

        # Generate unique agent ID
        self._agent_counter += 1
        agent_id = f"{name.lower().replace(' ', '_')}_{self._agent_counter:04d}"

        # Instantiate agent (strip kernel-reserved kwargs before passing to __init__)
        init_kwargs = {k: v for k, v in kwargs.items() if k not in ("agent_class",)}
        agent: Agent = agent_class(name=name, priority=priority, **init_kwargs)
        agent.kernel = self
        agent.agent_id = agent_id
        agent._spawn_tick = self._tick

        # Register with subsystems
        self.ipc.register_agent(agent_id)
        self.memory.create_agent_space(agent_id, max_keys=agent.memory_allocation)
        self.filesystem.mkdir(f"/agents/{agent_id}")
        self.filesystem.mkdir(f"/agents/{agent_id}/workspace")
        self.scheduler.add_process(
            agent_id=agent_id,
            name=name,
            priority=priority,
            time_slice=agent.time_slice,
            spawn_tick=self._tick,
        )

        self._agents[agent_id] = agent
        self._spawn_count += 1

        self.logger.system("kernel", f"Spawned agent {agent_id!r} (class={agent_class.__name__}, priority={priority})")

        # Call the agent's spawn hook
        try:
            agent.on_spawn()
        except Exception as exc:
            self.logger.error("kernel", f"on_spawn() failed for {agent_id!r}: {exc}")

        return agent_id

    def kill_agent(self, agent_id: str) -> bool:
        """Terminate an agent by ID."""
        agent = self._agents.get(agent_id)
        if agent is None:
            return False

        self.logger.system("kernel", f"Killing agent {agent_id!r}")

        # Call terminate hook
        try:
            agent.on_terminate()
        except Exception as exc:
            self.logger.error("kernel", f"on_terminate() failed for {agent_id!r}: {exc}")

        self.scheduler.terminate_process(agent_id)
        self.ipc.unregister_agent(agent_id)
        self.memory.delete_agent_space(agent_id)
        del self._agents[agent_id]
        self._kill_count += 1
        return True

    # ------------------------------------------------------------------
    # Syscall Dispatch
    # ------------------------------------------------------------------

    SYSCALLS = {
        "spawn_agent", "kill_agent", "send_message", "read_inbox",
        "read_memory", "write_memory", "access_tool", "list_agents",
        "get_status", "yield_cpu", "write_file", "read_file", "list_dir",
        "publish_topic", "subscribe",
    }

    def _dispatch_syscall(self, caller_id: str, name: str, **kwargs: Any) -> SyscallResult:
        """Route a syscall from an agent to the appropriate handler."""
        self._syscall_count += 1

        if name not in self.SYSCALLS:
            return SyscallResult(ok=False, error=f"Unknown syscall: {name!r}")

        handler = getattr(self, f"_syscall_{name}", None)
        if handler is None:
            return SyscallResult(ok=False, error=f"Syscall {name!r} not implemented")

        try:
            return handler(caller_id, **kwargs)
        except Exception as exc:
            if self.debug:
                traceback.print_exc()
            return SyscallResult(ok=False, error=str(exc))

    # ------------------------------------------------------------------
    # Syscall handlers
    # ------------------------------------------------------------------

    def _syscall_spawn_agent(
        self,
        caller_id: str,
        agent_class: Type[Agent],
        agent_name: str,
        priority: int = 5,
        **kwargs: Any,
    ) -> SyscallResult:
        agent_id = self.spawn_agent(agent_class, agent_name, priority, **kwargs)
        return SyscallResult(ok=True, value=agent_id)

    def _syscall_kill_agent(self, caller_id: str, target_id: str) -> SyscallResult:
        ok = self.kill_agent(target_id)
        return SyscallResult(ok=ok, error=None if ok else f"Agent {target_id!r} not found")

    def _syscall_send_message(
        self,
        caller_id: str,
        recipient_id: str,
        message_type: MessageType,
        payload: Any,
        correlation_id: Optional[str] = None,
        ttl: int = 0,
    ) -> SyscallResult:
        msg = Message(
            sender_id=caller_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            timestamp=self._tick,
            correlation_id=correlation_id,
            ttl=ttl,
        )
        ok = self.ipc.send(msg)
        if not ok and recipient_id != BROADCAST_ALL:
            return SyscallResult(ok=False, error=f"Delivery failed: {recipient_id!r} not found or mailbox full")
        return SyscallResult(ok=True, value=msg.message_id)

    def _syscall_read_inbox(self, caller_id: str) -> SyscallResult:
        mb = self.ipc.get_mailbox(caller_id)
        if mb is None:
            return SyscallResult(ok=False, error="No mailbox registered")
        messages = mb.receive_all(self._tick)
        return SyscallResult(ok=True, value=messages)

    def _syscall_read_memory(self, caller_id: str, key: str) -> SyscallResult:
        try:
            value = self.memory.agent_read(caller_id, key, self._tick)
            return SyscallResult(ok=True, value=value)
        except Exception as exc:
            return SyscallResult(ok=False, error=str(exc))

    def _syscall_write_memory(
        self,
        caller_id: str,
        key: str,
        value: Any,
        memory_type: MemoryType = MemoryType.LONG_TERM,
        ttl_ticks: Optional[int] = None,
    ) -> SyscallResult:
        try:
            entry = self.memory.agent_write(
                caller_id, key, value, memory_type, self._tick, ttl_ticks
            )
            return SyscallResult(ok=True, value=entry.key)
        except Exception as exc:
            return SyscallResult(ok=False, error=str(exc))

    def _syscall_access_tool(
        self,
        caller_id: str,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> SyscallResult:
        try:
            result = self.tools.execute(caller_id, tool_name, args or {})
            return SyscallResult(ok=True, value=result)
        except Exception as exc:
            return SyscallResult(ok=False, error=str(exc))

    def _syscall_list_agents(self, caller_id: str) -> SyscallResult:
        alive = [
            aid for aid, agent in self._agents.items()
        ]
        return SyscallResult(ok=True, value=alive)

    def _syscall_get_status(self, caller_id: str) -> SyscallResult:
        status = {
            "tick": self._tick,
            "agent_count": len(self._agents),
            "agents": {aid: repr(a) for aid, a in self._agents.items()},
            "scheduler_stats": self.scheduler.stats(),
            "ipc_stats": self.ipc.stats(),
            "memory_stats": self.memory.stats(),
            "tool_stats": self.tools.stats(),
            "fs_stats": self.filesystem.stats(),
            "spawn_count": self._spawn_count,
            "kill_count": self._kill_count,
            "syscall_count": self._syscall_count,
        }
        return SyscallResult(ok=True, value=status)

    def _syscall_yield_cpu(self, caller_id: str) -> SyscallResult:
        proc = self.scheduler.get_process(caller_id)
        if proc:
            proc.yield_cpu()
        return SyscallResult(ok=True)

    def _syscall_write_file(
        self,
        caller_id: str,
        path: str,
        data: Any,
        create_parents: bool = True,
        world_readable: bool = True,
    ) -> SyscallResult:
        try:
            self.filesystem.write_file(
                caller_id, path, data,
                create_parents=create_parents,
                world_readable=world_readable,
            )
            return SyscallResult(ok=True, value=path)
        except Exception as exc:
            return SyscallResult(ok=False, error=str(exc))

    def _syscall_read_file(self, caller_id: str, path: str) -> SyscallResult:
        try:
            data = self.filesystem.read_file(caller_id, path)
            return SyscallResult(ok=True, value=data)
        except Exception as exc:
            return SyscallResult(ok=False, error=str(exc))

    def _syscall_list_dir(self, caller_id: str, path: str) -> SyscallResult:
        try:
            entries = self.filesystem.list_dir(caller_id, path)
            return SyscallResult(ok=True, value=entries)
        except Exception as exc:
            return SyscallResult(ok=False, error=str(exc))

    def _syscall_publish_topic(
        self, caller_id: str, topic: str, value: Any
    ) -> SyscallResult:
        self.ipc.publish(topic, value, caller_id, self._tick)
        return SyscallResult(ok=True)

    def _syscall_subscribe(self, caller_id: str, topic: str) -> SyscallResult:
        self.ipc.subscribe(topic, caller_id)
        return SyscallResult(ok=True)

    # ------------------------------------------------------------------
    # Event Loop
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """
        Advance the OS by one tick.

        Order of operations each tick:
            1. Update tick counter in all subsystems
            2. Run the scheduler in a round to execute all ready agents once,
               in priority order (each agent may run up to its time_slice)
            3. Run memory garbage collection
            4. Collect terminated agents

        All READY agents get a turn each tick — the scheduler selects the
        next runnable agent repeatedly until each agent has been offered
        exactly one think() call this tick (tracked by a 'seen' set).
        """
        self._tick += 1
        # Update tick in subsystems
        self.logger._set_tick(self._tick)
        self.filesystem._set_tick(self._tick)
        self.tools._set_tick(self._tick)

        self.logger.debug("kernel", f"--- Tick {self._tick} ---")

        # Snapshot all ready agents at the START of this tick (in priority order)
        # so that agents spawned mid-tick don't run until next tick
        ready_this_tick: list = []
        for prio in range(self.scheduler.NUM_PRIORITIES):
            q = self.scheduler._queues[prio]
            for proc in list(q):
                if proc.state == AgentState.READY:
                    ready_this_tick.append(proc.agent_id)

        # Run each ready agent exactly once this tick
        for agent_id in ready_this_tick:
            # Skip agents that may have been killed mid-tick
            if agent_id not in self._agents:
                continue
            proc = self.scheduler.get_process(agent_id)
            if proc is None or proc.state == AgentState.TERMINATED:
                continue
            # Mark as running
            proc.state = AgentState.RUNNING
            proc.ticks_run += 1
            proc.remaining_ticks -= 1
            self.scheduler.total_scheduled += 1

            agent = self._agents.get(agent_id)
            if agent is not None:
                try:
                    agent._tick(self._tick)
                except Exception as exc:
                    self.logger.error(
                        "kernel",
                        f"Agent {agent_id!r} raised exception in think(): {exc}",
                    )
                    if self.debug:
                        traceback.print_exc()

            # After think(), return to READY unless yield/terminated
            if proc.state == AgentState.RUNNING:
                proc.state = AgentState.READY
                proc.reset_slice()
            elif proc.state == AgentState.WAITING or proc.state == AgentState.BLOCKED:
                # Remove from queue — will be re-added by unblock_process
                try:
                    self.scheduler._queues[proc.priority].remove(proc)
                except ValueError:
                    pass
            elif proc.yield_requested:
                proc.state = AgentState.READY
                proc.reset_slice()
                # Rotate to back of queue (already there from yield logic)

        # Collect terminated agents
        terminated = [
            aid for aid, proc in self.scheduler.all_processes().items()
            if proc.state == AgentState.TERMINATED
        ]
        for aid in terminated:
            if aid in self._agents:
                self.logger.system("kernel", f"Collecting terminated agent {aid!r}")
                self._agents.pop(aid)
                self.ipc.unregister_agent(aid)
                self.memory.delete_agent_space(aid)

        # Memory garbage collection
        evictions = self.memory.gc_tick(self._tick)
        if evictions:
            self.logger.debug("kernel", f"GC evicted: {evictions}")

    def run(self, ticks: Optional[int] = None) -> None:
        """
        Run the event loop for `ticks` iterations (default: self.max_ticks).
        """
        if not self._booted:
            raise KernelPanic("Call boot() before run()")

        n = ticks if ticks is not None else self.max_ticks
        self.logger.system("kernel", f"Event loop starting ({n} ticks)")

        for _ in range(n):
            if self._halted:
                break
            self.tick()

        self.logger.system("kernel", f"Event loop finished at tick {self._tick}")

    def halt(self) -> None:
        """Gracefully stop the event loop after the current tick."""
        self._halted = True
        self.logger.system("kernel", "Halt requested")

    # ------------------------------------------------------------------
    # System Report
    # ------------------------------------------------------------------

    def system_report(self) -> str:
        """Generate a formatted end-of-run system report."""
        sep = "=" * 70
        lines = [
            "",
            sep,
            f"  Battousai SYSTEM REPORT  —  v{self.VERSION}",
            sep,
            f"  Total ticks run       : {self._tick}",
            f"  Agents spawned        : {self._spawn_count}",
            f"  Agents killed         : {self._kill_count}",
            f"  Agents alive          : {len(self._agents)}",
            f"  Syscalls dispatched   : {self._syscall_count}",
            "",
            "  IPC",
            f"    Messages sent       : {self.ipc.total_sent}",
            f"    Messages dropped    : {self.ipc.total_dropped}",
            f"    Bulletin topics     : {len(self.ipc.bulletin_board.topics())}",
            "",
            "  Scheduler",
        ]
        sched = self.scheduler.stats()
        for state, count in sched["state_counts"].items():
            if count > 0:
                lines.append(f"    {state:<12}        : {count}")
        lines += [
            f"    Preemptions         : {sched['preemptions']}",
            f"    Voluntary yields    : {sched['voluntary_yields']}",
            f"    Total scheduled     : {sched['total_scheduled']}",
            "",
            "  Tools",
        ]
        tool_stats = self.tools.stats()
        lines.append(f"    Total tool calls    : {tool_stats['total_calls']}")
        for tool, count in sorted(tool_stats["calls_by_tool"].items()):
            lines.append(f"    {tool:<20}  : {count} calls")
        lines += [
            "",
            "  Memory",
        ]
        mem_stats = self.memory.stats()
        for aid, info in mem_stats["agents"].items():
            lines.append(f"    {aid:<30}: {info['used']}/{info['max']} keys")
        lines += [
            f"    GC runs             : {mem_stats['gc_runs']}",
            "",
            "  Filesystem",
        ]
        fs_stats = self.filesystem.stats()
        lines.append(f"    Total files         : {fs_stats['total_files']}")
        lines.append(f"    Total size          : {fs_stats['total_size_bytes']} bytes")
        for aid, count in fs_stats["files_by_agent"].items():
            lines.append(f"    {aid:<30}: {count} file(s)")
        lines += [
            "",
            "  Log summary           : " + self.logger.get_summary(),
            "",
            "  Filesystem tree:",
        ]
        lines.append(self.filesystem.tree("/"))
        lines.append(sep)

        # Print the summary file if it was created
        try:
            summary = self.filesystem.read_file("kernel", "/shared/results/summary.txt")
            lines += [
                "",
                "  /shared/results/summary.txt:",
                "-" * 60,
                summary,
                "-" * 60,
            ]
        except Exception:
            lines.append("  (no summary file found)")

        lines.append(sep)
        return "\n".join(lines)
