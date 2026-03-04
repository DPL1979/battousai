"""
scheduler.py — Battousai Priority-Based Preemptive Scheduler
=========================================================
Manages the execution order of agent processes in the Autonomous Intelligence
Operating System.

Scheduling Policy:
    Priority-based preemptive scheduling with round-robin within the same
    priority band.

    Priority levels: 0–9 (0 is the HIGHEST priority, 9 is lowest).
    Lower number = runs first. Real-time system agents use priority 0;
    normal worker agents default to priority 5.

Agent States:
    READY       — Eligible to run; waiting for CPU time
    RUNNING     — Currently executing its `think()` method this tick
    WAITING     — Blocked on a message/reply (will not be scheduled)
    BLOCKED     — Blocked on a resource or I/O (future use)
    TERMINATED  — Execution complete; will be collected and removed

Time-Slicing:
    Each agent is granted a configurable number of consecutive ticks
    (time_slice). After the slice expires the agent is preempted and
    moved to the back of its priority queue.

    Agents may yield voluntarily before their slice expires by calling
    `yield_cpu()` on the process descriptor.

Preemption:
    A higher-priority READY agent arriving mid-slice immediately preempts
    the current agent. The preempted agent retains its remaining slice for
    the next time it is scheduled.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple


class AgentState(Enum):
    READY      = auto()
    RUNNING    = auto()
    WAITING    = auto()
    BLOCKED    = auto()
    TERMINATED = auto()


@dataclass
class ProcessDescriptor:
    """
    The OS's view of a single agent process.

    The scheduler operates on these descriptors, not on agent objects directly.
    The kernel links each descriptor to its Agent instance.
    """
    agent_id: str
    name: str
    priority: int                    # 0 (highest) – 9 (lowest)
    state: AgentState = AgentState.READY
    time_slice: int = 3              # ticks per scheduling turn
    remaining_ticks: int = 0         # ticks left in current slice
    ticks_run: int = 0               # total ticks of CPU time consumed
    spawn_tick: int = 0
    yield_requested: bool = False    # Agent asked to give up CPU early
    wait_for: Optional[str] = None   # correlation_id of a pending reply

    def __post_init__(self) -> None:
        self.remaining_ticks = self.time_slice

    def yield_cpu(self) -> None:
        """Signal that the agent voluntarily yields the remainder of its slice."""
        self.yield_requested = True

    def reset_slice(self) -> None:
        self.remaining_ticks = self.time_slice
        self.yield_requested = False


class Scheduler:
    """
    Priority-based preemptive scheduler for Battousai.

    Internal structure:
        _queues: Dict[int, Deque[ProcessDescriptor]]
            One deque per priority level (0-9).
            Agents at the same priority rotate round-robin.

        _running: Optional[ProcessDescriptor]
            The currently executing process (at most one per tick).

        _all: Dict[str, ProcessDescriptor]
            Lookup table by agent_id for O(1) state queries.
    """

    NUM_PRIORITIES = 10

    def __init__(self, default_time_slice: int = 3) -> None:
        self.default_time_slice = default_time_slice
        self._queues: Dict[int, Deque[ProcessDescriptor]] = {
            i: deque() for i in range(self.NUM_PRIORITIES)
        }
        self._running: Optional[ProcessDescriptor] = None
        self._all: Dict[str, ProcessDescriptor] = {}
        self._tick: int = 0
        self.preemptions: int = 0
        self.voluntary_yields: int = 0
        self.total_scheduled: int = 0

    # ------------------------------------------------------------------
    # Process registration
    # ------------------------------------------------------------------

    def add_process(
        self,
        agent_id: str,
        name: str,
        priority: int,
        time_slice: Optional[int] = None,
        spawn_tick: int = 0,
    ) -> ProcessDescriptor:
        """Register a new process with the scheduler in READY state."""
        priority = max(0, min(9, priority))
        ts = time_slice if time_slice is not None else self.default_time_slice
        proc = ProcessDescriptor(
            agent_id=agent_id,
            name=name,
            priority=priority,
            time_slice=ts,
            spawn_tick=spawn_tick,
        )
        proc.remaining_ticks = ts
        self._all[agent_id] = proc
        self._queues[priority].append(proc)
        return proc

    def remove_process(self, agent_id: str) -> bool:
        """Remove a process from the scheduler entirely (after termination)."""
        proc = self._all.pop(agent_id, None)
        if proc is None:
            return False
        # Remove from priority queue if it is there
        q = self._queues[proc.priority]
        try:
            q.remove(proc)
        except ValueError:
            pass
        if self._running is proc:
            self._running = None
        return True

    def get_process(self, agent_id: str) -> Optional[ProcessDescriptor]:
        return self._all.get(agent_id)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def block_process(self, agent_id: str, wait_for: Optional[str] = None) -> None:
        """Move a process to WAITING (blocked on a reply/resource)."""
        proc = self._all.get(agent_id)
        if proc is None:
            return
        prev_state = proc.state
        proc.state = AgentState.WAITING
        proc.wait_for = wait_for
        # Remove from the ready queue if it is there
        q = self._queues[proc.priority]
        try:
            q.remove(proc)
        except ValueError:
            pass
        if self._running is proc:
            self._running = None

    def unblock_process(self, agent_id: str) -> None:
        """Return a WAITING/BLOCKED process to READY state."""
        proc = self._all.get(agent_id)
        if proc is None:
            return
        if proc.state in (AgentState.WAITING, AgentState.BLOCKED):
            proc.state = AgentState.READY
            proc.wait_for = None
            proc.reset_slice()
            self._queues[proc.priority].append(proc)

    def terminate_process(self, agent_id: str) -> None:
        """Mark a process as TERMINATED (will not be scheduled again)."""
        proc = self._all.get(agent_id)
        if proc is None:
            return
        proc.state = AgentState.TERMINATED
        q = self._queues[proc.priority]
        try:
            q.remove(proc)
        except ValueError:
            pass
        if self._running is proc:
            self._running = None

    def reprioritize(self, agent_id: str, new_priority: int) -> None:
        """Change an agent's priority. Takes effect on the next scheduling decision."""
        proc = self._all.get(agent_id)
        if proc is None:
            return
        new_priority = max(0, min(9, new_priority))
        old_q = self._queues[proc.priority]
        try:
            old_q.remove(proc)
        except ValueError:
            pass
        proc.priority = new_priority
        if proc.state == AgentState.READY:
            self._queues[new_priority].append(proc)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _pick_next(self) -> Optional[ProcessDescriptor]:
        """Return the highest-priority READY process, or None."""
        for priority in range(self.NUM_PRIORITIES):
            q = self._queues[priority]
            if q:
                # Check front of queue is actually READY
                while q and q[0].state != AgentState.READY:
                    q.popleft()  # Clean up stale entries
                if q:
                    return q[0]
        return None

    def tick(self, current_tick: int) -> Optional[ProcessDescriptor]:
        """
        Advance the scheduler by one tick.

        Returns the ProcessDescriptor of the agent that should run this tick,
        or None if there are no runnable agents.

        Algorithm:
        1. Determine the next candidate (highest-priority READY)
        2. If a different agent is currently running → preempt it
        3. Decrement the running agent's remaining_ticks
        4. If slice expires or yield requested → rotate to back of queue
        5. Return the selected process
        """
        self._tick = current_tick

        # Clean up terminated processes from all queues
        for priority in range(self.NUM_PRIORITIES):
            q = self._queues[priority]
            to_remove = [p for p in q if p.state == AgentState.TERMINATED]
            for p in to_remove:
                try:
                    q.remove(p)
                except ValueError:
                    pass

        next_proc = self._pick_next()

        if next_proc is None:
            if self._running is not None:
                self._running.state = AgentState.READY
                self._running = None
            return None

        # Preemption: higher-priority agent arrived
        if self._running is not None and self._running is not next_proc:
            if next_proc.priority < self._running.priority:
                # Preempt current
                self._running.state = AgentState.READY
                # Put back at front of its queue so it resumes with remaining slice
                self._queues[self._running.priority].appendleft(self._running)
                self.preemptions += 1
                self._running = None

        # Select the process to run
        if self._running is None or self._running.state != AgentState.RUNNING:
            if next_proc in self._queues[next_proc.priority]:
                self._queues[next_proc.priority].remove(next_proc)
            self._running = next_proc

        proc = self._running
        proc.state = AgentState.RUNNING
        proc.ticks_run += 1
        proc.remaining_ticks -= 1
        self.total_scheduled += 1

        # Slice exhausted or voluntary yield
        if proc.remaining_ticks <= 0 or proc.yield_requested:
            if proc.yield_requested:
                self.voluntary_yields += 1
            proc.state = AgentState.READY
            proc.reset_slice()
            self._queues[proc.priority].append(proc)
            self._running = None

        return proc

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def ready_queue_snapshot(self) -> List[Tuple[int, List[str]]]:
        """Return a list of (priority, [agent_ids]) for all non-empty queues."""
        return [
            (p, [proc.agent_id for proc in q])
            for p, q in sorted(self._queues.items())
            if q
        ]

    def all_processes(self) -> Dict[str, ProcessDescriptor]:
        return dict(self._all)

    def get_state(self, agent_id: str) -> Optional[AgentState]:
        proc = self._all.get(agent_id)
        return proc.state if proc else None

    def stats(self) -> Dict[str, object]:
        states: Dict[str, int] = {s.name: 0 for s in AgentState}
        for proc in self._all.values():
            states[proc.state.name] += 1
        return {
            "total_processes": len(self._all),
            "state_counts": states,
            "preemptions": self.preemptions,
            "voluntary_yields": self.voluntary_yields,
            "total_scheduled": self.total_scheduled,
            "ready_queue": self.ready_queue_snapshot(),
        }
