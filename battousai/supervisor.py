"""
supervisor.py — Agent Supervision Trees
==========================================
Erlang/OTP-inspired supervision for fault-tolerant agent hierarchies.

A Supervisor monitors its child agents and restarts them according to
a configurable restart strategy when they crash or terminate unexpectedly.

Restart Strategies:
    ONE_FOR_ONE  — Only restart the crashed child
    ONE_FOR_ALL  — Restart ALL children if one crashes
    REST_FOR_ONE — Restart the crashed child and all children spawned after it

Restart Intensity:
    max_restarts — max number of restarts within a time window
    window_ticks — the time window in ticks
    If exceeded, the supervisor itself terminates (escalates to parent supervisor)

Child Specs:
    Each child is described by a ChildSpec dataclass that defines:
    - agent_class, name, priority
    - restart_type: PERMANENT (always restart), TRANSIENT (restart on crash), TEMPORARY (never restart)
    - shutdown_timeout: ticks to wait for graceful shutdown
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Type

from battousai.agent import Agent, SyscallResult


class RestartStrategy(Enum):
    ONE_FOR_ONE  = auto()
    ONE_FOR_ALL  = auto()
    REST_FOR_ONE = auto()


class RestartType(Enum):
    PERMANENT  = auto()
    TRANSIENT  = auto()
    TEMPORARY  = auto()


@dataclass
class ChildSpec:
    agent_class: Type[Agent]
    name: str
    priority: int = 5
    restart_type: RestartType = RestartType.PERMANENT
    kwargs: Dict[str, Any] = field(default_factory=dict)
    shutdown_timeout: int = 3

    def __repr__(self) -> str:
        return (
            f"ChildSpec(name={self.name!r}, "
            f"class={self.agent_class.__name__}, "
            f"priority={self.priority}, "
            f"restart={self.restart_type.name})"
        )


@dataclass
class _RestartRecord:
    child_name: str
    tick: int


class SupervisorAgent(Agent):
    def __init__(
        self,
        name: str = "Supervisor",
        priority: int = 2,
        strategy: RestartStrategy = RestartStrategy.ONE_FOR_ONE,
        children: Optional[List[ChildSpec]] = None,
        max_restarts: int = 5,
        window_ticks: int = 20,
    ) -> None:
        super().__init__(
            name=name,
            priority=priority,
            memory_allocation=256,
            time_slice=2,
        )
        self._strategy: RestartStrategy = strategy
        self._child_specs: List[ChildSpec] = list(children or [])
        self._max_restarts: int = max_restarts
        self._window_ticks: int = window_ticks
        self._child_ids: Dict[str, Optional[str]] = {}
        self._restart_log: List[_RestartRecord] = []
        self._normal_exits: set = set()
        self._terminated: bool = False
        self._spawned: bool = False

    def on_spawn(self) -> None:
        self.log(
            f"[{self.name}] SupervisorAgent online. "
            f"Strategy={self._strategy.name}, "
            f"children={len(self._child_specs)}, "
            f"max_restarts={self._max_restarts}/{self._window_ticks}t"
        )

    def on_terminate(self) -> None:
        self.log(f"[{self.name}] Supervisor terminating -- shutting down children.")
        for spec_name, child_id in list(self._child_ids.items()):
            if child_id is not None:
                result = self.syscall("kill_agent", target_id=child_id)
                self.log(
                    f"[{self.name}] Shutdown child {spec_name!r} "
                    f"({child_id}): ok={result.ok}"
                )

    def think(self, tick: int) -> None:
        if not self._spawned:
            self._spawn_all_children(tick)
            self._spawned = True
            self.yield_cpu()
            return

        live_agents = set(self.list_agents())
        crashed: List[str] = []

        for spec_name, child_id in list(self._child_ids.items()):
            if child_id is None:
                continue
            if child_id not in live_agents:
                spec = self._spec_by_name(spec_name)
                if spec is None:
                    continue
                if spec.restart_type == RestartType.TEMPORARY:
                    self.log(
                        f"[{self.name}] Child {spec_name!r} ({child_id}) exited "
                        f"(TEMPORARY -- not restarting)."
                    )
                    self._child_ids[spec_name] = None
                    continue
                if spec.restart_type == RestartType.TRANSIENT:
                    if spec_name in self._normal_exits:
                        self.log(
                            f"[{self.name}] Child {spec_name!r} exited normally "
                            f"(TRANSIENT -- not restarting)."
                        )
                        self._child_ids[spec_name] = None
                        self._normal_exits.discard(spec_name)
                        continue
                self.log(
                    f"[{self.name}] Child {spec_name!r} ({child_id}) CRASHED "
                    f"at tick {tick}."
                )
                crashed.append(spec_name)
                self._child_ids[spec_name] = None

        if crashed:
            self._apply_strategy(crashed, tick)

        self.yield_cpu()

    def _apply_strategy(self, crashed_names: List[str], tick: int) -> None:
        if self._strategy == RestartStrategy.ONE_FOR_ONE:
            for spec_name in crashed_names:
                self._restart_child(spec_name, tick)

        elif self._strategy == RestartStrategy.ONE_FOR_ALL:
            self.log(
                f"[{self.name}] ONE_FOR_ALL triggered -- "
                f"restarting all {len(self._child_specs)} children."
            )
            for spec in self._child_specs:
                child_id = self._child_ids.get(spec.name)
                if child_id is not None:
                    self.syscall("kill_agent", target_id=child_id)
                    self._child_ids[spec.name] = None
            for spec in self._child_specs:
                self._restart_child(spec.name, tick)

        elif self._strategy == RestartStrategy.REST_FOR_ONE:
            spec_names = [s.name for s in self._child_specs]
            indices = [
                spec_names.index(name)
                for name in crashed_names
                if name in spec_names
            ]
            if not indices:
                return
            first_idx = min(indices)
            to_restart = spec_names[first_idx:]
            self.log(
                f"[{self.name}] REST_FOR_ONE -- restarting {len(to_restart)} "
                f"children from index {first_idx}."
            )
            for spec_name in to_restart:
                child_id = self._child_ids.get(spec_name)
                if child_id is not None:
                    self.syscall("kill_agent", target_id=child_id)
                    self._child_ids[spec_name] = None
            for spec_name in to_restart:
                self._restart_child(spec_name, tick)

    def _restart_child(self, spec_name: str, tick: int) -> None:
        if self._restart_intensity_exceeded(tick):
            self.log(
                f"[{self.name}] CRITICAL: Restart intensity exceeded "
                f"({self._max_restarts} restarts / {self._window_ticks} ticks). "
                f"Supervisor {self.name!r} terminating -- escalating to parent."
            )
            self._do_terminate()
            return

        spec = self._spec_by_name(spec_name)
        if spec is None:
            self.log(f"[{self.name}] Cannot restart {spec_name!r} -- spec not found.")
            return

        result = self._spawn_child_spec(spec)
        if result.ok:
            new_id = result.value
            self._child_ids[spec_name] = new_id
            self._restart_log.append(_RestartRecord(child_name=spec_name, tick=tick))
            self.log(
                f"[{self.name}] Restarted {spec_name!r} -> new agent_id={new_id!r} "
                f"(tick={tick}, restarts_in_window={self._count_restarts_in_window(tick)})"
            )
        else:
            self.log(
                f"[{self.name}] ERROR: Failed to restart {spec_name!r}: {result.error}"
            )

    def _spawn_child_spec(self, spec: ChildSpec) -> SyscallResult:
        return self.syscall(
            "spawn_agent",
            agent_class=spec.agent_class,
            agent_name=spec.name,
            priority=spec.priority,
            **spec.kwargs,
        )

    def _spawn_all_children(self, tick: int) -> None:
        self.log(
            f"[{self.name}] Spawning {len(self._child_specs)} initial children..."
        )
        for spec in self._child_specs:
            result = self._spawn_child_spec(spec)
            if result.ok:
                child_id = result.value
                self._child_ids[spec.name] = child_id
                self.log(
                    f"[{self.name}] Spawned child {spec.name!r} -> {child_id!r}"
                )
            else:
                self._child_ids[spec.name] = None
                self.log(
                    f"[{self.name}] ERROR: Failed to spawn {spec.name!r}: {result.error}"
                )

    def _count_restarts_in_window(self, current_tick: int) -> int:
        window_start = current_tick - self._window_ticks
        return sum(1 for r in self._restart_log if r.tick >= window_start)

    def _restart_intensity_exceeded(self, current_tick: int) -> bool:
        return self._count_restarts_in_window(current_tick) >= self._max_restarts

    def _do_terminate(self) -> None:
        for spec_name, child_id in list(self._child_ids.items()):
            if child_id is not None:
                self.syscall("kill_agent", target_id=child_id)
                self._child_ids[spec_name] = None
        self.syscall("kill_agent", target_id=self.agent_id)

    def _spec_by_name(self, name: str) -> Optional[ChildSpec]:
        for spec in self._child_specs:
            if spec.name == name:
                return spec
        return None

    def child_status(self) -> Dict[str, Any]:
        status = {}
        for spec in self._child_specs:
            child_id = self._child_ids.get(spec.name)
            status[spec.name] = {
                "spec": repr(spec),
                "agent_id": child_id,
                "alive": child_id is not None,
                "restart_type": spec.restart_type.name,
            }
        return status

    def restart_history(self) -> List[Dict[str, Any]]:
        return [
            {"child_name": r.child_name, "tick": r.tick}
            for r in self._restart_log
        ]

    def __repr__(self) -> str:
        alive = sum(1 for v in self._child_ids.values() if v is not None)
        return (
            f"SupervisorAgent(id={self.agent_id!r}, "
            f"name={self.name!r}, "
            f"strategy={self._strategy.name}, "
            f"children={len(self._child_specs)}, "
            f"alive={alive})"
        )


class SupervisorTree:
    def __init__(self, kernel: Any = None) -> None:
        self._kernel = kernel
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._root: Optional[str] = None

    def add_node(
        self,
        node_id: str,
        label: Optional[str] = None,
        parent: Optional[str] = None,
    ) -> None:
        self._nodes[node_id] = {
            "label": label or node_id,
            "children": [],
            "parent": parent,
        }
        if parent is None:
            self._root = node_id
        elif parent in self._nodes:
            self._nodes[parent]["children"].append(node_id)

    def render(self, node_id: Optional[str] = None, prefix: str = "", is_last: bool = True) -> str:
        if node_id is None:
            node_id = self._root
        if node_id is None or node_id not in self._nodes:
            return "(empty tree)"

        node = self._nodes[node_id]
        connector = "└── " if is_last else "├── "
        if prefix == "":
            connector = ""

        label = node["label"]
        if self._kernel is not None:
            agent = self._kernel._agents.get(node_id)
            if agent is not None:
                if isinstance(agent, SupervisorAgent):
                    alive = sum(1 for v in agent._child_ids.values() if v is not None)
                    label = f"{label} [{agent._strategy.name}, {alive}/{len(agent._child_specs)} alive]"
                else:
                    label = f"{label} [pid={node_id}]"

        lines = [f"{prefix}{connector}{label}"]

        children = node["children"]
        for i, child_id in enumerate(children):
            is_child_last = (i == len(children) - 1)
            if prefix == "":
                child_prefix = ""
            else:
                child_prefix = prefix + ("    " if is_last else "│   ")
            sub = self.render(child_id, prefix=child_prefix + "    ", is_last=is_child_last)
            lines.append(sub)

        return "\n".join(lines)

    def nodes(self) -> List[str]:
        return list(self._nodes.keys())

    def depth(self, node_id: Optional[str] = None) -> int:
        if node_id is None:
            node_id = self._root
        if node_id is None:
            return 0
        parent = self._nodes.get(node_id, {}).get("parent")
        if parent is None:
            return 0
        return 1 + self.depth(parent)


def build_supervision_tree(kernel: Any, tree_spec: Dict[str, Any]) -> str:
    return _build_node(kernel, tree_spec, parent_id=None)


def _build_node(kernel: Any, spec: Dict[str, Any], parent_id: Optional[str]) -> str:
    strategy_name = spec.get("strategy", "ONE_FOR_ONE")
    strategy = RestartStrategy[strategy_name] if isinstance(strategy_name, str) else strategy_name

    restart_name = spec.get("restart_type", "PERMANENT")
    restart_type = RestartType[restart_name] if isinstance(restart_name, str) else restart_name

    agent_class = spec.get("class", SupervisorAgent)
    name = spec.get("name", "Supervisor")
    priority = spec.get("priority", 2)
    max_restarts = spec.get("max_restarts", 5)
    window_ticks = spec.get("window_ticks", 20)

    child_specs: List[ChildSpec] = []
    raw_children = spec.get("children", [])

    for child_dict in raw_children:
        child_class = child_dict.get("class", Agent)
        child_is_supervisor = issubclass(child_class, SupervisorAgent)

        child_restart_name = child_dict.get("restart_type", "PERMANENT")
        child_restart = (
            RestartType[child_restart_name]
            if isinstance(child_restart_name, str)
            else child_restart_name
        )

        if child_is_supervisor:
            child_strategy_name = child_dict.get("strategy", "ONE_FOR_ONE")
            child_strategy = (
                RestartStrategy[child_strategy_name]
                if isinstance(child_strategy_name, str)
                else child_strategy_name
            )
            nested_child_specs = _build_child_specs(child_dict.get("children", []))
            cspec = ChildSpec(
                agent_class=child_class,
                name=child_dict.get("name", "SubSupervisor"),
                priority=child_dict.get("priority", 3),
                restart_type=child_restart,
                kwargs={
                    "strategy": child_strategy,
                    "children": nested_child_specs,
                    "max_restarts": child_dict.get("max_restarts", 3),
                    "window_ticks": child_dict.get("window_ticks", 10),
                },
                shutdown_timeout=child_dict.get("shutdown_timeout", 3),
            )
        else:
            cspec = ChildSpec(
                agent_class=child_class,
                name=child_dict.get("name", "Child"),
                priority=child_dict.get("priority", 5),
                restart_type=child_restart,
                kwargs=child_dict.get("kwargs", {}),
                shutdown_timeout=child_dict.get("shutdown_timeout", 3),
            )
        child_specs.append(cspec)

    agent_id = kernel.spawn_agent(
        SupervisorAgent,
        name=name,
        priority=priority,
        strategy=strategy,
        children=child_specs,
        max_restarts=max_restarts,
        window_ticks=window_ticks,
    )
    return agent_id


def _build_child_specs(raw_children: List[Dict[str, Any]]) -> List[ChildSpec]:
    specs: List[ChildSpec] = []
    for child_dict in raw_children:
        child_class = child_dict.get("class", Agent)
        child_restart_name = child_dict.get("restart_type", "PERMANENT")
        child_restart = (
            RestartType[child_restart_name]
            if isinstance(child_restart_name, str)
            else child_restart_name
        )
        cspec = ChildSpec(
            agent_class=child_class,
            name=child_dict.get("name", "Child"),
            priority=child_dict.get("priority", 5),
            restart_type=child_restart,
            kwargs=child_dict.get("kwargs", {}),
            shutdown_timeout=child_dict.get("shutdown_timeout", 3),
        )
        specs.append(cspec)
    return specs
