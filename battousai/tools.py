"""
tools.py — Battousai Tool Manager
==============================
Registry of capabilities (tools) available to agents in the Autonomous
Intelligence Operating System.

Tools are Python callables registered with the OS at boot time. Agents
request tool access via the kernel syscall `access_tool`. The tool manager
handles:
    - Registration of tools with metadata
    - Per-agent access control (allowlist)
    - Rate limiting (max calls per tick window)
    - Usage logging for audit and analytics
    - Built-in tool implementations

Built-in Tools:
    calculator     — Safe arithmetic expression evaluator
    web_search     — Simulated web search returning canned research data
    code_executor  — Simulated code execution (sandboxed, returns mock output)
    file_reader    — Read a file from the virtual filesystem
    file_writer    — Write a file to the virtual filesystem
"""

from __future__ import annotations

import math
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


class ToolError(Exception):
    """Base tool error."""


class ToolNotFoundError(ToolError):
    """Requested tool does not exist."""


class ToolAccessDeniedError(ToolError):
    """Agent is not allowed to use this tool."""


class ToolRateLimitError(ToolError):
    """Agent has exceeded the rate limit for this tool."""


class ToolExecutionError(ToolError):
    """The tool raised an exception during execution."""


@dataclass
class ToolSpec:
    name: str
    description: str
    callable: Callable[..., Any]
    allowed_agents: Set[str] = field(default_factory=set)
    rate_limit: int = 0
    rate_window: int = 10
    is_simulated: bool = False


@dataclass
class ToolUsageRecord:
    tool_name: str
    agent_id: str
    tick: int
    args: Dict[str, Any]
    result_summary: str
    success: bool
    error_message: Optional[str] = None


class ToolManager:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self._usage_log: List[ToolUsageRecord] = []
        self._call_history: Dict[tuple, List[int]] = {}
        self._current_tick: int = 0
        self._filesystem = None

    def _set_tick(self, tick: int) -> None:
        self._current_tick = tick

    def _inject_filesystem(self, fs) -> None:
        self._filesystem = fs

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ToolError(f"Tool {spec.name!r} is already registered.")
        self._tools[spec.name] = spec

    def deregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def list_tools(self) -> List[str]:
        return sorted(self._tools.keys())

    def get_spec(self, name: str) -> ToolSpec:
        spec = self._tools.get(name)
        if spec is None:
            raise ToolNotFoundError(f"Tool {name!r} is not registered.")
        return spec

    def grant_access(self, tool_name: str, agent_id: str) -> None:
        spec = self.get_spec(tool_name)
        spec.allowed_agents.add(agent_id)

    def revoke_access(self, tool_name: str, agent_id: str) -> None:
        spec = self.get_spec(tool_name)
        spec.allowed_agents.discard(agent_id)

    def _check_access(self, spec: ToolSpec, agent_id: str) -> None:
        if agent_id == "kernel":
            return
        if spec.allowed_agents and agent_id not in spec.allowed_agents:
            raise ToolAccessDeniedError(
                f"Agent {agent_id!r} is not allowed to use tool {spec.name!r}."
            )

    def _check_rate_limit(self, spec: ToolSpec, agent_id: str) -> None:
        if spec.rate_limit == 0:
            return
        key = (agent_id, spec.name)
        history = self._call_history.get(key, [])
        window_start = self._current_tick - spec.rate_window
        history = [t for t in history if t >= window_start]
        self._call_history[key] = history
        if len(history) >= spec.rate_limit:
            raise ToolRateLimitError(
                f"Agent {agent_id!r} exceeded rate limit for tool {spec.name!r} "
                f"({spec.rate_limit} calls per {spec.rate_window} ticks)."
            )

    def _record_call(self, spec: ToolSpec, agent_id: str) -> None:
        key = (agent_id, spec.name)
        self._call_history.setdefault(key, []).append(self._current_tick)

    def execute(
        self,
        agent_id: str,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if args is None:
            args = {}

        spec = self.get_spec(tool_name)
        self._check_access(spec, agent_id)
        self._check_rate_limit(spec, agent_id)
        self._record_call(spec, agent_id)

        try:
            result = spec.callable(**args)
            summary = str(result)[:120]
            self._usage_log.append(ToolUsageRecord(
                tool_name=tool_name,
                agent_id=agent_id,
                tick=self._current_tick,
                args=args,
                result_summary=summary,
                success=True,
            ))
            return result
        except Exception as exc:
            self._usage_log.append(ToolUsageRecord(
                tool_name=tool_name,
                agent_id=agent_id,
                tick=self._current_tick,
                args=args,
                result_summary="",
                success=False,
                error_message=str(exc),
            ))
            raise ToolExecutionError(f"Tool {tool_name!r} raised: {exc}") from exc

    def stats(self) -> Dict[str, Any]:
        by_tool: Dict[str, int] = {}
        by_agent: Dict[str, int] = {}
        for rec in self._usage_log:
            by_tool[rec.tool_name] = by_tool.get(rec.tool_name, 0) + 1
            by_agent[rec.agent_id] = by_agent.get(rec.agent_id, 0) + 1
        return {
            "total_calls": len(self._usage_log),
            "calls_by_tool": by_tool,
            "calls_by_agent": by_agent,
            "registered_tools": self.list_tools(),
        }

    def usage_log(self) -> List[ToolUsageRecord]:
        return list(self._usage_log)


_SEARCH_DB: Dict[str, str] = {
    "quantum computing basics": (
        "Quantum computing leverages quantum mechanical phenomena--superposition and entanglement--"
        "to process information in fundamentally different ways from classical computers. "
        "Qubits can represent 0 and 1 simultaneously (superposition), enabling massive parallelism. "
        "Key players: IBM (127-qubit Eagle processor), Google (achieved claimed quantum supremacy "
        "in 2019 with Sycamore), IonQ, Rigetti, and D-Wave (quantum annealing). "
        "Primary paradigms: gate-based universal quantum computing and quantum annealing."
    ),
    "quantum computing applications": (
        "Promising applications of quantum computing: "
        "1. Cryptography -- Shor's algorithm can factor large integers exponentially faster than "
        "   classical methods, threatening RSA encryption; post-quantum cryptography (NIST 2024 "
        "   standards: CRYSTALS-Kyber, CRYSTALS-Dilithium) is being deployed in response. "
        "2. Drug discovery -- simulating molecular interactions at quantum level for protein folding. "
        "3. Optimization -- logistics, portfolio optimization, supply chain scheduling. "
        "4. Machine learning -- quantum-enhanced gradient descent, kernel methods. "
        "5. Materials science -- discovering new superconductors and catalysts."
    ),
    "quantum computing challenges": (
        "Key technical challenges in quantum computing: "
        "Decoherence -- qubits lose their quantum state due to environmental noise; coherence times "
        "are measured in microseconds to milliseconds. Error rates -- current NISQ (Noisy "
        "Intermediate-Scale Quantum) devices have error rates of ~0.1-1% per gate. "
        "Scalability -- maintaining qubit quality while scaling to millions of logical qubits "
        "requires quantum error correction (surface codes need ~1,000 physical qubits per logical). "
        "Temperature -- superconducting qubits operate near absolute zero (~15 millikelvin)."
    ),
    "quantum supremacy milestones": (
        "Quantum supremacy/advantage milestones: "
        "2019: Google's Sycamore (54 qubits) solved a sampling problem in 200 seconds that Google "
        "claimed would take Summit supercomputer 10,000 years (disputed by IBM). "
        "2020: University of Science and Technology China demonstrated Jiuzhang photonic quantum "
        "computer achieving advantage in Gaussian boson sampling. "
        "2023: IBM unveiled 1,121-qubit Condor and 133-qubit Heron processors. "
        "2024: Microsoft announced topological qubits using Majorana-based hardware approach."
    ),
    "default": (
        "Search results: Quantum computing represents a paradigm shift in computational capability, "
        "combining principles of quantum mechanics with information theory. The field is advancing "
        "rapidly with major investments from tech giants, governments, and startups worldwide."
    ),
}


def _safe_calc(expression: str) -> str:
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
        "log10": math.log10, "floor": math.floor, "ceil": math.ceil,
        "pi": math.pi, "e": math.e, "pow": math.pow,
    }
    safe_pattern = re.compile(r"^[\d\s\+\-\*\/\(\)\.\%\,\_a-zA-Z]+$")
    if not safe_pattern.match(expression):
        return f"ERROR: Expression contains disallowed characters: {expression!r}"
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"ERROR: {e}"


def _simulated_web_search(query: str) -> Dict[str, Any]:
    query_lower = query.lower()
    result_text = _SEARCH_DB.get("default")
    best_match = None
    best_len = 0
    for key in _SEARCH_DB:
        if key == "default":
            continue
        key_words = key.split()
        matching_words = sum(1 for w in key_words if w in query_lower)
        if matching_words > 0 and len(key) > best_len:
            best_match = key
            best_len = len(key)
        if key in query_lower and len(key) > best_len:
            best_match = key
            best_len = len(key)
    if best_match:
        result_text = _SEARCH_DB[best_match]
    return {
        "query": query,
        "source": "SimulatedSearchEngine v1.0",
        "results": [
            {"title": f"Search result for: {query}", "snippet": result_text, "url": "sim://search/1"}
        ],
        "total_results": 1,
        "simulated": True,
    }


def _simulated_code_executor(code: str, language: str = "python") -> Dict[str, Any]:
    lines = code.strip().split("\n")
    return {
        "language": language,
        "lines_executed": len(lines),
        "stdout": f"[SIMULATED] Executed {len(lines)} line(s) of {language} code successfully.",
        "stderr": "",
        "exit_code": 0,
        "simulated": True,
    }


def register_builtin_tools(manager: ToolManager, filesystem=None) -> None:
    if filesystem is not None:
        manager._inject_filesystem(filesystem)

    manager.register(ToolSpec(
        name="calculator",
        description=(
            "Evaluate safe arithmetic expressions. Supports +, -, *, /, **, %, //, "
            "and math functions: abs, round, sqrt, log, floor, ceil, pi, e."
        ),
        callable=_safe_calc,
        is_simulated=False,
        rate_limit=50,
        rate_window=10,
    ))

    manager.register(ToolSpec(
        name="web_search",
        description=(
            "Search the web for information. Returns structured results with title, snippet, "
            "and source URL. (Simulated in this prototype.)"
        ),
        callable=_simulated_web_search,
        is_simulated=True,
        rate_limit=5,
        rate_window=10,
    ))

    manager.register(ToolSpec(
        name="code_executor",
        description=(
            "Execute code in a sandboxed environment. Supports Python and shell. "
            "(Simulated in this prototype.)"
        ),
        callable=_simulated_code_executor,
        is_simulated=True,
        rate_limit=3,
        rate_window=10,
    ))

    def _file_reader(path: str, agent_id: str = "kernel") -> Any:
        if manager._filesystem is None:
            return "ERROR: Filesystem not available"
        return manager._filesystem.read_file(agent_id, path)

    def _file_writer(path: str, data: Any, agent_id: str = "kernel") -> str:
        if manager._filesystem is None:
            return "ERROR: Filesystem not available"
        manager._filesystem.write_file(agent_id, path, data, create_parents=True)
        return f"OK: wrote {len(str(data))} chars to {path!r}"

    manager.register(ToolSpec(
        name="file_reader",
        description="Read the contents of a file in the Battousai virtual filesystem by path.",
        callable=_file_reader,
        is_simulated=False,
        rate_limit=20,
        rate_window=10,
    ))

    manager.register(ToolSpec(
        name="file_writer",
        description="Write data to a file in the Battousai virtual filesystem. Creates parent dirs.",
        callable=_file_writer,
        is_simulated=False,
        rate_limit=20,
        rate_window=10,
    ))
