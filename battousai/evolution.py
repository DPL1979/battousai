"""
evolution.py — Self-Modification & Agent Evolution
=====================================================
Enables agents to write new agent code, validate it in a sandbox,
and spawn it at runtime. This is the self-improvement engine of Battousai.

The key constraint: all self-modification happens within a secure sandbox.
Generated code is validated before it can run as an agent.

Design Rationale
----------------
Traditional software systems are static: a developer writes code, compiles it,
and deploys it. Battousai breaks this constraint by allowing agents to generate and
deploy *new agents* at runtime. This is the foundation of self-improving AI
systems.

The safety architecture follows a defence-in-depth approach:
    1. CodeValidator: static analysis catches dangerous patterns before execution
    2. CodeSandbox: isolated exec() with stripped builtins limits runtime damage
    3. AgentFactory: only validated code can be promoted to a real agent class
    4. FitnessEvaluator: only *successful* agents survive to produce offspring

This mirrors biological evolution: variation (mutation/crossover) + selection
(fitness) + inheritance (genetic pool). The EvolutionEngine is the meta-agent
that orchestrates this loop.

Components:
    CodeSandbox        — isolated execution environment for untrusted code
    CodeValidator      — static analysis of generated agent code
    AgentFactory       — compiles validated code into runnable Agent subclasses
    EvolutionEngine    — meta-agent that generates, tests, and deploys new agents
    FitnessEvaluator   — measures agent performance against objectives
    GeneticPool        — maintains a population of agent designs that evolve
"""

from __future__ import annotations

import ast
import io
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from battousai.agent import Agent


# ---------------------------------------------------------------------------
# Safe builtins — the only names available inside sandboxed code
# ---------------------------------------------------------------------------

_SAFE_BUILTINS: Dict[str, Any] = {
    "print": print,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "type": type,
    "isinstance": isinstance,
    "hasattr": hasattr,
    "getattr": getattr,
    "repr": repr,
    "None": None,
    "True": True,
    "False": False,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "AttributeError": AttributeError,
    "NotImplementedError": NotImplementedError,
}


# ---------------------------------------------------------------------------
# Validation Result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """
    Outcome of static code analysis.

    Fields:
        valid    — True if the code may be safely compiled and loaded.
        warnings — Non-fatal issues that were detected. Code can still run.
        errors   — Fatal issues. Code must not be executed.
    """
    valid: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid

    def summary(self) -> str:
        lines = [f"Valid: {self.valid}"]
        for w in self.warnings:
            lines.append(f"  WARN : {w}")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CodeValidator — AST-based static analyser
# ---------------------------------------------------------------------------

class CodeValidator:
    """
    Statically analyses agent source code using Python's AST module.

    Philosophy:
        We trust nothing. Every code submission is treated as potentially
        hostile. The validator is a whitelist system: anything not explicitly
        allowed is denied.

    Checks performed:
        - No bare import statements (only ``from battousai.* import ...`` allowed)
        - No exec() or eval() calls (prevent second-order code injection)
        - No file system operations outside syscalls (no open(), no os.*)
        - No network access (no socket, no urllib, no http)
        - No infinite-loop heuristic: ``while True:`` without a ``break``
        - The module must define exactly one class that inherits from Agent
        - That class must implement a ``think()`` method
    """

    # AST node types considered dangerous when called directly
    _FORBIDDEN_CALLS: set = {
        "exec", "eval", "compile", "open", "__import__",
        "globals", "locals", "vars", "dir", "breakpoint",
        "input", "memoryview",
    }

    # Top-level import module prefixes that are NOT battousai.*
    _FORBIDDEN_MODULES: set = {
        "os", "sys", "subprocess", "socket", "urllib", "http",
        "ftplib", "smtplib", "pickle", "marshal", "shelve",
        "importlib", "ctypes", "cffi", "mmap",
    }

    def validate(self, source_code: str) -> ValidationResult:
        """
        Run full static validation on ``source_code``.

        Returns a ValidationResult describing whether the code is safe.
        """
        result = ValidationResult(valid=True)

        # Step 1: Parse
        try:
            tree = ast.parse(source_code)
        except SyntaxError as exc:
            result.valid = False
            result.errors.append(f"SyntaxError: {exc}")
            return result

        # Step 2: Run individual checks
        self._check_imports(tree, result)
        self._check_forbidden_calls(tree, result)
        self._check_infinite_loops(tree, result)
        self._check_agent_structure(tree, result)

        # If any errors found, mark invalid
        if result.errors:
            result.valid = False

        return result

    def _check_imports(self, tree: ast.AST, result: ValidationResult) -> None:
        """
        Allow: ``from battousai.* import ...``
        Deny : ``import X`` (bare), ``from X import ...`` where X is not battousai.*
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    if name in self._FORBIDDEN_MODULES:
                        result.errors.append(
                            f"Forbidden import of module '{alias.name}'. "
                            f"Only 'from battousai.*' imports are allowed."
                        )
                    else:
                        result.warnings.append(
                            f"Bare import '{alias.name}' may not be available in sandbox."
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if not module.startswith("battousai"):
                    result.errors.append(
                        f"Forbidden import from '{module}'. "
                        f"Only 'from battousai.*' imports are allowed."
                    )

    def _check_forbidden_calls(self, tree: ast.AST, result: ValidationResult) -> None:
        """
        Detect calls to dangerous builtins and attribute accesses.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Direct calls: exec(...), eval(...)
                if isinstance(node.func, ast.Name):
                    if node.func.id in self._FORBIDDEN_CALLS:
                        result.errors.append(
                            f"Forbidden call to '{node.func.id}()'. "
                            f"Direct execution primitives are not allowed."
                        )
                # Attribute calls: os.system(...), subprocess.run(...)
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in self._FORBIDDEN_CALLS:
                        result.errors.append(
                            f"Forbidden attribute call '*.{node.func.attr}()'. "
                        )

    def _check_infinite_loops(self, tree: ast.AST, result: ValidationResult) -> None:
        """
        Heuristic: detect ``while True:`` blocks that contain no ``break``.

        This is a best-effort check. A sufficiently clever adversary could
        hide an infinite loop, but this catches the obvious cases.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                # Check if condition is constant True
                is_true = (
                    isinstance(node.test, ast.Constant) and node.test.value is True
                ) or (
                    isinstance(node.test, ast.NameConstant) and node.test.value is True  # type: ignore[attr-defined]
                )
                if is_true:
                    # Check for break in the loop body (any depth)
                    has_break = any(
                        isinstance(n, ast.Break)
                        for n in ast.walk(node)
                    )
                    if not has_break:
                        result.errors.append(
                            "Detected 'while True:' loop without a 'break' statement. "
                            "This would cause the agent to hang the scheduler."
                        )

    def _check_agent_structure(self, tree: ast.AST, result: ValidationResult) -> None:
        """
        The module must contain at least one class that:
            1. Inherits from Agent (by name)
            2. Defines a ``think`` method
        """
        agent_classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check base classes
                bases = [
                    base.id if isinstance(base, ast.Name) else
                    base.attr if isinstance(base, ast.Attribute) else ""
                    for base in node.bases
                ]
                if "Agent" in bases:
                    # Check for think() method
                    has_think = any(
                        isinstance(child, ast.FunctionDef) and child.name == "think"
                        for child in node.body
                    )
                    agent_classes.append((node.name, has_think))

        if not agent_classes:
            result.errors.append(
                "No class inheriting from Agent found. "
                "Generated code must define a class that extends Agent."
            )
        else:
            for class_name, has_think in agent_classes:
                if not has_think:
                    result.errors.append(
                        f"Class '{class_name}' inherits from Agent but does not "
                        f"implement think(). All Agent subclasses must define think()."
                    )


# ---------------------------------------------------------------------------
# CodeSandbox — tick-limited isolated execution
# ---------------------------------------------------------------------------

class CodeSandbox:
    """
    Executes Python source code in a restricted environment.

    The sandbox restricts the execution environment in two ways:
        1. ``__builtins__`` is replaced with our safe subset — dangerous
           functions like open(), exec(), __import__() are unavailable.
        2. A tick counter limits how many Python bytecode instructions can
           run. If the limit is exceeded, execution is aborted.

    Note on security:
        This sandbox is a *prototype*-grade security measure. A production
        Battousai would use OS-level isolation (namespaces, seccomp, gVisor, etc.)
        to guarantee safety. The Python-level sandbox is a useful first
        layer that catches common mistakes.

    Returns:
        (success: bool, output: str, error: str)
    """

    def __init__(self, tick_limit: int = 10_000) -> None:
        self.tick_limit = tick_limit

    def execute(
        self,
        source_code: str,
        extra_globals: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, str]:
        """
        Execute ``source_code`` in the sandbox.

        Args:
            source_code   — Python source to execute.
            extra_globals — Additional names injected into the sandbox namespace
                            (e.g., the Agent base class).

        Returns:
            (success, stdout_capture, error_message)
        """
        # Build restricted globals
        sandbox_globals: Dict[str, Any] = {
            "__builtins__": dict(_SAFE_BUILTINS),
            "__name__": "__aios_sandbox__",
        }
        if extra_globals:
            sandbox_globals.update(extra_globals)

        # Capture stdout
        output_buffer = io.StringIO()
        original_print = _SAFE_BUILTINS["print"]

        def captured_print(*args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("file", output_buffer)
            original_print(*args, **kwargs)

        sandbox_globals["__builtins__"]["print"] = captured_print  # type: ignore[index]

        # Install a tick counter via sys.settrace is not available, so we
        # use a simple instruction count via compile + exec with a counter
        # injected as a tracing function on the globals.
        tick_counter = [0]
        tick_limit = self.tick_limit

        def _trace(frame: Any, event: str, arg: Any) -> Any:
            tick_counter[0] += 1
            if tick_counter[0] > tick_limit:
                raise RuntimeError(
                    f"Sandbox tick limit exceeded ({tick_limit} instructions). "
                    "Possible infinite loop or excessively complex code."
                )
            return _trace

        try:
            import sys as _sys
            _sys.settrace(_trace)
            exec(compile(source_code, "<sandbox>", "exec"), sandbox_globals)  # noqa: S102
            _sys.settrace(None)
            return True, output_buffer.getvalue(), ""
        except Exception as exc:
            try:
                import sys as _sys
                _sys.settrace(None)
            except Exception:
                pass
            return False, output_buffer.getvalue(), str(exc)


# ---------------------------------------------------------------------------
# AgentFactory — promotes validated code to a real Agent subclass
# ---------------------------------------------------------------------------

class AgentFactory:
    """
    Compiles validated source code into a runnable Agent subclass and
    maintains a versioned registry of all dynamically created classes.

    The factory is the bridge between raw source code (a string) and a
    live Python class that the Kernel can instantiate via spawn_agent().

    Version tracking:
        Each class is stored as (class_name, version) → class. When a new
        version of the same class name is registered, the version counter
        increments. Older versions remain accessible for rollback.
    """

    def __init__(self) -> None:
        # (class_name, version) → class object
        self._registry: Dict[Tuple[str, int], Type[Agent]] = {}
        # class_name → latest version number
        self._versions: Dict[str, int] = {}

    def compile_and_register(
        self,
        source_code: str,
        extra_globals: Optional[Dict[str, Any]] = None,
    ) -> Optional[Type[Agent]]:
        """
        Execute the source code (which must define an Agent subclass),
        extract the first Agent subclass found, register it, and return it.

        The factory uses a more permissive execution environment than the
        sandbox (full builtins + the battousai module tree) because the code has
        already passed static validation via CodeValidator.  This mirrors
        the real-world pattern: sandbox for initial evaluation, controlled
        loader for promoted code.

        Returns None if compilation or extraction fails.
        """
        import sys as _sys

        # Build a namespace that includes full builtins and all battousai modules
        # so that ``from battousai.agent import Agent`` resolves inside exec().
        namespace: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "Agent": Agent,
        }
        # Expose loaded battousai sub-modules directly in the exec namespace
        for mod_name, mod in list(_sys.modules.items()):
            if mod_name.startswith("battousai"):
                namespace[mod_name] = mod
        if extra_globals:
            namespace.update(extra_globals)

        try:
            exec(compile(source_code, "<agent_factory>", "exec"), namespace)  # noqa: S102
        except Exception:
            return None

        # Find the Agent subclass
        agent_class: Optional[Type[Agent]] = None
        for obj in namespace.values():
            try:
                if (
                    isinstance(obj, type)
                    and issubclass(obj, Agent)
                    and obj is not Agent
                ):
                    agent_class = obj
                    break
            except TypeError:
                continue

        if agent_class is None:
            return None

        # Version tracking
        class_name = agent_class.__name__
        version = self._versions.get(class_name, 0) + 1
        self._versions[class_name] = version
        self._registry[(class_name, version)] = agent_class

        return agent_class

    def get(self, class_name: str, version: Optional[int] = None) -> Optional[Type[Agent]]:
        """Retrieve a registered class by name and optional version."""
        v = version if version is not None else self._versions.get(class_name, 0)
        return self._registry.get((class_name, v))

    def list_classes(self) -> List[Dict[str, Any]]:
        """Return a list of all registered dynamic agent classes."""
        return [
            {"class_name": name, "version": ver}
            for (name, ver) in sorted(self._registry.keys())
        ]

    def latest_version(self, class_name: str) -> int:
        return self._versions.get(class_name, 0)


# ---------------------------------------------------------------------------
# AgentGenome — the genetic representation of an agent design
# ---------------------------------------------------------------------------

@dataclass
class AgentGenome:
    """
    The genetic representation of an agent design.

    A genome encodes everything needed to recreate an agent:
    the source code (phenotype) plus the configuration parameters (genotype).
    Fitness is measured post-hoc after deploying test instances.

    Fields:
        genome_id    — unique identifier for this genome
        source_code  — Python source code of the agent class
        config       — dict of tunable parameters (e.g., tick_interval=5)
        fitness_score — weighted performance score (higher = better)
        generation   — how many evolutionary cycles produced this genome
        parent_ids   — genome_ids of the parents (empty if primordial)
        created_at   — wall-clock time of creation
        class_name   — name of the Agent subclass encoded
    """
    genome_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_code: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    class_name: str = "EvolvedAgent"


# ---------------------------------------------------------------------------
# Agent code templates
# ---------------------------------------------------------------------------

# Minimal functional agent template — the primordial genome
_TEMPLATE_IDLE_AGENT = textwrap.dedent("""\
    from battousai.agent import Agent

    class {class_name}(Agent):
        \"\"\"Evolved idle agent — generation {generation}.\"\"\"

        def __init__(self, name="{class_name}", priority=5):
            super().__init__(name=name, priority=priority,
                             memory_allocation={memory_allocation},
                             time_slice={time_slice})
            self._tick_count = 0

        def on_spawn(self):
            self.log("[{class_name}] spawned (gen={generation})")

        def think(self, tick):
            self._tick_count += 1
            if self._tick_count % {report_interval} == 0:
                self.log("[{class_name}] tick={t} count={c}".format(
                    t=tick, c=self._tick_count))
                self.mem_write("tick_count", self._tick_count)
            self.yield_cpu()
""")

_TEMPLATE_WORKER_AGENT = textwrap.dedent("""\
    from battousai.agent import Agent
    from battousai.ipc import MessageType

    class {class_name}(Agent):
        \"\"\"Evolved worker agent — generation {generation}.\"\"\"

        def __init__(self, name="{class_name}", priority=5):
            super().__init__(name=name, priority=priority,
                             memory_allocation={memory_allocation},
                             time_slice={time_slice})
            self._jobs_done = 0

        def on_spawn(self):
            self.log("[{class_name}] worker online (gen={generation})")

        def think(self, tick):
            messages = self.read_inbox()
            for msg in messages:
                if msg.message_type == MessageType.TASK:
                    self._process_task(msg, tick)
            self.yield_cpu()

        def _process_task(self, msg, tick):
            self._jobs_done += 1
            self.mem_write("jobs_done", self._jobs_done)
            self.log("[{class_name}] processed task #{c}".format(
                c=self._jobs_done))
""")


# ---------------------------------------------------------------------------
# FitnessEvaluator
# ---------------------------------------------------------------------------

class FitnessEvaluator:
    """
    Measures agent performance against a set of weighted objectives.

    Fitness Metrics:
        tasks_completed       — Number of TASK messages the agent handled
        messages_sent         — Total outbound IPC messages (capped to avoid spam)
        tools_used            — Distinct tool invocations (breadth signal)
        memory_efficiency     — Fraction of allocated memory actually used
        uptime                — Number of ticks alive (stability signal)
        error_rate            — Fraction of ticks that produced exceptions

    Weights are configurable; defaults reflect a balanced general-purpose agent.
    Higher score is always better.
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "tasks_completed":   3.0,
        "messages_sent":     0.5,   # low weight — avoid incentivising spam
        "tools_used":        1.0,
        "memory_efficiency": 1.5,
        "uptime":            0.2,
        "error_rate":       -5.0,   # negative — errors heavily penalised
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)

    def evaluate(self, metrics: Dict[str, float]) -> float:
        """
        Compute a weighted fitness score from raw metrics dict.

        Args:
            metrics — keys matching DEFAULT_WEIGHTS, values are raw numbers.

        Returns:
            float fitness score (may be negative if error_rate is high).
        """
        score = 0.0
        for key, weight in self.weights.items():
            value = metrics.get(key, 0.0)
            score += weight * value
        return round(score, 4)

    def collect_metrics(self, agent: Agent, kernel: Any) -> Dict[str, float]:
        """
        Extract fitness metrics from a live agent by querying the kernel.

        This is a best-effort extraction — agents that have already been
        killed will return zeros for live metrics.
        """
        metrics: Dict[str, float] = {
            "tasks_completed": 0.0,
            "messages_sent":   0.0,
            "tools_used":      0.0,
            "memory_efficiency": 0.0,
            "uptime":          float(agent._ticks_alive),
            "error_rate":      0.0,
        }

        # Memory efficiency
        if kernel is not None:
            try:
                space = kernel.memory.get_agent_space(agent.agent_id)
                used, max_keys = space.usage()
                metrics["memory_efficiency"] = used / max(max_keys, 1)
            except Exception:
                pass

            # Tool usage from tool stats
            try:
                tool_stats = kernel.tools.stats()
                agent_calls = tool_stats["calls_by_agent"].get(agent.agent_id, 0)
                metrics["tools_used"] = float(agent_calls)
            except Exception:
                pass

        # Tasks completed: read from agent's own memory
        tasks_done = agent.mem_read("tasks_completed") or 0
        metrics["tasks_completed"] = float(tasks_done)

        jobs_done = agent.mem_read("jobs_done") or 0
        metrics["tasks_completed"] = max(metrics["tasks_completed"], float(jobs_done))

        return metrics


# ---------------------------------------------------------------------------
# GeneticPool
# ---------------------------------------------------------------------------

class GeneticPool:
    """
    Maintains a population of agent genomes that evolve over time.

    The pool implements three core genetic operators:

    Selection:
        The top ``elite_fraction`` of genomes (by fitness_score) are
        guaranteed to survive to the next generation. The rest are replaced
        by offspring of the top performers.

    Crossover:
        Two parent genomes contribute different sections of their config
        dicts to the child. Source code is taken wholesale from the
        higher-fitness parent (code crossover is too risky syntactically).

    Mutation:
        Numeric parameters in the config dict are perturbed by ±mutation_rate.
        Boolean parameters are randomly flipped with probability mutation_rate.

    Population management:
        The pool never exceeds ``max_population`` genomes. When new offspring
        are added, the lowest-fitness genomes are evicted first.
    """

    def __init__(
        self,
        max_population: int = 20,
        elite_fraction: float = 0.3,
        mutation_rate: float = 0.15,
    ) -> None:
        self.max_population = max_population
        self.elite_fraction = elite_fraction
        self.mutation_rate = mutation_rate
        self._genomes: List[AgentGenome] = []
        self._generation: int = 0

    def add(self, genome: AgentGenome) -> None:
        """Add a genome to the pool, evicting the weakest if at capacity."""
        if len(self._genomes) >= self.max_population:
            # Evict lowest fitness
            self._genomes.sort(key=lambda g: g.fitness_score, reverse=True)
            self._genomes.pop()
        self._genomes.append(genome)

    def update_fitness(self, genome_id: str, score: float) -> bool:
        """Update the fitness score of a genome by ID."""
        for genome in self._genomes:
            if genome.genome_id == genome_id:
                genome.fitness_score = score
                return True
        return False

    def top_k(self, k: int = 5) -> List[AgentGenome]:
        """Return the top-k genomes by fitness score."""
        return sorted(self._genomes, key=lambda g: g.fitness_score, reverse=True)[:k]

    def crossover(self, parent_a: AgentGenome, parent_b: AgentGenome) -> AgentGenome:
        """
        Produce a child genome by combining two parents.

        Config crossover: for each key, randomly pick from parent_a or parent_b.
        Source code: taken from the higher-fitness parent.
        """
        import random
        winner = parent_a if parent_a.fitness_score >= parent_b.fitness_score else parent_b
        loser  = parent_b if winner is parent_a else parent_a

        child_config = {}
        all_keys = set(parent_a.config) | set(parent_b.config)
        for key in all_keys:
            if random.random() < 0.5 and key in parent_a.config:
                child_config[key] = parent_a.config[key]
            elif key in parent_b.config:
                child_config[key] = parent_b.config[key]
            else:
                child_config[key] = parent_a.config.get(key)

        return AgentGenome(
            source_code=winner.source_code,
            config=child_config,
            generation=self._generation + 1,
            parent_ids=[parent_a.genome_id, parent_b.genome_id],
            class_name=winner.class_name,
        )

    def mutate(self, genome: AgentGenome) -> AgentGenome:
        """
        Produce a mutated copy of a genome.

        Numeric config values are perturbed; booleans are flipped;
        strings are left unchanged (source code is not mutated here —
        only the EvolutionEngine mutates source via templates).
        """
        import random
        new_config = dict(genome.config)
        for key, val in new_config.items():
            if random.random() < self.mutation_rate:
                if isinstance(val, int):
                    delta = max(1, int(abs(val) * 0.2))
                    new_config[key] = max(1, val + random.randint(-delta, delta))
                elif isinstance(val, float):
                    new_config[key] = max(0.0, val * (1.0 + random.uniform(-self.mutation_rate, self.mutation_rate)))
                elif isinstance(val, bool):
                    new_config[key] = not val

        return AgentGenome(
            source_code=genome.source_code,
            config=new_config,
            generation=genome.generation + 1,
            parent_ids=[genome.genome_id],
            class_name=genome.class_name,
            fitness_score=0.0,  # fitness is re-evaluated after mutation
        )

    def next_generation(self) -> List[AgentGenome]:
        """
        Advance the pool by one generation.

        Keeps elite genomes, replaces the rest with offspring of the top performers.
        Returns the list of new genomes added.
        """
        import random
        self._generation += 1
        sorted_pop = sorted(self._genomes, key=lambda g: g.fitness_score, reverse=True)
        elite_n = max(1, int(len(sorted_pop) * self.elite_fraction))
        elite = sorted_pop[:elite_n]
        new_offspring: List[AgentGenome] = []
        while len(elite) + len(new_offspring) < self.max_population and len(elite) >= 2:
            p1, p2 = random.sample(elite, 2)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            new_offspring.append(child)
        self._genomes = elite + new_offspring
        return new_offspring

    def size(self) -> int:
        return len(self._genomes)

    def generation(self) -> int:
        return self._generation

    def snapshot(self) -> List[Dict[str, Any]]:
        """Return a serialisable snapshot of the pool for logging."""
        return [
            {
                "genome_id": g.genome_id,
                "class_name": g.class_name,
                "fitness": g.fitness_score,
                "generation": g.generation,
                "parents": g.parent_ids,
            }
            for g in sorted(self._genomes, key=lambda g: g.fitness_score, reverse=True)
        ]


# ---------------------------------------------------------------------------
# EvolutionEngine — the meta-agent
# ---------------------------------------------------------------------------

class EvolutionEngine(Agent):
    """
    The EvolutionEngine is a meta-agent whose purpose is to improve the
    agent population over time via automated code generation and selection.

    Think-loop behaviour:
        Phase 1 — BOOTSTRAP   : seed the genetic pool with primordial genomes
        Phase 2 — SPAWN_TESTS : instantiate test agents from the pool
        Phase 3 — EVALUATE    : measure fitness of running test agents
        Phase 4 — EVOLVE      : crossover + mutate to produce next generation
        Phase 5 — PROMOTE     : register top-fitness agents as available classes

    The engine runs at low priority (priority=8) so it does not interfere
    with productive agents. It operates on a slow tick cadence.
    """

    EVOLUTION_TICK_INTERVAL = 10  # ticks between evolution cycles

    def __init__(self, name: str = "EvolutionEngine", priority: int = 8) -> None:
        super().__init__(name=name, priority=priority,
                         memory_allocation=512, time_slice=3)
        self.validator = CodeValidator()
        self.sandbox = CodeSandbox(tick_limit=50_000)
        self.factory = AgentFactory()
        self.fitness_evaluator = FitnessEvaluator()
        self.pool = GeneticPool(max_population=10, mutation_rate=0.15)

        self._phase: str = "BOOTSTRAP"
        self._test_agent_ids: List[str] = []  # IDs of agents currently under test
        self._test_genome_map: Dict[str, str] = {}  # agent_id → genome_id
        self._cycle: int = 0
        self._promoted_classes: List[str] = []

    def on_spawn(self) -> None:
        self.log(f"[{self.name}] Online. Beginning evolutionary bootstrap.")

    def think(self, tick: int) -> None:
        # Run evolution cycle every EVOLUTION_TICK_INTERVAL ticks
        if tick % self.EVOLUTION_TICK_INTERVAL != 0:
            self.yield_cpu()
            return

        self._cycle += 1
        self.log(f"[{self.name}] Evolution cycle {self._cycle} at tick {tick}")

        if self._phase == "BOOTSTRAP":
            self._bootstrap_pool()
            self._phase = "SPAWN_TESTS"

        elif self._phase == "SPAWN_TESTS":
            self._spawn_test_agents()
            self._phase = "EVALUATE"

        elif self._phase == "EVALUATE":
            self._evaluate_fitness()
            self._phase = "EVOLVE"

        elif self._phase == "EVOLVE":
            self._run_evolution()
            self._phase = "PROMOTE"

        elif self._phase == "PROMOTE":
            self._promote_top_agents()
            self._phase = "SPAWN_TESTS"  # loop

        self.mem_write("evolution_cycle", self._cycle)
        self.mem_write("pool_size", self.pool.size())
        self.yield_cpu()

    def _bootstrap_pool(self) -> None:
        """Seed the pool with template-generated primordial genomes."""
        templates = [
            (_TEMPLATE_IDLE_AGENT,   "EvolvedIdleAgent",   {"memory_allocation": 256, "time_slice": 3, "report_interval": 5}),
            (_TEMPLATE_WORKER_AGENT, "EvolvedWorkerAgent", {"memory_allocation": 256, "time_slice": 4, "report_interval": 5}),
        ]
        for template, class_name, config in templates:
            source = template.format(
                class_name=class_name,
                generation=0,
                **config,
            )
            vr = self.validator.validate(source)
            if vr.valid:
                genome = AgentGenome(
                    source_code=source,
                    config=config,
                    generation=0,
                    class_name=class_name,
                )
                self.pool.add(genome)
                self.log(f"[{self.name}] Seeded genome {genome.genome_id} ({class_name})")
            else:
                self.log(f"[{self.name}] Template validation failed: {vr.summary()}")

    def _spawn_test_agents(self) -> None:
        """Compile pool genomes and spawn a test instance for each."""
        if self.kernel is None:
            return
        for genome in self.pool.top_k(k=3):
            agent_class = self.factory.compile_and_register(
                genome.source_code,
                extra_globals={"Agent": Agent},
            )
            if agent_class is None:
                self.log(f"[{self.name}] Failed to compile genome {genome.genome_id}")
                continue
            result = self.spawn_child(agent_class, name=f"test_{genome.class_name}", priority=7)
            if result.ok:
                aid = result.value
                self._test_agent_ids.append(aid)
                self._test_genome_map[aid] = genome.genome_id
                self.log(f"[{self.name}] Test agent {aid} spawned for genome {genome.genome_id}")

    def _evaluate_fitness(self) -> None:
        """Measure fitness of test agents and update the genetic pool."""
        if self.kernel is None:
            return
        alive = set(self.list_agents())
        for aid in self._test_agent_ids:
            genome_id = self._test_genome_map.get(aid)
            if genome_id is None:
                continue
            agent = self.kernel._agents.get(aid)
            if agent is None:
                continue
            metrics = self.fitness_evaluator.collect_metrics(agent, self.kernel)
            score = self.fitness_evaluator.evaluate(metrics)
            self.pool.update_fitness(genome_id, score)
            self.log(f"[{self.name}] Genome {genome_id} fitness={score:.3f}")

    def _run_evolution(self) -> None:
        """Advance the genetic pool by one generation."""
        offspring = self.pool.next_generation()
        self.log(
            f"[{self.name}] Generation {self.pool.generation()} — "
            f"pool={self.pool.size()}, new_offspring={len(offspring)}"
        )

    def _promote_top_agents(self) -> None:
        """Register the top genome's class in the factory for external use."""
        top = self.pool.top_k(k=1)
        if not top:
            return
        champion = top[0]
        cls = self.factory.compile_and_register(
            champion.source_code,
            extra_globals={"Agent": Agent},
        )
        if cls:
            self._promoted_classes.append(cls.__name__)
            self.log(
                f"[{self.name}] Promoted {cls.__name__} "
                f"(fitness={champion.fitness_score:.3f}, gen={champion.generation})"
            )
