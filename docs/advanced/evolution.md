# Self-Modification and Evolution

The `evolution.py` module is the self-improvement engine of Battousai. Agents can write new agent code, validate it in a sandbox, and deploy it at runtime. This enables genetic-algorithm-style evolution of agent behaviour.

---

## Safety Architecture (Defence in Depth)

```
Generated Code
      │
      ▼
┌─────────────────┐
│  CodeValidator  │  Static AST analysis: blocks dangerous imports,
│                 │  exec(), eval(), __import__, os/sys access
└────────┬────────┘
         │ passes
         ▼
┌─────────────────┐
│  CodeSandbox    │  Restricted exec() with stripped builtins:
│                 │  no file I/O, no network, no subprocess
└────────┬────────┘
         │ executes safely
         ▼
┌─────────────────┐
│  AgentFactory   │  Promotes validated code to a real Agent subclass;
│                 │  registers with the kernel
└────────┬────────┘
         │ produces agent class
         ▼
┌─────────────────┐
│ FitnessEvaluator│  Measures performance (messages sent, tools used,
│                 │  tasks completed); only fit agents produce offspring
└─────────────────┘
```

---

## `ValidationResult`

```python
@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]         # Error messages if invalid
    warnings: List[str]       # Non-fatal warnings
    ast_node_count: int       # Complexity metric
    dangerous_patterns: List[str]  # Blocked patterns found
```

---

## `CodeValidator`

Static AST analysis of generated agent code:

```python
from battousai.evolution import CodeValidator

validator = CodeValidator()

result = validator.validate("""
class MyAgent(Agent):
    def think(self, tick):
        self.log("hello")
        self.yield_cpu()
""")

print(result.valid)    # True
print(result.errors)   # []

# Dangerous code
result = validator.validate("""
import os
class EvilAgent(Agent):
    def think(self, tick):
        os.system("rm -rf /")
""")
print(result.valid)              # False
print(result.dangerous_patterns) # ["import os"]
```

**Blocked patterns:**
- `import` statements (any module)
- `exec()`, `eval()`
- `__import__`, `__builtins__`
- `open()`, `os.*`, `sys.*`
- `subprocess.*`
- `socket.*`
- Network access patterns

---

## `CodeSandbox`

Executes validated code in a restricted namespace:

```python
from battousai.evolution import CodeSandbox

sandbox = CodeSandbox()

# Execute code and extract the defined class
agent_code = """
class EvolvedAgent(Agent):
    def think(self, tick):
        self.log(f"Evolved agent tick {tick}")
        result = self.use_tool("calculator", expression="tick * 2")
        self.yield_cpu()
"""

result = sandbox.execute(agent_code, agent_base_class=Agent)
if result.success:
    agent_class = result.agent_class   # The extracted class
    print(agent_class.__name__)         # "EvolvedAgent"
else:
    print(f"Sandbox error: {result.error}")
```

Safe builtins available inside sandbox: `print`, `len`, `range`, `enumerate`, `zip`, `map`, `filter`, `sorted`, `min`, `max`, `sum`, `abs`, `round`, `str`, `int`, `float`, `bool`, `list`, `dict`, `tuple`, `set`, `type`, `isinstance`, `hasattr`, `getattr`, `repr` — plus exception types.

---

## `AgentFactory`

Promotes validated, sandboxed code to a registered agent class:

```python
from battousai.evolution import AgentFactory, CodeValidator, CodeSandbox

factory = AgentFactory(
    validator=CodeValidator(),
    sandbox=CodeSandbox(),
)

agent_code = """
class OptimizedWorker(Agent):
    def think(self, tick):
        r = self.use_tool("web_search", query="latest AI research")
        if r.ok:
            self.mem_write("latest_result", r.value)
        self.yield_cpu()
"""

result = factory.create(agent_code)
if result.success:
    OptimizedWorkerClass = result.agent_class

    # Spawn into the kernel
    agent_id = kernel.spawn_agent(OptimizedWorkerClass, name="OptimizedWorker", priority=4)
```

---

## `GeneticPool`

Maintains a population of agent code templates that can evolve across generations:

```python
from battousai.evolution import GeneticPool

pool = GeneticPool(
    max_population=20,    # maximum number of designs in the pool
    mutation_rate=0.1,    # 10% chance of mutating each gene
)

# Seed with a base template
pool.seed(base_code="""
class SeedAgent(Agent):
    def think(self, tick):
        self.use_tool("calculator", expression="tick * 2")
        self.yield_cpu()
""")

# Add a scored individual
pool.add(code="...", fitness_score=0.85, generation=1)

# Get the top N individuals for breeding
elite = pool.top_k(k=5)

# Mutate a design (introduces small random changes)
mutant_code = pool.mutate(elite[0].code)

# Crossover two designs (combines elements from both)
offspring = pool.crossover(elite[0].code, elite[1].code)

# Select next generation
pool.select(keep_top_k=5, generation=2)
```

---

## `FitnessEvaluator`

Scores agent performance against objectives:

```python
from battousai.evolution import FitnessEvaluator

evaluator = FitnessEvaluator(
    objectives={
        "tool_calls": 1.0,        # weight: reward tool usage
        "messages_sent": 0.5,     # weight: reward communication
        "files_written": 0.3,     # weight: reward output production
        "ticks_survived": 0.2,    # weight: reward longevity
        "errors": -2.0,           # weight: penalize errors
    }
)

# Score an agent after N evaluation ticks
metrics = {
    "tool_calls": 8,
    "messages_sent": 3,
    "files_written": 2,
    "ticks_survived": 20,
    "errors": 1,
}
score = evaluator.score(metrics)
# 8*1.0 + 3*0.5 + 2*0.3 + 20*0.2 - 1*2.0 = 8 + 1.5 + 0.6 + 4 - 2 = 12.1
print(f"Fitness: {score}")
```

---

## `EvolutionEngine`

The meta-agent that orchestrates the full evolution loop:

```python
from battousai.evolution import EvolutionEngine, GeneticPool

kernel.boot()

pool = GeneticPool(max_population=20, mutation_rate=0.05)
pool.seed(base_code="""
class BaseAgent(Agent):
    def think(self, tick):
        self.use_tool("calculator", expression=str(tick))
        self.yield_cpu()
""")

engine = EvolutionEngine(
    kernel=kernel,
    pool=pool,
    generations=5,         # number of evolutionary generations
    population_size=10,    # agents in each generation
    eval_ticks=20,         # ticks each candidate runs before scoring
    elite_size=3,          # top performers kept each generation
)

# Run evolution
best_class = engine.evolve()

# Deploy the best agent
agent_id = kernel.spawn_agent(best_class, name="BestEvolved", priority=4)
kernel.run()
```

### Evolution Cycle

Each generation:

```
1. GENERATE  — mutate/crossover top agents from GeneticPool
2. VALIDATE  — run through CodeValidator + CodeSandbox
3. DEPLOY    — instantiate via AgentFactory, spawn for eval_ticks
4. EVALUATE  — score with FitnessEvaluator
5. SELECT    — top performers added back to GeneticPool
6. REPEAT    — next generation begins
```

---

## Full Example: Evolving a Research Agent

```python
from battousai.kernel import Kernel
from battousai.evolution import EvolutionEngine, GeneticPool, CodeValidator, CodeSandbox, AgentFactory

kernel = Kernel(max_ticks=200)
kernel.boot()

# Define the base template
base_code = """
class ResearchAgent(Agent):
    def __init__(self):
        super().__init__(name="ResearchAgent", priority=4)
        self._queries = ["quantum computing", "machine learning", "robotics"]
        self._done = []

    def think(self, tick):
        if self._queries:
            query = self._queries.pop(0)
            r = self.use_tool("web_search", query=query)
            if r.ok:
                self.mem_write(f"result_{tick}", r.value)
                self._done.append(query)
        self.yield_cpu()
"""

pool = GeneticPool(max_population=15, mutation_rate=0.08)
pool.seed(base_code=base_code)

engine = EvolutionEngine(
    kernel=kernel,
    pool=pool,
    generations=3,
    population_size=8,
    eval_ticks=15,
    elite_size=3,
)

best_class = engine.evolve()
print(f"Best evolved agent: {best_class.__name__}")

# Deploy
kernel.spawn_agent(best_class, name="EvolvedResearcher", priority=4)
kernel.run(ticks=50)
print(kernel.system_report())
```

---

## Safety Guarantees

| Layer | What it prevents |
|---|---|
| `CodeValidator` | Dangerous imports, file/network access, `exec`/`eval` |
| `CodeSandbox` | Access to real builtins; only whitelisted functions available |
| `AgentFactory` | Non-`Agent` subclasses cannot be promoted |
| `FitnessEvaluator` | Only high-performing agents survive to produce offspring |

!!! warning "Sandbox limitations"
    The sandbox prevents the most common attack vectors but is not a hardware-level sandbox. For production deployments, run Battousai in a container or VM for additional isolation.

---

## Related Pages

- [Architecture Overview](../architecture/overview.md) — evolution layer in context
- [Custom Agents](../agents/custom.md) — building manually the agents evolution tries to discover automatically
- [Contracts](../security/contracts.md) — adding behavioral contracts to evolved agents
