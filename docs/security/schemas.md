# Typed Memory Schemas

The `schemas.py` module provides a type system for Battousai agent memory. Agents declare what keys they read and write, along with types and constraints. The kernel validates writes at runtime.

---

## Why Typed Memory?

In a multi-agent system, memory is a shared contract. Without types, one agent's `findings` key might be a string while another expects a list. Schemas make these contracts explicit and mechanically checked.

Schemas enable:
- **Runtime validation** — writes are rejected if the value fails type constraints
- **Cross-agent inspection** — agents can query what another agent produces
- **Data catalog** — `SchemaInspector.all_declared_keys()` provides a global view
- **Dynamic composition** — find agents that produce compatible data types

---

## `FieldType` Enum

```python
class FieldType(Enum):
    STRING   = auto()   # str
    INT      = auto()   # int (booleans excluded)
    FLOAT    = auto()   # float or int
    BOOL     = auto()   # bool
    LIST     = auto()   # list
    DICT     = auto()   # dict
    ANY      = auto()   # no type constraint
    OPTIONAL = auto()   # modifier; use nullable=True on FieldSpec instead
```

---

## `FieldSpec` Dataclass

Describes a single typed field in a schema:

```python
@dataclass
class FieldSpec:
    name: str
    field_type: FieldType
    required: bool = True
    nullable: bool = False        # if True, None is an acceptable value
    default: Any = None
    description: str = ""
    min_value: Optional[float] = None    # for INT/FLOAT
    max_value: Optional[float] = None    # for INT/FLOAT
    pattern: Optional[str] = None        # for STRING — regex fullmatch
    element_type: Optional[FieldType] = None  # for LIST — element type
    validator: Optional[Callable[[Any], bool]] = None  # custom validation
```

---

## `@schema` Class Decorator

Attach a `MemorySchema` to an `Agent` subclass and auto-register with the global schema registry:

```python
from battousai.schemas import schema, FieldSpec, FieldType
from battousai.agent import Agent

@schema(
    name="ResearchWorker",
    version="1.0",
    fields=[
        FieldSpec(
            "findings",
            FieldType.LIST,
            required=True,
            description="Research results collected so far",
        ),
        FieldSpec(
            "confidence",
            FieldType.FLOAT,
            required=True,
            description="Confidence score 0.0–1.0",
            min_value=0.0,
            max_value=1.0,
        ),
        FieldSpec(
            "sources",
            FieldType.LIST,
            required=False,
            description="Source URLs",
            nullable=True,
        ),
        FieldSpec(
            "status",
            FieldType.STRING,
            required=True,
            description="Current status",
            pattern=r"^(idle|working|done|error)$",
        ),
    ],
    reads=["task_assignment", "global.config"],
    writes=["findings", "confidence", "sources", "status"],
)
class ResearchWorkerAgent(Agent):
    def think(self, tick: int) -> None:
        # These writes are validated against the schema
        self.mem_write("findings", ["result 1", "result 2"])
        self.mem_write("confidence", 0.87)
        self.mem_write("status", "done")
        self.yield_cpu()
```

### Decorator Parameters

| Parameter | Description |
|---|---|
| `name` | Schema name (usually the class name) |
| `version` | Semver string (e.g. `"1.0"`, `"2.3.1"`) |
| `fields` | List of `FieldSpec` objects |
| `reads` | Memory keys the agent declares it reads |
| `writes` | Memory keys the agent declares it writes |
| `registry` | Override the registry (defaults to `GLOBAL_REGISTRY`) |

The schema is stored as `cls._memory_schema` and registered in the global registry under `"class:{ClassName}"`.

---

## `MemorySchema` Class

```python
class MemorySchema:
    def __init__(
        self,
        name: str,
        version: str = "1.0",
        fields: Optional[List[FieldSpec]] = None,
        read_keys: Optional[List[str]] = None,
        write_keys: Optional[List[str]] = None,
    ) -> None

    def get_field(self, name: str) -> Optional[FieldSpec]
    def validate(self, key: str, value: Any) -> bool   # raises SchemaValidationError on failure
    def required_fields(self) -> List[FieldSpec]
    def describe(self) -> str
```

---

## `SchemaValidator`

Validates a value against a `FieldSpec`:

```python
from battousai.schemas import SchemaValidator, FieldSpec, FieldType, SchemaValidationError

spec = FieldSpec(
    "confidence",
    FieldType.FLOAT,
    min_value=0.0,
    max_value=1.0,
    description="Confidence score",
)

# Raising validation
try:
    SchemaValidator.validate(spec, 1.5)   # exceeds max_value
except SchemaValidationError as e:
    print(e)  # "Field 'confidence': value 1.5 exceeds maximum 1.0."

# Non-raising validation
ok, error = SchemaValidator.safe_validate(spec, 0.75)
# ok=True, error=None

ok, error = SchemaValidator.safe_validate(spec, "not a number")
# ok=False, error="Field 'confidence': expected FLOAT..."
```

### Validated constraints

| Constraint | Applicable Types |
|---|---|
| Type check (`isinstance`) | STRING, INT, FLOAT, BOOL, LIST, DICT |
| `min_value` / `max_value` | INT, FLOAT |
| `pattern` (regex fullmatch) | STRING |
| `element_type` | LIST |
| `nullable` | All (None allowed if True) |
| Custom `validator` callable | All |

!!! note "INT vs BOOL"
    Since Python's `bool` is a subclass of `int`, writing a `bool` to an `INT` field raises `SchemaValidationError`. Use `FieldType.BOOL` for boolean fields.

---

## `SchemaRegistry`

Central registry mapping agent IDs to schemas:

```python
from battousai.schemas import SchemaRegistry, MemorySchema, FieldSpec, FieldType

registry = SchemaRegistry()

# Register manually
schema = MemorySchema(
    name="MyAgent",
    fields=[FieldSpec("count", FieldType.INT)],
    read_keys=[],
    write_keys=["count"],
)
registry.register("myagent_0001", schema)

# Retrieve
s = registry.get_schema("myagent_0001")
print(s.name)  # "MyAgent"

# Find all agents that write a specific key
writers = registry.find_agents_writing_key("findings")
# ["researchworker_0002", "researchworker_0003"]

# Find all agents that read a specific key
readers = registry.find_agents_reading_key("task_assignment")

# Validate a write before it reaches MemoryManager
try:
    registry.validate_write("myagent_0001", "count", "not_an_int")
except SchemaValidationError as e:
    print(e)  # "Field 'count': expected INT..."

# Summary
print(registry.summary())
# SchemaRegistry:
#   'myagent_0001' → MyAgent v1.0
#   'class:ResearchWorkerAgent' → ResearchWorker v1.0
```

The **global registry** is available at `battousai.schemas.GLOBAL_REGISTRY`.

---

## `SchemaInspector`

Introspect schemas for dynamic agent composition:

```python
from battousai.schemas import SchemaInspector, GLOBAL_REGISTRY, FieldType

inspector = SchemaInspector(GLOBAL_REGISTRY)

# Describe an agent's schema
print(inspector.describe("researchworker_0002"))
# Schema: ResearchWorker v1.0
#   Reads  : task_assignment, global.config
#   Writes : findings, confidence, sources, status
#   Fields:
#     findings : LIST [required] — Research results collected so far
#     confidence : FLOAT [required] — Confidence score 0.0–1.0
#     ...

# Find all agents that provide "findings" data
providers = inspector.find_providers("findings")
# ["researchworker_0002", "researchworker_0003"]

# Find all agents that consume "findings" data
consumers = inspector.find_consumers("findings")

# Find writers that produce a compatible type
compatible = inspector.compatible_writers("findings", required_type=FieldType.LIST)

# Get all declared keys across all schemas
all_keys = inspector.all_declared_keys()
# {"findings", "confidence", "status", "count", ...}
```

---

## Nested Schema Example

For complex structured data, combine `DICT` fields with custom validators:

```python
import json

@schema(
    name="PipelineStage",
    version="1.0",
    fields=[
        FieldSpec(
            "stage_config",
            FieldType.DICT,
            required=True,
            description="Pipeline stage configuration",
            validator=lambda v: "tool" in v and "args" in v,
        ),
        FieldSpec(
            "output",
            FieldType.ANY,
            required=False,
            nullable=True,
            description="Stage output (any type)",
        ),
        FieldSpec(
            "error_count",
            FieldType.INT,
            required=True,
            min_value=0,
            description="Number of errors encountered",
        ),
    ],
    reads=["pipeline_config"],
    writes=["stage_config", "output", "error_count"],
)
class PipelineAgent(Agent):
    def think(self, tick: int) -> None:
        self.mem_write("stage_config", {"tool": "web_search", "args": {"query": "AI"}})
        self.mem_write("output", {"results": []})
        self.mem_write("error_count", 0)
        self.yield_cpu()
```

---

## Related Pages

- [Capabilities](capabilities.md) — access control for who can read/write memory
- [Contracts](contracts.md) — behavioral contracts beyond just type checking
- [Memory](../architecture/memory.md) — the underlying memory subsystem
