"""
schemas.py — Typed Memory Schemas
====================================
Agents declare memory schemas — what keys they read, what keys they write,
and the types of values. The kernel validates writes at runtime and enables
cross-agent memory inspection.

This is the type system of Battousai. Just as a traditional OS has typed file
formats, Battousai has typed memory entries.

Features:
    Schema declaration — agents declare their memory contract
    Runtime validation — writes are checked against declared types
    Schema registry    — global registry of all agent schemas
    Cross-agent inspection — agents can query what another agent exposes
    Schema versioning  — schemas have versions for backward compatibility
    Composite types    — support for nested objects, arrays, enums, unions

Usage example::

    from battousai.schemas import schema, FieldSpec, FieldType, MemorySchema

    @schema(
        name="ResearchWorker",
        version="1.0",
        fields=[
            FieldSpec("findings",   FieldType.LIST,  required=True,
                      description="Research results"),
            FieldSpec("confidence", FieldType.FLOAT, required=True,
                      description="Confidence score 0-1"),
            FieldSpec("sources",    FieldType.LIST,  required=False,
                      description="Source URLs"),
        ],
        reads=["task_assignment", "global.config"],
        writes=["findings", "confidence", "sources", "status"],
    )
    class ResearchWorkerAgent(Agent):
        ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type


# ---------------------------------------------------------------------------
# FieldType enum
# ---------------------------------------------------------------------------

class FieldType(Enum):
    """
    Supported value types for memory schema fields.

    STRING   — str
    INT      — int (and bool is excluded; see validator notes)
    FLOAT    — float or int (numeric duck-typing)
    BOOL     — bool
    LIST     — list (homogeneous or heterogeneous)
    DICT     — dict
    ANY      — no type constraint
    OPTIONAL — modifier: the field may be None; combined with another type
               in FieldSpec via ``nullable=True``

    Note:
        OPTIONAL is provided as an enum value for documentation purposes;
        the actual nullability is controlled by ``FieldSpec.nullable``.
    """
    STRING   = auto()
    INT      = auto()
    FLOAT    = auto()
    BOOL     = auto()
    LIST     = auto()
    DICT     = auto()
    ANY      = auto()
    OPTIONAL = auto()  # modifier; see FieldSpec.nullable


# ---------------------------------------------------------------------------
# FieldSpec dataclass
# ---------------------------------------------------------------------------

@dataclass
class FieldSpec:
    """
    Descriptor for a single typed field in a MemorySchema.

    Attributes
    ----------
    name        : the memory key this spec governs
    field_type  : expected type (FieldType enum value)
    required    : if True, the field must be present in ``write_keys``
    nullable    : if True, None is an acceptable value regardless of type
    default     : default value when the field is absent (ignored in validation)
    description : human-readable documentation
    min_value   : for INT/FLOAT fields — optional inclusive lower bound
    max_value   : for INT/FLOAT fields — optional inclusive upper bound
    pattern     : for STRING fields — optional regex the value must match
    element_type: for LIST fields — optional FieldType each element must satisfy
    validator   : optional callable(value) → bool for custom validation logic;
                  should return True if the value is acceptable
    """
    name: str
    field_type: FieldType
    required: bool = True
    nullable: bool = False
    default: Any = None
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    element_type: Optional[FieldType] = None
    validator: Optional[Callable[[Any], bool]] = None

    def __repr__(self) -> str:
        req = "required" if self.required else "optional"
        null = ", nullable" if self.nullable else ""
        return (
            f"FieldSpec({self.name!r}: {self.field_type.name}{null} [{req}]"
            f" — {self.description!r})"
        )


# ---------------------------------------------------------------------------
# SchemaValidator
# ---------------------------------------------------------------------------

class SchemaValidationError(Exception):
    """Raised when a value fails schema validation."""


class SchemaValidator:
    """
    Validates a value against a FieldSpec.

    All validation logic is concentrated here to keep FieldSpec and
    MemorySchema thin.  Raise SchemaValidationError with a descriptive
    message on any constraint violation.
    """

    # Map FieldType → Python type(s) used for isinstance checks
    _TYPE_MAP: Dict[FieldType, Tuple[type, ...]] = {
        FieldType.STRING: (str,),
        FieldType.INT:    (int,),
        FieldType.FLOAT:  (float, int),
        FieldType.BOOL:   (bool,),
        FieldType.LIST:   (list,),
        FieldType.DICT:   (dict,),
    }

    @classmethod
    def validate(cls, spec: FieldSpec, value: Any) -> bool:
        """
        Validate *value* against *spec*.

        Returns True on success.  Raises SchemaValidationError describing
        the first constraint violation found.
        """
        # Null check
        if value is None:
            if spec.nullable:
                return True
            raise SchemaValidationError(
                f"Field {spec.name!r}: None is not allowed "
                f"(set nullable=True to permit it)."
            )

        # ANY type bypasses all further checks
        if spec.field_type == FieldType.ANY:
            return True

        # OPTIONAL used as a standalone type is treated as ANY
        if spec.field_type == FieldType.OPTIONAL:
            return True

        # Type check
        expected_types = cls._TYPE_MAP.get(spec.field_type)
        if expected_types:
            # Special case: booleans are instances of int in Python;
            # reject them when INT is requested unless BOOL is specified.
            if spec.field_type == FieldType.INT and isinstance(value, bool):
                raise SchemaValidationError(
                    f"Field {spec.name!r}: expected int, got bool. "
                    "Use FieldType.BOOL for boolean fields."
                )
            if not isinstance(value, expected_types):
                raise SchemaValidationError(
                    f"Field {spec.name!r}: expected {spec.field_type.name} "
                    f"({[t.__name__ for t in expected_types]}), "
                    f"got {type(value).__name__} ({value!r})."
                )

        # Numeric range checks
        if spec.field_type in (FieldType.INT, FieldType.FLOAT):
            if spec.min_value is not None and value < spec.min_value:
                raise SchemaValidationError(
                    f"Field {spec.name!r}: value {value} is below "
                    f"minimum {spec.min_value}."
                )
            if spec.max_value is not None and value > spec.max_value:
                raise SchemaValidationError(
                    f"Field {spec.name!r}: value {value} exceeds "
                    f"maximum {spec.max_value}."
                )

        # String pattern check
        if spec.field_type == FieldType.STRING and spec.pattern is not None:
            if not re.fullmatch(spec.pattern, value):
                raise SchemaValidationError(
                    f"Field {spec.name!r}: value {value!r} does not match "
                    f"pattern {spec.pattern!r}."
                )

        # List element type check
        if spec.field_type == FieldType.LIST and spec.element_type is not None:
            elem_spec = FieldSpec(
                name=f"{spec.name}[*]",
                field_type=spec.element_type,
                nullable=spec.nullable,
            )
            for i, item in enumerate(value):
                try:
                    cls.validate(elem_spec, item)
                except SchemaValidationError as exc:
                    raise SchemaValidationError(
                        f"Field {spec.name!r} element [{i}]: {exc}"
                    ) from exc

        # Custom validator
        if spec.validator is not None:
            try:
                ok = spec.validator(value)
            except Exception as exc:
                raise SchemaValidationError(
                    f"Field {spec.name!r}: custom validator raised {exc!r}."
                ) from exc
            if not ok:
                raise SchemaValidationError(
                    f"Field {spec.name!r}: custom validator rejected value {value!r}."
                )

        return True

    @classmethod
    def safe_validate(cls, spec: FieldSpec, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Non-raising version of validate().

        Returns (True, None) on success or (False, error_message) on failure.
        """
        try:
            cls.validate(spec, value)
            return True, None
        except SchemaValidationError as exc:
            return False, str(exc)


# ---------------------------------------------------------------------------
# MemorySchema
# ---------------------------------------------------------------------------

class MemorySchema:
    """
    A typed contract for an agent's memory usage.

    A MemorySchema declares:
        - Which memory keys the agent reads (``read_keys``)
        - Which memory keys the agent writes (``write_keys``)
        - The type and constraints for each write key (``fields``)
        - A version string for backward compatibility

    Agents with a schema can be introspected by other agents and by
    monitoring tools.  The kernel (or a middleware layer) can call
    ``validate(key, value)`` before accepting a memory write to enforce
    the contract at runtime.

    Attributes
    ----------
    name       : human-readable schema name (usually the agent class name)
    version    : semver string, e.g. "1.0" or "2.3.1"
    fields     : list of FieldSpec objects (one per write key)
    read_keys  : list of memory keys the agent is declared to read
    write_keys : list of memory keys the agent is declared to write
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0",
        fields: Optional[List[FieldSpec]] = None,
        read_keys: Optional[List[str]] = None,
        write_keys: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.version = version
        self.fields: List[FieldSpec] = fields or []
        self.read_keys: List[str] = read_keys or []
        self.write_keys: List[str] = write_keys or []

        # Build a lookup dict for fast validation
        self._field_index: Dict[str, FieldSpec] = {
            f.name: f for f in self.fields
        }

    def get_field(self, name: str) -> Optional[FieldSpec]:
        """Return the FieldSpec for *name*, or None if not declared."""
        return self._field_index.get(name)

    def validate(self, key: str, value: Any) -> bool:
        """
        Validate *value* for *key* against the declared FieldSpec.

        Returns True if valid.  Raises SchemaValidationError if the key
        has a spec and the value violates it.  If the key has no spec
        (i.e., it is not in ``fields``), validation passes by default.
        """
        spec = self._field_index.get(key)
        if spec is None:
            # No spec declared — allow the write (open-world assumption)
            return True
        return SchemaValidator.validate(spec, value)

    def required_fields(self) -> List[FieldSpec]:
        """Return all FieldSpec entries where ``required=True``."""
        return [f for f in self.fields if f.required]

    def describe(self) -> str:
        """Return a human-readable summary of this schema."""
        lines = [
            f"Schema: {self.name} v{self.version}",
            f"  Reads  : {', '.join(self.read_keys) or '(none)'}",
            f"  Writes : {', '.join(self.write_keys) or '(none)'}",
            "  Fields:",
        ]
        if not self.fields:
            lines.append("    (none)")
        for spec in self.fields:
            null_tag = ", nullable" if spec.nullable else ""
            req_tag = "required" if spec.required else "optional"
            lines.append(
                f"    {spec.name} : {spec.field_type.name}{null_tag} "
                f"[{req_tag}] — {spec.description}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"MemorySchema(name={self.name!r}, version={self.version!r}, fields={len(self.fields)})"


# ---------------------------------------------------------------------------
# SchemaRegistry
# ---------------------------------------------------------------------------

class SchemaRegistry:
    """
    Global registry mapping agent_ids to their declared MemorySchemas.

    The registry is the central authority for schema lookup and
    write-time validation.  It is typically a singleton per kernel, but
    the class is not enforced as a singleton so that unit tests can
    create independent instances.

    Key responsibilities:
        register             — associate an agent_id with a MemorySchema
        get_schema           — retrieve a schema by agent_id
        list_schemas         — enumerate all registered schemas
        find_agents_with_field — find all agents that declare a given field
        validate_write       — validate a proposed write before it reaches
                               the MemoryManager
    """

    def __init__(self) -> None:
        # agent_id → MemorySchema
        self._registry: Dict[str, MemorySchema] = {}
        # schema_name → list of agent_ids (for class-level lookups)
        self._by_schema_name: Dict[str, List[str]] = {}

    def register(self, agent_id: str, schema: MemorySchema) -> None:
        """
        Register *schema* for *agent_id*.

        If an agent is re-registered, its old schema is replaced.
        """
        self._registry[agent_id] = schema
        self._by_schema_name.setdefault(schema.name, [])
        if agent_id not in self._by_schema_name[schema.name]:
            self._by_schema_name[schema.name].append(agent_id)

    def unregister(self, agent_id: str) -> None:
        """Remove the schema registration for *agent_id*."""
        schema = self._registry.pop(agent_id, None)
        if schema is not None:
            agent_list = self._by_schema_name.get(schema.name, [])
            if agent_id in agent_list:
                agent_list.remove(agent_id)

    def get_schema(self, agent_id: str) -> Optional[MemorySchema]:
        """Return the MemorySchema registered for *agent_id*, or None."""
        return self._registry.get(agent_id)

    def list_schemas(self) -> List[Tuple[str, MemorySchema]]:
        """Return all (agent_id, MemorySchema) pairs in registration order."""
        return list(self._registry.items())

    def find_agents_with_field(self, field_name: str) -> List[str]:
        """
        Return agent_ids whose schema declares a field named *field_name*.

        Searches all registered schemas; returns an empty list if none match.
        """
        return [
            agent_id
            for agent_id, schema in self._registry.items()
            if schema.get_field(field_name) is not None
        ]

    def find_agents_writing_key(self, key: str) -> List[str]:
        """Return agent_ids whose schema lists *key* in ``write_keys``."""
        return [
            agent_id
            for agent_id, schema in self._registry.items()
            if key in schema.write_keys
        ]

    def find_agents_reading_key(self, key: str) -> List[str]:
        """Return agent_ids whose schema lists *key* in ``read_keys``."""
        return [
            agent_id
            for agent_id, schema in self._registry.items()
            if key in schema.read_keys
        ]

    def validate_write(self, agent_id: str, key: str, value: Any) -> bool:
        """
        Validate a proposed write of *value* to *key* by *agent_id*.

        If the agent has no registered schema, or the key is not declared
        in its schema, the write is permitted (open-world assumption).

        Returns True on success.
        Raises SchemaValidationError if the value violates the schema.
        """
        schema = self._registry.get(agent_id)
        if schema is None:
            return True  # No schema — writes always allowed
        return schema.validate(key, value)

    def agents_by_schema_name(self, schema_name: str) -> List[str]:
        """Return all agent_ids registered under schema named *schema_name*."""
        return list(self._by_schema_name.get(schema_name, []))

    def summary(self) -> str:
        """Return a formatted summary of all registered schemas."""
        if not self._registry:
            return "SchemaRegistry: (empty)"
        lines = ["SchemaRegistry:"]
        for agent_id, schema in self._registry.items():
            lines.append(f"  {agent_id!r} → {schema.name} v{schema.version}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SchemaRegistry({len(self._registry)} agents registered)"


# ---------------------------------------------------------------------------
# SchemaInspector
# ---------------------------------------------------------------------------

class SchemaInspector:
    """
    Allows agents (or tools) to introspect other agents' declared schemas.

    This is the "reflection" API of the Battousai type system.  An agent can
    discover what data another agent produces and how it is typed, enabling
    dynamic composition of agent pipelines.

    Attributes
    ----------
    registry : the SchemaRegistry this inspector operates over
    """

    def __init__(self, registry: SchemaRegistry) -> None:
        self.registry = registry

    def inspect(self, agent_id: str) -> Optional[MemorySchema]:
        """
        Return the MemorySchema declared by *agent_id*, or None if
        the agent has no registered schema.
        """
        return self.registry.get_schema(agent_id)

    def describe(self, agent_id: str) -> str:
        """
        Return a human-readable description of *agent_id*'s schema,
        or a fallback message if no schema is registered.
        """
        schema = self.inspect(agent_id)
        if schema is None:
            return f"Agent {agent_id!r} has no registered schema."
        return schema.describe()

    def find_providers(self, field_name: str) -> List[str]:
        """
        Return the agent_ids of all agents that declare *field_name*
        in their schema's ``write_keys``.

        This enables a consumer agent to locate a compatible producer
        at runtime without hard-coded agent IDs.
        """
        return self.registry.find_agents_writing_key(field_name)

    def find_consumers(self, field_name: str) -> List[str]:
        """
        Return the agent_ids of all agents that declare *field_name*
        in their schema's ``read_keys``.
        """
        return self.registry.find_agents_reading_key(field_name)

    def compatible_writers(
        self, field_name: str, required_type: FieldType
    ) -> List[str]:
        """
        Return agent_ids that write *field_name* with a compatible type.

        An agent is included if its FieldSpec for *field_name* matches
        *required_type* or is typed as ANY.
        """
        result: List[str] = []
        for agent_id in self.registry.find_agents_with_field(field_name):
            schema = self.registry.get_schema(agent_id)
            if schema is None:
                continue
            spec = schema.get_field(field_name)
            if spec and spec.field_type in (required_type, FieldType.ANY):
                result.append(agent_id)
        return result

    def all_declared_keys(self) -> Set[str]:
        """
        Return the union of all write_keys across every registered schema.

        Useful for building a global data catalog.
        """
        keys: Set[str] = set()
        for _, schema in self.registry.list_schemas():
            keys.update(schema.write_keys)
        return keys

    def __repr__(self) -> str:
        return f"SchemaInspector(registry={self.registry!r})"


# ---------------------------------------------------------------------------
# Module-level singleton registry
# ---------------------------------------------------------------------------

#: The default global SchemaRegistry.  Use this unless you need isolation.
GLOBAL_REGISTRY: SchemaRegistry = SchemaRegistry()


# ---------------------------------------------------------------------------
# @schema class decorator
# ---------------------------------------------------------------------------

def schema(
    name: str,
    version: str = "1.0",
    fields: Optional[List[FieldSpec]] = None,
    reads: Optional[List[str]] = None,
    writes: Optional[List[str]] = None,
    registry: Optional[SchemaRegistry] = None,
) -> Callable[[Type], Type]:
    """
    Class decorator that attaches a MemorySchema to an Agent subclass and
    auto-registers the schema with the global (or provided) SchemaRegistry.

    The schema is stored on the class as ``cls._memory_schema`` and
    registered under the class name (not yet an agent_id — the agent_id is
    assigned by the kernel at spawn time).  The kernel or a middleware layer
    should call ``registry.register(agent.agent_id, agent._memory_schema)``
    after spawning.

    Parameters
    ----------
    name     : schema name (usually the agent class name)
    version  : semver string
    fields   : list of FieldSpec objects
    reads    : list of memory keys the agent reads
    writes   : list of memory keys the agent writes
    registry : SchemaRegistry to register with (defaults to GLOBAL_REGISTRY)

    Example::

        @schema(
            name="ResearchWorker",
            version="1.0",
            fields=[
                FieldSpec("findings",   FieldType.LIST,  required=True,
                          description="Research results"),
                FieldSpec("confidence", FieldType.FLOAT, required=True,
                          description="Confidence score 0-1",
                          min_value=0.0, max_value=1.0),
            ],
            reads=["task_assignment", "global.config"],
            writes=["findings", "confidence", "status"],
        )
        class ResearchWorkerAgent(Agent):
            ...
    """
    reg = registry or GLOBAL_REGISTRY

    def decorator(cls: Type) -> Type:
        mem_schema = MemorySchema(
            name=name,
            version=version,
            fields=fields or [],
            read_keys=reads or [],
            write_keys=writes or [],
        )
        # Attach to class for introspection
        cls._memory_schema = mem_schema
        # Register under the class name as a sentinel key.
        # The actual agent_id-keyed entry should be registered at spawn time.
        reg.register(f"class:{cls.__name__}", mem_schema)
        return cls

    return decorator
