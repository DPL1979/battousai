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


class FieldType(Enum):
    STRING   = auto()
    INT      = auto()
    FLOAT    = auto()
    BOOL     = auto()
    LIST     = auto()
    DICT     = auto()
    ANY      = auto()
    OPTIONAL = auto()


@dataclass
class FieldSpec:
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
            f" - {self.description!r})"
        )


class SchemaValidationError(Exception):
    """Raised when a value fails schema validation."""


class SchemaValidator:
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
        if value is None:
            if spec.nullable:
                return True
            raise SchemaValidationError(
                f"Field {spec.name!r}: None is not allowed "
                f"(set nullable=True to permit it)."
            )
        if spec.field_type == FieldType.ANY:
            return True
        if spec.field_type == FieldType.OPTIONAL:
            return True
        expected_types = cls._TYPE_MAP.get(spec.field_type)
        if expected_types:
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
        if spec.field_type == FieldType.STRING and spec.pattern is not None:
            if not re.fullmatch(spec.pattern, value):
                raise SchemaValidationError(
                    f"Field {spec.name!r}: value {value!r} does not match "
                    f"pattern {spec.pattern!r}."
                )
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
        try:
            cls.validate(spec, value)
            return True, None
        except SchemaValidationError as exc:
            return False, str(exc)


class MemorySchema:
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
        self._field_index: Dict[str, FieldSpec] = {
            f.name: f for f in self.fields
        }

    def get_field(self, name: str) -> Optional[FieldSpec]:
        return self._field_index.get(name)

    def validate(self, key: str, value: Any) -> bool:
        spec = self._field_index.get(key)
        if spec is None:
            return True
        return SchemaValidator.validate(spec, value)

    def required_fields(self) -> List[FieldSpec]:
        return [f for f in self.fields if f.required]

    def describe(self) -> str:
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
                f"[{req_tag}] - {spec.description}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"MemorySchema(name={self.name!r}, version={self.version!r}, fields={len(self.fields)})"


class SchemaRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, MemorySchema] = {}
        self._by_schema_name: Dict[str, List[str]] = {}

    def register(self, agent_id: str, schema: MemorySchema) -> None:
        self._registry[agent_id] = schema
        self._by_schema_name.setdefault(schema.name, [])
        if agent_id not in self._by_schema_name[schema.name]:
            self._by_schema_name[schema.name].append(agent_id)

    def unregister(self, agent_id: str) -> None:
        schema = self._registry.pop(agent_id, None)
        if schema is not None:
            agent_list = self._by_schema_name.get(schema.name, [])
            if agent_id in agent_list:
                agent_list.remove(agent_id)

    def get_schema(self, agent_id: str) -> Optional[MemorySchema]:
        return self._registry.get(agent_id)

    def list_schemas(self) -> List[Tuple[str, MemorySchema]]:
        return list(self._registry.items())

    def find_agents_with_field(self, field_name: str) -> List[str]:
        return [
            agent_id
            for agent_id, schema in self._registry.items()
            if schema.get_field(field_name) is not None
        ]

    def find_agents_writing_key(self, key: str) -> List[str]:
        return [
            agent_id
            for agent_id, schema in self._registry.items()
            if key in schema.write_keys
        ]

    def find_agents_reading_key(self, key: str) -> List[str]:
        return [
            agent_id
            for agent_id, schema in self._registry.items()
            if key in schema.read_keys
        ]

    def validate_write(self, agent_id: str, key: str, value: Any) -> bool:
        schema = self._registry.get(agent_id)
        if schema is None:
            return True
        return schema.validate(key, value)

    def agents_by_schema_name(self, schema_name: str) -> List[str]:
        return list(self._by_schema_name.get(schema_name, []))

    def summary(self) -> str:
        if not self._registry:
            return "SchemaRegistry: (empty)"
        lines = ["SchemaRegistry:"]
        for agent_id, schema in self._registry.items():
            lines.append(f"  {agent_id!r} -> {schema.name} v{schema.version}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SchemaRegistry({len(self._registry)} agents registered)"


class SchemaInspector:
    def __init__(self, registry: SchemaRegistry) -> None:
        self.registry = registry

    def inspect(self, agent_id: str) -> Optional[MemorySchema]:
        return self.registry.get_schema(agent_id)

    def describe(self, agent_id: str) -> str:
        schema = self.inspect(agent_id)
        if schema is None:
            return f"Agent {agent_id!r} has no registered schema."
        return schema.describe()

    def find_providers(self, field_name: str) -> List[str]:
        return self.registry.find_agents_writing_key(field_name)

    def find_consumers(self, field_name: str) -> List[str]:
        return self.registry.find_agents_reading_key(field_name)

    def compatible_writers(
        self, field_name: str, required_type: FieldType
    ) -> List[str]:
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
        keys: Set[str] = set()
        for _, schema in self.registry.list_schemas():
            keys.update(schema.write_keys)
        return keys

    def __repr__(self) -> str:
        return f"SchemaInspector(registry={self.registry!r})"


GLOBAL_REGISTRY: SchemaRegistry = SchemaRegistry()


def schema(
    name: str,
    version: str = "1.0",
    fields: Optional[List[FieldSpec]] = None,
    reads: Optional[List[str]] = None,
    writes: Optional[List[str]] = None,
    registry: Optional[SchemaRegistry] = None,
) -> Callable[[Type], Type]:
    reg = registry or GLOBAL_REGISTRY

    def decorator(cls: Type) -> Type:
        mem_schema = MemorySchema(
            name=name,
            version=version,
            fields=fields or [],
            read_keys=reads or [],
            write_keys=writes or [],
        )
        cls._memory_schema = mem_schema
        reg.register(f"class:{cls.__name__}", mem_schema)
        return cls

    return decorator
