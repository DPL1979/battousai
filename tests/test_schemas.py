"""
test_schemas.py — Tests for battousai.schemas
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from battousai.schemas import (
    FieldSpec, FieldType, SchemaValidator, MemorySchema,
    SchemaRegistry, schema, GLOBAL_REGISTRY,
    SchemaValidationError,
)
from battousai.agent import Agent


class TestFieldSpec(unittest.TestCase):

    def test_field_spec_stores_name(self):
        f = FieldSpec(name="my_field", field_type=FieldType.STRING)
        self.assertEqual(f.name, "my_field")

    def test_field_spec_stores_type(self):
        f = FieldSpec(name="count", field_type=FieldType.INT)
        self.assertEqual(f.field_type, FieldType.INT)

    def test_field_type_enum_has_required_values(self):
        names = [ft.name for ft in FieldType]
        for expected in ["STRING", "INT", "FLOAT", "BOOL", "LIST", "DICT", "ANY"]:
            self.assertIn(expected, names)


class TestSchemaValidator(unittest.TestCase):
    """
    SchemaValidator.validate(spec, value) raises SchemaValidationError on failure,
    returns True on success.
    Use safe_validate(spec, value) → (bool, Optional[str]) for non-raising tests.
    """

    def setUp(self):
        self.validator = SchemaValidator()

    def test_validate_string_field_passes(self):
        spec = FieldSpec(name="name", field_type=FieldType.STRING, required=True)
        ok, err = SchemaValidator.safe_validate(spec, "hello")
        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_validate_string_field_wrong_type(self):
        spec = FieldSpec(name="name", field_type=FieldType.STRING, required=True)
        ok, err = SchemaValidator.safe_validate(spec, 42)
        self.assertFalse(ok)
        self.assertIsNotNone(err)

    def test_validate_int_field_passes(self):
        spec = FieldSpec(name="count", field_type=FieldType.INT, required=True)
        ok, err = SchemaValidator.safe_validate(spec, 5)
        self.assertTrue(ok)

    def test_validate_int_field_wrong_type(self):
        spec = FieldSpec(name="count", field_type=FieldType.INT, required=True)
        ok, err = SchemaValidator.safe_validate(spec, "five")
        self.assertFalse(ok)

    def test_validate_float_accepts_int_and_float(self):
        spec = FieldSpec(name="ratio", field_type=FieldType.FLOAT)
        ok, _ = SchemaValidator.safe_validate(spec, 3.14)
        self.assertTrue(ok)
        ok2, _ = SchemaValidator.safe_validate(spec, 3)
        self.assertTrue(ok2)

    def test_validate_bool_field(self):
        spec = FieldSpec(name="flag", field_type=FieldType.BOOL)
        ok, _ = SchemaValidator.safe_validate(spec, True)
        self.assertTrue(ok)
        ok2, _ = SchemaValidator.safe_validate(spec, "yes")
        self.assertFalse(ok2)

    def test_validate_nullable_field_accepts_none(self):
        spec = FieldSpec(name="opt", field_type=FieldType.STRING, nullable=True)
        ok, _ = SchemaValidator.safe_validate(spec, None)
        self.assertTrue(ok)

    def test_validate_non_nullable_field_rejects_none(self):
        spec = FieldSpec(name="req", field_type=FieldType.STRING, nullable=False)
        ok, err = SchemaValidator.safe_validate(spec, None)
        self.assertFalse(ok)

    def test_validate_list_field(self):
        spec = FieldSpec(name="items", field_type=FieldType.LIST)
        ok, _ = SchemaValidator.safe_validate(spec, [1, 2, 3])
        self.assertTrue(ok)
        ok2, _ = SchemaValidator.safe_validate(spec, "not a list")
        self.assertFalse(ok2)

    def test_validate_dict_field(self):
        spec = FieldSpec(name="data", field_type=FieldType.DICT)
        ok, _ = SchemaValidator.safe_validate(spec, {"key": "val"})
        self.assertTrue(ok)

    def test_validate_any_field_accepts_anything(self):
        spec = FieldSpec(name="wild", field_type=FieldType.ANY)
        # ANY bypasses type checks; None still fails if not nullable
        for val in [42, "string", [1, 2], {"k": "v"}, 3.14, True]:
            ok, _ = SchemaValidator.safe_validate(spec, val)
            self.assertTrue(ok, f"ANY field should accept {val!r}")
        # None with ANY and nullable=True
        spec_nullable = FieldSpec(name="wild_null", field_type=FieldType.ANY, nullable=True)
        ok, _ = SchemaValidator.safe_validate(spec_nullable, None)
        self.assertTrue(ok, "ANY nullable field should accept None")


class TestMemorySchema(unittest.TestCase):

    def setUp(self):
        self.schema_obj = MemorySchema(
            name="task_schema",
            version="1.0",
            fields=[
                FieldSpec("task_id", FieldType.STRING, required=True),
                FieldSpec("priority", FieldType.INT, required=True),
                FieldSpec("payload", FieldType.DICT, required=False, nullable=True),
            ],
            read_keys=["task_id", "priority", "payload"],
            write_keys=["task_id", "priority", "payload"],
        )

    def test_schema_stores_name(self):
        self.assertEqual(self.schema_obj.name, "task_schema")

    def test_schema_stores_version(self):
        self.assertEqual(self.schema_obj.version, "1.0")

    def test_schema_has_correct_field_count(self):
        self.assertEqual(len(self.schema_obj.fields), 3)

    def test_schema_validate_valid_data(self):
        # Use safe_validate via SchemaValidator (classmethod)
        for field in self.schema_obj.fields:
            if field.name == "task_id":
                ok, _ = SchemaValidator.safe_validate(field, "task_001")
                self.assertTrue(ok)
            elif field.name == "priority":
                ok, _ = SchemaValidator.safe_validate(field, 5)
                self.assertTrue(ok)


class TestSchemaRegistry(unittest.TestCase):
    """
    SchemaRegistry.register(agent_id, schema) — takes an agent_id and MemorySchema.
    SchemaRegistry.get_schema(agent_id) — returns the schema for an agent_id.
    SchemaRegistry.list_schemas() — returns [(agent_id, MemorySchema)] pairs.
    """

    def setUp(self):
        self.registry = SchemaRegistry()

    def test_register_and_get_schema(self):
        s = MemorySchema(
            name="reg_schema",
            version="1.0",
            fields=[FieldSpec("x", FieldType.INT)],
            read_keys=["x"],
            write_keys=["x"],
        )
        # register() takes (agent_id, schema)
        self.registry.register("agent_reg_001", s)
        retrieved = self.registry.get_schema("agent_reg_001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "reg_schema")

    def test_get_unknown_schema_returns_none(self):
        # get_schema() returns None for unknown agent_ids
        result = self.registry.get_schema("no_such_agent")
        self.assertIsNone(result)

    def test_list_schemas_includes_registered(self):
        s = MemorySchema(
            name="listed_schema",
            version="1.0",
            fields=[],
            read_keys=[],
            write_keys=[],
        )
        self.registry.register("agent_list_001", s)
        # list_schemas() returns [(agent_id, MemorySchema)] pairs
        pairs = self.registry.list_schemas()
        schema_names = [schema.name for _, schema in pairs]
        self.assertIn("listed_schema", schema_names)


class TestSchemaDecorator(unittest.TestCase):

    def test_schema_decorator_attaches_schema_to_class(self):
        @schema(
            name="decorated_schema",
            version="1.0",
            fields=[FieldSpec("value", FieldType.INT)],
            reads=["value"],
            writes=["value"],
        )
        class MyAgent(Agent):
            def think(self, tick):
                self.yield_cpu()

        self.assertTrue(hasattr(MyAgent, "_memory_schema"))
        self.assertEqual(MyAgent._memory_schema.name, "decorated_schema")


if __name__ == "__main__":
    unittest.main()
