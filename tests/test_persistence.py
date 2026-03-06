"""
tests/test_persistence.py — Unit tests for battousai.persistence
=================================================================
Tests PersistenceLayer: SQLite schema creation, save/load, checkpoint/restore.
Uses an in-memory SQLite database so no disk files are created.
"""

from __future__ import annotations

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from battousai.persistence import PersistenceLayer, _SCHEMA_VERSION
from battousai.memory import MemoryManager, MemoryType


class TestPersistenceLayerConnection(unittest.TestCase):

    def setUp(self):
        self.pl = PersistenceLayer(":memory:")
        self.pl.migrate()

    def tearDown(self):
        self.pl.close()

    def test_connect_returns_connection(self):
        import sqlite3
        conn = self.pl.connect()
        self.assertIsInstance(conn, sqlite3.Connection)

    def test_connect_idempotent(self):
        c1 = self.pl.connect()
        c2 = self.pl.connect()
        self.assertIs(c1, c2)

    def test_context_manager(self):
        with PersistenceLayer(":memory:") as pl:
            # Should be connected and migrated
            v = pl.get_schema_version()
            self.assertEqual(v, _SCHEMA_VERSION)


class TestSchemaMigration(unittest.TestCase):

    def setUp(self):
        self.pl = PersistenceLayer(":memory:")

    def tearDown(self):
        self.pl.close()

    def test_migrate_creates_all_tables(self):
        self.pl.migrate()
        conn = self.pl.connect()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
        for expected in (
            "schema_version",
            "agent_state",
            "memory_entries",
            "audit_log",
            "capability_grants",
        ):
            self.assertIn(expected, tables, f"Missing table: {expected}")

    def test_migrate_idempotent(self):
        self.pl.migrate()
        self.pl.migrate()  # Second call should not raise or duplicate version rows
        conn = self.pl.connect()
        count = conn.execute("SELECT COUNT(*) FROM schema_version;").fetchone()[0]
        self.assertEqual(count, 1)

    def test_schema_version_stored(self):
        self.pl.migrate()
        self.assertEqual(self.pl.get_schema_version(), _SCHEMA_VERSION)

    def test_wal_mode_enabled(self):
        self.pl.migrate()
        conn = self.pl.connect()
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        # In-memory DB may return 'memory' instead of 'wal'; that's acceptable.
        self.assertIn(mode, ("wal", "memory"))


class TestAgentStatePersistence(unittest.TestCase):

    def setUp(self):
        self.pl = PersistenceLayer(":memory:")
        self.pl.migrate()

    def tearDown(self):
        self.pl.close()

    def test_save_and_load_agent(self):
        self.pl.save_agent(
            agent_id="agent_001",
            name="TestWorker",
            agent_class="WorkerAgent",
            priority=3,
            status="active",
            metadata={"role": "worker"},
        )
        agents = self.pl.load_agents()
        self.assertEqual(len(agents), 1)
        a = agents[0]
        self.assertEqual(a["agent_id"], "agent_001")
        self.assertEqual(a["name"], "TestWorker")
        self.assertEqual(a["agent_class"], "WorkerAgent")
        self.assertEqual(a["priority"], 3)
        self.assertEqual(a["metadata"]["role"], "worker")

    def test_save_agent_upsert(self):
        self.pl.save_agent("a1", "Worker", "WorkerAgent", status="active")
        self.pl.save_agent("a1", "WorkerUpdated", "WorkerAgent", status="idle")
        agents = self.pl.load_agents()
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0]["name"], "WorkerUpdated")
        self.assertEqual(agents[0]["status"], "idle")

    def test_delete_agent(self):
        self.pl.save_agent("del_me", "Temp", "WorkerAgent")
        self.pl.delete_agent("del_me")
        agents = self.pl.load_agents()
        self.assertEqual(len(agents), 0)

    def test_load_agents_empty(self):
        agents = self.pl.load_agents()
        self.assertEqual(agents, [])

    def test_multiple_agents(self):
        for i in range(5):
            self.pl.save_agent(f"agent_{i}", f"Worker{i}", "WorkerAgent")
        agents = self.pl.load_agents()
        self.assertEqual(len(agents), 5)


class TestMemoryEntries(unittest.TestCase):

    def setUp(self):
        self.pl = PersistenceLayer(":memory:")
        self.pl.migrate()

    def tearDown(self):
        self.pl.close()

    def test_save_and_load_memory_entry(self):
        self.pl.save_memory_entry(
            agent_id="agent1",
            key="greeting",
            value="Hello, world!",
            memory_type="LONG_TERM",
            created_tick=5,
        )
        entries = self.pl.load_memory_entries("agent1")
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["key"], "greeting")
        self.assertEqual(e["value"], "Hello, world!")
        self.assertEqual(e["memory_type"], "LONG_TERM")
        self.assertEqual(e["created_tick"], 5)

    def test_save_memory_entry_with_dict_value(self):
        value = {"nested": {"x": 1, "y": [1, 2, 3]}}
        self.pl.save_memory_entry(
            agent_id="agent1",
            key="complex",
            value=value,
            memory_type="SHORT_TERM",
        )
        entries = self.pl.load_memory_entries("agent1")
        self.assertEqual(entries[0]["value"], value)

    def test_save_memory_entry_with_ttl(self):
        self.pl.save_memory_entry(
            agent_id="agent1",
            key="temp",
            value="ephemeral",
            memory_type="SHORT_TERM",
            ttl_ticks=10,
        )
        entries = self.pl.load_memory_entries("agent1")
        self.assertEqual(entries[0]["ttl_ticks"], 10)

    def test_load_shared_region_entries(self):
        self.pl.save_memory_entry(
            agent_id="agent1",
            key="shared_key",
            value="shared_value",
            memory_type="SHARED",
            region_name="global",
        )
        # Agent-only entries should not appear in shared query
        agent_entries = self.pl.load_memory_entries("agent1")
        self.assertEqual(len(agent_entries), 0)
        # Shared query
        shared_entries = self.pl.load_memory_entries("agent1", region_name="global")
        self.assertEqual(len(shared_entries), 1)
        self.assertEqual(shared_entries[0]["value"], "shared_value")

    def test_upsert_memory_entry(self):
        self.pl.save_memory_entry("a1", "k", "v1", "LONG_TERM")
        self.pl.save_memory_entry("a1", "k", "v2", "LONG_TERM")
        entries = self.pl.load_memory_entries("a1")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["value"], "v2")


class TestMemoryManagerSaveLoad(unittest.TestCase):

    def setUp(self):
        self.pl = PersistenceLayer(":memory:")
        self.pl.migrate()

    def tearDown(self):
        self.pl.close()

    def _make_mm(self) -> MemoryManager:
        mm = MemoryManager()
        mm.create_agent_space("agent1")
        mm.create_agent_space("agent2")
        mm.create_shared_region("global")
        return mm

    def test_save_and_load_agent_memory(self):
        mm = self._make_mm()
        mm.agent_write("agent1", "key1", "value1", MemoryType.LONG_TERM)
        mm.agent_write("agent1", "key2", 42, MemoryType.LONG_TERM)
        mm.agent_write("agent2", "keyA", [1, 2, 3], MemoryType.LONG_TERM)

        self.pl.save_memory_manager(mm)

        mm2 = MemoryManager()
        self.pl.load_memory_manager(mm2)

        self.assertEqual(mm2.agent_read("agent1", "key1"), "value1")
        self.assertEqual(mm2.agent_read("agent1", "key2"), 42)
        self.assertEqual(mm2.agent_read("agent2", "keyA"), [1, 2, 3])

    def test_save_and_load_shared_memory(self):
        mm = self._make_mm()
        mm.shared_write("global", "agent1", "shared_key", "shared_data")

        self.pl.save_memory_manager(mm)

        mm2 = MemoryManager()
        self.pl.load_memory_manager(mm2)

        value = mm2.shared_read("global", "agent1", "shared_key")
        self.assertEqual(value, "shared_data")

    def test_save_memory_clears_previous(self):
        """Saving twice should replace, not duplicate, entries."""
        mm = self._make_mm()
        mm.agent_write("agent1", "k", "first", MemoryType.LONG_TERM)
        self.pl.save_memory_manager(mm)

        mm.agent_write("agent1", "k", "second", MemoryType.LONG_TERM)
        self.pl.save_memory_manager(mm)

        entries = self.pl.load_memory_entries("agent1")
        matching = [e for e in entries if e["key"] == "k"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["value"], "second")

    def test_memory_type_preserved(self):
        mm = self._make_mm()
        mm.agent_write("agent1", "st_key", "val", MemoryType.SHORT_TERM, ttl_ticks=5)
        self.pl.save_memory_manager(mm)

        entries = self.pl.load_memory_entries("agent1")
        e = entries[0]
        self.assertEqual(e["memory_type"], "SHORT_TERM")
        self.assertEqual(e["ttl_ticks"], 5)

    def test_load_restores_correct_agent_spaces(self):
        mm = self._make_mm()
        mm.agent_write("agent1", "x", 100, MemoryType.LONG_TERM)
        mm.agent_write("agent2", "y", 200, MemoryType.LONG_TERM)
        self.pl.save_memory_manager(mm)

        mm2 = MemoryManager()
        self.pl.load_memory_manager(mm2)

        self.assertIn("agent1", mm2._agents)
        self.assertIn("agent2", mm2._agents)


class TestAuditLog(unittest.TestCase):

    def setUp(self):
        self.pl = PersistenceLayer(":memory:")
        self.pl.migrate()

    def tearDown(self):
        self.pl.close()

    def test_save_and_load_audit_entry(self):
        self.pl.save_audit_entry(
            entry_id="entry_001",
            agent_id="agent1",
            cap_type="FILE_READ",
            resource="/agents/agent1/data.txt",
            action="read",
            granted=True,
            tick=10,
        )
        entries = self.pl.load_audit_log()
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["entry_id"], "entry_001")
        self.assertEqual(e["agent_id"], "agent1")
        self.assertEqual(e["cap_type"], "FILE_READ")
        self.assertTrue(bool(e["granted"]))

    def test_load_audit_log_filtered_by_agent(self):
        for i in range(3):
            self.pl.save_audit_entry(
                entry_id=f"e{i}_a",
                agent_id="agentA",
                cap_type="TOOL_USE",
                resource="calc",
                action="use",
                granted=True,
            )
        self.pl.save_audit_entry(
            entry_id="e0_b",
            agent_id="agentB",
            cap_type="FILE_READ",
            resource="/x",
            action="read",
            granted=False,
        )
        entries_a = self.pl.load_audit_log(agent_id="agentA")
        entries_b = self.pl.load_audit_log(agent_id="agentB")
        self.assertEqual(len(entries_a), 3)
        self.assertEqual(len(entries_b), 1)

    def test_duplicate_entry_id_ignored(self):
        self.pl.save_audit_entry("dup_id", "a1", "FILE_READ", None, "read", True)
        self.pl.save_audit_entry("dup_id", "a1", "FILE_READ", None, "read", True)
        entries = self.pl.load_audit_log()
        self.assertEqual(len(entries), 1)

    def test_audit_log_limit(self):
        for i in range(20):
            self.pl.save_audit_entry(
                f"e{i}", "agent1", "TOOL_USE", "calc", "use", True
            )
        entries = self.pl.load_audit_log(limit=5)
        self.assertEqual(len(entries), 5)


class TestCapabilityGrants(unittest.TestCase):

    def setUp(self):
        self.pl = PersistenceLayer(":memory:")
        self.pl.migrate()

    def tearDown(self):
        self.pl.close()

    def test_save_and_load_grant(self):
        self.pl.save_capability_grant(
            token_id="tok_001",
            agent_id="agent1",
            cap_type="FILE_READ",
            resource_pattern="/agents/agent1/*",
            delegatable=True,
        )
        grants = self.pl.load_capability_grants()
        self.assertEqual(len(grants), 1)
        g = grants[0]
        self.assertEqual(g["token_id"], "tok_001")
        self.assertEqual(g["agent_id"], "agent1")
        self.assertEqual(g["cap_type"], "FILE_READ")
        self.assertTrue(g["delegatable"])

    def test_revoke_grant(self):
        self.pl.save_capability_grant("tok_del", "agent1", "NETWORK", None)
        self.pl.revoke_capability_grant("tok_del")
        grants = self.pl.load_capability_grants()
        self.assertEqual(len(grants), 0)

    def test_load_grants_filtered_by_agent(self):
        self.pl.save_capability_grant("t1", "agentA", "TOOL_USE", "calc")
        self.pl.save_capability_grant("t2", "agentA", "FILE_READ", "/a/*")
        self.pl.save_capability_grant("t3", "agentB", "SPAWN", "*")
        grants_a = self.pl.load_capability_grants(agent_id="agentA")
        self.assertEqual(len(grants_a), 2)


class TestCheckpointRestore(unittest.TestCase):

    def setUp(self):
        self.pl = PersistenceLayer(":memory:")
        self.pl.migrate()

    def tearDown(self):
        self.pl.close()

    def test_checkpoint_returns_string_id(self):
        mm = MemoryManager()
        cid = self.pl.checkpoint(mm)
        self.assertIsInstance(cid, str)
        self.assertTrue(len(cid) > 0)

    def test_checkpoint_and_restore(self):
        mm = MemoryManager()
        mm.create_agent_space("agent1")
        mm.agent_write("agent1", "remember_this", "important_value", MemoryType.LONG_TERM)

        cid = self.pl.checkpoint(mm)

        mm2 = MemoryManager()
        result = self.pl.restore(cid, mm2)
        self.assertTrue(result)
        self.assertEqual(mm2.agent_read("agent1", "remember_this"), "important_value")

    def test_restore_unknown_checkpoint_returns_false(self):
        mm = MemoryManager()
        result = self.pl.restore("nonexistent_id", mm)
        self.assertFalse(result)

    def test_checkpoint_records_sentinel_agent(self):
        mm = MemoryManager()
        cid = self.pl.checkpoint(mm)
        agents = self.pl.load_agents()
        sentinel_ids = [a["agent_id"] for a in agents if "checkpoint" in a["agent_id"]]
        self.assertTrue(len(sentinel_ids) > 0)
        self.assertIn(cid, sentinel_ids[0])

    def test_multiple_checkpoints_independent(self):
        mm = MemoryManager()
        mm.create_agent_space("agent1")
        mm.agent_write("agent1", "k", "v1", MemoryType.LONG_TERM)
        cid1 = self.pl.checkpoint(mm)

        mm.agent_write("agent1", "k", "v2", MemoryType.LONG_TERM)
        cid2 = self.pl.checkpoint(mm)

        self.assertNotEqual(cid1, cid2)


class TestTableRowCounts(unittest.TestCase):

    def setUp(self):
        self.pl = PersistenceLayer(":memory:")
        self.pl.migrate()

    def tearDown(self):
        self.pl.close()

    def test_table_row_counts_structure(self):
        counts = self.pl.table_row_counts()
        for table in ("schema_version", "agent_state", "memory_entries",
                      "audit_log", "capability_grants"):
            self.assertIn(table, counts)
            self.assertIsInstance(counts[table], int)

    def test_clear_all(self):
        self.pl.save_agent("a1", "Worker", "WorkerAgent")
        self.pl.save_memory_entry("a1", "k", "v", "LONG_TERM")
        self.pl.clear_all()
        counts = self.pl.table_row_counts()
        self.assertEqual(counts["agent_state"], 0)
        self.assertEqual(counts["memory_entries"], 0)


if __name__ == "__main__":
    unittest.main()
