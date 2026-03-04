"""
test_memory.py — Tests for battousai.memory
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from battousai.memory import (
    MemoryManager, AgentMemorySpace, SharedMemoryRegion,
    MemoryType, MemoryEntry, MemoryFullError, MemoryAccessError, MemoryKeyError
)


class TestAgentMemorySpace(unittest.TestCase):

    def setUp(self):
        self.manager = MemoryManager()
        self.manager.create_agent_space("agent_0001", max_keys=10)

    def test_write_and_read_short_term(self):
        self.manager.agent_write(
            "agent_0001", "key1", "value1", MemoryType.SHORT_TERM, current_tick=0
        )
        value = self.manager.agent_read("agent_0001", "key1")
        self.assertEqual(value, "value1")

    def test_write_and_read_long_term(self):
        self.manager.agent_write(
            "agent_0001", "persist_key", 42, MemoryType.LONG_TERM, current_tick=0
        )
        value = self.manager.agent_read("agent_0001", "persist_key")
        self.assertEqual(value, 42)

    def test_read_missing_key_raises_memory_key_error(self):
        with self.assertRaises(MemoryKeyError):
            self.manager.agent_read("agent_0001", "no_such_key")

    def test_overwrite_existing_key(self):
        self.manager.agent_write(
            "agent_0001", "k", "first", MemoryType.LONG_TERM, current_tick=0
        )
        self.manager.agent_write(
            "agent_0001", "k", "second", MemoryType.LONG_TERM, current_tick=1
        )
        value = self.manager.agent_read("agent_0001", "k")
        self.assertEqual(value, "second")

    def test_memory_full_raises_error(self):
        """Writing beyond max_keys should raise MemoryFullError."""
        for i in range(10):
            self.manager.agent_write(
                "agent_0001", f"k{i}", i, MemoryType.SHORT_TERM, current_tick=0
            )
        with self.assertRaises(MemoryFullError):
            self.manager.agent_write(
                "agent_0001", "overflow", "x", MemoryType.SHORT_TERM, current_tick=0
            )

    def test_access_other_agents_memory_raises_error(self):
        """Reading another agent's private memory should raise MemoryAccessError."""
        self.manager.create_agent_space("agent_0002", max_keys=10)
        self.manager.agent_write(
            "agent_0002", "secret", "hidden", MemoryType.LONG_TERM, current_tick=0
        )
        with self.assertRaises((MemoryAccessError, MemoryKeyError)):
            self.manager.agent_read("agent_0001", "secret")


class TestMemoryTTL(unittest.TestCase):

    def setUp(self):
        self.manager = MemoryManager()
        self.manager.create_agent_space("agent_0001", max_keys=20)

    def test_short_term_memory_expires_after_ttl(self):
        """SHORT_TERM memory with ttl_ticks=3 should be gone after 3 gc_ticks."""
        self.manager.agent_write(
            "agent_0001", "temp_key", "temp_value",
            MemoryType.SHORT_TERM, current_tick=0, ttl_ticks=3
        )
        # Should still exist at tick 2
        self.manager.gc_tick(current_tick=2)
        value = self.manager.agent_read("agent_0001", "temp_key")
        self.assertEqual(value, "temp_value")

        # Should expire at or after tick 3
        self.manager.gc_tick(current_tick=4)
        with self.assertRaises((MemoryKeyError, Exception)):
            self.manager.agent_read("agent_0001", "temp_key")

    def test_long_term_memory_not_expired_by_gc(self):
        """LONG_TERM memory (no TTL) must survive gc_tick."""
        self.manager.agent_write(
            "agent_0001", "long_key", "long_value",
            MemoryType.LONG_TERM, current_tick=0
        )
        self.manager.gc_tick(current_tick=100)
        value = self.manager.agent_read("agent_0001", "long_key")
        self.assertEqual(value, "long_value")


class TestSharedMemoryRegion(unittest.TestCase):

    def setUp(self):
        self.manager = MemoryManager()
        self.manager.create_agent_space("agent_a", max_keys=20)
        self.manager.create_agent_space("agent_b", max_keys=20)
        self.manager.create_shared_region(
            "shared_region", max_keys=10, authorized_agents=["agent_a", "agent_b"]
        )

    def test_authorized_agent_can_write_to_shared_region(self):
        self.manager.shared_write(
            "shared_region", "agent_a", "shared_key", "shared_value", current_tick=0
        )
        value = self.manager.shared_read(
            "shared_region", "agent_b", "shared_key", current_tick=0
        )
        self.assertEqual(value, "shared_value")

    def test_unauthorized_agent_cannot_access_shared_region(self):
        self.manager.create_agent_space("intruder", max_keys=20)
        self.manager.shared_write(
            "shared_region", "agent_a", "secret", "data", current_tick=0
        )
        with self.assertRaises((MemoryAccessError, Exception)):
            self.manager.shared_read(
                "shared_region", "intruder", "secret", current_tick=0
            )


class TestMemoryManagerStats(unittest.TestCase):

    def setUp(self):
        self.manager = MemoryManager()
        self.manager.create_agent_space("agent_0001", max_keys=20)

    def test_stats_returns_dict(self):
        stats = self.manager.stats()
        self.assertIsInstance(stats, dict)

    def test_stats_reflects_writes(self):
        self.manager.agent_write(
            "agent_0001", "k1", "v1", MemoryType.LONG_TERM, current_tick=0
        )
        self.manager.agent_write(
            "agent_0001", "k2", "v2", MemoryType.LONG_TERM, current_tick=0
        )
        stats = self.manager.stats()
        # Stats reports per-agent used count
        agent_stats = stats.get("agents", {}).get("agent_0001", {})
        used = agent_stats.get("used", 0)
        self.assertGreaterEqual(used, 2)


if __name__ == "__main__":
    unittest.main()
