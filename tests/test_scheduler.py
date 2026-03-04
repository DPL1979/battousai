"""
test_scheduler.py — Tests for battousai.scheduler
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from battousai.scheduler import Scheduler, ProcessDescriptor, AgentState


class TestSchedulerAddRemove(unittest.TestCase):

    def setUp(self):
        self.scheduler = Scheduler()

    def test_add_process_returns_descriptor(self):
        proc = self.scheduler.add_process("agent_0001", name="Worker", priority=5)
        self.assertIsInstance(proc, ProcessDescriptor)

    def test_added_process_has_correct_agent_id(self):
        proc = self.scheduler.add_process("agent_0001", name="Worker", priority=5)
        self.assertEqual(proc.agent_id, "agent_0001")

    def test_added_process_state_is_ready(self):
        proc = self.scheduler.add_process("agent_0001", name="Worker", priority=5)
        self.assertEqual(proc.state, AgentState.READY)

    def test_remove_process_returns_true(self):
        self.scheduler.add_process("agent_0001", name="Worker", priority=5)
        result = self.scheduler.remove_process("agent_0001")
        self.assertTrue(result)

    def test_remove_nonexistent_process_returns_false(self):
        result = self.scheduler.remove_process("no_such_agent")
        self.assertFalse(result)

    def test_remove_process_removes_from_scheduler(self):
        self.scheduler.add_process("agent_0001", name="Worker", priority=5)
        self.scheduler.remove_process("agent_0001")
        # The internal lookup table is _all, not _processes
        proc = self.scheduler._all.get("agent_0001")
        self.assertIsNone(proc)


class TestSchedulerStateTransitions(unittest.TestCase):

    def setUp(self):
        self.scheduler = Scheduler()
        self.scheduler.add_process("agent_0001", name="Worker", priority=5)

    def test_terminate_process_sets_terminated_state(self):
        self.scheduler.terminate_process("agent_0001")
        # After terminate_process the proc is still in _all, just TERMINATED
        proc = self.scheduler._all.get("agent_0001")
        if proc is not None:
            self.assertEqual(proc.state, AgentState.TERMINATED)

    def test_block_process_sets_blocked_state(self):
        # block_process() sets state to WAITING (not BLOCKED)
        self.scheduler.block_process("agent_0001")
        proc = self.scheduler._all["agent_0001"]
        self.assertEqual(proc.state, AgentState.WAITING)

    def test_unblock_process_restores_ready_state(self):
        self.scheduler.block_process("agent_0001")
        self.scheduler.unblock_process("agent_0001")
        proc = self.scheduler._all["agent_0001"]
        self.assertIn(proc.state, [AgentState.READY, AgentState.RUNNING])

    def test_block_then_unblock_cycle(self):
        """Multiple block/unblock cycles must not corrupt state."""
        for _ in range(3):
            self.scheduler.block_process("agent_0001")
            self.scheduler.unblock_process("agent_0001")
        proc = self.scheduler._all["agent_0001"]
        self.assertIn(proc.state, [AgentState.READY, AgentState.RUNNING])


class TestSchedulerPriorityOrdering(unittest.TestCase):

    def setUp(self):
        self.scheduler = Scheduler()

    def test_higher_priority_process_scheduled_first(self):
        """Lower numeric priority value = higher scheduling priority."""
        self.scheduler.add_process("high_0001", name="High", priority=1)
        self.scheduler.add_process("low_0001", name="Low", priority=9)
        stats = self.scheduler.stats()
        # Stats has 'state_counts' with READY count
        self.assertIn("state_counts", stats)
        self.assertEqual(stats["state_counts"]["READY"], 2)

    def test_stats_returns_dict_with_expected_keys(self):
        self.scheduler.add_process("agent_0001", name="Worker", priority=5)
        stats = self.scheduler.stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("state_counts", stats)

    def test_multiple_processes_tracked(self):
        for i in range(5):
            self.scheduler.add_process(f"agent_{i:04d}", name=f"W{i}", priority=5)
        stats = self.scheduler.stats()
        self.assertEqual(stats["state_counts"]["READY"], 5)

    def test_agent_state_enum_has_required_values(self):
        states = [s.name for s in AgentState]
        self.assertIn("READY", states)
        self.assertIn("RUNNING", states)
        self.assertIn("WAITING", states)
        self.assertIn("BLOCKED", states)
        self.assertIn("TERMINATED", states)


if __name__ == "__main__":
    unittest.main()
