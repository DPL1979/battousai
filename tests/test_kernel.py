"""
test_kernel.py — Tests for battousai.kernel.Kernel
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from battousai.kernel import Kernel, KernelPanic
from battousai.agent import Agent, CoordinatorAgent, WorkerAgent, MonitorAgent


class TestKernelBoot(unittest.TestCase):

    def setUp(self):
        self.kernel = Kernel(max_ticks=0, debug=False)

    def tearDown(self):
        pass

    def test_kernel_boot_initializes_all_subsystems(self):
        """Kernel.boot() must initialise scheduler, ipc, memory, fs, tools, logger."""
        self.kernel.boot()
        self.assertIsNotNone(self.kernel.scheduler)
        self.assertIsNotNone(self.kernel.ipc)
        self.assertIsNotNone(self.kernel.memory)
        self.assertIsNotNone(self.kernel.filesystem)
        self.assertIsNotNone(self.kernel.tools)
        self.assertIsNotNone(self.kernel.logger)

    def test_kernel_boot_creates_standard_filesystem_dirs(self):
        """After boot(), standard dirs /agents, /shared, /system/logs must exist."""
        self.kernel.boot()
        listing = self.kernel.filesystem.list_dir("kernel", "/")
        self.assertIn("agents", listing)
        self.assertIn("shared", listing)
        self.assertIn("system", listing)

    def test_kernel_double_boot_raises_kernel_panic(self):
        """Calling boot() twice on the same kernel should raise an exception."""
        self.kernel.boot()
        with self.assertRaises(Exception):
            self.kernel.boot()

    def test_kernel_spawn_agent_before_boot_raises_kernel_panic(self):
        """spawn_agent() without prior boot() must raise KernelPanic."""
        with self.assertRaises(KernelPanic):
            self.kernel.spawn_agent(CoordinatorAgent, name="Coord", priority=5)

    def test_kernel_spawn_agent_returns_string_id(self):
        """spawn_agent() must return a non-empty string agent_id."""
        self.kernel.boot()
        agent_id = self.kernel.spawn_agent(CoordinatorAgent, name="Coord", priority=5)
        self.assertIsInstance(agent_id, str)
        self.assertTrue(len(agent_id) > 0)

    def test_kernel_spawn_agent_id_format(self):
        """Agent IDs must follow the name_NNNN pattern."""
        self.kernel.boot()
        agent_id = self.kernel.spawn_agent(WorkerAgent, name="Worker", priority=5)
        self.assertIn("_", agent_id)
        parts = agent_id.rsplit("_", 1)
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[1].isdigit())

    def test_kernel_kill_agent_removes_agent(self):
        """kill_agent() must remove the agent from the kernel."""
        self.kernel.boot()
        agent_id = self.kernel.spawn_agent(WorkerAgent, name="Worker", priority=5)
        self.assertIn(agent_id, self.kernel._agents)
        result = self.kernel.kill_agent(agent_id)
        self.assertTrue(result)
        self.assertNotIn(agent_id, self.kernel._agents)

    def test_kernel_kill_nonexistent_agent_returns_false(self):
        """kill_agent() on unknown ID returns False without raising."""
        self.kernel.boot()
        result = self.kernel.kill_agent("nonexistent_9999")
        self.assertFalse(result)

    def test_kernel_tick_increments_tick_counter(self):
        """Each tick() call must advance the internal tick counter by 1."""
        self.kernel.boot()
        initial = self.kernel._tick
        self.kernel.tick()
        self.assertEqual(self.kernel._tick, initial + 1)

    def test_kernel_run_executes_correct_number_of_ticks(self):
        """run(n) must execute exactly n ticks."""
        self.kernel.boot()
        self.kernel.run(5)
        self.assertEqual(self.kernel._tick, 5)

    def test_kernel_system_report_returns_string(self):
        """system_report() must return a non-empty string."""
        self.kernel.boot()
        report = self.kernel.system_report()
        self.assertIsInstance(report, str)
        self.assertTrue(len(report) > 0)

    def test_kernel_dispatch_valid_syscall_returns_syscall_result(self):
        """_dispatch_syscall() for a valid syscall returns a SyscallResult."""
        from battousai.agent import SyscallResult
        self.kernel.boot()
        agent_id = self.kernel.spawn_agent(WorkerAgent, name="W", priority=5)
        result = self.kernel._dispatch_syscall(agent_id, "get_status")
        self.assertIsInstance(result, SyscallResult)

    def test_kernel_dispatch_unknown_syscall_returns_failure(self):
        """_dispatch_syscall() with an unknown syscall name returns ok=False."""
        from battousai.agent import SyscallResult
        self.kernel.boot()
        agent_id = self.kernel.spawn_agent(WorkerAgent, name="W", priority=5)
        result = self.kernel._dispatch_syscall(agent_id, "not_a_real_syscall")
        self.assertIsInstance(result, SyscallResult)
        self.assertFalse(result.ok)

    def test_kernel_multiple_agents_all_tracked(self):
        """Spawning multiple agents registers all of them."""
        self.kernel.boot()
        ids = [
            self.kernel.spawn_agent(WorkerAgent, name="W1", priority=5),
            self.kernel.spawn_agent(WorkerAgent, name="W2", priority=5),
            self.kernel.spawn_agent(CoordinatorAgent, name="Coord", priority=3),
        ]
        for agent_id in ids:
            self.assertIn(agent_id, self.kernel._agents)


class TestKernelSyscalls(unittest.TestCase):

    def setUp(self):
        self.kernel = Kernel(max_ticks=0, debug=False)
        self.kernel.boot()
        self.agent_id = self.kernel.spawn_agent(WorkerAgent, name="TestWorker", priority=5)

    def test_syscall_list_agents_returns_agent_list(self):
        """list_agents syscall returns a list containing the spawned agent."""
        from battousai.agent import SyscallResult
        result = self.kernel._dispatch_syscall(self.agent_id, "list_agents")
        self.assertTrue(result.ok)
        self.assertIsInstance(result.value, list)
        self.assertIn(self.agent_id, result.value)

    def test_syscall_yield_cpu_succeeds(self):
        """yield_cpu syscall must succeed."""
        result = self.kernel._dispatch_syscall(self.agent_id, "yield_cpu")
        self.assertTrue(result.ok)

    def test_syscall_write_then_read_memory(self):
        """write_memory followed by read_memory must return the same value."""
        self.kernel._dispatch_syscall(
            self.agent_id, "write_memory", key="testkey", value=42
        )
        result = self.kernel._dispatch_syscall(
            self.agent_id, "read_memory", key="testkey"
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.value, 42)


if __name__ == "__main__":
    unittest.main()
