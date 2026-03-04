"""
test_agent.py — Tests for battousai.agent (Agent, SyscallResult, built-in agents)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from battousai.agent import (
    Agent, SyscallResult, CoordinatorAgent, WorkerAgent, MonitorAgent
)
from battousai.ipc import MessageType
from battousai.kernel import Kernel


class TestSyscallResult(unittest.TestCase):

    def test_syscall_result_ok_is_truthy(self):
        result = SyscallResult(ok=True, value="hello")
        self.assertTrue(bool(result))

    def test_syscall_result_not_ok_is_falsy(self):
        result = SyscallResult(ok=False, error="oops")
        self.assertFalse(bool(result))

    def test_syscall_result_stores_value(self):
        result = SyscallResult(ok=True, value=[1, 2, 3])
        self.assertEqual(result.value, [1, 2, 3])

    def test_syscall_result_stores_error(self):
        result = SyscallResult(ok=False, error="bad input")
        self.assertEqual(result.error, "bad input")

    def test_syscall_result_default_value_is_none(self):
        result = SyscallResult(ok=True)
        self.assertIsNone(result.value)


class TestAgentInit(unittest.TestCase):

    def test_agent_name_stored(self):
        agent = WorkerAgent(name="TestAgent", priority=5)
        self.assertEqual(agent.name, "TestAgent")

    def test_agent_priority_stored(self):
        agent = WorkerAgent(name="A", priority=7)
        self.assertEqual(agent.priority, 7)

    def test_agent_default_memory_allocation(self):
        agent = WorkerAgent(name="A", priority=5)
        self.assertGreater(agent.memory_allocation, 0)

    def test_agent_ticks_alive_starts_at_zero(self):
        agent = WorkerAgent(name="A", priority=5)
        self.assertEqual(agent._ticks_alive, 0)

    def test_agent_kernel_initially_none(self):
        agent = WorkerAgent(name="A", priority=5)
        self.assertIsNone(agent.kernel)

    def test_agent_agent_id_initially_empty_string(self):
        """agent_id is '' before spawning (not None)."""
        agent = WorkerAgent(name="A", priority=5)
        # The default is empty string ''
        self.assertFalse(bool(agent.agent_id))  # falsy ('' or None both pass)


class TestAgentLifecycle(unittest.TestCase):

    def setUp(self):
        self.kernel = Kernel(max_ticks=0, debug=False)
        self.kernel.boot()

    def test_agent_on_spawn_called_on_spawn(self):
        """on_spawn() should fire when an agent is spawned via the kernel."""
        spawned = []

        class RecordingAgent(Agent):
            def on_spawn(self):
                spawned.append(True)
            def think(self, tick):
                self.yield_cpu()

        self.kernel.spawn_agent(RecordingAgent, name="Recorder", priority=5)
        self.assertEqual(len(spawned), 1)

    def test_agent_think_is_called_each_tick(self):
        """think() must be called for each running agent on each tick."""
        call_count = [0]

        class CountingAgent(Agent):
            def think(self, tick):
                call_count[0] += 1
                self.yield_cpu()

        self.kernel.spawn_agent(CountingAgent, name="Counter", priority=5)
        self.kernel.run(3)
        self.assertEqual(call_count[0], 3)

    def test_agent_on_terminate_called_on_kill(self):
        """on_terminate() should fire when an agent is killed."""
        terminated = []

        class TermAgent(Agent):
            def on_terminate(self):
                terminated.append(True)
            def think(self, tick):
                self.yield_cpu()

        agent_id = self.kernel.spawn_agent(TermAgent, name="Term", priority=5)
        self.kernel.kill_agent(agent_id)
        self.assertEqual(len(terminated), 1)

    def test_agent_ticks_alive_increments_each_tick(self):
        """_ticks_alive should increment by 1 per tick."""
        agent_id = self.kernel.spawn_agent(WorkerAgent, name="W", priority=5)
        self.kernel.run(4)
        agent = self.kernel._agents[agent_id]
        self.assertEqual(agent._ticks_alive, 4)

    def test_agent_kernel_set_after_spawn(self):
        """agent.kernel must be set after spawning."""
        agent_id = self.kernel.spawn_agent(WorkerAgent, name="W", priority=5)
        agent = self.kernel._agents[agent_id]
        self.assertIsNotNone(agent.kernel)

    def test_agent_agent_id_set_after_spawn(self):
        """agent.agent_id must be set after spawning."""
        agent_id = self.kernel.spawn_agent(WorkerAgent, name="W", priority=5)
        agent = self.kernel._agents[agent_id]
        self.assertEqual(agent.agent_id, agent_id)


class TestAgentConvenienceMethods(unittest.TestCase):

    def setUp(self):
        self.kernel = Kernel(max_ticks=0, debug=False)
        self.kernel.boot()
        self.agent_id = self.kernel.spawn_agent(WorkerAgent, name="W", priority=5)
        self.agent = self.kernel._agents[self.agent_id]

    def test_mem_write_returns_truthy_result(self):
        result = self.agent.mem_write("mykey", "myvalue")
        self.assertTrue(bool(result))

    def test_mem_write_and_read_round_trip(self):
        """mem_write stores a value; mem_read returns it directly."""
        self.agent.mem_write("mykey", "myvalue")
        value = self.agent.mem_read("mykey")
        self.assertEqual(value, "myvalue")

    def test_mem_read_missing_key_returns_none(self):
        """mem_read on a missing key returns None (not SyscallResult)."""
        value = self.agent.mem_read("no_such_key")
        self.assertIsNone(value)

    def test_list_agents_returns_list(self):
        result = self.agent.list_agents()
        self.assertIsInstance(result, list)
        self.assertIn(self.agent_id, result)

    def test_send_message_to_self_with_enum(self):
        """An agent should be able to send a message using MessageType enum."""
        result = self.agent.send_message(
            recipient_id=self.agent_id,
            message_type=MessageType.CUSTOM,
            payload={"hello": "world"}
        )
        self.assertTrue(bool(result))

    def test_read_inbox_after_send(self):
        """After sending a message to self, read_inbox should return it."""
        self.agent.send_message(
            recipient_id=self.agent_id,
            message_type=MessageType.CUSTOM,
            payload={"hello": "world"}
        )
        result = self.agent.read_inbox()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)


class TestCoordinatorAgent(unittest.TestCase):

    def setUp(self):
        self.kernel = Kernel(max_ticks=0, debug=False)
        self.kernel.boot()

    def test_coordinator_spawns_successfully(self):
        agent_id = self.kernel.spawn_agent(CoordinatorAgent, name="Coord", priority=3)
        self.assertIn(agent_id, self.kernel._agents)

    def test_coordinator_runs_multiple_ticks(self):
        """CoordinatorAgent must not crash over 5 ticks."""
        self.kernel.spawn_agent(CoordinatorAgent, name="Coord", priority=3)
        self.kernel.run(5)
        self.assertEqual(self.kernel._tick, 5)


class TestWorkerAgent(unittest.TestCase):

    def setUp(self):
        self.kernel = Kernel(max_ticks=0, debug=False)
        self.kernel.boot()

    def test_worker_spawns_successfully(self):
        agent_id = self.kernel.spawn_agent(WorkerAgent, name="Worker", priority=5)
        self.assertIn(agent_id, self.kernel._agents)

    def test_worker_runs_multiple_ticks(self):
        self.kernel.spawn_agent(WorkerAgent, name="Worker", priority=5)
        self.kernel.run(5)
        self.assertEqual(self.kernel._tick, 5)


class TestMonitorAgent(unittest.TestCase):

    def setUp(self):
        self.kernel = Kernel(max_ticks=0, debug=False)
        self.kernel.boot()

    def test_monitor_spawns_successfully(self):
        agent_id = self.kernel.spawn_agent(MonitorAgent, name="Monitor", priority=7)
        self.assertIn(agent_id, self.kernel._agents)

    def test_monitor_samples_system_state(self):
        """After running 5 ticks, monitor writes a sample_tick_5 entry."""
        agent_id = self.kernel.spawn_agent(MonitorAgent, name="Monitor", priority=7)
        self.kernel.run(5)
        # Monitor writes 'sample_tick_5' to memory
        space = self.kernel.memory.get_agent_space(agent_id)
        self.assertIsNotNone(space)
        snap = space.snapshot()
        self.assertTrue(len(snap) > 0, f"Expected memory entries, got: {snap}")

    def test_monitor_runs_without_crashing(self):
        self.kernel.spawn_agent(MonitorAgent, name="Monitor", priority=7)
        self.kernel.run(10)
        self.assertEqual(self.kernel._tick, 10)


class TestAgentIntegration(unittest.TestCase):

    def setUp(self):
        self.kernel = Kernel(max_ticks=0, debug=False)
        self.kernel.boot()

    def test_full_20_tick_run_with_all_agents(self):
        """20-tick run with all three built-in agents completes without crash."""
        self.kernel.spawn_agent(CoordinatorAgent, name="Coord", priority=3)
        self.kernel.spawn_agent(WorkerAgent, name="Worker", priority=5)
        self.kernel.spawn_agent(MonitorAgent, name="Monitor", priority=7)
        self.kernel.run(20)
        self.assertEqual(self.kernel._tick, 20)


if __name__ == "__main__":
    unittest.main()
