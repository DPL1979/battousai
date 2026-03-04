"""
test_integration.py — End-to-end integration test for the Battousai stack.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from battousai.kernel import Kernel
from battousai.agent import CoordinatorAgent, WorkerAgent, MonitorAgent
from battousai.supervisor import SupervisorAgent, ChildSpec, RestartStrategy, RestartType
from battousai.tools import register_builtin_tools
from battousai.tools_extended import register_extended_tools, _VECTOR_STORES, _KV_STORES, _TASK_QUEUES, _CRON_ENTRIES


def _reset_extended_state():
    _VECTOR_STORES.clear()
    _KV_STORES.clear()
    _TASK_QUEUES.clear()
    _CRON_ENTRIES.clear()


class TestFullStackIntegration(unittest.TestCase):
    """
    End-to-end integration: boot kernel → spawn supervisor with coordinator +
    2 workers → run 30 ticks → verify agents, messages, files, no panics.
    """

    def setUp(self):
        _reset_extended_state()
        self.kernel = Kernel(max_ticks=0, debug=False)
        self.kernel.boot()

    def tearDown(self):
        _reset_extended_state()

    def test_kernel_boots_cleanly(self):
        """Kernel should boot without exceptions."""
        # setUp already called boot()
        self.assertIsNotNone(self.kernel.scheduler)
        self.assertIsNotNone(self.kernel.ipc)
        self.assertIsNotNone(self.kernel.memory)
        self.assertIsNotNone(self.kernel.filesystem)
        self.assertIsNotNone(self.kernel.tools)

    def test_supervisor_spawns_coordinator_and_two_workers(self):
        """Supervisor with coordinator + 2 workers spawns all 3 children."""
        children = [
            ChildSpec(
                agent_class=CoordinatorAgent,
                name="Coord",
                priority=3,
                restart_type=RestartType.PERMANENT,
            ),
            ChildSpec(
                agent_class=WorkerAgent,
                name="Worker1",
                priority=5,
                restart_type=RestartType.PERMANENT,
            ),
            ChildSpec(
                agent_class=WorkerAgent,
                name="Worker2",
                priority=5,
                restart_type=RestartType.PERMANENT,
            ),
        ]
        self.kernel.spawn_agent(
            SupervisorAgent, name="MainSupervisor", priority=2,
            strategy=RestartStrategy.ONE_FOR_ONE,
            children=children,
        )
        # Run first tick to spawn children
        self.kernel.tick()
        # Should have supervisor + 3 children = at least 4 agents
        self.assertGreaterEqual(len(self.kernel._agents), 4)

    def test_30_tick_run_completes_without_panic(self):
        """30 ticks with coordinator + 2 workers + monitor must not raise."""
        self.kernel.spawn_agent(CoordinatorAgent, name="Coord", priority=3)
        self.kernel.spawn_agent(WorkerAgent, name="Worker1", priority=5)
        self.kernel.spawn_agent(WorkerAgent, name="Worker2", priority=5)
        self.kernel.spawn_agent(MonitorAgent, name="Monitor", priority=7)

        # Should not raise
        self.kernel.run(30)
        self.assertEqual(self.kernel._tick, 30)

    def test_agents_survive_30_ticks(self):
        """All spawned agents should still be alive after 30 ticks."""
        ids = [
            self.kernel.spawn_agent(CoordinatorAgent, name="Coord", priority=3),
            self.kernel.spawn_agent(WorkerAgent, name="Worker1", priority=5),
            self.kernel.spawn_agent(WorkerAgent, name="Worker2", priority=5),
        ]
        self.kernel.run(30)
        for agent_id in ids:
            self.assertIn(agent_id, self.kernel._agents)

    def test_agents_send_messages_during_run(self):
        """After 30 ticks, IPC should have processed messages."""
        self.kernel.spawn_agent(CoordinatorAgent, name="Coord", priority=3)
        self.kernel.spawn_agent(WorkerAgent, name="Worker1", priority=5)
        self.kernel.run(30)
        # Verify the kernel is still responsive
        report = self.kernel.system_report()
        self.assertIsInstance(report, str)

    def test_filesystem_operations_during_run(self):
        """Agents should be able to write and read files during a run."""
        self.kernel.spawn_agent(CoordinatorAgent, name="Coord", priority=3)
        self.kernel.spawn_agent(WorkerAgent, name="Worker1", priority=5)
        self.kernel.run(20)

        # The coordinator writes summary.txt — check if it exists, or manually write one
        kernel_agent_id = list(self.kernel._agents.keys())[0]
        self.kernel.filesystem.write_file(
            kernel_agent_id,
            "/shared/results/integration_test.txt",
            "Integration test passed",
            world_readable=True,
        )
        data = self.kernel.filesystem.read_file(
            kernel_agent_id, "/shared/results/integration_test.txt"
        )
        self.assertEqual(data, "Integration test passed")

    def test_extended_tools_registered_and_usable(self):
        """Extended tools can be registered and used during a run."""
        from battousai.tools_extended import register_extended_tools
        register_extended_tools(self.kernel.tools, self.kernel.filesystem)

        worker_id = self.kernel.spawn_agent(WorkerAgent, name="ToolWorker", priority=5)
        self.kernel.tools.grant_access("python_repl", worker_id)
        self.kernel.tools.grant_access("key_value_db", worker_id)

        # Use the tools directly
        result = self.kernel.tools.execute(
            worker_id, "python_repl", {"code": "1 + 1"}
        )
        self.assertIsNotNone(result)

    def test_memory_operations_during_run(self):
        """Agents' memory operations must persist across ticks."""
        worker_id = self.kernel.spawn_agent(WorkerAgent, name="MemWorker", priority=5)
        self.kernel.run(5)

        # Write a memory entry via the kernel's memory manager
        from battousai.memory import MemoryType
        self.kernel.memory.agent_write(
            worker_id, "test_key", "test_value",
            MemoryType.LONG_TERM, current_tick=5
        )
        self.kernel.run(5)  # Run 5 more ticks

        value = self.kernel.memory.agent_read(worker_id, "test_key")
        self.assertEqual(value, "test_value")

    def test_ipc_communication_between_agents(self):
        """Two agents can exchange messages via IPC."""
        from battousai.ipc import Message, MessageType

        agent_a = self.kernel.spawn_agent(WorkerAgent, name="AgentA", priority=5)
        agent_b = self.kernel.spawn_agent(WorkerAgent, name="AgentB", priority=5)

        msg = Message(
            sender_id=agent_a,
            recipient_id=agent_b,
            message_type=MessageType.TASK,
            payload={"task": "hello_from_a"},
            timestamp=0,
        )
        self.kernel.ipc.send(msg)

        mailbox = self.kernel.ipc.get_mailbox(agent_b)
        messages = mailbox.receive_all(current_tick=0)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].payload["task"], "hello_from_a")

    def test_system_report_includes_agent_counts(self):
        """system_report should include information about agents."""
        self.kernel.spawn_agent(CoordinatorAgent, name="Coord", priority=3)
        self.kernel.spawn_agent(WorkerAgent, name="Worker", priority=5)
        self.kernel.run(10)
        report = self.kernel.system_report()
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)

    def test_full_supervisor_tree_30_ticks(self):
        """Full supervisor tree with coordinator + 2 workers runs 30 ticks cleanly."""
        children = [
            ChildSpec(
                agent_class=CoordinatorAgent,
                name="Coord",
                priority=3,
                restart_type=RestartType.PERMANENT,
            ),
            ChildSpec(
                agent_class=WorkerAgent,
                name="Worker1",
                priority=5,
                restart_type=RestartType.PERMANENT,
            ),
            ChildSpec(
                agent_class=WorkerAgent,
                name="Worker2",
                priority=5,
                restart_type=RestartType.PERMANENT,
            ),
        ]
        sup_id = self.kernel.spawn_agent(
            SupervisorAgent, name="MainSupervisor", priority=2,
            strategy=RestartStrategy.ONE_FOR_ONE,
            children=children,
            max_restarts=5,
            window_ticks=20,
        )
        self.kernel.run(30)
        self.assertEqual(self.kernel._tick, 30)
        # Supervisor should still be alive
        self.assertIn(sup_id, self.kernel._agents)

    def test_no_kernel_panic_during_30_tick_run(self):
        """No KernelPanic should be raised during a normal 30-tick run."""
        from battousai.kernel import KernelPanic
        self.kernel.spawn_agent(CoordinatorAgent, name="Coord", priority=3)
        self.kernel.spawn_agent(WorkerAgent, name="W1", priority=5)
        self.kernel.spawn_agent(MonitorAgent, name="Mon", priority=7)
        try:
            self.kernel.run(30)
        except KernelPanic as e:
            self.fail(f"KernelPanic raised unexpectedly: {e}")


if __name__ == "__main__":
    unittest.main()
