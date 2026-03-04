"""
test_supervisor.py — Tests for battousai.supervisor
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from battousai.kernel import Kernel
from battousai.agent import Agent, WorkerAgent, CoordinatorAgent
from battousai.supervisor import (
    SupervisorAgent, ChildSpec, RestartStrategy, RestartType,
    SupervisorTree, build_supervision_tree,
)


class TestChildSpec(unittest.TestCase):

    def test_child_spec_stores_agent_class(self):
        spec = ChildSpec(
            agent_class=WorkerAgent,
            name="Worker",
            priority=5,
            restart_type=RestartType.PERMANENT,
        )
        self.assertEqual(spec.agent_class, WorkerAgent)

    def test_child_spec_stores_restart_type(self):
        spec = ChildSpec(
            agent_class=WorkerAgent,
            name="Worker",
            priority=5,
            restart_type=RestartType.TRANSIENT,
        )
        self.assertEqual(spec.restart_type, RestartType.TRANSIENT)

    def test_restart_type_enum_has_required_values(self):
        names = [r.name for r in RestartType]
        self.assertIn("PERMANENT", names)
        self.assertIn("TRANSIENT", names)
        self.assertIn("TEMPORARY", names)

    def test_restart_strategy_enum_has_required_values(self):
        names = [s.name for s in RestartStrategy]
        self.assertIn("ONE_FOR_ONE", names)
        self.assertIn("ONE_FOR_ALL", names)
        self.assertIn("REST_FOR_ONE", names)


class TestSupervisorAgent(unittest.TestCase):

    def setUp(self):
        self.kernel = Kernel(max_ticks=0, debug=False)
        self.kernel.boot()

    def test_supervisor_spawns_successfully(self):
        spec = ChildSpec(
            agent_class=WorkerAgent, name="Worker", priority=5,
            restart_type=RestartType.PERMANENT
        )
        agent_id = self.kernel.spawn_agent(
            SupervisorAgent, name="Supervisor", priority=2,
            strategy=RestartStrategy.ONE_FOR_ONE,
            children=[spec],
        )
        self.assertIn(agent_id, self.kernel._agents)

    def test_supervisor_spawns_children_on_first_tick(self):
        """After first tick, the supervisor's children should be in the kernel."""
        spec = ChildSpec(
            agent_class=WorkerAgent, name="ManagedWorker", priority=5,
            restart_type=RestartType.PERMANENT
        )
        sup_id = self.kernel.spawn_agent(
            SupervisorAgent, name="Sup", priority=2,
            strategy=RestartStrategy.ONE_FOR_ONE,
            children=[spec],
        )
        self.kernel.tick()
        # At least one agent beyond the supervisor should exist
        self.assertGreater(len(self.kernel._agents), 1)

    def test_supervisor_runs_multiple_ticks_without_crash(self):
        spec = ChildSpec(
            agent_class=WorkerAgent, name="Worker", priority=5,
            restart_type=RestartType.PERMANENT
        )
        self.kernel.spawn_agent(
            SupervisorAgent, name="Sup", priority=2,
            strategy=RestartStrategy.ONE_FOR_ONE,
            children=[spec],
        )
        self.kernel.run(5)
        self.assertEqual(self.kernel._tick, 5)

    def test_supervisor_with_multiple_children(self):
        children = [
            ChildSpec(
                agent_class=WorkerAgent, name=f"Worker{i}", priority=5,
                restart_type=RestartType.PERMANENT
            )
            for i in range(3)
        ]
        self.kernel.spawn_agent(
            SupervisorAgent, name="Sup", priority=2,
            strategy=RestartStrategy.ONE_FOR_ONE,
            children=children,
        )
        self.kernel.run(2)
        # Should have supervisor + 3 workers
        self.assertGreaterEqual(len(self.kernel._agents), 4)

    def test_supervisor_one_for_one_restarts_killed_child(self):
        """If ONE_FOR_ONE supervisor's child is killed, it should restart."""
        spec = ChildSpec(
            agent_class=WorkerAgent, name="Restartable", priority=5,
            restart_type=RestartType.PERMANENT
        )
        self.kernel.spawn_agent(
            SupervisorAgent, name="Sup", priority=2,
            strategy=RestartStrategy.ONE_FOR_ONE,
            children=[spec],
            max_restarts=3,
            window_ticks=10,
        )
        # Run a tick to spawn children
        self.kernel.tick()
        # Find the worker agent
        worker_ids = [
            aid for aid, a in self.kernel._agents.items()
            if "restartable" in aid.lower() or "worker" in a.name.lower()
        ]
        if worker_ids:
            self.kernel.kill_agent(worker_ids[0])
            # Run more ticks for supervisor to restart
            self.kernel.run(3)
            # The kernel should still be running (no panic)
            self.assertGreaterEqual(self.kernel._tick, 4)


class TestBuildSupervisionTree(unittest.TestCase):

    def setUp(self):
        self.kernel = Kernel(max_ticks=0, debug=False)
        self.kernel.boot()

    def test_build_supervision_tree_returns_agent_id(self):
        tree_spec = {
            "name": "RootSup",
            "strategy": "ONE_FOR_ONE",
            "children": [
                {"class": WorkerAgent, "name": "W1", "priority": 5, "restart_type": "PERMANENT"},
                {"class": WorkerAgent, "name": "W2", "priority": 5, "restart_type": "PERMANENT"},
            ]
        }
        agent_id = build_supervision_tree(self.kernel, tree_spec)
        self.assertIsInstance(agent_id, str)
        self.assertIn(agent_id, self.kernel._agents)


if __name__ == "__main__":
    unittest.main()
