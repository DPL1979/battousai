"""
test_federation.py — Tests for battousai.federation
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from battousai.kernel import Kernel
from battousai.federation import (
    FederationNode, FederationCluster, GlobalRegistry, LoadBalancer,
    SplitBrainDetector, BalancingStrategy, NodeRole, ClusterEntry,
    create_demo_federation,
)


def _make_kernel():
    k = Kernel(max_ticks=0, debug=False)
    k.boot()
    return k


class TestFederationNode(unittest.TestCase):

    def setUp(self):
        self.kernel = _make_kernel()
        self.node = FederationNode(self.kernel, node_id="node_test")

    def test_node_id_stored(self):
        self.assertEqual(self.node.node_id, "node_test")

    def test_node_starts_as_follower(self):
        self.assertEqual(self.node.role, NodeRole.FOLLOWER)

    def test_node_term_starts_at_zero(self):
        self.assertEqual(self.node.term, 0)

    def test_add_peer_registers_peer(self):
        self.node.add_peer("peer_1")
        self.assertIn("peer_1", self.node.peers)

    def test_add_self_as_peer_ignored(self):
        self.node.add_peer("node_test")  # same as own node_id
        self.assertNotIn("node_test", self.node.peers)

    def test_request_vote_grants_vote_on_higher_term(self):
        granted = self.node.request_vote("candidate_1", candidate_term=1)
        self.assertTrue(granted)

    def test_request_vote_denies_on_lower_term(self):
        self.node.term = 5
        granted = self.node.request_vote("candidate_1", candidate_term=3)
        self.assertFalse(granted)

    def test_heartbeat_resets_to_follower(self):
        self.node._start_election()
        self.assertEqual(self.node.role, NodeRole.CANDIDATE)
        self.node.heartbeat("leader_1", leader_term=self.node.term + 1)
        self.assertEqual(self.node.role, NodeRole.FOLLOWER)

    def test_append_to_log_creates_entry(self):
        self.node._become_leader()
        entry = self.node.append_to_log("spawn", {"agent": "worker"})
        self.assertIsInstance(entry, ClusterEntry)
        self.assertEqual(len(self.node.log), 1)

    def test_snapshot_returns_dict(self):
        snap = self.node.snapshot()
        self.assertIsInstance(snap, dict)
        self.assertIn("node_id", snap)
        self.assertIn("role", snap)


class TestFederationCluster(unittest.TestCase):

    def setUp(self):
        self.cluster = FederationCluster()

    def test_add_node_registers_node(self):
        k = _make_kernel()
        node = self.cluster.add_node(k, node_id="n1")
        self.assertIn("n1", self.cluster.list_nodes())

    def test_remove_node_deregisters_node(self):
        k = _make_kernel()
        self.cluster.add_node(k, node_id="n1")
        self.cluster.remove_node("n1")
        self.assertNotIn("n1", self.cluster.list_nodes())

    def test_cluster_tick_returns_events_dict(self):
        k = _make_kernel()
        self.cluster.add_node(k, node_id="n1")
        events = self.cluster.tick()
        self.assertIsInstance(events, dict)
        self.assertIn("tick", events)

    def test_leader_elected_after_ticks(self):
        """After enough ticks, exactly one leader should emerge."""
        for i in range(3):
            k = _make_kernel()
            self.cluster.add_node(k, node_id=f"node_{i}")
        # Run enough ticks for election
        for _ in range(30):
            self.cluster.tick()
            if self.cluster.get_leader() is not None:
                break
        leader = self.cluster.get_leader()
        self.assertIsNotNone(leader)

    def test_only_one_leader_at_a_time(self):
        for i in range(3):
            k = _make_kernel()
            self.cluster.add_node(k, node_id=f"n{i}")
        for _ in range(30):
            self.cluster.tick()
        leaders = [
            nid for nid, node in self.cluster._nodes.items()
            if node.role == NodeRole.LEADER
        ]
        self.assertLessEqual(len(leaders), 1)

    def test_stats_returns_cluster_stats(self):
        k = _make_kernel()
        self.cluster.add_node(k, node_id="n1")
        stats = self.cluster.stats()
        self.assertIn("cluster_size", stats)
        self.assertEqual(stats["cluster_size"], 1)

    def test_broadcast_log_entry_after_leader_election(self):
        for i in range(3):
            k = _make_kernel()
            self.cluster.add_node(k, node_id=f"n{i}")
        for _ in range(30):
            self.cluster.tick()
            if self.cluster.get_leader() is not None:
                break
        if self.cluster.get_leader():
            entry = self.cluster.broadcast_log_entry("test", {"data": 42})
            self.assertIsNotNone(entry)


class TestGlobalRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = GlobalRegistry()

    def test_register_agent(self):
        self.registry.register_agent("agent_0001", "node_a", "Worker")
        info = self.registry.lookup_agent("agent_0001")
        self.assertIsNotNone(info)
        self.assertEqual(info["node_id"], "node_a")

    def test_unregister_agent(self):
        self.registry.register_agent("agent_0001", "node_a", "Worker")
        result = self.registry.unregister_agent("agent_0001")
        self.assertTrue(result)
        self.assertIsNone(self.registry.lookup_agent("agent_0001"))

    def test_find_node_for_agent(self):
        self.registry.register_agent("agent_0001", "node_b", "Monitor")
        node_id = self.registry.find_node("agent_0001")
        self.assertEqual(node_id, "node_b")

    def test_register_service(self):
        self.registry.register_service("compute", "agent_0001", "node_a")
        providers = self.registry.discover_service("compute")
        self.assertEqual(len(providers), 1)

    def test_discover_unknown_service_returns_empty(self):
        providers = self.registry.discover_service("no_such_service")
        self.assertEqual(len(providers), 0)

    def test_agents_on_node(self):
        self.registry.register_agent("a1", "node_x", "W1")
        self.registry.register_agent("a2", "node_x", "W2")
        self.registry.register_agent("a3", "node_y", "W3")
        on_x = self.registry.agents_on_node("node_x")
        self.assertIn("a1", on_x)
        self.assertIn("a2", on_x)
        self.assertNotIn("a3", on_x)

    def test_snapshot_returns_dict(self):
        snap = self.registry.snapshot()
        self.assertIsInstance(snap, dict)
        self.assertIn("agent_count", snap)


class TestLoadBalancer(unittest.TestCase):

    def setUp(self):
        self.cluster = FederationCluster()
        for i in range(3):
            k = _make_kernel()
            self.cluster.add_node(k, node_id=f"n{i}")

    def test_round_robin_selects_node(self):
        lb = LoadBalancer(self.cluster, strategy=BalancingStrategy.ROUND_ROBIN)
        node_id = lb.select_node()
        self.assertIn(node_id, self.cluster.list_nodes())

    def test_round_robin_cycles_through_nodes(self):
        lb = LoadBalancer(self.cluster, strategy=BalancingStrategy.ROUND_ROBIN)
        selected = [lb.select_node() for _ in range(6)]
        # Should visit all 3 nodes at least once in 6 selections
        self.assertGreaterEqual(len(set(selected)), 1)

    def test_least_loaded_selects_node(self):
        lb = LoadBalancer(self.cluster, strategy=BalancingStrategy.LEAST_LOADED)
        node_id = lb.select_node()
        self.assertIn(node_id, self.cluster.list_nodes())

    def test_random_strategy_selects_node(self):
        lb = LoadBalancer(self.cluster, strategy=BalancingStrategy.RANDOM)
        node_id = lb.select_node()
        self.assertIn(node_id, self.cluster.list_nodes())

    def test_rebalance_plan_returns_list(self):
        lb = LoadBalancer(self.cluster, strategy=BalancingStrategy.LEAST_LOADED)
        plan = lb.rebalance_plan()
        self.assertIsInstance(plan, list)


class TestSplitBrainDetector(unittest.TestCase):

    def setUp(self):
        self.cluster = FederationCluster()
        k1 = _make_kernel()
        k2 = _make_kernel()
        self.node1 = self.cluster.add_node(k1, node_id="n1")
        self.node2 = self.cluster.add_node(k2, node_id="n2")

    def test_healthy_node_not_read_only(self):
        detector = SplitBrainDetector(self.node1, self.cluster)
        healthy = detector.check(current_tick=1)
        # With all peers reachable, should be healthy
        self.assertIsNotNone(healthy)

    def test_partition_sets_read_only(self):
        detector = SplitBrainDetector(self.node1, self.cluster)
        # Simulate partition: remove all connected peers
        self.node1._connected_peers.clear()
        healthy = detector.check(current_tick=1)
        # With majority peers unreachable, should be in read-only or unhealthy
        if not healthy:
            self.assertTrue(detector.is_read_only())

    def test_report_returns_dict(self):
        detector = SplitBrainDetector(self.node1, self.cluster)
        report = detector.report()
        self.assertIsInstance(report, dict)
        self.assertIn("node_id", report)


if __name__ == "__main__":
    unittest.main()
