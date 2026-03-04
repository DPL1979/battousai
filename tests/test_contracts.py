"""
test_contracts.py — Tests for battousai.contracts
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from battousai.contracts import (
    Precondition, Postcondition, Invariant, Contract, ContractMonitor,
    ContractViolation, PropertyChecker, SafetyEnvelope, SafetyEnvelopeConfig,
    DEFAULT_SAFETY_ENVELOPE, POLICY_WARN, POLICY_BLOCK, POLICY_KILL,
    contract,
)
from battousai.agent import Agent, WorkerAgent
from battousai.kernel import Kernel


class TestPrecondition(unittest.TestCase):

    def test_precondition_stores_name(self):
        p = Precondition(
            name="test_pre",
            description="Test precondition",
            check=lambda a: True,
        )
        self.assertEqual(p.name, "test_pre")

    def test_precondition_default_policy_is_warn(self):
        p = Precondition(name="p", description="", check=lambda a: True)
        self.assertEqual(p.on_violation, POLICY_WARN)

    def test_precondition_invalid_policy_raises(self):
        with self.assertRaises(ValueError):
            Precondition(name="p", description="", check=lambda a: True,
                         on_violation="INVALID_POLICY")

    def test_precondition_check_passes(self):
        agent = WorkerAgent(name="W", priority=5)
        p = Precondition(name="p", description="", check=lambda a: a.name == "W")
        self.assertTrue(p.check(agent))

    def test_precondition_check_fails(self):
        agent = WorkerAgent(name="W", priority=5)
        p = Precondition(name="p", description="", check=lambda a: a.name == "NotW")
        self.assertFalse(p.check(agent))


class TestPostcondition(unittest.TestCase):

    def test_postcondition_stores_name(self):
        p = Postcondition(name="post_test", description="", check=lambda a: True)
        self.assertEqual(p.name, "post_test")

    def test_postcondition_block_policy_valid(self):
        p = Postcondition(name="p", description="", check=lambda a: True,
                          on_violation=POLICY_BLOCK)
        self.assertEqual(p.on_violation, POLICY_BLOCK)


class TestInvariant(unittest.TestCase):

    def test_invariant_stores_description(self):
        inv = Invariant(
            name="inv_test",
            description="Priority must be valid",
            check=lambda a: 0 <= a.priority <= 9,
        )
        self.assertEqual(inv.description, "Priority must be valid")

    def test_invariant_kill_policy_valid(self):
        inv = Invariant(name="i", description="", check=lambda a: True,
                        on_violation=POLICY_KILL)
        self.assertEqual(inv.on_violation, POLICY_KILL)


class TestContract(unittest.TestCase):

    def setUp(self):
        self.contract = Contract(
            name="TestContract",
            agent_class_name="WorkerAgent",
        )

    def test_contract_stores_name(self):
        self.assertEqual(self.contract.name, "TestContract")

    def test_add_precondition_builder_style(self):
        p = Precondition(name="pre", description="", check=lambda a: True)
        result = self.contract.add_precondition(p)
        self.assertIs(result, self.contract)
        self.assertEqual(len(self.contract.preconditions), 1)

    def test_add_postcondition_builder_style(self):
        p = Postcondition(name="post", description="", check=lambda a: True)
        result = self.contract.add_postcondition(p)
        self.assertIs(result, self.contract)
        self.assertEqual(len(self.contract.postconditions), 1)

    def test_add_invariant_builder_style(self):
        inv = Invariant(name="inv", description="", check=lambda a: True)
        result = self.contract.add_invariant(inv)
        self.assertIs(result, self.contract)
        self.assertEqual(len(self.contract.invariants), 1)

    def test_contract_summary_returns_string(self):
        summary = self.contract.summary()
        self.assertIsInstance(summary, str)
        self.assertIn("TestContract", summary)

    def test_contract_chaining(self):
        pre = Precondition(name="pre", description="", check=lambda a: True)
        post = Postcondition(name="post", description="", check=lambda a: True)
        inv = Invariant(name="inv", description="", check=lambda a: True)
        c = (Contract(name="C", agent_class_name="A")
             .add_precondition(pre)
             .add_postcondition(post)
             .add_invariant(inv))
        self.assertEqual(len(c.preconditions), 1)
        self.assertEqual(len(c.postconditions), 1)
        self.assertEqual(len(c.invariants), 1)


class TestContractViolation(unittest.TestCase):

    def test_contract_violation_is_exception(self):
        v = ContractViolation(
            agent_id="a1", contract_name="C", condition_name="cond",
            condition_type="invariant", details="failed", tick=5
        )
        self.assertIsInstance(v, Exception)

    def test_contract_violation_stores_fields(self):
        v = ContractViolation(
            agent_id="a1", contract_name="C", condition_name="check_x",
            condition_type="precondition", details="x failed", tick=10
        )
        self.assertEqual(v.agent_id, "a1")
        self.assertEqual(v.condition_name, "check_x")
        self.assertEqual(v.tick, 10)


class TestContractMonitor(unittest.TestCase):

    def setUp(self):
        self.kernel = Kernel(max_ticks=0, debug=False)
        self.kernel.boot()

    def test_contract_monitor_spawns_successfully(self):
        agent_id = self.kernel.spawn_agent(
            ContractMonitor, name="ContractMonitor", priority=0
        )
        self.assertIn(agent_id, self.kernel._agents)

    def test_register_contract_and_check_invariant(self):
        monitor_id = self.kernel.spawn_agent(
            ContractMonitor, name="ContractMonitor", priority=0
        )
        worker_id = self.kernel.spawn_agent(WorkerAgent, name="W", priority=5)

        monitor = self.kernel._agents[monitor_id]
        contract_obj = Contract(
            name="WorkerContract",
            agent_class_name="WorkerAgent",
            invariants=[
                Invariant(
                    name="valid_priority",
                    description="Priority must be in range",
                    check=lambda a: 0 <= a.priority <= 9,
                    on_violation=POLICY_WARN,
                )
            ]
        )
        monitor.register_contract(worker_id, contract_obj)

        # Run a tick — invariant check should pass
        self.kernel.tick()
        self.assertEqual(monitor.stats()["total_violations"], 0)

    def test_contract_violation_logged_on_warn(self):
        monitor_id = self.kernel.spawn_agent(
            ContractMonitor, name="ContractMonitor", priority=0
        )
        worker_id = self.kernel.spawn_agent(WorkerAgent, name="W", priority=5)
        monitor = self.kernel._agents[monitor_id]
        worker = self.kernel._agents[worker_id]

        # Register a contract with an invariant that always fails
        contract_obj = Contract(
            name="FailContract",
            agent_class_name="WorkerAgent",
            invariants=[
                Invariant(
                    name="always_fail",
                    description="Always fails for testing",
                    check=lambda a: False,
                    on_violation=POLICY_WARN,
                )
            ]
        )
        monitor.register_contract(worker_id, contract_obj)
        self.kernel.tick()
        self.assertGreater(len(monitor.violation_log), 0)

    def test_contract_monitor_stats_returns_dict(self):
        monitor_id = self.kernel.spawn_agent(
            ContractMonitor, name="ContractMonitor", priority=0
        )
        monitor = self.kernel._agents[monitor_id]
        stats = monitor.stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("monitored_agents", stats)


class TestPropertyChecker(unittest.TestCase):

    def setUp(self):
        self.checker = PropertyChecker()
        self.agent = WorkerAgent(name="W", priority=5)

    def test_always_property_passes_when_true(self):
        self.checker.always(
            "valid_priority",
            lambda a: 0 <= a.priority <= 9
        )
        results = self.checker.check_all(self.agent, tick=1)
        self.assertTrue(results["valid_priority"])

    def test_always_property_fails_when_false(self):
        self.agent.priority = 99  # Invalid
        self.checker.always(
            "valid_priority",
            lambda a: 0 <= a.priority <= 9
        )
        results = self.checker.check_all(self.agent, tick=1)
        self.assertFalse(results["valid_priority"])

    def test_never_property_passes_when_predicate_is_false(self):
        self.checker.never(
            "never_negative",
            lambda a: a.priority < 0
        )
        results = self.checker.check_all(self.agent, tick=1)
        self.assertTrue(results["never_negative"])

    def test_eventually_property_satisfied_when_predicate_true(self):
        self.checker.eventually(
            "has_name",
            lambda a: len(a.name) > 0,
            within_ticks=5,
        )
        results = self.checker.check_all(self.agent, tick=1)
        self.assertTrue(results["has_name"])

    def test_violations_for_returns_list(self):
        self.checker.always("check_p", lambda a: a.priority >= 0)
        self.checker.check_all(self.agent, tick=1)
        viols = self.checker.violations_for("check_p")
        self.assertIsInstance(viols, list)

    def test_report_returns_string(self):
        self.checker.always("p", lambda a: True)
        self.checker.check_all(self.agent, tick=1)
        report = self.checker.report()
        self.assertIsInstance(report, str)


class TestSafetyEnvelope(unittest.TestCase):

    def setUp(self):
        config = SafetyEnvelopeConfig(
            max_messages_per_tick=3,
            max_memory_writes_per_tick=5,
            max_tool_calls_per_tick=2,
            max_spawn_per_tick=1,
        )
        self.envelope = SafetyEnvelope(config)

    def test_send_message_allowed_within_limit(self):
        for i in range(3):
            allowed = self.envelope.check_send_message("agent_0001", tick=1)
            self.assertTrue(allowed)

    def test_send_message_blocked_above_limit(self):
        for i in range(3):
            self.envelope.check_send_message("agent_0001", tick=1)
        # 4th call should be blocked
        blocked = self.envelope.check_send_message("agent_0001", tick=1)
        self.assertFalse(blocked)

    def test_memory_write_allowed_within_limit(self):
        for i in range(5):
            allowed = self.envelope.check_write_memory("agent_0001", tick=1)
            self.assertTrue(allowed)

    def test_tool_call_blocked_by_forbidden_tools(self):
        env = SafetyEnvelope(SafetyEnvelopeConfig(forbidden_tools=["hack_tool"]))
        allowed = env.check_tool_call("agent_0001", "hack_tool", tick=1)
        self.assertFalse(allowed)

    def test_tool_call_allowed_for_non_forbidden_tool(self):
        env = SafetyEnvelope(SafetyEnvelopeConfig(forbidden_tools=["hack_tool"]))
        allowed = env.check_tool_call("agent_0001", "calculator", tick=1)
        self.assertTrue(allowed)

    def test_spawn_blocked_above_total_agent_cap(self):
        env = SafetyEnvelope(SafetyEnvelopeConfig(max_total_agents=5))
        allowed = env.check_spawn_agent("agent_0001", tick=1, total_agents=5)
        self.assertFalse(allowed)

    def test_file_write_blocked_above_size_limit(self):
        env = SafetyEnvelope(SafetyEnvelopeConfig(max_file_size=10))
        large_data = "x" * 100
        allowed = env.check_write_file("agent_0001", large_data, tick=1)
        self.assertFalse(allowed)

    def test_blocked_log_records_violations(self):
        for i in range(5):
            self.envelope.check_send_message("agent_0001", tick=2)
        self.assertGreater(len(self.envelope.blocked_log), 0)

    def test_stats_returns_dict(self):
        stats = self.envelope.stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_blocked", stats)


class TestContractDecorator(unittest.TestCase):

    def test_contract_decorator_attaches_contract(self):
        @contract(
            preconditions=[
                Precondition("ready", "Agent ready", lambda a: True, POLICY_BLOCK)
            ],
            invariants=[
                Invariant("valid_prio", "Priority ok", lambda a: a.priority >= 0)
            ],
        )
        class DecoratedAgent(Agent):
            def think(self, tick):
                self.yield_cpu()

        self.assertTrue(hasattr(DecoratedAgent, "_contract"))
        c = DecoratedAgent._contract
        self.assertEqual(len(c.preconditions), 1)
        self.assertEqual(len(c.invariants), 1)

    def test_contract_decorator_sets_class_name(self):
        @contract()
        class NamedAgent(Agent):
            def think(self, tick):
                self.yield_cpu()

        c = NamedAgent._contract
        self.assertEqual(c.agent_class_name, "NamedAgent")


if __name__ == "__main__":
    unittest.main()
