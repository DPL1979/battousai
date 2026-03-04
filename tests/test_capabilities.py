"""
test_capabilities.py — Tests for battousai.capabilities
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from battousai.capabilities import (
    Capability, CapabilityType, CapabilitySet, CapabilityManager,
    CapabilityViolation, DEFAULT_POLICY, SecurityPolicy,
)


def _make_capability(
    cap_id="cap_001",
    cap_type=CapabilityType.TOOL_USE,
    resource_pattern="calculator",
    granted_to="agent_0001",
    granted_by="kernel",
    created_at=0,
):
    return Capability(
        cap_id=cap_id,
        cap_type=cap_type,
        resource_pattern=resource_pattern,
        granted_to=granted_to,
        granted_by=granted_by,
        created_at=created_at,
    )


class TestCapabilityCreation(unittest.TestCase):

    def test_capability_stores_cap_type(self):
        cap = _make_capability(cap_type=CapabilityType.TOOL_USE)
        self.assertEqual(cap.cap_type, CapabilityType.TOOL_USE)

    def test_capability_stores_resource_pattern(self):
        cap = _make_capability(
            cap_id="cap_002",
            cap_type=CapabilityType.FILE_READ,
            resource_pattern="/shared/*",
        )
        self.assertEqual(cap.resource_pattern, "/shared/*")

    def test_capability_type_enum_has_required_values(self):
        names = [ct.name for ct in CapabilityType]
        for expected in [
            "TOOL_USE", "FILE_READ", "FILE_WRITE",
            "MEMORY_READ", "MEMORY_WRITE", "SPAWN",
            "MESSAGE", "NETWORK", "ADMIN"
        ]:
            self.assertIn(expected, names)


class TestCapabilitySet(unittest.TestCase):

    def setUp(self):
        self.cap_set = CapabilitySet(agent_id="agent_0001")

    def test_empty_set_has_no_capabilities(self):
        self.assertEqual(len(self.cap_set.list()), 0)

    def test_grant_adds_capability(self):
        cap = _make_capability()
        self.cap_set.grant(cap)
        caps = self.cap_set.list()
        self.assertEqual(len(caps), 1)

    def test_has_capability_exact_match(self):
        cap = _make_capability()
        self.cap_set.grant(cap)
        self.assertTrue(
            self.cap_set.has_capability(CapabilityType.TOOL_USE, "calculator")
        )

    def test_has_capability_wildcard_pattern(self):
        cap = _make_capability(
            cap_id="c2",
            cap_type=CapabilityType.FILE_READ,
            resource_pattern="/shared/*",
        )
        self.cap_set.grant(cap)
        self.assertTrue(
            self.cap_set.has_capability(
                CapabilityType.FILE_READ, "/shared/results/report.txt"
            )
        )

    def test_revoke_removes_capability(self):
        cap = _make_capability(
            cap_id="c3",
            cap_type=CapabilityType.TOOL_USE,
            resource_pattern="web_search",
        )
        self.cap_set.grant(cap)
        self.cap_set.revoke("c3")
        self.assertFalse(
            self.cap_set.has_capability(CapabilityType.TOOL_USE, "web_search")
        )

    def test_missing_capability_returns_false(self):
        self.assertFalse(self.cap_set.has_capability(CapabilityType.ADMIN, "kernel"))


class TestCapabilityManager(unittest.TestCase):

    def setUp(self):
        self.manager = CapabilityManager()

    def test_create_capability_returns_capability(self):
        cap = self.manager.create_capability(
            cap_type=CapabilityType.TOOL_USE,
            resource_pattern="calculator",
            agent_id="agent_0001",
            current_tick=0,
        )
        self.assertIsInstance(cap, Capability)

    def test_check_granted_capability(self):
        self.manager.create_capability(
            cap_type=CapabilityType.TOOL_USE,
            resource_pattern="calculator",
            agent_id="agent_0001",
            current_tick=0,
        )
        self.assertTrue(
            self.manager.check("agent_0001", CapabilityType.TOOL_USE, "calculator")
        )

    def test_check_without_grant_returns_false(self):
        self.assertFalse(
            self.manager.check("agent_0001", CapabilityType.FILE_WRITE, "/system/logs")
        )

    def test_revoke_capability(self):
        cap = self.manager.create_capability(
            cap_type=CapabilityType.TOOL_USE,
            resource_pattern="web_search",
            agent_id="agent_0001",
            current_tick=0,
        )
        self.manager.revoke(cap.cap_id)
        self.assertFalse(
            self.manager.check("agent_0001", CapabilityType.TOOL_USE, "web_search")
        )

    def test_admin_capability_grants_all_access(self):
        self.manager.create_capability(
            cap_type=CapabilityType.ADMIN,
            resource_pattern="*",
            agent_id="admin_agent",
            current_tick=0,
        )
        self.assertTrue(
            self.manager.check("admin_agent", CapabilityType.ADMIN, "anything")
        )

    def test_audit_log_records_checks(self):
        self.manager.create_capability(
            cap_type=CapabilityType.TOOL_USE,
            resource_pattern="calculator",
            agent_id="agent_0001",
            current_tick=0,
        )
        self.manager.check("agent_0001", CapabilityType.TOOL_USE, "calculator")
        log = self.manager.audit_log()
        self.assertIsInstance(log, list)
        self.assertGreater(len(log), 0)


class TestCapabilityViolation(unittest.TestCase):

    def test_capability_violation_exception_stores_fields(self):
        violation = CapabilityViolation(
            agent_id="agent_0001",
            cap_type=CapabilityType.FILE_WRITE,
            resource="/system/config",
        )
        self.assertEqual(violation.agent_id, "agent_0001")
        self.assertEqual(violation.cap_type, CapabilityType.FILE_WRITE)

    def test_capability_violation_is_exception(self):
        violation = CapabilityViolation(
            agent_id="agent_0001",
            cap_type=CapabilityType.SPAWN,
            resource="any",
        )
        self.assertIsInstance(violation, Exception)


class TestDefaultPolicy(unittest.TestCase):

    def test_default_policy_is_security_policy(self):
        self.assertIsInstance(DEFAULT_POLICY, SecurityPolicy)

    def test_default_policy_has_rules(self):
        self.assertIsNotNone(DEFAULT_POLICY)


if __name__ == "__main__":
    unittest.main()
