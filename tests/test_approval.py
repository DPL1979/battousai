"""
test_approval.py — Comprehensive tests for battousai.approval
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import threading
import time
import unittest
from typing import List

from battousai.capabilities import CapabilityType
from battousai.approval import (
    ApprovalGate,
    ApprovalHandler,
    ApprovalMiddleware,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResult,
    AutoApproveHandler,
    CallbackApprovalHandler,
    CLIApprovalHandler,
    DEFAULT_RISK_MAP,
    RiskTier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gate(
    policy: ApprovalPolicy | None = None,
) -> ApprovalGate:
    """Build an ApprovalGate wired to an AutoApproveHandler."""
    p = policy or ApprovalPolicy()
    gate = ApprovalGate(p)
    handler = AutoApproveHandler(gate)
    gate.set_handler(handler)
    return gate


def _make_deny_gate(policy: ApprovalPolicy | None = None) -> ApprovalGate:
    """Build a gate whose handler immediately denies every request."""
    p = policy or ApprovalPolicy()
    gate = ApprovalGate(p)

    class _DenyHandler(ApprovalHandler):
        def request_approval(self, request: ApprovalRequest) -> None:
            gate.deny(request.request_id, denied_by="test_denier", reason="Test denial")

    gate.set_handler(_DenyHandler())
    return gate


# ---------------------------------------------------------------------------
# 1. RiskTier mapping and DEFAULT_RISK_MAP
# ---------------------------------------------------------------------------

class TestRiskTierDefaults(unittest.TestCase):

    def test_file_read_is_low(self):
        self.assertEqual(DEFAULT_RISK_MAP[CapabilityType.FILE_READ], RiskTier.LOW)

    def test_memory_read_is_low(self):
        self.assertEqual(DEFAULT_RISK_MAP[CapabilityType.MEMORY_READ], RiskTier.LOW)

    def test_message_is_low(self):
        self.assertEqual(DEFAULT_RISK_MAP[CapabilityType.MESSAGE], RiskTier.LOW)

    def test_file_write_is_medium(self):
        self.assertEqual(DEFAULT_RISK_MAP[CapabilityType.FILE_WRITE], RiskTier.MEDIUM)

    def test_memory_write_is_medium(self):
        self.assertEqual(DEFAULT_RISK_MAP[CapabilityType.MEMORY_WRITE], RiskTier.MEDIUM)

    def test_tool_use_is_medium(self):
        self.assertEqual(DEFAULT_RISK_MAP[CapabilityType.TOOL_USE], RiskTier.MEDIUM)

    def test_spawn_is_high(self):
        self.assertEqual(DEFAULT_RISK_MAP[CapabilityType.SPAWN], RiskTier.HIGH)

    def test_network_is_high(self):
        self.assertEqual(DEFAULT_RISK_MAP[CapabilityType.NETWORK], RiskTier.HIGH)

    def test_admin_is_critical(self):
        self.assertEqual(DEFAULT_RISK_MAP[CapabilityType.ADMIN], RiskTier.CRITICAL)

    def test_all_capability_types_mapped(self):
        for cap in CapabilityType:
            self.assertIn(cap, DEFAULT_RISK_MAP,
                          f"{cap.name} missing from DEFAULT_RISK_MAP")

    def test_risk_tier_enum_values(self):
        self.assertEqual(RiskTier.LOW.value, "low")
        self.assertEqual(RiskTier.MEDIUM.value, "medium")
        self.assertEqual(RiskTier.HIGH.value, "high")
        self.assertEqual(RiskTier.CRITICAL.value, "critical")


# ---------------------------------------------------------------------------
# 2. LOW risk auto-approval
# ---------------------------------------------------------------------------

class TestLowRiskAutoApproval(unittest.TestCase):

    def setUp(self):
        self.gate = _make_gate()

    def test_low_risk_approved(self):
        result = self.gate.check("agent_1", CapabilityType.FILE_READ, "read /tmp/x")
        self.assertTrue(result.approved)

    def test_low_risk_no_wait_required(self):
        result = self.gate.check("agent_1", CapabilityType.FILE_READ, "read /tmp/x")
        self.assertFalse(result.wait_required)

    def test_low_risk_request_stored(self):
        result = self.gate.check("agent_1", CapabilityType.FILE_READ, "read /tmp/x")
        req_id = result.request.request_id
        # Should appear in the audit trail
        trail = self.gate.get_audit_trail()
        ids = [e["request_id"] for e in trail]
        self.assertIn(req_id, ids)

    def test_low_risk_status_approved(self):
        result = self.gate.check("agent_1", CapabilityType.FILE_READ, "read /tmp/x")
        self.assertEqual(result.request.status, "approved")

    def test_low_risk_approved_by_system(self):
        result = self.gate.check("agent_1", CapabilityType.FILE_READ, "read /tmp/x")
        self.assertEqual(result.request.approved_by, "system:auto")

    def test_low_risk_multiple_calls_all_approved(self):
        for i in range(5):
            result = self.gate.check("agent_1", CapabilityType.FILE_READ, f"read /tmp/{i}")
            self.assertTrue(result.approved)


# ---------------------------------------------------------------------------
# 3. MEDIUM risk auto-approval with cooldown
# ---------------------------------------------------------------------------

class TestMediumRiskCooldown(unittest.TestCase):

    def _make_medium_gate(self, cooldown: float = 30.0) -> ApprovalGate:
        policy = ApprovalPolicy(auto_approve_cooldown=cooldown)
        return _make_gate(policy)

    def test_medium_risk_approved(self):
        gate = self._make_medium_gate()
        result = gate.check("agent_1", CapabilityType.FILE_WRITE, "write /tmp/a")
        self.assertTrue(result.approved)

    def test_medium_risk_cooldown_same_agent_cap(self):
        """Second call within cooldown window — still approved but no detail re-log."""
        gate = self._make_medium_gate(cooldown=60.0)
        r1 = gate.check("agent_1", CapabilityType.FILE_WRITE, "write /tmp/a")
        r2 = gate.check("agent_1", CapabilityType.FILE_WRITE, "write /tmp/b")
        self.assertTrue(r1.approved)
        self.assertTrue(r2.approved)

    def test_medium_risk_cooldown_different_agents(self):
        """Cooldown is per-agent — different agents track independently."""
        gate = self._make_medium_gate(cooldown=60.0)
        r1 = gate.check("agent_1", CapabilityType.FILE_WRITE, "write /tmp/a")
        r2 = gate.check("agent_2", CapabilityType.FILE_WRITE, "write /tmp/a")
        self.assertTrue(r1.approved)
        self.assertTrue(r2.approved)

    def test_medium_risk_expired_cooldown_resets(self):
        """After cooldown expires a new detailed log entry is written."""
        gate = self._make_medium_gate(cooldown=0.01)
        gate.check("agent_1", CapabilityType.FILE_WRITE, "write /tmp/a")
        time.sleep(0.05)  # exceed cooldown
        result = gate.check("agent_1", CapabilityType.FILE_WRITE, "write /tmp/b")
        self.assertTrue(result.approved)
        # Both should appear in the full audit trail
        trail = gate.get_audit_trail()
        self.assertGreaterEqual(len(trail), 2)

    def test_medium_risk_different_cap_types_not_shared(self):
        """Cooldown is per capability type, not just per agent."""
        gate = self._make_medium_gate(cooldown=60.0)
        r1 = gate.check("agent_1", CapabilityType.FILE_WRITE, "write /tmp/a")
        r2 = gate.check("agent_1", CapabilityType.MEMORY_WRITE, "write mem key")
        self.assertTrue(r1.approved)
        self.assertTrue(r2.approved)


# ---------------------------------------------------------------------------
# 4. HIGH risk blocks until approved
# ---------------------------------------------------------------------------

class TestHighRiskApproval(unittest.TestCase):

    def test_high_risk_auto_approve_handler_approves(self):
        gate = _make_gate()
        result = gate.check("agent_1", CapabilityType.SPAWN, "spawn worker")
        self.assertTrue(result.approved)

    def test_high_risk_request_has_correct_tier(self):
        gate = _make_gate()
        result = gate.check("agent_1", CapabilityType.SPAWN, "spawn worker")
        self.assertEqual(result.request.risk_tier, RiskTier.HIGH)

    def test_high_risk_status_approved_after_auto(self):
        gate = _make_gate()
        result = gate.check("agent_1", CapabilityType.SPAWN, "spawn worker")
        self.assertEqual(result.request.status, "approved")

    def test_high_risk_wait_required_false_after_resolution(self):
        gate = _make_gate()
        result = gate.check("agent_1", CapabilityType.SPAWN, "spawn worker")
        self.assertFalse(result.wait_required)

    def test_high_risk_callback_handler_approves_async(self):
        """Test that a callback handler can approve from another thread."""
        policy = ApprovalPolicy()
        gate   = ApprovalGate(policy)
        approved_requests: List[str] = []

        def cb(req: ApprovalRequest):
            approved_requests.append(req.request_id)
            gate.approve(req.request_id, approved_by="human_reviewer")

        gate.set_handler(CallbackApprovalHandler(cb))
        result = gate.check("agent_1", CapabilityType.SPAWN, "spawn worker")
        self.assertTrue(result.approved)
        self.assertEqual(len(approved_requests), 1)


# ---------------------------------------------------------------------------
# 5. HIGH risk blocks until denied
# ---------------------------------------------------------------------------

class TestHighRiskDenial(unittest.TestCase):

    def test_high_risk_deny_returns_not_approved(self):
        gate = _make_deny_gate()
        result = gate.check("agent_1", CapabilityType.SPAWN, "spawn worker")
        self.assertFalse(result.approved)

    def test_high_risk_denied_status(self):
        gate = _make_deny_gate()
        result = gate.check("agent_1", CapabilityType.SPAWN, "spawn worker")
        self.assertEqual(result.request.status, "denied")

    def test_high_risk_denied_recorded_in_audit(self):
        gate = _make_deny_gate()
        result = gate.check("agent_1", CapabilityType.SPAWN, "spawn worker")
        trail = gate.get_audit_trail()
        decisions = [e["decision"] for e in trail]
        self.assertIn("denied", decisions)

    def test_deny_nonexistent_request_returns_false(self):
        gate = _make_gate()
        ok = gate.deny("nonexistent-id", denied_by="tester")
        self.assertFalse(ok)

    def test_deny_already_approved_returns_false(self):
        gate = _make_gate()
        result = gate.check("agent_1", CapabilityType.FILE_READ, "read x")
        ok = gate.deny(result.request.request_id, denied_by="tester")
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# 6. CRITICAL tier requires justification
# ---------------------------------------------------------------------------

class TestCriticalTierJustification(unittest.TestCase):

    def test_critical_auto_approve_adds_justification(self):
        gate = _make_gate()
        result = gate.check("agent_1", CapabilityType.ADMIN, "admin action")
        self.assertEqual(result.request.risk_tier, RiskTier.CRITICAL)
        self.assertTrue(result.approved)
        # AutoApproveHandler supplies a justification
        self.assertIsNotNone(result.request.justification)

    def test_critical_manual_approve_records_justification(self):
        policy = ApprovalPolicy()
        gate   = ApprovalGate(policy)
        captured = []

        def cb(req: ApprovalRequest):
            captured.append(req)
            gate.approve(req.request_id, approved_by="admin",
                         justification="Required for maintenance window")

        gate.set_handler(CallbackApprovalHandler(cb))
        result = gate.check("agent_1", CapabilityType.ADMIN, "flush all logs")
        self.assertTrue(result.approved)
        self.assertEqual(result.request.justification, "Required for maintenance window")

    def test_critical_deny_records_reason(self):
        gate = _make_deny_gate()
        result = gate.check("agent_1", CapabilityType.ADMIN, "flush all logs")
        self.assertFalse(result.approved)
        self.assertEqual(result.request.status, "denied")

    def test_critical_tier_in_audit_trail(self):
        gate = _make_gate()
        gate.check("agent_1", CapabilityType.ADMIN, "admin op")
        trail = gate.get_audit_trail()
        critical_entries = [e for e in trail if e["risk_tier"] == "critical"]
        self.assertGreaterEqual(len(critical_entries), 1)


# ---------------------------------------------------------------------------
# 7. Expired request auto-denied
# ---------------------------------------------------------------------------

class TestExpiredRequests(unittest.TestCase):

    def test_expired_request_denied(self):
        """A request with a very short timeout should be marked expired."""
        policy = ApprovalPolicy(approval_timeout=0.01)
        gate   = ApprovalGate(policy)

        # Use a callback handler that does NOTHING (simulates no human present)
        gate.set_handler(CallbackApprovalHandler(lambda req: None))

        result = gate.check("agent_1", CapabilityType.SPAWN, "spawn worker")
        # Wait for expiry
        time.sleep(0.05)
        self.assertFalse(result.approved)
        self.assertEqual(result.request.status, "expired")

    def test_expired_not_in_pending(self):
        """Expired requests should not appear in list_pending()."""
        policy = ApprovalPolicy(approval_timeout=0.01)
        gate   = ApprovalGate(policy)
        gate.set_handler(CallbackApprovalHandler(lambda req: None))

        gate.check("agent_1", CapabilityType.SPAWN, "spawn worker")
        time.sleep(0.05)
        # Trigger stale cleanup
        gate._expire_stale_requests()
        pending = gate.list_pending()
        self.assertEqual(len(pending), 0)

    def test_expired_recorded_in_audit(self):
        policy = ApprovalPolicy(approval_timeout=0.01)
        gate   = ApprovalGate(policy)
        gate.set_handler(CallbackApprovalHandler(lambda req: None))

        gate.check("agent_1", CapabilityType.SPAWN, "spawn worker")
        time.sleep(0.05)
        gate._expire_stale_requests()
        trail = gate.get_audit_trail()
        expired_entries = [e for e in trail if e["decision"] == "expired"]
        self.assertGreaterEqual(len(expired_entries), 1)

    def test_approve_expired_request_returns_false(self):
        policy = ApprovalPolicy(approval_timeout=0.01)
        gate   = ApprovalGate(policy)
        gate.set_handler(CallbackApprovalHandler(lambda req: None))

        result = gate.check("agent_1", CapabilityType.SPAWN, "spawn worker")
        time.sleep(0.05)
        gate._expire_stale_requests()
        ok = gate.approve(result.request.request_id, approved_by="late_human")
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# 8. Audit trail records all decisions
# ---------------------------------------------------------------------------

class TestAuditTrail(unittest.TestCase):

    def setUp(self):
        self.gate = _make_gate()

    def test_audit_trail_initially_empty(self):
        gate = ApprovalGate(ApprovalPolicy())
        self.assertEqual(gate.get_audit_trail(), [])

    def test_audit_trail_grows_with_requests(self):
        self.gate.check("agent_1", CapabilityType.FILE_READ, "read x")
        self.gate.check("agent_1", CapabilityType.FILE_READ, "read y")
        trail = self.gate.get_audit_trail()
        self.assertGreaterEqual(len(trail), 2)

    def test_audit_entry_has_required_fields(self):
        self.gate.check("agent_1", CapabilityType.FILE_READ, "read x")
        entry = self.gate.get_audit_trail()[0]
        for field in ("request_id", "agent_id", "action_type", "decision",
                      "decided_by", "timestamp", "risk_tier"):
            self.assertIn(field, entry, f"Missing field: {field}")

    def test_audit_trail_records_denial(self):
        gate = _make_deny_gate()
        gate.check("agent_1", CapabilityType.SPAWN, "spawn")
        trail = gate.get_audit_trail()
        self.assertTrue(any(e["decision"] == "denied" for e in trail))

    def test_audit_trail_returns_copy(self):
        """Modifications to the returned list should not affect the gate's trail."""
        self.gate.check("agent_1", CapabilityType.FILE_READ, "read x")
        trail1 = self.gate.get_audit_trail()
        trail1.clear()
        trail2 = self.gate.get_audit_trail()
        self.assertGreater(len(trail2), 0)

    def test_audit_action_type_stored_as_name(self):
        self.gate.check("agent_1", CapabilityType.SPAWN, "spawn worker")
        trail = self.gate.get_audit_trail()
        spawn_entries = [e for e in trail if e["action_type"] == "SPAWN"]
        self.assertGreaterEqual(len(spawn_entries), 1)


# ---------------------------------------------------------------------------
# 9. list_pending returns only pending items
# ---------------------------------------------------------------------------

class TestListPending(unittest.TestCase):

    def test_pending_empty_initially(self):
        gate = ApprovalGate(ApprovalPolicy())
        self.assertEqual(gate.list_pending(), [])

    def test_no_pending_after_auto_approve(self):
        gate = _make_gate()
        gate.check("agent_1", CapabilityType.FILE_READ, "read x")
        self.assertEqual(gate.list_pending(), [])

    def test_pending_contains_unresolved_high_risk(self):
        """A handler that does NOT resolve the request leaves it pending."""
        policy = ApprovalPolicy(approval_timeout=60.0)
        gate   = ApprovalGate(policy)
        pending_requests: list = []

        def cb(req: ApprovalRequest):
            pending_requests.append(req)
            # Intentionally do NOT approve or deny

        gate.set_handler(CallbackApprovalHandler(cb))

        # Run check in a background thread so we don't block the test
        thread = threading.Thread(
            target=gate.check,
            args=("agent_1", CapabilityType.SPAWN, "spawn worker"),
            daemon=True,
        )
        thread.start()

        # Give the handler a moment to run
        time.sleep(0.05)
        pending = gate.list_pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].status, "pending")

        # Clean up — approve so the thread exits
        if pending_requests:
            gate.approve(pending_requests[0].request_id, approved_by="cleanup")
        thread.join(timeout=1.0)

    def test_resolved_requests_not_in_pending(self):
        gate = _make_gate()
        r = gate.check("agent_1", CapabilityType.SPAWN, "spawn")
        pending = gate.list_pending()
        self.assertNotIn(r.request, pending)


# ---------------------------------------------------------------------------
# 10. ApprovalMiddleware integration
# ---------------------------------------------------------------------------

class TestApprovalMiddleware(unittest.TestCase):

    def _make_mock_kernel(self):
        """Minimal fake kernel that records dispatched calls."""
        class _MockKernel:
            def __init__(self):
                self.calls = []

            def _dispatch_syscall(self, caller_id, name, **kwargs):
                self.calls.append((caller_id, name, kwargs))
                from collections import namedtuple
                SyscallResult = namedtuple("SyscallResult", ["ok", "error", "value"],
                                           defaults=[None, None])
                return SyscallResult(ok=True)

        return _MockKernel()

    def test_middleware_installs_and_intercepts(self):
        kernel = self._make_mock_kernel()
        gate   = _make_gate()
        mw     = ApprovalMiddleware(gate)
        mw.install(kernel)

        kernel._dispatch_syscall("agent_1", "read_file", path="/tmp/x")
        # Call should have passed through to real handler
        self.assertEqual(len(kernel.calls), 1)

    def test_middleware_denied_blocks_syscall(self):
        kernel = self._make_mock_kernel()
        policy = ApprovalPolicy()
        gate   = ApprovalGate(policy)
        gate.set_handler(CallbackApprovalHandler(
            lambda req: gate.deny(req.request_id, denied_by="test")
        ))
        mw = ApprovalMiddleware(gate)
        mw.install(kernel)

        result = kernel._dispatch_syscall("agent_1", "spawn_agent",
                                          agent_name="worker")
        # Original handler should NOT have been called
        self.assertEqual(len(kernel.calls), 0)
        self.assertFalse(result.ok)

    def test_middleware_uninstall_restores_original(self):
        kernel = self._make_mock_kernel()
        gate   = _make_gate()
        mw     = ApprovalMiddleware(gate)
        mw.install(kernel)
        mw.uninstall()

        kernel._dispatch_syscall("agent_1", "read_file", path="/tmp/x")
        self.assertEqual(len(kernel.calls), 1)

    def test_middleware_transparent_syscalls_pass_through(self):
        """Syscalls without a capability mapping should pass through unchanged."""
        kernel = self._make_mock_kernel()
        gate   = _make_gate()
        mw     = ApprovalMiddleware(gate)
        mw.install(kernel)

        kernel._dispatch_syscall("agent_1", "list_agents")
        self.assertEqual(len(kernel.calls), 1)

    def test_middleware_idempotent_install(self):
        """Installing twice should not double-wrap."""
        kernel = self._make_mock_kernel()
        gate   = _make_gate()
        mw     = ApprovalMiddleware(gate)
        mw.install(kernel)
        mw.install(kernel)  # second install — should be a no-op

        kernel._dispatch_syscall("agent_1", "read_file", path="/tmp/x")
        self.assertEqual(len(kernel.calls), 1)


# ---------------------------------------------------------------------------
# 11. CLIApprovalHandler output format (no real stdin — just verify structure)
# ---------------------------------------------------------------------------

class TestCLIApprovalHandler(unittest.TestCase):

    def test_cli_handler_instantiation(self):
        """CLIApprovalHandler should be constructable without errors."""
        gate    = _make_gate()
        handler = CLIApprovalHandler(gate)
        self.assertIsNotNone(handler)

    def test_cli_handler_is_approval_handler(self):
        gate    = _make_gate()
        handler = CLIApprovalHandler(gate)
        self.assertIsInstance(handler, ApprovalHandler)


# ---------------------------------------------------------------------------
# 12. CallbackApprovalHandler invocation
# ---------------------------------------------------------------------------

class TestCallbackApprovalHandler(unittest.TestCase):

    def test_callback_invoked_with_request(self):
        policy   = ApprovalPolicy()
        gate     = ApprovalGate(policy)
        received = []

        def cb(req: ApprovalRequest):
            received.append(req)
            gate.approve(req.request_id, approved_by="cb_approver")

        gate.set_handler(CallbackApprovalHandler(cb))
        gate.check("agent_1", CapabilityType.SPAWN, "spawn worker")
        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], ApprovalRequest)

    def test_callback_receives_correct_agent_id(self):
        policy   = ApprovalPolicy()
        gate     = ApprovalGate(policy)
        received = []

        def cb(req: ApprovalRequest):
            received.append(req)
            gate.approve(req.request_id, approved_by="cb_approver")

        gate.set_handler(CallbackApprovalHandler(cb))
        gate.check("my_special_agent", CapabilityType.SPAWN, "spawn worker")
        self.assertEqual(received[0].agent_id, "my_special_agent")

    def test_callback_receives_correct_action_type(self):
        policy   = ApprovalPolicy()
        gate     = ApprovalGate(policy)
        received = []

        def cb(req: ApprovalRequest):
            received.append(req)
            gate.approve(req.request_id, approved_by="cb_approver")

        gate.set_handler(CallbackApprovalHandler(cb))
        gate.check("agent_1", CapabilityType.NETWORK, "connect to node")
        self.assertEqual(received[0].action_type, CapabilityType.NETWORK)

    def test_callback_not_invoked_for_low_risk(self):
        """LOW-tier actions are auto-approved and should bypass the handler."""
        policy   = ApprovalPolicy()
        gate     = ApprovalGate(policy)
        received = []

        gate.set_handler(CallbackApprovalHandler(
            lambda req: received.append(req)
        ))
        gate.check("agent_1", CapabilityType.FILE_READ, "read file")
        self.assertEqual(len(received), 0)

    def test_callback_not_invoked_for_medium_risk(self):
        """MEDIUM-tier actions are auto-approved and bypass the handler."""
        policy   = ApprovalPolicy()
        gate     = ApprovalGate(policy)
        received = []

        gate.set_handler(CallbackApprovalHandler(
            lambda req: received.append(req)
        ))
        gate.check("agent_1", CapabilityType.FILE_WRITE, "write file")
        self.assertEqual(len(received), 0)


# ---------------------------------------------------------------------------
# 13. AutoApproveHandler approves everything
# ---------------------------------------------------------------------------

class TestAutoApproveHandler(unittest.TestCase):

    def test_approves_low(self):
        gate = _make_gate()
        r = gate.check("a", CapabilityType.FILE_READ, "read")
        self.assertTrue(r.approved)

    def test_approves_medium(self):
        gate = _make_gate()
        r = gate.check("a", CapabilityType.FILE_WRITE, "write")
        self.assertTrue(r.approved)

    def test_approves_high(self):
        gate = _make_gate()
        r = gate.check("a", CapabilityType.SPAWN, "spawn")
        self.assertTrue(r.approved)

    def test_approves_critical(self):
        gate = _make_gate()
        r = gate.check("a", CapabilityType.ADMIN, "admin op")
        self.assertTrue(r.approved)

    def test_auto_approve_provides_justification_for_critical(self):
        gate = _make_gate()
        r = gate.check("a", CapabilityType.ADMIN, "admin op")
        self.assertIsNotNone(r.request.justification)

    def test_custom_approver_name(self):
        policy  = ApprovalPolicy()
        gate    = ApprovalGate(policy)
        handler = AutoApproveHandler(gate, approver_name="ci_pipeline")
        gate.set_handler(handler)
        r = gate.check("a", CapabilityType.SPAWN, "spawn")
        self.assertEqual(r.request.approved_by, "ci_pipeline")


# ---------------------------------------------------------------------------
# 14. batch_low_risk option
# ---------------------------------------------------------------------------

class TestBatchLowRisk(unittest.TestCase):

    def test_batch_low_risk_does_not_write_to_audit_trail(self):
        policy = ApprovalPolicy(batch_low_risk=True)
        gate   = _make_gate(policy)
        gate.check("a", CapabilityType.FILE_READ, "read x")
        trail = gate.get_audit_trail()
        low_in_trail = [e for e in trail if e["risk_tier"] == "low"]
        self.assertEqual(len(low_in_trail), 0)

    def test_batch_low_risk_buffered_in_batch(self):
        policy = ApprovalPolicy(batch_low_risk=True)
        gate   = _make_gate(policy)
        gate.check("a", CapabilityType.FILE_READ, "read x")
        gate.check("a", CapabilityType.MESSAGE, "send msg")
        batch = gate.get_low_risk_batch()
        self.assertEqual(len(batch), 2)

    def test_batch_cleared_after_retrieval(self):
        policy = ApprovalPolicy(batch_low_risk=True)
        gate   = _make_gate(policy)
        gate.check("a", CapabilityType.FILE_READ, "read x")
        gate.get_low_risk_batch()          # consume
        batch2 = gate.get_low_risk_batch() # should be empty
        self.assertEqual(len(batch2), 0)

    def test_medium_risk_not_batched(self):
        """MEDIUM risk items are not added to the LOW batch."""
        policy = ApprovalPolicy(batch_low_risk=True)
        gate   = _make_gate(policy)
        gate.check("a", CapabilityType.FILE_WRITE, "write x")
        batch = gate.get_low_risk_batch()
        self.assertEqual(len(batch), 0)

    def test_batch_low_risk_false_uses_audit_trail(self):
        policy = ApprovalPolicy(batch_low_risk=False)
        gate   = _make_gate(policy)
        gate.check("a", CapabilityType.FILE_READ, "read x")
        trail = gate.get_audit_trail()
        self.assertGreaterEqual(len(trail), 1)


# ---------------------------------------------------------------------------
# 15. Custom risk_map overrides defaults
# ---------------------------------------------------------------------------

class TestCustomRiskMap(unittest.TestCase):

    def test_custom_map_elevates_file_read_to_high(self):
        """Override FILE_READ to HIGH so that it requires human approval."""
        custom_map = {**DEFAULT_RISK_MAP, CapabilityType.FILE_READ: RiskTier.HIGH}
        policy = ApprovalPolicy(risk_map=custom_map)
        gate   = _make_gate(policy)
        result = gate.check("agent_1", CapabilityType.FILE_READ, "read /secret")
        self.assertEqual(result.request.risk_tier, RiskTier.HIGH)
        self.assertTrue(result.approved)  # AutoApproveHandler still approves

    def test_custom_map_lowers_spawn_to_low(self):
        """Override SPAWN to LOW so it is auto-approved silently."""
        custom_map = {**DEFAULT_RISK_MAP, CapabilityType.SPAWN: RiskTier.LOW}
        policy = ApprovalPolicy(risk_map=custom_map)
        gate   = _make_gate(policy)
        result = gate.check("agent_1", CapabilityType.SPAWN, "spawn worker")
        self.assertEqual(result.request.risk_tier, RiskTier.LOW)
        self.assertTrue(result.approved)

    def test_custom_map_admin_as_medium(self):
        """Override ADMIN to MEDIUM — no human gate needed."""
        custom_map = {**DEFAULT_RISK_MAP, CapabilityType.ADMIN: RiskTier.MEDIUM}
        policy = ApprovalPolicy(risk_map=custom_map)
        gate   = _make_gate(policy)
        result = gate.check("agent_1", CapabilityType.ADMIN, "admin thing")
        self.assertEqual(result.request.risk_tier, RiskTier.MEDIUM)

    def test_unknown_capability_defaults_to_low(self):
        """
        If a CapabilityType is not in the risk map the gate falls back to LOW.
        We simulate this by passing a completely empty risk_map.
        """
        policy = ApprovalPolicy(risk_map={})
        gate   = _make_gate(policy)
        result = gate.check("agent_1", CapabilityType.NETWORK, "network call")
        # Falls back to LOW
        self.assertEqual(result.request.risk_tier, RiskTier.LOW)
        self.assertTrue(result.approved)


# ---------------------------------------------------------------------------
# 16. Dataclass integrity checks
# ---------------------------------------------------------------------------

class TestDataclasses(unittest.TestCase):

    def test_approval_request_fields(self):
        req = ApprovalRequest(
            request_id="r1",
            agent_id="a1",
            action_type=CapabilityType.FILE_READ,
            action_detail="read /tmp/x",
            risk_tier=RiskTier.LOW,
            timestamp=time.time(),
            status="pending",
            expires_at=time.time() + 300,
        )
        self.assertEqual(req.status, "pending")
        self.assertIsNone(req.approved_by)
        self.assertIsNone(req.justification)

    def test_approval_result_fields(self):
        req = ApprovalRequest(
            request_id="r2",
            agent_id="a1",
            action_type=CapabilityType.SPAWN,
            action_detail="spawn worker",
            risk_tier=RiskTier.HIGH,
            timestamp=time.time(),
            status="approved",
            expires_at=time.time() + 300,
        )
        result = ApprovalResult(approved=True, request=req, wait_required=False)
        self.assertTrue(result.approved)
        self.assertFalse(result.wait_required)

    def test_approval_policy_defaults(self):
        policy = ApprovalPolicy()
        self.assertEqual(policy.approval_timeout, 300.0)
        self.assertFalse(policy.batch_low_risk)
        self.assertIn(RiskTier.CRITICAL, policy.require_justification_for)


if __name__ == "__main__":
    unittest.main()
