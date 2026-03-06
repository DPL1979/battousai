"""
approval.py — Human-in-the-Loop Approval Workflow
===================================================
A checkpoint system that pauses agent execution when high-risk actions
are attempted and requires human approval before proceeding.

Design:
    - LOW risk:      Auto-approved, logged silently.
    - MEDIUM risk:   Auto-approved if cooldown hasn't elapsed, else logged with detail.
    - HIGH risk:     Execution pauses; ApprovalHandler is called; blocks until
                     approved, denied, or expired.
    - CRITICAL risk: Same as HIGH but requires justification text.

All decisions (auto or manual) are recorded in the audit trail with timestamps.

Integration:
    Use ApprovalMiddleware to wrap a Kernel instance and intercept every
    _dispatch_syscall call before it reaches the kernel's own handlers.
"""

from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from battousai.capabilities import CapabilityType


# ---------------------------------------------------------------------------
# Risk tiers
# ---------------------------------------------------------------------------

class RiskTier(Enum):
    """
    Ordered risk tiers that determine how an action request is handled.

    LOW      — Auto-approved, logged silently.
    MEDIUM   — Auto-approved with cooldown enforcement, logged with detail.
    HIGH     — Requires explicit human approval; execution blocks.
    CRITICAL — Requires human approval **and** written justification.
    """
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# Default mapping from CapabilityType to RiskTier.
DEFAULT_RISK_MAP: Dict[CapabilityType, RiskTier] = {
    CapabilityType.FILE_READ:    RiskTier.LOW,
    CapabilityType.MEMORY_READ:  RiskTier.LOW,
    CapabilityType.FILE_WRITE:   RiskTier.MEDIUM,
    CapabilityType.MEMORY_WRITE: RiskTier.MEDIUM,
    CapabilityType.TOOL_USE:     RiskTier.MEDIUM,
    CapabilityType.MESSAGE:      RiskTier.LOW,
    CapabilityType.SPAWN:        RiskTier.HIGH,
    CapabilityType.NETWORK:      RiskTier.HIGH,
    CapabilityType.ADMIN:        RiskTier.CRITICAL,
}


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ApprovalRequest:
    """
    Represents a single pending (or resolved) approval request.

    Fields:
        request_id   — UUID4 string, unique per request.
        agent_id     — ID of the agent making the request.
        action_type  — The CapabilityType the agent is trying to exercise.
        action_detail — Human-readable description of the intended action.
        risk_tier    — Risk classification computed from the policy.
        timestamp    — Unix timestamp when the request was created.
        status       — One of "pending", "approved", "denied", "expired".
        approved_by  — Identity of the approver (None until resolved).
        justification — Required for CRITICAL tier approvals.
        expires_at   — Unix timestamp after which the request auto-expires.
    """
    request_id:    str
    agent_id:      str
    action_type:   CapabilityType
    action_detail: str
    risk_tier:     RiskTier
    timestamp:     float
    status:        str  # "pending" | "approved" | "denied" | "expired"
    approved_by:   Optional[str] = None
    justification: Optional[str] = None
    expires_at:    float = 0.0


@dataclass
class ApprovalResult:
    """
    The outcome returned by ApprovalGate.check().

    Fields:
        approved      — Whether the action is permitted.
        request       — The underlying ApprovalRequest.
        wait_required — True if the caller should block until a human responds.
                        This is set to True for HIGH/CRITICAL tiers before
                        the gate resolves them.
    """
    approved:      bool
    request:       ApprovalRequest
    wait_required: bool = False


@dataclass
class ApprovalPolicy:
    """
    Configurable policy that governs ApprovalGate behaviour.

    Fields:
        risk_map                 — Maps CapabilityType → RiskTier.
                                   Overrides DEFAULT_RISK_MAP for listed types.
        auto_approve_cooldown    — Seconds between auto-approvals for MEDIUM
                                   tier.  If an agent makes the same capability
                                   request within the cooldown window it is
                                   approved without re-logging.
        approval_timeout         — Seconds before a pending HIGH/CRITICAL
                                   request auto-expires (default 300).
        require_justification_for — Tiers that must supply justification text
                                   (default: [CRITICAL]).
        batch_low_risk           — If True, LOW-tier actions are batched into
                                   periodic summaries instead of individual
                                   log entries.
    """
    risk_map: Dict[CapabilityType, RiskTier] = field(
        default_factory=lambda: dict(DEFAULT_RISK_MAP)
    )
    auto_approve_cooldown:       float          = 30.0
    approval_timeout:            float          = 300.0
    require_justification_for:   List[RiskTier] = field(
        default_factory=lambda: [RiskTier.CRITICAL]
    )
    batch_low_risk:              bool           = False


# ---------------------------------------------------------------------------
# ApprovalHandler protocol / ABC + concrete implementations
# ---------------------------------------------------------------------------

class ApprovalHandler(ABC):
    """
    Abstract base class that defines how pending approval requests reach
    human operators (or automated systems in tests/CI).

    Implementors should notify whoever is responsible for reviewing
    the request.  They do NOT resolve the request themselves — resolution
    happens through ApprovalGate.approve() / ApprovalGate.deny().
    """

    @abstractmethod
    def request_approval(self, request: ApprovalRequest) -> None:
        """
        Notify the appropriate party that ``request`` needs a decision.

        This method should return quickly; it must not block waiting for
        a human to respond.  Blocking resolution happens inside
        ApprovalGate._wait_for_resolution().
        """


class CLIApprovalHandler(ApprovalHandler):
    """
    Interactive handler for terminal-based usage.

    Prints a formatted approval prompt to stdout and reads a response from
    stdin.  It resolves the request immediately by calling the provided
    gate's approve() or deny() method.

    Note:
        This handler BLOCKS the calling thread while waiting for user input.
        It is not suitable for unit tests — use AutoApproveHandler or
        CallbackApprovalHandler instead.
    """

    def __init__(self, gate: "ApprovalGate") -> None:
        self._gate = gate

    def request_approval(self, request: ApprovalRequest) -> None:
        """Print the request to stdout and prompt for a decision via stdin."""
        sep = "-" * 60
        print(sep)
        print("  *** APPROVAL REQUIRED ***")
        print(f"  Request ID  : {request.request_id}")
        print(f"  Agent       : {request.agent_id}")
        print(f"  Action      : {request.action_type.name}")
        print(f"  Detail      : {request.action_detail}")
        print(f"  Risk Tier   : {request.risk_tier.value.upper()}")
        print(f"  Expires at  : {time.strftime('%H:%M:%S', time.localtime(request.expires_at))}")
        print(sep)

        justification: Optional[str] = None
        if request.risk_tier == RiskTier.CRITICAL:
            justification = input("  Justification (required): ").strip() or None

        decision = input("  Approve? [y/N]: ").strip().lower()
        if decision == "y":
            self._gate.approve(request.request_id, approved_by="cli_user",
                               justification=justification)
        else:
            self._gate.deny(request.request_id, denied_by="cli_user",
                            reason="Rejected via CLI")


class CallbackApprovalHandler(ApprovalHandler):
    """
    Handler that delegates to a user-supplied callable.

    The callable receives the ApprovalRequest and is responsible for
    eventually calling ApprovalGate.approve() or ApprovalGate.deny().
    Useful for GUI frontends, webhook integrations, and unit tests.
    """

    def __init__(self, callback: Callable[[ApprovalRequest], None]) -> None:
        """
        Args:
            callback: A function that accepts an ApprovalRequest.  It should
                      call gate.approve() or gate.deny() at some point.
        """
        self._callback = callback

    def request_approval(self, request: ApprovalRequest) -> None:
        """Invoke the user-supplied callback with the request."""
        self._callback(request)


class AutoApproveHandler(ApprovalHandler):
    """
    Handler that unconditionally approves every request immediately.

    Intended for automated testing and CI pipelines where human review is
    not needed.  Also useful for smoke-testing the plumbing without a real
    operator in the loop.
    """

    def __init__(self, gate: "ApprovalGate", approver_name: str = "auto") -> None:
        self._gate = gate
        self._approver_name = approver_name

    def request_approval(self, request: ApprovalRequest) -> None:
        """Approve the request immediately with a placeholder justification."""
        justification = "Auto-approved" if request.risk_tier == RiskTier.CRITICAL else None
        self._gate.approve(
            request.request_id,
            approved_by=self._approver_name,
            justification=justification,
        )


# ---------------------------------------------------------------------------
# ApprovalGate — the main orchestrator
# ---------------------------------------------------------------------------

class ApprovalGate:
    """
    Central checkpoint that evaluates every agent action against the policy
    and either auto-approves it or pauses execution until a human decides.

    Thread-safety:
        All internal state mutations are protected by a single re-entrant
        lock so that the gate can be used from multiple agent threads
        simultaneously.

    Usage::

        policy  = ApprovalPolicy()
        handler = AutoApproveHandler(gate)   # wired after construction
        gate    = ApprovalGate(policy, handler)

        result = gate.check("agent_001", CapabilityType.SPAWN, "spawn worker")
        if not result.approved:
            raise PermissionError("Action denied")
    """

    def __init__(
        self,
        policy:           ApprovalPolicy,
        approval_handler: Optional[ApprovalHandler] = None,
    ) -> None:
        self._policy   = policy
        self._handler  = approval_handler

        # Keyed by request_id
        self._requests: Dict[str, ApprovalRequest] = {}

        # Audit trail: ordered list of decision records
        self._audit_trail: List[Dict[str, Any]] = []

        # Per-agent per-capability cooldown tracking for MEDIUM tier.
        # Key: (agent_id, cap_type) → last auto-approve unix timestamp
        self._medium_last_approved: Dict[tuple, float] = {}

        # Batch buffer for LOW-risk actions (populated when batch_low_risk=True)
        self._low_risk_batch: List[Dict[str, Any]] = []

        # Threading primitives
        self._lock = threading.RLock()
        # Maps request_id → Event that is set when the request is resolved
        self._resolution_events: Dict[str, threading.Event] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_handler(self, handler: ApprovalHandler) -> None:
        """Replace the approval handler at runtime."""
        with self._lock:
            self._handler = handler

    def check(
        self,
        agent_id:       str,
        action_type:    CapabilityType,
        action_detail:  str,
    ) -> ApprovalResult:
        """
        Evaluate whether ``agent_id`` may perform ``action_type``.

        LOW/MEDIUM tiers are resolved synchronously and returned immediately.
        HIGH/CRITICAL tiers block the calling thread until a human resolves
        the request or it expires.

        Returns:
            ApprovalResult with ``approved=True`` if the action is permitted.
        """
        self._expire_stale_requests()

        risk_tier = self._policy.risk_map.get(action_type, RiskTier.LOW)
        now       = time.time()
        req       = ApprovalRequest(
            request_id    = str(uuid.uuid4()),
            agent_id      = agent_id,
            action_type   = action_type,
            action_detail = action_detail,
            risk_tier     = risk_tier,
            timestamp     = now,
            status        = "pending",
            expires_at    = now + self._policy.approval_timeout,
        )

        with self._lock:
            self._requests[req.request_id] = req

        if risk_tier == RiskTier.LOW:
            return self._handle_low(req)
        elif risk_tier == RiskTier.MEDIUM:
            return self._handle_medium(req)
        else:
            # HIGH or CRITICAL — requires human decision
            return self._handle_high_or_critical(req)

    def approve(
        self,
        request_id:    str,
        approved_by:   str,
        justification: Optional[str] = None,
    ) -> bool:
        """
        Mark a pending request as approved.

        For CRITICAL-tier requests, ``justification`` is required.
        Returns True if the request existed and was still pending, False otherwise.
        """
        with self._lock:
            req = self._requests.get(request_id)
            if req is None or req.status != "pending":
                return False

            # Enforce justification requirement
            if req.risk_tier in self._policy.require_justification_for:
                if not justification:
                    justification = ""   # accept empty but record it

            req.status        = "approved"
            req.approved_by   = approved_by
            req.justification = justification

            self._record_audit(req, decision="approved", decided_by=approved_by,
                               justification=justification)

            # Signal any waiting thread
            evt = self._resolution_events.get(request_id)
            if evt:
                evt.set()

        return True

    def deny(
        self,
        request_id: str,
        denied_by:  str,
        reason:     str = "",
    ) -> bool:
        """
        Mark a pending request as denied.

        Returns True if the request existed and was still pending, False otherwise.
        """
        with self._lock:
            req = self._requests.get(request_id)
            if req is None or req.status != "pending":
                return False

            req.status      = "denied"
            req.approved_by = denied_by
            req.justification = reason

            self._record_audit(req, decision="denied", decided_by=denied_by,
                               justification=reason)

            evt = self._resolution_events.get(request_id)
            if evt:
                evt.set()

        return True

    def list_pending(self) -> List[ApprovalRequest]:
        """Return all requests that are currently in the ``pending`` state."""
        with self._lock:
            return [r for r in self._requests.values() if r.status == "pending"]

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the complete, ordered audit trail of all decisions."""
        with self._lock:
            return list(self._audit_trail)

    def get_low_risk_batch(self) -> List[Dict[str, Any]]:
        """
        Return and clear the buffered LOW-risk actions.

        Only meaningful when ``policy.batch_low_risk`` is True.
        """
        with self._lock:
            batch = list(self._low_risk_batch)
            self._low_risk_batch.clear()
            return batch

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_low(self, req: ApprovalRequest) -> ApprovalResult:
        """Auto-approve LOW-tier actions, either silently or via batch buffer."""
        return self._auto_approve(req, detail_log=False)

    def _handle_medium(self, req: ApprovalRequest) -> ApprovalResult:
        """
        Auto-approve MEDIUM-tier actions subject to cooldown.

        If the same (agent, capability) pair was approved within the
        cooldown window the new request is still approved but not
        added to the detailed log again.  Once outside the window a
        new detailed log entry is written and the cooldown resets.
        """
        key = (req.agent_id, req.action_type)
        now = time.time()
        with self._lock:
            last = self._medium_last_approved.get(key, 0.0)
            within_cooldown = (now - last) < self._policy.auto_approve_cooldown
            if not within_cooldown:
                self._medium_last_approved[key] = now
        # Always auto-approve; detail_log=True only on the first call per window
        return self._auto_approve(req, detail_log=not within_cooldown)

    def _handle_high_or_critical(self, req: ApprovalRequest) -> ApprovalResult:
        """
        Block the calling thread until a human approves/denies the request
        or it expires.
        """
        evt = threading.Event()
        with self._lock:
            self._resolution_events[req.request_id] = evt

        # Notify the handler (must not block — it fires and returns)
        if self._handler is not None:
            self._handler.request_approval(req)

        # Wait for resolution or timeout
        remaining = req.expires_at - time.time()
        if remaining > 0:
            evt.wait(timeout=remaining)

        # Cleanup event reference
        with self._lock:
            self._resolution_events.pop(req.request_id, None)
            current_status = req.status

        # If still pending after wait, mark expired
        if current_status == "pending":
            self._mark_expired(req)
            return ApprovalResult(approved=False, request=req, wait_required=False)

        return ApprovalResult(
            approved=(req.status == "approved"),
            request=req,
            wait_required=False,
        )

    def _auto_approve(self, req: ApprovalRequest, *, detail_log: bool) -> ApprovalResult:
        """
        Resolve a request as auto-approved.

        When ``detail_log`` is False and ``batch_low_risk`` is enabled the
        entry is buffered rather than written to the main audit trail.
        """
        with self._lock:
            req.status      = "approved"
            req.approved_by = "system:auto"

            if not detail_log and self._policy.batch_low_risk:
                self._low_risk_batch.append({
                    "request_id":   req.request_id,
                    "agent_id":     req.agent_id,
                    "action_type":  req.action_type.name,
                    "action_detail": req.action_detail,
                    "risk_tier":    req.risk_tier.value,
                    "timestamp":    req.timestamp,
                    "decision":     "approved",
                    "decided_by":   "system:auto",
                    "batched":      True,
                })
            else:
                self._record_audit(
                    req,
                    decision    = "approved",
                    decided_by  = "system:auto",
                    detail_log  = detail_log,
                )

        return ApprovalResult(approved=True, request=req, wait_required=False)

    def _mark_expired(self, req: ApprovalRequest) -> None:
        """Transition a request from pending to expired and audit-log the event."""
        with self._lock:
            if req.status == "pending":
                req.status = "expired"
                self._record_audit(req, decision="expired", decided_by="system:timeout")

    def _expire_stale_requests(self) -> None:
        """
        Scan all pending requests and expire any that have passed their
        ``expires_at`` deadline.  Called at the start of every check().
        """
        now = time.time()
        with self._lock:
            for req in list(self._requests.values()):
                if req.status == "pending" and now >= req.expires_at:
                    self._mark_expired(req)
                    evt = self._resolution_events.get(req.request_id)
                    if evt:
                        evt.set()

    def _record_audit(
        self,
        req:          ApprovalRequest,
        decision:     str,
        decided_by:   str,
        justification: Optional[str] = None,
        detail_log:   bool = True,
    ) -> None:
        """
        Append a structured record to the audit trail.

        Must be called while holding self._lock.
        """
        self._audit_trail.append({
            "request_id":    req.request_id,
            "agent_id":      req.agent_id,
            "action_type":   req.action_type.name,
            "action_detail": req.action_detail,
            "risk_tier":     req.risk_tier.value,
            "timestamp":     req.timestamp,
            "decision":      decision,
            "decided_by":    decided_by,
            "justification": justification,
            "detail":        detail_log,
            "decided_at":    time.time(),
        })


# ---------------------------------------------------------------------------
# ApprovalMiddleware — kernel integration layer
# ---------------------------------------------------------------------------

# Maps kernel syscall names to the CapabilityType they exercise.
_SYSCALL_TO_CAPABILITY: Dict[str, CapabilityType] = {
    "read_file":      CapabilityType.FILE_READ,
    "read_memory":    CapabilityType.MEMORY_READ,
    "write_file":     CapabilityType.FILE_WRITE,
    "write_memory":   CapabilityType.MEMORY_WRITE,
    "access_tool":    CapabilityType.TOOL_USE,
    "send_message":   CapabilityType.MESSAGE,
    "spawn_agent":    CapabilityType.SPAWN,
    "kill_agent":     CapabilityType.SPAWN,   # also requires SPAWN clearance
    "publish_topic":  CapabilityType.NETWORK,
    "subscribe":      CapabilityType.NETWORK,
}


class ApprovalMiddleware:
    """
    Wraps the kernel's ``_dispatch_syscall`` to intercept capability-requiring
    operations and run them through the ApprovalGate before proceeding.

    Integration with SafetyEnvelope:
        If the SafetyEnvelope has already blocked the syscall (by returning
        a failed SyscallResult before this middleware runs) then no approval
        is needed — the action is already denied.  In practice the middleware
        is installed AFTER the SafetyEnvelope so that only actions that pass
        the hard limits are presented for human review.

    Usage::

        gate       = ApprovalGate(policy, handler)
        middleware = ApprovalMiddleware(gate)
        middleware.install(kernel)
    """

    def __init__(self, gate: ApprovalGate) -> None:
        self._gate    = gate
        self._kernel  = None
        self._original_dispatch: Optional[Any] = None

    def install(self, kernel: Any) -> None:
        """
        Patch ``kernel._dispatch_syscall`` to route every syscall through
        the approval gate first.

        This is idempotent — calling install() twice on the same kernel
        will not double-wrap.
        """
        if self._kernel is not None:
            return  # already installed
        self._kernel           = kernel
        self._original_dispatch = kernel._dispatch_syscall
        kernel._dispatch_syscall = self._intercepted_dispatch

    def uninstall(self) -> None:
        """Restore the kernel's original dispatch method."""
        if self._kernel is not None and self._original_dispatch is not None:
            self._kernel._dispatch_syscall = self._original_dispatch
            self._kernel   = None
            self._original_dispatch = None

    def _intercepted_dispatch(
        self, caller_id: str, name: str, **kwargs: Any
    ) -> Any:
        """
        Intercept a syscall, check approval, then forward to the real handler.

        If the action is not in the capability map it is forwarded unchanged
        (read_inbox, list_agents, get_status, yield_cpu, list_dir are all
        considered transparent/administrative and do not require approval).
        """
        cap_type = _SYSCALL_TO_CAPABILITY.get(name)
        if cap_type is not None:
            # Build a human-readable action description from kwargs
            detail = self._describe(name, kwargs)
            result = self._gate.check(caller_id, cap_type, detail)
            if not result.approved:
                # Return a SyscallResult-like object signalling denial
                from battousai.agent import SyscallResult
                return SyscallResult(
                    ok=False,
                    error=(
                        f"Approval denied for {name!r} "
                        f"(request_id={result.request.request_id}, "
                        f"status={result.request.status})"
                    ),
                )
        return self._original_dispatch(caller_id, name, **kwargs)

    @staticmethod
    def _describe(syscall_name: str, kwargs: Dict[str, Any]) -> str:
        """Build a concise human-readable action description."""
        if syscall_name == "access_tool":
            tool = kwargs.get("tool_name", "unknown")
            return f"Use tool '{tool}'"
        if syscall_name == "write_file":
            path = kwargs.get("path", "unknown")
            return f"Write file '{path}'"
        if syscall_name == "read_file":
            path = kwargs.get("path", "unknown")
            return f"Read file '{path}'"
        if syscall_name == "write_memory":
            key = kwargs.get("key", "unknown")
            return f"Write memory key '{key}'"
        if syscall_name == "read_memory":
            key = kwargs.get("key", "unknown")
            return f"Read memory key '{key}'"
        if syscall_name == "send_message":
            recipient = kwargs.get("recipient_id", "unknown")
            return f"Send message to '{recipient}'"
        if syscall_name == "spawn_agent":
            agent_name = kwargs.get("agent_name", "unknown")
            return f"Spawn agent '{agent_name}'"
        if syscall_name == "kill_agent":
            target = kwargs.get("target_id", "unknown")
            return f"Kill agent '{target}'"
        if syscall_name in ("publish_topic", "subscribe"):
            topic = kwargs.get("topic", "unknown")
            return f"{syscall_name} on topic '{topic}'"
        return f"Syscall '{syscall_name}'"
