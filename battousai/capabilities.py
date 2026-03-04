"""
capabilities.py — Capability-Based Security
===============================================
Agents receive a set of capabilities at spawn time that define exactly
what OS resources they can access. Capabilities are unforgeable tokens
managed by the kernel.

Design principles:
    - Principle of least privilege: agents get only what they need
    - Capabilities are delegatable: an agent can grant a subset of its
      capabilities to child agents
    - Capabilities are revocable: the kernel or a supervisor can revoke caps
    - No ambient authority: without a capability, an agent cannot access
      a resource

Capability Types:
    TOOL_USE(tool_name)      — permission to use a specific tool
    FILE_READ(path_pattern)  — permission to read files matching a glob
    FILE_WRITE(path_pattern) — permission to write files matching a glob
    MEMORY_READ(agent_id)    — permission to read another agent's memory
    MEMORY_WRITE(region)     — permission to write to a shared memory region
    SPAWN(agent_class)       — permission to spawn agents of a specific class
    MESSAGE(agent_pattern)   — permission to send messages to matching agents
    NETWORK(node_pattern)    — permission to communicate with remote nodes
    ADMIN                    — full access (only for kernel-level agents)
"""

from __future__ import annotations

import fnmatch
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type


class CapabilityType(Enum):
    TOOL_USE     = auto()
    FILE_READ    = auto()
    FILE_WRITE   = auto()
    MEMORY_READ  = auto()
    MEMORY_WRITE = auto()
    SPAWN        = auto()
    MESSAGE      = auto()
    NETWORK      = auto()
    ADMIN        = auto()


class CapabilityViolation(Exception):
    def __init__(self, agent_id: str, cap_type: CapabilityType, resource: str, message: str = "") -> None:
        self.agent_id = agent_id
        self.cap_type = cap_type
        self.resource = resource
        full_msg = message or f"Agent {agent_id!r} lacks {cap_type.name}({resource!r}) capability."
        super().__init__(full_msg)


@dataclass
class AuditEntry:
    timestamp: int
    agent_id: str
    action: str
    cap_type: CapabilityType
    resource: str
    allowed: bool
    details: str = ""
    cap_id: str = ""

    def __repr__(self) -> str:
        result = "ALLOW" if self.allowed else "DENY"
        return (f"AuditEntry(tick={self.timestamp}, agent={self.agent_id!r}, "
                f"action={self.action}, {self.cap_type.name}({self.resource!r}), {result})")


@dataclass
class Capability:
    cap_id: str
    cap_type: CapabilityType
    resource_pattern: str
    granted_to: str
    granted_by: str
    created_at: int
    expires_at: Optional[int] = None
    delegatable: bool = False
    revoked: bool = False

    def is_active(self, current_tick: int) -> bool:
        if self.revoked:
            return False
        if self.expires_at is not None and current_tick >= self.expires_at:
            return False
        return True

    def covers(self, resource: str) -> bool:
        return fnmatch.fnmatch(resource, self.resource_pattern)

    def __repr__(self) -> str:
        status = "active" if not self.revoked else "revoked"
        exp = f", expires={self.expires_at}" if self.expires_at is not None else ""
        return (f"Capability(id={self.cap_id[:8]}, {self.cap_type.name}"
                f"({self.resource_pattern!r}), to={self.granted_to!r}, {status}{exp})")


class CapabilitySet:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._caps: Dict[str, Capability] = {}

    def grant(self, capability: Capability) -> None:
        self._caps[capability.cap_id] = capability

    def revoke(self, cap_id: str) -> bool:
        cap = self._caps.get(cap_id)
        if cap is None:
            return False
        cap.revoked = True
        return True

    def remove(self, cap_id: str) -> Optional[Capability]:
        return self._caps.pop(cap_id, None)

    def has_capability(self, cap_type: CapabilityType, resource: str, current_tick: int = 0) -> bool:
        for cap in self._caps.values():
            if not cap.is_active(current_tick):
                continue
            if cap.cap_type == CapabilityType.ADMIN:
                return True
            if cap.cap_type == cap_type and cap.covers(resource):
                return True
        return False

    def delegate(self, cap_id: str, target_agent_id: str, current_tick: int, grantor_id: str,
                 expires_at: Optional[int] = None, delegatable: bool = False) -> Optional[Capability]:
        src_cap = self._caps.get(cap_id)
        if src_cap is None or not src_cap.is_active(current_tick):
            return None
        if not src_cap.delegatable:
            return None
        effective_expires: Optional[int] = expires_at
        if src_cap.expires_at is not None:
            if effective_expires is None:
                effective_expires = src_cap.expires_at
            else:
                effective_expires = min(effective_expires, src_cap.expires_at)
        new_cap = Capability(
            cap_id=str(uuid.uuid4()), cap_type=src_cap.cap_type,
            resource_pattern=src_cap.resource_pattern, granted_to=target_agent_id,
            granted_by=grantor_id, created_at=current_tick, expires_at=effective_expires,
            delegatable=delegatable, revoked=False,
        )
        return new_cap

    def list(self, include_revoked: bool = False) -> List[Capability]:
        if include_revoked:
            return list(self._caps.values())
        return [c for c in self._caps.values() if not c.revoked]

    def filter_by_type(self, cap_type: CapabilityType) -> List[Capability]:
        return [c for c in self._caps.values() if c.cap_type == cap_type and not c.revoked]

    def __len__(self) -> int:
        return len(self._caps)

    def __repr__(self) -> str:
        active = sum(1 for c in self._caps.values() if not c.revoked)
        return f"CapabilitySet(agent={self.agent_id!r}, active={active}/{len(self._caps)})"


class CapabilityManager:
    def __init__(self) -> None:
        self._agent_caps: Dict[str, CapabilitySet] = {}
        self._cap_index: Dict[str, Capability] = {}
        self._audit_log: List[AuditEntry] = []

    def register_agent(self, agent_id: str) -> CapabilitySet:
        cap_set = CapabilitySet(agent_id)
        self._agent_caps[agent_id] = cap_set
        return cap_set

    def unregister_agent(self, agent_id: str) -> None:
        self._agent_caps.pop(agent_id, None)

    def get_capability_set(self, agent_id: str) -> Optional[CapabilitySet]:
        return self._agent_caps.get(agent_id)

    def create_capability(self, cap_type: CapabilityType, resource_pattern: str, agent_id: str,
                          current_tick: int = 0, granted_by: str = "kernel", delegatable: bool = False,
                          expires_at: Optional[int] = None) -> Capability:
        cap = Capability(
            cap_id=str(uuid.uuid4()), cap_type=cap_type, resource_pattern=resource_pattern,
            granted_to=agent_id, granted_by=granted_by, created_at=current_tick,
            expires_at=expires_at, delegatable=delegatable, revoked=False,
        )
        cap_set = self._agent_caps.get(agent_id)
        if cap_set is None:
            cap_set = self.register_agent(agent_id)
        cap_set.grant(cap)
        self._cap_index[cap.cap_id] = cap
        self._audit(tick=current_tick, agent_id=agent_id, action="GRANT", cap_type=cap_type,
                    resource=resource_pattern, allowed=True,
                    details=f"granted_by={granted_by!r}, delegatable={delegatable}", cap_id=cap.cap_id)
        return cap

    def check(self, agent_id: str, cap_type: CapabilityType, resource: str, current_tick: int = 0) -> bool:
        if agent_id == "kernel":
            return True
        cap_set = self._agent_caps.get(agent_id)
        allowed = (cap_set is not None and cap_set.has_capability(cap_type, resource, current_tick))
        action = "CHECK_ALLOW" if allowed else "CHECK_DENY"
        self._audit(tick=current_tick, agent_id=agent_id, action=action, cap_type=cap_type,
                    resource=resource, allowed=allowed)
        return allowed

    def require(self, agent_id: str, cap_type: CapabilityType, resource: str, current_tick: int = 0) -> None:
        if not self.check(agent_id, cap_type, resource, current_tick):
            raise CapabilityViolation(agent_id, cap_type, resource)

    def revoke(self, cap_id: str, current_tick: int = 0) -> bool:
        cap = self._cap_index.get(cap_id)
        if cap is None:
            return False
        cap.revoked = True
        cap_set = self._agent_caps.get(cap.granted_to)
        if cap_set:
            cap_set.revoke(cap_id)
        self._audit(tick=current_tick, agent_id=cap.granted_to, action="REVOKE", cap_type=cap.cap_type,
                    resource=cap.resource_pattern, allowed=True, details=f"cap_id={cap_id!r}", cap_id=cap_id)
        return True

    def revoke_all(self, agent_id: str, current_tick: int = 0) -> int:
        cap_set = self._agent_caps.get(agent_id)
        if cap_set is None:
            return 0
        count = 0
        for cap in cap_set.list(include_revoked=False):
            if self.revoke(cap.cap_id, current_tick):
                count += 1
        self._audit(tick=current_tick, agent_id=agent_id, action="REVOKE", cap_type=CapabilityType.ADMIN,
                    resource="*", allowed=True, details=f"revoke_all: {count} capabilities revoked")
        return count

    def expire_caps(self, current_tick: int) -> List[str]:
        expired_ids: List[str] = []
        for cap in list(self._cap_index.values()):
            if not cap.revoked and cap.expires_at is not None and current_tick >= cap.expires_at:
                cap.revoked = True
                expired_ids.append(cap.cap_id)
                self._audit(tick=current_tick, agent_id=cap.granted_to, action="EXPIRE",
                            cap_type=cap.cap_type, resource=cap.resource_pattern, allowed=False,
                            details=f"Expired at tick {current_tick}", cap_id=cap.cap_id)
        return expired_ids

    def delegate(self, grantor_id: str, cap_id: str, target_agent_id: str, current_tick: int = 0,
                 expires_at: Optional[int] = None, delegatable: bool = False) -> Optional[Capability]:
        grantor_set = self._agent_caps.get(grantor_id)
        if grantor_set is None:
            return None
        new_cap = grantor_set.delegate(cap_id=cap_id, target_agent_id=target_agent_id,
                                       current_tick=current_tick, grantor_id=grantor_id,
                                       expires_at=expires_at, delegatable=delegatable)
        if new_cap is None:
            self._audit(tick=current_tick, agent_id=grantor_id, action="DELEGATE",
                        cap_type=CapabilityType.ADMIN, resource="?", allowed=False,
                        details=f"Delegation of {cap_id!r} to {target_agent_id!r} failed")
            return None
        target_set = self._agent_caps.get(target_agent_id)
        if target_set is None:
            target_set = self.register_agent(target_agent_id)
        target_set.grant(new_cap)
        self._cap_index[new_cap.cap_id] = new_cap
        self._audit(tick=current_tick, agent_id=grantor_id, action="DELEGATE", cap_type=new_cap.cap_type,
                    resource=new_cap.resource_pattern, allowed=True,
                    details=f"delegated to {target_agent_id!r}", cap_id=new_cap.cap_id)
        return new_cap

    def _audit(self, tick: int, agent_id: str, action: str, cap_type: CapabilityType, resource: str,
               allowed: bool, details: str = "", cap_id: str = "") -> None:
        entry = AuditEntry(timestamp=tick, agent_id=agent_id, action=action, cap_type=cap_type,
                           resource=resource, allowed=allowed, details=details, cap_id=cap_id)
        self._audit_log.append(entry)

    def audit_log(self) -> List[AuditEntry]:
        return list(self._audit_log)

    def audit_log_for_agent(self, agent_id: str) -> List[AuditEntry]:
        return [e for e in self._audit_log if e.agent_id == agent_id]

    def stats(self) -> Dict[str, Any]:
        total_caps = sum(len(cs) for cs in self._agent_caps.values())
        active_caps = sum(sum(1 for c in cs.list() if not c.revoked) for cs in self._agent_caps.values())
        denials = sum(1 for e in self._audit_log if e.action == "CHECK_DENY")
        return {"registered_agents": len(self._agent_caps), "total_caps_issued": len(self._cap_index),
                "active_caps": active_caps, "total_caps_in_sets": total_caps,
                "audit_log_entries": len(self._audit_log), "access_denials": denials}

    def __repr__(self) -> str:
        return f"CapabilityManager(agents={len(self._agent_caps)}, caps_issued={len(self._cap_index)})"


@dataclass
class SecurityPolicy:
    name: str
    class_policies: Dict[str, List[Tuple[CapabilityType, str, bool]]] = field(default_factory=dict)
    default_caps: List[Tuple[CapabilityType, str, bool]] = field(default_factory=list)

    def get_caps_for_class(self, class_name: str) -> List[Tuple[CapabilityType, str, bool]]:
        class_caps = self.class_policies.get(class_name, [])
        return list(self.default_caps) + list(class_caps)

    def apply(self, manager: CapabilityManager, agent_id: str, class_name: str, current_tick: int = 0) -> List[Capability]:
        caps: List[Capability] = []
        for cap_type, resource_pattern, delegatable in self.get_caps_for_class(class_name):
            resolved = resource_pattern.replace("{self}", agent_id)
            cap = manager.create_capability(cap_type=cap_type, resource_pattern=resolved,
                                            agent_id=agent_id, current_tick=current_tick,
                                            granted_by="policy", delegatable=delegatable)
            caps.append(cap)
        return caps

    def __repr__(self) -> str:
        return f"SecurityPolicy(name={self.name!r}, classes={list(self.class_policies.keys())})"


DEFAULT_POLICY: SecurityPolicy = SecurityPolicy(
    name="DefaultBattousaiPolicy",
    class_policies={
        "CoordinatorAgent": [
            (CapabilityType.SPAWN,      "*",            True),
            (CapabilityType.MESSAGE,    "*",            False),
            (CapabilityType.FILE_WRITE, "/shared/*",    False),
            (CapabilityType.FILE_READ,  "/shared/*",    False),
            (CapabilityType.TOOL_USE,   "*",            False),
            (CapabilityType.MEMORY_READ,"*",            False),
        ],
        "WorkerAgent": [
            (CapabilityType.TOOL_USE,   "*",                    False),
            (CapabilityType.MESSAGE,    "coordinator*",         False),
            (CapabilityType.FILE_WRITE, "/agents/{self}/*",     False),
            (CapabilityType.FILE_READ,  "/agents/{self}/*",     False),
            (CapabilityType.FILE_READ,  "/shared/*",            False),
        ],
        "MonitorAgent": [
            (CapabilityType.MEMORY_READ, "*",   False),
            (CapabilityType.FILE_READ,   "*",   False),
            (CapabilityType.MESSAGE,     "*",   False),
        ],
    },
    default_caps=[(CapabilityType.MEMORY_READ, "global", False)],
)


def requires_capability(
    cap_type: CapabilityType,
    resource: str = "*",
    capability_manager_attr: str = "_cap_manager",
    agent_id_attr: str = "agent_id",
    tick_attr: str = "_current_tick",
) -> Callable:
    """
    Decorator that guards an Agent method with a capability check.

    Usage::

        class MyAgent(Agent):
            @requires_capability(CapabilityType.TOOL_USE, "web_search")
            def do_search(self, query: str) -> None:
                ...
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(self_agent, *args, **kwargs):
            mgr: Optional[CapabilityManager] = getattr(self_agent, capability_manager_attr, None)
            if mgr is None:
                # No capability manager — allow through (permissive mode)
                return fn(self_agent, *args, **kwargs)
            agent_id: str = getattr(self_agent, agent_id_attr, "unknown")
            tick: int = getattr(self_agent, tick_attr, 0)
            resolved_resource = resource
            if resource.startswith("__arg:"):
                arg_name = resource[len("__arg:"):]
                resolved_resource = kwargs.get(arg_name, resource)
            mgr.require(agent_id, cap_type, resolved_resource, tick)
            return fn(self_agent, *args, **kwargs)
        wrapper._requires_cap_type = cap_type
        wrapper._requires_resource = resource
        return wrapper
    return decorator
