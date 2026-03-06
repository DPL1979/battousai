"""
integrity.py — Battousai Memory Integrity Module
=================================================
Provides tamper detection via hash chains, TTL/versioned memory entries,
and tool registry integrity verification.

Design goals:
    - Zero external dependencies: stdlib only (hashlib, json, time, dataclasses, uuid)
    - Append-only hash chain: any modification to past entries is detectable
    - SecureMemoryStore: wraps writes with hash-chain integrity and supports TTL
      and per-key versioning
    - ToolRegistryVerifier: snapshot tool definitions at registration time and
      detect changes later
    - IntegrityAuditor: periodic audit runner producing human-readable reports

Hash chain construction:
    Each entry stores:
        data_hash     = sha256(raw_data)
        previous_hash = chain_hash of the preceding entry  (or "genesis")
        chain_hash    = sha256(data_hash + previous_hash)

    Because chain_hash is baked into the *next* entry's previous_hash, any
    in-place modification cascades a chain break that verify() detects.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class IntegrityError(Exception):
    """Base class for all integrity-related errors."""


class IntegrityViolation(IntegrityError):
    """Raised when a tampered or corrupted entry is detected during a read."""

    def __init__(self, message: str, index: Optional[int] = None) -> None:
        super().__init__(message)
        self.index = index


class EntryNotFoundError(IntegrityError):
    """Raised when a requested key or version does not exist."""


class EntryExpiredError(IntegrityError):
    """Raised when a requested entry has passed its TTL."""


# ---------------------------------------------------------------------------
# HashChain primitives
# ---------------------------------------------------------------------------

_GENESIS_HASH = "genesis"


@dataclass
class HashChainEntry:
    """One link in the append-only hash chain."""

    index: int
    data_hash: str            # sha256(raw data bytes)
    previous_hash: str        # chain_hash of the prior entry, or "genesis"
    chain_hash: str           # sha256(data_hash + previous_hash)
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


class HashChain:
    """
    An append-only chain of SHA-256 hashes.

    Each entry's chain_hash is derived from both the content hash and the
    previous entry's chain_hash, forming an unbreakable chain.  Any tampering
    with a past entry's data_hash, previous_hash, or chain_hash is detectable
    by verify().
    """

    def __init__(self, algorithm: str = "sha256") -> None:
        self._entries: List[HashChainEntry] = []
        self._algorithm = algorithm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash(self, data: bytes) -> str:
        h = hashlib.new(self._algorithm)
        h.update(data)
        return h.hexdigest()

    def _compute_chain_hash(self, data_hash: str, previous_hash: str) -> str:
        combined = (data_hash + previous_hash).encode("utf-8")
        return self._hash(combined)

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(
        self,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HashChainEntry:
        """Append data to the chain and return the new entry."""
        data_hash = self._hash(data)
        previous_hash = (
            self._entries[-1].chain_hash if self._entries else _GENESIS_HASH
        )
        chain_hash = self._compute_chain_hash(data_hash, previous_hash)
        entry = HashChainEntry(
            index=len(self._entries),
            data_hash=data_hash,
            previous_hash=previous_hash,
            chain_hash=chain_hash,
            timestamp=time.time(),
            metadata=metadata,
        )
        self._entries.append(entry)
        return entry

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_entry(self, index: int) -> bool:
        """
        Verify a single entry's internal consistency:
          1. chain_hash == sha256(data_hash + previous_hash)
          2. previous_hash matches the chain_hash of the preceding entry
             (or "genesis" for index 0).
        """
        if index < 0 or index >= len(self._entries):
            return False
        entry = self._entries[index]

        # Check the chain_hash is correct
        expected_chain = self._compute_chain_hash(entry.data_hash, entry.previous_hash)
        if entry.chain_hash != expected_chain:
            return False

        # Check linkage to prior entry
        expected_prev = (
            self._entries[index - 1].chain_hash if index > 0 else _GENESIS_HASH
        )
        return entry.previous_hash == expected_prev

    def verify(self) -> IntegrityReport:
        """
        Verify the entire chain from genesis to tip.

        Returns an IntegrityReport summarising validity, how many entries
        were verified, and where the first tampered entry is (if any).
        """
        total = len(self._entries)
        first_tampered: Optional[int] = None

        for i in range(total):
            if not self.verify_entry(i):
                first_tampered = i
                break

        verified = first_tampered if first_tampered is not None else total
        valid = first_tampered is None

        if valid:
            details = (
                f"Chain is valid. All {total} entr{'y' if total == 1 else 'ies'} verified."
            )
        else:
            details = (
                f"Chain integrity VIOLATED at index {first_tampered}. "
                f"{verified} of {total} entries verified before first failure."
            )

        return IntegrityReport(
            valid=valid,
            total_entries=total,
            verified_entries=verified,
            first_tampered_index=first_tampered,
            details=details,
        )

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> HashChainEntry:
        return self._entries[index]


# ---------------------------------------------------------------------------
# IntegrityReport
# ---------------------------------------------------------------------------

@dataclass
class IntegrityReport:
    """Summary returned by HashChain.verify()."""

    valid: bool
    total_entries: int
    verified_entries: int
    first_tampered_index: Optional[int]
    details: str


# ---------------------------------------------------------------------------
# SecureMemoryStore
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """
    A single versioned record in the SecureMemoryStore.

    The value is stored as a JSON-serialised bytes string; every write to the
    same key increments the version counter.  Optionally the entry carries an
    absolute expiry timestamp.
    """

    key: str
    value: Any                  # original Python object (kept for convenience)
    agent_id: str               # author
    version: int                # 1-based, auto-incremented per key
    created_at: float           # wall-clock seconds
    expires_at: Optional[float] # wall-clock seconds, or None for no expiry
    chain_index: int            # index inside the backing HashChain
    data_hash: str              # sha256 of the serialised value

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or time.time()) >= self.expires_at


class SecureMemoryStore:
    """
    A key-value memory store where every write is appended to a hash chain.

    Reads optionally verify the chain before returning data (default: True).
    Supports per-entry TTL (time-to-live in seconds) and automatic versioning:
    every write to the same key creates a new version; old versions are
    preserved in the history log.

    Raises IntegrityViolation if the chain is broken when verify=True.
    """

    def __init__(self, ttl_seconds: Optional[float] = None) -> None:
        self._chain: HashChain = HashChain()
        # current "live" entry per key
        self._store: Dict[str, MemoryEntry] = {}
        # full version history per key (newest last)
        self._history: Dict[str, List[MemoryEntry]] = {}
        self._ttl = ttl_seconds
        self._version_counters: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize(value: Any) -> bytes:
        """Serialise an arbitrary value to bytes for hashing/storage."""
        return json.dumps(value, sort_keys=True, default=str).encode("utf-8")

    def _compute_data_hash(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, key: str, value: Any, agent_id: str) -> str:
        """
        Store *value* under *key*, attributed to *agent_id*.

        Every call appends a new entry to the internal hash chain and
        increments the per-key version counter.  Returns the chain entry's
        chain_hash, which the caller can use as a receipt.
        """
        data = self._serialize(value)
        data_hash = self._compute_data_hash(data)

        # Version counter
        version = self._version_counters.get(key, 0) + 1
        self._version_counters[key] = version

        now = time.time()
        expires_at = (now + self._ttl) if self._ttl is not None else None

        # Append to chain
        chain_entry = self._chain.append(
            data,
            metadata={
                "key": key,
                "agent_id": agent_id,
                "version": version,
            },
        )

        entry = MemoryEntry(
            key=key,
            value=value,
            agent_id=agent_id,
            version=version,
            created_at=now,
            expires_at=expires_at,
            chain_index=chain_entry.index,
            data_hash=data_hash,
        )

        self._store[key] = entry
        self._history.setdefault(key, []).append(entry)
        return chain_entry.chain_hash

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def _check_integrity(self) -> None:
        """Verify the full chain; raise IntegrityViolation if broken."""
        report = self._chain.verify()
        if not report.valid:
            raise IntegrityViolation(
                f"Hash chain integrity violation detected at index "
                f"{report.first_tampered_index}. {report.details}",
                index=report.first_tampered_index,
            )

    def read(self, key: str, verify: bool = True) -> Any:
        """
        Return the current value for *key*.

        If *verify* is True (default), the entire hash chain is verified first.
        Raises:
            EntryNotFoundError  — key does not exist
            EntryExpiredError   — entry exists but has passed its TTL
            IntegrityViolation  — chain is broken (only when verify=True)
        """
        if verify:
            self._check_integrity()

        entry = self._store.get(key)
        if entry is None:
            raise EntryNotFoundError(f"Key {key!r} not found in SecureMemoryStore.")
        if entry.is_expired():
            raise EntryExpiredError(
                f"Key {key!r} has expired (expired at {entry.expires_at:.3f})."
            )
        return entry.value

    def read_verified(self, key: str) -> Tuple[Any, bool]:
        """
        Return ``(value, integrity_ok)`` without raising on chain failures.

        *integrity_ok* is False if either:
          - the hash chain is broken, or
          - the entry is expired.
        Always returns the stored value (even if integrity is False), or None
        if the key does not exist.
        """
        entry = self._store.get(key)
        if entry is None:
            return None, False

        if entry.is_expired():
            return entry.value, False

        report = self._chain.verify()
        return entry.value, report.valid

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, key: str, agent_id: str) -> bool:
        """
        Remove the live entry for *key* (history is preserved).

        Returns True if the key existed, False otherwise.
        The deletion is recorded in the hash chain so the audit trail is
        never broken.
        """
        if key not in self._store:
            return False

        # Record the deletion in the chain so audit trail is complete
        self._chain.append(
            self._serialize({"__deleted__": key, "by": agent_id}),
            metadata={"key": key, "agent_id": agent_id, "op": "delete"},
        )
        del self._store[key]
        return True

    # ------------------------------------------------------------------
    # Integrity check on entire store
    # ------------------------------------------------------------------

    def verify_all(self) -> IntegrityReport:
        """Return a full IntegrityReport for the backing hash chain."""
        return self._chain.verify()

    # ------------------------------------------------------------------
    # History / versioning
    # ------------------------------------------------------------------

    def get_history(self, key: str) -> List[MemoryEntry]:
        """Return all historical versions of *key*, oldest first."""
        return list(self._history.get(key, []))

    def get_version(self, key: str, version: int) -> Any:
        """
        Return the value for a specific historical *version* of *key*.

        Raises EntryNotFoundError if the key or version does not exist.
        """
        history = self._history.get(key)
        if not history:
            raise EntryNotFoundError(f"No history found for key {key!r}.")
        for entry in history:
            if entry.version == version:
                return entry.value
        raise EntryNotFoundError(
            f"Version {version} of key {key!r} not found "
            f"(available: {[e.version for e in history]})."
        )

    # ------------------------------------------------------------------
    # TTL expiry
    # ------------------------------------------------------------------

    def expire_stale(self) -> int:
        """
        Remove all live entries that have passed their TTL.

        Returns the number of entries removed.  Deletions are NOT appended
        to the chain (bulk GC does not need an individual audit trail entry).
        """
        now = time.time()
        expired_keys = [
            k for k, v in self._store.items() if v.is_expired(now)
        ]
        for k in expired_keys:
            del self._store[k]
        return len(expired_keys)


# ---------------------------------------------------------------------------
# ToolRegistryVerifier
# ---------------------------------------------------------------------------

@dataclass
class RegistrySignature:
    """Immutable snapshot of a tool registry at a point in time."""

    timestamp: float
    tool_hashes: Dict[str, str]   # tool_name -> hash
    registry_hash: str            # sha256 of all tool_hashes combined
    tool_count: int


class ToolRegistryVerifier:
    """
    Compute and verify SHA-256 hashes of tool definitions.

    Detects if a tool's callable, description, or permissions were
    modified after the registry was signed.
    """

    # ------------------------------------------------------------------
    # Hashing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_tool(spec: Any) -> str:
        """
        Produce a stable hash of a ToolSpec.

        Includes: name, description, the callable's qualified name,
        allowed_agents (sorted), rate_limit, rate_window, is_simulated.
        """
        fingerprint = {
            "name": spec.name,
            "description": spec.description,
            # Use module + qualname to detect callable swaps
            "callable_module": getattr(spec.callable, "__module__", ""),
            "callable_qualname": getattr(spec.callable, "__qualname__", ""),
            "allowed_agents": sorted(spec.allowed_agents),
            "rate_limit": spec.rate_limit,
            "rate_window": spec.rate_window,
            "is_simulated": spec.is_simulated,
        }
        raw = json.dumps(fingerprint, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _hash_registry(tool_hashes: Dict[str, str]) -> str:
        """Produce a single hash representing the entire registry snapshot."""
        combined = json.dumps(tool_hashes, sort_keys=True).encode("utf-8")
        return hashlib.sha256(combined).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sign_tool(self, spec: Any) -> str:
        """Return the SHA-256 hash of a single ToolSpec."""
        return self._hash_tool(spec)

    def sign_registry(self, tool_manager: Any) -> RegistrySignature:
        """
        Snapshot the current state of *tool_manager* and return a
        RegistrySignature that can be used later to detect changes.
        """
        tool_hashes: Dict[str, str] = {}
        for name in tool_manager.list_tools():
            spec = tool_manager.get_spec(name)
            tool_hashes[name] = self._hash_tool(spec)

        registry_hash = self._hash_registry(tool_hashes)

        return RegistrySignature(
            timestamp=time.time(),
            tool_hashes=tool_hashes,
            registry_hash=registry_hash,
            tool_count=len(tool_hashes),
        )

    def verify_registry(
        self,
        tool_manager: Any,
        signature: RegistrySignature,
    ) -> bool:
        """
        Return True if the current registry matches *signature*.

        False if any tool was added, removed, or modified.
        """
        current = self.sign_registry(tool_manager)
        return current.registry_hash == signature.registry_hash

    def diff_registry(
        self,
        tool_manager: Any,
        signature: RegistrySignature,
    ) -> List[str]:
        """
        Return a list of human-readable change descriptions between the
        current registry and *signature*.

        An empty list means no changes were detected.
        """
        changes: List[str] = []
        current_hashes: Dict[str, str] = {}
        for name in tool_manager.list_tools():
            spec = tool_manager.get_spec(name)
            current_hashes[name] = self._hash_tool(spec)

        old_names = set(signature.tool_hashes.keys())
        new_names = set(current_hashes.keys())

        for name in sorted(old_names - new_names):
            changes.append(f"REMOVED: tool {name!r} was in signature but is no longer registered.")

        for name in sorted(new_names - old_names):
            changes.append(f"ADDED: tool {name!r} is registered but was not in signature.")

        for name in sorted(old_names & new_names):
            if signature.tool_hashes[name] != current_hashes[name]:
                changes.append(
                    f"MODIFIED: tool {name!r} hash changed "
                    f"({signature.tool_hashes[name][:12]}… → {current_hashes[name][:12]}…)."
                )

        return changes


# ---------------------------------------------------------------------------
# AuditResult
# ---------------------------------------------------------------------------

@dataclass
class AuditResult:
    """Output of a single IntegrityAuditor run."""

    timestamp: float
    memory_report: IntegrityReport
    registry_signature: Optional[RegistrySignature]  # None if no verifier configured
    registry_valid: Optional[bool]
    registry_diff: List[str]
    passed: bool            # True if all checks passed
    summary: str


# ---------------------------------------------------------------------------
# IntegrityAuditor
# ---------------------------------------------------------------------------

class IntegrityAuditor:
    """
    Standalone audit runner.

    Checks memory store integrity and (optionally) tool registry integrity,
    then produces a human-readable report.

    Usage::

        auditor = IntegrityAuditor(secure_store, registry_verifier)
        # Take an initial registry snapshot
        auditor.snapshot_registry(tool_manager)
        # Later…
        result = auditor.run_audit(tool_manager)
        print(auditor.format_report(result))
    """

    def __init__(
        self,
        secure_store: SecureMemoryStore,
        registry_verifier: ToolRegistryVerifier,
    ) -> None:
        self._store = secure_store
        self._verifier = registry_verifier
        self._baseline_signature: Optional[RegistrySignature] = None

    def snapshot_registry(self, tool_manager: Any) -> RegistrySignature:
        """
        Take a baseline snapshot of the tool registry.

        Must be called before run_audit() will produce registry comparison
        results.
        """
        self._baseline_signature = self._verifier.sign_registry(tool_manager)
        return self._baseline_signature

    def run_audit(
        self,
        tool_manager: Optional[Any] = None,
    ) -> AuditResult:
        """
        Execute all integrity checks and return an AuditResult.

        If *tool_manager* is provided and a baseline snapshot has been taken
        (via snapshot_registry), the registry is also verified.
        """
        now = time.time()

        # Memory chain
        memory_report = self._store.verify_all()

        # Registry
        registry_signature: Optional[RegistrySignature] = None
        registry_valid: Optional[bool] = None
        registry_diff: List[str] = []

        if tool_manager is not None and self._baseline_signature is not None:
            registry_signature = self._baseline_signature
            registry_valid = self._verifier.verify_registry(
                tool_manager, self._baseline_signature
            )
            if not registry_valid:
                registry_diff = self._verifier.diff_registry(
                    tool_manager, self._baseline_signature
                )

        passed = memory_report.valid and (
            registry_valid is None or registry_valid
        )

        summary_parts: List[str] = []
        summary_parts.append(
            f"Memory chain: {'OK' if memory_report.valid else 'VIOLATED'} "
            f"({memory_report.verified_entries}/{memory_report.total_entries} entries verified)"
        )
        if registry_valid is not None:
            summary_parts.append(
                f"Tool registry: {'OK' if registry_valid else 'TAMPERED'}"
            )
            if registry_diff:
                summary_parts.append(f"  Changes: {len(registry_diff)}")

        summary = " | ".join(summary_parts)

        return AuditResult(
            timestamp=now,
            memory_report=memory_report,
            registry_signature=registry_signature,
            registry_valid=registry_valid,
            registry_diff=registry_diff,
            passed=passed,
            summary=summary,
        )

    def format_report(self, result: AuditResult) -> str:
        """Render an AuditResult as a human-readable multi-line string."""
        lines: List[str] = []
        sep = "=" * 60
        lines.append(sep)
        lines.append("BATTOUSAI INTEGRITY AUDIT REPORT")
        lines.append(f"Timestamp : {result.timestamp:.6f}")
        lines.append(f"Overall   : {'PASS' if result.passed else 'FAIL'}")
        lines.append("")

        # Memory section
        mr = result.memory_report
        lines.append("-- Memory Chain ----------------------------------------")
        lines.append(f"  Status   : {'VALID' if mr.valid else 'VIOLATED'}")
        lines.append(f"  Entries  : {mr.verified_entries}/{mr.total_entries} verified")
        if not mr.valid:
            lines.append(f"  First tamper at index: {mr.first_tampered_index}")
        lines.append(f"  Details  : {mr.details}")
        lines.append("")

        # Registry section
        if result.registry_valid is not None:
            lines.append("-- Tool Registry ----------------------------------------")
            lines.append(f"  Status   : {'VALID' if result.registry_valid else 'TAMPERED'}")
            if result.registry_signature is not None:
                lines.append(
                    f"  Snapshot : {result.registry_signature.tool_count} tools, "
                    f"taken at {result.registry_signature.timestamp:.6f}"
                )
            if result.registry_diff:
                lines.append(f"  Changes ({len(result.registry_diff)}):")
                for change in result.registry_diff:
                    lines.append(f"    - {change}")
            lines.append("")

        lines.append(f"Summary: {result.summary}")
        lines.append(sep)
        return "\n".join(lines)
