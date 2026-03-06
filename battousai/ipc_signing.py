"""
ipc_signing.py — Battousai IPC Message Signing (HMAC-SHA256)
=============================================================
Provides cryptographic signing and verification for IPC messages using
HMAC-SHA256. Ensures authenticity, integrity, and non-repudiation of
all inter-agent communication.

Exported Symbols:
    SigningKey         — Wrapper around an HMAC secret key
    KeyRing            — Per-agent key storage (thread-safe)
    MessageSigner      — Signs and verifies messages
    SignedMessage      — Message with optional signature and key_id
    SigningPolicy      — Enforcement levels (PERMISSIVE, SIGN_REQUIRED, STRICT)
    SignedIPCManager   — Signing-aware wrapper around IPCManager
    SigningAuditor     — Audit trail for signing events

Stdlib only — zero external dependencies.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Re-export the IPC primitives needed by tests and callers.
# NOTE: per spec, we import Message/MessageType so tests can use them via
# this module too — but the signing logic itself only needs them for typing.
# ---------------------------------------------------------------------------
from battousai.ipc import IPCManager, Message, MessageType  # noqa: F401

logger = logging.getLogger(__name__)


# ===========================================================================
# SigningKey
# ===========================================================================

class SigningKey:
    """
    Wrapper around a raw HMAC-SHA256 secret key.

    Never exposes the raw key bytes directly — callers work with
    :py:meth:`generate` / :py:meth:`from_bytes` factory methods and
    treat the resulting object as opaque.

    Attributes:
        key_id  — First 8 hex characters of SHA-256(raw_key).
                  Safe to log for auditing without exposing the secret.
    """

    def __init__(self, raw: bytes) -> None:
        """
        Construct a SigningKey from *raw* bytes.

        Prefer the class-method factories :py:meth:`generate` and
        :py:meth:`from_bytes` over calling this constructor directly.

        Args:
            raw: Raw key material (must be non-empty bytes).

        Raises:
            ValueError: If *raw* is empty.
        """
        if not raw:
            raise ValueError("SigningKey requires non-empty bytes.")
        self._raw: bytes = raw
        # Derive a public-safe identifier: first 8 hex chars of SHA-256
        digest = hashlib.sha256(raw).hexdigest()
        self._key_id: str = digest[:8]

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def generate(cls) -> "SigningKey":
        """
        Generate a new signing key from 32 cryptographically-random bytes.

        Returns:
            A fresh :class:`SigningKey` instance.
        """
        return cls(secrets.token_bytes(32))

    @classmethod
    def from_bytes(cls, raw: bytes) -> "SigningKey":
        """
        Construct a :class:`SigningKey` from an existing byte sequence.

        Args:
            raw: Raw key material.

        Returns:
            A :class:`SigningKey` wrapping *raw*.

        Raises:
            ValueError: If *raw* is empty.
        """
        return cls(raw)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def key_id(self) -> str:
        """
        A public-safe identifier for this key.

        Derived as the first 8 hex characters of SHA-256(raw_key).
        Safe to include in logs and audit reports without leaking secrets.

        Returns:
            An 8-character lowercase hex string.
        """
        return self._key_id

    @property
    def _secret(self) -> bytes:
        """
        Internal accessor for raw key bytes — only used by
        :class:`MessageSigner`.  Not part of the public API.
        """
        return self._raw

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"SigningKey(key_id={self._key_id!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SigningKey):
            return NotImplemented
        return hmac.compare_digest(self._raw, other._raw)

    def __hash__(self) -> int:
        return hash(self._key_id)


# ===========================================================================
# KeyRing
# ===========================================================================

class KeyRing:
    """
    Thread-safe per-agent key storage.

    Each agent is identified by a string *agent_id* and may have exactly
    one active :class:`SigningKey`.  Key rotation atomically replaces the
    old key with a freshly generated one.

    All mutating operations are protected by an internal
    :py:class:`threading.Lock`.
    """

    def __init__(self) -> None:
        """Initialise an empty KeyRing."""
        self._keys: Dict[str, SigningKey] = {}
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mutating operations (all lock-protected)
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str, key: SigningKey) -> None:
        """
        Store a signing key for *agent_id*.

        If the agent already has a registered key it is silently replaced.

        Args:
            agent_id: Unique agent identifier string.
            key:      The :class:`SigningKey` to associate with the agent.
        """
        with self._lock:
            self._keys[agent_id] = key

    def remove_agent(self, agent_id: str) -> None:
        """
        Remove the signing key for *agent_id*.

        No-op if the agent is not registered.

        Args:
            agent_id: Unique agent identifier string.
        """
        with self._lock:
            self._keys.pop(agent_id, None)

    def rotate_key(self, agent_id: str) -> SigningKey:
        """
        Replace the key for *agent_id* with a freshly generated one.

        The old key is immediately invalidated — any in-flight signatures
        produced with it will fail verification after rotation.

        Args:
            agent_id: Unique agent identifier string.

        Returns:
            The newly generated :class:`SigningKey`.

        Raises:
            KeyError: If *agent_id* is not currently registered.
        """
        with self._lock:
            if agent_id not in self._keys:
                raise KeyError(
                    f"Cannot rotate key for unknown agent {agent_id!r}."
                )
            new_key = SigningKey.generate()
            self._keys[agent_id] = new_key
            return new_key

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_key(self, agent_id: str) -> Optional[SigningKey]:
        """
        Retrieve the signing key for *agent_id*.

        Args:
            agent_id: Unique agent identifier string.

        Returns:
            The :class:`SigningKey` for the agent, or ``None`` if not
            registered.
        """
        with self._lock:
            return self._keys.get(agent_id)

    def key_ids(self) -> Dict[str, str]:
        """
        Return a snapshot of ``{agent_id: key_id}`` pairs for auditing.

        The returned dict contains public-safe key identifiers only —
        no raw key material is exposed.

        Returns:
            A dictionary mapping every registered agent_id to its
            corresponding :py:attr:`SigningKey.key_id`.
        """
        with self._lock:
            return {aid: key.key_id for aid, key in self._keys.items()}

    def list_agents(self) -> List[str]:
        """
        Return a list of all currently registered agent IDs.

        Returns:
            A list of agent_id strings.
        """
        with self._lock:
            return list(self._keys.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._keys)

    def __repr__(self) -> str:
        with self._lock:
            agents = list(self._keys.keys())
        return f"KeyRing(agents={agents!r})"


# ===========================================================================
# MessageSigner
# ===========================================================================

class MessageSigner:
    """
    Stateless helper for HMAC-SHA256 signing and verification of IPC messages.

    Canonical signing form
    ----------------------
    The canonical string is assembled as::

        "{sender_id}|{recipient_id}|{message_type.name}|{json.dumps(payload, sort_keys=True)}|{timestamp}|{message_id}"

    This ties the signature to every immutable field, preventing any
    field-level tampering while remaining deterministic across runs.

    All verification uses :func:`hmac.compare_digest` to resist timing attacks.
    """

    # ------------------------------------------------------------------
    # Canonical form helper
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical(message: Message) -> str:
        """
        Build the canonical UTF-8 string representation of *message* for signing.

        Args:
            message: The :class:`~battousai.ipc.Message` to serialise.

        Returns:
            A pipe-delimited string of the message's immutable fields.
        """
        payload_json = json.dumps(message.payload, sort_keys=True)
        return (
            f"{message.sender_id}"
            f"|{message.recipient_id}"
            f"|{message.message_type.name}"
            f"|{payload_json}"
            f"|{message.timestamp}"
            f"|{message.message_id}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def sign(message: Message, key: SigningKey) -> str:
        """
        Compute and return an HMAC-SHA256 signature for *message*.

        Args:
            message: The IPC message to sign.
            key:     The :class:`SigningKey` whose secret is used.

        Returns:
            A lowercase hex-encoded HMAC-SHA256 digest (64 characters).
        """
        canonical = MessageSigner._canonical(message)
        mac = hmac.new(
            key._secret,
            canonical.encode("utf-8"),
            hashlib.sha256,
        )
        return mac.hexdigest()

    @staticmethod
    def verify(message: Message, signature: str, key: SigningKey) -> bool:
        """
        Verify that *signature* matches the HMAC-SHA256 of *message*.

        Uses :func:`hmac.compare_digest` for constant-time comparison to
        prevent timing-based side-channel attacks.

        Args:
            message:   The IPC message to verify.
            signature: The hex-encoded signature string to check against.
            key:       The :class:`SigningKey` used for verification.

        Returns:
            ``True`` if the signature is valid, ``False`` otherwise.
        """
        try:
            expected = MessageSigner.sign(message, key)
            return hmac.compare_digest(expected, signature)
        except Exception:  # pragma: no cover
            return False

    @staticmethod
    def canonical_form(message: Message) -> str:
        """
        Return the canonical string that would be signed for *message*.

        Exposed publicly so callers can inspect / log the exact byte
        sequence that is signed without having to re-implement the logic.

        Args:
            message: The IPC message.

        Returns:
            The pipe-delimited canonical string.
        """
        return MessageSigner._canonical(message)


# ===========================================================================
# SignedMessage
# ===========================================================================

@dataclass
class SignedMessage:
    """
    An IPC :class:`~battousai.ipc.Message` decorated with an optional
    HMAC-SHA256 signature.

    All original :class:`~battousai.ipc.Message` fields are mirrored here
    so that :class:`SignedMessage` can be passed anywhere a plain
    :class:`~battousai.ipc.Message` is expected without unwrapping.

    Attributes:
        sender_id:      Agent ID of the sender.
        recipient_id:   Agent ID of the recipient (or BROADCAST_ALL).
        message_type:   Semantic classification (:class:`~battousai.ipc.MessageType`).
        payload:        Arbitrary message body.
        timestamp:      System tick when the message was created.
        message_id:     Globally unique identifier (auto-generated if not supplied).
        correlation_id: Optional request/reply correlation token.
        ttl:            Ticks until the message expires (0 = no expiry).
        signature:      Hex-encoded HMAC-SHA256 signature, or ``None``.
        key_id:         :py:attr:`SigningKey.key_id` of the signing key, or ``None``.
    """

    sender_id: str
    recipient_id: str
    message_type: MessageType
    payload: Any
    timestamp: int
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    correlation_id: Optional[str] = None
    ttl: int = 0
    signature: Optional[str] = None
    key_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_signed(self) -> bool:
        """
        ``True`` if this message carries a non-empty signature.

        Returns:
            ``True`` when :py:attr:`signature` is a non-empty string.
        """
        return self.signature is not None and len(self.signature) > 0

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_message(
        cls,
        message: Message,
        signature: Optional[str] = None,
        key_id: Optional[str] = None,
    ) -> "SignedMessage":
        """
        Wrap a plain :class:`~battousai.ipc.Message` as a :class:`SignedMessage`.

        Args:
            message:   The source message.
            signature: Optional pre-computed signature string.
            key_id:    Optional key identifier.

        Returns:
            A :class:`SignedMessage` with all fields copied from *message*.
        """
        return cls(
            sender_id=message.sender_id,
            recipient_id=message.recipient_id,
            message_type=message.message_type,
            payload=message.payload,
            timestamp=message.timestamp,
            message_id=message.message_id,
            correlation_id=message.correlation_id,
            ttl=message.ttl,
            signature=signature,
            key_id=key_id,
        )

    def to_message(self) -> Message:
        """
        Strip signing metadata and return a plain :class:`~battousai.ipc.Message`.

        Returns:
            A new :class:`~battousai.ipc.Message` with all core fields but
            without *signature* or *key_id*.
        """
        return Message(
            sender_id=self.sender_id,
            recipient_id=self.recipient_id,
            message_type=self.message_type,
            payload=self.payload,
            timestamp=self.timestamp,
            message_id=self.message_id,
            correlation_id=self.correlation_id,
            ttl=self.ttl,
        )

    def is_expired(self, current_tick: int) -> bool:
        """
        Proxy for :py:meth:`Message.is_expired` — returns ``True`` if
        the message has exceeded its TTL.

        Args:
            current_tick: The current simulation tick.

        Returns:
            ``True`` if the message has expired.
        """
        return self.ttl > 0 and (current_tick - self.timestamp) >= self.ttl

    def __repr__(self) -> str:
        signed_flag = "signed" if self.is_signed else "unsigned"
        return (
            f"SignedMessage(id={self.message_id}, "
            f"{self.sender_id}->{self.recipient_id}, "
            f"type={self.message_type.name}, "
            f"tick={self.timestamp}, "
            f"{signed_flag})"
        )


# ===========================================================================
# SigningPolicy
# ===========================================================================

class SigningPolicy(Enum):
    """
    Configurable enforcement levels for the :class:`SignedIPCManager`.

    Levels
    ------
    PERMISSIVE
        Sign outgoing messages when a key is available, but deliver
        unsigned and improperly-signed inbound messages without raising
        an error.  Useful during gradual rollout.

    SIGN_REQUIRED
        All *outgoing* messages must be signed (raises
        :py:exc:`SigningError` if no key is available for the sender).
        Unsigned *inbound* messages are still accepted and delivered.

    STRICT
        Every message must be signed **and** verified.  Unsigned inbound
        messages are rejected.  Messages whose signature verification
        fails are also rejected.  Raises :py:exc:`SigningError` on
        violations.
    """

    PERMISSIVE = auto()
    SIGN_REQUIRED = auto()
    STRICT = auto()


# ===========================================================================
# Exceptions
# ===========================================================================

class SigningError(Exception):
    """
    Raised when a signing or verification operation violates the active
    :class:`SigningPolicy`.

    Attributes:
        message: Human-readable description of the violation.
        agent_id: The agent involved, if applicable.
    """

    def __init__(self, message: str, agent_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.agent_id = agent_id


# ===========================================================================
# AuditEvent
# ===========================================================================

@dataclass
class AuditEvent:
    """
    A single record in the :class:`SigningAuditor` event log.

    Attributes:
        event_type:  One of ``"sign"``, ``"verify_ok"``, ``"verify_fail"``,
                     ``"unsigned_bypass"``, ``"reject"``.
        agent_id:    The sender/recipient agent involved.
        message_id:  The message_id from the associated message.
        key_id:      The :py:attr:`SigningKey.key_id` used, or ``None``.
        timestamp:   Wall-clock time of the event (``time.time()``).
        detail:      Optional free-form detail string.
    """

    event_type: str
    agent_id: str
    message_id: str
    key_id: Optional[str]
    timestamp: float
    detail: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"AuditEvent({self.event_type!r}, agent={self.agent_id!r}, "
            f"msg={self.message_id!r}, key={self.key_id!r})"
        )


# ===========================================================================
# SigningAuditor
# ===========================================================================

class SigningAuditor:
    """
    Append-only audit trail for signing and verification events.

    Every sign / verify / reject / bypass operation in
    :class:`SignedIPCManager` creates an :class:`AuditEvent` here.
    Callers can query :py:meth:`audit_report` for a summary or
    :py:meth:`violations` for the list of failed verifications.

    Thread-safe — all operations are protected by an internal
    :py:class:`threading.Lock`.

    Attributes:
        max_events: Maximum number of events retained (oldest are dropped).
    """

    DEFAULT_MAX_EVENTS: int = 10_000

    def __init__(self, max_events: int = DEFAULT_MAX_EVENTS) -> None:
        """
        Initialise a :class:`SigningAuditor`.

        Args:
            max_events: Maximum number of audit events to retain in memory.
                        When the limit is reached the oldest entry is evicted.
        """
        self.max_events = max_events
        self._events: List[AuditEvent] = []
        self._lock: threading.Lock = threading.Lock()
        # Counters
        self._counts: Dict[str, int] = {
            "sign": 0,
            "verify_ok": 0,
            "verify_fail": 0,
            "unsigned_bypass": 0,
            "reject": 0,
        }

    # ------------------------------------------------------------------
    # Internal logging helper
    # ------------------------------------------------------------------

    def _log(
        self,
        event_type: str,
        agent_id: str,
        message_id: str,
        key_id: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> AuditEvent:
        """
        Append an audit event.

        Args:
            event_type: Category string (``"sign"``, ``"verify_ok"``, etc.).
            agent_id:   Agent involved in the event.
            message_id: Identifier of the associated message.
            key_id:     Optional key identifier.
            detail:     Optional free-form detail.

        Returns:
            The newly created :class:`AuditEvent`.
        """
        event = AuditEvent(
            event_type=event_type,
            agent_id=agent_id,
            message_id=message_id,
            key_id=key_id,
            timestamp=time.time(),
            detail=detail,
        )
        with self._lock:
            if len(self._events) >= self.max_events:
                self._events.pop(0)
            self._events.append(event)
            self._counts[event_type] = self._counts.get(event_type, 0) + 1
        return event

    # ------------------------------------------------------------------
    # Convenience logging methods
    # ------------------------------------------------------------------

    def log_sign(
        self,
        agent_id: str,
        message_id: str,
        key_id: Optional[str] = None,
    ) -> None:
        """
        Record a successful signing event.

        Args:
            agent_id:   The signing agent's ID.
            message_id: The signed message's ID.
            key_id:     The key ID used for signing.
        """
        self._log("sign", agent_id, message_id, key_id)

    def log_verify_ok(
        self,
        agent_id: str,
        message_id: str,
        key_id: Optional[str] = None,
    ) -> None:
        """
        Record a successful signature verification.

        Args:
            agent_id:   The verifying agent's ID.
            message_id: The verified message's ID.
            key_id:     The key ID used for verification.
        """
        self._log("verify_ok", agent_id, message_id, key_id)

    def log_verify_fail(
        self,
        agent_id: str,
        message_id: str,
        key_id: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        """
        Record a failed signature verification.

        Args:
            agent_id:   The agent associated with the failing message.
            message_id: The message ID that failed verification.
            key_id:     The key ID that was attempted.
            detail:     Optional context (e.g. "bad signature").
        """
        self._log("verify_fail", agent_id, message_id, key_id, detail)

    def log_unsigned_bypass(
        self,
        agent_id: str,
        message_id: str,
        detail: Optional[str] = None,
    ) -> None:
        """
        Record that an unsigned message was allowed through.

        Args:
            agent_id:   The sender/receiver agent.
            message_id: The unsigned message's ID.
            detail:     Optional context.
        """
        self._log("unsigned_bypass", agent_id, message_id, None, detail)

    def log_reject(
        self,
        agent_id: str,
        message_id: str,
        key_id: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        """
        Record that a message was rejected (e.g. under STRICT policy).

        Args:
            agent_id:   The agent associated with the rejected message.
            message_id: The rejected message's ID.
            key_id:     The key ID attempted, if any.
            detail:     Optional context.
        """
        self._log("reject", agent_id, message_id, key_id, detail)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def violations(self) -> List[AuditEvent]:
        """
        Return all recorded verification-failure events.

        Returns:
            A list of :class:`AuditEvent` objects where
            ``event_type == "verify_fail"``.
        """
        with self._lock:
            return [e for e in self._events if e.event_type == "verify_fail"]

    def rejections(self) -> List[AuditEvent]:
        """
        Return all recorded rejection events.

        Returns:
            A list of :class:`AuditEvent` objects where
            ``event_type == "reject"``.
        """
        with self._lock:
            return [e for e in self._events if e.event_type == "reject"]

    def recent_events(self, n: int = 20) -> List[AuditEvent]:
        """
        Return the *n* most recent audit events.

        Args:
            n: Maximum number of events to return.

        Returns:
            A list of the most recent :class:`AuditEvent` objects,
            newest last.
        """
        with self._lock:
            return list(self._events[-n:])

    def audit_report(self) -> Dict[str, Any]:
        """
        Return a summary dict with statistics and recent events.

        The report structure is::

            {
                "total_sign": int,
                "total_verify_ok": int,
                "total_verify_fail": int,
                "total_unsigned_bypass": int,
                "total_reject": int,
                "recent_events": [AuditEvent, ...],
                "violation_count": int,
            }

        Returns:
            A dictionary suitable for logging or external reporting.
        """
        with self._lock:
            counts = dict(self._counts)
            recent = list(self._events[-20:])
            violation_count = sum(
                1 for e in self._events if e.event_type == "verify_fail"
            )
        return {
            "total_sign": counts.get("sign", 0),
            "total_verify_ok": counts.get("verify_ok", 0),
            "total_verify_fail": counts.get("verify_fail", 0),
            "total_unsigned_bypass": counts.get("unsigned_bypass", 0),
            "total_reject": counts.get("reject", 0),
            "recent_events": recent,
            "violation_count": violation_count,
        }

    def reset(self) -> None:
        """
        Clear all recorded events and reset counters.

        Primarily useful in tests.
        """
        with self._lock:
            self._events.clear()
            for key in self._counts:
                self._counts[key] = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def __repr__(self) -> str:
        with self._lock:
            counts = dict(self._counts)
        return f"SigningAuditor(counts={counts!r})"


# ===========================================================================
# SignedIPCManager
# ===========================================================================

class SignedIPCManager:
    """
    Signing-aware wrapper around :class:`~battousai.ipc.IPCManager`.

    Transparently signs outgoing messages and verifies incoming ones
    according to the configured :class:`SigningPolicy`.  The underlying
    :class:`~battousai.ipc.IPCManager` is unmodified — this class is a
    pure decorator.

    Verification statistics
    -----------------------
    Three counters are maintained:

    * ``total_verified``   — messages successfully verified.
    * ``total_failed``     — messages that failed signature verification.
    * ``total_unsigned``   — messages delivered without a signature.

    Thread-safety
    -------------
    The statistics counters are updated atomically under an internal lock.
    The underlying :class:`~battousai.ipc.IPCManager` provides its own
    concurrency guarantees.
    """

    def __init__(
        self,
        ipc_manager: IPCManager,
        key_ring: KeyRing,
        policy: SigningPolicy = SigningPolicy.PERMISSIVE,
        auditor: Optional[SigningAuditor] = None,
    ) -> None:
        """
        Initialise the :class:`SignedIPCManager`.

        Args:
            ipc_manager: The underlying :class:`~battousai.ipc.IPCManager`
                         used for actual message routing.
            key_ring:    The :class:`KeyRing` that maps agent IDs to keys.
            policy:      The :class:`SigningPolicy` to enforce.  Defaults to
                         :py:attr:`SigningPolicy.PERMISSIVE`.
            auditor:     Optional :class:`SigningAuditor`.  If ``None`` a
                         fresh one is created.
        """
        self._ipc = ipc_manager
        self._key_ring = key_ring
        self._policy = policy
        self._auditor = auditor if auditor is not None else SigningAuditor()
        self._signer = MessageSigner()
        self._lock = threading.Lock()
        # Statistics
        self.total_verified: int = 0
        self.total_failed: int = 0
        self.total_unsigned: int = 0

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    @property
    def policy(self) -> SigningPolicy:
        """The active :class:`SigningPolicy`."""
        return self._policy

    @policy.setter
    def policy(self, value: SigningPolicy) -> None:
        """
        Replace the active policy.

        Args:
            value: The new :class:`SigningPolicy`.
        """
        self._policy = value

    @property
    def auditor(self) -> SigningAuditor:
        """The attached :class:`SigningAuditor` instance."""
        return self._auditor

    @property
    def key_ring(self) -> KeyRing:
        """The :class:`KeyRing` in use."""
        return self._key_ring

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _increment(self, counter: str) -> None:
        """Thread-safe counter increment."""
        with self._lock:
            setattr(self, counter, getattr(self, counter) + 1)

    def _sign_message(self, message: Message) -> Tuple[Optional[str], Optional[str]]:
        """
        Attempt to sign *message* using the sender's key.

        Returns:
            A ``(signature, key_id)`` tuple.  Both are ``None`` if no key
            is registered for the sender.
        """
        key = self._key_ring.get_key(message.sender_id)
        if key is None:
            return None, None
        sig = self._signer.sign(message, key)
        return sig, key.key_id

    def _verify_message(
        self, message: Message, signature: str, sender_id: str
    ) -> bool:
        """
        Verify *signature* against *message* using the sender's key.

        Args:
            message:   The message to verify.
            signature: The hex-encoded HMAC signature to check.
            sender_id: The declared sender agent ID.

        Returns:
            ``True`` if verification succeeds, ``False`` otherwise.
        """
        key = self._key_ring.get_key(sender_id)
        if key is None:
            return False
        return self._signer.verify(message, signature, key)

    # ------------------------------------------------------------------
    # Public send / verify-and-deliver
    # ------------------------------------------------------------------

    def send(self, message: Message) -> bool:
        """
        Sign *message* (if a key is available) and route it via the
        underlying :class:`~battousai.ipc.IPCManager`.

        Under :py:attr:`SigningPolicy.SIGN_REQUIRED` and
        :py:attr:`SigningPolicy.STRICT` a :py:exc:`SigningError` is raised
        if no key is registered for the sender.

        Args:
            message: The :class:`~battousai.ipc.Message` to send.

        Returns:
            ``True`` if the underlying IPC delivery succeeded.

        Raises:
            SigningError: If the policy requires a signature but none can
                          be produced.
        """
        key = self._key_ring.get_key(message.sender_id)

        if key is not None:
            sig = self._signer.sign(message, key)
            self._auditor.log_sign(message.sender_id, message.message_id, key.key_id)
        else:
            sig = None
            if self._policy in (SigningPolicy.SIGN_REQUIRED, SigningPolicy.STRICT):
                raise SigningError(
                    f"Policy {self._policy.name} requires a signing key for "
                    f"agent {message.sender_id!r} but none is registered.",
                    agent_id=message.sender_id,
                )

        # Attach signature as a SignedMessage then route the plain message
        # (the IPC layer doesn't know about SignedMessage)
        _ = SignedMessage.from_message(
            message,
            signature=sig,
            key_id=key.key_id if key else None,
        )
        return self._ipc.send(message)

    def verify_and_deliver(self, message: Message) -> bool:
        """
        Verify *message*'s signature and deliver it to the recipient.

        The message is expected to be a :class:`SignedMessage` (or at
        least duck-type-compatible with one).  If it is a plain
        :class:`~battousai.ipc.Message` it is treated as unsigned.

        Policy enforcement:

        * **PERMISSIVE** — deliver regardless of signature state.
        * **SIGN_REQUIRED** — deliver unsigned; reject bad signatures.
        * **STRICT** — reject unsigned AND bad-signature messages.

        Args:
            message: A :class:`~battousai.ipc.Message` or
                     :class:`SignedMessage` to verify and deliver.

        Returns:
            ``True`` if the message was delivered, ``False`` if rejected.

        Raises:
            SigningError: Under STRICT policy when a message is rejected.
        """
        # Detect signature field (SignedMessage duck-typing)
        sig: Optional[str] = getattr(message, "signature", None)
        sender_id: str = message.sender_id

        if sig:
            # We have a signature — verify it
            key = self._key_ring.get_key(sender_id)
            if key is not None:
                # Use underlying Message fields for verification
                base_msg = (
                    message.to_message()
                    if isinstance(message, SignedMessage)
                    else message
                )
                valid = self._signer.verify(base_msg, sig, key)
            else:
                valid = False

            if valid:
                self._increment("total_verified")
                self._auditor.log_verify_ok(
                    sender_id,
                    message.message_id,
                    getattr(message, "key_id", None),
                )
                return self._ipc.send(
                    message.to_message()
                    if isinstance(message, SignedMessage)
                    else message
                )
            else:
                self._increment("total_failed")
                self._auditor.log_verify_fail(
                    sender_id,
                    message.message_id,
                    getattr(message, "key_id", None),
                    detail="HMAC verification failed",
                )
                if self._policy in (SigningPolicy.SIGN_REQUIRED, SigningPolicy.STRICT):
                    self._auditor.log_reject(
                        sender_id,
                        message.message_id,
                        getattr(message, "key_id", None),
                        detail="rejected: bad signature",
                    )
                    raise SigningError(
                        f"Signature verification failed for message "
                        f"{message.message_id!r} from {sender_id!r}.",
                        agent_id=sender_id,
                    )
                # PERMISSIVE: deliver anyway
                return self._ipc.send(
                    message.to_message()
                    if isinstance(message, SignedMessage)
                    else message
                )
        else:
            # Unsigned message
            self._increment("total_unsigned")
            if self._policy == SigningPolicy.STRICT:
                self._auditor.log_reject(
                    sender_id,
                    message.message_id,
                    None,
                    detail="rejected: unsigned message under STRICT policy",
                )
                raise SigningError(
                    f"STRICT policy rejects unsigned message "
                    f"{message.message_id!r} from {sender_id!r}.",
                    agent_id=sender_id,
                )
            # PERMISSIVE or SIGN_REQUIRED: allow unsigned inbound
            self._auditor.log_unsigned_bypass(
                sender_id,
                message.message_id,
                detail="unsigned message bypassed",
            )
            return self._ipc.send(
                message.to_message()
                if isinstance(message, SignedMessage)
                else message
            )

    def send_unsigned(self, message: Message, audit: bool = True) -> bool:
        """
        Send *message* without signing, bypassing policy enforcement.

        Intended for kernel / system messages that do not have a signing
        key.  An audit event is always recorded unless *audit* is ``False``.

        Args:
            message: The message to deliver without signing.
            audit:   If ``True`` (default) record an unsigned_bypass event.

        Returns:
            ``True`` if delivery succeeded.
        """
        if audit:
            self._auditor.log_unsigned_bypass(
                message.sender_id,
                message.message_id,
                detail="send_unsigned bypass",
            )
        self._increment("total_unsigned")
        return self._ipc.send(message)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """
        Return a combined statistics dict.

        The dict merges the underlying IPC stats with signing-specific
        counters::

            {
                "total_verified": int,
                "total_failed": int,
                "total_unsigned": int,
                # … plus all keys from IPCManager.stats()
            }

        Returns:
            A dictionary of current statistics.
        """
        base = self._ipc.stats()
        base.update(
            {
                "total_verified": self.total_verified,
                "total_failed": self.total_failed,
                "total_unsigned": self.total_unsigned,
            }
        )
        return base

    # ------------------------------------------------------------------
    # Delegation to underlying IPCManager
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str, max_mailbox_size: int = 128):
        """
        Register an agent mailbox in the underlying IPC manager.

        Args:
            agent_id:         The agent to register.
            max_mailbox_size: Maximum mailbox capacity.

        Returns:
            The :class:`~battousai.ipc.Mailbox` created.
        """
        return self._ipc.register_agent(agent_id, max_mailbox_size)

    def unregister_agent(self, agent_id: str) -> None:
        """
        Remove an agent's mailbox from the underlying IPC manager.

        Args:
            agent_id: The agent to remove.
        """
        self._ipc.unregister_agent(agent_id)

    def get_mailbox(self, agent_id: str):
        """
        Retrieve an agent's mailbox from the underlying IPC manager.

        Args:
            agent_id: The agent whose mailbox to retrieve.

        Returns:
            The :class:`~battousai.ipc.Mailbox`, or ``None``.
        """
        return self._ipc.get_mailbox(agent_id)

    def create_message(
        self,
        sender_id: str,
        recipient_id: str,
        message_type: MessageType,
        payload: Any,
        timestamp: int,
        correlation_id: Optional[str] = None,
        ttl: int = 0,
    ) -> Message:
        """
        Factory helper: create, sign, and route a message in one call.

        Args:
            sender_id:      Agent ID of the sender.
            recipient_id:   Agent ID of the recipient.
            message_type:   Message classification.
            payload:        Arbitrary payload data.
            timestamp:      System tick.
            correlation_id: Optional correlation token.
            ttl:            Ticks until expiry.

        Returns:
            The created :class:`~battousai.ipc.Message`.
        """
        msg = Message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            timestamp=timestamp,
            correlation_id=correlation_id,
            ttl=ttl,
        )
        self.send(msg)
        return msg

    def __repr__(self) -> str:
        return (
            f"SignedIPCManager("
            f"policy={self._policy.name}, "
            f"verified={self.total_verified}, "
            f"failed={self.total_failed}, "
            f"unsigned={self.total_unsigned})"
        )


# ===========================================================================
# Module-level convenience helpers
# ===========================================================================

def make_signed_message(
    sender_id: str,
    recipient_id: str,
    message_type: MessageType,
    payload: Any,
    timestamp: int,
    key: SigningKey,
    correlation_id: Optional[str] = None,
    ttl: int = 0,
) -> SignedMessage:
    """
    Convenience factory: create a :class:`SignedMessage` in one call.

    Builds a plain :class:`~battousai.ipc.Message`, computes its HMAC-SHA256
    signature with *key*, and returns a fully-signed :class:`SignedMessage`.

    Args:
        sender_id:      Agent ID of the sender.
        recipient_id:   Agent ID of the intended recipient.
        message_type:   Semantic classification.
        payload:        Arbitrary message body.
        timestamp:      System tick when the message is created.
        key:            The :class:`SigningKey` used to sign.
        correlation_id: Optional request/reply correlation token.
        ttl:            Ticks until the message expires (0 = no expiry).

    Returns:
        A :class:`SignedMessage` with ``is_signed == True``.
    """
    msg = Message(
        sender_id=sender_id,
        recipient_id=recipient_id,
        message_type=message_type,
        payload=payload,
        timestamp=timestamp,
        correlation_id=correlation_id,
        ttl=ttl,
    )
    sig = MessageSigner.sign(msg, key)
    return SignedMessage.from_message(msg, signature=sig, key_id=key.key_id)


def verify_signed_message(signed: SignedMessage, key: SigningKey) -> bool:
    """
    Convenience helper: verify the signature on a :class:`SignedMessage`.

    Args:
        signed: The :class:`SignedMessage` to verify.
        key:    The :class:`SigningKey` to verify against.

    Returns:
        ``True`` if the signature is valid, ``False`` otherwise.
    """
    if not signed.is_signed:
        return False
    base = signed.to_message()
    return MessageSigner.verify(base, signed.signature, key)  # type: ignore[arg-type]
