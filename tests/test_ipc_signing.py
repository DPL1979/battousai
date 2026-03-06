"""
test_ipc_signing.py — Comprehensive tests for battousai/ipc_signing.py
=======================================================================
Tests cover:
- SigningKey generation, from_bytes, key_id derivation
- KeyRing CRUD, rotation, thread safety
- MessageSigner canonical form, sign/verify round-trip, tamper detection
- SignedMessage creation, properties, conversions
- SignedIPCManager transparent signing, verification, unsigned bypass
- SigningPolicy enforcement (PERMISSIVE, SIGN_REQUIRED, STRICT)
- SigningAuditor event logging and violation tracking
- Edge cases: empty payload, large payload, special characters, None correlation_id
- Replay protection (same message, different timestamps)
- Key rotation mid-stream (old signatures fail, new signatures pass)
"""

import hashlib
import hmac
import json
import threading
import time
import unittest

from battousai.ipc import IPCManager, Message, MessageType
from battousai.ipc_signing import (
    AuditEvent,
    KeyRing,
    MessageSigner,
    SignedIPCManager,
    SignedMessage,
    SigningAuditor,
    SigningError,
    SigningKey,
    SigningPolicy,
    make_signed_message,
    verify_signed_message,
)


# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------

def _msg(
    sender="alice",
    recipient="bob",
    mtype=MessageType.TASK,
    payload=None,
    timestamp=100,
    message_id=None,
    correlation_id=None,
    ttl=0,
):
    """Build a plain Message for tests."""
    kwargs = dict(
        sender_id=sender,
        recipient_id=recipient,
        message_type=mtype,
        payload=payload if payload is not None else {"data": "hello"},
        timestamp=timestamp,
        correlation_id=correlation_id,
        ttl=ttl,
    )
    if message_id:
        kwargs["message_id"] = message_id
    return Message(**kwargs)


# ===========================================================================
# SigningKey Tests
# ===========================================================================

class TestSigningKey(unittest.TestCase):
    """Tests for SigningKey."""

    def test_generate_returns_signing_key(self):
        key = SigningKey.generate()
        self.assertIsInstance(key, SigningKey)

    def test_generate_produces_unique_keys(self):
        k1 = SigningKey.generate()
        k2 = SigningKey.generate()
        self.assertNotEqual(k1, k2)

    def test_generate_key_has_32_bytes(self):
        key = SigningKey.generate()
        self.assertEqual(len(key._secret), 32)

    def test_from_bytes_round_trip(self):
        raw = b"\xAB" * 32
        key = SigningKey.from_bytes(raw)
        self.assertEqual(key._secret, raw)

    def test_from_bytes_preserves_short_key(self):
        raw = b"short"
        key = SigningKey.from_bytes(raw)
        self.assertEqual(key._secret, raw)

    def test_empty_bytes_raises(self):
        with self.assertRaises(ValueError):
            SigningKey.from_bytes(b"")

    def test_key_id_is_string(self):
        key = SigningKey.generate()
        self.assertIsInstance(key.key_id, str)

    def test_key_id_length_is_8(self):
        key = SigningKey.generate()
        self.assertEqual(len(key.key_id), 8)

    def test_key_id_is_hex(self):
        key = SigningKey.generate()
        # Should be valid hex
        int(key.key_id, 16)

    def test_key_id_derived_from_sha256(self):
        raw = b"deterministic_key_material_padded"
        key = SigningKey.from_bytes(raw)
        expected = hashlib.sha256(raw).hexdigest()[:8]
        self.assertEqual(key.key_id, expected)

    def test_same_raw_same_key_id(self):
        raw = b"same_raw_bytes_for_two_keys"
        k1 = SigningKey.from_bytes(raw)
        k2 = SigningKey.from_bytes(raw)
        self.assertEqual(k1.key_id, k2.key_id)

    def test_different_raw_different_key_id(self):
        k1 = SigningKey.from_bytes(b"key_one")
        k2 = SigningKey.from_bytes(b"key_two")
        self.assertNotEqual(k1.key_id, k2.key_id)

    def test_equality_same_raw(self):
        raw = b"equal_key_test_bytes_abc"
        k1 = SigningKey.from_bytes(raw)
        k2 = SigningKey.from_bytes(raw)
        self.assertEqual(k1, k2)

    def test_equality_different_raw(self):
        k1 = SigningKey.from_bytes(b"key_aaa")
        k2 = SigningKey.from_bytes(b"key_bbb")
        self.assertNotEqual(k1, k2)

    def test_repr_contains_key_id(self):
        key = SigningKey.generate()
        r = repr(key)
        self.assertIn(key.key_id, r)
        self.assertIn("SigningKey", r)

    def test_hash_same_raw_equal_hash(self):
        raw = b"hashable_key_bytes_xyz"
        k1 = SigningKey.from_bytes(raw)
        k2 = SigningKey.from_bytes(raw)
        self.assertEqual(hash(k1), hash(k2))

    def test_key_can_be_used_in_set(self):
        k1 = SigningKey.generate()
        k2 = SigningKey.generate()
        s = {k1, k2}
        self.assertEqual(len(s), 2)


# ===========================================================================
# KeyRing Tests
# ===========================================================================

class TestKeyRing(unittest.TestCase):
    """Tests for KeyRing."""

    def setUp(self):
        self.ring = KeyRing()
        self.key = SigningKey.generate()

    def test_register_and_get(self):
        self.ring.register_agent("alice", self.key)
        result = self.ring.get_key("alice")
        self.assertEqual(result, self.key)

    def test_get_unknown_returns_none(self):
        self.assertIsNone(self.ring.get_key("ghost"))

    def test_remove_agent(self):
        self.ring.register_agent("alice", self.key)
        self.ring.remove_agent("alice")
        self.assertIsNone(self.ring.get_key("alice"))

    def test_remove_nonexistent_is_noop(self):
        # Should not raise
        self.ring.remove_agent("does_not_exist")

    def test_rotate_key_changes_key(self):
        self.ring.register_agent("alice", self.key)
        new_key = self.ring.rotate_key("alice")
        self.assertNotEqual(new_key, self.key)
        stored = self.ring.get_key("alice")
        self.assertEqual(stored, new_key)

    def test_rotate_key_returns_signing_key(self):
        self.ring.register_agent("alice", self.key)
        new_key = self.ring.rotate_key("alice")
        self.assertIsInstance(new_key, SigningKey)

    def test_rotate_unknown_agent_raises(self):
        with self.assertRaises(KeyError):
            self.ring.rotate_key("nobody")

    def test_key_ids_returns_dict(self):
        k1 = SigningKey.generate()
        k2 = SigningKey.generate()
        self.ring.register_agent("alice", k1)
        self.ring.register_agent("bob", k2)
        ids = self.ring.key_ids()
        self.assertIsInstance(ids, dict)
        self.assertIn("alice", ids)
        self.assertIn("bob", ids)
        self.assertEqual(ids["alice"], k1.key_id)
        self.assertEqual(ids["bob"], k2.key_id)

    def test_key_ids_excludes_raw_keys(self):
        self.ring.register_agent("alice", self.key)
        ids = self.ring.key_ids()
        self.assertEqual(len(ids["alice"]), 8)

    def test_register_overwrites_existing(self):
        old_key = SigningKey.generate()
        new_key = SigningKey.generate()
        self.ring.register_agent("alice", old_key)
        self.ring.register_agent("alice", new_key)
        self.assertEqual(self.ring.get_key("alice"), new_key)

    def test_len_reflects_registrations(self):
        self.assertEqual(len(self.ring), 0)
        self.ring.register_agent("a", SigningKey.generate())
        self.ring.register_agent("b", SigningKey.generate())
        self.assertEqual(len(self.ring), 2)
        self.ring.remove_agent("a")
        self.assertEqual(len(self.ring), 1)

    def test_list_agents(self):
        self.ring.register_agent("x", SigningKey.generate())
        self.ring.register_agent("y", SigningKey.generate())
        agents = self.ring.list_agents()
        self.assertIn("x", agents)
        self.assertIn("y", agents)

    def test_thread_safety_concurrent_register(self):
        """Multiple threads registering different agents should not corrupt state."""
        errors = []

        def register(agent_id):
            try:
                self.ring.register_agent(agent_id, SigningKey.generate())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register, args=(f"agent_{i}",)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(self.ring), 50)

    def test_thread_safety_concurrent_get_rotate(self):
        """Concurrent get and rotate should not raise."""
        self.ring.register_agent("target", SigningKey.generate())
        errors = []

        def do_get():
            for _ in range(20):
                self.ring.get_key("target")

        def do_rotate():
            for _ in range(5):
                try:
                    self.ring.rotate_key("target")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=do_get) for _ in range(10)]
        threads += [threading.Thread(target=do_rotate) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)

    def test_repr_contains_keyring(self):
        r = repr(self.ring)
        self.assertIn("KeyRing", r)


# ===========================================================================
# MessageSigner Tests
# ===========================================================================

class TestMessageSigner(unittest.TestCase):
    """Tests for MessageSigner."""

    def setUp(self):
        self.key = SigningKey.from_bytes(b"test_signing_key_for_signer_tests")
        self.signer = MessageSigner()

    def test_sign_returns_string(self):
        msg = _msg()
        sig = MessageSigner.sign(msg, self.key)
        self.assertIsInstance(sig, str)

    def test_sign_returns_64_hex_chars(self):
        msg = _msg()
        sig = MessageSigner.sign(msg, self.key)
        self.assertEqual(len(sig), 64)
        int(sig, 16)  # Must be valid hex

    def test_sign_deterministic(self):
        msg = _msg(message_id="fixed_id")
        sig1 = MessageSigner.sign(msg, self.key)
        sig2 = MessageSigner.sign(msg, self.key)
        self.assertEqual(sig1, sig2)

    def test_verify_valid_signature(self):
        msg = _msg()
        sig = MessageSigner.sign(msg, self.key)
        self.assertTrue(MessageSigner.verify(msg, sig, self.key))

    def test_verify_invalid_signature(self):
        msg = _msg()
        self.assertFalse(MessageSigner.verify(msg, "bad_signature", self.key))

    def test_verify_wrong_key(self):
        msg = _msg()
        k1 = SigningKey.generate()
        k2 = SigningKey.generate()
        sig = MessageSigner.sign(msg, k1)
        self.assertFalse(MessageSigner.verify(msg, sig, k2))

    def test_tamper_sender_fails_verify(self):
        msg = _msg(sender="alice", message_id="m1")
        sig = MessageSigner.sign(msg, self.key)
        tampered = _msg(sender="mallory", message_id="m1")
        self.assertFalse(MessageSigner.verify(tampered, sig, self.key))

    def test_tamper_recipient_fails_verify(self):
        msg = _msg(recipient="bob", message_id="m2")
        sig = MessageSigner.sign(msg, self.key)
        tampered = _msg(recipient="eve", message_id="m2")
        self.assertFalse(MessageSigner.verify(tampered, sig, self.key))

    def test_tamper_payload_fails_verify(self):
        msg = _msg(payload={"secret": 42}, message_id="m3")
        sig = MessageSigner.sign(msg, self.key)
        tampered = _msg(payload={"secret": 99}, message_id="m3")
        self.assertFalse(MessageSigner.verify(tampered, sig, self.key))

    def test_tamper_timestamp_fails_verify(self):
        msg = _msg(timestamp=100, message_id="m4")
        sig = MessageSigner.sign(msg, self.key)
        tampered = _msg(timestamp=999, message_id="m4")
        self.assertFalse(MessageSigner.verify(tampered, sig, self.key))

    def test_tamper_message_type_fails_verify(self):
        msg = _msg(mtype=MessageType.TASK, message_id="m5")
        sig = MessageSigner.sign(msg, self.key)
        tampered = _msg(mtype=MessageType.ERROR, message_id="m5")
        self.assertFalse(MessageSigner.verify(tampered, sig, self.key))

    def test_canonical_form_pipe_delimited(self):
        msg = _msg(
            sender="alice",
            recipient="bob",
            mtype=MessageType.TASK,
            payload={"k": "v"},
            timestamp=42,
            message_id="abcd1234",
        )
        canon = MessageSigner.canonical_form(msg)
        parts = canon.split("|")
        self.assertEqual(len(parts), 6)
        self.assertEqual(parts[0], "alice")
        self.assertEqual(parts[1], "bob")
        self.assertEqual(parts[2], "TASK")
        # Part 3 is JSON payload
        self.assertEqual(json.loads(parts[3]), {"k": "v"})
        self.assertEqual(parts[4], "42")
        self.assertEqual(parts[5], "abcd1234")

    def test_canonical_payload_sort_keys(self):
        msg1 = _msg(payload={"b": 2, "a": 1}, message_id="cx1")
        msg2 = _msg(payload={"a": 1, "b": 2}, message_id="cx1")
        c1 = MessageSigner.canonical_form(msg1)
        c2 = MessageSigner.canonical_form(msg2)
        self.assertEqual(c1, c2)

    def test_sign_verify_with_empty_payload(self):
        msg = _msg(payload={})
        sig = MessageSigner.sign(msg, self.key)
        self.assertTrue(MessageSigner.verify(msg, sig, self.key))

    def test_sign_verify_with_none_payload(self):
        msg = _msg(payload=None)
        sig = MessageSigner.sign(msg, self.key)
        self.assertTrue(MessageSigner.verify(msg, sig, self.key))

    def test_sign_verify_large_payload(self):
        large = {"items": list(range(1000))}
        msg = _msg(payload=large)
        sig = MessageSigner.sign(msg, self.key)
        self.assertTrue(MessageSigner.verify(msg, sig, self.key))

    def test_sign_verify_special_chars_in_payload(self):
        special = {"text": "hello|world\nfoo\tbar", "unicode": "\u2603\u2665"}
        msg = _msg(payload=special)
        sig = MessageSigner.sign(msg, self.key)
        self.assertTrue(MessageSigner.verify(msg, sig, self.key))

    def test_sign_verify_none_correlation_id(self):
        msg = _msg(correlation_id=None)
        sig = MessageSigner.sign(msg, self.key)
        self.assertTrue(MessageSigner.verify(msg, sig, self.key))

    def test_replay_protection_different_timestamps(self):
        """Same message fields but different timestamp → different signatures."""
        msg1 = _msg(timestamp=100, message_id="replay1")
        msg2 = _msg(timestamp=200, message_id="replay1")
        sig1 = MessageSigner.sign(msg1, self.key)
        sig2 = MessageSigner.sign(msg2, self.key)
        self.assertNotEqual(sig1, sig2)
        # Cross-verify must fail
        self.assertFalse(MessageSigner.verify(msg2, sig1, self.key))
        self.assertFalse(MessageSigner.verify(msg1, sig2, self.key))

    def test_different_message_ids_different_sigs(self):
        msg1 = _msg(message_id="id_one_xx")
        msg2 = _msg(message_id="id_two_xx")
        sig1 = MessageSigner.sign(msg1, self.key)
        sig2 = MessageSigner.sign(msg2, self.key)
        self.assertNotEqual(sig1, sig2)

    def test_hmac_correctness_manual(self):
        """Spot-check: manually compute HMAC and compare."""
        raw = b"manual_key_bytes_for_verification"
        key = SigningKey.from_bytes(raw)
        msg = _msg(
            sender="s",
            recipient="r",
            mtype=MessageType.STATUS,
            payload={"x": 1},
            timestamp=5,
            message_id="aabbccdd",
        )
        canon = MessageSigner.canonical_form(msg)
        expected_hex = hmac.new(raw, canon.encode("utf-8"), "sha256").hexdigest()
        actual = MessageSigner.sign(msg, key)
        self.assertEqual(actual, expected_hex)


# ===========================================================================
# SignedMessage Tests
# ===========================================================================

class TestSignedMessage(unittest.TestCase):
    """Tests for SignedMessage."""

    def setUp(self):
        self.key = SigningKey.generate()
        self.base_msg = _msg()

    def test_create_signed_message_directly(self):
        sm = SignedMessage(
            sender_id="alice",
            recipient_id="bob",
            message_type=MessageType.TASK,
            payload={"k": 1},
            timestamp=10,
        )
        self.assertEqual(sm.sender_id, "alice")
        self.assertIsNone(sm.signature)

    def test_is_signed_false_when_no_signature(self):
        sm = SignedMessage.from_message(self.base_msg)
        self.assertFalse(sm.is_signed)

    def test_is_signed_true_when_signature_present(self):
        sig = MessageSigner.sign(self.base_msg, self.key)
        sm = SignedMessage.from_message(self.base_msg, signature=sig, key_id=self.key.key_id)
        self.assertTrue(sm.is_signed)

    def test_from_message_copies_all_fields(self):
        msg = _msg(
            sender="s1",
            recipient="r1",
            mtype=MessageType.REPLY,
            payload={"p": 7},
            timestamp=99,
            message_id="zz99",
            correlation_id="corr",
            ttl=5,
        )
        sm = SignedMessage.from_message(msg)
        self.assertEqual(sm.sender_id, "s1")
        self.assertEqual(sm.recipient_id, "r1")
        self.assertEqual(sm.message_type, MessageType.REPLY)
        self.assertEqual(sm.payload, {"p": 7})
        self.assertEqual(sm.timestamp, 99)
        self.assertEqual(sm.message_id, "zz99")
        self.assertEqual(sm.correlation_id, "corr")
        self.assertEqual(sm.ttl, 5)

    def test_to_message_strips_signature(self):
        sig = MessageSigner.sign(self.base_msg, self.key)
        sm = SignedMessage.from_message(self.base_msg, signature=sig, key_id=self.key.key_id)
        plain = sm.to_message()
        self.assertIsInstance(plain, Message)
        self.assertFalse(hasattr(plain, "signature"))

    def test_to_message_preserves_core_fields(self):
        sm = SignedMessage.from_message(self.base_msg)
        plain = sm.to_message()
        self.assertEqual(plain.sender_id, self.base_msg.sender_id)
        self.assertEqual(plain.message_id, self.base_msg.message_id)

    def test_is_expired_ttl_zero(self):
        sm = SignedMessage.from_message(_msg(ttl=0, timestamp=100))
        self.assertFalse(sm.is_expired(1000))

    def test_is_expired_past_ttl(self):
        sm = SignedMessage.from_message(_msg(ttl=5, timestamp=10))
        self.assertTrue(sm.is_expired(15))

    def test_repr_shows_signed(self):
        sig = MessageSigner.sign(self.base_msg, self.key)
        sm = SignedMessage.from_message(self.base_msg, signature=sig)
        r = repr(sm)
        self.assertIn("SignedMessage", r)
        self.assertIn("signed", r)

    def test_repr_shows_unsigned(self):
        sm = SignedMessage.from_message(self.base_msg)
        r = repr(sm)
        self.assertIn("unsigned", r)

    def test_key_id_stored(self):
        sig = MessageSigner.sign(self.base_msg, self.key)
        sm = SignedMessage.from_message(self.base_msg, signature=sig, key_id=self.key.key_id)
        self.assertEqual(sm.key_id, self.key.key_id)

    def test_default_message_id_generated(self):
        sm = SignedMessage(
            sender_id="a",
            recipient_id="b",
            message_type=MessageType.TASK,
            payload={},
            timestamp=1,
        )
        self.assertIsNotNone(sm.message_id)
        self.assertGreater(len(sm.message_id), 0)


# ===========================================================================
# make_signed_message / verify_signed_message helpers
# ===========================================================================

class TestConvenienceHelpers(unittest.TestCase):

    def setUp(self):
        self.key = SigningKey.generate()

    def test_make_signed_message_is_signed(self):
        sm = make_signed_message("a", "b", MessageType.TASK, {"x": 1}, 10, self.key)
        self.assertTrue(sm.is_signed)

    def test_make_signed_message_fields(self):
        sm = make_signed_message("a", "b", MessageType.QUERY, {"q": "yes"}, 77, self.key, correlation_id="cid", ttl=3)
        self.assertEqual(sm.sender_id, "a")
        self.assertEqual(sm.recipient_id, "b")
        self.assertEqual(sm.message_type, MessageType.QUERY)
        self.assertEqual(sm.payload, {"q": "yes"})
        self.assertEqual(sm.timestamp, 77)
        self.assertEqual(sm.correlation_id, "cid")
        self.assertEqual(sm.ttl, 3)
        self.assertEqual(sm.key_id, self.key.key_id)

    def test_verify_signed_message_valid(self):
        sm = make_signed_message("a", "b", MessageType.TASK, {"x": 1}, 10, self.key)
        self.assertTrue(verify_signed_message(sm, self.key))

    def test_verify_signed_message_wrong_key(self):
        other_key = SigningKey.generate()
        sm = make_signed_message("a", "b", MessageType.TASK, {}, 10, self.key)
        self.assertFalse(verify_signed_message(sm, other_key))

    def test_verify_unsigned_message_returns_false(self):
        sm = SignedMessage.from_message(_msg())
        self.assertFalse(verify_signed_message(sm, self.key))


# ===========================================================================
# SignedIPCManager Tests — PERMISSIVE Policy
# ===========================================================================

class TestSignedIPCManagerPermissive(unittest.TestCase):

    def setUp(self):
        self.ipc = IPCManager()
        self.ring = KeyRing()
        self.key = SigningKey.generate()
        self.ring.register_agent("alice", self.key)
        self.manager = SignedIPCManager(self.ipc, self.ring, SigningPolicy.PERMISSIVE)
        self.ipc.register_agent("alice")
        self.ipc.register_agent("bob")

    def test_send_with_key_succeeds(self):
        msg = _msg(sender="alice", recipient="bob")
        result = self.manager.send(msg)
        self.assertTrue(result)

    def test_send_without_key_permissive_succeeds(self):
        msg = _msg(sender="unknown_agent", recipient="bob")
        result = self.manager.send(msg)
        self.assertTrue(result)

    def test_verify_and_deliver_valid_signed(self):
        msg = _msg(sender="alice", recipient="bob")
        sig = MessageSigner.sign(msg, self.key)
        sm = SignedMessage.from_message(msg, signature=sig, key_id=self.key.key_id)
        result = self.manager.verify_and_deliver(sm)
        self.assertTrue(result)
        self.assertEqual(self.manager.total_verified, 1)

    def test_verify_and_deliver_unsigned_permissive(self):
        msg = _msg(sender="alice", recipient="bob")
        result = self.manager.verify_and_deliver(msg)
        self.assertTrue(result)
        self.assertEqual(self.manager.total_unsigned, 1)

    def test_verify_and_deliver_bad_sig_permissive_still_delivers(self):
        msg = _msg(sender="alice", recipient="bob")
        sm = SignedMessage.from_message(msg, signature="a" * 64, key_id=self.key.key_id)
        result = self.manager.verify_and_deliver(sm)
        # PERMISSIVE: deliver even on bad signature
        self.assertTrue(result)
        self.assertEqual(self.manager.total_failed, 1)

    def test_send_unsigned_increments_counter(self):
        msg = _msg(sender="kernel", recipient="bob")
        self.manager.send_unsigned(msg)
        self.assertEqual(self.manager.total_unsigned, 1)

    def test_stats_dict_contains_signing_keys(self):
        s = self.manager.stats()
        self.assertIn("total_verified", s)
        self.assertIn("total_failed", s)
        self.assertIn("total_unsigned", s)

    def test_repr_shows_policy(self):
        r = repr(self.manager)
        self.assertIn("PERMISSIVE", r)

    def test_register_agent_delegates(self):
        self.manager.register_agent("charlie")
        mb = self.ipc.get_mailbox("charlie")
        self.assertIsNotNone(mb)

    def test_get_mailbox_delegates(self):
        mb = self.manager.get_mailbox("alice")
        self.assertIsNotNone(mb)


# ===========================================================================
# SignedIPCManager Tests — SIGN_REQUIRED Policy
# ===========================================================================

class TestSignedIPCManagerSignRequired(unittest.TestCase):

    def setUp(self):
        self.ipc = IPCManager()
        self.ring = KeyRing()
        self.key = SigningKey.generate()
        self.ring.register_agent("alice", self.key)
        self.manager = SignedIPCManager(self.ipc, self.ring, SigningPolicy.SIGN_REQUIRED)
        self.ipc.register_agent("alice")
        self.ipc.register_agent("bob")

    def test_send_with_key_succeeds(self):
        msg = _msg(sender="alice", recipient="bob")
        result = self.manager.send(msg)
        self.assertTrue(result)

    def test_send_without_key_raises(self):
        msg = _msg(sender="no_key_agent", recipient="bob")
        with self.assertRaises(SigningError):
            self.manager.send(msg)

    def test_signing_error_has_agent_id(self):
        msg = _msg(sender="no_key_agent", recipient="bob")
        try:
            self.manager.send(msg)
        except SigningError as e:
            self.assertEqual(e.agent_id, "no_key_agent")

    def test_verify_unsigned_inbound_allowed(self):
        msg = _msg(sender="bob", recipient="alice")
        result = self.manager.verify_and_deliver(msg)
        self.assertTrue(result)

    def test_verify_bad_signature_raises(self):
        msg = _msg(sender="alice", recipient="bob")
        sm = SignedMessage.from_message(msg, signature="b" * 64, key_id=self.key.key_id)
        with self.assertRaises(SigningError):
            self.manager.verify_and_deliver(sm)


# ===========================================================================
# SignedIPCManager Tests — STRICT Policy
# ===========================================================================

class TestSignedIPCManagerStrict(unittest.TestCase):

    def setUp(self):
        self.ipc = IPCManager()
        self.ring = KeyRing()
        self.key = SigningKey.generate()
        self.ring.register_agent("alice", self.key)
        self.manager = SignedIPCManager(self.ipc, self.ring, SigningPolicy.STRICT)
        self.ipc.register_agent("alice")
        self.ipc.register_agent("bob")

    def test_send_without_key_raises(self):
        msg = _msg(sender="anon", recipient="bob")
        with self.assertRaises(SigningError):
            self.manager.send(msg)

    def test_verify_unsigned_raises(self):
        msg = _msg(sender="alice", recipient="bob")
        with self.assertRaises(SigningError):
            self.manager.verify_and_deliver(msg)

    def test_verify_bad_sig_raises(self):
        msg = _msg(sender="alice", recipient="bob")
        sm = SignedMessage.from_message(msg, signature="c" * 64)
        with self.assertRaises(SigningError):
            self.manager.verify_and_deliver(sm)

    def test_verify_valid_sig_succeeds(self):
        msg = _msg(sender="alice", recipient="bob")
        sig = MessageSigner.sign(msg, self.key)
        sm = SignedMessage.from_message(msg, signature=sig, key_id=self.key.key_id)
        result = self.manager.verify_and_deliver(sm)
        self.assertTrue(result)
        self.assertEqual(self.manager.total_verified, 1)
        self.assertEqual(self.manager.total_failed, 0)

    def test_send_unsigned_bypasses_strict(self):
        msg = _msg(sender="kernel", recipient="alice")
        # send_unsigned is explicitly exempt from policy
        result = self.manager.send_unsigned(msg)
        self.assertTrue(result)

    def test_policy_setter(self):
        self.manager.policy = SigningPolicy.PERMISSIVE
        self.assertEqual(self.manager.policy, SigningPolicy.PERMISSIVE)
        # Now unsigned should be fine
        msg = _msg(sender="anon", recipient="alice")
        result = self.manager.send(msg)
        self.assertTrue(result)


# ===========================================================================
# Key Rotation Mid-Stream Tests
# ===========================================================================

class TestKeyRotation(unittest.TestCase):

    def setUp(self):
        self.ipc = IPCManager()
        self.ring = KeyRing()
        self.key_v1 = SigningKey.generate()
        self.ring.register_agent("alice", self.key_v1)
        self.manager = SignedIPCManager(self.ipc, self.ring, SigningPolicy.SIGN_REQUIRED)
        self.ipc.register_agent("alice")
        self.ipc.register_agent("bob")

    def test_old_signature_fails_after_rotation(self):
        msg = _msg(sender="alice", recipient="bob", message_id="rot1")
        old_sig = MessageSigner.sign(msg, self.key_v1)
        # Rotate
        self.ring.rotate_key("alice")
        # Old sig should no longer verify
        self.assertFalse(MessageSigner.verify(msg, old_sig, self.ring.get_key("alice")))

    def test_new_signature_passes_after_rotation(self):
        msg = _msg(sender="alice", recipient="bob", message_id="rot2")
        key_v2 = self.ring.rotate_key("alice")
        new_sig = MessageSigner.sign(msg, key_v2)
        self.assertTrue(MessageSigner.verify(msg, new_sig, key_v2))

    def test_verify_and_deliver_fails_with_old_sig_after_rotation(self):
        msg = _msg(sender="alice", recipient="bob", message_id="rot3")
        old_sig = MessageSigner.sign(msg, self.key_v1)
        sm = SignedMessage.from_message(msg, signature=old_sig, key_id=self.key_v1.key_id)
        self.ring.rotate_key("alice")
        with self.assertRaises(SigningError):
            self.manager.verify_and_deliver(sm)

    def test_verify_and_deliver_passes_with_new_sig_after_rotation(self):
        msg = _msg(sender="alice", recipient="bob", message_id="rot4")
        key_v2 = self.ring.rotate_key("alice")
        new_sig = MessageSigner.sign(msg, key_v2)
        sm = SignedMessage.from_message(msg, signature=new_sig, key_id=key_v2.key_id)
        result = self.manager.verify_and_deliver(sm)
        self.assertTrue(result)

    def test_key_ring_key_id_changes_on_rotation(self):
        old_id = self.ring.key_ids()["alice"]
        self.ring.rotate_key("alice")
        new_id = self.ring.key_ids()["alice"]
        self.assertNotEqual(old_id, new_id)


# ===========================================================================
# SigningAuditor Tests
# ===========================================================================

class TestSigningAuditor(unittest.TestCase):

    def setUp(self):
        self.auditor = SigningAuditor()

    def test_log_sign_recorded(self):
        self.auditor.log_sign("alice", "msg1", "keyabc1")
        report = self.auditor.audit_report()
        self.assertEqual(report["total_sign"], 1)

    def test_log_verify_ok_recorded(self):
        self.auditor.log_verify_ok("alice", "msg1", "keyabc2")
        report = self.auditor.audit_report()
        self.assertEqual(report["total_verify_ok"], 1)

    def test_log_verify_fail_recorded(self):
        self.auditor.log_verify_fail("bob", "msg2", "keyabc3", "bad sig")
        report = self.auditor.audit_report()
        self.assertEqual(report["total_verify_fail"], 1)

    def test_log_unsigned_bypass_recorded(self):
        self.auditor.log_unsigned_bypass("kernel", "msg3")
        report = self.auditor.audit_report()
        self.assertEqual(report["total_unsigned_bypass"], 1)

    def test_log_reject_recorded(self):
        self.auditor.log_reject("bad_agent", "msg4", detail="strict policy")
        report = self.auditor.audit_report()
        self.assertEqual(report["total_reject"], 1)

    def test_violations_returns_only_fail_events(self):
        self.auditor.log_sign("alice", "m1")
        self.auditor.log_verify_ok("alice", "m2")
        self.auditor.log_verify_fail("bob", "m3", detail="fail")
        violations = self.auditor.violations()
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].event_type, "verify_fail")

    def test_violations_empty_initially(self):
        self.assertEqual(self.auditor.violations(), [])

    def test_audit_report_structure(self):
        report = self.auditor.audit_report()
        for key in ("total_sign", "total_verify_ok", "total_verify_fail",
                    "total_unsigned_bypass", "total_reject", "recent_events",
                    "violation_count"):
            self.assertIn(key, report)

    def test_recent_events_limited(self):
        for i in range(30):
            self.auditor.log_sign("a", f"m{i}")
        recent = self.auditor.recent_events(10)
        self.assertLessEqual(len(recent), 10)

    def test_max_events_evicts_oldest(self):
        auditor = SigningAuditor(max_events=5)
        for i in range(10):
            auditor.log_sign("a", f"m{i}")
        self.assertLessEqual(len(auditor), 5)

    def test_reset_clears_events(self):
        self.auditor.log_sign("a", "m1")
        self.auditor.log_verify_fail("b", "m2")
        self.auditor.reset()
        self.assertEqual(len(self.auditor), 0)
        report = self.auditor.audit_report()
        self.assertEqual(report["total_sign"], 0)
        self.assertEqual(report["total_verify_fail"], 0)

    def test_violation_count_in_report(self):
        self.auditor.log_verify_fail("x", "m1")
        self.auditor.log_verify_fail("y", "m2")
        report = self.auditor.audit_report()
        self.assertEqual(report["violation_count"], 2)

    def test_audit_event_repr(self):
        self.auditor.log_sign("alice", "m_repr", "kid1")
        events = self.auditor.recent_events(1)
        r = repr(events[0])
        self.assertIn("AuditEvent", r)
        self.assertIn("sign", r)

    def test_thread_safe_logging(self):
        errors = []

        def log_many():
            try:
                for i in range(50):
                    self.auditor.log_sign("t", f"msg_{i}_{threading.current_thread().name}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=log_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)

    def test_auditor_attached_to_manager(self):
        ipc = IPCManager()
        ring = KeyRing()
        auditor = SigningAuditor()
        manager = SignedIPCManager(ipc, ring, auditor=auditor)
        self.assertIs(manager.auditor, auditor)

    def test_auditor_logs_sign_events_via_manager(self):
        ipc = IPCManager()
        ring = KeyRing()
        key = SigningKey.generate()
        ring.register_agent("alice", key)
        auditor = SigningAuditor()
        manager = SignedIPCManager(ipc, ring, auditor=auditor)
        ipc.register_agent("bob")
        msg = _msg(sender="alice", recipient="bob")
        manager.send(msg)
        report = auditor.audit_report()
        self.assertGreater(report["total_sign"], 0)

    def test_auditor_logs_reject_events_under_strict(self):
        ipc = IPCManager()
        ring = KeyRing()
        key = SigningKey.generate()
        ring.register_agent("alice", key)
        auditor = SigningAuditor()
        manager = SignedIPCManager(ipc, ring, SigningPolicy.STRICT, auditor=auditor)
        ipc.register_agent("alice")
        ipc.register_agent("bob")
        msg = _msg(sender="alice", recipient="bob")
        sm = SignedMessage.from_message(msg, signature="d" * 64)
        try:
            manager.verify_and_deliver(sm)
        except SigningError:
            pass
        report = auditor.audit_report()
        self.assertGreater(report["total_reject"], 0)

    def test_rejections_list(self):
        self.auditor.log_reject("bad", "m99", detail="test")
        rejections = self.auditor.rejections()
        self.assertEqual(len(rejections), 1)
        self.assertEqual(rejections[0].event_type, "reject")


# ===========================================================================
# SigningPolicy Tests
# ===========================================================================

class TestSigningPolicy(unittest.TestCase):

    def test_policy_enum_values(self):
        self.assertIsNotNone(SigningPolicy.PERMISSIVE)
        self.assertIsNotNone(SigningPolicy.SIGN_REQUIRED)
        self.assertIsNotNone(SigningPolicy.STRICT)

    def test_policy_all_distinct(self):
        self.assertNotEqual(SigningPolicy.PERMISSIVE, SigningPolicy.SIGN_REQUIRED)
        self.assertNotEqual(SigningPolicy.SIGN_REQUIRED, SigningPolicy.STRICT)
        self.assertNotEqual(SigningPolicy.PERMISSIVE, SigningPolicy.STRICT)

    def test_permissive_allows_no_key(self):
        ipc = IPCManager()
        ring = KeyRing()
        m = SignedIPCManager(ipc, ring, SigningPolicy.PERMISSIVE)
        ipc.register_agent("bob")
        msg = _msg(sender="unknown", recipient="bob")
        # Should not raise
        m.send(msg)

    def test_sign_required_enforces_outbound_key(self):
        ipc = IPCManager()
        ring = KeyRing()
        m = SignedIPCManager(ipc, ring, SigningPolicy.SIGN_REQUIRED)
        ipc.register_agent("bob")
        msg = _msg(sender="no_key", recipient="bob")
        with self.assertRaises(SigningError):
            m.send(msg)

    def test_strict_enforces_both_inbound_and_outbound(self):
        ipc = IPCManager()
        ring = KeyRing()
        m = SignedIPCManager(ipc, ring, SigningPolicy.STRICT)
        ipc.register_agent("alice")
        # Unsigned inbound should raise
        msg = _msg(sender="alice", recipient="alice")
        with self.assertRaises(SigningError):
            m.verify_and_deliver(msg)


# ===========================================================================
# Edge Case / Integration Tests
# ===========================================================================

class TestEdgeCases(unittest.TestCase):

    def test_sign_verify_broadcast_message(self):
        from battousai.ipc import BROADCAST_ALL
        key = SigningKey.generate()
        msg = _msg(recipient=BROADCAST_ALL)
        sig = MessageSigner.sign(msg, key)
        self.assertTrue(MessageSigner.verify(msg, sig, key))

    def test_sign_verify_heartbeat_type(self):
        key = SigningKey.generate()
        msg = _msg(mtype=MessageType.HEARTBEAT, payload={"tick": 999})
        sig = MessageSigner.sign(msg, key)
        self.assertTrue(MessageSigner.verify(msg, sig, key))

    def test_sign_verify_error_type(self):
        key = SigningKey.generate()
        msg = _msg(mtype=MessageType.ERROR, payload={"code": 500})
        sig = MessageSigner.sign(msg, key)
        self.assertTrue(MessageSigner.verify(msg, sig, key))

    def test_sign_int_payload(self):
        key = SigningKey.generate()
        msg = _msg(payload=42)
        sig = MessageSigner.sign(msg, key)
        self.assertTrue(MessageSigner.verify(msg, sig, key))

    def test_sign_string_payload(self):
        key = SigningKey.generate()
        msg = _msg(payload="just a string")
        sig = MessageSigner.sign(msg, key)
        self.assertTrue(MessageSigner.verify(msg, sig, key))

    def test_sign_list_payload(self):
        key = SigningKey.generate()
        msg = _msg(payload=[1, 2, 3])
        sig = MessageSigner.sign(msg, key)
        self.assertTrue(MessageSigner.verify(msg, sig, key))

    def test_multiple_agents_multiple_keys(self):
        ring = KeyRing()
        ipc = IPCManager()
        keys = {}
        for name in ["alice", "bob", "charlie"]:
            k = SigningKey.generate()
            ring.register_agent(name, k)
            keys[name] = k
            ipc.register_agent(name)

        manager = SignedIPCManager(ipc, ring, SigningPolicy.SIGN_REQUIRED)
        for sender, recipient in [("alice", "bob"), ("bob", "charlie"), ("charlie", "alice")]:
            msg = _msg(sender=sender, recipient=recipient)
            self.assertTrue(manager.send(msg))

    def test_send_unsigned_no_audit(self):
        ipc = IPCManager()
        ring = KeyRing()
        auditor = SigningAuditor()
        manager = SignedIPCManager(ipc, ring, auditor=auditor)
        ipc.register_agent("alice")
        msg = _msg(sender="kernel", recipient="alice")
        manager.send_unsigned(msg, audit=False)
        # With audit=False, no unsigned_bypass event recorded (counter still increments)
        report = auditor.audit_report()
        self.assertEqual(report["total_unsigned_bypass"], 0)

    def test_create_message_via_manager(self):
        ipc = IPCManager()
        ring = KeyRing()
        key = SigningKey.generate()
        ring.register_agent("alice", key)
        manager = SignedIPCManager(ipc, ring, SigningPolicy.PERMISSIVE)
        ipc.register_agent("bob")
        msg = manager.create_message("alice", "bob", MessageType.TASK, {"data": 1}, 10)
        self.assertIsInstance(msg, Message)

    def test_signing_error_is_exception(self):
        e = SigningError("test error", agent_id="agent_x")
        self.assertIsInstance(e, Exception)
        self.assertEqual(e.agent_id, "agent_x")

    def test_signed_ipc_manager_key_ring_property(self):
        ipc = IPCManager()
        ring = KeyRing()
        manager = SignedIPCManager(ipc, ring)
        self.assertIs(manager.key_ring, ring)


if __name__ == "__main__":
    unittest.main()
