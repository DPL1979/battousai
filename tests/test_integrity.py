"""
tests/test_integrity.py — Comprehensive tests for battousai.integrity
=====================================================================

Covers:
    HashChain          — append, verify (clean + tampered + inserted + deleted), genesis
    SecureMemoryStore  — write/read, versioning, TTL, integrity violation,
                         read_verified, delete, expire_stale
    ToolRegistryVerifier — sign/verify (clean), detect added/removed/modified tool,
                            diff_registry
    IntegrityAuditor   — run_audit and format_report
"""

from __future__ import annotations

import time
import unittest
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from unittest.mock import MagicMock

from battousai.integrity import (
    AuditResult,
    EntryExpiredError,
    EntryNotFoundError,
    HashChain,
    HashChainEntry,
    IntegrityAuditor,
    IntegrityReport,
    IntegrityViolation,
    MemoryEntry,
    RegistrySignature,
    SecureMemoryStore,
    ToolRegistryVerifier,
)


# ---------------------------------------------------------------------------
# Minimal ToolSpec / ToolManager stubs (avoid importing tools.py side-effects)
# ---------------------------------------------------------------------------

@dataclass
class _ToolSpec:
    """Minimal ToolSpec compatible with ToolRegistryVerifier."""
    name: str
    description: str
    callable: Callable[..., Any]
    allowed_agents: Set[str] = field(default_factory=set)
    rate_limit: int = 0
    rate_window: int = 10
    is_simulated: bool = False


class _ToolManager:
    """Minimal ToolManager stub."""

    def __init__(self) -> None:
        self._tools: Dict[str, _ToolSpec] = {}

    def register(self, spec: _ToolSpec) -> None:
        self._tools[spec.name] = spec

    def deregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def list_tools(self) -> List[str]:
        return sorted(self._tools.keys())

    def get_spec(self, name: str) -> _ToolSpec:
        return self._tools[name]


def _noop(*args, **kwargs) -> None:
    return None


def _noop2(*args, **kwargs) -> None:
    return None


# ===========================================================================
# HashChain tests
# ===========================================================================

class TestHashChainAppendAndVerify(unittest.TestCase):
    """Basic append and clean-chain verification."""

    def test_empty_chain_is_valid(self):
        chain = HashChain()
        report = chain.verify()
        self.assertTrue(report.valid)
        self.assertEqual(report.total_entries, 0)
        self.assertEqual(report.verified_entries, 0)
        self.assertIsNone(report.first_tampered_index)

    def test_single_entry_chain_is_valid(self):
        chain = HashChain()
        entry = chain.append(b"hello world")
        self.assertEqual(entry.index, 0)
        self.assertEqual(entry.previous_hash, "genesis")
        report = chain.verify()
        self.assertTrue(report.valid)
        self.assertEqual(report.total_entries, 1)
        self.assertEqual(report.verified_entries, 1)

    def test_multi_entry_chain_is_valid(self):
        chain = HashChain()
        for i in range(10):
            chain.append(f"entry {i}".encode())
        report = chain.verify()
        self.assertTrue(report.valid)
        self.assertEqual(report.total_entries, 10)
        self.assertEqual(report.verified_entries, 10)

    def test_entry_has_correct_index(self):
        chain = HashChain()
        for i in range(5):
            e = chain.append(f"x{i}".encode())
            self.assertEqual(e.index, i)

    def test_entry_previous_hash_links(self):
        chain = HashChain()
        e0 = chain.append(b"first")
        e1 = chain.append(b"second")
        e2 = chain.append(b"third")
        self.assertEqual(e1.previous_hash, e0.chain_hash)
        self.assertEqual(e2.previous_hash, e1.chain_hash)

    def test_entry_metadata_stored(self):
        chain = HashChain()
        entry = chain.append(b"data", metadata={"key": "val"})
        self.assertEqual(entry.metadata, {"key": "val"})

    def test_len_equals_entries_appended(self):
        chain = HashChain()
        for i in range(7):
            chain.append(str(i).encode())
        self.assertEqual(len(chain), 7)

    def test_getitem_returns_correct_entry(self):
        chain = HashChain()
        entries = [chain.append(str(i).encode()) for i in range(5)]
        for i, e in enumerate(entries):
            self.assertEqual(chain[i].chain_hash, e.chain_hash)

    def test_verify_entry_single(self):
        chain = HashChain()
        chain.append(b"solo")
        self.assertTrue(chain.verify_entry(0))

    def test_verify_entry_out_of_range(self):
        chain = HashChain()
        chain.append(b"x")
        self.assertFalse(chain.verify_entry(5))
        self.assertFalse(chain.verify_entry(-1))


class TestHashChainGenesisEntry(unittest.TestCase):
    """First entry in a chain uses 'genesis' as its previous_hash."""

    def test_first_entry_previous_hash_is_genesis(self):
        chain = HashChain()
        entry = chain.append(b"genesis block")
        self.assertEqual(entry.previous_hash, "genesis")

    def test_chain_hash_differs_from_data_hash(self):
        chain = HashChain()
        entry = chain.append(b"some data")
        self.assertNotEqual(entry.chain_hash, entry.data_hash)


class TestHashChainTampering(unittest.TestCase):
    """Verify that modifications are detected."""

    def _build_chain(self, n: int = 5) -> HashChain:
        chain = HashChain()
        for i in range(n):
            chain.append(f"entry {i}".encode())
        return chain

    def test_tamper_data_hash_breaks_chain(self):
        chain = self._build_chain(5)
        # Corrupt the data_hash of entry 2
        chain._entries[2].data_hash = "deadbeef" * 8
        report = chain.verify()
        self.assertFalse(report.valid)
        self.assertEqual(report.first_tampered_index, 2)

    def test_tamper_chain_hash_breaks_chain(self):
        chain = self._build_chain(5)
        chain._entries[3].chain_hash = "00" * 32
        report = chain.verify()
        self.assertFalse(report.valid)
        # Either entry 3 or 4 must be flagged (linkage breaks at 4)
        self.assertIsNotNone(report.first_tampered_index)
        self.assertLessEqual(report.first_tampered_index, 4)

    def test_tamper_previous_hash_breaks_chain(self):
        chain = self._build_chain(5)
        chain._entries[1].previous_hash = "cafebabe" * 8
        report = chain.verify()
        self.assertFalse(report.valid)
        self.assertEqual(report.first_tampered_index, 1)

    def test_tamper_first_entry_detected(self):
        chain = self._build_chain(3)
        chain._entries[0].data_hash = "aa" * 32
        report = chain.verify()
        self.assertFalse(report.valid)
        self.assertEqual(report.first_tampered_index, 0)

    def test_inserted_entry_breaks_chain(self):
        """Simulate an entry inserted mid-chain (shifts subsequent previous_hashes)."""
        chain = self._build_chain(4)
        # Build a fake entry and splice it in at position 2
        fake = HashChainEntry(
            index=2,
            data_hash="ff" * 32,
            previous_hash=chain._entries[1].chain_hash,
            chain_hash="ee" * 32,
            timestamp=time.time(),
        )
        chain._entries.insert(2, fake)
        # Re-index
        for i, e in enumerate(chain._entries):
            e.index = i
        report = chain.verify()
        self.assertFalse(report.valid)

    def test_deleted_entry_breaks_chain(self):
        """Removing an entry mid-chain should break subsequent linkage."""
        chain = self._build_chain(5)
        del chain._entries[2]
        # Re-index
        for i, e in enumerate(chain._entries):
            e.index = i
        report = chain.verify()
        self.assertFalse(report.valid)

    def test_report_details_contains_index_on_violation(self):
        chain = self._build_chain(4)
        chain._entries[1].data_hash = "00" * 32
        report = chain.verify()
        self.assertIn("1", report.details)


# ===========================================================================
# SecureMemoryStore tests
# ===========================================================================

class TestSecureMemoryStoreWriteRead(unittest.TestCase):

    def setUp(self):
        self.store = SecureMemoryStore()

    def test_write_returns_string(self):
        receipt = self.store.write("k1", "hello", "agent-a")
        self.assertIsInstance(receipt, str)
        self.assertTrue(len(receipt) > 0)

    def test_read_returns_correct_value(self):
        self.store.write("k1", {"x": 42}, "agent-a")
        val = self.store.read("k1")
        self.assertEqual(val, {"x": 42})

    def test_read_various_types(self):
        self.store.write("int_key", 99, "a")
        self.store.write("list_key", [1, 2, 3], "a")
        self.store.write("none_key", None, "a")
        self.assertEqual(self.store.read("int_key"), 99)
        self.assertEqual(self.store.read("list_key"), [1, 2, 3])
        self.assertIsNone(self.store.read("none_key"))

    def test_read_missing_key_raises(self):
        with self.assertRaises(EntryNotFoundError):
            self.store.read("does_not_exist")

    def test_read_without_verify(self):
        self.store.write("k", "val", "a")
        val = self.store.read("k", verify=False)
        self.assertEqual(val, "val")

    def test_overwrite_returns_latest(self):
        self.store.write("k", "v1", "a")
        self.store.write("k", "v2", "a")
        val = self.store.read("k")
        self.assertEqual(val, "v2")


class TestSecureMemoryStoreVersioning(unittest.TestCase):

    def setUp(self):
        self.store = SecureMemoryStore()

    def test_version_increments_per_key(self):
        self.store.write("k", "v1", "a")
        self.store.write("k", "v2", "a")
        self.store.write("k", "v3", "a")
        history = self.store.get_history("k")
        self.assertEqual([e.version for e in history], [1, 2, 3])

    def test_different_keys_version_independently(self):
        self.store.write("a", 1, "agent")
        self.store.write("b", 10, "agent")
        self.store.write("a", 2, "agent")
        self.assertEqual(self.store.get_history("a")[-1].version, 2)
        self.assertEqual(self.store.get_history("b")[-1].version, 1)

    def test_get_version_retrieves_old_value(self):
        self.store.write("k", "first", "a")
        self.store.write("k", "second", "a")
        self.store.write("k", "third", "a")
        self.assertEqual(self.store.get_version("k", 1), "first")
        self.assertEqual(self.store.get_version("k", 2), "second")
        self.assertEqual(self.store.get_version("k", 3), "third")

    def test_get_version_missing_key_raises(self):
        with self.assertRaises(EntryNotFoundError):
            self.store.get_version("missing", 1)

    def test_get_version_missing_version_raises(self):
        self.store.write("k", "v1", "a")
        with self.assertRaises(EntryNotFoundError):
            self.store.get_version("k", 99)

    def test_get_history_empty_for_missing_key(self):
        self.assertEqual(self.store.get_history("nonexistent"), [])

    def test_history_oldest_first(self):
        for i in range(5):
            self.store.write("k", i, "a")
        history = self.store.get_history("k")
        self.assertEqual([e.value for e in history], list(range(5)))


class TestSecureMemoryStoreTTL(unittest.TestCase):

    def test_read_before_expiry(self):
        store = SecureMemoryStore(ttl_seconds=10.0)
        store.write("k", "val", "a")
        val = store.read("k")
        self.assertEqual(val, "val")

    def test_read_after_expiry_raises(self):
        store = SecureMemoryStore(ttl_seconds=0.05)
        store.write("k", "val", "a")
        time.sleep(0.1)
        with self.assertRaises(EntryExpiredError):
            store.read("k")

    def test_no_ttl_does_not_expire(self):
        store = SecureMemoryStore(ttl_seconds=None)
        store.write("k", "forever", "a")
        time.sleep(0.05)
        self.assertEqual(store.read("k"), "forever")


class TestSecureMemoryStoreExpireStale(unittest.TestCase):

    def test_expire_stale_removes_expired_entries(self):
        store = SecureMemoryStore(ttl_seconds=0.05)
        store.write("k1", "v1", "a")
        store.write("k2", "v2", "a")
        time.sleep(0.1)
        count = store.expire_stale()
        self.assertEqual(count, 2)
        with self.assertRaises(EntryNotFoundError):
            store.read("k1", verify=False)

    def test_expire_stale_keeps_live_entries(self):
        store = SecureMemoryStore(ttl_seconds=0.05)
        store.write("short", "v1", "a")
        time.sleep(0.1)
        # Write a fresh entry AFTER sleep so it has a fresh expiry
        store = SecureMemoryStore(ttl_seconds=10.0)
        store.write("long", "v2", "a")
        count = store.expire_stale()
        self.assertEqual(count, 0)
        self.assertEqual(store.read("long"), "v2")

    def test_expire_stale_returns_zero_when_nothing_expired(self):
        store = SecureMemoryStore(ttl_seconds=60.0)
        store.write("k", "v", "a")
        count = store.expire_stale()
        self.assertEqual(count, 0)


class TestSecureMemoryStoreIntegrityViolation(unittest.TestCase):

    def test_tampered_chain_raises_on_read(self):
        store = SecureMemoryStore()
        store.write("k", "secret", "a")
        # Corrupt the first chain entry's data_hash directly
        store._chain._entries[0].data_hash = "bad" * 21 + "b"
        with self.assertRaises(IntegrityViolation):
            store.read("k", verify=True)

    def test_tampered_chain_does_not_raise_without_verify(self):
        store = SecureMemoryStore()
        store.write("k", "secret", "a")
        store._chain._entries[0].data_hash = "bad" * 21 + "b"
        # Should not raise — verify=False skips the chain check
        val = store.read("k", verify=False)
        self.assertEqual(val, "secret")

    def test_verify_all_returns_report_on_tamper(self):
        store = SecureMemoryStore()
        store.write("k", "v", "a")
        store._chain._entries[0].data_hash = "00" * 32
        report = store.verify_all()
        self.assertFalse(report.valid)
        self.assertEqual(report.first_tampered_index, 0)

    def test_verify_all_passes_clean_store(self):
        store = SecureMemoryStore()
        for i in range(10):
            store.write(f"k{i}", i, "a")
        report = store.verify_all()
        self.assertTrue(report.valid)


class TestSecureMemoryStoreReadVerified(unittest.TestCase):

    def test_read_verified_returns_value_and_true_when_clean(self):
        store = SecureMemoryStore()
        store.write("k", "good", "a")
        val, ok = store.read_verified("k")
        self.assertEqual(val, "good")
        self.assertTrue(ok)

    def test_read_verified_returns_false_on_tamper(self):
        store = SecureMemoryStore()
        store.write("k", "secret", "a")
        store._chain._entries[0].data_hash = "ff" * 32
        val, ok = store.read_verified("k")
        self.assertFalse(ok)

    def test_read_verified_returns_none_false_for_missing_key(self):
        store = SecureMemoryStore()
        val, ok = store.read_verified("missing")
        self.assertIsNone(val)
        self.assertFalse(ok)

    def test_read_verified_returns_false_for_expired_entry(self):
        store = SecureMemoryStore(ttl_seconds=0.05)
        store.write("k", "x", "a")
        time.sleep(0.1)
        val, ok = store.read_verified("k")
        self.assertFalse(ok)


class TestSecureMemoryStoreDelete(unittest.TestCase):

    def test_delete_existing_key_returns_true(self):
        store = SecureMemoryStore()
        store.write("k", "v", "a")
        result = store.delete("k", "a")
        self.assertTrue(result)

    def test_delete_missing_key_returns_false(self):
        store = SecureMemoryStore()
        result = store.delete("nope", "a")
        self.assertFalse(result)

    def test_read_after_delete_raises(self):
        store = SecureMemoryStore()
        store.write("k", "v", "a")
        store.delete("k", "a")
        with self.assertRaises(EntryNotFoundError):
            store.read("k", verify=False)

    def test_history_preserved_after_delete(self):
        store = SecureMemoryStore()
        store.write("k", "v", "a")
        store.delete("k", "a")
        history = store.get_history("k")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].value, "v")


# ===========================================================================
# ToolRegistryVerifier tests
# ===========================================================================

class TestToolRegistryVerifierSignAndVerify(unittest.TestCase):

    def _make_manager(self) -> _ToolManager:
        mgr = _ToolManager()
        mgr.register(_ToolSpec(name="tool_a", description="A", callable=_noop))
        mgr.register(_ToolSpec(name="tool_b", description="B", callable=_noop))
        return mgr

    def test_sign_returns_registry_signature(self):
        verifier = ToolRegistryVerifier()
        mgr = self._make_manager()
        sig = verifier.sign_registry(mgr)
        self.assertIsInstance(sig, RegistrySignature)
        self.assertEqual(sig.tool_count, 2)
        self.assertIn("tool_a", sig.tool_hashes)
        self.assertIn("tool_b", sig.tool_hashes)

    def test_verify_clean_registry_returns_true(self):
        verifier = ToolRegistryVerifier()
        mgr = self._make_manager()
        sig = verifier.sign_registry(mgr)
        self.assertTrue(verifier.verify_registry(mgr, sig))

    def test_sign_tool_returns_string(self):
        verifier = ToolRegistryVerifier()
        spec = _ToolSpec(name="t", description="d", callable=_noop)
        h = verifier.sign_tool(spec)
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)  # SHA-256 hex

    def test_different_specs_have_different_hashes(self):
        verifier = ToolRegistryVerifier()
        s1 = _ToolSpec(name="t", description="desc1", callable=_noop)
        s2 = _ToolSpec(name="t", description="desc2", callable=_noop)
        self.assertNotEqual(verifier.sign_tool(s1), verifier.sign_tool(s2))

    def test_same_spec_has_same_hash(self):
        verifier = ToolRegistryVerifier()
        s = _ToolSpec(name="t", description="d", callable=_noop)
        self.assertEqual(verifier.sign_tool(s), verifier.sign_tool(s))


class TestToolRegistryVerifierDetectsChanges(unittest.TestCase):

    def _make_manager_and_sig(self):
        verifier = ToolRegistryVerifier()
        mgr = _ToolManager()
        mgr.register(_ToolSpec(name="alpha", description="Alpha tool", callable=_noop))
        mgr.register(_ToolSpec(name="beta", description="Beta tool", callable=_noop))
        sig = verifier.sign_registry(mgr)
        return verifier, mgr, sig

    def test_detect_added_tool(self):
        verifier, mgr, sig = self._make_manager_and_sig()
        mgr.register(_ToolSpec(name="gamma", description="Gamma tool", callable=_noop))
        self.assertFalse(verifier.verify_registry(mgr, sig))

    def test_detect_removed_tool(self):
        verifier, mgr, sig = self._make_manager_and_sig()
        mgr.deregister("beta")
        self.assertFalse(verifier.verify_registry(mgr, sig))

    def test_detect_modified_tool_description(self):
        verifier, mgr, sig = self._make_manager_and_sig()
        mgr._tools["alpha"].description = "Modified description"
        self.assertFalse(verifier.verify_registry(mgr, sig))

    def test_detect_modified_tool_callable(self):
        verifier, mgr, sig = self._make_manager_and_sig()
        mgr._tools["alpha"].callable = _noop2
        self.assertFalse(verifier.verify_registry(mgr, sig))

    def test_detect_modified_tool_rate_limit(self):
        verifier, mgr, sig = self._make_manager_and_sig()
        mgr._tools["beta"].rate_limit = 9999
        self.assertFalse(verifier.verify_registry(mgr, sig))

    def test_detect_modified_allowed_agents(self):
        verifier, mgr, sig = self._make_manager_and_sig()
        mgr._tools["alpha"].allowed_agents.add("rogue-agent")
        self.assertFalse(verifier.verify_registry(mgr, sig))


class TestToolRegistryVerifierDiffRegistry(unittest.TestCase):

    def _make_manager_and_sig(self):
        verifier = ToolRegistryVerifier()
        mgr = _ToolManager()
        mgr.register(_ToolSpec(name="tool1", description="T1", callable=_noop))
        mgr.register(_ToolSpec(name="tool2", description="T2", callable=_noop))
        sig = verifier.sign_registry(mgr)
        return verifier, mgr, sig

    def test_no_changes_returns_empty_diff(self):
        verifier, mgr, sig = self._make_manager_and_sig()
        diff = verifier.diff_registry(mgr, sig)
        self.assertEqual(diff, [])

    def test_added_tool_in_diff(self):
        verifier, mgr, sig = self._make_manager_and_sig()
        mgr.register(_ToolSpec(name="tool3", description="T3", callable=_noop))
        diff = verifier.diff_registry(mgr, sig)
        self.assertTrue(any("ADDED" in d and "tool3" in d for d in diff))

    def test_removed_tool_in_diff(self):
        verifier, mgr, sig = self._make_manager_and_sig()
        mgr.deregister("tool2")
        diff = verifier.diff_registry(mgr, sig)
        self.assertTrue(any("REMOVED" in d and "tool2" in d for d in diff))

    def test_modified_tool_in_diff(self):
        verifier, mgr, sig = self._make_manager_and_sig()
        mgr._tools["tool1"].description = "Changed"
        diff = verifier.diff_registry(mgr, sig)
        self.assertTrue(any("MODIFIED" in d and "tool1" in d for d in diff))

    def test_multiple_changes_all_appear_in_diff(self):
        verifier, mgr, sig = self._make_manager_and_sig()
        mgr.deregister("tool1")
        mgr.register(_ToolSpec(name="tool3", description="T3", callable=_noop))
        diff = verifier.diff_registry(mgr, sig)
        self.assertEqual(len(diff), 2)


# ===========================================================================
# IntegrityAuditor tests
# ===========================================================================

class TestIntegrityAuditor(unittest.TestCase):

    def _make_auditor(self):
        store = SecureMemoryStore()
        verifier = ToolRegistryVerifier()
        auditor = IntegrityAuditor(store, verifier)
        return auditor, store, verifier

    def _make_manager(self):
        mgr = _ToolManager()
        mgr.register(_ToolSpec(name="t1", description="Tool 1", callable=_noop))
        return mgr

    def test_run_audit_without_manager_passes_on_empty_store(self):
        auditor, store, _ = self._make_auditor()
        result = auditor.run_audit()
        self.assertTrue(result.passed)
        self.assertIsNone(result.registry_valid)

    def test_run_audit_with_clean_store_passes(self):
        auditor, store, _ = self._make_auditor()
        store.write("k", "v", "a")
        result = auditor.run_audit()
        self.assertTrue(result.passed)
        self.assertTrue(result.memory_report.valid)

    def test_run_audit_detects_chain_violation(self):
        auditor, store, _ = self._make_auditor()
        store.write("k", "v", "a")
        store._chain._entries[0].data_hash = "bad" * 21 + "b"
        result = auditor.run_audit()
        self.assertFalse(result.passed)
        self.assertFalse(result.memory_report.valid)

    def test_run_audit_with_registry_snapshot_passes_if_clean(self):
        auditor, store, _ = self._make_auditor()
        mgr = self._make_manager()
        auditor.snapshot_registry(mgr)
        result = auditor.run_audit(tool_manager=mgr)
        self.assertTrue(result.passed)
        self.assertTrue(result.registry_valid)

    def test_run_audit_detects_registry_tampering(self):
        auditor, store, _ = self._make_auditor()
        mgr = self._make_manager()
        auditor.snapshot_registry(mgr)
        mgr._tools["t1"].description = "tampered!"
        result = auditor.run_audit(tool_manager=mgr)
        self.assertFalse(result.passed)
        self.assertFalse(result.registry_valid)

    def test_run_audit_provides_registry_diff_on_change(self):
        auditor, store, _ = self._make_auditor()
        mgr = self._make_manager()
        auditor.snapshot_registry(mgr)
        mgr.register(_ToolSpec(name="new_tool", description="X", callable=_noop))
        result = auditor.run_audit(tool_manager=mgr)
        self.assertTrue(len(result.registry_diff) > 0)

    def test_run_audit_no_registry_diff_when_no_manager(self):
        auditor, store, _ = self._make_auditor()
        result = auditor.run_audit()
        self.assertEqual(result.registry_diff, [])

    # Format report tests
    def test_format_report_is_string(self):
        auditor, store, _ = self._make_auditor()
        result = auditor.run_audit()
        report_str = auditor.format_report(result)
        self.assertIsInstance(report_str, str)

    def test_format_report_contains_pass_on_clean(self):
        auditor, store, _ = self._make_auditor()
        result = auditor.run_audit()
        report_str = auditor.format_report(result)
        self.assertIn("PASS", report_str)

    def test_format_report_contains_fail_on_violation(self):
        auditor, store, _ = self._make_auditor()
        store.write("k", "v", "a")
        store._chain._entries[0].data_hash = "00" * 32
        result = auditor.run_audit()
        report_str = auditor.format_report(result)
        self.assertIn("FAIL", report_str)

    def test_format_report_contains_memory_section(self):
        auditor, store, _ = self._make_auditor()
        result = auditor.run_audit()
        report_str = auditor.format_report(result)
        self.assertIn("Memory Chain", report_str)

    def test_format_report_contains_registry_section_when_checked(self):
        auditor, store, _ = self._make_auditor()
        mgr = self._make_manager()
        auditor.snapshot_registry(mgr)
        result = auditor.run_audit(tool_manager=mgr)
        report_str = auditor.format_report(result)
        self.assertIn("Tool Registry", report_str)

    def test_format_report_shows_changes(self):
        auditor, store, _ = self._make_auditor()
        mgr = self._make_manager()
        auditor.snapshot_registry(mgr)
        mgr.deregister("t1")
        result = auditor.run_audit(tool_manager=mgr)
        report_str = auditor.format_report(result)
        self.assertIn("REMOVED", report_str)

    def test_audit_result_summary_is_non_empty(self):
        auditor, store, _ = self._make_auditor()
        result = auditor.run_audit()
        self.assertTrue(len(result.summary) > 0)


# ===========================================================================
# Additional edge-case / integration tests
# ===========================================================================

class TestIntegrationHashChainAndStore(unittest.TestCase):

    def test_multiple_writes_produce_growing_chain(self):
        store = SecureMemoryStore()
        for i in range(20):
            store.write(f"k{i}", i, "agent")
        self.assertEqual(len(store._chain), 20)
        report = store.verify_all()
        self.assertTrue(report.valid)

    def test_delete_appends_to_chain(self):
        store = SecureMemoryStore()
        store.write("k", "v", "a")
        initial_len = len(store._chain)
        store.delete("k", "a")
        self.assertGreater(len(store._chain), initial_len)

    def test_large_chain_verify_performance(self):
        """Verifying 1000-entry chain should complete quickly."""
        store = SecureMemoryStore()
        for i in range(1000):
            store.write("k", i, "a")
        start = time.time()
        report = store.verify_all()
        elapsed = time.time() - start
        self.assertTrue(report.valid)
        self.assertLess(elapsed, 5.0)  # generous bound


if __name__ == "__main__":
    unittest.main()
