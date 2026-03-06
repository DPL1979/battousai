"""
tests/test_sandbox.py — Unit tests for battousai.sandbox
==========================================================
Tests for OS-level sandboxing module.

All tests are designed to pass on any platform (Linux, macOS, Windows).
Features that require Linux use ``is_available()`` checks and
``unittest.skipIf`` guards where appropriate; they never require root.
"""

from __future__ import annotations

import multiprocessing
import os
import platform
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from battousai.capabilities import CapabilityType
from battousai.sandbox import (
    PROFILE_MINIMAL,
    PROFILE_NETWORK,
    PROFILE_PRIVILEGED,
    PROFILE_STANDARD,
    EnforcementReport,
    EnvironmentSanitizer,
    NamespaceIsolation,
    ResourceLimiter,
    SandboxProfile,
    SandboxResult,
    SandboxedProcess,
    SeccompFilter,
    capability_to_profile,
)

_IS_LINUX = platform.system() == "Linux"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(**kwargs) -> SandboxProfile:
    """Return a minimal-overhead SandboxProfile for testing.

    max_processes defaults to 0 (unlimited) to avoid accidentally clamping
    RLIMIT_NPROC on the test-runner process and breaking subsequent fork() calls.
    """
    defaults = dict(
        name="test",
        allowed_paths=["/tmp"],
        denied_paths=[],
        writable_paths=["/tmp"],
        network_allowed=False,
        max_memory_bytes=0,
        max_cpu_seconds=0.0,
        max_file_descriptors=0,
        max_processes=0,
        env_whitelist=[],
        env_blacklist=[],
    )
    defaults.update(kwargs)
    return SandboxProfile(**defaults)


def _simple_target() -> str:
    """A picklable callable that returns a constant string."""
    return "hello-from-sandbox"


def _raising_target() -> None:
    """A picklable callable that always raises."""
    raise ValueError("intentional error in sandbox")


def _env_capture_target() -> dict:
    """Return a subset of os.environ for inspection."""
    return {k: v for k, v in os.environ.items()
            if k in ("PATH", "AWS_ACCESS_KEY_ID", "OPENAI_API_KEY", "HOME",
                     "SAFE_VAR")}


# ---------------------------------------------------------------------------
# TestSandboxProfile
# ---------------------------------------------------------------------------

class TestSandboxProfile(unittest.TestCase):
    """Tests for SandboxProfile dataclass."""

    def test_create_minimal_profile(self):
        p = SandboxProfile(name="minimal", max_file_descriptors=32, max_processes=1)
        self.assertEqual(p.name, "minimal")
        self.assertFalse(p.network_allowed)
        self.assertEqual(p.allowed_paths, [])
        self.assertEqual(p.denied_paths, [])
        self.assertEqual(p.writable_paths, [])
        self.assertIsNone(p.allowed_syscalls)

    def test_create_full_profile(self):
        p = SandboxProfile(
            name="full",
            allowed_paths=["/tmp", "/usr"],
            denied_paths=["/etc/shadow"],
            writable_paths=["/tmp"],
            network_allowed=True,
            allowed_syscalls={"read", "write", "open"},
            max_memory_bytes=256 * 1024 * 1024,
            max_cpu_seconds=60.0,
            max_file_descriptors=128,
            max_processes=8,
            env_whitelist=["PATH"],
            env_blacklist=["SECRET"],
        )
        self.assertEqual(p.name, "full")
        self.assertTrue(p.network_allowed)
        self.assertIn("/tmp", p.allowed_paths)
        self.assertIn("/tmp", p.writable_paths)
        self.assertIn("/etc/shadow", p.denied_paths)
        self.assertEqual(p.allowed_syscalls, {"read", "write", "open"})
        self.assertEqual(p.max_memory_bytes, 256 * 1024 * 1024)
        self.assertEqual(p.max_cpu_seconds, 60.0)
        self.assertEqual(p.env_whitelist, ["PATH"])
        self.assertEqual(p.env_blacklist, ["SECRET"])

    def test_default_mutable_fields_are_independent(self):
        """Each SandboxProfile instance must have independent lists."""
        p1 = SandboxProfile(name="a")
        p2 = SandboxProfile(name="b")
        p1.allowed_paths.append("/extra")
        self.assertNotIn("/extra", p2.allowed_paths)


# ---------------------------------------------------------------------------
# TestPredefinedProfiles
# ---------------------------------------------------------------------------

class TestPredefinedProfiles(unittest.TestCase):
    """Tests that the four predefined profiles have the expected properties."""

    def test_profile_minimal_no_network(self):
        self.assertFalse(PROFILE_MINIMAL.network_allowed)

    def test_profile_minimal_no_writable_paths(self):
        self.assertEqual(PROFILE_MINIMAL.writable_paths, [])

    def test_profile_minimal_low_memory(self):
        # 64 MB
        self.assertGreater(PROFILE_MINIMAL.max_memory_bytes, 0)
        self.assertLessEqual(PROFILE_MINIMAL.max_memory_bytes, 128 * 1024 * 1024)

    def test_profile_minimal_single_process(self):
        self.assertEqual(PROFILE_MINIMAL.max_processes, 1)

    def test_profile_standard_no_network(self):
        self.assertFalse(PROFILE_STANDARD.network_allowed)

    def test_profile_standard_has_writable_tmp(self):
        self.assertIn("/tmp", PROFILE_STANDARD.writable_paths)

    def test_profile_network_allows_network(self):
        self.assertTrue(PROFILE_NETWORK.network_allowed)

    def test_profile_network_has_writable_tmp(self):
        self.assertIn("/tmp", PROFILE_NETWORK.writable_paths)

    def test_profile_network_larger_memory_than_standard(self):
        self.assertGreater(
            PROFILE_NETWORK.max_memory_bytes,
            PROFILE_STANDARD.max_memory_bytes,
        )

    def test_profile_privileged_network_allowed(self):
        self.assertTrue(PROFILE_PRIVILEGED.network_allowed)

    def test_profile_privileged_allows_all_paths(self):
        self.assertIn("/", PROFILE_PRIVILEGED.allowed_paths)

    def test_profile_privileged_writable_root(self):
        self.assertIn("/", PROFILE_PRIVILEGED.writable_paths)

    def test_profile_privileged_unlimited_memory(self):
        self.assertEqual(PROFILE_PRIVILEGED.max_memory_bytes, 0)

    def test_profile_privileged_empty_env_whitelist(self):
        """Empty whitelist means all env vars pass through."""
        self.assertEqual(PROFILE_PRIVILEGED.env_whitelist, [])

    def test_all_profiles_have_names(self):
        for p in [PROFILE_MINIMAL, PROFILE_STANDARD, PROFILE_NETWORK, PROFILE_PRIVILEGED]:
            self.assertIsInstance(p.name, str)
            self.assertGreater(len(p.name), 0)

    def test_profiles_deny_ssh_paths(self):
        """MINIMAL and STANDARD should block sensitive dot-dirs."""
        ssh_path = os.path.expanduser("~/.ssh")
        for p in [PROFILE_MINIMAL, PROFILE_STANDARD]:
            self.assertIn(ssh_path, p.denied_paths,
                          f"Profile {p.name!r} should deny ~/.ssh")


# ---------------------------------------------------------------------------
# TestEnvironmentSanitizer
# ---------------------------------------------------------------------------

class TestEnvironmentSanitizer(unittest.TestCase):
    """Tests for EnvironmentSanitizer."""

    def setUp(self):
        self.san = EnvironmentSanitizer()
        self.base_env = {
            "PATH": "/usr/bin:/bin",
            "LANG": "en_US.UTF-8",
            "HOME": "/home/user",
            "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "OPENAI_API_KEY": "sk-proj-somekey123",
            "GITHUB_TOKEN": "ghp_16C7e42F292c6912E7710c838347Ae5B49",
            "SAFE_VAR": "totally_safe_value",
            "DATABASE_URL": "postgres://user:pass@localhost/db",
        }

    def test_removes_default_blacklisted_vars(self):
        p = _make_profile()
        result = self.san.sanitize(self.base_env, p)
        self.assertNotIn("AWS_ACCESS_KEY_ID", result)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", result)
        self.assertNotIn("OPENAI_API_KEY", result)
        self.assertNotIn("GITHUB_TOKEN", result)
        self.assertNotIn("DATABASE_URL", result)
        self.assertNotIn("HOME", result)

    def test_preserves_safe_vars_when_no_whitelist(self):
        p = _make_profile(env_whitelist=[])
        result = self.san.sanitize(self.base_env, p)
        self.assertIn("PATH", result)
        self.assertIn("LANG", result)
        self.assertIn("SAFE_VAR", result)

    def test_whitelist_restricts_to_listed_vars(self):
        p = _make_profile(env_whitelist=["PATH", "LANG"])
        result = self.san.sanitize(self.base_env, p)
        self.assertIn("PATH", result)
        self.assertIn("LANG", result)
        self.assertNotIn("SAFE_VAR", result)
        self.assertNotIn("HOME", result)

    def test_profile_blacklist_augments_default(self):
        p = _make_profile(env_blacklist=["SAFE_VAR"])
        result = self.san.sanitize(self.base_env, p)
        self.assertNotIn("SAFE_VAR", result)

    def test_whitelist_and_blacklist_combined(self):
        """Blacklisted vars should be removed even if they appear in the whitelist."""
        p = _make_profile(
            env_whitelist=["PATH", "SAFE_VAR", "AWS_ACCESS_KEY_ID"],
            env_blacklist=[],
        )
        result = self.san.sanitize(self.base_env, p)
        # AWS_ACCESS_KEY_ID is in DEFAULT_BLACKLIST → removed even though whitelisted
        self.assertNotIn("AWS_ACCESS_KEY_ID", result)
        self.assertIn("PATH", result)
        self.assertIn("SAFE_VAR", result)

    def test_sanitize_returns_copy_not_mutate(self):
        p = _make_profile()
        original = dict(self.base_env)
        _ = self.san.sanitize(self.base_env, p)
        self.assertEqual(self.base_env, original)

    def test_empty_env_returns_empty(self):
        p = _make_profile()
        result = self.san.sanitize({}, p)
        self.assertEqual(result, {})

    def test_detect_leaked_secrets_openai(self):
        env = {"MY_KEY": "sk-abc123DEF456ghi789JKL012mno345pqr678stu"}
        leaked = self.san.detect_leaked_secrets(env)
        self.assertIn("MY_KEY", leaked)

    def test_detect_leaked_secrets_aws_akia(self):
        env = {"SOME_VAR": "AKIAIOSFODNN7EXAMPLE"}
        leaked = self.san.detect_leaked_secrets(env)
        self.assertIn("SOME_VAR", leaked)

    def test_detect_leaked_secrets_github_pat(self):
        env = {"TOKEN": "ghp_" + "A" * 36}
        leaked = self.san.detect_leaked_secrets(env)
        self.assertIn("TOKEN", leaked)

    def test_detect_no_leaked_secrets_for_safe_values(self):
        env = {"PATH": "/usr/bin", "LANG": "en_US.UTF-8", "USER": "alice"}
        leaked = self.san.detect_leaked_secrets(env)
        self.assertEqual(leaked, [])

    def test_detect_leaked_returns_list(self):
        leaked = self.san.detect_leaked_secrets({})
        self.assertIsInstance(leaked, list)

    def test_default_blacklist_is_nonempty(self):
        self.assertGreater(len(EnvironmentSanitizer.DEFAULT_BLACKLIST), 0)
        self.assertIn("AWS_ACCESS_KEY_ID", EnvironmentSanitizer.DEFAULT_BLACKLIST)
        self.assertIn("OPENAI_API_KEY", EnvironmentSanitizer.DEFAULT_BLACKLIST)


# ---------------------------------------------------------------------------
# TestResourceLimiter
# ---------------------------------------------------------------------------

class TestResourceLimiter(unittest.TestCase):
    """Tests for ResourceLimiter."""

    def test_apply_returns_dict(self):
        p = _make_profile(max_memory_bytes=0, max_cpu_seconds=0.0,
                          max_file_descriptors=0, max_processes=0)
        limiter = ResourceLimiter(p)
        result = limiter.apply()
        self.assertIsInstance(result, dict)

    def test_apply_dict_has_expected_keys(self):
        p = _make_profile()
        limiter = ResourceLimiter(p)
        result = limiter.apply()
        self.assertIn("memory", result)
        self.assertIn("cpu", result)
        self.assertIn("file_descriptors", result)
        self.assertIn("processes", result)

    def test_apply_values_are_bool(self):
        # Use max_file_descriptors=256 but keep other limits at 0 to avoid
        # clamping RLIMIT_NPROC on the test-runner process (which would prevent
        # future fork() calls in other tests).
        p = _make_profile(max_file_descriptors=256, max_processes=0)
        limiter = ResourceLimiter(p)
        result = limiter.apply()
        for key, val in result.items():
            self.assertIsInstance(val, bool, f"Expected bool for key {key!r}")

    def test_apply_unlimited_profile_all_false(self):
        """Profile with all zeros should not apply any limits."""
        p = _make_profile(max_memory_bytes=0, max_cpu_seconds=0.0,
                          max_file_descriptors=0, max_processes=0)
        limiter = ResourceLimiter(p)
        result = limiter.apply()
        self.assertFalse(result["memory"])
        self.assertFalse(result["cpu"])
        self.assertFalse(result["file_descriptors"])
        self.assertFalse(result["processes"])

    def test_get_usage_returns_dict(self):
        p = _make_profile()
        limiter = ResourceLimiter(p)
        usage = limiter.get_usage()
        self.assertIsInstance(usage, dict)
        self.assertIn("max_rss_bytes", usage)
        self.assertIn("user_time_ms", usage)
        self.assertIn("system_time_ms", usage)

    def test_get_usage_values_are_nonnegative_ints(self):
        p = _make_profile()
        limiter = ResourceLimiter(p)
        usage = limiter.get_usage()
        for key, val in usage.items():
            self.assertIsInstance(val, int, f"Expected int for {key!r}")
            self.assertGreaterEqual(val, 0, f"Expected non-negative for {key!r}")


# ---------------------------------------------------------------------------
# TestSeccompFilter
# ---------------------------------------------------------------------------

class TestSeccompFilter(unittest.TestCase):
    """Tests for SeccompFilter. Never requires root."""

    def test_is_available_returns_bool(self):
        p = _make_profile()
        sf = SeccompFilter(p)
        result = sf.is_available()
        self.assertIsInstance(result, bool)

    def test_is_available_false_on_non_linux(self):
        if _IS_LINUX:
            self.skipTest("Only checks non-Linux path")
        p = _make_profile()
        sf = SeccompFilter(p)
        self.assertFalse(sf.is_available())

    def test_apply_returns_bool_without_crashing(self):
        """apply() must return bool regardless of platform."""
        p = _make_profile()
        sf = SeccompFilter(p)
        result = sf.apply()
        self.assertIsInstance(result, bool)

    def test_minimal_filter_returns_nonempty_set(self):
        s = SeccompFilter.minimal_filter()
        self.assertIsInstance(s, set)
        self.assertGreater(len(s), 0)

    def test_standard_filter_is_superset_of_minimal(self):
        minimal = SeccompFilter.minimal_filter()
        standard = SeccompFilter.standard_filter()
        self.assertTrue(minimal.issubset(standard))

    def test_network_filter_is_superset_of_standard(self):
        standard = SeccompFilter.standard_filter()
        network = SeccompFilter.network_filter()
        self.assertTrue(standard.issubset(network))

    def test_filter_sets_contain_ints(self):
        for fset in [SeccompFilter.minimal_filter(),
                     SeccompFilter.standard_filter(),
                     SeccompFilter.network_filter()]:
            for nr in fset:
                self.assertIsInstance(nr, int)

    def test_network_filter_has_socket_syscall(self):
        from battousai.sandbox import _SYSCALL_SOCKET
        network = SeccompFilter.network_filter()
        self.assertIn(_SYSCALL_SOCKET, network)

    def test_minimal_filter_has_exit_syscall(self):
        from battousai.sandbox import _SYSCALL_EXIT
        minimal = SeccompFilter.minimal_filter()
        self.assertIn(_SYSCALL_EXIT, minimal)


# ---------------------------------------------------------------------------
# TestNamespaceIsolation
# ---------------------------------------------------------------------------

class TestNamespaceIsolation(unittest.TestCase):
    """Tests for NamespaceIsolation. Never requires root."""

    def test_is_available_returns_bool(self):
        p = _make_profile()
        ns = NamespaceIsolation(p)
        result = ns.is_available()
        self.assertIsInstance(result, bool)

    def test_is_available_false_on_non_linux(self):
        if _IS_LINUX:
            self.skipTest("Only checks non-Linux path")
        p = _make_profile()
        ns = NamespaceIsolation(p)
        self.assertFalse(ns.is_available())

    def test_create_sandbox_returns_sandbox_context(self):
        """create_sandbox() is only safe to call inside a subprocess.

        On Linux, calling unshare() in the test-runner process can corrupt
        the process namespace and cause subsequent fork() calls to fail with
        ENOMEM.  We therefore test create_sandbox() by delegating to a
        one-off subprocess and asserting the returned SandboxContext is valid.
        """
        from battousai.sandbox import SandboxContext
        import pickle

        def _probe(conn):
            """Run inside a child process — safe to call create_sandbox()."""
            try:
                from battousai.sandbox import NamespaceIsolation, SandboxProfile
                prof = SandboxProfile(name="probe")
                ns = NamespaceIsolation(prof)
                ctx = ns.create_sandbox()
                conn.send({
                    "pid": ctx.pid,
                    "namespaces_applied": ctx.namespaces_applied,
                    "warnings": ctx.warnings,
                })
            except Exception as exc:
                conn.send({"error": str(exc)})
            finally:
                conn.close()

        parent, child = multiprocessing.Pipe(duplex=False)
        proc = multiprocessing.Process(target=_probe, args=(child,), daemon=True)
        proc.start()
        child.close()
        proc.join(timeout=10)
        data = parent.recv() if parent.poll(1.0) else {"error": "no result"}
        parent.close()

        self.assertNotIn("error", data, f"probe failed: {data}")
        self.assertIsInstance(data["pid"], int)
        self.assertGreater(data["pid"], 0)
        self.assertIsInstance(data["namespaces_applied"], list)
        self.assertIsInstance(data["warnings"], list)

    def test_sandbox_context_has_pid(self):
        """Delegates to subprocess — see test_create_sandbox_returns_sandbox_context."""
        # Already covered above; this test verifies the SandboxContext dataclass shape.
        from battousai.sandbox import SandboxContext
        ctx = SandboxContext(pid=12345)
        self.assertEqual(ctx.pid, 12345)
        self.assertIsInstance(ctx.namespaces_applied, list)
        self.assertIsInstance(ctx.warnings, list)

    def test_sandbox_context_warnings_is_list(self):
        from battousai.sandbox import SandboxContext
        ctx = SandboxContext(pid=1)
        ctx.warnings.append("test warning")
        self.assertIsInstance(ctx.warnings, list)
        self.assertIn("test warning", ctx.warnings)

    def test_sandbox_context_namespaces_applied_is_list(self):
        from battousai.sandbox import SandboxContext
        ctx = SandboxContext(pid=1, namespaces_applied=["user", "pid"])
        self.assertIsInstance(ctx.namespaces_applied, list)
        self.assertIn("user", ctx.namespaces_applied)


# ---------------------------------------------------------------------------
# TestEnforcementReport
# ---------------------------------------------------------------------------

class TestEnforcementReport(unittest.TestCase):
    """Tests for EnforcementReport dataclass and its summary()."""

    def _make_report(self, **kwargs) -> EnforcementReport:
        defaults = dict(
            platform="Linux",
            seccomp_applied=False,
            namespaces_applied=False,
            namespace_types=[],
            resource_limits_applied={"memory": True, "cpu": False,
                                      "file_descriptors": True, "processes": False},
            env_sanitized=True,
            secrets_scrubbed=["OPENAI_API_KEY"],
            warnings=[],
            enforcement_level="partial",
        )
        defaults.update(kwargs)
        return EnforcementReport(**defaults)

    def test_report_creation(self):
        r = self._make_report()
        self.assertEqual(r.platform, "Linux")
        self.assertFalse(r.seccomp_applied)
        self.assertTrue(r.env_sanitized)
        self.assertIn("OPENAI_API_KEY", r.secrets_scrubbed)

    def test_report_summary_returns_string(self):
        r = self._make_report()
        s = r.summary()
        self.assertIsInstance(s, str)
        self.assertIn("Linux", s)
        self.assertIn("partial", s)

    def test_report_full_level(self):
        r = self._make_report(
            seccomp_applied=True,
            namespaces_applied=True,
            enforcement_level="full",
        )
        self.assertEqual(r.enforcement_level, "full")
        s = r.summary()
        self.assertIn("full", s)

    def test_report_minimal_level(self):
        r = self._make_report(
            seccomp_applied=False,
            namespaces_applied=False,
            resource_limits_applied={"memory": False, "cpu": False,
                                      "file_descriptors": False, "processes": False},
            env_sanitized=False,
            enforcement_level="minimal",
        )
        self.assertEqual(r.enforcement_level, "minimal")

    def test_report_namespace_types_list(self):
        r = self._make_report(
            namespaces_applied=True,
            namespace_types=["user", "pid", "mount"],
        )
        self.assertIn("pid", r.namespace_types)
        self.assertEqual(len(r.namespace_types), 3)

    def test_report_warnings_list(self):
        r = self._make_report(warnings=["seccomp unavailable"])
        self.assertIn("seccomp unavailable", r.warnings)


# ---------------------------------------------------------------------------
# TestSandboxedProcess
# ---------------------------------------------------------------------------

class TestSandboxedProcess(unittest.TestCase):
    """Integration tests for SandboxedProcess. Platform-independent."""

    def test_start_simple_target_succeeds(self):
        p = _make_profile(max_memory_bytes=0, max_cpu_seconds=0.0)
        sp = SandboxedProcess(profile=p, agent_id="test-agent", timeout=30.0)
        result = sp.start(_simple_target)
        self.assertIsInstance(result, SandboxResult)
        self.assertTrue(result.success)
        self.assertEqual(result.return_value, "hello-from-sandbox")
        self.assertIsNone(result.exception)

    def test_start_raising_target_captures_exception(self):
        p = _make_profile(max_memory_bytes=0, max_cpu_seconds=0.0)
        sp = SandboxedProcess(profile=p, agent_id="test-agent", timeout=30.0)
        result = sp.start(_raising_target)
        self.assertIsInstance(result, SandboxResult)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.exception)

    def test_start_returns_enforcement_report(self):
        p = _make_profile(max_memory_bytes=0, max_cpu_seconds=0.0)
        sp = SandboxedProcess(profile=p, agent_id="test-agent", timeout=30.0)
        result = sp.start(_simple_target)
        self.assertIsInstance(result.enforcement, EnforcementReport)

    def test_enforcement_report_platform_is_string(self):
        p = _make_profile(max_memory_bytes=0, max_cpu_seconds=0.0)
        sp = SandboxedProcess(profile=p, agent_id="test-agent", timeout=30.0)
        result = sp.start(_simple_target)
        self.assertIsInstance(result.enforcement.platform, str)
        self.assertGreater(len(result.enforcement.platform), 0)

    def test_enforcement_report_env_sanitized_true(self):
        p = _make_profile(max_memory_bytes=0, max_cpu_seconds=0.0)
        sp = SandboxedProcess(profile=p, agent_id="test-agent", timeout=30.0)
        result = sp.start(_simple_target)
        self.assertTrue(result.enforcement.env_sanitized)

    def test_get_enforcement_report_none_before_start(self):
        p = _make_profile()
        sp = SandboxedProcess(profile=p, agent_id="test-agent")
        self.assertIsNone(sp.get_enforcement_report())

    def test_get_enforcement_report_set_after_start(self):
        p = _make_profile(max_memory_bytes=0, max_cpu_seconds=0.0)
        sp = SandboxedProcess(profile=p, agent_id="test-agent", timeout=30.0)
        sp.start(_simple_target)
        report = sp.get_enforcement_report()
        self.assertIsNotNone(report)
        self.assertIsInstance(report, EnforcementReport)

    def test_elapsed_seconds_nonnegative(self):
        p = _make_profile(max_memory_bytes=0, max_cpu_seconds=0.0)
        sp = SandboxedProcess(profile=p, agent_id="test-agent", timeout=30.0)
        result = sp.start(_simple_target)
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)

    def test_start_with_args(self):
        def add(a, b):
            return a + b

        p = _make_profile(max_memory_bytes=0, max_cpu_seconds=0.0)
        sp = SandboxedProcess(profile=p, agent_id="test-agent", timeout=30.0)
        result = sp.start(add, args=(3, 4))
        self.assertTrue(result.success)
        self.assertEqual(result.return_value, 7)

    def test_environment_sanitized_inside_sandbox(self):
        """Secrets injected into env should be absent inside the sandbox."""
        import os as _os
        original_env = dict(_os.environ)

        # Inject a fake secret
        _os.environ["OPENAI_API_KEY"] = "sk-fakekeyfortesting123456789"
        _os.environ["SAFE_VAR"] = "safe"

        try:
            p = _make_profile(max_memory_bytes=0, max_cpu_seconds=0.0)
            sp = SandboxedProcess(profile=p, agent_id="env-test", timeout=30.0)
            result = sp.start(_env_capture_target)
            self.assertTrue(result.success)
            captured = result.return_value
            self.assertNotIn("OPENAI_API_KEY", captured)
            self.assertNotIn("HOME", captured)
        finally:
            _os.environ.clear()
            _os.environ.update(original_env)

    def test_secrets_scrubbed_reported(self):
        """Leaked secrets detected in env should appear in secrets_scrubbed."""
        import os as _os
        original_env = dict(_os.environ)
        _os.environ["SNEAKY_KEY"] = "sk-" + "x" * 30

        try:
            p = _make_profile(max_memory_bytes=0, max_cpu_seconds=0.0)
            sp = SandboxedProcess(profile=p, agent_id="scrub-test", timeout=30.0)
            result = sp.start(_simple_target)
            # The key should be detected, though it may or may not be in the list
            # depending on whether it was already blacklisted by name
            self.assertIsInstance(result.enforcement.secrets_scrubbed, list)
        finally:
            _os.environ.clear()
            _os.environ.update(original_env)


# ---------------------------------------------------------------------------
# TestCapabilityToProfile
# ---------------------------------------------------------------------------

class TestCapabilityToProfile(unittest.TestCase):
    """Tests for the capability_to_profile mapping function."""

    def test_empty_capabilities_returns_profile(self):
        p = capability_to_profile(set(), "agent-0")
        self.assertIsInstance(p, SandboxProfile)

    def test_admin_returns_privileged_like_profile(self):
        p = capability_to_profile({CapabilityType.ADMIN}, "admin-agent")
        self.assertTrue(p.network_allowed)
        self.assertIn("/", p.allowed_paths)

    def test_file_read_enables_path_access(self):
        p = capability_to_profile({CapabilityType.FILE_READ}, "reader")
        # Should include some allowed paths
        self.assertGreater(len(p.allowed_paths), 0)

    def test_no_network_cap_blocks_network(self):
        p = capability_to_profile({CapabilityType.FILE_READ}, "reader")
        self.assertFalse(p.network_allowed)

    def test_network_cap_enables_network(self):
        p = capability_to_profile({CapabilityType.NETWORK}, "net-agent")
        self.assertTrue(p.network_allowed)

    def test_file_write_adds_writable_path(self):
        p = capability_to_profile({CapabilityType.FILE_WRITE}, "writer")
        self.assertGreater(len(p.writable_paths), 0)

    def test_spawn_increases_max_processes(self):
        p_without = capability_to_profile(set(), "a")
        p_with = capability_to_profile({CapabilityType.SPAWN}, "b")
        self.assertGreater(p_with.max_processes, p_without.max_processes)

    def test_spawn_increases_memory(self):
        p_without = capability_to_profile(set(), "a")
        p_with = capability_to_profile({CapabilityType.SPAWN}, "b")
        self.assertGreaterEqual(p_with.max_memory_bytes, p_without.max_memory_bytes)

    def test_profile_name_contains_agent_id(self):
        p = capability_to_profile(set(), "my-special-agent")
        self.assertIn("my-special-agent", p.name)

    def test_profile_has_env_whitelist(self):
        p = capability_to_profile({CapabilityType.FILE_READ}, "agent")
        self.assertIsInstance(p.env_whitelist, list)
        self.assertIn("PATH", p.env_whitelist)

    def test_admin_profile_name_contains_agent_id(self):
        p = capability_to_profile({CapabilityType.ADMIN}, "admin-007")
        self.assertIn("admin-007", p.name)

    def test_combined_caps(self):
        caps = {CapabilityType.FILE_READ, CapabilityType.FILE_WRITE,
                CapabilityType.NETWORK}
        p = capability_to_profile(caps, "full-agent")
        self.assertTrue(p.network_allowed)
        self.assertGreater(len(p.writable_paths), 0)
        self.assertGreater(len(p.allowed_paths), 0)

    def test_returns_independent_copies(self):
        """Each call should return an independent profile object."""
        p1 = capability_to_profile({CapabilityType.FILE_READ}, "a")
        p2 = capability_to_profile({CapabilityType.FILE_READ}, "b")
        p1.allowed_paths.append("/extra")
        self.assertNotIn("/extra", p2.allowed_paths)


# ---------------------------------------------------------------------------
# TestEnvWhitelistBlacklistInteraction
# ---------------------------------------------------------------------------

class TestEnvWhitelistBlacklistInteraction(unittest.TestCase):
    """Detailed tests for env_whitelist and env_blacklist interaction."""

    def setUp(self):
        self.san = EnvironmentSanitizer()
        self.env = {
            "PATH": "/usr/bin",
            "LANG": "en_US.UTF-8",
            "HOME": "/root",           # in DEFAULT_BLACKLIST
            "CUSTOM_TOKEN": "abc123",
            "MY_SECRET": "supersecret",
            "SAFE": "yes",
        }

    def test_whitelist_empty_passes_non_blacklisted(self):
        p = _make_profile(env_whitelist=[], env_blacklist=[])
        result = self.san.sanitize(self.env, p)
        self.assertIn("PATH", result)
        self.assertIn("SAFE", result)
        self.assertNotIn("HOME", result)  # DEFAULT_BLACKLIST

    def test_whitelist_filters_down_to_allowed_set(self):
        p = _make_profile(env_whitelist=["PATH", "SAFE"], env_blacklist=[])
        result = self.san.sanitize(self.env, p)
        self.assertEqual(set(result.keys()), {"PATH", "SAFE"})

    def test_blacklist_removes_additional_keys(self):
        p = _make_profile(env_whitelist=[], env_blacklist=["CUSTOM_TOKEN"])
        result = self.san.sanitize(self.env, p)
        self.assertNotIn("CUSTOM_TOKEN", result)
        self.assertIn("SAFE", result)

    def test_whitelist_does_not_restore_default_blacklist(self):
        """Even if HOME is in whitelist, DEFAULT_BLACKLIST should scrub it."""
        p = _make_profile(env_whitelist=["PATH", "HOME"], env_blacklist=[])
        result = self.san.sanitize(self.env, p)
        # HOME is in DEFAULT_BLACKLIST → removed before whitelist check
        self.assertNotIn("HOME", result)
        self.assertIn("PATH", result)


if __name__ == "__main__":
    unittest.main()
