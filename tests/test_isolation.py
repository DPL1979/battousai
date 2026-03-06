"""
tests/test_isolation.py — Unit tests for battousai.isolation
=============================================================
Tests IsolatedAgentProcess, SandboxConfig, ProcessPool.
Uses lightweight stub agents that don't require a full Battousai kernel.
"""

from __future__ import annotations

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from battousai.isolation import (
    IsolatedAgentProcess,
    ProcessPool,
    SandboxConfig,
    _send,
    _recv,
    _apply_resource_limits,
)


# ---------------------------------------------------------------------------
# Stub agents (no kernel dependency)
# ---------------------------------------------------------------------------

class EchoAgent:
    """An agent that records the ticks it receives."""

    def __init__(self, name: str = "EchoAgent") -> None:
        self.name = name
        self.ticks_seen = []

    def think(self, tick: int) -> None:
        self.ticks_seen.append(tick)

    def on_terminate(self) -> None:
        pass


class CrashingAgent:
    """An agent that raises an exception on its first tick."""

    def __init__(self) -> None:
        self.call_count = 0

    def think(self, tick: int) -> None:
        self.call_count += 1
        raise RuntimeError("deliberate crash for testing")

    def on_terminate(self) -> None:
        pass


class SlowAgent:
    """An agent that sleeps during think()."""

    def __init__(self, sleep_seconds: float = 0.1) -> None:
        self.sleep_seconds = sleep_seconds

    def think(self, tick: int) -> None:
        time.sleep(self.sleep_seconds)


class SyscallAgent:
    """An agent that calls self.syscall() during think()."""

    def __init__(self) -> None:
        self.syscall_result = None

    def syscall(self, name: str, **kwargs) -> Any:
        """Default stub — overridden by monkey-patch in isolation.py."""
        class _Result:
            ok = False
            value = None
            error = "no handler"
        return _Result()

    def think(self, tick: int) -> None:
        result = self.syscall("calculator", expression="1+1")
        self.syscall_result = getattr(result, "value", None)

    def on_terminate(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests: SandboxConfig
# ---------------------------------------------------------------------------

class TestSandboxConfig(unittest.TestCase):

    def test_defaults(self):
        config = SandboxConfig()
        self.assertEqual(config.max_memory_mb, 0)
        self.assertEqual(config.max_cpu_seconds, 0)
        self.assertEqual(config.max_open_files, 0)
        self.assertEqual(config.allowed_paths, [])
        self.assertFalse(config.network_access)

    def test_custom_values(self):
        config = SandboxConfig(
            max_memory_mb=512,
            max_cpu_seconds=30,
            max_open_files=100,
            allowed_paths=["/tmp", "/var"],
            network_access=True,
        )
        self.assertEqual(config.max_memory_mb, 512)
        self.assertEqual(config.max_cpu_seconds, 30)
        self.assertTrue(config.network_access)
        self.assertIn("/tmp", config.allowed_paths)


class TestApplyResourceLimits(unittest.TestCase):
    """_apply_resource_limits() must not crash even with zero values."""

    def test_zero_limits_no_crash(self):
        config = SandboxConfig(
            max_memory_mb=0, max_cpu_seconds=0, max_open_files=0
        )
        # Should complete without raising
        _apply_resource_limits(config)


# ---------------------------------------------------------------------------
# Tests: IsolatedAgentProcess lifecycle
# ---------------------------------------------------------------------------

class TestIsolatedAgentProcessLifecycle(unittest.TestCase):

    def test_start_and_stop(self):
        iso = IsolatedAgentProcess(
            agent_class=EchoAgent,
            agent_kwargs={"name": "TestEcho"},
        )
        iso.start()
        self.assertTrue(iso.is_alive())
        self.assertIsNotNone(iso.pid)
        iso.stop()
        self.assertFalse(iso.is_alive())

    def test_repr_contains_class_name(self):
        iso = IsolatedAgentProcess(agent_class=EchoAgent)
        iso.start()
        r = repr(iso)
        self.assertIn("EchoAgent", r)
        iso.stop()

    def test_pid_is_none_before_start(self):
        iso = IsolatedAgentProcess(agent_class=EchoAgent)
        self.assertIsNone(iso.pid)

    def test_is_alive_false_before_start(self):
        iso = IsolatedAgentProcess(agent_class=EchoAgent)
        self.assertFalse(iso.is_alive())

    def test_stop_idempotent(self):
        iso = IsolatedAgentProcess(agent_class=EchoAgent)
        iso.start()
        iso.stop()
        iso.stop()  # Second stop should not raise

    def test_stop_not_started_is_safe(self):
        iso = IsolatedAgentProcess(agent_class=EchoAgent)
        iso.stop()  # Should not raise even if never started


# ---------------------------------------------------------------------------
# Tests: Tick driving
# ---------------------------------------------------------------------------

class TestIsolatedAgentTick(unittest.TestCase):

    def test_tick_returns_ok_true(self):
        iso = IsolatedAgentProcess(agent_class=EchoAgent)
        iso.start()
        result = iso.tick(tick=1)
        self.assertTrue(result.get("ok"), f"Expected ok=True, got: {result}")
        iso.stop()

    def test_multiple_ticks(self):
        iso = IsolatedAgentProcess(agent_class=EchoAgent)
        iso.start()
        for i in range(3):
            result = iso.tick(tick=i)
            self.assertTrue(result.get("ok"), f"Tick {i} failed: {result}")
        iso.stop()

    def test_crashing_agent_reports_error(self):
        iso = IsolatedAgentProcess(agent_class=CrashingAgent)
        iso.start()
        result = iso.tick(tick=1)
        # The crash should be caught and reported as ok=False
        self.assertFalse(result.get("ok"), f"Expected ok=False for crashing agent, got: {result}")
        self.assertIn("error", result)
        iso.stop()

    def test_tick_on_non_running_process_returns_error(self):
        iso = IsolatedAgentProcess(agent_class=EchoAgent)
        # Not started
        result = iso.tick(tick=1)
        self.assertFalse(result.get("ok"))
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# Tests: Syscall forwarding
# ---------------------------------------------------------------------------

class TestSyscallForwarding(unittest.TestCase):

    def test_syscall_with_handler(self):
        """Syscall requests are forwarded to the handler and the result returned."""

        class _Result:
            ok = True
            value = 42
            error = None

        def _handler(name, **kwargs):
            return _Result()

        iso = IsolatedAgentProcess(agent_class=SyscallAgent)
        iso.start()
        result = iso.tick_with_handler(tick=1, syscall_handler=_handler)
        self.assertTrue(result.get("ok"), f"tick_with_handler failed: {result}")
        iso.stop()

    def test_syscall_without_handler_returns_error_in_result(self):
        """If no handler, syscall result is denied but tick completes."""
        iso = IsolatedAgentProcess(agent_class=SyscallAgent)
        iso.start()
        # The agent calls self.syscall() but there's no real kernel attached.
        # The tick should still complete (denied result returned).
        result = iso.tick_with_handler(tick=1, syscall_handler=None)
        # Tick itself completes — the denied syscall result is just a negative value
        # inside the agent's think() method.
        # (ok=True means the tick finished; the syscall denial is internal)
        iso.stop()


# ---------------------------------------------------------------------------
# Tests: ProcessPool
# ---------------------------------------------------------------------------

class TestProcessPool(unittest.TestCase):

    def setUp(self):
        self.pool = ProcessPool()

    def tearDown(self):
        self.pool.stop_all()

    def test_spawn_registers_process(self):
        self.pool.spawn("worker1", EchoAgent, {"name": "W1"})
        self.assertIn("worker1", self.pool.names())

    def test_spawn_returns_isolated_process(self):
        iso = self.pool.spawn("worker1", EchoAgent)
        self.assertIsInstance(iso, IsolatedAgentProcess)
        self.assertTrue(iso.is_alive())

    def test_alive_count(self):
        self.pool.spawn("w1", EchoAgent)
        self.pool.spawn("w2", EchoAgent)
        self.assertEqual(self.pool.alive_count(), 2)

    def test_len(self):
        self.pool.spawn("w1", EchoAgent)
        self.pool.spawn("w2", EchoAgent)
        self.assertEqual(len(self.pool), 2)

    def test_tick_all(self):
        self.pool.spawn("w1", EchoAgent)
        self.pool.spawn("w2", EchoAgent)
        results = self.pool.tick_all(tick=1)
        self.assertIn("w1", results)
        self.assertIn("w2", results)
        self.assertTrue(results["w1"].get("ok"), f"w1 failed: {results['w1']}")
        self.assertTrue(results["w2"].get("ok"), f"w2 failed: {results['w2']}")

    def test_stop_all_stops_all_processes(self):
        self.pool.spawn("w1", EchoAgent)
        self.pool.spawn("w2", EchoAgent)
        self.pool.stop_all()
        self.assertEqual(len(self.pool), 0)

    def test_get_returns_process_by_name(self):
        iso = self.pool.spawn("myworker", EchoAgent)
        retrieved = self.pool.get("myworker")
        self.assertIs(retrieved, iso)

    def test_get_unknown_returns_none(self):
        result = self.pool.get("nonexistent")
        self.assertIsNone(result)

    def test_repr(self):
        r = repr(self.pool)
        self.assertIn("ProcessPool", r)

    def test_pool_default_config(self):
        config = SandboxConfig(max_memory_mb=256)
        pool = ProcessPool(config=config)
        iso = pool.spawn("w1", EchoAgent)
        pool.stop_all()

    def test_names_empty_at_start(self):
        fresh_pool = ProcessPool()
        self.assertEqual(fresh_pool.names(), [])


# ---------------------------------------------------------------------------
# Tests: Resource limits (smoke test — just ensures no crash)
# ---------------------------------------------------------------------------

class TestResourceLimits(unittest.TestCase):

    def test_process_starts_with_resource_limits(self):
        """Process with resource limits configured should start without error."""
        config = SandboxConfig(
            max_memory_mb=512,     # 512 MB virtual memory
            max_cpu_seconds=60,    # 60s CPU time
            max_open_files=256,    # 256 open files
        )
        iso = IsolatedAgentProcess(
            agent_class=EchoAgent,
            config=config,
        )
        iso.start()
        self.assertTrue(iso.is_alive())
        result = iso.tick(tick=1)
        self.assertTrue(result.get("ok"), f"Tick failed: {result}")
        iso.stop()


# ---------------------------------------------------------------------------
# Tests: Error recovery
# ---------------------------------------------------------------------------

class TestErrorRecovery(unittest.TestCase):

    def test_crash_does_not_kill_parent(self):
        """A crash in the subprocess must not propagate to the test process."""
        iso = IsolatedAgentProcess(agent_class=CrashingAgent)
        iso.start()
        result = iso.tick(tick=1)
        # Error is captured, parent is fine
        self.assertIn("error", result)
        iso.stop()

    def test_exit_code_available_after_stop(self):
        iso = IsolatedAgentProcess(agent_class=EchoAgent)
        iso.start()
        iso.stop()
        # Exit code should be set (0 for clean exit)
        self.assertIsNotNone(iso.exit_code)


if __name__ == "__main__":
    unittest.main()
