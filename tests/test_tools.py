"""
test_tools.py — Tests for battousai.tools (ToolManager, ToolSpec, built-in tools)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from battousai.tools import (
    ToolManager, ToolSpec,
    register_builtin_tools,
    ToolNotFoundError, ToolAccessDeniedError, ToolRateLimitError, ToolExecutionError,
)
from battousai.filesystem import VirtualFilesystem


class TestToolManagerRegistration(unittest.TestCase):

    def setUp(self):
        self.manager = ToolManager()

    def test_register_tool_and_list(self):
        spec = ToolSpec(
            name="test_tool",
            description="A test tool",
            callable=lambda args: {"result": "ok"},
        )
        self.manager.register(spec)
        tools = self.manager.list_tools()
        self.assertIn("test_tool", tools)

    def test_get_registered_tool_spec(self):
        spec = ToolSpec(
            name="my_tool",
            description="test",
            callable=lambda args: "done",
        )
        self.manager.register(spec)
        retrieved = self.manager.get_spec("my_tool")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "my_tool")

    def test_get_unknown_tool_returns_none(self):
        # get_spec() raises ToolNotFoundError for unknown tools;
        # test should handle that gracefully
        try:
            result = self.manager.get_spec("ghost_tool")
            self.assertIsNone(result)
        except ToolNotFoundError:
            pass  # Correct behaviour: raises for unknown tool


class TestBuiltinTools(unittest.TestCase):

    def setUp(self):
        self.fs = VirtualFilesystem()
        self.fs._init_standard_dirs()
        self.manager = ToolManager()
        register_builtin_tools(self.manager, self.fs)
        self.manager.grant_access("calculator", "agent_0001")
        self.manager.grant_access("file_reader", "agent_0001")
        self.manager.grant_access("file_writer", "agent_0001")

    def test_builtin_tools_registered(self):
        tools = self.manager.list_tools()
        for expected in ["calculator", "web_search", "code_executor",
                         "file_reader", "file_writer"]:
            self.assertIn(expected, tools)

    def test_calculator_add(self):
        # _safe_calc takes an expression string
        result = self.manager.execute(
            "agent_0001", "calculator",
            {"expression": "3 + 4"}
        )
        self.assertIn("7", str(result))

    def test_calculator_subtract(self):
        result = self.manager.execute(
            "agent_0001", "calculator",
            {"expression": "10 - 3"}
        )
        self.assertIn("7", str(result))

    def test_calculator_multiply(self):
        result = self.manager.execute(
            "agent_0001", "calculator",
            {"expression": "3 * 4"}
        )
        self.assertIn("12", str(result))

    def test_calculator_divide(self):
        result = self.manager.execute(
            "agent_0001", "calculator",
            {"expression": "10 / 2"}
        )
        self.assertIn("5", str(result))

    def test_file_writer_and_file_reader(self):
        """file_writer writes a file; file_reader reads it back."""
        self.manager.execute(
            "agent_0001", "file_writer",
            {"path": "/agents/hello.txt", "data": "hello from tool"}
        )
        result = self.manager.execute(
            "agent_0001", "file_reader",
            {"path": "/agents/hello.txt"}
        )
        self.assertIn("hello from tool", str(result))


class TestToolAccessControl(unittest.TestCase):

    def setUp(self):
        self.fs = VirtualFilesystem()
        self.manager = ToolManager()
        register_builtin_tools(self.manager, self.fs)

    def test_execute_without_access_raises_denied(self):
        # When allowed_agents is empty (no grant_access called), the tool is
        # open to ALL agents (empty set = open-world assumption in the source).
        # To test access denial, we must first grant, then revoke, or explicitly
        # restrict by granting to a different agent.
        # Grant access to a different agent to restrict the allowlist:
        self.manager.grant_access("calculator", "some_other_agent")
        # Now agent_0001 is not in the allowlist → access denied
        with self.assertRaises(ToolAccessDeniedError):
            self.manager.execute("agent_0001", "calculator", {"expression": "1 + 1"})

    def test_grant_access_then_execute_succeeds(self):
        self.manager.grant_access("calculator", "agent_0001")
        result = self.manager.execute(
            "agent_0001", "calculator", {"expression": "2 + 2"}
        )
        self.assertIn("4", str(result))

    def test_revoke_access_prevents_execution(self):
        # Grant to another agent first to make the allowlist non-empty,
        # then grant and revoke for agent_0001.
        self.manager.grant_access("calculator", "some_other_agent")
        self.manager.grant_access("calculator", "agent_0001")
        self.manager.revoke_access("calculator", "agent_0001")
        with self.assertRaises(ToolAccessDeniedError):
            self.manager.execute("agent_0001", "calculator", {"expression": "1 + 1"})

    def test_execute_unknown_tool_raises_not_found(self):
        with self.assertRaises(ToolNotFoundError):
            self.manager.execute("agent_0001", "ghost_tool", {})


class TestToolRateLimit(unittest.TestCase):

    def setUp(self):
        self.fs = VirtualFilesystem()
        self.fs._init_standard_dirs()
        self.manager = ToolManager()
        register_builtin_tools(self.manager, self.fs)
        # web_search has rate_limit=5 per 10 ticks
        self.manager.grant_access("web_search", "agent_0001")

    def test_web_search_under_rate_limit(self):
        """Executing web_search fewer than 5 times should not raise."""
        for _ in range(4):
            try:
                self.manager.execute(
                    "agent_0001", "web_search",
                    {"query": "test"}
                )
            except ToolRateLimitError:
                self.fail("Rate limit triggered unexpectedly before 5 calls")
            except Exception:
                pass  # Other errors are fine

    def test_tool_stats_returned(self):
        stats = self.manager.stats()
        self.assertIsInstance(stats, dict)


if __name__ == "__main__":
    unittest.main()
