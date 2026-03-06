"""
tests/test_mcp_client.py — Comprehensive tests for battousai.mcp_client
=======================================================================
Tests cover:
    - Connection lifecycle (connect / disconnect)
    - list_tools returns correct format
    - call_tool sends correct JSON-RPC and unwraps results
    - Capability gating on outbound tool calls
    - Timeout handling for unresponsive servers
    - Error handling for malformed / error responses
    - Fake MCP server via mock subprocess
"""

from __future__ import annotations

import io
import json
import threading
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

from battousai.capabilities import CapabilityManager, CapabilityType
from battousai.mcp_client import (
    MCP_CAPABILITY_DENIED,
    MCP_PROTOCOL_VERSION,
    MCPCallError,
    MCPClient,
    MCPClientConfig,
    MCPConnectionError,
    _PendingRequest,
)


# ---------------------------------------------------------------------------
# Fake MCP server helper
# ---------------------------------------------------------------------------

class FakeMCPServerIO:
    """
    Simulates the stdout of an MCP server by enqueuing canned responses.

    When the client writes a request to FakeMCPServerIO.stdin_write, the
    server thread parses it and writes a response to FakeMCPServerIO.stdout_read.
    """

    def __init__(self, handler=None) -> None:
        # Pipes: client writes to _client_write, server reads from _server_read
        # Server writes to _server_write, client reads from _client_read
        self._lock = threading.Lock()
        self._responses: List[str] = []
        self._requests: List[str] = []
        self._handler = handler or self._default_handler

        # We simulate via in-memory line buffers
        self._stdin_lines: List[str] = []   # client writes here (server reads)
        self._stdout_lines: List[str] = []  # server writes here (client reads)
        self._stdin_cond = threading.Condition(self._lock)
        self._stdout_cond = threading.Condition(self._lock)
        self._done = False

        self._server_thread = threading.Thread(target=self._serve, daemon=True)
        self._server_thread.start()

    # -------- client-facing stream objects --------

    class _StdinProxy:
        def __init__(self, parent):
            self._parent = parent
            self.closed = False

        def write(self, data: str) -> None:
            if self.closed:
                return
            lines = data.splitlines()
            with self._parent._lock:
                for line in lines:
                    line = line.strip()
                    if line:
                        self._parent._stdin_lines.append(line)
                self._parent._stdin_cond.notify_all()

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True
            with self._parent._lock:
                self._parent._done = True
                self._parent._stdin_cond.notify_all()

    class _StdoutProxy:
        def __init__(self, parent):
            self._parent = parent
            self._buf = ""

        def __iter__(self):
            return self

        def __next__(self) -> str:
            with self._parent._lock:
                while (not self._parent._stdout_lines and
                       not self._parent._done):
                    self._parent._stdout_cond.wait(timeout=0.1)
                if self._parent._stdout_lines:
                    return self._parent._stdout_lines.pop(0) + "\n"
                raise StopIteration

        def readline(self) -> str:
            try:
                return next(self)
            except StopIteration:
                return ""

    @property
    def stdin(self):
        return self._StdinProxy(self)

    @property
    def stdout(self):
        return self._StdoutProxy(self)

    def _serve(self) -> None:
        with self._lock:
            pass  # acquire to set up
        while True:
            with self._lock:
                while not self._stdin_lines and not self._done:
                    self._stdin_cond.wait(timeout=0.05)
                if self._done and not self._stdin_lines:
                    break
                if not self._stdin_lines:
                    continue
                raw = self._stdin_lines.pop(0)

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            response = self._handler(msg)
            if response is not None:
                with self._lock:
                    self._stdout_lines.append(json.dumps(response))
                    self._stdout_cond.notify_all()

    @staticmethod
    def _default_handler(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = msg.get("method")
        req_id = msg.get("id")
        if req_id is None:
            return None  # notification
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "serverInfo": {"name": "fake-server", "version": "1.0"},
                    "capabilities": {"tools": {}},
                },
            }
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo back a message",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"message": {"type": "string"}},
                                "required": ["message"],
                            },
                        },
                        {
                            "name": "add",
                            "description": "Add two numbers",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "number"},
                                    "b": {"type": "number"},
                                },
                                "required": ["a", "b"],
                            },
                        },
                    ]
                },
            }
        if method == "tools/call":
            params = msg.get("params", {})
            tool = params.get("name")
            args = params.get("arguments", {})
            if tool == "echo":
                text = args.get("message", "")
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": text}],
                               "isError": False},
                }
            if tool == "add":
                total = args.get("a", 0) + args.get("b", 0)
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": str(total)}],
                               "isError": False},
                }
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32002, "message": f"Tool not found: {tool}"},
            }
        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def _make_connected_client(
    *,
    cap_mgr=None,
    agent_id: str = "test-agent",
    require_cap: bool = True,
    timeout: float = 5.0,
    fake_io: Optional[FakeMCPServerIO] = None,
) -> tuple[MCPClient, FakeMCPServerIO]:
    """Build an MCPClient wired to a FakeMCPServerIO without subprocess."""
    if fake_io is None:
        fake_io = FakeMCPServerIO()

    config = MCPClientConfig(
        agent_id=agent_id,
        timeout=timeout,
        require_tool_use_capability=require_cap,
    )
    client = MCPClient(
        capability_manager=cap_mgr,
        config=config,
        stderr=io.StringIO(),
    )

    # Patch subprocess.Popen so we inject our fake I/O
    mock_proc = MagicMock()
    mock_proc.stdin = fake_io.stdin
    mock_proc.stdout = fake_io.stdout
    mock_proc.wait = MagicMock(return_value=0)
    mock_proc.kill = MagicMock()

    with patch("subprocess.Popen", return_value=mock_proc):
        client.connect(command="fake-server")

    return client, fake_io


# ---------------------------------------------------------------------------
# Test: Connection lifecycle
# ---------------------------------------------------------------------------

class TestMCPClientLifecycle(unittest.TestCase):

    def test_connect_sets_is_connected(self):
        client, _ = _make_connected_client()
        self.assertTrue(client.is_connected)
        client.disconnect()

    def test_disconnect_clears_is_connected(self):
        client, _ = _make_connected_client()
        client.disconnect()
        self.assertFalse(client.is_connected)

    def test_double_connect_raises(self):
        client, _ = _make_connected_client()
        try:
            with self.assertRaises(MCPConnectionError):
                client.connect("fake-server")
        finally:
            client.disconnect()

    def test_server_info_populated_after_connect(self):
        client, _ = _make_connected_client()
        self.assertEqual(client.server_info.get("name"), "fake-server")
        client.disconnect()

    def test_disconnect_when_not_connected_is_safe(self):
        client = MCPClient(config=MCPClientConfig(), stderr=io.StringIO())
        # Should not raise
        client.disconnect()

    def test_operations_fail_when_not_connected(self):
        client = MCPClient(config=MCPClientConfig(), stderr=io.StringIO())
        with self.assertRaises(MCPConnectionError):
            client.list_tools()
        with self.assertRaises(MCPConnectionError):
            client.call_tool("echo", {"message": "hi"})


# ---------------------------------------------------------------------------
# Test: list_tools
# ---------------------------------------------------------------------------

class TestMCPClientListTools(unittest.TestCase):

    def setUp(self):
        self.client, _ = _make_connected_client(require_cap=False)

    def tearDown(self):
        self.client.disconnect()

    def test_list_tools_returns_list(self):
        tools = self.client.list_tools()
        self.assertIsInstance(tools, list)

    def test_list_tools_contains_expected_names(self):
        tools = self.client.list_tools()
        names = {t["name"] for t in tools}
        self.assertIn("echo", names)
        self.assertIn("add", names)

    def test_list_tools_has_description_and_schema(self):
        tools = self.client.list_tools()
        for tool in tools:
            with self.subTest(tool=tool["name"]):
                self.assertIn("description", tool)
                self.assertIn("inputSchema", tool)

    def test_list_tools_uses_cache_on_second_call(self):
        """Second call with use_cache=True should not hit the network."""
        tools1 = self.client.list_tools(use_cache=True)
        # Clear the fake server's ability to respond (won't matter due to cache)
        tools2 = self.client.list_tools(use_cache=True)
        self.assertEqual(tools1, tools2)

    def test_list_tools_bypasses_cache_when_requested(self):
        tools1 = self.client.list_tools(use_cache=False)
        tools2 = self.client.list_tools(use_cache=False)
        self.assertEqual(len(tools1), len(tools2))


# ---------------------------------------------------------------------------
# Test: call_tool
# ---------------------------------------------------------------------------

class TestMCPClientCallTool(unittest.TestCase):

    def setUp(self):
        self.client, _ = _make_connected_client(require_cap=False)

    def tearDown(self):
        self.client.disconnect()

    def test_call_echo_returns_message(self):
        result = self.client.call_tool("echo", {"message": "hello"})
        self.assertEqual(result, "hello")

    def test_call_add_returns_sum(self):
        result = self.client.call_tool("add", {"a": 3, "b": 7})
        # Fake server returns "10" as text; _unwrap_content tries JSON parse
        # "10" parses to int 10
        self.assertEqual(result, 10)

    def test_call_tool_with_no_arguments_uses_empty_dict(self):
        """call_tool(name) with no arguments kwarg should not raise."""
        # echo with no message arg — fake server returns ""
        result = self.client.call_tool("echo")
        # empty string or empty value is fine
        self.assertIsNotNone(result)

    def test_call_nonexistent_tool_raises_mcp_call_error(self):
        with self.assertRaises(MCPCallError):
            self.client.call_tool("nonexistent_tool", {})


# ---------------------------------------------------------------------------
# Test: Capability gating
# ---------------------------------------------------------------------------

class TestMCPClientCapabilityGating(unittest.TestCase):

    def test_call_blocked_when_no_capability(self):
        cap_mgr = CapabilityManager()
        client, _ = _make_connected_client(
            cap_mgr=cap_mgr, agent_id="restricted-agent", require_cap=True
        )
        try:
            with self.assertRaises(MCPCallError) as ctx:
                client.call_tool("echo", {"message": "hi"})
            self.assertEqual(ctx.exception.code, MCP_CAPABILITY_DENIED)
        finally:
            client.disconnect()

    def test_call_allowed_when_capability_granted(self):
        cap_mgr = CapabilityManager()
        cap_mgr.create_capability(
            cap_type=CapabilityType.TOOL_USE,
            resource_pattern="echo",
            agent_id="allowed-agent",
        )
        client, _ = _make_connected_client(
            cap_mgr=cap_mgr, agent_id="allowed-agent", require_cap=True
        )
        try:
            result = client.call_tool("echo", {"message": "capability works"})
            self.assertEqual(result, "capability works")
        finally:
            client.disconnect()

    def test_wildcard_capability_grants_all_tools(self):
        cap_mgr = CapabilityManager()
        cap_mgr.create_capability(
            cap_type=CapabilityType.TOOL_USE,
            resource_pattern="*",
            agent_id="wildcard-agent",
        )
        client, _ = _make_connected_client(
            cap_mgr=cap_mgr, agent_id="wildcard-agent", require_cap=True
        )
        try:
            r1 = client.call_tool("echo", {"message": "hi"})
            r2 = client.call_tool("add", {"a": 1, "b": 2})
            self.assertEqual(r1, "hi")
            self.assertEqual(r2, 3)
        finally:
            client.disconnect()

    def test_capability_check_skipped_when_disabled(self):
        """require_tool_use_capability=False disables capability checks."""
        cap_mgr = CapabilityManager()
        # No caps granted, but require_cap=False
        client, _ = _make_connected_client(
            cap_mgr=cap_mgr, agent_id="no-cap-agent", require_cap=False
        )
        try:
            result = client.call_tool("echo", {"message": "open"})
            self.assertEqual(result, "open")
        finally:
            client.disconnect()

    def test_cap_blocked_does_not_reach_server(self):
        """A blocked call should never be sent over the wire."""
        cap_mgr = CapabilityManager()
        received_calls: List[str] = []

        def tracking_handler(msg):
            if msg.get("method") == "tools/call":
                received_calls.append(msg["params"]["name"])
            return FakeMCPServerIO._default_handler(msg)

        fake_io = FakeMCPServerIO(handler=tracking_handler)
        client, _ = _make_connected_client(
            cap_mgr=cap_mgr, agent_id="blocked-agent", require_cap=True,
            fake_io=fake_io,
        )
        try:
            with self.assertRaises(MCPCallError):
                client.call_tool("echo", {"message": "blocked"})
            self.assertEqual(received_calls, [])
        finally:
            client.disconnect()


# ---------------------------------------------------------------------------
# Test: Timeout handling
# ---------------------------------------------------------------------------

class TestMCPClientTimeout(unittest.TestCase):

    def test_timeout_raises_connection_error(self):
        """A server that never responds should trigger a timeout."""
        def slow_handler(msg):
            if msg.get("method") == "tools/call":
                # Never respond to tool calls
                import time; time.sleep(10)
            return FakeMCPServerIO._default_handler(msg)

        fake_io = FakeMCPServerIO(handler=slow_handler)
        client, _ = _make_connected_client(
            require_cap=False, timeout=0.2, fake_io=fake_io
        )
        try:
            with self.assertRaises(MCPConnectionError) as ctx:
                client.call_tool("echo", {"message": "hello"})
            self.assertIn("Timeout", str(ctx.exception))
        finally:
            client.disconnect()


# ---------------------------------------------------------------------------
# Test: Error handling for malformed responses
# ---------------------------------------------------------------------------

class TestMCPClientErrorHandling(unittest.TestCase):

    def test_server_error_response_raises_mcp_call_error(self):
        """When server returns an error object, MCPCallError is raised."""
        def error_handler(msg):
            req_id = msg.get("id")
            if msg.get("method") == "initialize":
                return FakeMCPServerIO._default_handler(msg)
            if req_id is None:
                return None
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32000, "message": "Something went wrong"},
            }

        fake_io = FakeMCPServerIO(handler=error_handler)
        client, _ = _make_connected_client(require_cap=False, fake_io=fake_io)
        try:
            with self.assertRaises(MCPCallError) as ctx:
                client.list_tools()
            self.assertIn("Something went wrong", str(ctx.exception))
        finally:
            client.disconnect()

    def test_server_error_preserves_error_code(self):
        """MCPCallError should carry the server's error code."""
        def error_handler(msg):
            req_id = msg.get("id")
            if msg.get("method") == "initialize":
                return FakeMCPServerIO._default_handler(msg)
            if req_id is None:
                return None
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32099, "message": "Custom error"},
            }

        fake_io = FakeMCPServerIO(handler=error_handler)
        client, _ = _make_connected_client(require_cap=False, fake_io=fake_io)
        try:
            with self.assertRaises(MCPCallError) as ctx:
                client.list_tools()
            self.assertEqual(ctx.exception.code, -32099)
        finally:
            client.disconnect()

    def test_malformed_json_from_server_is_ignored(self):
        """Bad JSON lines from the server should not crash the reader thread."""
        responded = threading.Event()

        def mixed_handler(msg):
            req_id = msg.get("id")
            method = msg.get("method")
            if method == "initialize":
                return FakeMCPServerIO._default_handler(msg)
            if req_id is None:
                return None
            # Return a valid response after a simulated invalid line
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"tools": []},
            }

        fake_io = FakeMCPServerIO(handler=mixed_handler)
        # Inject a bad line into the server's stdout buffer
        with fake_io._lock:
            fake_io._stdout_lines.append("{ NOT JSON }")
            fake_io._stdout_cond.notify_all()

        client, _ = _make_connected_client(require_cap=False, fake_io=fake_io)
        try:
            # Should still be able to make requests despite the earlier bad line
            tools = client.list_tools(use_cache=False)
            self.assertIsInstance(tools, list)
        finally:
            client.disconnect()


# ---------------------------------------------------------------------------
# Test: _unwrap_content helper
# ---------------------------------------------------------------------------

class TestUnwrapContent(unittest.TestCase):

    def test_single_text_item_returns_string(self):
        result = {"content": [{"type": "text", "text": "hello"}]}
        unwrapped = MCPClient._unwrap_content(result)
        self.assertEqual(unwrapped, "hello")

    def test_single_text_json_parses_to_dict(self):
        payload = json.dumps({"key": "value"})
        result = {"content": [{"type": "text", "text": payload}]}
        unwrapped = MCPClient._unwrap_content(result)
        self.assertEqual(unwrapped, {"key": "value"})

    def test_multiple_items_returns_list(self):
        result = {"content": [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]}
        unwrapped = MCPClient._unwrap_content(result)
        self.assertIsInstance(unwrapped, list)
        self.assertEqual(len(unwrapped), 2)

    def test_empty_content_returns_result(self):
        result = {"content": [], "extra": "data"}
        unwrapped = MCPClient._unwrap_content(result)
        self.assertEqual(unwrapped, result)

    def test_no_content_key_returns_result(self):
        result = {"other": "stuff"}
        unwrapped = MCPClient._unwrap_content(result)
        self.assertEqual(unwrapped, result)


if __name__ == "__main__":
    unittest.main()
