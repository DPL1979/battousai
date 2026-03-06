"""
tests/test_mcp_server.py — Comprehensive tests for battousai.mcp_server
=======================================================================
Tests cover:
    - initialize handshake
    - tools/list returns correct tool schemas
    - tools/call executes tools correctly
    - tools/call respects capability restrictions
    - invalid JSON-RPC requests return proper errors
    - protocol version negotiation
    - ping keepalive
    - notification handling (no response returned)
    - audit log population
    - exposed_capabilities filtering
"""

from __future__ import annotations

import io
import json
import unittest
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from battousai.capabilities import (
    CapabilityManager,
    CapabilityType,
)
from battousai.mcp_server import (
    MCP_CAPABILITY_DENIED,
    MCP_NOT_INITIALIZED,
    MCP_PROTOCOL_VERSION,
    MCP_TOOL_NOT_FOUND,
    JSONRPC_INVALID_PARAMS,
    JSONRPC_INVALID_REQUEST,
    JSONRPC_METHOD_NOT_FOUND,
    JSONRPC_PARSE_ERROR,
    MCPProtocolError,
    MCPServer,
    MCPServerConfig,
)
from battousai.tools import ToolManager, ToolSpec, register_builtin_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_server(
    *,
    require_cap: bool = True,
    exposed: Optional[list] = None,
    cap_mgr: Optional[CapabilityManager] = None,
) -> tuple[MCPServer, ToolManager]:
    """Build a pre-configured MCPServer backed by built-in tools."""
    tool_mgr = ToolManager()
    register_builtin_tools(tool_mgr)
    config = MCPServerConfig(
        name="test-server",
        version="0.0.1",
        require_tool_use_capability=require_cap,
        exposed_capabilities=exposed or [],
    )
    server = MCPServer(
        tool_manager=tool_mgr,
        capability_manager=cap_mgr,
        config=config,
        stderr=io.StringIO(),
    )
    return server, tool_mgr


def _send(server: MCPServer, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Call handle_message and parse the JSON response."""
    raw = json.dumps(message)
    resp_str = server.handle_message(raw)
    if resp_str is None:
        return None
    return json.loads(resp_str)


def _initialize(server: MCPServer) -> Dict[str, Any]:
    """Send an initialize request and return the response."""
    return _send(server, {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    })


# ---------------------------------------------------------------------------
# Test: Initialize
# ---------------------------------------------------------------------------

class TestMCPServerInitialize(unittest.TestCase):

    def test_initialize_returns_protocol_version(self):
        server, _ = _make_server()
        resp = _initialize(server)
        self.assertIn("result", resp)
        self.assertEqual(resp["result"]["protocolVersion"], MCP_PROTOCOL_VERSION)

    def test_initialize_returns_server_info(self):
        server, _ = _make_server()
        resp = _initialize(server)
        server_info = resp["result"]["serverInfo"]
        self.assertEqual(server_info["name"], "test-server")
        self.assertEqual(server_info["version"], "0.0.1")

    def test_initialize_returns_capabilities(self):
        server, _ = _make_server()
        resp = _initialize(server)
        self.assertIn("capabilities", resp["result"])
        self.assertIn("tools", resp["result"]["capabilities"])

    def test_initialize_sets_initialized_flag(self):
        server, _ = _make_server()
        self.assertFalse(server._initialized)
        _initialize(server)
        self.assertTrue(server._initialized)

    def test_initialize_with_wrong_protocol_version_still_succeeds(self):
        """Server should accept any version and return its own."""
        server, _ = _make_server()
        resp = _send(server, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "1999-01-01"},
        })
        # Should still respond successfully
        self.assertIn("result", resp)
        self.assertEqual(resp["result"]["protocolVersion"], MCP_PROTOCOL_VERSION)

    def test_tools_list_before_initialize_returns_error(self):
        server, _ = _make_server()
        resp = _send(server, {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        })
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], MCP_NOT_INITIALIZED)


# ---------------------------------------------------------------------------
# Test: Ping
# ---------------------------------------------------------------------------

class TestMCPServerPing(unittest.TestCase):

    def test_ping_before_initialize_succeeds(self):
        """ping is always allowed."""
        server, _ = _make_server()
        resp = _send(server, {"jsonrpc": "2.0", "id": 99, "method": "ping", "params": {}})
        self.assertIn("result", resp)
        self.assertEqual(resp["result"], {})

    def test_ping_after_initialize_succeeds(self):
        server, _ = _make_server()
        _initialize(server)
        resp = _send(server, {"jsonrpc": "2.0", "id": 5, "method": "ping", "params": {}})
        self.assertIn("result", resp)


# ---------------------------------------------------------------------------
# Test: tools/list
# ---------------------------------------------------------------------------

class TestMCPServerToolsList(unittest.TestCase):

    def setUp(self):
        self.server, self.tool_mgr = _make_server()
        _initialize(self.server)

    def test_tools_list_returns_all_registered_tools(self):
        resp = _send(self.server, {
            "jsonrpc": "2.0", "id": 10, "method": "tools/list", "params": {},
        })
        self.assertIn("result", resp)
        names = {t["name"] for t in resp["result"]["tools"]}
        self.assertIn("calculator", names)
        self.assertIn("web_search", names)
        self.assertIn("code_executor", names)

    def test_tools_list_includes_description(self):
        resp = _send(self.server, {
            "jsonrpc": "2.0", "id": 11, "method": "tools/list", "params": {},
        })
        for tool in resp["result"]["tools"]:
            with self.subTest(tool=tool["name"]):
                self.assertIn("description", tool)
                self.assertIsInstance(tool["description"], str)

    def test_tools_list_includes_input_schema(self):
        resp = _send(self.server, {
            "jsonrpc": "2.0", "id": 12, "method": "tools/list", "params": {},
        })
        for tool in resp["result"]["tools"]:
            with self.subTest(tool=tool["name"]):
                self.assertIn("inputSchema", tool)
                schema = tool["inputSchema"]
                self.assertEqual(schema["type"], "object")
                self.assertIn("properties", schema)

    def test_tools_list_filtered_by_exposed_capabilities(self):
        server, _ = _make_server(exposed=["calculator"])
        _initialize(server)
        resp = _send(server, {
            "jsonrpc": "2.0", "id": 13, "method": "tools/list", "params": {},
        })
        names = [t["name"] for t in resp["result"]["tools"]]
        self.assertEqual(names, ["calculator"])
        self.assertNotIn("web_search", names)


# ---------------------------------------------------------------------------
# Test: tools/call — success paths
# ---------------------------------------------------------------------------

class TestMCPServerToolsCallSuccess(unittest.TestCase):

    def setUp(self):
        # Permissive: no capability manager, no cap required
        self.server, self.tool_mgr = _make_server(require_cap=False)
        _initialize(self.server)

    def test_tools_call_calculator_returns_result(self):
        resp = _send(self.server, {
            "jsonrpc": "2.0", "id": 20, "method": "tools/call",
            "params": {"name": "calculator", "arguments": {"expression": "2 + 2"}},
        })
        self.assertIn("result", resp)
        result = resp["result"]
        self.assertFalse(result["isError"])
        # Content is a list with a text entry
        content = result["content"]
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[0]["text"], "4")

    def test_tools_call_web_search_returns_content(self):
        resp = _send(self.server, {
            "jsonrpc": "2.0", "id": 21, "method": "tools/call",
            "params": {"name": "web_search", "arguments": {"query": "quantum computing"}},
        })
        self.assertIn("result", resp)
        self.assertFalse(resp["result"]["isError"])
        text = resp["result"]["content"][0]["text"]
        self.assertIn("quantum", text.lower())

    def test_tools_call_result_content_is_list(self):
        resp = _send(self.server, {
            "jsonrpc": "2.0", "id": 22, "method": "tools/call",
            "params": {"name": "calculator", "arguments": {"expression": "10 * 5"}},
        })
        self.assertIsInstance(resp["result"]["content"], list)

    def test_tools_call_records_audit_entry(self):
        _send(self.server, {
            "jsonrpc": "2.0", "id": 23, "method": "tools/call",
            "params": {"name": "calculator", "arguments": {"expression": "1+1"}},
        })
        log = self.server.audit_log()
        self.assertGreater(len(log), 0)
        last = log[-1]
        self.assertEqual(last.tool_name, "calculator")
        self.assertTrue(last.success)


# ---------------------------------------------------------------------------
# Test: tools/call — capability restrictions
# ---------------------------------------------------------------------------

class TestMCPServerCapabilityGating(unittest.TestCase):

    def setUp(self):
        self.cap_mgr = CapabilityManager()
        self.server, _ = _make_server(require_cap=True, cap_mgr=self.cap_mgr)
        _initialize(self.server)

    def _call(self, tool: str, agent: str = "agent-1") -> Dict[str, Any]:
        return _send(self.server, {
            "jsonrpc": "2.0", "id": 30, "method": "tools/call",
            "_meta": {"x-battousai-agent-id": agent},
            "params": {"name": tool, "arguments": {"expression": "1+1"}},
        })

    def test_call_blocked_when_no_capability(self):
        """Agent with no TOOL_USE cap should be denied."""
        resp = self._call("calculator", agent="agent-no-cap")
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], MCP_CAPABILITY_DENIED)

    def test_call_allowed_when_capability_granted(self):
        """Agent with TOOL_USE(calculator) cap should succeed."""
        self.cap_mgr.create_capability(
            cap_type=CapabilityType.TOOL_USE,
            resource_pattern="calculator",
            agent_id="agent-has-cap",
        )
        resp = self._call("calculator", agent="agent-has-cap")
        self.assertIn("result", resp)
        self.assertFalse(resp["result"]["isError"])

    def test_call_blocked_for_different_tool(self):
        """Cap for calculator should not grant access to web_search."""
        self.cap_mgr.create_capability(
            cap_type=CapabilityType.TOOL_USE,
            resource_pattern="calculator",
            agent_id="agent-calc-only",
        )
        resp = _send(self.server, {
            "jsonrpc": "2.0", "id": 31, "method": "tools/call",
            "_meta": {"x-battousai-agent-id": "agent-calc-only"},
            "params": {"name": "web_search", "arguments": {"query": "test"}},
        })
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], MCP_CAPABILITY_DENIED)

    def test_kernel_agent_always_allowed(self):
        """The special 'kernel' agent bypasses capability checks."""
        resp = self._call("calculator", agent="kernel")
        self.assertIn("result", resp)
        self.assertFalse(resp["result"]["isError"])

    def test_admin_capability_grants_all_tools(self):
        """An ADMIN capability should grant access to all tools."""
        self.cap_mgr.create_capability(
            cap_type=CapabilityType.ADMIN,
            resource_pattern="*",
            agent_id="admin-agent",
        )
        resp = self._call("calculator", agent="admin-agent")
        self.assertIn("result", resp)
        self.assertFalse(resp["result"]["isError"])

    def test_denied_call_recorded_in_audit_log(self):
        _resp = self._call("calculator", agent="agent-denied")
        log = self.server.audit_log()
        denied_entries = [e for e in log if not e.allowed]
        self.assertGreater(len(denied_entries), 0)


# ---------------------------------------------------------------------------
# Test: tools/call — error handling
# ---------------------------------------------------------------------------

class TestMCPServerToolsCallErrors(unittest.TestCase):

    def setUp(self):
        self.server, _ = _make_server(require_cap=False)
        _initialize(self.server)

    def test_call_unknown_tool_returns_error(self):
        resp = _send(self.server, {
            "jsonrpc": "2.0", "id": 40, "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        })
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], MCP_TOOL_NOT_FOUND)

    def test_call_missing_name_returns_invalid_params(self):
        resp = _send(self.server, {
            "jsonrpc": "2.0", "id": 41, "method": "tools/call",
            "params": {"arguments": {}},
        })
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], JSONRPC_INVALID_PARAMS)

    def test_call_non_object_params_returns_invalid_params(self):
        resp = _send(self.server, {
            "jsonrpc": "2.0", "id": 42, "method": "tools/call",
            "params": "not-an-object",
        })
        self.assertIn("error", resp)

    def test_call_tool_not_in_exposed_list_returns_error(self):
        server, _ = _make_server(require_cap=False, exposed=["calculator"])
        _initialize(server)
        resp = _send(server, {
            "jsonrpc": "2.0", "id": 43, "method": "tools/call",
            "params": {"name": "web_search", "arguments": {"query": "hi"}},
        })
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], MCP_TOOL_NOT_FOUND)


# ---------------------------------------------------------------------------
# Test: Invalid JSON-RPC requests
# ---------------------------------------------------------------------------

class TestMCPServerProtocolErrors(unittest.TestCase):

    def setUp(self):
        self.server, _ = _make_server()

    def test_invalid_json_returns_parse_error(self):
        resp_str = self.server.handle_message("{not valid json")
        resp = json.loads(resp_str)
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], JSONRPC_PARSE_ERROR)

    def test_non_object_json_returns_invalid_request(self):
        resp_str = self.server.handle_message('"just a string"')
        resp = json.loads(resp_str)
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], JSONRPC_INVALID_REQUEST)

    def test_missing_method_returns_invalid_request(self):
        resp_str = self.server.handle_message(json.dumps({"jsonrpc": "2.0", "id": 1}))
        resp = json.loads(resp_str)
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], JSONRPC_INVALID_REQUEST)

    def test_unknown_method_after_init_returns_method_not_found(self):
        _initialize(self.server)
        resp = _send(self.server, {
            "jsonrpc": "2.0", "id": 50, "method": "not/a/method", "params": {},
        })
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], JSONRPC_METHOD_NOT_FOUND)

    def test_notification_returns_none(self):
        """Notifications (no 'id') must not produce a response."""
        result = self.server.handle_message(json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }))
        self.assertIsNone(result)

    def test_error_response_has_jsonrpc_2_header(self):
        resp_str = self.server.handle_message("bad json {{")
        resp = json.loads(resp_str)
        self.assertEqual(resp["jsonrpc"], "2.0")

    def test_id_preserved_in_error_response(self):
        _initialize(self.server)
        resp = _send(self.server, {
            "jsonrpc": "2.0", "id": 777, "method": "nonexistent", "params": {},
        })
        self.assertEqual(resp["id"], 777)


# ---------------------------------------------------------------------------
# Test: run() with StringIO transport
# ---------------------------------------------------------------------------

class TestMCPServerRun(unittest.TestCase):

    def test_run_processes_multiple_messages(self):
        """server.run() should handle a stream of newline-delimited messages."""
        tool_mgr = ToolManager()
        register_builtin_tools(tool_mgr)
        config = MCPServerConfig(require_tool_use_capability=False)

        messages = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": MCP_PROTOCOL_VERSION,
                                   "clientInfo": {"name": "t", "version": "1"}}}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
            json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping", "params": {}}),
        ]
        stdin_io = io.StringIO("\n".join(messages) + "\n")
        stdout_io = io.StringIO()
        stderr_io = io.StringIO()

        server = MCPServer(
            tool_manager=tool_mgr,
            config=config,
            stdin=stdin_io,
            stdout=stdout_io,
            stderr=stderr_io,
        )
        server.run()

        output = stdout_io.getvalue().strip().split("\n")
        self.assertEqual(len(output), 3)
        responses = [json.loads(line) for line in output]
        # id 1 → initialize
        self.assertEqual(responses[0]["id"], 1)
        self.assertIn("result", responses[0])
        # id 2 → tools/list
        self.assertEqual(responses[1]["id"], 2)
        self.assertIn("tools", responses[1]["result"])
        # id 3 → ping
        self.assertEqual(responses[2]["id"], 3)


if __name__ == "__main__":
    unittest.main()
