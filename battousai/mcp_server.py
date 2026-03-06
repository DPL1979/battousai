"""
mcp_server.py — Battousai MCP Server Adapter
=============================================
Exposes Battousai's tool registry as an MCP-compatible server endpoint,
implementing JSON-RPC 2.0 over stdio (stdin/stdout).

MCP Protocol Version: 2024-11-05

The server implements the following MCP methods:
    initialize      — Protocol handshake, return server capabilities
    tools/list      — List all registered tools with JSON schemas
    tools/call      — Execute a tool through the capability-gated ToolManager
    ping            — Keepalive / health check

All tool calls are gated through Battousai's CapabilityManager, and every
invocation is appended to an in-process audit trail.

Usage::

    from battousai.mcp_server import MCPServer, MCPServerConfig
    from battousai.tools import ToolManager, register_builtin_tools
    from battousai.capabilities import CapabilityManager

    cap_mgr = CapabilityManager()
    tool_mgr = ToolManager()
    register_builtin_tools(tool_mgr)

    config = MCPServerConfig(name="battousai", version="0.3.0")
    server = MCPServer(tool_manager=tool_mgr, capability_manager=cap_mgr, config=config)
    server.run()          # blocks, reads JSON-RPC from stdin, writes to stdout
"""

from __future__ import annotations

import io
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TextIO


# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

MCP_PROTOCOL_VERSION = "2024-11-05"

# JSON-RPC 2.0 error codes
JSONRPC_PARSE_ERROR      = -32700
JSONRPC_INVALID_REQUEST  = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS   = -32602
JSONRPC_INTERNAL_ERROR   = -32603

# MCP-specific error codes (in the -32000 to -32099 range)
MCP_CAPABILITY_DENIED    = -32001
MCP_TOOL_NOT_FOUND       = -32002
MCP_TOOL_EXECUTION_ERROR = -32003
MCP_NOT_INITIALIZED      = -32004


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MCPProtocolError(Exception):
    """Raised when the MCP server encounters a protocol-level error."""

    def __init__(self, message: str, code: int = JSONRPC_INTERNAL_ERROR,
                 data: Optional[Any] = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class MCPServerConfig:
    """Configuration for the Battousai MCP server.

    Attributes:
        name: Human-readable server name returned during initialization.
        version: Server version string.
        exposed_capabilities: Subset of registered tools to expose over MCP.
            An empty set means *all* registered tools are exposed.
        require_tool_use_capability: When True, callers must hold a
            ``CapabilityType.TOOL_USE`` token for the specific tool.
        audit_log_max_entries: Maximum number of audit entries to keep in
            memory (oldest entries are dropped when the limit is reached).
        agent_id_header: JSON-RPC request metadata key used to identify the
            calling agent.  Falls back to ``"mcp_client"`` when absent.
    """

    name: str = "battousai-mcp-server"
    version: str = "0.3.0"
    exposed_capabilities: List[str] = field(default_factory=list)
    require_tool_use_capability: bool = True
    audit_log_max_entries: int = 10_000
    agent_id_header: str = "x-battousai-agent-id"


# ---------------------------------------------------------------------------
# Audit entry
# ---------------------------------------------------------------------------

@dataclass
class MCPAuditEntry:
    """A single audit record for an MCP tool call."""

    timestamp: float
    request_id: Optional[Any]
    agent_id: str
    method: str
    tool_name: Optional[str]
    allowed: bool
    success: bool
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "method": self.method,
            "tool_name": self.tool_name,
            "allowed": self.allowed,
            "success": self.success,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# MCPServer
# ---------------------------------------------------------------------------

class MCPServer:
    """MCP server that bridges Battousai's tool registry to the MCP protocol.

    The server speaks JSON-RPC 2.0 over stdio.  Each line on stdin is expected
    to be a complete JSON-RPC message; each response is written as a single
    line to stdout followed by a newline.

    Args:
        tool_manager: The Battousai :class:`~battousai.tools.ToolManager`
            instance whose tools are to be exposed.
        capability_manager: Optional :class:`~battousai.capabilities.CapabilityManager`
            for enforcing tool-use permissions.  When ``None``, all tool calls
            are permitted (permissive mode).
        config: Server configuration.  Defaults to :class:`MCPServerConfig`.
        stdin: Input stream (defaults to ``sys.stdin``).
        stdout: Output stream (defaults to ``sys.stdout``).
        stderr: Error/log stream (defaults to ``sys.stderr``).
    """

    def __init__(
        self,
        tool_manager: Any,
        capability_manager: Optional[Any] = None,
        config: Optional[MCPServerConfig] = None,
        stdin: Optional[TextIO] = None,
        stdout: Optional[TextIO] = None,
        stderr: Optional[TextIO] = None,
    ) -> None:
        self._tool_manager = tool_manager
        self._cap_manager = capability_manager
        self._config = config or MCPServerConfig()
        self._stdin: TextIO = stdin or sys.stdin
        self._stdout: TextIO = stdout or sys.stdout
        self._stderr: TextIO = stderr or sys.stderr
        self._initialized: bool = False
        self._client_info: Dict[str, Any] = {}
        self._audit_log: List[MCPAuditEntry] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Block and process JSON-RPC messages from stdin until EOF."""
        self._log("MCP server starting (protocol version %s)", MCP_PROTOCOL_VERSION)
        for line in self._stdin:
            line = line.strip()
            if not line:
                continue
            response = self._handle_raw(line)
            if response is not None:
                self._write_response(response)
        self._log("MCP server: stdin closed, shutting down.")

    def handle_message(self, raw: str) -> Optional[str]:
        """Process a single raw JSON-RPC message string.

        Returns the serialised JSON response string, or ``None`` for
        notifications (which have no ``id``).
        """
        response = self._handle_raw(raw)
        if response is None:
            return None
        return json.dumps(response)

    def audit_log(self) -> List[MCPAuditEntry]:
        """Return a copy of the in-process audit log."""
        return list(self._audit_log)

    # ------------------------------------------------------------------
    # Transport helpers
    # ------------------------------------------------------------------

    def _write_response(self, response: Dict[str, Any]) -> None:
        self._stdout.write(json.dumps(response) + "\n")
        self._stdout.flush()

    def _log(self, fmt: str, *args: Any) -> None:
        msg = fmt % args if args else fmt
        self._stderr.write(f"[battousai-mcp] {msg}\n")
        self._stderr.flush()

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    def _handle_raw(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse *raw* and dispatch to the appropriate handler."""
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            return self._error_response(None, JSONRPC_PARSE_ERROR,
                                        f"Parse error: {exc}")

        if not isinstance(message, dict):
            return self._error_response(None, JSONRPC_INVALID_REQUEST,
                                        "Request must be a JSON object")

        req_id = message.get("id")  # None for notifications
        method = message.get("method")
        params = message.get("params", {})

        if not isinstance(method, str) or not method:
            return self._error_response(req_id, JSONRPC_INVALID_REQUEST,
                                        "Missing or invalid 'method' field")

        # Notifications (no id) — process but return nothing
        is_notification = "id" not in message

        try:
            result = self._dispatch(method, params, message)
        except MCPProtocolError as exc:
            if is_notification:
                return None
            return self._error_response(req_id, exc.code, str(exc), exc.data)
        except Exception as exc:  # noqa: BLE001
            if is_notification:
                return None
            return self._error_response(req_id, JSONRPC_INTERNAL_ERROR,
                                        f"Internal error: {exc}")

        if is_notification:
            return None
        return self._success_response(req_id, result)

    def _dispatch(self, method: str, params: Any, raw_message: Dict[str, Any]) -> Any:
        """Route a method name to the appropriate handler."""
        # initialize is allowed before the server is considered initialised
        if method == "initialize":
            return self._handle_initialize(params, raw_message)

        if method == "ping":
            return self._handle_ping()

        # All other methods require initialization first
        if not self._initialized:
            raise MCPProtocolError(
                "Server not initialized — send 'initialize' first",
                code=MCP_NOT_INITIALIZED,
            )

        if method == "tools/list":
            return self._handle_tools_list(params)

        if method == "tools/call":
            return self._handle_tools_call(params, raw_message)

        raise MCPProtocolError(
            f"Method not found: {method!r}",
            code=JSONRPC_METHOD_NOT_FOUND,
        )

    # ------------------------------------------------------------------
    # MCP method handlers
    # ------------------------------------------------------------------

    def _handle_initialize(
        self, params: Any, raw_message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle the MCP ``initialize`` handshake."""
        if not isinstance(params, dict):
            params = {}

        self._client_info = params.get("clientInfo", {})
        client_protocol = params.get("protocolVersion", MCP_PROTOCOL_VERSION)

        # We support only one protocol version; log a warning if mismatched
        if client_protocol != MCP_PROTOCOL_VERSION:
            self._log(
                "Protocol version mismatch: client=%s, server=%s",
                client_protocol,
                MCP_PROTOCOL_VERSION,
            )

        self._initialized = True

        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
            },
            "serverInfo": {
                "name": self._config.name,
                "version": self._config.version,
            },
        }

    def _handle_ping(self) -> Dict[str, Any]:
        """Handle the MCP ``ping`` keepalive."""
        return {}

    def _handle_tools_list(self, params: Any) -> Dict[str, Any]:
        """Handle the ``tools/list`` method — returns all exposed tools."""
        tools = []
        tool_names = self._tool_manager.list_tools()

        # Filter to exposed_capabilities if configured
        if self._config.exposed_capabilities:
            tool_names = [t for t in tool_names
                          if t in self._config.exposed_capabilities]

        for name in tool_names:
            try:
                spec = self._tool_manager.get_spec(name)
            except Exception:  # noqa: BLE001
                continue
            tools.append(self._tool_spec_to_mcp(spec))

        return {"tools": tools}

    def _handle_tools_call(
        self, params: Any, raw_message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle the ``tools/call`` method — execute a tool."""
        if not isinstance(params, dict):
            raise MCPProtocolError("'params' must be an object", code=JSONRPC_INVALID_PARAMS)

        tool_name: Optional[str] = params.get("name")
        arguments: Dict[str, Any] = params.get("arguments", {})
        meta: Dict[str, Any] = raw_message.get("_meta", {})

        if not tool_name:
            raise MCPProtocolError(
                "Missing required parameter 'name'", code=JSONRPC_INVALID_PARAMS
            )
        if not isinstance(arguments, dict):
            raise MCPProtocolError(
                "'arguments' must be an object", code=JSONRPC_INVALID_PARAMS
            )

        # Determine caller identity
        agent_id = meta.get(self._config.agent_id_header, "mcp_client")

        # Capability check
        allowed = self._check_tool_capability(agent_id, tool_name)

        self._record_audit(
            request_id=raw_message.get("id"),
            agent_id=agent_id,
            method="tools/call",
            tool_name=tool_name,
            allowed=allowed,
            success=False,  # will update below
        )

        if not allowed:
            # update the last audit entry to final state
            self._audit_log[-1].success = False
            raise MCPProtocolError(
                f"Agent {agent_id!r} lacks TOOL_USE capability for {tool_name!r}",
                code=MCP_CAPABILITY_DENIED,
                data={"tool": tool_name, "agent": agent_id},
            )

        # Verify the tool exists
        try:
            self._tool_manager.get_spec(tool_name)
        except Exception:
            self._audit_log[-1].success = False
            self._audit_log[-1].error = f"Tool not found: {tool_name}"
            raise MCPProtocolError(
                f"Tool not found: {tool_name!r}", code=MCP_TOOL_NOT_FOUND
            )

        # Verify the tool is in exposed_capabilities (if restricted)
        if (self._config.exposed_capabilities and
                tool_name not in self._config.exposed_capabilities):
            self._audit_log[-1].success = False
            self._audit_log[-1].error = f"Tool not exposed: {tool_name}"
            raise MCPProtocolError(
                f"Tool {tool_name!r} is not exposed by this server",
                code=MCP_TOOL_NOT_FOUND,
            )

        # Execute via the ToolManager (which enforces its own access control)
        try:
            result = self._tool_manager.execute(
                agent_id=agent_id,
                tool_name=tool_name,
                args=arguments,
            )
            self._audit_log[-1].success = True
        except Exception as exc:
            self._audit_log[-1].success = False
            self._audit_log[-1].error = str(exc)
            raise MCPProtocolError(
                f"Tool execution failed: {exc}",
                code=MCP_TOOL_EXECUTION_ERROR,
                data={"tool": tool_name, "error": str(exc)},
            )

        # Normalise result to MCP content format
        content = self._result_to_content(result)
        return {"content": content, "isError": False}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_tool_capability(self, agent_id: str, tool_name: str) -> bool:
        """Return True if *agent_id* is allowed to use *tool_name*."""
        if not self._config.require_tool_use_capability:
            return True
        if self._cap_manager is None:
            # No capability manager configured — permissive
            return True
        if agent_id == "kernel":
            return True
        from battousai.capabilities import CapabilityType
        return self._cap_manager.check(
            agent_id=agent_id,
            cap_type=CapabilityType.TOOL_USE,
            resource=tool_name,
        )

    def _tool_spec_to_mcp(self, spec: Any) -> Dict[str, Any]:
        """Convert a Battousai :class:`~battousai.tools.ToolSpec` to MCP tool schema."""
        import inspect

        # Build a minimal JSON Schema from the callable's signature
        try:
            sig = inspect.signature(spec.callable)
            properties: Dict[str, Any] = {}
            required: List[str] = []
            for param_name, param in sig.parameters.items():
                if param_name in ("self",):
                    continue
                param_schema: Dict[str, Any] = {"type": "string"}
                if param.annotation != inspect.Parameter.empty:
                    ann = param.annotation
                    if ann in (int,):
                        param_schema = {"type": "integer"}
                    elif ann in (float,):
                        param_schema = {"type": "number"}
                    elif ann in (bool,):
                        param_schema = {"type": "boolean"}
                    elif ann in (dict, Dict):
                        param_schema = {"type": "object"}
                    elif ann in (list, List):
                        param_schema = {"type": "array"}
                properties[param_name] = param_schema
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)
        except (ValueError, TypeError):
            properties = {}
            required = []

        return {
            "name": spec.name,
            "description": spec.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    @staticmethod
    def _result_to_content(result: Any) -> List[Dict[str, Any]]:
        """Convert a raw tool result to MCP content array."""
        if isinstance(result, str):
            text = result
        elif isinstance(result, (dict, list)):
            text = json.dumps(result, ensure_ascii=False)
        else:
            text = str(result)
        return [{"type": "text", "text": text}]

    def _record_audit(
        self,
        request_id: Optional[Any],
        agent_id: str,
        method: str,
        tool_name: Optional[str],
        allowed: bool,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        entry = MCPAuditEntry(
            timestamp=time.time(),
            request_id=request_id,
            agent_id=agent_id,
            method=method,
            tool_name=tool_name,
            allowed=allowed,
            success=success,
            error=error,
        )
        self._audit_log.append(entry)
        # Cap the log size
        if len(self._audit_log) > self._config.audit_log_max_entries:
            self._audit_log = self._audit_log[-self._config.audit_log_max_entries:]

    # ------------------------------------------------------------------
    # JSON-RPC response builders
    # ------------------------------------------------------------------

    @staticmethod
    def _success_response(req_id: Any, result: Any) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error_response(
        req_id: Optional[Any],
        code: int,
        message: str,
        data: Optional[Any] = None,
    ) -> Dict[str, Any]:
        error_obj: Dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error_obj["data"] = data
        return {"jsonrpc": "2.0", "id": req_id, "error": error_obj}
