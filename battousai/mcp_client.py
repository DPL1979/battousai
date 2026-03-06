"""
mcp_client.py — Battousai MCP Client Adapter
=============================================
Allows Battousai agents to connect to *external* MCP servers, discover their
tools, and call them — all with Battousai's capability model enforced on every
outbound call.

MCP Protocol Version: 2024-11-05

The client spawns the external MCP server as a subprocess and communicates over
its stdin/stdout using JSON-RPC 2.0.  A background reader thread continuously
drains the subprocess's stdout so that responses are matched to outstanding
requests without deadlocks.

Usage::

    from battousai.mcp_client import MCPClient, MCPClientConfig
    from battousai.capabilities import CapabilityManager, CapabilityType

    cap_mgr = CapabilityManager()
    # Grant the agent permission to use the remote 'read_file' tool
    cap_mgr.create_capability(CapabilityType.TOOL_USE, "read_file", "agent-1")

    config = MCPClientConfig(agent_id="agent-1", timeout=10.0)
    client = MCPClient(capability_manager=cap_mgr, config=config)
    client.connect(command="python", args=["-m", "some_mcp_server"])

    tools = client.list_tools()
    result = client.call_tool("read_file", {"path": "/tmp/hello.txt"})
    client.disconnect()
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Protocol constants (mirror from mcp_server)
# ---------------------------------------------------------------------------

MCP_PROTOCOL_VERSION = "2024-11-05"

JSONRPC_PARSE_ERROR      = -32700
JSONRPC_INVALID_REQUEST  = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS   = -32602
JSONRPC_INTERNAL_ERROR   = -32603
MCP_CAPABILITY_DENIED    = -32001


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MCPConnectionError(Exception):
    """Raised for connection-level failures (spawn, timeout, protocol)."""


class MCPCallError(Exception):
    """Raised when the remote server returns a JSON-RPC error result."""

    def __init__(self, message: str, code: int = JSONRPC_INTERNAL_ERROR,
                 data: Optional[Any] = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MCPClientConfig:
    """Configuration for the Battousai MCP client.

    Attributes:
        agent_id: Battousai agent identifier — sent in every request and used
            for capability checks.
        timeout: Seconds to wait for a response before raising
            :class:`MCPConnectionError`.
        require_tool_use_capability: When True the client checks Battousai's
            ``CapabilityManager`` before sending a ``tools/call`` request.
        client_name: Reported to the remote server during handshake.
        client_version: Reported to the remote server during handshake.
        max_pending_requests: Maximum number of in-flight requests.
    """

    agent_id: str = "battousai-agent"
    timeout: float = 30.0
    require_tool_use_capability: bool = True
    client_name: str = "battousai-mcp-client"
    client_version: str = "0.3.0"
    max_pending_requests: int = 64


# ---------------------------------------------------------------------------
# Internal pending-request bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class _PendingRequest:
    """Represents an in-flight JSON-RPC request awaiting a response."""

    request_id: int
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[Dict[str, Any]] = field(default=None)


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------

class MCPClient:
    """MCP client that connects to an external MCP server subprocess.

    All outbound ``tools/call`` requests are gated through Battousai's
    :class:`~battousai.capabilities.CapabilityManager` before being forwarded
    to the remote server.

    Args:
        capability_manager: Optional ``CapabilityManager``.  When ``None``,
            capability checking is disabled (permissive mode).
        config: Client configuration.  Defaults to :class:`MCPClientConfig`.
        stderr: Stream for diagnostic output (defaults to ``sys.stderr``).
    """

    def __init__(
        self,
        capability_manager: Optional[Any] = None,
        config: Optional[MCPClientConfig] = None,
        stderr: Optional[Any] = None,
    ) -> None:
        self._cap_manager = capability_manager
        self._config = config or MCPClientConfig()
        self._stderr = stderr or sys.stderr

        self._process: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        self._connected: bool = False
        self._next_id: int = 1
        self._id_lock = threading.Lock()

        # Map of request_id -> _PendingRequest
        self._pending: Dict[int, _PendingRequest] = {}
        self._pending_lock = threading.Lock()

        # Background reader thread
        self._reader_thread: Optional[threading.Thread] = None
        self._stopping: bool = False

        # Cached tool listing
        self._tools_cache: Optional[List[Dict[str, Any]]] = None

        # Server info from handshake
        self._server_info: Dict[str, Any] = {}
        self._server_capabilities: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, command: str, args: Optional[List[str]] = None) -> None:
        """Spawn the MCP server subprocess and perform the MCP handshake.

        Args:
            command: Executable to run (e.g. ``"python"`` or ``"npx"``).
            args: Additional arguments (e.g. ``["-m", "my_mcp_server"]``).

        Raises:
            MCPConnectionError: If the subprocess cannot be started or the
                handshake times out / fails.
        """
        if self._connected:
            raise MCPConnectionError("Already connected — call disconnect() first")

        cmd = [command] + (args or [])
        self._log("Spawning MCP server: %s", " ".join(cmd))

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered
            )
        except OSError as exc:
            raise MCPConnectionError(f"Failed to spawn MCP server: {exc}") from exc

        self._stopping = False
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="battousai-mcp-reader",
        )
        self._reader_thread.start()

        # MCP handshake
        try:
            self._do_initialize()
        except Exception as exc:
            self._cleanup()
            raise MCPConnectionError(f"Handshake failed: {exc}") from exc

        self._connected = True
        self._log("Connected to MCP server: %s", self._server_info)

    def disconnect(self) -> None:
        """Cleanly shut down the connection to the MCP server."""
        if not self._connected and self._process is None:
            return
        self._log("Disconnecting from MCP server")
        self._cleanup()
        self._connected = False
        self._tools_cache = None
        self._log("Disconnected")

    @property
    def is_connected(self) -> bool:
        """True if the client has a live connection to an MCP server."""
        return self._connected

    @property
    def server_info(self) -> Dict[str, Any]:
        """Server metadata returned during the initialize handshake."""
        return dict(self._server_info)

    # ------------------------------------------------------------------
    # Public MCP operations
    # ------------------------------------------------------------------

    def list_tools(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Discover tools available on the remote MCP server.

        Args:
            use_cache: If True and a previous listing is cached, return it
                without a round-trip.

        Returns:
            A list of tool descriptors, each a dict with keys ``name``,
            ``description``, and ``inputSchema``.

        Raises:
            MCPConnectionError: If not connected or the request times out.
        """
        self._require_connected()
        if use_cache and self._tools_cache is not None:
            return list(self._tools_cache)

        result = self._rpc_call("tools/list", {})
        tools = result.get("tools", [])
        self._tools_cache = tools
        return list(tools)

    def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Call a tool on the remote MCP server.

        Before the request is sent, Battousai's :class:`CapabilityManager` is
        consulted to verify that the configured ``agent_id`` holds a
        ``TOOL_USE`` capability for ``name``.

        Args:
            name: Tool name as reported by :meth:`list_tools`.
            arguments: Keyword arguments for the tool.

        Returns:
            The tool result extracted from the MCP ``content`` array.

        Raises:
            MCPConnectionError: If not connected or the request times out.
            MCPCallError: If the server returns an error *or* the capability
                check fails.
        """
        self._require_connected()

        if arguments is None:
            arguments = {}

        # Capability gate
        self._require_tool_capability(name)

        params: Dict[str, Any] = {"name": name, "arguments": arguments}
        result = self._rpc_call("tools/call", params)

        # Unwrap MCP content array into plain Python value
        return self._unwrap_content(result)

    # ------------------------------------------------------------------
    # Internal: JSON-RPC transport
    # ------------------------------------------------------------------

    def _next_request_id(self) -> int:
        with self._id_lock:
            rid = self._next_id
            self._next_id += 1
        return rid

    def _rpc_call(self, method: str, params: Any) -> Any:
        """Send a JSON-RPC request and block until the response arrives."""
        self._require_connected()

        rid = self._next_request_id()
        message = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": method,
            "params": params,
        }

        pending = _PendingRequest(request_id=rid)
        with self._pending_lock:
            if len(self._pending) >= self._config.max_pending_requests:
                raise MCPConnectionError("Too many pending requests")
            self._pending[rid] = pending

        raw = json.dumps(message) + "\n"
        try:
            self._process.stdin.write(raw)  # type: ignore[union-attr]
            self._process.stdin.flush()     # type: ignore[union-attr]
        except OSError as exc:
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise MCPConnectionError(f"Failed to send request: {exc}") from exc

        # Wait for the response
        signalled = pending.event.wait(timeout=self._config.timeout)
        with self._pending_lock:
            self._pending.pop(rid, None)

        if not signalled:
            raise MCPConnectionError(
                f"Timeout waiting for response to {method!r} "
                f"(>{self._config.timeout}s)"
            )

        response = pending.response
        if response is None:
            raise MCPConnectionError("Reader thread produced no response")

        if "error" in response:
            err = response["error"]
            raise MCPCallError(
                err.get("message", "Unknown error"),
                code=err.get("code", JSONRPC_INTERNAL_ERROR),
                data=err.get("data"),
            )

        return response.get("result", {})

    def _rpc_notify(self, method: str, params: Any) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        raw = json.dumps(message) + "\n"
        try:
            self._process.stdin.write(raw)  # type: ignore[union-attr]
            self._process.stdin.flush()     # type: ignore[union-attr]
        except OSError:
            pass

    def _reader_loop(self) -> None:
        """Background thread: read lines from the subprocess stdout."""
        assert self._process is not None
        try:
            for line in self._process.stdout:  # type: ignore[union-attr]
                if self._stopping:
                    break
                line = line.strip()
                if not line:
                    continue
                self._handle_incoming(line)
        except Exception as exc:  # noqa: BLE001
            if not self._stopping:
                self._log("Reader loop error: %s", exc)
        finally:
            # Wake up any threads waiting on pending requests
            with self._pending_lock:
                for pending in self._pending.values():
                    pending.event.set()

    def _handle_incoming(self, raw: str) -> None:
        """Parse and dispatch an incoming JSON-RPC message from the server."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._log("Received invalid JSON: %s | error: %s", raw[:200], exc)
            return

        if not isinstance(msg, dict):
            return

        msg_id = msg.get("id")
        if msg_id is None:
            # Notification from server — currently ignore
            return

        with self._pending_lock:
            pending = self._pending.get(msg_id)

        if pending is not None:
            pending.response = msg
            pending.event.set()
        else:
            self._log("Received response for unknown id=%s", msg_id)

    # ------------------------------------------------------------------
    # Internal: MCP handshake
    # ------------------------------------------------------------------

    def _do_initialize(self) -> None:
        """Perform the MCP initialize handshake."""
        rid = self._next_request_id()
        message = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "clientInfo": {
                    "name": self._config.client_name,
                    "version": self._config.client_version,
                },
                "capabilities": {
                    "tools": {},
                },
            },
        }

        pending = _PendingRequest(request_id=rid)
        with self._pending_lock:
            self._pending[rid] = pending

        raw = json.dumps(message) + "\n"
        self._process.stdin.write(raw)   # type: ignore[union-attr]
        self._process.stdin.flush()      # type: ignore[union-attr]

        signalled = pending.event.wait(timeout=self._config.timeout)
        with self._pending_lock:
            self._pending.pop(rid, None)

        if not signalled:
            raise MCPConnectionError(
                f"Timeout during initialize (>{self._config.timeout}s)"
            )

        response = pending.response
        if response is None:
            raise MCPConnectionError("No response to initialize")
        if "error" in response:
            err = response["error"]
            raise MCPConnectionError(
                f"initialize failed: {err.get('message', 'unknown')}"
            )

        result = response.get("result", {})
        self._server_info = result.get("serverInfo", {})
        self._server_capabilities = result.get("capabilities", {})

        # Send the required ``notifications/initialized`` notification
        self._rpc_notify("notifications/initialized", {})

    # ------------------------------------------------------------------
    # Capability check
    # ------------------------------------------------------------------

    def _require_tool_capability(self, tool_name: str) -> None:
        """Raise :class:`MCPCallError` if the agent lacks TOOL_USE for *tool_name*."""
        if not self._config.require_tool_use_capability:
            return
        if self._cap_manager is None:
            return  # permissive when no manager configured
        agent_id = self._config.agent_id
        if agent_id == "kernel":
            return
        from battousai.capabilities import CapabilityType
        allowed = self._cap_manager.check(
            agent_id=agent_id,
            cap_type=CapabilityType.TOOL_USE,
            resource=tool_name,
        )
        if not allowed:
            raise MCPCallError(
                f"Agent {agent_id!r} lacks TOOL_USE capability for {tool_name!r}",
                code=MCP_CAPABILITY_DENIED,
                data={"tool": tool_name, "agent": agent_id},
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected:
            raise MCPConnectionError("Not connected — call connect() first")

    def _cleanup(self) -> None:
        """Stop the reader thread and terminate the subprocess."""
        self._stopping = True
        if self._process is not None:
            try:
                self._process.stdin.close()  # type: ignore[union-attr]
            except OSError:
                pass
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)
            self._reader_thread = None

    def _log(self, fmt: str, *args: Any) -> None:
        msg = fmt % args if args else fmt
        try:
            self._stderr.write(f"[battousai-mcp-client] {msg}\n")
            self._stderr.flush()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _unwrap_content(result: Dict[str, Any]) -> Any:
        """Extract the primary value from an MCP ``content`` response."""
        content = result.get("content", [])
        if not content:
            return result
        # If there's exactly one text item, return its string value
        if len(content) == 1 and content[0].get("type") == "text":
            text = content[0].get("text", "")
            # Try to parse as JSON for rich results
            try:
                return json.loads(text)
            except (json.JSONDecodeError, ValueError):
                return text
        # Multiple items: return as a list
        return content
