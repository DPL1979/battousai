"""
isolation.py — Process-based Agent Isolation
=============================================
Runs each agent in a separate subprocess with restricted permissions.
Communication happens over multiprocessing Pipes.

Architecture
------------
The parent process (kernel side) creates an ``IsolatedAgentProcess`` which
wraps an ``Agent`` subclass in a ``multiprocessing.Process``.  The two sides
communicate via a pair of ``multiprocessing.Pipe`` connections:

    Kernel ──[parent_conn]──⟩ IPC ⟨──[child_conn]── Agent subprocess

Protocol
--------
Requests (kernel → agent subprocess):
    {"type": "syscall", "name": <str>, "kwargs": <dict>}
    {"type": "tick",    "tick": <int>}
    {"type": "shutdown"}

Responses (agent subprocess → kernel):
    {"type": "syscall_result", "ok": <bool>, "value": <any>, "error": <str|None>}
    {"type": "tick_done"}
    {"type": "error", "message": <str>}

Resource Limits (Linux only)
------------------------------
``SandboxConfig`` specifies optional resource limits applied inside the child
process via the ``resource`` stdlib module.  On non-Linux platforms the limits
are silently ignored so the module remains portable.

Security Model
--------------
- The agent subprocess is isolated in its own address space.
- Syscall requests from the agent are forwarded through the pipe to the
  kernel, which validates capabilities and executes the real I/O.
- Agent crashes (exceptions, OOM) are caught and reported via the pipe
  without affecting the kernel process.
"""

from __future__ import annotations

import logging
import multiprocessing
import multiprocessing.connection
import os
import signal
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SandboxConfig
# ---------------------------------------------------------------------------

@dataclass
class SandboxConfig:
    """
    Resource constraints and permissions for an isolated agent process.

    Attributes
    ----------
    max_memory_mb  : int  — Maximum virtual memory in megabytes (0 = unlimited).
    max_cpu_seconds: int  — Maximum CPU time in seconds (0 = unlimited).
    max_open_files : int  — Maximum open file descriptors (0 = OS default).
    allowed_paths  : list — Filesystem paths the agent is allowed to access.
                            Passed as metadata; enforcement is done by
                            SandboxedFilesystem in the kernel.
    network_access : bool — Whether the agent is permitted network calls.
                            Currently informational; actual enforcement
                            requires OS-level sandboxing (e.g. seccomp).
    """

    max_memory_mb: int = 0
    max_cpu_seconds: int = 0
    max_open_files: int = 0
    allowed_paths: List[str] = field(default_factory=list)
    network_access: bool = False


# ---------------------------------------------------------------------------
# Internal protocol helpers
# ---------------------------------------------------------------------------

def _send(conn: multiprocessing.connection.Connection, msg: Dict[str, Any]) -> None:
    """Send a dict over a multiprocessing Pipe connection."""
    conn.send(msg)


def _recv(
    conn: multiprocessing.connection.Connection,
    timeout: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Receive a dict from a multiprocessing Pipe, with optional timeout."""
    if timeout is not None:
        if not conn.poll(timeout):
            return None
    return conn.recv()


# ---------------------------------------------------------------------------
# Agent subprocess entry point
# ---------------------------------------------------------------------------

def _apply_resource_limits(config: SandboxConfig) -> None:
    """
    Apply resource limits inside the child process (Linux only).

    Gracefully skips if the ``resource`` module is unavailable (e.g. Windows).
    """
    try:
        import resource as _resource  # type: ignore
    except ImportError:
        return

    try:
        if config.max_memory_mb > 0:
            limit_bytes = config.max_memory_mb * 1024 * 1024
            _resource.setrlimit(
                _resource.RLIMIT_AS, (limit_bytes, limit_bytes)
            )

        if config.max_cpu_seconds > 0:
            _resource.setrlimit(
                _resource.RLIMIT_CPU,
                (config.max_cpu_seconds, config.max_cpu_seconds),
            )

        if config.max_open_files > 0:
            _resource.setrlimit(
                _resource.RLIMIT_NOFILE,
                (config.max_open_files, config.max_open_files),
            )
    except (ValueError, _resource.error) as exc:
        logger.warning("Could not apply resource limit: %s", exc)


def _agent_subprocess_main(
    child_conn: multiprocessing.connection.Connection,
    agent_class: Type,
    agent_kwargs: Dict[str, Any],
    config: SandboxConfig,
) -> None:
    """
    Entry point for the isolated agent subprocess.

    Protocol loop:
        1. Receive a message from the parent.
        2. Process it (run a tick, execute a syscall stub, or shutdown).
        3. Send a response.

    Syscall stubs
    -------------
    Inside the subprocess, ``Agent.syscall()`` is monkey-patched to forward
    the request to the parent (kernel) via the pipe and block until a result
    is received.  This preserves the kernel's authority over all I/O.
    """
    _apply_resource_limits(config)

    # Instantiate the agent inside the subprocess
    try:
        agent = agent_class(**agent_kwargs)
    except Exception as exc:
        _send(child_conn, {"type": "error", "message": str(exc)})
        child_conn.close()
        return

    # Monkey-patch the syscall method so calls go through the pipe
    def _remote_syscall(name: str, **kwargs: Any) -> Any:
        _send(child_conn, {"type": "syscall", "name": name, "kwargs": kwargs})
        result = _recv(child_conn, timeout=30.0)
        if result is None:
            # Timeout — return a failed SyscallResult-like object
            class _Timeout:
                ok = False
                value = None
                error = "syscall timeout in isolated process"
            return _Timeout()
        # Reconstruct a lightweight result wrapper
        class _Result:
            ok = result.get("ok", False)
            value = result.get("value")
            error = result.get("error")
        return _Result()

    # Only patch if the agent has a syscall method
    if hasattr(agent, "syscall"):
        agent.syscall = _remote_syscall  # type: ignore[method-assign]

    # Signal readiness
    _send(child_conn, {"type": "ready"})

    # Main message loop
    try:
        while True:
            msg = _recv(child_conn, timeout=None)  # block indefinitely
            if msg is None:
                break

            msg_type = msg.get("type")

            if msg_type == "tick":
                tick = msg.get("tick", 0)
                try:
                    if hasattr(agent, "think"):
                        agent.think(tick)  # type: ignore[arg-type]
                    _send(child_conn, {"type": "tick_done"})
                except Exception as exc:
                    _send(child_conn, {
                        "type": "error",
                        "message": f"tick {tick} raised: {exc}",
                        "traceback": traceback.format_exc(),
                    })

            elif msg_type == "syscall_result":
                # These are replies to our own outbound syscall requests —
                # they should never arrive on this branch; they are consumed
                # inside _remote_syscall above.
                pass

            elif msg_type == "shutdown":
                if hasattr(agent, "on_terminate"):
                    try:
                        agent.on_terminate()
                    except Exception:
                        pass
                _send(child_conn, {"type": "shutdown_ack"})
                break

            else:
                _send(child_conn, {
                    "type": "error",
                    "message": f"unknown message type: {msg_type!r}",
                })

    except (EOFError, BrokenPipeError):
        pass  # Parent closed the pipe — clean exit
    except Exception as exc:
        try:
            _send(child_conn, {
                "type": "error",
                "message": f"subprocess fatal: {exc}",
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
    finally:
        child_conn.close()


# ---------------------------------------------------------------------------
# IsolatedAgentProcess
# ---------------------------------------------------------------------------

class IsolatedAgentProcess:
    """
    Wraps an Agent subclass in an isolated subprocess.

    The subprocess communicates with the caller (the kernel) via a
    ``multiprocessing.Pipe``.  The kernel can:

    - Send ticks to drive the agent's ``think()`` loop.
    - Receive and service syscall requests from the agent.
    - Receive crash reports without the crash propagating to the kernel.

    Parameters
    ----------
    agent_class  : Type
        The Agent subclass to instantiate in the subprocess.
    agent_kwargs : dict
        Keyword arguments forwarded to the agent constructor.
    config       : SandboxConfig, optional
        Resource limits and security config for the subprocess.
    start_timeout: float
        Seconds to wait for the subprocess to signal readiness (default 10).

    Example
    -------
    ::

        from battousai.isolation import IsolatedAgentProcess, SandboxConfig
        from battousai.agent import WorkerAgent

        config = SandboxConfig(max_memory_mb=256, max_cpu_seconds=30)
        iso = IsolatedAgentProcess(
            agent_class=WorkerAgent,
            agent_kwargs={"name": "Worker1"},
            config=config,
        )
        iso.start()

        # Drive one tick
        result = iso.tick(tick=1)

        # Clean shutdown
        iso.stop()
    """

    def __init__(
        self,
        agent_class: Type,
        agent_kwargs: Optional[Dict[str, Any]] = None,
        config: Optional[SandboxConfig] = None,
        start_timeout: float = 10.0,
    ) -> None:
        self.agent_class = agent_class
        self.agent_kwargs = agent_kwargs or {}
        self.config = config or SandboxConfig()
        self.start_timeout = start_timeout

        self._process: Optional[multiprocessing.Process] = None
        self._parent_conn: Optional[multiprocessing.connection.Connection] = None
        self._child_conn: Optional[multiprocessing.connection.Connection] = None
        self._running: bool = False
        self._error: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the agent subprocess.

        Raises
        ------
        RuntimeError
            If the subprocess does not become ready within ``start_timeout``
            seconds, or if it immediately reports an error.
        """
        self._parent_conn, self._child_conn = multiprocessing.Pipe(duplex=True)

        self._process = multiprocessing.Process(
            target=_agent_subprocess_main,
            args=(
                self._child_conn,
                self.agent_class,
                self.agent_kwargs,
                self.config,
            ),
            daemon=True,
        )
        self._process.start()
        # Close the child end in the parent process
        self._child_conn.close()
        self._child_conn = None

        # Wait for "ready" signal
        ready_msg = _recv(self._parent_conn, timeout=self.start_timeout)
        if ready_msg is None:
            self._process.terminate()
            raise RuntimeError(
                f"IsolatedAgentProcess: subprocess did not become ready within "
                f"{self.start_timeout}s."
            )
        if ready_msg.get("type") == "error":
            self._process.terminate()
            raise RuntimeError(
                f"IsolatedAgentProcess: subprocess startup error: "
                f"{ready_msg.get('message', '?')}"
            )
        self._running = True
        logger.info(
            "IsolatedAgentProcess: %s started (pid=%d)",
            self.agent_class.__name__,
            self._process.pid,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """
        Gracefully shut down the agent subprocess.

        Sends a ``shutdown`` message, waits for acknowledgement, then
        terminates the process if it does not exit within ``timeout``.
        """
        if not self._running or self._process is None:
            return
        try:
            _send(self._parent_conn, {"type": "shutdown"})
            ack = _recv(self._parent_conn, timeout=timeout)
            if ack and ack.get("type") == "shutdown_ack":
                logger.debug("IsolatedAgentProcess: clean shutdown acknowledged.")
        except (BrokenPipeError, EOFError, OSError):
            pass

        self._process.join(timeout=timeout)
        if self._process.is_alive():
            logger.warning(
                "IsolatedAgentProcess: process %d did not exit — terminating.",
                self._process.pid,
            )
            self._process.terminate()
            self._process.join(2)

        if self._parent_conn:
            self._parent_conn.close()
        self._running = False
        logger.info("IsolatedAgentProcess: %s stopped.", self.agent_class.__name__)

    def is_alive(self) -> bool:
        """Return ``True`` if the subprocess is still running."""
        return (
            self._running
            and self._process is not None
            and self._process.is_alive()
        )

    # ------------------------------------------------------------------
    # Tick driving
    # ------------------------------------------------------------------

    def tick(self, tick: int = 0, timeout: float = 30.0) -> Dict[str, Any]:
        """
        Send a tick to the agent subprocess and wait for completion.

        Syscall requests emitted by the agent during the tick are serviced
        by the ``syscall_handler`` callback (if provided) — see
        ``tick_with_handler``.

        Parameters
        ----------
        tick    : int   — current kernel tick number
        timeout : float — seconds to wait for tick completion

        Returns
        -------
        dict
            ``{"ok": True}`` on success, or ``{"ok": False, "error": ...}``
            on agent crash / timeout.
        """
        return self.tick_with_handler(tick, syscall_handler=None, timeout=timeout)

    def tick_with_handler(
        self,
        tick: int = 0,
        syscall_handler: Optional[Any] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Send a tick to the subprocess, servicing any syscall requests.

        The agent's ``think()`` method may call ``syscall()`` which forwards
        requests through the pipe.  This method intercepts those requests,
        invokes ``syscall_handler(name, kwargs)`` on the kernel side, and
        returns the result back into the subprocess.

        Parameters
        ----------
        tick            : int
        syscall_handler : callable(name, **kwargs) → SyscallResult-like, optional
        timeout         : float — total seconds to wait for tick completion

        Returns
        -------
        dict with keys ``ok`` (bool) and optionally ``error`` (str).
        """
        if not self.is_alive():
            return {"ok": False, "error": "subprocess not running"}

        _send(self._parent_conn, {"type": "tick", "tick": tick})

        import time as _time
        deadline = _time.monotonic() + timeout

        while True:
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                return {"ok": False, "error": f"tick timeout after {timeout}s"}

            msg = _recv(self._parent_conn, timeout=min(remaining, 1.0))
            if msg is None:
                continue  # poll again

            msg_type = msg.get("type")

            if msg_type == "tick_done":
                return {"ok": True}

            if msg_type == "error":
                return {
                    "ok": False,
                    "error": msg.get("message", "unknown error"),
                    "traceback": msg.get("traceback"),
                }

            if msg_type == "syscall":
                # Service the syscall request
                if syscall_handler is not None:
                    try:
                        result = syscall_handler(
                            msg["name"], **msg.get("kwargs", {})
                        )
                        _send(self._parent_conn, {
                            "type": "syscall_result",
                            "ok": getattr(result, "ok", True),
                            "value": getattr(result, "value", None),
                            "error": getattr(result, "error", None),
                        })
                    except Exception as exc:
                        _send(self._parent_conn, {
                            "type": "syscall_result",
                            "ok": False,
                            "value": None,
                            "error": str(exc),
                        })
                else:
                    # No handler — deny the syscall
                    _send(self._parent_conn, {
                        "type": "syscall_result",
                        "ok": False,
                        "value": None,
                        "error": "no syscall handler configured",
                    })

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def pid(self) -> Optional[int]:
        """Return the OS PID of the subprocess, or ``None`` if not started."""
        return self._process.pid if self._process else None

    @property
    def exit_code(self) -> Optional[int]:
        """Return the exit code of the subprocess (``None`` if still running)."""
        if self._process:
            return self._process.exitcode
        return None

    def __repr__(self) -> str:
        state = "running" if self.is_alive() else "stopped"
        return (
            f"IsolatedAgentProcess({self.agent_class.__name__}, "
            f"pid={self.pid}, state={state})"
        )


# ---------------------------------------------------------------------------
# ProcessPool — manage a collection of isolated agents
# ---------------------------------------------------------------------------

class ProcessPool:
    """
    Manage a pool of ``IsolatedAgentProcess`` instances.

    Provides a simple interface for spawning, ticking, and stopping
    multiple isolated agents at once.

    Parameters
    ----------
    config : SandboxConfig, optional
        Default resource config applied to all spawned processes.
    """

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self.default_config = config or SandboxConfig()
        self._pool: Dict[str, IsolatedAgentProcess] = {}

    def spawn(
        self,
        name: str,
        agent_class: Type,
        agent_kwargs: Optional[Dict[str, Any]] = None,
        config: Optional[SandboxConfig] = None,
    ) -> IsolatedAgentProcess:
        """
        Spawn an isolated agent process and register it under ``name``.

        Parameters
        ----------
        name         : str  — unique logical name for the process
        agent_class  : Type — Agent subclass
        agent_kwargs : dict — constructor kwargs for the agent
        config       : SandboxConfig, optional — overrides the pool default

        Returns
        -------
        IsolatedAgentProcess
        """
        iso = IsolatedAgentProcess(
            agent_class=agent_class,
            agent_kwargs=agent_kwargs,
            config=config or self.default_config,
        )
        iso.start()
        self._pool[name] = iso
        return iso

    def tick_all(
        self,
        tick: int = 0,
        syscall_handler: Optional[Any] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Send a tick to all running processes.

        Returns
        -------
        dict mapping name → tick result dict.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for name, iso in list(self._pool.items()):
            if iso.is_alive():
                results[name] = iso.tick_with_handler(
                    tick, syscall_handler=syscall_handler, timeout=timeout
                )
            else:
                results[name] = {"ok": False, "error": "process not alive"}
        return results

    def stop_all(self) -> None:
        """Stop and remove all processes from the pool."""
        for name, iso in list(self._pool.items()):
            try:
                iso.stop()
            except Exception as exc:
                logger.warning("ProcessPool: error stopping %r: %s", name, exc)
        self._pool.clear()

    def get(self, name: str) -> Optional[IsolatedAgentProcess]:
        """Return a process by name, or ``None`` if not found."""
        return self._pool.get(name)

    def names(self) -> List[str]:
        """Return the names of all registered processes."""
        return list(self._pool.keys())

    def alive_count(self) -> int:
        """Return the number of currently alive processes."""
        return sum(1 for iso in self._pool.values() if iso.is_alive())

    def __len__(self) -> int:
        return len(self._pool)

    def __repr__(self) -> str:
        return (
            f"ProcessPool(total={len(self._pool)}, alive={self.alive_count()})"
        )
