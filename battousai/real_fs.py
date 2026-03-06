"""
real_fs.py — Real OS Filesystem Provider
=========================================
Wraps actual OS file I/O with Battousai's capability-based permission model.
Each agent gets a jailed root directory. All paths are resolved relative to
the agent's jail — no path traversal escapes.

Security Model
--------------
- Each agent's files live under ``<root_dir>/agents/<agent_id>/``.
- A shared area is available at ``<root_dir>/shared/``.
- All user-supplied paths are treated as relative to the agent's jail root.
- Symlinks are resolved with ``os.path.realpath()`` before any jail check.
- Absolute path components and ``..`` sequences are stripped before joining.
- If the resolved real path is not under the jail, ``PermissionError`` is raised.

API Compatibility
-----------------
SandboxedFilesystem mirrors the VirtualFilesystem public API so it can be
used as a drop-in replacement:
    write_file, read_file, delete_file, list_dir, exists, mkdir,
    get_metadata, stats, tree
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Re-export the FileMetadata dataclass from filesystem so consumers can use
# a single import.  We extend it very slightly for real-FS metadata.
# ---------------------------------------------------------------------------
from battousai.filesystem import FileMetadata


# ---------------------------------------------------------------------------
# Filesystem errors (mirror VirtualFilesystem)
# ---------------------------------------------------------------------------

class FSError(Exception):
    """Base error for SandboxedFilesystem operations."""

class FileNotFoundError(FSError):
    """Path does not exist inside the jail."""

class PermissionError(FSError):
    """Agent lacks permission, or attempted path traversal."""

class NotADirectoryError(FSError):
    """Expected a directory but found a file."""

class FileExistsError(FSError):
    """File already exists and overwrite was not requested."""

class PathTraversalError(PermissionError):
    """Resolved path escapes the agent's jail directory."""


# ---------------------------------------------------------------------------
# SandboxedFilesystem
# ---------------------------------------------------------------------------

class SandboxedFilesystem:
    """
    Real-OS filesystem provider with per-agent directory jails.

    Directory Layout
    ----------------
    ::

        <root_dir>/
        ├── agents/
        │   ├── <agent_id_1>/
        │   │   └── ...              ← agent-private files
        │   └── <agent_id_2>/
        │       └── ...
        └── shared/
            └── ...                  ← cross-agent shared files

    Path Resolution
    ---------------
    All caller-supplied paths are interpreted as **relative** to the agent's
    jail directory (``<root_dir>/agents/<agent_id>/``).  Leading slashes and
    ``..`` components are stripped before joining to the jail root, then
    ``os.path.realpath()`` is used to resolve any remaining symlinks.  If the
    result is not a sub-path of the jail, ``PathTraversalError`` is raised.

    The agent_id ``"shared"`` (or paths prefixed with ``"shared/"`` when using
    the shared-path helper) maps to the shared directory.  The agent_id
    ``"kernel"`` bypasses permission checks (same convention as the virtual FS).

    Parameters
    ----------
    root_dir : str
        Absolute path to the root directory on the real OS.  Created if it
        does not exist.
    """

    # Tick counter — callers can call ``_set_tick()`` to advance it, or it
    # defaults to 0 (matching VirtualFilesystem behaviour when not driven by
    # the kernel scheduler).
    _current_tick: int = 0

    def __init__(self, root_dir: str) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self._agents_dir = os.path.join(self.root_dir, "agents")
        self._shared_dir = os.path.join(self.root_dir, "shared")

        # Ensure the top-level structure exists.
        os.makedirs(self._agents_dir, exist_ok=True)
        os.makedirs(self._shared_dir, exist_ok=True)

        # Lightweight in-memory metadata cache (key = real abs path).
        self._metadata: Dict[str, FileMetadata] = {}
        self._total_files: int = 0
        self._total_size: int = 0
        self._files_by_agent: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Tick management (mirrors VirtualFilesystem._set_tick)
    # ------------------------------------------------------------------

    def _set_tick(self, tick: int) -> None:
        self._current_tick = tick

    # ------------------------------------------------------------------
    # Internal: jail management
    # ------------------------------------------------------------------

    def _agent_jail(self, agent_id: str) -> str:
        """Return (creating if necessary) the jail directory for ``agent_id``."""
        if agent_id == "shared":
            return self._shared_dir
        jail = os.path.join(self._agents_dir, agent_id)
        os.makedirs(jail, exist_ok=True)
        return jail

    def _resolve_path(self, agent_id: str, user_path: str) -> str:
        """
        Convert a user-supplied path to an absolute real OS path inside the
        agent's jail.  Raises ``PathTraversalError`` if the result escapes.

        Algorithm
        ---------
        1. Strip any leading ``/`` and collapse ``..`` components before join.
        2. Join to the jail directory.
        3. Resolve symlinks with ``os.path.realpath()``.
        4. Assert the result starts with the real jail path.
        """
        jail = self._agent_jail(agent_id)
        # Ensure the jail itself is resolved (in case root_dir contains symlinks)
        real_jail = os.path.realpath(jail)

        # Sanitise the user-supplied path:
        # - Strip leading slashes (prevent absolute-path injection).
        # - Normalise separators.
        sanitised = os.path.normpath(user_path.lstrip("/\\"))

        # Build the candidate path inside the jail.
        candidate = os.path.join(real_jail, sanitised)

        # Resolve to strip any symlink hops that might escape the jail.
        real_candidate = os.path.realpath(candidate)

        # Security check: real_candidate must be the jail itself OR inside it.
        if real_candidate != real_jail and not real_candidate.startswith(
            real_jail + os.sep
        ):
            raise PathTraversalError(
                f"Path {user_path!r} resolves to {real_candidate!r} which is "
                f"outside the jail {real_jail!r} for agent {agent_id!r}."
            )

        return real_candidate

    # ------------------------------------------------------------------
    # Internal: metadata helpers
    # ------------------------------------------------------------------

    def _load_metadata(self, real_path: str, agent_id: str) -> Optional[FileMetadata]:
        """Return cached metadata for a real path, or None."""
        return self._metadata.get(real_path)

    def _store_metadata(self, real_path: str, meta: FileMetadata) -> None:
        self._metadata[real_path] = meta

    def _drop_metadata(self, real_path: str) -> Optional[FileMetadata]:
        return self._metadata.pop(real_path, None)

    def _make_metadata(
        self,
        agent_id: str,
        size: int,
        world_readable: bool = True,
        world_writable: bool = False,
    ) -> FileMetadata:
        return FileMetadata(
            created_by=agent_id,
            created_at=self._current_tick,
            modified_at=self._current_tick,
            size=size,
            owner_read=True,
            owner_write=True,
            world_read=world_readable,
            world_write=world_writable,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_file(
        self,
        agent_id: str,
        path: str,
        data: Any,
        create_parents: bool = True,
        overwrite: bool = True,
        world_readable: bool = True,
        world_writable: bool = False,
    ) -> str:
        """
        Write ``data`` to a file at ``path`` inside the agent's jail.

        Parameters
        ----------
        agent_id       : str  — writing agent's ID (``"kernel"`` skips perm check)
        path           : str  — path relative to the agent's jail root
        data           : Any  — data to write (converted to str if not bytes)
        create_parents : bool — auto-create parent directories
        overwrite      : bool — allow replacing an existing file
        world_readable : bool — other agents can read the file
        world_writable : bool — other agents can write/overwrite the file

        Returns
        -------
        str
            Absolute real path of the written file.

        Raises
        ------
        PathTraversalError  — path escapes the jail.
        FileExistsError     — file exists and ``overwrite=False``.
        PermissionError     — agent cannot overwrite an existing file.
        FileNotFoundError   — parent dir missing and ``create_parents=False``.
        """
        real_path = self._resolve_path(agent_id, path)
        parent_dir = os.path.dirname(real_path)

        if not os.path.exists(parent_dir):
            if create_parents:
                os.makedirs(parent_dir, exist_ok=True)
            else:
                raise FileNotFoundError(
                    f"Parent directory for {path!r} does not exist inside agent jail."
                )

        # Encode data
        if isinstance(data, bytes):
            raw = data
            size = len(raw)
        else:
            text = str(data)
            raw = text.encode("utf-8")
            size = len(raw)

        if os.path.exists(real_path):
            if not overwrite:
                raise FileExistsError(f"File {path!r} already exists in jail.")
            existing_meta = self._load_metadata(real_path, agent_id)
            if existing_meta is not None and agent_id != "kernel":
                if not existing_meta.can_write(agent_id):
                    raise PermissionError(
                        f"Agent {agent_id!r} cannot overwrite {path!r}."
                    )
            # Update size tracking
            old_size = existing_meta.size if existing_meta else 0
            self._total_size += size - old_size
            with open(real_path, "wb") as fh:
                fh.write(raw)
            meta = self._load_metadata(real_path, agent_id) or self._make_metadata(
                agent_id, size, world_readable, world_writable
            )
            meta.modified_at = self._current_tick
            meta.size = size
            self._store_metadata(real_path, meta)
        else:
            with open(real_path, "wb") as fh:
                fh.write(raw)
            meta = self._make_metadata(agent_id, size, world_readable, world_writable)
            self._store_metadata(real_path, meta)
            self._total_files += 1
            self._total_size += size
            self._files_by_agent[agent_id] = self._files_by_agent.get(agent_id, 0) + 1

        return real_path

    def read_file(self, agent_id: str, path: str) -> str:
        """
        Read the contents of a file and return as a UTF-8 string.

        Raises
        ------
        FileNotFoundError  — file does not exist.
        PermissionError    — agent cannot read the file.
        PathTraversalError — path escapes the jail.
        """
        real_path = self._resolve_path(agent_id, path)

        if not os.path.exists(real_path):
            raise FileNotFoundError(f"File not found: {path!r} in agent {agent_id!r} jail.")
        if os.path.isdir(real_path):
            raise NotADirectoryError(f"{path!r} is a directory, not a file.")

        meta = self._load_metadata(real_path, agent_id)
        if meta is not None and agent_id != "kernel":
            if not meta.can_read(agent_id):
                raise PermissionError(
                    f"Agent {agent_id!r} cannot read {path!r}."
                )

        with open(real_path, "rb") as fh:
            return fh.read().decode("utf-8", errors="replace")

    def delete_file(self, agent_id: str, path: str) -> bool:
        """
        Delete a file.  Returns ``True`` if deleted, ``False`` if not found.

        Raises
        ------
        PermissionError    — agent cannot delete the file.
        PathTraversalError — path escapes the jail.
        """
        real_path = self._resolve_path(agent_id, path)

        if not os.path.exists(real_path):
            return False
        if os.path.isdir(real_path):
            return False

        meta = self._load_metadata(real_path, agent_id)
        if meta is not None and agent_id != "kernel":
            if not meta.can_write(agent_id):
                raise PermissionError(
                    f"Agent {agent_id!r} cannot delete {path!r}."
                )

        old_size = meta.size if meta else 0
        os.remove(real_path)
        self._drop_metadata(real_path)
        self._total_files = max(0, self._total_files - 1)
        self._total_size = max(0, self._total_size - old_size)
        return True

    def list_dir(self, agent_id: str, path: str) -> List[str]:
        """
        List the entries in a directory inside the agent's jail.

        Raises
        ------
        FileNotFoundError  — directory does not exist.
        NotADirectoryError — path is a file.
        PathTraversalError — path escapes the jail.
        """
        real_path = self._resolve_path(agent_id, path)

        if not os.path.exists(real_path):
            raise FileNotFoundError(
                f"Directory not found: {path!r} in agent {agent_id!r} jail."
            )
        if not os.path.isdir(real_path):
            raise NotADirectoryError(f"{path!r} is a file, not a directory.")

        return sorted(os.listdir(real_path))

    def exists(self, agent_id: str, path: str) -> bool:
        """Return ``True`` if the path exists inside the agent's jail."""
        try:
            real_path = self._resolve_path(agent_id, path)
            return os.path.exists(real_path)
        except PathTraversalError:
            return False

    def mkdir(self, agent_id: str, path: str) -> str:
        """
        Create a directory (and all parents) inside the agent's jail.

        Returns
        -------
        str
            Absolute real path of the created directory.
        """
        real_path = self._resolve_path(agent_id, path)
        os.makedirs(real_path, exist_ok=True)
        return real_path

    def get_metadata(self, agent_id: str, path: str) -> Optional[FileMetadata]:
        """Return cached FileMetadata for a path, or ``None``."""
        try:
            real_path = self._resolve_path(agent_id, path)
            return self._load_metadata(real_path, agent_id)
        except (PathTraversalError, FileNotFoundError):
            return None

    def get_shared_path(self, path: str) -> str:
        """
        Resolve a path inside the shared directory (outside any agent jail).

        Useful for kernel-level shared access.
        """
        sanitised = os.path.normpath(path.lstrip("/\\"))
        real_shared = os.path.realpath(self._shared_dir)
        candidate = os.path.realpath(os.path.join(real_shared, sanitised))
        if candidate != real_shared and not candidate.startswith(
            real_shared + os.sep
        ):
            raise PathTraversalError(
                f"Shared path {path!r} resolves outside shared directory."
            )
        return candidate

    # ------------------------------------------------------------------
    # Statistics (mirrors VirtualFilesystem.stats)
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return filesystem usage statistics."""
        return {
            "root_dir": self.root_dir,
            "total_files": self._total_files,
            "total_size_bytes": self._total_size,
            "files_by_agent": dict(self._files_by_agent),
        }

    def tree(self, agent_id: str = "kernel", path: str = "") -> str:
        """
        Return a textual directory tree for the agent's jail (or root if kernel).

        Parameters
        ----------
        agent_id : str  — agent whose jail to inspect (``"kernel"`` → entire root)
        path     : str  — sub-path within the jail (default: jail root)
        """
        if agent_id == "kernel":
            base = self.root_dir
        else:
            try:
                base = self._resolve_path(agent_id, path or ".")
            except (PathTraversalError, FileNotFoundError):
                return "<not found>"

        lines: List[str] = []
        self._build_tree(base, "", lines)
        return "\n".join(lines)

    def _build_tree(self, real_path: str, prefix: str, lines: List[str]) -> None:
        name = os.path.basename(real_path) or real_path
        if os.path.isdir(real_path):
            lines.append(f"{prefix}{name}/")
            try:
                children = sorted(os.listdir(real_path))
            except PermissionError:
                children = []
            for child in children:
                self._build_tree(
                    os.path.join(real_path, child), prefix + "  ", lines
                )
        else:
            try:
                size = os.path.getsize(real_path)
            except OSError:
                size = 0
            lines.append(f"{prefix}{name} ({size}b)")

    # ------------------------------------------------------------------
    # Cleanup helper (useful in tests)
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """
        Remove the entire root directory tree.

        **WARNING**: This deletes all files permanently.  Only use in tests.
        """
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)
