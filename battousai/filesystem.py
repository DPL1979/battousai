"""
filesystem.py — Battousai Virtual Filesystem
=========================================
A hierarchical virtual filesystem for the Autonomous Intelligence Operating System.

Structure:
    /                          — Root
    ├── agents/
    │   └── {agent_id}/
    │       └── workspace/     — Private working area for each agent
    ├── shared/
    │   └── results/           — Cross-agent result publication
    └── system/
        └── logs/              — OS log files

Files store arbitrary data (str, bytes-like, or JSON-serialisable objects).
Directories are implicit — created automatically when a file path is written
if `create_parents=True` is set.

Permissions:
    Each file has an owner agent, and read/write flags for:
        owner  — the creating agent
        group  — explicitly listed agents
        world  — all agents

    The kernel always bypasses permission checks (agent_id="kernel").
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


class FSError(Exception):
    """Base filesystem error."""


class FileNotFoundError(FSError):
    """Path does not exist."""


class PermissionError(FSError):
    """Agent lacks permission to perform the operation."""


class NotADirectoryError(FSError):
    """Expected a directory but found a file."""


class FileExistsError(FSError):
    """File already exists and overwrite was not requested."""


@dataclass
class FileMetadata:
    """Metadata stored alongside every file node."""
    created_by: str
    created_at: int         # tick
    modified_at: int        # tick
    size: int               # approximate character/byte count
    owner_read: bool = True
    owner_write: bool = True
    world_read: bool = True
    world_write: bool = False
    group_agents: Set[str] = field(default_factory=set)
    wall_created: float = field(default_factory=time.time)

    def can_read(self, agent_id: str) -> bool:
        if agent_id == "kernel":
            return True
        if agent_id == self.created_by:
            return self.owner_read
        if agent_id in self.group_agents:
            return True
        return self.world_read

    def can_write(self, agent_id: str) -> bool:
        if agent_id == "kernel":
            return True
        if agent_id == self.created_by:
            return self.owner_write
        if agent_id in self.group_agents:
            return True
        return self.world_write


class FSNode:
    """Base class for filesystem nodes (files and directories)."""
    def __init__(self, name: str, parent: Optional["FSDirectory"] = None) -> None:
        self.name = name
        self.parent = parent

    @property
    def is_file(self) -> bool:
        return isinstance(self, FSFile)

    @property
    def is_directory(self) -> bool:
        return isinstance(self, FSDirectory)


class FSFile(FSNode):
    """A file node storing data and metadata."""
    def __init__(
        self,
        name: str,
        data: Any,
        metadata: FileMetadata,
        parent: Optional["FSDirectory"] = None,
    ) -> None:
        super().__init__(name, parent)
        self.data = data
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"FSFile({self.name!r}, size={self.metadata.size})"


class FSDirectory(FSNode):
    """A directory node containing child nodes."""
    def __init__(self, name: str, parent: Optional["FSDirectory"] = None) -> None:
        super().__init__(name, parent)
        self._children: Dict[str, FSNode] = {}

    def add(self, node: FSNode) -> None:
        self._children[node.name] = node
        node.parent = self

    def remove(self, name: str) -> Optional[FSNode]:
        return self._children.pop(name, None)

    def get(self, name: str) -> Optional[FSNode]:
        return self._children.get(name)

    def children(self) -> Dict[str, FSNode]:
        return dict(self._children)

    def list(self) -> List[str]:
        return sorted(self._children.keys())

    def __repr__(self) -> str:
        return f"FSDirectory({self.name!r}, children={self.list()})"


def _split_path(path: str) -> List[str]:
    """Split a Unix-style path into components, ignoring empty parts."""
    return [p for p in path.strip("/").split("/") if p]


class VirtualFilesystem:
    """
    Virtual in-memory hierarchical filesystem for Battousai.

    Provides:
    - Path-based file creation, reading, writing, deletion
    - Directory listing and existence checks
    - Automatic parent directory creation
    - Permission enforcement (bypassed for agent_id="kernel")
    - File metadata: owner, timestamps, size
    - Statistics: total files, total size, files per agent
    """

    def __init__(self) -> None:
        self._root = FSDirectory("/")
        self._total_files: int = 0
        self._total_size: int = 0
        self._current_tick: int = 0
        self._files_by_agent: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_tick(self, tick: int) -> None:
        self._current_tick = tick

    def _resolve(self, path: str) -> Optional[FSNode]:
        """Walk the tree and return the node at path, or None."""
        parts = _split_path(path)
        node: FSNode = self._root
        for part in parts:
            if not isinstance(node, FSDirectory):
                return None
            node = node.get(part)
            if node is None:
                return None
        return node

    def _resolve_parent(self, path: str) -> tuple[Optional[FSDirectory], str]:
        """Return (parent_directory, filename) for a given path."""
        parts = _split_path(path)
        if not parts:
            return None, ""
        filename = parts[-1]
        if len(parts) == 1:
            return self._root, filename
        parent_path = "/" + "/".join(parts[:-1])
        parent_node = self._resolve(parent_path)
        if parent_node is None or not parent_node.is_directory:
            return None, filename
        return parent_node, filename  # type: ignore[return-value]

    def _make_dirs(self, path: str) -> FSDirectory:
        """Ensure all directories along path exist, creating them as needed."""
        parts = _split_path(path)
        current: FSDirectory = self._root
        for part in parts:
            child = current.get(part)
            if child is None:
                new_dir = FSDirectory(part, current)
                current.add(new_dir)
                current = new_dir
            elif isinstance(child, FSDirectory):
                current = child
            else:
                raise NotADirectoryError(f"Path component {part!r} is a file, not a directory.")
        return current

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
    ) -> FSFile:
        """
        Write data to a file at path.

        Creates parent directories if create_parents=True.
        Raises FileExistsError if the file exists and overwrite=False.
        Raises PermissionError if the agent cannot write to an existing file.
        """
        parent, filename = self._resolve_parent(path)

        if parent is None:
            if create_parents:
                parts = _split_path(path)
                dir_path = "/" + "/".join(parts[:-1])
                parent = self._make_dirs(dir_path)
                filename = parts[-1]
            else:
                raise FileNotFoundError(f"Parent directory for {path!r} does not exist.")

        # Check for existing file
        existing = parent.get(filename)
        if existing is not None:
            if not overwrite:
                raise FileExistsError(f"File {path!r} already exists.")
            if not isinstance(existing, FSFile):
                raise NotADirectoryError(f"{path!r} is a directory.")
            if not existing.metadata.can_write(agent_id):
                raise PermissionError(f"Agent {agent_id!r} cannot write to {path!r}.")
            # Update existing
            old_size = existing.metadata.size
            new_size = len(str(data))
            existing.data = data
            existing.metadata.modified_at = self._current_tick
            existing.metadata.size = new_size
            self._total_size += new_size - old_size
            return existing

        # Create new file
        size = len(str(data))
        meta = FileMetadata(
            created_by=agent_id,
            created_at=self._current_tick,
            modified_at=self._current_tick,
            size=size,
            world_read=world_readable,
            world_write=world_writable,
        )
        file_node = FSFile(filename, data, meta, parent)
        parent.add(file_node)
        self._total_files += 1
        self._total_size += size
        self._files_by_agent[agent_id] = self._files_by_agent.get(agent_id, 0) + 1
        return file_node

    def read_file(self, agent_id: str, path: str) -> Any:
        """
        Read and return the contents of a file.
        Raises FileNotFoundError if path does not exist.
        Raises PermissionError if agent cannot read the file.
        """
        node = self._resolve(path)
        if node is None:
            raise FileNotFoundError(f"File not found: {path!r}")
        if not isinstance(node, FSFile):
            raise NotADirectoryError(f"{path!r} is a directory, not a file.")
        if not node.metadata.can_read(agent_id):
            raise PermissionError(f"Agent {agent_id!r} cannot read {path!r}.")
        return node.data

    def delete_file(self, agent_id: str, path: str) -> bool:
        """Delete a file. Returns True if deleted, False if not found."""
        parent, filename = self._resolve_parent(path)
        if parent is None:
            return False
        node = parent.get(filename)
        if node is None:
            return False
        if isinstance(node, FSFile):
            if not node.metadata.can_write(agent_id):
                raise PermissionError(f"Agent {agent_id!r} cannot delete {path!r}.")
            self._total_files -= 1
            self._total_size -= node.metadata.size
        parent.remove(filename)
        return True

    def list_dir(self, agent_id: str, path: str) -> List[str]:
        """Return the names of entries in a directory."""
        node = self._resolve(path)
        if node is None:
            raise FileNotFoundError(f"Directory not found: {path!r}")
        if not isinstance(node, FSDirectory):
            raise NotADirectoryError(f"{path!r} is a file, not a directory.")
        return node.list()

    def exists(self, path: str) -> bool:
        return self._resolve(path) is not None

    def get_metadata(self, path: str) -> Optional[FileMetadata]:
        node = self._resolve(path)
        if isinstance(node, FSFile):
            return node.metadata
        return None

    def mkdir(self, path: str) -> FSDirectory:
        """Create a directory (and parents) at path."""
        return self._make_dirs(path)

    # ------------------------------------------------------------------
    # Tree rendering (for debug/reports)
    # ------------------------------------------------------------------

    def tree(self, path: str = "/", indent: int = 0) -> str:
        node = self._resolve(path)
        if node is None:
            return f"<not found: {path}>"
        lines = []
        prefix = "  " * indent
        if isinstance(node, FSDirectory):
            lines.append(f"{prefix}{node.name}/")
            for child_name in sorted(node._children.keys()):
                child_path = path.rstrip("/") + "/" + child_name
                lines.append(self.tree(child_path, indent + 1))
        else:
            assert isinstance(node, FSFile)
            lines.append(f"{prefix}{node.name} ({node.metadata.size}b)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "total_files": self._total_files,
            "total_size_bytes": self._total_size,
            "files_by_agent": dict(self._files_by_agent),
        }

    def _init_standard_dirs(self) -> None:
        """Bootstrap the standard Battousai directory tree."""
        for path in ["/agents", "/shared", "/shared/results", "/system", "/system/logs"]:
            self._make_dirs(path)
