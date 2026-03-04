"""
memory.py — Battousai Memory Manager
=================================
Manages memory spaces for agents in the Autonomous Intelligence Operating System.

Memory Architecture:
    PRIVATE  — Each agent owns an isolated key-value store. Other agents cannot
               access it directly; they must use the kernel syscall interface.
    SHARED   — Named shared regions that multiple agents can read and write.
               Useful for blackboards, result aggregation, etc.

Memory Types (within any space):
    SHORT_TERM — Automatically expires after a configurable TTL (in ticks).
                 Use for working memory, intermediate computation results.
    LONG_TERM  — Persists for the lifetime of the OS session.
    SHARED     — Stored in a named shared region accessible by authorized agents.

Memory Limits:
    Each agent has a configurable maximum number of keys. Writes that would
    exceed the limit raise MemoryFullError (agents must evict old entries).

Garbage Collection:
    The kernel calls `gc_tick(current_tick)` each tick to evict expired
    SHORT_TERM entries across all agent spaces and shared regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


class MemoryType(Enum):
    """Lifetime classification of a memory entry."""
    SHORT_TERM = auto()   # Expires after ttl_ticks
    LONG_TERM  = auto()   # Lives for the session
    SHARED     = auto()   # Lives in a named shared region


class MemoryError(Exception):
    """Base class for memory subsystem errors."""


class MemoryFullError(MemoryError):
    """Raised when an agent's memory allocation is exhausted."""


class MemoryAccessError(MemoryError):
    """Raised when an agent tries to access memory it does not own."""


class MemoryKeyError(MemoryError):
    """Raised when a key does not exist in the requested memory space."""


@dataclass
class MemoryEntry:
    """A single item in an agent's or shared memory space."""
    key: str
    value: Any
    memory_type: MemoryType
    created_tick: int
    ttl_ticks: Optional[int] = None    # Only used for SHORT_TERM
    owner_agent_id: str = ""

    @property
    def expires_at(self) -> Optional[int]:
        if self.memory_type == MemoryType.SHORT_TERM and self.ttl_ticks is not None:
            return self.created_tick + self.ttl_ticks
        return None

    def is_expired(self, current_tick: int) -> bool:
        exp = self.expires_at
        return exp is not None and current_tick >= exp

    def __repr__(self) -> str:
        exp = f", expires={self.expires_at}" if self.expires_at else ""
        return f"MemoryEntry(key={self.key!r}, type={self.memory_type.name}{exp})"


class AgentMemorySpace:
    """
    Isolated memory space for a single agent.

    Provides a key-value store with per-entry memory type tracking,
    TTL-based expiration for SHORT_TERM entries, and a hard cap on
    the total number of entries.
    """

    def __init__(self, agent_id: str, max_keys: int = 256) -> None:
        self.agent_id = agent_id
        self.max_keys = max_keys
        self._store: Dict[str, MemoryEntry] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        key: str,
        value: Any,
        memory_type: MemoryType = MemoryType.LONG_TERM,
        current_tick: int = 0,
        ttl_ticks: Optional[int] = None,
    ) -> MemoryEntry:
        """Store a value. Overwrites existing entry with the same key."""
        if key not in self._store and len(self._store) >= self.max_keys:
            raise MemoryFullError(
                f"Agent {self.agent_id!r} memory full ({self.max_keys} keys). "
                f"Evict entries before writing new ones."
            )
        entry = MemoryEntry(
            key=key,
            value=value,
            memory_type=memory_type,
            created_tick=current_tick,
            ttl_ticks=ttl_ticks,
            owner_agent_id=self.agent_id,
        )
        self._store[key] = entry
        return entry

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, key: str, current_tick: int = 0) -> Any:
        """Read a value by key. Raises MemoryKeyError if missing or expired."""
        entry = self._store.get(key)
        if entry is None:
            raise MemoryKeyError(f"Key {key!r} not found in agent {self.agent_id!r} memory.")
        if entry.is_expired(current_tick):
            del self._store[key]
            raise MemoryKeyError(f"Key {key!r} in agent {self.agent_id!r} memory has expired.")
        return entry.value

    def read_entry(self, key: str, current_tick: int = 0) -> MemoryEntry:
        """Return the full MemoryEntry (including metadata)."""
        self.read(key, current_tick)  # validates existence and expiry
        return self._store[key]

    def exists(self, key: str, current_tick: int = 0) -> bool:
        """Return True if key exists and has not expired."""
        try:
            self.read(key, current_tick)
            return True
        except MemoryKeyError:
            return False

    # ------------------------------------------------------------------
    # Delete / evict
    # ------------------------------------------------------------------

    def delete(self, key: str) -> bool:
        """Remove a key. Returns True if it existed."""
        return self._store.pop(key, None) is not None

    def gc(self, current_tick: int) -> List[str]:
        """Garbage-collect expired SHORT_TERM entries. Returns list of evicted keys."""
        expired = [k for k, v in self._store.items() if v.is_expired(current_tick)]
        for k in expired:
            del self._store[k]
        return expired

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def keys(self) -> List[str]:
        return list(self._store.keys())

    def usage(self) -> Tuple[int, int]:
        """Return (used, max) key counts."""
        return len(self._store), self.max_keys

    def snapshot(self) -> Dict[str, Any]:
        """Return a shallow dict copy of all values (for debugging/logging)."""
        return {k: v.value for k, v in self._store.items()}


class SharedMemoryRegion:
    """
    A named shared memory area accessible by multiple agents.

    Access control is enforced by listing authorized agent IDs.
    An empty authorized_agents set means open access to all.
    """

    def __init__(
        self,
        name: str,
        max_keys: int = 512,
        authorized_agents: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.max_keys = max_keys
        self.authorized_agents: List[str] = authorized_agents or []
        self._store: Dict[str, MemoryEntry] = {}

    def _check_auth(self, agent_id: str) -> None:
        if self.authorized_agents and agent_id not in self.authorized_agents:
            raise MemoryAccessError(
                f"Agent {agent_id!r} is not authorized to access shared region {self.name!r}."
            )

    def write(
        self,
        agent_id: str,
        key: str,
        value: Any,
        memory_type: MemoryType = MemoryType.SHARED,
        current_tick: int = 0,
        ttl_ticks: Optional[int] = None,
    ) -> MemoryEntry:
        self._check_auth(agent_id)
        if key not in self._store and len(self._store) >= self.max_keys:
            raise MemoryFullError(f"Shared region {self.name!r} is full ({self.max_keys} keys).")
        entry = MemoryEntry(
            key=key,
            value=value,
            memory_type=memory_type,
            created_tick=current_tick,
            ttl_ticks=ttl_ticks,
            owner_agent_id=agent_id,
        )
        self._store[key] = entry
        return entry

    def read(self, agent_id: str, key: str, current_tick: int = 0) -> Any:
        self._check_auth(agent_id)
        entry = self._store.get(key)
        if entry is None:
            raise MemoryKeyError(f"Key {key!r} not found in shared region {self.name!r}.")
        if entry.is_expired(current_tick):
            del self._store[key]
            raise MemoryKeyError(f"Key {key!r} in shared region {self.name!r} has expired.")
        return entry.value

    def delete(self, agent_id: str, key: str) -> bool:
        self._check_auth(agent_id)
        return self._store.pop(key, None) is not None

    def gc(self, current_tick: int) -> List[str]:
        expired = [k for k, v in self._store.items() if v.is_expired(current_tick)]
        for k in expired:
            del self._store[k]
        return expired

    def keys(self) -> List[str]:
        return list(self._store.keys())

    def snapshot(self) -> Dict[str, Any]:
        return {k: v.value for k, v in self._store.items()}


class MemoryManager:
    """
    Central memory manager for Battousai.

    Responsibilities:
    - Create and delete per-agent memory spaces
    - Create and manage named shared memory regions
    - Provide a unified read/write API used by the kernel syscall layer
    - Run garbage collection each tick
    - Report memory usage statistics
    """

    def __init__(self, default_agent_max_keys: int = 256) -> None:
        self.default_agent_max_keys = default_agent_max_keys
        self._agents: Dict[str, AgentMemorySpace] = {}
        self._shared: Dict[str, SharedMemoryRegion] = {}
        self._gc_count: int = 0

    # ------------------------------------------------------------------
    # Agent space lifecycle
    # ------------------------------------------------------------------

    def create_agent_space(
        self, agent_id: str, max_keys: Optional[int] = None
    ) -> AgentMemorySpace:
        max_keys = max_keys or self.default_agent_max_keys
        space = AgentMemorySpace(agent_id, max_keys)
        self._agents[agent_id] = space
        return space

    def delete_agent_space(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    def get_agent_space(self, agent_id: str) -> AgentMemorySpace:
        space = self._agents.get(agent_id)
        if space is None:
            raise MemoryAccessError(f"No memory space found for agent {agent_id!r}.")
        return space

    # ------------------------------------------------------------------
    # Shared region lifecycle
    # ------------------------------------------------------------------

    def create_shared_region(
        self,
        name: str,
        max_keys: int = 512,
        authorized_agents: Optional[List[str]] = None,
    ) -> SharedMemoryRegion:
        region = SharedMemoryRegion(name, max_keys, authorized_agents)
        self._shared[name] = region
        return region

    def get_shared_region(self, name: str) -> SharedMemoryRegion:
        region = self._shared.get(name)
        if region is None:
            raise MemoryKeyError(f"Shared memory region {name!r} does not exist.")
        return region

    # ------------------------------------------------------------------
    # Unified syscall-facing API
    # ------------------------------------------------------------------

    def agent_write(
        self,
        agent_id: str,
        key: str,
        value: Any,
        memory_type: MemoryType = MemoryType.LONG_TERM,
        current_tick: int = 0,
        ttl_ticks: Optional[int] = None,
    ) -> MemoryEntry:
        space = self.get_agent_space(agent_id)
        return space.write(key, value, memory_type, current_tick, ttl_ticks)

    def agent_read(self, agent_id: str, key: str, current_tick: int = 0) -> Any:
        space = self.get_agent_space(agent_id)
        return space.read(key, current_tick)

    def agent_delete(self, agent_id: str, key: str) -> bool:
        space = self.get_agent_space(agent_id)
        return space.delete(key)

    def shared_write(
        self,
        region_name: str,
        agent_id: str,
        key: str,
        value: Any,
        current_tick: int = 0,
        ttl_ticks: Optional[int] = None,
    ) -> MemoryEntry:
        region = self.get_shared_region(region_name)
        return region.write(agent_id, key, value, MemoryType.SHARED, current_tick, ttl_ticks)

    def shared_read(self, region_name: str, agent_id: str, key: str, current_tick: int = 0) -> Any:
        region = self.get_shared_region(region_name)
        return region.read(agent_id, key, current_tick)

    # ------------------------------------------------------------------
    # Garbage collection
    # ------------------------------------------------------------------

    def gc_tick(self, current_tick: int) -> Dict[str, List[str]]:
        """
        Run garbage collection across all agent spaces and shared regions.
        Returns a dict mapping space/region name to list of evicted keys.
        """
        evictions: Dict[str, List[str]] = {}
        for agent_id, space in self._agents.items():
            evicted = space.gc(current_tick)
            if evicted:
                evictions[f"agent:{agent_id}"] = evicted
        for name, region in self._shared.items():
            evicted = region.gc(current_tick)
            if evicted:
                evictions[f"shared:{name}"] = evicted
        self._gc_count += 1
        return evictions

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        agent_stats = {}
        for aid, space in self._agents.items():
            used, max_k = space.usage()
            agent_stats[aid] = {"used": used, "max": max_k}
        shared_stats = {
            name: {"keys": len(region.keys()), "max": region.max_keys}
            for name, region in self._shared.items()
        }
        return {
            "agents": agent_stats,
            "shared_regions": shared_stats,
            "gc_runs": self._gc_count,
        }
