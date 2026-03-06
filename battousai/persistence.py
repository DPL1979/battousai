"""
persistence.py — SQLite-based Persistence Layer
=================================================
Persists agent state, memory, audit logs, and capability grants
to a SQLite database. Zero external dependencies.

Schema
------
agent_state       — Snapshot of agent registry metadata.
memory_entries    — Per-agent and shared memory entries.
audit_log         — Capability audit trail from CapabilityManager.
capability_grants — Active capability token grants per agent.
schema_version    — Single-row table for migration tracking.

Features
--------
- WAL (Write-Ahead Logging) mode for concurrent readers.
- ``migrate()`` method for non-destructive schema upgrades.
- Checkpoint/restore for full kernel state serialisation.
- Save/load helpers for MemoryManager and CapabilityManager state.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Current schema version — bump when adding columns / tables.
_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    version  INTEGER NOT NULL,
    applied_at REAL NOT NULL
);
"""

_CREATE_AGENT_STATE = """
CREATE TABLE IF NOT EXISTS agent_state (
    agent_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    agent_class TEXT NOT NULL,
    priority    INTEGER NOT NULL DEFAULT 5,
    status      TEXT NOT NULL DEFAULT 'active',
    spawned_at  REAL NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);
"""

_CREATE_MEMORY_ENTRIES = """
CREATE TABLE IF NOT EXISTS memory_entries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id     TEXT NOT NULL,
    region_name  TEXT NOT NULL DEFAULT '',
    key          TEXT NOT NULL,
    value        TEXT NOT NULL,
    memory_type  TEXT NOT NULL,
    created_tick INTEGER NOT NULL DEFAULT 0,
    ttl_ticks    INTEGER,
    saved_at     REAL NOT NULL,
    UNIQUE(agent_id, region_name, key)
);
"""

_CREATE_AUDIT_LOG = """
CREATE TABLE IF NOT EXISTS audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id     TEXT NOT NULL UNIQUE,
    agent_id     TEXT NOT NULL,
    cap_type     TEXT NOT NULL,
    resource     TEXT,
    action       TEXT NOT NULL,
    granted      INTEGER NOT NULL DEFAULT 1,
    tick         INTEGER NOT NULL DEFAULT 0,
    timestamp    REAL NOT NULL
);
"""

_CREATE_CAPABILITY_GRANTS = """
CREATE TABLE IF NOT EXISTS capability_grants (
    token_id        TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    cap_type        TEXT NOT NULL,
    resource_pattern TEXT,
    delegatable     INTEGER NOT NULL DEFAULT 0,
    granted_at      REAL NOT NULL,
    expires_at      REAL,
    metadata        TEXT NOT NULL DEFAULT '{}'
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_memory_agent ON memory_entries(agent_id);",
    "CREATE INDEX IF NOT EXISTS idx_memory_region ON memory_entries(region_name);",
    "CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_log(agent_id);",
    "CREATE INDEX IF NOT EXISTS idx_caps_agent ON capability_grants(agent_id);",
]


# ---------------------------------------------------------------------------
# PersistenceLayer
# ---------------------------------------------------------------------------

class PersistenceLayer:
    """
    SQLite-backed persistence for Battousai kernel state.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.  Use ``":memory:"`` for ephemeral
        in-process storage (tests, demos).

    Usage
    -----
    ::

        pl = PersistenceLayer("/var/battousai/state.db")
        pl.migrate()                    # create / upgrade schema

        # Save all agent memory
        pl.save_memory_manager(kernel.memory)

        # Restore after restart
        pl.load_memory_manager(kernel.memory)

        # Checkpoint full kernel state
        checkpoint_id = pl.checkpoint(kernel)
        pl.restore(checkpoint_id, kernel)
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """Return (and cache) the SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0,
            )
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for concurrent reads
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "PersistenceLayer":
        self.connect()
        self.migrate()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def migrate(self) -> None:
        """
        Create or upgrade database schema to the current version.

        This method is idempotent — safe to call on every startup.
        Existing data is never destroyed during migration.
        """
        conn = self.connect()
        with conn:
            # Core tables
            conn.execute(_CREATE_SCHEMA_VERSION)
            conn.execute(_CREATE_AGENT_STATE)
            conn.execute(_CREATE_MEMORY_ENTRIES)
            conn.execute(_CREATE_AUDIT_LOG)
            conn.execute(_CREATE_CAPABILITY_GRANTS)
            for idx_sql in _INDEXES:
                conn.execute(idx_sql)

            # Record version if not already present
            row = conn.execute("SELECT version FROM schema_version LIMIT 1;").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?);",
                    (_SCHEMA_VERSION, time.time()),
                )
            elif row["version"] < _SCHEMA_VERSION:
                self._apply_migrations(conn, from_version=row["version"])
                conn.execute(
                    "UPDATE schema_version SET version=?, applied_at=?;",
                    (_SCHEMA_VERSION, time.time()),
                )

        logger.debug("PersistenceLayer: schema at version %d", _SCHEMA_VERSION)

    def _apply_migrations(
        self, conn: sqlite3.Connection, from_version: int
    ) -> None:
        """Apply incremental migrations from ``from_version`` to ``_SCHEMA_VERSION``."""
        # Future migrations would be added here as elif blocks.
        # Example:
        #   if from_version < 2:
        #       conn.execute("ALTER TABLE agent_state ADD COLUMN ...")
        logger.info(
            "PersistenceLayer: migrating from v%d to v%d",
            from_version,
            _SCHEMA_VERSION,
        )

    def get_schema_version(self) -> int:
        """Return the current schema version stored in the DB."""
        conn = self.connect()
        row = conn.execute("SELECT version FROM schema_version LIMIT 1;").fetchone()
        return row["version"] if row else 0

    # ------------------------------------------------------------------
    # Agent state
    # ------------------------------------------------------------------

    def save_agent(
        self,
        agent_id: str,
        name: str,
        agent_class: str,
        priority: int = 5,
        status: str = "active",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or replace an agent record."""
        conn = self.connect()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_state
                    (agent_id, name, agent_class, priority, status, spawned_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    agent_id, name, agent_class, priority, status,
                    time.time(), json.dumps(metadata or {}),
                ),
            )

    def load_agents(self) -> List[Dict[str, Any]]:
        """Return all agent records as dicts."""
        conn = self.connect()
        rows = conn.execute("SELECT * FROM agent_state;").fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["metadata"] = json.loads(d.get("metadata") or "{}")
            result.append(d)
        return result

    def delete_agent(self, agent_id: str) -> None:
        """Remove an agent record."""
        conn = self.connect()
        with conn:
            conn.execute("DELETE FROM agent_state WHERE agent_id=?;", (agent_id,))

    # ------------------------------------------------------------------
    # Memory persistence
    # ------------------------------------------------------------------

    def save_memory_entry(
        self,
        agent_id: str,
        key: str,
        value: Any,
        memory_type: str,
        created_tick: int = 0,
        ttl_ticks: Optional[int] = None,
        region_name: Optional[str] = None,
    ) -> None:
        """Upsert a single memory entry."""
        conn = self.connect()
        try:
            serialised = json.dumps(value)
        except (TypeError, ValueError):
            serialised = json.dumps(str(value))

        # Normalise NULL region_name to empty string for UNIQUE constraint compatibility
        effective_region = region_name if region_name is not None else ""

        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_entries
                    (agent_id, region_name, key, value, memory_type,
                     created_tick, ttl_ticks, saved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    agent_id,
                    effective_region,
                    key,
                    serialised,
                    memory_type,
                    created_tick,
                    ttl_ticks,
                    time.time(),
                ),
            )

    def load_memory_entries(
        self,
        agent_id: str,
        region_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load all memory entries for an agent (or a shared region).

        Parameters
        ----------
        agent_id    : str
        region_name : str, optional — shared region name

        Returns
        -------
        list of dicts with keys: key, value, memory_type, created_tick, ttl_ticks
        """
        conn = self.connect()
        if region_name is not None:
            rows = conn.execute(
                "SELECT * FROM memory_entries WHERE agent_id=? AND region_name=?;",
                (agent_id, region_name),
            ).fetchall()
        else:
            # Agent-private entries use empty string for region_name
            rows = conn.execute(
                "SELECT * FROM memory_entries WHERE agent_id=? AND region_name='';",
                (agent_id,),
            ).fetchall()

        result = []
        for row in rows:
            d = dict(row)
            try:
                d["value"] = json.loads(d["value"])
            except (json.JSONDecodeError, TypeError):
                pass
            result.append(d)
        return result

    def save_memory_manager(self, memory_manager: Any) -> None:
        """
        Serialise the entire MemoryManager state to the database.

        Parameters
        ----------
        memory_manager : battousai.memory.MemoryManager
        """
        conn = self.connect()
        # Clear existing entries to avoid stale data
        with conn:
            conn.execute("DELETE FROM memory_entries;")

        # Agent spaces
        for agent_id, space in memory_manager._agents.items():
            for key, entry in space._store.items():
                self.save_memory_entry(
                    agent_id=agent_id,
                    key=key,
                    value=entry.value,
                    memory_type=entry.memory_type.name,
                    created_tick=entry.created_tick,
                    ttl_ticks=entry.ttl_ticks,
                    region_name=None,
                )

        # Shared regions
        for region_name, region in memory_manager._shared.items():
            for key, entry in region._store.items():
                self.save_memory_entry(
                    agent_id=entry.owner_agent_id or "kernel",
                    key=key,
                    value=entry.value,
                    memory_type=entry.memory_type.name,
                    created_tick=entry.created_tick,
                    ttl_ticks=entry.ttl_ticks,
                    region_name=region_name,
                )

        logger.debug(
            "PersistenceLayer: saved memory for %d agents, %d shared regions",
            len(memory_manager._agents),
            len(memory_manager._shared),
        )

    def load_memory_manager(self, memory_manager: Any) -> None:
        """
        Restore MemoryManager state from the database.

        Creates agent spaces and shared regions as needed.

        Parameters
        ----------
        memory_manager : battousai.memory.MemoryManager
        """
        from battousai.memory import MemoryType

        conn = self.connect()
        rows = conn.execute("SELECT * FROM memory_entries;").fetchall()

        for row in rows:
            agent_id = row["agent_id"]
            key = row["key"]
            region_name = row["region_name"]
            created_tick = row["created_tick"]
            ttl_ticks = row["ttl_ticks"]
            try:
                value = json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                value = row["value"]

            mem_type_str = row["memory_type"]
            try:
                mem_type = MemoryType[mem_type_str]
            except KeyError:
                mem_type = MemoryType.LONG_TERM

            if region_name:  # non-empty string means it's a shared region
                # Shared region
                if region_name not in memory_manager._shared:
                    memory_manager.create_shared_region(region_name)
                region = memory_manager._shared[region_name]
                region.write(
                    agent_id=agent_id,
                    key=key,
                    value=value,
                    memory_type=mem_type,
                    current_tick=created_tick,
                    ttl_ticks=ttl_ticks,
                )
            else:
                # Agent-private space (region_name == "")
                if agent_id not in memory_manager._agents:
                    memory_manager.create_agent_space(agent_id)
                space = memory_manager._agents[agent_id]
                space.write(
                    key=key,
                    value=value,
                    memory_type=mem_type,
                    current_tick=created_tick,
                    ttl_ticks=ttl_ticks,
                )

        logger.debug("PersistenceLayer: restored %d memory entries", len(rows))

    # ------------------------------------------------------------------
    # Audit log persistence
    # ------------------------------------------------------------------

    def save_audit_entry(
        self,
        entry_id: str,
        agent_id: str,
        cap_type: str,
        resource: Optional[str],
        action: str,
        granted: bool,
        tick: int = 0,
        timestamp: Optional[float] = None,
    ) -> None:
        """Append a capability audit log entry."""
        conn = self.connect()
        with conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO audit_log
                    (entry_id, agent_id, cap_type, resource, action, granted, tick, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    entry_id, agent_id, cap_type, resource, action,
                    int(granted), tick, timestamp or time.time(),
                ),
            )

    def load_audit_log(
        self,
        agent_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Load audit log entries, optionally filtered by agent.

        Parameters
        ----------
        agent_id : str, optional — filter to one agent
        limit    : int           — maximum rows to return

        Returns
        -------
        list of dicts (most recent first)
        """
        conn = self.connect()
        if agent_id:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE agent_id=? ORDER BY id DESC LIMIT ?;",
                (agent_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?;",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_capability_manager_audit(self, capability_manager: Any) -> None:
        """
        Persist the audit log from a CapabilityManager instance.

        Parameters
        ----------
        capability_manager : battousai.capabilities.CapabilityManager
        """
        if not hasattr(capability_manager, "_audit_log"):
            return

        for entry in capability_manager._audit_log:
            self.save_audit_entry(
                entry_id=getattr(entry, "entry_id", str(id(entry))),
                agent_id=getattr(entry, "agent_id", ""),
                cap_type=str(getattr(entry, "cap_type", "")),
                resource=getattr(entry, "resource", None),
                action=getattr(entry, "action", ""),
                granted=bool(getattr(entry, "granted", True)),
                tick=getattr(entry, "tick", 0),
                timestamp=getattr(entry, "timestamp", time.time()),
            )

    # ------------------------------------------------------------------
    # Capability grants persistence
    # ------------------------------------------------------------------

    def save_capability_grant(
        self,
        token_id: str,
        agent_id: str,
        cap_type: str,
        resource_pattern: Optional[str] = None,
        delegatable: bool = False,
        expires_at: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a capability grant token."""
        conn = self.connect()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO capability_grants
                    (token_id, agent_id, cap_type, resource_pattern,
                     delegatable, granted_at, expires_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    token_id, agent_id, cap_type, resource_pattern,
                    int(delegatable), time.time(), expires_at,
                    json.dumps(metadata or {}),
                ),
            )

    def load_capability_grants(
        self, agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return persisted capability grants, optionally filtered by agent."""
        conn = self.connect()
        if agent_id:
            rows = conn.execute(
                "SELECT * FROM capability_grants WHERE agent_id=?;",
                (agent_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM capability_grants;").fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["metadata"] = json.loads(d.get("metadata") or "{}")
            d["delegatable"] = bool(d.get("delegatable", 0))
            result.append(d)
        return result

    def revoke_capability_grant(self, token_id: str) -> None:
        """Remove a capability grant by token ID."""
        conn = self.connect()
        with conn:
            conn.execute(
                "DELETE FROM capability_grants WHERE token_id=?;", (token_id,)
            )

    # ------------------------------------------------------------------
    # Checkpoint / restore
    # ------------------------------------------------------------------

    def checkpoint(
        self,
        memory_manager: Any,
        capability_manager: Optional[Any] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a full checkpoint of memory + capability state.

        Parameters
        ----------
        memory_manager     : battousai.memory.MemoryManager
        capability_manager : battousai.capabilities.CapabilityManager, optional
        extra_metadata     : dict, optional — arbitrary extra data

        Returns
        -------
        str
            Checkpoint identifier (timestamp + microseconds for uniqueness).
        """
        import datetime

        now = datetime.datetime.utcnow()
        checkpoint_id = now.strftime("%Y%m%dT%H%M%S") + f"_{now.microsecond:06d}Z"
        logger.info("PersistenceLayer: creating checkpoint %s", checkpoint_id)

        self.save_memory_manager(memory_manager)

        if capability_manager is not None:
            self.save_capability_manager_audit(capability_manager)

        # Save a checkpoint record in agent_state with a sentinel ID
        self.save_agent(
            agent_id=f"__checkpoint_{checkpoint_id}",
            name="checkpoint",
            agent_class="Checkpoint",
            status="checkpoint",
            metadata={
                "checkpoint_id": checkpoint_id,
                "created_at": time.time(),
                "extra": extra_metadata or {},
            },
        )

        logger.info("PersistenceLayer: checkpoint %s saved", checkpoint_id)
        return checkpoint_id

    def restore(
        self,
        checkpoint_id: str,
        memory_manager: Any,
    ) -> bool:
        """
        Restore memory state from a checkpoint.

        Parameters
        ----------
        checkpoint_id  : str — as returned by ``checkpoint()``
        memory_manager : battousai.memory.MemoryManager

        Returns
        -------
        bool
            ``True`` if the checkpoint was found and loaded; ``False`` otherwise.
        """
        conn = self.connect()
        row = conn.execute(
            "SELECT * FROM agent_state WHERE agent_id=?;",
            (f"__checkpoint_{checkpoint_id}",),
        ).fetchone()

        if row is None:
            logger.warning(
                "PersistenceLayer: checkpoint %s not found", checkpoint_id
            )
            return False

        self.load_memory_manager(memory_manager)
        logger.info("PersistenceLayer: restored checkpoint %s", checkpoint_id)
        return True

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear_all(self) -> None:
        """Delete all rows from all tables (destructive — use in tests only)."""
        conn = self.connect()
        with conn:
            for table in (
                "memory_entries",
                "audit_log",
                "capability_grants",
                "agent_state",
            ):
                conn.execute(f"DELETE FROM {table};")

    def table_row_counts(self) -> Dict[str, int]:
        """Return a dict of table → row count for diagnostics."""
        conn = self.connect()
        counts: Dict[str, int] = {}
        for table in (
            "schema_version",
            "agent_state",
            "memory_entries",
            "audit_log",
            "capability_grants",
        ):
            row = conn.execute(f"SELECT COUNT(*) as c FROM {table};").fetchone()
            counts[table] = row["c"] if row else 0
        return counts
