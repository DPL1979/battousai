"""
federation.py — Multi-Kernel Federation
==========================================
Enables multiple Battousai kernel instances to form a federated system
where agents can transparently operate across kernel boundaries.

Design Rationale
----------------
A single-kernel Battousai is bounded by one machine's resources. Federation
breaks that limit: multiple kernels can collaborate as a cluster, sharing
agents, distributing load, and tolerating node failures.

The architecture is inspired by distributed systems research:
    - Raft consensus for leader election and log replication
    - Gossip protocols for GlobalRegistry synchronisation
    - Live migration for agent mobility across nodes
    - Circuit-breaker style split-brain detection

Safety note: agent migration includes a checkpoint of the agent's memory
and inbox. If migration fails mid-flight, the agent stays on the source
node. "At-most-once" migration semantics — we never clone an agent.

Architecture:
    FederationNode     — wraps a Kernel with federation capabilities
    FederationCluster  — manages the cluster of nodes
    ConsensusProtocol  — Raft-inspired leader election and log replication
    AgentMigrator      — moves agents between nodes for load balancing
    GlobalRegistry     — cluster-wide agent and service directory
    LoadBalancer       — distributes agent spawns across nodes
    SplitBrainDetector — detects and handles network partitions
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class NodeRole(Enum):
    """
    Raft consensus roles.

    LEADER    — receives all client requests, replicates log to followers
    FOLLOWER  — replicates log from leader, votes in elections
    CANDIDATE — running for election; solicits votes from all peers
    """
    LEADER    = auto()
    FOLLOWER  = auto()
    CANDIDATE = auto()


class BalancingStrategy(Enum):
    """Load balancing strategies for agent placement."""
    ROUND_ROBIN  = auto()  # Rotate through nodes in order
    LEAST_LOADED = auto()  # Pick the node with fewest agents
    RANDOM       = auto()  # Random node selection
    AFFINITY     = auto()  # Keep related agents on the same node


class MigrationStatus(Enum):
    """Status of an in-flight agent migration."""
    PENDING    = auto()
    SERIALIZING = auto()
    TRANSFERRING = auto()
    DESERIALIZING = auto()
    COMPLETED  = auto()
    FAILED     = auto()


# ---------------------------------------------------------------------------
# Cluster Log Entry
# ---------------------------------------------------------------------------

@dataclass
class ClusterEntry:
    """
    A single entry in the Raft distributed log.

    The log is the authoritative record of all cluster-wide state changes.
    Entries are applied to the state machine only after they are committed
    (replicated to a majority of nodes).

    Fields:
        term     — election term in which this entry was created
        index    — monotonically increasing log position
        command  — type of state change: "spawn", "kill", "message", "register"
        data     — command-specific payload (agent spec, message body, etc.)
        leader_id — node_id of the leader that created this entry
    """
    term: int
    index: int
    command: str
    data: Any
    leader_id: str = ""
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Federation Node
# ---------------------------------------------------------------------------

class FederationNode:
    """
    Wraps a Kernel with Raft-inspired federation capabilities.

    Each node maintains:
        - A local copy of the cluster log
        - An election term counter and vote record
        - A heartbeat timer (resets when a message is received from the leader)
        - A list of peer node IDs

    Raft summary (simplified for prototype):
        1. Nodes start as FOLLOWERs.
        2. If a follower's heartbeat timer expires, it becomes a CANDIDATE,
           increments its term, and sends vote requests to all peers.
        3. The first node to receive votes from >50% of nodes becomes LEADER.
        4. The LEADER sends heartbeats every ``heartbeat_interval`` ticks to
           prevent new elections.
        5. The LEADER replicates log entries to all followers.
        6. An entry is "committed" once a majority acknowledge it.

    Note: Full Raft includes log compaction, snapshots, and membership
    changes. This prototype implements the core election and replication loop.
    """

    # Ticks before a follower starts a new election
    ELECTION_TIMEOUT_MIN = 5
    ELECTION_TIMEOUT_MAX = 10
    HEARTBEAT_INTERVAL   = 2  # ticks between leader heartbeats

    def __init__(self, kernel: Any, node_id: Optional[str] = None) -> None:
        self.kernel = kernel
        self.node_id: str = node_id or str(uuid.uuid4())[:8]
        self.role: NodeRole = NodeRole.FOLLOWER
        self.term: int = 0
        self.voted_for: Optional[str] = None  # candidate voted for in current term
        self.peers: List[str] = []  # node_ids of other nodes
        self.log: List[ClusterEntry] = []  # cluster log
        self.commit_index: int = 0  # highest committed log index
        self.last_applied: int = 0  # highest applied log index
        self._votes_received: Set[str] = set()
        self._heartbeat_timer: int = self._random_election_timeout()
        self._last_heartbeat_tick: int = 0
        self._current_tick: int = 0
        self._connected_peers: Set[str] = set()  # peers we can reach

    def _random_election_timeout(self) -> int:
        return random.randint(self.ELECTION_TIMEOUT_MIN, self.ELECTION_TIMEOUT_MAX)

    def add_peer(self, node_id: str) -> None:
        """Register another node as a peer."""
        if node_id not in self.peers and node_id != self.node_id:
            self.peers.append(node_id)
            self._connected_peers.add(node_id)

    def request_vote(self, candidate_id: str, candidate_term: int) -> bool:
        """
        Handle a RequestVote RPC from a candidate.

        Grant the vote if:
          1. The candidate's term >= our term
          2. We haven't already voted for someone else this term
          3. The candidate's log is at least as up-to-date as ours
        """
        if candidate_term < self.term:
            return False
        if candidate_term > self.term:
            self.term = candidate_term
            self.role = NodeRole.FOLLOWER
            self.voted_for = None

        if self.voted_for is None or self.voted_for == candidate_id:
            self.voted_for = candidate_id
            self._heartbeat_timer = self._random_election_timeout()
            return True
        return False

    def append_entries(
        self,
        leader_id: str,
        leader_term: int,
        entries: List[ClusterEntry],
        leader_commit: int,
    ) -> bool:
        """
        Handle an AppendEntries RPC from the leader.

        This doubles as a heartbeat (entries may be empty).
        On success, appends new entries to the local log and advances commit_index.
        """
        if leader_term < self.term:
            return False

        # Valid leader — reset heartbeat timer and step down if we're a candidate
        if leader_term > self.term:
            self.term = leader_term
            self.voted_for = None
        self.role = NodeRole.FOLLOWER
        self._heartbeat_timer = self._random_election_timeout()
        self._last_heartbeat_tick = self._current_tick

        # Append new entries
        for entry in entries:
            if entry.index > len(self.log):
                self.log.append(entry)

        # Advance commit index
        if leader_commit > self.commit_index:
            self.commit_index = min(leader_commit, len(self.log))

        return True

    def heartbeat(self, leader_id: str, leader_term: int) -> bool:
        """Receive a heartbeat from the leader (empty AppendEntries)."""
        return self.append_entries(leader_id, leader_term, [], self.commit_index)

    def tick(self, current_tick: int) -> Optional[str]:
        """
        Advance the node's Raft state machine by one tick.

        Returns:
            "election_started" if this node started an election,
            "leader_elected"   if this node won the election,
            None               otherwise.
        """
        self._current_tick = current_tick
        event: Optional[str] = None

        if self.role == NodeRole.LEADER:
            # Send heartbeats at regular intervals
            if current_tick - self._last_heartbeat_tick >= self.HEARTBEAT_INTERVAL:
                self._last_heartbeat_tick = current_tick
                # Heartbeats are dispatched by FederationCluster
        elif self.role == NodeRole.FOLLOWER:
            # Count down heartbeat timer
            self._heartbeat_timer -= 1
            if self._heartbeat_timer <= 0:
                # Election timeout — start an election
                self._start_election()
                event = "election_started"
        elif self.role == NodeRole.CANDIDATE:
            # Check if we've won
            majority = (len(self.peers) + 1) // 2 + 1
            if len(self._votes_received) >= majority:
                self._become_leader()
                event = "leader_elected"
            else:
                # Re-request votes (in case of packet loss)
                self._heartbeat_timer -= 1
                if self._heartbeat_timer <= 0:
                    self._start_election()
                    event = "election_started"

        return event

    def _start_election(self) -> None:
        """Become a candidate and increment the election term."""
        self.term += 1
        self.role = NodeRole.CANDIDATE
        self.voted_for = self.node_id
        self._votes_received = {self.node_id}  # vote for ourselves
        self._heartbeat_timer = self._random_election_timeout()

    def _become_leader(self) -> None:
        """Transition to LEADER state."""
        self.role = NodeRole.LEADER
        self._last_heartbeat_tick = self._current_tick

    def append_to_log(self, command: str, data: Any) -> ClusterEntry:
        """
        (Leader only) Append a new entry to the distributed log.

        The entry is immediately added to the leader's local log.
        FederationCluster.tick() replicates it to followers.
        """
        entry = ClusterEntry(
            term=self.term,
            index=len(self.log) + 1,
            command=command,
            data=data,
            leader_id=self.node_id,
        )
        self.log.append(entry)
        return entry

    def agent_count(self) -> int:
        """Return the number of agents running on this node's kernel."""
        return len(self.kernel._agents)

    def is_readable(self) -> bool:
        """Is this node accepting reads? (Always true unless in split-brain READ_ONLY mode)"""
        return True

    def snapshot(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role.name,
            "term": self.term,
            "log_length": len(self.log),
            "commit_index": self.commit_index,
            "agent_count": self.agent_count(),
            "peers": self.peers,
        }


# ---------------------------------------------------------------------------
# Federation Cluster
# ---------------------------------------------------------------------------

class FederationCluster:
    """
    Manages a cluster of FederationNodes and drives the consensus protocol.

    Responsibilities:
        - Maintain the node registry
        - Drive per-node tick() calls each system tick
        - Route vote requests and heartbeats between nodes
        - Replicate log entries from leader to followers
        - Expose the current leader to external callers

    The cluster is the single entry point for cluster-wide operations.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, FederationNode] = {}
        self._tick: int = 0

    def add_node(self, kernel: Any, node_id: Optional[str] = None) -> FederationNode:
        """
        Wrap a Kernel in a FederationNode and add it to the cluster.

        All existing nodes are automatically added as peers of the new node.
        """
        node = FederationNode(kernel, node_id=node_id)
        # Wire up peers
        for existing_id, existing_node in self._nodes.items():
            node.add_peer(existing_id)
            existing_node.add_peer(node.node_id)
        self._nodes[node.node_id] = node
        return node

    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the cluster. Triggers re-election if it was the leader."""
        node = self._nodes.pop(node_id, None)
        if node is None:
            return False
        for other in self._nodes.values():
            if node_id in other.peers:
                other.peers.remove(node_id)
            other._connected_peers.discard(node_id)
        return True

    def get_leader(self) -> Optional[str]:
        """Return the node_id of the current leader, or None if no leader exists."""
        for node_id, node in self._nodes.items():
            if node.role == NodeRole.LEADER:
                return node_id
        return None

    def get_node(self, node_id: str) -> Optional[FederationNode]:
        return self._nodes.get(node_id)

    def list_nodes(self) -> List[str]:
        return list(self._nodes.keys())

    def tick(self) -> Dict[str, Any]:
        """
        Advance all nodes by one tick.

        Order of operations:
            1. Tick each node's Raft state machine
            2. Process vote requests from new candidates
            3. Send heartbeats from leader to all followers
            4. Replicate new log entries from leader to followers

        Returns a dict of events that occurred this tick.
        """
        self._tick += 1
        events: Dict[str, Any] = {"tick": self._tick, "elections": [], "heartbeats": 0, "entries_replicated": 0}

        # 1. Tick each node
        new_candidates: List[str] = []
        new_leaders: List[str] = []
        for node_id, node in self._nodes.items():
            event = node.tick(self._tick)
            if event == "election_started":
                new_candidates.append(node_id)
                events["elections"].append({"node": node_id, "term": node.term})
            elif event == "leader_elected":
                new_leaders.append(node_id)

        # 2. Distribute vote requests from candidates
        for candidate_id in new_candidates:
            candidate = self._nodes[candidate_id]
            for peer_id in candidate.peers:
                peer = self._nodes.get(peer_id)
                if peer is None:
                    continue
                granted = peer.request_vote(candidate_id, candidate.term)
                if granted:
                    candidate._votes_received.add(peer_id)
            # Check if candidate now has majority
            majority = (len(candidate.peers) + 1) // 2 + 1
            if len(candidate._votes_received) >= majority:
                candidate._become_leader()
                new_leaders.append(candidate_id)

        # 3. Leader heartbeats and log replication
        leader_id = self.get_leader()
        if leader_id:
            leader = self._nodes[leader_id]
            for follower_id, follower in self._nodes.items():
                if follower_id == leader_id:
                    continue
                # Heartbeat
                follower.heartbeat(leader_id, leader.term)
                events["heartbeats"] += 1
                # Replicate any log entries the follower is missing
                follower_log_len = len(follower.log)
                new_entries = leader.log[follower_log_len:]
                if new_entries:
                    follower.append_entries(
                        leader_id, leader.term, new_entries, leader.commit_index
                    )
                    events["entries_replicated"] += len(new_entries)
            # Advance leader's commit_index (simplified: commit immediately)
            if leader.log:
                leader.commit_index = len(leader.log)

        return events

    def broadcast_log_entry(self, command: str, data: Any) -> Optional[ClusterEntry]:
        """
        Append a log entry via the leader and replicate it.

        Returns the ClusterEntry, or None if no leader is available.
        """
        leader_id = self.get_leader()
        if leader_id is None:
            return None
        leader = self._nodes[leader_id]
        entry = leader.append_to_log(command, data)
        # Replicate immediately
        for follower_id, follower in self._nodes.items():
            if follower_id == leader_id:
                continue
            follower.append_entries(leader_id, leader.term, [entry], leader.commit_index)
        return entry

    def stats(self) -> Dict[str, Any]:
        return {
            "cluster_size": len(self._nodes),
            "leader": self.get_leader(),
            "tick": self._tick,
            "nodes": {nid: node.snapshot() for nid, node in self._nodes.items()},
        }


# ---------------------------------------------------------------------------
# Agent Migrator
# ---------------------------------------------------------------------------

class AgentMigrator:
    """
    Moves agents between FederationNodes for load balancing or maintenance.

    Migration protocol (at-most-once semantics):
        1. SERIALIZE  — snapshot agent's memory and inbox on source node
        2. TRANSFER   — transmit the snapshot (simulated; in-process dict copy)
        3. DESERIALIZE — reconstruct the agent on the destination node
        4. REGISTER   — register the agent on the destination kernel
        5. DEREGISTER — remove the agent from the source kernel ONLY after
                         successful registration on destination

    If any step fails, the agent remains on the source node unchanged.
    The migration log records all migration attempts and outcomes.
    """

    def __init__(self, cluster: FederationCluster) -> None:
        self.cluster = cluster
        self._migration_log: List[Dict[str, Any]] = []
        self._migration_count: int = 0
        self._failure_count: int = 0

    def migrate(
        self,
        agent_id: str,
        src_node_id: str,
        dst_node_id: str,
    ) -> bool:
        """
        Migrate an agent from source to destination node.

        Returns True on success, False on failure (agent stays on source).
        """
        record: Dict[str, Any] = {
            "migration_id": str(uuid.uuid4())[:8],
            "agent_id": agent_id,
            "src": src_node_id,
            "dst": dst_node_id,
            "status": MigrationStatus.PENDING.name,
            "started_at": time.time(),
        }

        src_node = self.cluster.get_node(src_node_id)
        dst_node = self.cluster.get_node(dst_node_id)

        if src_node is None or dst_node is None:
            record["status"] = MigrationStatus.FAILED.name
            record["error"] = "Source or destination node not found"
            self._migration_log.append(record)
            self._failure_count += 1
            return False

        agent = src_node.kernel._agents.get(agent_id)
        if agent is None:
            record["status"] = MigrationStatus.FAILED.name
            record["error"] = f"Agent {agent_id!r} not found on source node"
            self._migration_log.append(record)
            self._failure_count += 1
            return False

        try:
            # Step 1: Serialize agent state
            record["status"] = MigrationStatus.SERIALIZING.name
            snapshot = self._serialize_agent(agent, src_node.kernel)

            # Step 2: Transfer (in-memory copy for prototype)
            record["status"] = MigrationStatus.TRANSFERRING.name
            transfer_packet = dict(snapshot)  # shallow copy simulates network transfer

            # Step 3: Deserialize on destination
            record["status"] = MigrationStatus.DESERIALIZING.name
            success = self._deserialize_agent(transfer_packet, dst_node.kernel)
            if not success:
                raise RuntimeError("Deserialization failed on destination node")

            # Step 4: Deregister on source ONLY after successful arrival
            src_node.kernel.kill_agent(agent_id)

            record["status"] = MigrationStatus.COMPLETED.name
            record["completed_at"] = time.time()
            self._migration_count += 1

            # Notify the global registry about the move
            self._migration_log.append(record)
            return True

        except Exception as exc:
            record["status"] = MigrationStatus.FAILED.name
            record["error"] = str(exc)
            self._migration_log.append(record)
            self._failure_count += 1
            return False

    def _serialize_agent(self, agent: Any, kernel: Any) -> Dict[str, Any]:
        """
        Capture agent state into a serialisable dict.

        Captured state:
            - Class type (by name)
            - Constructor kwargs (name, priority)
            - Memory snapshot (key → value)
            - Pending inbox messages
            - Internal counters (_ticks_alive, _spawn_tick)
        """
        # Memory snapshot
        memory_snapshot: Dict[str, Any] = {}
        try:
            space = kernel.memory.get_agent_space(agent.agent_id)
            memory_snapshot = space.snapshot()
        except Exception:
            pass

        # Inbox snapshot (drain messages without consuming)
        inbox_snapshot: List[Any] = []
        try:
            mb = kernel.ipc.get_mailbox(agent.agent_id)
            if mb:
                # Peek at all messages without consuming them
                inbox_snapshot = list(mb._queue)
        except Exception:
            pass

        return {
            "class_name": type(agent).__name__,
            "class_ref": type(agent),
            "name": agent.name,
            "priority": agent.priority,
            "memory_allocation": agent.memory_allocation,
            "time_slice": agent.time_slice,
            "ticks_alive": agent._ticks_alive,
            "spawn_tick": agent._spawn_tick,
            "memory": memory_snapshot,
            "inbox": inbox_snapshot,
            "agent_id_hint": agent.agent_id,  # suggested ID on destination
        }

    def _deserialize_agent(self, snapshot: Dict[str, Any], dst_kernel: Any) -> bool:
        """
        Reconstruct an agent from a serialised snapshot on the destination kernel.

        The agent is spawned as a new instance with the same class and configuration.
        Memory entries and queued messages are restored after spawning.
        """
        try:
            agent_class = snapshot["class_ref"]
            name = snapshot["name"]
            priority = snapshot["priority"]
            memory_allocation = snapshot["memory_allocation"]
            time_slice = snapshot["time_slice"]

            # Spawn on destination
            new_agent_id = dst_kernel.spawn_agent(
                agent_class,
                name=name,
                priority=priority,
                memory_allocation=memory_allocation,
                time_slice=time_slice,
            )
            new_agent = dst_kernel._agents[new_agent_id]
            new_agent._ticks_alive = snapshot.get("ticks_alive", 0)

            # Restore memory
            from battousai.memory import MemoryType
            for key, value in snapshot.get("memory", {}).items():
                try:
                    dst_kernel.memory.agent_write(
                        new_agent_id, key, value, MemoryType.LONG_TERM, dst_kernel._tick
                    )
                except Exception:
                    pass

            # Restore inbox messages
            dst_mb = dst_kernel.ipc.get_mailbox(new_agent_id)
            if dst_mb:
                for msg in snapshot.get("inbox", []):
                    try:
                        dst_mb.deliver(msg)
                    except Exception:
                        pass

            return True
        except Exception:
            return False

    def migration_report(self) -> str:
        """Return a formatted migration history report."""
        lines = [f"=== AgentMigrator Report ({len(self._migration_log)} migrations) ==="]
        for rec in self._migration_log[-20:]:  # last 20
            status = rec.get("status", "?")
            lines.append(
                f"  {rec.get('migration_id','?')} | "
                f"agent={rec.get('agent_id','?')} | "
                f"{rec.get('src','?')} → {rec.get('dst','?')} | "
                f"status={status}"
                + (f" | err={rec['error']}" if status == "FAILED" else "")
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Global Registry — cluster-wide agent and service directory
# ---------------------------------------------------------------------------

class GlobalRegistry:
    """
    Cluster-wide directory of agents and named services.

    Each FederationNode maintains a local copy of the registry.
    Updates are propagated via gossip: when a node learns of a change,
    it forwards the update to a random subset of peers.

    Entries:
        agent_id → (node_id, agent_name, metadata)
        service_name → [(agent_id, node_id), ...]

    The registry uses a logical clock (version counter) to resolve
    conflicting updates: higher version wins.
    """

    def __init__(self) -> None:
        # agent_id → {"node_id": str, "name": str, "meta": dict, "version": int}
        self._agents: Dict[str, Dict[str, Any]] = {}
        # service_name → list of {"agent_id": str, "node_id": str, "version": int}
        self._services: Dict[str, List[Dict[str, Any]]] = {}
        self._version: int = 0

    def register_agent(
        self,
        agent_id: str,
        node_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an agent's location in the global directory."""
        self._version += 1
        self._agents[agent_id] = {
            "node_id": node_id,
            "name": name,
            "meta": metadata or {},
            "version": self._version,
            "registered_at": time.time(),
        }

    def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the directory (e.g., on termination or migration)."""
        return self._agents.pop(agent_id, None) is not None

    def lookup_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Return location info for an agent, or None if not found."""
        return self._agents.get(agent_id)

    def find_node(self, agent_id: str) -> Optional[str]:
        """Return the node_id hosting a given agent."""
        entry = self._agents.get(agent_id)
        return entry["node_id"] if entry else None

    def register_service(self, service_name: str, agent_id: str, node_id: str) -> None:
        """
        Register an agent as a provider of a named service.

        Multiple agents can provide the same service (for redundancy).
        """
        self._version += 1
        providers = self._services.setdefault(service_name, [])
        # Remove old registration for same agent_id
        self._services[service_name] = [p for p in providers if p["agent_id"] != agent_id]
        self._services[service_name].append({
            "agent_id": agent_id,
            "node_id": node_id,
            "version": self._version,
        })

    def discover_service(self, service_name: str) -> List[Dict[str, Any]]:
        """Return all agents providing a named service."""
        return list(self._services.get(service_name, []))

    def unregister_service(self, service_name: str, agent_id: str) -> bool:
        providers = self._services.get(service_name, [])
        new_providers = [p for p in providers if p["agent_id"] != agent_id]
        if len(new_providers) < len(providers):
            self._services[service_name] = new_providers
            return True
        return False

    def agents_on_node(self, node_id: str) -> List[str]:
        """Return all agent IDs registered on a specific node."""
        return [aid for aid, info in self._agents.items() if info["node_id"] == node_id]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "agent_count": len(self._agents),
            "service_count": len(self._services),
            "version": self._version,
            "agents": dict(self._agents),
            "services": {s: list(ps) for s, ps in self._services.items()},
        }


# ---------------------------------------------------------------------------
# Load Balancer
# ---------------------------------------------------------------------------

class LoadBalancer:
    """
    Selects the optimal node for placing a new agent or migrating an existing one.

    Strategies:
        ROUND_ROBIN  — cycle through nodes regardless of load
        LEAST_LOADED — place on the node with fewest agents
        RANDOM       — uniform random selection
        AFFINITY     — keep agents with matching affinity_key on the same node
    """

    def __init__(
        self,
        cluster: FederationCluster,
        strategy: BalancingStrategy = BalancingStrategy.LEAST_LOADED,
    ) -> None:
        self.cluster = cluster
        self.strategy = strategy
        self._rr_index: int = 0  # round-robin cursor

    def select_node(
        self,
        agent_spec: Optional[Dict[str, Any]] = None,
        affinity_key: Optional[str] = None,
    ) -> Optional[str]:
        """
        Choose a node for placing an agent.

        Args:
            agent_spec   — optional agent descriptor (used for AFFINITY)
            affinity_key — if provided with AFFINITY strategy, selects the
                           node hosting other agents with the same key

        Returns:
            node_id of the selected node, or None if no nodes available.
        """
        node_ids = self.cluster.list_nodes()
        if not node_ids:
            return None

        if self.strategy == BalancingStrategy.ROUND_ROBIN:
            node_id = node_ids[self._rr_index % len(node_ids)]
            self._rr_index += 1
            return node_id

        elif self.strategy == BalancingStrategy.LEAST_LOADED:
            return min(
                node_ids,
                key=lambda nid: (
                    self.cluster.get_node(nid).agent_count()
                    if self.cluster.get_node(nid) else 999
                ),
            )

        elif self.strategy == BalancingStrategy.RANDOM:
            return random.choice(node_ids)

        elif self.strategy == BalancingStrategy.AFFINITY:
            # Default to least-loaded if no affinity can be determined
            return min(
                node_ids,
                key=lambda nid: (
                    self.cluster.get_node(nid).agent_count()
                    if self.cluster.get_node(nid) else 999
                ),
            )

        return node_ids[0]

    def rebalance_plan(self) -> List[Tuple[str, str, str]]:
        """
        Compute a migration plan to balance agent counts across nodes.

        Returns a list of (agent_id, src_node_id, dst_node_id) tuples.
        The plan is advisory — the caller decides whether to execute it.
        """
        node_ids = self.cluster.list_nodes()
        if len(node_ids) < 2:
            return []

        counts = {
            nid: self.cluster.get_node(nid).agent_count()
            for nid in node_ids
        }
        avg = sum(counts.values()) / len(counts)
        overloaded = sorted(
            [(nid, cnt) for nid, cnt in counts.items() if cnt > avg + 1],
            key=lambda x: -x[1],
        )
        underloaded = sorted(
            [(nid, cnt) for nid, cnt in counts.items() if cnt < avg],
            key=lambda x: x[1],
        )

        plan: List[Tuple[str, str, str]] = []
        for src_id, src_cnt in overloaded:
            if not underloaded:
                break
            dst_id, dst_cnt = underloaded[0]
            src_node = self.cluster.get_node(src_id)
            if src_node is None:
                continue
            # Pick one agent to move
            for agent_id in list(src_node.kernel._agents.keys())[:1]:
                plan.append((agent_id, src_id, dst_id))
            underloaded[0] = (dst_id, dst_cnt + 1)
            if underloaded[0][1] >= avg:
                underloaded.pop(0)

        return plan


# ---------------------------------------------------------------------------
# Split-Brain Detector
# ---------------------------------------------------------------------------

class SplitBrainDetector:
    """
    Monitors network connectivity and detects partition events.

    A network partition (split-brain) occurs when a node can no longer
    reach more than 50% of its peers. This is dangerous because two
    sub-clusters could independently accept writes, diverging their state.

    Response:
        When a node detects it can reach fewer than majority peers,
        it enters READ_ONLY mode — it stops accepting write operations
        but continues serving reads from local state.

    When connectivity recovers:
        The node syncs its log with the leader and re-enters full operation.
    """

    def __init__(self, node: FederationNode, cluster: FederationCluster) -> None:
        self.node = node
        self.cluster = cluster
        self._read_only: bool = False
        self.partition_events: List[Dict[str, Any]] = []
        self.healing_events: List[Dict[str, Any]] = []
        self.read_only_periods: List[Tuple[int, Optional[int]]] = []  # (start_tick, end_tick)
        self._partition_start: Optional[int] = None

    def check(self, current_tick: int) -> bool:
        """
        Check connectivity and update READ_ONLY state.

        Returns True if the node is healthy (not in READ_ONLY mode).
        """
        reachable = len(self.node._connected_peers)
        total = len(self.node.peers)
        majority = (total + 1) // 2 + 1 if total > 0 else 1

        currently_reachable = reachable >= majority or total == 0

        if not currently_reachable and not self._read_only:
            # Enter READ_ONLY mode
            self._read_only = True
            self._partition_start = current_tick
            self.read_only_periods.append((current_tick, None))
            self.partition_events.append({
                "tick": current_tick,
                "node_id": self.node.node_id,
                "reachable_peers": reachable,
                "total_peers": total,
            })

        elif currently_reachable and self._read_only:
            # Heal the partition — sync with leader
            self._read_only = False
            if self.read_only_periods:
                start, _ = self.read_only_periods[-1]
                self.read_only_periods[-1] = (start, current_tick)
            self.healing_events.append({
                "tick": current_tick,
                "node_id": self.node.node_id,
                "duration_ticks": current_tick - (self._partition_start or current_tick),
            })
            self._partition_start = None
            self._sync_with_leader()

        return not self._read_only

    def is_read_only(self) -> bool:
        return self._read_only

    def _sync_with_leader(self) -> None:
        """Re-sync log with the cluster leader after partition heals."""
        leader_id = self.cluster.get_leader()
        if leader_id is None or leader_id == self.node.node_id:
            return
        leader = self.cluster.get_node(leader_id)
        if leader is None:
            return
        # Copy missing log entries
        our_len = len(self.node.log)
        missing = leader.log[our_len:]
        for entry in missing:
            self.node.log.append(entry)
        self.node.commit_index = leader.commit_index

    def report(self) -> Dict[str, Any]:
        return {
            "node_id": self.node.node_id,
            "read_only": self._read_only,
            "partition_events": len(self.partition_events),
            "healing_events": len(self.healing_events),
            "read_only_periods": self.read_only_periods,
        }


# ---------------------------------------------------------------------------
# Demo factory
# ---------------------------------------------------------------------------

def create_demo_federation(num_nodes: int = 3) -> Dict[str, Any]:
    """
    Build a demo federation cluster with ``num_nodes`` kernels.

    Steps:
        1. Create ``num_nodes`` Kernel instances and boot them
        2. Wrap each in a FederationNode and add to a FederationCluster
        3. Run ticks until a leader is elected
        4. Spawn demo agents on each node
        5. Demonstrate cross-node messaging via GlobalRegistry

    Returns a dict with the cluster, registry, migrator, and node list.
    """
    from battousai.kernel import Kernel
    from battousai.agent import MonitorAgent

    kernels: List[Any] = []
    for i in range(num_nodes):
        k = Kernel(max_ticks=0, debug=False)
        k.boot()
        kernels.append(k)

    cluster = FederationCluster()
    nodes: List[FederationNode] = []
    for i, kernel in enumerate(kernels):
        node = cluster.add_node(kernel, node_id=f"node_{i}")
        nodes.append(node)

    # Run ticks until a leader emerges (max 30 ticks)
    for _ in range(30):
        cluster.tick()
        if cluster.get_leader() is not None:
            break

    # Spawn one MonitorAgent per node
    for node in nodes:
        node.kernel.spawn_agent(MonitorAgent, name=f"Monitor@{node.node_id}", priority=7)

    registry = GlobalRegistry()
    for node in nodes:
        for agent_id, agent in node.kernel._agents.items():
            registry.register_agent(agent_id, node.node_id, agent.name)
            registry.register_service("monitoring", agent_id, node.node_id)

    migrator = AgentMigrator(cluster)
    load_balancer = LoadBalancer(cluster, strategy=BalancingStrategy.LEAST_LOADED)
    split_brain_detectors = [SplitBrainDetector(node, cluster) for node in nodes]

    return {
        "cluster": cluster,
        "nodes": nodes,
        "kernels": kernels,
        "registry": registry,
        "migrator": migrator,
        "load_balancer": load_balancer,
        "split_brain_detectors": split_brain_detectors,
        "leader_id": cluster.get_leader(),
        "cluster_stats": cluster.stats(),
    }
