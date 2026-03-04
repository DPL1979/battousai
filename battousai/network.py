"""
network.py — Distributed Agent Network Stack
================================================
Enables communication between agents on different Battousai kernel instances.

Architecture:
    NetworkInterface   — a kernel's network adapter
    Packet             — unit of transmission between kernels
    VirtualWire        — simulated network link between two kernels
    NetworkTopology    — graph of kernel nodes and their connections
    RemoteProxy        — local proxy for a remote agent
    GossipProtocol     — epidemic-style state propagation
    ServiceDiscovery   — agents advertise and discover services across kernels

Protocol:
    Each kernel has a unique node_id. Packets carry:
        src_node, dst_node, src_agent, dst_agent, packet_type, payload,
        hop_count, ttl

    Supported packet types:
        AGENT_MESSAGE — forward an IPC message to a remote agent
        DISCOVERY     — announce/query available services
        HEARTBEAT     — liveness check between nodes
        MIGRATION     — serialize and transfer an agent to another kernel
        GOSSIP        — state propagation for eventual consistency
        SYNC          — request full state sync between nodes

    Routing:
        Direct routing: if dst_node is a neighbor, send directly
        Multi-hop: simple flooding with TTL for discovery
        Gossip: probabilistic fan-out to random subset of neighbors

Design note:
    Because this is a pure-Python prototype we simulate the network layer
    entirely in-process.  VirtualWire queues packets and applies latency
    and packet-loss rules deterministically so the simulation is
    reproducible.  Swapping VirtualWire for a real asyncio TCP transport
    is the intended upgrade path.
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from battousai.kernel import Kernel

from battousai.agent import Agent, SyscallResult


# ---------------------------------------------------------------------------
# Packet types
# ---------------------------------------------------------------------------

class PacketType(Enum):
    """Semantic classification of a network packet."""
    AGENT_MESSAGE = auto()  # Forward an IPC message to a remote agent
    DISCOVERY     = auto()  # Announce or query available services
    HEARTBEAT     = auto()  # Liveness check between nodes
    MIGRATION     = auto()  # Serialize and transfer an agent to another kernel
    GOSSIP        = auto()  # State propagation for eventual consistency
    SYNC          = auto()  # Request full state sync between nodes


# ---------------------------------------------------------------------------
# Packet dataclass
# ---------------------------------------------------------------------------

@dataclass
class Packet:
    """
    The fundamental unit of transmission between Battousai kernel nodes.

    Fields
    ------
    src_node       : node_id of the originating kernel
    dst_node       : node_id of the destination kernel (or "*" for broadcast)
    src_agent      : agent_id on the source kernel that sent this packet
    dst_agent      : agent_id on the destination kernel that should receive it
    packet_type    : semantic classification (PacketType)
    payload        : arbitrary serialisable content
    hop_count      : incremented by each router; used to detect routing loops
    ttl            : maximum hops allowed before the packet is dropped
    sequence_number: monotonic counter per (src_node, src_agent) pair
    checksum       : SHA-256 truncated hash of the serialised payload,
                     used for integrity verification
    packet_id      : globally unique identifier (UUID4 short form)
    """
    src_node: str
    dst_node: str
    src_agent: str
    dst_agent: str
    packet_type: PacketType
    payload: Any
    hop_count: int = 0
    ttl: int = 8
    sequence_number: int = 0
    checksum: str = field(default="")
    packet_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    def __post_init__(self) -> None:
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute a short SHA-256 hash of the payload for integrity checks."""
        raw = json.dumps(self.payload, default=str, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def verify_checksum(self) -> bool:
        """Return True if the payload has not been corrupted in transit."""
        return self.checksum == self._compute_checksum()

    def is_expired(self) -> bool:
        """Return True if the packet has exceeded its maximum hop count."""
        return self.hop_count >= self.ttl

    def __repr__(self) -> str:
        return (
            f"Packet(id={self.packet_id}, type={self.packet_type.name}, "
            f"{self.src_node}/{self.src_agent} → {self.dst_node}/{self.dst_agent}, "
            f"hops={self.hop_count}/{self.ttl})"
        )


# ---------------------------------------------------------------------------
# VirtualWire
# ---------------------------------------------------------------------------

class VirtualWire:
    """
    Simulated point-to-point network link between two NetworkInterface nodes.

    Models real-world link imperfections:
        latency_ticks      — minimum number of ticks before a packet is
                             deliverable at the far end.  Simulates propagation
                             delay.
        bandwidth          — maximum packets deliverable per tick (0 = infinite).
        packet_loss_rate   — probability in [0.0, 1.0] that any given packet
                             is silently dropped, simulating unreliable networks.

    Internal state
    --------------
    _in_flight : Deque[(deliver_at_tick, Packet)]
        Packets are enqueued with their computed delivery tick.  The
        ``tick()`` method returns packets whose delivery tick has arrived.
    """

    def __init__(
        self,
        node_a: str,
        node_b: str,
        latency_ticks: int = 1,
        bandwidth: int = 0,
        packet_loss_rate: float = 0.0,
    ) -> None:
        self.node_a = node_a
        self.node_b = node_b
        self.latency_ticks = max(0, latency_ticks)
        self.bandwidth = bandwidth          # 0 = unlimited
        self.packet_loss_rate = max(0.0, min(1.0, packet_loss_rate))
        # Queue of (deliver_at_tick, packet) tuples
        self._in_flight: Deque[Tuple[int, Packet]] = deque()
        self.packets_sent: int = 0
        self.packets_dropped: int = 0
        self.packets_delivered: int = 0

    def transmit(self, packet: Packet, current_tick: int) -> bool:
        """
        Enqueue a packet for delivery after ``latency_ticks``.

        Returns True if accepted, False if dropped (loss simulation).
        """
        if random.random() < self.packet_loss_rate:
            self.packets_dropped += 1
            return False
        deliver_at = current_tick + self.latency_ticks
        self._in_flight.append((deliver_at, packet))
        self.packets_sent += 1
        return True

    def tick(self, current_tick: int) -> List[Packet]:
        """
        Return all packets whose delivery tick has arrived.

        Respects ``bandwidth`` limit: at most ``bandwidth`` packets are
        returned per tick (FIFO).  Remaining packets stay in the queue.
        """
        ready: List[Packet] = []
        remaining: Deque[Tuple[int, Packet]] = deque()
        delivered_this_tick = 0

        for deliver_at, pkt in self._in_flight:
            if deliver_at <= current_tick:
                if self.bandwidth == 0 or delivered_this_tick < self.bandwidth:
                    ready.append(pkt)
                    delivered_this_tick += 1
                    self.packets_delivered += 1
                else:
                    remaining.append((deliver_at, pkt))
            else:
                remaining.append((deliver_at, pkt))

        self._in_flight = remaining
        return ready

    def queue_depth(self) -> int:
        """Return the number of packets currently in flight."""
        return len(self._in_flight)

    def stats(self) -> Dict[str, Any]:
        return {
            "node_a": self.node_a,
            "node_b": self.node_b,
            "latency_ticks": self.latency_ticks,
            "bandwidth": self.bandwidth,
            "packet_loss_rate": self.packet_loss_rate,
            "in_flight": self.queue_depth(),
            "packets_sent": self.packets_sent,
            "packets_dropped": self.packets_dropped,
            "packets_delivered": self.packets_delivered,
        }

    def __repr__(self) -> str:
        return (
            f"VirtualWire({self.node_a!r} ↔ {self.node_b!r}, "
            f"latency={self.latency_ticks}, loss={self.packet_loss_rate:.0%})"
        )


# ---------------------------------------------------------------------------
# NetworkInterface
# ---------------------------------------------------------------------------

class NetworkInterface:
    """
    The network adapter for a single Battousai kernel node.

    Each kernel that participates in the distributed network owns one
    NetworkInterface.  The interface is responsible for:
        - Maintaining a directory of outgoing VirtualWire links
          (keyed by the remote node_id)
        - Sending packets to neighbours (direct) or flooding (multi-hop)
        - Collecting packets from all wires each tick and placing them in
          the inbox for consumption by the owning kernel

    Attributes
    ----------
    node_id       : unique identifier for this node in the network
    neighbors     : dict mapping node_id → VirtualWire for direct links
    outbox        : packets staged for transmission this tick
    inbox         : packets received and waiting to be processed
    _seq_counter  : monotonic per-node sequence counter
    """

    def __init__(self, node_id: str) -> None:
        self.node_id: str = node_id
        self.neighbors: Dict[str, VirtualWire] = {}
        self.outbox: Deque[Packet] = deque()
        self.inbox: Deque[Packet] = deque()
        self._seq_counter: int = 0
        self._seen_packet_ids: Set[str] = set()  # flood-loop prevention
        self.packets_routed: int = 0
        self.packets_received: int = 0

    # ------------------------------------------------------------------
    # Neighbor management
    # ------------------------------------------------------------------

    def add_neighbor(self, node_id: str, wire: VirtualWire) -> None:
        """Register a direct link to another node."""
        self.neighbors[node_id] = wire

    def remove_neighbor(self, node_id: str) -> None:
        """Disconnect from a neighbor."""
        self.neighbors.pop(node_id, None)

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def _next_seq(self) -> int:
        self._seq_counter += 1
        return self._seq_counter

    def send_packet(self, packet: Packet, current_tick: int) -> bool:
        """
        Route a packet toward its destination.

        Direct route: if dst_node is a direct neighbour, use that wire.
        Flooding: otherwise, broadcast to all neighbours (TTL guards against
        infinite loops).  Each intermediate node forwards on receive.

        Returns True if at least one wire accepted the packet.
        """
        if packet.is_expired():
            return False

        packet.sequence_number = self._next_seq()
        wire = self.neighbors.get(packet.dst_node)
        if wire is not None:
            # Direct delivery
            ok = wire.transmit(packet, current_tick)
            if ok:
                self.packets_routed += 1
            return ok

        # Multi-hop: flood to all neighbours, let TTL prevent loops
        sent = 0
        for neighbor_id, w in self.neighbors.items():
            if neighbor_id == packet.src_node:
                continue  # don't send back to origin
            forwarded = Packet(
                src_node=packet.src_node,
                dst_node=packet.dst_node,
                src_agent=packet.src_agent,
                dst_agent=packet.dst_agent,
                packet_type=packet.packet_type,
                payload=packet.payload,
                hop_count=packet.hop_count + 1,
                ttl=packet.ttl,
                sequence_number=packet.sequence_number,
                checksum=packet.checksum,
                packet_id=packet.packet_id,
            )
            if w.transmit(forwarded, current_tick):
                sent += 1
        if sent:
            self.packets_routed += sent
        return sent > 0

    def broadcast(self, packet: Packet, current_tick: int) -> int:
        """
        Send a packet to ALL direct neighbours unconditionally.

        Useful for HEARTBEAT, GOSSIP, and DISCOVERY floods.
        Returns the number of wires that accepted the packet.
        """
        sent = 0
        for neighbor_id, wire in self.neighbors.items():
            p = Packet(
                src_node=self.node_id,
                dst_node=neighbor_id,
                src_agent=packet.src_agent,
                dst_agent=packet.dst_agent,
                packet_type=packet.packet_type,
                payload=packet.payload,
                hop_count=packet.hop_count,
                ttl=packet.ttl,
                sequence_number=self._next_seq(),
                packet_id=packet.packet_id,
            )
            if wire.transmit(p, current_tick):
                sent += 1
        return sent

    # ------------------------------------------------------------------
    # Receiving
    # ------------------------------------------------------------------

    def receive_packets(self, current_tick: int) -> List[Packet]:
        """
        Collect all packets that have arrived from every connected wire.

        Deduplicates by packet_id to handle flooding convergence.
        Places new, non-duplicate packets into ``self.inbox``.
        Returns the list of newly received packets.
        """
        received: List[Packet] = []
        for wire in self.neighbors.values():
            for pkt in wire.tick(current_tick):
                if pkt.packet_id not in self._seen_packet_ids:
                    self._seen_packet_ids.add(pkt.packet_id)
                    self.inbox.append(pkt)
                    received.append(pkt)
                    self.packets_received += 1
        return received

    def drain_inbox(self) -> List[Packet]:
        """Return and clear all packets currently in the inbox."""
        packets = list(self.inbox)
        self.inbox.clear()
        return packets

    def stats(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "neighbors": list(self.neighbors.keys()),
            "inbox_depth": len(self.inbox),
            "outbox_depth": len(self.outbox),
            "packets_routed": self.packets_routed,
            "packets_received": self.packets_received,
        }


# ---------------------------------------------------------------------------
# NetworkTopology
# ---------------------------------------------------------------------------

class NetworkTopology:
    """
    Graph of all kernel nodes and their VirtualWire connections.

    Provides:
        add_node      — register a NetworkInterface in the topology
        add_link      — create a bidirectional VirtualWire between two nodes
        remove_node   — detach a node from the topology
        get_neighbors — return the direct neighbours of a node
        shortest_path — BFS shortest path (by hop count) between two nodes

    This class serves as the "routing table" for the distributed system.
    In production, this information would be distributed; here it is
    centralised for simplicity.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, NetworkInterface] = {}
        # Adjacency: node_id → set of neighbour node_ids
        self._adjacency: Dict[str, Set[str]] = {}
        # Wire registry: frozenset({a, b}) → VirtualWire
        self._wires: Dict[frozenset, VirtualWire] = {}

    def add_node(self, interface: NetworkInterface) -> None:
        """Register a NetworkInterface node in the topology."""
        nid = interface.node_id
        self._nodes[nid] = interface
        self._adjacency.setdefault(nid, set())

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node and all of its links from the topology.

        Also disconnects the node's wires from all still-connected neighbours.
        """
        iface = self._nodes.pop(node_id, None)
        if iface is None:
            return
        # Remove all wires touching this node
        for neighbor_id in list(self._adjacency.get(node_id, [])):
            self._adjacency.get(neighbor_id, set()).discard(node_id)
            self._wires.pop(frozenset({node_id, neighbor_id}), None)
            # Tell the neighbour to disconnect
            ni = self._nodes.get(neighbor_id)
            if ni:
                ni.remove_neighbor(node_id)
        self._adjacency.pop(node_id, None)

    def add_link(
        self,
        node_a: str,
        node_b: str,
        latency_ticks: int = 1,
        bandwidth: int = 0,
        packet_loss_rate: float = 0.0,
    ) -> VirtualWire:
        """
        Create a bidirectional VirtualWire between node_a and node_b.

        Both NetworkInterface objects are updated to know about each other.
        Returns the new VirtualWire (shared by both directions).
        """
        if node_a not in self._nodes:
            raise ValueError(f"Node {node_a!r} is not registered in the topology.")
        if node_b not in self._nodes:
            raise ValueError(f"Node {node_b!r} is not registered in the topology.")

        wire = VirtualWire(node_a, node_b, latency_ticks, bandwidth, packet_loss_rate)
        key = frozenset({node_a, node_b})
        self._wires[key] = wire

        self._nodes[node_a].add_neighbor(node_b, wire)
        self._nodes[node_b].add_neighbor(node_a, wire)
        self._adjacency[node_a].add(node_b)
        self._adjacency[node_b].add(node_a)
        return wire

    def get_neighbors(self, node_id: str) -> List[str]:
        """Return the node_ids of all directly connected neighbours."""
        return list(self._adjacency.get(node_id, set()))

    def shortest_path(self, src: str, dst: str) -> List[str]:
        """
        Compute the shortest path from src to dst by BFS hop count.

        Returns a list of node_ids [src, ..., dst], or an empty list if no
        path exists.
        """
        if src not in self._adjacency or dst not in self._adjacency:
            return []
        if src == dst:
            return [src]
        visited: Set[str] = {src}
        queue: Deque[List[str]] = deque([[src]])
        while queue:
            path = queue.popleft()
            current = path[-1]
            for neighbor in self._adjacency.get(current, set()):
                if neighbor == dst:
                    return path + [dst]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return []  # no path

    def all_nodes(self) -> List[str]:
        """Return all registered node_ids."""
        return list(self._nodes.keys())

    def get_interface(self, node_id: str) -> Optional[NetworkInterface]:
        return self._nodes.get(node_id)

    def wire_stats(self) -> Dict[str, Any]:
        """Return statistics for all wires in the topology."""
        return {
            str(sorted(key)): wire.stats()
            for key, wire in self._wires.items()
        }

    def __repr__(self) -> str:
        edges = [f"{sorted(k)}" for k in self._wires]
        return f"NetworkTopology(nodes={self.all_nodes()}, edges={edges})"


# ---------------------------------------------------------------------------
# GossipProtocol
# ---------------------------------------------------------------------------

class GossipProtocol:
    """
    Epidemic-style state propagation for eventual consistency across nodes.

    How it works:
        1. Each node maintains a *state store*: a dict of
           ``key → (value, vector_clock_tick)`` pairs.
        2. Every ``gossip_interval`` ticks, the node selects
           ``fanout`` random neighbours and sends them a *digest*
           (key → clock_tick) via a GOSSIP packet.
        3. The recipient compares the digest to its own store and sends back
           any entries it has that the sender is missing or that are newer.
        4. This converges to a global consistent state in O(log N) rounds.

    The implementation is simplified for the prototype: the full value is
    sent in the digest rather than a separate anti-entropy pull round.

    Attributes
    ----------
    node_id         : the owning node
    state           : dict of key → (value, tick)  — the gossip state store
    fanout          : number of random neighbours contacted per gossip round
    gossip_interval : ticks between gossip rounds
    rounds          : total gossip rounds executed
    """

    def __init__(
        self,
        node_id: str,
        fanout: int = 2,
        gossip_interval: int = 3,
    ) -> None:
        self.node_id = node_id
        self.state: Dict[str, Tuple[Any, int]] = {}   # key → (value, tick)
        self.fanout = fanout
        self.gossip_interval = gossip_interval
        self.rounds: int = 0
        self._last_gossip_tick: int = 0

    def set(self, key: str, value: Any, tick: int) -> None:
        """
        Write a key-value pair into the local gossip state.

        Only updates if the provided ``tick`` is newer than the stored one.
        """
        existing = self.state.get(key)
        if existing is None or tick > existing[1]:
            self.state[key] = (value, tick)

    def get(self, key: str) -> Optional[Any]:
        """Return the current value for *key*, or None if not present."""
        entry = self.state.get(key)
        return entry[0] if entry else None

    def build_digest(self) -> Dict[str, Tuple[Any, int]]:
        """Return the full state dict (used as gossip payload)."""
        return dict(self.state)

    def merge_digest(self, remote_digest: Dict[str, Tuple[Any, int]]) -> List[str]:
        """
        Merge a gossip digest received from a remote node.

        Applies the last-write-wins rule based on the vector clock tick.
        Returns the list of keys that were updated.
        """
        updated: List[str] = []
        for key, (value, remote_tick) in remote_digest.items():
            local = self.state.get(key)
            if local is None or remote_tick > local[1]:
                self.state[key] = (value, remote_tick)
                updated.append(key)
        return updated

    def should_gossip(self, current_tick: int) -> bool:
        """Return True if it is time to start a new gossip round."""
        return (current_tick - self._last_gossip_tick) >= self.gossip_interval

    def select_gossip_targets(self, neighbors: List[str]) -> List[str]:
        """Randomly select up to ``fanout`` neighbours to gossip with."""
        if len(neighbors) <= self.fanout:
            return list(neighbors)
        return random.sample(neighbors, self.fanout)

    def tick(
        self,
        current_tick: int,
        interface: NetworkInterface,
        topology: NetworkTopology,
    ) -> List[Packet]:
        """
        Execute one gossip tick.

        If it is time to gossip, generate GOSSIP packets for a random
        subset of neighbours and return them for transmission.
        """
        if not self.should_gossip(current_tick):
            return []

        self._last_gossip_tick = current_tick
        self.rounds += 1
        neighbors = topology.get_neighbors(self.node_id)
        targets = self.select_gossip_targets(neighbors)
        digest = self.build_digest()
        packets: List[Packet] = []

        for target in targets:
            pkt = Packet(
                src_node=self.node_id,
                dst_node=target,
                src_agent="gossip",
                dst_agent="gossip",
                packet_type=PacketType.GOSSIP,
                payload=digest,
                ttl=2,
            )
            interface.send_packet(pkt, current_tick)
            packets.append(pkt)

        return packets

    def convergence_score(self, expected_keys: List[str]) -> float:
        """
        Return a convergence score in [0.0, 1.0].

        1.0 means all expected keys are present in the local state.
        """
        if not expected_keys:
            return 1.0
        present = sum(1 for k in expected_keys if k in self.state)
        return present / len(expected_keys)

    def __repr__(self) -> str:
        return (
            f"GossipProtocol(node={self.node_id!r}, "
            f"keys={len(self.state)}, rounds={self.rounds})"
        )


# ---------------------------------------------------------------------------
# ServiceDiscovery
# ---------------------------------------------------------------------------

class ServiceDiscovery:
    """
    Cross-kernel service advertisement and lookup via gossip propagation.

    Agents register services of the form::

        "tool:web_search"
        "capability:research"
        "agent_class:CoordinatorAgent"

    Service records propagate through GossipProtocol so that agents on
    any node can discover providers on any other node.

    Attributes
    ----------
    node_id          : the owning node's id
    gossip           : the GossipProtocol instance used for propagation
    _local_services  : dict of service_name → set of agent_ids (local only)
    """

    # Gossip key prefix for service entries
    _KEY_PREFIX = "svc:"

    def __init__(self, node_id: str, gossip: GossipProtocol) -> None:
        self.node_id = node_id
        self.gossip = gossip
        self._local_services: Dict[str, Set[str]] = {}

    def register(self, service_name: str, agent_id: str, tick: int) -> None:
        """
        Advertise that *agent_id* on this node provides *service_name*.

        The record is written into the gossip state for network-wide
        propagation.
        """
        # Track locally
        self._local_services.setdefault(service_name, set()).add(agent_id)

        # Propagate via gossip: value is a list of (node_id, agent_id) tuples
        gossip_key = f"{self._KEY_PREFIX}{service_name}"
        current: List[Tuple[str, str]] = list(
            self.gossip.get(gossip_key) or []
        )
        entry = (self.node_id, agent_id)
        if entry not in current:
            current.append(entry)
        self.gossip.set(gossip_key, current, tick)

    def deregister(self, service_name: str, agent_id: str, tick: int) -> None:
        """Remove a service advertisement."""
        self._local_services.get(service_name, set()).discard(agent_id)
        gossip_key = f"{self._KEY_PREFIX}{service_name}"
        current: List[Tuple[str, str]] = list(
            self.gossip.get(gossip_key) or []
        )
        updated = [e for e in current if e != (self.node_id, agent_id)]
        self.gossip.set(gossip_key, updated, tick)

    def query(self, service_name: str) -> List[Tuple[str, str]]:
        """
        Return all known providers of *service_name*.

        Returns a list of (node_id, agent_id) tuples.  The list includes
        providers discovered through gossip on remote nodes.
        """
        gossip_key = f"{self._KEY_PREFIX}{service_name}"
        return list(self.gossip.get(gossip_key) or [])

    def list_services(self) -> List[str]:
        """Return all service names known on this node (including gossiped)."""
        prefix_len = len(self._KEY_PREFIX)
        return [
            key[prefix_len:]
            for key in self.gossip.state
            if key.startswith(self._KEY_PREFIX)
        ]

    def find_providers(self, service_name: str) -> List[str]:
        """
        Convenience method: return full remote addresses "node:agent_id"
        for all providers of *service_name*.
        """
        return [
            f"{node}:{agent}"
            for node, agent in self.query(service_name)
        ]

    def __repr__(self) -> str:
        return (
            f"ServiceDiscovery(node={self.node_id!r}, "
            f"local_services={list(self._local_services.keys())})"
        )


# ---------------------------------------------------------------------------
# RemoteAgentProxy
# ---------------------------------------------------------------------------

class RemoteAgentProxy(Agent):
    """
    A local stub that looks like a regular Agent but forwards all
    communication as network packets to a remote agent.

    When the IPC layer detects a recipient_id of the form
    ``"<node_id>:<remote_agent_id>"`` it can instantiate a
    RemoteAgentProxy and route messages through it.

    The proxy translates Agent.send_message() calls into AGENT_MESSAGE
    packets and injects arriving AGENT_MESSAGE packets back as local
    IPC messages.

    Attributes
    ----------
    remote_node     : node_id of the kernel hosting the real agent
    remote_agent_id : agent_id on the remote kernel
    interface       : the local NetworkInterface used to send packets
    """

    def __init__(
        self,
        remote_node: str,
        remote_agent_id: str,
        interface: NetworkInterface,
        local_node: str,
    ) -> None:
        proxy_name = f"proxy:{remote_node}:{remote_agent_id}"
        super().__init__(name=proxy_name, priority=5)
        self.remote_node = remote_node
        self.remote_agent_id = remote_agent_id
        self.interface = interface
        self.local_node = local_node

    def forward_message(self, payload: Any, current_tick: int) -> bool:
        """
        Wrap *payload* in an AGENT_MESSAGE packet and transmit it.

        Returns True if the packet was accepted by the wire.
        """
        pkt = Packet(
            src_node=self.local_node,
            dst_node=self.remote_node,
            src_agent=self.agent_id or "proxy",
            dst_agent=self.remote_agent_id,
            packet_type=PacketType.AGENT_MESSAGE,
            payload=payload,
        )
        return self.interface.send_packet(pkt, current_tick)

    def think(self, tick: int) -> None:
        """
        The proxy has no autonomous behaviour; it is purely reactive.

        Real forwarding logic is driven by the kernel's network tick handler.
        """
        self.yield_cpu()

    def __repr__(self) -> str:
        return (
            f"RemoteAgentProxy({self.local_node}/{self.agent_id} → "
            f"{self.remote_node}/{self.remote_agent_id})"
        )


# ---------------------------------------------------------------------------
# AgentMigration
# ---------------------------------------------------------------------------

class AgentMigration:
    """
    Utilities to serialize an agent's runtime state into a dict and
    restore it on a destination kernel.

    The migration protocol uses MIGRATION packets to carry the agent
    snapshot across the network.  The destination kernel deserialises
    the snapshot and recreates the agent.

    Only serialisable state (memory snapshot, metadata) is transferred.
    The agent's class must be importable on the destination kernel.
    """

    @staticmethod
    def serialize(agent: Agent, current_tick: int) -> Dict[str, Any]:
        """
        Produce a JSON-serialisable snapshot of *agent*'s state.

        Captures:
            - Class name and module (for reconstruction)
            - Agent metadata (name, priority, memory_allocation, time_slice)
            - Memory snapshot (all LONG_TERM and SHORT_TERM entries)
            - Spawn tick and ticks alive
            - Any public instance attributes that are JSON-serialisable
        """
        memory_snapshot: Dict[str, Any] = {}
        if agent.kernel is not None:
            try:
                space = agent.kernel.memory.get_agent_space(agent.agent_id)
                memory_snapshot = space.snapshot()
            except Exception:
                pass

        return {
            "class_name": type(agent).__name__,
            "module": type(agent).__module__,
            "name": agent.name,
            "priority": agent.priority,
            "memory_allocation": agent.memory_allocation,
            "time_slice": agent.time_slice,
            "spawn_tick": agent._spawn_tick,
            "ticks_alive": agent._ticks_alive,
            "memory_snapshot": memory_snapshot,
            "migrated_at_tick": current_tick,
        }

    @staticmethod
    def build_migration_packet(
        agent: Agent,
        src_node: str,
        dst_node: str,
        current_tick: int,
    ) -> Packet:
        """Create a MIGRATION packet carrying the agent's serialised state."""
        snapshot = AgentMigration.serialize(agent, current_tick)
        return Packet(
            src_node=src_node,
            dst_node=dst_node,
            src_agent=agent.agent_id,
            dst_agent="kernel",
            packet_type=PacketType.MIGRATION,
            payload=snapshot,
            ttl=4,
        )

    @staticmethod
    def restore_memory(
        snapshot: Dict[str, Any],
        agent: Agent,
        kernel: "Kernel",
        current_tick: int,
    ) -> None:
        """
        Write the memory snapshot from a MIGRATION payload back into
        the agent's memory space on the destination kernel.
        """
        memory_snapshot: Dict[str, Any] = snapshot.get("memory_snapshot", {})
        for key, value in memory_snapshot.items():
            try:
                from battousai.memory import MemoryType
                kernel.memory.agent_write(
                    agent.agent_id, key, value,
                    MemoryType.LONG_TERM, current_tick
                )
            except Exception:
                pass  # Non-fatal: partial restore is acceptable


# ---------------------------------------------------------------------------
# Demo helper
# ---------------------------------------------------------------------------

def create_demo_network(num_nodes: int = 3) -> Dict[str, Any]:
    """
    Create a demo network of *num_nodes* Battousai kernels connected in a ring.

    Each adjacent pair of nodes is connected by a VirtualWire with
    simulated latency.  Gossip and ServiceDiscovery are initialised for
    every node.

    Returns a dict with keys:
        "topology"   : NetworkTopology
        "interfaces" : dict of node_id → NetworkInterface
        "gossip"     : dict of node_id → GossipProtocol
        "discovery"  : dict of node_id → ServiceDiscovery

    Usage::

        net = create_demo_network(num_nodes=3)
        topo = net["topology"]
        ifaces = net["interfaces"]
        # Tick the wires manually:
        for nid, iface in ifaces.items():
            packets = iface.receive_packets(current_tick=1)
    """
    if num_nodes < 2:
        raise ValueError("A demo network requires at least 2 nodes.")

    node_ids = [f"node{i}" for i in range(1, num_nodes + 1)]
    topology = NetworkTopology()
    interfaces: Dict[str, NetworkInterface] = {}
    gossip_protocols: Dict[str, GossipProtocol] = {}
    discovery_services: Dict[str, ServiceDiscovery] = {}

    # Create interfaces and register them in the topology
    for nid in node_ids:
        iface = NetworkInterface(nid)
        topology.add_node(iface)
        interfaces[nid] = iface
        gp = GossipProtocol(node_id=nid, fanout=2, gossip_interval=3)
        gossip_protocols[nid] = gp
        discovery_services[nid] = ServiceDiscovery(nid, gp)

    # Connect in a ring: node1–node2–node3–…–nodeN–node1
    for i in range(num_nodes):
        a = node_ids[i]
        b = node_ids[(i + 1) % num_nodes]
        topology.add_link(a, b, latency_ticks=1, packet_loss_rate=0.0)

    return {
        "topology": topology,
        "interfaces": interfaces,
        "gossip": gossip_protocols,
        "discovery": discovery_services,
    }
