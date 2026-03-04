# Networking

The `network.py` module enables communication between agents on different Battousai kernel instances. The prototype uses in-process simulation; swapping `VirtualWire` for a real TCP transport is the intended upgrade path.

---

## Architecture

```
Kernel A                              Kernel B
┌──────────────────────┐              ┌──────────────────────┐
│ NetworkInterface     │              │ NetworkInterface     │
│   node_id="node_a"  │◄─ VirtualWire─►   node_id="node_b"  │
│                      │              │                      │
│  RemoteProxy         │              │  agent_007           │
│  (for agent_007)     │─ Packet ────►│  (real agent)        │
└──────────────────────┘              └──────────────────────┘
         │                                      │
         └──── NetworkTopology ─────────────────┘
                (graph of nodes + links)
```

---

## `PacketType` Enum

```python
class PacketType(Enum):
    AGENT_MESSAGE = auto()  # Forward an IPC message to a remote agent
    DISCOVERY     = auto()  # Announce or query available services
    HEARTBEAT     = auto()  # Liveness check between nodes
    MIGRATION     = auto()  # Serialize and transfer an agent to another kernel
    GOSSIP        = auto()  # State propagation for eventual consistency
    SYNC          = auto()  # Request full state sync between nodes
```

---

## `Packet` Dataclass

```python
@dataclass
class Packet:
    src_node: str           # Source kernel node_id
    dst_node: str           # Destination kernel node_id (or "*" for broadcast)
    src_agent: str          # Agent_id on source kernel
    dst_agent: str          # Agent_id on destination kernel
    packet_type: PacketType
    payload: Any            # Arbitrary serialisable content
    hop_count: int = 0      # Incremented by each router (loop detection)
    ttl: int = 10           # Max hops before drop
    sequence_number: int = 0
    checksum: str = ""      # SHA-256 truncated hash of payload
    packet_id: str = ...    # Auto-generated UUID4 short form
```

---

## `VirtualWire`

A simulated network link between two nodes:

```python
from battousai.network import VirtualWire

wire = VirtualWire(
    latency_ticks=1,        # Packets arrive after N ticks
    packet_loss_rate=0.0,   # 0.0 = no loss, 0.1 = 10% loss
    bandwidth_limit=0,      # 0 = unlimited packets per tick
)
```

The `VirtualWire` queues packets and delivers them after `latency_ticks`. Setting `packet_loss_rate > 0` introduces probabilistic loss for testing resilience.

---

## `NetworkTopology`

Graph of kernel nodes and their connections:

```python
from battousai.network import NetworkTopology, VirtualWire

topology = NetworkTopology()

# Add a bidirectional link
wire = VirtualWire(latency_ticks=1)
topology.connect("node_a", "node_b", wire)
topology.connect("node_b", "node_c", VirtualWire(latency_ticks=2))

# Check connectivity
neighbors = topology.neighbors("node_a")
# ["node_b"]

# Get the wire between two nodes
wire = topology.get_wire("node_a", "node_b")
```

---

## `NetworkInterface`

Attaches a kernel to the network topology:

```python
from battousai.network import NetworkInterface

iface_a = NetworkInterface(
    kernel=kernel_a,
    node_id="node_a",
    topology=topology,
)

iface_b = NetworkInterface(
    kernel=kernel_b,
    node_id="node_b",
    topology=topology,
)
```

---

## Setting Up a Multi-Node Network

```python
from battousai.kernel import Kernel
from battousai.network import NetworkTopology, VirtualWire, NetworkInterface

# Create two kernels
kernel_a = Kernel(max_ticks=50)
kernel_b = Kernel(max_ticks=50)
kernel_a.boot()
kernel_b.boot()

# Connect with a simulated wire (1 tick latency, no loss)
topology = NetworkTopology()
wire = VirtualWire(latency_ticks=1, packet_loss_rate=0.0)
topology.connect("node_a", "node_b", wire)

iface_a = NetworkInterface(kernel=kernel_a, node_id="node_a", topology=topology)
iface_b = NetworkInterface(kernel=kernel_b, node_id="node_b", topology=topology)

# Now agents on kernel_a can send messages to agents on kernel_b
```

---

## Gossip Protocol

The gossip protocol propagates state changes (agent directories, service registrations, config updates) across the cluster without a central coordinator.

```
Each tick, a node fans out to a random subset of its neighbors.
Each neighbor fans out to its own random subset.
Convergence: O(log N) ticks for N nodes.
```

```python
from battousai.network import GossipProtocol

gossip = GossipProtocol(
    node_id="node_a",
    fan_out=3,           # forward to 3 random neighbors per round
    max_ttl=5,           # messages expire after 5 hops
)

# Push a state update
gossip.push(key="agent_directory", value={"agent_007": "node_a"})

# Pull current state for a key
state = gossip.read("agent_directory")
```

---

## Service Discovery

Agents advertise services; consumers find them by name:

```python
from battousai.network import ServiceDiscovery

discovery = ServiceDiscovery(node_id="node_a")

# Advertise a service
discovery.register(
    service_name="embedding_service",
    agent_id="embedder_0001",
    node_id="node_a",
    metadata={"model": "text-embedding-3-small", "dims": 1536},
)

# Find a service (queries local cache first, propagated via gossip)
result = discovery.lookup("embedding_service")
# {"agent_id": "embedder_0001", "node_id": "node_a", "latency_ms": 2, "metadata": {...}}

# Unregister when the agent terminates
discovery.unregister("embedding_service", "embedder_0001")
```

---

## Agent Migration

Agent migration uses **at-most-once semantics** — an agent is never cloned:

```
1. Source kernel serialises agent state (memory snapshot + inbox snapshot)
2. Serialised state transmitted as a MIGRATION packet
3. Destination kernel deserialises and spawns the agent
4. Source kernel marks agent as migrated (not killed until ACK)
5. On failure at any step → agent stays on source kernel
```

```python
# Trigger migration via network interface
iface_a.migrate_agent(
    agent_id="worker_0003",
    target_node="node_b",
)
```

The `MIGRATION` packet carries:
- Agent class name
- Serialised memory space (`space.snapshot()`)
- Serialised inbox contents
- Agent priority and configuration

---

## `RemoteProxy`

A `RemoteProxy` is a local placeholder for an agent that lives on a remote kernel. It forwards messages via the network instead of delivering them locally:

```python
from battousai.network import RemoteProxy

# Automatically created when an AGENT_MESSAGE arrives for a known remote agent
proxy = RemoteProxy(
    remote_agent_id="agent_007",
    remote_node_id="node_b",
    local_interface=iface_a,
)

# Sending to a remote proxy looks identical to sending to a local agent
proxy.send(message)
```

---

## Packet Routing

| Destination | Strategy |
|---|---|
| Direct neighbor | Send directly via `VirtualWire` |
| Non-neighbor | Simple flooding with TTL decrement |
| Broadcast (`dst_node="*"`) | Deliver to all connected nodes |
| Gossip | Probabilistic fan-out to random subset of neighbors |

---

## Related Pages

- [Federation](federation.md) — multi-kernel consensus and load balancing
- [Architecture Overview](../architecture/overview.md) — network layer in context
- [Agent API](../agents/api.md) — agents send/receive messages the same way locally and remotely
