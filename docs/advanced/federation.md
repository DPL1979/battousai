# Federation

The `federation.py` module enables multiple Battousai kernel instances to form a federated cluster. It provides Raft-inspired consensus, leader election, agent migration, and load balancing.

---

## Architecture

```
FederationCluster
├── FederationNode (kernel_a, node_id="node_a")
│   └── ConsensusProtocol (role=LEADER)
├── FederationNode (kernel_b, node_id="node_b")
│   └── ConsensusProtocol (role=FOLLOWER)
└── FederationNode (kernel_c, node_id="node_c")
    └── ConsensusProtocol (role=FOLLOWER)

Shared:
├── GlobalRegistry  — cluster-wide agent and service directory
├── LoadBalancer    — distributes spawns across nodes
└── SplitBrainDetector — partition detection and read-only mode
```

---

## `NodeRole` Enum

```python
class NodeRole(Enum):
    LEADER    = auto()  # Receives all writes, replicates to followers
    FOLLOWER  = auto()  # Replicates from leader, votes in elections
    CANDIDATE = auto()  # Soliciting votes; transitions to LEADER or FOLLOWER
```

---

## `BalancingStrategy` Enum

```python
class BalancingStrategy(Enum):
    ROUND_ROBIN  = auto()  # Rotate through nodes in order
    LEAST_LOADED = auto()  # Pick node with fewest running agents
    RANDOM       = auto()  # Random node selection
    AFFINITY     = auto()  # Keep related agents on the same node
```

---

## `ClusterEntry` — Raft Log

```python
@dataclass
class ClusterEntry:
    term: int       # Election term when entry was created
    index: int      # Monotonically increasing log position
    command: str    # "spawn", "kill", "message", "register"
    data: Any       # Command-specific payload
    leader_id: str  # Node that created this entry
```

Log entries are committed only after a majority of nodes acknowledge them.

---

## Setting Up a Federation

```python
from battousai.kernel import Kernel
from battousai.federation import FederationCluster, FederationNode, BalancingStrategy

# Create kernels
kernel_a = Kernel(max_ticks=100)
kernel_b = Kernel(max_ticks=100)
kernel_c = Kernel(max_ticks=100)
kernel_a.boot()
kernel_b.boot()
kernel_c.boot()

# Create federation
cluster = FederationCluster(strategy=BalancingStrategy.LEAST_LOADED)

node_a = FederationNode(kernel=kernel_a, node_id="node_a")
node_b = FederationNode(kernel=kernel_b, node_id="node_b")
node_c = FederationNode(kernel=kernel_c, node_id="node_c")

cluster.add_node(node_a)
cluster.add_node(node_b)
cluster.add_node(node_c)

cluster.start()  # triggers leader election
```

---

## `FederationCluster` API

```python
class FederationCluster:
    def __init__(self, strategy: BalancingStrategy = BalancingStrategy.ROUND_ROBIN) -> None

    def add_node(self, node: FederationNode) -> None
    def remove_node(self, node_id: str) -> bool
    def start(self) -> None           # triggers leader election
    def stop(self) -> None            # graceful shutdown

    # Agent management
    def spawn_agent(self, agent_class, name, priority=5, **kwargs) -> str
    def kill_agent(self, agent_id: str) -> bool
    def migrate_agent(self, agent_id: str, target_node: str) -> bool

    # Cluster state
    def leader(self) -> Optional[str]       # node_id of current leader
    def nodes(self) -> List[str]            # all node_ids in cluster
    def node_load(self) -> Dict[str, int]   # node_id → agent count
    def registry(self) -> GlobalRegistry
    def stats(self) -> Dict[str, Any]
```

---

## Raft-Inspired Consensus

The consensus protocol ensures all cluster state changes are agreed upon by a majority:

```
Log Entry Lifecycle:
  1. Client sends command to LEADER
  2. LEADER appends to local log
  3. LEADER sends AppendEntries RPC to all FOLLOWERs
  4. FOLLOWER acknowledges receipt
  5. Once majority ACK → entry is COMMITTED
  6. LEADER applies entry to state machine
  7. LEADER notifies FOLLOWERs to apply
```

### Leader Election

When a follower's election timeout expires without a heartbeat:

1. Node increments its term → transitions to `CANDIDATE`
2. Broadcasts `RequestVote` to all peers
3. Each peer votes for the first candidate it sees per term (first-come-first-served)
4. If majority votes received → becomes `LEADER`
5. If another leader's heartbeat arrives → reverts to `FOLLOWER`

```python
from battousai.federation import FederationNode, ConsensusProtocol

node = FederationNode(kernel=kernel_a, node_id="node_a")
# Access the consensus state
protocol = node.consensus
print(protocol.role)          # NodeRole.LEADER or NodeRole.FOLLOWER
print(protocol.current_term)  # current election term
print(protocol.voted_for)     # node_id this node voted for in current term
```

---

## `FederationNode` API

```python
class FederationNode:
    def __init__(self, kernel: Kernel, node_id: str) -> None

    def spawn_agent(self, agent_class, name, priority=5, **kwargs) -> str
    def kill_agent(self, agent_id: str) -> bool
    def agent_count(self) -> int
    def running_agents(self) -> List[str]
    def is_leader(self) -> bool
    def node_info(self) -> Dict[str, Any]
```

---

## `GlobalRegistry`

Cluster-wide directory of agents and services:

```python
from battousai.federation import GlobalRegistry

registry = GlobalRegistry()

# Register an agent's location
registry.register_agent("agent_007", node_id="node_b", agent_class="WorkerAgent")

# Look up where an agent lives
location = registry.find_agent("agent_007")
# {"node_id": "node_b", "agent_class": "WorkerAgent", "registered_at": tick}

# Register a service
registry.register_service("embedding_svc", node_id="node_a", agent_id="embedder_0001")

# Look up a service
svc = registry.find_service("embedding_svc")
# {"node_id": "node_a", "agent_id": "embedder_0001"}

# List all agents
agents = registry.all_agents()  # Dict[agent_id → location_info]
```

---

## Load Balancing Strategies

| Strategy | Algorithm | Best For |
|---|---|---|
| `ROUND_ROBIN` | Cycle through nodes in registration order | Equal distribution regardless of load |
| `LEAST_LOADED` | Pick node with fewest running agents | Balancing computational load |
| `RANDOM` | Random selection | Avoiding coordination overhead |
| `AFFINITY` | Keep related agents on the same node | Reducing cross-node IPC latency |

```python
# Spawn with load balancing
agent_id = cluster.spawn_agent(WorkerAgent, name="DistWorker", priority=4)
# Cluster picks node based on configured strategy

# Migrate to a specific node
cluster.migrate_agent(agent_id, target_node="node_c")
```

---

## Agent Migration

```python
from battousai.federation import AgentMigrator, MigrationStatus

migrator = AgentMigrator()
status = migrator.migrate(
    agent_id="worker_0003",
    source_kernel=kernel_a,
    target_kernel=kernel_b,
    source_node="node_a",
    target_node="node_b",
)

if status == MigrationStatus.COMPLETED:
    print("Migration successful")
elif status == MigrationStatus.FAILED:
    print("Migration failed — agent remains on source")
```

Migration steps:
1. `SERIALIZING` — snapshot agent memory and inbox
2. `TRANSFERRING` — send state to target kernel
3. `DESERIALIZING` — reconstruct agent on target
4. `COMPLETED` or `FAILED`

At-most-once semantics: the agent is never cloned. If any step fails, the agent stays on the source kernel and a rollback occurs.

---

## Split-Brain Detection

If a network partition divides the cluster into two groups both believing they are the majority:

```python
from battousai.federation import SplitBrainDetector

detector = SplitBrainDetector(cluster)
result = detector.check()
# {"partitioned": False, "minority_nodes": [], "majority_nodes": ["node_a", "node_b", "node_c"]}

# If partition detected:
# {"partitioned": True, "minority_nodes": ["node_c"], "majority_nodes": ["node_a", "node_b"]}
```

Minority partition nodes enter **read-only mode** until the partition heals. They can still serve reads but reject all writes until they reconnect to the majority.

---

## Example: Three-Node Cluster

```python
from battousai.kernel import Kernel
from battousai.federation import FederationCluster, FederationNode, BalancingStrategy
from battousai.agent import WorkerAgent, MonitorAgent

# Boot three kernels
kernels = [Kernel(max_ticks=50) for _ in range(3)]
for k in kernels:
    k.boot()

# Create cluster
cluster = FederationCluster(strategy=BalancingStrategy.LEAST_LOADED)
nodes = [
    FederationNode(kernel=kernels[i], node_id=f"node_{i}")
    for i in range(3)
]
for node in nodes:
    cluster.add_node(node)

cluster.start()

# Print leader
print(f"Leader: {cluster.leader()}")

# Spawn an agent on the least-loaded node
worker_id = cluster.spawn_agent(WorkerAgent, name="DistWorker", priority=4)
print(f"Spawned {worker_id} on: {cluster.registry().find_agent(worker_id)['node_id']}")

# Migrate to node_2
cluster.migrate_agent(worker_id, target_node="node_2")

# Load report
for node_id, count in cluster.node_load().items():
    print(f"  {node_id}: {count} agents")
```

---

## Related Pages

- [Networking](networking.md) — the lower-level packet transport layer
- [Architecture Overview](../architecture/overview.md) — federation in the network layer
