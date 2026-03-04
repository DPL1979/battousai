# Changelog

All notable changes to Battousai are documented here.

---

## [0.2.0] — 2026-03-03

### Added

- **LLM Integration** (`llm.py`, 1,258 lines)
  - `LLMProvider` abstract interface for any LLM backend
  - `MockLLMProvider` — deterministic keyword-matching provider for testing and demos, no API key required
  - `OpenAIProvider` — template for OpenAI Chat Completions API
  - `AnthropicProvider` — template for Anthropic Messages API (Claude)
  - `LLMRouter` — routes requests to the appropriate provider with fallback support
  - `ContextWindow` — maps agent memory to LLM context (long-term memory → system prompt, short-term → turns)
  - `LLMAgent` — `Agent` subclass with full cognitive loop and action-tag parsing
  - Action format: `[ACTION:SEND]`, `[ACTION:TOOL]`, `[ACTION:WRITE]`, `[ACTION:SPAWN]`, `[ACTION:THINK]`
  - `create_mock_router()` convenience factory

- **Supervision Trees** (`supervisor.py`, 844 lines)
  - `SupervisorAgent` with Erlang/OTP-style restart strategies
  - `RestartStrategy` enum: `ONE_FOR_ONE`, `ONE_FOR_ALL`, `REST_FOR_ONE`
  - `RestartType` enum: `PERMANENT`, `TRANSIENT`, `TEMPORARY`
  - `ChildSpec` dataclass for child agent blueprints
  - Restart intensity (max_restarts / window_ticks) with escalation to parent supervisor
  - `SupervisorTree` display class for ASCII hierarchy rendering
  - `build_supervision_tree()` convenience factory for nested hierarchies

- **Extended Tools** (`tools_extended.py`, 1,457 lines)
  - `http_client` — simulated HTTP GET/POST/PUT/DELETE client
  - `python_repl` — safe Python expression evaluator with restricted builtins
  - `json_processor` — parse, query (dot-notation), transform JSON; operations: parse, stringify, query, set, delete, keys, merge
  - `text_analyzer` — word count, sentiment analysis, Flesch-Kincaid readability
  - `vector_store` — in-memory cosine similarity search (pure Python, no NumPy)
  - `key_value_db` — persistent KV store with per-key TTL
  - `task_queue` — priority min-heap task queue
  - `cron_scheduler` — tick-based recurring job scheduler
  - `data_pipeline` — chain multiple tools sequentially

- **Capability-Based Security** (`capabilities.py`, 914 lines)
  - `CapabilityType` enum: `TOOL_USE`, `FILE_READ`, `FILE_WRITE`, `MEMORY_READ`, `MEMORY_WRITE`, `SPAWN`, `MESSAGE`, `NETWORK`, `ADMIN`
  - `CapabilityManager` — central authority for creating, checking, revoking capabilities
  - `Capability` dataclass — unforgeable UUID tokens with expiry and delegatable flag
  - `CapabilitySet` — per-agent collection with fast `has_capability()` checks
  - `SecurityPolicy` — declarative default capabilities by agent class with `{self}` interpolation
  - `DEFAULT_POLICY` — built-in policy for `CoordinatorAgent`, `WorkerAgent`, `MonitorAgent`
  - `@requires_capability` decorator for guarding agent methods
  - Full audit log (`AuditEntry`) for every grant/revoke/check event

- **Typed Memory Schemas** (`schemas.py`, 629 lines)
  - `FieldType` enum: `STRING`, `INT`, `FLOAT`, `BOOL`, `LIST`, `DICT`, `ANY`, `OPTIONAL`
  - `FieldSpec` dataclass with type constraints, range bounds, regex patterns, element types, custom validators
  - `SchemaValidator` with raising and non-raising validation modes
  - `MemorySchema` — typed contract for an agent's memory usage
  - `SchemaRegistry` — global registry with cross-agent lookup (`find_agents_writing_key`, etc.)
  - `SchemaInspector` — introspect schemas, find compatible providers/consumers
  - `@schema` class decorator for declarative schema attachment
  - `GLOBAL_REGISTRY` module-level singleton

- **Network Stack** (`network.py`, 988 lines)
  - `PacketType` enum: `AGENT_MESSAGE`, `DISCOVERY`, `HEARTBEAT`, `MIGRATION`, `GOSSIP`, `SYNC`
  - `Packet` dataclass with checksum, TTL, sequence numbers
  - `VirtualWire` — simulated link with configurable latency, packet loss, bandwidth limits
  - `NetworkTopology` — graph of kernel nodes and their connections
  - `NetworkInterface` — attaches a kernel to the network
  - `RemoteProxy` — local placeholder for remote agents
  - `GossipProtocol` — probabilistic state propagation (fan-out, TTL, O(log N) convergence)
  - `ServiceDiscovery` — register and look up services across nodes

- **Multi-Kernel Federation** (`federation.py`, 1,045 lines)
  - `NodeRole` enum: `LEADER`, `FOLLOWER`, `CANDIDATE`
  - `BalancingStrategy` enum: `ROUND_ROBIN`, `LEAST_LOADED`, `RANDOM`, `AFFINITY`
  - `MigrationStatus` enum
  - `ClusterEntry` — Raft distributed log entry
  - `FederationNode` — wraps a kernel with federation capabilities
  - `FederationCluster` — manages multi-node cluster lifecycle
  - `ConsensusProtocol` — Raft-inspired leader election and log replication
  - `GlobalRegistry` — cluster-wide agent and service directory
  - `AgentMigrator` — at-most-once agent migration with checkpoint/rollback
  - `SplitBrainDetector` — partition detection with read-only mode fallback

- **Self-Modification and Evolution** (`evolution.py`, 991 lines)
  - `ValidationResult` dataclass
  - `CodeValidator` — static AST analysis blocking dangerous patterns
  - `CodeSandbox` — restricted `exec()` with stripped builtins
  - `AgentFactory` — promotes validated code to real `Agent` subclasses
  - `FitnessEvaluator` — multi-objective scoring (tool calls, messages, files, errors)
  - `GeneticPool` — population management with mutation, crossover, selection
  - `EvolutionEngine` — meta-agent orchestrating full generate/validate/deploy/evaluate/select loop

- **Hardware Abstraction Layer** (`hal.py`, 1,100 lines)
  - `DeviceType` enum: `GPIO`, `SENSOR`, `CAMERA`, `ACTUATOR`, `DISPLAY`, `NETWORK_HW`, `STORAGE`, `COMPUTE`
  - `DeviceState` enum: `UNINITIALIZED`, `READY`, `BUSY`, `ERROR`, `OFFLINE`
  - `HardwareDevice` ABC — base class for all devices
  - `DeviceManager` — device registry and lifecycle management
  - `SimulatedHardware` — realistic synthetic data generation
    - Temperature: diurnal sine-wave cycle with Gaussian noise (σ=0.5°C)
    - Camera: deterministic base64-encoded synthetic frames
    - Actuator: position/velocity state tracking
    - Compute: workload queue simulation

- **Formal Verification and Contracts** (`contracts.py`, 1,027 lines)
  - `POLICY_WARN`, `POLICY_BLOCK`, `POLICY_KILL` violation policies
  - `Precondition` — checked before `think()` begins
  - `Postcondition` — checked after `think()` completes
  - `Invariant` — checked every tick
  - `SafetyEnvelope` — hard rate limits per tick (tools, messages, files, spawns)
  - `Contract` — full behavioral specification combining all conditions
  - `ContractMonitor` — background supervisor checking contracts every tick
  - `ContractViolation` exception
  - `PropertyChecker` — temporal property verification (`always`, `eventually`, `until`, `never`)

- **Comprehensive test suite** — 150+ tests covering all 20 modules

---

## [0.1.0] — 2026-03-03

### Added

- **Core Kernel** (`kernel.py`)
  - `Kernel` class with boot sequence, tick loop, syscall dispatch
  - 15 syscalls: spawn_agent, kill_agent, send_message, read_inbox, read_memory, write_memory, access_tool, list_agents, get_status, yield_cpu, write_file, read_file, list_dir, publish_topic, subscribe
  - `KernelPanic` exception

- **Scheduler** (`scheduler.py`)
  - Priority-based (0–9) preemptive scheduler
  - Round-robin within priority bands
  - `AgentState` enum: READY, RUNNING, WAITING, BLOCKED, TERMINATED
  - `ProcessDescriptor` dataclass
  - Voluntary yield, preemption, block/unblock, reprioritize

- **IPC** (`ipc.py`)
  - `MessageType` enum: TASK, RESULT, STATUS, QUERY, REPLY, BROADCAST, HEARTBEAT, ERROR, CUSTOM
  - `Message` dataclass with TTL, correlation_id
  - `Mailbox` — FIFO queue with max_size=128
  - `BulletinBoard` — pub/sub for named topics
  - `IPCManager` — unicast, broadcast, statistics

- **Memory Manager** (`memory.py`)
  - `MemoryType` enum: SHORT_TERM, LONG_TERM, SHARED
  - Per-agent `AgentMemorySpace` with max_keys enforcement
  - `SharedMemoryRegion` with optional authorized_agents
  - TTL-based garbage collection via `gc_tick()`

- **Virtual Filesystem** (`filesystem.py`)
  - Hierarchical in-memory filesystem
  - `/agents/`, `/shared/results/`, `/system/logs/` standard layout
  - `FileMetadata` with owner/group/world permissions
  - `write_file`, `read_file`, `delete_file`, `list_dir`, `mkdir`
  - `tree()` rendering and statistics

- **Tool Manager** (`tools.py`)
  - `ToolManager` with per-agent access control, rate limiting, usage log
  - `ToolSpec` dataclass
  - Built-in tools: `calculator`, `web_search`, `code_executor`, `file_reader`, `file_writer`

- **Agent Runtime** (`agent.py`)
  - `Agent` base class with lifecycle hooks, syscall wrappers
  - `SyscallResult` dataclass
  - `CoordinatorAgent` — decomposes tasks, spawns workers, synthesises results
  - `WorkerAgent` — executes subtasks using tools, reports results
  - `MonitorAgent` — passively observes and publishes metrics

- **Structured Logging** (`logger.py`)
  - Tick-stamped log entries with levels: DEBUG, INFO, WARN, ERROR, SYSTEM
  - Filesystem persistence (writes to `/system/logs/`)

- **Multi-Agent Research Demo** (`main.py`)
  - `python -m battousai.main` entry point
  - Quantum computing research scenario with Coordinator + 2 Workers + Monitor
  - Full system report with filesystem tree
