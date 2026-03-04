# Architecture Overview

Battousai is structured as a layered operating system where each layer provides services to the layer above it. Agents never access layers below them directly — all interaction goes through the kernel's syscall interface.

---

## Layer Diagram

```
╔══════════════════════════════════════════════════════════════════════════╗
║                            Battousai Kernel                                   ║
║                                                                          ║
║  ┌─────────────────────────────── Core Layer ──────────────────────────┐ ║
║  │                                                                      │ ║
║  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │ ║
║  │  │Scheduler │ │   IPC    │ │  Memory  │ │  Files   │ │  Logger  │ │ ║
║  │  │ Priority │ │Mailboxes │ │ Private  │ │ Virtual  │ │ Levels:  │ │ ║
║  │  │ 0–9 band │ │Broadcast │ │ Shared   │ │ /agents/ │ │ DEBUG    │ │ ║
║  │  │  Round   │ │ Pub/Sub  │ │ TTL GC   │ │ /shared/ │ │ INFO     │ │ ║
║  │  │  Robin   │ │ Req/Rply │ │          │ │ /system/ │ │ WARN     │ │ ║
║  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │ ║
║  │       └────────────┴────────────┴─────────────┴────────────┘       │ ║
║  │                         Syscall Interface                            │ ║
║  │   spawn_agent | send_message | read_memory | access_tool            │ ║
║  │   write_file  | list_agents  | get_status  | yield_cpu              │ ║
║  └──────────────────────────────────────────────────────────────────────┘ ║
║                                                                          ║
║  ┌─────────────────────────────── Agent Layer ─────────────────────────┐ ║
║  │                                                                      │ ║
║  │  ┌────────────────┐   ┌────────────────┐   ┌────────────────────┐  │ ║
║  │  │  Agent Runtime │   │ LLM Integration│   │  Supervision Trees │  │ ║
║  │  │  (agent.py)    │   │  (llm.py)      │   │  (supervisor.py)   │  │ ║
║  │  │ CoordAgent     │   │ LLMProvider    │   │ ONE_FOR_ONE        │  │ ║
║  │  │ WorkerAgent    │   │ MockLLM        │   │ ONE_FOR_ALL        │  │ ║
║  │  │ MonitorAgent   │   │ LLMAgent       │   │ REST_FOR_ONE       │  │ ║
║  │  │ + custom...    │   │ ContextWindow  │   │ ChildSpec          │  │ ║
║  │  └────────────────┘   └────────────────┘   └────────────────────┘  │ ║
║  └──────────────────────────────────────────────────────────────────────┘ ║
║                                                                          ║
║  ┌────── Tool Layer ──────────────────────────────────────────────────┐  ║
║  │  tools.py: calculator | web_search | code_executor | file_r/w      │  ║
║  │  tools_extended.py:  http_client | python_repl | json_processor    │  ║
║  │                      text_analyzer | vector_store | key_value_db   │  ║
║  │                      task_queue | cron_scheduler | data_pipeline   │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
║                                                                          ║
║  ┌── Security Layer ──────────────────────┐  ┌── Network Layer ───────┐  ║
║  │  capabilities.py  — least priv.        │  │  network.py            │  ║
║  │  schemas.py       — typed memory       │  │   Packets, VirtualWire │  ║
║  │  contracts.py     — DbC runtime        │  │   Gossip, ServiceDisc  │  ║
║  └────────────────────────────────────────┘  │  federation.py         │  ║
║                                              │   Raft consensus, LB   │  ║
║  ┌── Hardware Layer ──────────────────────┐  └────────────────────────┘  ║
║  │  hal.py                               │                              ║
║  │   GPIO, Sensor, Camera, Actuator      │  ┌── Evolution Layer ──────┐  ║
║  │   SimulatedHardware, DeviceManager    │  │  evolution.py           │  ║
║  └────────────────────────────────────────┘  │   CodeSandbox, Factory  │  ║
║                                              │   GeneticPool, Fitness  │  ║
║                                              └────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Layer Descriptions

### Core Layer

The foundation of Battousai. Five subsystems that every agent depends on:

| Module | Role |
|---|---|
| [`scheduler.py`](scheduler.md) | Priority-based (0–9) preemptive scheduler with round-robin within bands |
| [`ipc.py`](ipc.md) | Async mailboxes, broadcast, request/reply with correlation IDs, pub/sub bulletin board |
| [`memory.py`](memory.md) | Per-agent private memory + named shared regions with TTL-based GC |
| [`filesystem.py`](filesystem.md) | Virtual hierarchical filesystem with permission model |
| `logger.py` | Structured log entries with tick timestamps, levels, and FS persistence |

These subsystems are created by the `Kernel` constructor and wired together during `kernel.boot()`. Agents never hold references to these objects directly — they go through the syscall interface.

### Agent Layer

The intelligence layer. Everything that runs as an agent lives here:

| Module | Role |
|---|---|
| [`agent.py`](../agents/api.md) | `Agent` base class; `CoordinatorAgent`, `WorkerAgent`, `MonitorAgent` |
| [`llm.py`](../agents/llm.md) | `LLMProvider` abstract interface; `MockLLMProvider`, `OpenAIProvider`, `AnthropicProvider`; `LLMAgent`; `ContextWindow` |
| [`supervisor.py`](../agents/supervision.md) | `SupervisorAgent` with Erlang/OTP restart strategies |

### Tool Layer

Tools are Python callables registered with the OS. Agents invoke them via the `access_tool` syscall:

| Module | Tools |
|---|---|
| [`tools.py`](../tools/builtin.md) | `calculator`, `web_search`, `code_executor`, `file_reader`, `file_writer` |
| [`tools_extended.py`](../tools/extended.md) | `http_client`, `python_repl`, `json_processor`, `text_analyzer`, `vector_store`, `key_value_db`, `task_queue`, `cron_scheduler`, `data_pipeline` |

### Security Layer

Optional enforcement layers that can be added on top of the core:

| Module | Role |
|---|---|
| [`capabilities.py`](../security/capabilities.md) | Unforgeable capability tokens; least-privilege enforcement; delegation and revocation |
| [`schemas.py`](../security/schemas.md) | Typed memory schemas; `@schema` decorator; runtime validation |
| [`contracts.py`](../security/contracts.md) | Design-by-Contract; `Precondition`, `Postcondition`, `Invariant`, `SafetyEnvelope` |

### Network Layer

Enables communication between agents on different kernel instances:

| Module | Role |
|---|---|
| [`network.py`](../advanced/networking.md) | `Packet`, `VirtualWire`, `NetworkTopology`, gossip protocol, service discovery, agent migration |
| [`federation.py`](../advanced/federation.md) | Multi-kernel cluster; Raft-inspired consensus; `FederationCluster`, `GlobalRegistry`, load balancing |

### Hardware Layer

Insulates agents from physical device specifics:

| Module | Role |
|---|---|
| [`hal.py`](../advanced/hal.md) | `HardwareDevice` ABC; `DeviceManager`; `SimulatedHardware` with GPIO, sensors, cameras, actuators, accelerators |

### Evolution Layer

Self-modification with safety sandboxing:

| Module | Role |
|---|---|
| [`evolution.py`](../advanced/evolution.md) | `CodeSandbox`, `CodeValidator` (AST analysis), `AgentFactory`, `EvolutionEngine`, `GeneticPool`, `FitnessEvaluator` |

---

## How Layers Connect

### The Syscall Interface

Agents access all OS resources through `self.syscall(name, **kwargs)`. This single interface provides isolation — no module-level globals, no direct subsystem access:

```
Agent.think()
    │
    ▼
Agent.syscall("access_tool", tool_name="calculator", args={...})
    │
    ▼
Kernel._dispatch_syscall(caller_id, "access_tool", ...)
    │
    ▼
Kernel._syscall_access_tool(caller_id, tool_name, args)
    │
    ▼
ToolManager.execute(caller_id, tool_name, args)
    │
    ▼
tool_callable(**args) → result
```

### Boot Sequence

The kernel initialises subsystems in a specific order to satisfy dependencies:

```
1. Logger
2. Filesystem
3. Memory Manager
4. IPC Manager
5. Tool Manager
6. Scheduler
7. Register built-in tools
8. Spawn initial agents
9. Event loop
```

Each subsystem is initialised before the one that depends on it. The filesystem, for example, must exist before the logger can persist log files to it.

### The Tick Loop

Each call to `kernel.tick()` executes this sequence:

```
1. Increment tick counter; update subsystems
2. Snapshot all READY agents in priority order
3. For each ready agent:
   a. Set state = RUNNING
   b. Call agent._tick(current_tick) → agent.think(current_tick)
   c. Return to READY (or WAITING/TERMINATED based on state)
4. Collect terminated agents (remove from kernel, IPC, memory)
5. Run memory GC (evict expired SHORT_TERM entries)
```

---

## Design Principles

1. **Agents as processes** — every computation is an agent; the OS has no "main thread" beyond the event loop
2. **No ambient globals** — agents access all OS services through the syscall interface
3. **Observability by default** — every syscall, message, memory write, and tool call is logged
4. **Graceful degradation** — a crashing agent does not crash the OS; exceptions are caught and the agent remains in READY state
5. **Pure Python** — no external dependencies; runs with `python -m battousai.main`
6. **Least privilege** — agents are granted only the capabilities they need
7. **Fault tolerance** — supervision trees ensure agent failures are contained and recovered
8. **Composability** — every layer is independently usable

---

## Component Summary Table

| Module | Layer | Lines | Key Classes |
|---|---|---|---|
| `kernel.py` | Core | 583 | `Kernel`, `KernelPanic` |
| `scheduler.py` | Core | 332 | `Scheduler`, `ProcessDescriptor`, `AgentState` |
| `ipc.py` | Core | 310 | `IPCManager`, `Mailbox`, `BulletinBoard`, `Message`, `MessageType` |
| `memory.py` | Core | 395 | `MemoryManager`, `AgentMemorySpace`, `SharedMemoryRegion`, `MemoryType` |
| `filesystem.py` | Core | 375 | `VirtualFilesystem`, `FSFile`, `FSDirectory`, `FileMetadata` |
| `agent.py` | Agent | 552 | `Agent`, `CoordinatorAgent`, `WorkerAgent`, `MonitorAgent`, `SyscallResult` |
| `llm.py` | Agent | 1,258 | `LLMProvider`, `MockLLMProvider`, `OpenAIProvider`, `AnthropicProvider`, `LLMRouter`, `ContextWindow`, `LLMAgent` |
| `supervisor.py` | Agent | 844 | `SupervisorAgent`, `ChildSpec`, `RestartStrategy`, `RestartType` |
| `tools.py` | Tools | 447 | `ToolManager`, `ToolSpec`, `ToolUsageRecord` |
| `tools_extended.py` | Tools | 1,457 | 9 tool functions + `register_extended_tools` |
| `capabilities.py` | Security | 914 | `CapabilityManager`, `Capability`, `CapabilityType`, `CapabilitySet` |
| `schemas.py` | Security | 629 | `MemorySchema`, `FieldSpec`, `FieldType`, `SchemaRegistry`, `SchemaValidator` |
| `contracts.py` | Security | 1,027 | `Contract`, `Precondition`, `Postcondition`, `Invariant`, `SafetyEnvelope`, `ContractMonitor` |
| `network.py` | Network | 988 | `Packet`, `PacketType`, `VirtualWire`, `NetworkTopology`, `NetworkInterface`, `GossipProtocol` |
| `federation.py` | Network | 1,045 | `FederationCluster`, `FederationNode`, `ConsensusProtocol`, `GlobalRegistry`, `BalancingStrategy` |
| `hal.py` | Hardware | 1,100 | `HardwareDevice`, `DeviceManager`, `SimulatedHardware`, `DeviceType` |
| `evolution.py` | Evolution | 991 | `CodeSandbox`, `CodeValidator`, `AgentFactory`, `EvolutionEngine`, `GeneticPool`, `FitnessEvaluator` |
