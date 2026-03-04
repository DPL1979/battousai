# Battousai — Autonomous Intelligence Operating System

```
  █████╗ ██╗ ██████╗ ███████╗
 ██╔══██╗██║██╔═══██╗██╔════╝
 ███████║██║██║   ██║███████╗
 ██╔══██║██║██║   ██║╚════██║
 ██║  ██║██║╚██████╔╝███████║
 ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚══════╝
 Autonomous Intelligence Operating System  v0.2.0
```

> **An operating system designed exclusively for AI agents.**
> No human users. No GUI. No terminal. Agents are first-class citizens.

---

## What is Battousai?

Conventional operating systems were designed for human users who interact through terminals, GUIs, and applications. Battousai inverts this assumption entirely.

In Battousai, **AI agents are the only processes**. There is no concept of a human typing a command — only agents spawning other agents, exchanging messages, using tools, and writing their findings to a shared filesystem. The "user interface" is the agent runtime itself.

Battousai provides the substrate for autonomous AI workloads: scheduling, isolation, communication, memory, storage, tools, security, networking, hardware abstraction, formal verification, and self-improvement — all in a pure Python, zero-dependency package.

---

## Key Features

- **Agent-first design** — Agents are processes; the OS has no concept of a human user
- **Priority-based scheduler** — 10-band preemptive scheduler with round-robin within bands; agents yield voluntarily via `yield_cpu()`
- **Structured IPC** — Async mailboxes, broadcast, request/reply with correlation IDs, pub/sub bulletin board
- **Typed memory** — Per-agent private memory + shared regions; TTL-based garbage collection; runtime schema validation
- **Virtual filesystem** — Hierarchical `/agents/`, `/shared/`, `/system/` layout with permission enforcement
- **Rich tool ecosystem** — 5 built-in tools (calculator, web search, code executor, file I/O) + 9 extended tools (HTTP client, vector store, task queue, data pipeline, and more)
- **LLM integration** — Pluggable `LLMProvider` interface; `LLMAgent` with action-tag parsing; `ContextWindow` memory-to-context mapping; `MockLLMProvider` for testing; templates for OpenAI and Anthropic
- **Fault-tolerant supervision** — Erlang/OTP-style `SupervisorAgent` with `ONE_FOR_ONE`, `ONE_FOR_ALL`, and `REST_FOR_ONE` restart strategies
- **Capability-based security** — Unforgeable capability tokens, least-privilege enforcement, delegatable/revocable, full audit log
- **Distributed networking** — Multi-kernel network stack with gossip protocol, service discovery, and agent migration
- **Multi-kernel federation** — Raft-inspired consensus, leader election, `GlobalRegistry`, load balancing across kernel nodes
- **Hardware abstraction** — `DeviceManager`, `SimulatedHardware`, 8 device types (GPIO, sensors, cameras, actuators, accelerators)
- **Self-modification engine** — `CodeSandbox`, AST validation, `AgentFactory`, `GeneticPool`, fitness evaluation
- **Formal contracts** — Design-by-Contract runtime verification; preconditions, postconditions, invariants, `SafetyEnvelope`
- **Pure Python** — Python 3.10+, zero external dependencies, 13,691 lines across 20 modules

---

## Architecture

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
║  │  │                │   │                │   │                    │  │ ║
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
║  ┌────── Security Layer ──────────────────┐  ┌────── Network Layer ───┐  ║
║  │  capabilities.py  — least priv.        │  │  network.py            │  ║
║  │  schemas.py       — typed memory       │  │   Packets, VirtualWires│  ║
║  │  contracts.py     — DbC runtime        │  │   Gossip, ServiceDisc  │  ║
║  └────────────────────────────────────────┘  │  federation.py         │  ║
║                                              │   Raft consensus, LB   │  ║
║  ┌────── Hardware Layer ──────────────────┐  └────────────────────────┘  ║
║  │  hal.py                               │                              ║
║  │   GPIO, Sensor, Camera, Actuator      │  ┌────── Evolution Layer ──┐  ║
║  │   SimulatedHardware, DeviceManager    │  │  evolution.py           │  ║
║  └────────────────────────────────────────┘  │   CodeSandbox, Factory  │  ║
║                                              │   GeneticPool, Fitness  │  ║
║                                              └────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Quick Start

**Requirements:** Python 3.10+ — no external dependencies.

```bash
# Clone and run the demo
git clone https://github.com/battousai-project/battousai.git
cd battousai
python -m battousai.main
```

```bash
# Run with custom options
python -m battousai.main --ticks 100     # more ticks
python -m battousai.main --debug         # verbose output
python -m battousai.main --no-banner     # suppress ASCII art
```

### Programmatic usage

```python
from battousai.kernel import Kernel
from battousai.agent import Agent

class HelloAgent(Agent):
    def think(self, tick: int) -> None:
        self.log(f"Hello from tick {tick}!")
        self.yield_cpu()

kernel = Kernel(max_ticks=10)
kernel.boot()
kernel.spawn_agent(HelloAgent, name="Hello", priority=5)
kernel.run()
report = kernel.system_report()
print(report)
```

---

## Documentation Sections

| Section | Description |
|---|---|
| [Installation](getting-started/installation.md) | Requirements, install from PyPI or source |
| [Quickstart](getting-started/quickstart.md) | 5-minute guide to boot, spawn, and message |
| [Demo Walkthrough](getting-started/demo.md) | Tick-by-tick explanation of `python -m battousai.main` |
| [Architecture Overview](architecture/overview.md) | All layers explained with diagrams |
| [Kernel](architecture/kernel.md) | Boot sequence, tick loop, syscall dispatch |
| [Scheduler](architecture/scheduler.md) | Priority bands, round-robin, preemption |
| [IPC](architecture/ipc.md) | Messages, mailboxes, broadcast, pub/sub |
| [Memory](architecture/memory.md) | Private/shared regions, TTL GC |
| [Filesystem](architecture/filesystem.md) | Virtual FS layout, permissions, operations |
| [Agent API](agents/api.md) | Base class, lifecycle hooks, all methods |
| [LLM Integration](agents/llm.md) | Providers, LLMAgent, ContextWindow, action format |
| [Supervision Trees](agents/supervision.md) | SupervisorAgent, strategies, ChildSpec |
| [Custom Agents](agents/custom.md) | Guide to building your own agents |
| [Built-in Tools](tools/builtin.md) | calculator, web_search, code_executor, file I/O |
| [Extended Tools](tools/extended.md) | HTTP, REPL, vector store, task queue, pipeline |
| [Capabilities](security/capabilities.md) | Least-privilege tokens, delegation, audit |
| [Schemas](security/schemas.md) | Typed memory, @schema decorator, validation |
| [Contracts](security/contracts.md) | Design-by-Contract, preconditions, safety envelopes |
| [Networking](advanced/networking.md) | Packets, VirtualWire, gossip, service discovery |
| [Federation](advanced/federation.md) | Multi-kernel cluster, Raft consensus, migration |
| [HAL](advanced/hal.md) | Hardware abstraction, simulated devices |
| [Evolution](advanced/evolution.md) | Self-modification, code sandbox, genetic pool |
| [Contributing](contributing.md) | How to contribute to Battousai |
| [Changelog](changelog.md) | Version history |
