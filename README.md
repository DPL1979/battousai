# Battousai

**A lightweight Python runtime for sandboxed AI agents — capability-based security, fault-tolerant supervision, and memory isolation in under 16K lines with zero dependencies.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 857 passing](https://img.shields.io/badge/tests-857%20passing-brightgreen.svg)](#testing)
[![Dependencies: 0](https://img.shields.io/badge/dependencies-0-orange.svg)](#zero-dependencies)

---

## Why Battousai?

Every AI agent framework lets agents do things. Battousai makes sure they **only** do what they're allowed to.

```
pip install battousai   # zero dependencies, works offline
```

Most agent frameworks (LangGraph, CrewAI, AutoGen) assume you trust your agents. They run in-process, share memory, and have access to your entire host environment. When an agent goes rogue — through prompt injection, memory poisoning, or simple misalignment — there's nothing between it and your SSH keys.

Battousai is different. Every agent runs inside a **capability-gated sandbox** where file access, network calls, tool invocations, and memory writes require explicit capability tokens. If an agent doesn't have the token, the action is blocked and logged. No exceptions.

### What makes it different

| Feature | Battousai | LangGraph | CrewAI | AutoGen | E2B | Modal |
|---------|-----------|-----------|--------|---------|-----|-------|
| **Capability-token security** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Erlang supervision trees** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Design-by-Contract invariants** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Zero external dependencies** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Self-hosted / air-gapped** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Process-level isolation** | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Per-agent filesystem jails** | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **SQLite persistence (WAL)** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

---

## Quick Start

### Hello World

```python
from battousai.kernel import Kernel
from battousai.agent import WorkerAgent
from battousai.llm import MockLLMProvider

# Boot the kernel
kernel = Kernel(max_ticks=5)
kernel.boot()

# Spawn a sandboxed agent
agent_id = kernel.spawn_agent(WorkerAgent, name="researcher", priority=5)

# Run the event loop
kernel.run()
```

### Real LLM Providers (v0.3.0)

```python
from battousai.providers import RealOpenAIProvider, RealAnthropicProvider, OllamaProvider

# OpenAI — uses stdlib urllib, no openai package needed
openai = RealOpenAIProvider(api_key="sk-...", model="gpt-4o")
response = openai.generate("Analyze this security report", max_tokens=2000)

# Anthropic — same pattern
claude = RealAnthropicProvider(api_key="sk-ant-...", model="claude-sonnet-4-20250514")

# Ollama — local models, no API key
ollama = OllamaProvider(model="llama3")
```

### Sandboxed Filesystem

```python
from battousai.real_fs import SandboxedFilesystem

# Each agent gets a jail — path traversal is impossible
fs = SandboxedFilesystem(base_dir="/tmp/battousai-sandbox")
agent_root = fs.create_agent_jail("agent_001")

fs.write_file("agent_001", "notes.txt", "Research findings...")
content = fs.read_file("agent_001", "notes.txt")

# This raises SecurityViolation — agent can't escape its jail
fs.read_file("agent_001", "../../etc/passwd")
```

### Checkpoint & Restore

```python
from battousai.persistence import PersistenceLayer

db = PersistenceLayer("/tmp/battousai.db")

# Save agent state (uses SQLite WAL mode for concurrent access)
db.save_checkpoint("agent_001", {"task": "research", "progress": 0.7})

# Restore after crash
state = db.load_checkpoint("agent_001")
```

### Process Isolation

```python
from battousai.isolation import IsolatedAgentProcess, SandboxConfig

config = SandboxConfig(
    max_memory_mb=256,
    max_cpu_time_seconds=30,
    allowed_paths=["/tmp/agent-workspace"],
    network_enabled=False
)

# Agent runs in a separate process with enforced limits
proc = IsolatedAgentProcess(agent_id="untrusted_agent", config=config)
proc.start()
```

---

## Security Model

Battousai enforces security at three levels:

### 1. Capability Tokens
Every resource access requires an explicit capability token. No ambient authority.

```python
from battousai.capabilities import CapabilityManager, CapabilityType

cap_mgr = CapabilityManager()

# Grant specific capabilities — agent can read files but not write
cap_mgr.grant(agent_id, CapabilityType.FILE_READ)
# cap_mgr.grant(agent_id, CapabilityType.FILE_WRITE)  # NOT granted

# Every syscall checks capabilities before executing
# Unauthorized access → blocked + audit logged
```

### 2. Safety Envelope
Hard limits that override all other policies:

```python
from battousai.contracts import SafetyEnvelope, SafetyEnvelopeConfig

envelope = SafetyEnvelope(SafetyEnvelopeConfig(
    max_messages_per_tick=10,
    max_tool_calls_per_tick=5,
    max_file_size=1_000_000,         # 1MB max per file write
    forbidden_tools=["shell_exec"],   # Absolute blocklist
    max_total_agents=20,              # Prevent fork bombs
))
```

### 3. Design-by-Contract
Agents declare behavioral contracts verified at runtime:

```python
from battousai.contracts import Contract, Invariant, POLICY_KILL

contract = (Contract(name="SafeResearcher", agent_class_name="WorkerAgent")
    .add_invariant(Invariant(
        name="memory_bound",
        description="Agent memory must stay under limit",
        check=lambda agent: agent.memory_usage < 100_000,
        on_violation="KILL"  # Terminate immediately if violated
    )))
```

### OWASP Agentic AI Alignment

Battousai maps to all 10 risks in the [OWASP Top 10 for Agentic Applications (2026)](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/). See [`docs/owasp-alignment.md`](docs/owasp-alignment.md) for the full analysis.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  BATTOUSAI RUNTIME                       │
├──────────────┬──────────────┬──────────────┬────────────┤
│   Kernel     │  Supervisor  │  Contracts   │  Scheduler │
│  (syscalls,  │  (Erlang     │  (pre/post/  │  (priority │
│   security)  │   trees)     │   invariant) │   ticks)   │
├──────────────┼──────────────┼──────────────┼────────────┤
│ Capabilities │   Memory     │  Filesystem  │    IPC     │
│  (tokens,    │  (isolated   │  (per-agent  │  (mailbox, │
│   audit)     │   regions)   │   jails)     │   board)   │
├──────────────┼──────────────┼──────────────┼────────────┤
│  Providers   │ Persistence  │  Isolation   │   Tools    │
│  (OpenAI,    │  (SQLite     │  (process    │  (registry │
│   Claude,    │   WAL,       │   pools,     │   + gate)  │
│   Ollama)    │   restore)   │   sandbox)   │            │
└──────────────┴──────────────┴──────────────┴────────────┘
```

### Core Modules (29 files, ~21,000 lines)

| Module | Purpose |
|--------|---------|
| `kernel.py` | Central coordinator, syscall dispatch, event loop |
| `agent.py` | Agent base classes, lifecycle, state machines |
| `capabilities.py` | Capability tokens, security policies, audit trail |
| `contracts.py` | Design-by-Contract, SafetyEnvelope, PropertyChecker |
| `supervisor.py` | Erlang-style supervision trees (one_for_one, one_for_all, rest_for_one) |
| `memory.py` | Per-agent memory spaces, shared regions, typed schemas |
| `filesystem.py` | Virtual filesystem with permissions |
| `real_fs.py` | Sandboxed real filesystem with per-agent jails |
| `providers.py` | Real LLM providers (OpenAI, Anthropic, Ollama) via stdlib |
| `persistence.py` | SQLite persistence with WAL mode, checkpoint/restore |
| `isolation.py` | Process-level agent isolation, sandbox configs |
| `mcp_server.py` | MCP JSON-RPC 2.0 server with capability-gated tool access |
| `mcp_client.py` | MCP client for connecting to external MCP servers |
| `approval.py` | Human-in-the-loop approval workflow (risk tiers, audit trail) |
| `integrity.py` | SHA-256 hash-chain memory integrity, tamper detection |
| `sandbox.py` | OS-level sandboxing: seccomp, namespaces, rlimits, env sanitization |
| `scheduler.py` | Priority-based tick scheduler (10 levels) |
| `ipc.py` | Inter-agent messaging, mailboxes, bulletin board |
| `network.py` | Gossip protocol, service discovery |
| `federation.py` | Raft consensus, multi-kernel coordination |
| `evolution.py` | Code sandbox, genetic algorithms |
| `tools.py` | Tool registry and capability-gated invocation |

---

## Fault Tolerance

Battousai uses **Erlang-style supervision trees** — the same pattern that gives Erlang systems nine-nines uptime.

```python
from battousai.supervisor import SupervisorAgent

# If a child agent crashes, the supervisor restarts it automatically
# Strategies: one_for_one, one_for_all, rest_for_one
supervisor = SupervisorAgent(
    name="research_team",
    strategy="one_for_one",
    max_restarts=5,
    restart_window=20  # ticks
)
```

When an agent fails:
- **one_for_one** — only the failed agent restarts
- **one_for_all** — all siblings restart (for tightly coupled workflows)
- **rest_for_one** — the failed agent and all agents started after it restart

---

## Testing

```bash
# Run all 857 tests (takes ~3.5 seconds)
python -m unittest discover -s tests -v

# Run a specific module
python -m unittest tests.test_contracts -v
python -m unittest tests.test_providers -v
python -m unittest tests.test_isolation -v
```

29 test files covering every module. Zero external test dependencies.

---

## Zero Dependencies

Battousai uses **only the Python standard library**. No pip packages, no native extensions, no Docker required.

- HTTP requests → `urllib.request`
- JSON handling → `json`
- Persistence → `sqlite3`
- Process isolation → `multiprocessing`
- Crypto hashing → `hashlib`
- Logging → built-in `logging`

This means:
- `pip install battousai` just works, everywhere
- Air-gapped deployments need nothing beyond Python 3.10+
- No supply chain risk from third-party packages
- No version conflicts with your existing project

---

## Examples

```bash
# Security demo — shows capability enforcement in action
python examples/security_demo.py

# Minimal quickstart
python examples/quickstart.py

# Run the full kernel with agents
python -m battousai.main
```

---

## Documentation

- [`docs/owasp-alignment.md`](docs/owasp-alignment.md) — Maps all 10 OWASP Agentic AI risks to Battousai
- [`docs/comparison.md`](docs/comparison.md) — Honest comparison vs CrewAI, AutoGen, LangGraph, E2B, Modal, Daytona

---

## Roadmap

### v0.4.0 — Production Hardening
- [x] MCP server adapter (expose tools as MCP-compatible)
- [x] MCP client adapter (connect to external MCP servers with capability gating)
- [x] Human-in-the-loop approval workflow for high-risk actions
- [x] Memory integrity hashing (tamper detection)
- [x] OS-level sandboxing (seccomp, namespaces, rlimits)
- [ ] IPC message signing (HMAC)
- [ ] Agent behavioral drift detection

### v0.5.0 — Launch
- [ ] Comprehensive security audit
- [ ] Performance benchmarks against E2B/Modal
- [ ] PyPI stable release
- [ ] GitHub Codespace one-click demo

### v1.0.0 — Production Ready
- [ ] seccomp + namespace hardening with test coverage
- [ ] Formal verification of capability model
- [ ] Plugin architecture for custom security policies

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/DPL1979/battousai.git
cd battousai
python -m unittest discover -s tests -v  # all tests should pass
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Battousai (抜刀斎) — "one who draws the sword." Named for decisive, swift action under constraint.*
