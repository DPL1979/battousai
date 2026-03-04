# Installation

Battousai has **zero external dependencies**. It runs on the Python standard library alone.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10 or later |
| External packages | None |
| Operating system | Any (Linux, macOS, Windows) |

!!! note "Why no dependencies?"
    Battousai is intentionally dependency-free. Every module (`scheduler.py`, `ipc.py`, `memory.py`, etc.) uses only the Python standard library. This means you can clone the repo and run it immediately on any Python 3.10+ installation with no `pip install` step.

---

## Install from PyPI

```bash
pip install battousai-os
```

Verify the installation:

```bash
python -c "import battousai; print(battousai.__version__)"
# 0.2.0

python -m battousai.main --no-banner
# [tick=0000] SYSTEM  kernel   Battousai v0.2.0 booting...
```

---

## Install from Source

```bash
# Clone the repository
git clone https://github.com/battousai-project/battousai.git
cd battousai

# Install in editable (development) mode
pip install -e .

# Or simply run directly without installing
python -m battousai.main
```

!!! tip "Editable mode"
    `pip install -e .` installs Battousai in editable mode — changes to the source files take effect immediately without reinstalling.

---

## Verify the Installation

Run the built-in demo to confirm everything is working:

```bash
python -m battousai.main
```

Expected first lines of output:

```
  █████╗ ██╗ ██████╗ ███████╗
 ██╔══██╗██║██╔═══██╗██╔════╝
 ...
 Autonomous Intelligence Operating System  v0.2.0

[tick=0000] SYSTEM  kernel               Battousai v0.2.0 booting...
[tick=0000] SYSTEM  kernel               Filesystem initialised (/agents, /shared, /system/logs)
[tick=0000] SYSTEM  kernel               Memory manager online (global shared region created)
[tick=0000] SYSTEM  kernel               Tools registered: ['calculator', 'code_executor', 'file_reader', 'file_writer', 'web_search']
[tick=0000] SYSTEM  kernel               Boot sequence complete. Ready.
```

The demo runs for 50 ticks by default and prints a full system report at the end.

---

## Run Options

```bash
# Default: 50 ticks
python -m battousai.main

# More ticks
python -m battousai.main --ticks 100

# Enable debug-level logging (very verbose)
python -m battousai.main --debug

# Suppress the ASCII art banner
python -m battousai.main --no-banner
```

---

## Project Structure

```
battousai/
├── __init__.py         Version info and top-level exports
├── agent.py            Base Agent class + CoordinatorAgent, WorkerAgent, MonitorAgent
├── capabilities.py     Capability-based security; CapabilityManager
├── contracts.py        Design-by-Contract runtime verification; ContractMonitor
├── evolution.py        Self-modification engine; CodeSandbox, AgentFactory, GeneticPool
├── federation.py       Multi-kernel federation; Raft consensus, agent migration
├── filesystem.py       Virtual hierarchical filesystem with permission model
├── hal.py              Hardware abstraction layer; simulated devices
├── ipc.py              Mailboxes, broadcast, request/reply, pub/sub bulletin board
├── kernel.py           Central coordinator, event loop, syscall dispatcher
├── llm.py              LLM integration; LLMProvider, LLMAgent, ContextWindow
├── logger.py           Structured logging with tick timestamps
├── main.py             Demo entry point
├── memory.py           Private + shared memory with TTL GC
├── network.py          Distributed network stack; VirtualWire, Gossip
├── scheduler.py        Priority-based preemptive scheduler with round-robin
├── schemas.py          Typed memory schemas; @schema decorator, validation
├── supervisor.py       Erlang/OTP supervision trees
├── tools.py            Tool registry with access control, built-in tools
└── tools_extended.py   9 extended tools
```

---

## Next Steps

- [Quickstart](quickstart.md) — boot a kernel, spawn agents, send messages
- [Demo Walkthrough](demo.md) — understand the built-in research scenario tick-by-tick
- [Architecture Overview](../architecture/overview.md) — understand the system design
