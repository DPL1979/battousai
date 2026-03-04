# Changelog

## [0.2.0] - 2026-03-03

### Added
- LLM integration layer with pluggable providers (MockLLM, OpenAI, Anthropic templates)
- LLMAgent with cognitive loop and action parsing
- Erlang/OTP-style supervision trees (ONE_FOR_ONE, ONE_FOR_ALL, REST_FOR_ONE)
- 9 extended tools: HTTP client, Python REPL, JSON processor, text analyzer, vector store, KV database, task queue, cron scheduler, data pipeline
- Distributed agent network stack with gossip protocol and service discovery
- Typed memory schemas with @schema decorator and runtime validation
- Capability-based security with unforgeable tokens and audit logging
- Self-modification sandbox with AST code validation and genetic agent evolution
- Hardware abstraction layer with simulated GPIO, sensors, cameras, actuators, compute accelerators
- Multi-kernel federation with Raft-inspired consensus and leader election
- Formal verification with behavioral contracts and safety envelopes
- Comprehensive test suite (150+ tests)

## [0.1.0] - 2026-03-03

### Added
- Initial release
- Core kernel with event loop and syscall interface
- Priority-based preemptive scheduler
- IPC with async mailboxes, broadcast, pub/sub bulletin board
- Memory manager with private/shared regions and TTL garbage collection
- Virtual hierarchical filesystem with permissions
- Tool manager with 5 built-in tools
- Agent runtime with Coordinator, Worker, and Monitor agents
- Structured logging with filesystem persistence
- Multi-agent research demo scenario
