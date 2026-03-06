# Battousai OWASP Alignment — Top 10 for Agentic Applications 2026

## Introduction

The [OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/) is a peer-reviewed security framework released December 9, 2025, produced by 100+ security researchers across industry and academia. It supersedes the earlier LLM-focused OWASP Top 10 with a new list specifically targeting the risks that emerge when AI models are given agency — the ability to autonomously call tools, delegate to sub-agents, retain memory across sessions, and execute code.

Where the LLM Top 10 focused on threats to the model (prompt injection, training data poisoning), the Agentic Top 10 focuses on threats **through** the model to systems: what happens when a compromised or misaligned agent has access to your production database, your filesystem, your API keys, or other agents.

Battousai is designed from the ground up as a security-first agent sandboxing runtime. This document maps each of the 10 risks to Battousai's architecture, identifies gaps with full honesty, and tracks the roadmap to close them.

For additional context on the agentic threat landscape, see the [Astrix Security explainer on OWASP Agentic Top 10](https://astrix.security/learn/blog/the-owasp-agentic-top-10-just-dropped-heres-what-you-need-to-know/) and the [Giskard guide to OWASP Agentic Applications 2026](https://www.giskard.ai/knowledge/owasp-top-10-for-agentic-application-2026).

---

## ASI01 — Agent Goal Hijack

**Status: `PARTIAL`**

### Risk Description

An attacker redirects agent objectives by manipulating inputs, tool outputs, or external content the agent reads. This is the agentic extension of prompt injection: rather than extracting information, the goal is to change *what the agent is trying to accomplish* — turning a customer service agent into a data exfiltration pipeline, for example, by poisoning a document it reads.

### How Battousai Addresses It

- **`contracts.py` — Design-by-Contract invariants.** Agents in Battousai are defined with formal behavioral guarantees: preconditions that must be true before a task runs, postconditions that must hold after, and invariants that can never be violated regardless of instructions. A goal-hijacking attack that tries to redirect the agent past its stated objective will violate these invariants and cause a controlled exception rather than silent misbehavior.

- **`capabilities.py` — Capability scope enforcement.** Even if an agent's instructions are hijacked, its capability token determines which operations it can actually perform. An agent that is manipulated into "now exfiltrate the database" still cannot make a network call if `NETWORK` is not in its capability set. The blast radius of a successful goal hijack is bounded by the agent's granted capabilities.

### Current Gaps

- **No input sanitization layer.** Battousai does not currently inspect or sanitize the content flowing into an agent's context window. A prompt injection attack embedded in a tool response or retrieved document will be passed to the LLM as-is.
- **No semantic drift detection.** There is no mechanism to detect when an agent's behavior is drifting from its declared objective over multiple turns. Invariant violations are caught, but subtle goal drift below the invariant threshold is not.

### Planned Mitigations

- Input sanitization middleware at the tool response boundary — strip or flag content matching known injection patterns before it reaches the LLM context
- Semantic drift detection via embedding-based comparison of agent outputs against the declared objective at each step
- Signed context receipts to prove that what the agent acted on matches what was retrieved

---

## ASI02 — Tool Misuse & Exploitation

**Status: `GOOD`**

### Risk Description

An agent misuses legitimate tools via prompt injection, misalignment, or unsafe delegation. This covers the case where the tools themselves are authorized — the threat is the agent using them in ways the operator never intended. Examples: using a legitimate `send_email` tool to exfiltrate data, using a `file_read` tool to read credential files, chaining tools together to achieve effects no individual tool would permit.

### How Battousai Addresses It

- **`capabilities.py` — Per-tool capability gating.** Every tool call in Battousai requires a matching capability token. Tool access is not a binary `allowed/denied` per agent — it is a typed, scoped permission: `FILE_READ`, `FILE_WRITE`, `NETWORK`, `EXEC`, `ADMIN`, etc. An agent that was granted `FILE_READ` on a specific directory cannot use that permission to read `/etc/passwd`.

- **Fine-grained permission model.** Capability tokens are designed to be narrowly scoped. Rather than granting an agent `NETWORK` in general, operators can scope network access to specific domains. The architecture supports deny-by-default with explicit allow, which is the correct security posture.

- **Tool registry integrity.** All tools are registered in `tools.py` at startup. There is no mechanism for agents to register new tools at runtime without going through the registry, which prevents tool substitution attacks.

### Current Gaps

- **No anomaly detection on tool call patterns.** Battousai does not currently monitor for unusual sequences of tool calls that might indicate misuse (e.g., 50 file reads in 2 seconds, or a specific access pattern consistent with credential harvesting).
- **No rate limiting.** A compromised agent can call the same tool in a tight loop. Per-tool rate limits are not yet implemented.

### Planned Mitigations

- Tool call rate limiting per agent, configurable per capability type
- Anomaly detection layer that flags statistically unusual access patterns and optionally pauses execution for human review

---

## ASI03 — Identity & Privilege Abuse

**Status: `GOOD`**

### Risk Description

An agent exploits inherited credentials, cached permissions, or agent-to-agent trust to access resources beyond its intended scope. In traditional systems this is privilege escalation. In agentic systems it takes a new form: an agent that is trusted by another agent (e.g., a sub-agent trusted by an orchestrator) can potentially leverage that trust to access resources the orchestrator would never grant directly.

### How Battousai Addresses It

- **No ambient authority.** This is a core design principle of Battousai. Agents do not inherit permissions by virtue of who spawned them. Every agent has an explicitly defined capability set, and capabilities are not automatically passed from parent to child. A supervisor agent that has `ADMIN` capability does not automatically grant its sub-agents `ADMIN`.

- **Explicit capability delegation.** When an orchestrator spawns a sub-agent and needs to delegate a capability, that delegation is explicit in the code. There is no implicit inheritance.

- **Per-agent identity.** Each agent is a distinct entity with its own defined capability set. The identity model in `agent.py` ensures that agents cannot claim to be other agents to inherit their permissions.

### Current Gaps

- **No cryptographic identity for inter-agent messages.** The mailbox system in `ipc.py` uses Python dicts in-process. A compromised agent could in principle craft a message claiming to originate from the orchestrator. Messages are not signed, so their claimed origin cannot be verified.
- **No session expiry.** Capability tokens granted to agents do not expire within a session. A long-running agent retains its full capability set indefinitely once granted.

### Planned Mitigations

- HMAC signing of inter-agent messages in `ipc.py` — each message carries a signature verifiable against the sender's identity key
- Capability token TTL — tokens expire after a configurable duration and must be re-granted by the supervisor
- Cryptographic agent identity, likely using Python's `secrets` module for key generation (zero external dependencies)

---

## ASI04 — Agentic Supply Chain Vulnerabilities

**Status: `PARTIAL`**

### Risk Description

Malicious tools, MCP servers, or agent personas are introduced into the agent's environment at runtime — a supply chain attack targeted specifically at agentic systems. The [OWASP Agentic Top 10 explicitly calls out MCP server impersonation](https://astrix.security/learn/blog/the-owasp-agentic-top-10-just-dropped-heres-what-you-need-to-know/) as a real-world vector: a "lookalike" MCP server that silently replaces a trusted one can redirect tool calls and exfiltrate data while appearing legitimate to the agent.

### How Battousai Addresses It

- **Internal tool registry.** All tools available to Battousai agents are defined in `tools.py` at startup. There is no mechanism for runtime injection of tool definitions by default — the attack surface is significantly smaller than systems that accept arbitrary tool schemas at runtime.

- **No dynamic tool loading by default.** Unlike MCP-native frameworks that accept tool definitions from external servers, Battousai's current architecture keeps tools in-process and code-defined.

### Current Gaps

- **No integrity verification for tools loaded at runtime.** When Battousai adds MCP support (a priority on the roadmap), tools will be loaded from external MCP servers. There is currently no mechanism to verify that a tool definition hasn't been tampered with.
- **No MCP server validation.** There is no mechanism to verify the identity of an MCP server before connecting to it, authenticate its certificate chain, or check a known-good registry of MCP server descriptors.
- **No signed tool registry.** Even internal tools are not hash-verified at load time.

### Planned Mitigations

- Hash-based tool registry signing: tool definitions are SHA-256 hashed at registration time and verified before each call
- MCP server identity verification when MCP client support is added — TLS certificate pinning and server descriptor hashing
- `integrity.py` module that can verify the entire tool registry as an audit step

---

## ASI05 — Unexpected Code Execution (RCE)

**Status: `PARTIAL`**

### Risk Description

Agents generate or execute attacker-controlled code, leading to remote code execution on the host system. This is the highest-severity risk in agentic systems: if an agent generates Python or shell code and executes it, and that code was influenced by a prompt injection attack, the attacker has a code execution primitive on whatever system the agent runs on.

### How Battousai Addresses It

- **`evolution.py` — Code execution sandbox.** Battousai includes a sandboxed execution environment for agent-generated code. The capability model gates access to `EXEC` operations, meaning an agent must have explicit `EXEC` capability to run generated code.

- **Subprocess isolation (planned).** The architecture is explicitly designed for subprocess isolation — every OS-level access is meant to go through a capability-gated boundary. The design exists; the implementation is in progress.

### Current Gaps

- **`evolution.py` sandbox is simulated, not real.** The current sandbox is a logical boundary enforced at the Python level, not at the OS level. A sufficiently crafty piece of generated code could escape the logical sandbox through Python's reflection capabilities, `ctypes`, or other mechanisms.
- **No `seccomp`/namespace isolation.** The subprocess is not yet launched with Linux seccomp-bpf syscall filtering or namespace isolation. The `EXEC` capability token blocks the *intent* to execute, but if a code execution path is found that bypasses the capability check, there is no OS-level backstop.

### Planned Mitigations

- **Real subprocess isolation** (Priority 4 on the roadmap): run each agent in a subprocess with proper stdin/stdout IPC; apply `seccomp-bpf` syscall filtering and Linux namespaces on Linux; use `sandbox-exec` (macOS Seatbelt) on macOS
- The capability model in `capabilities.py` becomes the policy that drives the seccomp/sandbox profile — a clean separation between policy definition and enforcement mechanism
- Target: agents *literally cannot* execute syscalls their capability set does not permit, enforced at the kernel level

---

## ASI06 — Memory & Context Poisoning

**Status: `PARTIAL`**

### Risk Description

Persistent corruption of agent memory, RAG stores, or shared context. This is among the most insidious agentic risks because the damage is invisible and persistent: if an attacker can write malicious instructions into an agent's memory or a shared knowledge store, every subsequent agent that reads that memory is compromised. The [ClawSandbox benchmark found that 4 out of 4 memory poisoning attacks succeeded](https://news.ycombinator.com/item?id=47246778) across tested agent frameworks — config files like `AGENTS.md` and `.cursorrules` are writable by agents and loaded without integrity checks.

### How Battousai Addresses It

- **`memory.py` — Typed schemas with validation.** Memory entries in Battousai are typed and schema-validated on write. This prevents an agent from writing arbitrary unstructured data into memory and having it interpreted as instructions on read.

- **Capability-gated memory writes.** Memory write operations require the agent to have the appropriate capability. An agent that only needs to read from memory is not granted write capability.

### Current Gaps

- **No integrity hashing on memory writes.** There is no hash-chain or Merkle tree protecting memory state. A compromised agent with write access could modify existing memory entries without leaving a detectable trace.
- **No expiry or versioning.** Memory entries do not expire. Poisoned instructions persist indefinitely until explicitly overwritten.
- **SQLite persistence not yet implemented.** Memory is currently in-process. Once SQLite persistence is added, the attack surface for memory poisoning grows (disk-level tampering becomes possible) without additional mitigations.

### Planned Mitigations

- **Memory integrity (Priority 3 on the roadmap):** hash-chain writes to `memory.py` — each write records a SHA-256 hash of the previous state; tampering is detectable because the hash chain breaks
- Memory entry TTL and versioning — poisoned instructions expire; old versions can be reviewed
- `integrity.py` module — standalone memory integrity checker runnable as an audit step before agent startup
- Signed tool registry — tool definitions hash-verified at load time to catch `AGENTS.md`-style configuration poisoning

---

## ASI07 — Insecure Inter-Agent Communication

**Status: `WEAK`**

### Risk Description

Inter-agent messages are spoofed, manipulated, or intercepted. In a multi-agent system, agents communicate with each other to delegate tasks, share context, and coordinate actions. If those communication channels lack authentication and integrity protection, an attacker who can inject a message can impersonate any agent — including the orchestrator — and cause other agents to take arbitrary actions.

### How Battousai Addresses It

There is currently limited coverage here. The `ipc.py` Mailbox system provides a structured message-passing interface between agents, which is better than ad-hoc function calls, but the security properties are insufficient for the threat model.

### Current Gaps

- **Messages are plain Python dicts in-process.** There is no message signing, no sender authentication, no encryption, and no replay protection. Any agent with access to the mailbox can send a message claiming to be from any other agent.
- **No authentication.** The Mailbox does not verify that a message originating from "AgentA" was actually sent by the agent instance registered as AgentA.
- **No encryption.** While in-process communication is not subject to network interception, this will become a real exposure when Battousai agents communicate across process or machine boundaries (a planned feature).
- **No replay protection.** A captured message could be replayed to trigger an action a second time.

### Planned Mitigations

- **HMAC signing of inter-agent messages** — each message carries an HMAC signature using the sender's session key; recipients verify before processing
- **Message sequence numbers** — prevent replay attacks
- **Cryptographic agent identity** — agent startup generates a session keypair; the public key is registered with the supervisor; all messages are signed
- Encryption of message payloads at the transport layer when cross-process communication is implemented

---

## ASI08 — Cascading Failures

**Status: `STRONG`**

### Risk Description

Single-point faults propagate through multi-agent workflows, causing systemic failures. In a multi-agent system, one agent crashing, entering an infinite loop, or producing garbage output can cascade to downstream agents that depend on its outputs — amplifying errors across the entire workflow and potentially causing large-scale harmful actions before any human can intervene.

### How Battousai Addresses It

This is Battousai's strongest coverage area. The supervision tree system in `supervisor.py` is directly modeled on Erlang/OTP's battle-tested fault tolerance philosophy.

- **`supervisor.py` — Erlang-style supervision trees.** Agents are organized in hierarchical supervision trees. When a child agent fails, the supervisor responds according to a configurable policy:
  - `one_for_one` — restart only the failed agent; siblings are unaffected
  - `one_for_all` — restart all agents in the supervision group; used when agents are tightly interdependent
  - `rest_for_one` — restart the failed agent and all agents that were started after it; appropriate for pipeline topologies where downstream agents depend on upstream ones

- **Failure isolation by design.** The supervision tree architecture means that failures are contained to their subtree by default. An agent crash in a sub-tree does not automatically propagate to the root supervisor.

- **Predictable recovery.** Because restart policies are declared in code, the system's behavior under failure is auditable and testable — not emergent.

### Current Gaps

- **Supervision only handles crashes.** The current supervision model handles process-level failures (exceptions, crashes). It does not handle semantic misbehavior: an agent that produces subtly wrong outputs but never throws an exception will not be detected or restarted.
- **No circuit breakers.** There is no mechanism to automatically suspend an agent that is consistently producing failed outputs — a necessary protection against an agent that is "working" from the runtime's perspective but harmful from a semantic one.

### Planned Mitigations

- Semantic health checks: supervisors can be configured with an output validator that checks agent outputs for expected properties before forwarding them downstream
- Circuit breaker pattern: automatic agent suspension after N consecutive output validation failures, with escalation to the human-in-the-loop approval workflow

---

## ASI09 — Human-Agent Trust Exploitation

**Status: `WEAK`**

### Risk Description

Agents manipulate humans into approving harmful actions through social engineering, false urgency, or misleading framing. This risk operates in both directions: agents may deceive human reviewers into approving dangerous actions (trust exploitation), and poorly designed human-in-the-loop systems may create "approval fatigue" that causes reviewers to rubber-stamp everything.

### How Battousai Addresses It

This is currently the most significant unaddressed risk in Battousai. There is no human-in-the-loop system, no approval workflow, and no mechanism for agents to present their intended actions to a human before execution.

### Current Gaps

- **No human-in-the-loop checkpoints.** Agents execute tool calls autonomously without any mechanism to pause and request human approval, even for high-impact operations like `FILE_WRITE` to sensitive paths or `NETWORK` calls to external services.
- **No approval workflow.** There is no interface — CLI, webhook, or web UI — for a human to review and approve or reject a pending agent action.
- **No explanation audit trail.** There is no record of what justification an agent provided for a given action, making it impossible to detect if an agent is systematically misrepresenting its intentions to humans.

### Planned Mitigations

- **`approval.py` module (Priority 2 on the roadmap):** a checkpoint system integrated with the capability model — tool calls above a configurable risk tier (e.g., `NETWORK`, `FILE_WRITE`, `ADMIN`) pause execution and emit an approval request
- Approval channels: CLI prompt, webhook (for CI/CD integration), and a simple built-in local web UI
- Audit trail: every approval decision records who approved, when, what the agent stated as justification, and what the actual tool call was — enabling post-hoc analysis for trust exploitation detection
- Risk-tiered approvals to mitigate approval fatigue: low-risk actions are batched and presented as a summary; only high-risk actions require individual review

---

## ASI10 — Rogue Agents

**Status: `PARTIAL`**

### Risk Description

Compromised or misaligned agents diverge from their intended behavior — acting outside their designated roles, pursuing emergent sub-goals, or actively working against the system's intended purpose. This is the most general risk: it covers everything from an agent that drifts due to context accumulation to a deliberately compromised agent pursuing an attacker's goals.

### How Battousai Addresses It

- **`MonitorAgent` in `agent.py`.** Battousai includes a dedicated monitoring agent that observes other agents' behavior. This is an architectural acknowledgment that behavioral monitoring is a first-class concern, not an afterthought.

- **`contracts.py` — Invariant enforcement.** Design-by-contract invariants define the behavioral boundaries within which agents must operate. Violations are detected and raise exceptions. This is the strongest tool Battousai currently has against rogue behavior — formal specification of what an agent is and is not allowed to do.

### Current Gaps

- **No behavioral drift detection.** The invariant system catches hard violations, but subtle drift — an agent that is gradually changing its behavior in ways that individually don't violate any invariant but collectively represent dangerous misalignment — is not detected.
- **No agent kill switch with graceful audit.** There is no `battousai.kill_agent(id, reason="policy_violation")` primitive that would immediately halt an agent, record the reason and state at time of termination, and emit an alert. The supervision tree can crash an agent but it will restart it per policy.

### Planned Mitigations

- Behavioral drift detection using embedding-based comparison of agent outputs over time against a baseline profile established at agent creation
- Graceful agent kill switch with audit log: halt execution, snapshot final state, record reason, prevent restart until human review
- Integration with the human-in-the-loop approval workflow: detected drift automatically escalates to human review rather than silent restart

---

## Summary Coverage Table

| Risk | Name | Status | Battousai Coverage |
|------|------|--------|--------------------|
| ASI01 | Agent Goal Hijack | `PARTIAL` | Contracts + capability scope bound the blast radius; no input sanitization or drift detection |
| ASI02 | Tool Misuse & Exploitation | `GOOD` | Per-tool capability tokens gate every call; deny-by-default design |
| ASI03 | Identity & Privilege Abuse | `GOOD` | No ambient authority; explicit capability delegation; no implicit inheritance |
| ASI04 | Agentic Supply Chain Vulnerabilities | `PARTIAL` | Internal tool registry limits attack surface; no integrity verification for runtime-loaded tools |
| ASI05 | Unexpected Code Execution | `PARTIAL` | `evolution.py` sandbox exists; capability gates EXEC; not yet OS-level enforced |
| ASI06 | Memory & Context Poisoning | `PARTIAL` | Typed memory schemas; no hash-chain integrity or TTL |
| ASI07 | Insecure Inter-Agent Communication | `WEAK` | Mailbox provides structured IPC; no authentication, signing, or encryption |
| ASI08 | Cascading Failures | `STRONG` | Erlang-style supervision trees with `one_for_one`, `one_for_all`, `rest_for_one` policies |
| ASI09 | Human-Agent Trust Exploitation | `WEAK` | Not currently addressed; no approval workflow exists |
| ASI10 | Rogue Agents | `PARTIAL` | MonitorAgent + contracts; no drift detection or graceful kill switch |

**Coverage summary:**
- **STRONG:** 1 (ASI08)
- **GOOD:** 2 (ASI02, ASI03)
- **PARTIAL:** 5 (ASI01, ASI04, ASI05, ASI06, ASI10)
- **WEAK:** 2 (ASI07, ASI09)

No competing framework — CrewAI, LangGraph, AutoGen, or others — has published an OWASP Agentic Top 10 alignment analysis. Battousai's architecture addresses more of these risks by design than any alternative, and this analysis documents an honest path to full coverage.

---

## Roadmap to Full Coverage

The following work items, prioritized by security impact and implementation effort, would bring Battousai to **GOOD** or **STRONG** coverage across all 10 risks.

### Near-term (1–2 weeks each)

**1. Memory integrity and signed config** (closes ASI06 gap; improves ASI04, ASI10)
- Hash-chain writes to `memory.py` using stdlib `hashlib.sha256`
- Memory entry TTL and versioning
- `integrity.py` module as a standalone audit step
- Effort: Low — pure stdlib, no external dependencies

**2. Human-in-the-loop approval workflow** (closes ASI09; improves ASI01, ASI10)
- `approval.py` checkpoint system integrated with capability model
- Risk-tiered: high-impact tool calls pause for review; low-risk batched for summary review
- CLI + webhook approval channels initially; local web UI as a follow-on
- Effort: Medium — natural extension of the existing capability token model

**3. HMAC inter-agent message signing** (closes ASI07)
- Session keypair generated at agent startup using stdlib `secrets` + `hmac`
- All mailbox messages carry HMAC signature; recipients verify on receipt
- Sequence numbers for replay protection
- Effort: Low — pure stdlib, no external dependencies

### Medium-term (2–4 weeks each)

**4. Real subprocess isolation with seccomp/namespaces** (closes ASI05 gap; strengthens ASI01, ASI02, ASI04)
- `subprocess_agent.py` — agents run in separate processes with stdin/stdout IPC
- Linux: `seccomp-bpf` syscall filtering + Linux namespaces
- macOS: `sandbox-exec` (Seatbelt)
- `capabilities.py` model drives the seccomp profile — policy and enforcement stay in sync
- Effort: High — this is the single most impactful security improvement

**5. Tool integrity verification and MCP server validation** (closes ASI04 gap)
- SHA-256 hash of every tool definition at registration; verify before each call
- MCP server identity verification: TLS certificate pinning + descriptor hashing
- Effort: Medium — required before MCP support is production-ready

### Longer-term

**6. Behavioral drift detection** (strengthens ASI01, ASI10)
- Embedding-based comparison of agent outputs against declared objective and baseline profile
- Automatic escalation to human review on detected drift

**7. Circuit breakers for semantic misbehavior** (strengthens ASI08)
- Automatic agent suspension after N consecutive output validation failures
- Escalation to human-in-the-loop workflow

**8. Graceful agent kill switch** (closes ASI10 gap)
- `battousai.kill_agent(id, reason)` — halt, snapshot, audit log, block restart until review

---

*Last updated: March 6, 2026. Sources: [OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/), [Astrix Security OWASP Agentic Top 10 explainer](https://astrix.security/learn/blog/the-owasp-agentic-top-10-just-dropped-heres-what-you-need-to-know/), [ClawSandbox memory poisoning benchmark](https://news.ycombinator.com/item?id=47246778).*
