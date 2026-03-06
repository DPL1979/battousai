# Battousai vs. The Field — An Honest Comparison

## Quick Positioning

Battousai is not an agent orchestration framework. This distinction matters.

Frameworks like CrewAI, AutoGen, and LangGraph answer the question: *"How do I wire agents together to accomplish a task?"* They provide role models, conversation patterns, graph topologies, and LLM provider abstractions. They are excellent at what they do.

Battousai answers a different question: *"How do I run agents safely, with auditable security properties, bounded blast radius, and predictable failure behavior?"* It is a **sandboxing runtime** — the layer that sits beneath or beside an orchestration framework and enforces that agents can only do what they are explicitly permitted to do.

The infrastructure analogy: CrewAI/LangGraph are application frameworks; Battousai is closer to an OS process supervisor and capabilities system. The two are complementary. You can use LangGraph for orchestration and run each agent through Battousai's capability and supervision model for security enforcement.

The cloud sandbox analogy: E2B and Modal isolate agents at the OS level via microVMs and containers, respectively. Battousai operates at the agent behavior level — it enforces *which tool calls* an agent is permitted to make, audits *why* it made them, and constrains *what it can do* within a session. These are different and complementary guarantees.

---

## Feature Comparison Table

| Feature | Battousai | CrewAI | AutoGen / AG2 | LangGraph | E2B | Modal | Daytona |
|---------|-----------|--------|---------------|-----------|-----|-------|---------|
| **Capability-based security** | ✅ Per-tool, typed capability tokens; deny-by-default | ❌ None | ❌ None | ❌ None | ✅ OS-level only (microVM) | ✅ OS-level only (gVisor) | ✅ OS-level only (Docker) |
| **Process isolation** | ⚠️ In-process today; subprocess isolation on roadmap | ❌ In-process | ❌ In-process | ❌ In-process | ✅ Firecracker microVM (hardware-level) | ✅ gVisor container | ✅ Docker (shared kernel) |
| **Memory isolation** | ⚠️ Typed schemas; no hash-chain integrity yet | ❌ None | ❌ None | ❌ None | ✅ Isolated VM memory | ✅ Isolated container memory | ✅ Isolated container memory |
| **Supervision / fault tolerance** | ✅ Erlang-style supervision trees (`one_for_one`, `one_for_all`, `rest_for_one`) | ❌ None | ❌ None | ⚠️ Error handling at graph edges | ❌ None | ❌ None | ❌ None |
| **Audit trail** | ✅ Per-tool-call audit logging via capability model | ❌ None | ⚠️ Conversation logs | ⚠️ State graph history | ⚠️ Session logs (not agent-semantic) | ⚠️ Function logs | ⚠️ Workspace history |
| **Human-in-the-loop** | ⚠️ On roadmap (`approval.py`) | ⚠️ Manual integration | ✅ Core design pattern | ⚠️ Interrupt nodes in graph | ❌ None | ❌ None | ❌ None |
| **MCP support** | ⚠️ On roadmap (MCP server + client adapters) | ✅ Native | ✅ Native | ✅ Via LangChain tools | ❌ Not applicable | ❌ Not applicable | ❌ Not applicable |
| **Zero external dependencies** | ✅ Pure Python stdlib | ❌ Heavy (pydantic, litellm, openai, etc.) | ❌ Heavy (docker, pydantic, etc.) | ❌ Heavy (LangChain ecosystem) | ❌ Requires E2B SDK + cloud account | ❌ Requires Modal SDK + cloud account | ❌ Requires Docker + cloud/K8s |
| **Self-hostable** | ✅ Fully — it's a library | ✅ Yes (runs locally) | ✅ Yes (runs locally) | ✅ Yes (runs locally) | ⚠️ Enterprise tier only (AWS BYOC) | ❌ No self-hosting option | ⚠️ Helm charts available; complex setup |
| **Codebase complexity** | ✅ Small, auditable codebase | ❌ Large; hard to audit | ❌ Large; heavy abstractions | ❌ Large; LangChain dependency tree | ❌ Managed service; closed-source core | ❌ Managed service; closed-source core | ❌ Managed service; partially open |
| **LLM provider support** | ✅ Provider-agnostic by design | ✅ Multi-provider (litellm) | ✅ Multi-provider | ✅ Multi-provider (LangChain) | ⚠️ Agnostic (you bring your LLM) | ⚠️ Agnostic (you bring your LLM) | ⚠️ Agnostic (you bring your LLM) |
| **Multi-agent orchestration** | ✅ Supervision tree hierarchy | ✅ Role-based crew model | ✅ Conversational multi-agent | ✅ Graph-based state machine | ❌ Not an orchestration framework | ❌ Not an orchestration framework | ❌ Not an orchestration framework |
| **Persistence** | ⚠️ SQLite-backed memory on roadmap | ⚠️ External storage required | ⚠️ External storage required | ⚠️ LangGraph Cloud or self-managed | ✅ Session-based snapshots | ✅ Ephemeral + volume snapshots | ✅ Persistent workspaces |
| **Design-by-contract invariants** | ✅ `contracts.py` — formal pre/postconditions and invariants on agent behavior | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None |
| **Production readiness** | ⚠️ Core is solid; subprocess isolation and persistence still needed | ⚠️ Used in production but reliability critiqued | ⚠️ Non-deterministic outputs in multi-agent scenarios | ⚠️ Frequently breaking changes | ✅ Production-ready managed service | ✅ Production-ready managed service | ✅ Production-ready managed service |

*Legend: ✅ Supported / strong, ⚠️ Partial / on roadmap, ❌ Not available*

---

## Where Battousai Is Stronger

### 1. Capability-based security at the agent behavior level

E2B and Modal sandbox at the OS level — agents cannot escape the VM or container. But neither platform constrains *which tool calls* an agent is allowed to make within that sandbox, or enforces any semantic policy on agent behavior. An agent with internet access in an E2B microVM can still exfiltrate data through DNS if that capability isn't blocked at the network level. An agent with filesystem access can still read credential files.

Battousai's capability tokens operate at the agent logic level: `FILE_READ` can be scoped to a specific directory, `NETWORK` can be restricted to approved domains, and `ADMIN` operations require explicit elevation. This is complementary to OS-level isolation, not a replacement — and no competing platform provides it.

CrewAI, AutoGen, and LangGraph provide no capability model at all. Security is fully delegated to the developer. As noted in [HN discussions on agent sandboxes](https://news.ycombinator.com/item?id=47154803): "LangGraph, CrewAI, etc. assume you trust your agents."

### 2. Erlang-style supervision trees

No other agent framework or sandbox platform provides fault tolerance modeled on Erlang/OTP's supervision tree philosophy. LangGraph handles errors at graph edges — you can configure what happens when a node throws an exception — but there is no hierarchical supervisor that orchestrates the restart strategy for an entire agent subtree. If a key agent fails in LangGraph, the graph fails. In Battousai, the supervisor decides whether to restart it, restart the group, or cascade the failure in a controlled way according to the declared policy.

This is a direct answer to the pain point identified in [LangChain's State of AI Agents report](https://www.langchain.com/stateofaiagents): only 51% of companies use agents in production, with error compounding across agentic chains cited as the top barrier.

### 3. Zero external dependencies

Battousai runs as a pure Python library with no external dependencies. `pip install battousai` works offline, in air-gapped environments, in regulated industries where adding npm or pip packages requires security review, and in environments where you don't want to expose your agent workloads to a third-party cloud service.

E2B requires an E2B SDK, a cloud account, and network connectivity to E2B's infrastructure. Modal requires the Modal CLI, a Modal account, and Modal's managed cloud. Daytona requires Docker and either a cloud account or a Kubernetes cluster with Helm charts.

This is the Ollama model for agent security: local, no account required, works in 30 seconds.

### 4. Design-by-contract behavioral invariants

`contracts.py` gives agents formal behavioral guarantees analogous to Eiffel's design-by-contract: preconditions that must hold before a task runs, postconditions that must hold after, and class invariants that can never be violated regardless of what instructions the agent receives. This is the correct answer to goal hijacking attacks (ASI01): even if an attacker manipulates the agent's instructions, the invariant system provides a formal backstop.

No competing platform — framework or sandbox — has this.

### 5. Embeddable, not hosted

E2B, Modal, and Daytona are cloud services your application calls out to. There is a network round-trip for every sandboxed operation. Battousai embeds in your application as a library. This makes it suitable for:
- Air-gap deployments (classified environments, defense contractors, healthcare systems)
- Low-latency use cases where round-trips to an external sandbox are unacceptable
- Developers who want to own their security infrastructure end-to-end
- Local development without cloud credentials

---

## Where Competitors Are Stronger — An Honest Assessment

This section matters. Building trust requires honesty about where Battousai falls short today.

### E2B: Real microVM isolation is genuinely stronger

E2B uses [Firecracker microVMs](https://northflank.com/blog/e2b-vs-modal) — the same technology AWS uses for Lambda. Each agent session runs in a hardware-isolated virtual machine. Even a kernel exploit in the agent's process cannot escape the VM. This is a categorically stronger isolation guarantee than Battousai's current in-process model, and it will remain stronger than subprocess isolation with seccomp even after Battousai's roadmap is complete.

**The honest comparison:** Battousai's capability model operates at a higher semantic level than E2B's isolation, and the two are genuinely complementary. But if your threat model includes "agent achieves code execution and tries to break out," E2B's microVM provides a more robust OS-level containment boundary today. E2B's limitation is that it doesn't constrain agent behavior *within* the sandbox — Battousai's limitation is that the sandbox itself isn't yet OS-enforced.

### CrewAI: Better developer experience for simple use cases

CrewAI's role-based model — define agents as "roles" (Researcher, Writer, Analyst) with natural-language descriptions, assemble them into a "crew" — is extremely accessible. The time from `pip install crewai` to a working multi-agent pipeline is roughly 15 minutes. The metaphor maps well to common use cases. [CrewAI has ~44K GitHub stars](https://hunted.space/product/crewai) for good reason.

**The honest comparison:** For a developer who wants to build a content generation pipeline or a research assistant and doesn't have complex security requirements, CrewAI will get them to a working prototype faster. Battousai's explicit capability model and supervision tree configuration require more upfront investment. Battousai's value proposition increases as the stakes increase and as the deployment environment requires accountability.

### LangGraph: More community adoption and ecosystem

LangGraph has [~85K GitHub stars](https://www.datacamp.com/tutorial/crewai-vs-langgraph-vs-autogen) and the full LangChain ecosystem behind it — hundreds of pre-built integrations, extensive documentation, a large community, and LangSmith for observability. When you hit an unusual problem with LangGraph, there is a high probability someone on the internet has hit it before and written about it.

**The honest comparison:** Battousai is new. The community is small, the documentation is in progress, and when you hit an edge case you are more likely to be the first person to hit it. For teams that need to move fast on established patterns with community backing, LangGraph has a real advantage. Battousai trades ecosystem breadth for architectural correctness and security properties.

### AutoGen/AG2: Best-in-class human-in-the-loop patterns

AutoGen was designed from the ground up around conversational multi-agent patterns with human-in-the-loop as a core concept. Its `UserProxyAgent` pattern is the most mature implementation of human oversight in any open-source agent framework. For use cases where the human is continuously in the loop and wants to have a conversation with the agent system, AutoGen's UX is hard to beat.

**The honest comparison:** AutoGen's non-determinism makes it difficult to deploy in production scenarios that require predictable, auditable behavior. But for research, rapid prototyping, and human-in-the-loop workflows, it is excellent. Battousai's planned `approval.py` will provide human oversight, but it will not replicate AutoGen's conversational model.

### Modal: Best production infrastructure story

Modal's execution model — write Python, deploy instantly, auto-scale, pay per millisecond — is genuinely impressive. The developer experience for deploying agents to production with Modal is better than any self-hosted alternative. Their observability tooling, cold start performance, and persistence model are polished.

**The honest comparison:** Modal has no self-hosting option and runs on Modal's infrastructure. For teams with data residency requirements, compliance constraints, or simply a preference for owning their infrastructure, Modal is not an option. But for teams without those constraints who want the best managed agent execution environment, Modal is strong competition.

### Daytona: Git and devcontainer integration

Daytona's developer workspace model — Git-native, devcontainer-compatible, persistent workspaces — is extremely popular with teams that want their agent's execution environment to match their development environment. The ability to give an agent a workspace that is a clone of your repo and have it work the same way a developer would is a compelling DX story.

**The honest comparison:** Battousai has no developer workspace story. There is no integration with Git, no devcontainer support, and no concept of a persistent workspace tied to a project. This is a gap for teams building coding agents or agents that need to work within a development environment context.

---

## When to Use Battousai vs. Alternatives

### Use Battousai when:

- **Security is a primary requirement.** You need auditable, enforceable policies on what agents can and cannot do — not just sandboxing but behavioral contracts.
- **Air-gap or self-hosted deployment is required.** Regulated industries (finance, healthcare, defense), enterprises with data residency requirements, or any environment where agents cannot call out to external services.
- **Production reliability matters.** Your agentic workflow needs predictable failure behavior, not crash-and-fail. The supervision tree model is the right answer for production workloads.
- **You want to own your security primitives.** Rather than trusting a managed sandbox service's security model, you want to audit and understand exactly what isolation guarantees your agents operate under.
- **Your threat model includes insider threats or compromised agents.** Capability tokens and design-by-contract invariants are specifically designed for environments where you can't fully trust the agent itself.
- **You need a zero-dependency library.** Adding E2B or Modal to a project means adding cloud accounts, API keys, and network dependencies. `pip install battousai` works everywhere.

### Use CrewAI when:

- Your use case maps naturally to "roles working together" (content teams, research pipelines, customer support flows)
- Developer experience and time-to-working-demo are the top priority
- You don't have significant security or isolation requirements
- You want a large community and many pre-built examples to reference

### Use LangGraph when:

- You need fine-grained control over agent flow with explicit state management
- You are already in the LangChain ecosystem and want its integrations
- You need cyclical workflows (agents that can loop back, revisit steps, branch dynamically)
- Community support and ecosystem breadth are important

### Use AutoGen/AG2 when:

- Human-in-the-loop is the central pattern of your application
- You are prototyping or doing research where non-determinism is acceptable
- You want a conversational interface to your multi-agent system

### Use E2B when:

- You need the strongest available OS-level isolation (Firecracker microVM)
- You are running untrusted or user-supplied code (e.g., a coding assistant that executes code written by end users)
- Cloud deployment is acceptable and you want a managed service
- The threat model includes attempts to escape the execution environment at the OS level

### Use Modal when:

- You want production-grade managed infrastructure with auto-scaling and pay-per-use
- Data residency and self-hosting are not requirements
- You want the best cloud-native agent execution experience

### Use Daytona when:

- You are building coding agents or developer-tooling agents
- Git integration and devcontainer support are required
- You want persistent, project-scoped agent workspaces

---

## What's on the Roadmap to Close Gaps

Battousai's gaps are known, specific, and addressable. None of them require architectural rethinking — the core design is correct and the gaps are implementation work against the existing architecture.

### Gap 1: Real process isolation (addresses the E2B comparison)

**Target:** subprocess isolation with seccomp-bpf on Linux and sandbox-exec on macOS, with `capabilities.py` driving the OS-level policy.

**Result:** Battousai will offer OS-level enforcement in addition to its semantic capability model — two layers of defense, neither of which any competitor currently provides simultaneously. The claim becomes: "Battousai's capability model enforces at the agent behavior level AND the OS kernel level. Agents cannot read files they're not permitted to read, period."

**Effort:** High. This is the highest-effort item but the single biggest credibility gap. Target: before any production-readiness claims.

### Gap 2: Human-in-the-loop approval workflow (addresses the AutoGen comparison)

**Target:** `approval.py` module — checkpoints integrated with the capability model, risk-tiered (high-impact calls pause; low-risk batched), CLI + webhook + web UI approval channels, full audit trail.

**Result:** Battousai will offer more structured and auditable human oversight than AutoGen's conversational model — not a chat interface, but a formal approval protocol with accountability.

**Effort:** Medium. Natural extension of the existing capability model.

### Gap 3: SQLite persistence and memory integrity (addresses persistence gap)

**Target:** SQLite-backed memory with hash-chain integrity, TTL, and versioning.

**Result:** Persistent agent memory with tamper-evident audit trail. No competitor offers tamper detection on agent memory.

**Effort:** Low (SQLite is stdlib). The integrity mechanism is a week of work.

### Gap 4: MCP support (addresses ecosystem gap)

**Target:** MCP server and client adapters — expose Battousai's tool registry as a standard MCP server; allow agents to connect to external MCP servers with capability enforcement on incoming tool calls.

**Result:** Battousai becomes the secure runtime for the MCP ecosystem, which is now the [industry-standard tool integration protocol](https://en.wikipedia.org/wiki/Model_Context_Protocol) adopted by OpenAI, Google DeepMind, Microsoft, and virtually every major AI platform. The positioning: "Battousai is the only MCP runtime with built-in capability enforcement and audit logging."

**Effort:** Medium. Pure stdlib JSON-RPC + HTTP; no external dependencies required.

### Gap 5: Developer experience and documentation (addresses CrewAI/LangGraph comparison)

**Target:** A compelling self-contained demo (`demo.py`) that shows Battousai blocking a dangerous agent action in under 10 seconds; comprehensive docs; examples; a README that answers "why should I care" in 30 seconds.

**Result:** Lowers the barrier to evaluation for developers who currently choose CrewAI or LangGraph for their ease-of-onboarding.

**Effort:** Low. The functionality exists; this is packaging and documentation.

---

*Last updated: March 6, 2026. Sources: [Northflank: Daytona vs E2B](https://northflank.com/blog/daytona-vs-e2b-ai-code-execution-sandboxes), [Northflank: E2B vs Modal](https://northflank.com/blog/e2b-vs-modal), [Superagent AI Code Sandbox Benchmark 2026](https://www.superagent.sh/blog/ai-code-sandbox-benchmark-2026), [DataCamp: CrewAI vs LangGraph vs AutoGen](https://www.datacamp.com/tutorial/crewai-vs-langgraph-vs-autogen), [LangChain State of AI Agents](https://www.langchain.com/stateofaiagents), [HN: Sandboxes won't save you](https://news.ycombinator.com/item?id=47154803), [HN: Ask HN: The new wave of AI agent sandboxes](https://news.ycombinator.com/item?id=47254841).*
