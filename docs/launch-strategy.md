# Battousai Launch Strategy

**Prepared:** March 6, 2026
**Target Launch:** Tuesday, March 10, 2026 at 13:00 UTC (optimal HN window)

---

## Positioning One-Liner

> "Battousai is a lightweight Python runtime for sandboxed AI agents — capability-based security, fault-tolerant supervision, and memory isolation in under 16K lines with zero dependencies."

---

## Launch Channels (in order of execution)

### 1. Hacker News — Show HN (Primary)

**When:** Tuesday March 10, 2026 at 13:00 UTC (8:00 AM ET)
**Why this time:** [arXiv research](https://arxiv.org/html/2511.04453v1) shows 12–17 UTC window generates ~200 more stars than off-peak. Tuesday/Wednesday are highest engagement.

**Title options (A/B test mentally):**
- "Show HN: Battousai – A zero-dep Python runtime for sandboxed AI agents"
- "Show HN: Battousai – Capability-based security for AI agents in pure Python"

**Body draft:**

```
I built Battousai because every agent framework (LangGraph, CrewAI, AutoGen) assumes
you trust your agents. They run in-process, share memory, and have access to your
entire host environment.

Battousai is different. Every agent runs inside a capability-gated sandbox where file
access, network calls, tool invocations, and memory writes require explicit capability
tokens. No token, no access — blocked and logged.

What's in it:
- Capability-token security (no ambient authority)
- Erlang-style supervision trees for fault tolerance
- Design-by-Contract runtime verification
- MCP server/client with capability enforcement
- Per-agent filesystem jails with path traversal prevention
- SQLite persistence with WAL mode
- Process-level isolation
- 607 tests, 20K lines of pure Python, zero external dependencies

OWASP alignment: We mapped all 10 OWASP Agentic AI risks and Battousai addresses 8.
None of the framework competitors address any.

pip install battousai  # no Docker, no API keys, no cloud account

GitHub: https://github.com/DPL1979/battousai
Security benchmark: python examples/security_benchmark.py
```

**Engagement plan:**
- Respond to every comment within 2 hours
- Have technical depth ready for security questions
- Acknowledge limitations honestly (no seccomp yet, not production-hardened)

### 2. Reddit (Same Day, Staggered)

**Subreddits:**
- r/Python (12:00 UTC) — "I built a zero-dependency Python runtime for sandboxed AI agents"
- r/MachineLearning (14:00 UTC) — Focus on OWASP alignment angle
- r/LocalLLaMA (15:00 UTC) — Emphasize Ollama integration, local-first, no cloud
- r/AI_Agents (16:00 UTC) — "Every framework assumes you trust your agents. Here's one that doesn't."

### 3. Twitter/X (Same Day)

**Thread structure (5 tweets):**

1. "Every AI agent framework assumes you trust your agents. LangGraph, CrewAI, AutoGen — agents run in-process, share memory, access your entire host. What could go wrong? 🧵"

2. "I built Battousai — a Python runtime where every agent action requires a capability token. No token for FILE_WRITE? Blocked. No NETWORK access? Blocked. Every denied action is logged."

3. "What's inside: Erlang supervision trees, Design-by-Contract verification, MCP compatibility, per-agent filesystem jails, SQLite persistence. 607 tests. Zero external dependencies."

4. "We mapped all 10 OWASP Agentic AI risks. Battousai addresses 8 of them. The major frameworks address 0. Security benchmark: 11/11 path traversal attacks blocked."

5. "pip install battousai — no Docker, no cloud account, no API keys. Pure Python 3.10+. Try the security demo: python examples/security_benchmark.py. GitHub: [link]"

### 4. AI Engineer World's Fair CFP (Before March 30)

**Talk submission:** "We Ran the OWASP Agentic Top 10 Against 6 Agent Frameworks. Here's What Survived."

**Abstract (250 words):**

The OWASP Top 10 for Agentic Applications (2026) identifies 10 critical security risks
specific to AI agent systems. We evaluated 6 popular frameworks — LangGraph, CrewAI,
AutoGen, E2B, Modal, and Battousai — against each risk category.

The results are sobering. None of the major agent orchestration frameworks (LangGraph,
CrewAI, AutoGen) provide execution isolation, capability enforcement, or runtime behavioral
verification. The sandbox platforms (E2B, Modal) isolate at the OS level but don't constrain
agent behavior — a sandboxed agent can still read your SSH keys within its container.

Battousai takes a different approach: capability-based security at the agent behavior level.
Every tool call, file access, memory write, and network request requires an explicit capability
token. Erlang-style supervision trees handle failures. Design-by-Contract invariants verify
behavior at runtime.

This talk presents: (1) our methodology for mapping each OWASP risk to concrete framework
features, (2) a live demo of attacks that succeed against popular frameworks but fail against
capability-gated agents, (3) lessons for framework authors on what "agent security" actually
requires beyond sandboxing.

Attendees will leave with a concrete checklist for evaluating agent framework security and
an open-source tool they can use immediately.

---

## Pre-Launch Checklist

### Must Have (by March 9)
- [ ] All code pushed to GitHub (v0.3.0 tag)
- [ ] README has compelling code examples
- [ ] Security benchmark runs cleanly
- [ ] `pip install battousai` works from GitHub (`pip install git+https://github.com/DPL1979/battousai.git`)
- [ ] Examples run without errors
- [ ] OWASP alignment doc reviewed for accuracy

### Nice to Have
- [ ] GIF of security_benchmark.py output for README
- [ ] GitHub Codespace devcontainer for one-click demo
- [ ] PyPI package published
- [ ] Blog post with deeper technical narrative

---

## Key Messages (Talking Points)

1. **"The Ollama of agent security"** — pip install, works offline, zero config
2. **"Capability tokens, not sandboxes"** — Sandboxes contain blast radius. Capability tokens prevent the blast.
3. **"OWASP-aligned, competitors aren't"** — Provable, specific, differentiated
4. **"Erlang for agents"** — Supervision trees are battle-tested (Ericsson, WhatsApp)
5. **"Zero dependencies means zero supply chain risk"** — In a world where left-pad and XZ taught us dependency costs

---

## Success Metrics

| Metric | Target (Week 1) | Stretch (Month 1) |
|--------|-----------------|-------------------|
| GitHub stars | 100 | 500 |
| HN points | 50 | — |
| PyPI installs | 200 | 1,000 |
| Contributors | 3 | 10 |
| Issues filed | 10 | 50 |
| Conference CFP accepted | — | 1 |

---

## Post-Launch Priorities

1. **Respond to all GitHub issues within 24 hours** — first impression matters
2. **Label good-first-issues** — convert HN interest into contributors
3. **Weekly "office hours" on Discord/GitHub Discussions** — build community
4. **Monthly security audit updates** — publish a SECURITY.md with responsible disclosure process
5. **v0.4.0 targets:** Memory integrity hashing, human-in-the-loop approval workflow, real subprocess isolation with seccomp
