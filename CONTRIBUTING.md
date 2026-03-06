# Contributing to Battousai

Thank you for your interest in contributing to Battousai. This document covers everything you need to get started.

---

## Development Environment

**Requirements:** Python 3.10 or later. Nothing else — Battousai has zero external dependencies.

```bash
git clone https://github.com/DPL1979/battousai.git
cd battousai
python --version  # confirm 3.10+
```

No virtual environment required (there's nothing to install), but you can use one if you prefer:

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## Running Tests

```bash
# Run the full test suite (544 tests, ~0.2 seconds)
python -m unittest discover -s tests -v

# Run a specific test file
python -m unittest tests.test_contracts -v
python -m unittest tests.test_providers -v
python -m unittest tests.test_isolation -v

# Run a specific test class
python -m unittest tests.test_contracts.TestSafetyEnvelope -v

# Run a single test
python -m unittest tests.test_contracts.TestSafetyEnvelope.test_send_message_blocked_above_limit -v
```

All tests must pass before submitting a PR. The test suite runs in under a second — there's no reason not to run it often.

---

## Code Style

### The Rules

1. **Type hints everywhere.** Every function signature, every return type. Use `from __future__ import annotations` at the top of every module.

2. **Docstrings on public classes and methods.** Explain what it does, not how. Include design rationale for non-obvious decisions.

3. **Pure Python, zero external dependencies.** This is non-negotiable. If you need HTTP, use `urllib.request`. If you need JSON, use `json`. If you need persistence, use `sqlite3`. If the stdlib can't do it, we probably don't need it.

4. **No global mutable state outside the Kernel.** All state flows through the kernel's syscall interface. This is what makes capability enforcement possible.

5. **Tests for everything.** Every new module gets a corresponding `tests/test_<module>.py`. Aim for both happy paths and edge cases.

### Naming Conventions

- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: prefix with `_`

### File Organization

```
battousai/
├── __init__.py          # Version, public API exports
├── kernel.py            # Central coordinator
├── agent.py             # Agent base classes
├── capabilities.py      # Security model
├── contracts.py         # Behavioral verification
├── supervisor.py        # Fault tolerance
├── ...                  # Other modules
tests/
├── test_kernel.py       # One test file per module
├── test_agent.py
├── ...
examples/
├── quickstart.py        # Minimal working example
├── security_demo.py     # Security features showcase
docs/
├── owasp-alignment.md   # OWASP mapping
├── comparison.md        # Competitor comparison
```

---

## Architecture Overview

Before contributing, understand the core design principles:

### 1. Everything Goes Through Syscalls
Agents don't access resources directly. They make syscalls to the kernel, which checks capability tokens before granting access. This is the fundamental security invariant — **no ambient authority**.

### 2. Capability Tokens Gate Every Action
File reads, file writes, network calls, tool invocations, memory writes, agent spawning — everything requires a capability token. If an agent doesn't have the right token, the action is blocked and logged.

### 3. Supervision Trees Handle Failures
When an agent fails, its supervisor decides what to do (restart it, restart all siblings, or escalate). This follows Erlang's "let it crash" philosophy.

### 4. Contracts Verify Behavior
Agents declare preconditions, postconditions, and invariants. The `ContractMonitor` verifies these at runtime. The `SafetyEnvelope` provides hard limits that override everything else.

---

## How to Contribute

### Reporting Bugs

Open an issue with:
- Python version and OS
- Minimal reproduction case
- Expected vs actual behavior
- Full traceback

### Suggesting Features

Open an issue describing:
- The problem you're trying to solve
- Why existing functionality doesn't cover it
- Your proposed approach (if you have one)

### Submitting Code

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write your code + tests
4. Run the full test suite: `python -m unittest discover -s tests -v`
5. Commit with a clear message
6. Open a PR against `main`

### Good First Issues

Look for issues labeled `good first issue`. These are well-scoped, well-documented, and a great way to learn the codebase.

Areas where contributions are especially welcome:
- **MCP adapter** — implementing MCP server/client using stdlib
- **Human-in-the-loop approval** — pause/approve workflow for high-risk actions
- **Memory integrity** — hash-chain verification for agent memory writes
- **Documentation** — tutorials, examples, architecture guides
- **Security auditing** — finding and responsibly reporting vulnerabilities

---

## PR Review Process

1. All tests must pass
2. No external dependencies introduced
3. Type hints on all new code
4. Docstrings on public APIs
5. At least one reviewer approves

PRs that break any of these will be asked for revisions before merge.

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
