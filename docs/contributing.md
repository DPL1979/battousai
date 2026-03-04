# Contributing to Battousai

Thank you for your interest in contributing to Battousai! This guide covers everything you need to get started.

---

## Development Environment

**Requirements:** Python 3.10 or later. No external dependencies — Battousai is pure standard library.

```bash
git clone https://github.com/battousai-project/battousai.git
cd battousai
python --version  # confirm 3.10+
```

That's it. No `pip install`, no virtual environment required (though using one is always a good practice).

---

## Running the Tests

Battousai uses Python's built-in `unittest` framework:

```bash
python -m unittest discover -s tests -v
```

All tests are in the `tests/` directory. The test suite covers the kernel, scheduler, IPC, memory manager, filesystem, tools, LLM layer, supervision trees, capabilities, schemas, network stack, federation, HAL, and evolution engine (150+ tests).

---

## Code Style

Battousai follows a small set of conventions designed to keep the codebase approachable and dependency-free:

- **Type hints everywhere.** Every function signature should be fully annotated. Use `from __future__ import annotations` at the top of new files if needed for forward references.
- **Docstrings on public classes and methods.** One-line docstrings are fine for simple methods; multi-line docstrings should follow the Google style (Args / Returns / Raises sections).
- **Pure Python — no external dependencies.** All modules must work with `python -m battousai.main` on a fresh Python 3.10+ install. Do not add `import requests`, `import numpy`, or any third-party library.
- **No global mutable state** outside of the `Kernel` object. Singletons are a code smell; pass references explicitly.
- **Keep modules focused.** If a new file exceeds ~1,500 lines, consider splitting it.

---

## Adding a New Tool

Tools are registered in `battousai/tools_extended.py`. To add a new tool:

1. Open `battousai/tools_extended.py`.
2. Define a function that accepts keyword arguments and returns a result dict:

    ```python
    def my_new_tool(**kwargs) -> dict:
        """One-line description of what the tool does.

        Args:
            param_name: Description of param.

        Returns:
            dict with result, success, error keys.
        """
        param = kwargs.get("param_name", "default")
        try:
            result = {"output": f"processed: {param}"}
            return {"result": result, "success": True, "error": ""}
        except Exception as e:
            return {"result": None, "success": False, "error": str(e)}
    ```

3. Register it inside `register_extended_tools()`:

    ```python
    def register_extended_tools(tool_manager, filesystem=None):
        # ... existing registrations ...
        tool_manager.register(ToolSpec(
            name="my_new_tool",
            description="Converts input to uppercase and returns its length.",
            callable=my_new_tool,
            rate_limit=10,
            rate_window=10,
        ))
    ```

4. Add at least one test in `tests/test_tools_extended.py`.

---

## Adding a New Agent Type

All agent types subclass `Agent` from `battousai/agent.py`. To add a new agent:

1. Create or extend a file (e.g. `battousai/agent.py` or a new module like `battousai/my_agents.py`).
2. Subclass `Agent` and implement `think()`:

    ```python
    from battousai.agent import Agent

    class MyNewAgent(Agent):
        """Brief description of what this agent does."""

        def __init__(self, name: str = "MyNewAgent", priority: int = 5):
            super().__init__(name=name, priority=priority)

        def on_spawn(self) -> None:
            """Called once when the agent is first scheduled."""
            self.log("MyNewAgent online.")

        def think(self, tick: int) -> None:
            """Main cognitive loop — called every tick."""
            messages = self.read_inbox()
            for msg in messages:
                self.log(f"Received: {msg.message_type.name}")

            # ... agent logic ...

            self.yield_cpu()

        def on_terminate(self) -> None:
            """Called when the agent is about to be killed."""
            self.log("Shutting down.")
    ```

3. Spawn it from the kernel or another agent:

    ```python
    kernel.spawn_agent(MyNewAgent, name="MyNewAgent", priority=5)
    ```

4. Add tests in `tests/test_agent.py` or a new `tests/test_my_agents.py`.

---

## Submitting a Pull Request

1. **Fork** the repository and create a feature branch off `main`:

    ```bash
    git checkout -b feature/my-feature
    ```

2. **Write tests** for your changes. PRs without tests will not be merged.

3. **Run the full test suite** and confirm everything passes:

    ```bash
    python -m unittest discover -s tests -v
    ```

4. **Keep commits focused.** One logical change per commit. Write descriptive commit messages in the imperative mood (`Add X`, `Fix Y`, `Refactor Z`).

5. **Open a Pull Request** against `main`. Fill out the PR template:
    - What does this PR do?
    - Which issue does it close (if any)?
    - How was it tested?

6. A maintainer will review within a few days. Be responsive to feedback — PRs that go stale for 30 days without activity may be closed.

---

## Reporting Issues

When opening a bug report, please include:

- **Python version** (`python --version`)
- **Operating system**
- **Steps to reproduce** — a minimal script that triggers the issue
- **Expected behavior** vs. **actual behavior**
- **Full traceback** if applicable

For feature requests, describe the use case first — what problem are you trying to solve? — before proposing a solution.

---

## Questions?

Open a [GitHub Discussion](https://github.com/battousai-project/battousai/discussions) for design questions, usage help, or ideas you'd like to socialise before committing to an implementation.
