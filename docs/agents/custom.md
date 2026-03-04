# Custom Agents

This guide walks through practical patterns for building your own Battousai agents, from a minimal skeleton to multi-step workflows.

---

## Minimal Agent

Every custom agent subclasses `Agent` and implements `think()`:

```python
from battousai.agent import Agent

class MinimalAgent(Agent):
    def __init__(self):
        super().__init__(name="Minimal", priority=5)

    def think(self, tick: int) -> None:
        self.log(f"Running at tick {tick}")
        self.yield_cpu()
```

Spawn it:

```python
from battousai.kernel import Kernel

kernel = Kernel(max_ticks=10)
kernel.boot()
kernel.spawn_agent(MinimalAgent, name="Minimal", priority=5)
kernel.run()
```

---

## Monitoring Agent

Monitor system health, react to alerts, and publish metrics:

```python
from battousai.agent import Agent
from battousai.ipc import MessageType, BROADCAST_ALL

class SystemHealthAgent(Agent):
    """Samples health metrics every 5 ticks and alerts on anomalies."""

    def __init__(self, alert_threshold: int = 10):
        super().__init__(
            name="HealthMonitor",
            priority=7,         # low priority — runs after workers
            memory_allocation=512,
        )
        self._alert_threshold = alert_threshold

    def on_spawn(self) -> None:
        self.log(f"Health monitor online. Alert threshold: {self._alert_threshold} agents")
        self.mem_write("sample_count", 0)
        self.mem_write("anomaly_count", 0)

    def think(self, tick: int) -> None:
        # Read any broadcasts
        for msg in self.read_inbox():
            if msg.message_type == MessageType.BROADCAST:
                self.log(f"Broadcast: {str(msg.payload)[:80]}")

        # Sample every 5 ticks
        if tick % 5 == 0 and tick > 0:
            status = self.get_status()
            if status.ok:
                metrics = status.value
                agent_count = metrics["agent_count"]
                msgs_sent = metrics["ipc_stats"]["total_sent"]

                sample = {"tick": tick, "agents": agent_count, "msgs": msgs_sent}

                # Store sample
                count = self.mem_read("sample_count") or 0
                self.mem_write(f"sample_{count}", sample)
                self.mem_write("sample_count", count + 1)

                # Publish to bulletin board
                self.syscall("publish_topic", topic="health", value=sample)

                # Alert on high agent count
                if agent_count > self._alert_threshold:
                    anomalies = self.mem_read("anomaly_count") or 0
                    self.mem_write("anomaly_count", anomalies + 1)
                    self.log(f"ALERT: {agent_count} agents (threshold={self._alert_threshold})")
                    self.send_message(
                        BROADCAST_ALL,
                        MessageType.STATUS,
                        {"alert": "high_agent_count", "count": agent_count},
                    )

        self.yield_cpu()
```

---

## Data Processing Agent

A stateful agent that processes data in multiple phases:

```python
from battousai.agent import Agent
from battousai.ipc import MessageType

class DataProcessorAgent(Agent):
    """Receives data batches, processes them, and writes results."""

    def __init__(self):
        super().__init__(name="DataProcessor", priority=4, memory_allocation=1024)
        self._phase = "IDLE"
        self._batch = []
        self._processed = []

    def on_spawn(self) -> None:
        self.log("DataProcessor ready")
        self.mem_write("status", "idle")

    def think(self, tick: int) -> None:
        # Accept incoming data
        for msg in self.read_inbox():
            if msg.message_type == MessageType.TASK:
                payload = msg.payload
                if isinstance(payload, dict) and "data" in payload:
                    self._batch = payload["data"]
                    self._phase = "PROCESSING"
                    self.mem_write("status", "processing")
                    self.mem_write("batch_size", len(self._batch))
                    self.log(f"Received batch of {len(self._batch)} items")

        # Process one item per tick (non-blocking pattern)
        if self._phase == "PROCESSING" and self._batch:
            item = self._batch.pop(0)
            result = self._process_item(item)
            self._processed.append(result)

            if not self._batch:
                self._phase = "REPORTING"
                self.mem_write("status", "done")
                self.log(f"Processing complete. {len(self._processed)} results.")

        # Report when done
        elif self._phase == "REPORTING":
            self.write_file(
                f"/agents/{self.agent_id}/workspace/results.txt",
                "\n".join(str(r) for r in self._processed),
            )
            # Report back to coordinator
            agents = self.list_agents()
            coord = next((a for a in agents if "coordinator" in a), None)
            if coord:
                self.send_message(
                    coord,
                    MessageType.RESULT,
                    {"results": self._processed, "count": len(self._processed)},
                )
            self._phase = "IDLE"

        self.yield_cpu()

    def _process_item(self, item):
        # Use calculator for numeric processing
        if isinstance(item, (int, float)):
            r = self.use_tool("calculator", expression=f"sqrt({abs(item)})")
            return float(r.value) if r.ok else 0.0
        return str(item).upper()

    def on_terminate(self) -> None:
        self.log(f"Processed {len(self._processed)} total items")
```

---

## Multi-Step Workflow Agent

An agent that orchestrates a multi-step research workflow using a state machine:

```python
from battousai.agent import Agent
from battousai.ipc import MessageType

class ResearchOrchestrator(Agent):
    """Runs a multi-step research pipeline: search → analyze → summarize."""

    STEPS = ["search", "analyze", "summarize", "done"]

    def __init__(self, topic: str = "machine learning"):
        super().__init__(name="Orchestrator", priority=3, memory_allocation=512)
        self._topic = topic
        self._step_idx = 0
        self._search_results = []

    def on_spawn(self) -> None:
        self.log(f"Starting research on: {self._topic}")
        self.mem_write("topic", self._topic)
        self.mem_write("step", "search")

    def think(self, tick: int) -> None:
        step = self.STEPS[self._step_idx]

        if step == "search":
            self._do_search()
        elif step == "analyze":
            self._do_analyze()
        elif step == "summarize":
            self._do_summarize()
        elif step == "done":
            self.log("Research complete.")

        self.yield_cpu()

    def _do_search(self) -> None:
        r = self.use_tool("web_search", query=self._topic)
        if r.ok:
            self._search_results = r.value.get("results", [])
            snippets = [res["snippet"] for res in self._search_results]
            self.mem_write("raw_snippets", snippets)
            self.log(f"Search complete: {len(snippets)} snippets")
            self._advance_step()

    def _do_analyze(self) -> None:
        snippets = self.mem_read("raw_snippets") or []
        combined_text = " ".join(snippets)
        r = self.use_tool("text_analyzer", text=combined_text)
        if r.ok:
            analysis = r.value
            self.mem_write("analysis", analysis)
            self.log(
                f"Analysis: {analysis['word_count']} words, "
                f"sentiment={analysis['sentiment']}"
            )
            self._advance_step()

    def _do_summarize(self) -> None:
        snippets = self.mem_read("raw_snippets") or []
        analysis = self.mem_read("analysis") or {}
        summary = {
            "topic": self._topic,
            "snippets_found": len(snippets),
            "word_count": analysis.get("word_count", 0),
            "sentiment": analysis.get("sentiment", "neutral"),
            "top_words": analysis.get("top_words", []),
        }
        self.write_file("/shared/results/research_summary.txt", str(summary))
        self.log(f"Summary written to /shared/results/research_summary.txt")
        self._advance_step()

    def _advance_step(self) -> None:
        if self._step_idx < len(self.STEPS) - 1:
            self._step_idx += 1
            self.mem_write("step", self.STEPS[self._step_idx])
```

!!! tip "One action per tick"
    The pattern of doing one unit of work per tick (one search, one analysis, etc.) keeps agents responsive to messages and allows the scheduler to share CPU time fairly. Avoid long loops inside `think()`.

---

## Agent with Child Agents

An agent that dynamically spawns children to parallelise work:

```python
from battousai.agent import Agent, WorkerAgent
from battousai.ipc import MessageType

class ParallelCoordinator(Agent):
    """Fans out work to N workers in parallel, collects results."""

    def __init__(self, queries: list):
        super().__init__(name="ParallelCoord", priority=2, memory_allocation=512)
        self._queries = queries
        self._worker_ids = []
        self._results = {}
        self._spawned = False

    def on_spawn(self) -> None:
        self.mem_write("pending_queries", self._queries)

    def think(self, tick: int) -> None:
        # First tick: spawn one worker per query
        if not self._spawned:
            for i, query in enumerate(self._queries):
                result = self.spawn_child(
                    WorkerAgent,
                    name=f"Worker-{i+1}",
                    priority=4,
                    subtask={"description": query, "queries": [query]},
                )
                if result.ok:
                    self._worker_ids.append(result.value)
                    self.log(f"Spawned {result.value} for: {query}")
            self._spawned = True

        # Collect results from workers
        for msg in self.read_inbox():
            if msg.message_type == MessageType.RESULT:
                self._results[msg.sender_id] = msg.payload
                self.log(f"Got result from {msg.sender_id}")

        # Check if all results are in
        if self._spawned and len(self._results) == len(self._worker_ids):
            self._synthesize()

        self.yield_cpu()

    def _synthesize(self) -> None:
        summary = []
        for worker_id, result in self._results.items():
            if isinstance(result, dict):
                for finding in result.get("findings", []):
                    summary.append(f"{finding['query']}: {finding['result'][:100]}")

        self.write_file("/shared/results/parallel_summary.txt", "\n".join(summary))
        self.log("Synthesis complete!")
```

---

## Tips and Best Practices

!!! tip "Always yield CPU"
    Call `self.yield_cpu()` at the end of every `think()` call. Agents that don't yield hold their state as RUNNING until end of tick, which blocks preemption.

!!! tip "One tick, one action"
    Structure `think()` as a state machine where each tick advances one step. This keeps agents responsive, prevents blocking, and makes scheduling fair.

!!! warning "No blocking operations"
    Never call `time.sleep()` or perform I/O operations inside `think()`. Battousai is single-threaded — blocking `think()` blocks the entire OS.

!!! tip "Use memory for state"
    Store persistent state in `self.mem_write()` rather than instance variables, especially for anything you want to survive a potential supervisor restart.

!!! tip "Handle None from mem_read"
    `self.mem_read(key)` returns `None` if the key doesn't exist or has expired. Always guard with `or default`:
    ```python
    count = self.mem_read("count") or 0
    ```

---

## Related Pages

- [Agent API](api.md) — full reference for all agent methods
- [LLM Integration](llm.md) — let an LLM drive agent decisions
- [Supervision Trees](supervision.md) — make agents fault-tolerant
- [Tools: Built-in](../tools/builtin.md) — tools available in custom agents
- [Tools: Extended](../tools/extended.md) — more tools for complex workflows
