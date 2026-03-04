# LLM Integration

The `llm.py` module connects Battousai agents to language models. The key insight: **an agent's memory space is its context window** — long-term memory becomes the system prompt, short-term memory becomes recent conversation turns.

---

## Architecture

```
LLMRouter
    │
    ├── MockLLMProvider     (deterministic, no API key)
    ├── OpenAIProvider      (template; requires requests + API key)
    └── AnthropicProvider   (template; requires requests + API key)

ContextWindow
    │
    └── Maps AgentMemorySpace → List[LLMMessage]

LLMAgent (subclass of Agent)
    │
    ├── On each tick:
    │   1. read_inbox() → format as user message
    │   2. ContextWindow.build(memory_snapshot) → messages list
    │   3. LLMRouter.complete(messages) → LLMResponse
    │   4. _parse_actions(response.content) → List[action_dict]
    │   5. Execute each action via syscalls
    │   6. Store response to SHORT_TERM memory as a turn
```

---

## Core Data Types

### `LLMMessage`

```python
@dataclass
class LLMMessage:
    role: str     # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]
```

Mirrors the role/content structure used by OpenAI, Anthropic, and most modern chat APIs.

### `LLMResponse`

```python
@dataclass
class LLMResponse:
    content: str               # The model's text output
    tokens_used: int           # Total tokens consumed (prompt + completion)
    model: str                 # Identifier of the model
    metadata: Dict[str, Any]   # Provider-specific extras (finish_reason, etc.)
```

---

## `LLMProvider` Interface

Any LLM backend must implement two methods:

```python
from abc import ABC, abstractmethod
from battousai.llm import LLMProvider, LLMMessage, LLMResponse

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages: List[LLMMessage], **kwargs) -> LLMResponse: ...

    @abstractmethod
    def embed(self, text: str) -> List[float]: ...

    @property
    def provider_name(self) -> str: ...
```

---

## `MockLLMProvider`

A deterministic provider for testing and demos — no API key or network access required:

```python
from battousai.llm import MockLLMProvider

provider = MockLLMProvider(model_name="mock-gpt-1")

response = provider.complete([
    LLMMessage(role="system", content="You are a research assistant."),
    LLMMessage(role="user", content="Summarize quantum computing."),
])
print(response.content)
# "Summary of key points: ..."
```

Responses are selected by keyword matching — the template with the most matching keywords wins. Variation suffixes are added on repeated calls to the same template.

`embed(text)` returns a 26-dimensional character-frequency vector (for structural compatibility — not semantic similarity).

---

## `OpenAIProvider` Template

```python
from battousai.llm import OpenAIProvider

class LiveOpenAIProvider(OpenAIProvider):
    def _request(self, endpoint: str, payload: dict) -> dict:
        import requests
        resp = requests.post(
            self.base_url + endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

# Usage
provider = LiveOpenAIProvider(
    api_key="sk-...",
    model="gpt-4o",              # default
    base_url="https://api.openai.com/v1",  # default
)
```

The base class handles request serialization and response parsing. You only need to implement `_request()`.

---

## `AnthropicProvider` Template

```python
from battousai.llm import AnthropicProvider

class LiveAnthropicProvider(AnthropicProvider):
    def _request(self, endpoint: str, payload: dict) -> dict:
        import requests
        resp = requests.post(
            self.base_url + endpoint,
            json=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

provider = LiveAnthropicProvider(
    api_key="sk-ant-...",
    model="claude-3-5-sonnet-20241022",  # default
)
```

Anthropic differences from OpenAI:
- System prompt is passed as a top-level `"system"` field
- Response structure uses `content[0].text`
- No embedding endpoint (`embed()` raises `NotImplementedError`)

---

## `LLMRouter`

Routes requests to the appropriate provider with fallback support:

```python
from battousai.llm import LLMRouter, MockLLMProvider

router = LLMRouter()
router.register_provider("mock", MockLLMProvider())
router.register_provider("openai", LiveOpenAIProvider(api_key="sk-..."))
router.set_default("openai")
router.set_fallback(["mock"])   # try mock if openai fails

# Complete
response = router.complete(messages, preferred_provider="openai")

# Embed
vector = router.embed("hello world")

# Stats
stats = router.stats()
# {"providers": [...], "default": "openai", "total_requests": 42, "total_tokens": 8192, ...}
```

### Convenience Factory

```python
from battousai.llm import create_mock_router

router = create_mock_router(model_name="mock-gpt-1")
# Creates LLMRouter with MockLLMProvider registered as default
```

---

## `ContextWindow`

Converts agent memory to an ordered `List[LLMMessage]`:

```python
from battousai.llm import ContextWindow

ctx = ContextWindow(max_turns=20)
messages = ctx.build(
    memory_snapshot=agent_memory.snapshot(),
    extra_user_message="New task: analyse the data",
)
```

Memory key conventions:

| Key Pattern | Role | Notes |
|---|---|---|
| `"system_prompt"` | `system` | Stored as `LONG_TERM`; becomes the system message |
| `"fact_*"` | `system` (appended) | `LONG_TERM` facts appended to system prompt |
| `"turn_NNNN_user"` | `user` | `SHORT_TERM` conversation turns |
| `"turn_NNNN_assistant"` | `assistant` | `SHORT_TERM` model responses |
| `"tool_result_*"` | `assistant` | Tool execution results |

Turns older than `max_turns` are automatically trimmed (oldest first).

---

## `LLMAgent`

A ready-to-use `Agent` subclass that replaces hard-coded `think()` logic with LLM inference:

```python
from battousai.llm import LLMAgent, create_mock_router

router = create_mock_router()

kernel.boot()
agent_id = kernel.spawn_agent(
    LLMAgent,
    name="ResearchBot",
    priority=5,
    llm_router=router,
    provider_name="mock",          # which registered provider to use
    system_prompt="You are a research assistant. Use tools to gather information.",
    max_turns=20,                  # context window size
)
kernel.run(ticks=10)
```

### Constructor Parameters

| Parameter | Default | Description |
|---|---|---|
| `name` | `"LLMAgent"` | Agent name |
| `priority` | `5` | Scheduler priority |
| `llm_router` | `None` | Configured `LLMRouter` instance |
| `provider_name` | `None` | Specific provider to use (overrides default) |
| `system_prompt` | `"You are a helpful AI agent running inside Battousai."` | Initial system instructions |
| `max_turns` | `20` | Max conversation turns to keep in context |

### Cognitive Loop (per tick)

```
1. Drain inbox → format as "user" turn string
2. Get memory snapshot → build ContextWindow
3. Call LLMRouter.complete(messages)
4. Parse [ACTION:...] tags from response
5. Execute each action via syscalls
6. Store response to SHORT_TERM memory as turn_NNNN_assistant
```

---

## Action Format

The LLM communicates intent through structured tags embedded in its response text:

```
[ACTION:SEND target_agent_id] Message content to send

[ACTION:TOOL tool_name] {"arg_key": "arg_value"}

[ACTION:WRITE /path/to/file] File content here...

[ACTION:SPAWN AgentName priority] System prompt for new agent

[ACTION:THINK] Internal reasoning or analysis text
```

Any response without a recognized tag is treated as `THINK` (logged but no action taken).

### Action Execution

| Action | Syscall |
|---|---|
| `SEND target_id` | `send_message(target_id, MessageType.CUSTOM, payload)` |
| `TOOL tool_name` | `access_tool(tool_name, args)` |
| `WRITE /path` | `write_file(path, content)` |
| `SPAWN AgentName priority` | `spawn_agent(LLMAgent, agent_name, priority, system_prompt=...)` |
| `THINK` | `mem_write(f"turn_{n}_think", thought, SHORT_TERM, ttl=max_turns)` |

---

## Example: LLM-Powered Research Agent

```python
from battousai.kernel import Kernel
from battousai.llm import LLMRouter, MockLLMProvider, LLMAgent

# Setup
kernel = Kernel(max_ticks=20)
kernel.boot()

router = LLMRouter()
router.register_provider("mock", MockLLMProvider())
router.set_default("mock")

# Spawn the LLM agent
agent_id = kernel.spawn_agent(
    LLMAgent,
    name="Researcher",
    priority=4,
    llm_router=router,
    system_prompt=(
        "You are a research agent. Use the web_search tool to gather information "
        "and write findings to /shared/results/. "
        "After 3 searches, summarize your findings."
    ),
)

# Seed with initial task via IPC
from battousai.ipc import MessageType
kernel.ipc.create_message(
    sender_id="kernel",
    recipient_id=agent_id,
    message_type=MessageType.TASK,
    payload={"task": "Research quantum computing applications"},
    timestamp=0,
)

kernel.run()
```

---

## Related Pages

- [Agent API](api.md) — base Agent class that LLMAgent extends
- [Tools: Built-in](../tools/builtin.md) — tools available via `[ACTION:TOOL ...]`
- [Tools: Extended](../tools/extended.md) — additional tools for LLM agents
- [Custom Agents](custom.md) — how to combine LLMAgent with custom logic
