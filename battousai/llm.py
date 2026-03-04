"""
llm.py — LLM Integration Layer
================================
Provides a pluggable interface for connecting AI agents to language models.

Architecture:
    LLMProvider (abstract) — defines the interface for any LLM backend
    MockLLMProvider      — deterministic provider for testing/demos
    OpenAIProvider       — template for OpenAI API (requires `requests`)
    AnthropicProvider    — template for Anthropic API (requires `requests`)
    LLMRouter            — routes agent requests to the appropriate provider
    ContextWindow        — maps agent memory to LLM context

The key insight: an agent's memory space IS its context window.
Short-term memory maps to recent conversation, long-term memory maps
to system prompt and persistent facts.

Usage::

    from battousai.llm import LLMRouter, MockLLMProvider, LLMAgent

    # Register a provider
    router = LLMRouter()
    router.register_provider("mock", MockLLMProvider())
    router.set_default("mock")

    # Or use LLMAgent directly (requires a booted kernel)
    kernel.boot()
    agent_id = kernel.spawn_agent(
        LLMAgent,
        name="MyLLMAgent",
        provider_name="mock",
        system_prompt="You are a helpful assistant.",
    )
    kernel.run(10)
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from battousai.agent import Agent, SyscallResult
from battousai.ipc import MessageType
from battousai.memory import MemoryType

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class LLMMessage:
    """
    A single message in an LLM conversation context.

    Mirrors the role/content structure used by OpenAI, Anthropic, and most
    modern chat-completion APIs so that ContextWindow output can be fed
    directly into any provider.

    Roles:
        system    — Initial instructions / persona for the model.
        user      — Human or agent turn; input to the model.
        assistant — Previous model responses; context for the next turn.
    """
    role: str      # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Serialise to the wire format expected by most LLM APIs."""
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """
    Structured response returned by any LLMProvider.

    Fields:
        content     — The model's text output.
        tokens_used — Total tokens consumed (prompt + completion).
                      MockLLMProvider estimates this as len(content) // 4.
        model       — Identifier of the model that produced the response.
        metadata    — Provider-specific extra data (finish reason, logprobs…).
    """
    content: str
    tokens_used: int
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return (
            f"LLMResponse(model={self.model!r}, "
            f"tokens={self.tokens_used}, "
            f"content={preview!r}...)"
        )


# ---------------------------------------------------------------------------
# Abstract provider interface
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """
    Abstract base class for all LLM backends.

    Any concrete provider must implement:
        complete(messages, **kwargs) → LLMResponse
        embed(text) → List[float]

    Providers are registered with LLMRouter and selected either explicitly
    (by agent config) or automatically (by model name prefix matching).
    """

    @abstractmethod
    def complete(
        self,
        messages: List[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Request a chat completion from the model.

        Args:
            messages — Ordered list of conversation turns. The last message
                       is typically a "user" turn asking for a response.
            **kwargs — Provider-specific overrides (temperature, max_tokens…).

        Returns:
            LLMResponse with the model's reply.
        """
        ...

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Compute a vector embedding for the given text.

        Returns a list of floats representing the embedding vector.
        The dimensionality depends on the provider and model.
        """
        ...

    @property
    def provider_name(self) -> str:
        """Human-readable name for this provider (used in logs)."""
        return type(self).__name__


# ---------------------------------------------------------------------------
# Mock provider — deterministic, no external dependencies
# ---------------------------------------------------------------------------

# Keyword → response template mapping.  Each entry is (keywords_list, response).
# The MockLLMProvider scores each template by counting matching keywords.
_MOCK_RESPONSE_TEMPLATES: List[Tuple[List[str], str]] = [
    (
        ["research", "quantum", "physics", "science"],
        (
            "Based on my analysis of the available information:\n\n"
            "Quantum computing leverages superposition and entanglement to process information "
            "exponentially faster than classical computers for specific problem classes. "
            "Key milestones include Google's claimed quantum supremacy (2019) and IBM's "
            "1,121-qubit Condor processor (2023). Primary challenges remain decoherence, "
            "error correction, and scalability.\n\n"
            "[ACTION:THINK] Research task complete — key findings documented."
        ),
    ),
    (
        ["decompose", "subtask", "plan", "split", "breakdown", "task"],
        (
            "I will break this task into three parallel subtasks:\n\n"
            "1. Data gathering — collect relevant information from available sources.\n"
            "2. Analysis — process and cross-reference the gathered data.\n"
            "3. Synthesis — combine findings into a coherent summary.\n\n"
            "[ACTION:THINK] Task decomposition complete. Spawning workers.\n"
            "[ACTION:SPAWN WorkerAgent 4] You are a data gathering worker. "
            "Collect information and report back with your findings."
        ),
    ),
    (
        ["summarize", "summary", "summarise", "overview", "recap"],
        (
            "Summary of key points:\n\n"
            "• The primary objective has been addressed through systematic analysis.\n"
            "• Multiple data sources were consulted to ensure completeness.\n"
            "• Findings converge on a consistent conclusion.\n"
            "• Recommended next steps are outlined in the full report.\n\n"
            "[ACTION:THINK] Summary generated successfully."
        ),
    ),
    (
        ["code", "python", "function", "implement", "write", "program"],
        (
            "Here is the implementation:\n\n"
            "```python\ndef solution(data):\n"
            "    \"\"\"Process the input data and return results.\"\"\"\n"
            "    results = []\n"
            "    for item in data:\n"
            "        processed = item  # Apply transformation logic here\n"
            "        results.append(processed)\n"
            "    return results\n```\n\n"
            "[ACTION:THINK] Code generation complete."
        ),
    ),
    (
        ["error", "failed", "crash", "exception", "bug", "fix"],
        (
            "I've identified the issue. The error likely stems from:\n\n"
            "1. An unexpected None value in the processing pipeline.\n"
            "2. Possible race condition between concurrent operations.\n"
            "3. Missing error handling for edge cases.\n\n"
            "Recommended fix: add explicit null checks and wrap critical "
            "sections in try/except blocks with appropriate fallback logic.\n\n"
            "[ACTION:THINK] Diagnosis complete — awaiting confirmation to apply fix."
        ),
    ),
    (
        ["analyze", "analyse", "data", "metrics", "statistics", "report"],
        (
            "Analysis results:\n\n"
            "The dataset shows a clear upward trend over the observed period. "
            "Key statistical properties:\n"
            "  • Mean: approximately within normal operating range\n"
            "  • Variance: low, indicating stable behaviour\n"
            "  • Outliers: 2 anomalous data points detected (ticks 12 and 34)\n\n"
            "[ACTION:THINK] Analysis complete — writing report to filesystem.\n"
            "[ACTION:WRITE /agents/self/analysis_report.txt] "
            "Analysis complete. Trend: stable. Outliers at ticks 12, 34."
        ),
    ),
    (
        ["send", "message", "notify", "tell", "inform", "alert"],
        (
            "Understood. I will relay the information to the appropriate agent.\n\n"
            "[ACTION:THINK] Composing notification message.\n"
        ),
    ),
    (
        ["search", "find", "lookup", "query", "web"],
        (
            "Executing search query to retrieve relevant information.\n\n"
            "[ACTION:TOOL web_search] {\"query\": \"latest information on topic\"}\n\n"
            "[ACTION:THINK] Search initiated — will process results on next tick."
        ),
    ),
    (
        ["calculate", "compute", "math", "formula", "number"],
        (
            "Performing the requested calculation.\n\n"
            "[ACTION:TOOL calculator] {\"expression\": \"42 * 1024\"}\n\n"
            "[ACTION:THINK] Calculation submitted — awaiting result."
        ),
    ),
    (
        # Default fallback — always matches
        [],
        (
            "I have processed your request and here is my response:\n\n"
            "The task has been acknowledged and I am working through it "
            "systematically. I will continue to the next logical step based "
            "on the available context and constraints.\n\n"
            "[ACTION:THINK] Processing complete for this tick."
        ),
    ),
]

# A simple deterministic hash-style function to vary responses when the same
# template matches repeatedly, avoiding repetitive identical outputs.
_VARIATION_SUFFIXES = [
    "\n\nNote: proceeding with standard protocol.",
    "\n\nNote: all preconditions satisfied.",
    "\n\nNote: context window updated with latest state.",
    "\n\nNote: ready for next instruction.",
    "\n\nNote: monitoring system health in background.",
]


class MockLLMProvider(LLMProvider):
    """
    Deterministic mock LLM provider for unit testing and demos.

    Responses are selected by scoring keyword matches against the combined
    text of all input messages. The highest-scoring template is used, with
    small variations applied on repeated calls to the same template.

    Embedding is implemented as a simple bag-of-characters frequency vector
    (26 dims, one per letter a-z), normalised to unit length.  This is
    intentionally naive — it exists to make vector operations testable
    without any ML dependencies.

    Usage::

        provider = MockLLMProvider(model_name="mock-v1")
        response = provider.complete([
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Summarize quantum computing."),
        ])
        print(response.content)
    """

    def __init__(self, model_name: str = "mock-gpt-1") -> None:
        self.model_name = model_name
        self._call_count: int = 0
        self._last_template_idx: int = -1

    def complete(
        self,
        messages: List[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Return a deterministic response based on keyword matching.

        Scoring: for each template, count how many of its keywords appear
        in the combined lowercased text of all messages. The template with
        the highest score wins; ties go to the first (higher-priority) entry.
        The final empty-keyword template acts as the universal fallback.
        """
        combined = " ".join(m.content.lower() for m in messages)

        best_score = -1
        best_idx = len(_MOCK_RESPONSE_TEMPLATES) - 1  # default = fallback

        for idx, (keywords, _) in enumerate(_MOCK_RESPONSE_TEMPLATES):
            if not keywords:
                # Fallback — only use if nothing else matched
                continue
            score = sum(1 for kw in keywords if kw in combined)
            if score > best_score:
                best_score = score
                best_idx = idx

        _, base_response = _MOCK_RESPONSE_TEMPLATES[best_idx]

        # Add variation suffix if we're repeating the same template
        suffix = ""
        if best_idx == self._last_template_idx:
            suffix = _VARIATION_SUFFIXES[self._call_count % len(_VARIATION_SUFFIXES)]
        self._last_template_idx = best_idx

        content = base_response + suffix
        self._call_count += 1

        # Estimate token usage (rough heuristic: 1 token ≈ 4 chars)
        prompt_chars = sum(len(m.content) for m in messages)
        tokens_used = (prompt_chars + len(content)) // 4

        return LLMResponse(
            content=content,
            tokens_used=tokens_used,
            model=self.model_name,
            metadata={"template_idx": best_idx, "call_count": self._call_count},
        )

    def embed(self, text: str) -> List[float]:
        """
        Compute a simple character-frequency embedding (26 dims, a–z).

        Each dimension is the normalised frequency of the corresponding
        letter.  This is purely for structural compatibility — real
        semantic similarity requires a true embedding model.
        """
        text_lower = text.lower()
        counts = [0.0] * 26
        total = 0
        for ch in text_lower:
            if "a" <= ch <= "z":
                counts[ord(ch) - ord("a")] += 1
                total += 1
        if total > 0:
            counts = [c / total for c in counts]
        return counts

    @property
    def provider_name(self) -> str:
        return f"MockLLMProvider({self.model_name})"


# ---------------------------------------------------------------------------
# OpenAI provider template
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """
    Template provider for the OpenAI Chat Completions API.

    This class constructs the correct HTTP request payload and handles
    response parsing.  The actual HTTP call is delegated to ``_request()``,
    which raises NotImplementedError until the ``requests`` library is
    available and the method is overridden.

    To make this functional, subclass and override ``_request``::

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

    Args:
        api_key   — OpenAI API key (sk-…).
        model     — Model identifier, e.g. "gpt-4o", "gpt-3.5-turbo".
        base_url  — Override for custom OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an HTTP POST to the OpenAI API.

        Raises:
            NotImplementedError — Install the ``requests`` library and override
                                  this method to enable live API calls:
                                  ``pip install requests``
        """
        raise NotImplementedError(
            "OpenAIProvider._request() requires the 'requests' library. "
            "Install it with: pip install requests\n"
            "Then subclass OpenAIProvider and override _request() with a real HTTP call."
        )

    def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Call the OpenAI /chat/completions endpoint.

        Constructs the request payload from the messages list, calls
        ``_request()``, and parses the response into an LLMResponse.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        data = self._request("/chat/completions", payload)
        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            tokens_used=usage.get("total_tokens", 0),
            model=data.get("model", self.model),
            metadata={
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
            },
        )

    def embed(self, text: str, embedding_model: str = "text-embedding-3-small") -> List[float]:
        """
        Call the OpenAI /embeddings endpoint.

        Args:
            text            — Input text to embed.
            embedding_model — Model to use for embeddings.
        """
        payload = {"model": embedding_model, "input": text}
        data = self._request("/embeddings", payload)
        return data["data"][0]["embedding"]

    @property
    def provider_name(self) -> str:
        return f"OpenAIProvider(model={self.model})"


# ---------------------------------------------------------------------------
# Anthropic provider template
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """
    Template provider for the Anthropic Messages API (Claude models).

    Follows the same pattern as OpenAIProvider — the ``_request()`` method
    raises NotImplementedError until overridden with a real HTTP call.

    Anthropic's API has two key differences from OpenAI:
        1. The system prompt is passed as a top-level field, not a message.
        2. The response structure uses ``content[0].text`` instead of
           ``choices[0].message.content``.

    To activate::

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

    Args:
        api_key  — Anthropic API key.
        model    — Model identifier, e.g. "claude-3-5-sonnet-20241022".
        base_url — Override for custom endpoints.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        base_url: str = "https://api.anthropic.com/v1",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an HTTP POST to the Anthropic API.

        Raises:
            NotImplementedError — Install ``requests`` and override this method.
        """
        raise NotImplementedError(
            "AnthropicProvider._request() requires the 'requests' library. "
            "Install it with: pip install requests\n"
            "Then subclass AnthropicProvider and override _request() with a real HTTP call."
        )

    def complete(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Call the Anthropic /messages endpoint.

        Splits the messages list into a system prompt string (from any
        "system" role message) and the human/assistant conversation turns.
        """
        system_parts: List[str] = []
        conversation: List[Dict[str, str]] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                # Anthropic uses "user" and "assistant" roles
                conversation.append({"role": msg.role, "content": msg.content})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        data = self._request("/messages", payload)
        content = data["content"][0]["text"]
        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            tokens_used=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            model=data.get("model", self.model),
            metadata={
                "stop_reason": data.get("stop_reason"),
                "usage": usage,
            },
        )

    def embed(self, text: str) -> List[float]:
        """
        Anthropic does not currently offer a standalone embedding API.

        This method raises NotImplementedError. Use OpenAIProvider or a
        dedicated embedding provider (e.g. Cohere, Voyage AI) for embeddings.
        """
        raise NotImplementedError(
            "Anthropic does not provide an embedding endpoint. "
            "Use a dedicated embedding provider such as OpenAIProvider or VoyageProvider."
        )

    @property
    def provider_name(self) -> str:
        return f"AnthropicProvider(model={self.model})"


# ---------------------------------------------------------------------------
# LLM Router
# ---------------------------------------------------------------------------

class LLMRouter:
    """
    Routes LLM requests from agents to the appropriate provider.

    Providers are registered by name.  A default provider handles requests
    that don't specify a preference.  Fallback providers are tried in order
    if the primary provider raises an exception.

    Usage::

        router = LLMRouter()
        router.register_provider("mock", MockLLMProvider())
        router.register_provider("openai", OpenAIProvider(api_key="sk-..."))
        router.set_default("mock")
        router.set_fallback(["openai"])

        response = router.complete(messages, preferred_provider="openai")
    """

    def __init__(self) -> None:
        self._providers: Dict[str, LLMProvider] = {}
        self._default: Optional[str] = None
        self._fallback_order: List[str] = []
        self._total_requests: int = 0
        self._total_tokens: int = 0
        self._errors: int = 0

    def register_provider(self, name: str, provider: LLMProvider) -> None:
        """
        Register a provider under a given name.

        Args:
            name     — Lookup key (e.g. "mock", "openai", "anthropic").
            provider — Any LLMProvider instance.
        """
        self._providers[name] = provider
        # Auto-set default if none configured yet
        if self._default is None:
            self._default = name

    def set_default(self, name: str) -> None:
        """Set the default provider by name."""
        if name not in self._providers:
            raise KeyError(f"Provider {name!r} not registered. Call register_provider() first.")
        self._default = name

    def set_fallback(self, ordered_names: List[str]) -> None:
        """
        Set an ordered list of fallback provider names.

        When the primary provider raises an exception, the router tries each
        fallback in order until one succeeds.
        """
        for name in ordered_names:
            if name not in self._providers:
                raise KeyError(f"Fallback provider {name!r} not registered.")
        self._fallback_order = list(ordered_names)

    def complete(
        self,
        messages: List[LLMMessage],
        preferred_provider: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Request a completion, routing to the preferred (or default) provider.

        Falls back through self._fallback_order if the primary raises.

        Args:
            messages           — Conversation context.
            preferred_provider — Name of the provider to use (overrides default).
            **kwargs           — Forwarded to provider.complete().

        Raises:
            RuntimeError — If no provider is registered or all attempts fail.
        """
        if not self._providers:
            raise RuntimeError("No LLM providers registered. Call register_provider() first.")

        provider_name = preferred_provider or self._default
        if provider_name is None:
            raise RuntimeError("No default LLM provider set. Call set_default() first.")

        attempt_order = [provider_name] + [
            f for f in self._fallback_order if f != provider_name
        ]
        last_error: Optional[Exception] = None

        for name in attempt_order:
            provider = self._providers.get(name)
            if provider is None:
                continue
            try:
                response = provider.complete(messages, **kwargs)
                self._total_requests += 1
                self._total_tokens += response.tokens_used
                return response
            except Exception as exc:
                self._errors += 1
                last_error = exc
                # Try next fallback
                continue

        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_error}"
        )

    def embed(
        self,
        text: str,
        preferred_provider: Optional[str] = None,
    ) -> List[float]:
        """
        Request an embedding from the preferred (or default) provider.

        Args:
            text               — Text to embed.
            preferred_provider — Provider name override.
        """
        if not self._providers:
            raise RuntimeError("No LLM providers registered.")
        provider_name = preferred_provider or self._default
        provider = self._providers.get(provider_name)
        if provider is None:
            raise KeyError(f"Provider {provider_name!r} not found.")
        return provider.embed(text)

    def list_providers(self) -> List[str]:
        """Return names of all registered providers."""
        return list(self._providers.keys())

    def stats(self) -> Dict[str, Any]:
        """Return router-level usage statistics."""
        return {
            "providers": self.list_providers(),
            "default": self._default,
            "fallback_order": self._fallback_order,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "errors": self._errors,
        }


# ---------------------------------------------------------------------------
# Context Window — maps agent memory to LLM messages
# ---------------------------------------------------------------------------

class ContextWindow:
    """
    Converts an agent's memory space into a structured messages list.

    Memory key conventions:
        "system_prompt"         — LONG_TERM: becomes the system message.
        "turn_NNNN_user"        — SHORT_TERM: user turn (NNNN = zero-padded seq).
        "turn_NNNN_assistant"   — SHORT_TERM: assistant turn.
        "tool_result_*"         — Any: appended as assistant messages describing
                                  tool execution results.
        "fact_*"                — LONG_TERM: appended to the system prompt as
                                  known facts / persistent context.

    Usage::

        ctx = ContextWindow(max_turns=20)
        messages = ctx.build(agent_memory_snapshot, extra_user_message)
    """

    def __init__(self, max_turns: int = 20) -> None:
        """
        Args:
            max_turns — Maximum number of conversation turns to include.
                        Older turns are truncated to keep within context limits.
        """
        self.max_turns = max_turns

    def build(
        self,
        memory_snapshot: Dict[str, Any],
        extra_user_message: Optional[str] = None,
    ) -> List[LLMMessage]:
        """
        Build a messages list from the agent's current memory snapshot.

        Args:
            memory_snapshot    — Dict mapping key → value from the agent's
                                 memory space (obtained via space.snapshot()).
            extra_user_message — An additional user-role message to append
                                 at the end (e.g. the latest inbox content).

        Returns:
            Ordered list of LLMMessage objects ready to be sent to a provider.
        """
        messages: List[LLMMessage] = []

        # --- System prompt ---
        system_parts: List[str] = []
        system_prompt = memory_snapshot.get("system_prompt")
        if system_prompt:
            system_parts.append(str(system_prompt))

        # Append any "fact_*" keys to the system prompt
        facts = sorted(
            (k, v) for k, v in memory_snapshot.items() if k.startswith("fact_")
        )
        if facts:
            system_parts.append("\nKnown facts:")
            for key, val in facts:
                system_parts.append(f"  - {key[5:]}: {val}")

        if system_parts:
            messages.append(LLMMessage(role="system", content="\n".join(system_parts)))

        # --- Conversation history from turn_NNNN_* keys ---
        turn_keys = sorted(
            k for k in memory_snapshot if re.match(r"^turn_\d{4}_(user|assistant)$", k)
        )

        # Group by turn number
        turns: Dict[str, Dict[str, str]] = {}
        for key in turn_keys:
            parts_k = key.split("_")
            turn_num = parts_k[1]
            role = parts_k[2]
            turns.setdefault(turn_num, {})[role] = str(memory_snapshot[key])

        # Flatten to messages, respecting max_turns (take the most recent)
        sorted_nums = sorted(turns.keys())
        if len(sorted_nums) > self.max_turns:
            sorted_nums = sorted_nums[-self.max_turns:]

        for num in sorted_nums:
            turn = turns[num]
            if "user" in turn:
                messages.append(LLMMessage(role="user", content=turn["user"]))
            if "assistant" in turn:
                messages.append(LLMMessage(role="assistant", content=turn["assistant"]))

        # --- Tool results (appended as assistant messages) ---
        tool_keys = sorted(
            k for k in memory_snapshot if k.startswith("tool_result_")
        )
        for key in tool_keys:
            messages.append(LLMMessage(
                role="assistant",
                content=f"[Tool result for {key[12:]}]: {memory_snapshot[key]}",
            ))

        # --- Extra user message (e.g. from inbox) ---
        if extra_user_message:
            messages.append(LLMMessage(role="user", content=extra_user_message))

        # Ensure there is always at least one non-system message
        if not any(m.role != "system" for m in messages):
            messages.append(LLMMessage(role="user", content="What should I do next?"))

        return messages


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

# Regex patterns for each action tag the LLMAgent recognises
_ACTION_PATTERNS = {
    "SEND":  re.compile(r"\[ACTION:SEND\s+(\S+)\]\s*(.+?)(?=\[ACTION:|$)", re.DOTALL),
    "TOOL":  re.compile(r"\[ACTION:TOOL\s+(\S+)\]\s*(\{.*?\})(?=\[ACTION:|$)", re.DOTALL),
    "WRITE": re.compile(r"\[ACTION:WRITE\s+(\S+)\]\s*(.+?)(?=\[ACTION:|$)", re.DOTALL),
    "SPAWN": re.compile(r"\[ACTION:SPAWN\s+(\S+)\s+(\d+)\]\s*(.+?)(?=\[ACTION:|$)", re.DOTALL),
    "THINK": re.compile(r"\[ACTION:THINK\]\s*(.+?)(?=\[ACTION:|$)", re.DOTALL),
}


def _parse_actions(text: str) -> List[Dict[str, Any]]:
    """
    Parse structured action tags from an LLM response.

    Recognises these tags (case-sensitive):
        [ACTION:SEND target_id] message content
        [ACTION:TOOL tool_name] {"arg": "value"}
        [ACTION:WRITE /path] file content
        [ACTION:SPAWN AgentName priority] system prompt for new agent
        [ACTION:THINK] internal reasoning text

    Returns a list of action dicts, e.g.:
        {"type": "SEND", "target": "agent_id", "content": "hello"}
        {"type": "TOOL", "name": "calculator", "args": {"expression": "1+1"}}
        {"type": "WRITE", "path": "/foo/bar.txt", "content": "data"}
        {"type": "SPAWN", "name": "AgentName", "priority": 4, "prompt": "..."}
        {"type": "THINK", "content": "internal thought"}
    """
    actions: List[Dict[str, Any]] = []

    for m in _ACTION_PATTERNS["SEND"].finditer(text):
        actions.append({"type": "SEND", "target": m.group(1).strip(), "content": m.group(2).strip()})

    for m in _ACTION_PATTERNS["TOOL"].finditer(text):
        tool_name = m.group(1).strip()
        raw_args = m.group(2).strip()
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {"raw": raw_args}
        actions.append({"type": "TOOL", "name": tool_name, "args": args})

    for m in _ACTION_PATTERNS["WRITE"].finditer(text):
        actions.append({"type": "WRITE", "path": m.group(1).strip(), "content": m.group(2).strip()})

    for m in _ACTION_PATTERNS["SPAWN"].finditer(text):
        actions.append({
            "type": "SPAWN",
            "name": m.group(1).strip(),
            "priority": int(m.group(2).strip()),
            "prompt": m.group(3).strip(),
        })

    for m in _ACTION_PATTERNS["THINK"].finditer(text):
        actions.append({"type": "THINK", "content": m.group(1).strip()})

    return actions


# ---------------------------------------------------------------------------
# LLMAgent
# ---------------------------------------------------------------------------

class LLMAgent(Agent):
    """
    An Battousai agent that uses an LLM provider for its cognitive loop.

    Unlike hard-coded agents (CoordinatorAgent, WorkerAgent) which follow
    fixed decision trees, LLMAgent delegates all reasoning to a language
    model.  Each tick it:

        1. Reads its inbox and drains pending messages.
        2. Builds a context window from its memory + inbox content.
        3. Calls the LLM router for a completion.
        4. Parses structured action tags from the LLM response.
        5. Executes each action via the kernel syscall interface.
        6. Stores the LLM's response back to SHORT_TERM memory as a turn.

    Action format (parsed from LLM output):
        [ACTION:SEND target_id] message content
        [ACTION:TOOL tool_name] {"arg": "value"}
        [ACTION:WRITE /path] file content
        [ACTION:SPAWN AgentName priority] system prompt for new agent
        [ACTION:THINK] internal reasoning text

    Any text outside of action tags is treated as narrative and logged.

    Args:
        name          — Human-readable name for the agent.
        priority      — Scheduler priority (0=highest, 9=lowest).
        llm_router    — A configured LLMRouter instance.
        provider_name — Which registered provider to use (overrides default).
        system_prompt — Initial system-role instructions for the LLM.
        max_turns     — How many conversation turns to keep in context.

    Usage::

        router = LLMRouter()
        router.register_provider("mock", MockLLMProvider())

        kernel.boot()
        agent_id = kernel.spawn_agent(
            LLMAgent,
            name="ResearchBot",
            llm_router=router,
            system_prompt="You are a research assistant. Use tools to answer questions.",
        )
        kernel.run(10)
    """

    def __init__(
        self,
        name: str = "LLMAgent",
        priority: int = 5,
        llm_router: Optional[LLMRouter] = None,
        provider_name: Optional[str] = None,
        system_prompt: str = "You are a helpful AI agent running inside Battousai.",
        max_turns: int = 20,
    ) -> None:
        super().__init__(
            name=name,
            priority=priority,
            memory_allocation=512,
            time_slice=4,
        )
        # LLM infrastructure
        self._router: Optional[LLMRouter] = llm_router
        self._provider_name: Optional[str] = provider_name
        self._system_prompt: str = system_prompt
        self._max_turns: int = max_turns

        # Conversation state
        self._turn_counter: int = 0
        self._total_tokens: int = 0

        # Context window builder
        self._ctx_builder = ContextWindow(max_turns=max_turns)

    def on_spawn(self) -> None:
        """Initialise memory with the system prompt on first spawn."""
        self.mem_write("system_prompt", self._system_prompt, memory_type=MemoryType.LONG_TERM)
        self.log(
            f"[{self.name}] LLMAgent online. "
            f"Provider: {self._provider_name or 'default'}. "
            f"System prompt: {self._system_prompt[:80]!r}..."
        )

    def on_terminate(self) -> None:
        """Log total token usage on shutdown."""
        self.log(
            f"[{self.name}] Terminating. "
            f"Total LLM tokens used: {self._total_tokens}. "
            f"Turns: {self._turn_counter}."
        )

    def think(self, tick: int) -> None:
        """
        Main cognitive loop — called every tick.

        Steps:
            1. Drain inbox → format as user message.
            2. Build context window from memory + inbox.
            3. Call LLM.
            4. Parse actions from response.
            5. Execute actions via syscalls.
            6. Store response turn in SHORT_TERM memory.
        """
        if self._router is None:
            self.log(f"[{self.name}] No LLM router configured — idle.")
            self.yield_cpu()
            return

        # Step 1: drain inbox
        messages = self.read_inbox()
        inbox_text = self._format_inbox(messages)

        # Step 2: build context window
        mem_snapshot = self._get_memory_snapshot()
        llm_messages = self._ctx_builder.build(mem_snapshot, inbox_text or None)

        # Step 3: call LLM
        try:
            response = self._router.complete(
                llm_messages,
                preferred_provider=self._provider_name,
            )
        except Exception as exc:
            self.log(f"[{self.name}] LLM call failed: {exc}")
            self.yield_cpu()
            return

        self._total_tokens += response.tokens_used
        self.log(
            f"[{self.name}] LLM response ({response.tokens_used} tokens): "
            f"{response.content[:100]!r}..."
        )

        # Step 4 & 5: parse and execute actions
        actions = _parse_actions(response.content)
        if not actions:
            # No structured actions — treat the whole response as a THINK
            actions = [{"type": "THINK", "content": response.content}]

        for action in actions:
            self._execute_action(action, tick)

        # Step 6: store this turn in SHORT_TERM memory (TTL = 2 * max_turns)
        self._turn_counter += 1
        turn_key = f"turn_{self._turn_counter:04d}_assistant"
        self.mem_write(
            turn_key,
            response.content,
            memory_type=MemoryType.SHORT_TERM,
            ttl=self._max_turns * 2,
        )
        if inbox_text:
            user_key = f"turn_{self._turn_counter:04d}_user"
            self.mem_write(
                user_key,
                inbox_text,
                memory_type=MemoryType.SHORT_TERM,
                ttl=self._max_turns * 2,
            )

        self.yield_cpu()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_inbox(self, messages: list) -> str:
        """
        Convert a list of Message objects into a single user-turn string.

        Each message is formatted as:
            [TYPE from sender_id]: payload
        """
        if not messages:
            return ""
        parts = []
        for msg in messages:
            payload_str = str(msg.payload)[:200]
            parts.append(
                f"[{msg.message_type.name} from {msg.sender_id}]: {payload_str}"
            )
        return "\n".join(parts)

    def _get_memory_snapshot(self) -> Dict[str, Any]:
        """
        Read all keys from this agent's memory space.

        Uses the kernel's memory manager directly (via syscall read loop)
        to reconstruct a snapshot dict.
        """
        snapshot: Dict[str, Any] = {}
        if self.kernel is None:
            return snapshot
        space = self.kernel.memory._agents.get(self.agent_id)
        if space is not None:
            snapshot = space.snapshot()
        return snapshot

    def _execute_action(self, action: Dict[str, Any], tick: int) -> None:
        """
        Dispatch a parsed LLM action to the appropriate syscall.

        SEND  → send_message syscall to target agent.
        TOOL  → access_tool syscall with provided args.
        WRITE → write_file syscall at the given path.
        SPAWN → spawn_agent syscall with a new LLMAgent child.
        THINK → store as short-term memory (internal monologue).
        """
        action_type = action.get("type", "THINK")

        if action_type == "SEND":
            target = action.get("target", "")
            content = action.get("content", "")
            if target:
                result = self.syscall(
                    "send_message",
                    recipient_id=target,
                    message_type=MessageType.CUSTOM,
                    payload={"content": content, "from_llm": True},
                )
                self.log(f"[{self.name}] SEND → {target}: {content[:60]!r} (ok={result.ok})")
            else:
                self.log(f"[{self.name}] SEND action missing target — skipping.")

        elif action_type == "TOOL":
            tool_name = action.get("name", "")
            args = action.get("args", {})
            if tool_name:
                result = self.syscall("access_tool", tool_name=tool_name, args=args)
                result_key = f"tool_result_{tool_name}_{tick}"
                if result.ok:
                    self.mem_write(
                        result_key,
                        str(result.value)[:500],
                        memory_type=MemoryType.SHORT_TERM,
                        ttl=10,
                    )
                    self.log(f"[{self.name}] TOOL {tool_name!r}: ok, result stored as {result_key!r}")
                else:
                    self.log(f"[{self.name}] TOOL {tool_name!r} failed: {result.error}")
            else:
                self.log(f"[{self.name}] TOOL action missing name — skipping.")

        elif action_type == "WRITE":
            path = action.get("path", "")
            content = action.get("content", "")
            if path:
                result = self.syscall("write_file", path=path, data=content)
                self.log(f"[{self.name}] WRITE {path!r}: ok={result.ok}")
            else:
                self.log(f"[{self.name}] WRITE action missing path — skipping.")

        elif action_type == "SPAWN":
            child_name = action.get("name", "ChildAgent")
            priority = action.get("priority", 5)
            prompt = action.get("prompt", "You are a helpful AI agent.")
            if self._router is not None:
                result = self.syscall(
                    "spawn_agent",
                    agent_class=LLMAgent,
                    agent_name=child_name,
                    priority=priority,
                    llm_router=self._router,
                    provider_name=self._provider_name,
                    system_prompt=prompt,
                )
                self.log(
                    f"[{self.name}] SPAWN {child_name!r} "
                    f"(priority={priority}): ok={result.ok}, id={result.value!r}"
                )
            else:
                self.log(f"[{self.name}] SPAWN: no router configured — cannot spawn.")

        elif action_type == "THINK":
            thought = action.get("content", "")
            if thought:
                think_key = f"turn_{self._turn_counter:04d}_think"
                self.mem_write(
                    think_key,
                    thought[:500],
                    memory_type=MemoryType.SHORT_TERM,
                    ttl=self._max_turns,
                )
                self.log(f"[{self.name}] THINK: {thought[:80]!r}")

        else:
            self.log(f"[{self.name}] Unknown action type {action_type!r} — skipping.")


# ---------------------------------------------------------------------------
# Module-level convenience factory
# ---------------------------------------------------------------------------

def create_mock_router(model_name: str = "mock-gpt-1") -> LLMRouter:
    """
    Create and return a pre-configured LLMRouter backed by MockLLMProvider.

    Convenience function for tests and demos::

        from battousai.llm import create_mock_router, LLMAgent

        router = create_mock_router()
        kernel.spawn_agent(LLMAgent, name="Bot", llm_router=router)
    """
    router = LLMRouter()
    router.register_provider("mock", MockLLMProvider(model_name=model_name))
    router.set_default("mock")
    return router
