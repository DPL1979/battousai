"""
providers.py — Production LLM Providers (zero dependencies)
============================================================
Real HTTP-based providers using only Python stdlib.
Supports any OpenAI-compatible API (OpenAI, Ollama, vLLM, LiteLLM, etc.)

Providers
---------
HTTPProvider          — Base class for HTTP POST with urllib.request.
RealOpenAIProvider    — OpenAI /chat/completions and /embeddings endpoints.
RealAnthropicProvider — Anthropic /messages endpoint (Claude models).
OllamaProvider        — Local Ollama via OpenAI-compatible API.

All providers extend battousai.llm.LLMProvider and return LLMResponse
objects.  No third-party packages required.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from battousai.llm import LLMProvider, LLMResponse, LLMMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class LLMProviderError(Exception):
    """Raised when an LLM provider request fails after all retries."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(LLMProviderError):
    """HTTP 429 — provider is rate-limiting requests."""


class AuthenticationError(LLMProviderError):
    """HTTP 401/403 — invalid or missing API key."""


class TimeoutError(LLMProviderError):
    """Request timed out before the provider responded."""


# ---------------------------------------------------------------------------
# HTTP base class
# ---------------------------------------------------------------------------

class HTTPProvider:
    """
    Base class for providers that communicate via HTTP POST.

    Features
    --------
    - Real HTTP using stdlib ``urllib.request`` (zero deps).
    - JSON request serialisation and response parsing.
    - Auth header injection (Bearer or custom).
    - Configurable timeout and retry logic with exponential back-off.
    - Graceful error mapping: 401/403 → AuthenticationError,
      429 → RateLimitError, socket/url errors → LLMProviderError.
    """

    def __init__(
        self,
        base_url: str,
        *,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if default_headers:
            self._headers.update(default_headers)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an HTTP POST to ``base_url + endpoint`` with JSON payload.

        Retries on 5xx or connection errors (up to ``max_retries`` times).
        Does NOT retry on 4xx (client errors).

        Returns
        -------
        dict
            Parsed JSON response body.

        Raises
        ------
        AuthenticationError  — 401 or 403 response.
        RateLimitError       — 429 response.
        LLMProviderError     — Any other HTTP error or connection failure.
        TimeoutError         — Request timed out.
        """
        url = self.base_url + endpoint
        body = json.dumps(payload).encode("utf-8")

        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(
                url,
                data=body,
                headers=self._headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    return json.loads(raw)

            except urllib.error.HTTPError as exc:
                status = exc.code
                try:
                    error_body = json.loads(exc.read())
                except Exception:
                    error_body = {}
                error_msg = (
                    error_body.get("error", {}).get("message", "")
                    or error_body.get("error", str(exc))
                    or str(exc)
                )
                logger.warning(
                    "HTTP %d from %s (attempt %d/%d): %s",
                    status, url, attempt, self.max_retries, error_msg,
                )
                if status in (401, 403):
                    raise AuthenticationError(
                        f"Authentication failed (HTTP {status}): {error_msg}",
                        status_code=status,
                    ) from exc
                if status == 429:
                    raise RateLimitError(
                        f"Rate limit exceeded (HTTP 429): {error_msg}",
                        status_code=429,
                    ) from exc
                if 400 <= status < 500:
                    # Other 4xx — no point retrying
                    raise LLMProviderError(
                        f"Client error (HTTP {status}): {error_msg}",
                        status_code=status,
                    ) from exc
                # 5xx — retry
                last_exc = LLMProviderError(
                    f"Server error (HTTP {status}): {error_msg}",
                    status_code=status,
                )

            except (urllib.error.URLError, OSError) as exc:
                cause = str(exc)
                if "timed out" in cause.lower() or "timeout" in cause.lower():
                    raise TimeoutError(
                        f"Request to {url} timed out after {self.timeout}s"
                    ) from exc
                logger.warning(
                    "Connection error to %s (attempt %d/%d): %s",
                    url, attempt, self.max_retries, cause,
                )
                last_exc = LLMProviderError(
                    f"Connection error: {cause}"
                )

            if attempt < self.max_retries:
                sleep_time = self.retry_delay * (2 ** (attempt - 1))
                logger.info("Retrying in %.1fs…", sleep_time)
                time.sleep(sleep_time)

        raise last_exc or LLMProviderError(
            f"All {self.max_retries} attempts to {url} failed."
        )


# ---------------------------------------------------------------------------
# RealOpenAIProvider
# ---------------------------------------------------------------------------

class RealOpenAIProvider(LLMProvider):
    """
    Production provider for the OpenAI Chat Completions API.

    Also works with any OpenAI-compatible endpoint:
        - Ollama (when ``base_url`` points to Ollama's OpenAI shim)
        - vLLM, LiteLLM, Together, Groq, etc.

    Parameters
    ----------
    api_key  : str, optional
        OpenAI API key.  Falls back to ``BATTOUSAI_OPENAI_API_KEY`` env var.
    model    : str
        Chat completion model, e.g. ``"gpt-4o"`` or ``"gpt-3.5-turbo"``.
    base_url : str
        Base URL of the API.  Override for custom compatible endpoints.
    timeout  : int
        Per-request timeout in seconds (default 60).
    max_retries : int
        Number of retry attempts for transient errors (default 3).

    Raises
    ------
    ValueError
        If no API key is provided or found in the environment.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        resolved_key = api_key or os.environ.get("BATTOUSAI_OPENAI_API_KEY", "")
        if not resolved_key:
            logger.warning(
                "RealOpenAIProvider: no API key supplied. "
                "Set BATTOUSAI_OPENAI_API_KEY or pass api_key=."
            )
        self.api_key = resolved_key
        self.model = model

        headers: Dict[str, str] = {}
        if resolved_key:
            headers["Authorization"] = f"Bearer {resolved_key}"

        self._http = HTTPProvider(
            base_url=base_url,
            default_headers=headers,
            timeout=timeout,
            max_retries=max_retries,
        )

    def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Call ``/chat/completions`` and return an ``LLMResponse``.

        Parameters
        ----------
        messages     : list of LLMMessage
        temperature  : float
        max_tokens   : int
        **kwargs     : forwarded to the API payload (e.g. ``top_p``)
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        logger.debug("RealOpenAIProvider: completing %d messages", len(messages))
        data = self._http._post("/chat/completions", payload)
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

    def embed(
        self,
        text: str,
        embedding_model: str = "text-embedding-3-small",
    ) -> List[float]:
        """
        Call ``/embeddings`` and return the embedding vector.

        Parameters
        ----------
        text            : str  — input text
        embedding_model : str  — embedding model name
        """
        payload = {"model": embedding_model, "input": text}
        logger.debug("RealOpenAIProvider: embedding text (%d chars)", len(text))
        data = self._http._post("/embeddings", payload)
        return data["data"][0]["embedding"]

    @property
    def provider_name(self) -> str:
        return f"RealOpenAIProvider(model={self.model})"


# ---------------------------------------------------------------------------
# RealAnthropicProvider
# ---------------------------------------------------------------------------

class RealAnthropicProvider(LLMProvider):
    """
    Production provider for the Anthropic Messages API (Claude models).

    Parameters
    ----------
    api_key  : str, optional
        Anthropic API key.  Falls back to ``BATTOUSAI_ANTHROPIC_API_KEY``.
    model    : str
        Model identifier, e.g. ``"claude-3-5-sonnet-20241022"``.
    base_url : str
        Base URL (override for proxies / custom endpoints).
    timeout  : int
        Per-request timeout in seconds (default 60).
    max_retries : int
        Number of retry attempts for transient errors (default 3).
    anthropic_version : str
        Value for the ``anthropic-version`` header.

    Notes
    -----
    Anthropic does not currently provide a public embedding endpoint.
    Calling ``embed()`` raises ``NotImplementedError``.
    """

    _ANTHROPIC_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        base_url: str = "https://api.anthropic.com/v1",
        timeout: int = 60,
        max_retries: int = 3,
        anthropic_version: str = "2023-06-01",
    ) -> None:
        resolved_key = api_key or os.environ.get("BATTOUSAI_ANTHROPIC_API_KEY", "")
        if not resolved_key:
            logger.warning(
                "RealAnthropicProvider: no API key supplied. "
                "Set BATTOUSAI_ANTHROPIC_API_KEY or pass api_key=."
            )
        self.api_key = resolved_key
        self.model = model

        headers: Dict[str, str] = {
            "anthropic-version": anthropic_version,
        }
        if resolved_key:
            headers["x-api-key"] = resolved_key

        self._http = HTTPProvider(
            base_url=base_url,
            default_headers=headers,
            timeout=timeout,
            max_retries=max_retries,
        )

    def complete(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Call the Anthropic ``/messages`` endpoint.

        System-role messages are extracted from ``messages`` and passed as
        the top-level ``system`` field, as required by the Anthropic API.
        """
        system_parts: List[str] = []
        conversation: List[Dict[str, str]] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
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

        logger.debug("RealAnthropicProvider: completing %d messages", len(messages))
        data = self._http._post("/messages", payload)
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
        Not supported by Anthropic.

        Raises
        ------
        NotImplementedError
            Anthropic does not provide a public embedding endpoint.
            Use ``RealOpenAIProvider`` or a dedicated embedding service.
        """
        raise NotImplementedError(
            "Anthropic does not provide an embedding endpoint. "
            "Use RealOpenAIProvider or a dedicated embedding provider."
        )

    @property
    def provider_name(self) -> str:
        return f"RealAnthropicProvider(model={self.model})"


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------

class OllamaProvider(RealOpenAIProvider):
    """
    Provider for local Ollama instances via the OpenAI-compatible shim.

    Ollama exposes an OpenAI-compatible API at ``/v1`` by default.
    No API key is required for local use.

    Parameters
    ----------
    model    : str
        Ollama model tag, e.g. ``"llama3.2"``, ``"mistral"``, ``"phi3"``.
    base_url : str
        Ollama server base URL.  Defaults to ``http://localhost:11434/v1``.
    timeout  : int
        Per-request timeout in seconds (default 120 — local models can be slow).
    max_retries : int
        Number of retry attempts (default 2).

    Example
    -------
    ::

        from battousai.providers import OllamaProvider
        from battousai.llm import LLMMessage, LLMRouter

        provider = OllamaProvider(model="llama3.2")
        router = LLMRouter()
        router.register_provider("ollama", provider)
        router.set_default("ollama")
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
        timeout: int = 120,
        max_retries: int = 2,
    ) -> None:
        # No API key needed for local Ollama
        super().__init__(
            api_key="ollama",      # Ollama ignores the key, but requires a non-empty header
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def provider_name(self) -> str:
        return f"OllamaProvider(model={self.model})"
