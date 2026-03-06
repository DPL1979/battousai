"""
tests/test_providers.py — Unit tests for battousai.providers
=============================================================
Tests RealOpenAIProvider, RealAnthropicProvider, and OllamaProvider.

HTTP calls are mocked at the urllib.request level so no real API keys
or network access are required.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from typing import Any
from unittest.mock import MagicMock, patch, call
from io import BytesIO

# Ensure the workspace root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from battousai.providers import (
    AuthenticationError,
    HTTPProvider,
    LLMProviderError,
    OllamaProvider,
    RateLimitError,
    RealAnthropicProvider,
    RealOpenAIProvider,
    TimeoutError,
)
from battousai.llm import LLMMessage, LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_response(body: dict, status: int = 200) -> MagicMock:
    """Return a mock that mimics urllib.request.urlopen context manager."""
    raw = json.dumps(body).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _http_error(status: int, body: dict = None) -> "urllib.error.HTTPError":
    import urllib.error
    raw = json.dumps(body or {}).encode("utf-8")
    err = urllib.error.HTTPError(
        url="https://example.com",
        code=status,
        msg=f"HTTP {status}",
        hdrs={},
        fp=BytesIO(raw),
    )
    return err


_OPENAI_COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from GPT-4o!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
}

_OPENAI_EMBEDDING = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 5, "total_tokens": 5},
}

_ANTHROPIC_MESSAGE = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello from Claude!"}],
    "model": "claude-3-5-sonnet-20241022",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 12, "output_tokens": 7},
}


# ---------------------------------------------------------------------------
# HTTPProvider tests
# ---------------------------------------------------------------------------

class TestHTTPProvider(unittest.TestCase):
    """Unit tests for HTTPProvider._post()."""

    @patch("urllib.request.urlopen")
    def test_successful_post_returns_parsed_json(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response({"ok": True})
        provider = HTTPProvider("https://api.example.com")
        result = provider._post("/test", {"key": "value"})
        self.assertEqual(result, {"ok": True})

    @patch("urllib.request.urlopen")
    def test_sets_content_type_header(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response({"ok": True})
        provider = HTTPProvider("https://api.example.com")
        provider._post("/test", {})
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("Content-type"), "application/json")

    @patch("urllib.request.urlopen")
    def test_custom_default_headers_are_sent(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response({"ok": True})
        provider = HTTPProvider(
            "https://api.example.com",
            default_headers={"Authorization": "Bearer secret"},
        )
        provider._post("/test", {})
        req = mock_urlopen.call_args[0][0]
        self.assertIn("Bearer secret", req.get_header("Authorization"))

    @patch("urllib.request.urlopen")
    def test_raises_authentication_error_on_401(self, mock_urlopen):
        mock_urlopen.side_effect = _http_error(
            401, {"error": {"message": "invalid key"}}
        )
        provider = HTTPProvider("https://api.example.com", max_retries=1)
        with self.assertRaises(AuthenticationError) as ctx:
            provider._post("/test", {})
        self.assertEqual(ctx.exception.status_code, 401)

    @patch("urllib.request.urlopen")
    def test_raises_authentication_error_on_403(self, mock_urlopen):
        mock_urlopen.side_effect = _http_error(403, {})
        provider = HTTPProvider("https://api.example.com", max_retries=1)
        with self.assertRaises(AuthenticationError) as ctx:
            provider._post("/test", {})
        self.assertEqual(ctx.exception.status_code, 403)

    @patch("urllib.request.urlopen")
    def test_raises_rate_limit_error_on_429(self, mock_urlopen):
        mock_urlopen.side_effect = _http_error(
            429, {"error": {"message": "rate limited"}}
        )
        provider = HTTPProvider("https://api.example.com", max_retries=1)
        with self.assertRaises(RateLimitError):
            provider._post("/test", {})

    @patch("urllib.request.urlopen")
    def test_raises_llm_provider_error_on_400(self, mock_urlopen):
        mock_urlopen.side_effect = _http_error(400, {"error": {"message": "bad request"}})
        provider = HTTPProvider("https://api.example.com", max_retries=1)
        with self.assertRaises(LLMProviderError) as ctx:
            provider._post("/test", {})
        self.assertEqual(ctx.exception.status_code, 400)

    @patch("urllib.request.urlopen")
    @patch("time.sleep", return_value=None)
    def test_retries_on_500_server_error(self, mock_sleep, mock_urlopen):
        mock_urlopen.side_effect = _http_error(500, {})
        provider = HTTPProvider(
            "https://api.example.com", max_retries=3, retry_delay=0.01
        )
        with self.assertRaises(LLMProviderError):
            provider._post("/test", {})
        self.assertEqual(mock_urlopen.call_count, 3)

    @patch("urllib.request.urlopen")
    def test_base_url_trailing_slash_stripped(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response({"ok": True})
        provider = HTTPProvider("https://api.example.com/")
        provider._post("/endpoint", {})
        req = mock_urlopen.call_args[0][0]
        self.assertFalse(req.full_url.startswith("https://api.example.com//"))


# ---------------------------------------------------------------------------
# RealOpenAIProvider tests
# ---------------------------------------------------------------------------

class TestRealOpenAIProvider(unittest.TestCase):

    def _make_provider(self, **kwargs) -> RealOpenAIProvider:
        return RealOpenAIProvider(api_key="sk-test", **kwargs)

    @patch("urllib.request.urlopen")
    def test_complete_returns_llm_response(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response(_OPENAI_COMPLETION)
        provider = self._make_provider()
        messages = [LLMMessage(role="user", content="Hello!")]
        response = provider.complete(messages)
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "Hello from GPT-4o!")
        self.assertEqual(response.tokens_used, 18)
        self.assertEqual(response.model, "gpt-4o")

    @patch("urllib.request.urlopen")
    def test_complete_sends_correct_payload(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response(_OPENAI_COMPLETION)
        provider = self._make_provider(model="gpt-3.5-turbo")
        messages = [
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Hi"),
        ]
        provider.complete(messages, temperature=0.5, max_tokens=256)
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        self.assertEqual(body["model"], "gpt-3.5-turbo")
        self.assertEqual(body["temperature"], 0.5)
        self.assertEqual(body["max_tokens"], 256)
        self.assertEqual(len(body["messages"]), 2)

    @patch("urllib.request.urlopen")
    def test_embed_returns_float_list(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response(_OPENAI_EMBEDDING)
        provider = self._make_provider()
        embedding = provider.embed("test text")
        self.assertIsInstance(embedding, list)
        self.assertEqual(embedding, [0.1, 0.2, 0.3])

    @patch("urllib.request.urlopen")
    def test_provider_name(self, mock_urlopen):
        provider = self._make_provider(model="gpt-4o")
        self.assertIn("RealOpenAIProvider", provider.provider_name)
        self.assertIn("gpt-4o", provider.provider_name)

    def test_api_key_from_env_variable(self):
        with patch.dict(os.environ, {"BATTOUSAI_OPENAI_API_KEY": "env-key"}):
            provider = RealOpenAIProvider()
            self.assertEqual(provider.api_key, "env-key")

    def test_explicit_key_overrides_env(self):
        with patch.dict(os.environ, {"BATTOUSAI_OPENAI_API_KEY": "env-key"}):
            provider = RealOpenAIProvider(api_key="explicit-key")
            self.assertEqual(provider.api_key, "explicit-key")

    @patch("urllib.request.urlopen")
    def test_bearer_auth_header_sent(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response(_OPENAI_COMPLETION)
        provider = self._make_provider()
        provider.complete([LLMMessage(role="user", content="hi")])
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("Authorization"), "Bearer sk-test")

    @patch("urllib.request.urlopen")
    def test_auth_error_propagates(self, mock_urlopen):
        mock_urlopen.side_effect = _http_error(401, {"error": {"message": "bad key"}})
        provider = self._make_provider()
        with self.assertRaises(AuthenticationError):
            provider.complete([LLMMessage(role="user", content="hi")])

    @patch("urllib.request.urlopen")
    def test_metadata_contains_finish_reason(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response(_OPENAI_COMPLETION)
        provider = self._make_provider()
        response = provider.complete([LLMMessage(role="user", content="hi")])
        self.assertEqual(response.metadata["finish_reason"], "stop")

    @patch("urllib.request.urlopen")
    def test_custom_base_url_used(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response(_OPENAI_COMPLETION)
        provider = RealOpenAIProvider(
            api_key="key",
            base_url="http://localhost:8080/v1",
        )
        provider.complete([LLMMessage(role="user", content="hi")])
        req = mock_urlopen.call_args[0][0]
        self.assertIn("localhost:8080", req.full_url)


# ---------------------------------------------------------------------------
# RealAnthropicProvider tests
# ---------------------------------------------------------------------------

class TestRealAnthropicProvider(unittest.TestCase):

    def _make_provider(self, **kwargs) -> RealAnthropicProvider:
        return RealAnthropicProvider(api_key="test-anthropic-key", **kwargs)

    @patch("urllib.request.urlopen")
    def test_complete_returns_llm_response(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response(_ANTHROPIC_MESSAGE)
        provider = self._make_provider()
        messages = [LLMMessage(role="user", content="Hello Claude!")]
        response = provider.complete(messages)
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "Hello from Claude!")
        self.assertEqual(response.tokens_used, 19)

    @patch("urllib.request.urlopen")
    def test_system_messages_extracted(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response(_ANTHROPIC_MESSAGE)
        provider = self._make_provider()
        messages = [
            LLMMessage(role="system", content="You are Claude."),
            LLMMessage(role="user", content="Hi"),
        ]
        provider.complete(messages)
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        # System role must appear as top-level "system" field, not in messages list
        self.assertIn("system", body)
        self.assertEqual(body["system"], "You are Claude.")
        for msg in body["messages"]:
            self.assertNotEqual(msg["role"], "system")

    @patch("urllib.request.urlopen")
    def test_anthropic_version_header_sent(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response(_ANTHROPIC_MESSAGE)
        provider = self._make_provider()
        provider.complete([LLMMessage(role="user", content="hi")])
        req = mock_urlopen.call_args[0][0]
        self.assertIn("2023-06-01", req.get_header("Anthropic-version"))

    @patch("urllib.request.urlopen")
    def test_api_key_in_x_api_key_header(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response(_ANTHROPIC_MESSAGE)
        provider = self._make_provider()
        provider.complete([LLMMessage(role="user", content="hi")])
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("X-api-key"), "test-anthropic-key")

    def test_embed_raises_not_implemented(self):
        provider = self._make_provider()
        with self.assertRaises(NotImplementedError):
            provider.embed("some text")

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"BATTOUSAI_ANTHROPIC_API_KEY": "env-claude-key"}):
            provider = RealAnthropicProvider()
            self.assertEqual(provider.api_key, "env-claude-key")

    @patch("urllib.request.urlopen")
    def test_metadata_contains_stop_reason(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response(_ANTHROPIC_MESSAGE)
        provider = self._make_provider()
        response = provider.complete([LLMMessage(role="user", content="hi")])
        self.assertEqual(response.metadata["stop_reason"], "end_turn")

    @patch("urllib.request.urlopen")
    def test_provider_name(self, mock_urlopen):
        provider = self._make_provider()
        self.assertIn("RealAnthropicProvider", provider.provider_name)

    @patch("urllib.request.urlopen")
    def test_multiple_system_messages_joined(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response(_ANTHROPIC_MESSAGE)
        provider = self._make_provider()
        messages = [
            LLMMessage(role="system", content="Part 1."),
            LLMMessage(role="system", content="Part 2."),
            LLMMessage(role="user", content="Hi"),
        ]
        provider.complete(messages)
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        self.assertIn("Part 1.", body["system"])
        self.assertIn("Part 2.", body["system"])


# ---------------------------------------------------------------------------
# OllamaProvider tests
# ---------------------------------------------------------------------------

class TestOllamaProvider(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_defaults_to_localhost(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response(_OPENAI_COMPLETION)
        provider = OllamaProvider(model="llama3.2")
        provider.complete([LLMMessage(role="user", content="hi")])
        req = mock_urlopen.call_args[0][0]
        self.assertIn("localhost:11434", req.full_url)

    @patch("urllib.request.urlopen")
    def test_complete_returns_llm_response(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response(_OPENAI_COMPLETION)
        provider = OllamaProvider(model="mistral")
        response = provider.complete([LLMMessage(role="user", content="Hello!")])
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "Hello from GPT-4o!")

    @patch("urllib.request.urlopen")
    def test_embed_works_via_openai_shim(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response(_OPENAI_EMBEDDING)
        provider = OllamaProvider()
        embedding = provider.embed("test")
        self.assertIsInstance(embedding, list)

    def test_provider_name(self):
        provider = OllamaProvider(model="phi3")
        self.assertIn("OllamaProvider", provider.provider_name)
        self.assertIn("phi3", provider.provider_name)

    @patch("urllib.request.urlopen")
    def test_custom_base_url(self, mock_urlopen):
        mock_urlopen.return_value = _fake_response(_OPENAI_COMPLETION)
        provider = OllamaProvider(base_url="http://192.168.1.5:11434/v1")
        provider.complete([LLMMessage(role="user", content="hi")])
        req = mock_urlopen.call_args[0][0]
        self.assertIn("192.168.1.5", req.full_url)

    @patch("urllib.request.urlopen")
    def test_connection_error_raises_llm_provider_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        provider = OllamaProvider(max_retries=1)
        with self.assertRaises(LLMProviderError):
            provider.complete([LLMMessage(role="user", content="hi")])


# ---------------------------------------------------------------------------
# Error handling edge cases
# ---------------------------------------------------------------------------

class TestProviderErrorHandling(unittest.TestCase):

    @patch("urllib.request.urlopen")
    @patch("time.sleep", return_value=None)
    def test_connection_error_retried(self, mock_sleep, mock_urlopen):
        """URLError is retried, then raises LLMProviderError."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Network unreachable")
        provider = HTTPProvider(
            "https://api.example.com", max_retries=2, retry_delay=0.0
        )
        with self.assertRaises(LLMProviderError):
            provider._post("/test", {})
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("urllib.request.urlopen")
    @patch("time.sleep", return_value=None)
    def test_success_after_retry(self, mock_sleep, mock_urlopen):
        """Succeeds on the second attempt after a 500 error."""
        fail_err = _http_error(500, {})
        success = _fake_response({"ok": True})
        mock_urlopen.side_effect = [fail_err, success]
        provider = HTTPProvider(
            "https://api.example.com", max_retries=3, retry_delay=0.0
        )
        result = provider._post("/test", {})
        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("urllib.request.urlopen")
    def test_400_not_retried(self, mock_urlopen):
        """Client errors (4xx) are not retried."""
        mock_urlopen.side_effect = _http_error(400, {"error": {"message": "bad input"}})
        provider = HTTPProvider(
            "https://api.example.com", max_retries=3, retry_delay=0.0
        )
        with self.assertRaises(LLMProviderError):
            provider._post("/test", {})
        # Should only be called once — no retry on 4xx
        self.assertEqual(mock_urlopen.call_count, 1)

    def test_llm_provider_error_has_status_code(self):
        err = LLMProviderError("test", status_code=503)
        self.assertEqual(err.status_code, 503)

    def test_rate_limit_error_is_llm_provider_error(self):
        err = RateLimitError("too fast")
        self.assertIsInstance(err, LLMProviderError)

    def test_authentication_error_is_llm_provider_error(self):
        err = AuthenticationError("denied")
        self.assertIsInstance(err, LLMProviderError)


if __name__ == "__main__":
    unittest.main()
