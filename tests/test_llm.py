"""
test_llm.py — Tests for battousai.llm
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from battousai.llm import (
    MockLLMProvider, LLMRouter, LLMResponse, LLMMessage, ContextWindow,
    LLMAgent, _parse_actions, create_mock_router,
)
from battousai.kernel import Kernel


class TestMockLLMProvider(unittest.TestCase):

    def setUp(self):
        self.provider = MockLLMProvider(model_name="mock-v1")

    def test_complete_returns_llm_response(self):
        msgs = [LLMMessage(role="user", content="hello")]
        response = self.provider.complete(msgs)
        self.assertIsInstance(response, LLMResponse)

    def test_complete_response_has_text(self):
        msgs = [LLMMessage(role="user", content="hi")]
        response = self.provider.complete(msgs)
        # LLMResponse exposes text via .content or .text
        text = response.content if hasattr(response, 'content') else response.text
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_complete_response_has_model(self):
        msgs = [LLMMessage(role="user", content="hi")]
        response = self.provider.complete(msgs)
        self.assertEqual(response.model, "mock-v1")

    def test_embed_returns_list_of_floats(self):
        vec = self.provider.embed("hello world")
        self.assertIsInstance(vec, list)
        for v in vec:
            self.assertIsInstance(v, float)

    def test_embed_returns_26_dimensional_vector(self):
        """MockLLMProvider.embed returns a 26-dim character frequency vector."""
        vec = self.provider.embed("abcdef")
        self.assertEqual(len(vec), 26)

    def test_embed_different_texts_differ(self):
        vec_a = self.provider.embed("aaaa")
        vec_b = self.provider.embed("zzzz")
        self.assertNotEqual(vec_a, vec_b)


class TestLLMRouter(unittest.TestCase):

    def setUp(self):
        self.router = LLMRouter()
        self.provider = MockLLMProvider(model_name="mock-v1")
        self.router.register_provider("mock", self.provider)
        self.router.set_default("mock")

    def _msgs(self, content="hello"):
        return [LLMMessage(role="user", content=content)]

    def test_register_provider(self):
        # Already done in setUp; just verify list_providers works
        names = self.router.list_providers()
        self.assertIn("mock", names)

    def test_set_default_and_complete(self):
        response = self.router.complete(self._msgs("hello"))
        self.assertIsInstance(response, LLMResponse)

    def test_complete_with_named_provider(self):
        response = self.router.complete(
            self._msgs("hello"),
            preferred_provider="mock"
        )
        self.assertIsInstance(response, LLMResponse)

    def test_set_fallback_provider(self):
        """set_fallback takes a list of ordered provider names."""
        fallback = MockLLMProvider(model_name="fallback-v1")
        self.router.register_provider("fallback", fallback)
        self.router.set_fallback(["fallback"])  # Must be a list
        # Should complete without error
        response = self.router.complete(
            self._msgs("test"),
            preferred_provider="mock"
        )
        self.assertIsInstance(response, LLMResponse)

    def test_create_mock_router_convenience(self):
        router = create_mock_router()
        self.assertIsInstance(router, LLMRouter)
        msgs = [LLMMessage(role="user", content="hi")]
        response = router.complete(msgs)
        self.assertIsInstance(response, LLMResponse)


class TestContextWindow(unittest.TestCase):

    def setUp(self):
        self.ctx = ContextWindow(max_turns=5)

    def test_build_returns_list_of_messages(self):
        messages = self.ctx.build(memory_snapshot={})
        self.assertIsInstance(messages, list)

    def test_build_with_extra_user_message(self):
        """build() with extra_user_message — that message appears as an LLMMessage."""
        messages = self.ctx.build(
            memory_snapshot={},
            extra_user_message="What should I do?"
        )
        # LLMMessage has .content attribute (not dict .get())
        contents = [m.content for m in messages if hasattr(m, 'content')]
        self.assertTrue(
            any("What should I do?" in c for c in contents)
        )

    def test_build_with_memory_snapshot(self):
        snapshot = {"goal": "process data", "status": "idle"}
        messages = self.ctx.build(memory_snapshot=snapshot)
        self.assertIsInstance(messages, list)
        self.assertGreater(len(messages), 0)


class TestParseActions(unittest.TestCase):

    def test_parse_send_action(self):
        text = "[ACTION:SEND agent_0001] hello world message"
        actions = _parse_actions(text)
        send_actions = [a for a in actions if a.get("type") == "SEND"]
        self.assertGreater(len(send_actions), 0)

    def test_parse_think_action(self):
        text = "[ACTION:THINK] analyzing the situation carefully"
        actions = _parse_actions(text)
        think_actions = [a for a in actions if a.get("type") == "THINK"]
        self.assertGreater(len(think_actions), 0)

    def test_parse_empty_text_returns_list(self):
        actions = _parse_actions("")
        self.assertIsInstance(actions, list)

    def test_parse_tool_action(self):
        text = '[ACTION:TOOL calculator] {"operation": "add", "a": 1, "b": 2}'
        actions = _parse_actions(text)
        self.assertIsInstance(actions, list)


class TestLLMAgent(unittest.TestCase):

    def setUp(self):
        self.kernel = Kernel(max_ticks=0, debug=False)
        self.kernel.boot()
        self.router = create_mock_router()

    def test_llm_agent_spawns_successfully(self):
        agent_id = self.kernel.spawn_agent(
            LLMAgent, name="LLMWorker", priority=5,
            llm_router=self.router,
            system_prompt="You are a helpful agent."
        )
        self.assertIn(agent_id, self.kernel._agents)

    def test_llm_agent_runs_multiple_ticks(self):
        self.kernel.spawn_agent(
            LLMAgent, name="LLMWorker", priority=5,
            llm_router=self.router,
            system_prompt="You are a helpful agent."
        )
        self.kernel.run(3)
        self.assertEqual(self.kernel._tick, 3)

    def test_llm_agent_with_named_provider(self):
        agent_id = self.kernel.spawn_agent(
            LLMAgent, name="LLMWorker2", priority=5,
            llm_router=self.router,
            provider_name="mock",
            system_prompt="Agent system prompt."
        )
        self.assertIn(agent_id, self.kernel._agents)


if __name__ == "__main__":
    unittest.main()
