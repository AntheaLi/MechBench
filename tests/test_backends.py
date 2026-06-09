"""Tests for LLM provider backends.

These tests use mock HTTP responses — no real API calls are made.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

import httpx

from agents.backends import (
    AnthropicBackend,
    OpenAIBackend,
    TokenUsage,
    make_backend,
)


def _mock_anthropic_response(text: str = '{"answer": 42}', input_tokens: int = 100, output_tokens: int = 50):
    """Build a mock httpx.Response matching the Anthropic Messages API shape."""
    body = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
    response = httpx.Response(200, json=body, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    return response


def _mock_openai_response(text: str = '{"answer": 42}', prompt_tokens: int = 100, completion_tokens: int = 50):
    """Build a mock httpx.Response matching the OpenAI Chat Completions shape."""
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    }
    response = httpx.Response(200, json=body, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    return response


class TokenUsageTest(unittest.TestCase):
    def test_record_accumulates(self):
        usage = TokenUsage()
        usage.record(100, 50)
        usage.record(200, 80)
        self.assertEqual(usage.input_tokens, 300)
        self.assertEqual(usage.output_tokens, 130)
        self.assertEqual(usage.calls, 2)
        d = usage.to_dict()
        self.assertEqual(d, {"input_tokens": 300, "output_tokens": 130, "calls": 2})


class AnthropicBackendTest(unittest.TestCase):
    def _make_backend(self) -> AnthropicBackend:
        return AnthropicBackend(api_key="test-key", max_retries=0)

    @patch.object(httpx.Client, "post")
    def test_complete_returns_text(self, mock_post):
        mock_post.return_value = _mock_anthropic_response('{"plan": "run experiments"}')
        backend = self._make_backend()
        messages = [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "Do something."},
        ]
        result = backend.complete(messages, purpose="plan")
        self.assertEqual(result, '{"plan": "run experiments"}')

        # Verify the POST was called with correct structure
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        self.assertEqual(body["system"], "You are a test agent.")
        self.assertEqual(len(body["messages"]), 1)  # system extracted, only user remains
        self.assertEqual(body["messages"][0]["role"], "user")

    @patch.object(httpx.Client, "post")
    def test_usage_tracked(self, mock_post):
        mock_post.return_value = _mock_anthropic_response(input_tokens=150, output_tokens=75)
        backend = self._make_backend()
        backend.complete([{"role": "user", "content": "hi"}], purpose="test")
        self.assertEqual(backend.usage.input_tokens, 150)
        self.assertEqual(backend.usage.output_tokens, 75)
        self.assertEqual(backend.usage.calls, 1)

    @patch.object(httpx.Client, "post")
    def test_api_error_raises(self, mock_post):
        mock_post.return_value = httpx.Response(
            401, json={"error": {"message": "invalid key"}},
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        backend = self._make_backend()
        with self.assertRaises(RuntimeError) as ctx:
            backend.complete([{"role": "user", "content": "hi"}], purpose="test")
        self.assertIn("401", str(ctx.exception))

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                AnthropicBackend()

    @patch.object(httpx.Client, "post")
    def test_no_system_message(self, mock_post):
        mock_post.return_value = _mock_anthropic_response()
        backend = self._make_backend()
        backend.complete([{"role": "user", "content": "hi"}], purpose="test")
        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        self.assertNotIn("system", body)


class OpenAIBackendTest(unittest.TestCase):
    def _make_backend(self) -> OpenAIBackend:
        return OpenAIBackend(api_key="test-key", max_retries=0)

    @patch.object(httpx.Client, "post")
    def test_complete_returns_text(self, mock_post):
        mock_post.return_value = _mock_openai_response('{"plan": "run experiments"}')
        backend = self._make_backend()
        messages = [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "Do something."},
        ]
        result = backend.complete(messages, purpose="plan")
        self.assertEqual(result, '{"plan": "run experiments"}')

        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        # OpenAI keeps system in messages
        self.assertEqual(len(body["messages"]), 2)
        self.assertEqual(body["messages"][0]["role"], "system")

    @patch.object(httpx.Client, "post")
    def test_usage_tracked(self, mock_post):
        mock_post.return_value = _mock_openai_response(prompt_tokens=200, completion_tokens=100)
        backend = self._make_backend()
        backend.complete([{"role": "user", "content": "hi"}], purpose="test")
        self.assertEqual(backend.usage.input_tokens, 200)
        self.assertEqual(backend.usage.output_tokens, 100)
        self.assertEqual(backend.usage.calls, 1)

    @patch.object(httpx.Client, "post")
    def test_api_error_raises(self, mock_post):
        mock_post.return_value = httpx.Response(
            429, json={"error": {"message": "rate limited"}},
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        backend = self._make_backend()
        with self.assertRaises(RuntimeError) as ctx:
            backend.complete([{"role": "user", "content": "hi"}], purpose="test")
        self.assertIn("429", str(ctx.exception))

    @patch.object(httpx.Client, "post")
    def test_empty_choices_raises(self, mock_post):
        body = {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 0}}
        mock_post.return_value = httpx.Response(
            200, json=body,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        backend = self._make_backend()
        with self.assertRaises(RuntimeError) as ctx:
            backend.complete([{"role": "user", "content": "hi"}], purpose="test")
        self.assertIn("no choices", str(ctx.exception))

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                OpenAIBackend()

    @patch.object(httpx.Client, "post")
    def test_custom_base_url(self, mock_post):
        mock_post.return_value = _mock_openai_response()
        backend = OpenAIBackend(api_key="test-key", base_url="https://custom.api.com/v1", max_retries=0)
        backend.complete([{"role": "user", "content": "hi"}], purpose="test")
        call_args = mock_post.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        # The URL should use the custom base
        self.assertIn("custom.api.com", str(url))


class MakeBackendFactoryTest(unittest.TestCase):
    def test_anthropic(self):
        backend = make_backend("anthropic", api_key="test-key", model="claude-haiku-4-5-20251001")
        self.assertIsInstance(backend, AnthropicBackend)
        self.assertEqual(backend.model, "claude-haiku-4-5-20251001")

    def test_openai(self):
        backend = make_backend("openai", api_key="test-key", model="gpt-4o-mini")
        self.assertIsInstance(backend, OpenAIBackend)
        self.assertEqual(backend.model, "gpt-4o-mini")

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            make_backend("gemini", api_key="test")


class CLIIntegrationTest(unittest.TestCase):
    """Test that the CLI wiring for LLM agents works without API calls."""

    def test_llm_mock_agent_through_cli_evaluate(self):
        from mechbench.cli import main
        # llm_mock doesn't need an API key — it's the heuristic mock
        ret = main(["evaluate", "--agent", "llm_mock", "--worlds", "simulator/seed_hacking_001"])
        self.assertEqual(ret, 0)

    def test_llm_study_with_mock(self):
        from mechbench.cli import main
        ret = main(["llm-study", "--agent", "llm_mock", "--worlds", "simulator/seed_hacking_001,simulator/true_mechanism_001"])
        self.assertEqual(ret, 0)


if __name__ == "__main__":
    unittest.main()
