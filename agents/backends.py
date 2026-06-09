"""Real LLM provider backends for the scaffold agent.

Each backend implements the ``LLMBackend`` protocol from ``llm_scaffold.py``.
They use raw ``httpx`` to avoid hard SDK dependencies — only the stdlib and
httpx are required.

Environment variables
---------------------
ANTHROPIC_API_KEY   – required for AnthropicBackend
OPENAI_API_KEY      – required for OpenAIBackend
OPENAI_BASE_URL     – optional, defaults to https://api.openai.com/v1

Usage::

    from agents.backends import AnthropicBackend, OpenAIBackend
    from agents.llm_scaffold import LLMScaffoldAgent

    agent = LLMScaffoldAgent(backend=AnthropicBackend())
    agent = LLMScaffoldAgent(backend=OpenAIBackend(model="gpt-4o"))
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    """Cumulative token counter across calls."""
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calls": self.calls,
        }


def _retry_request(
    func,
    *,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    retryable_statuses: tuple[int, ...] = (429, 500, 502, 503, 529),
) -> httpx.Response:
    """Call *func* with exponential backoff on transient HTTP errors."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = func()
            if response.status_code in retryable_statuses and attempt < max_retries:
                wait = backoff_base ** attempt
                # Respect Retry-After header if present
                retry_after = response.headers.get("retry-after")
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except ValueError:
                        pass
                logger.warning(
                    "retryable %d on attempt %d, waiting %.1fs",
                    response.status_code, attempt + 1, wait,
                )
                time.sleep(wait)
                continue
            return response
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = backoff_base ** attempt
                logger.warning("network error on attempt %d: %s, waiting %.1fs", attempt + 1, exc, wait)
                time.sleep(wait)
            else:
                raise
    raise last_exc  # type: ignore[misc]   # unreachable in practice


# ---------------------------------------------------------------------------
# Anthropic Messages API
# ---------------------------------------------------------------------------

class AnthropicBackend:
    """Calls the Anthropic Messages API via httpx.

    Parameters
    ----------
    model : str
        Model name. Defaults to ``claude-sonnet-4-20250514``.
    api_key : str | None
        API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
    max_tokens : int
        Maximum tokens in the response.
    temperature : float
        Sampling temperature (0 = deterministic).
    timeout : float
        HTTP request timeout in seconds.
    max_retries : int
        Number of automatic retries on transient errors.
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "AnthropicBackend requires an API key. "
                "Set ANTHROPIC_API_KEY or pass api_key=."
            )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.usage = TokenUsage()
        self._client = httpx.Client(timeout=self.timeout)

    def complete(self, messages: list[dict[str, str]], *, purpose: str) -> str:
        """Send messages to the Anthropic Messages API and return the text response."""
        # Anthropic API expects system as a top-level field, not in messages
        system_text = ""
        api_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_text = msg["content"]
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": api_messages,
        }
        if system_text:
            body["system"] = system_text

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }

        logger.info("anthropic request: model=%s purpose=%s messages=%d", self.model, purpose, len(api_messages))

        response = _retry_request(
            lambda: self._client.post(self.API_URL, json=body, headers=headers),
            max_retries=self.max_retries,
        )

        if response.status_code != 200:
            error_body = response.text[:500]
            raise RuntimeError(
                f"Anthropic API error {response.status_code}: {error_body}"
            )

        data = response.json()

        # Track usage
        usage = data.get("usage", {})
        self.usage.record(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

        # Extract text from content blocks
        content = data.get("content", [])
        texts = [block["text"] for block in content if block.get("type") == "text"]
        result = "\n".join(texts)

        logger.info(
            "anthropic response: purpose=%s tokens=%d+%d",
            purpose,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
        return result

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# OpenAI Chat Completions API (also covers compatible providers)
# ---------------------------------------------------------------------------

class OpenAIBackend:
    """Calls the OpenAI Chat Completions API (or any compatible endpoint).

    Parameters
    ----------
    model : str
        Model name. Defaults to ``gpt-4o``.
    api_key : str | None
        API key. Falls back to ``OPENAI_API_KEY`` env var.
    base_url : str | None
        API base URL. Falls back to ``OPENAI_BASE_URL`` or the default
        OpenAI endpoint. Useful for Azure, Together, Fireworks, etc.
    max_tokens : int
        Maximum tokens in the response.
    temperature : float
        Sampling temperature.
    timeout : float
        HTTP request timeout in seconds.
    max_retries : int
        Number of automatic retries on transient errors.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OpenAIBackend requires an API key. "
                "Set OPENAI_API_KEY or pass api_key=."
            )
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.usage = TokenUsage()
        self._client = httpx.Client(timeout=self.timeout)

    def complete(self, messages: list[dict[str, str]], *, purpose: str) -> str:
        """Send messages to the OpenAI Chat Completions API and return the text."""
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": msg["role"], "content": msg["content"]} for msg in messages],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"

        logger.info("openai request: model=%s purpose=%s url=%s", self.model, purpose, url)

        response = _retry_request(
            lambda: self._client.post(url, json=body, headers=headers),
            max_retries=self.max_retries,
        )

        if response.status_code != 200:
            error_body = response.text[:500]
            raise RuntimeError(
                f"OpenAI API error {response.status_code}: {error_body}"
            )

        data = response.json()

        # Track usage
        usage = data.get("usage", {})
        self.usage.record(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

        # Extract text
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenAI API returned no choices")
        result = choices[0].get("message", {}).get("content", "")

        logger.info(
            "openai response: purpose=%s tokens=%d+%d",
            purpose,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
        return result

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_backend(
    provider: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> AnthropicBackend | OpenAIBackend:
    """Create a backend by provider name.

    Parameters
    ----------
    provider : str
        One of ``"anthropic"`` or ``"openai"``.
    model : str | None
        Override the default model for the provider.
    api_key : str | None
        Override the API key (otherwise uses env var).
    **kwargs
        Additional kwargs forwarded to the backend constructor.
    """
    if provider == "anthropic":
        kw: dict[str, Any] = {**kwargs}
        if model:
            kw["model"] = model
        if api_key:
            kw["api_key"] = api_key
        return AnthropicBackend(**kw)
    if provider == "openai":
        kw = {**kwargs}
        if model:
            kw["model"] = model
        if api_key:
            kw["api_key"] = api_key
        return OpenAIBackend(**kw)
    raise ValueError(f"unknown provider: {provider!r}  (expected 'anthropic' or 'openai')")
