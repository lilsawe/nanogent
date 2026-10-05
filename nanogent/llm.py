"""
LLM client for DeepSeek API (OpenAI-compatible format).

Provides a thin wrapper around the OpenAI SDK configured for DeepSeek's
endpoint, with support for tool/function calling.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from openai import AsyncOpenAI

from nanogent.errors import ConfigError, LLMError

API_KEY_ENV_VARS = ("NANOGENT_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY")


def _first_env(names: tuple[str, ...]) -> str | None:
    """Return the first environment variable that is set and non-empty."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


@dataclass
class ToolCall:
    """Represents a tool call requested by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Structured response from the LLM."""
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@runtime_checkable
class ChatModel(Protocol):
    """Agent 需要的最小模型接口。

    任何实现 "model" 属性与 "async chat(messages, tools)" 的对象都可注入 Agent——
    这让离线评测可以注入脚本化模型，测试也不必依赖真实 API。
    """

    model: str

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> "LLMResponse": ...


class LLMClient:
    """
    Async client for DeepSeek's chat completion API.

    Usage:
        client = LLMClient(api_key="sk-xxx")
        response = await client.chat(messages, tools=[...])
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key or _first_env(API_KEY_ENV_VARS)
        if not self.api_key:
            raise ConfigError(
                "No API key found. Set one of " + ", ".join(API_KEY_ENV_VARS)
                + " (e.g. export NANOGENT_API_KEY=sk-...), or pass --api-key."
            )

        self.model: str = model or os.getenv("NANOGENT_MODEL") or "deepseek-chat"
        self.base_url = base_url or os.getenv("NANOGENT_BASE_URL", "https://api.deepseek.com")
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request to DeepSeek.

        Args:
            messages: Conversation history in OpenAI format.
            tools: Optional list of tool definitions (OpenAI function format).

        Returns:
            LLMResponse with either text content or tool calls.

        Raises:
            Exception: On API errors, with the error message included.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            completion = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise LLMError(f"LLM API error: {e}") from e

        choice = completion.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                # Parse JSON arguments string into dict
                import json

                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
        )
