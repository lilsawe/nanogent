"""Tests for the LLM client (no network: the OpenAI client is replaced by a fake)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanogent.errors import ConfigError, LLMError
from nanogent.llm import LLMClient

ENV_VARS = ("NANOGENT_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY")


def _completion(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _client_with(return_value=None, side_effect=None) -> LLMClient:
    client = LLMClient(api_key="sk-test")
    create = AsyncMock(return_value=return_value, side_effect=side_effect)
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    client._create = create
    return client


def test_missing_api_key_raises_config_error(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ConfigError) as excinfo:
        LLMClient()

    assert "No API key found" in str(excinfo.value)
    # 兼容既有调用方：ConfigError 同时是 ValueError
    assert isinstance(excinfo.value, ValueError)


def test_api_key_can_come_from_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NANOGENT_API_KEY", "sk-from-env")

    assert LLMClient().api_key == "sk-from-env"


def test_model_defaults_to_deepseek_chat(monkeypatch):
    monkeypatch.delenv("NANOGENT_MODEL", raising=False)
    assert LLMClient(api_key="sk-test").model == "deepseek-chat"


def test_chat_returns_plain_content():
    client = _client_with(return_value=_completion(content="hello"))
    response = asyncio.run(client.chat([{"role": "user", "content": "hi"}]))

    assert response.content == "hello"
    assert response.tool_calls == []


def test_chat_parses_tool_calls():
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="calculator", arguments=json.dumps({"expression": "1+1"})),
    )
    client = _client_with(return_value=_completion(content=None, tool_calls=[tool_call]))
    response = asyncio.run(client.chat([{"role": "user", "content": "1+1?"}]))

    assert response.has_tool_calls
    assert response.tool_calls[0].name == "calculator"
    assert response.tool_calls[0].arguments == {"expression": "1+1"}


def test_chat_passes_tools_when_provided():
    client = _client_with(return_value=_completion(content="ok"))
    tool_defs = [{"type": "function", "function": {"name": "glob", "parameters": {}}}]

    asyncio.run(client.chat([{"role": "user", "content": "hi"}], tools=tool_defs))

    kwargs = client._create.await_args.kwargs
    assert kwargs["tools"] == tool_defs
    assert kwargs["tool_choice"] == "auto"


def test_chat_wraps_api_errors():
    client = _client_with(side_effect=RuntimeError("boom"))

    with pytest.raises(LLMError) as excinfo:
        asyncio.run(client.chat([{"role": "user", "content": "hi"}]))

    assert "LLM API error" in str(excinfo.value)
