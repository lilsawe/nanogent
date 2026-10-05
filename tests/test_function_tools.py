"""Tests for function tools (@tool decorator) and schema generation."""

from __future__ import annotations

import asyncio

from nanogent.tools import FunctionTool, Tool, schema_from_callable, tool


def run(coro):
    return asyncio.run(coro)


@tool
def word_count(text: str) -> int:
    """统计文本的单词数。

    Args:
        text: 待统计的文本
    """
    return len(text.split())


@tool(name="shout")
def to_upper(text: str, times: int = 1) -> str:
    """把文本变大写并重复若干次。

    Args:
        text: 输入文本
        times: 重复次数
    """
    return text.upper() * times


def test_decorator_returns_tool():
    assert isinstance(word_count, Tool)
    assert word_count.name == "word_count"
    assert word_count.description == "统计文本的单词数。"


def test_schema_generated_from_annotations():
    schema = schema_from_callable(word_count._fn)
    assert schema["properties"]["text"]["type"] == "string"
    assert schema["properties"]["text"]["description"] == "待统计的文本"
    assert schema["required"] == ["text"]


def test_optional_parameter_is_not_required():
    schema = to_upper.parameters
    assert schema["properties"]["times"]["type"] == "integer"
    assert schema["required"] == ["text"]


def test_type_mapping_for_containers_and_optionals():
    def sample(items: list[str], mapping: dict, flag: bool = False, ratio: float = 0.5, note: str | None = None) -> str:
        """示例函数。"""
        return "ok"

    props = schema_from_callable(sample)["properties"]
    assert props["items"]["type"] == "array"
    assert props["mapping"]["type"] == "object"
    assert props["flag"]["type"] == "boolean"
    assert props["ratio"]["type"] == "number"
    assert props["note"]["type"] == "string"


def test_custom_name_overrides_function_name():
    assert to_upper.name == "shout"


def test_execute_sync_function():
    assert run(word_count.execute(text="a b c")) == "3"


def test_execute_serializes_non_string_results():
    assert run(word_count.execute(text="one two")) == "2"


def test_execute_async_function():
    @tool
    async def double(value: int) -> int:
        """翻倍。

        Args:
            value: 输入值
        """
        return value * 2

    assert run(double.execute(value=21)) == "42"


def test_execute_reports_bad_arguments_instead_of_raising():
    result = run(word_count.execute(wrong="x"))
    assert result.startswith("Error: 工具参数不匹配")


def test_execute_catches_tool_exceptions():
    @tool
    def boom() -> str:
        """总是抛异常。"""
        raise ValueError("boom")

    result = run(boom.execute())
    assert "Error: ValueError: boom" in result


def test_to_openai_format_shape():
    spec = word_count.to_openai_format()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "word_count"
    assert "parameters" in spec["function"]


def test_function_tool_requires_callable():
    import pytest

    with pytest.raises(TypeError):
        FunctionTool("not callable")
