"""Tests for plugin discovery and registry filtering."""

from __future__ import annotations

import asyncio

from nanogent.tools import (
    Tool,
    create_default_registry,
    load_plugin_tools,
    tool,
)


@tool
def ping(host: str) -> str:
    """探测主机。

    Args:
        host: 主机名
    """
    return f"pong:{host}"


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "回显输入"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs) -> str:
        return "echo!"


class FakeEntryPoint:
    def __init__(self, obj, should_fail=False, name="fake"):
        self._obj = obj
        self._should_fail = should_fail
        self.name = name

    def load(self):
        if self._should_fail:
            raise ImportError("broken plugin")
        return self._obj


def fake_entry_points(group=None):
    return [
        FakeEntryPoint(ping),
        FakeEntryPoint(EchoTool()),
        FakeEntryPoint([ping, EchoTool()]),
        FakeEntryPoint(None, should_fail=True),
        FakeEntryPoint(12345),
    ]


def test_load_plugin_tools_accepts_callables_tools_and_lists(monkeypatch):
    monkeypatch.setattr("nanogent.tools.entry_points", fake_entry_points)

    tools = load_plugin_tools()

    names = [t.name for t in tools]
    assert "ping" in names
    assert "echo" in names
    assert len(tools) == 4  # 1 callable + 1 Tool + 列表里的 2 个（坏插件与非可调用对象被跳过）


def test_broken_plugin_is_skipped(monkeypatch):
    monkeypatch.setattr("nanogent.tools.entry_points", lambda group=None: [FakeEntryPoint(None, should_fail=True)])
    assert load_plugin_tools() == []


def test_plugin_tools_are_usable(monkeypatch):
    monkeypatch.setattr("nanogent.tools.entry_points", lambda group=None: [FakeEntryPoint(ping)])
    tools = {t.name: t for t in load_plugin_tools()}
    assert asyncio.run(tools["ping"].execute(host="db")) == "pong:db"


def test_registry_skips_plugins_when_disabled(monkeypatch):
    monkeypatch.setattr("nanogent.tools.entry_points", fake_entry_points)

    with_plugins = create_default_registry()
    without = create_default_registry(plugins=False)

    assert "ping" in with_plugins.tools
    assert "ping" not in without.tools
    assert set(without.tools) == {"execute_python", "read_file", "write_file",
                                  "calculator", "web_search", "glob", "grep"}


def test_registry_include_and_exclude():
    only = create_default_registry(include=["glob", "grep"], plugins=False)
    assert set(only.tools) == {"glob", "grep"}

    trimmed = create_default_registry(exclude=["execute_python", "web_search"], plugins=False)
    assert "execute_python" not in trimmed.tools
    assert "read_file" in trimmed.tools
