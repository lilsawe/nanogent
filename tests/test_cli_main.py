"""Tests for CLI entrypoints: --tools / --list-sessions / one-shot / --json / REPL."""

from __future__ import annotations

import asyncio
import builtins
import json
from types import SimpleNamespace

import pytest

from nanogent.cli import amain, build_parser
from nanogent.errors import ConfigError
from nanogent.tools import create_default_registry


class FakeAgent:
    """最小 Agent 替身：只实现 CLI 用到的接口。"""

    def __init__(self, answer: str = "fake answer"):
        self.llm = SimpleNamespace(model="fake-model")
        self.messages = [{"role": "system", "content": "system"}]
        self.tracer = None
        self._answer = answer

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": "system"}]

    async def run(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})
        return self._answer


def _patch_agent(monkeypatch, answer="fake answer"):
    def fake_build_agent(args, trace_path=None):
        return FakeAgent(answer), create_default_registry(plugins=False), SimpleNamespace(model="fake-model")

    monkeypatch.setattr("nanogent.cli.build_agent", fake_build_agent)


def _patch_sessions(monkeypatch, tmp_path):
    monkeypatch.setattr("nanogent.session.DEFAULT_SESSION_DIR", tmp_path)


def run_amain(argv):
    return asyncio.run(amain(build_parser().parse_args(argv)))


def test_tools_listing(capsys):
    assert run_amain(["--tools"]) == 0
    out = capsys.readouterr().out

    assert "execute_python" in out
    assert "[builtin]" in out


def test_list_sessions_when_empty(monkeypatch, tmp_path, capsys):
    _patch_sessions(monkeypatch, tmp_path)

    assert run_amain(["--list-sessions"]) == 0
    assert "暂无会话" in capsys.readouterr().out


def test_one_shot_prints_answer(monkeypatch, capsys):
    _patch_agent(monkeypatch, answer="42")
    assert run_amain(["-q", "计算 6*7"]) == 0
    assert "42" in capsys.readouterr().out


def test_one_shot_json_output(monkeypatch, capsys):
    _patch_agent(monkeypatch, answer="done")
    assert run_amain(["--json", "干活"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["answer"] == "done"
    assert payload["model"] == "fake-model"
    assert payload["tool_calls"] == 0


def test_one_shot_saves_session(monkeypatch, tmp_path, capsys):
    _patch_agent(monkeypatch, answer="saved")
    _patch_sessions(monkeypatch, tmp_path)

    assert run_amain(["-q", "--session", "work", "记住这件事"]) == 0
    assert (tmp_path / "work.json").exists()


def test_missing_api_key_returns_exit_code_2(monkeypatch, capsys):
    def boom(args, trace_path=None):
        raise ConfigError("No API key found")

    monkeypatch.setattr("nanogent.cli.build_agent", boom)

    assert run_amain(["hello"]) == 2
    err = capsys.readouterr().err
    assert "API Key" in err


def test_repl_exits_on_slash_quit(monkeypatch, capsys):
    _patch_agent(monkeypatch)
    answers = iter(["/quit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))

    assert run_amain(["-q"]) == 0
    assert "Bye" in capsys.readouterr().out


def test_repl_handles_reset_and_tools(monkeypatch, capsys):
    _patch_agent(monkeypatch, answer="answer")
    answers = iter(["/reset", "/tools", "/unknown", "/quit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))

    assert run_amain(["-q"]) == 0
    out = capsys.readouterr().out
    assert "对话已清空" in out
    assert "execute_python" in out
    assert "未知命令" in out
