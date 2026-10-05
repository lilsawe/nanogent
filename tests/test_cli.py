"""Tests for CLI parsing and helpers (no network calls)."""

from __future__ import annotations

import pytest

from nanogent.cli import build_parser, compose_prompt, count_tool_calls
from nanogent.tracing import TraceRecorder


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.prompt == []
    assert args.json is False
    assert args.max_iterations == 10
    assert args.resume is False


def test_parser_positional_and_flags():
    args = build_parser().parse_args(["总结这段日志", "--json", "-q", "--session", "work"])
    assert args.prompt == ["总结这段日志"]
    assert args.json is True
    assert args.quiet is True
    assert args.session == "work"


def test_version_flag_exits():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0


def test_compose_prompt_merges_stdin():
    merged = compose_prompt(["分析根因"], "line1\nline2")
    assert "分析根因" in merged
    assert "line1" in merged


def test_compose_prompt_stdin_only():
    assert compose_prompt([], "only stdin") == "only stdin"


def test_compose_prompt_empty():
    assert compose_prompt([], None) is None


def test_count_tool_calls():
    tracer = TraceRecorder()
    tracer.record("tool_call", name="glob")
    tracer.record("tool_result", name="glob")
    tracer.record("llm_request")
    assert count_tool_calls(tracer) == 2
    assert count_tool_calls(None) == 0
