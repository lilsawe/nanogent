"""Evaluation runner for nanogent tool-calling behavior."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanogent.agent import Agent
from nanogent.llm import LLMClient, LLMResponse, ToolCall
from nanogent.tools import create_default_registry
from nanogent.tracing import TraceRecorder


@dataclass
class EvalTask:
    """One evaluation task loaded from a JSONL file."""

    id: str
    prompt: str
    expected_tool: str | None = None
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    expected_substrings: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result for one evaluation task."""

    task_id: str
    passed: bool
    final_answer: str
    called_tools: list[str]
    trace_path: str
    failures: list[str] = field(default_factory=list)


class ScriptedEvalLLM:
    """
    Deterministic offline LLM for validating the eval pipeline.

    Offline mode is not a model-quality benchmark. It exercises the agent loop,
    tool execution, tracing, and report generation without requiring an API key.
    """

    def __init__(self, task: EvalTask):
        self.task = task
        self.calls = 0
        self.model = "scripted-eval-llm"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls += 1

        if self.calls == 1 and self.task.expected_tool:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=f"{self.task.id}_call_1",
                        name=self.task.expected_tool,
                        arguments=self.task.tool_arguments,
                    )
                ],
            )

        tool_messages = [m["content"] for m in messages if m.get("role") == "tool"]
        if tool_messages:
            return LLMResponse(content=f"Tool result: {tool_messages[-1]}")
        return LLMResponse(content="No tool was required.")


def load_tasks(path: str | Path) -> list[EvalTask]:
    tasks: list[EvalTask] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            try:
                tasks.append(EvalTask(**data))
            except TypeError as e:
                raise ValueError(f"Invalid eval task at line {line_no}: {e}") from e
    return tasks


async def run_eval_tasks(
    tasks: list[EvalTask],
    *,
    offline: bool,
    out_dir: str | Path,
) -> list[EvalResult]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: list[EvalResult] = []
    for task in tasks:
        trace_path = out_path / f"{task.id}.trace.jsonl"
        if trace_path.exists():
            trace_path.unlink()

        tracer = TraceRecorder(trace_path, run_id=task.id)
        llm = ScriptedEvalLLM(task) if offline else LLMClient()
        agent = Agent(llm=llm, tools=create_default_registry(), tracer=tracer)
        final_answer = await agent.run(task.prompt)

        called_tools = [
            event["tool_name"]
            for event in tracer.events
            if event["event"] == "tool_call"
        ]
        failures: list[str] = []

        if task.expected_tool and task.expected_tool not in called_tools:
            failures.append(f"expected tool '{task.expected_tool}', got {called_tools}")

        for needle in task.expected_substrings:
            if needle not in final_answer:
                failures.append(f"missing expected substring: {needle!r}")

        results.append(
            EvalResult(
                task_id=task.id,
                passed=not failures,
                final_answer=final_answer,
                called_tools=called_tools,
                trace_path=str(trace_path),
                failures=failures,
            )
        )

    write_reports(results, out_path)
    return results


def write_reports(results: list[EvalResult], out_dir: Path) -> None:
    data = [result.__dict__ for result in results]
    (out_dir / "eval_results.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    passed = sum(1 for result in results if result.passed)
    lines = [
        "# nanogent Eval Report",
        "",
        f"Passed: {passed} / {len(results)}",
        "",
        "| Task | Passed | Tools | Trace |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        status = "yes" if result.passed else "no"
        tools = ", ".join(result.called_tools) or "-"
        lines.append(f"| {result.task_id} | {status} | {tools} | `{result.trace_path}` |")
        if result.failures:
            lines.append(f"| {result.task_id} failures |  | {'; '.join(result.failures)} |  |")

    (out_dir / "eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def _amain(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run nanogent evaluation tasks.")
    parser.add_argument(
        "--tasks",
        default="evals/tool_call_tasks.jsonl",
        help="Path to JSONL eval tasks.",
    )
    parser.add_argument(
        "--out-dir",
        default="eval_runs",
        help="Directory for traces and reports.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use a deterministic scripted LLM instead of a real API call.",
    )
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks)
    results = await run_eval_tasks(tasks, offline=args.offline, out_dir=args.out_dir)
    passed = sum(1 for result in results if result.passed)
    print(f"nanogent eval: {passed}/{len(results)} passed")
    print(f"Report: {Path(args.out_dir) / 'eval_report.md'}")
    return 0 if passed == len(results) else 1


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(_amain(argv)))


if __name__ == "__main__":
    main()
