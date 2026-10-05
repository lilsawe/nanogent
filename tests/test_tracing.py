import asyncio
import json
from dataclasses import dataclass
from typing import Any

from nanogent.agent import Agent
from nanogent.tools import CalculatorTool, ToolRegistry
from nanogent.tracing import TraceRecorder


@dataclass
class FakeToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class FakeResponse:
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class FakeLLM:
    responses: list[FakeResponse]
    calls: int = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> FakeResponse:
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_agent_writes_trace_events(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    tracer = TraceRecorder(trace_path, run_id="test_run")
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    llm = FakeLLM(
        responses=[
            FakeResponse(
                tool_calls=[
                    FakeToolCall(
                        id="call_1",
                        name="calculator",
                        arguments={"expression": "2 + 3"},
                    )
                ],
            ),
            FakeResponse(content="2 + 3 = 5"),
        ]
    )
    agent = Agent(llm=llm, tools=registry, tracer=tracer)

    result = asyncio.run(agent.run("What is 2 + 3?"))

    assert result == "2 + 3 = 5"
    event_names = [event["event"] for event in tracer.events]
    assert "llm_request" in event_names
    assert "tool_call" in event_names
    assert "tool_result" in event_names
    assert "final_response" in event_names

    persisted = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert persisted[0]["run_id"] == "test_run"
    assert any(event["event"] == "tool_call" for event in persisted)
