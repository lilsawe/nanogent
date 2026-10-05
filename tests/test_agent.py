import asyncio
from dataclasses import dataclass
from typing import Any

from nanogent.agent import Agent
from nanogent.tools import CalculatorTool, ToolRegistry


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


def test_agent_executes_tool_and_returns_final_answer() -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    llm = FakeLLM(
        responses=[
            FakeResponse(
                content=None,
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
    agent = Agent(llm=llm, tools=registry)

    result = asyncio.run(agent.run("What is 2 + 3?"))

    assert result == "2 + 3 = 5"
    assert llm.calls == 2
    assert any(
        message["role"] == "tool" and message["content"] == "5"
        for message in agent.messages
    )
