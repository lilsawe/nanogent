"""
Core agent loop — the heart of the nanogent.

Implements the perception → thinking → action cycle:
1. Receive user input (perception)
2. Send conversation + tool definitions to the LLM (thinking)
3. If the LLM returns text → deliver to user (done)
4. If the LLM returns tool calls → execute them (action)
5. Feed tool results back into the conversation → go to step 2
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from nanogent.tools import ToolRegistry
from nanogent.tracing import TraceRecorder, preview

if TYPE_CHECKING:
    from nanogent.llm import ChatModel


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to tools. Follow these rules:

1. THINK before acting. If a task requires multiple steps, plan briefly before
   using tools.
2. Use tools when you need to: run code, read/write files, do math, or search
   for information.
3. When a tool returns an error, try to diagnose the problem and fix it. Do NOT
   repeat the same failing call.
4. After tool results come back, synthesize a clear answer for the user. Do NOT
   just dump raw tool output — explain what you found or did.
5. If you can answer without tools, just answer directly.
6. Respond in the same language the user used.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    An autonomous agent that uses an LLM to decide what to do next.

    The agent maintains a conversation history and runs a loop:
      LLM thinks → maybe calls tools → tools execute → results fed back → repeat

    Attributes:
        llm: The LLM client for chat completions.
        tools: Registry of available tools.
        max_iterations: Safety limit on how many LLM calls per user turn.
        messages: The full conversation history (system + user + assistant + tool).
    """

    def __init__(
        self,
        llm: ChatModel,
        tools: ToolRegistry,
        max_iterations: int = 10,
        system_prompt: str = SYSTEM_PROMPT,
        tracer: TraceRecorder | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        self.messages: list[dict[str, Any]] = []
        self.tracer = tracer

    def reset(self) -> None:
        """Clear conversation history, keeping only the system message."""
        self.messages = [
            {"role": "system", "content": self.system_prompt},
        ]

    async def run(self, user_input: str) -> str:
        """
        Run the agent on a single user message.

        Args:
            user_input: The user's message.

        Returns:
            The agent's final text response.
        """
        if not self.messages:
            self.reset()
        self.messages.append({"role": "user", "content": user_input})
        if self.tracer:
            self.tracer.record("user_message", content_preview=preview(user_input))

        tool_defs = self.tools.list_definitions()

        for iteration in range(self.max_iterations):
            # --- Step 1: Call the LLM ---
            start = self.tracer.timer() if self.tracer else None
            if self.tracer:
                self.tracer.record(
                    "llm_request",
                    iteration=iteration,
                    message_count=len(self.messages),
                    tool_count=len(tool_defs),
                )
            response = await self.llm.chat(self.messages, tools=tool_defs)
            if self.tracer:
                self.tracer.record(
                    "llm_response",
                    iteration=iteration,
                    elapsed_ms=self.tracer.elapsed_ms(start or self.tracer.timer()),
                    has_tool_calls=response.has_tool_calls,
                    content_preview=preview(response.content or ""),
                    tool_names=[tc.name for tc in (response.tool_calls or [])],
                )

            # --- Step 2a: LLM returned plain text → we're done ---
            if not response.has_tool_calls and response.content:
                self.messages.append({
                    "role": "assistant",
                    "content": response.content,
                })
                if self.tracer:
                    self.tracer.record(
                        "final_response",
                        iteration=iteration,
                        content_preview=preview(response.content),
                    )
                return response.content

            # --- Step 2b: LLM wants to call tools ---
            if response.has_tool_calls:
                # Record the assistant message with tool calls
                assistant_msg = {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
                self.messages.append(assistant_msg)

                # Execute each tool call and collect results
                tool_results = []
                for tc in response.tool_calls:
                    tool_start = self.tracer.timer() if self.tracer else None
                    if self.tracer:
                        self.tracer.record(
                            "tool_call",
                            iteration=iteration,
                            tool_name=tc.name,
                            arguments=tc.arguments,
                        )
                    result = await self.tools.execute(tc.name, tc.arguments)
                    if self.tracer:
                        self.tracer.record(
                            "tool_result",
                            iteration=iteration,
                            tool_name=tc.name,
                            elapsed_ms=self.tracer.elapsed_ms(
                                tool_start or self.tracer.timer()
                            ),
                            result_preview=preview(result),
                        )
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                # Feed results back into the conversation
                self.messages.extend(tool_results)

                # Loop back to let the LLM process the results
                continue

            # --- Step 2c: LLM returned neither text nor tool calls (edge case) ---
            if self.tracer:
                self.tracer.record("empty_model_response", iteration=iteration)
            return "(Agent produced no output — this may indicate an API issue.)"

        # --- Max iterations reached ---
        if self.tracer:
            self.tracer.record("max_iterations_reached", max_iterations=self.max_iterations)
        return (
            f"Agent stopped after {self.max_iterations} iterations without "
            "producing a final answer. The task may be too complex or the "
            "agent may be stuck in a loop. Try simplifying your request."
        )
