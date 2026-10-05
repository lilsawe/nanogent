#!/usr/bin/env python3
"""
nanogent CLI — a minimal but complete AI agent you can run locally.

Usage:
    export DEEPSEEK_API_KEY="your-deepseek-api-key"
    nanogent

Then type your requests. The agent will use tools (execute code, read/write
files, calculate, search) to help you. Type /quit to exit, /reset to clear
conversation history.

Example session:
    You > What is 15 * 23 + the square root of 144?
    Agent > Let me calculate that.
            [tool: calculator] → 15 * 23 + sqrt(144)
            [tool result: 357.0]
            Agent > 15 × 23 + √144 = 357
"""

from __future__ import annotations

import asyncio
import sys

from nanogent.agent import Agent
from nanogent.llm import LLMClient
from nanogent.tools import create_default_registry


# ---------------------------------------------------------------------------
# Color helpers for terminal output
# ---------------------------------------------------------------------------

class Colors:
    """ANSI escape codes for terminal coloring."""
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_banner() -> None:
    print(Colors.CYAN + r"""
 _ __   __ _ _ __   ___   __ _  ___ _ __ | |_
| '_ \ / _` | '_ \ / _ \ / _` |/ _ \ '_ \| __|
| | | | (_| | | | | (_) | (_| |  __/ | | | |_
|_| |_|\__,_|_| |_|\___/ \__, |\___|_| |_|\__|
                         |___/
    """ + Colors.RESET)
    print(Colors.DIM + "  Type /help for commands, /quit to exit\n" + Colors.RESET)


def print_help() -> None:
    print(f"""
{Colors.CYAN}Commands:{Colors.RESET}
  {Colors.GREEN}/quit{Colors.RESET}    Exit the agent
  {Colors.GREEN}/reset{Colors.RESET}   Clear conversation history
  {Colors.GREEN}/tools{Colors.RESET}   List available tools
  {Colors.GREEN}/help{Colors.RESET}    Show this message
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print_banner()

    # --- Initialize ---
    try:
        llm = LLMClient()
    except ValueError as e:
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        print("Set your DeepSeek API key:  export DEEPSEEK_API_KEY='your-deepseek-api-key'")
        sys.exit(1)

    tools = create_default_registry()
    agent = Agent(llm=llm, tools=tools)
    agent.reset()

    print(f"{Colors.GREEN}✓{Colors.RESET} LLM: {llm.model}")
    print(f"{Colors.GREEN}✓{Colors.RESET} Tools: {', '.join(tools.tools.keys())}")
    print()

    # --- REPL loop ---
    while True:
        try:
            user_input = input(f"{Colors.CYAN}You > {Colors.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/quit":
                print("Goodbye!")
                break
            elif cmd == "/reset":
                agent.reset()
                print(f"{Colors.DIM}Conversation reset.{Colors.RESET}")
            elif cmd == "/tools":
                for tool in tools.tools.values():
                    print(f"  {Colors.GREEN}{tool.name}{Colors.RESET} — {tool.description[:100]}...")
            elif cmd == "/help":
                print_help()
            else:
                print(f"Unknown command: {user_input}")
            continue

        # --- Run agent ---
        print(f"{Colors.DIM}Thinking...{Colors.RESET}", end="\r")
        try:
            response = await agent.run(user_input)
            print(f"{Colors.GREEN}Agent > {Colors.RESET}{response}")
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        print()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
