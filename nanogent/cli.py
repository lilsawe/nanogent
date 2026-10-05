#!/usr/bin/env python3
"""
nanogent CLI - a small but genuinely useful AI agent you can run locally.

Three ways to use it:

  1. Interactive:      nanogent
  2. One-shot:         nanogent "计算 15*23 并把结果写到 result.txt"
  3. Piped input:      cat error.log | nanogent "这段日志的根因是什么？"
                       git diff | nanogent "帮我写 commit message"

Script-friendly output:
  nanogent --json "..."        # 输出结构化 JSON
  nanogent --quiet "..."       # 只打印答案（适合管道/CI）

Sessions:
  nanogent --session work      # 保存到命名会话
  nanogent --continue          # 继续最近一次会话
  nanogent --list-sessions     # 列出所有会话
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from nanogent import __version__
from nanogent.agent import Agent
from nanogent.llm import LLMClient
from nanogent.session import SessionStore
from nanogent.tools import create_default_registry, load_plugin_tools
from nanogent.tracing import TraceRecorder


class Colors:
    """ANSI escape codes; disabled automatically when not a TTY."""

    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        for name in ("CYAN", "GREEN", "YELLOW", "RED", "DIM", "RESET"):
            setattr(cls, name, "")


_BT = chr(96)
BANNER = (
    "\n"
    " _ __   __ _ _ __   ___   __ _  ___ _ __ | |_\n"
    "| '_ \\ / " + _BT + " | '_ \\ / _ \\ / " + _BT + " |/ _ \\ '_ \\| __|\n"
    "| | | | (_| | | | | (_) | (_| |  __/ | | | |_\n"
    "|_| |_|\\__,_|_| |_|\\___/ \\__, |\\___|_| |_|\\__|\n"
    "                         |___/\n"
)


def print_banner() -> None:
    print(Colors.CYAN + BANNER + Colors.RESET)
    print(Colors.DIM + "  /help 查看命令 · /quit 退出 · 也可 nanogent \"问题\" 单次执行\n" + Colors.RESET)


def print_help() -> None:
    print(f"""
{Colors.CYAN}Commands:{Colors.RESET}
  {Colors.GREEN}/quit{Colors.RESET}            退出
  {Colors.GREEN}/reset{Colors.RESET}           清空对话历史
  {Colors.GREEN}/tools{Colors.RESET}           列出可用工具
  {Colors.GREEN}/save [name]{Colors.RESET}     保存会话（默认 default）
  {Colors.GREEN}/sessions{Colors.RESET}        列出已保存会话
  {Colors.GREEN}/help{Colors.RESET}            显示本帮助
""")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanogent",
        description="A small AI agent runtime: LLM + tools + tracing + eval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  nanogent                              # 交互模式\n"
            "  nanogent \"把 data.csv 转成 JSON\"       # 单次执行\n"
            "  cat app.log | nanogent \"分析根因\"      # 管道输入\n"
            "  nanogent --json \"...\"                 # JSON 输出\n"
            "  nanogent --continue                   # 继续上次会话\n"
        ),
    )
    parser.add_argument("prompt", nargs="*", help="单次执行的任务；省略则进入交互模式")
    parser.add_argument("-m", "--model", help="模型名（默认读 NANOGENT_MODEL，或 deepseek-chat）")
    parser.add_argument("--base-url", help="OpenAI 兼容 API 地址（默认读 NANOGENT_BASE_URL）")
    parser.add_argument("--api-key", help="API Key（默认读 NANOGENT_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY）")
    parser.add_argument("--max-iterations", type=int, default=10, help="单轮最多 LLM 调用次数（默认 10）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（answer/model/elapsed_ms/tool_calls）")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默：不打印 banner 与状态提示")
    parser.add_argument("--color", choices=["auto", "always", "never"], default="auto", help="颜色开关")
    parser.add_argument("--session", help="会话名（用于保存/续接）")
    parser.add_argument("--continue", dest="resume", action="store_true", help="继续最近一次会话")
    parser.add_argument("--list-sessions", action="store_true", help="列出已保存会话后退出")
    parser.add_argument("--tools", action="store_true", help="列出可用工具后退出")
    parser.add_argument("--no-plugins", action="store_true", help="不加载第三方工具插件")
    parser.add_argument("--version", action="version", version=f"nanogent {__version__}")
    return parser


def read_piped_stdin() -> str | None:
    """Return piped stdin content, or None when running interactively."""
    try:
        if sys.stdin is None or sys.stdin.isatty():
            return None
        data = sys.stdin.read()
    except Exception:
        return None
    data = (data or "").strip()
    return data or None


def compose_prompt(parts: list[str], stdin_text: str | None) -> str | None:
    prompt = " ".join(parts).strip()
    if stdin_text and prompt:
        return f"{prompt}\n\n--- 输入内容 ---\n{stdin_text}"
    if stdin_text:
        return stdin_text
    return prompt or None


def build_agent(args: argparse.Namespace, trace_path: Path | None = None):
    llm = LLMClient(api_key=args.api_key, model=args.model, base_url=args.base_url)
    tools = create_default_registry(plugins=not getattr(args, "no_plugins", False))
    tracer = TraceRecorder(path=trace_path) if trace_path else None
    agent = Agent(llm=llm, tools=tools, max_iterations=args.max_iterations, tracer=tracer)
    agent.reset()
    return agent, tools, llm


def count_tool_calls(tracer: TraceRecorder | None) -> int:
    if not tracer:
        return 0
    return sum(1 for event in tracer.events if "tool" in str(event.get("event", "")))


async def run_one_shot(agent, tools, prompt: str, args: argparse.Namespace) -> int:
    tracer = agent.tracer
    started = time.perf_counter()
    try:
        answer = await agent.run(prompt)
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        else:
            print(f"{Colors.RED}Error: {type(exc).__name__}: {exc}{Colors.RESET}", file=sys.stderr)
        return 1

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if args.json:
        print(json.dumps({
            "answer": answer,
            "model": agent.llm.model,
            "elapsed_ms": elapsed_ms,
            "tool_calls": count_tool_calls(tracer),
            "session": args.session,
        }, ensure_ascii=False, indent=2))
    else:
        print(answer)

    if args.session:
        path = SessionStore().save(args.session, agent.messages, model=agent.llm.model)
        if not args.quiet and not args.json:
            print(f"{Colors.DIM}会话已保存: {path}{Colors.RESET}", file=sys.stderr)
    return 0


async def run_repl(agent, tools, args: argparse.Namespace) -> int:
    store = SessionStore()
    session_name = args.session or "default"

    if not args.quiet:
        print_banner()
        print(f"{Colors.GREEN}✓{Colors.RESET} 模型: {agent.llm.model}")
        print(f"{Colors.GREEN}✓{Colors.RESET} 工具: {', '.join(tools.tools.keys())}")
        print()

    while True:
        try:
            # input() 是阻塞调用，放到线程里，避免卡住事件循环
            user_input = (await asyncio.to_thread(input, f"{Colors.CYAN}You > {Colors.RESET}")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd, _, rest = user_input.partition(" ")
            cmd = cmd.lower()
            if cmd in ("/quit", "/exit"):
                print("Bye!")
                break
            if cmd == "/reset":
                agent.reset()
                print(f"{Colors.DIM}对话已清空{Colors.RESET}")
            elif cmd == "/tools":
                for tool in tools.tools.values():
                    print(f"  {Colors.GREEN}{tool.name}{Colors.RESET} - {tool.description[:90]}")
            elif cmd == "/save":
                name = rest.strip() or session_name
                path = store.save(name, agent.messages, model=agent.llm.model)
                print(f"{Colors.DIM}已保存: {path}{Colors.RESET}")
            elif cmd == "/sessions":
                items = store.list()
                if not items:
                    print(f"{Colors.DIM}（暂无会话）{Colors.RESET}")
                for item in items:
                    label = f"  {Colors.GREEN}{item['name']}{Colors.RESET}"
                    print(f"{label} - {item['message_count']} 条消息 - {item['saved_at']}")
            elif cmd == "/help":
                print_help()
            else:
                print(f"未知命令: {user_input}（用 /help 查看）")
            continue

        try:
            answer = await agent.run(user_input)
            print(f"{Colors.GREEN}Agent >{Colors.RESET} {answer}\n")
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}已中断{Colors.RESET}")
        except Exception as exc:
            print(f"{Colors.RED}Error: {type(exc).__name__}: {exc}{Colors.RESET}\n")
    return 0


async def amain(args: argparse.Namespace) -> int:
    if args.color == "never" or (args.color == "auto" and not sys.stdout.isatty()):
        Colors.disable()

    store = SessionStore()

    if args.list_sessions:
        items = store.list()
        if not items:
            print("（暂无会话）")
        for item in items:
            print(f"{item['name']}\t{item['message_count']} 条消息\t{item['saved_at']}")
        return 0

    if args.tools:
        plugin_names = set()
        if not args.no_plugins:
            plugin_names = {t.name for t in load_plugin_tools()}
        for tool in create_default_registry(plugins=not args.no_plugins).tools.values():
            source = "plugin" if tool.name in plugin_names else "builtin"
            print(f"{tool.name}\t[{source}]\t{tool.description.splitlines()[0][:90]}")
        return 0

    stdin_text = read_piped_stdin()
    prompt = compose_prompt(args.prompt, stdin_text)

    trace_path = Path.home() / ".nanogent" / "traces" / f"run-{int(time.time())}.jsonl"

    try:
        agent, tools, _llm = build_agent(args, trace_path=trace_path)
    except ValueError as exc:
        print(f"{Colors.RED}{exc}{Colors.RESET}", file=sys.stderr)
        print("\n可用的 API Key 环境变量：NANOGENT_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY", file=sys.stderr)
        print("示例： export NANOGENT_API_KEY=sk-xxx", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"{Colors.RED}初始化失败: {type(exc).__name__}: {exc}{Colors.RESET}", file=sys.stderr)
        return 2

    # 续接会话
    if args.resume or args.session:
        name = args.session or store.latest_name()
        if name:
            messages = store.load(name)
            if messages:
                agent.messages = messages
                if not args.quiet:
                    print(f"{Colors.DIM}已恢复会话 {name}（{len(messages)} 条消息）{Colors.RESET}", file=sys.stderr)

    if prompt:
        return await run_one_shot(agent, tools, prompt, args)
    return await run_repl(agent, tools, args)


def run() -> None:
    """Console script entry point."""
    args = build_parser().parse_args()
    try:
        sys.exit(asyncio.run(amain(args)))
    except KeyboardInterrupt:
        sys.exit(130)


main = run

if __name__ == "__main__":
    run()
