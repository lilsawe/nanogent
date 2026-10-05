# Mini-Agent

> A minimal AI Agent Runtime built from scratch in Python — agent loop, function calling, tool registry, JSONL tracing, and an evaluation harness.

[![CI](https://github.com/lilsawe/mini-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/lilsawe/mini-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Agent Runtime](https://img.shields.io/badge/Agent-Runtime-6f42c1.svg)](#architecture)

作者：[李耀彬 (Li Yaobin) · @lilsawe](https://github.com/lilsawe)　|　个人主页：https://github.com/lilsawe

一个从零实现的轻量级 AI Agent Runtime，用少量 Python 代码展示 Agent 的核心工作流：LLM 推理、工具调用、工具结果回传、多轮对话、CLI 交互，以及 Agent 行为 tracing / eval。

这个项目不是对 LangChain、CrewAI 等成熟框架的封装。代码刻意保持可读，方便快速看到 Agent loop、tool registry、function calling、异步执行、JSONL trace 和 evaluation harness 的具体实现。

## Highlights

- 从零实现 `perceive -> think -> act -> observe` Agent 循环
- 基于 DeepSeek 的 OpenAI-compatible function calling
- 可扩展工具系统：抽象 `Tool` 基类、注册表、统一 JSON Schema 描述
- 内置 5 个工具：Python 执行、文件读写、计算器、模拟搜索
- 支持多轮对话上下文和 `/reset` 重置
- 使用 `asyncio` 封装 LLM 调用和工具执行
- JSONL tracing：记录 user message、LLM request/response、tool_call、tool_result、final_response
- Evaluation harness：用 JSONL 任务集评估工具调用链路，输出 JSON + Markdown 报告
- 带单元测试，覆盖工具注册、计算器安全边界、Agent tool-call、tracing 和 eval 流程

## Architecture

```text
User
  |
  v
CLI / REPL
  |
  v
Agent.run()
  |
  +--> LLMClient.chat(messages, tools)
  |       |
  |       +--> text response ---------------> final answer
  |       |
  |       +--> tool_calls
  |              |
  |              v
  |        ToolRegistry.execute()
  |              |
  |              v
  |        tool result messages
  |              |
  +--------------+
```

## Project Structure

```text
.
├── mini_agent/
│   ├── __init__.py       # Public package exports
│   ├── __main__.py       # Enables module execution
│   ├── cli.py            # CLI / REPL entrypoint
│   ├── agent.py          # Core Agent loop and conversation state
│   ├── eval.py           # Tool-calling eval runner
│   ├── llm.py            # DeepSeek/OpenAI-compatible chat client
│   ├── tools.py          # Tool abstraction, registry, built-in tools
│   └── tracing.py        # JSONL trace recorder
├── evals/                # JSONL eval task sets
├── tests/                # Unit tests for tools and agent loop
├── pyproject.toml        # Package metadata, console script, pytest config
└── .env.example          # API key template
```

## Quick Start

### 1. Install dependencies

Using conda:

```bash
conda create -n mini-agent python=3.11
conda activate mini-agent
pip install -e ".[dev]"
```

### 2. Configure API key

```bash
cp .env.example .env
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

### 3. Run the agent

```bash
mini-agent
```

Example prompts:

```text
计算 (15 * 23 + sqrt(144)) / 2 的结果
在当前目录创建一个 hello.py，内容是打印 "Hello Agent"
读取刚才创建的 hello.py，然后用 Python 执行它
帮我分析一下：1 米/秒 的风速下，一个半径 5 米的水平轴风力发电机理论功率是多少？用贝茨极限算。
```

## Commands

```text
/help   Show CLI commands
/tools  List available tools
/reset  Clear conversation history
/quit   Exit the CLI
```

## Agent Eval & Tracing

Run the offline evaluation pipeline:

```bash
mini-agent-eval --offline --tasks evals/tool_call_tasks.jsonl --out-dir eval_runs
```

Offline mode uses a deterministic scripted LLM, so it does not need an API key.
It is useful for validating the Agent runtime, tool execution, trace capture,
and report generation. To evaluate a real model, omit `--offline` after setting
`DEEPSEEK_API_KEY`.

Generated artifacts:

```text
eval_runs/
├── calculator_basic.trace.jsonl
├── python_execution.trace.jsonl
├── eval_results.json
└── eval_report.md
```

Trace events include:

- `llm_request` / `llm_response`
- `tool_call` / `tool_result`
- `final_response`

## Tests

Run tests:

```bash
pytest
```

The tests use fake/scripted LLMs, so they do not require a real API key.

## Core Design

### Agent loop

`mini_agent/agent.py` keeps the conversation state and repeatedly calls the LLM until it receives a final text response or reaches the iteration limit. When the LLM returns tool calls, the agent executes them through `ToolRegistry` and appends results back into the message history.

When a `TraceRecorder` is attached, the loop records each LLM request/response,
tool call, tool result, and final answer as JSONL events. This makes it easier
to debug failed tool calls and build eval reports.

### Tool system

Each tool implements:

- `name`: function name exposed to the LLM
- `description`: natural-language instruction for when to use it
- `parameters`: JSON Schema argument definition
- `execute()`: async implementation

This mirrors OpenAI-compatible function calling while keeping the implementation easy to inspect.

### LLM client

`mini_agent/llm.py` wraps the OpenAI SDK and points it at DeepSeek's compatible endpoint. The wrapper normalizes model responses into a small `LLMResponse` dataclass so the rest of the project does not depend on SDK-specific response objects.

### Eval runner

`mini_agent/eval.py` loads JSONL tasks, runs them through the same Agent loop,
checks expected tool usage and answer substrings, then emits a Markdown report
plus machine-readable JSON results.

## Built-in Tools

| Tool | Purpose |
| --- | --- |
| `execute_python` | Runs a self-contained Python snippet in a subprocess |
| `read_file` | Reads UTF-8 text files with output truncation |
| `write_file` | Creates or overwrites text files |
| `calculator` | Evaluates math expressions with a restricted namespace |
| `web_search` | Simulated search placeholder for demonstrating tool flow |

## Interview Talking Points

- Why implement the Agent loop directly instead of hiding it behind a framework
- How function calling maps to tool registration and tool result messages
- Why the project uses async interfaces even though the demo is small
- How iteration limits and structured error strings prevent simple failure loops
- How JSONL traces help debug tool selection, tool failures, and loop behavior
- How to turn prompt/task examples into repeatable Agent evals
- Where production hardening would be added: filesystem sandboxing, user confirmation before writes, real search API, streaming output, persistent memory, and observability

## Security Notes

This is a local learning project. `execute_python` runs code in a subprocess, but it is not a full security sandbox. In production, code execution and file writing should be isolated with stricter permissions, path allowlists, resource limits, and explicit user confirmation.

## 设计取舍 / 已知限制 / 下一步

### 设计取舍（为什么这么做）

| 决策 | 选择 | 为什么 |
|---|---|---|
| 实现方式 | 从零手写，不封装 LangChain / CrewAI | 目标是**看清 Agent loop 的真实实现**，封装会遮蔽核心 |
| 观测方式 | 自研 JSONL tracing，不用 OpenTelemetry | 零依赖、可直接 diff 与断言，适合教学与面试演示 |
| 交互形态 | CLI / REPL，不做 Web UI | 降低运行门槛；前端不是本项目要证明的能力 |
| 工具系统 | 抽象 Tool 基类 + 注册表 + JSON Schema | 新增工具只需实现一个类，体现可扩展设计 |
| 模型接入 | 走 OpenAI-compatible 协议（默认 DeepSeek） | 换模型只改环境变量，不绑死供应商 |

### 已知限制（诚实说明）

- **单进程单会话**：没有并发、多用户隔离与鉴权
- **无持久化记忆**：上下文只存在内存里，重启即丢
- **无重试 / 限流 / 成本控制**：LLM 调用失败直接抛出
- **工具沙箱很弱**：Python 执行工具仅做基础限制，不能用于不可信输入
- **评测集规模小**：eval 只覆盖工具调用链路的少量任务，不是 benchmark

### 下一步（如果继续做）

1. 支持 MCP（Model Context Protocol），接入外部工具生态
2. 记忆持久化（向量检索 + 会话摘要），支持长对话
3. 流式输出与工具调用并行，降低首字延迟
4. 评测集扩展为多步任务，输出通过率趋势
5. 增加 token / 成本统计与限流保护
