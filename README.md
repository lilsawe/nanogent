# nanogent

> A minimal AI Agent Runtime built from scratch in Python — agent loop, function calling, tool registry, JSONL tracing, and an evaluation harness.

[![CI](https://github.com/lilsawe/nanogent/actions/workflows/ci.yml/badge.svg)](https://github.com/lilsawe/nanogent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Agent Runtime](https://img.shields.io/badge/Agent-Runtime-6f42c1.svg)](#architecture)

作者：[李耀彬 (Li Yaobin) · @lilsawe](https://github.com/lilsawe)　|　个人主页：https://github.com/lilsawe

一个从零实现的轻量级 AI Agent Runtime，用少量 Python 代码展示 Agent 的核心工作流：LLM 推理、工具调用、工具结果回传、多轮对话、CLI 交互，以及 Agent 行为 tracing / eval。

这个项目不是对 LangChain、CrewAI 等成熟框架的封装。代码刻意保持可读，方便快速看到 Agent loop、tool registry、function calling、异步执行、JSONL trace 和 evaluation harness 的具体实现。

## 60 秒上手

§§§bash
pip install -e .                     # 或 pipx install .
export NANOGENT_API_KEY=sk-xxx       # 也支持 DEEPSEEK_API_KEY / OPENAI_API_KEY

nanogent "把 data.csv 里的空值统计出来"        # 单次执行，跑完即退
cat app.log | nanogent "这段日志的根因是什么？"  # 管道输入
nanogent --json "..."                          # 结构化输出，喂给脚本
nanogent                                        # 交互模式
§§§

不需要配置文件、不需要起服务、不需要 Docker——装完就能用。
## Highlights

- 从零实现 `perceive -> think -> act -> observe` Agent 循环
- 基于 DeepSeek 的 OpenAI-compatible function calling
- **既是一个库，也是一个 CLI**：`from nanogent import Agent` 直接嵌进你的程序（PEP 561 类型标注）
- **函数即工具**：一个 `@tool` 装饰器，Schema 由类型注解与 docstring 自动生成
- **插件式工具发现**：第三方包声明 entry point，装完即被自动发现（`nanogent --tools` 标注 `[plugin]`）
- 可扩展工具系统：抽象 `Tool` 基类、注册表、统一 JSON Schema 描述
- 内置 **7 个工具**：Python 执行、文件读写、计算器、模拟搜索、**glob（按模式找文件）**、**grep（正则搜内容）**
- 支持多轮对话上下文和 `/reset` 重置
- 使用 `asyncio` 封装 LLM 调用和工具执行
- JSONL tracing：记录 user message、LLM request/response、tool_call、tool_result、final_response
- Evaluation harness：用 JSONL 任务集评估工具调用链路，输出 JSON + Markdown 报告
- **66 个单元测试**，覆盖工具系统（含函数工具与插件发现）、计算器安全边界、Agent tool-call、tracing、eval、会话与 CLI

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
├── nanogent/
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
conda create -n nanogent python=3.11
conda activate nanogent
pip install -e ".[dev]"
```

### 2. Configure API key

```bash
cp .env.example .env
export NANOGENT_API_KEY="your-api-key"   # 也支持 DEEPSEEK_API_KEY / OPENAI_API_KEY
```

### 3. Run the agent

```bash
nanogent
```

Example prompts:

```text
计算 (15 * 23 + sqrt(144)) / 2 的结果
在当前目录创建一个 hello.py，内容是打印 "Hello Agent"
读取刚才创建的 hello.py，然后用 Python 执行它
帮我分析一下：1 米/秒 的风速下，一个半径 5 米的水平轴风力发电机理论功率是多少？用贝茨极限算。
```

## 作为库使用

nanogent **首先是一个库**，§pip install§ 之后可以直接嵌进你的程序：

§§§python
import asyncio
from nanogent import Agent, LLMClient, create_default_registry

agent = Agent(llm=LLMClient(), tools=create_default_registry())
agent.reset()
print(asyncio.run(agent.run("把 data.csv 里的空值统计出来")))
§§§

完整示例见 §examples/custom_tool.py§。

## 自定义工具：一行装饰器

不需要手写 JSON Schema——**注解决定类型，docstring 决定描述**：

§§§python
from nanogent import tool, create_default_registry

@tool
def word_count(text: str) -> int:
    """统计文本的单词数。

    Args:
        text: 待统计的文本
    """
    return len(text.split())

registry = create_default_registry()
registry.register(word_count)      # 注册即可用
§§§

自动生成的工具 Schema：

§§§json
{"type": "object",
 "properties": {"text": {"type": "string", "description": "待统计的文本"}},
 "required": ["text"]}
§§§

支持同步/异步函数、§list§/§dict§/§Optional§ 等注解，也兼容 §from __future__ import annotations§（字符串注解）。

## 插件：第三方包自动提供工具

第三方包只要声明 entry point，**安装后即被发现**：

§§§toml
[project.entry-points."nanogent.tools"]
weather = "nanogent_weather.tools:ALL_TOOLS"
§§§

仓库里的 §examples/plugin_weather/§ 就是一个可直接安装的示例插件：

§§§bash
pip install -e examples/plugin_weather
nanogent --tools              # 列表里会多出 weather  [plugin]
nanogent --no-plugins         # 需要时可关闭插件发现
§§§

加载失败的插件会被**跳过**，不会影响主流程；CI 里也真的会安装这个示例插件并断言它被发现。
## 三种用法

| 场景 | 命令 | 说明 |
|---|---|---|
| **交互** | §nanogent§ | REPL，支持 §/tools§ §/save§ §/sessions§ §/reset§ |
| **单次** | §nanogent "任务"§ | 跑完即退，适合写进脚本/CI |
| **管道** | §cat x | nanogent "分析"§ | stdin 自动作为输入内容拼进 prompt |

## 脚本化输出

§§§bash
# 只要答案（适合管道/CI，无 banner 无颜色）
nanogent -q "把 result.txt 里的数字求和"

# 结构化 JSON：answer / model / elapsed_ms / tool_calls
nanogent --json "统计 src 下的 Python 文件数"
§§§

§§§json
{
  "answer": "共 12 个 .py 文件",
  "model": "deepseek-chat",
  "elapsed_ms": 3210,
  "tool_calls": 2,
  "session": null
}
§§§

## 会话（跨次对话）

§§§bash
nanogent --session refactor "先看一下 order.py 的结构"   # 保存到命名会话
nanogent --continue "接着改：把幂等校验抽成独立函数"       # 继续最近一次会话
nanogent --list-sessions                                # 列出所有会话
§§§

会话存于 §~/.nanogent/sessions/*.json§，纯文本可读、可 diff、可手动删。
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
nanogent-eval --offline --tasks evals/tool_call_tasks.jsonl --out-dir eval_runs
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

`nanogent/agent.py` keeps the conversation state and repeatedly calls the LLM until it receives a final text response or reaches the iteration limit. When the LLM returns tool calls, the agent executes them through `ToolRegistry` and appends results back into the message history.

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

`nanogent/llm.py` wraps the OpenAI SDK and points it at DeepSeek's compatible endpoint. The wrapper normalizes model responses into a small `LLMResponse` dataclass so the rest of the project does not depend on SDK-specific response objects.

### Eval runner

`nanogent/eval.py` loads JSONL tasks, runs them through the same Agent loop,
checks expected tool usage and answer substrings, then emits a Markdown report
plus machine-readable JSON results.

## Built-in Tools

| Tool | Purpose |
| --- | --- |
| `execute_python` | Runs a self-contained Python snippet in a subprocess |
| `read_file` | Reads UTF-8 text files with output truncation |
| `write_file` | Creates or overwrites text files |
| `calculator` | Evaluates math expressions with a restricted namespace |
| `web_search` | Placeholder search tool that shows how a tool call flows through the loop |
| `glob` | Finds files by glob pattern (e.g. **/*.py), newest first, skips .git/venv/target |
| `grep` | Regex search across files, returns file:line: text; supports include filter |

## 评测（Eval）

任务集是 JSONL，每条声明期望调用的工具、参数与期望输出片段：

§§§json
{"id": "multi_step_grep_then_read",
 "prompt": "先在 evals/fixtures 里搜索 TODO，然后读取命中的文件。",
 "steps": [{"tool": "grep", "arguments": {"pattern": "TODO", "path": "evals/fixtures"}},
           {"tool": "read_file", "arguments": {"path": "evals/fixtures/sample.py"}}],
 "expected_tools": ["grep", "read_file"],
 "expected_substrings": ["TODO"]}
§§§

§§§bash
# 离线：用脚本化模型验证「Agent 循环 + 工具执行 + trace + 报告」这条链路（无需 API Key）
nanogent-eval --offline --tasks evals/tool_call_tasks.jsonl --out-dir eval_runs

# 真实模型：给出真实通过率
export NANOGENT_API_KEY=sk-xxx
nanogent-eval --tasks evals/tool_call_tasks.jsonl --out-dir eval_runs
§§§

**当前离线自检结果**：

| 指标 | 数值 |
|---|---|
| 任务数 | **19** |
| 通过率 | **19 / 19（100%）** |
| 覆盖工具 | **7 / 7**（calculator · execute_python · read_file · write_file · glob · grep · web_search） |

> ⚠️ 离线模式用**脚本化模型**，验证的是「评测链路本身是通的」，**不代表模型能力**；
> 真实通过率需要带 API Key 跑一次（上面第二条命令）。CI 只跑离线自检，保证任务集与评测代码不腐化。

支持的能力：单步 / **多步任务**、**forbidden_tools**（验证「不该调工具时不调」）、**有序子序列校验**（允许中间夹杂其它工具调用）、错误场景（文件不存在 / 除零 / 无匹配）。

## 代码质量

| 检查 | 现状 |
|---|---|
| 测试 | **66 个**，覆盖率 **82%** |
| lint（ruff） | 通过（规则与例外写在 §pyproject.toml§） |
| 类型检查（mypy） | 通过：10 个文件 0 错误，随包发布 §py.typed§ |
| 评测 | 19 条任务，CI 跑离线自检 |
| CI | Python 3.11 / 3.12 / 3.13 三版本矩阵 |

## Interview Talking Points

- Why implement the Agent loop directly instead of hiding it behind a framework
- How function calling maps to tool registration and tool result messages
- Why the project uses async interfaces even though the codebase is small
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
