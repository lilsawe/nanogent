"""
Tool system for the nanogent.

Defines a Tool base class, a ToolRegistry for managing tools, and five
built-in tools: execute_python, read_file, write_file, calculator, web_search.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import subprocess
import sys
import types
import typing
from importlib.metadata import entry_points
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

class Tool(ABC):
    """
    Abstract base class for all tools.

    Each tool provides:
    - name: unique identifier (used in tool-call matching)
    - description: natural-language description (shown to the LLM)
    - parameters: JSON Schema describing the expected arguments
    - execute(): the actual implementation
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]: ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str: ...

    def to_openai_format(self) -> dict[str, Any]:
        """Convert tool definition to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class ExecutePythonTool(Tool):
    """Execute a snippet of Python code and return stdout/stderr."""

    @property
    def name(self) -> str:
        return "execute_python"

    @property
    def description(self) -> str:
        return (
            "Execute a Python code snippet in an isolated subprocess. "
            "Use this for calculations, data processing, or any logic that "
            "requires running code. The code must be a complete, self-contained "
            "script. Stdout and stderr are both captured and returned."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Must be self-contained.",
                },
            },
            "required": ["code"],
        }

    async def execute(self, code: str, **kwargs: Any) -> str:
        try:
            proc = await _run_subprocess(
                [sys.executable, "-c", code],
                timeout=10,
            )
            output = proc["stdout"]
            if proc["stderr"]:
                output += "\n[stderr]\n" + proc["stderr"]
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: code execution timed out (10 seconds)."
        except Exception as e:
            return f"Error executing code: {e}"


class ReadFileTool(Tool):
    """Read the contents of a file at the given path."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file at the specified path. "
            "Returns the file content as text. Use this when you need to "
            "inspect a file's contents."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file.",
                },
            },
            "required": ["path"],
        }

    async def execute(self, path: str, **kwargs: Any) -> str:
        try:
            with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
                content = f.read()
            if not content:
                return "(file is empty)"
            # Truncate if too long to avoid blowing up context
            if len(content) > 8000:
                content = content[:8000] + "\n... (truncated)"
            return content
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except PermissionError:
            return f"Error: permission denied: {path}"
        except Exception as e:
            return f"Error reading file: {e}"


class WriteFileTool(Tool):
    """Write content to a file, creating it if it doesn't exist."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write content to a file at the specified path. "
            "Creates the file if it does not exist, overwrites if it does. "
            "Use this to save output, create code files, or persist results."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path where the file will be written.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        try:
            expanded = os.path.expanduser(path)
            os.makedirs(os.path.dirname(expanded) or ".", exist_ok=True)
            with open(expanded, "w", encoding="utf-8") as f:
                f.write(content)
            return f"File written successfully: {path} ({len(content)} characters)"
        except Exception as e:
            return f"Error writing file: {e}"


class CalculatorTool(Tool):
    """Evaluate a mathematical expression safely."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate a mathematical expression and return the result. "
            "Supports basic arithmetic (+, -, *, /, **), math functions "
            "(sqrt, sin, cos, log, etc.), and constants (pi, e). "
            "Use this for precise numerical calculations."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g. 'sqrt(144) + 3 * 7'.",
                },
            },
            "required": ["expression"],
        }

    async def execute(self, expression: str, **kwargs: Any) -> str:
        # Build a safe namespace with math functions
        safe_namespace: dict[str, Any] = {
            name: getattr(math, name)
            for name in dir(math)
            if not name.startswith("_")
        }
        safe_namespace["__builtins__"] = {}

        try:
            result = eval(expression, {"__builtins__": {}}, safe_namespace)
            return str(result)
        except SyntaxError as e:
            return f"Syntax error in expression: {e}"
        except Exception as e:
            return f"Error evaluating expression: {e}"


class WebSearchTool(Tool):
    """
    Simulated web search tool.

    In a production agent this would call a real search API (Brave, SerpAPI,
    etc.). Here it returns a placeholder to show the tool-calling flow
    without requiring an additional API key.
    """

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for information on a given query. "
            "Returns a summary of search results. Use this when you need "
            "up-to-date information or facts you are uncertain about."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, **kwargs: Any) -> str:
        # In production, integrate a real search API here.
        # We return a clear placeholder so the agent knows the
        # tool was called but real search is not wired up.
        return (
            f"[Simulated search] Query: '{query}'\n"
            "To enable real search, integrate a search API (e.g. Brave Search, "
            "SerpAPI) in the WebSearchTool.execute() method."
        )


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

@dataclass
class ToolRegistry:
    """Stores available tools and handles lookup + execution."""

    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        return self.tools.get(name)

    def list_definitions(self) -> list[dict[str, Any]]:
        """Return all tools in OpenAI function-calling format."""
        return [t.to_openai_format() for t in self.tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool by name with the given arguments.

        Returns the tool's output as a string. Errors are caught and returned
        as error strings so the agent loop never crashes on a tool failure.
        """
        tool = self.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'. Available tools: {list(self.tools.keys())}"
        try:
            return await tool.execute(**arguments)
        except TypeError as e:
            return f"Error: invalid arguments for tool '{name}': {e}"
        except Exception as e:
            return f"Error: tool '{name}' failed: {e}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def builtin_tools() -> list[Tool]:
    """全部内置工具。"""
    return [
        ExecutePythonTool(),
        ReadFileTool(),
        WriteFileTool(),
        CalculatorTool(),
        WebSearchTool(),
        GlobFilesTool(),
        GrepFilesTool(),
    ]


def create_default_registry(
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    plugins: bool = True,
) -> ToolRegistry:
    """
    创建工具注册表。

    Args:
        include: 只加载这些工具名（None 表示全部）
        exclude: 排除这些工具名
        plugins: 是否发现第三方插件（entry_points 组 nanogent.tools）
    """
    registry = ToolRegistry()
    candidates = builtin_tools()
    if plugins:
        candidates = candidates + load_plugin_tools()

    for candidate in candidates:
        if include is not None and candidate.name not in include:
            continue
        if exclude is not None and candidate.name in exclude:
            continue
        registry.register(candidate)
    return registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _run_subprocess(
    cmd: list[str],
    timeout: int = 10,
) -> dict[str, str]:
    """Run a subprocess asynchronously and return stdout + stderr."""
    proc = await _async_subprocess_run(cmd, timeout)
    return {"stdout": proc["stdout"], "stderr": proc["stderr"]}


async def _async_subprocess_run(
    cmd: list[str], timeout: int
) -> dict[str, str]:
    """Thin async wrapper around subprocess.run."""
    import asyncio

    loop = asyncio.get_running_loop()

    def _run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    result = await loop.run_in_executor(None, _run)
    return {"stdout": result.stdout, "stderr": result.stderr}


class GlobFilesTool(Tool):
    """Find files by glob pattern (e.g. **/*.py)."""

    MAX_RESULTS = 200
    SKIP_DIRS = {
        ".git", ".venv", "venv", "__pycache__", "node_modules",
        ".idea", ".vscode", "target", "dist", "build", ".mypy_cache", ".pytest_cache",
    }

    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return (
            "Find files matching a glob pattern, e.g. '**/*.py' or 'src/**/*.java'. "
            "Returns matching paths (newest first). Use this to locate files before reading them."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, e.g. **/*.py",
                },
                "path": {
                    "type": "string",
                    "description": "Root directory to search in. Defaults to the current directory.",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, pattern: str, path: str = ".", **kwargs: Any) -> str:
        root = Path(os.path.expanduser(path or "."))
        if not root.exists():
            return f"Error: directory not found: {path}"
        if not root.is_dir():
            return f"Error: not a directory: {path}"

        matches: list[Path] = []
        try:
            for candidate in root.glob(pattern):
                if not candidate.is_file():
                    continue
                if any(part in self.SKIP_DIRS for part in candidate.parts):
                    continue
                matches.append(candidate)
        except Exception as exc:
            return f"Error: invalid glob pattern '{pattern}': {exc}"

        if not matches:
            return f"No files matched '{pattern}' under {root}"

        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        shown = matches[: self.MAX_RESULTS]
        lines = [str(p) for p in shown]
        if len(matches) > len(shown):
            lines.append(f"... ({len(matches) - len(shown)} more files)")
        return "\n".join(lines)


class GrepFilesTool(Tool):
    """Search file contents with a regular expression."""

    MAX_RESULTS = 100
    MAX_FILE_BYTES = 1_000_000
    SKIP_DIRS = GlobFilesTool.SKIP_DIRS
    SKIP_SUFFIXES = {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz",
        ".jar", ".class", ".so", ".dylib", ".pyc", ".woff", ".woff2", ".mp4", ".mov",
    }

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search file contents with a regular expression (Python re syntax). "
            "Returns matches as 'file:line: text'. Optional 'include' filters file names, e.g. '*.py'."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression to search for."},
                "path": {"type": "string", "description": "Root directory. Defaults to the current directory."},
                "include": {"type": "string", "description": "Only search files matching this glob, e.g. *.py"},
                "max_results": {"type": "integer", "description": "Maximum matches to return (default 100)."},
            },
            "required": ["pattern"],
        }

    def _iter_files(self, root: Path, include: str | None):
        candidates = root.rglob(include) if include else root.rglob("*")
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if any(part in self.SKIP_DIRS for part in candidate.parts):
                continue
            if candidate.suffix.lower() in self.SKIP_SUFFIXES:
                continue
            try:
                if candidate.stat().st_size > self.MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield candidate

    async def execute(self, pattern: str, path: str = ".", include: str | None = None,
                      max_results: int | None = None, **kwargs: Any) -> str:
        root = Path(os.path.expanduser(path or "."))
        if not root.exists() or not root.is_dir():
            return f"Error: not a directory: {path}"

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regular expression: {exc}"

        limit = int(max_results) if max_results else self.MAX_RESULTS
        results: list[str] = []
        for candidate in self._iter_files(root, include):
            try:
                text = candidate.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    results.append(f"{candidate}:{lineno}: {line.strip()[:200]}")
                    if len(results) >= limit:
                        results.append(f"... (stopped at {limit} matches)")
                        return "\n".join(results)
        return "\n".join(results) if results else f"No matches for '{pattern}' under {root}"


# ---------------------------------------------------------------------------
# Function tools（把普通函数变成工具）
# ---------------------------------------------------------------------------

PLUGIN_ENTRY_POINT_GROUP = "nanogent.tools"

_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

#: 注解被字符串化时的兜底（模块里写了 from __future__ import annotations 就会这样）
_JSON_TYPES_BY_NAME: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "Any": "string",
}


def _resolved_hints(fn: Any) -> dict[str, Any]:
    """解析函数注解；对字符串注解会尝试求值，失败则返回空表。"""
    try:
        return typing.get_type_hints(fn)
    except Exception:
        return {}


def _json_type(annotation: Any) -> str:
    """把 Python 类型注解映射成 JSON Schema 的 type。"""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return "string"
    if isinstance(annotation, str):  # 字符串注解（PEP 563 延迟求值）
        return _JSON_TYPES_BY_NAME.get(annotation, "string")

    origin = typing.get_origin(annotation)
    if origin in (list, set, tuple, frozenset):
        return "array"
    if origin is dict:
        return "object"
    if origin in (typing.Union, types.UnionType):  # Optional[X] / X | None
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _json_type(args[0])
        return "string"
    return _JSON_TYPES.get(annotation, "string")


def _docstring_parts(fn: Any) -> tuple[str, dict[str, str]]:
    """取函数 docstring 的摘要行与 Args 段的参数说明。"""
    doc = inspect.getdoc(fn) or ""
    lines = doc.splitlines()
    summary = lines[0].strip() if lines else ""
    params: dict[str, str] = {}
    in_args = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("args:", "arguments:", "parameters:", "参数:"):
            in_args = True
            continue
        if in_args:
            if not stripped:
                continue
            match = re.match(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$", stripped)
            if match:
                params[match.group(1)] = match.group(2).strip()
    return summary, params


def schema_from_callable(fn: Any) -> dict[str, Any]:
    """根据函数签名与类型注解，自动生成工具参数的 JSON Schema。"""
    signature = inspect.signature(fn)
    _, param_docs = _docstring_parts(fn)
    hints = _resolved_hints(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(name, param.annotation)
        prop: dict[str, Any] = {"type": _json_type(annotation)}
        if name in param_docs:
            prop["description"] = param_docs[name]
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {"type": "object", "properties": properties, "required": required}


class FunctionTool(Tool):
    """
    把普通函数（同步或异步）包装成 Tool。

    参数 Schema 由类型注解与 docstring 自动生成：注解决定 JSON 类型，docstring 第一行
    作为工具描述，Args 段落里「参数名: 说明」作为参数描述。因此写一个工具只需要写业务
    逻辑本身，不需要手写 JSON Schema。
    """

    def __init__(
        self,
        fn: Any,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        if not callable(fn):
            raise TypeError("FunctionTool 需要一个可调用对象")
        summary, _ = _docstring_parts(fn)
        self._fn = fn
        self._name = name or getattr(fn, "__name__", "unnamed_tool")
        self._description = description or summary or f"调用 {self._name}"
        self._parameters = parameters or schema_from_callable(fn)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        try:
            result = self._fn(**kwargs)
        except TypeError as exc:
            return f"Error: 工具参数不匹配（{exc}）"
        except Exception as exc:  # 工具异常不应让 Agent 崩掉
            return f"Error: {type(exc).__name__}: {exc}"
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)


def tool(fn: Any = None, *, name: str | None = None, description: str | None = None):
    """
    装饰器：把函数注册为工具。

        @tool                                  # 直接用函数名与 docstring
        @tool(name="weather")                  # 自定义名字
    """

    def wrap(func: Any) -> FunctionTool:
        return FunctionTool(func, name=name, description=description)

    return wrap if fn is None else wrap(fn)


# ---------------------------------------------------------------------------
# 插件发现（第三方包通过 entry_points 提供工具）
# ---------------------------------------------------------------------------

def _as_tools(obj: Any) -> list[Tool]:
    """把 entry point 加载出的对象统一成 Tool 列表。"""
    if isinstance(obj, Tool):
        return [obj]
    if isinstance(obj, (list, tuple, set)):
        collected: list[Tool] = []
        for item in obj:
            collected.extend(_as_tools(item))
        return collected
    if callable(obj):
        return [FunctionTool(obj)]
    return []


def load_plugin_tools(group: str = PLUGIN_ENTRY_POINT_GROUP) -> list[Tool]:
    """
    发现已安装的第三方工具插件。

    第三方包只要在 pyproject.toml 里声明：

        [project.entry-points."nanogent.tools"]
        my_tools = "my_package.tools:ALL_TOOLS"

    安装后即被自动发现（加载失败的插件会被跳过，不影响主流程）。
    """
    tools: list[Tool] = []
    try:
        discovered = entry_points(group=group)
    except TypeError:  # pragma: no cover - 旧版本 API 兼容
        discovered = entry_points().get(group, [])  # type: ignore[attr-defined]

    for entry_point in discovered:
        try:
            loaded = entry_point.load()
        except Exception:
            continue
        tools.extend(_as_tools(loaded))
    return tools
