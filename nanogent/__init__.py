"""nanogent package."""

__version__ = "0.2.0"

from nanogent.agent import Agent
from nanogent.llm import LLMClient
from nanogent.tracing import TraceRecorder
from nanogent.session import SessionStore
from nanogent.tools import (
    FunctionTool,
    Tool,
    ToolRegistry,
    builtin_tools,
    create_default_registry,
    load_plugin_tools,
    schema_from_callable,
    tool,
)

__all__ = [
    "__version__",
    "Agent",
    "LLMClient",
    "TraceRecorder",
    "Tool",
    "ToolRegistry",
    "FunctionTool",
    "SessionStore",
    "builtin_tools",
    "create_default_registry",
    "load_plugin_tools",
    "schema_from_callable",
    "tool",
]
