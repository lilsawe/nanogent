"""nanogent package."""

__version__ = "0.2.0"

from nanogent.agent import Agent
from nanogent.llm import LLMClient
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
from nanogent.tracing import TraceRecorder

__all__ = [
    "Agent",
    "FunctionTool",
    "LLMClient",
    "SessionStore",
    "Tool",
    "ToolRegistry",
    "TraceRecorder",
    "__version__",
    "builtin_tools",
    "create_default_registry",
    "load_plugin_tools",
    "schema_from_callable",
    "tool",
]
