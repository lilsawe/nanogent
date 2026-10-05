import asyncio
import math

from nanogent.tools import CalculatorTool, ToolRegistry, create_default_registry


def test_calculator_supports_math_functions() -> None:
    tool = CalculatorTool()

    result = asyncio.run(tool.execute("sqrt(144) + sin(pi / 2)"))

    assert math.isclose(float(result), 13.0)


def test_calculator_rejects_builtins() -> None:
    tool = CalculatorTool()

    result = asyncio.run(tool.execute("__import__('os').system('echo unsafe')"))

    assert result.startswith("Error evaluating expression:")


def test_registry_reports_unknown_tool() -> None:
    registry = ToolRegistry()

    result = asyncio.run(registry.execute("missing_tool", {}))

    assert "unknown tool" in result


def test_default_registry_exposes_expected_tools() -> None:
    # 显式关闭插件发现：本机若安装了第三方工具插件，结果才可复现
    registry = create_default_registry(plugins=False)

    assert set(registry.tools) == {
        "execute_python",
        "read_file",
        "write_file",
        "calculator",
        "web_search",
        "glob",
        "grep",
    }
