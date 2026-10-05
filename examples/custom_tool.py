#!/usr/bin/env python3
"""
示例：把 nanogent 当作库使用，并注册一个自定义工具。

运行：
    pip install -e .
    python examples/custom_tool.py            # 需要设置 API Key 才会真正调用模型
    python examples/custom_tool.py --dry-run  # 只展示工具定义，不调用模型
"""

from __future__ import annotations

import asyncio
import sys

from nanogent import Agent, LLMClient, create_default_registry, tool


@tool
def word_count(text: str) -> int:
    """统计文本的单词数。

    Args:
        text: 待统计的文本
    """
    return len(text.split())


def main() -> int:
    registry = create_default_registry()
    registry.register(word_count)          # 注册自定义工具（一行）

    if "--dry-run" in sys.argv:
        for definition in registry.list_definitions():
            fn = definition["function"]
            print(f"{fn['name']}: {fn['description']}")
            print(f"  parameters = {fn['parameters']}")
        return 0

    agent = Agent(llm=LLMClient(), tools=registry)
    agent.reset()
    answer = asyncio.run(agent.run("用 word_count 统计这句话有几个单词：nanogent is a small agent runtime"))
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
