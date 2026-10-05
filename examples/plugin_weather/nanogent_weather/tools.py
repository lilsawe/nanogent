"""插件提供的工具：只用 @tool 装饰器，无需手写 JSON Schema。"""

from __future__ import annotations

from nanogent import tool

_FAKE_DATA = {
    "深圳": "晴，26°C，东南风 2 级",
    "北京": "多云，18°C，北风 3 级",
    "上海": "小雨，21°C，东风 2 级",
}


@tool
def weather(city: str) -> str:
    """查询指定城市的天气（示例插件，返回内置假数据）。

    Args:
        city: 城市名，例如 深圳
    """
    return _FAKE_DATA.get(city, f"{city}：暂无数据")


#: 插件入口：entry point 指向这个对象（Tool 或 Tool 列表都可以）
ALL_TOOLS = [weather]
