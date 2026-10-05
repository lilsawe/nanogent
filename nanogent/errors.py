"""nanogent 的异常层次。

目的是让调用方可以按类型区分「配置错误」「模型调用失败」「工具执行失败」，
而不是笼统地捕获 Exception。
"""

from __future__ import annotations


class NanogentError(Exception):
    """所有 nanogent 异常的基类。"""


class ConfigError(NanogentError, ValueError):
    """配置缺失或非法，例如未找到 API Key。

    同时继承 ValueError，兼容既有调用方的 except ValueError 写法。
    """


class LLMError(NanogentError):
    """调用模型 API 失败（网络、鉴权、限流等）。"""


class ToolError(NanogentError):
    """工具执行失败。"""
