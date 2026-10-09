"""AQuant 包的全局常量与配置默认值。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConstants:
    """引擎级别的固定常量（研究口径，非可调参数）。"""

    cash: float = 1e8          # 研究本金：足够大，消除「100 股整手」对小权重的扭曲
    warmup: int = 250          # 指标预热窗口（滚动前推时必须保留，否则长均线被冤枉）
    market_code: str = "sh000300"   # 沪深 300 作基准 + 统一交易日轴


CONST = EngineConstants()
