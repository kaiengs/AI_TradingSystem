"""AQuant —— A 股量化回测与策略研究框架（可 pip 安装的封装层）。

把自包含的研究引擎（aquant._engine.aq）包成带类型注解、可返回 pandas
DataFrame 的干净接口。最常见的两种用法：

    from aquant import run_backtest, target_weights

    # 1) 跑一次回测
    r = run_backtest(factor="bp", tau=3.0, freq="monthly")
    print(r.total_return, r.benchmark_return, r.excess_pp, r.annual_turnover)
    df = r.to_frame()                 # 净值/基准曲线，索引为交易日

    # 2) 只拿某天的目标权重当信号
    w = target_weights("2026-10-08", factor="bp", tau=3.0)
    #    w 是 DataFrame: columns=['symbol', 'weight'], sum(weight)==1.0
"""
from __future__ import annotations

from aquant.backtest import BacktestConfig, BacktestResult, run_backtest
from aquant.config import CONST, EngineConstants
from aquant.data import load_universe
from aquant.signal import SignalConfig, target_weights

# 直接把底层引擎也暴露出来，方便高级用法（自定义策略 / 直接调 Backtest）
from aquant._engine.aq.backtest import Backtest, Cost
from aquant._engine.aq.fundamentals import FACTORS, Fund
from aquant._engine.aq.strategies import REGISTRY, fund_tilt, multi_tilt

__version__ = "0.1.0"

__all__ = [
    "run_backtest",
    "BacktestConfig",
    "BacktestResult",
    "target_weights",
    "SignalConfig",
    "load_universe",
    "Backtest",
    "Cost",
    "Fund",
    "FACTORS",
    "REGISTRY",
    "fund_tilt",
    "multi_tilt",
    "CONST",
    "EngineConstants",
    "__version__",
]
