"""回测入口：把一个「因子 + 用法旋钮」组合跑成完整回测，返回结构化结果。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from aquant._engine.aq.backtest import Backtest, Cost
from aquant._engine.aq.fundamentals import Fund
from aquant._engine.aq.strategies import fund_tilt
from aquant.config import CONST
from aquant.data import load_universe


@dataclass
class BacktestConfig:
    """回测参数。所有字段均可被 run_backtest(**overrides) 覆盖。"""

    factor: str = "bp"               # 因子名，须在 aquant._engine.aq.fundamentals.FACTORS 中
    tau: float = 3.0                 # 倾斜锐度：tau→∞ 退化为等权宽基；越小越向高分票集中
    floor: float = 0.0               # 与等权混合比例（0=纯倾斜，0.5=半等权半倾斜）
    freq: str = "monthly"            # 调仓频率：daily|weekly|monthly|quarterly|semiannual
    pool_top: int = 300              # 股票池规模（随机抽样，避免前视偏差）
    cash: float = CONST.cash         # 初始本金
    cost_slippage: float = 0.001     # 双边滑点
    market_code: str = CONST.market_code
    warmup: int = CONST.warmup       # 预热窗口（不计入收益）
    max_weight: float = 0.0          # 单票权重上限（0=不限）
    band: float = 0.0                # 再平衡带宽（相对比例）：小偏离就不动
    min_hold: int = 0                # 建仓后 min_hold 天内不允许减仓


@dataclass
class BacktestResult:
    """回测结果（结构化）。用 ratio 表示收益率，.to_frame() 转成 DataFrame。"""

    dates: List[str] = field(repr=False)
    equity: List[float] = field(repr=False)
    benchmark_curve: List[float] = field(repr=False)
    trades: List[Dict[str, Any]] = field(repr=False)
    n_trades: int
    total_return: float
    benchmark_return: float
    excess_return: float             # = total_return - benchmark_return（ratio 口径）
    annual_turnover: float
    config: BacktestConfig = field(repr=False)

    @property
    def excess_pp(self) -> float:
        """超额收益，以百分点（pp）表示。"""
        return self.excess_return * 100.0

    def to_frame(self) -> pd.DataFrame:
        """净值与基准曲线，索引为交易日。"""
        return pd.DataFrame(
            {"equity": self.equity, "benchmark": self.benchmark_curve},
            index=self.dates,
        )


def run_backtest(
    config: Optional[BacktestConfig] = None,
    **overrides: Any,
) -> BacktestResult:
    """运行一次回测。

    参数
    ----
    config   : 可选的 BacktestConfig；为 None 时用默认值。
    **overrides : 直接覆盖 config 的任意字段，例如 factor="ep", tau=6.0。

    返回
    ----
    BacktestResult
    """
    cfg = config or BacktestConfig()
    for key, val in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"未知的回测参数: {key}")
        setattr(cfg, key, val)

    A, dates, mkt = load_universe(cfg.pool_top, cfg.market_code)
    strat = fund_tilt(cfg.factor, tau=cfg.tau, floor=cfg.floor)
    fund = Fund()

    bt = Backtest(
        A,
        strat,
        cash=cfg.cash,
        cost=Cost(slippage=cfg.cost_slippage),
        rebalance=cfg.freq,
        max_weight=cfg.max_weight,
        band=cfg.band,
        min_hold=cfg.min_hold,
        market={"date": dates, "close": mkt},
        fund=fund,
    )
    r = bt.run()

    off = cfg.warmup
    eq, bc = r["equity"], r["benchmark"]["curve"]
    ret = eq[-1] / eq[off] - 1
    bret = bc[-1] / bc[off] - 1
    amt = sum(t["amount"] for t in r["trades"])
    yrs = max((len(dates) - 1 - cfg.warmup) / 252.0, 1e-6)

    return BacktestResult(
        dates=dates,
        equity=eq,
        benchmark_curve=bc,
        trades=r["trades"],
        n_trades=r["n_trades"],
        total_return=ret,
        benchmark_return=bret,
        excess_return=ret - bret,
        annual_turnover=amt / cfg.cash / yrs,
        config=cfg,
    )
