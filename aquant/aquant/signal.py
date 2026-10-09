"""信号生成：不跑完整回测，只算某一交易日的「目标权重」。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from aquant._engine.aq.factors import _zstat
from aquant._engine.aq.fundamentals import FACTORS, Fund
from aquant.config import CONST
from aquant.data import load_universe


@dataclass
class SignalConfig:
    """信号参数。"""

    factor: str = "bp"
    tau: float = 3.0
    floor: float = 0.0
    pool_top: int = 300
    market_code: str = CONST.market_code
    min_frac: float = 0.4           # 至少有 min_frac 比例的票有因子值才出信号


def target_weights(
    as_of: str,
    config: Optional[SignalConfig] = None,
    **overrides: Any,
) -> pd.DataFrame:
    """计算 as_of 这一天的目标权重。

    复刻 fund_tilt 的权重数学：point-in-time 基本面 → 横截面 z-score → softmax 倾斜。
    返回的 DataFrame 含 ['symbol', 'weight'] 两列，按权重降序，sum(weight)==1.0；
    as_of 实际生效日期记录在 df.attrs['as_of']。

    用法场景：每天收盘后调用一次，拿返回的权重去调仓。
    """
    cfg = config or SignalConfig()
    for key, val in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"未知的信号参数: {key}")
        setattr(cfg, key, val)

    A, dates, _ = load_universe(cfg.pool_top, cfg.market_code)
    if as_of not in dates:
        as_of = max(d for d in dates if d <= as_of)   # 取不晚于 as_of 的最近交易日

    fund = Fund()
    fn = FACTORS[cfg.factor][0]
    is_val = cfg.factor in ("ep", "bp")               # 估值类因子需要价格
    i = _index_of_date(dates, as_of)
    price_on = {s: (bars[i][2] if i is not None else None) for s, bars in A.items()}

    raw: Dict[str, float] = {}
    for s in A:
        price = price_on.get(s)
        if not price or price <= 0:
            continue
        st = fund.pit(s, as_of)
        if st is None:
            continue
        v = fn(st, price) if is_val else fn(st)
        if v is None:
            continue
        raw[s] = v

    if len(raw) < max(20, int(cfg.min_frac * len(A))):
        # 覆盖率不足：返回空信号（调用方应沿用上次权重）
        df = pd.DataFrame(columns=["symbol", "weight"])
        df.attrs["as_of"] = as_of
        df.attrs["coverage"] = len(raw)
        return df

    vals = list(raw.values())
    m, sd = _zstat(vals)
    sd = sd if sd > 1e-9 else 1.0
    w = {s: math.exp((v - m) / sd / cfg.tau) for s, v in raw.items()}
    tot = sum(w.values()) or 1.0
    w = {s: x / tot for s, x in w.items()}
    if cfg.floor > 0:
        eq = 1.0 / len(w)
        w = {s: cfg.floor * eq + (1.0 - cfg.floor) * x for s, x in w.items()}

    df = pd.DataFrame({"symbol": list(w), "weight": [w[s] for s in w]})
    df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    df.attrs["as_of"] = as_of
    df.attrs["coverage"] = len(raw)
    return df


def _index_of_date(dates: list, target: str) -> Optional[int]:
    """在升序 dates 中二分查找 target 的索引；找不到返回其前驱索引。"""
    lo, hi = 0, len(dates) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if dates[mid] == target:
            return mid
        if dates[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi if hi >= 0 else None
