#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aq_client.py —— AQuant 量化系统的「对外调用封装」
================================================
把研究脚本里散落的 helper 收拢成一个干净入口，让工作项目能直接调用，
而不必去 import 互相耦合的 fund_*.py / optimize.py。

依赖：仅 aq/ 包（纯标准库，无 requests/pandas/numpy 依赖，可直接拷进任何项目）。
首次运行会联网拉一次行情（东方财富 HTTPS），之后落本地 JSON 缓存。

两种用法：
  A. 跑回测（验证 / 研究）：
        from aq_client import run_backtest
        r = run_backtest(factor="bp", tau=3.0, floor=0.0, freq="monthly")
        print(r["ret"], r["bench"], r["excess"], r["turnover"])

  B. 直接拿「目标权重」做信号（不跑完整回测）：
        from aq_client import target_weights
        w = target_weights("2026-10-08", factor="bp", tau=3.0, floor=0.0)
        # w = {"600519": 0.012, ...}  sum(w) == 1.0
"""
import os
import sys
import math
import json
import argparse

# ---- 把本文件所在目录（即 aq/ 的父目录）加进路径，确保能 import aq ----
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from aq.data import kline, build_pool
from aq.fundamentals import Fund
from aq.strategies import fund_tilt
from aq.backtest import Backtest, Cost
from aq.factors import _zstat

CASH = 1e8              # 研究本金：足够大，消除「100股整手」对小权重的扭曲
WARMUP = 250            # 指标预热窗口
MKT_CODE = "sh000300"   # 沪深300 作基准 + 统一交易日轴


# ---------------------------------------------------------------------------
# 内部：把个股 K线对齐到统一日期轴（复刻 optimize.fill_to_axis，避免依赖顶层脚本）
# ---------------------------------------------------------------------------
def _fill_to_axis(d, axis):
    out = {}
    for sym, bars in d.items():
        if not bars or bars[0][0] > axis[0]:   # 轴起点之后才上市，剔除
            continue
        p = {b[0]: b for b in bars}
        filled, last, ok = [], None, True
        for dt in axis:
            b = p.get(dt)
            if b is None:
                if last is None:
                    ok = False
                    break
                filled.append((dt, last[2], last[2], last[2], last[2], 0.0))
            else:
                last = b
                filled.append(b)
        if ok:
            out[sym] = filled
    return out


def _load_universe(pool_top=300, mkt_code=MKT_CODE):
    """构建无偏宽基池 + 行情 + 指数轴，返回 (A, dates, mkt)。"""
    pool = build_pool(top=pool_top, mode="random", min_bars=2000)
    d = {s: kline(s, bars=0) for s in pool}
    mkt_k = kline(mkt_code, bars=0)
    axis = [x[0] for x in mkt_k]
    mkt = [x[2] for x in mkt_k]
    A = _fill_to_axis(d, axis)
    return A, axis, mkt


# ---------------------------------------------------------------------------
# A. 跑回测
# ---------------------------------------------------------------------------
def run_backtest(factor="bp", tau=3.0, floor=0.0, freq="monthly",
                 pool_top=300, cash=CASH, cost_slip=0.001,
                 mkt_code=MKT_CODE, warmup=WARMUP):
    """
    对一个「基本面因子 + 用法旋钮」组合跑完整回测。

    参数
    ----
    factor   : 因子名，须在 aq.fundamentals.FACTORS 中（P2 验证过用 'bp'）
    tau      : 倾斜锐度。tau→∞ 退化为等权宽基；越小越向高分票集中
    floor    : 与等权混合比例（0=纯倾斜，0.5=半等权半倾斜）
    freq     : 调仓频率 daily|weekly|monthly|quarterly|semiannual（经引擎 rebalance 接入）
    pool_top : 股票池规模（随机抽样，避免前视偏差）
    cost_slip: 双边滑点

    返回
    ----
    dict: ret(总收益) / bench(基准收益) / excess(超额) / turnover(年换手)
          / equity(list) / benchmark_curve(list) / trades / n_trades
    """
    A, dates, mkt = _load_universe(pool_top, mkt_code)
    strat = fund_tilt(factor, tau=tau, floor=floor)
    fund = Fund()
    bt = Backtest(
        A, strat, cash=cash, cost=Cost(slippage=cost_slip),
        rebalance=freq, max_weight=0.0, band=0.0, min_hold=0,
        market={"date": dates, "close": mkt}, fund=fund,
    )
    r = bt.run()
    off = warmup
    eq, bc = r["equity"], r["benchmark"]["curve"]
    ret = eq[-1] / eq[off] - 1
    bret = bc[-1] / bc[off] - 1
    amt = sum(t["amount"] for t in r["trades"])
    yrs = max((len(dates) - 1 - warmup) / 252.0, 1e-6)
    return {
        "ret": ret, "bench": bret, "excess": ret - bret,
        "turnover": amt / cash / yrs, "equity": eq,
        "benchmark_curve": bc, "trades": r["trades"], "n_trades": r["n_trades"],
    }


# ---------------------------------------------------------------------------
# B. 直接生成「目标权重」信号（不跑回测，只算某一日的持仓占比）
# ---------------------------------------------------------------------------
def target_weights(as_of, factor="bp", tau=3.0, floor=0.0, pool_top=300,
                   mkt_code=MKT_CODE, min_frac=0.4):
    """
    计算 as_of 这一天的目标权重 {symbol: weight}，sum == 1.0。
    复刻 fund_tilt 的权重数学：point-in-time 基本面 → 横截面 z-score → softmax 倾斜。

    用法场景：每天收盘后调用一次，拿返回的权重去调仓。
    """
    A, dates, mkt = _load_universe(pool_top, mkt_code)
    if as_of not in dates:
        # 取不晚于 as_of 的最近交易日
        as_of = max(d for d in dates if d <= as_of)
    fund = Fund()
    fn = Fund_fact(factor)        # 见下方辅助
    is_val = factor in ("ep", "bp")
    # 取 as_of 当天（或之前最近）的收盘价
    price_on = {}
    for s, bars in A.items():
        # bars 已对齐到 axis，二分找 as_of 索引
        i = _index_of_date(dates, as_of)
        price_on[s] = bars[i][2] if i is not None else None

    raw = {}
    for s in A:
        price = price_on.get(s)
        if price is None or price <= 0:
            continue
        st = fund.pit(s, as_of)
        if st is None:
            continue
        v = fn(st, price) if is_val else fn(st)
        if v is None:
            continue
        raw[s] = v
    if len(raw) < max(20, int(min_frac * len(A))):
        return {}   # 覆盖率不足，返回空（沿用上次权重）

    vals = list(raw.values())
    m, sd = _zstat(vals)
    sd = sd if sd > 1e-9 else 1.0
    w = {s: math.exp((v - m) / sd / tau) for s, v in raw.items()}
    tot = sum(w.values()) or 1.0
    w = {s: v / tot for s, v in w.items()}
    if floor > 0:
        eq = 1.0 / len(w)
        w = {s: floor * eq + (1.0 - floor) * v for s, v in w.items()}
    return w


def Fund_fact(factor):
    """从 FACTORS 取因子函数（与 strategies.fund_tilt 内部一致）。"""
    from aq import fundamentals as fnd
    return fnd.FACTORS[factor][0]


def _index_of_date(dates, target):
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


# ---------------------------------------------------------------------------
# CLI：python aq_client.py            -> 跑一次默认回测
#       python aq_client.py --signal 2026-10-08 -> 打印目标权重
# ---------------------------------------------------------------------------
def _main():
    ap = argparse.ArgumentParser(description="AQuant 调用封装")
    ap.add_argument("--factor", default="bp")
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--floor", type=float, default=0.0)
    ap.add_argument("--freq", default="monthly")
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--signal", metavar="DATE", default=None,
                    help="给一个日期则只输出该日目标权重")
    a = ap.parse_args()

    if a.signal:
        w = target_weights(a.signal, factor=a.factor, tau=a.tau,
                           floor=a.floor, pool_top=a.pool)
        tot = sum(w.values())
        print(f"# 目标权重 {a.signal} | factor={a.factor} tau={a.tau} "
              f"floor={a.floor} | 共 {len(w)} 只 sum={tot:.4f}")
        for s, x in sorted(w.items(), key=lambda kv: -kv[1])[:30]:
            print(f"  {s}  {x*100:6.3f}%")
    else:
        r = run_backtest(factor=a.factor, tau=a.tau, floor=a.floor,
                         freq=a.freq, pool_top=a.pool)
        print(f"# 回测 factor={a.factor} tau={a.tau} floor={a.floor} freq={a.freq}")
        print(f"  总收益  {r['ret']*100:+.2f}%")
        print(f"  基准    {r['bench']*100:+.2f}%")
        print(f"  超额    {r['excess']*100:+.2f}pp")
        print(f"  年换手  {r['turnover']*100:.0f}%")
        print(f"  成交笔数 {r['n_trades']}")


if __name__ == "__main__":
    _main()
