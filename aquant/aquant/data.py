"""行情与股票池的加载（封装 aq 数据层，对外暴露干净的接口）。"""
from __future__ import annotations

from typing import Dict, List, Tuple

from aquant._engine.aq.data import kline, build_pool


def load_universe(
    pool_top: int = 300,
    market_code: str = "sh000300",
) -> Tuple[Dict[str, list], List[str], List[float]]:
    """构建无偏宽基池 + 行情 + 指数轴。

    返回
    ----
    (A, dates, mkt)
        A     : {symbol: [(date,o,c,h,l,v), ...]} —— 已对齐到统一日期轴
        dates : 统一交易日轴（字符串，升序）
        mkt   : 指数收盘价序列（与 dates 对齐）
    """
    pool = build_pool(top=pool_top, mode="random", min_bars=2000)
    d = {s: kline(s, bars=0) for s in pool}
    mkt_k = kline(market_code, bars=0)
    axis = [x[0] for x in mkt_k]
    mkt = [x[2] for x in mkt_k]
    A = _fill_to_axis(d, axis)
    return A, axis, mkt


def _fill_to_axis(
    d: Dict[str, list], axis: List[str]
) -> Dict[str, list]:
    """把个股 K 线对齐到统一日期轴（复刻 optimize.fill_to_axis）。

    缺失日期（长期停牌）用前收盘价填充、成交量置 0 —— 价格不变则指标不变，
    不会凭空产生信号，同时避免「一只停牌股把整个日期交集拖短」。
    """
    out: Dict[str, list] = {}
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
