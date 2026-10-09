#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指标层：纯 Python，输入 list 输出 list（等长，前段为 None）"""


def sma(x, n):
    out, s = [], 0.0
    for i, v in enumerate(x):
        s += v
        if i >= n:
            s -= x[i - n]
        out.append(s / n if i >= n - 1 else None)
    return out


def ema(x, n):
    out, k, prev = [], 2.0 / (n + 1), None
    for v in x:
        prev = v if prev is None else v * k + prev * (1 - k)
        out.append(prev)
    return out


def rsi(close, n=14):
    """Wilder RSI"""
    out = [None] * len(close)
    if len(close) < n + 1:
        return out
    gains = losses = 0.0
    for i in range(1, n + 1):
        d = close[i] - close[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    ag, al = gains / n, losses / n
    out[n] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n + 1, len(close)):
        d = close[i] - close[i - 1]
        ag = (ag * (n - 1) + max(d, 0.0)) / n
        al = (al * (n - 1) + max(-d, 0.0)) / n
        out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def atr(high, low, close, n=14):
    tr = []
    for i in range(len(close)):
        if i == 0:
            tr.append(high[0] - low[0])
        else:
            tr.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
    out = [None] * len(close)
    if len(tr) < n:
        return out
    prev = sum(tr[:n]) / n
    out[n - 1] = prev
    for i in range(n, len(tr)):
        prev = (prev * (n - 1) + tr[i]) / n
        out[i] = prev
    return out


def stddev(x, n):
    out = [None] * len(x)
    for i in range(n - 1, len(x)):
        seg = x[i - n + 1:i + 1]
        m = sum(seg) / n
        out[i] = (sum((v - m) ** 2 for v in seg) / n) ** 0.5
    return out


def momentum(close, n):
    out = [None] * len(close)
    for i in range(n, len(close)):
        out[i] = close[i] / close[i - n] - 1.0
    return out


def position_pct(close, n=60):
    """当前价在近 n 日高低区间的百分位"""
    out = [None] * len(close)
    for i in range(n - 1, len(close)):
        hi = max(close[i - n + 1:i + 1]); lo = min(close[i - n + 1:i + 1])
        out[i] = (close[i] - lo) / (hi - lo) * 100 if hi > lo else 50.0
    return out


def max_drawdown(equity):
    """输入净值序列，返回 (最大回撤比例, 回撤序列)"""
    peak, mdd, dd = equity[0], 0.0, []
    for v in equity:
        peak = max(peak, v)
        d = v / peak - 1.0
        dd.append(d)
        mdd = min(mdd, d)
    return mdd, dd


# ============================================================
# 预计算层：一次性算好全序列，回测循环中 O(1) 查表
# 这是支撑 500 只 × 2000 根大样本回测的关键（否则纯 Python 逐日
# 计算是 O(标的数 × 天数 × 窗口)，大样本下直接跑到小时级）
# ============================================================

_BUILDERS = {
    "ma": lambda c, n: sma(c["close"], n),
    "ema": lambda c, n: ema(c["close"], n),
    "rsi": lambda c, n: rsi(c["close"], n),
    "atr": lambda c, n: atr(c["high"], c["low"], c["close"], n),
    "mom": lambda c, n: momentum(c["close"], n),
    "std": lambda c, n: stddev(c["close"], n),
    "pos": lambda c, n: position_pct(c["close"], n),
}


def precompute(cols, spec):
    """
    cols: {'open','close','high','low','volume'} 全序列
    spec: {'ma':[5,20], 'rsi':[14], ...}
    返回 {'ma': {5:[...], 20:[...]}, 'rsi': {14:[...]}}
    """
    out = {}
    for kind, ns in (spec or {}).items():
        b = _BUILDERS.get(kind)
        if b is None:
            continue
        d = out.setdefault(kind, {})
        for n in ns:
            if n not in d:
                d[n] = b(cols, n)
    return out


def merge_spec(*specs):
    """合并多个 spec（例如基础策略 + 市场门控各声明一部分）"""
    out = {}
    for s in specs:
        for k, ns in (s or {}).items():
            out.setdefault(k, [])
            for n in ns:
                if n not in out[k]:
                    out[k].append(n)
    return out
