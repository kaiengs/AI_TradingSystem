#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多因子横截面模型（v3 的核心新增）
==================================================
v2 的结论：风控（门控+低频）已成立，但**选股本身没有 alpha**——
单一反转因子在随机选股对照里排第 10 百分位。

v3 的解法：不再依赖单一因子，而是建立一个
**多因子横截面打分模型**，并用 rank-IC 验证每个因子是否真有预测力，
只把通过检验的因子合成进组合。

设计要点：
  * 所有因子都能从 OHLCV 历史计算（不依赖无法回测的基本面快照），
    所以结论不掺假。
  * 因子在 prescan 层计算每个调仓日的横截面取值，做 z-score 标准化后
    按理论方向加权合成，排名靠前者入选。
  * 因子的"方向"由金融学理论固定（见 SIGNS），先用 IC 验证方向是否
    与理论一致，不一致的因子直接剔除，避免"按样本内结果倒推方向"的作弊。

因子清单（覆盖经典风格）：
  rev20    短期反转：过去 20 日跌幅大的，未来更可能反弹
  mom12_1  中期动量（跳过最近 1 月）：过去约 12 月赢家持续
  vol60    低波动异常：波动越低的股票未来风险调整收益越好
  hml250   接近一年新高（high-minus-low）：强势股延续
  tsmom200 时序动量/趋势：价格在长均线之上
"""

import math


# ------------------------------------------------------------
# 基础取值函数（全部返回 float，数据不足返回 None）
# 入参 item 即 cols[sym] = {'open','close','high','low','volume'}(对齐到交易日轴)
# j 为当前 bar 在 item 中的下标
# ------------------------------------------------------------
def _ret(closes, j, back):
    if j < back:
        return None
    a = closes[j - back]
    b = closes[j]
    if a <= 0:
        return None
    return b / a - 1.0


def f_rev20(item, j):
    """过去 20 日收益率（短期反转因子，原始值）"""
    return _ret(item["close"], j, 20)


def f_mom12_1(item, j):
    """跳过最近 1 月的 12 月动量：close[j-20] / close[j-260] - 1"""
    if j < 260:
        return None
    a = item["close"][j - 260]
    b = item["close"][j - 20]
    if a <= 0 or b <= 0:
        return None
    return b / a - 1.0


def f_vol60(item, j):
    """过去 60 日收益率标准差（已实现波动）"""
    if j < 60:
        return None
    cl = item["close"]
    rets = [cl[k] / cl[k - 1] - 1 for k in range(j - 59, j + 1) if cl[k - 1] > 0]
    if len(rets) < 40:
        return None
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / len(rets)
    return var ** 0.5


def f_hml(item, j, w=250):
    """接近过去 w 日最高价的程度：close[j] / max(high[-w:]) - 1"""
    if j < w:
        return None
    hi = max(item["high"][j - w + 1:j + 1])
    if hi <= 0:
        return None
    return item["close"][j] / hi - 1.0


def f_tsmom(item, j, w=200):
    """时序动量：close[j] / MA(w) - 1"""
    if j < w:
        return None
    cl = item["close"][j - w + 1:j + 1]
    if len(cl) < w:
        return None
    ma = sum(cl) / w
    if ma <= 0:
        return None
    return item["close"][j] / ma - 1.0


# 原始因子函数（不带方向）
RAW = {
    "rev20": f_rev20,
    "mom12_1": f_mom12_1,
    "vol60": f_vol60,
    "hml250": f_hml,
    "tsmom200": f_tsmom,
}

# 理论方向：+1 表示"因子越大越该持有"，-1 表示"因子越小越该持有"
SIGNS = {
    "rev20": -1.0,    # 反转：持有过去跌得多的
    "mom12_1": +1.0,  # 动量：持有过去涨得多的
    "vol60": -1.0,    # 低波动：持有波动小的
    "hml250": +1.0,   # 接近新高：持有强势的
    "tsmom200": +1.0, # 趋势：持有在长均线上方的
}


# ------------------------------------------------------------
# 横截面合成 prescan
# ------------------------------------------------------------
def _zstat(vals):
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    sd = math.sqrt(var)
    return m, (sd if sd > 1e-12 else 1.0)


def make_composite(fkeys, signs, min_frac=0.6):
    """
    返回一个 prescan 函数：在每个调仓日，对 fkeys 里的因子做横截面 z-score，
    按 signs 方向加权求和得到 composite 分数，再排名。
    fkeys : 入选因子名列表
    signs : 与 fkeys 等长的方向数组（+1/-1），通常直接取 SIGNS 的子集
    min_frac : 有效截面占全样本比例低于此值则该日不建仓（避免样本过薄时噪声主导）
    """
    def psc(cols, idx, t, indc):
        raw = {}
        for s in cols:
            j = idx[s][t]
            vs = []
            ok = True
            for k in fkeys:
                v = RAW[k](cols[s], j)
                if v is None:
                    ok = False
                    break
                vs.append(v)
            if ok:
                raw[s] = vs
        if len(raw) < max(20, int(min_frac * len(cols))):
            return {"rank": {}, "n": 0}
        # 逐因子横截面标准化
        cols_z = list(zip(*raw.values()))
        means, sds = [], []
        for col in cols_z:
            m, sd = _zstat(col)
            means.append(m)
            sds.append(sd)
        comp = {}
        for s, vs in raw.items():
            z = 0.0
            for i, v in enumerate(vs):
                z += (v - means[i]) / sds[i] * signs[i]
            comp[s] = z
        order = sorted(comp, key=lambda s: -comp[s])
        return {"rank": {s: i for i, s in enumerate(order)}, "n": len(order)}
    psc.spec = {}
    return psc
