#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""策略层：每个策略返回目标权重(0~1)，None 表示维持不变

签名: fn(sym, hist, i, state, ctx) -> float | None
  hist : 该标的的**完整**序列 {'open','close','high','low','volume'}（不切片，零拷贝）
  i    : 当前bar索引（用 hist['close'][i] 取当前价）
  ctx  : {'date','t','equity','scan','ind','mkt'}
         ctx['ind'] 已按 f.spec 预计算好全序列指标 -> O(1) 查表
         ctx['mkt'] 市场状态 {'close','ma','above'}

性能约定（重要）：
  策略必须通过 `f.spec = {'ma':[5,20], 'rsi':[14], ...}` 声明所需指标，
  引擎会一次性预计算。禁止在策略里调用 ind.sma(c, n)[i] 这类全序列函数——
  那是 O(总长度)，500只×2000根下会把回测拖到小时级。
"""

import math
import random

from . import indicators as ind
from . import factors as fac
from . import fundamentals as fnd
from .factors import _zstat


# ============================================================
# 1. 双均线趋势跟踪
# ============================================================
def ma_cross(fast=5, slow=20):
    def f(sym, hist, i, state, ctx):
        ma = ctx["ind"]["ma"]
        mf, ms = ma[fast], ma[slow]
        if None in (mf[i], ms[i], mf[i - 1], ms[i - 1]):
            return 0.0
        if mf[i - 1] <= ms[i - 1] and mf[i] > ms[i]:
            return 1.0          # 金叉 -> 买入
        if mf[i - 1] >= ms[i - 1] and mf[i] < ms[i]:
            return 0.0          # 死叉 -> 清仓
        return None             # 持有
    f.__name__ = f"双均线({fast}/{slow})"
    f.spec = {"ma": [fast, slow]}
    return f


# ============================================================
# 2. 双均线 + 趋势过滤（MA60之上才做多）
# ============================================================
def ma_cross_filter(fast=5, slow=20, trend=60):
    def f(sym, hist, i, state, ctx):
        ma = ctx["ind"]["ma"]
        mf, ms, mt = ma[fast], ma[slow], ma[trend]
        if None in (mf[i], ms[i], mt[i], mf[i - 1], ms[i - 1]):
            return 0.0
        if hist["close"][i] < mt[i]:
            return 0.0                                   # 趋势线下不做多
        if mf[i - 1] <= ms[i - 1] and mf[i] > ms[i]:
            return 1.0
        if mf[i - 1] >= ms[i - 1] and mf[i] < ms[i]:
            return 0.0
        return None
    f.__name__ = f"双均线+MA{trend}过滤({fast}/{slow})"
    f.spec = {"ma": [fast, slow, trend]}
    return f


# ============================================================
# 3. 双重确认趋势持有（低换手版）—— 新增
#    入场：金叉 且 价在趋势线上方
#    出场：价跌破趋势线 且 死叉（两个条件同时成立才卖）
#    目的：用"出场更难"换取换手率下降，直接对冲成本敏感性测试暴露的问题
# ============================================================
def trend_hold(fast=5, slow=20, trend=60):
    def f(sym, hist, i, state, ctx):
        ma = ctx["ind"]["ma"]
        mf, ms, mt = ma[fast], ma[slow], ma[trend]
        if None in (mf[i], ms[i], mt[i], mf[i - 1], ms[i - 1], mt[i - 1]):
            return 0.0
        c = hist["close"][i]
        if c > mt[i] and mf[i - 1] <= ms[i - 1] and mf[i] > ms[i]:
            return 1.0
        if c <= mt[i] and mf[i - 1] >= ms[i - 1] and mf[i] < ms[i]:
            return 0.0
        return None
    f.__name__ = f"双重确认趋势({fast}/{slow}/MA{trend})"
    f.spec = {"ma": [fast, slow, trend]}
    return f


# ============================================================
# 4. RSI 均值回归
# ============================================================
def rsi_revert(period=14, buy=30, sell=70, weight=0.5):
    def f(sym, hist, i, state, ctx):
        r = ctx["ind"]["rsi"][period]
        if r[i] is None:
            return 0.0
        if r[i] < buy:
            return weight
        if r[i] > sell:
            return 0.0
        return None
    f.__name__ = f"RSI均值回归({period},{buy}/{sell})"
    f.spec = {"rsi": [period]}
    return f


# ============================================================
# 5. 绝对动量择时
# ============================================================
def abs_momentum(n=20):
    def f(sym, hist, i, state, ctx):
        m = ctx["ind"]["mom"][n]
        if m[i] is None:
            return 0.0
        return 1.0 if m[i] > 0 else 0.0
    f.__name__ = f"绝对动量({n}日)"
    f.spec = {"mom": [n]}
    return f


# ============================================================
# 6. 动量轮动（横截面，需 prescan）
# ============================================================
def rotation(n=20, top=5, reverse=False):
    """
    reverse=False 动量轮动（买涨幅最高的 top 只）
    reverse=True  反转轮动（买跌幅最大的 top 只）—— A股实证上反转效应通常强于动量
    """
    prescan = _mom_prescan(n)

    def f(sym, hist, i, state, ctx):
        scan = (ctx or {}).get("scan") or {}
        rk = (scan.get("rank") or {}).get(sym)
        if rk is None:
            return 0.0
        if reverse:
            total = scan.get("n") or 0
            return 1.0 / top if rk >= total - top else 0.0
        return 1.0 / top if rk < top else 0.0
    f.__name__ = f"{'反转' if reverse else '动量'}轮动({n}日,Top{top})"
    f.spec = {"mom": [n]}
    f.prescan = prescan
    return f


def _mom_prescan(n):
    def psc(cols, idx, t, indc):
        moms = {}
        for sym in cols:
            j = idx[sym][t]
            seq = (indc.get(sym) or {}).get("mom", {}).get(n)
            moms[sym] = seq[j] if (seq is not None and j < len(seq)) else None
        valid = {k: v for k, v in moms.items() if v is not None}
        order = sorted(valid.keys(), key=lambda k: -valid[k])
        rank = {k: i for i, k in enumerate(order)}   # 别用 order.index，那是 O(n^2)
        return {"mom": moms, "rank": rank, "n": len(order)}
    psc.spec = {"mom": [n]}
    return psc


# ============================================================
# 11. 多因子横截面选股（v3 核心）
# ============================================================
def multifactor(fkeys, signs, top):
    """
    fkeys : 入选因子名列表（须都在 fac.RAW 中）
    signs : 与 fkeys 等长的方向数组（+1/-1）
    top   : 每次调仓持有前 top 名
    选股逻辑完全在 fac.make_composite 的 prescan 里完成（横截面 z-score 合成 + 排名）。
    """
    psc = fac.make_composite(fkeys, signs)
    def f(sym, hist, i, state, ctx):
        scan = (ctx or {}).get("scan") or {}
        rk = (scan.get("rank") or {}).get(sym)
        if rk is None:
            return 0.0
        return 1.0 / top if rk < top else 0.0
    f.__name__ = "多因子(%s·Top%d)" % ("+".join(fkeys), top)
    f.prescan = psc
    return f


# ============================================================
# 12. 多因子"极端分位滞回"择股（压换手专用）
#     只在进入最极端分位（enter_q，如 Top5%）时才买，
#     跌出较宽分位（hold_q，如 Top15%）才卖 —— 形成滞回（hysteresis），
#     中间的小幅排名波动不会触发交易，天然把换手压到最低。
#     返回 1.0/0.0，由引擎在"持有的标的"之间自动等权（sum>1 会归一）。
# ============================================================
def multifactor_q(fkeys, signs, enter_q, hold_q):
    """
    enter_q : 进入分位（如 0.05 = 仅当排名进入前 5% 才买入）
    hold_q  : 持有分位（如 0.15 = 排名跌出前 15% 才卖出）
    hold_q > enter_q 形成滞回带：买点更严格、卖点更宽松 -> 减少来回交易
    """
    psc = fac.make_composite(fkeys, signs)

    def f(sym, hist, i, state, ctx):
        scan = (ctx or {}).get("scan") or {}
        rank = (scan.get("rank") or {}).get(sym)
        n = scan.get("n") or 0
        if rank is None or n == 0:
            state["held"] = False
            return 0.0
        held = state.get("held", False)
        enter_r = int(enter_q * n)
        hold_r = int(hold_q * n)
        # 已持有：跌出持有分位才卖；未持有：进入进入分位才买
        keep = rank < hold_r if held else rank < enter_r
        state["held"] = keep
        return 1.0 if keep else 0.0

    f.__name__ = "多因子_极端分位(进%d%%/持%d%%)" % (int(enter_q * 100), int(hold_q * 100))
    f.spec = {}
    f.prescan = psc
    return f


# ============================================================
# 7. ATR 通道突破
# ============================================================
def atr_break(n=20, atr_n=14, k=1.5):
    def f(sym, hist, i, state, ctx):
        ind_ = ctx["ind"]
        a = ind_["atr"][atr_n]
        ma = ind_["ma"][n]
        if ma[i] is None or a[i] is None or i < n:
            return 0.0
        up = max(hist["high"][i - n:i])
        if hist["close"][i] > up:
            return 1.0
        if hist["close"][i] < ma[i] - k * a[i]:
            return 0.0
        return None
    f.__name__ = f"ATR突破({n}日)"
    f.spec = {"atr": [atr_n], "ma": [n]}
    return f


# ============================================================
# 8. 市场状态门控（包装器）—— 新增，对应"缺少市场状态判断"
#    指数在其 MA 之上 -> 正常仓位；之下 -> 仓位压到 bear_w（0=清仓）
# ============================================================
def market_gate(base, bear_w=0.0, cutoff=1.0):
    """
    市场状态门控（包装器）

    base   : 底层策略
    bear_w : 熊市仓位（0=清仓，1=满仓）
    cutoff : 门控放宽系数（默认 1.0 = 原语义：价在 MA 之上才满仓）
             cutoff < 1.0 表示「放宽」：只有当价跌破 cutoff×MA 才减仓，
             即容忍一定幅度的跌破（如 0.90 = 跌破 MA 10% 以内仍满仓）。

    P2 动机：P1 揭示「二元门控在 2024-26 牛市踏空」把价值组合的好成绩吃掉
    （无门控的纯价值 +69.16% 反而是最高）。放宽门控就是让门控只在
    **确定性熊市**减仓，减少牛市中的误踏空。
    """
    def f(sym, hist, i, state, ctx):
        m = (ctx or {}).get("mkt") or {}
        c, ma = m.get("close"), m.get("ma")
        if c is None or ma is None or ma <= 0:
            above = m.get("above")
            gate = 1.0 if above is None else (1.0 if above else bear_w)
        else:
            gate = 1.0 if c >= cutoff * ma else bear_w
        if gate <= 0.0:
            return 0.0
        w = base(sym, hist, i, state, ctx)
        if w is None:
            return None
        return float(w) * gate
    f.__name__ = f"{base.__name__}+市场门控(熊市{bear_w:.0%},放宽{cutoff:.2f})"
    f.spec = getattr(base, "spec", None)
    f.prescan = getattr(base, "prescan", None)
    return f


# ============================================================
# 8a. 组合层波动率目标（正交风控，P4 核心）
# ============================================================
def port_vol_target(base, target=0.20, cap=1.0, floor=0.0):
    """
    组合层波动率目标：按**指数已实现波动**决定总暴露，再统一缩放全组合

        e = clip(target / 指数年化已实现波动率, floor, cap)

    与逐票 `vol_target` 的本质区别（重要）：
      * `vol_target` 按**每只票自己的波动**缩放该票权重 —— 那会系统性超配低波动股，
        等于偷偷引入一个"低波因子"，**与选股层耦合、不正交**。
      * 这里用**指数**的波动算出一个**全组合共用的标量** `e`，
        只缩放总暴露、完全不改变各票相对权重 —— 与选股层严格正交。

    为什么它有望修好 MA200 门控的毛病：
      波动率目标**按风险而非方向**调仓。牛市若走得平稳（低波），`e` 会被 cap 到 1.0
      —— 满仓参与，不踏空；而崩盘通常伴随波动飙升，`e` 自动降下来削平回撤。
      MA200 则是方向性开关，本样本下它把 2024-26 牛市也切掉了一块（P3 诊断）。
    """
    def f(sym, hist, i, state, ctx):
        m = (ctx or {}).get("mkt") or {}
        v = m.get("vol")
        if v is None or v <= 1e-9:
            e = cap                       # 波动不可用（预热期）-> 不干预
        else:
            e = target / v
            e = floor if e < floor else (cap if e > cap else e)
        if e <= 1e-9:
            return 0.0
        w = base(sym, hist, i, state, ctx)
        if w is None:
            return None
        return float(w) * e
    f.__name__ = f"{base.__name__}+组合波动率目标({target:.0%})"
    f.spec = getattr(base, "spec", None)
    f.prescan = getattr(base, "prescan", None)
    return f


# ============================================================
# 8b. 连续仓位门控（正交择时，P3 核心）
# ============================================================
def ramp_gate(base, lo=0.85, hi=1.05):
    """
    连续仓位门控：按指数相对 MA 的偏离度**平滑缩放总仓位**
        ratio = close / MA
        exposure = clip((ratio - lo) / (hi - lo), 0, 1)
    ratio >= hi 满仓；ratio <= lo 空仓；中间线性过渡。

    为什么这是「正交」的（P2 遗留矛盾的解法）：
      * **择时层**只缩放**总暴露**（一个标量 gate）
      * **选股层**（fund_tilt）决定**各票相对权重**（一个向量）
      两层互不干涉 —— 门控不再强迫卖出全部持仓，价值倾斜的 alpha 在
      **任何仓位水平下都继续工作**（只是按比例缩放），不会被"清仓"整个抹掉。
      这正是二元门控做不到的：它一空仓，alpha 就归零（P2：倾斜+原门控
      +64.06% ≈ 纯门控 +65.43%，alpha 被完全抵消）。

    顺带治 P2 诊断出的 whipsaw：平滑过渡不像二元开关那样来回翻转，
    避免了频繁进出带来的换手与损耗。
    """
    def f(sym, hist, i, state, ctx):
        m = (ctx or {}).get("mkt") or {}
        c, ma = m.get("close"), m.get("ma")
        if c is None or ma is None or ma <= 0:
            gate = 1.0
        else:
            ratio = c / ma
            gate = 1.0 if hi <= lo else (ratio - lo) / (hi - lo)
            gate = 0.0 if gate < 0.0 else (1.0 if gate > 1.0 else gate)
        if gate <= 1e-9:
            return 0.0
        w = base(sym, hist, i, state, ctx)
        if w is None:
            return None
        return float(w) * gate
    f.__name__ = f"{base.__name__}+连续仓位门控({lo:.2f}~{hi:.2f})"
    f.spec = getattr(base, "spec", None)
    f.prescan = getattr(base, "prescan", None)
    return f


# ============================================================
# 9. 波动率目标仓位（包装器）—— 新增
#    年化波动越高仓位越低，把组合波动拉向 target
# ============================================================
def vol_target(base, vol_n=60, target=0.20, cap=1.0):
    def f(sym, hist, i, state, ctx):
        sd = (ctx["ind"].get("std") or {}).get(vol_n)
        w = base(sym, hist, i, state, ctx)
        if sd is None or sd[i] is None:
            return w
        c = hist["close"][i]
        if c <= 0:
            return w
        ann = sd[i] / c * (252 ** 0.5)
        scale = min(cap, target / ann) if ann > 1e-6 else cap
        if w is None:
            return None
        return float(w) * scale
    f.__name__ = f"{base.__name__}+波动率目标({target:.0%})"
    f.spec = ind.merge_spec(getattr(base, "spec", None), {"std": [vol_n]})
    f.prescan = getattr(base, "prescan", None)
    return f


# ============================================================
# 10. 买入持有（基准）
# ============================================================
def buy_hold():
    """真正买入并持有：首日建仓一次，此后不再交易"""
    def f(sym, hist, i, state, ctx):
        if state.get("bought"):
            return None          # 维持当前权重，不再调仓
        state["bought"] = True
        return 1.0
    f.__name__ = "买入持有(等权)"
    return f


def always_on():
    """
    始终给出满仓目标（等权）。与 buy_hold 的区别：buy_hold 返回 None（真的不调仓），
    这里每天都返回 1.0，配合调仓带宽使用 —— 用于被 market_gate 包装，
    这样熊市清仓后、牛市回来时能自动买回（buy_hold 的 None 语义会导致买不回来）。
    """
    def f(sym, hist, i, state, ctx):
        return 1.0
    f.__name__ = "始终满仓(等权)"
    return f


# ============================================================
# 13. 多因子横截面选股（point-in-time 基本面，v5 核心）
# ============================================================
def fund_composite(fnames, signs, top, min_frac=0.4):
    """
    用 point-in-time 基本面因子做横截面合成选股（与 multifactor 同构：返回带 .prescan 的策略）。
    fnames : 因子名（须都在 fnd.FACTORS 中，如 'ep'/'bp'/'roe'/'revyoy'...）
    signs  : 与 fnames 等长的方向数组（+1 越大越好 / -1 越小越好）
    top    : 每次调仓持有前 top 名
    第 5 个参数 fund 由引擎透传（fnd.Fund 实例，含 pit(code,date)）。
    每个调仓日：取各票「当时真实可得」的基本面快照，逐因子横截面 z-score × 方向加权，
    排名靠前(top)者由外层策略取 1/top 权重。估值类(ep/bp)当场用价格推导，天然 point-in-time。

    健壮性：当有效覆盖率不足（某期财报真空/大量停牌）时，**沿用上次排名不调仓**，
    而不是返回空 rank —— 否则引擎会对所有票返回 0 权重，把组合误清空。
    """
    state = {"rank": {}, "n": 0}

    def psc(cols, idx, t, indc, fund):
        s0 = next(iter(cols))
        date = cols[s0]["date"][idx[s0][t]]      # 所有标的对齐到同一日期轴
        raw = {}
        for s in cols:
            st = fund.pit(s, date)
            if st is None:
                continue
            price = cols[s]["close"][idx[s][t]]
            if price is None or price <= 0:
                continue
            vs, ok = [], True
            for k in fnames:
                fn = fnd.FACTORS[k][0]
                v = fn(st, price) if k in ("ep", "bp") else fn(st)
                if v is None:
                    ok = False
                    break
                vs.append(v)
            if ok:
                raw[s] = vs
        if len(raw) < max(20, int(min_frac * len(cols))):
            return {"rank": state["rank"], "n": state["n"]}   # 覆盖率不足：沿用上次，不误清空
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
        rank = {s: i for i, s in enumerate(order)}
        state["rank"], state["n"] = rank, len(order)
        return {"rank": rank, "n": len(order)}
    psc.spec = {}

    def f(sym, hist, i, state_, ctx):
        scan = (ctx or {}).get("scan") or {}
        rk = (scan.get("rank") or {}).get(sym)
        if rk is None:
            return 0.0
        return 1.0 / top if rk < top else 0.0
    f.__name__ = "基本面多因子(%s·Top%d)" % ("+".join(fnames), top)
    f.spec = {}
    f.prescan = psc
    return f


def rand_fund_composite(seed, fnames, top, min_frac=0.4):
    """同因子、同覆盖、同成本，但信号随机化（rank 随机打乱）——选股 alpha 照妖镜。
    与被检验策略保持完全相同的调仓频率、持仓数、市场门控，唯一区别是「按因子打分」
    换成「随机抽」，比的是纯选股信号是否携带信息。"""
    rnd = random.Random(seed)
    state = {"rank": {}, "n": 0}

    def psc(cols, idx, t, indc, fund):
        s0 = next(iter(cols))
        date = cols[s0]["date"][idx[s0][t]]
        raw = {}
        for s in cols:
            st = fund.pit(s, date)
            if st is None:
                continue
            price = cols[s]["close"][idx[s][t]]
            if price is None or price <= 0:
                continue
            vs, ok = [], True
            for k in fnames:
                fn = fnd.FACTORS[k][0]
                v = fn(st, price) if k in ("ep", "bp") else fn(st)
                if v is None:
                    ok = False
                    break
                vs.append(v)
            if ok:
                raw[s] = vs
        if len(raw) < max(20, int(min_frac * len(cols))):
            return {"rank": state["rank"], "n": state["n"]}
        syms = list(raw.keys())
        rnd.shuffle(syms)
        rank = {s: i for i, s in enumerate(syms)}
        state["rank"], state["n"] = rank, len(syms)
        return {"rank": rank, "n": len(syms)}
    psc.spec = {}

    def f(sym, hist, i, state_, ctx):
        scan = (ctx or {}).get("scan") or {}
        rk = (scan.get("rank") or {}).get(sym)
        if rk is None:
            return 0.0
        return 1.0 / top if rk < top else 0.0
    f.__name__ = "随机信号选股Top%d#%d" % (top, seed)
    f.spec = {}
    f.prescan = psc
    return f


# ============================================================
# 14. 基本面「价值倾斜」（smart beta，P2 核心）
#     与 fund_composite 的「Top-N 集中」相对：这里**持有宽基全集**，
#     保留市场 beta 与分散度，只对各票按因子 z-score 做 softmax 倾斜：
#         w_i ∝ exp(z_i / tau)
#     目的是只吃价值溢价（BP 已被证实 OOS 稳健），而不吃集中带来的个股风险。
#     返回**连续权重**，由引擎按 sum(targets) 归一化 -> 自动成为「倾斜后的宽基」。
#
#     P1 的两个失败根因，这里各解一个：
#       * 集中牺牲分散 -> 倾斜保留全样本持仓，beta 不丢
#       * 二元门控牛市踏空 -> 配合 market_gate(cutoff<1.0) 放宽门控
# ============================================================
def fund_tilt(fname, tau=10.0, floor=0.0, min_frac=0.4):
    """
    fname : 因子名（须在 fnd.FACTORS 中；P2 用 'bp'，即已验证的价值因子）
    tau   : 倾斜锐度。越大越接近等权（tau→∞ 即退化为等权宽基），越小越向高分票集中
    floor : 与等权混合的比例（0=纯 softmax 倾斜，0.5=一半等权一半倾斜）。
            用于在「倾斜强度」与「分散度」之间取舍，也是防止极端 tau 下过度集中的保险。
    """
    state = {"weight": {}, "n": 0}

    def psc(cols, idx, t, indc, fund):
        s0 = next(iter(cols))
        date = cols[s0]["date"][idx[s0][t]]
        fn = fnd.FACTORS[fname][0]
        is_val = fname in ("ep", "bp")
        raw = {}
        for s in cols:
            st = fund.pit(s, date)
            if st is None:
                continue
            price = cols[s]["close"][idx[s][t]]
            if price is None or price <= 0:
                continue
            v = fn(st, price) if is_val else fn(st)
            if v is None:
                continue
            raw[s] = v
        if len(raw) < max(20, int(min_frac * len(cols))):
            return {"weight": state["weight"], "n": state["n"]}   # 覆盖率不足：沿用上次
        vals = list(raw.values())
        m, sd = _zstat(vals)
        sd = sd if sd > 1e-9 else 1.0
        w = {s: math.exp((v - m) / sd / tau) for s, v in raw.items()}
        tot = sum(w.values()) or 1.0
        w = {s: v / tot for s, v in w.items()}          # 归一到 sum=1（便于与等权混合）
        if floor > 0:
            eq = 1.0 / len(w)
            w = {s: floor * eq + (1.0 - floor) * v for s, v in w.items()}
        state["weight"], state["n"] = w, len(w)
        return {"weight": w, "n": len(w)}
    psc.spec = {}

    def f(sym, hist, i, state_, ctx):
        scan = (ctx or {}).get("scan") or {}
        wt = (scan.get("weight") or {}).get(sym)
        if wt is None:
            return 0.0
        return wt
    f.__name__ = f"基本面倾斜({fname},tau={tau:.0f},floor={floor:.2f})"
    f.spec = {}
    f.prescan = psc
    return f


def multi_tilt(fnames, weights=None, tau=10.0, floor=0.0, min_frac=0.4, req_frac=0.5):
    """
    多因子合成倾斜：把若干个 point-in-time 基本面因子的横截面 z-score 按权重合成为
    一个综合分数，再对该分数做 softmax 倾斜（与 `fund_tilt` 的单因子版本同构）。

    fnames  : 因子名列表（须在 fnd.FACTORS 中）
    weights : 各因子权重；None 表示等权。权重会乘以因子的理论方向
    tau     : 倾斜锐度（同 fund_tilt）
    floor   : 与等权混合比例（同 fund_tilt）
    req_frac: 单只票至少要有多少**权重比例**的因子有值才纳入（缺失因子按剩余权重重新
              归一化，等价于"缺失=中性"，不因缺一个因子就把票踢掉）

    为什么不直接用因子值的交集：各因子覆盖率不同（如 EP 要求盈利为正），
    取交集会让样本量大幅缩水并引入"数据完整度"这个与 alpha 无关的筛选偏差。
    """
    names = list(fnames)
    if weights is None:
        weights = [1.0 / len(names)] * len(names)
    weights = [w * fnd.FACTORS[f][1] for f, w in zip(names, list(weights))]
    tot_w = sum(abs(x) for x in weights) or 1.0
    state = {"weight": {}, "n": 0}

    def _scan(cols, idx, t, fund):
        s0 = next(iter(cols))
        date = cols[s0]["date"][idx[s0][t]]
        raw = {f: {} for f in names}
        for s in cols:
            stt = fund.pit(s, date)
            if stt is None:
                continue
            price = cols[s]["close"][idx[s][t]]
            if price is None or price <= 0:
                continue
            for f in names:
                fn = fnd.FACTORS[f][0]
                v = fn(stt, price) if f in ("ep", "bp") else fn(stt)
                if v is not None:
                    raw[f][s] = v
        z = {}
        for f in names:
            vals = list(raw[f].values())
            if len(vals) < 20:
                z[f] = None
                continue
            m, sd = _zstat(vals)
            sd = sd if sd > 1e-9 else 1.0
            z[f] = {s: (v - m) / sd for s, v in raw[f].items()}
        score = {}
        for s in cols:
            num, den = 0.0, 0.0
            for f, wf in zip(names, weights):
                zz = z.get(f)
                if zz and s in zz:
                    num += wf * zz[s]
                    den += abs(wf)
            if den < tot_w * req_frac - 1e-9:
                continue
            score[s] = num / den
        return score

    def _soft(score, ncol):
        if len(score) < max(20, int(min_frac * ncol)):
            return None
        w = {s: math.exp(v / tau) for s, v in score.items()}
        t_ = sum(w.values()) or 1.0
        w = {s: v / t_ for s, v in w.items()}
        if floor > 0:
            eq = 1.0 / len(w)
            w = {s: floor * eq + (1.0 - floor) * v for s, v in w.items()}
        return w

    def psc(cols, idx, t, indc, fund):
        w = _soft(_scan(cols, idx, t, fund), len(cols))
        if w is None:
            return {"weight": state["weight"], "n": state["n"]}   # 覆盖率不足：沿用上次
        state["weight"], state["n"] = w, len(w)
        return {"weight": w, "n": len(w)}
    psc.spec = {}

    def f(sym, hist, i, state_, ctx):
        scan = (ctx or {}).get("scan") or {}
        wt = (scan.get("weight") or {}).get(sym)
        if wt is None:
            return 0.0
        return wt
    f.__name__ = f"多因子倾斜({'+'.join(names)},tau={tau:.0f},floor={floor:.2f})"
    f.spec = {}
    f.prescan = psc
    return f


def rand_multi_tilt(seed, fnames, weights=None, tau=10.0, floor=0.0,
                    min_frac=0.4, req_frac=0.5):
    """多因子倾斜的**随机对照**：合成方式、倾斜强度、覆盖、调仓、风控全部相同，
    唯一区别是把「哪个票拿哪个综合分数」随机打乱 —— 权重分布不变（换手量级相同），
    信号退化为噪声。用于检验合成后的 alpha 是真的来自因子还是来自"倾斜"这个动作本身。"""
    base = multi_tilt(fnames, weights, tau=tau, floor=floor,
                      min_frac=min_frac, req_frac=req_frac)
    rnd = random.Random(seed)
    state = {"weight": {}, "n": 0}
    inner = base.prescan

    def psc(cols, idx, t, indc, fund):
        out = inner(cols, idx, t, indc, fund)
        w = out.get("weight") or {}
        if not w:
            return out
        syms = list(w.keys())
        shuffled = [w[s] for s in syms]
        rnd.shuffle(shuffled)
        w2 = dict(zip(syms, shuffled))
        state["weight"], state["n"] = w2, len(w2)
        return {"weight": w2, "n": len(w2)}
    psc.spec = {}

    def f(sym, hist, i, state_, ctx):
        scan = (ctx or {}).get("scan") or {}
        wt = (scan.get("weight") or {}).get(sym)
        if wt is None:
            return 0.0
        return wt
    f.__name__ = f"随机多因子倾斜#{seed}(tau={tau:.0f})"
    f.spec = {}
    f.prescan = psc
    return f


def rand_fund_tilt(seed, fname, tau=10.0, floor=0.0, min_frac=0.4):
    """同倾斜强度、同覆盖、同成本、同门控，但把「哪个票拿哪个权重」随机打乱。
    权重**分布完全不变**（因而换手量级相同），唯一区别是信号变成噪声 —— 选股 alpha 照妖镜。"""
    rnd = random.Random(seed)
    state = {"weight": {}, "n": 0}

    def psc(cols, idx, t, indc, fund):
        s0 = next(iter(cols))
        date = cols[s0]["date"][idx[s0][t]]
        fn = fnd.FACTORS[fname][0]
        is_val = fname in ("ep", "bp")
        raw = {}
        for s in cols:
            st = fund.pit(s, date)
            if st is None:
                continue
            price = cols[s]["close"][idx[s][t]]
            if price is None or price <= 0:
                continue
            v = fn(st, price) if is_val else fn(st)
            if v is None:
                continue
            raw[s] = v
        if len(raw) < max(20, int(min_frac * len(cols))):
            return {"weight": state["weight"], "n": state["n"]}
        vals = list(raw.values())
        m, sd = _zstat(vals)
        sd = sd if sd > 1e-9 else 1.0
        w = {s: math.exp((v - m) / sd / tau) for s, v in raw.items()}
        tot = sum(w.values()) or 1.0
        w = {s: v / tot for s, v in w.items()}
        if floor > 0:
            eq = 1.0 / len(w)
            w = {s: floor * eq + (1.0 - floor) * v for s, v in w.items()}
        syms = list(w.keys())
        shuffled = [w[s] for s in syms]
        rnd.shuffle(shuffled)                            # 只打乱"谁拿哪个权重"
        w = dict(zip(syms, shuffled))
        state["weight"], state["n"] = w, len(w)
        return {"weight": w, "n": len(w)}
    psc.spec = {}

    def f(sym, hist, i, state_, ctx):
        scan = (ctx or {}).get("scan") or {}
        wt = (scan.get("weight") or {}).get(sym)
        if wt is None:
            return 0.0
        return wt
    f.__name__ = f"随机倾斜#{seed}(tau={tau:.0f})"
    f.spec = {}
    f.prescan = psc
    return f


REGISTRY = {
    "ma_cross": ma_cross,
    "ma_filter": ma_cross_filter,
    "trend_hold": trend_hold,
    "rsi": rsi_revert,
    "momentum": abs_momentum,
    "rotation": rotation,
    "atr": atr_break,
    "gate": market_gate,
    "vol": vol_target,
    "hold": buy_hold,
    "always": always_on,
    "multifactor": multifactor,
    "multifactor_q": multifactor_q,
    "fund_composite": fund_composite,
    "rand_fund_composite": rand_fund_composite,
    "fund_tilt": fund_tilt,
    "rand_fund_tilt": rand_fund_tilt,
    "multi_tilt": multi_tilt,
    "rand_multi_tilt": rand_multi_tilt,
    "ramp_gate": ramp_gate,
    "port_vol_target": port_vol_target,
}
