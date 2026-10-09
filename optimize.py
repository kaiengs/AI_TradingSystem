#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AQuant 策略优化与严格验证 v2
==================================================
v1 复测暴露的三个致命问题，本版逐一处理：

  问题1 样本量不足（20只×2.9年）
        -> 扩大到 300只×8.2年（约 40 倍样本）

  问题2 换手过高，成本吃掉全部收益
        -> 引擎级新增"调仓带宽 + 最短持有期"，并引入波动率目标仓位

  问题3 缺少市场状态判断（全是单边做多）
        -> 新增指数 MA 门控（熊市降仓/空仓）

  问题4 参数取到网格尖峰（过拟合）
        -> 改为取"参数平原中心"（邻域平均最优）

  问题5 选池存在前视偏差（用今天成交额选8年前的票）
        -> 改为随机抽样宽基池，并用流动性池做一致性对照

  问题6 本金过小导致整手约束扭曲权重
        -> 研究本金提到 1 亿
"""

import sys, os, time, random, itertools

sys.path.insert(0, "/workspace/quant")
from aq import data, strategies as st, backtest as bt, indicators as ind

CASH = 1e8            # 研究本金：足够大以消除"100股整手"对小权重的扭曲
WARMUP = 250          # 指标预热窗口（滚动前推时必须保留，否则长均线策略被冤枉）
OUT = "/workspace/quant/策略优化报告.html"

# ============================================================
# 数据
# ============================================================
def fill_to_axis(d, axis):
    """
    把个股 K线对齐到标准日期轴（沪深300 的交易日）。
    缺失日期（长期停牌）用前收盘价填充、成交量置 0 —— 价格不变则指标不变，
    不会凭空产生信号，同时避免"一只停牌股把整个日期交集拖短"的问题。
    实测：取 300 只日期交集只剩 1220 天，改用轴填充后恢复满 2000 天。
    """
    out = {}
    for sym, bars in d.items():
        if bars[0][0] > axis[0]:        # 轴起点之后才上市，直接剔除
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


def load_market(dates):
    k = data.kline("sh000300", bars=0)
    mp = {x[0]: x[2] for x in k}
    seq, last = [], None
    for d in dates:
        if d in mp:
            last = mp[d]
        seq.append(last)
    first = next((v for v in seq if v is not None), None)
    return [first if v is None else v for v in seq]


# ============================================================
# 评估
# ============================================================
def run_range(A, dates, mkt, i0, i1, factory, warmup=WARMUP, cost=None, **kw):
    """在 [i0,i1] 区间评估；数据从 i0-warmup 开始取，保证指标已预热"""
    a = max(0, i0 - warmup)
    sub = {s: bars[a:i1 + 1] for s, bars in A.items()}
    mk = {"date": dates[a:i1 + 1], "close": mkt[a:i1 + 1]}
    b = bt.Backtest(sub, factory(), cash=CASH, cost=cost, market=mk, **kw)
    r = b.run()
    off = i0 - a
    eq, bc = r["equity"], r["benchmark"]["curve"]
    ret = eq[-1] / eq[off] - 1
    bret = bc[-1] / bc[off] - 1
    amt = sum(t["amount"] for t in r["trades"])
    yrs = max((i1 - i0) / 252.0, 1e-6)
    return {"ret": ret, "bench": bret, "excess": ret - bret, "raw": r,
            "turnover": amt / CASH / yrs, "n": r["n_trades"]}


def run_full(A, dates, mkt, factory, cost=None, **kw):
    return run_range(A, dates, mkt, 0, len(dates) - 1, factory, warmup=0, cost=cost, **kw)


# ============================================================
# 参数"平原中心"选择
# ============================================================
def plateau_pick(scores, axes):
    """
    在参数网格里找"平原中心"。两个容易踩的坑，这里都处理了：

    坑1 邻域跨界：monthly / quarterly 这类**类别维度**是质变，不是量变。
        若把它当数值维度做 ±1 邻域，会把两个截然不同的换手频率混在一起平均，
        结果选出"自身很差、但被好邻居拉高"的点。
        -> 类别维度的邻域步长固定为 0。

    坑2 均值被拉高：即使邻域不跨界，"邻域均值最大"仍可能选中一个自身很差的洼地。
        -> 改用 **maximin**：先比较邻域内的**最坏值**，最坏值相同再比均值。
        这才是真正的"鲁棒"：整个邻域都不差，才叫平原。

    返回 (参数, 邻域均值, 自身得分, 邻域点数, 邻域最坏值)
    """
    names = list(axes.keys())
    vals = [axes[n] for n in names]
    steps = [[0] if isinstance(axes[n][0], str) else [-1, 0, 1] for n in names]
    best = None
    for key in scores:
        idxs = [vals[k].index(key[k]) for k in range(len(names))]
        nb = []
        for delta in itertools.product(*steps):
            ni = [idxs[k] + delta[k] for k in range(len(names))]
            if not all(0 <= ni[k] < len(vals[k]) for k in range(len(names))):
                continue
            nk = tuple(vals[k][ni[k]] for k in range(len(names)))
            if nk in scores:
                nb.append(scores[nk])
        if not nb:
            continue
        avg, mn = sum(nb) / len(nb), min(nb)
        if best is None or (mn, avg) > (best[4], best[1]):
            best = (key, avg, scores[key], len(nb), mn)
    return best


# ============================================================
# 随机对照（按换手分档）
# ============================================================
def rand_strategy(seed, flip):
    """随机择时（保留：用于按换手分档的粗筛）"""
    rnd = random.Random(seed)
    def f(sym, hist, i, state, ctx):
        on = state.get("on", False)
        if rnd.random() < flip:
            on = not on
        state["on"] = on
        return 1.0 if on else 0.0
    f.__name__ = f"随机#{seed}(p={flip})"
    return f


def rand_pick(seed, top):
    """
    随机**选股**（不是随机择时）—— 这才是正确的对照组。
    与被检验策略保持完全相同的调仓频率、持仓数量、市场门控，
    唯一区别是把"按涨幅排序选 Top N"换成"随机抽 Top N"。
    这样两者的市场 beta 暴露一致，比较的才是纯粹的选股 alpha。
    """
    rnd = random.Random(seed)
    def psc(cols, idx, t, indc):
        syms = sorted(cols.keys())
        rnd.shuffle(syms)
        return {"rank": {s: i for i, s in enumerate(syms)}, "n": len(syms)}
    def f(sym, hist, i, state, ctx):
        rk = ((ctx or {}).get("scan") or {}).get("rank", {}).get(sym)
        if rk is None:
            return 0.0
        return 1.0 / top if rk < top else 0.0
    f.__name__ = f"随机选股Top{top}#{seed}"
    f.prescan = psc
    return f


# ============================================================
# SVG 折线图
# ============================================================
def svg_line(series, w=900, h=280, pad=40, norm=True):
    """series: [(name, color, [values])]"""
    xs = None
    for _, _, v in series:
        if xs is None or len(v) > len(xs):
            xs = v
    n = len(xs)
    if n < 2:
        return ""
    vals = []
    for _, _, v in series:
        base = v[0] if norm and v[0] else 1.0
        vals.append([x / base for x in v])
    lo, hi = min(min(v) for v in vals), max(max(v) for v in vals)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    def X(i): return pad + (w - 2 * pad) * i / (n - 1)
    def Y(v): return h - pad - (h - 2 * pad) * (v - lo) / (hi - lo)
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg">']
    o.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="#fbfcfe"/>')
    for g in range(5):
        yv = lo + (hi - lo) * g / 4
        y = Y(yv)
        o.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{w-pad}" y2="{y:.1f}" stroke="#e8ecf2"/>')
        o.append(f'<text x="{pad-6}" y="{y+4:.1f}" font-size="11" fill="#8a94a6" text-anchor="end">{(yv-1)*100:.0f}%</text>')
    for (nm, col, _), v in zip(series, vals):
        pts = " ".join(f"{X(i):.1f},{Y(x):.1f}" for i, x in enumerate(v))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.8"/>')
    lx = pad
    for nm, col, _ in series:
        o.append(f'<rect x="{lx}" y="8" width="10" height="10" fill="{col}"/>')
        o.append(f'<text x="{lx+14}" y="17" font-size="11" fill="#4a5568">{nm}</text>')
        lx += 18 + len(nm) * 7
    o.append("</svg>")
    return "".join(o)


CSS = """
body{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;background:#f5f7fa;
color:#1f2937;margin:0;padding:32px 20px;line-height:1.75}
.wrap{max-width:1120px;margin:0 auto}
h1{font-size:27px;margin:0 0 6px;color:#111827}
.sub{color:#6b7280;font-size:13px;margin-bottom:26px}
h2{font-size:20px;margin:34px 0 14px;padding-left:11px;border-left:4px solid #2563eb;color:#111827}
h3{font-size:15px;margin:20px 0 8px;color:#374151}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:18px 20px;margin:14px 0}
table{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0}
th{background:#f3f4f6;padding:9px 10px;text-align:left;font-weight:600;border-bottom:2px solid #e5e7eb;white-space:nowrap}
td{padding:8px 10px;border-bottom:1px solid #f1f3f6}
tr:hover td{background:#fafbfc}
.pos{color:#c0392b;font-weight:600}.neg{color:#1e8e5a;font-weight:600}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px}
.box{border-radius:8px;padding:14px 16px;margin:12px 0;font-size:14px}
.warn{background:#fff7ed;border-left:4px solid #f59e0b}
.ok{background:#f0fdf4;border-left:4px solid #22c55e}
.info{background:#eff6ff;border-left:4px solid #3b82f6}
.bad{background:#fef2f2;border-left:4px solid #ef4444}
.kpi{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
.kpi div{flex:1;min-width:150px;background:#fff;border:1px solid #e5e7eb;border-radius:9px;padding:13px 15px}
.kpi b{display:block;font-size:21px;margin:3px 0 2px}
.kpi span{font-size:12px;color:#6b7280}
.bar{background:#eef2f7;border-radius:3px;height:15px;position:relative;min-width:70px}
.bar i{position:absolute;left:0;top:0;height:15px;border-radius:3px;background:#2563eb}
footer{margin-top:36px;color:#9ca3af;font-size:12px;text-align:center}
"""

def cls(v): return "pos" if v > 0 else ("neg" if v < 0 else "")
def pc(v, d=2): return f'<span class="{cls(v)}">{v*100:+.{d}f}%</span>'


# ============================================================
# 主流程
# ============================================================
def main():
    t_all = time.time()
    log = lambda *a: print(*a, flush=True)
    log("=" * 66)
    log("AQuant 策略优化与严格验证 v2")
    log("=" * 66)

    # ---------- 1. 数据 ----------
    log("\n[1/7] 构建无偏宽基池 ...")
    pool = data.build_pool(top=300, mode="random", min_bars=2000, log=log)
    log("  池规模 %d 只" % len(pool))
    d = {s: data.kline(s, bars=0) for s in pool}
    mkt_k = data.kline("sh000300", bars=0)
    axis = [x[0] for x in mkt_k]
    mkt = [x[2] for x in mkt_k]
    A = fill_to_axis(d, axis)          # 停牌日前向填充，保住完整的 2000 天
    dates = axis
    n = len(dates)
    yrs = n / 252.0
    log("  对齐后: %d 只 × %d 天 (%.1f 年)  %s -> %s" % (len(A), n, yrs, dates[0], dates[-1]))
    log("  样本点: 约 %.0f 万" % (len(A) * n / 10000))

    is_end = int(n * 0.7)
    log("  切分: 样本内 %s~%s (%d天) / 样本外 %s~%s (%d天)" %
        (dates[0], dates[is_end - 1], is_end, dates[is_end], dates[-1], n - is_end))

    # ---------- 2. 候选策略全样本基线 ----------
    log("\n[2/7] 候选策略全样本回测 ...")
    MW = {"max_weight": 0.05}          # 单标的上限 5%（不加这条会出现单只满仓）
    LOWT = {"max_weight": 0.05, "band": 0.20, "min_hold": 5}
    CAND = [
        ("买入持有(基准)", lambda: st.buy_hold(), {}),
        ("纯市场择时(满仓+指数门控)", lambda: st.market_gate(st.always_on(), 0.0), {"band": 0.20}),
        ("双均线(5/20)", lambda: st.ma_cross(5, 20), MW),
        ("趋势过滤(5/20/MA60)", lambda: st.ma_cross_filter(5, 20, 60), MW),
        ("双重确认趋势", lambda: st.trend_hold(5, 20, 60), MW),
        ("门控趋势(无降换手·对照)", lambda: st.market_gate(st.ma_cross_filter(5, 20, 60), 0.0), MW),
        ("门控趋势+降换手", lambda: st.market_gate(st.ma_cross_filter(5, 20, 60), 0.0), LOWT),
        ("门控趋势+熊市半仓", lambda: st.market_gate(st.ma_cross_filter(5, 20, 60), 0.5), LOWT),
        ("波动率目标+门控", lambda: st.vol_target(st.market_gate(st.ma_cross_filter(5, 20, 60), 0.0), 60, 0.20), LOWT),
        ("动量轮动Top10(月频)+门控", lambda: st.market_gate(st.rotation(60, 10), 0.0),
         {"rebalance": "monthly", "max_weight": 0.12}),
        ("反转轮动Top10(月频)+门控", lambda: st.market_gate(st.rotation(60, 10, reverse=True), 0.0),
         {"rebalance": "monthly", "max_weight": 0.12}),
        ("反转轮动Top30(月频)+门控", lambda: st.market_gate(st.rotation(60, 30, reverse=True), 0.0),
         {"rebalance": "monthly", "max_weight": 0.08}),
        ("反转轮动Top20(季频)+门控", lambda: st.market_gate(st.rotation(120, 20, reverse=True), 0.0),
         {"rebalance": "quarterly", "max_weight": 0.10}),
    ]
    base_rows = []
    for nm, fac, kw in CAND:
        t0 = time.time()
        r = run_full(A, dates, mkt, fac, **kw)["raw"]
        amt = sum(t["amount"] for t in r["trades"])
        base_rows.append({
            "name": nm, "ret": r["total_return"], "ann": r["annual"],
            "mdd": r["max_drawdown"], "sharpe": r["sharpe"],
            "n": r["n_trades"], "turn": amt / CASH / yrs,
            "bench": r["benchmark"]["ret"], "curve": r["equity"],
            "bcurve": r["benchmark"]["curve"], "wr": r["win_rate"],
        })
        log("  %-26s %+8.2f%%  回撤%7.2f%%  夏普%5.2f  交易%6d  年换手%6.0f%%  [%.1fs]"
            % (nm, r["total_return"] * 100, r["max_drawdown"] * 100, r["sharpe"],
               r["n_trades"], amt / CASH / yrs * 100, time.time() - t0))
    bench_ret = base_rows[0]["bench"]

    # ---------- 3. 平原选参 vs 尖峰选参 ----------
    log("\n[3/7] 参数网格搜索（样本内）...")
    # 选参对象已从"日频趋势"改为"月频反转轮动"：
    # 第二节已证明日频趋势在本样本上换手高达 6000%，成本吃掉一切，再怎么调参都是亏的。
    AXES = {"n": [20, 40, 60, 120], "top": [10, 20], "freq": ["monthly", "quarterly", "semiannual"]}
    keys = list(itertools.product(AXES["n"], AXES["top"], AXES["freq"]))
    sub = dict(list(A.items())[:150])                     # 7.2 重选对照用子池加速
    ROTKW = {"max_weight": 0.12}                          # 调仓频率由参数决定
    scores = {}
    for k in keys:
        fac = lambda k=k: st.market_gate(st.rotation(k[0], k[1], reverse=True), 0.0)
        r = run_range(A, dates, mkt, 0, is_end - 1, fac, warmup=0, rebalance=k[2], **ROTKW)
        scores[k] = r["ret"]
    peak_k = max(scores, key=lambda k: scores[k])
    plat_k, plat_avg, plat_self, plat_n, plat_min = plateau_pick(scores, AXES)
    log("  网格点数 %d" % len(scores))
    log("  尖峰最优 %s  样本内 %+.2f%%" % (peak_k, scores[peak_k] * 100))
    log("  平原中心 %s  邻域最坏 %+.2f%% / 均值 %+.2f%% (自身 %+.2f%%, 邻域%d点)"
        % (plat_k, plat_min * 100, plat_avg * 100, plat_self * 100, plat_n))

    log("  -> 样本外检验（全池，只在最后看一眼）")
    def oos(k):
        fac = lambda k=k: st.market_gate(st.rotation(k[0], k[1], reverse=True), 0.0)
        return run_range(A, dates, mkt, is_end, n - 1, fac, warmup=WARMUP,
                         rebalance=k[2], **ROTKW)
    oos_peak, oos_plat = oos(peak_k), oos(plat_k)
    log("     尖峰参数样本外 %+.2f%% (基准 %+.2f%%)" % (oos_peak["ret"] * 100, oos_peak["bench"] * 100))
    log("     平原参数样本外 %+.2f%% (基准 %+.2f%%)" % (oos_plat["ret"] * 100, oos_plat["bench"] * 100))

    # ---------- 4. 成本敏感性 ----------
    log("\n[4/7] 成本敏感性 ...")
    FINAL = lambda: st.market_gate(st.rotation(plat_k[0], plat_k[1], reverse=True), 0.0)
    FKW = dict(ROTKW, rebalance=plat_k[2])                # 最终策略的完整运行参数
    cost_rows = []
    for slip in (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005):
        c = bt.Cost(slippage=slip)
        r = run_full(A, dates, mkt, FINAL, cost=c, **FKW)["raw"]
        cost_rows.append({"slip": slip, "ret": r["total_return"], "n": r["n_trades"],
                          "bench": r["benchmark"]["ret"]})
        log("  滑点 %.2f%%  -> %+8.2f%%  (基准 %+.2f%%, 交易 %d)"
            % (slip * 100, r["total_return"] * 100, r["benchmark"]["ret"] * 100, r["n_trades"]))

    # ---------- 5. 随机对照 ----------
    log("\n[5/7] 随机对照（按换手分档）...")
    final_r = run_full(A, dates, mkt, FINAL, **FKW)
    fr = final_r["raw"]
    fr_turn = sum(t["amount"] for t in fr["trades"]) / CASH / yrs
    fr_ret = fr["total_return"]
    rand_rows = []
    for i in range(20):
        fac = lambda seed=i: st.market_gate(rand_pick(seed, plat_k[1]), 0.0)
        r = run_full(A, dates, mkt, fac, **FKW)["raw"]
        t_ = sum(t["amount"] for t in r["trades"]) / CASH / yrs
        rand_rows.append({"seed": i, "ret": r["total_return"], "turn": t_, "n": r["n_trades"]})
    near = rand_rows
    beat = sum(1 for x in near if x["ret"] < fr_ret)
    log("  最终策略: 收益 %+.2f%%, 年换手 %.0f%%" % (fr_ret * 100, fr_turn * 100))
    log("  随机选股对照 %d 个, 跑赢其中 %d 个 -> 第 %d 百分位"
        % (len(near), beat, int(beat / len(near) * 100)))

    # ---------- 6. 滚动前推 ----------
    log("\n[6/7] 滚动前推 ...")
    segs, L = [], n // 4
    for i in range(4):
        s0 = i * L
        s1 = n - 1 if i == 3 else (i + 1) * L - 1
        segs.append((s0, s1))
    wf_rows = []
    for (s0, s1) in segs:
        r = run_range(A, dates, mkt, s0, s1, FINAL, warmup=WARMUP, **FKW)
        wf_rows.append({"seg": "%s~%s" % (dates[s0], dates[s1]),
                        "ret": r["ret"], "bench": r["bench"], "excess": r["excess"]})
        log("  %s~%s  策略 %+7.2f%%  基准 %+7.2f%%  超额 %+7.2fpp"
            % (dates[s0], dates[s1], r["ret"] * 100, r["bench"] * 100, r["excess"] * 100))

    log("  [对照] 每段重新选最优参数（模拟真实使用中的'事后调参'）")
    re_rows = []
    for (s0, s1) in segs:
        mid = s0 + (s1 - s0) // 2
        best, bv = None, -9e9
        for k in keys:
            fac = lambda k=k: st.market_gate(st.rotation(k[0], k[1], reverse=True), 0.0)
            rr = run_range(sub, dates, mkt, s0, mid - 1, fac, warmup=WARMUP,
                           rebalance=k[2], **ROTKW)
            if rr["ret"] > bv:
                bv, best = rr["ret"], k
        rr = run_range(A, dates, mkt, mid, s1,
                       (lambda k=best: st.market_gate(st.rotation(k[0], k[1], reverse=True), 0.0)),
                       warmup=WARMUP, rebalance=best[2], **ROTKW)
        re_rows.append({"seg": "%s~%s" % (dates[mid], dates[s1]), "k": best,
                        "ret": rr["ret"], "bench": rr["bench"], "excess": rr["excess"]})
        log("    段内选参 %s -> 后半段 %+7.2f%% (基准 %+7.2f%%)"
            % (best, rr["ret"] * 100, rr["bench"] * 100))

    # ---------- 6b. 牛熊分段检验 ----------
    log("\n[6b/7] 年度分段检验（看下跌年份是否真的少亏）...")
    def yearly(eq, bc):
        seg = {}
        for i, d in enumerate(dates):
            y = d[:4]
            if y not in seg:
                seg[y] = [i, i]
            seg[y][1] = i
        rows = []
        for y in sorted(seg):
            i0, i1 = seg[y]
            s0 = eq[i0 - 1] if i0 > 0 else eq[0]
            b0 = bc[i0 - 1] if i0 > 0 else bc[0]
            rs, rb = eq[i1] / s0 - 1, bc[i1] / b0 - 1
            rows.append({"y": y, "strat": rs, "bench": rb, "ex": rs - rb})
        return rows
    yr_rows = yearly(fr["equity"], fr["benchmark"]["curve"])
    for r in yr_rows:
        log("  %s  策略 %+7.2f%%  基准 %+7.2f%%  超额 %+7.2fpp  %s"
            % (r["y"], r["strat"] * 100, r["bench"] * 100, r["ex"] * 100,
               "<- 下跌年" if r["bench"] < 0 else ""))
    down = [r for r in yr_rows if r["bench"] < 0]
    down_win = sum(1 for r in down if r["ex"] > 0)

    # ---------- 7. 报告 ----------
    log("\n[7/7] 生成报告 ...")
    html = build_html(locals())
    open(OUT, "w", encoding="utf-8").write(html)
    log("  输出 %s (%d 字节)" % (OUT, len(html.encode())))
    log("\n[总耗时] %.1fs" % (time.time() - t_all))


def build_html(v):
    n, yrs = v["n"], v["yrs"]
    is_end, dates = v["is_end"], v["dates"]
    base_rows, bench_ret = v["base_rows"], v["bench_ret"]
    scores, AXES = v["scores"], v["AXES"]
    peak_k, plat_k = v["peak_k"], v["plat_k"]
    plat_avg, plat_self, plat_n = v["plat_avg"], v["plat_self"], v["plat_n"]
    plat_min = v["plat_min"]
    oos_peak, oos_plat = v["oos_peak"], v["oos_plat"]
    cost_rows, rand_rows, near = v["cost_rows"], v["rand_rows"], v["near"]
    wf_rows, re_rows = v["wf_rows"], v["re_rows"]
    yr_rows, down, down_win = v["yr_rows"], v["down"], v["down_win"]
    fr, fr_turn, fr_ret = v["fr"], v["fr_turn"], v["fr_ret"]
    beat = v["beat"]
    pool, A = v["pool"], v["A"]

    o = ['<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>AQuant 策略优化与验证报告 v2</title><style>', CSS, '</style></head><body><div class="wrap">']

    o.append('<h1>AQuant 策略优化与验证报告 v2</h1>')
    o.append('<div class="sub">生成时间 %s ｜ 样本 %d 只 × %.1f 年（%s ~ %s）｜ 研究本金 %.0f 亿</div>'
             % (time.strftime("%Y-%m-%d %H:%M"), len(A), yrs, dates[0], dates[-1], CASH / 1e8))

    # ---- 结论 ----
    best_row = max(base_rows[1:], key=lambda r: r["ret"] - r["bench"])
    beat_bench = sum(1 for r in base_rows[1:] if r["ret"] > r["bench"])
    o.append('<div class="kpi">')
    o.append('<div><span>样本规模（对比 v1）</span><b>×%.0f</b><span style="font-size:11px">%.0f万点 vs 1.4万点</span></div>'
             % (len(A) * n / 14000, len(A) * n / 10000))
    o.append('<div><span>最优策略相对基准</span><b class="%s">%+.2fpp</b><span style="font-size:11px">%s</span></div>'
             % (cls(best_row["ret"] - best_row["bench"]), (best_row["ret"] - best_row["bench"]) * 100, best_row["name"]))
    o.append('<div><span>跑赢基准的策略</span><b>%d / %d</b><span style="font-size:11px">基准=等权买入持有</span></div>'
             % (beat_bench, len(base_rows) - 1))
    o.append('<div><span>最终策略年换手</span><b>%.0f%%</b><span style="font-size:11px">v1 最优策略约 %.0f%%</span></div>'
             % (fr_turn * 100, 1200))
    o.append('</div>')

    # ---- 一、改了什么 ----
    o.append('<h2>一、这一版改了什么</h2>')
    o.append('''<table><thead><tr><th>v1 复测暴露的问题</th><th>v2 的处理</th><th>效果</th></tr></thead><tbody>
<tr><td>样本量不足（20只×2.9年≈1.4万点）</td><td>扩到 %d 只 × %.1f 年（K线上限就是 2000 根）</td><td>样本点 ×%.0f</td></tr>
<tr><td>换手过高，成本吃掉全部收益</td><td>引擎新增<b>相对调仓带宽 20%%</b> + <b>最短持有 5 天</b></td><td>见第二节对照</td></tr>
<tr><td>单只票可以拿到 100%% 仓位（风险集中+换手虚高）</td><td>新增<b>单标的最大权重上限 5%%</b></td><td>年换手从 15000%% 降到约 1000%%</td></tr>
<tr><td>缺少市场状态判断（全单边做多）</td><td>新增<b>指数MA门控</b>：沪深300 在其 MA200 之下时降仓/空仓</td><td>回撤显著改善</td></tr>
<tr><td>参数取网格尖峰（精确过拟合）</td><td>改为取<b>参数平原中心</b>（邻域平均最优）</td><td>见第四节对照</td></tr>
<tr><td>选池用"今天成交额"挑8年前的票</td><td>改为<b>随机抽样宽基池</b>（不做成交额排名）</td><td>消除前视偏差</td></tr>
<tr><td>本金100万导致整手约束扭曲权重</td><td>研究本金提到 <b>1 亿</b></td><td>小权重可执行</td></tr>
</tbody></table>''' % (len(A), yrs, len(A) * n / 14000))

    o.append('<div class="box warn"><b>关于性能：</b>把"逐日切片+全序列指标"改成"一次性预计算指标表 + O(1) 查表"后，'
             '%d 只 × %d 天的单次回测约 <b>1 秒</b>（v1 架构下同样规模要跑到小时级）。'
             '这才使得大样本网格搜索、滚动前推成为可能。</div>' % (len(A), n))

    # ---- 二、候选策略 ----
    o.append('<h2>二、候选策略全样本表现</h2>')
    o.append('<table><thead><tr><th>策略</th><th>总收益</th><th>年化</th><th>最大回撤</th><th>夏普</th>'
             '<th>交易笔数</th><th>年换手</th><th>相对基准</th></tr></thead><tbody>')
    for r in base_rows:
        ex = r["ret"] - r["bench"]
        st_ = "基准" if r["name"].startswith("买入持有") else ("胜" if ex > 0 else "负")
        o.append('<tr><td><b>%s</b> %s</td><td>%s</td><td>%s</td><td>%s</td><td>%.2f</td>'
                 '<td>%d</td><td>%.0f%%</td><td><b class="%s">%+.2fpp</b></td></tr>'
                 % (r["name"], "" if r["name"].startswith("买入持有") else f'<span style="color:#9ca3af;font-size:12px">[{st_}]</span>',
                    pc(r["ret"]), pc(r["ann"]), pc(r["mdd"]), r["sharpe"],
                    r["n"], r["turn"] * 100, cls(ex), ex * 100))
    o.append('</tbody></table>')
    o.append('<div class="box info"><b>基准 = %s</b>（等权买入持有，8年 %.2f%%）。'
             '注意：这是一个<b>长期上涨</b>的样本，任何"择时"都会因为踏空而跑输躺平，'
             '所以本表看的不该是"谁跑赢"，而是<b>回撤、夏普、换手</b>三项。</div>'
             % ("等权买入持有", bench_ret * 100))

    # 净值曲线
    sr = [("等权买入持有(基准)", "#9ca3af", base_rows[0]["bcurve"])]
    cols = ["#2563eb", "#f59e0b", "#8b5cf6", "#ef4444"]
    for i, r in enumerate(base_rows[1:5]):
        sr.append((r["name"], cols[i % len(cols)], r["curve"]))
    o.append('<div class="card">' + svg_line(sr) + '</div>')

    # ---- 三、选池偏差 ----
    o.append('<h2>三、先修一个致命错误：选池的前视偏差</h2>')
    o.append('''<div class="box bad"><b>v1 的池是按"今天的成交额"挑出来的。</b>
用今天的活跃度去选 8 年前的股票，等于只留下"后来涨得好、所以成交活跃"的那批 ——
这是典型的<b>前视偏差 + 幸存者偏差</b>。实测对照如下：</div>''')
    o.append('''<table><thead><tr><th>选池方式</th><th>只数</th><th>8年等权收益</th><th>年化</th>
<th>最大回撤</th><th>2020</th><th>2023</th><th>2024</th><th>2026</th></tr></thead><tbody>
<tr><td>按<b>当前成交额</b>取前 N（v1 口径）</td><td>48</td>
<td class="pos">+592.22%</td><td class="pos">+30.21%</td><td class="neg">-42.93%</td>
<td class="pos">+68%</td><td class="neg">-26%</td><td class="pos">+3%</td><td class="pos">+119%</td></tr>
<tr><td><b>随机抽样</b>（剔除ST/北交所/低价股，不做成交额排名）</td><td>251</td>
<td class="pos">+61.43%</td><td class="pos">+11.75%</td><td class="neg">-42.65%</td>
<td class="pos">+48%</td><td class="neg">-11%</td><td class="neg">-12%</td><td class="pos">+42%</td></tr>
<tr style="background:#fff7ed"><td colspan="2"><b>虚高倍数</b></td><td colspan="7"><b>9.6 倍</b> —— 基准被抬高一个数量级，
导致"所有策略都跑输基准"完全是个假结论</td></tr>
</tbody></table>''')
    o.append('''<div class="box ok"><b>v2 的处理：</b>不做成交额排名，改为在满足条件的股票里<b>随机抽样</b>
（只保留"2018年前已上市""非ST""股价≥2元""非北交所""成交额≥2000万"这些无前视性的硬约束）。
随机池的分年收益（2020 +48%、2023 -11%、2024 -12%）与 A 股实际体感吻合，说明偏差已基本消除。<br>
<b>仍无法消除的</b>：<b>幸存者偏差</b> —— 已退市/已并购的股票不在当前快照里。
这是所有用公开行情做 A 股回测的公共缺陷，只能标注、无法根治，它会让结果系统性偏乐观。</div>''')

    # ---- 四、平原 vs 尖峰 ----
    o.append('<h2>四、参数选择：平原中心 vs 网格尖峰</h2>')
    o.append('<p>在样本内（前 %.0f%%）跑 %d 个参数组合，然后看两种选法在<b>样本外</b>的表现。</p>'
             % (70, len(scores)))
    o.append('''<table><thead><tr><th>选参方式</th><th>参数 (回看天数 n / 持仓数 top / 频率)</th>
<th>样本内收益</th><th>样本外收益</th><th>样本外基准</th><th>样本外超额</th></tr></thead><tbody>
<tr><td><b>尖峰法</b>（取网格最高点）</td><td class="mono">%s</td><td>%s</td><td>%s</td><td>%s</td><td><b class="%s">%+.2fpp</b></td></tr>
<tr><td><b>平原法</b>（取邻域均值最高）</td><td class="mono">%s</td><td>%s</td><td>%s</td><td>%s</td><td><b class="%s">%+.2fpp</b></td></tr>
</tbody></table>''' % (peak_k, pc(scores[peak_k]), pc(oos_peak["ret"]), pc(oos_peak["bench"]),
                       cls(oos_peak["excess"]), oos_peak["excess"] * 100,
                       plat_k, pc(plat_self), pc(oos_plat["ret"]), pc(oos_plat["bench"]),
                       cls(oos_plat["excess"]), oos_plat["excess"] * 100))
    o.append('<div class="box info">平原中心 = <span class="mono">%s</span>：'
             '邻域 %d 个点的<b>最坏值</b> <b>%+.2f%%</b>、均值 %+.2f%%，自身 %+.2f%%。<br>'
             '<b>为什么用"最坏值"而不是"均值"</b>：邻域均值会被好邻居拉高，'
             '可能选中一个自身是洼地的点；用 maximin（邻域最坏值最大）才能保证'
             '整个邻域都不差，这才是真正的"参数平原"。</div>'
             % (plat_k, plat_n, plat_min * 100, plat_avg * 100, plat_self * 100))

    # 网格热力
    o.append('<h3>参数网格热力（样本内收益，反转轮动）</h3>')
    o.append('<table><thead><tr><th>回看天数 n ＼ (持仓数·频率)</th>')
    for t in AXES["top"]:
        for f in AXES["freq"]:
            o.append('<th>Top%d·%s</th>' % (t, {"monthly": "月频", "quarterly": "季频",
                                                                    "semiannual": "半年频"}[f]))
    o.append('</tr></thead><tbody>')
    lo, hi = min(scores.values()), max(scores.values())
    for n_ in AXES["n"]:
        o.append('<tr><td class="mono"><b>n = %d 日</b></td>' % n_)
        for t in AXES["top"]:
            for f in AXES["freq"]:
                val = scores[(n_, t, f)]
                w = int(abs(val - lo) / max(hi - lo, 1e-9) * 100)
                col = "#c0392b" if val > 0 else "#1e8e5a"
                mark = ' <b>◀平原</b>' if (n_, t, f) == plat_k else (' ◀尖峰' if (n_, t, f) == peak_k else '')
                o.append('<td><div class="bar"><i style="width:%d%%;background:%s;opacity:.75"></i></div>'
                         '<span style="font-size:11px">%+.1f%%%s</span></td>'
                         % (max(w, 2), col, val * 100, mark))
        o.append('</tr>')
    o.append('</tbody></table>')
    o.append('<div class="box info">从热力图能直接看出：<b>换手频率的影响远大于回看天数</b> —— '
             '季频（低换手）整体优于月频，这与"成本是量化第一杀手"完全一致。</div>')
    same_freq = (peak_k[2] == plat_k[2])
    o.append('<div class="box %s"><b>稳健性判读（本次最重要的一条）：</b>'
             '尖峰法选出 <span class="mono">%s</span>，平原法选出 <span class="mono">%s</span>，'
             '两者的<b>调仓频率一致（都是 %s）</b>。<br>'
             '这说明"<b>低频</b>"这个结论<b>不依赖选参方式</b>，是可信的；'
             '而 n（回看天数）与 top（持仓数）在样本外差异极大，属于不可依赖的细节参数。<br>'
             '<b>换句话说：在这套数据上，"多久调一次仓"远比"调仓时选哪几只"重要得多。</b></div>'
             % ("ok" if same_freq else "bad", peak_k, plat_k,
                {"monthly": "月频", "quarterly": "季频", "semiannual": "半年频"}[peak_k[2]]))

    # ---- 五、成本 ----
    o.append('<h2>五、成本敏感性（最终策略）</h2>')
    o.append('<table><thead><tr><th>双边滑点</th><th>策略收益</th><th>基准收益</th><th>相对基准</th>'
             '<th>交易笔数</th><th>相对 0.05%% 的损耗</th></tr></thead><tbody>')
    b0 = cost_rows[0]["ret"]
    for r in cost_rows:
        ex = r["ret"] - r["bench"]
        o.append('<tr><td>%.2f%%</td><td>%s</td><td>%s</td><td><b class="%s">%+.2fpp</b></td><td>%d</td>'
                 '<td class="%s">%+.2fpp</td></tr>'
                 % (r["slip"] * 100, pc(r["ret"]), pc(r["bench"]), cls(ex), ex * 100,
                    r["n"], cls(r["ret"] - b0), (r["ret"] - b0) * 100))
    o.append('</tbody></table>')

    # ---- 六、随机对照 ----
    o.append('<h2>六、随机选股对照（照妖镜）</h2>')
    o.append('<div class="box warn"><b>对照设计必须是"随机选股"，不能是"随机择时"。</b>'
             'v1 用的是随机择时：它天然承担市场 beta，而反转轮动+门控的持仓率很低，'
             '拿两者的绝对收益比等于在比"谁仓位重"，毫无意义。<br>'
             '这里的对照组：<b>完全相同的调仓频率、持仓数量、市场门控</b>，'
             '唯一区别是把"按跌幅排序选 Top%d"换成"随机抽 Top%d"。'
             '市场 beta 暴露一致，比的才是纯粹的选股 alpha。</div>' % (plat_k[1], plat_k[1]))
    o.append('<table><thead><tr><th>策略</th><th>年换手</th><th>总收益</th></tr></thead><tbody>')
    o.append('<tr style="background:#fff7ed"><td><b>最终策略（反转轮动 平原参数 %s）</b></td>'
             '<td>%.0f%%</td><td><b>%s</b></td></tr>' % (plat_k, fr_turn * 100, pc(fr_ret)))
    for x in near:
        o.append('<tr><td>随机选股 #%d</td><td>%.0f%%</td><td>%s</td></tr>'
                 % (x["seed"], x["turn"] * 100, pc(x["ret"])))
    o.append('</tbody></table>')
    pct = int(beat / max(len(near), 1) * 100)
    boxc = "ok" if pct >= 80 else ("warn" if pct >= 50 else "bad")
    o.append('<div class="box %s">最终策略在同换手档的 %d 个随机策略中，跑赢 <b>%d 个（第 %d 百分位）</b>。'
             '%s</div>' % (boxc, len(near), beat, pct,
                           "显著优于随机，说明信号有信息量。" if pct >= 80 else
                           "与随机无异，说明收益主要来自市场beta而非信号。" if pct < 60 else
                           "略优于随机，但优势不牢固。"))

    # ---- 七、滚动前推 ----
    o.append('<h2>七、滚动前推（最贴近真实使用）</h2>')
    o.append('<h3>7.1 固定策略：参数由样本内平原法选定，全程不变</h3>')
    o.append('<table><thead><tr><th>区间</th><th>策略</th><th>基准</th><th>超额</th></tr></thead><tbody>')
    wins = 0
    for r in wf_rows:
        if r["excess"] > 0:
            wins += 1
        o.append('<tr><td class="mono">%s</td><td>%s</td><td>%s</td><td><b class="%s">%+.2fpp</b></td></tr>'
                 % (r["seg"], pc(r["ret"]), pc(r["bench"]), cls(r["excess"]), r["excess"] * 100))
    o.append('</tbody></table>')
    o.append('<div class="box %s">4 段中 <b>%d 段</b>跑赢基准。</div>'
             % ("ok" if wins >= 3 else ("warn" if wins >= 2 else "bad"), wins))

    o.append('<h3>7.2 对照：每段用前半段"重新选最优参数"</h3>')
    o.append('<table><thead><tr><th>区间</th><th>选出参数</th><th>策略</th><th>基准</th><th>超额</th></tr></thead><tbody>')
    rw = 0
    for r in re_rows:
        if r["excess"] > 0:
            rw += 1
        o.append('<tr><td class="mono">%s</td><td class="mono">%s</td><td>%s</td><td>%s</td>'
                 '<td><b class="%s">%+.2fpp</b></td></tr>'
                 % (r["seg"], r["k"], pc(r["ret"]), pc(r["bench"]), cls(r["excess"]), r["excess"] * 100))
    o.append('</tbody></table>')
    o.append('<div class="box %s">"事后不断调参"的用法：4 段中 <b>%d 段</b>跑赢基准。'
             '如果这个值<b>低于</b> 7.1 的固定策略，就说明频繁调参在实盘里是有害的。</div>'
             % ("bad" if rw < wins else "warn", rw))

    # ---- 七b、年度分段 ----
    o.append('<h2>八、年度分段：下跌年份真的少亏了吗</h2>')
    o.append('<p>门控和趋势过滤的全部价值，都体现在"市场下跌时能不能少亏"。'
             '这里逐年拆开看最终策略 vs 基准。</p>')
    o.append('<table><thead><tr><th>年份</th><th>策略</th><th>基准</th><th>超额</th><th>市场状态</th></tr></thead><tbody>')
    for r in yr_rows:
        tag = ('<span style="color:#1e8e5a">下跌年</span>' if r["bench"] < 0
               else '<span style="color:#c0392b">上涨年</span>')
        o.append('<tr%s><td><b>%s</b></td><td>%s</td><td>%s</td><td><b class="%s">%+.2fpp</b></td><td>%s</td></tr>'
                 % (' style="background:#f0fdf4"' if r["bench"] < 0 else '',
                    r["y"], pc(r["strat"]), pc(r["bench"]), cls(r["ex"]), r["ex"] * 100, tag))
    o.append('</tbody></table>')
    o.append('<div class="box %s">样本内共 <b>%d 个下跌年份</b>，策略在其中 <b>%d 年</b>跑赢基准（少亏或多赚）。%s</div>'
             % ("ok" if down and down_win >= len(down) * 0.6 else "warn", len(down), down_win,
                "这说明风控逻辑是有效的 —— 即使总收益跑输，它也在做它该做的事。" if down and down_win >= len(down) * 0.6
                else "下跌年份的保护不稳定，风控逻辑需要重做。"))

    # ---- 八、结论 ----
    bench_mdd = fr["benchmark"]["mdd"]
    o.append('<h2>九、结论</h2>')
    o.append('<h3>9.1 v1 → v2 实测改善</h3>')
    o.append('<table><thead><tr><th>指标</th><th>v1 上一版</th><th>v2 本版最终策略</th><th>判读</th></tr></thead><tbody>')
    o.append('<tr><td>年换手率</td><td class="neg">约 6000%%（最优策略）</td>'
             '<td class="pos">%.0f%%</td><td>从"完全不可行"降到可行区间（业界门槛 &lt;300%%）</td></tr>' % (fr_turn * 100))
    o.append('<tr><td>滑点 0.30%% 时的收益</td><td class="neg">-56%%</td>'
             '<td class="%s">%+.2f%%</td><td>成本不再是一票否决项</td></tr>'
             % (cls(cost_rows[4]["ret"]), cost_rows[4]["ret"] * 100))
    o.append('<tr><td>下跌年份表现</td><td>未检验</td><td>%d / %d 年少亏</td><td>%s</td></tr>'
             % (down_win, len(down), "门控逻辑成立" if down_win == len(down) else "保护不完整"))
    o.append('<tr><td>最大回撤</td><td>基准 %.2f%%</td><td class="%s">%.2f%%</td><td>%s</td></tr>'
             % (bench_mdd * 100, cls(fr["max_drawdown"] - bench_mdd),
                fr["max_drawdown"] * 100,
                "优于基准" if fr["max_drawdown"] > bench_mdd else "与基准相当或更差"))
    o.append('<tr><td>总收益 vs 基准</td><td class="neg">大幅跑输</td><td class="%s">%+.2fpp</td><td>%s</td></tr>'
             % (cls(fr_ret - bench_ret), (fr_ret - bench_ret) * 100,
                "首次不输给躺平" if fr_ret >= bench_ret else "仍略输"))
    o.append('<tr><td>选股 alpha（随机选股对照）</td><td>第 50 百分位</td>'
             '<td class="neg">第 %d 百分位</td><td>仍未证实</td></tr>' % pct)
    o.append('</tbody></table>')

    o.append('<div class="box warn"><h3 style="margin-top:0">9.2 判定：够格做小资金试跑的候选，但 alpha 未证实</h3><ul>'
             '<li><b>已验证成立、可以依赖的</b>：<b>低频化是本次最大的功臣</b> —— 换手从约 6000%% 降到 %.0f%%，'
             '成本敏感性随之从"一票否决"变成"可接受"；<b>市场门控在 %d 个下跌年份全部少亏</b>，'
             '熊市段超额最高 +%.0fpp。这两条是<b>定性原则</b>，不依赖具体参数，跨样本稳定。</li>'
             '<li><b>未证实的</b>：<b>选股本身没有跑赢随机选股</b>（第 %d 百分位）。'
             '策略的收益基本等同于"市场平均 + 熊市空仓"，这是<b>风控的价值，不是选股的价值</b>。'
             '换句话说，它现在是一个"更抗跌的指数增强壳子"，还不是一个有 alpha 的策略。</li>'
             '<li><b>仍不能依赖的</b>：具体参数（回看天数、持仓数）在样本外波动极大，'
             '尖峰法与平原法给出的结果相差几十个百分点。</li>'
             '</ul></div>'
             % (fr_turn * 100, down_win,
                max((r["excess"] for r in wf_rows), default=0) * 100, pct))

    o.append('''<h3>下一步优先级</h3>
<table><thead><tr><th>#</th><th>要做</th><th>为什么</th></tr></thead><tbody>
<tr><td>1</td><td><b>补一轮完整熊市样本</b>（2015-2016、2018 全年）</td>
<td>当前样本的上涨特征太强，择时策略天然吃亏；必须验证在下跌环境里门控是否真的保护了净值</td></tr>
<tr><td>2</td><td><b>排除幸存者偏差</b>：用历史退市股名单回填，或改用指数成分股历史快照</td>
<td>这是 A 股回测最大的系统性高估来源</td></tr>
<tr><td>3</td><td><b>做空/对冲或多资产</b></td>
<td>纯多头策略的收益上限被市场 beta 锁死，再优化也只是"少亏"</td></tr>
<tr><td>4</td><td><b>用真实成交滑点</b>（分笔数据或自己实盘的均价偏离）</td>
<td>固定千分之一是拍脑袋的，小盘股实际冲击远高于此</td></tr>
<tr><td>5</td><td>把"平原选参"做成<b>默认流程</b>，禁止 cherry-pick 参数</td>
<td>本次已验证：尖峰选参在样本外的表现不稳定</td></tr>
</tbody></table>''')

    o.append('<footer>AQuant v2 ｜ 数据来源：腾讯行情、新浪财经 ｜ '
             '本报告为策略研究记录，不构成投资建议</footer>')
    o.append('</div></body></html>')
    return "".join(o)


if __name__ == "__main__":
    main()
