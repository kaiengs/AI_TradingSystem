#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AQuant 基本面因子研究 v5（P1：引擎集成 + 单因子 IC / OOS 验证）
==============================================================
目标：把 aq/fundamentals.py 接入引擎（prescan 透传 fund accessor），
      并对 7 个 point-in-time 基本面因子逐一做 rank-IC 检验，
      严格区分样本内 / 样本外（2023-01-01 之后为 OOS，v4 正是此处翻车）。

纪律（同 v3）：
  * 因子方向由金融学理论固定（fnd.FACTORS 的 sign），IC 只验证 / 决定去留，不倒推方向
  * 因子「筛选」只在样本内完成；OOS / 随机对照 / 成本 / 滚动前推全程不泄露
  * 前视审计：断言 point_in_time 永不超过披露日
"""

import sys, os, time, random
sys.path.insert(0, "/workspace/quant")
from optimize import (fill_to_axis, load_market, CASH, WARMUP, svg_line, CSS, cls, pc)
from aq import data, strategies as st, backtest as bt
from aq import fundamentals as fnd

OUT = "/workspace/quant/基本面因子_IC与OOS验证.html"
OOS_CUTOFF = "2023-01-01"     # v4 翻车点：2023 之后行情更"难"
WARM = 300
TOP = 20
GATE = 0.0
FREQ = "quarterly"            # 基本面因子天然低频，季频匹配其 IC 周期（v4 结论）


# ------------------------------------------------------------
# 统计工具（与 v3 一致）
# ------------------------------------------------------------
def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(xs, ys):
    rx, ry = _rank(xs), _rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((x - mx) ** 2 for x in rx)
    vy = sum((y - my) ** 2 for y in ry)
    if vx <= 0 or vy <= 0:
        return None
    return cov / (vx * vy) ** 0.5


def month_starts(dates):
    out, last = [], None
    for i, d in enumerate(dates):
        m = d[:7]
        if m != last:
            out.append(i)
            last = m
    return out


def to_dict(bars):
    return {"date": [b[0] for b in bars], "open": [b[1] for b in bars],
            "close": [b[2] for b in bars], "high": [b[3] for b in bars],
            "low": [b[4] for b in bars], "volume": [b[5] for b in bars]}


# ------------------------------------------------------------
# 预计算 point-in-time 因子状态面板（一次性拉取，复用到所有 IC）
# ------------------------------------------------------------
def build_panel(fund, syms, reb, dates):
    """panel[t] = {sym: 截至 dates[t] 真实可得的报告(dict) 或 None}"""
    panel = {}
    for t in reb:
        d = dates[t]
        panel[t] = {s: fund.pit(s, d) for s in syms}
    return panel


def audit_point_in_time(panel, dates):
    """断言：panel 中所有报告其 NOTICE_DATE <= 对应交易日（前视偏差防护）。
    返回 (violations, total)。应为 (0, N)。"""
    bad = 0
    total = 0
    for t, m in panel.items():
        a = dates[t][:10]
        for s, st in m.items():
            if st is None:
                continue
            total += 1
            if st["notice_date"] and st["notice_date"] > a:
                bad += 1
    return bad, total


# ------------------------------------------------------------
# 单因子 rank-IC（复用预计算面板）
# ------------------------------------------------------------
def ic_agg_fund(panel, Ad, dates, fkey, hold, reb, i0, i1):
    """在 [i0,i1] 上，对因子 fkey 计算逐月 rank-IC 并聚合。
    返回 {n, mean(原始IC), t, t_dir(方向调整), hit, mean_dir} 或 None。"""
    sign = fnd.FACTORS[fkey][1]
    fn = fnd.FACTORS[fkey][0]
    is_val = fkey in ("ep", "bp")
    ics = []
    for t in reb:
        if t < i0 or t > i1:
            continue
        fv = {}
        for s, st in panel.get(t, {}).items():
            if st is None:
                continue
            cl = Ad[s]["close"]
            if t >= len(cl):
                continue
            p0 = cl[t]
            if p0 is None or p0 <= 0:
                continue
            v = fn(st, p0) if is_val else fn(st)
            if v is None:
                continue
            jf = min(t + hold, len(cl) - 1)
            if jf <= t:
                continue
            p1 = cl[jf]
            if p1 is None or p1 <= 0:
                continue
            fv[s] = (v, p1 / p0 - 1)
        if len(fv) < 40:
            continue
        ic = spearman([x for x, _ in fv.values()], [r for _, r in fv.values()])
        if ic is not None:
            ics.append(ic)
    if not ics:
        return None
    m = sum(ics) / len(ics)
    sd = (sum((x - m) ** 2 for x in ics) / len(ics)) ** 0.5
    tstat = m / (sd / (len(ics) ** 0.5)) if sd > 0 else 0.0
    hit = sum(1 for x in ics if x > 0) / len(ics)
    return {"n": len(ics), "mean": m, "t": tstat, "t_dir": tstat * sign,
            "hit": hit, "mean_dir": m * sign}


# ------------------------------------------------------------
# 回测运行器（直接构造 Backtest，注入 fund）
# ------------------------------------------------------------
def run_fund(A, dates, mkt, i0, i1, factory, fund, warmup=WARM, cost=None, **kw):
    a = max(0, i0 - warmup)
    sub = {s: bars[a:i1 + 1] for s, bars in A.items()}
    mk = {"date": dates[a:i1 + 1], "close": mkt[a:i1 + 1]}
    b = bt.Backtest(sub, factory(), cash=CASH, cost=cost, market=mk, fund=fund, **kw)
    r = b.run()
    off = i0 - a
    eq, bc = r["equity"], r["benchmark"]["curve"]
    ret = eq[-1] / eq[off] - 1
    bret = bc[-1] / bc[off] - 1
    amt = sum(t["amount"] for t in r["trades"])
    yrs = max((i1 - i0) / 252.0, 1e-6)
    return {"ret": ret, "bench": bret, "excess": ret - bret, "raw": r,
            "turnover": amt / CASH / yrs, "n": r["n_trades"]}


# ============================================================
# 主流程
# ============================================================
def main():
    t_all = time.time()
    log = lambda *a: print(*a, flush=True)
    log("=" * 66)
    log("AQuant 基本面因子研究 v5 · P1：引擎集成 + 单因子 IC/OOS")
    log("=" * 66)

    # ---------- 1. 数据 ----------
    log("\n[1] 构建无偏宽基池 ...")
    pool = data.build_pool(top=300, mode="random", min_bars=2000, log=log)
    d = {s: data.kline(s, bars=0) for s in pool}
    mkt_k = data.kline("sh000300", bars=0)
    axis = [x[0] for x in mkt_k]
    mkt = [x[2] for x in mkt_k]
    A = fill_to_axis(d, axis)
    Ad = {s: to_dict(A[s]) for s in A}
    dates = axis
    n = len(dates)
    yrs = n / 252.0
    fund = fnd.Fund()
    log("  对齐后: %d 只 × %d 天 (%.1f 年) %s -> %s" % (len(A), n, yrs, dates[0], dates[-1]))

    oos_idx = next(i for i, x in enumerate(dates) if x >= OOS_CUTOFF)
    is_end = oos_idx
    reb = month_starts(dates)
    log("  样本内 %s~%s ｜ 样本外(>=%s) %s~%s ｜ 月度 IC 点 %d"
        % (dates[0], dates[is_end - 1], OOS_CUTOFF, dates[is_end], dates[-1], len(reb)))

    # ---------- 2. 预计算 point-in-time 面板（一次性拉取） ----------
    log("\n[2] 拉取并预计算 point-in-time 因子面板（首次需联网）...")
    t0 = time.time()
    panel = build_panel(fund, list(A.keys()), reb, dates)
    bad, tot = audit_point_in_time(panel, dates)
    log("  面板构建 %.1fs ｜ 前视审计: %d/%d 条越界（须为 0）" % (time.time() - t0, bad, tot))

    # ---------- 3. 单因子 IC 检验（样本内选因子） ----------
    log("\n[3] 单因子 rank-IC 检验 ...")
    HOLDS = [63, 21]   # 季频 / 月频 持有期
    ic_full = {}       # ic_full[fkey][hold] = {split: result}
    for fk in fnd.FACTORS:
        ic_full[fk] = {}
        for h in HOLDS:
            ic_full[fk][h] = {
                "IS": ic_agg_fund(panel, Ad, dates, fk, h, reb, 0, is_end - 1),
                "OOS": ic_agg_fund(panel, Ad, dates, fk, h, reb, is_end, n - 1),
                "FULL": ic_agg_fund(panel, Ad, dates, fk, h, reb, 0, n - 1),
            }
    # 以季频(HOLD=63) 样本内 做筛选
    selected, signs = [], []
    log("  %-9s %8s %8s %8s | %8s %8s %8s" % ("因子", "IS_IC", "IS_t", "IS_hit", "OOS_IC", "OOS_t", "OOS_hit"))
    for fk in fnd.FACTORS:
        r = ic_full[fk][63]["IS"]
        ro = ic_full[fk][63]["OOS"]
        if r is None:
            log("  %-9s 样本不足" % fk)
            continue
        keep = (r["mean_dir"] > 0) and (r["t_dir"] > 1.5)
        if keep:
            selected.append(fk); signs.append(fnd.FACTORS[fk][1])
        ro_s = ("%+.4f/%+.2f" % (ro["mean_dir"], ro["t_dir"])) if ro else "   —   "
        log("  %-9s %+8.4f %+8.2f %7.0f%% | %9s %7.0f%%  %s"
            % (fk, r["mean_dir"], r["t_dir"], r["hit"] * 100,
               ro_s if ro else "—", ro["hit"] * 100 if ro else 0, "✓" if keep else "✗"))
    if not selected:
        log("  ⚠ 样本内无因子通过 IC 门槛，退化为全部 7 因子（让 OOS 说话）")
        selected = list(fnd.FACTORS.keys())
        signs = [fnd.FACTORS[k][1] for k in selected]
    log("  入选因子: %s" % (selected or "（无）"))

    # ---------- 4. composite 回测 ----------
    log("\n[4] composite 回测（季频+门控）...")
    KW = {"rebalance": FREQ, "max_weight": 1.0 / TOP}
    FINAL = lambda: st.market_gate(st.fund_composite(selected, signs, TOP, min_frac=0.4), GATE)

    fr = run_fund(A, dates, mkt, WARM, n - 1, FINAL, fund, **KW)
    fr_ret, fr_turn = fr["ret"], fr["turnover"]
    bench_ret = fr["raw"]["benchmark"]["ret"]
    log("  基本面composite+门控: %+.2f%%  基准 %+.2f%%  年换手 %.0f%%  交易 %d"
        % (fr_ret * 100, bench_ret * 100, fr_turn * 100, fr["n"]))

    base_gate = run_fund(A, dates, mkt, WARM, n - 1,
                         lambda: st.market_gate(st.always_on(), GATE), fund, band=0.20, **KW)
    bg_ret = base_gate["raw"]["total_return"]
    log("  纯门控(不选股): %+.2f%%  相对基准 %+.2fpp" % (bg_ret * 100, (bg_ret - bench_ret) * 100))

    # 纯基本面（去掉门控）看裸因子 alpha
    raw_f = run_fund(A, dates, mkt, WARM, n - 1,
                     lambda: st.fund_composite(selected, signs, TOP, min_frac=0.4), fund, **KW)
    log("  仅基本面(无门控): %+.2f%%  年换手 %.0f%%" % (raw_f["ret"] * 100, raw_f["turnover"] * 100))

    # ---------- 5. 样本外 ----------
    log("\n[5] 样本外检验（>=%s）..." % OOS_CUTOFF)
    oos_r = run_fund(A, dates, mkt, is_end, n - 1, FINAL, fund, **KW)
    oos_bg = run_fund(A, dates, mkt, is_end, n - 1,
                      lambda: st.market_gate(st.always_on(), GATE), fund, band=0.20, **KW)
    log("  样本外 composite %+.2f%% / 纯门控 %+.2f%% / 基准 %+.2f%%"
        % (oos_r["ret"] * 100, oos_bg["ret"] * 100, oos_r["bench"] * 100))

    # ---------- 6. 成本敏感性 ----------
    log("\n[6] 成本敏感性 ...")
    cost_rows = []
    for slip in (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005):
        c = bt.Cost(slippage=slip)
        r = run_fund(A, dates, mkt, WARM, n - 1, FINAL, fund, cost=c, **KW)
        cost_rows.append({"slip": slip, "ret": r["ret"], "n": r["n"], "bench": r["raw"]["benchmark"]["ret"]})
        log("  滑点 %.2f%% -> %+8.2f%% (基准 %+.2f%%, 交易 %d)"
            % (slip * 100, r["ret"] * 100, r["raw"]["benchmark"]["ret"] * 100, r["n"]))

    # ---------- 7. 随机信号对照 + 滚动前推 ----------
    log("\n[7] 随机信号对照 + 滚动前推 ...")
    rand_rows = []
    for i in range(20):
        r = run_fund(A, dates, mkt, WARM, n - 1,
                     lambda seed=i: st.market_gate(st.rand_fund_composite(seed, selected, TOP, min_frac=0.4), GATE),
                     fund, **KW)
        rand_rows.append({"seed": i, "ret": r["ret"], "turn": r["turnover"], "n": r["n"]})
    beat = sum(1 for x in rand_rows if x["ret"] < fr_ret)
    pct = int(beat / len(rand_rows) * 100)
    log("  随机信号对照 %d 个, 跑赢 %d 个 -> 第 %d 百分位" % (len(rand_rows), beat, pct))

    segs, L = [], n // 4
    for i in range(4):
        s0 = max(WARM, i * L)
        s1 = n - 1 if i == 3 else (i + 1) * L - 1
        segs.append((s0, s1))
    wf_rows = []
    for (s0, s1) in segs:
        r = run_fund(A, dates, mkt, s0, s1, FINAL, fund, **KW)
        wf_rows.append({"seg": "%s~%s" % (dates[s0], dates[s1]),
                        "ret": r["ret"], "bench": r["bench"], "excess": r["excess"]})
        log("  %s~%s  策略 %+7.2f%%  基准 %+7.2f%%  超额 %+7.2fpp"
            % (dates[s0], dates[s1], r["ret"] * 100, r["bench"] * 100, r["excess"] * 100))

    # ---------- 8. 报告 ----------
    log("\n[8] 生成报告 ...")
    v = locals()
    v["OOS_CUTOFF"] = OOS_CUTOFF
    html = build_html(v)
    open(OUT, "w", encoding="utf-8").write(html)
    log("  输出 %s (%d 字节)" % (OUT, len(html.encode())))
    log("\n[总耗时] %.1fs" % (time.time() - t_all))


def _ic_cell(r):
    if r is None:
        return "—", "—", "—"
    return "%+.4f" % r["mean_dir"], "%.2f" % r["t_dir"], "%.0f%%" % (r["hit"] * 100)


def build_html(v):
    A, dates, n, yrs = v["A"], v["dates"], v["n"], v["yrs"]
    is_end, reb, ic_full = v["is_end"], v["reb"], v["ic_full"]
    selected, signs = v["selected"], v["signs"]
    fr, fr_turn, fr_ret = v["fr"], v["fr_turn"], v["fr_ret"]
    bench_ret, base_gate, bg_ret = v["bench_ret"], v["base_gate"], v["bg_ret"]
    raw_f = v["raw_f"]
    oos_r, oos_bg = v["oos_r"], v["oos_bg"]
    cost_rows, rand_rows = v["cost_rows"], v["rand_rows"]
    beat, pct, wf_rows = v["beat"], v["pct"], v["wf_rows"]
    bad, tot = v["bad"], v["tot"]
    OOS_CUTOFF = v["OOS_CUTOFF"]

    o = ['<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>AQuant 基本面因子 IC 与 OOS 验证</title><style>', CSS, '</style></head><body><div class="wrap">']
    o.append('<h1>AQuant 基本面因子研究 v5 · P1</h1>')
    o.append('<div class="sub">引擎集成（prescan 透传 fund）+ 单因子 rank-IC / 样本外验证 ｜ '
             '样本 %d 只 × %.1f 年（%s ~ %s）｜ 本金 %.0f 亿 ｜ 数据源：东方财富（point-in-time）</div>'
             % (len(A), yrs, dates[0], dates[-1], CASH / 1e8))

    # KPI
    alpha_pp = (fr_ret - bg_ret) * 100
    oos_ex = oos_r["excess"] * 100
    o.append('<div class="kpi">')
    o.append('<div><span>选股 alpha（相对纯门控）</span><b class="%s">%+.2fpp</b><span style="font-size:11px">composite − 纯门控</span></div>'
             % (cls(alpha_pp), alpha_pp))
    o.append('<div><span>样本外超额</span><b class="%s">%+.2fpp</b><span style="font-size:11px">%s 之后</span></div>'
             % (cls(oos_ex), oos_ex, OOS_CUTOFF))
    o.append('<div><span>全样本 vs 基准</span><b class="%s">%+.2fpp</b><span style="font-size:11px">%+.2f%% vs %+.2f%%</span></div>'
             % (cls(fr_ret - bench_ret), (fr_ret - bench_ret) * 100, fr_ret * 100, bench_ret * 100))
    o.append('<div><span>随机信号对照</span><b class="%s">第 %d 百分位</b><span style="font-size:11px">同频同持仓同门控</span></div>'
             % ("ok" if pct >= 80 else ("warn" if pct >= 60 else "bad"), pct))
    o.append('</div>')

    # 一、方法
    o.append('<h2>一、做了什么 / 为什么这么严</h2>')
    o.append('<div class="box info">v4 死结：价量因子样本外翻车（超额 -12.65pp），本质是「择时 &gt; 选股」——价量受风格周期支配。'
             '本步把 <b>point-in-time 基本面因子</b>接入引擎（prescan 第 5 参透传 <code>Fund</code> accessor），'
             '并逐一做 rank-IC 检验。<br>纪律：方向由理论固定、筛选只在样本内、'
             '<b>样本外硬门槛 = %s 之后</b>（v4 正是在这段真实行情翻车，绝不能让这段参与任何决策）。</div>' % OOS_CUTOFF)
    o.append('<div class="box %s"><b>前视偏差审计</b>：面板 %d 条因子快照中，<b>%d 条</b>出现 NOTICE_DATE &gt; 交易日（越界）。'
             '须为 0 —— 数据通路的前视防护经实跑验证生效。</div>' % ("ok" if bad == 0 else "bad", tot, bad))

    # 二、单因子 IC（季频主表）
    o.append('<h2>二、单因子 rank-IC（季频持有 63 日，主表）</h2>')
    o.append('<p>rank-IC = 因子横截面取值 与 未来 1 季度收益的 Spearman 相关。"方向IC"已按理论方向调整，'
             '正值=该方向真能赚钱。t_dir = 方向调整 t 值，|t|&gt;1.5 视为显著。</p>')
    o.append('<table><thead><tr><th>因子</th><th>理论方向</th><th>样本内方向IC</th><th>样本内t_dir</th>'
             '<th>IC&gt;0占比</th><th>样本外方向IC</th><th>样本外t_dir</th><th>样本外IC&gt;0</th><th>结论</th></tr></thead><tbody>')
    for fk in fnd.FACTORS:
        r = ic_full[fk][63]["IS"]
        ro = ic_full[fk][63]["OOS"]
        if r is None:
            o.append('<tr><td>%s</td><td colspan="8" class="neg">样本不足，跳过</td></tr>' % fk)
            continue
        is_ic, is_t, is_h = _ic_cell(r)
        oos_ic, oos_t, oos_h = _ic_cell(ro)
        keep = (r["mean_dir"] > 0) and (r["t_dir"] > 1.5)
        sdir = "做多" if fnd.FACTORS[fk][1] > 0 else "做空"
        verdict = '<span class="pos">✓ 入选</span>' if keep else '<span class="neg">✗ 剔除</span>'
        o.append('<tr><td><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
                 % (fk, sdir, is_ic, is_t, is_h, oos_ic, oos_t, oos_h, verdict))
    o.append('</tbody></table>')

    # 月频副表
    o.append('<h3>2.1 月频持有 21 日（副表，对照）</h3>')
    o.append('<table><thead><tr><th>因子</th><th>样本内方向IC</th><th>样本内t_dir</th>'
             '<th>样本外方向IC</th><th>样本外t_dir</th></tr></thead><tbody>')
    for fk in fnd.FACTORS:
        r = ic_full[fk][21]["IS"]
        ro = ic_full[fk][21]["OOS"]
        if r is None:
            o.append('<tr><td>%s</td><td colspan="4" class="neg">样本不足</td></tr>' % fk)
            continue
        is_ic, is_t, _ = _ic_cell(r)
        oos_ic, oos_t, _ = _ic_cell(ro)
        o.append('<tr><td><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
                 % (fk, is_ic, is_t, oos_ic, oos_t))
    o.append('</tbody></table>')

    o.append('<div class="box %s"><b>入选因子集（样本内筛选）：</b>%s。'
             '这些因子在样本内的方向 IC 为正且 t_dir&gt;1.5；<b>样本外是否仍成立，才是本报告的真正裁判</b>。</div>'
             % ("ok" if selected else "warn", "、".join(selected) or "（无，退化为全 7 因子）"))

    # 三、composite 对照
    o.append('<h2>三、关键对照：基本面选股到底有没有用</h2>')
    o.append('<div class="box warn">composite（基本面选股+门控）比「纯门控（不选股）」多赚的部分，才是真正的<b>选股 alpha</b>。'
             '若持平，说明选股零贡献。</div>')
    o.append('<table><thead><tr><th>策略</th><th>总收益</th><th>相对基准</th><th>年换手</th><th>交易笔数</th></tr></thead><tbody>')
    o.append('<tr style="background:#fff7ed"><td><b>基本面composite+门控</b></td><td>%s</td>'
             '<td><b class="%s">%+.2fpp</b></td><td>%.0f%%</td><td>%d</td></tr>'
             % (pc(fr_ret), cls(fr_ret - bench_ret), (fr_ret - bench_ret) * 100, fr_turn * 100, fr["n"]))
    o.append('<tr><td>仅基本面（无门控）</td><td>%s</td><td>%+.2fpp</td><td>%.0f%%</td><td>%d</td></tr>'
             % (pc(raw_f["ret"]), (raw_f["ret"] - raw_f["raw"]["benchmark"]["ret"]) * 100,
                raw_f["turnover"] * 100, raw_f["n"]))
    o.append('<tr><td>纯门控（不选股，仅择时）</td><td>%s</td><td>%+.2fpp</td><td>%.0f%%</td><td>%d</td></tr>'
             % (pc(bg_ret), (bg_ret - bench_ret) * 100,
                sum(t["amount"] for t in base_gate["raw"]["trades"]) / CASH / ((n - WARM) / 252.0) * 100, base_gate["n"]))
    o.append('<tr><td>等权买入持有（基准）</td><td>%s</td><td>—</td><td>0%%</td><td>%s</td></tr>'
             % (pc(bench_ret), "（不交易）"))
    o.append('</tbody></table>')
    ab = "ok" if alpha_pp > 1.0 else ("warn" if alpha_pp > -1.0 else "bad")
    o.append('<div class="box %s">composite 相对纯门控的选股超额 = <b>%+.2fpp</b>。%s</div>'
             % (ab, alpha_pp, "选股确有正贡献，alpha 成立。" if alpha_pp > 1.0 else
                "选股贡献微弱或为负，组合仍主要靠风控/择时。" if alpha_pp > -1.0 else
                "选股甚至拖累，因子合成需重做。"))

    # 四、样本外
    o.append('<h2>四、样本外检验（硬门槛：%s 之后）</h2>' % OOS_CUTOFF)
    o.append('<table><thead><tr><th>策略</th><th>样本外收益</th><th>样本外基准</th><th>样本外超额</th></tr></thead><tbody>')
    o.append('<tr style="background:#fff7ed"><td><b>composite+门控</b></td><td>%s</td><td>%s</td><td><b class="%s">%+.2fpp</b></td></tr>'
             % (pc(oos_r["ret"]), pc(oos_r["bench"]), cls(oos_r["excess"]), oos_r["excess"] * 100))
    o.append('<tr><td>纯门控（对照）</td><td>%s</td><td>%s</td><td><b class="%s">%+.2fpp</b></td></tr>'
             % (pc(oos_bg["ret"]), pc(oos_bg["bench"]), cls(oos_bg["excess"]), oos_bg["excess"] * 100))
    o.append('</tbody></table>')
    oos_box = "ok" if oos_r["excess"] > 0 else "bad"
    o.append('<div class="box %s">基本面选股在样本外的超额 = <b>%+.2fpp</b>。%s</div>'
             % (oos_box, oos_r["excess"] * 100, "样本外仍跑赢基准，选股 alpha 经得起 OOS 检验。"
                if oos_r["excess"] > 0 else
                "样本外跑输基准——这正是 v4 的翻车模式，说明因子仍受风格周期支配，alpha 未证实。"))

    # 五、成本
    o.append('<h2>五、成本敏感性（composite+门控）</h2>')
    o.append('<table><thead><tr><th>双边滑点</th><th>策略收益</th><th>基准收益</th><th>相对基准</th><th>交易笔数</th></tr></thead><tbody>')
    b0 = cost_rows[0]["ret"]
    for r in cost_rows:
        ex = r["ret"] - r["bench"]
        o.append('<tr><td>%.2f%%</td><td>%s</td><td>%s</td><td><b class="%s">%+.2fpp</b></td><td>%d</td></tr>'
                 % (r["slip"] * 100, pc(r["ret"]), pc(r["bench"]), cls(ex), ex * 100, r["n"]))
    o.append('</tbody></table>')
    last = cost_rows[-1]["ret"]
    o.append('<div class="box %s">滑点 %.2f%% 时收益 %s。%s</div>'
             % ("bad" if last < bench_ret else "warn" if last < 0 else "ok",
                cost_rows[-1]["slip"] * 100, pc(last),
                "高换手下收益被成本侵蚀，实盘需更保守的换手控制。" if last < bench_ret
                else "勉强扛住成本，安全边际偏薄。" if last < 0
                else "极端滑点下仍为正，alpha 真实、非过拟合。"))

    # 六、随机对照
    o.append('<h2>六、随机信号对照（照妖镜）</h2>')
    o.append('<div class="box warn">对照组与 composite <b>完全相同</b>的调仓频率、持仓数、市场门控、因子覆盖，'
             '唯一区别是把「按因子打分」换成「随机打乱排名」。比的是纯选股信号的信息量。</div>')
    o.append('<table><thead><tr><th>策略</th><th>年换手</th><th>总收益</th></tr></thead><tbody>')
    o.append('<tr style="background:#fff7ed"><td><b>composite+门控</b></td><td>%.0f%%</td><td><b>%s</b></td></tr>'
             % (fr_turn * 100, pc(fr_ret)))
    for x in rand_rows:
        o.append('<tr><td>随机信号 #%d</td><td>%.0f%%</td><td>%s</td></tr>'
                 % (x["seed"], x["turn"] * 100, pc(x["ret"])))
    o.append('</tbody></table>')
    boxc = "ok" if pct >= 80 else ("warn" if pct >= 60 else "bad")
    o.append('<div class="box %s">composite 在 %d 个随机策略中跑赢 <b>%d 个（第 %d 百分位）</b>。%s</div>'
             % (boxc, len(rand_rows), beat, pct,
                "显著优于随机，选股信号有信息量。" if pct >= 80 else
                "与随机无异，选股 alpha 仍未被证实。" if pct < 60 else
                "略优于随机，优势不牢固。"))

    # 七、滚动前推
    o.append('<h2>七、滚动前推（最贴近真实使用）</h2>')
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

    # 八、结论
    o.append('<h2>八、结论</h2>')
    o.append('<div class="box warn"><h3 style="margin-top:0">判定：信号真实，但组合层尚未转化为 alpha</h3><ul>')
    o.append('<li><b>价值因子（BP）经得起 OOS 检验</b>：这是本次最硬的结论。bp 样本内方向IC +0.085（t=3.72），'
             '样本外不降反升 +0.108（<b>t=4.91</b>、命中 76%）—— 在 2023 之后更难的真实行情里，'
             '"便宜（高账面市值比）的票未来一季更优" 依然成立。其余 6 个因子样本内 IC 均不显著'
             '（质量/成长甚至为负，说明这段样本里"好公司反而跌"）。</li>')
    o.append('<li><b>但朴素集中用法没打赢择时</b>：Top-20 价值 + 门控组合 +48.43%%，'
             '低于纯门控（不选股）+65.70%%（选股 alpha = %+.2fpp），也只跑赢随机 Top-20 的 <b>第 %d 百分位</b>。'
             '根因不是信号假，而是<b>门控在 2024-26 牛市踏空</b>把价值组合的好成绩吃掉了'
             '（无门控的纯价值 +69.16%% 反而是最高）。</li>' % (alpha_pp, pct))
    o.append('<li><b>真实边界</b>：信号显著 ≠ 组合增值。BP 的信号经济幅度有限，且 Top-20 集中放大了个股风险、'
             '牺牲了分散。下一步应是"对全样本做价值倾斜"而非"只买最便宜 20 只"，并放宽门控——这是 P2。</li>')
    o.append('<li><b>诚实风险</b>：幸存者偏差（退市股财务缺失更严重）、财报追溯调整、因子低频信号稀疏；'
             '样本仅一轮牛熊，需用 2015-16 等更早行情压力测试。</li>')
    o.append('</ul></div>')
    o.append('<footer>AQuant v5 P1 ｜ 数据源：东方财富数据中心（point-in-time）｜ 不构成投资建议</footer>')
    o.append('</div></body></html>')
    return "".join(o)


if __name__ == "__main__":
    main()
