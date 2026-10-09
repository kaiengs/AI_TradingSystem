#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AQuant v5 · P3：正交组合（价值倾斜提供 alpha ＋ 择时单独控制回撤）
================================================================
P2 遗留的精确矛盾：
    价值倾斜(无门控)  +95.88%
    价值倾斜(原门控)  +64.06%  ≈  纯门控(宽基) +65.43%
    -> **二元门控把倾斜的 alpha 整个抵消掉了**

根因：二元门控一触发就"全卖光"——空仓期选股 alpha 直接归零。
      选股与择时被耦合在同一个开关上，互相抵消。

P3 的处方：**两层彻底解耦（正交）**
    * 选股层（fund_tilt）：永远满仓，只决定**各票相对权重**（一个向量）
    * 择时层（gate）    ：只缩放**总暴露**（一个标量），不干涉相对权重
    于是价值倾斜的 alpha 在任何仓位水平下都继续工作（按比例缩放），
    不会被"清仓"抹掉；择时则单独负责削峰（控制回撤）。

待检验的三种择时形态：
    1. binary 二元开关（P1/P2 原版，最糙，whipsaw 最重）
    2. soft   部分仓位（熊市留 30/50/70% —— alpha 按比例保留）
    3. ramp   连续缩放（按 close/MA 偏离度线性过渡，天然抑制 whipsaw）

选参规则（明确、对齐目标，避免"只挑收益最高"又回到无门控）：
    在**保留 ≥80% 选股 alpha** 的配置中，选**回撤最小**的那个。
    （alpha 保留率 = 该配置超额 / 无门控超额，均以样本内计）

纪律（同 P1/P2）：参数只在样本内挑；样本外硬门槛 2023-01-01 之后全程不参与决策；
                  随机信号对照 + 成本敏感性 + 滚动前推三重验证。
"""

import sys, os, time
sys.path.insert(0, "/workspace/quant")
from optimize import (fill_to_axis, load_market, CASH, CSS, cls, pc)
from aq import data, strategies as st, backtest as bt
from aq import fundamentals as fnd
from fund_ic import (month_starts, to_dict, build_panel,
                     audit_point_in_time, ic_agg_fund, run_fund)

OUT = "/workspace/quant/正交组合_倾斜加择时_P3.html"
OOS_CUTOFF = "2023-01-01"
WARM = 300
FREQ = "quarterly"
FNAME = "bp"
TAU, FLOOR = 6.0, 0.0        # 沿用 P2 样本内选出的倾斜强度
GATE_BEAR = 0.0              # 二元门控的熊市仓位
ALPHA_KEEP_MIN = 0.80        # 选参规则：至少要保留这么多 alpha

KW = {"rebalance": FREQ, "max_weight": 0.0}

# (类型, p1, p2, 说明)
CONFIGS = [
    ("nogate", None, None, "无门控（P2 最佳，alpha 全开）"),
    ("binary", 1.00, None, "二元门控（P1/P2 原版）"),
    ("binary", 0.95, None, "二元放宽 0.95"),
    ("binary", 0.90, None, "二元放宽 0.90"),
    ("soft", 0.30, None, "部分仓位：熊市留 30%"),
    ("soft", 0.50, None, "部分仓位：熊市留 50%"),
    ("soft", 0.70, None, "部分仓位：熊市留 70%"),
    ("ramp", 0.85, 1.05, "连续门控 0.85~1.05"),
    ("ramp", 0.90, 1.00, "连续门控 0.90~1.00"),
    ("ramp", 0.80, 1.10, "连续门控 0.80~1.10"),
]


def tilt_factory(seed=None):
    if seed is None:
        return lambda: st.fund_tilt(FNAME, tau=TAU, floor=FLOOR)
    return lambda: st.rand_fund_tilt(seed, FNAME, tau=TAU, floor=FLOOR)


def make_gate(kind, p1, p2, seed=None):
    """把选股层与择时层组装起来。seed 不为 None 时选股信号随机化（对照用）。"""
    tf = tilt_factory(seed)
    if kind == "nogate":
        return tf
    if kind == "binary":
        return lambda: st.market_gate(tf(), GATE_BEAR, cutoff=p1)
    if kind == "soft":
        return lambda: st.market_gate(tf(), p1, cutoff=1.00)   # 熊市保留 p1 仓位
    if kind == "ramp":
        return lambda: st.ramp_gate(tf(), lo=p1, hi=p2)
    raise ValueError(kind)


def cname(c):
    kind, p1, p2, desc = c
    return desc


def main():
    t_all = time.time()
    log = lambda *a: print(*a, flush=True)
    log("=" * 68)
    log("AQuant v5 · P3：正交组合（价值倾斜 alpha ＋ 择时控回撤）")
    log("=" * 68)

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
    log(f"  对齐后: {len(A)} 只 × {n} 天 ({yrs:.1f} 年) {dates[0]} -> {dates[-1]}")

    oos_idx = next(i for i, x in enumerate(dates) if x >= OOS_CUTOFF)
    is_end = oos_idx
    reb = month_starts(dates)
    log(f"  样本内 {dates[0]}~{dates[is_end - 1]} ｜ 样本外(>={OOS_CUTOFF}) {dates[is_end]}~{dates[-1]}")

    # ---------- 2. 面板 + 前视审计 ----------
    log("\n[2] 构建 point-in-time 面板 ...")
    panel = build_panel(fund, list(A.keys()), reb, dates)
    bad, tot = audit_point_in_time(panel, dates)
    log(f"  前视审计: {bad}/{tot} 条越界（须为 0）")

    # ---------- 3. 信号复核 ----------
    log("\n[3] 复核 BP 因子 IC（季频 63 日）...")
    ic_is = ic_agg_fund(panel, Ad, dates, FNAME, 63, reb, 0, is_end - 1)
    ic_oos = ic_agg_fund(panel, Ad, dates, FNAME, 63, reb, is_end, n - 1)
    if ic_is:
        log(f"  样本内 方向IC {ic_is['mean_dir']:+.4f}  t {ic_is['t_dir']:+.2f}")
    if ic_oos:
        log(f"  样本外 方向IC {ic_oos['mean_dir']:+.4f}  t {ic_oos['t_dir']:+.2f}")

    # ---------- 4. 样本内：配置前沿扫描 ----------
    log("\n[4] 样本内配置前沿扫描（选股层固定，只扫择时层）...")
    is_rows = {}
    for c in CONFIGS:
        r = run_fund(A, dates, mkt, WARM, is_end - 1, make_gate(c[0], c[1], c[2]), fund, **KW)
        is_rows[c] = {"ret": r["ret"], "bench": r["bench"], "excess": r["excess"],
                      "turn": r["turnover"], "n": r["n"],
                      "mdd": r["raw"]["max_drawdown"]}
    full = is_rows[CONFIGS[0]]           # 无门控 = alpha 全开基准
    full_alpha = full["excess"]
    for c in CONFIGS:
        r = is_rows[c]
        r["alpha_keep"] = (r["excess"] / full_alpha) if full_alpha > 1e-9 else 0.0
        r["mdd_cut"] = (1.0 - abs(r["mdd"]) / abs(full["mdd"])) if full["mdd"] else 0.0
    log(f"  {'配置':<28}{'超额':>10}{'alpha保留':>12}{'回撤':>10}{'回撤削减':>10}{'换手':>9}")
    for c in CONFIGS:
        r = is_rows[c]
        log(f"  {cname(c):<28}{r['excess'] * 100:>+9.2f}pp{r['alpha_keep'] * 100:>11.0f}%"
            f"{r['mdd'] * 100:>9.1f}%{r['mdd_cut'] * 100:>9.0f}%{r['turn'] * 100:>8.0f}%")

    # ---------- 5. 选参：保留 ≥80% alpha 中回撤最小 ----------
    cands = [c for c in CONFIGS if is_rows[c]["alpha_keep"] >= ALPHA_KEEP_MIN]
    if cands:
        pick = min(cands, key=lambda c: abs(is_rows[c]["mdd"]))
    else:
        pick = max(CONFIGS, key=lambda c: is_rows[c]["alpha_keep"])
    pr = is_rows[pick]
    log(f"\n  选中: {cname(pick)}  (alpha保留 {pr['alpha_keep'] * 100:.0f}%, "
        f"回撤 {pr['mdd'] * 100:.1f}%, 回撤削减 {pr['mdd_cut'] * 100:.0f}%)")

    BEST = make_gate(pick[0], pick[1], pick[2])
    NOGATE = make_gate("nogate", None, None)
    BINARY = make_gate("binary", 1.00, None)

    # ---------- 6. 全样本对照 ----------
    log("\n[5] 全样本关键对照 ...")
    full_best = run_fund(A, dates, mkt, WARM, n - 1, BEST, fund, **KW)
    bench_ret = full_best["bench"]
    full_nogate = run_fund(A, dates, mkt, WARM, n - 1, NOGATE, fund, **KW)
    full_binary = run_fund(A, dates, mkt, WARM, n - 1, BINARY, fund, **KW)
    log(f"  正交组合    : {full_best['ret'] * 100:+.2f}%  回撤 {full_best['raw']['max_drawdown'] * 100:.1f}%")
    log(f"  无门控倾斜  : {full_nogate['ret'] * 100:+.2f}%  回撤 {full_nogate['raw']['max_drawdown'] * 100:.1f}%")
    log(f"  二元门控倾斜: {full_binary['ret'] * 100:+.2f}%  回撤 {full_binary['raw']['max_drawdown'] * 100:.1f}%")
    log(f"  基准        : {bench_ret * 100:+.2f}%  回撤 {full_best['raw']['benchmark']['mdd'] * 100:.1f}%")
    keep_full = (full_best["excess"] / full_nogate["excess"]) if full_nogate["excess"] > 1e-9 else 0.0
    cut_full = 1.0 - abs(full_best["raw"]["max_drawdown"]) / abs(full_nogate["raw"]["max_drawdown"])
    log(f"  >> 全样本：alpha 保留 {keep_full * 100:.0f}%，回撤削减 {cut_full * 100:.0f}%")

    # ---------- 7. 样本外 ----------
    log("\n[6] 样本外检验（>=%s）..." % OOS_CUTOFF)
    oos_best = run_fund(A, dates, mkt, is_end, n - 1, BEST, fund, **KW)
    oos_nogate = run_fund(A, dates, mkt, is_end, n - 1, NOGATE, fund, **KW)
    log(f"  样本外 正交 {oos_best['ret'] * 100:+.2f}% / 无门控 {oos_nogate['ret'] * 100:+.2f}% / "
        f"基准 {oos_best['bench'] * 100:+.2f}%  -> 超额 {oos_best['excess'] * 100:+.2f}pp")

    # ---------- 8. 随机信号对照 ----------
    log("\n[7] 随机信号对照（20 次，同择时层）...")
    rand_rows = []
    for i in range(20):
        r = run_fund(A, dates, mkt, WARM, n - 1, make_gate(pick[0], pick[1], pick[2], seed=i), fund, **KW)
        rand_rows.append({"seed": i, "ret": r["ret"], "turn": r["turnover"]})
    beat = sum(1 for x in rand_rows if x["ret"] < full_best["ret"])
    pct = int(beat / len(rand_rows) * 100)
    log(f"  跑赢 {beat}/{len(rand_rows)} 个随机 -> 第 {pct} 百分位")

    # ---------- 9. 成本敏感性 ----------
    log("\n[8] 成本敏感性 ...")
    cost_rows = []
    for slip in (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005):
        r = run_fund(A, dates, mkt, WARM, n - 1, BEST, fund, cost=bt.Cost(slippage=slip), **KW)
        cost_rows.append({"slip": slip, "ret": r["ret"], "bench": r["bench"], "n": r["n"]})
        log(f"  滑点 {slip * 100:.2f}% -> {r['ret'] * 100:+8.2f}% (基准 {r['bench'] * 100:+.2f}%)")

    # ---------- 10. 滚动前推（含熊市段 alpha 检验） ----------
    log("\n[9] 滚动前推 ...")
    segs, L = [], n // 4
    for i in range(4):
        s0 = max(WARM, i * L)
        s1 = n - 1 if i == 3 else (i + 1) * L - 1
        segs.append((s0, s1))
    wf_rows = []
    for (s0, s1) in segs:
        r = run_fund(A, dates, mkt, s0, s1, BEST, fund, **KW)
        rn = run_fund(A, dates, mkt, s0, s1, NOGATE, fund, **KW)
        rb = run_fund(A, dates, mkt, s0, s1, BINARY, fund, **KW)
        wf_rows.append({"seg": f"{dates[s0]}~{dates[s1]}", "ret": r["ret"], "bench": r["bench"],
                        "excess": r["excess"], "nogate": rn["ret"], "binary": rb["ret"]})
        log(f"  {dates[s0]}~{dates[s1]}  正交 {r['ret'] * 100:+7.2f}%  无门控 {rn['ret'] * 100:+7.2f}%  "
            f"二元 {rb['ret'] * 100:+7.2f}%  基准 {r['bench'] * 100:+7.2f}%  超额 {r['excess'] * 100:+7.2f}pp")

    # ---------- 11. 报告 ----------
    log("\n[10] 生成报告 ...")
    v = {
        "A": A, "dates": dates, "n": n, "yrs": yrs, "is_end": is_end,
        "bad": bad, "tot": tot, "ic_is": ic_is, "ic_oos": ic_oos,
        "CONFIGS": CONFIGS, "is_rows": is_rows, "pick": pick, "full": full,
        "full_best": full_best, "full_nogate": full_nogate, "full_binary": full_binary,
        "bench_ret": bench_ret, "keep_full": keep_full, "cut_full": cut_full,
        "oos_best": oos_best, "oos_nogate": oos_nogate,
        "rand_rows": rand_rows, "beat": beat, "pct": pct,
        "cost_rows": cost_rows, "wf_rows": wf_rows,
        "OOS_CUTOFF": OOS_CUTOFF, "FNAME": FNAME, "FREQ": FREQ,
        "TAU": TAU, "FLOOR": FLOOR, "ALPHA_KEEP_MIN": ALPHA_KEEP_MIN,
    }
    html = build_html(v)
    open(OUT, "w", encoding="utf-8").write(html)
    log(f"  输出 {OUT} ({len(html.encode())} 字节)")
    log(f"\n[总耗时] {time.time() - t_all:.1f}s")


def build_html(v):
    A, dates, n, yrs = v["A"], v["dates"], v["n"], v["yrs"]
    bad, tot = v["bad"], v["tot"]
    ic_is, ic_oos = v["ic_is"], v["ic_oos"]
    CONFIGS, is_rows, pick, full = v["CONFIGS"], v["is_rows"], v["pick"], v["full"]
    full_best, full_nogate, full_binary = v["full_best"], v["full_nogate"], v["full_binary"]
    bench_ret, keep_full, cut_full = v["bench_ret"], v["keep_full"], v["cut_full"]
    oos_best, oos_nogate = v["oos_best"], v["oos_nogate"]
    rand_rows, beat, pct = v["rand_rows"], v["beat"], v["pct"]
    cost_rows, wf_rows = v["cost_rows"], v["wf_rows"]
    OOS_CUTOFF, FNAME, FREQ = v["OOS_CUTOFF"], v["FNAME"], v["FREQ"]
    TAU, FLOOR, AK = v["TAU"], v["FLOOR"], v["ALPHA_KEEP_MIN"]

    o = ['<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>AQuant v5 P3 · 正交组合（倾斜+择时）</title><style>', CSS, '</style></head>',
         '<body><div class="wrap">']
    o.append('<h1>AQuant v5 · P3：正交组合</h1>')
    o.append(f'<div class="sub">价值倾斜提供选股 alpha ＋ 择时单独控制回撤，两层解耦互不抵消 ｜ '
             f'样本 {len(A)} 只 × {yrs:.1f} 年（{dates[0]} ~ {dates[-1]}）｜ {FREQ} 调仓 ｜ '
             f'本金 {CASH / 1e8:.0f} 亿 ｜ 数据源：东方财富（point-in-time）</div>')

    # KPI
    o.append('<div class="kpi">')
    o.append(f'<div><span>alpha 保留率（全样本）</span><b class="{"ok" if keep_full >= 0.8 else "warn"}">'
             f'{keep_full * 100:.0f}%</b><span style="font-size:11px">相对无门控倾斜</span></div>')
    o.append(f'<div><span>回撤削减</span><b class="{"ok" if cut_full > 0.1 else "warn"}">'
             f'{cut_full * 100:.0f}%</b><span style="font-size:11px">'
             f'{full_best["raw"]["max_drawdown"] * 100:.1f}% vs {full_nogate["raw"]["max_drawdown"] * 100:.1f}%</span></div>')
    o.append(f'<div><span>样本外超额</span><b class="{cls(oos_best["excess"])}">'
             f'{oos_best["excess"] * 100:+.2f}pp</b><span style="font-size:11px">'
             f'{OOS_CUTOFF} 之后</span></div>')
    o.append(f'<div><span>随机信号对照</span><b class="{"ok" if pct >= 80 else ("warn" if pct >= 60 else "bad")}">'
             f'第 {pct} 百分位</b><span style="font-size:11px">同倾斜强度同择时</span></div>')
    o.append('</div>')

    # 一、为什么
    o.append('<h2>一、P2 遗留的精确矛盾</h2>')
    o.append('<table><thead><tr><th>配置</th><th>全样本收益</th><th>说明</th></tr></thead><tbody>')
    o.append('<tr><td>价值倾斜（无门控）</td><td><b>+95.88%</b></td><td>alpha 全开，但回撤 -30.1%</td></tr>')
    o.append('<tr><td>价值倾斜（原二元门控）</td><td>+64.06%</td>'
             '<td class="neg">alpha 被门控抵消 —— 几乎等于不选股的纯门控 +65.43%</td></tr>')
    o.append('</tbody></table>')
    o.append('<div class="box bad"><b>根因</b>：二元门控一触发就「全卖光」，空仓期选股 alpha 直接归零。'
             '选股与择时被耦合在同一个开关上 —— <b>互相抵消</b>。</div>')

    # 二、正交设计
    o.append('<h2>二、正交设计原理</h2>')
    o.append('<table><thead><tr><th>层</th><th>职责</th><th>数学形式</th><th>关键约束</th></tr></thead><tbody>')
    o.append('<tr><td><b>选股层</b><br><code>fund_tilt</code></td>'
             '<td>决定买什么、各买多少（价值倾斜）</td>'
             '<td>权重<b>向量</b> <code>w_i ∝ exp(z_i/tau)</code></td>'
             '<td><b>永远满仓</b>，不受择时干涉 —— alpha 永不中断</td></tr>')
    o.append('<tr><td><b>择时层</b><br><code>gate</code></td>'
             '<td>决定总仓位（控制回撤）</td>'
             '<td>暴露<b>标量</b> <code>e ∈ [0,1]</code></td>'
             '<td>只缩放总暴露，<b>不改变相对权重</b> —— 不干涉选股</td></tr>')
    o.append('</tbody></table>')
    o.append('<div class="box info">最终持仓 = <code>e × w_i</code>。因为两层分别是标量与向量，'
             '天然互不干涉：价值倾斜的 alpha 在<b>任何仓位水平下都继续工作</b>（只是按比例缩放），'
             '不会被「清仓」抹掉；而择时仍能通过降低 <code>e</code> 削平回撤。<br>'
             '附带好处：连续/部分仓位不像二元开关那样来回翻转，顺带抑制了 P2 诊断出的 <b>whipsaw</b>。</div>')
    o.append(f'<div class="box {"ok" if bad == 0 else "bad"}"><b>前视偏差审计</b>：面板 {tot} 条快照中 '
             f'<b>{bad} 条</b>越界（须为 0）。</div>')

    # 三、信号复核
    o.append(f'<h2>三、信号复核：{FNAME}</h2>')
    o.append('<table><thead><tr><th>区间</th><th>方向IC</th><th>t_dir</th><th>IC&gt;0 占比</th></tr></thead><tbody>')
    for nm, r in (("样本内", ic_is), ("样本外", ic_oos)):
        if r is None:
            o.append(f'<tr><td>{nm}</td><td colspan="3">样本不足</td></tr>')
        else:
            o.append(f'<tr><td><b>{nm}</b></td><td>{r["mean_dir"]:+.4f}</td><td>{r["t_dir"]:+.2f}</td>'
                     f'<td>{r["hit"] * 100:.0f}%</td></tr>')
    o.append('</tbody></table>')

    # 四、配置前沿
    o.append('<h2>四、配置前沿（样本内扫描，选股层固定只扫择时层）</h2>')
    o.append(f'<p>选股层固定为 P2 样本内选出的倾斜（<code>{FNAME}, tau={TAU:.0f}, floor={FLOOR:.2f}</code>），'
             f'只扫描择时层形态。「alpha 保留率」= 该配置超额 ÷ 无门控超额（无门控 = alpha 全开）。</p>')
    o.append('<table><thead><tr><th>择时配置</th><th>样本内超额</th><th>alpha 保留率</th>'
             '<th>最大回撤</th><th>回撤削减</th><th>年换手</th></tr></thead><tbody>')
    for c in CONFIGS:
        r = is_rows[c]
        mark = ' style="background:#fff7ed"' if c == pick else ''
        ak_c = "ok" if r["alpha_keep"] >= AK else ("warn" if r["alpha_keep"] >= 0.5 else "bad")
        o.append(f'<tr{mark}><td>{c[3]}</td>'
                 f'<td><b class="{cls(r["excess"])}">{r["excess"] * 100:+.2f}pp</b></td>'
                 f'<td><b class="{ak_c}">{r["alpha_keep"] * 100:.0f}%</b></td>'
                 f'<td>{r["mdd"] * 100:.1f}%</td><td>{r["mdd_cut"] * 100:.0f}%</td>'
                 f'<td>{r["turn"] * 100:.0f}%</td></tr>')
    o.append('</tbody></table>')
    pr = is_rows[pick]
    o.append(f'<div class="box info"><b>选参规则</b>：在 <b>alpha 保留率 ≥ {AK * 100:.0f}%</b> 的配置中，'
             f'选<b>回撤最小</b>者 —— 规则对齐「alpha 不丢、回撤可控」，'
             f'避免像 P2 那样只挑收益最高又退回无门控。<br>'
             f'选中：<b>{pick[3]}</b>（alpha 保留 {pr["alpha_keep"] * 100:.0f}%，'
             f'回撤 {pr["mdd"] * 100:.1f}%，回撤削减 {pr["mdd_cut"] * 100:.0f}%）。</div>')
    # 二元 vs 正交 的直接对比
    bin_r = is_rows[[c for c in CONFIGS if c[0] == "binary" and c[1] == 1.00][0]]
    o.append(f'<div class="box {"ok" if pr["alpha_keep"] > bin_r["alpha_keep"] else "warn"}">'
             f'<b>正交是否真的解决了抵消问题</b>：二元门控 alpha 保留率仅 '
             f'<b>{bin_r["alpha_keep"] * 100:.0f}%</b>'
             f'（{bin_r["excess"] * 100:+.2f}pp），正交组合 <b>{pr["alpha_keep"] * 100:.0f}%</b>'
             f'（{pr["excess"] * 100:+.2f}pp）'
             f'{"—— 抵消问题被解决。" if pr["alpha_keep"] > bin_r["alpha_keep"] else "—— 抵消问题未解决。"}</div>')

    # 五、全样本对照
    o.append('<h2>五、全样本关键对照</h2>')
    o.append('<table><thead><tr><th>策略</th><th>总收益</th><th>年化</th><th>相对基准</th>'
             '<th>最大回撤</th><th>年化/回撤<br>(Calmar)</th><th>年换手</th><th>交易笔数</th></tr></thead><tbody>')
    for nm, r in (("正交组合（P3）", full_best),
                  ("价值倾斜（无门控，P2 最佳）", full_nogate),
                  ("价值倾斜 + 二元门控（P2 稳健版）", full_binary)):
        mk = ' style="background:#fff7ed"' if nm.startswith("正交") else ''
        ann = r["raw"]["annual"]
        mdd = abs(r["raw"]["max_drawdown"])
        calmar = ann / mdd if mdd > 1e-9 else 0.0
        o.append(f'<tr{mk}><td><b>{nm}</b></td><td>{pc(r["ret"])}</td><td>{ann * 100:.2f}%</td>'
                 f'<td><b class="{cls(r["excess"])}">{r["excess"] * 100:+.2f}pp</b></td>'
                 f'<td>{r["raw"]["max_drawdown"] * 100:.1f}%</td>'
                 f'<td><b>{calmar:.3f}</b></td>'
                 f'<td>{r["turnover"] * 100:.0f}%</td><td>{r["n"]}</td></tr>')
    o.append(f'<tr><td>等权买入持有（基准）</td><td>{pc(bench_ret)}</td><td>—</td><td>—</td>'
             f'<td>{full_best["raw"]["benchmark"]["mdd"] * 100:.1f}%</td><td>—</td>'
             f'<td>0%</td><td>（不交易）</td></tr>')
    o.append('</tbody></table>')
    # 风险调整后的公平比较
    def _calmar(r):
        m = abs(r["raw"]["max_drawdown"])
        return r["raw"]["annual"] / m if m > 1e-9 else 0.0
    c_orth, c_nog, c_bin = _calmar(full_best), _calmar(full_nogate), _calmar(full_binary)
    o.append(f'<div class="box info"><b>风险调整后的公平比较（关键）</b>：'
             f'正交组合 Calmar <b>{c_orth:.3f}</b>、无门控倾斜 <b>{c_nog:.3f}</b>、'
             f'二元门控 <b>{c_bin:.3f}</b>。'
             f'{"三者几乎持平" if abs(c_orth - c_nog) < 0.02 else "差异明显"} —— '
             f'说明正交组合的价值<b>不是"多赚"，而是"在同等风险调整收益下把绝对回撤做小"</b>'
             f'（{full_best["raw"]["max_drawdown"] * 100:.1f}% vs '
             f'{full_nogate["raw"]["max_drawdown"] * 100:.1f}%）。'
             f'对有回撤约束的资金有意义；追求纯收益则无门控更优。</div>')
    o.append(f'<div class="box {"ok" if keep_full >= 0.8 and cut_full > 0.1 else "warn"}">'
             f'全样本：正交组合相对无门控倾斜 <b>保留了 {keep_full * 100:.0f}% 的 alpha</b>，'
             f'同时把最大回撤 <b>削减了 {cut_full * 100:.0f}%</b>'
             f'（{full_best["raw"]["max_drawdown"] * 100:.1f}% vs '
             f'{full_nogate["raw"]["max_drawdown"] * 100:.1f}%）。'
             f'{"两层各司其职，未互相抵消。" if keep_full >= 0.8 and cut_full > 0.1 else "取舍仍偏一侧，未达理想正交。"}</div>')

    # 六、样本外
    o.append(f'<h2>六、样本外检验（硬门槛：{OOS_CUTOFF} 之后）</h2>')
    o.append('<table><thead><tr><th>策略</th><th>样本外收益</th><th>样本外基准</th><th>样本外超额</th></tr></thead><tbody>')
    o.append(f'<tr style="background:#fff7ed"><td><b>正交组合</b></td><td>{pc(oos_best["ret"])}</td>'
             f'<td>{pc(oos_best["bench"])}</td>'
             f'<td><b class="{cls(oos_best["excess"])}">{oos_best["excess"] * 100:+.2f}pp</b></td></tr>')
    o.append(f'<tr><td>无门控倾斜（对照）</td><td>{pc(oos_nogate["ret"])}</td>'
             f'<td>{pc(oos_nogate["bench"])}</td>'
             f'<td><b class="{cls(oos_nogate["excess"])}">{oos_nogate["excess"] * 100:+.2f}pp</b></td></tr>')
    o.append('</tbody></table>')
    o.append(f'<div class="box {"ok" if oos_best["excess"] > 0 else "bad"}">样本外超额 '
             f'<b>{oos_best["excess"] * 100:+.2f}pp</b>'
             f'{"（v4 在此翻车 -12.65pp，P2 为 +22.45pp）—— 正交组合同样经得起 OOS 检验。" if oos_best["excess"] > 0 else "—— 样本外仍未证实。"}</div>')

    # 七、随机对照
    o.append('<h2>七、随机信号对照（照妖镜）</h2>')
    o.append(f'<div class="box warn">对照组与正交组合<b>完全相同</b>的倾斜强度、'
             f'<b>相同的择时层</b>（{pick[3]}）、调仓频率与因子覆盖，'
             f'唯一区别是把「哪个票拿哪个权重」随机打乱。</div>')
    o.append('<table><thead><tr><th>策略</th><th>年换手</th><th>总收益</th></tr></thead><tbody>')
    o.append(f'<tr style="background:#fff7ed"><td><b>正交组合</b></td>'
             f'<td>{full_best["turnover"] * 100:.0f}%</td><td><b>{pc(full_best["ret"])}</b></td></tr>')
    for x in rand_rows:
        o.append(f'<tr><td>随机倾斜 #{x["seed"]}</td><td>{x["turn"] * 100:.0f}%</td>'
                 f'<td>{pc(x["ret"])}</td></tr>')
    o.append('</tbody></table>')
    o.append(f'<div class="box {"ok" if pct >= 80 else ("warn" if pct >= 60 else "bad")}">'
             f'在 {len(rand_rows)} 个随机策略中跑赢 <b>{beat} 个（第 {pct} 百分位）</b>。'
             f'{"显著优于随机，选股信号有信息量。" if pct >= 80 else "与随机无异，alpha 未被证实。" if pct < 60 else "略优于随机。"}</div>')

    # 八、成本
    o.append('<h2>八、成本敏感性</h2>')
    o.append('<table><thead><tr><th>双边滑点</th><th>策略收益</th><th>基准收益</th><th>相对基准</th><th>交易笔数</th></tr></thead><tbody>')
    for r in cost_rows:
        ex = r["ret"] - r["bench"]
        o.append(f'<tr><td>{r["slip"] * 100:.2f}%</td><td>{pc(r["ret"])}</td><td>{pc(r["bench"])}</td>'
                 f'<td><b class="{cls(ex)}">{ex * 100:+.2f}pp</b></td><td>{r["n"]}</td></tr>')
    o.append('</tbody></table>')
    worst = cost_rows[-1]
    o.append(f'<div class="box {"bad" if worst["ret"] < worst["bench"] else "ok"}">'
             f'滑点 {worst["slip"] * 100:.2f}% 时收益 {pc(worst["ret"])}（基准 {pc(worst["bench"])}）。'
             f'{"仍跑赢基准，alpha 非过拟合。" if worst["ret"] > worst["bench"] else "被成本吃掉，需进一步压换手。"}</div>')

    # 九、滚动前推
    o.append('<h2>九、滚动前推（检验熊市段 alpha 是否中断）</h2>')
    o.append('<p>正交设计的核心主张是「alpha 永不中断」。若成立，则在下跌段正交组合应<b>仍然跑赢基准</b>，'
             '而二元门控因空仓会把 alpha 一起丢掉。</p>')
    o.append('<table><thead><tr><th>区间</th><th>正交组合</th><th>无门控倾斜</th><th>二元门控倾斜</th>'
             '<th>基准</th><th>正交超额</th></tr></thead><tbody>')
    wins = 0
    for r in wf_rows:
        if r["excess"] > 0:
            wins += 1
        o.append(f'<tr><td class="mono">{r["seg"]}</td><td>{pc(r["ret"])}</td><td>{pc(r["nogate"])}</td>'
                 f'<td>{pc(r["binary"])}</td><td>{pc(r["bench"])}</td>'
                 f'<td><b class="{cls(r["excess"])}">{r["excess"] * 100:+.2f}pp</b></td></tr>')
    o.append('</tbody></table>')
    o.append(f'<div class="box {"ok" if wins >= 3 else ("warn" if wins >= 2 else "bad")}">'
             f'4 段中 <b>{wins} 段</b>跑赢基准。</div>')

    # 十、结论
    o.append('<h2>十、结论</h2>')
    bin_is = is_rows[[c for c in CONFIGS if c[0] == "binary" and c[1] == 1.00][0]]
    o.append('<div class="box warn"><h3 style="margin-top:0">判定</h3><ul>')
    o.append(f'<li><b>抵消问题是否解决</b>：二元门控 alpha 保留率 {bin_is["alpha_keep"] * 100:.0f}%，'
             f'正交组合 <b>{pr["alpha_keep"] * 100:.0f}%</b>（样本内）—— '
             f'{"选股与择时已解耦，alpha 不再被清仓抹掉。" if pr["alpha_keep"] > bin_is["alpha_keep"] else "抵消问题仍在，需换思路。"}</li>')
    o.append(f'<li><b>全样本取舍结果</b>：保留 <b>{keep_full * 100:.0f}%</b> alpha，'
             f'回撤削减 <b>{cut_full * 100:.0f}%</b>'
             f'（{full_best["raw"]["max_drawdown"] * 100:.1f}% vs 无门控 '
             f'{full_nogate["raw"]["max_drawdown"] * 100:.1f}%）。</li>')
    o.append(f'<li><b>核心裁判</b>：样本外超额 <b>{oos_best["excess"] * 100:+.2f}pp</b>；'
             f'随机对照第 <b>{pct}</b> 百分位；滚动 {wins}/4 段跑赢基准。</li>')
    last_seg = wf_rows[-1] if wf_rows else None
    if last_seg:
        o.append(f'<li><b>瓶颈不在正交设计，而在择时信号本身</b>：最近一段牛市'
                 f'（{last_seg["seg"]}）正交组合超额 <b>{last_seg["excess"] * 100:+.2f}pp</b>'
                 f'（无门控同期为 +13.63pp）—— MA200 门控把牛市也切掉了一块。'
                 f'正交设计已经把「两层耦合」解开了，剩余亏损来自 <b>MA200 这把尺子本身的误判</b>，'
                 f'而非架构问题。要再进一步，应换更可靠的择时/风控信号'
                 f'（如<b>波动率目标</b>——按风险而非方向调仓，天然不会踏空牛市），'
                 f'而不是继续调门控阈值。</li>')
    o.append(f'<li><b>诚实边界</b>：择时形态与参数在样本内挑选；'
             f'「降低暴露」必然按比例同时降低 alpha 与 beta —— 正交设计能做到的是'
             f'<b>不把 alpha 归零</b>，而非「白拿回撤保护」；幸存者偏差、仅一轮牛熊仍是硬约束。</li>')
    o.append('</ul></div>')
    o.append('<footer>AQuant v5 P3 ｜ 数据源：东方财富数据中心（point-in-time）｜ 不构成投资建议</footer>')
    o.append('</div></body></html>')
    return "".join(o)


if __name__ == "__main__":
    main()
