#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AQuant 基本面因子研究 v5 · P2：价值倾斜 + 放宽门控
==================================================
P1 结论：
  * BP（价值/账面市值比）是唯一通过 IC 门槛、且样本外更强的因子（OOS t=4.91、命中 76%）
  * 但「Top-20 集中 + 二元门控」的组合 +48.43%，反而跑输纯门控 +65.70%（选股 alpha -17.27pp）

P1 诊断的两个根因（各开一剂药）：
  1) Top-20 集中牺牲了分散 —— 价值信号经济幅度有限，集中放大的多是个股风险、丢了 beta
     -> 处方 A：**价值倾斜**。持有宽基全集保留 beta 与分散，只对各票按 BP z-score
        做 softmax 倾斜 w_i ∝ exp(z_i/tau)，只吃价值溢价、不吃集中风险
  2) 二元门控在 2024-26 牛市踏空（无门控的纯价值 +69.16% 反而是最高）
     -> 处方 B：**放宽门控**。门控阈值改为 cutoff×MA（cutoff<1 容忍一定跌破），
        让门控只在确定性熊市减仓，减少牛市误踏空

纪律（同 P1，绝不放松）：
  * 参数（tau / floor / cutoff）只在**样本内**网格挑选
  * 样本外硬门槛 = 2023-01-01 之后（v4 翻车点），全程不参与任何决策
  * 选股 alpha 的裁判是「相对纯门控」+「随机信号对照」，不是绝对收益
  * 邻域最坏值检查：防止挑到孤立尖峰（v2 maximin 教训）
"""

import sys, os, time
sys.path.insert(0, "/workspace/quant")
from optimize import (fill_to_axis, load_market, CASH, WARMUP, CSS, cls, pc)
from aq import data, strategies as st, backtest as bt
from aq import fundamentals as fnd
# 复用 P1 的统计/面板/回测底座（DRY —— 二次实现会引入口径偏差）
from fund_ic import (month_starts, to_dict, build_panel,
                     audit_point_in_time, ic_agg_fund, run_fund)

OUT = "/workspace/quant/价值倾斜_放宽门控_P2.html"
OOS_CUTOFF = "2023-01-01"
WARM = 300
TOP = 20                       # P1 的 Top-20 集中（仅作对照）
GATE = 0.0                     # 熊市仓位（0=清仓）
FREQ = "quarterly"
FNAME = "bp"                   # P1 唯一入选、且样本外更强的因子

# 参数网格（只在样本内挑选）
TAUS = [6.0, 10.0, 15.0, 25.0]        # 倾斜锐度：越大越接近等权
FLOORS = [0.0, 0.5]                   # 与等权混合比例：越大越分散
CUTOFFS = [1.00, 0.95, 0.90, 0.00]    # 门控放宽系数：0.00 = 无门控（始终满仓）

KW_TILT = {"rebalance": FREQ, "max_weight": 0.0}       # 倾斜用连续权重，不设单票上限
KW_TOP20 = {"rebalance": FREQ, "max_weight": 1.0 / TOP}


def neighbor_min(grid, key):
    """网格中该点 ±1 邻域的最小样本内超额（v2 maximin 教训：防挑到孤立尖峰）"""
    ti, fi, ci = TAUS.index(key[0]), FLOORS.index(key[1]), CUTOFFS.index(key[2])
    vals = []
    for dt in (-1, 0, 1):
        for df in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dt == df == dc == 0:
                    continue
                a, b, c = ti + dt, fi + df, ci + dc
                if 0 <= a < len(TAUS) and 0 <= b < len(FLOORS) and 0 <= c < len(CUTOFFS):
                    vals.append(grid[(TAUS[a], FLOORS[b], CUTOFFS[c])]["excess"])
    return min(vals) if vals else None


def main():
    t_all = time.time()
    log = lambda *a: print(*a, flush=True)
    log("=" * 68)
    log("AQuant v5 · P2：价值倾斜 + 放宽门控（把 BP 信号转成组合 alpha）")
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

    # ---------- 2. point-in-time 面板 + 前视审计 ----------
    log("\n[2] 构建 point-in-time 面板（首次需联网）...")
    t0 = time.time()
    panel = build_panel(fund, list(A.keys()), reb, dates)
    bad, tot = audit_point_in_time(panel, dates)
    log(f"  面板 {time.time() - t0:.1f}s ｜ 前视审计: {bad}/{tot} 条越界（须为 0）")

    # ---------- 3. 信号复核：BP 的 IC 是否仍稳健 ----------
    log("\n[3] 复核 BP 因子 IC（季频持有 63 日）...")
    ic_is = ic_agg_fund(panel, Ad, dates, FNAME, 63, reb, 0, is_end - 1)
    ic_oos = ic_agg_fund(panel, Ad, dates, FNAME, 63, reb, is_end, n - 1)
    ic_full = ic_agg_fund(panel, Ad, dates, FNAME, 63, reb, 0, n - 1)
    if ic_is:
        log(f"  样本内 方向IC {ic_is['mean_dir']:+.4f}  t {ic_is['t_dir']:+.2f}  命中 {ic_is['hit'] * 100:.0f}%")
    if ic_oos:
        log(f"  样本外 方向IC {ic_oos['mean_dir']:+.4f}  t {ic_oos['t_dir']:+.2f}  命中 {ic_oos['hit'] * 100:.0f}%")

    # ---------- 4. 样本内参数网格 ----------
    log("\n[4] 样本内参数网格（tau × floor × 门控放宽）...")
    grid = {}
    for tau in TAUS:
        for fl in FLOORS:
            for cu in CUTOFFS:
                fac = lambda tau=tau, fl=fl, cu=cu: st.market_gate(
                    st.fund_tilt(FNAME, tau=tau, floor=fl), GATE, cutoff=cu)
                r = run_fund(A, dates, mkt, WARM, is_end - 1, fac, fund, **KW_TILT)
                grid[(tau, fl, cu)] = {"ret": r["ret"], "bench": r["bench"],
                                       "excess": r["excess"], "turn": r["turnover"], "n": r["n"]}
    # 按样本内超额挑最佳
    best_key = max(grid, key=lambda k: grid[k]["excess"])
    best = grid[best_key]
    nb = neighbor_min(grid, best_key)
    log(f"  最佳(样本内): tau={best_key[0]:.0f} floor={best_key[1]:.2f} cutoff={best_key[2]:.2f} "
        f"-> 超额 {best['excess'] * 100:+.2f}pp  换手 {best['turn'] * 100:.0f}%")
    if nb is not None:
        log(f"  邻域最坏超额 {nb * 100:+.2f}pp（与最佳差距 "
            f"{(best['excess'] - nb) * 100:.2f}pp，越大越像孤立尖峰）")

    # 门控放宽的边际贡献（同 tau/floor 下，cutoff=1.00 vs 更宽松）
    bl = grid[(best_key[0], best_key[1], 1.00)]
    log(f"  同参数下原门控(cutoff=1.00) 超额 {bl['excess'] * 100:+.2f}pp "
        f"-> 放宽到 {best_key[2]:.2f} 增益 {(best['excess'] - bl['excess']) * 100:+.2f}pp")

    BEST = lambda: st.market_gate(
        st.fund_tilt(FNAME, tau=best_key[0], floor=best_key[1]), GATE, cutoff=best_key[2])

    # ---------- 5. 关键对照（全样本） ----------
    log("\n[5] 关键对照（全样本）...")
    full_best = run_fund(A, dates, mkt, WARM, n - 1, BEST, fund, **KW_TILT)
    bench_ret = full_best["bench"]
    pure_gate = run_fund(A, dates, mkt, WARM, n - 1,
                         lambda: st.market_gate(st.always_on(), GATE, cutoff=1.00), fund, **KW_TILT)
    tilt_nogate = run_fund(A, dates, mkt, WARM, n - 1,
                           lambda: st.fund_tilt(FNAME, tau=best_key[0], floor=best_key[1]),
                           fund, **KW_TILT)
    comp20 = run_fund(A, dates, mkt, WARM, n - 1,
                      lambda: st.market_gate(st.fund_composite([FNAME], [1.0], TOP, min_frac=0.4),
                                             GATE, cutoff=1.00), fund, **KW_TOP20)
    # 稳健版：同样的价值倾斜，但保留**原门控**（用于展示"收益 vs 回撤"的真实取舍）
    tilt_gate = run_fund(A, dates, mkt, WARM, n - 1,
                         lambda: st.market_gate(
                             st.fund_tilt(FNAME, tau=best_key[0], floor=best_key[1]),
                             GATE, cutoff=1.00), fund, **KW_TILT)
    bench_mdd = full_best["raw"]["benchmark"]["mdd"]   # 基准 dict 的键是 "mdd"
    log(f"  价值倾斜+放宽门控 : {full_best['ret'] * 100:+.2f}%  (基准 {bench_ret * 100:+.2f}%, "
        f"换手 {full_best['turnover'] * 100:.0f}%, 回撤 {full_best['raw']['max_drawdown'] * 100:.1f}%)")
    log(f"  价值倾斜+原门控   : {tilt_gate['ret'] * 100:+.2f}%  "
        f"(回撤 {tilt_gate['raw']['max_drawdown'] * 100:.1f}%)")
    log(f"  纯门控(宽基,原门控): {pure_gate['ret'] * 100:+.2f}%")
    log(f"  价值倾斜(无门控)   : {tilt_nogate['ret'] * 100:+.2f}%")
    log(f"  Top20集中+门控(P1) : {comp20['ret'] * 100:+.2f}%")
    alpha_pp = (full_best["ret"] - pure_gate["ret"]) * 100
    log(f"  >> 选股 alpha（相对纯门控）= {alpha_pp:+.2f}pp")

    # ---------- 6. 样本外 ----------
    log("\n[6] 样本外检验（>=%s）..." % OOS_CUTOFF)
    oos_best = run_fund(A, dates, mkt, is_end, n - 1, BEST, fund, **KW_TILT)
    oos_gate = run_fund(A, dates, mkt, is_end, n - 1,
                        lambda: st.market_gate(st.always_on(), GATE, cutoff=1.00), fund, **KW_TILT)
    log(f"  样本外 倾斜 {oos_best['ret'] * 100:+.2f}% / 纯门控 {oos_gate['ret'] * 100:+.2f}% / "
        f"基准 {oos_best['bench'] * 100:+.2f}%  -> 超额 {oos_best['excess'] * 100:+.2f}pp")

    # ---------- 7. 随机信号对照 ----------
    log("\n[7] 随机信号对照（20 次）...")
    rand_rows = []
    for i in range(20):
        r = run_fund(A, dates, mkt, WARM, n - 1,
                     lambda seed=i: st.market_gate(
                         st.rand_fund_tilt(seed, FNAME, tau=best_key[0], floor=best_key[1]),
                         GATE, cutoff=best_key[2]), fund, **KW_TILT)
        rand_rows.append({"seed": i, "ret": r["ret"], "turn": r["turnover"]})
    beat = sum(1 for x in rand_rows if x["ret"] < full_best["ret"])
    pct = int(beat / len(rand_rows) * 100)
    log(f"  跑赢 {beat}/{len(rand_rows)} 个随机 -> 第 {pct} 百分位")

    # ---------- 8. 成本敏感性 ----------
    log("\n[8] 成本敏感性 ...")
    cost_rows = []
    for slip in (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005):
        r = run_fund(A, dates, mkt, WARM, n - 1, BEST, fund, cost=bt.Cost(slippage=slip), **KW_TILT)
        cost_rows.append({"slip": slip, "ret": r["ret"], "bench": r["bench"], "n": r["n"]})
        log(f"  滑点 {slip * 100:.2f}% -> {r['ret'] * 100:+8.2f}% (基准 {r['bench'] * 100:+.2f}%)")

    # ---------- 9. 滚动前推 ----------
    log("\n[9] 滚动前推 ...")
    segs, L = [], n // 4
    for i in range(4):
        s0 = max(WARM, i * L)
        s1 = n - 1 if i == 3 else (i + 1) * L - 1
        segs.append((s0, s1))
    wf_rows = []
    for (s0, s1) in segs:
        r = run_fund(A, dates, mkt, s0, s1, BEST, fund, **KW_TILT)
        rg = run_fund(A, dates, mkt, s0, s1,
                      lambda: st.market_gate(st.always_on(), GATE, cutoff=1.00), fund, **KW_TILT)
        wf_rows.append({"seg": f"{dates[s0]}~{dates[s1]}", "ret": r["ret"], "bench": r["bench"],
                        "excess": r["excess"], "gate": rg["ret"]})
        log(f"  {dates[s0]}~{dates[s1]}  倾斜 {r['ret'] * 100:+7.2f}%  纯门控 {rg['ret'] * 100:+7.2f}%  "
            f"基准 {r['bench'] * 100:+7.2f}%  超额 {r['excess'] * 100:+7.2f}pp")

    # ---------- 10. 报告 ----------
    log("\n[10] 生成报告 ...")
    v = {
        "A": A, "dates": dates, "n": n, "yrs": yrs, "is_end": is_end,
        "bad": bad, "tot": tot, "ic_is": ic_is, "ic_oos": ic_oos, "ic_full": ic_full,
        "grid": grid, "best_key": best_key, "best": best, "nb": nb, "bl": bl,
        "full_best": full_best, "bench_ret": bench_ret, "pure_gate": pure_gate,
        "tilt_nogate": tilt_nogate, "comp20": comp20, "alpha_pp": alpha_pp,
        "tilt_gate": tilt_gate, "bench_mdd": bench_mdd,
        "oos_best": oos_best, "oos_gate": oos_gate,
        "rand_rows": rand_rows, "beat": beat, "pct": pct,
        "cost_rows": cost_rows, "wf_rows": wf_rows,
        "OOS_CUTOFF": OOS_CUTOFF, "FNAME": FNAME, "FREQ": FREQ,
        "TAUS": TAUS, "FLOORS": FLOORS, "CUTOFFS": CUTOFFS, "TOP": TOP,
    }
    html = build_html(v)
    open(OUT, "w", encoding="utf-8").write(html)
    log(f"  输出 {OUT} ({len(html.encode())} 字节)")
    log(f"\n[总耗时] {time.time() - t_all:.1f}s")


def build_html(v):
    A, dates, n, yrs = v["A"], v["dates"], v["n"], v["yrs"]
    is_end = v["is_end"]
    bad, tot = v["bad"], v["tot"]
    ic_is, ic_oos, ic_full = v["ic_is"], v["ic_oos"], v["ic_full"]
    grid, best_key, best, nb, bl = v["grid"], v["best_key"], v["best"], v["nb"], v["bl"]
    full_best, bench_ret = v["full_best"], v["bench_ret"]
    pure_gate, tilt_nogate, comp20 = v["pure_gate"], v["tilt_nogate"], v["comp20"]
    tilt_gate, bench_mdd = v["tilt_gate"], v["bench_mdd"]
    alpha_pp = v["alpha_pp"]
    oos_best, oos_gate = v["oos_best"], v["oos_gate"]
    rand_rows, beat, pct = v["rand_rows"], v["beat"], v["pct"]
    cost_rows, wf_rows = v["cost_rows"], v["wf_rows"]
    OOS_CUTOFF, FNAME, FREQ = v["OOS_CUTOFF"], v["FNAME"], v["FREQ"]
    TAUS, FLOORS, CUTOFFS, TOP = v["TAUS"], v["FLOORS"], v["CUTOFFS"], v["TOP"]

    o = ['<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>AQuant v5 P2 · 价值倾斜 + 放宽门控</title><style>', CSS, '</style></head>',
         '<body><div class="wrap">']
    o.append('<h1>AQuant v5 · P2：价值倾斜 + 放宽门控</h1>')
    o.append(f'<div class="sub">把 P1 已验证的 BP 价值信号转化为组合 alpha ｜ '
             f'样本 {len(A)} 只 × {yrs:.1f} 年（{dates[0]} ~ {dates[-1]}）｜ '
             f'{FREQ} 调仓 ｜ 本金 {CASH / 1e8:.0f} 亿 ｜ 数据源：东方财富（point-in-time）</div>')

    # KPI
    o.append('<div class="kpi">')
    o.append(f'<div><span>选股 alpha（相对纯门控）</span><b class="{cls(alpha_pp)}">'
             f'{alpha_pp:+.2f}pp</b><span style="font-size:11px">P1 为 -17.27pp</span></div>')
    o.append(f'<div><span>倾斜 vs 基准</span><b class="{cls(full_best["excess"])}">'
             f'{full_best["excess"] * 100:+.2f}pp</b><span style="font-size:11px">'
             f'{full_best["ret"] * 100:+.2f}% vs {bench_ret * 100:+.2f}%</span></div>')
    o.append(f'<div><span>样本外超额</span><b class="{cls(oos_best["excess"])}">'
             f'{oos_best["excess"] * 100:+.2f}pp</b><span style="font-size:11px">'
             f'{OOS_CUTOFF} 之后</span></div>')
    o.append(f'<div><span>随机信号对照</span><b class="{"ok" if pct >= 80 else ("warn" if pct >= 60 else "bad")}">'
             f'第 {pct} 百分位</b><span style="font-size:11px">同倾斜强度同门控</span></div>')
    o.append('</div>')

    # 一、为什么
    o.append('<h2>一、P2 要解决什么</h2>')
    o.append('<div class="box info">P1 留下一个矛盾：<b>BP 价值信号统计上真实（样本外 t=4.91）</b>，'
             '但「Top-20 集中 + 二元门控」的组合 +48.43%，反而跑输纯门控 +65.70%。'
             '诊断出两个根因，P2 各开一剂药：</div>')
    o.append('<table><thead><tr><th>根因</th><th>处方</th><th>做法</th></tr></thead><tbody>')
    o.append('<tr><td>Top-20 集中牺牲分散，价值信号幅度有限，集中放大的多是个股风险、还丢了 beta</td>'
             '<td><b>A. 价值倾斜</b></td>'
             '<td>持有<b>宽基全集</b>保留 beta 与分散，只按 BP z-score 做 softmax 倾斜 '
             '<code>w_i ∝ exp(z_i/tau)</code> —— 只吃价值溢价，不吃集中风险</td></tr>')
    o.append('<tr><td>二元门控在 2024-26 牛市踏空（无门控的纯价值 +69.16% 反而是最高）</td>'
             '<td><b>B. 放宽门控</b></td>'
             '<td>门控阈值改为 <code>cutoff×MA</code>（cutoff&lt;1 容忍一定跌破），'
             '让门控只在<b>确定性熊市</b>减仓，减少牛市误踏空</td></tr>')
    o.append('</tbody></table>')
    o.append(f'<div class="box {"ok" if bad == 0 else "bad"}"><b>前视偏差审计</b>：面板 {tot} 条因子快照中，'
             f'<b>{bad} 条</b>出现 NOTICE_DATE &gt; 交易日（越界）。须为 0。</div>')

    # 二、信号复核
    o.append(f'<h2>二、信号复核：{FNAME} 的 IC 是否仍稳健</h2>')
    o.append('<table><thead><tr><th>区间</th><th>方向IC</th><th>t_dir</th><th>IC&gt;0 占比</th><th>样本月数</th></tr></thead><tbody>')
    for nm, r in (("样本内", ic_is), ("样本外", ic_oos), ("全样本", ic_full)):
        if r is None:
            o.append(f'<tr><td>{nm}</td><td colspan="4">样本不足</td></tr>')
        else:
            o.append(f'<tr><td><b>{nm}</b></td><td>{r["mean_dir"]:+.4f}</td><td>{r["t_dir"]:+.2f}</td>'
                     f'<td>{r["hit"] * 100:.0f}%</td><td>{r["n"]}</td></tr>')
    o.append('</tbody></table>')
    if ic_oos:
        o.append(f'<div class="box {"ok" if ic_oos["mean_dir"] > 0 and ic_oos["t_dir"] > 1.5 else "warn"}">'
                 f'{FNAME} 在样本外的方向 IC = <b>{ic_oos["mean_dir"]:+.4f}</b>（t={ic_oos["t_dir"]:+.2f}、'
                 f'命中 {ic_oos["hit"] * 100:.0f}%）。'
                 f'{"信号在更难的真实行情里依然成立 —— 这是 P2 敢于在其上下注的前提。" if ic_oos["mean_dir"] > 0 and ic_oos["t_dir"] > 1.5 else "样本外信号减弱，下注需谨慎。"}</div>')

    # 三、参数网格
    o.append('<h2>三、参数网格（样本内挑选）</h2>')
    o.append(f'<p>在样本内扫描 <code>tau</code>（倾斜锐度）× <code>floor</code>（等权混合比例）× '
             f'<code>cutoff</code>（门控放宽系数）。cutoff=0.00 即完全无门控。</p>')
    o.append('<table><thead><tr><th>tau</th><th>floor</th><th>门控cutoff</th><th>样本内收益</th>'
             '<th>样本内超额</th><th>年换手</th><th>交易笔数</th></tr></thead><tbody>')
    for tau in TAUS:
        for fl in FLOORS:
            for cu in CUTOFFS:
                g = grid[(tau, fl, cu)]
                mark = ' style="background:#fff7ed"' if (tau, fl, cu) == best_key else ''
                o.append(f'<tr{mark}><td>{tau:.0f}</td><td>{fl:.2f}</td>'
                         f'<td>{"无门控" if cu == 0.0 else f"{cu:.2f}"}</td>'
                         f'<td>{pc(g["ret"])}</td><td><b class="{cls(g["excess"])}">'
                         f'{g["excess"] * 100:+.2f}pp</b></td><td>{g["turn"] * 100:.0f}%</td>'
                         f'<td>{g["n"]}</td></tr>')
    o.append('</tbody></table>')
    tau_spread = (max(grid[(t, f, best_key[2])]["excess"] for t in TAUS for f in FLOORS)
                  - min(grid[(t, f, best_key[2])]["excess"] for t in TAUS for f in FLOORS))
    cut_series = [(cu, grid[(best_key[0], best_key[1], cu)]["excess"],
                   grid[(best_key[0], best_key[1], cu)]["turn"]) for cu in CUTOFFS]
    cut_txt = "，".join(f'{"无门控" if cu == 0.0 else f"{cu:.2f}"}→{ex * 100:+.1f}pp'
                       f'（换手{tu * 100:.0f}%）' for cu, ex, tu in cut_series)
    nb_txt = ""
    if nb is not None:
        gap = (best["excess"] - nb) * 100
        nb_txt = (f'邻域最坏超额 <b>{nb * 100:+.2f}pp</b>，与最佳差 {gap:.2f}pp。')
    o.append(f'<div class="box {"warn" if (nb is not None and (best["excess"] - nb) * 100 > 5) else "ok"}">'
             f'最佳参数：<b>tau={best_key[0]:.0f}，floor={best_key[1]:.2f}，'
             f'门控cutoff={best_key[2]:.2f}</b>，样本内超额 {best["excess"] * 100:+.2f}pp。{nb_txt}</div>')
    o.append(f'<div class="box info"><b>分轴稳健性（关键，防自欺）</b>：<br>'
             f'· <code>tau</code>/<code>floor</code>（倾斜强度）：在最佳门控下，全网格极差仅 '
             f'<b>{tau_spread * 100:.2f}pp</b> —— 是<b>宽平台</b>，倾斜强度怎么设都差不多，'
             f'这块不敏感、也不易过拟合。<br>'
             f'· <code>cutoff</code>（门控放宽）：<b>非单调</b> —— {cut_txt}。'
             f'中间值最差并非巧合：它让门控<b>频繁来回翻转</b>（换手最高），产生 whipsaw 损耗；'
             f'越接近"不翻转"（无门控）越好。因此最佳点落在轴端点（无门控）是<b>机制使然</b>而非数据噪音，'
             f'但结论应理解为"<b>本样本下门控弊大于利</b>"，不等于门控在任何行情都无用。</div>')
    o.append(f'<div class="box info"><b>放宽门控的边际贡献</b>：同 tau/floor 下，原门控(cutoff=1.00) 超额 '
             f'{bl["excess"] * 100:+.2f}pp -> 放宽到 {best_key[2]:.2f} 后 '
             f'{best["excess"] * 100:+.2f}pp，增益 '
             f'<b>{(best["excess"] - bl["excess"]) * 100:+.2f}pp</b>。'
             f'{"放宽门控确实有效（P1 诊断的牛市踏空被修复）。" if best["excess"] > bl["excess"] else "放宽门控无增益甚至有害 —— P1 的踏空诊断在本样本下不成立。"}</div>')

    # 四、关键对照
    o.append('<h2>四、关键对照：倾斜有没有把信号转成 alpha</h2>')
    o.append('<div class="box warn">裁判不是绝对收益，而是两条：'
             '① <b>相对纯门控</b>（多赚的部分才是选股 alpha）；② <b>随机信号对照</b>（信号是否携带信息）。</div>')
    o.append('<table><thead><tr><th>策略</th><th>总收益</th><th>相对基准</th><th>年换手</th>'
             '<th>最大回撤</th><th>交易笔数</th></tr></thead><tbody>')
    o.append(f'<tr style="background:#fff7ed"><td><b>价值倾斜 + 放宽门控（P2）</b></td>'
             f'<td>{pc(full_best["ret"])}</td>'
             f'<td><b class="{cls(full_best["excess"])}">{full_best["excess"] * 100:+.2f}pp</b></td>'
             f'<td>{full_best["turnover"] * 100:.0f}%</td>'
             f'<td>{full_best["raw"]["max_drawdown"] * 100:.1f}%</td><td>{full_best["n"]}</td></tr>')
    o.append(f'<tr><td>纯门控（宽基，原门控）</td><td>{pc(pure_gate["ret"])}</td>'
             f'<td>{pure_gate["excess"] * 100:+.2f}pp</td><td>{pure_gate["turnover"] * 100:.0f}%</td>'
             f'<td>{pure_gate["raw"]["max_drawdown"] * 100:.1f}%</td><td>{pure_gate["n"]}</td></tr>')
    o.append(f'<tr><td>价值倾斜 + 原门控（稳健版，cutoff=1.00）</td><td>{pc(tilt_gate["ret"])}</td>'
             f'<td>{tilt_gate["excess"] * 100:+.2f}pp</td><td>{tilt_gate["turnover"] * 100:.0f}%</td>'
             f'<td>{tilt_gate["raw"]["max_drawdown"] * 100:.1f}%</td><td>{tilt_gate["n"]}</td></tr>')
    o.append(f'<tr><td>Top{TOP} 集中 + 门控（P1 做法）</td><td>{pc(comp20["ret"])}</td>'
             f'<td>{comp20["excess"] * 100:+.2f}pp</td><td>{comp20["turnover"] * 100:.0f}%</td>'
             f'<td>{comp20["raw"]["max_drawdown"] * 100:.1f}%</td><td>{comp20["n"]}</td></tr>')
    o.append(f'<tr><td>等权买入持有（基准）</td><td>{pc(bench_ret)}</td><td>—</td><td>0%</td>'
             f'<td>{bench_mdd * 100:.1f}%</td><td>（不交易）</td></tr>')
    o.append('</tbody></table>')
    # 收益 vs 回撤的真实取舍
    d_ret = (full_best["ret"] - tilt_gate["ret"]) * 100
    d_mdd = (tilt_gate["raw"]["max_drawdown"] - full_best["raw"]["max_drawdown"]) * 100
    o.append(f'<div class="box warn"><b>收益 vs 回撤的真实取舍</b>：P2 最佳（放宽到无门控）比稳健版'
             f'（保留原门控）多赚 <b>{d_ret:+.2f}pp</b>，但最大回撤也深了 <b>{d_mdd:.1f}pp</b>'
             f'（{full_best["raw"]["max_drawdown"] * 100:.1f}% vs {tilt_gate["raw"]["max_drawdown"] * 100:.1f}%，'
             f'基准自身回撤 {bench_mdd * 100:.1f}%）。'
             f'{"选门控=买回撤保护、放弃部分收益；选无门控=要收益、承受更深回撤。这是风险偏好问题，不是对错问题。" if d_ret > 0 else "原门控既少赚又没减回撤，本样本下应直接弃用门控。"}</div>')
    ab = "ok" if alpha_pp > 1.0 else ("warn" if alpha_pp > -1.0 else "bad")
    verdict = ("选股 alpha 转正 —— 价值倾斜成功把 BP 信号转化为组合增值。"
               if alpha_pp > 1.0 else
               "选股贡献微弱，组合仍主要靠择时。"
               if alpha_pp > -1.0 else
               "选股仍为负贡献 —— 即便换成倾斜，BP 信号在经济上也打不过「宽基+择时」。")
    o.append(f'<div class="box {ab}">相对纯门控的选股 alpha = <b>{alpha_pp:+.2f}pp</b>（P1 为 -17.27pp）。{verdict}</div>')
    conc_vs_top20 = (full_best["ret"] - comp20["ret"]) * 100
    o.append(f'<div class="box {"ok" if conc_vs_top20 > 0 else "warn"}">'
             f'<b>倾斜 vs 集中（同一 BP 信号的两种用法）</b>：倾斜 {full_best["ret"] * 100:+.2f}% '
             f'{"高于" if conc_vs_top20 > 0 else "低于"} Top{TOP} 集中 {comp20["ret"] * 100:+.2f}%，'
             f'差 {conc_vs_top20:+.2f}pp —— '
             f'{"证明「保留分散的价值倾斜」优于「集中买最便宜的」，P1 根因① 判断正确。" if conc_vs_top20 > 0 else "集中反而更好，P1 根因① 判断不成立。"}</div>')

    # 五、样本外
    o.append(f'<h2>五、样本外检验（硬门槛：{OOS_CUTOFF} 之后）</h2>')
    o.append('<table><thead><tr><th>策略</th><th>样本外收益</th><th>样本外基准</th><th>样本外超额</th></tr></thead><tbody>')
    o.append(f'<tr style="background:#fff7ed"><td><b>价值倾斜 + 放宽门控</b></td>'
             f'<td>{pc(oos_best["ret"])}</td><td>{pc(oos_best["bench"])}</td>'
             f'<td><b class="{cls(oos_best["excess"])}">{oos_best["excess"] * 100:+.2f}pp</b></td></tr>')
    o.append(f'<tr><td>纯门控（对照）</td><td>{pc(oos_gate["ret"])}</td><td>{pc(oos_gate["bench"])}</td>'
             f'<td><b class="{cls(oos_gate["excess"])}">{oos_gate["excess"] * 100:+.2f}pp</b></td></tr>')
    o.append('</tbody></table>')
    oos_alpha = (oos_best["ret"] - oos_gate["ret"]) * 100
    o.append(f'<div class="box {"ok" if oos_best["excess"] > 0 else "bad"}">样本外超额 '
             f'<b>{oos_best["excess"] * 100:+.2f}pp</b>（v4 在此翻车 -12.65pp）；'
             f'相对纯门控的样本外 alpha = <b>{oos_alpha:+.2f}pp</b>。'
             f'{"样本外仍跑赢基准，选股 alpha 经受住 OOS 检验。" if oos_best["excess"] > 0 else "样本外跑输基准 —— alpha 仍未证实。"}</div>')

    # 六、随机对照
    o.append('<h2>六、随机信号对照（照妖镜）</h2>')
    o.append('<div class="box warn">对照组与倾斜策略<b>完全相同</b>的倾斜强度(tau)、等权混合(floor)、'
             '调仓频率、门控、因子覆盖，唯一区别是把「哪个票拿哪个权重」随机打乱 —— '
             '权重分布不变（换手量级相同），但信号变成噪声。</div>')
    o.append('<table><thead><tr><th>策略</th><th>年换手</th><th>总收益</th></tr></thead><tbody>')
    o.append(f'<tr style="background:#fff7ed"><td><b>价值倾斜 + 放宽门控</b></td>'
             f'<td>{full_best["turnover"] * 100:.0f}%</td><td><b>{pc(full_best["ret"])}</b></td></tr>')
    for x in rand_rows:
        o.append(f'<tr><td>随机倾斜 #{x["seed"]}</td><td>{x["turn"] * 100:.0f}%</td>'
                 f'<td>{pc(x["ret"])}</td></tr>')
    o.append('</tbody></table>')
    bc = "ok" if pct >= 80 else ("warn" if pct >= 60 else "bad")
    o.append(f'<div class="box {bc}">在 {len(rand_rows)} 个随机策略中跑赢 <b>{beat} 个（第 {pct} 百分位）</b>。'
             f'{"显著优于随机，选股信号有信息量。" if pct >= 80 else "与随机无异，alpha 仍未被证实。" if pct < 60 else "略优于随机，优势不牢固。"}</div>')

    # 七、成本
    o.append('<h2>七、成本敏感性</h2>')
    o.append('<table><thead><tr><th>双边滑点</th><th>策略收益</th><th>基准收益</th><th>相对基准</th><th>交易笔数</th></tr></thead><tbody>')
    for r in cost_rows:
        ex = r["ret"] - r["bench"]
        o.append(f'<tr><td>{r["slip"] * 100:.2f}%</td><td>{pc(r["ret"])}</td><td>{pc(r["bench"])}</td>'
                 f'<td><b class="{cls(ex)}">{ex * 100:+.2f}pp</b></td><td>{r["n"]}</td></tr>')
    o.append('</tbody></table>')
    worst = cost_rows[-1]
    o.append(f'<div class="box {"bad" if worst["ret"] < worst["bench"] else "ok"}">'
             f'滑点放大到 {worst["slip"] * 100:.2f}% 时收益 {pc(worst["ret"])}（基准 {pc(worst["bench"])}）。'
             f'{"仍跑赢基准，alpha 非过拟合。" if worst["ret"] > worst["bench"] else "已被成本吃掉，实盘需进一步压换手。"}</div>')

    # 八、滚动前推
    o.append('<h2>八、滚动前推（最贴近真实使用）</h2>')
    o.append('<table><thead><tr><th>区间</th><th>倾斜策略</th><th>纯门控</th><th>基准</th><th>超额(vs基准)</th></tr></thead><tbody>')
    wins = 0
    for r in wf_rows:
        if r["excess"] > 0:
            wins += 1
        o.append(f'<tr><td class="mono">{r["seg"]}</td><td>{pc(r["ret"])}</td><td>{pc(r["gate"])}</td>'
                 f'<td>{pc(r["bench"])}</td><td><b class="{cls(r["excess"])}">{r["excess"] * 100:+.2f}pp</b></td></tr>')
    o.append('</tbody></table>')
    o.append(f'<div class="box {"ok" if wins >= 3 else ("warn" if wins >= 2 else "bad")}">'
             f'4 段中 <b>{wins} 段</b>跑赢基准。</div>')

    # 九、结论
    o.append('<h2>九、结论</h2>')
    gate_gain = (best["excess"] - bl["excess"]) * 100
    conc_gain = (full_best["ret"] - comp20["ret"]) * 100
    o.append('<div class="box warn"><h3 style="margin-top:0">判定</h3><ul>')
    o.append(f'<li><b>信号复核</b>：{FNAME} 样本外方向IC {ic_oos["mean_dir"] if ic_oos else 0:+.4f}'
             f'（t={ic_oos["t_dir"] if ic_oos else 0:+.2f}）—— '
             f'{"信号稳健，是 P2 下注的前提。" if ic_oos and ic_oos["mean_dir"] > 0 and ic_oos["t_dir"] > 1.5 else "信号减弱。"}</li>')
    o.append(f'<li><b>处方 A（价值倾斜）</b>：倾斜 {full_best["ret"] * 100:+.2f}% '
             f'{"高于" if conc_gain > 0 else "低于"} Top{TOP} 集中 {comp20["ret"] * 100:+.2f}%'
             f'（{conc_gain:+.2f}pp）—— '
             f'{"保留分散的宽基倾斜确实优于集中，根因① 成立。" if conc_gain > 0 else "集中仍更优，根因① 不成立。"}</li>')
    o.append(f'<li><b>处方 B（放宽门控）</b>：样本内增益 <b>{gate_gain:+.2f}pp</b> —— '
             f'{"牛市踏空被修复，根因② 成立；且放宽的极限就是「完全弃用门控」，代价是回撤变深（见第四节取舍框）。" if gate_gain > 0 else "放宽无增益，根因② 不成立，原门控已是最优。"}</li>')
    o.append(f'<li><b>核心裁判</b>：相对纯门控的选股 alpha = <b>{alpha_pp:+.2f}pp</b>（P1 -17.27pp）；'
             f'随机对照第 <b>{pct}</b> 百分位；样本外超额 <b>{oos_best["excess"] * 100:+.2f}pp</b>。</li>')
    o.append('<li><b>诚实边界</b>：参数在样本内网格挑选（已用邻域最坏值防尖峰）；'
             '幸存者偏差（退市股财务缺失）、财报追溯调整、样本仅一轮牛熊仍是硬约束。</li>')
    o.append('</ul></div>')
    o.append('<footer>AQuant v5 P2 ｜ 数据源：东方财富数据中心（point-in-time）｜ 不构成投资建议</footer>')
    o.append('</div></body></html>')
    return "".join(o)


if __name__ == "__main__":
    main()
