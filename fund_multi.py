#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AQuant v5 · P5：把选股层从「单因子 BP」升级为「多因子合成倾斜」
==============================================================
P4 的结论：风控层只能沿着同一条风险-收益线**移动**仓位，抬不高这条线。
所以想再往上走，只能动选股层 —— 这是 P5 的动机。

已有基础：
  * P1 单因子 IC 检验：BP 入选（样本外 t=+4.91），但 Top-20 集中的 alpha 是 **-17.27pp**；
  * P2 把 BP 改成「保留分散度的宽基倾斜」，alpha 翻正为 **+30.45pp**，OOS +22.45pp。
    即：**怎么用**因子比**用哪个**因子更关键。
  * P4 证明风控层不改变选股 alpha，只缩放暴露。

P5 要做的事：在 P2 验证过的「倾斜」用法之上，把单因子换成多因子合成。
框架里现成有 7 个 point-in-time 因子（估值 2 + 质量 3 + 成长 2）：
    ep / bp / roe / gmarg / nmarg / revyoy / profyoy

三个必须回答的问题：
  1. 单因子 IC：哪些在样本内显著？（P1 只报了 BP 入选，其余没公开）
  2. 合成是否有增量：多因子 vs 单因子 BP，在**同一个倾斜机制**下比较
     （这一步极易做错：若顺带改了倾斜强度或集中度，增量就归因不清）
  3. 增量是真 alpha 还是过拟合：样本外硬门槛 + 随机信号对照 + 成本 + 滚动前推

纪律（同 P1~P4）：
  * 因子筛选与合成权重**只用样本内 IC**决定，样本外（2023-01-01 之后）全程不参与任何决策；
  * 风控层**固定不动**（主线用无风控，即 alpha 全开），只换选股层 —— 否则选股与风控的
    增量会混在一起，无法归因；
  * 所有配置共享同一个倾斜强度 tau / floor / 调仓频率，保证「唯一差别是因子」。
"""

import sys, os, time
sys.path.insert(0, "/workspace/quant")
from optimize import (fill_to_axis, CASH, CSS, cls, pc)
from aq import data, strategies as st, backtest as bt
from aq import fundamentals as fnd
from fund_ic import (month_starts, to_dict, build_panel,
                     audit_point_in_time, ic_agg_fund, run_fund)

OUT = "/workspace/quant/多因子合成倾斜_P5.html"
OOS_CUTOFF = "2023-01-01"
WARM = 300
FREQ = "quarterly"
TAU, FLOOR = 6.0, 0.0        # 沿用 P2 样本内选出的倾斜强度（不重新调，避免归因混淆）
HOLD = 63                     # IC 检验的前瞻收益窗口（季频）

KW = {"rebalance": FREQ, "max_weight": 0.0}

# 全部候选因子（估值 / 质量 / 成长）
ALL_FACTORS = ["ep", "bp", "roe", "gmarg", "nmarg", "revyoy", "profyoy"]
VAL_FACTORS = ("ep", "bp")     # 需要 price 的估值类

# 样本内入选门槛：方向 IC 的 t 值
T_MIN = 2.0


def main():
    t_all = time.time()
    log = lambda *a: print(*a, flush=True)
    log("=" * 68)
    log("AQuant v5 · P5：选股层升级为多因子合成倾斜")
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

    # ---------- 2. 面板 + 前视审计 ----------
    log("\n[2] 构建 point-in-time 面板 ...")
    panel = build_panel(fund, list(A.keys()), reb, dates)
    bad, tot = audit_point_in_time(panel, dates)
    log(f"  前视审计: {bad}/{tot} 条越界（须为 0）")

    # ---------- 3. 单因子 IC 全家福（样本内选因子，样本外只作事后观察） ----------
    log(f"\n[3] 单因子 IC 扫描（前瞻 {HOLD} 日，样本内/样本外分开）...")
    ic_rows = []
    for f in ALL_FACTORS:
        a = ic_agg_fund(panel, Ad, dates, f, HOLD, reb, 0, is_end - 1)
        b = ic_agg_fund(panel, Ad, dates, f, HOLD, reb, is_end, n - 1)
        ic_rows.append({"f": f, "is": a, "oos": b})
        si = f"{a['mean_dir']:+.4f} (t {a['t_dir']:+.2f})" if a else "样本不足"
        so = f"{b['mean_dir']:+.4f} (t {b['t_dir']:+.2f})" if b else "样本不足"
        log(f"  {f:<9} 样本内 {si:<22} 样本外 {so}")

    # 入选规则（先验）：样本内 t 值显著且方向与理论一致（方向 IC 为正）
    picked = []
    for r in ic_rows:
        a = r["is"]
        if a and a["t_dir"] >= T_MIN and a["mean_dir"] > 0:
            picked.append(r)
    picked_names = [r["f"] for r in picked]
    log(f"  入选（样本内 t>={T_MIN} 且 IC>0）：{picked_names if picked_names else '无'}")

    # ---------- 4. 合成方式对比（样本内，选股层唯一差别） ----------
    log("\n[4] 样本内合成方式对比（倾斜强度/风控/频率全部固定）...")
    def mk(names, weights=None, seed=None):
        if seed is None:
            if len(names) == 1 and weights is None:
                return lambda: st.fund_tilt(names[0], tau=TAU, floor=FLOOR)
            return lambda: st.multi_tilt(names, weights, tau=TAU, floor=FLOOR)
        return lambda: st.rand_multi_tilt(seed, names, weights, tau=TAU, floor=FLOOR)

    bp_only = lambda: st.fund_tilt("bp", tau=TAU, floor=FLOOR)

    # 三种合成权重：等权 / IC 均值加权 / IR(IC均值/IC标准差) 加权
    def w_ic(rs):
        return [max(r["is"]["mean_dir"], 0.0) for r in rs]

    def w_ir(rs):
        # ic_agg_fund 不直接返回 IC 标准差，但 t = IR × √n，故 IR ≈ t/√n
        out = []
        for r in rs:
            nn = r["is"]["n"]
            out.append(abs(r["is"]["t_dir"]) / (nn ** 0.5) if nn > 0 else 0.0)
        return out

    ic_std_ok = True
    # 全因子 IC **带符号**加权：权重直接取样本内 mean_dir（可为负，即反向使用）。
    # 这是探索性配置 —— 方向取自样本内 IC，本身就是选参，过拟合风险高于其他配置，
    # 只有样本外能站住才算数。
    def w_signed(rs):
        return [r["is"]["mean_dir"] for r in rs]

    # 只用显著因子（|t|>=T_MIN），方向按样本内 IC 的实际符号
    sig_f = [r for r in ic_rows if r["is"] and abs(r["is"]["t_dir"]) >= T_MIN]
    sig_names = [r["f"] for r in sig_f]

    combos = [("BP 单因子（P2 基线）", ["bp"], None)]
    if len(picked_names) >= 2:
        combos.append((f"入选因子等权（{'+'.join(picked_names)}）", picked_names, None))
        combos.append((f"入选因子 IC 加权（{'+'.join(picked_names)}）", picked_names, w_ic(picked)))
        combos.append((f"入选因子 IR 加权（{'+'.join(picked_names)}）", picked_names, w_ir(picked)))
    # 全因子等权（不筛选）——检验"筛选"这一步到底有没有用
    combos.append(("全 7 因子等权（不筛选）", ALL_FACTORS, None))
    # 全因子 IC 带符号加权（含反向使用负向因子）
    combos.append(("全 7 因子 IC 带符号加权", ALL_FACTORS, w_signed(ic_rows)))
    if len(sig_names) >= 2:
        combos.append((f"显著因子带符号加权（{'+'.join(sig_names)}）",
                       sig_names, w_signed(sig_f)))

    is_rows = {}
    for name, names, w in combos:
        fct = (lambda: st.fund_tilt(names[0], tau=TAU, floor=FLOOR)) \
            if (len(names) == 1 and w is None) else mk(names, w)
        r = run_fund(A, dates, mkt, WARM, is_end - 1, fct, fund, **KW)
        is_rows[name] = {"r": r, "names": names, "w": w, "excess": r["excess"],
                         "mdd": r["raw"]["max_drawdown"], "turn": r["turnover"]}
        log(f"  {name:<34} 超额 {r['excess'] * 100:>+8.2f}pp  回撤 {r['raw']['max_drawdown'] * 100:>5.1f}%  "
            f"换手 {r['turnover'] * 100:>5.0f}%")

    base_ex = is_rows["BP 单因子（P2 基线）"]["excess"]
    for name, v in is_rows.items():
        v["lift"] = v["excess"] - base_ex

    # 选参规则（先验，与 P4 R4 同源）：样本内超额最高的合成方式；
    # 但若多因子相对 BP 的增量 <1pp，视为无增量，仍用 BP（奥卡姆：不增加复杂度）
    best_name = max(is_rows, key=lambda k: is_rows[k]["excess"])
    best = is_rows[best_name]
    use_multi = (best_name != "BP 单因子（P2 基线）") and (best["lift"] > 0.01)
    if not use_multi:
        best_name = "BP 单因子（P2 基线）"
        best = is_rows[best_name]
    final_names, final_w = best["names"], best["w"]
    log(f"\n  >> 选中：{best_name}（样本内超额 {best['excess'] * 100:+.2f}pp，"
        f"相对 BP 增量 {best['lift'] * 100:+.2f}pp）")
    if not use_multi:
        log("     增量未达 1pp 门槛 -> 按奥卡姆原则仍用单因子 BP")

    FINAL = mk(final_names, final_w)
    BP = lambda: st.fund_tilt("bp", tau=TAU, floor=FLOOR)

    # 探索性对照：样本内表现最好的「非 BP」配置。即使它没被选中，也要看它的样本外，
    # 否则无法判断 1pp 门槛是否设错（若它样本外明显更好，说明门槛过严）。
    cands = [(k, v) for k, v in is_rows.items() if k != "BP 单因子（P2 基线）"]
    runner = max(cands, key=lambda kv: kv[1]["excess"]) if cands else None
    runner_name = runner[0] if runner else None

    # ---------- 5. 全样本 ----------
    log("\n[5] 全样本 ...")
    full_final = run_fund(A, dates, mkt, WARM, n - 1, FINAL, fund, **KW)
    full_bp = run_fund(A, dates, mkt, WARM, n - 1, BP, fund, **KW)
    bench_ret = full_bp["bench"]
    keep = (full_final["excess"] / full_bp["excess"]) if full_bp["excess"] > 1e-9 else 0.0
    log(f"  多因子(最终) {full_final['ret'] * 100:+.2f}%  回撤 {full_final['raw']['max_drawdown'] * 100:.1f}%")
    log(f"  BP 单因子    {full_bp['ret'] * 100:+.2f}%  回撤 {full_bp['raw']['max_drawdown'] * 100:.1f}%")
    log(f"  基准 {bench_ret * 100:+.2f}%")

    # ---------- 6. 样本外 ----------
    log(f"\n[6] 样本外检验（>= {OOS_CUTOFF}）...")
    oos_final = run_fund(A, dates, mkt, is_end, n - 1, FINAL, fund, **KW)
    oos_bp = run_fund(A, dates, mkt, is_end, n - 1, BP, fund, **KW)
    log(f"  样本外 多因子 {oos_final['ret'] * 100:+.2f}% / BP {oos_bp['ret'] * 100:+.2f}% / "
        f"基准 {oos_final['bench'] * 100:+.2f}%")
    log(f"  样本外超额 多因子 {oos_final['excess'] * 100:+.2f}pp / BP {oos_bp['excess'] * 100:+.2f}pp")

    runner_full = runner_oos = None
    if runner is not None:
        log(f"\n[6b] 探索性对照（样本内次优，未被选中）：{runner_name} ...")
        rf = mk(runner[1]["names"], runner[1]["w"])
        runner_full = run_fund(A, dates, mkt, WARM, n - 1, rf, fund, **KW)
        runner_oos = run_fund(A, dates, mkt, is_end, n - 1, rf, fund, **KW)
        log(f"  全样本 {runner_full['ret'] * 100:+.2f}%  样本外超额 {runner_oos['excess'] * 100:+.2f}pp "
            f"（BP 样本外 {oos_bp['excess'] * 100:+.2f}pp）")

    # ---------- 7. 随机对照 ----------
    log("\n[7] 随机信号对照（20 次，同合成同倾斜）...")
    rand_rows = []
    for i in range(20):
        r = run_fund(A, dates, mkt, WARM, n - 1, mk(final_names, final_w, seed=i), fund, **KW)
        rand_rows.append({"seed": i, "ret": r["ret"], "turn": r["turnover"]})
    beat = sum(1 for x in rand_rows if x["ret"] < full_final["ret"])
    pct = int(beat / len(rand_rows) * 100)
    log(f"  跑赢 {beat}/{len(rand_rows)} 个随机 -> 第 {pct} 百分位")

    # ---------- 8. 成本 ----------
    log("\n[8] 成本敏感性 ...")
    cost_rows = []
    for slip in (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005):
        r = run_fund(A, dates, mkt, WARM, n - 1, FINAL, fund, cost=bt.Cost(slippage=slip), **KW)
        cost_rows.append({"slip": slip, "ret": r["ret"], "bench": r["bench"], "n": r["n"]})
        log(f"  滑点 {slip * 100:.2f}% -> {r['ret'] * 100:+8.2f}% (基准 {r['bench'] * 100:+.2f}%)")

    # ---------- 9. 滚动前推 ----------
    log("\n[9] 滚动前推（多因子 vs BP vs 基准）...")
    segs, L = [], n // 4
    for i in range(4):
        s0 = max(WARM, i * L)
        s1 = n - 1 if i == 3 else (i + 1) * L - 1
        segs.append((s0, s1))
    wf_rows = []
    wins = 0
    for (s0, s1) in segs:
        r = run_fund(A, dates, mkt, s0, s1, FINAL, fund, **KW)
        rb = run_fund(A, dates, mkt, s0, s1, BP, fund, **KW)
        wf_rows.append({"seg": f"{dates[s0]}~{dates[s1]}", "ret": r["ret"], "bench": r["bench"],
                        "excess": r["excess"], "bp": rb["ret"], "bp_excess": rb["excess"]})
        if r["excess"] > 0:
            wins += 1
        log(f"  {dates[s0]}~{dates[s1]}  多因子 {r['ret'] * 100:+7.2f}%  BP {rb['ret'] * 100:+7.2f}%  "
            f"基准 {r['bench'] * 100:+7.2f}%  多因子超额 {r['excess'] * 100:+7.2f}pp  "
            f"BP超额 {rb['excess'] * 100:+7.2f}pp")

    # ---------- 10. 叠加 P4 风控层（可落地的完整方案） ----------
    log("\n[10] 叠加 P4 风控层（波动率目标 25%，回看 60 日）...")
    def with_vol(fct):
        return lambda: st.port_vol_target(fct(), target=0.25, cap=1.0)
    full_vol = run_fund(A, dates, mkt, WARM, n - 1, with_vol(FINAL), fund,
                        **{**KW, "mkt_vol_n": 60})
    oos_vol = run_fund(A, dates, mkt, is_end, n - 1, with_vol(FINAL), fund,
                       **{**KW, "mkt_vol_n": 60})
    log(f"  全样本 {full_vol['ret'] * 100:+.2f}%  回撤 {full_vol['raw']['max_drawdown'] * 100:.1f}%")
    log(f"  样本外超额 {oos_vol['excess'] * 100:+.2f}pp")

    # ---------- 11. 报告 ----------
    log("\n[10] 生成报告 ...")
    v = {
        "A": A, "dates": dates, "n": n, "yrs": yrs, "is_end": is_end,
        "bad": bad, "tot": tot, "ic_rows": ic_rows, "picked": picked_names,
        "T_MIN": T_MIN, "is_rows": is_rows, "best_name": best_name, "best": best,
        "use_multi": use_multi, "final_names": final_names,
        "full_final": full_final, "full_bp": full_bp, "bench_ret": bench_ret, "keep": keep,
        "oos_final": oos_final, "oos_bp": oos_bp,
        "rand_rows": rand_rows, "beat": beat, "pct": pct,
        "cost_rows": cost_rows, "wf_rows": wf_rows, "wins": wins,
        "full_vol": full_vol, "oos_vol": oos_vol,
        "runner_name": runner_name, "runner_full": runner_full, "runner_oos": runner_oos,
        "OOS_CUTOFF": OOS_CUTOFF, "FREQ": FREQ, "TAU": TAU, "FLOOR": FLOOR, "HOLD": HOLD,
        "ALL_FACTORS": ALL_FACTORS,
    }
    html = build_html(v)
    open(OUT, "w", encoding="utf-8").write(html)
    log(f"  输出 {OUT} ({len(html.encode())} 字节)")
    log(f"\n[总耗时] {time.time() - t_all:.1f}s")


def build_html(v):
    A, dates, n, yrs = v["A"], v["dates"], v["n"], v["yrs"]
    bad, tot = v["bad"], v["tot"]
    ic_rows, picked, T_MIN = v["ic_rows"], v["picked"], v["T_MIN"]
    is_rows, best_name, best = v["is_rows"], v["best_name"], v["best"]
    use_multi, final_names = v["use_multi"], v["final_names"]
    full_final, full_bp = v["full_final"], v["full_bp"]
    bench_ret, keep = v["bench_ret"], v["keep"]
    oos_final, oos_bp = v["oos_final"], v["oos_bp"]
    rand_rows, beat, pct = v["rand_rows"], v["beat"], v["pct"]
    cost_rows, wf_rows, wins = v["cost_rows"], v["wf_rows"], v["wins"]
    full_vol, oos_vol = v["full_vol"], v["oos_vol"]
    runner_name, runner_full, runner_oos = v["runner_name"], v["runner_full"], v["runner_oos"]
    OOS_CUTOFF, FREQ = v["OOS_CUTOFF"], v["FREQ"]
    TAU, FLOOR, HOLD = v["TAU"], v["FLOOR"], v["HOLD"]

    def _calmar(r):
        m = abs(r["raw"]["max_drawdown"])
        return r["raw"]["annual"] / m if m > 1e-9 else 0.0

    o = ['<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>AQuant v5 P5 · 多因子合成倾斜</title><style>', CSS, '</style></head>',
         '<body><div class="wrap">']
    o.append('<h1>AQuant v5 · P5：选股层升级为多因子合成倾斜</h1>')
    o.append(f'<div class="sub">风控层固定不动，只换选股层：单因子 BP → 多因子合成 ｜ '
             f'样本 {len(A)} 只 × {yrs:.1f} 年（{dates[0]} ~ {dates[-1]}）｜ {FREQ} 调仓 ｜ '
             f'倾斜 tau={TAU:.0f} floor={FLOOR:.2f} ｜ 本金 {CASH / 1e8:.0f} 亿 ｜ '
             f'数据源：东方财富（point-in-time）</div>')

    # KPI
    d_ex = (oos_final["excess"] - oos_bp["excess"]) * 100
    o.append('<div class="kpi">')
    o.append(f'<div><span>样本内增量</span><b class="{cls(best["lift"])}">'
             f'{best["lift"] * 100:+.2f}pp</b><span style="font-size:11px">多因子 vs BP</span></div>')
    o.append(f'<div><span>样本外超额</span><b class="{cls(oos_final["excess"])}">'
             f'{oos_final["excess"] * 100:+.2f}pp</b><span style="font-size:11px">'
             f'BP {oos_bp["excess"] * 100:+.2f}pp</span></div>')
    o.append(f'<div><span>样本外增量</span><b class="{cls(d_ex / 100)}">'
             f'{d_ex:+.2f}pp</b><span style="font-size:11px">相对 BP 单因子</span></div>')
    o.append(f'<div><span>随机信号对照</span><b class="{"ok" if pct >= 80 else ("warn" if pct >= 60 else "bad")}">'
             f'第 {pct} 百分位</b><span style="font-size:11px">同合成同倾斜</span></div>')
    o.append('</div>')

    # 一、动机
    o.append('<h2>一、为什么现在动选股层</h2>')
    o.append('<div class="box info"><b>P4 的结论</b>：风控层（无论是 MA200 还是波动率目标）'
             '只能沿着同一条风险-收益线<b>移动</b>仓位，两族各赢一半口径，'
             '没有任何一个能把这条线本身抬上去。<br>'
             '<b>P2 的结论</b>：<b>怎么用</b>因子比<b>用哪个</b>因子更关键 —— '
             'BP 从「Top-20 集中」（alpha <b>-17.27pp</b>）改成「保留分散度的宽基倾斜」后，'
             'alpha 翻正到 <b>+30.45pp</b>。<br>'
             '两条合起来指向同一件事：剩下的空间在<b>选股层</b>，不在风控层。</div>')
    o.append('<div class="box warn"><b>本实验的控制变量</b>：倾斜强度 tau / floor、'
             '调仓频率、覆盖率阈值、风控层（主线为无风控，alpha 全开）<b>全部固定</b>，'
             '唯一的差别是「喂给倾斜机制的分数怎么算」。'
             '否则选股增量与"倾斜强度变化"会混在一起，归因不清。</div>')

    # 二、单因子 IC 全家福
    o.append(f'<h2>二、单因子 IC 全家福（前瞻 {HOLD} 日）</h2>')
    o.append('<p>P1 只公开了 BP 入选的结果，其余因子从未披露。这里把 7 个因子全部摊开，'
             '样本内用于<b>选因子</b>，样本外只作<b>事后观察</b>（不参与任何决策）。</p>')
    o.append('<table><thead><tr><th>因子</th><th>类别</th><th>样本内 方向IC</th><th>样本内 t</th>'
             '<th>样本外 方向IC</th><th>样本外 t</th><th>入选</th></tr></thead><tbody>')
    CAT = {"ep": "估值", "bp": "估值", "roe": "质量", "gmarg": "质量",
           "nmarg": "质量", "revyoy": "成长", "profyoy": "成长"}
    for r in ic_rows:
        f = r["f"]
        a, b = r["is"], r["oos"]
        ok = f in picked
        mk = ' style="background:#fff7ed"' if ok else ''
        flag = '<b class="ok">是</b>' if ok else '否'
        o.append(f'<tr{mk}><td><b>{f}</b></td><td>{CAT.get(f, "-")}</td>'
                 f'<td>{a["mean_dir"]:+.4f}</td><td>{a["t_dir"]:+.2f}</td>'
                 f'<td>{b["mean_dir"]:+.4f}</td><td>{b["t_dir"]:+.2f}</td>'
                 f'<td>{flag}</td></tr>')
    o.append('</tbody></table>')
    o.append(f'<div class="box info"><b>入选规则（先验）</b>：样本内方向 IC 的 t 值 ≥ {T_MIN} '
             f'且 IC 为正 —— 只在样本内看，样本外不参与。'
             f'入选：<b>{picked if picked else "无"}</b>。</div>')
    neg = [r for r in ic_rows if r["is"] and r["is"]["t_dir"] <= -T_MIN]
    if neg:
        lst = "、".join(f'<b>{r["f"]}</b>（t {r["is"]["t_dir"]:+.2f}）' for r in neg)
        o.append(f'<div class="box bad"><b>本轮最值得注意的发现：质量因子在本样本显著为负。</b>'
                 f'{lst} —— 即「高 ROE / 高毛利率 / 高净利率」在这段时间里<b>跑输</b>，'
                 f'与教科书理论方向相反。'
                 f'<br>可能解释：A 股高 ROE、高毛利的集中在白酒/医药等板块，'
                 f'2018-2026 恰好处在这些板块<b>杀估值</b>的过程里，'
                 f'质量信号被估值因子掩盖（价值与质量在 A 股常呈负相关）。'
                 f'<br><b>警告</b>：这些因子样本外的 t 值明显衰减'
                 f'（如 roe 从 {[r["is"]["t_dir"] for r in ic_rows if r["f"] == "roe"][0]:+.2f} 降到 '
                 f'{[r["oos"]["t_dir"] for r in ic_rows if r["f"] == "roe"][0]:+.2f}），'
                 f'说明<b>负向关系并不稳定</b>，不能据此反向建仓。</div>')

    # 三、合成方式对比
    o.append('<h2>三、合成方式对比（样本内）</h2>')
    o.append('<table><thead><tr><th>选股层</th><th>样本内超额</th><th>相对 BP 增量</th>'
             '<th>最大回撤</th><th>年换手</th></tr></thead><tbody>')
    for name, vv in is_rows.items():
        hi = ' style="background:#fff7ed"' if name == best_name else ''
        star = ' ★' if name == best_name else ''
        warn = ' <span style="font-size:11px;color:#b45309">（探索性：方向取自样本内 IC）</span>' \
            if ("带符号" in name) else ''
        o.append(f'<tr{hi}><td>{name}{star}{warn}</td>'
                 f'<td><b class="{cls(vv["excess"])}">{vv["excess"] * 100:+.2f}pp</b></td>'
                 f'<td><b class="{cls(vv["lift"])}">{vv["lift"] * 100:+.2f}pp</b></td>'
                 f'<td>{vv["mdd"] * 100:.1f}%</td><td>{vv["turn"] * 100:.0f}%</td></tr>')
    o.append('</tbody></table>')
    if use_multi:
        o.append(f'<div class="box ok"><b>选中 {best_name}</b>：样本内超额 '
                 f'{best["excess"] * 100:+.2f}pp，相对 BP 单因子增量 '
                 f'<b>{best["lift"] * 100:+.2f}pp</b>。</div>')
    else:
        o.append(f'<div class="box warn"><b>未切换</b>：样本内表现最好的是 {best_name}，'
                 f'但相对 BP 单因子的增量只有 {best["lift"] * 100:+.2f}pp，'
                 f'未达到 1pp 门槛 —— 按奥卡姆原则<b>不增加复杂度，仍用单因子 BP</b>。'
                 f'（这是一个负面结果：多因子在本样本没有带来可用增量。）</div>')

    # 四、全样本
    o.append('<h2>四、全样本对照</h2>')
    o.append('<table><thead><tr><th>策略</th><th>总收益</th><th>年化</th><th>相对基准</th>'
             '<th>最大回撤</th><th>Calmar</th><th>年换手</th></tr></thead><tbody>')
    for nm, r, hi in ((f"多因子合成（{'+'.join(final_names)}）", full_final, True),
                      ("BP 单因子（P2 基线）", full_bp, False)):
        mk = ' style="background:#fff7ed"' if hi else ''
        o.append(f'<tr{mk}><td><b>{nm}</b></td><td>{pc(r["ret"])}</td>'
                 f'<td>{r["raw"]["annual"] * 100:.2f}%</td>'
                 f'<td><b class="{cls(r["excess"])}">{r["excess"] * 100:+.2f}pp</b></td>'
                 f'<td>{r["raw"]["max_drawdown"] * 100:.1f}%</td>'
                 f'<td><b>{_calmar(r):.3f}</b></td><td>{r["turnover"] * 100:.0f}%</td></tr>')
    o.append(f'<tr><td>等权买入持有（基准）</td><td>{pc(bench_ret)}</td><td>—</td><td>—</td>'
             f'<td>{full_bp["raw"]["benchmark"]["mdd"] * 100:.1f}%</td><td>—</td><td>0%</td></tr>')
    o.append('</tbody></table>')

    # 五、样本外
    o.append(f'<h2>五、样本外检验（硬门槛：{OOS_CUTOFF} 之后）</h2>')
    o.append('<table><thead><tr><th>策略</th><th>样本外收益</th><th>样本外基准</th>'
             '<th>样本外超额</th></tr></thead><tbody>')
    for nm, r, hi in ((f"多因子合成（{'+'.join(final_names)}）", oos_final, True),
                      ("BP 单因子（P2 基线）", oos_bp, False)):
        mk = ' style="background:#fff7ed"' if hi else ''
        o.append(f'<tr{mk}><td><b>{nm}</b></td><td>{pc(r["ret"])}</td><td>{pc(r["bench"])}</td>'
                 f'<td><b class="{cls(r["excess"])}">{r["excess"] * 100:+.2f}pp</b></td></tr>')
    o.append('</tbody></table>')
    verdict = "样本外增量为正 —— 多因子确实带来了<b>可迁移</b>的 alpha。" if d_ex > 0 else \
              "样本外增量为负 —— 样本内的增量<b>不可迁移</b>，是过拟合。"
    o.append(f'<div class="box {"ok" if d_ex > 0 else "bad"}">'
             f'样本外增量 <b>{d_ex:+.2f}pp</b>（多因子 {oos_final["excess"] * 100:+.2f}pp vs '
             f'BP {oos_bp["excess"] * 100:+.2f}pp）。{verdict}'
             f'这是本次实验<b>唯一的决定性判据</b>，因为样本内增量总可以通过加因子刷出来。</div>')

    # 探索性对照：即使没被选中，也要看样本外，否则无法判断门槛是否设错
    if runner_oos is not None:
        rd = (runner_oos["excess"] - oos_bp["excess"]) * 100
        if not use_multi:
            note = (f'这是<b>对 1pp 门槛的敏感性检验</b>：若它在样本外明显优于 BP，'
                    f'说明门槛设得过严、多因子其实可用；反之说明奥卡姆原则的选择是对的。')
        else:
            note = ''
        o.append(f'<h3>5b. 探索性对照：样本内次优方案（{runner_name}）</h3>')
        o.append('<table><thead><tr><th>方案</th><th>样本内超额</th><th>相对 BP 增量</th>'
                 '<th>全样本收益</th><th>样本外超额</th><th>样本外增量</th></tr></thead><tbody>')
        o.append(f'<tr><td><b>{runner_name}</b></td>'
                 f'<td>{is_rows[runner_name]["excess"] * 100:+.2f}pp</td>'
                 f'<td><b class="{cls(is_rows[runner_name]["lift"])}">'
                 f'{is_rows[runner_name]["lift"] * 100:+.2f}pp</b></td>'
                 f'<td>{pc(runner_full["ret"])}</td>'
                 f'<td><b class="{cls(runner_oos["excess"])}">'
                 f'{runner_oos["excess"] * 100:+.2f}pp</b></td>'
                 f'<td><b class="{cls(rd / 100)}">{rd:+.2f}pp</b></td></tr>')
        o.append(f'<tr style="background:#fff7ed"><td>BP 单因子（已采用）</td>'
                 f'<td>{is_rows["BP 单因子（P2 基线）"]["excess"] * 100:+.2f}pp</td><td>—</td>'
                 f'<td>{pc(full_bp["ret"])}</td>'
                 f'<td>{oos_bp["excess"] * 100:+.2f}pp</td><td>—</td></tr>')
        o.append('</tbody></table>')
        o.append(f'<div class="box {"ok" if rd > 1.0 else ("warn" if rd > 0 else "bad")}">'
                 f'样本外增量 <b>{rd:+.2f}pp</b>。{note}'
                 f'{"<b>它在样本外确实更好</b> —— 1pp 门槛偏严，多因子是有价值的。" if rd > 1.0 else ""}'
                 f'{"增量很小（0~1pp），门槛的选择对结果影响不大。" if 0 < rd <= 1.0 else ""}'
                 f'{"<b>样本外反而不如 BP</b> —— 样本内那点增量是过拟合，奥卡姆原则的选择正确。" if rd <= 0 else ""}'
                 f'</div>')

    # 六、随机对照
    o.append('<h2>六、随机信号对照（照妖镜）</h2>')
    o.append(f'<div class="box warn">对照组与最终方案<b>完全相同</b>的合成方式、倾斜强度、'
             f'调仓频率与因子覆盖，唯一区别是把「哪个票拿哪个综合分数」随机打乱 —— '
             f'权重分布不变（换手量级相同），信号退化为噪声。</div>')
    o.append('<table><thead><tr><th>策略</th><th>年换手</th><th>总收益</th></tr></thead><tbody>')
    o.append(f'<tr style="background:#fff7ed"><td><b>多因子合成（真实信号）</b></td>'
             f'<td>{full_final["turnover"] * 100:.0f}%</td>'
             f'<td><b>{pc(full_final["ret"])}</b></td></tr>')
    for x in rand_rows[:10]:
        o.append(f'<tr><td>随机 #{x["seed"]}</td><td>{x["turn"] * 100:.0f}%</td>'
                 f'<td>{pc(x["ret"])}</td></tr>')
    o.append('</tbody></table>')
    if pct >= 80:
        pct_note = '（≥80：收益来自<b>因子信号</b>，而不是「倾斜」这个动作本身）'
    else:
        pct_note = '（偏低：收益有相当部分来自<b>倾斜机制</b>而非因子信号）'
    o.append(f'<div class="box {"ok" if pct >= 80 else ("warn" if pct >= 60 else "bad")}">'
             f'真实信号跑赢 <b>{beat}/{len(rand_rows)}</b> 个随机对照，位于第 <b>{pct}</b> 百分位。'
             f'{pct_note}</div>')

    # 七、成本
    o.append('<h2>七、成本敏感性</h2>')
    o.append('<table><thead><tr><th>双边滑点</th><th>策略收益</th><th>基准收益</th>'
             '<th>相对基准</th><th>交易笔数</th></tr></thead><tbody>')
    for x in cost_rows:
        o.append(f'<tr><td>{x["slip"] * 100:.2f}%</td><td>{pc(x["ret"])}</td>'
                 f'<td>{pc(x["bench"])}</td>'
                 f'<td><b class="{cls(x["ret"] - x["bench"])}">'
                 f'{(x["ret"] - x["bench"]) * 100:+.2f}pp</b></td><td>{x["n"]}</td></tr>')
    o.append('</tbody></table>')

    # 八、滚动
    o.append('<h2>八、滚动前推（多因子 vs BP vs 基准）</h2>')
    o.append('<table><thead><tr><th>区间</th><th>多因子</th><th>BP 单因子</th><th>基准</th>'
             '<th>多因子超额</th><th>BP 超额</th><th>增量</th></tr></thead><tbody>')
    for x in wf_rows:
        d = (x["excess"] - x["bp_excess"]) * 100
        o.append(f'<tr><td>{x["seg"]}</td><td>{x["ret"] * 100:+.2f}%</td>'
                 f'<td>{x["bp"] * 100:+.2f}%</td><td>{x["bench"] * 100:+.2f}%</td>'
                 f'<td><b class="{cls(x["excess"])}">{x["excess"] * 100:+.2f}pp</b></td>'
                 f'<td>{x["bp_excess"] * 100:+.2f}pp</td>'
                 f'<td><b class="{cls(d / 100)}">{d:+.2f}pp</b></td></tr>')
    o.append('</tbody></table>')
    nseg_win = sum(1 for x in wf_rows if (x["excess"] - x["bp_excess"]) > 0)
    o.append(f'<div class="box {"ok" if nseg_win >= 3 else "warn"}">'
             f'多因子在 <b>{nseg_win}/4</b> 段跑赢 BP 单因子；'
             f'相对基准 {wins}/4 段为正。</div>')

    # 九、叠加风控
    o.append('<h2>九、叠加 P4 风控层（可落地的完整方案）</h2>')
    o.append('<table><thead><tr><th>方案</th><th>全样本收益</th><th>最大回撤</th><th>Calmar</th>'
             '<th>样本外超额</th></tr></thead><tbody>')
    o.append(f'<tr style="background:#fff7ed"><td><b>多因子 + 波动率目标 25%</b></td>'
             f'<td>{pc(full_vol["ret"])}</td><td>{full_vol["raw"]["max_drawdown"] * 100:.1f}%</td>'
             f'<td><b>{_calmar(full_vol):.3f}</b></td>'
             f'<td><b class="{cls(oos_vol["excess"])}">{oos_vol["excess"] * 100:+.2f}pp</b></td></tr>')
    o.append(f'<tr><td>多因子（无风控）</td><td>{pc(full_final["ret"])}</td>'
             f'<td>{full_final["raw"]["max_drawdown"] * 100:.1f}%</td>'
             f'<td>{_calmar(full_final):.3f}</td>'
             f'<td>{oos_final["excess"] * 100:+.2f}pp</td></tr>')
    o.append(f'<tr><td>BP 单因子（无风控，P2 基线）</td><td>{pc(full_bp["ret"])}</td>'
             f'<td>{full_bp["raw"]["max_drawdown"] * 100:.1f}%</td>'
             f'<td>{_calmar(full_bp):.3f}</td>'
             f'<td>{oos_bp["excess"] * 100:+.2f}pp</td></tr>')
    o.append('</tbody></table>')

    # 十、结论
    o.append('<h2>十、结论</h2>')
    o.append('<div class="box warn"><h3 style="margin-top:0">判定</h3><ul>')
    o.append(f'<li><b>多因子是否有样本内增量</b>：{best["lift"] * 100:+.2f}pp'
             f'（{best_name} vs BP 单因子）。</li>')
    o.append(f'<li><b>增量是否可迁移（决定性）</b>：样本外增量 <b>{d_ex:+.2f}pp</b>'
             f'{"—— 可迁移，多因子成立。" if d_ex > 0 else "—— 不可迁移，样本内增量是过拟合。"}</li>')
    o.append(f'<li><b>是否只是"倾斜"动作的功劳</b>：随机对照第 <b>{pct}</b> 百分位'
             f'{"（是因子信号而非倾斜机制）" if pct >= 80 else "（收益有相当部分来自倾斜机制本身）"}。</li>')
    o.append(f'<li><b>是否稳健</b>：滚动 {nseg_win}/4 段跑赢 BP；'
             f'滑点 0.5% 时仍 {pc(cost_rows[-1]["ret"])}。</li>')
    o.append(f'<li><b>诚实边界</b>：因子筛选与权重均用样本内 IC 决定；'
             f'因子间相关性未做正交化处理（多重共线性会让"多因子"实际只贡献一两个自由度）；'
             f'幸存者偏差、仅一轮牛熊仍是硬约束。</li>')
    o.append('</ul></div>')

    o.append(f'<div class="box {"ok" if d_ex > 0 else "bad"}"><b>一句话</b>：'
             + ('多因子合成在本样本<b>确实把选股 alpha 抬高了</b>，且抬高的部分能迁移到样本外 —— '
                '选股层还有空间，P4「线抬不上去」的结论只适用于风控层。' if d_ex > 0 else
                '多因子合成在本样本<b>没有带来可迁移的增量</b> —— '
                '结合 P4 的结论，这条路（加因子 + 换风控）已经接近本框架的边际。')
             + '</div>')

    o.append('<div class="sub" style="margin-top:22px">AQuant v5 P5 ｜ '
             '数据源：东方财富数据中心（point-in-time）｜ 不构成投资建议</div>')
    o.append('</div></body></html>')
    return "".join(o)


if __name__ == "__main__":
    main()
