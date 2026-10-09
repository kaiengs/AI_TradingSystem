#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AQuant v5 · P6：系统扫描「因子的用法」
======================================
P5 的结论：加因子这条路到头了 —— 7 个因子只有 bp 入选，多因子合成没有可迁移增量，
次优方案的样本内 +0.40pp 在样本外翻成 -0.50pp（过拟合）。

但有一条线索一直指着别的方向：**同一因子的不同用法，差距远大于换因子**。
  * P1：bp 做 Top-20 集中        -> alpha **-17.27pp**
  * P2：bp 做「保留分散度的倾斜」 -> alpha **+30.45pp**
  同一个因子、同一段数据，差 **47.72pp**。

所以 P6 不换因子（固定 bp），只系统扫描**怎么用**：
    tau（倾斜锐度） / floor（与等权混合） / max_weight（单票上限）
    min_hold（最短持有） / band（再平衡带宽） / rebalance（调仓频率）

选点规则与 P4 的 R4 **同源**（保持系列方法论一致）：
  * 目标函数 = α-Calmar（年化超额 / 最大回撤）—— 用超额口径是为了排除 beta 的干扰；
  * 护栏 = 样本内超额 > 0 —— 用法再怎么调也不能把信号毁成负数；
  * 邻域最坏值 —— 取该轴上相邻档中 α-Calmar 的最小值再排序，防样本内过拟合尖峰。

纪律：所有参数只在样本内挑选；样本外（2023-01-01 之后）全程不参与任何决策；
      随机信号对照 + 成本敏感性 + 滚动前推三重验证；
      每轴都报告**年换手** —— 用法的变化首先体现在交易频率上，忽略换手等于自欺。
"""

import sys, os, time
sys.path.insert(0, "/workspace/quant")
from optimize import (fill_to_axis, CASH, CSS, cls, pc)
from aq import data, strategies as st, backtest as bt
from aq import fundamentals as fnd
from fund_ic import (month_starts, to_dict, build_panel,
                     audit_point_in_time, ic_agg_fund, run_fund)

OUT = "/workspace/quant/因子用法扫描_P6.html"
OOS_CUTOFF = "2023-01-01"
WARM = 300
FNAME = "bp"          # 因子固定不动（P1~P5 的结论）

# 基线 = P2 选出的用法（当前最佳）
BASE = {"tau": 6.0, "floor": 0.0, "max_weight": 0.0,
        "min_hold": 0, "band": 0.0, "freq": "quarterly"}

# 六个用法轴（逐轴扫描，其余轴固定在基线/已选值上）
AXES = [
    ("tau", "倾斜锐度 tau", "越小越向高分票集中（tau→∞ 退化为等权宽基）",
     [3.0, 4.5, 6.0, 9.0, 12.0, 18.0, 25.0, 40.0], "{:.1f}"),
    ("floor", "等权混合 floor", "0=纯倾斜，1=完全等权（稀释信号但降低集中度）",
     [0.0, 0.15, 0.30, 0.45, 0.60, 0.80], "{:.2f}"),
    ("max_weight", "单票权重上限", "硬性压制单票集中度（重仓踩雷保护）",
     [0.0, 0.03, 0.05, 0.07, 0.10, 0.15], "{:.2f}"),
    ("min_hold", "最短持有天数", "买入后至少持有 N 天才允许减仓（抑制来回交易）",
     [0, 10, 20, 40, 80, 160], "{:d}"),
    ("band", "再平衡带宽", "目标权重偏离超过此比例才交易（过滤小调仓）",
     [0.0, 0.05, 0.10, 0.20, 0.30], "{:.2f}"),
    ("freq", "调仓频率", "多久重新计算一次目标权重",
     ["monthly", "quarterly", "semiannual"], "{}"),
]


def cfg_factory(c):
    """由用法配置生成 (factory, engine_kw) —— 用默认参数绑定避免闭包陷阱"""
    def _f(tau=c["tau"], floor=c["floor"]):
        return lambda: st.fund_tilt(FNAME, tau=tau, floor=floor)
    kw = {"rebalance": c["freq"], "max_weight": c["max_weight"],
          "min_hold": c["min_hold"], "band": c["band"]}
    return _f(), kw


def main():
    t_all = time.time()
    log = lambda *a: print(*a, flush=True)
    log("=" * 70)
    log("AQuant v5 · P6：系统扫描「因子的用法」（因子固定为 bp）")
    log("=" * 70)

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

    # ---------- 3. 信号复核 ----------
    log(f"\n[3] 复核 {FNAME} 因子 IC（季频 63 日）...")
    ic_is = ic_agg_fund(panel, Ad, dates, FNAME, 63, reb, 0, is_end - 1)
    ic_oos = ic_agg_fund(panel, Ad, dates, FNAME, 63, reb, is_end, n - 1)
    if ic_is:
        log(f"  样本内 方向IC {ic_is['mean_dir']:+.4f}  t {ic_is['t_dir']:+.2f}")
    if ic_oos:
        log(f"  样本外 方向IC {ic_oos['mean_dir']:+.4f}  t {ic_oos['t_dir']:+.2f}")

    yrs_is = max((is_end - 1 - WARM) / 252.0, 1e-6)
    cache = {}

    def run_is(c):
        key = tuple(sorted(c.items()))
        if key in cache:
            return cache[key]
        f, kw = cfg_factory(c)
        r = run_fund(A, dates, mkt, WARM, is_end - 1, f, fund, **kw)
        cg = (max(1.0 + r["ret"], 1e-9)) ** (1.0 / yrs_is) - 1
        cb = (max(1.0 + r["bench"], 1e-9)) ** (1.0 / yrs_is) - 1
        mdd = r["raw"]["max_drawdown"]
        out = {"ret": r["ret"], "excess": r["excess"], "mdd": mdd,
               "turn": r["turnover"], "n": r["n"], "cagr_ex": cg - cb,
               "acal": (cg - cb) / abs(mdd) if mdd else 0.0}
        cache[key] = out
        return out

    def pick_axis(key, vals, others):
        """在一个轴上按 R4 同源规则选点：邻域最坏 α-Calmar 最大 + 超额>0 护栏"""
        rows = []
        for v in vals:
            c = dict(others)
            c[key] = v
            rows.append({"v": v, "c": c, **run_is(c)})
        for i, rw in enumerate(rows):
            nb = [rows[j]["acal"] for j in (i - 1, i, i + 1) if 0 <= j < len(rows)]
            rw["acal_nb"] = min(nb)
        ok = [r for r in rows if r["excess"] > 0] or rows
        bestp = max(ok, key=lambda r: (r["acal_nb"], r["excess"]))
        return rows, bestp

    # ---------- 4. 逐轴扫描 ----------
    log("\n[4] 逐轴扫描（目标 α-Calmar，邻域最坏值 + 超额>0 护栏）...")
    base_row = run_is(BASE)
    log(f"  基线用法（P2）: 超额 {base_row['excess'] * 100:+.2f}pp  "
        f"回撤 {base_row['mdd'] * 100:.1f}%  α-Calmar {base_row['acal']:.3f}  "
        f"换手 {base_row['turn'] * 100:.0f}%")

    cur = dict(BASE)
    axis_rows = {}
    for key, name, desc, vals, fmt in AXES:
        rows, bestp = pick_axis(key, vals, cur)
        axis_rows[key] = {"name": name, "desc": desc, "fmt": fmt,
                          "rows": rows, "best": bestp, "base_v": BASE[key]}
        log(f"\n  —— {name}（{desc}）——")
        log(f"     {'值':>8} {'超额':>10} {'α-Calmar':>10} {'邻域最坏':>9} {'回撤':>8} {'换手':>7}")
        for rw in rows:
            star = " *" if rw is bestp else ""
            log(f"     {fmt.format(rw['v']):>8} {rw['excess'] * 100:>+9.2f}pp "
                f"{rw['acal']:>10.3f} {rw['acal_nb']:>9.3f} "
                f"{rw['mdd'] * 100:>7.1f}% {rw['turn'] * 100:>6.0f}%{star}")
        cur[key] = bestp["v"]
        log(f"     -> 选 {fmt.format(bestp['v'])}"
            f"（年换手 {bestp['turn'] * 100:.0f}%，真实边际贡献见第九节组合检验）")

    FINAL_CFG = dict(cur)
    final_row = run_is(FINAL_CFG)
    lift_is = (final_row["excess"] - base_row["excess"]) * 100
    log(f"\n  >> 组合后用法: 超额 {final_row['excess'] * 100:+.2f}pp"
        f"（基线 {base_row['excess'] * 100:+.2f}pp，样本内增量 {lift_is:+.2f}pp）")
    log(f"     回撤 {final_row['mdd'] * 100:.1f}%  α-Calmar {final_row['acal']:.3f}  "
        f"换手 {final_row['turn'] * 100:.0f}%")
    log("     注：逐轴贪心不等于全局最优，各轴收益也不一定可叠加 —— 见第九节组合检验")

    # 真实边际贡献：保持其余轴在最终配置，仅把本轴回退到基线，衡量本轴的独立增量
    # （不能用 扫描时的 best_excess - base_row，因为 pick_axis 扫描时 cur 是累积贪心配置，
    #  后扫描的轴会把前轴收益也算进自己头上 —— 见第九节组合检验的对照）
    axis_marg = {}
    for key, name, *_ in AXES:
        cfg_off = dict(FINAL_CFG)
        cfg_off[key] = BASE[key]
        r_off = run_is(cfg_off)
        axis_marg[key] = (final_row["excess"] - r_off["excess"]) * 100
    log("\n  各轴真实边际贡献（其余轴固定为最终配置，仅本轴回退基线）:")
    for key, name, *_ in AXES:
        tag = " *非基线" if FINAL_CFG[key] != BASE[key] else ""
        log(f"    {name}: {axis_marg[key]:+.2f}pp{tag}")

    final_f, final_kw = cfg_factory(FINAL_CFG)
    base_f, base_kw = cfg_factory(BASE)

    # ---------- 5. 全样本 ----------
    log("\n[5] 全样本 ...")
    full_final = run_fund(A, dates, mkt, WARM, n - 1, final_f, fund, **final_kw)
    full_base = run_fund(A, dates, mkt, WARM, n - 1, base_f, fund, **base_kw)
    bench_ret = full_base["bench"]
    log(f"  优化用法 {full_final['ret'] * 100:+.2f}%  回撤 {full_final['raw']['max_drawdown'] * 100:.1f}%  "
        f"换手 {full_final['turnover'] * 100:.0f}%")
    log(f"  基线(P2) {full_base['ret'] * 100:+.2f}%  回撤 {full_base['raw']['max_drawdown'] * 100:.1f}%  "
        f"换手 {full_base['turnover'] * 100:.0f}%")
    log(f"  基准 {bench_ret * 100:+.2f}%")

    # ---------- 6. 样本外 ----------
    log(f"\n[6] 样本外检验（>= {OOS_CUTOFF}）...")
    oos_final = run_fund(A, dates, mkt, is_end, n - 1, final_f, fund, **final_kw)
    oos_base = run_fund(A, dates, mkt, is_end, n - 1, base_f, fund, **base_kw)
    d_ex = (oos_final["excess"] - oos_base["excess"]) * 100
    log(f"  样本外超额 优化用法 {oos_final['excess'] * 100:+.2f}pp / "
        f"基线 {oos_base['excess'] * 100:+.2f}pp  -> 增量 {d_ex:+.2f}pp")

    # ---------- 7. 随机对照 ----------
    log("\n[7] 随机信号对照（20 次，同用法）...")
    rand_rows = []
    for i in range(20):
        def rf(seed=i, tau=FINAL_CFG["tau"], floor=FINAL_CFG["floor"]):
            return lambda: st.rand_fund_tilt(seed, FNAME, tau=tau, floor=floor)
        r = run_fund(A, dates, mkt, WARM, n - 1, rf(), fund, **final_kw)
        rand_rows.append({"seed": i, "ret": r["ret"], "turn": r["turnover"]})
    beat = sum(1 for x in rand_rows if x["ret"] < full_final["ret"])
    pct = int(beat / len(rand_rows) * 100)
    log(f"  跑赢 {beat}/{len(rand_rows)} 个随机 -> 第 {pct} 百分位")

    # ---------- 8. 成本 ----------
    log("\n[8] 成本敏感性 ...")
    cost_rows = []
    for slip in (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005):
        r = run_fund(A, dates, mkt, WARM, n - 1, final_f, fund,
                     cost=bt.Cost(slippage=slip), **final_kw)
        cost_rows.append({"slip": slip, "ret": r["ret"], "bench": r["bench"], "n": r["n"]})
        log(f"  滑点 {slip * 100:.2f}% -> {r['ret'] * 100:+8.2f}% (基准 {r['bench'] * 100:+.2f}%)")

    # ---------- 9. 滚动前推 ----------
    log("\n[9] 滚动前推 ...")
    segs, L = [], n // 4
    for i in range(4):
        s0 = max(WARM, i * L)
        s1 = n - 1 if i == 3 else (i + 1) * L - 1
        segs.append((s0, s1))
    wf_rows, wins = [], 0
    for (s0, s1) in segs:
        r = run_fund(A, dates, mkt, s0, s1, final_f, fund, **final_kw)
        rb = run_fund(A, dates, mkt, s0, s1, base_f, fund, **base_kw)
        wf_rows.append({"seg": f"{dates[s0]}~{dates[s1]}", "ret": r["ret"], "bench": r["bench"],
                        "excess": r["excess"], "base": rb["ret"], "base_excess": rb["excess"],
                        "turn": r["turnover"]})
        if r["excess"] > 0:
            wins += 1
        log(f"  {dates[s0]}~{dates[s1]}  优化 {r['ret'] * 100:+7.2f}%  基线 {rb['ret'] * 100:+7.2f}%  "
            f"基准 {r['bench'] * 100:+7.2f}%  超额 {r['excess'] * 100:+7.2f}pp  "
            f"换手 {r['turnover'] * 100:>5.0f}%")

    # ---------- 10. 报告 ----------
    log("\n[10] 生成报告 ...")
    v = {
        "A": A, "dates": dates, "n": n, "yrs": yrs, "is_end": is_end,
        "bad": bad, "tot": tot, "ic_is": ic_is, "ic_oos": ic_oos,
        "BASE": BASE, "AXES": AXES, "axis_rows": axis_rows, "axis_marg": axis_marg,
        "FINAL_CFG": FINAL_CFG, "base_row": base_row, "final_row": final_row,
        "lift_is": lift_is, "d_ex": d_ex,
        "full_final": full_final, "full_base": full_base, "bench_ret": bench_ret,
        "oos_final": oos_final, "oos_base": oos_base,
        "rand_rows": rand_rows, "beat": beat, "pct": pct,
        "cost_rows": cost_rows, "wf_rows": wf_rows, "wins": wins,
        "OOS_CUTOFF": OOS_CUTOFF, "FNAME": FNAME,
    }
    html = build_html(v)
    open(OUT, "w", encoding="utf-8").write(html)
    log(f"  输出 {OUT} ({len(html.encode())} 字节)")
    log(f"\n[总耗时] {time.time() - t_all:.1f}s")


def build_html(v):
    A, dates, n, yrs = v["A"], v["dates"], v["n"], v["yrs"]
    bad, tot = v["bad"], v["tot"]
    ic_is, ic_oos = v["ic_is"], v["ic_oos"]
    BASE, AXES, axis_rows = v["BASE"], v["AXES"], v["axis_rows"]
    axis_marg = v["axis_marg"]
    FINAL_CFG = v["FINAL_CFG"]
    base_row, final_row, lift_is = v["base_row"], v["final_row"], v["lift_is"]
    d_ex = v["d_ex"]
    full_final, full_base = v["full_final"], v["full_base"]
    bench_ret = v["bench_ret"]
    oos_final, oos_base = v["oos_final"], v["oos_base"]
    rand_rows, beat, pct = v["rand_rows"], v["beat"], v["pct"]
    cost_rows, wf_rows, wins = v["cost_rows"], v["wf_rows"], v["wins"]
    OOS_CUTOFF, FNAME = v["OOS_CUTOFF"], v["FNAME"]

    def _calmar(r):
        m = abs(r["raw"]["max_drawdown"])
        return r["raw"]["annual"] / m if m > 1e-9 else 0.0

    o = ['<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>AQuant v5 P6 · 因子用法扫描</title><style>', CSS, '</style></head>',
         '<body><div class="wrap">']
    o.append('<h1>AQuant v5 · P6：系统扫描「因子的用法」</h1>')
    o.append(f'<div class="sub">因子固定为 <code>{FNAME}</code>，只改用法 ｜ '
             f'样本 {len(A)} 只 × {yrs:.1f} 年（{dates[0]} ~ {dates[-1]}）｜ '
             f'本金 {CASH / 1e8:.0f} 亿 ｜ 数据源：东方财富（point-in-time）</div>')

    o.append('<div class="kpi">')
    o.append(f'<div><span>样本内增量</span><b class="{cls(lift_is / 100)}">'
             f'{lift_is:+.2f}pp</b><span style="font-size:11px">组合用法 vs P2 基线</span></div>')
    o.append(f'<div><span>样本外增量</span><b class="{cls(d_ex / 100)}">'
             f'{d_ex:+.2f}pp</b><span style="font-size:11px">决定性判据</span></div>')
    turn_cls = "ok" if final_row["turn"] <= base_row["turn"] else "warn"
    o.append(f'<div><span>年换手变化</span><b class="{turn_cls}">'
             f'{(final_row["turn"] - base_row["turn"]) * 100:+.0f}pp</b>'
             f'<span style="font-size:11px">{final_row["turn"] * 100:.0f}% vs '
             f'{base_row["turn"] * 100:.0f}%</span></div>')
    pct_cls = "ok" if pct >= 80 else ("warn" if pct >= 60 else "bad")
    o.append(f'<div><span>随机信号对照</span><b class="{pct_cls}">'
             f'第 {pct} 百分位</b><span style="font-size:11px">同用法对照</span></div>')
    o.append('</div>')

    # 一、动机
    o.append('<h2>一、为什么转而扫描「用法」</h2>')
    o.append('<div class="box info"><b>P5 的结论</b>：7 个因子只有 bp 入选，'
             '多因子合成没有可迁移增量 —— 次优方案样本内 +0.40pp，样本外 <b>-0.50pp</b>（过拟合）。'
             '<b>加因子这条路到头了。</b><br>'
             '<b>但同一因子的不同用法，差距远大于换因子</b>：'
             '<br>· P1：bp 做 Top-20 <b>集中</b> → alpha <b>-17.27pp</b>'
             '<br>· P2：bp 做<b>保留分散度的倾斜</b> → alpha <b>+30.45pp</b>'
             '<br>同一个因子、同一段数据，差 <b>47.72pp</b>。'
             '所以 P6 固定因子，只系统扫描怎么用。</div>')

    # 二、扫描设计
    o.append('<h2>二、扫描设计</h2>')
    o.append('<table><thead><tr><th>轴</th><th>含义</th><th>取值</th><th>基线</th></tr></thead><tbody>')
    for key, name, desc, vals, fmt in AXES:
        o.append(f'<tr><td><b>{name}</b><br><code>{key}</code></td>'
                 f'<td style="text-align:left">{desc}</td>'
                 f'<td>{"、".join(fmt.format(x) for x in vals)}</td>'
                 f'<td><b>{fmt.format(BASE[key])}</b></td></tr>')
    o.append('</tbody></table>')
    o.append('<div class="box info"><b>选点规则（与 P4 的 R4 同源，保持系列方法论一致）</b>：'
             '<br>① 目标函数 = <code>α-Calmar</code>（年化超额 / 最大回撤）—— '
             '用超额口径是为了排除 beta 干扰；'
             '<br>② 护栏 = 样本内超额 &gt; 0 —— 用法再怎么调也不能把信号毁成负数；'
             '<br>③ 邻域最坏值 = 取该轴上<b>相邻档</b>中 α-Calmar 的最小值再排序，防样本内过拟合尖峰。'
             '<br><b>纪律</b>：所有参数只用样本内挑选；样本外（'
             f'{OOS_CUTOFF} 之后）全程不参与任何决策。</div>')
    o.append(f'<div class="box {"ok" if bad == 0 else "bad"}"><b>前视偏差审计</b>：面板 {tot} 条快照中 '
             f'<b>{bad} 条</b>越界（须为 0）。</div>')
    if ic_is and ic_oos:
        o.append(f'<div class="box info"><b>信号复核（{FNAME}）</b>：样本内方向IC '
                 f'<b>{ic_is["mean_dir"]:+.4f}</b>（t {ic_is["t_dir"]:+.2f}），样本外 '
                 f'<b>{ic_oos["mean_dir"]:+.4f}</b>（t {ic_oos["t_dir"]:+.2f}）。'
                 f'信号在整个扫描过程中<b>固定不变</b>，变的只是用法。</div>')

    # 三~八：逐轴
    sec = 3
    CN = ["三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二", "十三", "十四"]
    for key, name, desc, vals, fmt in AXES:
        ar = axis_rows[key]
        o.append(f'<h2>{CN[sec - 3]}、{name}（<code>{key}</code>）</h2>')
        o.append(f'<p>{desc}。其余轴固定在当前已选值上。</p>')
        o.append('<table><thead><tr><th>取值</th><th>样本内超额</th><th>α-Calmar</th>'
                 '<th>邻域最坏</th><th>最大回撤</th><th>年换手</th></tr></thead><tbody>')
        for rw in ar["rows"]:
            isbest = rw is ar["best"]
            isbase = rw["v"] == ar["base_v"]
            if isbest:
                mk = ' style="background:#fff7ed"'
            elif isbase:
                mk = ' style="background:#f0f7ff"'
            else:
                mk = ''
            tag = ' ★选中' if isbest else ''
            if isbase:
                tag += ' <span style="font-size:11px">基线</span>'
            o.append(f'<tr{mk}><td><b>{fmt.format(rw["v"])}</b>{tag}</td>'
                     f'<td><b class="{cls(rw["excess"])}">{rw["excess"] * 100:+.2f}pp</b></td>'
                     f'<td>{rw["acal"]:.3f}</td><td><b>{rw["acal_nb"]:.3f}</b></td>'
                     f'<td>{rw["mdd"] * 100:.1f}%</td><td>{rw["turn"] * 100:.0f}%</td></tr>')
        o.append('</tbody></table>')
        b = ar["best"]
        same = b["v"] == ar["base_v"]
        marg = axis_marg[key]
        if same:
            boxcls, tail_txt = "info", "（与基线相同 —— 该轴上基线已是最优点，无独立改进）"
        else:
            boxcls = "ok" if marg > 0 else "warn"
            tail_txt = f'（在最终组合语境下的真实边际贡献 {marg:+.2f}pp）'
        o.append(f'<div class="box {boxcls}">选中 <b>{fmt.format(b["v"])}</b>{tail_txt}'
                 f'。年换手 <b>{b["turn"] * 100:.0f}%</b>（基线 {base_row["turn"] * 100:.0f}%）。</div>')
        sec += 1

    # 九、组合检验
    o.append(f'<h2>{CN[sec - 3]}、组合起来：逐轴最优能否叠加</h2>')
    o.append('<table><thead><tr><th>配置</th><th>样本内超额</th><th>α-Calmar</th>'
             '<th>最大回撤</th><th>年换手</th></tr></thead><tbody>')
    o.append(f'<tr style="background:#fff7ed"><td><b>逐轴最优组合</b></td>'
             f'<td><b class="{cls(final_row["excess"])}">{final_row["excess"] * 100:+.2f}pp</b></td>'
             f'<td>{final_row["acal"]:.3f}</td><td>{final_row["mdd"] * 100:.1f}%</td>'
             f'<td>{final_row["turn"] * 100:.0f}%</td></tr>')
    o.append(f'<tr style="background:#f0f7ff"><td>P2 基线用法</td>'
             f'<td><b class="{cls(base_row["excess"])}">{base_row["excess"] * 100:+.2f}pp</b></td>'
             f'<td>{base_row["acal"]:.3f}</td><td>{base_row["mdd"] * 100:.1f}%</td>'
             f'<td>{base_row["turn"] * 100:.0f}%</td></tr>')
    o.append('</tbody></table>')
    sum_marg = sum(axis_marg[k] for k, *_ in AXES)
    cant_add = abs(lift_is - sum_marg) > 1.0
    o.append(f'<div class="box {"warn" if cant_add else "info"}">'
             f'<b>叠加性检验（真实边际分解）</b>：各轴边际贡献之和 = <b>{sum_marg:+.2f}pp</b>，'
             f'实际组合后相对基线的总增量 = <b>{lift_is:+.2f}pp</b>。'
             f'{"两者相差超过 1pp，说明<b>各轴的收益不可叠加</b> —— 存在交互效应，某一轴的收益依赖另一轴的取值（例如 tau 更锐只在月频下才有效）。" if cant_add else "两者接近，说明<b>各轴收益基本可叠加</b>，逐轴贪心没有严重漏掉交互。"}'
             f'注：这里每个轴的边际贡献都独立测量 —— 保持其余轴在最终配置、仅把本轴回退到基线后损失的超额，'
             f'因此不存在「后扫描的轴把前轴收益算进自己头上」的污染（旧的 best_excess−base_row 度量有此缺陷，已弃用）。</div>')
    cfg_txt = "、".join(f'<code>{k}={fmt.format(FINAL_CFG[k])}</code>'
                       for k, _, _, _, fmt in AXES)
    o.append(f'<div class="box info"><b>最终用法配置</b>：{cfg_txt}</div>')
    sec += 1

    # 十、全样本 + 样本外
    o.append(f'<h2>{CN[sec - 3]}、全样本与样本外</h2>')
    o.append('<table><thead><tr><th>策略</th><th>总收益</th><th>年化</th><th>相对基准</th>'
             '<th>最大回撤</th><th>Calmar</th><th>年换手</th></tr></thead><tbody>')
    for nm, r, hi in (("优化用法", full_final, True), ("P2 基线用法", full_base, False)):
        mk = ' style="background:#fff7ed"' if hi else ''
        o.append(f'<tr{mk}><td><b>{nm}</b></td><td>{pc(r["ret"])}</td>'
                 f'<td>{r["raw"]["annual"] * 100:.2f}%</td>'
                 f'<td><b class="{cls(r["excess"])}">{r["excess"] * 100:+.2f}pp</b></td>'
                 f'<td>{r["raw"]["max_drawdown"] * 100:.1f}%</td>'
                 f'<td><b>{_calmar(r):.3f}</b></td><td>{r["turnover"] * 100:.0f}%</td></tr>')
    o.append(f'<tr><td>等权买入持有（基准）</td><td>{pc(bench_ret)}</td><td>—</td><td>—</td>'
             f'<td>{full_base["raw"]["benchmark"]["mdd"] * 100:.1f}%</td><td>—</td><td>0%</td></tr>')
    o.append('</tbody></table>')
    o.append(f'<h3>样本外（硬门槛：{OOS_CUTOFF} 之后）</h3>')
    o.append('<table><thead><tr><th>策略</th><th>样本外收益</th><th>样本外基准</th>'
             '<th>样本外超额</th></tr></thead><tbody>')
    for nm, r, hi in (("优化用法", oos_final, True), ("P2 基线用法", oos_base, False)):
        mk = ' style="background:#fff7ed"' if hi else ''
        o.append(f'<tr{mk}><td><b>{nm}</b></td><td>{pc(r["ret"])}</td><td>{pc(r["bench"])}</td>'
                 f'<td><b class="{cls(r["excess"])}">{r["excess"] * 100:+.2f}pp</b></td></tr>')
    o.append('</tbody></table>')
    if d_ex > 0:
        verdict = '样本外增量为正 —— 用法的改进是<b>可迁移</b>的。'
    else:
        verdict = '样本外增量为负 —— 用法的改进<b>不可迁移</b>，样本内的提升是调参调出来的。'
    o.append(f'<div class="box {"ok" if d_ex > 0 else "bad"}">样本外增量 <b>{d_ex:+.2f}pp</b>'
             f'（优化用法 {oos_final["excess"] * 100:+.2f}pp vs 基线 '
             f'{oos_base["excess"] * 100:+.2f}pp）。{verdict}'
             f'这是<b>决定性判据</b>：用法参数比因子更容易被调过拟合，'
             f'所以样本内的提升必须拿到样本外才算数。</div>')
    sec += 1

    # 十一、随机对照
    o.append(f'<h2>{CN[sec - 3]}、随机信号对照（照妖镜）</h2>')
    o.append(f'<div class="box warn">对照组与最终配置<b>完全相同</b>的倾斜锐度、'
             f'等权混合比例、单票上限、最短持有期、再平衡带宽、调仓频率，'
             f'唯一区别是把「哪个票拿哪个权重」随机打乱 —— '
             f'权重分布不变（换手量级相同），信号退化为噪声。</div>')
    o.append('<table><thead><tr><th>策略</th><th>年换手</th><th>总收益</th></tr></thead><tbody>')
    o.append(f'<tr style="background:#fff7ed"><td><b>优化用法（真实信号）</b></td>'
             f'<td>{full_final["turnover"] * 100:.0f}%</td>'
             f'<td><b>{pc(full_final["ret"])}</b></td></tr>')
    for x in rand_rows[:10]:
        o.append(f'<tr><td>随机 #{x["seed"]}</td><td>{x["turn"] * 100:.0f}%</td>'
                 f'<td>{pc(x["ret"])}</td></tr>')
    o.append('</tbody></table>')
    pct_note = ('（≥80：收益来自<b>因子信号</b>）' if pct >= 80
                else '（偏低：收益有相当部分来自<b>用法结构</b>而非因子信号）')
    o.append(f'<div class="box {"ok" if pct >= 80 else ("warn" if pct >= 60 else "bad")}">'
             f'真实信号跑赢 <b>{beat}/{len(rand_rows)}</b> 个随机对照，位于第 <b>{pct}</b> 百分位。'
             f'{pct_note}</div>')
    sec += 1

    # 十二、成本
    o.append(f'<h2>{CN[sec - 3]}、成本敏感性</h2>')
    o.append('<table><thead><tr><th>双边滑点</th><th>策略收益</th><th>基准收益</th>'
             '<th>相对基准</th><th>交易笔数</th></tr></thead><tbody>')
    for x in cost_rows:
        dv = x["ret"] - x["bench"]
        o.append(f'<tr><td>{x["slip"] * 100:.2f}%</td><td>{pc(x["ret"])}</td>'
                 f'<td>{pc(x["bench"])}</td>'
                 f'<td><b class="{cls(dv)}">{dv * 100:+.2f}pp</b></td><td>{x["n"]}</td></tr>')
    o.append('</tbody></table>')
    sec += 1

    # 十三、滚动
    o.append(f'<h2>{CN[sec - 3]}、滚动前推</h2>')
    o.append('<table><thead><tr><th>区间</th><th>优化用法</th><th>P2 基线</th><th>基准</th>'
             '<th>超额</th><th>增量</th><th>年换手</th></tr></thead><tbody>')
    segwin = 0
    for x in wf_rows:
        d = (x["excess"] - x["base_excess"]) * 100
        if d > 0:
            segwin += 1
        o.append(f'<tr><td>{x["seg"]}</td><td>{x["ret"] * 100:+.2f}%</td>'
                 f'<td>{x["base"] * 100:+.2f}%</td><td>{x["bench"] * 100:+.2f}%</td>'
                 f'<td><b class="{cls(x["excess"])}">{x["excess"] * 100:+.2f}pp</b></td>'
                 f'<td><b class="{cls(d / 100)}">{d:+.2f}pp</b></td>'
                 f'<td>{x["turn"] * 100:.0f}%</td></tr>')
    o.append('</tbody></table>')
    o.append(f'<div class="box {"ok" if segwin >= 3 else "warn"}">'
             f'优化用法在 <b>{segwin}/4</b> 段跑赢 P2 基线；相对基准 {wins}/4 段为正。</div>')
    sec += 1

    # 选到非基线值的轴（用于结论计数）
    lifted = [k for k, *_ in AXES if FINAL_CFG[k] != BASE[k]]

    # 十四、结论
    o.append(f'<h2>{CN[sec - 3]}、结论</h2>')
    o.append('<div class="box warn"><h3 style="margin-top:0">判定</h3><ul>')
    o.append(f'<li><b>用法是否有改进空间</b>：样本内增量 <b>{lift_is:+.2f}pp</b>'
             f'（{len(lifted)}/6 个轴选到了非基线值）。</li>')
    o.append(f'<li><b>增量是否可迁移（决定性）</b>：样本外增量 <b>{d_ex:+.2f}pp</b>'
             f'{"—— 可迁移。" if d_ex > 0 else "—— 不可迁移，是样本内调参的产物。"}</li>')
    add_txt = ('，相差较大 —— 各轴存在交互效应，收益不可简单相加。' if cant_add else '，基本可叠加。')
    o.append(f'<li><b>各轴收益是否可叠加</b>：边际贡献之和 {sum_marg:+.2f}pp vs '
             f'组合后总增量 {lift_is:+.2f}pp{add_txt}</li>')
    o.append(f'<li><b>是否为因子信号本身</b>：随机对照第 <b>{pct}</b> 百分位。</li>')
    o.append(f'<li><b>诚实边界</b>：<b>逐轴贪心不等于全局最优</b>（未做全网格搜索，'
             f'6 轴全组合数量级很大）；'
             f'用法参数在样本内挑选，存在过拟合风险；'
             f'幸存者偏差、仅一轮牛熊仍是硬约束。</li>')
    o.append('</ul></div>')

    if d_ex > 0:
        tail = ('用法扫描在本样本<b>确实找到了可迁移的改进</b> —— '
                '这与 P5「加因子没有增量」形成对照，印证了 P2 的线索：'
                '<b>怎么用比用哪个更关键</b>。')
    else:
        tail = ('用法扫描<b>没有带来可迁移的增量</b>。结合 P5，'
                '加因子、换风控、调用法三条路都走完了 —— '
                '<b>这个框架在本样本上的边际已经到顶</b>，'
                '再往下走需要的是新数据（分钟级 / 另类数据）或新维度（行业中性 / 多空对冲），'
                '而不是在同一个面上继续调。')
    o.append(f'<div class="box {"ok" if d_ex > 0 else "bad"}"><b>一句话</b>：{tail}</div>')

    o.append('<div class="sub" style="margin-top:22px">AQuant v5 P6 ｜ '
             '数据源：东方财富数据中心（point-in-time）｜ 不构成投资建议</div>')
    o.append('</div></body></html>')
    return "".join(o)


if __name__ == "__main__":
    main()
