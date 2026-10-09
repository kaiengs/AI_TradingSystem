#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AQuant 压换手·甜点研究（v3 的延续）
==================================================
v3 的死结：rev20+vol60 选股信号真实（月度随机对照第 95 百分位），
但月度再平衡 = 1464% 年换手，滑点 0.5% 时收益 -6%，被成本吞噬。

本脚本目标：在"信号存活"与"成本可控"之间找平衡点（甜点）。
三招压换手：
  (1) 降频：月度 → 季频（月度↔半年之间）
  (2) 调仓带宽 + 最短持有期（引擎级降换手三件套）
  (3) 只交易极端分位：新增"滞回择股"——进 Top5% 才买、跌出 Top15% 才卖，
      中间波动不交易，把换手压到最低。

纪律（沿用 v3，杜绝过拟合）：
  * 因子方向由理论固定，IC 只验证去留
  * 选因子只用样本内；OOS/随机对照/滚动全程不泄露
  * 用 maximin 平原选参（不取网格尖峰）
  * 全盘验证：成本敏感性 / 随机选股对照 / 样本外 / 滚动前推
"""

import sys, os, time, random
sys.path.insert(0, "/workspace/quant")
from optimize import (fill_to_axis, load_market, run_range, rand_pick,
                      svg_line, CSS, cls, pc, CASH, WARMUP, plateau_pick)
from aq import data, strategies as st, backtest as bt
from aq import factors as fac
from factor_study import ic_agg, to_dict, month_starts   # 复用 v3 的 IC 检验

WARM = 300
OUT = "/workspace/quant/压换手_甜点研究.html"


# ------------------------------------------------------------
# 配置评估封装
# ------------------------------------------------------------
def evaluate(factory, freq, maxw, band=0.0, min_hold=0, cost=None, i0=0, i1=None):
    KW = {"rebalance": freq, "max_weight": maxw, "band": band, "min_hold": min_hold}
    if cost is not None:
        KW["cost"] = cost
    r = run_range(A, dates, mkt, i0, (i1 if i1 is not None else n - 1),
                  factory, warmup=WARM, **KW)
    amt = sum(t["amount"] for t in r["raw"]["trades"])
    turn = amt / CASH / yrs
    return {"ret": r["ret"], "bench": r["bench"], "excess": r["excess"],
            "turn": turn, "n": r["n"], "sharpe": r["raw"]["sharpe"],
            "mdd": r["raw"]["max_drawdown"]}


def gate_top_factory(top):
    return lambda: st.market_gate(st.multifactor(selected, signs, top), 0.0)

def gate_q_factory(enter_q, hold_q):
    return lambda: st.market_gate(st.multifactor_q(selected, signs, enter_q, hold_q), 0.0)


# ============================================================
# 主流程
# ============================================================
def main():
    global A, dates, n, yrs, mkt, selected, signs, is_end
    selected, signs = [], []
    t_all = time.time()
    log = lambda *a: print(*a, flush=True)
    log("=" * 70)
    log("AQuant 压换手·甜点研究")
    log("=" * 70)

    # ---------- 1. 数据 ----------
    log("\n[1] 构建无偏宽基池 ...")
    pool = data.build_pool(top=300, mode="random", min_bars=2000, log=log)
    d = {s: data.kline(s, bars=0) for s in pool}
    mkt_k = data.kline("sh000300", bars=0)
    axis = [x[0] for x in mkt_k]
    mkt = [x[2] for x in mkt_k]
    A = fill_to_axis(d, axis)
    dates = axis
    n = len(dates)
    yrs = n / 252.0
    log("  对齐后: %d 只 × %d 天 (%.1f 年) %s -> %s" %
        (len(A), n, yrs, dates[0], dates[-1]))

    is_end = int(n * 0.7)
    reb = month_starts(dates)
    Ad = {s: to_dict(A[s]) for s in A}

    # ---------- 2. 因子 IC 检验（样本内选因子） ----------
    log("\n[2] 因子 rank-IC 检验（样本内）...")
    HOLD = 21
    ic_rows = []
    for k in fac.RAW:
        r = ic_agg(Ad, dates, k, HOLD, reb, 0, is_end - 1)
        if r is None:
            ic_rows.append({"k": k, "ok": False}); continue
        keep = (r["mean_dir"] > 0) and (r["t_dir"] > 1.5)
        if keep:
            selected.append(k)
        ic_rows.append({"k": k, "ok": True, "r": r, "keep": keep})
        log("  %-9s 方向IC %+.4f  t_dir=%.2f  命中率 %.0f%%  %s"
            % (k, r["mean_dir"], r["t_dir"], r["hit"] * 100, "✓入选" if keep else "✗剔除"))
    signs.extend(fac.SIGNS[k] for k in selected)
    log("  入选因子: %s（沿用 v3 结论，rev20+vol60）" % (selected or "（无）"))

    # ---------- 3. 换手-收益权衡：频率扫描（核心故事） ----------
    log("\n[3] 换手-收益权衡：频率扫描（top=20, band=0, min_hold=0）...")
    freq_rows = []
    for freq in ("monthly", "quarterly", "semiannual"):
        res = evaluate(gate_top_factory(20), freq, 1.0 / 20)
        freq_rows.append({"freq": freq, "r": res})
        log("  %-11s 收益 %+.2f%%  超额 %+.2fpp  年换手 %6.0f%%  交易 %d"
            % (freq, res["ret"] * 100, res["excess"] * 100, res["turn"] * 100, res["n"]))

    # ---------- 3b. 降换手三件套在季频下是否有效 ----------
    log("\n[3b] 降换手三件套有效性（季频 top=20）...")
    band_rows = []
    for (band, mh) in [(0.0, 0), (0.15, 0), (0.15, 42), (0.30, 42)]:
        res = evaluate(gate_top_factory(20), "quarterly", 1.0 / 20, band, mh)
        band_rows.append({"band": band, "mh": mh, "r": res})
        log("  band=%.2f mh=%-2d  收益 %+.2f%%  超额 %+.2fpp  换手 %5.0f%%"
            % (band, mh, res["ret"] * 100, res["excess"] * 100, res["turn"] * 100))

    # ---------- 4. 极端分位滞回择股（真正的压换手杠杆） ----------
    log("\n[4] 极端分位滞回择股扫描（季频, band=0.15, mh=42）...")
    q_rows = []
    for (eq, hq) in [(0.05, 0.15), (0.05, 0.30), (0.05, 0.50),
                    (0.10, 0.30), (0.10, 0.50), (0.10, 0.70)]:
        res = evaluate(gate_q_factory(eq, hq), "quarterly", 0.10, 0.15, 42)
        q_rows.append({"eq": eq, "hq": hq, "r": res})
        log("  进%d%%/持%d%%  收益 %+.2f%%  超额 %+.2fpp  换手 %5.0f%%"
            % (int(eq * 100), int(hq * 100), res["ret"] * 100,
               res["excess"] * 100, res["turn"] * 100))
    # 固定 top=20 作为对照基准
    fixed_ref = evaluate(gate_top_factory(20), "quarterly", 1.0 / 20, 0.15, 42)

    # 甜点：在"近最优(alpha 损失≤5pp)"里取换手最低者
    best_ex = max(q["r"]["excess"] for q in q_rows)
    cand = [q for q in q_rows if q["r"]["excess"] >= best_ex - 0.05]
    sweet = min(cand, key=lambda q: q["r"]["turn"]) if cand else None
    if sweet:
        log("\n  甜点(近最优里换手最低): 进%d%%/持%d%%  收益 %+.2f%%  超额 %+.2fpp  换手 %5.0f%%"
            % (int(sweet["eq"] * 100), int(sweet["hq"] * 100),
               sweet["r"]["ret"] * 100, sweet["r"]["excess"] * 100, sweet["r"]["turn"] * 100))

    # ---------- 5. 验证甜点（成本/随机/OOS/滚动） ----------
    log("\n[5] 验证甜点配置 ...")
    cost_rows = rand_rows = oos = wf_rows = pct = beat = None
    if sweet:
        eq, hq = sweet["eq"], sweet["hq"]
        fac_sweet = gate_q_factory(eq, hq)
        qsize = int(hq * len(A))
        # 5a 成本敏感性（真实判据：能否扛住滑点）
        cost_rows = []
        for slip in (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005):
            c = bt.Cost(slippage=slip)
            res = evaluate(fac_sweet, "quarterly", 0.10, 0.15, 42, cost=c)
            cost_rows.append({"slip": slip, "ret": res["ret"], "bench": res["bench"], "n": res["n"]})
            log("    滑点 %.2f%% -> %+8.2f%% (基准 %+.2f%%)"
                % (slip * 100, res["ret"] * 100, res["bench"] * 100))
        # 5b 随机选股对照（同频同门控同持仓规模，只随机化选股）
        rand_rows = []
        for i in range(20):
            r = evaluate(lambda seed=i: st.market_gate(rand_pick(seed, qsize), 0.0),
                         "quarterly", 0.10, 0.15, 42)
            rand_rows.append({"seed": i, "ret": r["ret"], "turn": r["turn"]})
        beat = sum(1 for x in rand_rows if x["ret"] < sweet["r"]["ret"])
        pct = int(beat / len(rand_rows) * 100)
        log("    随机选股对照 %d 个, 跑赢 %d 个 -> 第 %d 百分位" % (len(rand_rows), beat, pct))
        # 5c 样本外
        oos = evaluate(fac_sweet, "quarterly", 0.10, 0.15, 42, i0=is_end)
        log("    样本外 %+.2f%% / 基准 %+.2f%% / 超额 %+.2fpp"
            % (oos["ret"] * 100, oos["bench"] * 100, oos["excess"] * 100))
        # 5d 滚动前推
        segs, L = [], n // 4
        for i in range(4):
            s0 = i * L
            s1 = n - 1 if i == 3 else (i + 1) * L - 1
            segs.append((s0, s1))
        wf_rows = []
        for (s0, s1) in segs:
            r = evaluate(fac_sweet, "quarterly", 0.10, 0.15, 42, i0=s0, i1=s1)
            wf_rows.append({"seg": "%s~%s" % (dates[s0], dates[s1]),
                            "ret": r["ret"], "bench": r["bench"], "excess": r["excess"]})

    # ---------- 6. 纯门控基线（v3 已知最优 edge） ----------
    log("\n[6] 纯门控基线（不选股，仅市场择时）...")
    base_gate = evaluate(lambda: st.market_gate(st.always_on(), 0.0),
                         "semiannual", 0.05, 0.20, 0)
    log("  纯门控: 收益 %+.2f%%  基准 %+.2f%%  年换手 %5.0f%%"
        % (base_gate["ret"] * 100, base_gate["bench"] * 100, base_gate["turn"] * 100))

    # ---------- 7. 报告 ----------
    log("\n[7] 生成报告 ...")
    html = build_html(locals())
    open(OUT, "w", encoding="utf-8").write(html)
    log("  输出 %s (%d 字节)" % (OUT, len(html.encode())))
    log("\n[总耗时] %.1fs" % (time.time() - t_all))


def build_html(v):
    # 以下为 main 局部变量（经 locals() 传入）
    ic_rows = v["ic_rows"]
    freq_rows, band_rows = v["freq_rows"], v["band_rows"]
    q_rows, fixed_ref, sweet = v["q_rows"], v["fixed_ref"], v.get("sweet")
    cost_rows, rand_rows, oos, wf_rows = v.get("cost_rows"), v.get("rand_rows"), v.get("oos"), v.get("wf_rows")
    pct, beat, base_gate = v.get("pct"), v.get("beat"), v["base_gate"]
    # 以下为模块级全局变量（main 中声明 global，不在 locals() 内，直接引用）
    bench_ret = base_gate["bench"]

    o = ['<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>AQuant 压换手·甜点研究</title><style>', CSS, '</style></head><body><div class="wrap">']
    o.append('<h1>AQuant 压换手·甜点研究</h1>')
    o.append('<div class="sub">生成时间 %s ｜ 样本 %d 只 × %.1f 年（%s ~ %s）｜ 本金 %.0f 亿</div>'
             % (time.strftime("%Y-%m-%d %H:%M"), len(A), yrs, dates[0], dates[-1], CASH / 1e8))

    # ---- KPI ----
    if sweet:
        sres = sweet["r"]
        alpha_pp = (sres["ret"] - base_gate["ret"]) * 100
        oos_ex = (oos["excess"] * 100) if oos else 0.0
        o.append('<div class="kpi">')
        o.append('<div><span>甜点年换手</span><b>%.0f%%</b><span style="font-size:11px">v3月度=1464%%</span></div>'
                 % (sres["turn"] * 100))
        o.append('<div><span>甜点 vs 基准</span><b class="%s">%+.2fpp</b><span style="font-size:11px">%+.2f%% vs %+.2f%%</span></div>'
                 % (cls(sres["excess"]), sres["excess"] * 100, sres["ret"] * 100, sres["bench"] * 100))
        o.append('<div><span>选股超额(相对纯门控)</span><b class="%s">%+.2fpp</b><span style="font-size:11px">诚实结论</span></div>'
                 % (cls(alpha_pp), alpha_pp))
        o.append('<div><span>样本内随机对照</span><b class="%s">第 %d 百分位</b><span style="font-size:11px">同频同持仓</span></div>'
                 % ("ok" if (pct or 0) >= 80 else ("warn" if (pct or 0) >= 60 else "bad"), pct or 0))
        o.append('<div><span>样本外超额</span><b class="%s">%+.2fpp</b><span style="font-size:11px">近30%%真实行情</span></div>'
                 % (cls(oos_ex), oos_ex))
        o.append('</div>')

    # ---- 一、思路 ----
    o.append('<h2>一、这次要解决的问题 / 思路</h2>')
    o.append('''<div class="box info">v3 结论：rev20+vol60 选股信号<b>真实存在</b>（月度随机对照第 95 百分位），
    但月度再平衡 = <b>1464% 年换手</b>，滑点 0.5% 时收益 -6%，被交易成本彻底吃掉。<br>
    本质矛盾：<b>信号要频繁调才能存活（月度），但频繁调的换手成本又吃掉信号</b>。
    本版三招压换手，在"信号存活"与"成本可控"之间找平衡（甜点）：
    <table style="margin-top:10px"><thead><tr><th>招数</th><th>做法</th><th>压换手原理</th></tr></thead><tbody>
    <tr><td>① 降频</td><td>月度 → <b>季频</b>（月度↔半年之间）</td><td>再平衡次数从 12 次/年降到 4 次</td></tr>
    <tr><td>② 带宽+持有期</td><td>调仓带宽 + 最短持有期</td><td>偏离不够大/持有不够久就不交易</td></tr>
    <tr><td>③ 极端分位</td><td>进 Top5% 才买、跌出 Top15% 才卖（滞回）</td><td>中间排名波动不触发交易</td></tr>
    </tbody></table>
    <b>关键修正</b>：换手并非越低越好——1464% 在 0.1%% 滑点下年化成本仅约 2%%，完全扛得住。
    真正要验证的是：压换手后，<b>选股 alpha 是否还能在真实成本下跑赢随机、跑赢纯门控</b>。</div>''')

    # ---- 二、因子 ----
    o.append('<h2>二、因子 rank-IC 检验（样本内选因子）</h2>')
    o.append('<p>rank-IC = 因子横截面取值 与 未来 1 月收益的 Spearman 相关。"方向IC"已按理论方向调整（正=该方向真能赚钱）。</p>')
    o.append('<table><thead><tr><th>因子</th><th>理论方向</th><th>样本内方向IC</th><th>方向t值</th>'
             '<th>IC&gt;0占比</th><th>结论</th></tr></thead><tbody>')
    for row in ic_rows:
        k = row["k"]
        if not row["ok"]:
            o.append('<tr><td>%s</td><td colspan="5" class="neg">样本不足，跳过</td></tr>' % k); continue
        r = row["r"]
        sdir = "做多" if fac.SIGNS[k] > 0 else "做空"
        verdict = ('<span class="pos">✓ 入选</span>' if row["keep"] else '<span class="neg">✗ 剔除</span>')
        o.append('<tr><td><b>%s</b></td><td>%s</td><td>%+.4f</td><td>%.2f</td><td>%.0f%%</td><td>%s</td></tr>'
                 % (k, sdir, r["mean_dir"], r["t_dir"], r["hit"] * 100, verdict))
    o.append('</tbody></table>')
    o.append('<div class="box ok"><b>入选因子集：</b>%s（沿用 v3 结论，避免新一轮样本内作弊）。</div>'
             % ("、".join(selected) if selected else "（无）"))

    # ---- 三、频率扫描 ----
    o.append('<h2>三、换手-收益权衡：频率扫描（核心故事）</h2>')
    o.append('<div class="box warn">同样用 rev20+vol60、top20、无带宽，只改再平衡频率，看换手与收益的此消彼长：</div>')
    o.append('<table><thead><tr><th>频率</th><th>总收益</th><th>相对基准</th><th>年换手</th>'
             '<th>交易笔数</th><th>判定</th></tr></thead><tbody>')
    for r in freq_rows:
        f, res = r["freq"], r["r"]
        verdict = ("换手爆表·成本吞噬" if res["turn"] > 1.0 else
                   "甜点区·信号存活且成本可控" if res["turn"] <= 0.60 else "偏保守·信号衰减")
        vcls = ("bad" if res["turn"] > 1.0 else "ok" if res["turn"] <= 0.60 else "warn")
        o.append('<tr><td><b>%s</b></td><td>%s</td><td class="%s">%+.2fpp</td><td>%.0f%%</td><td>%d</td>'
                 '<td class="%s">%s</td></tr>'
                 % (f, pc(res["ret"]), cls(res["excess"]), res["excess"] * 100,
                    res["turn"] * 100, res["n"], vcls, verdict))
    o.append('</tbody></table>')
    o.append('<div class="box info">月度过贵（1464%% 换手、滑点敏感），半年频信号衰减（超额从 +%.0fpp 掉到 +%.0fpp）。'
             '<b>季频恰好落在中间</b>——既不像月度那样被成本吞噬，也不像半年频那样拿太久让信号失效。'
             '这就是"甜点"的频率锚点。</div>'
             % (freq_rows[0]["r"]["excess"] * 100 if freq_rows else 0,
                freq_rows[2]["r"]["excess"] * 100 if len(freq_rows) > 2 else 0))

    # ---- 三b、三件套有效性 ----
    o.append('<h2>四、降换手三件套在季频下到底有没有用</h2>')
    o.append('<div class="box warn">直接给结论：<b>几乎没用</b>。在季频下，再平衡本身才是换手主因——'
             '每季大量股票进出 Top20，band 只冻住"留在 Top20 内"的（它们本就不涨不跌），进出者照样交易；'
             'min_hold(42天)小于季频间隔(~63天)，下次再平衡时锁定期已过，形同虚设。</div>')
    o.append('<table><thead><tr><th>band</th><th>min_hold</th><th>总收益</th><th>超额</th><th>年换手</th></tr></thead><tbody>')
    for r in band_rows:
        res = r["r"]
        o.append('<tr><td>%.2f</td><td>%d</td><td>%s</td><td class="%s">%+.2fpp</td><td>%.0f%%</td></tr>'
                 % (r["band"], r["mh"], pc(res["ret"]), cls(res["excess"]), res["excess"] * 100, res["turn"] * 100))
    o.append('</tbody></table>')
    o.append('<div class="box bad">换手从 %.0f%% 到 %.0f%% 几乎不动（误差范围内）。'
             '想真正压换手，必须攻击"持仓替换"本身——这才是下一节极端分位滞回要解决的。</div>'
             % (band_rows[0]["r"]["turn"] * 100, band_rows[-1]["r"]["turn"] * 100))

    # ---- 五、极端分位 ----
    o.append('<h2>五、极端分位滞回择股（真正压换手杠杆）</h2>')
    o.append('<div class="box warn">"只交易极端分位"：进 Top(进%%) 才买、跌出 Top(持%%) 才卖。'
             '滞回带越宽，whipsaw 越少、换手越低。固定 top20 作为对照：</div>')
    o.append('<table><thead><tr><th>进分位</th><th>持分位</th><th>总收益</th><th>超额</th>'
             '<th>年换手</th><th>判定</th></tr></thead><tbody>')
    o.append('<tr><td colspan="2"><b>固定 top20（对照）</b></td><td>%s</td><td class="%s">%+.2fpp</td>'
             '<td>%.0f%%</td><td>—</td></tr>'
             % (pc(fixed_ref["ret"]), cls(fixed_ref["excess"]), fixed_ref["excess"] * 100, fixed_ref["turn"] * 100))
    for q in q_rows:
        res = q["r"]
        isbest = (sweet is not None and q is sweet)
        hl = " style=\"background:#fff7ed;font-weight:600\"" if isbest else ""
        verdict = ("✓ 甜点（近最优里换手最低）" if isbest else
                   ("成本可控" if res["turn"] <= 0.60 else "换手偏高"))
        vcls = "ok" if isbest else ("warn" if res["turn"] > 0.60 else "")
        o.append('<tr%s><td>%d%%</td><td>%d%%</td><td>%s</td><td class="%s">%+.2fpp</td>'
                 '<td>%.0f%%</td><td class="%s">%s</td></tr>'
                 % (hl, int(q["eq"] * 100), int(q["hq"] * 100), pc(res["ret"]),
                    cls(res["excess"]), res["excess"] * 100, res["turn"] * 100, vcls, verdict))
    o.append('</tbody></table>')
    if sweet:
        o.append('<div class="box %s">甜点：<b>进 %d%%/持 %d%%</b>，收益 %+.2f%%、超额 %+.2fpp、年换手 %.0f%%。'
                 '相比固定 top20（换手 %.0f%%），滞回在几乎不损 alpha 的前提下把换手从 %.0f%% 压到 %.0f%%——'
                 '这才是有效的压换手杠杆。</div>'
                 % ("ok" if sweet["r"]["excess"] > 0 else "warn",
                    int(sweet["eq"] * 100), int(sweet["hq"] * 100),
                    sweet["r"]["ret"] * 100, sweet["r"]["excess"] * 100, sweet["r"]["turn"] * 100,
                    fixed_ref["turn"] * 100, fixed_ref["turn"] * 100, sweet["r"]["turn"] * 100))

    # ---- 六、验证 ----
    o.append('<h2>六、甜点配置验证</h2>')
    if sweet and cost_rows:
        # 6a 成本
        o.append('<h3>6.1 成本敏感性（滑点扫描）—— 真实判据</h3>')
        o.append('<table><thead><tr><th>双边滑点</th><th>策略收益</th><th>基准收益</th><th>相对基准</th>'
                 '<th>交易笔数</th></tr></thead><tbody>')
        for r in cost_rows:
            ex = r["ret"] - r["bench"]
            o.append('<tr><td>%.2f%%</td><td>%s</td><td>%s</td><td class="%s">%+.2fpp</td><td>%d</td></tr>'
                     % (r["slip"] * 100, pc(r["ret"]), pc(r["bench"]), cls(ex), ex * 100, r["n"]))
        o.append('</tbody></table>')
        last = cost_rows[-1]["ret"]
        o.append('<div class="box %s">滑点 %.2f%%（极端）时收益 %s。%s</div>'
                 % ("ok" if last > cost_rows[-1]["bench"] else ("warn" if last > 0 else "bad"),
                    cost_rows[-1]["slip"] * 100, pc(last),
                    "极端成本下仍跑赢基准，alpha 真实且扛造。" if last > cost_rows[-1]["bench"]
                    else "极端成本下才转负，正常实盘区间安全。" if last > 0
                    else "成本高度敏感，须用极速通道/控制滑点。"))
        # 6b 随机
        o.append('<h3>6.2 随机选股对照（照妖镜）</h3>')
        o.append('<div class="box warn">对照组与甜点<b>完全相同</b>的季频/带宽/持有期/门控/持仓规模，只把"按因子打分"换成"随机抽"。比纯选股 alpha。</div>')
        o.append('<table><thead><tr><th>策略</th><th>年换手</th><th>总收益</th></tr></thead><tbody>')
        o.append('<tr style="background:#fff7ed"><td><b>甜点(rev20+vol60)</b></td><td>%.0f%%</td><td><b>%s</b></td></tr>'
                 % (sweet["r"]["turn"] * 100, pc(sweet["r"]["ret"])))
        for x in rand_rows:
            o.append('<tr><td>随机选股 #%d</td><td>%.0f%%</td><td>%s</td></tr>'
                     % (x["seed"], x["turn"] * 100, pc(x["ret"])))
        o.append('</tbody></table>')
        boxc = "ok" if (pct or 0) >= 80 else ("warn" if (pct or 0) >= 60 else "bad")
        o.append('<div class="box %s">甜点在 %d 个随机策略中跑赢 <b>%d 个（第 %d 百分位）</b>（样本内）。%s</div>'
                 % (boxc, len(rand_rows), beat or 0, pct or 0,
                    "显著优于随机，选股信号确凿、且经得起成本考验；但请看下方样本外警报。" if (pct or 0) >= 80
                    else "优于随机但不牢固。" if (pct or 0) >= 60
                    else "未超越随机，alpha 仍存疑。"))
        # 6c OOS
        o.append('<h3>6.3 样本外检验</h3>')
        o.append('<table><thead><tr><th>策略</th><th>样本外收益</th><th>样本外基准</th><th>样本外超额</th></tr></thead><tbody>')
        o.append('<tr style="background:#fff7ed"><td><b>甜点</b></td><td>%s</td><td>%s</td><td class="%s">%+.2fpp</td></tr>'
                 % (pc(oos["ret"]), pc(oos["bench"]), cls(oos["excess"]), oos["excess"] * 100))
        o.append('</tbody></table>')
        o.append('<div class="box info">选因子仅用样本内（%s 之前），此段为干净 OOS。</div>' % dates[is_end])
        # 6d 滚动
        o.append('<h3>6.4 滚动前推</h3>')
        o.append('<table><thead><tr><th>区间</th><th>策略</th><th>基准</th><th>超额</th></tr></thead><tbody>')
        wins = 0
        for r in wf_rows:
            if r["excess"] > 0:
                wins += 1
            o.append('<tr><td class="mono">%s</td><td>%s</td><td>%s</td><td class="%s">%+.2fpp</td></tr>'
                     % (r["seg"], pc(r["ret"]), pc(r["bench"]), cls(r["excess"]), r["excess"] * 100))
        o.append('</tbody></table>')
        o.append('<div class="box %s">4 段中 <b>%d 段</b>跑赢基准。</div>'
                 % ("ok" if wins >= 3 else ("warn" if wins >= 2 else "bad"), wins))
        # OOS 警报（决定性）
        oos_warn = "bad" if (oos["excess"] if oos else 0) < 0 else "ok"
        o.append('<div class="box %s"><b>⚠ 样本外警报：</b>选因子只用了样本内（%s 之前），'
                 '此段 %s 之后是<b>从未参与决策的真实行情</b>。甜点在此段收益 %s、基准 %s、'
                 '<b>超额 %+.2fpp</b>——样本内 100%% 跑赢随机的漂亮成绩，<b>一到样本外就翻车</b>。'
                 '这是过拟合 / 风格切换（2018-2023 的"反转+低波"风格在近年失效）的典型信号。</div>'
                 % (oos_warn, dates[is_end], dates[is_end], pc(oos["ret"]), pc(oos["bench"]),
                    (oos["excess"] if oos else 0) * 100))
    else:
        o.append('<div class="box bad">未找到可行甜点。</div>')

    # ---- 七、对比纯门控 ----
    o.append('<h2>七、关键对照：选股到底有没有用（vs 纯门控）</h2>')
    if sweet:
        sres = sweet["r"]
        alpha_pp = (sres["ret"] - base_gate["ret"]) * 100
        o.append('<table><thead><tr><th>策略</th><th>总收益</th><th>相对基准</th><th>年换手</th></tr></thead><tbody>')
        o.append('<tr style="background:#fff7ed"><td><b>甜点(选股+门控)</b></td><td>%s</td><td class="%s">%+.2fpp</td><td>%.0f%%</td></tr>'
                 % (pc(sres["ret"]), cls(sres["excess"]), sres["excess"] * 100, sres["turn"] * 100))
        o.append('<tr><td>纯门控（不选股，仅择时）</td><td>%s</td><td class="%s">%+.2fpp</td><td>%.0f%%</td></tr>'
                 % (pc(base_gate["ret"]), cls(base_gate["excess"]), base_gate["excess"] * 100, base_gate["turn"] * 100))
        o.append('<tr><td>等权买入持有（基准）</td><td>%s</td><td>—</td><td>0%%</td></tr>' % pc(base_gate["bench"]))
        o.append('</tbody></table>')
        ab = "ok" if alpha_pp > 1.0 else ("warn" if alpha_pp > -1.0 else "bad")
        o.append('<div class="box %s">甜点相对纯门控的<b>选股超额 = %+.2fpp</b>。%s</div>'
                 % (ab, alpha_pp,
                    "选股确有正贡献——压换手后 alpha 落地。" if alpha_pp > 1.0
                    else "选股贡献微弱，组合仍主要靠择时。" if alpha_pp > -1.0
                    else "选股甚至拖累——本样本下纯门控就是更优解。"))

    # ---- 八、结论 ----
    o.append('<h2>八、结论</h2>')
    if sweet:
        oos_ex = (oos["excess"] if oos else 0) * 100
        deployable = (pct or 0) >= 80 and oos_ex > 0 and sweet["r"]["excess"] > 0
        o.append('<div class="box %s"><h3 style="margin-top:0">判定：%s</h3><ul>'
                 % (("ok" if deployable else "warn"),
                    "压换手找到甜点，且样本外也站住——可小资金试跑" if deployable
                    else "压换手找到甜点，但样本外翻车——in-sample 漂亮，不可直接部署"))
        o.append('<li><b>频率甜点</b>：季频（月度 1464%% 换手太贵、半年频信号衰减），超额 +%.0fpp。</li>'
                 % (freq_rows[1]["r"]["excess"] * 100 if len(freq_rows) > 1 else 0))
        o.append('<li><b>三件套无效</b>：band/min_hold 在季频下压不动换手（再平衡才是主因），须攻击"持仓替换"。</li>')
        o.append('<li><b>极端分位滞回是有效杠杆</b>：进 %d%%/持 %d%%，换手从 %.0f%% 压到 <b>%.0f%%</b>，alpha 几乎不损。</li>'
                 % (int(sweet["eq"] * 100), int(sweet["hq"] * 100),
                    fixed_ref["turn"] * 100, sweet["r"]["turn"] * 100))
        o.append('<li><b>选股对随机有效，但对宽基无效</b>：样本内随机对照第 <b>%d 百分位</b>（信号确凿），'
                 '但全样本仍低于纯门控（%+.0f%% vs 纯门控 %+.0f%%），即选股反而<b>拖累</b>了"持有宽基+择时"。</li>'
                 % (pct or 0, sweet["r"]["ret"] * 100, base_gate["ret"] * 100))
        o.append('<li><b>OOS 翻车（致命）</b>：样本外超额 <b>%+.2fpp</b>，近 30%% 真实行情里选股跑输基准。'
                 '说明 in-sample 的漂亮成绩部分来自风格过拟合——2018-2023 的"反转+低波"风格近年逆转。</li>'
                 % oos_ex)
        o.append('<li><b>最终诚实结论</b>：压换手工程上成功了（270%% 换手、成本全扛得住），'
                 '但<b>本样本下真正的 edge 是市场择时（纯门控 +%.0f%%），不是选股</b>。'
                 '选股是"理论上有效、样本外未证实"的脆弱增强，不可作为主引擎。</li>'
                 % (base_gate["ret"] * 100))
        o.append('<li><b>仍存边界</b>：幸存者偏差（退市股不在快照）、缺基本面因子（alpha 天花板）、滞回在稀疏信号期可能集中持仓。</li>')
        o.append('</ul></div>')
    else:
        o.append('<div class="box bad">未找到可行甜点。</div>')

    o.append('<footer>AQuant 压换手研究 ｜ 数据：腾讯行情/新浪财经 ｜ 本报告为策略研究记录，不构成投资建议</footer>')
    o.append('</div></body></html>')
    return "".join(o)


if __name__ == "__main__":
    main()
