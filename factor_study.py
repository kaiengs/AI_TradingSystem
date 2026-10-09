#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AQuant 多因子研究 v3
==================================================
目标：v2 已证明"风控成立、选股无 alpha"。本脚本验证——

  (1) 用 rank-IC 检验 5 个经典因子的横截面预测力（样本内选因子）
  (2) 把通过检验的因子合成 composite，做低频 + 门控 的选股策略
  (3) 全套验证：样本外 / 成本敏感性 / 随机选股对照 / 滚动前推
  (4) 关键对照：composite 是否真的比"纯门控（不选股）"多赚 —— 即 alpha 是否成立

设计纪律（避免 v1 的过拟合陷阱）：
  * 因子方向由金融学理论固定（fac.SIGNS），IC 只用来"验证方向 + 决定去留"，不倒推方向
  * 因子选择在样本内完成；样本外 / 滚动前推 / 随机对照 全程不泄露给选因子环节
  * 组合换手频率、持仓数沿用 v2 已证伪/证实的结论（低频、分散、门控），不再 grid 搜参
"""

import sys, os, time, random
sys.path.insert(0, "/workspace/quant")
from optimize import (fill_to_axis, load_market, run_range, run_full, rand_pick,
                      svg_line, CSS, cls, pc, CASH, WARMUP)
from aq import data, strategies as st, backtest as bt
from aq import factors as fac

WARM = 300          # 因子最长需要 260 根预热，给足余量
OUT = "/workspace/quant/因子研究_多因子模型.html"

# ------------------------------------------------------------
# 统计工具
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
    """把对齐后的 (date,o,c,h,l,v) 列表转成因子期望的 dict 形式
    （引擎内部也是这么转的，这里让 IC 计算与引擎 prescan 拿到一致的数据结构）"""
    return {"date": [b[0] for b in bars], "open": [b[1] for b in bars],
            "close": [b[2] for b in bars], "high": [b[3] for b in bars],
            "low": [b[4] for b in bars], "volume": [b[5] for b in bars]}

def ic_agg(A, dates, fkey, hold, reb, i0, i1):
    """在 [i0,i1] 上，对因子 fkey 计算逐月（reb）的 rank-IC 并聚合。
    返回 {n, mean(原始IC), t, hit}，以及 mean(方向调整后IC)。"""
    fn = fac.RAW[fkey]
    sign = fac.SIGNS[fkey]
    ics = []
    for t in reb:
        if t < i0 or t > i1:
            continue
        fv = {}
        for s in A:
            v = fn(A[s], t)
            if v is None:
                continue
            cl = A[s]["close"]
            jf = min(t + hold, len(cl) - 1)
            if jf <= t or cl[t] <= 0:
                continue
            fwd = cl[jf] / cl[t] - 1
            fv[s] = (v, fwd)
        if len(fv) < 40:
            continue
        xs = [v for v, _ in fv.values()]
        ys = [r for _, r in fv.values()]
        ic = spearman(xs, ys)
        if ic is not None:
            ics.append(ic)
    if not ics:
        return None
    m = sum(ics) / len(ics)
    sd = (sum((x - m) ** 2 for x in ics) / len(ics)) ** 0.5
    tstat = m / (sd / (len(ics) ** 0.5)) if sd > 0 else 0.0
    hit = sum(1 for x in ics if x > 0) / len(ics)
    # 方向调整后的统计量：t_dir = t_raw × sign。
    # 注意：方向 IC 为正不代表显著；必须用方向调整后的 t 判显著，
    # 否则会出现"原始 IC 显著为负、方向翻转后为正却被 t 误杀"的 bug（rev20/vol60 就是）。
    return {"n": len(ics), "mean": m, "t": tstat, "t_dir": tstat * sign,
            "hit": hit, "mean_dir": m * sign}


# ============================================================
# 主流程
# ============================================================
def main():
    t_all = time.time()
    log = lambda *a: print(*a, flush=True)
    log("=" * 66)
    log("AQuant 多因子研究 v3")
    log("=" * 66)

    # ---------- 1. 数据 ----------
    log("\n[1/6] 构建无偏宽基池 ...")
    pool = data.build_pool(top=300, mode="random", min_bars=2000, log=log)
    d = {s: data.kline(s, bars=0) for s in pool}
    mkt_k = data.kline("sh000300", bars=0)
    axis = [x[0] for x in mkt_k]
    mkt = [x[2] for x in mkt_k]
    A = fill_to_axis(d, axis)
    Ad = {s: to_dict(A[s]) for s in A}        # 因子/prescan 需要 dict 形式（引擎内部同理）
    dates = axis
    n = len(dates)
    yrs = n / 252.0
    log("  对齐后: %d 只 × %d 天 (%.1f 年) %s -> %s" % (len(A), n, yrs, dates[0], dates[-1]))

    is_end = int(n * 0.7)
    reb = month_starts(dates)
    log("  切分 样本内 %s~%s / 样本外 %s~%s ｜ 月度 IC 样本点 %d" %
        (dates[0], dates[is_end - 1], dates[is_end], dates[-1], len(reb)))

    # ---------- 2. 因子 IC 检验（样本内选因子） ----------
    log("\n[2/6] 因子 rank-IC 检验（样本内）...")
    HOLD = 21          # 用 1 个月持有期衡量短期横截面预测力
    ic_rows = []
    selected = []
    for k in fac.RAW:
        r = ic_agg(Ad, dates, k, HOLD, reb, 0, is_end - 1)
        if r is None:
            ic_rows.append({"k": k, "ok": False})
            continue
        keep = (r["mean_dir"] > 0) and (r["t_dir"] > 1.5)
        if keep:
            selected.append(k)
        ic_rows.append({"k": k, "ok": True, "r": r, "keep": keep})
        log("  %-9s 方向IC %+.4f  t=%.2f  命中率 %.0f%%  %s"
            % (k, r["mean_dir"], r["t"], r["hit"] * 100, "✓入选" if keep else "✗剔除"))

    # 也给出全样本 IC（仅展示，不参与选择，避免泄漏）
    full_ic = {k: ic_agg(Ad, dates, k, HOLD, reb, 0, n - 1) for k in fac.RAW}
    log("  入选因子: %s" % (selected or "（无，退化到纯门控）"))
    signs = [fac.SIGNS[k] for k in selected]

    # ---------- 3. 组合配置（沿用 v2 已证结论：低频+分散+门控） ----------
    TOP = 20
    FREQ = "semiannual"
    GATE = 0.0
    KW = {"rebalance": FREQ, "max_weight": 1.0 / TOP}
    FINAL_FAC = lambda: st.market_gate(
        st.multifactor(selected, signs, TOP), GATE) if selected else st.market_gate(st.always_on(), GATE)

    # 全样本
    log("\n[3/6] 全样本回测 ...")
    fr = run_range(A, dates, mkt, 0, n - 1, FINAL_FAC, warmup=WARM, **KW)["raw"]
    fr_turn = sum(t["amount"] for t in fr["trades"]) / CASH / yrs
    fr_ret = fr["total_return"]
    bench_ret = fr["benchmark"]["ret"]
    log("  composite+门控: 收益 %+.2f%%  基准 %+.2f%%  年换手 %.0f%%  交易 %d"
        % (fr_ret * 100, bench_ret * 100, fr_turn * 100, fr["n_trades"]))

    # 纯门控（不选股）基线：检验"选股到底有没有用"
    base_gate = run_range(A, dates, mkt, 0, n - 1,
                          lambda: st.market_gate(st.always_on(), GATE),
                          warmup=WARM, band=0.20, **KW)["raw"]
    bg_ret = base_gate["total_return"]
    log("  纯门控(不选股): %+.2f%%  相对基准 %+.2fpp"
        % (bg_ret * 100, (bg_ret - bench_ret) * 100))

    # ---------- 4. 样本外 ----------
    log("\n[4/6] 样本外检验 ...")
    oos_r = run_range(A, dates, mkt, is_end, n - 1, FINAL_FAC, warmup=WARM, **KW)
    oos_sel = run_range(A, dates, mkt, is_end, n - 1,
                        lambda: st.market_gate(st.always_on(), GATE),
                        warmup=WARM, band=0.20, **KW)
    log("  样本外 composite %+.2f%% / 纯门控 %+.2f%% / 基准 %+.2f%%"
        % (oos_r["ret"] * 100, oos_sel["ret"] * 100, oos_r["bench"] * 100))

    # ---------- 5. 成本敏感性 ----------
    log("\n[5/6] 成本敏感性 ...")
    cost_rows = []
    for slip in (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005):
        c = bt.Cost(slippage=slip)
        r = run_range(A, dates, mkt, 0, n - 1, FINAL_FAC, warmup=WARM, cost=c, **KW)["raw"]
        cost_rows.append({"slip": slip, "ret": r["total_return"], "n": r["n_trades"],
                          "bench": r["benchmark"]["ret"]})
        log("  滑点 %.2f%% -> %+8.2f%% (基准 %+.2f%%, 交易 %d)"
            % (slip * 100, r["total_return"] * 100, r["benchmark"]["ret"] * 100, r["n_trades"]))

    # ---------- 6. 随机选股对照 + 滚动前推 ----------
    log("\n[6/6] 随机选股对照 + 滚动前推 ...")
    rand_rows = []
    for i in range(20):
        r = run_range(A, dates, mkt, 0, n - 1,
                      lambda seed=i: st.market_gate(rand_pick(seed, TOP), GATE),
                      warmup=WARM, **KW)["raw"]
        t_ = sum(t["amount"] for t in r["trades"]) / CASH / yrs
        rand_rows.append({"seed": i, "ret": r["total_return"], "turn": t_, "n": r["n_trades"]})
    beat = sum(1 for x in rand_rows if x["ret"] < fr_ret)
    pct = int(beat / len(rand_rows) * 100)
    log("  随机选股对照 %d 个, 跑赢 %d 个 -> 第 %d 百分位" % (len(rand_rows), beat, pct))

    segs, L = [], n // 4
    for i in range(4):
        s0 = i * L
        s1 = n - 1 if i == 3 else (i + 1) * L - 1
        segs.append((s0, s1))
    wf_rows = []
    for (s0, s1) in segs:
        r = run_range(A, dates, mkt, s0, s1, FINAL_FAC, warmup=WARM, **KW)
        wf_rows.append({"seg": "%s~%s" % (dates[s0], dates[s1]),
                        "ret": r["ret"], "bench": r["bench"], "excess": r["excess"]})
        log("  %s~%s  策略 %+7.2f%%  基准 %+7.2f%%  超额 %+7.2fpp"
            % (dates[s0], dates[s1], r["ret"] * 100, r["bench"] * 100, r["excess"] * 100))

    # ---------- 6b. 持有期错配检验：月度再平衡（匹配 1 月 IC 周期） ----------
    log("\n[6b] 月度再平衡对照（检验持有期 vs IC 周期错配）...")
    KW_M = {"rebalance": "monthly", "max_weight": 1.0 / TOP}
    fr_m = run_range(A, dates, mkt, 0, n - 1, FINAL_FAC, warmup=WARM, **KW_M)["raw"]
    fr_m_turn = sum(t["amount"] for t in fr_m["trades"]) / CASH / yrs
    fr_m_ret = fr_m["total_return"]
    rand_m = []
    for i in range(20):
        r = run_range(A, dates, mkt, 0, n - 1,
                      lambda seed=i: st.market_gate(rand_pick(seed, TOP), GATE),
                      warmup=WARM, **KW_M)["raw"]
        rand_m.append(r["total_return"])
    beat_m = sum(1 for x in rand_m if x < fr_m_ret)
    pct_m = int(beat_m / len(rand_m) * 100)
    log("  月度 composite %+.2f%%  年换手 %.0f%%  随机对照第 %d 百分位"
        % (fr_m_ret * 100, fr_m_turn * 100, pct_m))

    # 月度成本敏感性（1464% 年换手必须过这一关）
    cost_m_rows = []
    for slip in (0.001, 0.002, 0.003, 0.005):
        c = bt.Cost(slippage=slip)
        r = run_range(A, dates, mkt, 0, n - 1, FINAL_FAC, warmup=WARM, cost=c, **KW_M)["raw"]
        cost_m_rows.append({"slip": slip, "ret": r["total_return"], "n": r["n_trades"],
                           "bench": r["benchmark"]["ret"]})
        log("    月度 滑点 %.2f%% -> %+8.2f%% (基准 %+.2f%%, 交易 %d)"
            % (slip * 100, r["total_return"] * 100, r["benchmark"]["ret"] * 100, r["n_trades"]))

    # ---------- 7. 报告 ----------
    log("\n[7/7] 生成报告 ...")
    html = build_html(locals())
    open(OUT, "w", encoding="utf-8").write(html)
    log("  输出 %s (%d 字节)" % (OUT, len(html.encode())))
    log("\n[总耗时] %.1fs" % (time.time() - t_all))


def build_html(v):
    A, dates, n, yrs = v["A"], v["dates"], v["n"], v["yrs"]
    is_end, reb, ic_rows, selected = v["is_end"], v["reb"], v["ic_rows"], v["selected"]
    full_ic, fr, fr_turn, fr_ret = v["fr"], v["fr"], v["fr_turn"], v["fr_ret"]
    bench_ret, base_gate, bg_ret = v["bench_ret"], v["base_gate"], v["bg_ret"]
    oos_r, oos_sel = v["oos_r"], v["oos_sel"]
    cost_rows, rand_rows, near = v["cost_rows"], v["rand_rows"], v["rand_rows"]
    beat, pct, wf_rows = v["beat"], v["pct"], v["wf_rows"]
    fr_m_ret, fr_m_turn, pct_m = v["fr_m_ret"], v["fr_m_turn"], v["pct_m"]
    cost_m_rows = v.get("cost_m_rows", [])
    TOP, FREQ = v["TOP"], v["FREQ"]

    o = ['<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>AQuant 多因子研究 v3</title><style>', CSS, '</style></head><body><div class="wrap">']
    o.append('<h1>AQuant 多因子研究 v3</h1>')
    o.append('<div class="sub">生成时间 %s ｜ 样本 %d 只 × %.1f 年（%s ~ %s）｜ 研究本金 %.0f 亿</div>'
             % (time.strftime("%Y-%m-%d %H:%M"), len(A), yrs, dates[0], dates[-1], CASH / 1e8))

    # ---- KPI ----
    alpha_pp = (fr_ret - bg_ret) * 100          # composite 相对"纯门控"多赚的（选股 alpha 代理）
    o.append('<div class="kpi">')
    o.append('<div><span>选股 alpha（相对纯门控）</span><b class="%s">%+.2fpp</b><span style="font-size:11px">composite − 纯门控</span></div>'
             % (cls(alpha_pp), alpha_pp))
    o.append('<div><span>最终策略 vs 基准</span><b class="%s">%+.2fpp</b><span style="font-size:11px">%+.2f%% vs %+.2f%%</span></div>'
             % (cls(fr_ret - bench_ret), (fr_ret - bench_ret) * 100, fr_ret * 100, bench_ret * 100))
    o.append('<div><span>年换手（半年频）</span><b>%.0f%%</b><span style="font-size:11px">月度化将达 1464%%</span></div>' % (fr_turn * 100))
    o.append('<div><span>月度随机对照</span><b class="%s">第 %d 百分位</b><span style="font-size:11px">对齐 IC 周期后</span></div>'
             % ("ok" if pct_m >= 80 else ("warn" if pct_m >= 60 else "bad"), pct_m))
    o.append('</div>')

    # ---- 一、方法 ----
    o.append('<h2>一、这次改了什么 / 为什么这么做</h2>')
    o.append('''<div class="box info">v2 的死结是：<b>风控成立、选股无 alpha</b>。单一反转因子在随机选股对照里
    排第 10 百分位——它的收益 ≈ "市场平均 + 熊市空仓"，<b>没有选股能力</b>。<br>
    这一版不再依赖单一因子，而是建立<b>多因子横截面模型</b>：先用 rank-IC 验证每个因子是否真有预测力，
    只把通过检验的因子合成进组合。这是把"单因子赌博"升级为"因子投资"的标准做法。</div>''')
    o.append('''<table><thead><tr><th>环节</th><th>做法</th><th>目的</th></tr></thead><tbody>
    <tr><td>因子库</td><td>5 个经典因子：短期反转 / 中期动量 / 低波动 / 接近一年新高 / 时序趋势</td><td>覆盖主要风格，且全部可由 OHLCV 历史计算（不掺假基本面）</td></tr>
    <tr><td>方向固定</td><td>每个因子方向由金融学理论定（fac.SIGNS），IC 只验证不去改</td><td>禁止"看结果倒推方向"的作弊</td></tr>
    <tr><td>选因子</td><td>在样本内用 rank-IC 的 t 值筛选（方向对 + t&gt;1.5）</td><td>只用样本内信息决定因子集</td></tr>
    <tr><td>验证</td><td>样本外 / 成本 / 随机选股对照 / 滚动前推，全程不泄露给选因子</td><td>真实检验 alpha</td></tr>
    <tr><td>组合</td><td>低频（半年频）+ 分散（Top%d）+ 指数门控，沿用 v2 已证结论</td><td>不重复 grid 搜参，避免新一轮过拟合</td></tr>
    </tbody></table>''' % TOP)

    # ---- 二、因子 IC ----
    o.append('<h2>二、因子 rank-IC 检验</h2>')
    o.append('<p>rank-IC = 因子横截面取值 与 未来 1 个月收益的 Spearman 相关系数。'
             '它能直接告诉我们：<b>按这个因子排序选股，下一月到底有没有用</b>。'
             '下表"方向IC"已按理论方向调整（正值=该因子指向的方向真能赚钱）。</p>')
    o.append('<table><thead><tr><th>因子</th><th>理论方向</th><th>样本内方向IC</th>'
             '<th>方向t值</th><th>IC&gt;0 占比</th><th>全样本方向IC</th><th>结论</th></tr></thead><tbody>')
    for row in ic_rows:
        k = row["k"]
        if not row["ok"]:
            o.append('<tr><td>%s</td><td colspan="6" class="neg">样本不足，跳过</td></tr>' % k)
            continue
        r = row["r"]
        fi = full_ic.get(k)
        fi_dir = "%.4f" % fi["mean_dir"] if fi else "—"
        sdir = "做多" if fac.SIGNS[k] > 0 else "做空"
        verdict = ('<span class="pos">✓ 入选</span>' if row["keep"]
                   else '<span class="neg">✗ 剔除（方向错或 t 不足）</span>')
        o.append('<tr><td><b>%s</b></td><td>%s</td><td>%s</td><td>%.2f</td><td>%.0f%%</td><td>%s</td><td>%s</td></tr>'
                 % (k, sdir, ("%+.4f" % r["mean_dir"]), r["t_dir"], r["hit"] * 100, fi_dir, verdict))
    o.append('</tbody></table>')
    if selected:
        o.append('<div class="box ok"><b>入选因子集：</b>%s。这些因子在样本内的方向 IC 为正且统计显著（t&gt;1.5），'
                 '说明它们携带真实的横截面预测信息，不是噪声。</div>' % "、".join(selected))
    else:
        o.append('<div class="box bad"><b>没有任何因子通过检验。</b>这本身就是重要结论——'
                 '在当前样本（含幸存者偏差的宽基池）里，这些经典因子没有显著 alpha，'
                 '组合退化为"纯门控"，即 v2 的"更抗跌指数壳子"。</div>')

    # ---- 三、composite vs 纯门控 vs 基准 ----
    o.append('<h2>三、关键对照：选股到底有没有用</h2>')
    o.append('<div class="box warn">这是 v3 的<b>核心问题</b>：composite（多因子选股+门控）比"纯门控（不选股，只做市场择时）"'
             '多赚的部分，才是真正的<b>选股 alpha</b>。如果两者持平，说明选股是零贡献。</div>')
    o.append('<table><thead><tr><th>策略</th><th>总收益</th><th>相对基准</th><th>年换手</th><th>交易笔数</th></tr></thead><tbody>')
    o.append('<tr style="background:#fff7ed"><td><b>多因子+门控（composite）</b></td><td>%s</td>'
             '<td><b class="%s">%+.2fpp</b></td><td>%.0f%%</td><td>%d</td></tr>'
             % (pc(fr_ret), cls(fr_ret - bench_ret), (fr_ret - bench_ret) * 100, fr_turn * 100, fr["n_trades"]))
    o.append('<tr><td>纯门控（不选股，仅市场择时）</td><td>%s</td><td>%+.2fpp</td><td>%.0f%%</td><td>%d</td></tr>'
             % (pc(bg_ret), (bg_ret - bench_ret) * 100,
                sum(t["amount"] for t in base_gate["trades"]) / CASH / yrs * 100, base_gate["n_trades"]))
    o.append('<tr><td>等权买入持有（基准）</td><td>%s</td><td>—</td><td>0%%</td><td>%s</td></tr>'
             % (pc(bench_ret), "（不交易）"))
    o.append('</tbody></table>')
    ab = "ok" if alpha_pp > 1.0 else ("warn" if alpha_pp > -1.0 else "bad")
    o.append('<div class="box %s">composite 相对纯门控的选股超额 = <b>%+.2fpp</b>。%s</div>'
             % (ab, alpha_pp,
                "选股确有正贡献，alpha 成立。" if alpha_pp > 1.0 else
                "选股贡献微弱或为负，组合仍主要靠风控/择时。" if alpha_pp > -1.0 else
                "选股甚至拖累，因子合成需要重做。"))

    # ---- 四、样本外 ----
    o.append('<h2>四、样本外检验</h2>')
    o.append('<table><thead><tr><th>策略</th><th>样本外收益</th><th>样本外基准</th><th>样本外超额</th></tr></thead><tbody>')
    o.append('<tr style="background:#fff7ed"><td><b>composite+门控</b></td><td>%s</td><td>%s</td><td><b class="%s">%+.2fpp</b></td></tr>'
             % (pc(oos_r["ret"]), pc(oos_r["bench"]), cls(oos_r["excess"]), oos_r["excess"] * 100))
    o.append('<tr><td>纯门控（对照）</td><td>%s</td><td>%s</td><td><b class="%s">%+.2fpp</b></td></tr>'
             % (pc(oos_sel["ret"]), pc(oos_sel["bench"]), cls(oos_sel["excess"]), oos_sel["excess"] * 100))
    o.append('</tbody></table>')
    o.append('<div class="box info">选因子只用了样本内，这一段（%s 之后）从未参与任何参数/因子决策，'
             '是干净的 Out-of-Sample 检验。</div>' % dates[is_end])

    # ---- 五、成本 ----
    o.append('<h2>五、成本敏感性（composite+门控）</h2>')
    o.append('<table><thead><tr><th>双边滑点</th><th>策略收益</th><th>基准收益</th><th>相对基准</th>'
             '<th>交易笔数</th><th>相对 0.05%% 损耗</th></tr></thead><tbody>')
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
    o.append('<div class="box warn">对照组与 composite <b>完全相同</b>的调仓频率、持仓数、市场门控，'
             '唯一区别是把"按因子打分选 Top%d"换成"随机抽 Top%d"。比的是纯选股 alpha。</div>' % (TOP, TOP))
    o.append('<table><thead><tr><th>策略</th><th>年换手</th><th>总收益</th></tr></thead><tbody>')
    o.append('<tr style="background:#fff7ed"><td><b>composite+门控</b></td><td>%.0f%%</td><td><b>%s</b></td></tr>'
             % (fr_turn * 100, pc(fr_ret)))
    for x in near:
        o.append('<tr><td>随机选股 #%d</td><td>%.0f%%</td><td>%s</td></tr>'
                 % (x["seed"], x["turn"] * 100, pc(x["ret"])))
    o.append('</tbody></table>')
    boxc = "ok" if pct >= 80 else ("warn" if pct >= 60 else "bad")
    o.append('<div class="box %s">composite 在 %d 个随机策略中跑赢 <b>%d 个（第 %d 百分位）</b>。%s</div>'
             % (boxc, len(near), beat, pct,
                "显著优于随机，选股信号有信息量。" if pct >= 80 else
                "与随机无异，选股 alpha 仍未被证实。" if pct < 60 else
                "略优于随机，优势不牢固。"))

    # ---- 七、滚动前推 ----
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

    # ---- 八、持有期错配检验 ----
    o.append('<h2>八、持有期错配检验：为什么选股变成了负 alpha</h2>')
    o.append('<div class="box info">因子 IC 是<b>按月</b>测的（预测未来 1 个月），但上面的 composite 是<b>半年</b>才调一次仓。'
             '信号说"下个月谁涨"，组合却要拿满半年——中间信号早就反转了。这是因子投资最常见的坑：'
             '<b>IC 的预测周期 ≠ 持仓周期</b>。下面把组合改成月度再平衡（匹配 IC 周期）看 alpha 是否翻转。</div>')
    o.append('<table><thead><tr><th>配置</th><th>总收益</th><th>年换手</th><th>相对基准</th>'
             '<th>随机对照百分位</th></tr></thead><tbody>')
    o.append('<tr style="background:#fff7ed"><td><b>半年频 composite+门控</b></td><td>%s</td><td>%.0f%%</td>'
             '<td><b class="%s">%+.2fpp</b></td><td>第 %d 百分位</td></tr>'
             % (pc(fr_ret), fr_turn * 100, cls(fr_ret - bench_ret), (fr_ret - bench_ret) * 100, pct))
    o.append('<tr><td>月频 composite+门控</td><td>%s</td><td>%.0f%%</td>'
             '<td><b class="%s">%+.2fpp</b></td><td>第 %d 百分位</td></tr>'
             % (pc(fr_m_ret), fr_m_turn * 100, cls(fr_m_ret - bench_ret), (fr_m_ret - bench_ret) * 100, pct_m))
    o.append('<tr><td>半年频 纯门控（不选股）</td><td>%s</td><td>%.0f%%</td><td>%+.2fpp</td><td>—</td></tr>'
             % (pc(bg_ret), sum(t["amount"] for t in base_gate["trades"]) / CASH / yrs * 100,
                (bg_ret - bench_ret) * 100))
    o.append('</tbody></table>')
    o.append('<h3>8.1 月度 composite 的成本敏感性（换手 1464%/年，必须过这关）</h3>')
    o.append('<table><thead><tr><th>双边滑点</th><th>策略收益</th><th>基准收益</th><th>相对基准</th>'
             '<th>交易笔数</th></tr></thead><tbody>')
    for r in cost_m_rows:
        ex = r["ret"] - r["bench"]
        o.append('<tr><td>%.2f%%</td><td>%s</td><td>%s</td><td><b class="%s">%+.2fpp</b></td><td>%d</td></tr>'
                 % (r["slip"] * 100, pc(r["ret"]), pc(r["bench"]), cls(ex), ex * 100, r["n"]))
    o.append('</tbody></table>')
    last_m = cost_m_rows[-1]["ret"]
    o.append('<div class="box %s">月度 composite 在滑点 %.2f%% 时收益 %s。%s</div>'
             % ("bad" if last_m < bench_ret else "warn" if last_m < 0 else "ok",
                cost_m_rows[-1]["slip"] * 100, pc(last_m),
                "高换手下收益被成本严重侵蚀，实盘几乎不可行——这是月度化的致命伤。" if last_m < bench_ret
                else "勉强扛住成本，但安全边际极薄，实盘需更保守的换手控制。" if last_m < 0
                else "在极端滑点下仍为正，说明 alpha 真实存在、非过拟合。"))

    flip = "ok" if pct_m >= 60 else ("warn" if pct_m > pct else "bad")
    o.append('<div class="box %s">月度化后，随机对照百分位从第 <b>%d</b> 升到第 <b>%d</b>。%s</div>'
             % (flip, pct, pct_m,
                "持仓周期对齐 IC 后，选股 alpha 显著改善——说明之前是'拿太久'吃掉了信号。" if pct_m >= 60
                else "即便对齐周期，选股仍未超越随机——因子本身在这样本上就是弱信号。" if pct_m > pct
                else "月度化反而更差，错配不是主因，因子本身无效。"))

    # ---- 九、结论 ----
    o.append('<h2>九、结论</h2>')
    if selected:
        o.append('<div class="box %s"><h3 style="margin-top:0">判定：%s</h3><ul>'
                 % (boxc if pct >= 80 else "warn",
                    "多因子选股有真实 alpha，可进小资金试跑" if pct >= 80 and alpha_pp > 1.0
                    else "选股方向对、但优势不稳固，仍需打磨" if pct >= 60
                    else "仍未超越随机，alpha 未证实"))
        o.append('<li><b>入选因子</b>：%s（样本内方向 IC 为正且 t&gt;1.5）。</li>' % "、".join(selected))
        o.append('<li><b>选股 alpha</b>：composite 相对纯门控 %+.2fpp；样本外超额 %+.2fpp；'
                 '随机对照第 %d 百分位。</li>' % (alpha_pp, oos_r["excess"] * 100, pct))
        o.append('<li><b>仍未解决</b>：幸存者偏差（退市股不在快照）、缺少基本面/估值因子（这是 alpha 的天花板）。</li>')
        o.append('</ul></div>')
    else:
        o.append('<div class="box bad"><h3 style="margin-top:0">判定：alpha 仍未找到</h3>'
                 '没有任何经典因子在当前样本通过 IC 检验，组合退化为 v2 的"更抗跌指数壳子"。'
                 '突破口在引入<b>估值/质量/基本面因子</b>——但那需要历史财务快照，当前免费数据源拿不到，是下一步必须补的。</div>')

    o.append('<footer>AQuant v3 ｜ 数据来源：腾讯行情、新浪财经 ｜ 本报告为策略研究记录，不构成投资建议</footer>')
    o.append('</div></body></html>')
    return "".join(o)


if __name__ == "__main__":
    main()
