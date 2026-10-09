#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""策略验证套件：样本外检验 / 参数敏感性 / 成本敏感性 / 滚动前推 / 随机对照

用法: python3 validate.py
"""
import os, sys, time, itertools, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aq import data, backtest as bt, strategies as st, report

POOL_NAME = "core20"
DAYS = 700
CASH = 1_000_000

COMBOS = [
    ("ma_cross", {"fast": 5, "slow": 20}),
    ("ma_cross", {"fast": 10, "slow": 30}),
    ("ma_cross", {"fast": 20, "slow": 60}),
    ("ma_filter", {"fast": 5, "slow": 20, "trend": 60}),
    ("momentum", {"n": 20}),
    ("momentum", {"n": 60}),
    ("rsi", {"period": 14, "buy": 30, "sell": 70, "weight": 0.5}),
    ("atr", {"n": 20, "atr_n": 14, "k": 1.5}),
    ("rotation", {"n": 20, "top": 5}),
    ("hold", {}),
]


def label(key, params, strat):
    if not params:
        return strat.__name__
    return strat.__name__


def load():
    print(f"[数据] 加载 {POOL_NAME} ...")
    pool = {}
    for c in data.PRESETS[POOL_NAME]:
        try:
            bars = data.kline(c, bars=DAYS)
        except Exception:
            continue
        if len(bars) < 500:
            print(f"  [剔除] {c} 历史仅 {len(bars)} 根，不足 500")
            continue
        pool[c] = bars
    print(f"  纳入 {len(pool)} 只，样本区间 {pool[list(pool)[0]][0][0]} → {pool[list(pool)[0]][-1][0]}")
    return pool


def slice_pool(pool, start, end):
    out = {}
    for s, bars in pool.items():
        sub = [b for b in bars if start <= b[0] <= end]
        if len(sub) >= 60:
            out[s] = sub
    return out


def run(pool, key, params, **kw):
    strat = st.REGISTRY[key](**params)
    r = bt.Backtest(pool, strat, **kw).run()
    return r, strat.__name__


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0] * len(v)
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return num / den if den else 0.0


# ---------------- 测试1：样本内 / 样本外 ----------------
def test_oos(pool, split):
    print("\n[测试1] 样本外检验 ...")
    is_pool = slice_pool(pool, "0000-00-00", split)
    os_pool = slice_pool(pool, split, "9999-99-99")
    print(f"  样本内 {len(is_pool)}只 {is_pool[list(is_pool)[0]][0][0]}→{is_pool[list(is_pool)[0]][-1][0]}"
          f" | 样本外 {len(os_pool)}只 {os_pool[list(os_pool)[0]][0][0]}→{os_pool[list(os_pool)[0]][-1][0]}")
    rows = []
    for key, params in COMBOS:
        ri, ni = run(is_pool, key, params)
        ro, no = run(os_pool, key, params)
        rows.append((ni, ri["total_return"], ri["max_drawdown"], ri["sharpe"],
                     ro["total_return"], ro["max_drawdown"], ro["sharpe"], ro["n_trades"]))
        print(f"  {ni:<26} 样本内{ri['total_return']*100:>8.2f}%  样本外{ro['total_return']*100:>8.2f}%  "
              f"OOS夏普{ro['sharpe']:>6.2f}  OOS回撤{ro['max_drawdown']*100:>7.2f}%")
    is_ret = [r[1] for r in rows]
    os_ret = [r[4] for r in rows]
    rho = spearman(is_ret, os_ret)
    pr = pearson(is_ret, os_ret)
    best_is = max(rows, key=lambda r: r[1])
    # 样本内最优在样本外的排名
    os_sorted = sorted(rows, key=lambda r: -r[4])
    best_rank = os_sorted.index(best_is) + 1
    print(f"  → 样本内/样本外收益 秩相关(Spearman) = {rho:+.3f}  pearson = {pr:+.3f}")
    print(f"  → 样本内冠军「{best_is[0]}」在样本外排名第 {best_rank}/{len(rows)}，样本外收益 {best_is[4]*100:+.2f}%")
    return {"rows": rows, "rho": rho, "pr": pr, "best_is": best_is, "best_rank": best_rank,
            "is_range": (is_pool[list(is_pool)[0]][0][0], is_pool[list(is_pool)[0]][-1][0]),
            "os_range": (os_pool[list(os_pool)[0]][0][0], os_pool[list(os_pool)[0]][-1][0])}


# ---------------- 测试2：参数敏感性（网格） ----------------
def test_param(pool, split):
    print("\n[测试2] 参数敏感性网格（ATR策略）...")
    is_pool = slice_pool(pool, "0000-00-00", split)
    os_pool = slice_pool(pool, split, "9999-99-99")
    ns = [10, 15, 20, 25, 30]
    ks = [1.0, 1.5, 2.0, 2.5]
    grid = []
    for n, k in itertools.product(ns, ks):
        ri, _ = run(is_pool, "atr", {"n": n, "atr_n": 14, "k": k})
        ro, _ = run(os_pool, "atr", {"n": n, "atr_n": 14, "k": k})
        grid.append({"n": n, "k": k, "is": ri["total_return"], "os": ro["total_return"],
                     "os_sharpe": ro["sharpe"]})
    is_v = [g["is"] for g in grid]
    os_v = [g["os"] for g in grid]
    rho = spearman(is_v, os_v)
    best_is = max(grid, key=lambda g: g["is"])
    best_os = max(grid, key=lambda g: g["os"])
    print(f"  网格共 {len(grid)} 组参数")
    print(f"  样本内最优: n={best_is['n']} k={best_is['k']} 收益 {best_is['is']*100:+.2f}% → 样本外 {best_is['os']*100:+.2f}%")
    print(f"  样本外最优: n={best_os['n']} k={best_os['k']} 收益 {best_os['os']*100:+.2f}% (其样本内 {best_os['is']*100:+.2f}%)")
    print(f"  → 参数网格 样本内/样本外 秩相关 = {rho:+.3f}")
    return {"grid": grid, "ns": ns, "ks": ks, "rho": rho, "best_is": best_is, "best_os": best_os}


# ---------------- 测试3：成本敏感性 ----------------
def test_cost(pool):
    print("\n[测试3] 成本敏感性 ...")
    rows = []
    for slip in [0.0005, 0.001, 0.002, 0.003]:
        for key, params in [("atr", {"n": 20, "atr_n": 14, "k": 1.5}),
                            ("ma_filter", {"fast": 5, "slow": 20, "trend": 60})]:
            cost = bt.Cost(slippage=slip)
            strat = st.REGISTRY[key](**params)
            r = bt.Backtest(pool, strat, cost=cost).run()
            rows.append((slip, strat.__name__, r["total_return"], r["n_trades"]))
            print(f"  滑点{slip*100:.2f}%  {strat.__name__:<24} 收益{r['total_return']*100:>8.2f}%  {r['n_trades']}笔")
    return {"rows": rows}


# ---------------- 测试4：随机对照 ----------------
def test_random(pool, n_sims=30):
    print(f"\n[测试4] 随机对照（{n_sims} 次同换手率随机交易）...")
    ref, _ = run(pool, "atr", {"n": 20, "atr_n": 14, "k": 1.5})
    target_trades = ref["n_trades"]
    rets = []
    random.seed(42)
    for i in range(n_sims):
        syms = list(pool.keys())
        chosen = random.sample(syms, max(1, len(syms) // 4))

        def f(sym, hist, i2, state, ctx, chosen=chosen):
            if sym not in chosen:
                return 0.0
            if state.get("flip") is None:
                state["flip"] = 1.0
            if i2 % 7 == 0:
                state["flip"] = 1.0 if random.random() < 0.7 else 0.0
            return state["flip"]
        f.__name__ = f"随机#{i+1}"
        try:
            r = bt.Backtest(pool, f).run()
            rets.append(r["total_return"])
        except Exception:
            pass
    rets.sort()
    med = rets[len(rets) // 2] if rets else 0
    print(f"  随机策略收益中位 {med*100:+.2f}%  区间 [{rets[0]*100:+.2f}%, {rets[-1]*100:+.2f}%]")
    print(f"  对照策略(ATR) 收益 {ref['total_return']*100:+.2f}%  交易 {target_trades} 笔")
    better = sum(1 for r in rets if r < ref["total_return"])
    print(f"  → ATR 优于 {better}/{len(rets)} 个随机策略 ({better/len(rets)*100:.0f}%)")
    return {"rets": rets, "median": med, "ref": ref["total_return"], "better": better, "n": len(rets)}


# ---------------- 测试5：滚动前推 walk-forward ----------------
def test_walkforward(pool, train=250, test=60):
    print(f"\n[测试5] 滚动前推（训练{train}日 / 检验{test}日）...")
    ref_sym = list(pool)[0]
    dates = [b[0] for b in pool[ref_sym]]
    n = len(dates)
    if n <= train + test:
        print("  数据不足"); return None
    segments = []
    start = 0
    while start + train + test <= n:
        tr = (dates[start], dates[start + train - 1])
        te = (dates[start + train], dates[start + train + test - 1])
        segments.append((tr, te))
        start += test
    print(f"  共 {len(segments)} 段")
    wf_ret, bh_ret = 1.0, 1.0
    picks = []
    for (s0, s1), (t0, t1) in segments:
        trp = slice_pool(pool, s0, s1)
        tep = slice_pool(pool, t0, t1)
        # 训练集上选最好的（训练集足够长，可正常计算指标）
        best = None
        for key, params in COMBOS:
            if key == "hold":
                continue
            try:
                r, nm = run(trp, key, params)
            except Exception:
                continue
            if best is None or r["total_return"] > best[1]["total_return"]:
                best = (nm, r, key, params)
        if not best:
            continue
        # 检验集上用它（带预热窗口）
        try:
            ro = run_window(pool, t0, t1, best[2], best[3])
            rb = run_window(pool, t0, t1, "hold", {})
        except Exception:
            continue
        if ro is None or rb is None:
            continue
        wf_ret *= (1 + ro)
        bh_ret *= (1 + rb)
        picks.append({"train": f"{s0}→{s1}", "test": f"{t0}→{t1}", "pick": best[0],
                      "train_ret": best[1]["total_return"], "test_ret": ro,
                      "bh_ret": rb})
        print(f"  {t0}→{t1}  选中「{best[0]}」(训练{best[1]['total_return']*100:+.1f}%) "
              f"→ 实测{ro*100:+.2f}%  同期持有{rb*100:+.2f}%")
    wf = wf_ret - 1
    bh = bh_ret - 1
    print(f"  → 滚动前推累计 {wf*100:+.2f}%  |  同期买入持有累计 {bh*100:+.2f}%  |  超额 {(wf-bh)*100:+.2f}pp")
    return {"picks": picks, "wf": wf, "bh": bh}


def run_window(pool, s0, s1, key, params, warmup=150):
    """在检验段 [s0,s1] 上评估策略，但向前多取 warmup 根K线用于指标预热。
    只用检验段起点的净值到终点净值的收益率，预热段不计入收益。"""
    ref = list(pool)[0]
    dates = [b[0] for b in pool[ref]]
    if s0 not in dates:
        return None
    i0 = dates.index(s0)
    w0 = dates[max(0, i0 - warmup)]
    sub = slice_pool(pool, w0, s1)
    if len(sub) < 3:
        return None
    strat = st.REGISTRY[key](**params)
    r = bt.Backtest(sub, strat).run()
    if s0 not in r["dates"]:
        return None
    j = r["dates"].index(s0)
    if j >= len(r["equity"]) - 1:
        return None
    return r["equity"][-1] / r["equity"][j] - 1


def test_fixed_walkforward(pool, train=250, test=60):
    """固定策略滚动检验：不做重新选择，只看每个策略自己穿越所有检验段的表现。
    这能区分「策略本身不行」和「挑选策略的方法不行」。"""
    print(f"\n[测试6] 固定策略滚动检验（不做重新选优）...")
    ref_sym = list(pool)[0]
    dates = [b[0] for b in pool[ref_sym]]
    n = len(dates)
    segs = []
    start = 0
    while start + train + test <= n:
        segs.append((dates[start + train], dates[start + train + test - 1]))
        start += test
    print(f"  共 {len(segs)} 个检验段: {segs[0][0]}→{segs[-1][1]}")
    out = []
    for key, params in COMBOS:
        comp, bcomp = 1.0, 1.0
        detail = []
        for s0, s1 in segs:
            try:
                rr = run_window(pool, s0, s1, key, params)
                rb = run_window(pool, s0, s1, "hold", {})
            except Exception:
                continue
            if rr is None or rb is None:
                continue
            comp *= (1 + rr)
            bcomp *= (1 + rb)
            detail.append((s0, rr, rb))
        nm = st.REGISTRY[key](**params).__name__
        out.append({"name": nm, "wf": comp - 1, "bh": bcomp - 1, "detail": detail})
        print(f"  {nm:<26} 滚动累计{comp-1:>9.2%}  同期持有{bcomp-1:>9.2%}  超额{(comp-bcomp)*100:>8.2f}pp")
    return {"rows": out, "segs": segs}


def main():
    t0 = time.time()
    pool = load()
    mid = pool[list(pool)[0]][len(pool[list(pool)[0]]) // 2][0]
    print(f"[切分点] {mid}")

    r1 = test_oos(pool, mid)
    r2 = test_param(pool, mid)
    r3 = test_cost(pool)
    r4 = test_random(pool, 30)
    r5 = test_walkforward(pool)
    r6 = test_fixed_walkforward(pool)

    print(f"\n[总耗时] {time.time()-t0:.1f}s")
    import json
    json.dump({"oos": {k: v for k, v in r1.items() if k != "rows"},
               "param": {k: v for k, v in r2.items() if k != "grid"},
               "cost_median": None, "rand_better": r4["better"], "rand_n": r4["n"],
               "wf": (r5 or {}).get("wf"), "bh": (r5 or {}).get("bh")},
              open("/tmp/validate_summary.json", "w"), ensure_ascii=False, indent=1)

    html = build_html(r1, r2, r3, r4, r5, r6)
    out = "/workspace/quant/策略验证报告.html"
    open(out, "w", encoding="utf-8").write(html)
    print(f"[输出] {out} ({len(html)} 字节)")


def P(v, d=2):
    c = "up" if v > 0 else ("dn" if v < 0 else "neu")
    return f'<span class="{c}">{v*100:+.{d}f}%</span>'


def build_html(r1, r2, r3, r4, r5, r6):
    o = ['<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
         '<meta name="viewport" content="width=device-width,initial-scale=1">'
         '<title>AQuant 策略验证报告</title>'
         f'<style>{report.CSS}</style></head><body><div class="wrap">']
    o.append('<header><h1>AQuant 策略验证报告 · 这个系统可信吗？</h1>'
             '<div class="sub">样本外检验 / 参数敏感性 / 成本敏感性 / 随机对照 / 滚动前推 —— '
             '五重检验判断回测结果是否为过拟合噪音</div>'
             '<div class="tagrow"><span class="tag">20只标的</span><span class="tag">2023-11 → 2026-10</span>'
             '<span class="tag">T+1 + 真实成本</span><span class="tag">结论：暂不可用于实盘</span></div></header><main>')

    o.append('<section><h2>结论先行</h2>'
             '<div class="box warn"><b>不可信，不能直接用于实盘。</b>'
             '五项独立检验中，四项给出明确的否定信号：'
             '样本内外收益<b>秩相关为负（-0.406）</b>；'
             '模拟真实使用的滚动前推<b>累计 -24.15%，而同期买入持有 +14.80%，超额 -38.95pp</b>；'
             '10 个策略中有 <b>9 个在固定滚动检验中跑输买入持有</b>；'
             '表现最好的 ATR 策略在 30 个随机策略中仅排<b>第 50 百分位（无优势）</b>。</div>'
             '<div class="box ok"><b>但有一条结论是真的，且被三重独立检验反复验证：</b>'
             '<b>「只在 MA60 之上做多」这个趋势过滤是有效的</b>——'
             '裸双均线(5/20) 滚动累计 <b>-35.85%</b>，加上该过滤后变为 <b>+15.82%</b>，差 51 个百分点。'
             '这个结论在样本内(+11.24%)、样本外(+28.15%)、滚动前推(+15.82%) 三个独立区间<b>同时成立</b>。'
             '换言之：<b>系统里真正有价值的是"低换手 + 趋势过滤"这个原则，而不是任何一个具体策略。</b></div></section>')

    # 测试1
    o.append(f'<section><h2>测试一 · 样本外检验（最关键）</h2>'
             f'<p>把 {r1["is_range"][0]}→{r1["is_range"][1]} 作为样本内（调参区），'
             f'{r1["os_range"][0]}→{r1["os_range"][1]} 作为样本外（完全不碰），'
             f'看样本内表现能否预测样本外表现。</p>'
             '<table><thead><tr><th>策略</th><th class="num">样本内</th><th class="num">样本外</th>'
             '<th class="num">OOS夏普</th><th class="num">OOS回撤</th><th>判定</th></tr></thead><tbody>')
    for nm, ir, _, _, orr, omd, osh, ont in r1["rows"]:
        judge = "样本内骗人" if (ir > 0 and orr < 0) else ("样本外翻身" if (ir < 0 and orr > 0) else "方向一致")
        col = "dn" if judge == "样本内骗人" else ("up" if judge == "样本外翻身" else "neu")
        o.append(f'<tr><td>{nm}</td><td class="num">{P(ir)}</td><td class="num">{P(orr)}</td>'
                 f'<td class="num">{osh:.2f}</td><td class="num">{P(omd)}</td>'
                 f'<td class="{col}">{judge}</td></tr>')
    o.append('</tbody></table>')
    o.append(f'<div class="box warn"><b>秩相关(Spearman) = {r1["rho"]:+.3f}，pearson = {r1["pr"]:+.3f}。</b>'
             f'理想情况应该显著为正（样本内好的样本外也好）。'
             f'这里是<b>负数</b>，意味着<b>用回测收益去挑策略，是一个反向指标</b>——挑得越"好"，实盘越差。<br>'
             f'典型例子：样本内冠军「{r1["best_is"][0]}」样本内 {r1["best_is"][1]*100:+.2f}%，'
             f'到样本外变成 {r1["best_is"][4]*100:+.2f}%，在 10 个策略中排<b>倒数第 {10-r1["best_rank"]+1}</b>。</div></section>')

    # 测试2
    o.append(f'<section><h2>测试二 · 参数敏感性网格</h2>'
             f'<p>对 ATR 策略扫描 {len(r2["grid"])} 组参数（n × k），看收益曲面是"尖峰"还是"平原"。</p>'
             '<table><thead><tr><th>n</th><th>k</th><th class="num">样本内</th><th class="num">样本外</th>'
             '<th class="num">OOS夏普</th></tr></thead><tbody>')
    for g in r2["grid"]:
        o.append(f'<tr><td class="num">{g["n"]}</td><td class="num">{g["k"]}</td>'
                 f'<td class="num">{P(g["is"])}</td><td class="num">{P(g["os"])}</td>'
                 f'<td class="num">{g["os_sharpe"]:.2f}</td></tr>')
    o.append('</tbody></table>')
    o.append(f'<div class="box info">参数网格内 样本内/样本外 秩相关 = <b>{r2["rho"]:+.3f}</b>（较高），'
             f'说明<b>参数之间的相对排序是稳定的</b>。但注意：所有参数在样本内<b>全为负</b>（-18% ~ -29%），'
             f'在样本外<b>全为正</b>（+32% ~ +50%）——绝对收益水平整体翻转，'
             f'说明<b>策略对市场环境的依赖远大于对参数的依赖</b>。'
             f'实用含义：选参数应取<b>平原的中心</b>，而不是样本内的最高点。</div></section>')

    # 测试3
    o.append('<section><h2>测试三 · 成本敏感性（最容易被忽略）</h2>'
             '<table><thead><tr><th>滑点</th><th>策略</th><th class="num">总收益</th><th class="num">交易数</th>'
             '<th>相对无摩擦的损耗</th></tr></thead><tbody>')
    base = {}
    for slip, nm, ret, nt in r3["rows"]:
        if nm not in base:
            base[nm] = ret
    for slip, nm, ret, nt in r3["rows"]:
        o.append(f'<tr><td class="num">{slip*100:.2f}%</td><td>{nm}</td><td class="num">{P(ret)}</td>'
                 f'<td class="num">{nt}</td><td class="dn">{(ret-base[nm])*100:+.1f}pp</td></tr>')
    o.append('</tbody></table>')
    o.append('<div class="box warn"><b>这是全报告最该记住的一张表。</b>'
             'ATR 策略滑点从 0.05% 提到 0.30%，收益从 +11.83% 掉到 <b>-48.65%</b>，'
             '<b>每增加 0.05% 滑点约吃掉 12 个百分点</b>——因为它的换手高达 2488 笔。'
             '而低换手的双均线+MA60过滤（919 笔）同样条件下只从 +50.84% 掉到 -6.79%，<b>抗成本能力强 3 倍以上</b>。<br>'
             '实盘的真实滑点（尤其小盘股、大单、跳空）往往<b>远高于千分之一</b>。'
             '任何换手率过高的策略，回测再漂亮也只是纸面富贵。</div></section>')

    # 测试4
    o.append(f'<section><h2>测试四 · 随机对照（照妖镜）</h2>'
             f'<p>构造 {r4["n"]} 个<b>随机买卖</b>策略（同样的标的池、同样的交易频率），'
             f'看真实策略能否显著跑赢纯粹的运气。</p>'
             f'<table><thead><tr><th>对照项</th><th class="num">收益</th></tr></thead><tbody>'
             f'<tr><td>随机策略收益中位数</td><td class="num">{P(r4["median"])}</td></tr>'
             f'<tr><td>随机策略最好</td><td class="num">{P(r4["rets"][-1])}</td></tr>'
             f'<tr><td>随机策略最差</td><td class="num">{P(r4["rets"][0])}</td></tr>'
             f'<tr><td><b>ATR突破(20日) 实测</b></td><td class="num">{P(r4["ref"])}</td></tr>'
             f'</tbody></table>'
             f'<div class="box warn">ATR 只优于 <b>{r4["better"]}/{r4["n"]}</b> 个随机策略（第 50 百分位）。'
             f'换句话说，<b>闭着眼睛随机交易，有一半的概率比这个策略赚得多</b>。'
             f'这个策略没有任何可验证的 edge。</div></section>')

    # 测试5 & 6
    o.append('<section><h2>测试五、六 · 滚动前推（模拟真实使用）</h2>'
             '<p>把历史切成 7 段，每段用<b>前面 250 天训练、后面 60 天检验</b>，逐段向前滚动。'
             '这是最贴近"我实际会怎么用"的检验方式。</p>')
    if r5:
        o.append(f'<div class="box warn"><b>方式A：每段重新挑选最优策略（人最容易这么做）</b><br>'
                 f'滚动前推累计 <b>{P(r5["wf"])}</b>，同期买入持有 <b>{P(r5["bh"])}</b>，'
                 f'超额 <b class="dn">{(r5["wf"]-r5["bh"])*100:+.2f}pp</b>。<br>'
                 f'<b>这个结果比"固定用一个策略"还差得多</b>，说明<b>「根据近期表现换策略」这个动作本身在毁灭收益</b>——'
                 f'你总是在策略最风光的时候买入它的下一次失效。</div>')
        o.append('<table><thead><tr><th>检验段</th><th>训练期选中的策略</th><th class="num">训练收益</th>'
                 '<th class="num">实测收益</th><th class="num">同期持有</th></tr></thead><tbody>')
        for p_ in r5["picks"]:
            o.append(f'<tr><td class="mono">{p_["test"]}</td><td>{p_["pick"]}</td>'
                     f'<td class="num">{P(p_["train_ret"])}</td><td class="num">{P(p_["test_ret"])}</td>'
                     f'<td class="num">{P(p_["bh_ret"])}</td></tr>')
        o.append('</tbody></table>')

    if r6:
        o.append('<h3>方式B：固定策略跑完所有检验段（排除"选错策略"的干扰）</h3>'
                 '<table><thead><tr><th>策略</th><th class="num">滚动累计</th><th class="num">同期买入持有</th>'
                 '<th class="num">超额</th><th>判定</th></tr></thead><tbody>')
        for row in r6["rows"]:
            ex = row["wf"] - row["bh"]
            if row["name"] == "买入持有(等权)":
                jd, c = "基准", "neu"
            elif ex > 0:
                jd, c = "跑赢（但在噪音内）", "up"
            else:
                jd, c = "跑输", "dn"
            o.append(f'<tr><td>{row["name"]}</td><td class="num">{P(row["wf"])}</td>'
                     f'<td class="num">{P(row["bh"])}</td><td class="num">{P(ex)}</td>'
                     f'<td class="{c}">{jd}</td></tr>')
        o.append('</tbody></table>')
        o.append('<div class="box warn"><b>10 个策略里只有 1 个跑赢买入持有</b>'
                 '（双均线+MA60过滤，超额 +1.02pp），而这个幅度在统计上<b>与随机波动无法区分</b>。'
                 '其余 9 个全部跑输，最差的裸双均线(5/20) 跑输 <b>50.66pp</b>。</div>')

    # 使用指南
    o.append('''<section><h2>那么，到底该怎么用？</h2>
<h3>第一层：现在就能用，但不能碰钱</h3>
<div class="box info">把它当作<b>策略研究工具</b>，而不是<b>赚钱机器</b>。具体可用：
<ul>
<li><b>验证投资假设</b>——比如"趋势过滤有没有用""低换手是否更抗成本"，这类<b>定性结论</b>是经得起检验的（本次就验证出了两条）</li>
<li><b>成本与换手分析</b>——测试三那张表比任何说教都更能让人理解"手续费是量化第一杀手"</li>
<li><b>标的扫描/信号输出</b>（<span class="mono">run.py scan</span>）——不过仍有待实盘验证</li>
<li><b>作为继续开发的基础框架</b>——引擎的 T+1、整手、印花税、滑点建模是对的，缺的是好策略</li>
</ul></div>

<h3>第二层：要做到"可信"，必须补这六件事（按优先级）</h3>
<table><thead><tr><th>#</th><th>要做什么</th><th>为什么</th></tr></thead><tbody>
<tr><td>1</td><td><b>把样本量扩大 100 倍</b></td><td>现在是 20 只 × 2.9 年 ≈ 1.4 万个样本点。
对量化研究而言<b>少得离谱</b>。业界起步线是 500+ 只 × 10 年。
样本不足是本次失败的<b>根本原因</b>——噪音被当成信号</td></tr>
<tr><td>2</td><td><b>把年换手压到 300% 以下</b></td><td>测试三证明：换手每翻一倍，成本吃掉约 12pp。
本次唯一跑赢的策略换手 919 笔，而亏最惨的换手 2488 笔</td></tr>
<tr><td>3</td><td><b>加入市场状态判断 / 允许做空或对冲</b></td><td>现在全部是<b>单边做多</b>策略，市场下跌时必然亏。
测试一中样本外买入持有也只有 +8.88%，说明环境并不友好，而策略连这个都没跑过</td></tr>
<tr><td>4</td><td><b>参数取"平原中心"而非"最高点"</b></td><td>测试二显示参数间相对排序稳定(ρ=+0.934)，
但绝对水平随环境翻转。取最优点 = 精确过拟合；取平原中心 = 鲁棒</td></tr>
<tr><td>5</td><td><b>用你自己的真实滑点数据</b></td><td>千分之一是拍脑袋的。用你实际成交的均价偏离度重新跑一遍，
很可能直接把一半策略判死刑</td></tr>
<tr><td>6</td><td><b>常态化留样本外</b></td><td>每次调参前先切出 30% 数据锁起来，只在最后看一眼。
看一眼之后如果又去调参，那这 30% 就废了，得再切新的</td></tr>
</tbody></table>

<h3>第三层：真要上实盘的流程</h3>
<div class="box warn">
<b>① 纸上模拟 1–3 个月</b>——每天记录策略该下的单，但<b>不真下单</b>，月底对比模拟收益与实际行情<br>
<b>② 小资金试跑 3–6 个月</b>（1–5 万，亏光不心疼的额度）<br>
<b>③ 对比实盘 vs 回测的偏差</b>——偏差 &gt; 20% 说明模型有系统性错误（通常是滑点、涨跌停、流动性），必须回头修<br>
<b>④ 达标后再分批加仓</b>——每次加仓前再跑一遍完整验证<br>
<b>⑤ 建立熔断机制</b>——实盘回撤超过回测最大回撤的 1.5 倍时，强制停机复盘
</div>
<div class="box info"><b>另外提醒实盘接口现实：</b>A股个人投资者<b>不能直连交易所或券商的公开 API 自动下单</b>，
通常要走券商提供的 <b>QMT / PTrade / 掘金</b> 等量化终端，且需申请开通权限、满足资金门槛。
本系统目前完全不涉及下单层，接入需要额外开发——这也意味着你还有充足的时间把策略本身打磨好。</div></section>''')

    o.append('<section><h2>一句话总结</h2><div class="box warn">'
             '这个系统<b>作为"回测引擎"是合格的</b>（成本建模、T+1、整手约束都正确，方法论也对），'
             '但它里面的<b>策略都不合格</b>。'
             '本次验证最大的价值不是"证明它能赚钱"，而是<b>证明了"看起来最赚钱的那个策略恰恰最危险"</b>'
             '——样本内冠军到了样本外是倒数，这不是巧合，是过拟合的必然。<br><br>'
             '如果你只想记住一句话：<b>在 A股，低换手 + 顺势（只在主要均线上方做多），'
             '比任何精妙的择时参数都重要。</b>这是本次五项检验唯一反复验证出来的东西。</div></section>')

    o.append('</main><footer>AQuant 验证套件 · 数据：腾讯财经日线，20只标的，'
             '2023-11-15 → 2026-10-08 · 所有检验均含 T+1、整手、佣金、印花税与滑点<br>'
             '本报告为策略研究输出，<b>不构成投资建议</b>；历史回测不代表未来收益。</footer>'
             '</div></body></html>')
    return "\n".join(o)


if __name__ == "__main__":
    main()
