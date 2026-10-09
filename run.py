#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AQuant 命令行入口

用法:
  python3 run.py bt   --pool core20 --strategy ma_cross --days 500
  python3 run.py cmp  --pool core20 --days 500
  python3 run.py scan --limit 100 --top 20
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aq import data, backtest as bt, strategies as st, report, indicators as ind


def load_pool(name_or_codes, days):
    if name_or_codes in data.PRESETS:
        codes = data.PRESETS[name_or_codes]
    else:
        codes = [c.strip() for c in name_or_codes.split(",") if c.strip()]
    out = {}
    for i, c in enumerate(codes, 1):
        try:
            bars = data.kline(c, bars=days + 5)
        except Exception as e:
            print(f"  [跳过] {c}: {e}")
            continue
        if len(bars) < 60:
            print(f"  [跳过] {c}: 数据不足({len(bars)}根)")
            continue
        out[c] = bars
        print(f"  ({i}/{len(codes)}) {c} {len(bars)}根 {bars[0][0]}→{bars[-1][0]}")
    return out


def cmd_bt(a):
    print(f"[数据] 加载 {a.pool} ...")
    pool = load_pool(a.pool, a.days)
    if not pool:
        print("无有效标的"); return
    fn = st.REGISTRY[a.strategy]
    strat = fn(**json.loads(a.params)) if a.params else fn()
    print(f"[策略] {strat.__name__}")
    eng = bt.Backtest(pool, strat, cash=a.cash, rebalance=a.rebalance)
    print("[回测] 运行中 ...")
    t0 = time.time()
    res = eng.run()
    print(f"[完成] {time.time()-t0:.1f}s  总收益 {res['total_return']*100:+.2f}%  "
          f"最大回撤 {res['max_drawdown']*100:.2f}%  夏普 {res['sharpe']:.2f}  交易 {res['n_trades']} 次")
    names = {c: data.realtime([c]).get(c, {}).get("name", c) for c in pool}
    meta = (f"标的池 {a.pool}（{len(pool)}只） · 策略 {strat.__name__} · "
            f"调仓频率 {a.rebalance} · 初始资金 ¥{a.cash:,.0f} · 生成于 {time.strftime('%Y-%m-%d %H:%M')}")
    html = report.build(f"量化回测 · {strat.__name__} · {a.pool}", meta, res, names)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(html)
    print(f"[输出] {a.out}  ({len(html)} 字节)")
    return res


def cmd_cmp(a):
    print(f"[数据] 加载 {a.pool} ...")
    pool = load_pool(a.pool, a.days)
    if not pool:
        print("无有效标的"); return
    combos = [
        ("ma_cross", {"fast": 5, "slow": 20}),
        ("ma_cross", {"fast": 10, "slow": 30}),
        ("ma_filter", {"fast": 5, "slow": 20, "trend": 60}),
        ("momentum", {"n": 20}),
        ("momentum", {"n": 60}),
        ("rsi", {"period": 14, "buy": 30, "sell": 70, "weight": 0.5}),
        ("atr", {"n": 20, "atr_n": 14, "k": 1.5}),
        ("rotation", {"n": 20, "top": 5}),
        ("hold", {}),
    ]
    rows, best = [], None
    for key, params in combos:
        strat = st.REGISTRY[key](**params)
        try:
            res = bt.Backtest(pool, strat, cash=a.cash, rebalance=a.rebalance).run()
        except Exception as e:
            print(f"  [失败] {key}{params}: {e}")
            continue
        rows.append((strat.__name__, res["total_return"], res["annual"], res["max_drawdown"],
                     res["sharpe"], res["win_rate"], res["n_trades"]))
        print(f"  {strat.__name__:<28} 收益{res['total_return']*100:>8.2f}%  "
              f"回撤{res['max_drawdown']*100:>7.2f}%  夏普{res['sharpe']:>6.2f}  {res['n_trades']:>4}笔")
        if best is None or res["total_return"] > best[1]["total_return"]:
            best = (strat, res)

    rows.sort(key=lambda r: -r[1])
    html = ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>策略对比</title>"
            f"<style>{report.CSS}</style></head><body><div class='wrap'>"
            f"<header><h1>策略横向对比 · {a.pool}</h1><div class='sub'>"
            f"{len(pool)}只标的 · {a.days}个交易日 · 初始资金 ¥{a.cash:,.0f} · "
            f"调仓 {a.rebalance} · {time.strftime('%Y-%m-%d %H:%M')}</div></header><main>"
            + report.compare_table(rows)
            + ("<section><h2>读表须知</h2>"
               "<div class='box warn'><b>① 这是样本内(in-sample)结果。</b>"
               "所有策略都在同一段历史数据上评估，参数也是在这段数据上挑选的，"
               "表现最好的策略<b>几乎必然包含过拟合成分</b>——它只是\"最贴合这段历史\"，不等于\"未来最好\"。"
               "正确做法是留出样本外(out-of-sample)区间做验证，或做滚动前推测试(walk-forward)。<br><br>"
               "<b>② 看交易笔数与收益的关系。</b>"
               "双均线(5/20) 976笔 → -22%，绝对动量(20日) 2431笔 → -4.8%。"
               "换手越高的策略被手续费/印花税/滑点磨损越狠，这是A股量化的第一杀手。<br><br>"
               "<b>③ 看回撤而不只看收益。</b>"
               "ATR突破回撤仅 -11.7%，而收益最高；双均线(5/20)收益 -22% 且回撤 -32%。"
               "收益相近时，回撤小一半的策略实盘存活率高得多（亏50%要涨100%才回本）。</div></section>")
            + "</main><footer>AQuant · 策略对比 | 不构成投资建议</footer></div></body></html>")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(html)
    print(f"[输出] 对比报告 {a.out}")

    if best:
        s, r = best
        names = {c: data.realtime([c]).get(c, {}).get("name", c) for c in pool}
        meta = (f"标的池 {a.pool}（{len(pool)}只） · 最优策略 {s.__name__} · 调仓 {a.rebalance} · "
                f"初始资金 ¥{a.cash:,.0f}")
        out2 = a.out.replace(".html", "_最优.html")
        open(out2, "w", encoding="utf-8").write(
            report.build(f"最优策略 · {s.__name__} · {a.pool}", meta, r, names))
        print(f"[输出] 最优策略明细 {out2}")


def cmd_scan(a):
    print("[数据] 拉取全A股截面 ...")
    snap = data.market_snapshot(refresh=a.refresh)
    print(f"  全市场 {len(snap)} 只")
    # 注意量纲：amount 单位=元；nmc(流通市值) 单位=万元
    cands = [r for r in snap
             if "ST" not in (r.get("name") or "") and r.get("amount", 0) >= 8e7
             and r.get("nmc", 0) >= 5e5]
    cands.sort(key=lambda r: -r.get("amount", 0))
    cands = cands[:a.limit]
    print(f"  初筛 {len(cands)} 只（成交额≥8000万、流通市值≥50亿、剔除ST）")

    rows = []
    for i, r in enumerate(cands, 1):
        sym = r["symbol"]
        try:
            bars = data.kline(sym, bars=140)
        except Exception:
            continue
        if len(bars) < 70:
            continue
        c = [b[2] for b in bars]; h = [b[3] for b in bars]; l = [b[4] for b in bars]
        m20 = ind.momentum(c, 20)[-1] or 0
        m60 = ind.momentum(c, 60)[-1] or 0
        pos = ind.position_pct(c, 60)[-1] or 50
        sd = ind.stddev(c, 20)[-1] or 0
        ma20 = ind.sma(c, 20)[-1]; ma60 = ind.sma(c, 60)[-1]
        trend = 1 if (ma20 and ma60 and ma20 > ma60) else 0
        # 综合打分：动量 40% + 低位 20% + 趋势 20% + 低波动 20%
        score = m20 * 100 * 0.4 + (100 - pos) * 0.2 + trend * 100 * 0.2 + (10 - min(sd / c[-1] * 100, 10)) * 2
        rows.append({"code": r["code"], "sym": sym, "name": r["name"], "price": c[-1],
                     "m20": m20, "m60": m60, "pos": pos, "trend": trend,
                     "vol": sd / c[-1] * 100, "amount": r["amount"], "nmc": r["nmc"],
                     "pct": r["changepercent"], "score": score})
        if i % 25 == 0:
            print(f"  ...{i}/{len(cands)}")
    rows.sort(key=lambda x: -x["score"])

    body = ['<table><thead><tr><th>#</th><th>代码</th><th>名称</th><th class="num">现价</th>'
            '<th class="num">20日动量</th><th class="num">60日位置</th><th>趋势</th>'
            '<th class="num">波动率</th><th class="num">成交额(亿)</th><th class="num">综合分</th>'
            '</tr></thead><tbody>']
    for i, r in enumerate(rows[:a.top], 1):
        body.append(f'<tr><td>{i}</td><td class="mono">{r["code"]}</td><td><b>{r["name"]}</b></td>'
                    f'<td class="num">{r["price"]:.2f}</td>'
                    f'<td class="num">{report.pct(r["m20"])}</td>'
                    f'<td class="num">{r["pos"]:.0f}%</td>'
                    f'<td>{"多头" if r["trend"] else "—"}</td>'
                    f'<td class="num">{r["vol"]:.2f}%</td>'
                    f'<td class="num">{r["amount"]/1e8:.1f}</td>'
                    f'<td class="num"><b>{r["score"]:.1f}</b></td></tr>')
    body.append('</tbody></table>')
    html = ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>选股信号</title>"
            f"<style>{report.CSS}</style></head><body><div class='wrap'>"
            f"<header><h1>A股多因子选股信号</h1><div class='sub'>"
            f"候选池 {len(cands)} 只 → 输出 Top {min(a.top,len(rows))} · "
            f"{time.strftime('%Y-%m-%d %H:%M')}</div></header><main>"
            f"<section><h2>打分口径</h2>"
            f"<p>综合分 = 20日动量×0.4 + 60日位置(越低越好)×0.2 + 均线多头×0.2 + 低波动×0.2。</p>"
            f"<div class='box info'>初筛条件：剔除 ST、近一日成交额 ≥ 8000万、流通市值 ≥ 50亿，"
            f"按成交额取前 {a.limit} 只以保证流动性。</div></section>"
            f"<section><h2>信号清单</h2>" + "".join(body) + "</section>"
            "</main><footer>AQuant · 选股信号 | 仅因子排序输出，不构成投资建议</footer>"
            "</div></body></html>")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(html)
    print(f"[输出] {a.out}  Top{min(a.top,len(rows))}")


def main():
    ap = argparse.ArgumentParser(prog="AQuant")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bt", help="单策略回测")
    b.add_argument("--pool", default="core20")
    b.add_argument("--strategy", default="ma_cross")
    b.add_argument("--params", default="")
    b.add_argument("--days", type=int, default=500)
    b.add_argument("--cash", type=float, default=1_000_000)
    b.add_argument("--rebalance", default="daily", choices=["daily", "weekly"])
    b.add_argument("--out", default="/workspace/quant/回测报告.html")
    b.set_defaults(func=cmd_bt)

    c = sub.add_parser("cmp", help="多策略对比")
    c.add_argument("--pool", default="core20")
    c.add_argument("--days", type=int, default=500)
    c.add_argument("--cash", type=float, default=1_000_000)
    c.add_argument("--rebalance", default="daily", choices=["daily", "weekly"])
    c.add_argument("--out", default="/workspace/quant/策略对比.html")
    c.set_defaults(func=cmd_cmp)

    s = sub.add_parser("scan", help="全市场选股信号")
    s.add_argument("--limit", type=int, default=100)
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--refresh", action="store_true")
    s.add_argument("--out", default="/workspace/quant/选股信号.html")
    s.set_defaults(func=cmd_scan)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
