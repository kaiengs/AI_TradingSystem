#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""报告层：生成回测看板 HTML"""

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;background:#f4f6f9;color:#1a1d24;line-height:1.65;padding:28px 14px}
.wrap{max-width:1120px;margin:0 auto;background:#fff;border-radius:14px;box-shadow:0 2px 22px rgba(0,0,0,.07);overflow:hidden}
header{background:linear-gradient(135deg,#0d1b2a,#163b5c);color:#fff;padding:30px 36px}
header h1{font-size:23px;font-weight:700}
header .sub{margin-top:8px;font-size:13px;opacity:.82}
.tagrow{margin-top:12px;display:flex;gap:7px;flex-wrap:wrap}
.tag{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.26);padding:2px 10px;border-radius:20px;font-size:11.5px}
main{padding:6px 36px 34px}
section{padding:22px 0;border-bottom:1px solid #edeff3}
section:last-child{border:none}
h2{font-size:18px;color:#0d1b2a;margin-bottom:10px;padding-left:11px;border-left:4px solid #1c6fb8}
p{font-size:14px;color:#333;margin:8px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:14px 0}
.kpi{background:#f8fafc;border:1px solid #e6eaf0;border-radius:10px;padding:13px 15px}
.kpi .lb{font-size:11.5px;color:#7b8698;letter-spacing:.3px}
.kpi .vl{font-size:21px;font-weight:700;margin-top:3px;font-variant-numeric:tabular-nums}
.kpi .ex{font-size:11px;color:#98a2b3;margin-top:2px}
.up{color:#d93025}.dn{color:#1a9c52}.neu{color:#334}
table{width:100%;border-collapse:collapse;margin:10px 0;font-size:12.5px}
th{background:#eef2f7;color:#0d1b2a;padding:8px 9px;text-align:left;border-bottom:2px solid #dbe2ec;white-space:nowrap}
td{padding:7px 9px;border-bottom:1px solid #eef0f4;font-variant-numeric:tabular-nums}
tr:hover td{background:#fafbfd}
.num{text-align:right}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px}
.box{padding:11px 15px;border-radius:0 8px 8px 0;font-size:13.5px;margin:12px 0}
.warn{background:#fff6f6;border-left:3px solid #d93025}
.info{background:#f0f6fc;border-left:3px solid #1c6fb8}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.legend{display:flex;gap:16px;font-size:12px;margin:6px 0 2px}
.legend i{display:inline-block;width:16px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}
footer{padding:16px 36px;background:#f7f8fa;font-size:11.5px;color:#8a94a6;border-top:1px solid #edeff3}
@media(max-width:760px){.grid2{grid-template-columns:1fr}main,header,footer{padding-left:18px;padding-right:18px}}
"""


def line_chart(series, width=680, height=240, pad_l=52, pad_b=26, pad_t=12, pad_r=12):
    """series: [(name, values, color)]，values 为净值序列(起点归一)"""
    vals = [v for _, vs, _ in series for v in vs if v is not None]
    if not vals:
        return '<p>（无数据）</p>'
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        hi = lo + 1
    n = max(len(vs) for _, vs, _ in series)
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b

    def X(i):
        return pad_l + (i / max(n - 1, 1)) * iw

    def Y(v):
        return pad_t + (1 - (v - lo) / (hi - lo)) * ih

    o = [f'<svg viewBox="0 0 {width} {height}" style="width:100%;height:auto">']
    o.append(f'<rect x="{pad_l}" y="{pad_t}" width="{iw}" height="{ih}" fill="#fbfcfe" stroke="#e6eaf0"/>')
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = Y(v)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+iw}" y2="{y:.1f}" stroke="#eef1f6"/>')
        o.append(f'<text x="{pad_l-6}" y="{y+3.5:.1f}" text-anchor="end" font-size="9.5" fill="#98a2b3">{v:.2f}</text>')
    for name, vs, color in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vs) if v is not None)
        o.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
    o.append(f'<line x1="{pad_l}" y1="{Y(1.0):.1f}" x2="{pad_l+iw}" y2="{Y(1.0):.1f}" '
             f'stroke="#c8ced8" stroke-width="1" stroke-dasharray="4 3"/>')
    o.append('</svg>')
    return "\n".join(o)


def area_chart(values, width=680, height=140, pad_l=52, pad_b=22, pad_t=10, pad_r=12):
    """回撤序列（负值）"""
    lo = min(values) if values else -0.01
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b

    def X(i):
        return pad_l + (i / max(len(values) - 1, 1)) * iw

    def Y(v):
        return pad_t + (v / lo) * ih      # lo 为负 -> 0 在顶部

    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(values))
    base = f"{pad_l},{pad_t} {pad_l+iw},{pad_t}"
    o = [f'<svg viewBox="0 0 {width} {height}" style="width:100%;height:auto">']
    o.append(f'<rect x="{pad_l}" y="{pad_t}" width="{iw}" height="{ih}" fill="#fbfcfe" stroke="#e6eaf0"/>')
    o.append(f'<polygon points="{base} {pts}" fill="#d93025" opacity=".18"/>')
    o.append(f'<polyline points="{pts}" fill="none" stroke="#d93025" stroke-width="1.5"/>')
    for k in range(3):
        v = lo * (k + 1) / 3
        y = Y(v)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+iw}" y2="{y:.1f}" stroke="#eef1f6"/>')
        o.append(f'<text x="{pad_l-6}" y="{y+3.5:.1f}" text-anchor="end" font-size="9.5" fill="#98a2b3">{v*100:.0f}%</text>')
    o.append('</svg>')
    return "\n".join(o)


def pct(v, digits=2, sign=True):
    cls = "up" if v > 0 else ("dn" if v < 0 else "neu")
    s = f"{v*100:+.{digits}f}%" if sign else f"{v*100:.{digits}f}%"
    return f'<span class="{cls}">{s}</span>'


def build(title, meta, res, pool_names=None, extra_html=""):
    b = res.get("benchmark") or {}
    eq = res["equity"]
    base = eq[0]
    norm = [v / base for v in eq]
    bcurve = [v / b["curve"][0] for v in b["curve"]] if b.get("curve") else None

    p = ['<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
         '<meta name="viewport" content="width=device-width,initial-scale=1">'
         f'<title>{title}</title><style>{CSS}</style></head><body><div class="wrap">']
    p.append(f'<header><h1>{title}</h1><div class="sub">{meta}</div>'
             f'<div class="tagrow">' + "".join(f'<span class="tag">{t}</span>' for t in
             [f'区间 {res["dates"][0]} → {res["dates"][-1]}', f'{len(res["dates"])} 个交易日',
              f'初始资金 ¥{int(base):,}', f'交易 {res["n_trades"]} 次']) + '</div></header><main>')

    # KPI
    p.append('<section><h2>一、绩效总览</h2><div class="kpis">')
    for lb, vl, ex in [
        ("总收益率", pct(res["total_return"]), "策略"),
        ("年化收益", pct(res["annual"]), "按252交易日折算"),
        ("最大回撤", pct(res["max_drawdown"]), "峰值到谷底"),
        ("夏普比率", f'<span class="neu">{res["sharpe"]:.2f}</span>', "无风险利率2%"),
        ("胜率", f'<span class="neu">{res["win_rate"]*100:.1f}%</span>', f'{res["n_closed"]} 笔平仓'),
        ("盈亏比", f'<span class="neu">{res["profit_factor"]:.2f}</span>' if res["profit_factor"] != float("inf") else '<span class="neu">∞</span>', "总盈利/总亏损"),
    ]:
        p.append(f'<div class="kpi"><div class="lb">{lb}</div><div class="vl">{vl}</div><div class="ex">{ex}</div></div>')
    if b:
        for lb, vl, ex in [
            ("基准收益", pct(b["ret"]), b.get("name", "基准")),
            ("超额收益", pct(res["total_return"] - b["ret"]), "策略 − 基准"),
        ]:
            p.append(f'<div class="kpi"><div class="lb">{lb}</div><div class="vl">{vl}</div><div class="ex">{ex}</div></div>')
    p.append('</div></section>')

    # 净值曲线
    p.append('<section><h2>二、净值曲线</h2>')
    p.append('<div class="legend">'
             '<span><i style="background:#1c6fb8"></i>策略净值</span>'
             + (f'<span><i style="background:#9aa3b0"></i>{b.get("name","基准")}</span>' if b else '')
             + '<span><i style="background:#c8ced8"></i>成本线(1.00)</span></div>')
    ss = [("策略", norm, "#1c6fb8")]
    if bcurve:
        ss.append((b.get("name", "基准"), bcurve, "#9aa3b0"))
    p.append(line_chart(ss))
    p.append('<h2 style="margin-top:18px">回撤走势</h2>')
    p.append(area_chart(res["drawdown"]))
    p.append('<div class="box info">净值以初始资金为 1.00 归一。回撤图为策略相对历史净值高点的回落幅度，'
             '越深的坑代表该策略在实盘中最难熬的阶段。</div></section>')

    # 持仓
    fp = res.get("final_positions") or {}
    p.append('<section><h2>三、期末持仓</h2>')
    if fp:
        p.append('<table><thead><tr><th>代码</th><th>名称</th><th class="num">持股</th>'
                 '<th class="num">成本</th><th class="num">市值(元)</th></tr></thead><tbody>')
        for s, v in sorted(fp.items(), key=lambda x: -x[1]["value"]):
            nm = (pool_names or {}).get(s, s)
            p.append(f'<tr><td class="mono">{s}</td><td>{nm}</td><td class="num">{v["shares"]}</td>'
                     f'<td class="num">{v["cost"]:.2f}</td><td class="num">{v["value"]:,.0f}</td></tr>')
        p.append('</tbody></table>')
    else:
        p.append('<p>期末空仓。</p>')
    p.append('</section>')

    # 交易明细
    trades = res["trades"]
    p.append('<section><h2>四、交易明细（最近 40 笔）</h2>')
    if trades:
        p.append('<table><thead><tr><th>日期</th><th>代码</th><th>方向</th><th class="num">股数</th>'
                 '<th class="num">价格</th><th class="num">金额</th><th class="num">费用</th>'
                 '<th class="num">实现盈亏</th></tr></thead><tbody>')
        for t in trades[-40:][::-1]:
            buy = t["side"] == "BUY"
            pnl = "" if buy else pct(t["pnl"], 0)
            p.append(f'<tr><td>{t["date"]}</td><td class="mono">{t["sym"]}</td>'
                     f'<td class="{"up" if buy else "dn"}">{"买入" if buy else "卖出"}</td>'
                     f'<td class="num">{t["shares"]}</td><td class="num">{t["price"]:.2f}</td>'
                     f'<td class="num">{t["amount"]:,.0f}</td><td class="num">{t["fee"]:,.1f}</td>'
                     f'<td class="num">{pnl}</td></tr>')
        p.append('</tbody></table>')
    else:
        p.append('<p>区间内无交易。</p>')
    p.append('</section>')

    if extra_html:
        p.append(extra_html)

    p.append('<section><h2>五、成本与规则假设</h2>'
             '<table><thead><tr><th>项目</th><th>设定</th></tr></thead><tbody>'
             '<tr><td>佣金</td><td>万分之 2.5，双边，单笔最低 5 元</td></tr>'
             '<tr><td>印花税</td><td>千分之 1，仅卖出端</td></tr>'
             '<tr><td>滑点</td><td>千分之 1，买卖双向各计</td></tr>'
             '<tr><td>交易制度</td><td>T+1（当日买入不可卖出）；100 股整手</td></tr>'
             '<tr><td>撮合价格</td><td>当日收盘价</td></tr>'
             '<tr><td>资金约束</td><td>100股整手，若单手金额 &gt; 该标的可用额度则跳过建仓'
             '（如100万本金分20份时，茅台单手需15万+，会被跳过）</td></tr>'
             '<tr><td>分红送转</td><td>未处理（使用前复权数据可部分规避）</td></tr>'
             '</tbody></table>'
             '<div class="box warn"><b>必读风险提示：</b>回测结果基于历史数据，'
             '<b>历史表现不代表未来收益</b>；本模型未考虑涨跌停无法成交、停牌、流动性冲击、'
             '盘中跳空等实盘摩擦，实盘结果通常<b>显著劣于</b>回测。本页仅为策略研究工具输出，不构成投资建议。</div>'
             '</section>')

    p.append('</main><footer>AQuant · A股量化回测引擎 | 数据源：腾讯财经日线 | '
             '本报告由本地引擎自动生成，仅供策略研究与教学，不构成任何投资建议。</footer>'
             '</div></body></html>')
    return "\n".join(p)


def compare_table(rows, title="策略对比"):
    """rows: [(策略名, 总收益, 年化, 最大回撤, 夏普, 胜率, 交易数)]"""
    p = [f'<section><h2>{title}</h2><table><thead><tr><th>策略</th><th class="num">总收益</th>'
         '<th class="num">年化</th><th class="num">最大回撤</th><th class="num">夏普</th>'
         '<th class="num">胜率</th><th class="num">交易数</th></tr></thead><tbody>']
    for name, tr, an, md, sh, wr, nt in rows:
        p.append(f'<tr><td><b>{name}</b></td><td class="num">{pct(tr)}</td><td class="num">{pct(an)}</td>'
                 f'<td class="num">{pct(md)}</td><td class="num">{sh:.2f}</td>'
                 f'<td class="num">{wr*100:.1f}%</td><td class="num">{nt}</td></tr>')
    p.append('</tbody></table></section>')
    return "\n".join(p)
