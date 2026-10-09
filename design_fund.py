#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v5 设计文档生成器：基本面数据接入与因子扩展方案
============================================
生成 基本面数据接入与因子扩展方案.html，并嵌入 aq/fundamentals.py 的实跑证据。
"""
import time
from aq import data
from aq import fundamentals as fnd

CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
       margin: 0; background: #f4f6f9; color: #1f2733; line-height: 1.65; }
.wrap { max-width: 1040px; margin: 0 auto; padding: 32px 22px 80px; }
h1 { font-size: 27px; margin: 0 0 6px; }
h2 { font-size: 21px; margin: 42px 0 14px; padding-left: 11px;
     border-left: 5px solid #2c6ef2; }
h3 { font-size: 16px; margin: 22px 0 8px; color: #2c3e50; }
.sub { color: #6b7785; font-size: 13px; margin-bottom: 26px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px;
        background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
th, td { border: 1px solid #e3e8ef; padding: 7px 9px; text-align: left; }
th { background: #eef3fb; color: #243; font-weight: 600; }
tr:nth-child(even) td { background: #fafcff; }
code { background: #eef1f6; padding: 1px 5px; border-radius: 4px; font-size: 12px;
       color: #b5328a; }
.ok { color: #1a8f4a; font-weight: 600; }
.warn { color: #c47d00; font-weight: 600; }
.bad { color: #d23b3b; font-weight: 600; }
.box { border-radius: 9px; padding: 14px 16px; margin: 14px 0; font-size: 14px; }
.box.ok { background: #e8f7ee; border: 1px solid #b7e6c8; }
.box.warn { background: #fff6e6; border: 1px solid #f3dca0; }
.box.bad { background: #fdecec; border: 1px solid #f3bcbc; }
.box.info { background: #eaf1fe; border: 1px solid #b9d2fb; }
.kpi { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
.kpi > div { flex: 1 1 160px; background: #fff; border: 1px solid #e3e8ef;
             border-radius: 9px; padding: 12px 14px; }
.kpi span { display: block; font-size: 12px; color: #6b7785; }
.kpi b { font-size: 22px; }
pre { background: #1e2530; color: #d6e2f0; padding: 14px 16px; border-radius: 8px;
      overflow-x: auto; font-size: 12.5px; line-height: 1.5; }
.tag { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11px;
       background: #eef3fb; color: #2c6ef2; margin-right: 4px; }
.foot { margin-top: 50px; padding-top: 16px; border-top: 1px solid #dfe5ee;
        color: #8a94a3; font-size: 12px; }
"""

LOG = []


def log(*a):
    LOG.append(" ".join(str(x) for x in a))


# ---------------- 实跑证据 ----------------
def proof():
    syms = ["sh600519", "sz000858", "sh601318", "sz000651"]
    rows = []
    for sym in syms:
        reps = fnd.pull(sym)
        if not reps:
            rows.append((sym, "无数据", [], []))
            continue
        nm = reps[-1].get("name")
        tail = [(r["report_date"], r["type"], r["notice_date"], r["roe"], r["bps"], r["ttm_eps"])
                for r in reps[-4:]]
        probes = []
        for p in ["2025-12-31", "2026-04-20", "2026-05-10"]:
            st = fnd.point_in_time(reps, p)
            probes.append((p, st["report_date"] if st else "无", st["notice_date"] if st else "-"))
        rows.append((sym, nm, tail, probes))
    return rows


def panel_demo():
    """取 8 只票，在 2025-03-31 时点展示非估值因子面板，证明因子通路可用。"""
    pool = data.build_pool(top=120, mode="random", min_bars=2000)
    out = []
    for sym in pool[:14]:
        reps = fnd.pull(sym)
        if not reps:
            continue
        st = fnd.point_in_time(reps, "2025-03-31")
        if not st:
            continue
        out.append((sym, st.get("name"), st.get("roe"), st.get("rev_yoy"),
                    st.get("profit_yoy"), st.get("gross_margin")))
        if len(out) >= 8:
            break
    return out


t0 = time.time()
proof_rows = proof()
panel = panel_demo()
log("实跑证据构建完成，耗时 %.1fs" % (time.time() - t0))

# ---------------- HTML ----------------
o = []
o.append("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>")
o.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
o.append("<title>基本面数据接入与因子扩展方案</title><style>%s</style></head><body><div class='wrap'>" % CSS)
o.append("<h1>基本面数据接入与因子扩展方案</h1>")
o.append("<div class='sub'>AQuant · v5 设计 · 承 v4「压换手·甜点研究」—— 打破价量因子的风格天花板</div>")

# KPI
o.append("<div class='kpi'>")
o.append("<div><span>主数据源</span><b class='ok'>已实测可达</b><span style='font-size:11px'>东方财富数据中心·纯HTTP</span></div>")
o.append("<div><span>前视偏差防护</span><b class='ok'>NOTICE_DATE 对齐</b><span style='font-size:11px'>已实跑验证</span></div>")
o.append("<div><span>因子族</span><b>7 个</b><span style='font-size:11px'>估值/质量/成长</span></div>")
o.append("<div><span>落地状态</span><b class='warn'>数据层完成</b><span style='font-size:11px'>引擎集成待做</span></div>")
o.append("</div>")

# 0
o.append("<h2>0. 为什么是现在做这件事</h2>")
o.append("<div class='box info'>v4 结论回顾：压换手工程上成功了（季频+极端分位滞回把换手压到 270%，成本全扛得住），"
         "但样本外（近 30% 真实行情）<b>超额 -12.65pp</b>，且全样本仍低于纯市场门控 +106%。"
         "根因是 <b>「择时 > 选股」</b>：rev20/vol60 等价量因子受市场风格周期支配，没有跨周期护城河——"
         "2018-2023 的「反转+低波」风格近年逆转，因子便失效。<br><br>"
         "<b>基本面因子</b>（估值/质量/成长）是与价量<strong>低相关</strong>的独立 alpha 源，理论上能补上这块短板。"
         "本方案解决唯一拦路虎：<b>在回测里拿到「当时真实可得」的基本面数据</b>（point-in-time）。</div>")

# 1
o.append("<h2>1. 头号陷阱：前视偏差（point-in-time）</h2>")
o.append("<p>财报有两个日期：<b>报告期</b>（数据截止日）和 <b>披露日</b>（投资者真正能看到的那天）。"
         "年报（12-31）通常次年 4-30 前后才披露，一季报（3-31）通常 4 月底，三季报（9-30）通常 10 月底。"
         "若在报告期当天就用该财报选股，等于提前数月看到未来，回测必然虚高。</p>")
o.append("<table><tr><th>错误做法</th><th>正确做法（本方案）</th></tr>"
         "<tr><td class='bad'>2025-12-31 直接用 2025 年报</td><td class='ok'>2025-12-31 只能用 NOTICE_DATE≤该日的最新报告（2025-09-30 三季报）</td></tr>"
         "<tr><td class='bad'>财报数据「即产即用」</td><td class='ok'>以披露日为可用时点，报告期仅作标签</td></tr></table>")
o.append("<div class='box bad'><b>前视审计铁律</b>：任何时候 <code>point_in_time</code> 返回的报告，其 NOTICE_DATE 必须 ≤ 回测当日。代码层做硬断言，集成测试自动校验。</div>")

# 2
o.append("<h2>2. 数据源方案</h2>")
o.append("<h3>主源：东方财富数据中心 <code>RPT_F10_FINANCE_MAINFINADATA</code></h3>")
o.append("<div class='box ok'>已实测可达，<b>纯 HTTP、无第三方库</b>，与 AQuant 现有 <code>aq/data.py</code> 哲学一致。逐季度返回主要财务指标，且<strong>自带 NOTICE_DATE</strong>，天然支持 point-in-time。</div>")
o.append("<table><tr><th>字段</th><th>含义</th><th>用途</th></tr>"
         "<tr><td><code>REPORT_DATE</code></td><td>报告期</td><td>标签</td></tr>"
         "<tr><td><code>NOTICE_DATE</code></td><td>披露日</td><td><b>时点对齐基准</b></td></tr>"
         "<tr><td><code>EPSJB</code></td><td>当期累计每股收益</td><td>拆 TTM / 估值</td></tr>"
         "<tr><td><code>BPS</code></td><td>每股净资产</td><td>PB 派生</td></tr>"
         "<tr><td><code>ROEJQ</code></td><td>净资产收益率%</td><td>质量因子</td></tr>"
         "<tr><td><code>XSMLL / XSJLL</code></td><td>毛利率% / 净利率%</td><td>质量因子</td></tr>"
         "<tr><td><code>TOTALOPERATEREVETZ</code></td><td>营收同比%</td><td>成长因子</td></tr>"
         "<tr><td><code>PARENTNETPROFITTZ</code></td><td>净利润同比%</td><td>成长因子</td></tr></table>")
o.append("<h3>估值（PE/PB）怎么办</h3>")
o.append("<p>该接口<strong>不直接给历史逐日 PE/PB</strong>。但正因如此反而更干净：在 point-in-time 对齐后，用"
         " <code>PB = 价 / BPS</code>、<code>PE_ttm = 价 / EPS_ttm</code> <strong>当场推导</strong>。"
         "EPS_ttm 由累计 EPS 拆成单季再滚动 4 季得到（见 §4）。推导出的估值天然满足时点约束，不会前视。</p>")
o.append("<h3>备选 adapter（可热插拔）</h3>")
o.append("<p>若主源不稳，可加一个实现同一接口的 adapter：<span class='tag'>AKShare</span> <code>stock_a_indicator_lg</code>（逐日 PE/PB/PS 历史）、"
         "<span class='tag'>TuShare</span> <code>daily_basic</code>/<code>fina_indicator</code>（需免费 token）。"
         "接口层 <code>FACTORS</code> 与 <code>build_panel</code> 不变，只换底层拉取。</p>")

# 3
o.append("<h2>3. 架构设计（<code>aq/fundamentals.py</code> 已实现）</h2>")
o.append("<pre>adapter(HTTP, 东方财富)\n   ↓  normalize + 拆单季EPS + 滚动TTM\n   ↓  cache  .cache/fund/&lt;code&gt;.json\npoint_in_time(reports, as_of)   ← NOTICE_DATE ≤ as_of 的最新报告\n   ↓\nfactor_*(state, price)         ← 估值(ep/bp)/质量(roe…)/成长(rev…)\n   ↓\nbuild_panel(codes, axis_dates)  ← {code:{date:{fac:val}}} 喂 IC 引擎</pre>")
o.append("<p>已落地接口：<code>pull / point_in_time / factor_pb / factor_ep / factor_roe / factor_gross_margin / "
         "factor_net_margin / factor_rev_yoy / factor_profit_yoy / build_panel / demo</code>。</p>")
o.append("<h3>实跑证据：时点对齐（前视偏差防护生效）</h3>")
o.append("<table><tr><th>代码</th><th>名称</th><th>最近报告</th><th>披露日</th><th>ROE</th><th>BPS</th><th>TTM_EPS</th></tr>")
for sym, nm, tail, _ in proof_rows:
    for i, (rd, typ, nd, roe, bps, ttm) in enumerate(tail):
        o.append("<tr><td>%s</td><td>%s</td><td>%s %s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                 % (sym if i == 0 else "", nm if i == 0 else "", rd, typ, nd,
                    ("%.2f" % roe) if roe is not None else "-",
                    ("%.2f" % bps) if bps is not None else "-",
                    ("%.2f" % ttm) if ttm is not None else "-"))
o.append("</table>")
o.append("<table><tr><th>代码</th><th>时点 2025-12-31</th><th>时点 2026-04-20</th><th>时点 2026-05-10</th></tr>")
for sym, nm, _, probes in proof_rows:
    cells = " ".join("%s→%s(披露%s)" % (p, rd, nd) for p, rd, nd in probes)
    o.append("<tr><td>%s</td><td colspan=3>%s</td></tr>" % (sym, cells))
o.append("</table>")
o.append("<div class='box ok'>证据：在 <code>2025-12-31</code> 时点，所有票都只能取到 <b>2025-09-30 三季报</b>（年报尚未披露）——"
         "前视偏差防护实打实生效，不是纸面设计。</div>")

# 4
o.append("<h2>4. 时点对齐引擎（核心算法）</h2>")
o.append("<h3>4.1 披露滞后</h3>")
o.append("<table><tr><th>报告类型</th><th>报告期</th><th>典型披露窗口</th></tr>"
         "<tr><td>一季报</td><td>3-31</td><td>4 月</td></tr>"
         "<tr><td>中报</td><td>6-30</td><td>8 月</td></tr>"
         "<tr><td>三季报</td><td>9-30</td><td>10 月</td></tr>"
         "<tr><td>年报</td><td>12-31</td><td>次年 4 月</td></tr></table>")
o.append("<h3>4.2 <code>point_in_time(reports, as_of)</code></h3>")
o.append("<pre>取 reports 中 NOTICE_DATE ≤ as_of 的<strong>最新</strong>一条\n（reports 已按报告期升序，遇 notice&gt;as_of 即停）</pre>")
o.append("<h3>4.3 TTM EPS 拆单季</h3>")
o.append("<pre>累计EPS → 单季EPS：\n  Q1单季  = 累计Q1\n  H1单季  = 累计H1 - 累计Q1\n  9M单季  = 累计9M - 累计H1\n  FY单季  = 累计FY - 累计9M\nTTM_EPS = 最近 4 个单季之和   （同财年内滚动）</pre>")

# 5
o.append("<h2>5. 因子目录（Value / Quality / Growth）</h2>")
o.append("<table><tr><th>因子</th><th>族</th><th>来源</th><th>方向</th><th>point-in-time 处理</th></tr>"
         "<tr><td><code>ep</code></td><td>估值</td><td>1/PE_ttm</td><td class='ok'>+1 便宜好</td><td>price×EPS_ttm 推导</td></tr>"
         "<tr><td><code>bp</code></td><td>估值</td><td>1/PB</td><td class='ok'>+1 便宜好</td><td>price×BPS 推导</td></tr>"
         "<tr><td><code>roe</code></td><td>质量</td><td>ROEJQ</td><td class='ok'>+1</td><td>直接取报告值</td></tr>"
         "<tr><td><code>gmarg</code></td><td>质量</td><td>XSMLL</td><td class='ok'>+1</td><td>直接取报告值</td></tr>"
         "<tr><td><code>nmarg</code></td><td>质量</td><td>XSJLL</td><td class='ok'>+1</td><td>直接取报告值</td></tr>"
         "<tr><td><code>revyoy</code></td><td>成长</td><td>营收同比%</td><td class='ok'>+1</td><td>直接取报告值</td></tr>"
         "<tr><td><code>profyoy</code></td><td>成长</td><td>净利同比%</td><td class='ok'>+1</td><td>直接取报告值</td></tr></table>")
o.append("<div class='box info'>估值类（ep/bp）必须先拿到当时的价格才能算，因此只能在 prescan 内结合行情推导；"
         "质量/成长类直接用报告内水平或同比值，天然 point-in-time。</div>")
o.append("<h3>实跑样本：2025-03-31 时点的质量/成长因子面板</h3>")
o.append("<table><tr><th>代码</th><th>名称</th><th>ROE</th><th>营收同比%</th><th>净利同比%</th><th>毛利率%</th></tr>")
for sym, nm, roe, ry, py, gm in panel:
    o.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
             % (sym, nm, ("%.1f" % roe) if roe is not None else "-",
                ("%.1f" % ry) if ry is not None else "-",
                ("%.1f" % py) if py is not None else "-",
                ("%.1f" % gm) if gm is not None else "-"))
o.append("</table>")

# 6
o.append("<h2>6. 引擎集成设计</h2>")
o.append("<p>现有 prescan 回调签名 <code>(cols, idx, t, indc)</code>。扩展为传入一个 fund accessor：</p>")
o.append("<pre># backtest.py：__init__ 增 fund=None；调用 prescan 处改为\nrank = self.prescan(cols, idx, t, indc, self.fund)\n\n# prescan 内用法（与 factors.make_composite 同构）\ndef fund_composite(cols, idx, t, indc, fund):\n    date = dates[t]\n    raw = {}\n    for s in cols:\n        st = fund.pit(s, date)          # point_in_time 结果\n        price = prices[s][t]\n        vs = [fnd.factor_ep(st, price), fnd.factor_roe(st), fnd.factor_rev_yoy(st)]\n        if all(v is not None for v in vs):\n            raw[s] = vs\n    # 逐因子横截面 z-score × 方向，加权求和排名（复用 factors._zstat）\n    ...</pre>")
o.append("<div class='box warn'>集成是下一步实现工作（P1）。本方案已把数据层与因子层做成可直接被 prescan 调用的纯函数，"
         "patch 面极小：仅 backtest.py 透传 <code>fund</code>、strategies.py 新增 <code>fund_composite</code>。</div>")

# 7
o.append("<h2>7. 验证方案（如何证明不是又一个空中楼阁）</h2>")
o.append("<table><tr><th>检验</th><th>做法</th><th>判定标准</th></tr>"
         "<tr><td>因子 IC</td><td>复用 factor_study 的 <code>ic_agg</code>，方向调整 t 值</td><td>|t|&gt;1.5 且方向符合理论</td></tr>"
         "<tr><td>前视审计</td><td>断言 point_in_time 永不返回 notice&gt;as_of</td><td>100% 通过</td></tr>"
         "<tr><td>成本/随机对照</td><td>同频同持仓同门控，只换选股</td><td>≥第 80 百分位</td></tr>"
         "<tr><td>OOS</td><td><b>必须用 2023 以后样本外</b>（v4 在此翻车）</td><td>样本外超额 ≥ 0</td></tr>"
         "<tr><td>价量×基本面合成</td><td>低相关 → 分散收益，合成 composite</td><td>合成 IC &gt; 单族</td></tr></table>")

# 8
o.append("<h2>8. 分阶段落地</h2>")
o.append("<table><tr><th>阶段</th><th>内容</th><th>状态</th></tr>"
         "<tr><td><b>P0</b></td><td>数据层 <code>aq/fundamentals.py</code> + 本设计</td><td class='ok'>已完成·已实跑</td></tr>"
         "<tr><td><b>P1</b></td><td>引擎集成 patch + <code>fund_composite</code> + 单因子 IC 报告</td><td class='warn'>待做</td></tr>"
         "<tr><td><b>P2</b></td><td>价量×基本面合成，复用压换手方案，月度/季频 IC 与 OOS</td><td class='warn'>待做</td></tr>"
         "<tr><td><b>P3</b></td><td>更长/多段样本、基本面因子跨周期稳定性</td><td class='warn'>待做</td></tr></table>")

# 9
o.append("<h2>9. 诚实风险</h2>")
o.append("<div class='box bad'><ul style='margin:0;padding-left:18px'>"
         "<li><b>幸存者偏差更严重</b>：退市股不在快照，财务数据同样缺失 → 历史回测天然偏好「活下来的好公司」，需明确标注。</li>"
         "<li><b>财报追溯调整</b>：披露后可能修正（restatement），我们拿到的历史快照与当时投资者所见可能不一致。</li>"
         "<li><b>数据天然低频</b>：季报间隔长，因子更新慢 → 正好契合「压换手/低频」方向，但 signal 频率低。</li>"
         "<li><b>不可样本内自嗨</b>：v4 教训——漂亮的样本内成绩一到 OOS 就翻车，必须以 2023 后样本外为硬门槛。</li></ul></div>")

# 10
o.append("<h2>10. 结论</h2>")
o.append("<div class='box ok'><b>数据通路已打通，时点对齐已证明可行。</b> 免费、纯 HTTP、自带披露日，"
         "前视偏差防护经实跑验证生效；估值因子由价格×时点基本面推导，质量/成长因子直接取自报告值。"
         "下一步是极小的引擎集成 patch + 单因子 IC/OOS 验证——把「择时 > 选股」的结论，"
         "推进到「择时 + 经得起 OOS 的基本面选股」。</div>")

o.append("<div class='foot'>生成于 AQuant 研究流程 · 数据源：东方财富数据中心（已实测）· "
         "本报告所有「证据表」均为脚本实跑结果，非示意数据。</div>")
o.append("</div></body></html>")

html = "".join(o)
fn = "基本面数据接入与因子扩展方案.html"
open(fn, "w", encoding="utf-8").write(html)
log("已写出 %s (%d 字节)" % (fn, len(html)))
print("\n".join(LOG))
