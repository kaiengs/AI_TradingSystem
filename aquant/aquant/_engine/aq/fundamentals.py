#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本面数据接入层（v4 之后：补基本面因子，打破价量因子的风格天花板）
=======================================================================

设计目标
--------
v4 结论：价量因子（rev20/vol60）在样本内漂亮，但样本外翻车，本质是
「择时 > 选股」——价量因子受市场风格周期支配，没有跨周期的护城河。
基本面因子（估值/质量/成长）是另一条独立的 alpha 来源，且与价量低相关，
理论上能补上这块短板。

本模块只解决一件事：**在回测里拿到「当时真实可得」的基本面数据**，
也就是 point-in-time（时点）对齐，杜绝前视偏差。

为什么前视偏差是头号陷阱
------------------------
财报有「报告期」和「披露日」两个日期：
  * 报告期 2025-12-31（年报）通常要到 2026-04-30 前后才披露；
  * 一季报（3-31）通常 4 月底披露，三季报（9-30）通常 10 月底披露。
如果在 2025-12-31 这一天就用 2025 全年财报做选股，等于提前 4 个月
看到了未来——回测必然虚高。正确做法：某交易日 t 只能用
NOTICE_DATE <= t 的报告。东方财富 MAINFINADATA 恰好返回 NOTICE_DATE，
本模块据此做时点对齐。

数据源（已实测可达，纯 HTTP，无第三方库）
----------------------------------------
主源：东方财富数据中心 RPT_F10_FINANCE_MAINFINADATA
  * 逐季度主要财务指标，含 REPORT_DATE（报告期）、NOTICE_DATE（披露日）、
    EPSJB（每股收益）、BPS（每股净资产）、ROEJQ（ROE）、
    XSMLL（毛利率%）、XSJLL（净利率%）、TOTALOPERATEREVETZ（营收同比%）、
    PARENTNETPROFITTZ（净利润同比%）等。
  * 自带披露日，天然支持 point-in-time。
  * 估值（PE/PB）不直接提供历史序列，但可由 price × BPS/EPS_ttm 在
    时点对齐后**推导**，反而比拿现成估值序列更干净、更不易前视。
备选（文档中说明，本模块不强制依赖）：
  * AKShare `stock_a_indicator_lg` 直接给逐日 PE/PB/PS 历史；
  * TuShare `daily_basic` / `fina_indicator`（需免费 token）。
  若主源不稳，可加一个 adapter 类实现同一接口即可热插拔。

对外接口
--------
  pull(code, refresh=False)        -> [report_dict]（已归一化、按报告期升序、带 ttm_eps）
  point_in_time(reports, as_of)    -> 截至 as_of 真实可得的最新 report（dict 或 None）
  factor_pb(state, price)          -> 时点 PB 派生因子原始值（或 None）
  factor_ep(state, price)          -> 时点 EP=1/PE_ttm（或 None）
  factor_roe / factor_gross_margin / factor_rev_yoy / factor_profit_yoy
  build_panel(codes, axis_dates)   -> {code: {date: {fac: val}}} 喂给 IC 引擎的横截面面板
  demo()                           自测：拉茅台 + 几个池内票，打印时点对齐证据

数据缓存：`.cache/fund/<6位code>.json`
"""

import os
import json
import time

from . import data as _data

FUND_DIR = os.path.join(_data.CACHE, "fund")
os.makedirs(FUND_DIR, exist_ok=True)

API = ("https://datacenter-web.eastmoney.com/api/data/v1/get?"
       "reportName=RPT_F10_FINANCE_MAINFINADATA&columns=ALL"
       "&filter=(SECURITY_CODE%3D%22{code}%22)"
       "&pageSize=100&pageNum={page}&sortColumns=REPORT_DATE&sortTypes=-1")

# 报告类型 ->  Fiscal quarter 序号（1=Q1,2=H1,3=Q3,4=FY），用于 TTM 拆单季
_TYPE_Q = {"一季报": 1, "中报": 2, "三季报": 3, "年报": 4}


def _sec_code(symbol):
    """sh600519 -> 600519 ; sz000594 -> 000594 ; 北交所跳过"""
    s = symbol.replace("sh", "").replace("sz", "").replace("bj", "")
    return s


def _norm(r):
    """把东方财富一条记录归一化为内部结构，并补算单季 EPS / TTM EPS。"""
    def f(k, d=None):
        v = r.get(k)
        if v in (None, "", "null"):
            return d
        try:
            return float(v)
        except Exception:
            return d
    rep = (r.get("REPORT_DATE") or "")[:10]
    notice = (r.get("NOTICE_DATE") or "")[:10]
    typ = r.get("REPORT_TYPE") or ""
    return {
        "code": r.get("SECURITY_CODE"),
        "name": r.get("SECURITY_NAME_ABBR"),
        "report_date": rep,
        "notice_date": notice,
        "type": typ,
        "q": _TYPE_Q.get(typ, 0),
        "eps": f("EPSJB"),          # 当期累计基本每股收益
        "bps": f("BPS"),            # 每股净资产（时点快照，用于 PB）
        "roe": f("ROEJQ"),          # 净资产收益率(%)
        "gross_margin": f("XSMLL"), # 销售毛利率(%)
        "net_margin": f("XSJLL"),   # 销售净利率(%)
        "rev_yoy": f("TOTALOPERATEREVETZ"),     # 营收同比(%)
        "profit_yoy": f("PARENTNETPROFITTZ"),   # 净利润同比(%)
        "ocfps": f("MGJYXJJE"),     # 每股经营现金流
        "single_eps": None,         # 单季 EPS（下面填）
        "ttm_eps": None,            # TTM EPS（下面填）
    }


def _attach_ttm(reports):
    """按报告期升序，把累计 EPS 拆成单季 EPS，再滚动 4 季得 TTM EPS。"""
    last_cum_year = {}   # year -> 上一期累计 EPS（同财年内）
    single_list = []     # 按序累计的单季 EPS
    for rp in reports:
        yr = rp["report_date"][:4]
        cum = rp["eps"]
        if cum is None:
            rp["single_eps"] = None
        else:
            prev = last_cum_year.get(yr)
            if prev is None:
                rp["single_eps"] = cum          # 一季报：单季=累计
            else:
                rp["single_eps"] = cum - prev   # 其余：累计-上期累计
            last_cum_year[yr] = cum
            single_list.append(rp["single_eps"])
        # TTM = 最近 4 个单季之和
        if len(single_list) >= 4 and all(x is not None for x in single_list[-4:]):
            rp["ttm_eps"] = sum(single_list[-4:])
        else:
            rp["ttm_eps"] = None
    return reports


def pull(code, refresh=False):
    """拉取并缓存某只票的全部主要财务指标（按报告期升序）。"""
    sc = _sec_code(code)
    fn = os.path.join(FUND_DIR, f"{sc}.json")
    if not refresh and os.path.exists(fn):
        try:
            return json.load(open(fn))
        except Exception:
            pass
    out = []
    total_pages = 1
    for page in range(1, 40):          # 降序翻页：先取最新，再往回翻，确保近期数据不丢
        try:
            raw = _data._get(API.format(code=sc, page=page), enc="utf-8", timeout=30)
            j = json.loads(raw)
        except Exception:
            break
        res = j.get("result") or {}
        arr = res.get("data") or []
        total_pages = res.get("pages") or total_pages
        if not arr:
            break
        out.extend(_norm(r) for r in arr)
        if page >= total_pages:
            break
        time.sleep(0.05)
    # 升序、去重、补 TTM
    seen = set()
    uniq = []
    for r in sorted(out, key=lambda x: x["report_date"]):
        k = (r["report_date"], r["type"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    _attach_ttm(uniq)
    json.dump(uniq, open(fn, "w"), ensure_ascii=False)
    return uniq


def point_in_time(reports, as_of):
    """返回截至 as_of（'YYYY-MM-DD'）真实可得的最新报告；无则返回 None。
    关键：只用 NOTICE_DATE <= as_of 的报告，杜绝前视。"""
    a = as_of[:10]
    best = None
    for r in reports:
        if r["notice_date"] and r["notice_date"] <= a:
            best = r
        # reports 已升序，遇到 notice > a 之后不再有更早的可披露报告
    return best


# ------------------------------------------------------------
# 因子原始值计算（全部 point-in-time，输入 state=point_in_time 结果）
# 返回 float 或 None（数据不足/无效）
# ------------------------------------------------------------
def factor_pb(state, price):
    """PB = 价 / 每股净资产（时点）。返回 BP=1/PB 越大越便宜（估值因子原始用 BP）。"""
    if not state or price <= 0:
        return None
    bps = state.get("bps")
    if not bps or bps <= 0:
        return None
    pb = price / bps
    if pb <= 0:
        return None
    return 1.0 / pb          # BP：账面市值比，越大越便宜


def factor_ep(state, price):
    """EP = 1 / PE_ttm（时点）。盈利为负则 None。"""
    if not state or price <= 0:
        return None
    ttm = state.get("ttm_eps")
    if not ttm or ttm <= 0:
        return None
    pe = price / ttm
    if pe <= 0:
        return None
    return 1.0 / pe          # 盈利收益率，越大越便宜


def factor_roe(state, price=None):
    if not state:
        return None
    v = state.get("roe")
    return v if (v is not None and v > -100) else None


def factor_gross_margin(state, price=None):
    if not state:
        return None
    v = state.get("gross_margin")
    return v if v is not None else None


def factor_net_margin(state, price=None):
    if not state:
        return None
    v = state.get("net_margin")
    return v if v is not None else None


def factor_rev_yoy(state, price=None):
    if not state:
        return None
    v = state.get("rev_yoy")
    return v if v is not None else None


def factor_profit_yoy(state, price=None):
    if not state:
        return None
    v = state.get("profit_yoy")
    return v if v is not None else None


# 因子目录：名称 -> (计算函数, 理论方向)。方向 +1=越大越好, -1=越小越好
FACTORS = {
    "ep":   (factor_ep, +1.0),          # 估值：盈利收益率（便宜）
    "bp":   (factor_pb, +1.0),          # 估值：账面市值比（便宜）
    "roe":  (factor_roe, +1.0),         # 质量：净资产收益率
    "gmarg":(factor_gross_margin, +1.0),# 质量：毛利率
    "nmarg":(factor_net_margin, +1.0),  # 质量：净利率
    "revyoy":(factor_rev_yoy, +1.0),    # 成长：营收同比
    "profyoy":(factor_profit_yoy, +1.0),# 成长：净利同比
}


def build_panel(codes, axis_dates, prices=None):
    """构造横截面因子面板，喂给 IC 引擎。

    codes       : ['sh600519', ...]
    axis_dates  : ['2023-01-03', ...] 与回测交易日轴对齐
    prices      : {code: [price_per_axis_date]} 可选；不传则估值类因子(ep/bp)返回 None
    返回 {code: {date: {fac: val}}}，缺值处直接不出现该键（IC 引擎自行跳过）
    """
    panel = {}
    for code in codes:
        reps = pull(code)
        prc = (prices or {}).get(code)
        row = {}
        for d in axis_dates:
            st = point_in_time(reps, d)
            facs = {}
            for name, (fn, _sign) in FACTORS.items():
                p = prc[axis_dates.index(d)] if (prc and name in ("ep", "bp")) else None
                try:
                    v = fn(st, p) if name in ("ep", "bp") else fn(st)
                except Exception:
                    v = None
                if v is not None:
                    facs[name] = v
            if facs:
                row[d] = facs
        panel[code] = row
    return panel


def demo():
    print("=" * 70)
    print("fundamentals 自测：时点对齐证据")
    print("=" * 70)
    for sym in ["sh600519", "sz000858", "sh601318"]:
        reps = pull(sym)
        if not reps:
            print(f"  {sym}: 无数据")
            continue
        nm = reps[-1].get("name")
        print(f"\n  {sym} {nm}：共 {len(reps)} 条报告")
        # 展示最近 4 条的报告期/披露日/ROE/BPS
        for r in reps[-4:]:
            print(f"    报告期 {r['report_date']} {r['type']:>3} | 披露 {r['notice_date']}"
                  f" | ROE {r['roe']} | BPS {r['bps']} | TTM_EPS {r['ttm_eps']}")
        # 时点对齐：在 2025-12-31 这天，真实可得的最新报告应是一季报/更早（年报未披露）
        for probe in ["2025-12-31", "2026-04-20", "2026-05-10"]:
            st = point_in_time(reps, probe)
            rd = st["report_date"] if st else "无"
            nd = st["notice_date"] if st else "-"
            print(f"    时点 {probe} -> 可用报告期 {rd}（披露 {nd}）")


if __name__ == "__main__":
    demo()


class Fund:
    """point-in-time 基本面 accessor：喂给引擎 prescan 的第 5 个参数。

    用法：fund.pit(code, date) -> 截至 date 真实可得的最新报告（dict）或 None。
    code 接受 'sh600519' 这类带前缀符号，内部自动剥离。
    报告按 code 缓存，避免重复拉取/解析。
    """

    def __init__(self):
        self._cache = {}

    def pit(self, code, date):
        reps = self._cache.get(code)
        if reps is None:
            reps = pull(code)
            self._cache[code] = reps
        return point_in_time(reps, date)
