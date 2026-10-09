#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据层：行情 / K线 / 全市场截面，带本地缓存"""

import json, os, time, urllib.request

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
os.makedirs(CACHE, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/kline/kline?param={},day,,,{}"
RT_URL = "https://qt.gtimg.cn/q="
SINA_URL = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
            "Market_Center.getHQNodeData?page={}&num={}&sort=symbol&asc=1&node=hs_a&symbol=&_s_r_a=page")


def _get(url, enc="utf-8", retries=3, timeout=20):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=timeout).read().decode(enc, "ignore")
        except Exception as e:
            last = e
            time.sleep(0.5 * (i + 1))
    raise last


MAX_FETCH = 2000  # 缓存始终按最大长度抓取，避免小请求覆盖大缓存导致标的池不一致
                  # 实测腾讯上限约 2000 根（≈2018-07 至今，8.2 年），再大返回结构异常


def kline(code, bars=600, refresh=False):
    """日线，返回 [(date, open, close, high, low, volume)]，日期升序"""
    fn = os.path.join(CACHE, f"{code}.json")
    data = None
    if not refresh and os.path.exists(fn):
        try:
            data = json.load(open(fn))
        except Exception:
            data = None
    if not data:
        raw = json.loads(_get(KLINE_URL.format(code, MAX_FETCH)))["data"][code]
        arr = raw.get("day") or raw.get("qfqday") or []
        data = [[r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in arr]
        json.dump(data, open(fn, "w"))
    return [tuple(r) for r in (data[-bars:] if bars else data)]


def realtime(codes):
    """批量实时快照 -> {code: {...}}"""
    out = {}
    for i in range(0, len(codes), 60):
        chunk = codes[i:i + 60]
        raw = _get(RT_URL + ",".join(chunk), enc="gbk")
        for line in raw.strip().split(";"):
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            code = key.split("_")[-1].strip()
            f = val.strip().strip('"').split("~")
            if len(f) < 45:
                continue
            def g(i, d=0.0):
                try:
                    return float(f[i])
                except Exception:
                    return d
            out[code] = {"code": code, "name": f[1], "price": g(3), "prev": g(4), "open": g(5),
                         "pct": g(32), "high": g(33), "low": g(34), "volume": g(36),
                         "amount": g(37), "turnover": g(38), "pe": g(39), "pb": g(46),
                         "nmc": g(44) * 1e4, "mcap": g(45) * 1e4}
    return out


def market_snapshot(pages=140, num=40, refresh=False):
    """全A股截面快照（沪深京）"""
    fn = os.path.join(CACHE, "allA.json")
    if not refresh and os.path.exists(fn):
        try:
            return json.load(open(fn))
        except Exception:
            pass
    rows = []
    for p in range(1, pages + 1):
        raw = _get(SINA_URL.format(p, num), enc="gbk")
        if not raw or len(raw) < 10:
            break
        try:
            arr = json.loads(raw)
        except Exception:
            break
        if not arr:
            break
        rows.extend(arr)
        if len(arr) < num:
            break
    out = []
    for r in rows:
        def g(k, d=0.0):
            try:
                return float(r.get(k))
            except Exception:
                return d
        if g("trade") <= 0:
            continue
        out.append({"symbol": r.get("symbol"), "code": r.get("code"), "name": r.get("name"),
                    "trade": g("trade"), "settlement": g("settlement"), "changepercent": g("changepercent"),
                    "amount": g("amount"), "nmc": g("nmc"), "mcap": g("mcap"),
                    "turnoverratio": g("turnoverratio"), "pe": g("pe"), "pb": g("pb")})
    json.dump(out, open(fn, "w"))
    return out


def build_pool(top=400, mode="random", min_amount=2e7, min_bars=2000, min_price=2.0,
               exclude_bj=True, seed=20261008, refresh=False, log=print):
    """
    构建大样本标的池（解决"样本量不足"这一根本问题）。

    mode = "random"（默认，推荐）
        在满足硬约束的股票里**随机抽样**，不做成交额排名。
        实测：按"当前成交额排名"选池存在严重前视偏差 —— 用今天的活跃度
        挑 8 年前的票，等于只留下"后来涨得好所以成交活跃"的那批，
        等权买入持有基准被虚高约 9.6 倍（+592% vs +61%），
        会让"所有策略都跑输基准"变成一个假结论。

    mode = "liquid"
        按当前成交额降序取前 N。仅用于对照实验，**不可用于正式回测**。

    共同硬约束（均无前视性）：
      1) 剔除 ST / 退市 / 北交所（涨跌幅规则不同）
      2) 股价 >= min_price（低价股受整手约束扭曲严重）
      3) 成交额 >= min_amount（仅剔除僵尸股/停牌，很宽松）
      4) K线长度 >= min_bars：回测取的是所有标的的**日期交集**，
         混入次新股会把整个样本窗口拖短，必须卡死
    """
    fn = os.path.join(CACHE, f"pool_{mode}_{top}_{int(min_amount)}_{min_bars}.json")
    if not refresh and os.path.exists(fn):
        try:
            return json.load(open(fn))
        except Exception:
            pass

    snap = market_snapshot(refresh=refresh)
    cands = []
    for r in snap:
        sym = r.get("symbol") or ""
        name = r.get("name") or ""
        if "ST" in name or "退" in name:
            continue
        if exclude_bj and sym.startswith("bj"):
            continue
        if r.get("trade", 0) < min_price:
            continue
        if r.get("amount", 0) < min_amount:
            continue
        cands.append(r)

    if mode == "liquid":
        cands.sort(key=lambda r: -r.get("amount", 0))
        cands = cands[:int(top * 2.2)]
    else:
        import random as _rd
        _rd.Random(seed).shuffle(cands)
        cands = cands[:int(top * 2.8)]      # 约六成满足 K线长度，多留余量

    out, tried = [], 0
    for r in cands:
        if len(out) >= top:
            break
        tried += 1
        try:
            k = kline(r["symbol"], bars=0)
        except Exception:
            continue
        if len(k) >= min_bars:
            out.append(r["symbol"])
        if log and tried % 100 == 0:
            log(f"  池构建中 {tried}/{len(cands)}，已入选 {len(out)}")
    json.dump(out, open(fn, "w"))
    return out


PRESETS = {
    "core20": ["sh600519", "sh601318", "sh600036", "sh600276", "sh600030", "sh601899", "sh600887",
               "sz000858", "sz000333", "sz000651", "sh600989", "sz300433", "sh601012", "sh600900",
               "sz002594", "sh603259", "sh600309", "sz000001", "sh601166", "sh600585"],
    "tech10": ["sz300433", "sh688185", "sz002230", "sh688981", "sz300750", "sz002594",
               "sh603501", "sz300308", "sh688036", "sz002371"],
    "pharma10": ["sh600276", "sh600789", "sh600812", "sz300760", "sh600380", "sz002422",
                 "sz300122", "sh688185", "sz000513", "sh600420"],
    "idx": ["sh000300", "sh000905", "sh000852"],
}
