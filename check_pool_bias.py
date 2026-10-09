#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：按"当前成交额"选池是否存在前视/幸存者偏差"""
import sys, time, random
sys.path.insert(0, "/workspace/quant")
from aq import data, strategies as st, backtest as bt

snap = data.market_snapshot()
print("全市场快照:", len(snap))

base = [r for r in snap
        if "ST" not in (r.get("name") or "") and "退" not in (r.get("name") or "")
        and not (r.get("symbol") or "").startswith("bj")
        and r.get("trade", 0) >= 2.0]
print("剔除ST/北交所/低价股后:", len(base))

# ---- 1. 流动性前 60（v2 原口径，有前视嫌疑）----
liq = sorted(base, key=lambda r: -r.get("amount", 0))[:60]
liq_syms = [r["symbol"] for r in liq]

# ---- 2. 随机抽样（无成交额筛选，作为无偏对照）----
rnd = random.Random(20261008)
sample = rnd.sample(base, 420)
syms_rnd = [r["symbol"] for r in sample]

t0 = time.time()
def load(syms):
    out = {}
    for s in syms:
        try:
            k = data.kline(s, bars=0)
        except Exception:
            continue
        if len(k) >= 2000:
            out[s] = k
    return out

d_liq = load(liq_syms)
d_rnd = load(syms_rnd)
print("K线>=2000根: 流动性池 %d/%d, 随机池 %d/%d  [%.1fs]"
      % (len(d_liq), len(liq_syms), len(d_rnd), len(syms_rnd), time.time() - t0))

d_rnd = dict(list(d_rnd.items())[:300])

for name, d in [("流动性前%d" % len(d_liq), d_liq), ("随机%d" % len(d_rnd), d_rnd)]:
    b = bt.Backtest(d, st.buy_hold(), cash=1e8)
    r = b.run()
    eq = r["equity"]
    # 逐年收益
    yr = {}
    for d_, v in zip(r["dates"], eq):
        yr.setdefault(d_[:4], v)
    years = sorted(yr)
    seq = []
    prev = None
    for y in years:
        v = yr[y]
        seq.append("%s:%+.0f%%" % (y, (v / prev - 1) * 100 if prev else 0))
        prev = v
    print("\n== %s ==" % name)
    print("  等权买入持有 %.2f%%  年化 %.2f%%  回撤 %.2f%%" %
          (r["total_return"] * 100, r["annual"] * 100, r["max_drawdown"] * 100))
    print("  分年:", " ".join(seq))
print("\n总耗时 %.1fs" % (time.time() - t0))
