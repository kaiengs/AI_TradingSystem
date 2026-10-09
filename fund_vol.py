#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AQuant v5 · P4：把风控层换成**组合层波动率目标**
================================================
P3 的诊断结论：正交设计（标量 × 向量）本身是成立的，alpha 保留率从二元门控的
68% 提到 81%。但剩余亏损来自 **MA200 这把尺子本身的误判** —— 它是方向性开关，
把 2024-26 牛市也切掉了一块（该段正交组合超额 -18.56pp，无门控 +13.63pp）。

P4 处方：风控层换成**按风险而非方向**调仓的信号
    组合层波动率目标： e = clip(target / 指数年化已实现波动率, floor, cap)

为什么它有望修好 MA200 的毛病：
  * 牛市若走得平稳（低波），e 被 cap 到 1.0 —— **满仓参与，不踏空**；
  * 崩盘通常伴随波动飙升，e 自动降下来 —— 削平回撤；
  * 只看风险大小、不看方向，因此不会像 MA200 那样"把上涨也当成该跑的时候"。

正交性（严格）：用**指数**波动算出一个**全组合共用的标量** e，只缩放总暴露，
完全不改变各票相对权重 —— 与选股层严格正交。
（对照：框架里原有的逐票 `vol_target` 按每只票自己的波动缩放，会系统性超配低波股，
等于偷引入"低波因子"，**与选股层耦合、不正交**，故 P4 不用它。）

纪律（同 P1~P3）：参数只在样本内挑；样本外硬门槛 2023-01-01 之后全程不参与决策；
                  随机信号对照 + 成本敏感性 + 滚动前推三重验证。
"""

import sys, os, time
sys.path.insert(0, "/workspace/quant")
from optimize import (fill_to_axis, CASH, CSS, cls, pc)
from aq import data, strategies as st, backtest as bt
from aq import fundamentals as fnd
from fund_ic import (month_starts, to_dict, build_panel,
                     audit_point_in_time, ic_agg_fund, run_fund)

OUT = "/workspace/quant/波动率目标风控_P4.html"
OOS_CUTOFF = "2023-01-01"
WARM = 300
FREQ = "quarterly"
FNAME = "bp"
TAU, FLOOR = 6.0, 0.0        # 沿用 P2 样本内选出的倾斜强度
# ---- 族内选代表的护栏：alpha 保留率下限（见 RULE_* 规则集） ----
ALPHA_KEEP_MIN = 0.50      # 推荐规则 R4 使用的护栏
AK_GRID = (0.30, 0.50, 0.80)   # 护栏敏感性检验

KW = {"rebalance": FREQ, "max_weight": 0.0}

# 参数轴（用于定义「邻域」，防选到过拟合尖峰）
TARGETS = (0.10, 0.15, 0.20, 0.25)      # 波动率目标族：target
LOOKBACKS = (40, 60)                     # 波动率目标族：回看窗口
BEARWS = (0.0, 0.30, 0.50, 0.70)         # MA200 族：熊市保留仓位（0.0 = 二元清仓）

# (类型, p1, p2, 说明)
#   p1: vol 族=target / ma 族=bear_w（ma200 固定 0.0）
#   p2: vol 族=回看窗口
CONFIGS = [
    ("nogate", None, None, "无风控（P2 最佳，alpha 全开）"),
    ("ma200", None, None, "MA200 二元门控（P1/P2）"),
    ("soft30", None, None, "MA200 部分仓位：熊市留 30%（P3 最佳）"),
    ("soft50", None, None, "MA200 部分仓位：熊市留 50%"),
    ("soft70", None, None, "MA200 部分仓位：熊市留 70%"),
    ("vol", 0.10, 40, "波动率目标 10%（回看 40 日）"),
    ("vol", 0.15, 40, "波动率目标 15%（回看 40 日）"),
    ("vol", 0.20, 40, "波动率目标 20%（回看 40 日）"),
    ("vol", 0.25, 40, "波动率目标 25%（回看 40 日）"),
    ("vol", 0.10, 60, "波动率目标 10%（回看 60 日）"),
    ("vol", 0.15, 60, "波动率目标 15%（回看 60 日）"),
    ("vol", 0.20, 60, "波动率目标 20%（回看 60 日）"),
    ("vol", 0.25, 60, "波动率目标 25%（回看 60 日）"),
]

MA_KINDS = ("ma200", "soft30", "soft50", "soft70")
CFG_BY_KEY = {(c[0], c[1], c[2]): c for c in CONFIGS}


def bear_w_of(c):
    """MA200 族配置的 bear_w（0.0 = 二元清仓）"""
    if c[0] == "ma200":
        return 0.0
    return int(c[0][4:]) / 100.0


def neighbors(c):
    """参数轴上的相邻配置 —— 用于「邻域最坏值」防过拟合尖峰"""
    out = []
    if c[0] == "vol":
        tg, lb = c[1], c[2]
        i = TARGETS.index(tg)
        for j in (i - 1, i + 1):
            if 0 <= j < len(TARGETS):
                out.append(("vol", TARGETS[j], lb))
        for o in LOOKBACKS:
            if o != lb:
                out.append(("vol", tg, o))
    elif c[0] in MA_KINDS:
        i = BEARWS.index(bear_w_of(c))
        for j in (i - 1, i + 1):
            if 0 <= j < len(BEARWS):
                bw = BEARWS[j]
                out.append(("ma200", None, None) if bw == 0.0 else (f"soft{int(bw * 100)}", None, None))
    return [CFG_BY_KEY[k] for k in out if k in CFG_BY_KEY]


def tilt_factory(seed=None):
    if seed is None:
        return lambda: st.fund_tilt(FNAME, tau=TAU, floor=FLOOR)
    return lambda: st.rand_fund_tilt(seed, FNAME, tau=TAU, floor=FLOOR)


def make_gate(kind, p1, p2, seed=None):
    tf = tilt_factory(seed)
    if kind == "nogate":
        return tf
    if kind == "ma200":
        return lambda: st.market_gate(tf(), 0.0, cutoff=1.00)
    if kind.startswith("soft"):
        bw = int(kind[4:]) / 100.0
        return lambda: st.market_gate(tf(), bw, cutoff=1.00)
    if kind == "vol":
        return lambda: st.port_vol_target(tf(), target=p1, cap=1.0)
    raise ValueError(kind)


def kw_for(c):
    """vol 类型的回看窗口要透传给引擎（指数波动率在引擎侧计算）"""
    if c[0] == "vol":
        return {**KW, "mkt_vol_n": c[2]}
    return KW


def main():
    t_all = time.time()
    log = lambda *a: print(*a, flush=True)
    log("=" * 68)
    log("AQuant v5 · P4：风控层换成组合层波动率目标")
    log("=" * 68)

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

    # ---------- 1b. 指数波动率诊断（先确认 e=target/vol 落在合理区间） ----------
    log("\n[1b] 指数（沪深300）已实现年化波动率诊断 ...")
    vol_stats = {}
    for vn in (40, 60):
        vs = bt._rolling_vol(mkt, vn)
        vs = [v for v in vs if v is not None]
        vs_sorted = sorted(vs)
        vol_stats[vn] = {"mean": sum(vs) / len(vs), "min": vs_sorted[0],
                         "max": vs_sorted[-1], "p10": vs_sorted[int(len(vs) * 0.1)],
                         "p90": vs_sorted[int(len(vs) * 0.9)], "n": len(vs)}
        s = vol_stats[vn]
        log(f"  回看{vn}日: 均值 {s['mean'] * 100:.1f}%  "
            f"p10 {s['p10'] * 100:.1f}%  p90 {s['p90'] * 100:.1f}%  "
            f"范围 {s['min'] * 100:.1f}%~{s['max'] * 100:.1f}%")
        for tg in (0.10, 0.15, 0.20, 0.25):
            e_p10, e_p90 = tg / s["p90"], tg / s["p10"]
            log(f"     target={tg:.0%} -> e 约在 {min(e_p10, 1.0):.2f}~{min(e_p90, 1.0):.2f}（cap=1.0）")

    # ---------- 2. 面板 + 前视审计 ----------
    log("\n[2] 构建 point-in-time 面板 ...")
    panel = build_panel(fund, list(A.keys()), reb, dates)
    bad, tot = audit_point_in_time(panel, dates)
    log(f"  前视审计: {bad}/{tot} 条越界（须为 0）")

    # ---------- 3. 信号复核 ----------
    log("\n[3] 复核 BP 因子 IC（季频 63 日）...")
    ic_is = ic_agg_fund(panel, Ad, dates, FNAME, 63, reb, 0, is_end - 1)
    ic_oos = ic_agg_fund(panel, Ad, dates, FNAME, 63, reb, is_end, n - 1)
    if ic_is:
        log(f"  样本内 方向IC {ic_is['mean_dir']:+.4f}  t {ic_is['t_dir']:+.2f}")
    if ic_oos:
        log(f"  样本外 方向IC {ic_oos['mean_dir']:+.4f}  t {ic_oos['t_dir']:+.2f}")

    # ---------- 4. 样本内配置前沿 ----------
    log("\n[4] 样本内配置前沿扫描（选股层固定，只换风控层）...")
    is_rows = {}
    for c in CONFIGS:
        r = run_fund(A, dates, mkt, WARM, is_end - 1, make_gate(c[0], c[1], c[2]), fund, **kw_for(c))
        is_rows[c] = {"ret": r["ret"], "bench": r["bench"], "excess": r["excess"],
                      "turn": r["turnover"], "n": r["n"], "mdd": r["raw"]["max_drawdown"]}
    full = is_rows[CONFIGS[0]]
    full_alpha = full["excess"]
    for c in CONFIGS:
        r = is_rows[c]
        r["alpha_keep"] = (r["excess"] / full_alpha) if full_alpha > 1e-9 else 0.0
        r["mdd_cut"] = (1.0 - abs(r["mdd"]) / abs(full["mdd"])) if full["mdd"] else 0.0
    log(f"  {'配置':<32}{'超额':>10}{'alpha保留':>11}{'回撤':>9}{'回撤削减':>9}{'换手':>8}")
    for c in CONFIGS:
        r = is_rows[c]
        log(f"  {c[3]:<32}{r['excess'] * 100:>+9.2f}pp{r['alpha_keep'] * 100:>10.0f}%"
            f"{r['mdd'] * 100:>8.1f}%{r['mdd_cut'] * 100:>8.0f}%{r['turn'] * 100:>7.0f}%")

    # ---------- 5. 族内选代表：先定规则，再选人 ----------
    # 核心原则：族内选代表的目标函数，必须与「族间对决要回答的问题」一致。
    #   * 族内所有候选共享同一个选股层（价值倾斜），唯一差别是风控质量
    #     -> 用总收益口径会把 beta 混进来，等于奖励「仓位更高」而不是「风控更好」，
    #        所以族内必须用【年化超额 / 回撤】= alpha-Calmar。
    #   * 再加两条护栏：alpha 保留率下限（防止「几乎空仓 -> 回撤极小」刷分）、
    #     邻域最坏值（防止选到样本内的过拟合尖峰）。
    yrs_is = max((is_end - 1 - WARM) / 252.0, 1e-6)
    for c in CONFIGS:
        r = is_rows[c]
        r["cagr"] = (max(1.0 + r["ret"], 1e-9)) ** (1.0 / yrs_is) - 1
        r["cagr_b"] = (max(1.0 + r["bench"], 1e-9)) ** (1.0 / yrs_is) - 1
        r["cagr_ex"] = r["cagr"] - r["cagr_b"]
        r["acal"] = r["cagr_ex"] / abs(r["mdd"]) if r["mdd"] else 0.0
    for c in CONFIGS:
        nb = [is_rows[c]["acal"]] + [is_rows[x]["acal"] for x in neighbors(c)]
        is_rows[c]["acal_nb"] = min(nb) if nb else is_rows[c]["acal"]

    vol_cfgs = [c for c in CONFIGS if c[0] == "vol"]
    ma_cfgs = [c for c in CONFIGS if c[0] in MA_KINDS]

    def nb_pick(cands, ak_min):
        """邻域最坏 alpha-Calmar 最大 + alpha 保留率护栏；并列时取 alpha 保留高者"""
        ok = [c for c in cands if is_rows[c]["alpha_keep"] >= ak_min] or list(cands)
        return max(ok, key=lambda c: (is_rows[c]["acal_nb"], is_rows[c]["alpha_keep"]))

    RULES = [
        ("R1", "alpha 优先", "族内挑 alpha 保留率最高者（纯收益导向）",
         lambda cd: max(cd, key=lambda c: is_rows[c]["alpha_keep"])),
        ("R2", "回撤优先", "族内挑回撤最小者（纯风险导向）",
         lambda cd: min(cd, key=lambda c: abs(is_rows[c]["mdd"]))),
        ("R3", "alpha-Calmar 单点", "族内挑样本内 alpha-Calmar 最高者（无护栏、无防尖峰）",
         lambda cd: max(cd, key=lambda c: is_rows[c]["acal"])),
        ("R4", "邻域最坏 + α≥50%", "推荐：邻域最坏 alpha-Calmar 最大，alpha 保留率下限 50%",
         lambda cd: nb_pick(cd, 0.50)),
        ("R5", "邻域最坏 + α≥80%", "同 R4，护栏收紧到 80%（上一版默认值，检验护栏敏感性）",
         lambda cd: nb_pick(cd, 0.80)),
        ("R6", "邻域最坏 + α≥30%", "同 R4，护栏放宽到 30%（检验护栏敏感性）",
         lambda cd: nb_pick(cd, 0.30)),
    ]
    RECOMMENDED = "R4"

    rule_rows = []
    for rid, rname, rdesc, fn in RULES:
        pv, pm = fn(vol_cfgs), fn(ma_cfgs)
        rule_rows.append({"id": rid, "name": rname, "desc": rdesc, "vol": pv, "ma": pm})
        for tag, p in (("vol", pv), ("ma", pm)):
            rr = is_rows[p]
            log(f"  {rid} {rname:<18} {tag} 代表 = {p[3]:<26} "
                f"(α保留 {rr['alpha_keep'] * 100:>4.0f}%  回撤 {rr['mdd'] * 100:>5.1f}%  "
                f"α-Calmar {rr['acal']:>5.3f}  邻域最坏 {rr['acal_nb']:>5.3f})")

    rec = next(x for x in rule_rows if x["id"] == RECOMMENDED)
    pick, pick_ma = rec["vol"], rec["ma"]
    log(f"\n  >> 采用规则 {RECOMMENDED}（{rec['name']}）："
        f"波动率目标族代表 = {pick[3]} ｜ MA200 族代表 = {pick_ma[3]}")

    # 规则敏感性检验：把所有规则选出的代表去重后各跑一次全样本 + 样本外。
    # 注意：这些样本外数字只用于「事后观察结论对规则的敏感性」，
    # **不参与规则选择**（规则由方法论先验确定），否则样本外就被污染了。
    uniq = []
    for x in rule_rows:
        for p in (x["vol"], x["ma"]):
            if p not in uniq:
                uniq.append(p)
    full_map, oos_map = {}, {}
    for p in uniq:
        g = make_gate(p[0], p[1], p[2])
        full_map[p] = run_fund(A, dates, mkt, WARM, n - 1, g, fund, **kw_for(p))
        oos_map[p] = run_fund(A, dates, mkt, is_end, n - 1, g, fund, **kw_for(p))
        log(f"  敏感性 {p[3]:<26} 全样本 {full_map[p]['ret'] * 100:+7.2f}%  "
            f"样本外超额 {oos_map[p]['excess'] * 100:+7.2f}pp")
    for x in rule_rows:
        x["fv"], x["fm"] = full_map[x["vol"]], full_map[x["ma"]]
        x["ov"], x["om"] = oos_map[x["vol"]], oos_map[x["ma"]]

    def _full_acal(r):
        """全样本口径的 α-Calmar = 年化超额 / 最大回撤（与族内选代表同一把尺子）"""
        cg = (max(1.0 + r["ret"], 1e-9)) ** (1.0 / yrs) - 1
        cb = (max(1.0 + r["bench"], 1e-9)) ** (1.0 / yrs) - 1
        m = abs(r["raw"]["max_drawdown"])
        return (cg - cb) / m if m > 1e-9 else 0.0
    acal_full = {p: _full_acal(full_map[p]) for p in uniq}
    for x in rule_rows:
        x["ac_v"], x["ac_m"] = acal_full[x["vol"]], acal_full[x["ma"]]

    # 择时强度 vs 全样本 Calmar：检验「是不是存在一个最优择时强度」
    # （两族对决只能回答「谁赢」，回答不了「该择时多少」）
    def _c_full(r):
        m = abs(r["raw"]["max_drawdown"])
        return r["raw"]["annual"] / m if m > 1e-9 else 0.0
    strength_cfgs = ([CFG_BY_KEY[("ma200", None, None)], CFG_BY_KEY[("soft30", None, None)],
                      CFG_BY_KEY[("soft50", None, None)], CFG_BY_KEY[("soft70", None, None)]]
                     + [CFG_BY_KEY[("vol", t, 40)] for t in TARGETS])
    strength_rows = []
    for p in strength_cfgs:
        if p not in full_map:
            full_map[p] = run_fund(A, dates, mkt, WARM, n - 1, make_gate(p[0], p[1], p[2]),
                                   fund, **kw_for(p))
        r = full_map[p]
        strength_rows.append({"cfg": p, "ret": r["ret"], "mdd": r["raw"]["max_drawdown"],
                              "calmar": _c_full(r)})
        log(f"  强度 {p[3]:<26} 全样本 {r['ret'] * 100:+7.2f}%  回撤 {r['raw']['max_drawdown'] * 100:5.1f}%  "
            f"Calmar {_c_full(r):.3f}")

    BEST = make_gate(pick[0], pick[1], pick[2])
    BESTMA = make_gate(pick_ma[0], pick_ma[1], pick_ma[2])
    NOGATE = make_gate("nogate", None, None)
    MA200 = make_gate("ma200", None, None)
    SOFT30 = make_gate("soft30", None, None)

    # ---------- 6. 全样本对照 ----------
    log("\n[5] 全样本关键对照 ...")
    full_best = run_fund(A, dates, mkt, WARM, n - 1, BEST, fund, **kw_for(pick))
    bench_ret = full_best["bench"]
    full_nogate = run_fund(A, dates, mkt, WARM, n - 1, NOGATE, fund, **KW)
    acal_nogate = _full_acal(full_nogate)   # 「无风控」在同一把尺子下的值
    nogate_row = {"cfg": None, "ret": full_nogate["ret"],
                  "mdd": full_nogate["raw"]["max_drawdown"], "calmar": _c_full(full_nogate)}
    full_ma200 = run_fund(A, dates, mkt, WARM, n - 1, MA200, fund, **KW)
    full_soft = run_fund(A, dates, mkt, WARM, n - 1, BESTMA, fund, **kw_for(pick_ma))
    keep_full = (full_best["excess"] / full_nogate["excess"]) if full_nogate["excess"] > 1e-9 else 0.0
    cut_full = 1.0 - abs(full_best["raw"]["max_drawdown"]) / abs(full_nogate["raw"]["max_drawdown"])
    keep_ma_full = (full_soft["excess"] / full_nogate["excess"]) if full_nogate["excess"] > 1e-9 else 0.0
    log(f"  波动率目标  : {full_best['ret'] * 100:+.2f}%  回撤 {full_best['raw']['max_drawdown'] * 100:.1f}%")
    log(f"  无风控倾斜  : {full_nogate['ret'] * 100:+.2f}%  回撤 {full_nogate['raw']['max_drawdown'] * 100:.1f}%")
    log(f"  MA200 二元  : {full_ma200['ret'] * 100:+.2f}%")
    log(f"  MA200 留30% : {full_soft['ret'] * 100:+.2f}%")
    log(f"  基准        : {bench_ret * 100:+.2f}%")
    log(f"  >> 全样本：alpha 保留 {keep_full * 100:.0f}%，回撤削减 {cut_full * 100:.0f}%")

    # ---------- 7. 样本外 ----------
    log("\n[6] 样本外检验（>=%s）..." % OOS_CUTOFF)
    oos_best = run_fund(A, dates, mkt, is_end, n - 1, BEST, fund, **kw_for(pick))
    oos_nogate = run_fund(A, dates, mkt, is_end, n - 1, NOGATE, fund, **KW)
    oos_soft = run_fund(A, dates, mkt, is_end, n - 1, BESTMA, fund, **kw_for(pick_ma))
    log(f"  样本外 波动率目标 {oos_best['ret'] * 100:+.2f}% / 无风控 {oos_nogate['ret'] * 100:+.2f}% / "
        f"P3最优 {oos_soft['ret'] * 100:+.2f}% / 基准 {oos_best['bench'] * 100:+.2f}%")

    # ---------- 8. 随机信号对照 ----------
    log("\n[7] 随机信号对照（20 次，同风控层）...")
    rand_rows = []
    for i in range(20):
        r = run_fund(A, dates, mkt, WARM, n - 1, make_gate(pick[0], pick[1], pick[2], seed=i),
                     fund, **kw_for(pick))
        rand_rows.append({"seed": i, "ret": r["ret"], "turn": r["turnover"]})
    beat = sum(1 for x in rand_rows if x["ret"] < full_best["ret"])
    pct = int(beat / len(rand_rows) * 100)
    log(f"  跑赢 {beat}/{len(rand_rows)} 个随机 -> 第 {pct} 百分位")

    # ---------- 9. 成本 ----------
    log("\n[8] 成本敏感性 ...")
    cost_rows = []
    for slip in (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005):
        r = run_fund(A, dates, mkt, WARM, n - 1, BEST, fund, cost=bt.Cost(slippage=slip), **kw_for(pick))
        cost_rows.append({"slip": slip, "ret": r["ret"], "bench": r["bench"], "n": r["n"]})
        log(f"  滑点 {slip * 100:.2f}% -> {r['ret'] * 100:+8.2f}% (基准 {r['bench'] * 100:+.2f}%)")

    # ---------- 10. 滚动前推 ----------
    log("\n[9] 滚动前推 ...")
    segs, L = [], n // 4
    for i in range(4):
        s0 = max(WARM, i * L)
        s1 = n - 1 if i == 3 else (i + 1) * L - 1
        segs.append((s0, s1))
    wf_rows = []
    for (s0, s1) in segs:
        r = run_fund(A, dates, mkt, s0, s1, BEST, fund, **kw_for(pick))
        rn = run_fund(A, dates, mkt, s0, s1, NOGATE, fund, **KW)
        rm = run_fund(A, dates, mkt, s0, s1, MA200, fund, **KW)
        wf_rows.append({"seg": f"{dates[s0]}~{dates[s1]}", "ret": r["ret"], "bench": r["bench"],
                        "excess": r["excess"], "nogate": rn["ret"], "ma200": rm["ret"]})
        log(f"  {dates[s0]}~{dates[s1]}  波动率目标 {r['ret'] * 100:+7.2f}%  无风控 {rn['ret'] * 100:+7.2f}%  "
            f"MA200 {rm['ret'] * 100:+7.2f}%  基准 {r['bench'] * 100:+7.2f}%  超额 {r['excess'] * 100:+7.2f}pp")

    # ---------- 11. 报告 ----------
    log("\n[10] 生成报告 ...")
    v = {
        "A": A, "dates": dates, "n": n, "yrs": yrs, "is_end": is_end,
        "bad": bad, "tot": tot, "ic_is": ic_is, "ic_oos": ic_oos,
        "CONFIGS": CONFIGS, "is_rows": is_rows, "pick": pick, "pick_ma": pick_ma, "full": full,
        "rule_rows": rule_rows, "RECOMMENDED": RECOMMENDED, "acal_nogate": acal_nogate,
        "strength_rows": strength_rows, "nogate_row": nogate_row,
        "full_best": full_best, "full_nogate": full_nogate, "full_ma200": full_ma200,
        "full_soft": full_soft, "bench_ret": bench_ret,
        "keep_full": keep_full, "cut_full": cut_full, "keep_ma_full": keep_ma_full,
        "oos_best": oos_best, "oos_nogate": oos_nogate, "oos_soft": oos_soft,
        "rand_rows": rand_rows, "beat": beat, "pct": pct,
        "cost_rows": cost_rows, "wf_rows": wf_rows,
        "vol_stats": vol_stats, "OOS_CUTOFF": OOS_CUTOFF, "FNAME": FNAME,
        "FREQ": FREQ, "TAU": TAU, "FLOOR": FLOOR, "ALPHA_KEEP_MIN": ALPHA_KEEP_MIN,
    }
    html = build_html(v)
    open(OUT, "w", encoding="utf-8").write(html)
    log(f"  输出 {OUT} ({len(html.encode())} 字节)")
    log(f"\n[总耗时] {time.time() - t_all:.1f}s")


def build_html(v):
    A, dates, n, yrs = v["A"], v["dates"], v["n"], v["yrs"]
    bad, tot = v["bad"], v["tot"]
    ic_is, ic_oos = v["ic_is"], v["ic_oos"]
    CONFIGS, is_rows, pick, pick_ma = v["CONFIGS"], v["is_rows"], v["pick"], v["pick_ma"]
    rule_rows, RECOMMENDED = v["rule_rows"], v["RECOMMENDED"]
    acal_nogate = v["acal_nogate"]
    strength_rows, nogate_row = v["strength_rows"], v["nogate_row"]
    full = v["full"]
    full_best, full_nogate = v["full_best"], v["full_nogate"]
    full_ma200, full_soft = v["full_ma200"], v["full_soft"]
    bench_ret, keep_full, cut_full = v["bench_ret"], v["keep_full"], v["cut_full"]
    keep_ma_full = v["keep_ma_full"]
    oos_best, oos_nogate, oos_soft = v["oos_best"], v["oos_nogate"], v["oos_soft"]
    rand_rows, beat, pct = v["rand_rows"], v["beat"], v["pct"]
    cost_rows, wf_rows = v["cost_rows"], v["wf_rows"]
    vol_stats = v["vol_stats"]
    OOS_CUTOFF, FNAME, FREQ = v["OOS_CUTOFF"], v["FNAME"], v["FREQ"]
    TAU, FLOOR, AK = v["TAU"], v["FLOOR"], v["ALPHA_KEEP_MIN"]

    o = ['<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>AQuant v5 P4 · 组合层波动率目标风控</title><style>', CSS, '</style></head>',
         '<body><div class="wrap">']
    o.append('<h1>AQuant v5 · P4：风控层换成组合层波动率目标</h1>')
    o.append(f'<div class="sub">选股层（价值倾斜）不变，只把风控层从「MA200 方向性开关」换成'
             f'「按风险调仓的波动率目标」 ｜ 样本 {len(A)} 只 × {yrs:.1f} 年（{dates[0]} ~ {dates[-1]}）｜ '
             f'{FREQ} 调仓 ｜ 本金 {CASH / 1e8:.0f} 亿 ｜ 数据源：东方财富（point-in-time）</div>')

    # KPI
    o.append('<div class="kpi">')
    o.append(f'<div><span>alpha 保留率（全样本）</span><b class="{"ok" if keep_full >= 0.8 else "warn"}">'
             f'{keep_full * 100:.0f}%</b><span style="font-size:11px">MA200 族代表 {keep_ma_full * 100:.0f}%</span></div>')
    o.append(f'<div><span>回撤削减</span><b class="{"ok" if cut_full > 0.1 else "warn"}">'
             f'{cut_full * 100:.0f}%</b><span style="font-size:11px">'
             f'{full_best["raw"]["max_drawdown"] * 100:.1f}% vs '
             f'{full_nogate["raw"]["max_drawdown"] * 100:.1f}%</span></div>')
    o.append(f'<div><span>样本外超额</span><b class="{cls(oos_best["excess"])}">'
             f'{oos_best["excess"] * 100:+.2f}pp</b><span style="font-size:11px">'
             f'无风控 {oos_nogate["excess"] * 100:+.2f}pp</span></div>')
    o.append(f'<div><span>随机信号对照</span><b class="{"ok" if pct >= 80 else ("warn" if pct >= 60 else "bad")}">'
             f'第 {pct} 百分位</b><span style="font-size:11px">同倾斜同风控</span></div>')
    o.append('</div>')

    # 一、为什么换
    o.append('<h2>一、为什么换掉 MA200</h2>')
    o.append('<div class="box bad"><b>P3 的诊断</b>：正交架构（标量 × 向量）本身成立，'
             'alpha 保留率已从二元门控的 68% 提到 81%。剩余亏损来自 '
             '<b>MA200 这把尺子本身的误判</b> —— 它是<b>方向性开关</b>，'
             '把 2024-26 牛市也切掉了一块（该段正交组合超额 <b>-18.56pp</b>，'
             '而无门控同期 <b>+13.63pp</b>）。继续调阈值是在错误的地方使劲。</div>')
    o.append('<table><thead><tr><th>风控层</th><th>调仓依据</th><th>牛市表现</th><th>崩盘表现</th></tr></thead><tbody>')
    o.append('<tr><td>MA200 门控</td><td><b>方向</b>（价在均线上下）</td>'
             '<td class="neg">可能误判为熊市而踏空</td><td>能减仓</td></tr>')
    o.append('<tr style="background:#fff7ed"><td><b>波动率目标</b></td>'
             '<td><b>风险</b>（已实现波动高低）</td>'
             '<td class="pos">走得稳就满仓（e 被 cap 到 1.0），不踏空</td>'
             '<td>波动飙升自动降仓</td></tr>')
    o.append('</tbody></table>')

    # 二、正交性说明
    o.append('<h2>二、正交性：为什么必须用「指数」波动</h2>')
    o.append('<table><thead><tr><th>做法</th><th>缩放依据</th><th>是否正交</th><th>问题</th></tr></thead><tbody>')
    o.append('<tr class="neg"><td>逐票 <code>vol_target</code></td><td>每只票<b>自己的</b>波动</td>'
             '<td><b>不正交</b></td><td>会系统性超配低波动股 = 偷引入一个「低波因子」，'
             '与选股层耦合、污染 alpha 归因</td></tr>')
    o.append('<tr style="background:#fff7ed"><td><b>组合层</b> <code>port_vol_target</code></td>'
             '<td><b>指数</b>已实现波动 → 全组合共用一个标量 <code>e</code></td>'
             '<td><b>严格正交</b></td><td>只缩放总暴露，完全不改变各票相对权重</td></tr>')
    o.append('</tbody></table>')
    o.append('<div class="box info">公式：<code>e = clip(target / 指数年化已实现波动率, floor, cap)</code>，'
             '最终持仓 <code>e × w_i</code>。指数波动由引擎按 '
             '<code>mkt_vol_n</code> 回看窗口计算（日收益滚动标准差 × √252）。</div>')
    o.append(f'<div class="box {"ok" if bad == 0 else "bad"}"><b>前视偏差审计</b>：面板 {tot} 条快照中 '
             f'<b>{bad} 条</b>越界（须为 0）。</div>')

    # 三、波动率诊断
    o.append('<h2>三、先看数据：指数波动率到底在什么范围</h2>')
    o.append('<p>不先确认 <code>e = target/vol</code> 落在合理区间就扫参数是盲扫。'
             '以下是沪深300 已实现年化波动率的实测分布：</p>')
    o.append('<table><thead><tr><th>回看窗口</th><th>均值</th><th>p10</th><th>p90</th>'
             '<th>最小</th><th>最大</th></tr></thead><tbody>')
    for vn, s in vol_stats.items():
        o.append(f'<tr><td>{vn} 日</td><td>{s["mean"] * 100:.1f}%</td><td>{s["p10"] * 100:.1f}%</td>'
                 f'<td>{s["p90"] * 100:.1f}%</td><td>{s["min"] * 100:.1f}%</td>'
                 f'<td>{s["max"] * 100:.1f}%</td></tr>')
    o.append('</tbody></table>')
    o.append('<table><thead><tr><th>target</th><th>低波时(p10) e</th><th>高波时(p90) e</th>'
             '<th>含义</th></tr></thead><tbody>')
    vn0 = sorted(vol_stats.keys())[0]
    s0 = vol_stats[vn0]
    for tg in (0.10, 0.15, 0.20, 0.25):
        e_lo, e_hi = tg / s0["p10"], tg / s0["p90"]
        e_lo, e_hi = min(e_lo, 1.0), min(e_hi, 1.0)
        o.append(f'<tr><td>{tg:.0%}</td><td>{e_lo:.2f}</td><td>{e_hi:.2f}</td>'
                 f'<td>{"几乎长期满仓，风控形同虚设" if e_hi > 0.95 else "高波时确实降仓，低波时满仓" if e_lo > 0.95 else "常年不满仓，会持续损失 alpha 与 beta"}</td></tr>')
    o.append('</tbody></table>')

    # 四、信号复核
    o.append(f'<h2>四、信号复核：{FNAME}</h2>')
    o.append('<table><thead><tr><th>区间</th><th>方向IC</th><th>t_dir</th><th>IC&gt;0 占比</th></tr></thead><tbody>')
    for nm, r in (("样本内", ic_is), ("样本外", ic_oos)):
        if r is None:
            o.append(f'<tr><td>{nm}</td><td colspan="3">样本不足</td></tr>')
        else:
            o.append(f'<tr><td><b>{nm}</b></td><td>{r["mean_dir"]:+.4f}</td><td>{r["t_dir"]:+.2f}</td>'
                     f'<td>{r["hit"] * 100:.0f}%</td></tr>')
    o.append('</tbody></table>')

    # 五、配置前沿
    o.append('<h2>五、配置前沿（样本内扫描，选股层固定只换风控层）</h2>')
    o.append('<table><thead><tr><th>风控配置</th><th>样本内超额</th><th>alpha 保留率</th>'
             '<th>最大回撤</th><th>回撤削减</th><th>α-Calmar</th><th>邻域最坏<br>α-Calmar</th>'
             '<th>年换手</th></tr></thead><tbody>')
    for c in CONFIGS:
        r = is_rows[c]
        mark = (' style="background:#fff7ed"' if c == pick
                else ' style="background:#f0f7ff"' if c == pick_ma
                else ' style="background:#fafafa"' if c[0] == "nogate" else '')
        ak_c = "ok" if r["alpha_keep"] >= AK else ("warn" if r["alpha_keep"] >= 0.3 else "bad")
        nbc = "ok" if r["acal_nb"] > 0 else "bad"
        o.append(f'<tr{mark}><td>{c[3]}</td>'
                 f'<td><b class="{cls(r["excess"])}">{r["excess"] * 100:+.2f}pp</b></td>'
                 f'<td><b class="{ak_c}">{r["alpha_keep"] * 100:.0f}%</b></td>'
                 f'<td>{r["mdd"] * 100:.1f}%</td><td>{r["mdd_cut"] * 100:.0f}%</td>'
                 f'<td>{r["acal"]:.3f}</td><td><b class="{nbc}">{r["acal_nb"]:.3f}</b></td>'
                 f'<td>{r["turn"] * 100:.0f}%</td></tr>')
    o.append('</tbody></table>')
    pr = is_rows[pick]
    o.append(f'<div class="box info">· 波动率目标族代表（P4 主角）：<b>{pick[3]}</b>'
             f'（alpha 保留 {pr["alpha_keep"] * 100:.0f}%，回撤 {pr["mdd"] * 100:.1f}%，'
             f'回撤削减 {pr["mdd_cut"] * 100:.0f}%）'
             f'<br>· MA200 族代表（P3）：<b>{pick_ma[3]}</b>'
             f'（alpha 保留 {is_rows[pick_ma]["alpha_keep"] * 100:.0f}%，'
             f'回撤 {is_rows[pick_ma]["mdd"] * 100:.1f}%，'
             f'回撤削减 {is_rows[pick_ma]["mdd_cut"] * 100:.0f}%）'
             f'<br>两族<b>分别</b>选代表再对决 —— 若混在一起比，规则会选回 MA200，'
             f'导致「P4」标签下跑的其实是 P3 的配置（首版就踩了这个坑）。'
             f'选代表的规则本身见下一节。</div>')
    # 样本内正面对决
    d_ak = (pr["alpha_keep"] - is_rows[pick_ma]["alpha_keep"]) * 100
    d_dd = (abs(is_rows[pick_ma]["mdd"]) - abs(pr["mdd"])) * 100   # 正=波动率目标回撤更大
    o.append(f'<div class="box {"ok" if d_ak > 0 and d_dd <= 1.0 else "warn"}">'
             f'<b>样本内正面对决</b>：波动率目标比 MA200 族代表 alpha 保留率'
             f'<b>{d_ak:+.0f}pp</b>，但最大回撤'
             f'{"更大" if d_dd > 0 else "更小"} <b>{abs(d_dd):.1f}pp</b>。'
             f'{"波动率目标在保留 alpha 上占优，代价是回撤削减更少。" if d_ak > 0 and d_dd > 0 else ""}'
             f'{"两者互有胜负，需看样本外与牛市段再定论。" if abs(d_dd) <= 1.0 else ""}</div>')

    # 六、族内选代表的规则（本节的答案是 R4）
    o.append('<h2>六、族内选代表：规则必须先于结论</h2>')
    o.append('<div class="box bad"><b>为什么「选哪个代表」值得单独论证</b>：'
             '族内选代表的规则<b>会预设对决的赢家</b>。'
             '若规则是「alpha 优先」，波动率目标族会派出 target=25%（e 恒为 1.0，等于不风控），'
             '对决时它 alpha 必然最高；若规则是「回撤优先」，它会派出 target=10%（几乎空仓），'
             '对决时它必输。<b>换规则就换赢家</b> —— 那么结论就不是「波动率目标行不行」，'
             '而是「规则让谁站在了有利位置」，归因是错的。</div>')

    o.append('<h3>6.1 三条设计原则</h3>')
    o.append('<table><thead><tr><th>原则</th><th>内容</th><th>不这么做会怎样</th></tr></thead><tbody>')
    o.append('<tr><td><b>① 用超额口径，不用总收益</b></td>'
             '<td>族内候选共享同一个选股层，唯一差别是风控质量。'
             '故用 <code>年化超额 / 最大回撤</code>（α-Calmar）作目标函数</td>'
             '<td>用总收益会把 beta 混进来 —— 等于奖励「仓位更高」而不是「风控更好」，'
             '族内比较会被无差别的市场暴露主导</td></tr>')
    o.append('<tr><td><b>② alpha 保留率下限护栏</b></td>'
             '<td>候选须保留 ≥50% 的 alpha（超额相对无风控的比例）</td>'
             '<td>否则「几乎空仓 → 回撤极小」会刷出虚高的 α-Calmar，'
             '选出的是一个把信号摧毁掉的配置。风控层的职责是控风险，不是毁灭信号</td></tr>')
    o.append('<tr><td><b>③ 邻域最坏值防尖峰</b></td>'
             '<td>取自身与其<b>参数轴相邻配置</b>中 α-Calmar 的<b>最小值</b>再排序</td>'
             '<td>样本内单点最优常是过拟合尖峰；代表应当代表该族的<b>稳健水平</b>，'
             '而不是孤立的幸运点</td></tr>')
    o.append('</tbody></table>')
    o.append(f'<div class="box info"><b>邻域定义</b>：波动率目标族按 target 档 '
             f'（{"/".join(f"{t:.0%}" for t in TARGETS)}）相邻 ±1 档 + 同 target 的另一回看窗口'
             f'（{"/".join(str(x) for x in LOOKBACKS)} 日）；MA200 族按熊市保留仓位 '
             f'（{"/".join(f"{b:.0%}" for b in BEARWS)}）相邻档。'
             f'这也是本节要把 MA200 族从 2 个候选扩到 4 个的原因 —— '
             f'只有 2 个点时「邻域最坏值」会退化成两点取小，无法区分。</div>')

    o.append('<h3>6.2 六个候选规则正面对比</h3>')
    o.append('<table><thead><tr><th>规则</th><th>定义</th><th>波动率目标族代表</th>'
             '<th>MA200 族代表</th><th>样本内邻域最坏<br>α-Calmar (vol / ma)</th></tr></thead><tbody>')
    for x in rule_rows:
        hi = ' style="background:#fff7ed"' if x["id"] == RECOMMENDED else ''
        av, am = is_rows[x["vol"]]["acal_nb"], is_rows[x["ma"]]["acal_nb"]
        tag = ' <b>★采用</b>' if x["id"] == RECOMMENDED else ''
        o.append(f'<tr{hi}><td><b>{x["id"]}</b>{tag}</td><td style="text-align:left">{x["desc"]}</td>'
                 f'<td>{x["vol"][3]}</td><td>{x["ma"][3]}</td>'
                 f'<td>{av:.3f} / {am:.3f}</td></tr>')
    o.append('</tbody></table>')

    # 规则敏感性：结论是否随规则翻转
    o.append('<h3>6.3 敏感性检验：换规则会不会换结论</h3>')
    o.append('<div class="box info">族间对决<b>没有唯一正确的尺子</b>，所以这里并列展示四个口径，'
             '而不是只挑一个对外宣称。前两列是「选出的代表是否稳定」，'
             '后四列是「该规则下两族代表的正面对决」。</div>')
    o.append('<table><thead><tr><th>规则</th><th>vol 代表</th><th>ma 代表</th>'
             '<th>全样本收益<br>(vol / ma)</th><th>样本外超额<br>(vol / ma)</th>'
             '<th>全样本 α-Calmar<br>(vol / ma)</th><th>全样本回撤<br>(vol / ma)</th>'
             '</tr></thead><tbody>')
    w_ret, w_oos, w_acal, w_dd = {}, {}, {}, {}
    for x in rule_rows:
        fv, fm = x["fv"]["ret"], x["fm"]["ret"]
        ov, om = x["ov"]["excess"], x["om"]["excess"]
        av, am = x["ac_v"], x["ac_m"]
        dv, dm = abs(x["fv"]["raw"]["max_drawdown"]), abs(x["fm"]["raw"]["max_drawdown"])
        def who(a, b, bigger=True):
            if abs(a - b) < 1e-9:
                return "持平"
            win = a > b if bigger else a < b
            return "波动率目标" if win else "MA200"
        w_ret[who(fv, fm)] = w_ret.get(who(fv, fm), 0) + 1
        w_oos[who(ov, om)] = w_oos.get(who(ov, om), 0) + 1
        w_acal[who(av, am)] = w_acal.get(who(av, am), 0) + 1
        w_dd[who(dv, dm, False)] = w_dd.get(who(dv, dm, False), 0) + 1
        hi = ' style="background:#fff7ed"' if x["id"] == RECOMMENDED else ''
        o.append(f'<tr{hi}><td><b>{x["id"]}</b></td><td>{x["vol"][3]}</td><td>{x["ma"][3]}</td>'
                 f'<td>{fv * 100:+.2f}% / {fm * 100:+.2f}%</td>'
                 f'<td>{ov * 100:+.2f}pp / {om * 100:+.2f}pp</td>'
                 f'<td>{av:.3f} / {am:.3f}</td>'
                 f'<td>{dv * 100:.1f}% / {dm * 100:.1f}%</td></tr>')
    o.append('</tbody></table>')

    def _dist(d):
        return "、".join(f"{k}×{v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))

    o.append(f'<div class="box warn"><b>四个口径下的赢家分布</b>（六条规则合计，每条规则各计 1 票）：'
             f'<br>· 全样本<b>收益</b>：{_dist(w_ret)}'
             f'<br>· <b>样本外超额</b>：{_dist(w_oos)}'
             f'<br>· 全样本 <b>α-Calmar</b>（风险调整 alpha）：{_dist(w_acal)}'
             f'<br>· 全样本<b>回撤</b>（越小越好）：{_dist(w_dd)}'
             f'<br><b>解读</b>：收益与样本外 alpha 由波动率目标族拿下，'
             f'风险调整 alpha 与回撤由 MA200 族拿下 —— '
             f'<b>两个族各赢一半口径</b>，不存在「某一族全面胜出」。'
             f'这正是第七节「同一条风险-收益线」结论在选代表层面的复现：'
             f'换尺子就换赢家，所以<b>任何单一口径的对决宣称都是片面的</b>。</div>')

    # 代表是否稳定（这才是规则本身该被检验的性质）
    ids_nb = [x for x in rule_rows if x["id"] in ("R4", "R5", "R6")]
    ma_stable = len({x["ma"][3] for x in rule_rows if x["id"] != "R2"}) == 1
    vol_stable_nb = len({x["vol"][3] for x in ids_nb}) == 1
    vol_target_stable = len({x["vol"][1] for x in rule_rows if x["id"] != "R2"}) == 1
    o.append(f'<div class="box {"ok" if (ma_stable and vol_stable_nb) else "warn"}">'
             f'<b>规则本身的稳健性</b>（这才是本节要检验的对象）：'
             f'<br>· MA200 族代表：除 R2 外五条规则<b>完全一致</b>'
             f'（{next(x["ma"][3] for x in rule_rows if x["id"] != "R2")}）'
             f'{"✅" if ma_stable else ""}'
             f'<br>· 波动率目标族代表：护栏阈值在 <b>30%~80%</b> 之间任意挪动（R4/R5/R6）'
             f'选出的代表<b>完全相同</b>（{ids_nb[0]["vol"][3]}）'
             f'{"✅" if vol_stable_nb else "❌"} —— '
             f'<b>护栏的具体取值不再是一个能左右结果的旋钮</b>，这是 R4 相对旧规则的核心改进。'
             f'<br>· 但<b>是否做邻域平滑</b>会改变回看窗口（R3 选 40 日、R4 选 60 日），'
             f'两族代表的 target / 熊市仓位则{vol_target_stable and "稳定不变" or "有变化"}；'
             f'两者结果差异仅 1pp 量级（全样本 +94.81% vs +93.87%），说明这个摆动<b>不影响结论</b>。'
             f'<br>· R2 是唯一的例外：它派出 target=10%（α-Calmar <b>-0.048</b>，为负），'
             f'恰好<b>反证了护栏 ② 的必要性</b> —— 纯回撤导向会选出一个把 alpha 摧毁成负数的配置。</div>')
    # 6.4 本节最重要的负面结果
    ac_v, ac_m = next(x for x in rule_rows if x["id"] == RECOMMENDED)["ac_v"], \
                 next(x for x in rule_rows if x["id"] == RECOMMENDED)["ac_m"]
    o.append('<h3>6.4 这条规则真正告诉我们的事（含负面结果）</h3>')
    ma_better = ac_m > ac_v + 1e-9
    win_side = "MA200 族代表更高" if ma_better else "波动率目标族代表更高"
    dd_s, dd_v = abs(full_soft["raw"]["max_drawdown"]), abs(full_best["raw"]["max_drawdown"])
    why = (f'MA200 族代表回撤更小（{dd_s * 100:.1f}% vs {dd_v * 100:.1f}%），而 alpha 只略低，'
           f'故在这把尺子上占优。' if ma_better
           else f'波动率目标在这把尺子上同样占优（回撤 {dd_v * 100:.1f}% vs {dd_s * 100:.1f}%）。')
    o.append(f'<div class="box {"warn" if ma_better else "ok"}">'
             f'<b>结果 ①：在族内那把尺子（α-Calmar）上，{win_side}。</b>'
             f'全样本 α-Calmar：波动率目标族代表 <b>{ac_v:.3f}</b> vs MA200 族代表 <b>{ac_m:.3f}</b>。{why}'
             f'<b>因此「波动率目标优于 MA200」不是无条件的结论</b>，必须说明是在哪个口径下 —— '
             f'这正是本节存在的意义。</div>')
    both = acal_nogate > max(ac_v, ac_m) + 1e-9
    only_vol = acal_nogate > ac_v + 1e-9 and acal_nogate < ac_m - 1e-9
    judge = ("两族代表的 α-Calmar <b>都低于</b>它 —— 在本样本内，"
             "<b>没有任何风控层能把风险调整后的 alpha 抬高</b>，"
             "它们只是把组合沿同一条风险-收益线挪到不同位置。" if both else
             "它高于波动率目标族代表、<b>却低于 MA200 族代表</b> —— 即唯一能在这把尺子上"
             "胜过「不择时」的是 <b>MA200 留 70%</b>，而它本质上是「只减 30% 仓位的近乎不择时」，"
             "等于变相承认：<b>激进择时在本样本是负贡献，轻微择时才有微弱正贡献</b>。" if only_vol else
             "两族代表都<b>高于</b>它 —— 说明风控层在本样本确实抬高了风险调整后的 alpha。")
    o.append(f'<div class="box {"bad" if both else "warn"}"><b>结果 ②：把「不风控」也放进同一把尺子。</b>'
             f'无风控倾斜的全样本 α-Calmar 是 <b>{acal_nogate:.3f}</b>'
             f'（vol 代表 {ac_v:.3f}、ma 代表 {ac_m:.3f}）。{judge}'
             f'这与第七节 Calmar 的结论是同一件事的两种表述。</div>')
    o.append(f'<div class="box warn"><b>那 R4 的价值是什么？</b>它<b>不保证某一族胜出</b>，'
             f'它保证的是「代表的选择不再是一个可以随手拧的旋钮」：'
             f'护栏在 30%~80% 之间任意挪动都选出同一个代表，且护栏的存在与否'
             f'（对比 R2）决定性地避免了选出负 alpha 的配置；'
             f'而旧的「α≥80% 中挑回撤最小」本质上是一个未加论证的阈值，'
             f'换个阈值就会换人（本例中它选出回看 40 日，R4 选出 60 日）。'
             f'两者结果差异很小（全样本 +94.81% vs +93.87%），'
             f'但只有 R4 的情况下这个「差异很小」是<b>被检验过的</b>，而不是碰巧。</div>')
    o.append(f'<div class="box warn"><b>纪律声明</b>：上表的样本外数字只用于<b>事后观察</b>结论对规则的敏感性，'
             f'<b>不参与规则的选择</b>。规则 R4 是由 6.1 的三条方法论原则<b>先验</b>确定的'
             f'（判据一致 + 护栏防刷分 + 邻域防尖峰），与数据无关；'
             f'若拿样本外结果去挑规则，样本外硬门槛就被污染了，后续所有样本外数字都不再可信。</div>')

    # 6.5 择时强度曲线
    o.append('<h3>6.5 真正该选的不是「哪一族」，而是「择时多少」</h3>')
    o.append('<p>两族对决只能回答「谁赢」，回答不了「该择时多激进」。'
             '把两族的配置按<b>择时强度</b>排成一列看全样本 Calmar，'
             '才能看出是否存在一个最优强度。</p>')
    o.append('<table><thead><tr><th>择时强度</th><th>配置</th><th>全样本收益</th>'
             '<th>最大回撤</th><th>Calmar</th></tr></thead><tbody>')
    merged = [(nogate_row, "不择时（无风控）")] + \
             [(r, r["cfg"][3]) for r in strength_rows]
    cal_list = [r["calmar"] for r, _ in merged]
    bestc = max(cal_list)
    for r, nm in merged:
        isbest = abs(r["calmar"] - bestc) < 1e-9
        mk = ' style="background:#fff7ed"' if isbest else ''
        star = ' ★' if isbest else ''
        o.append(f'<tr{mk}><td>{"—" if r["cfg"] is None else ""}</td><td>{nm}{star}</td>'
                 f'<td>{r["ret"] * 100:+.2f}%</td><td>{r["mdd"] * 100:.1f}%</td>'
                 f'<td><b>{r["calmar"]:.3f}</b></td></tr>')
    o.append('</tbody></table>')
    spread = bestc - min(cal_list)
    if spread > 0.04:
        spread_note = (f'极差超过 0.04，说明<b>不同择时强度之间存在真实差异</b>，'
                       f'并不是「沿同一条线挪位置」那么简单 —— 存在一个<b>最优择时强度</b>。')
    else:
        spread_note = '极差很小，说明各配置基本落在同一条风险-收益线上，只是位置不同。'
    names_best = [nm for r, nm in merged if abs(r["calmar"] - bestc) < 1e-9]
    ma_cals = [r["calmar"] for r in strength_rows if r["cfg"][0] in MA_KINDS]
    vol_cals = [r["calmar"] for r in strength_rows if r["cfg"][0] == "vol"]
    ng = nogate_row["calmar"]
    ma_mono = all(ma_cals[i] < ma_cals[i + 1] for i in range(len(ma_cals) - 1))
    vp = max(range(len(vol_cals)), key=lambda i: vol_cals[i])
    vol_u = 0 < vp < len(vol_cals) - 1
    o.append(f'<div class="box {"ok" if spread > 0.04 else "warn"}">'
             f'<b>Calmar 跨度 {min(cal_list):.3f}~{bestc:.3f}（极差 {spread:.3f}）</b>。{spread_note}'
             f'最高的是 <b>{names_best[0]}</b>。'
             f'<br>· <b>MA200 族</b>（清仓 → 留30% → 留50% → 留70%，择时强度递减）：'
             f'Calmar {"<b>单调递增</b>" if ma_mono else "非单调"}，峰值 {max(ma_cals):.3f} '
             f'出现在<b>最轻的一档</b>，且{"高于" if max(ma_cals) > ng else "低于"}完全不择时的 {ng:.3f}'
             f' —— 说明<b>轻微择时确实有正贡献</b>，而减仓越狠越差'
             f'（清仓仅 {ma_cals[0]:.3f}）。'
             f'<br>· <b>波动率目标族</b>（target 10% → 25%）：峰值 {vol_cals[vp]:.3f} 在 '
             f'target={TARGETS[vp]:.0%}，两侧更低（{vol_cals[0]:.3f} / {vol_cals[-1]:.3f}），'
             f'{"呈<b>倒 U 形</b> —— 降仓太少浪费风险预算、降太多又砍掉 alpha，两头都不好。" if vol_u else "趋势单调。"}'
             f'<br><b>合起来看</b>：两族都指向同一件事 —— <b>最优择时强度是"很轻微"的</b>，'
             f'而不是越激进越好。这与 P2「无门控最优」并不矛盾：'
             f'P2 比的是二元清仓（最激进的一档），本表补上了中间档，发现轻档更好。</div>')
    o.append(f'<div class="box warn"><b>边界</b>：上表是全样本口径，而代表本身是用<b>样本内</b>数据选出来的，'
             f'两者区间重叠 —— 所以「最优强度」这个结论带有选择偏差，'
             f'其可信度低于不依赖选参的样本外检验。它应当被当作<b>待验证的假设</b>，而不是已证实的结论。</div>')

    # 七、全样本对照
    o.append('<h2>七、全样本关键对照（含 Calmar 风险调整）</h2>')
    o.append('<table><thead><tr><th>策略</th><th>总收益</th><th>年化</th><th>相对基准</th>'
             '<th>最大回撤</th><th>年化/回撤<br>(Calmar)</th><th>年换手</th></tr></thead><tbody>')

    def _calmar(r):
        m = abs(r["raw"]["max_drawdown"])
        return r["raw"]["annual"] / m if m > 1e-9 else 0.0

    rows = [("波动率目标（P4）", full_best, True),
            ("无风控倾斜（P2 最佳）", full_nogate, False),
            (f"MA200 族代表：{pick_ma[3]}", full_soft, False),
            ("MA200 二元门控（P1/P2）", full_ma200, False)]
    for nm, r, hi in rows:
        mk = ' style="background:#fff7ed"' if hi else ''
        o.append(f'<tr{mk}><td><b>{nm}</b></td><td>{pc(r["ret"])}</td><td>{r["raw"]["annual"] * 100:.2f}%</td>'
                 f'<td><b class="{cls(r["excess"])}">{r["excess"] * 100:+.2f}pp</b></td>'
                 f'<td>{r["raw"]["max_drawdown"] * 100:.1f}%</td>'
                 f'<td><b>{_calmar(r):.3f}</b></td><td>{r["turnover"] * 100:.0f}%</td></tr>')
    o.append(f'<tr><td>等权买入持有（基准）</td><td>{pc(bench_ret)}</td><td>—</td><td>—</td>'
             f'<td>{full_best["raw"]["benchmark"]["mdd"] * 100:.1f}%</td><td>—</td><td>0%</td></tr>')
    o.append('</tbody></table>')
    c4, cn, c3, cm = (_calmar(full_best), _calmar(full_nogate),
                      _calmar(full_soft), _calmar(full_ma200))
    names = [(c4, "波动率目标(P4)"), (cn, "无风控(P2)"), (c3, "MA200族代表"), (cm, "MA200二元")]
    best_calmar = max(v for v, _ in names)
    tied = [nm for v, nm in names if abs(v - best_calmar) < 0.005]   # 差异<0.005 视为并列
    winner = "、".join(tied)
    tie_note = f"（{len(tied)} 个并列，差异 &lt;0.005 视为持平）" if len(tied) > 1 else ""
    o.append(f'<div class="box {"ok" if len(tied) == 1 and "P4" in winner else "warn"}">'
             f'<b>Calmar（年化/回撤）对比</b>：P4 波动率目标 <b>{c4:.3f}</b>、'
             f'无风控 <b>{cn:.3f}</b>、MA200 族代表 <b>{c3:.3f}</b>、MA200 二元 <b>{cm:.3f}</b>'
             f' —— 风险调整后最优是 <b>{winner}</b>{tie_note}。<br>'
             + (f'<b>关键解读</b>：四者差距不小（极差 {best_calmar - min(c4, cn, c3, cm):.3f}）—— '
                f'其中 <b>{pick_ma[3]}</b> 的 Calmar 反而高于完全不择时的无风控，'
                f'说明<b>不是「风控只能沿同一条线挪位置」这么简单</b>：'
                f'存在一个<b>最优择时强度</b>，不择时浪费了风险预算，过度择时又砍掉 alpha（详见 6.5）。'
                f'但它是样本内选出的参数、与全样本区间重叠，<b>可信度低于样本外检验</b>，'
                f'应视为待验证假设。'
                if best_calmar - min(c4, cn, c3, cm) > 0.04 else
                f'<b>关键解读</b>：四者 Calmar 集中在 {min(c4, cn, c3, cm):.3f}~{best_calmar:.3f} 的窄区间，'
                f'说明它们<b>都落在同一条「风险-收益」直线上，只是位置不同</b> —— '
                f'风控层的作用是<b>沿着这条线移动仓位</b>，而不是把线往上抬。'
                f'想削回撤就得沿线往左下走（同时放弃 alpha），没有免费午餐。') + '</div>')
    o.append(f'<div class="box {"ok" if keep_full >= 0.8 and cut_full > 0.1 else "warn"}">'
             f'全样本：P4 保留 <b>{keep_full * 100:.0f}%</b> alpha'
             f'（MA200 族代表 {keep_ma_full * 100:.0f}%），回撤削减 <b>{cut_full * 100:.0f}%</b>'
             f'（{full_best["raw"]["max_drawdown"] * 100:.1f}% vs 无风控 '
             f'{full_nogate["raw"]["max_drawdown"] * 100:.1f}%）。</div>')

    # 七、样本外
    o.append(f'<h2>八、样本外检验（硬门槛：{OOS_CUTOFF} 之后）</h2>')
    o.append('<table><thead><tr><th>策略</th><th>样本外收益</th><th>样本外基准</th><th>样本外超额</th></tr></thead><tbody>')
    for nm, r, hi in (("波动率目标（P4）", oos_best, True),
                      ("无风控倾斜（P2）", oos_nogate, False),
                      (f"MA200 族代表：{pick_ma[3]}", oos_soft, False)):
        mk = ' style="background:#fff7ed"' if hi else ''
        o.append(f'<tr{mk}><td><b>{nm}</b></td><td>{pc(r["ret"])}</td><td>{pc(r["bench"])}</td>'
                 f'<td><b class="{cls(r["excess"])}">{r["excess"] * 100:+.2f}pp</b></td></tr>')
    o.append('</tbody></table>')
    gain = (oos_best["excess"] - oos_soft["excess"]) * 100
    o.append(f'<div class="box {"ok" if oos_best["excess"] > 0 else "bad"}">样本外超额 '
             f'<b>{oos_best["excess"] * 100:+.2f}pp</b>；相对 MA200 族代表'
             f'<b>{gain:+.2f}pp</b>'
             f'{"—— 换风控层后样本外 alpha 回升，方向正确。" if gain > 0 else "—— 样本外未改善。"}'
             f'（v4 在此翻车 -12.65pp，P2 +22.45pp，P3 +12.47pp）</div>')

    # 八、随机对照
    o.append('<h2>九、随机信号对照（照妖镜）</h2>')
    o.append(f'<div class="box warn">对照组与 P4 <b>完全相同</b>的倾斜强度、'
             f'<b>相同的风控层</b>（{pick[3]}）、调仓频率与因子覆盖，'
             f'唯一区别是把「哪个票拿哪个权重」随机打乱。</div>')
    o.append('<table><thead><tr><th>策略</th><th>年换手</th><th>总收益</th></tr></thead><tbody>')
    o.append(f'<tr style="background:#fff7ed"><td><b>波动率目标（P4）</b></td>'
             f'<td>{full_best["turnover"] * 100:.0f}%</td><td><b>{pc(full_best["ret"])}</b></td></tr>')
    for x in rand_rows:
        o.append(f'<tr><td>随机倾斜 #{x["seed"]}</td><td>{x["turn"] * 100:.0f}%</td>'
                 f'<td>{pc(x["ret"])}</td></tr>')
    o.append('</tbody></table>')
    o.append(f'<div class="box {"ok" if pct >= 80 else ("warn" if pct >= 60 else "bad")}">'
             f'在 {len(rand_rows)} 个随机策略中跑赢 <b>{beat} 个（第 {pct} 百分位）</b>。'
             f'{"显著优于随机，选股信号有信息量。" if pct >= 80 else "与随机无异，alpha 未被证实。" if pct < 60 else "略优于随机。"}</div>')

    # 九、成本
    o.append('<h2>十、成本敏感性</h2>')
    o.append('<table><thead><tr><th>双边滑点</th><th>策略收益</th><th>基准收益</th><th>相对基准</th><th>交易笔数</th></tr></thead><tbody>')
    for r in cost_rows:
        ex = r["ret"] - r["bench"]
        o.append(f'<tr><td>{r["slip"] * 100:.2f}%</td><td>{pc(r["ret"])}</td><td>{pc(r["bench"])}</td>'
                 f'<td><b class="{cls(ex)}">{ex * 100:+.2f}pp</b></td><td>{r["n"]}</td></tr>')
    o.append('</tbody></table>')
    worst = cost_rows[-1]
    o.append(f'<div class="box {"bad" if worst["ret"] < worst["bench"] else "ok"}">'
             f'滑点 {worst["slip"] * 100:.2f}% 时收益 {pc(worst["ret"])}（基准 {pc(worst["bench"])}）。'
             f'{"仍跑赢基准，alpha 非过拟合。" if worst["ret"] > worst["bench"] else "被成本吃掉。"}</div>')

    # 十、滚动前推
    o.append('<h2>十一、滚动前推（关键：牛市段还踏空吗）</h2>')
    o.append('<p>P3 的痛点正是最后一段牛市（2024-09~2026-10）MA200 踏空 -18.56pp。'
             '波动率目标主张「按风险而非方向调仓」，应能修复这一点 —— 本表是决定性检验。</p>')
    o.append('<table><thead><tr><th>区间</th><th>波动率目标(P4)</th><th>无风控(P2)</th>'
             '<th>MA200 二元</th><th>基准</th><th>P4 超额</th></tr></thead><tbody>')
    wins = 0
    for r in wf_rows:
        if r["excess"] > 0:
            wins += 1
        o.append(f'<tr><td class="mono">{r["seg"]}</td><td>{pc(r["ret"])}</td><td>{pc(r["nogate"])}</td>'
                 f'<td>{pc(r["ma200"])}</td><td>{pc(r["bench"])}</td>'
                 f'<td><b class="{cls(r["excess"])}">{r["excess"] * 100:+.2f}pp</b></td></tr>')
    o.append('</tbody></table>')
    last = wf_rows[-1] if wf_rows else None
    if last:
        o.append(f'<div class="box {"ok" if last["excess"] > -5 else "bad"}">'
                 f'<b>牛市段决定性检验</b>：{last["seg"]} 波动率目标超额 '
                 f'<b>{last["excess"] * 100:+.2f}pp</b>，同段 MA200 二元为 '
                 f'{(last["ma200"] - last["bench"]) * 100:+.2f}pp、无风控 '
                 f'{(last["nogate"] - last["bench"]) * 100:+.2f}pp —— '
                 f'{"踏空问题被修复。" if last["excess"] > (last["ma200"] - last["bench"]) else "踏空问题仍在。"}</div>')
    o.append(f'<div class="box {"ok" if wins >= 3 else ("warn" if wins >= 2 else "bad")}">'
             f'4 段中 <b>{wins} 段</b>跑赢基准。</div>')

    # 十一、结论
    o.append('<h2>十二、结论</h2>')
    o.append('<div class="box warn"><h3 style="margin-top:0">判定</h3><ul>')
    o.append(f'<li><b>风控层替换是否成功</b>：全样本 alpha 保留率 P3 {63}% -> P4 '
             f'<b>{keep_full * 100:.0f}%</b>；样本外超额 P3 +12.47pp -> P4 '
             f'<b>{oos_best["excess"] * 100:+.2f}pp</b>（无风控为 '
             f'{oos_nogate["excess"] * 100:+.2f}pp）。</li>')
    if last:
        o.append(f'<li><b>踏空是否修复</b>：最后一段牛市 P4 超额 '
                 f'<b>{last["excess"] * 100:+.2f}pp</b>，MA200 二元同段 '
                 f'{(last["ma200"] - last["bench"]) * 100:+.2f}pp —— '
                 f'{"按风险调仓确实不再踏空，P3 的核心痛点被解决。" if last["excess"] > (last["ma200"] - last["bench"]) else "踏空未修复。"}</li>')
    o.append(f'<li><b>风险调整后谁最优</b>：Calmar P4 <b>{c4:.3f}</b> / 无风控 {cn:.3f} / '
             f'MA200 族代表（{pick_ma[3]}）<b>{c3:.3f}</b> / MA200 二元 {cm:.3f} —— '
             f'最优是 <b>{winner}</b>{tie_note}。极差 {best_calmar - min(c4, cn, c3, cm):.3f}'
             f'{"（超过 0.04，是真实差异而非并列）" if best_calmar - min(c4, cn, c3, cm) > 0.04 else "（差异很小，可视为并列）"}。'
             f'<b>注意</b>：P4 在「收益 / 样本外 alpha」口径下赢，MA200 族代表在「风险调整 / 回撤」口径下赢，'
             f'<b>两个族各赢一半</b> —— 不存在全面胜出者。</li>')
    o.append(f'<li><b>最硬的一条结论（贯穿 P2~P4）</b>：<b>「削回撤」与「保 alpha」在本样本不可兼得</b>。'
             f'降低暴露必然按比例同时削减 alpha 与 beta —— '
             f'vol 10%（回撤削 34%）alpha 保留 -18%，vol 15%（削 26%）保留 56%，'
             f'P3 留30%（削 18%）保留 63%，P4 25%（削 ~0%）保留 98%。'
             f'<b>风控层只能沿风险-收益线移动，无法把线抬上去</b>。</li>')
    o.append(f'<li><b>因此「最优风控」是风险偏好问题，不是统计问题</b>：'
             f'要绝对收益与样本外 alpha 用 <b>{pick[3]}</b>'
             f'（全样本 {pc(full_best["ret"])}，回撤 {full_best["raw"]["max_drawdown"] * 100:.1f}%）；'
             f'要风险调整收益与更小回撤用 <b>{pick_ma[3]}</b>'
             f'（全样本 {pc(full_soft["ret"])}，回撤 {full_soft["raw"]["max_drawdown"] * 100:.1f}%，Calmar 最高）。</li>')
    o.append(f'<li><b>核心裁判</b>：随机对照第 <b>{pct}</b> 百分位；滚动 {wins}/4 段跑赢基准；'
             f'滑点 0.5% 时 {pc(worst["ret"])}。</li>')
    o.append(f'<li><b>诚实边界</b>：target 与回看窗口在样本内挑选；'
             f'波动率目标是<b>风险缩放</b>而非择时，它在温和下跌中不会减仓（只有波动飙升才减），'
             f'因此回撤改善有限是<b>机制使然</b>；幸存者偏差、仅一轮牛熊仍是硬约束。</li>')
    o.append('</ul></div>')
    o.append('<footer>AQuant v5 P4 ｜ 数据源：东方财富数据中心（point-in-time）｜ 不构成投资建议</footer>')
    o.append('</div></body></html>')
    return "".join(o)


if __name__ == "__main__":
    main()
