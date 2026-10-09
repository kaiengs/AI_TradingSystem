#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回测引擎：目标权重驱动，支持 A股 T+1、佣金/印花税/滑点"""

import inspect

from . import indicators as ind


class Cost:
    """A股交易成本模型"""

    def __init__(self, commission=0.00025, min_commission=5.0, stamp_tax=0.001, slippage=0.001):
        self.commission = commission      # 双边佣金率
        self.min_commission = min_commission
        self.stamp_tax = stamp_tax        # 卖出印花税
        self.slippage = slippage          # 滑点（双向各计）

    def buy(self, amount):
        return amount * self.slippage + max(amount * self.commission, self.min_commission)

    def sell(self, amount):
        return amount * self.slippage + max(amount * self.commission, self.min_commission) + amount * self.stamp_tax


class Position:
    def __init__(self):
        self.shares = 0          # 持股数（100股整数倍）
        self.cost = 0.0          # 平均成本
        self.buy_day = None      # 最近一次建仓日索引，用于 T+1
        self.realized = 0.0      # 已实现盈亏
        self.wins = 0
        self.losses = 0
        self.gross_win = 0.0
        self.gross_loss = 0.0


class Backtest:
    """
    strategy: fn(sym, hist, i, state) -> target weight (0~1 占总权益比例) 或 None(=维持)
    hist: dict of lists {'date','open','close','high','low','volume'} 截至第 i 根（含）
    """

    def __init__(self, data, strategy, cash=1_000_000.0, cost=None,
                 t_plus_1=True, lot=100, allow_leverage=False, rebalance="daily",
                 prescan=None, band=0.0, min_hold=0, market=None, mkt_ma=200,
                 max_weight=0.0, slippage_mode="fixed", fund=None, mkt_vol_n=60):
        self.data = data                  # {sym: [(date,o,c,h,l,v),...]}
        self.strategy = strategy
        self.prescan = prescan or getattr(strategy, "prescan", None)
        # ---- 基本面 accessor（point-in-time）----
        # 透传给支持第 5 个参数的 prescan（如 fund_composite）。
        # 用 inspect 探测参数个数，旧版 4 参 prescan（make_composite / rotation）不受影响。
        self.fund = fund
        self._prescan_fund = False
        if self.prescan is not None:
            try:
                pk = list(inspect.signature(self.prescan).parameters.values())
                self._prescan_fund = (len(pk) >= 5) or any(
                    p.kind == inspect.Parameter.VAR_POSITIONAL for p in pk)
            except (ValueError, TypeError):
                self._prescan_fund = False
        self.cash0 = cash
        self.cost = cost or Cost()
        self.t_plus_1 = t_plus_1
        self.lot = lot
        self.allow_leverage = allow_leverage
        self.rebalance = rebalance        # daily | weekly | monthly
        # --- 降换手三件套 ---
        self.band = band                  # 调仓带宽（相对比例）：小偏离就不动
        self.min_hold = min_hold          # 建仓后 min_hold 天内不允许减仓
        self.max_weight = max_weight      # 单标的最大权重（0=不限）
        # --- 市场状态 ---
        self.market = market              # 指数序列 {'date':[], 'close':[]}
        self.mkt_ma = mkt_ma
        self.mkt_vol_n = mkt_vol_n        # 指数已实现波动率回看窗口（组合层风控用）
        self.slippage_mode = slippage_mode  # fixed | impact（按成交额估算冲击成本）

    @staticmethod
    def _lots(v):
        """整数手数，加极小 epsilon 抵消浮点截断（否则会凭空产生 1 手的卖出）"""
        return int(v + 1e-9)

    def _align(self):
        """取所有标的的日期交集，返回统一日期轴与各标的行号映射"""
        series = {}
        for sym, bars in self.data.items():
            series[sym] = {b[0]: i for i, b in enumerate(bars)}
        common = None
        for s in series.values():
            keys = set(s.keys())
            common = keys if common is None else (common & keys)
        dates = sorted(common)
        idx = {sym: [series[sym][d] for d in dates] for sym in self.data}
        return dates, idx

    def _prepare(self, cols):
        """按策略声明的 spec 一次性算好指标全序列，循环中 O(1) 查表"""
        spec = ind.merge_spec(getattr(self.strategy, "spec", None),
                              getattr(self.prescan, "spec", None))
        cache = {}
        for sym in self.data:
            cache[sym] = ind.precompute(cols[sym], spec) if spec else {}
        return cache

    def _call_prescan(self, cols, idx, t, indc):
        """调用横截面 prescan。支持第 5 个参数 fund（point-in-time 基本面 accessor）。"""
        if self._prescan_fund:
            return self.prescan(cols, idx, t, indc, self.fund)
        return self.prescan(cols, idx, t, indc)

    def _mkt_state(self, dates):
        """把指数序列对齐到回测日期轴，产出 [{close, ma, above}]"""
        if not self.market:
            return None
        mk = {d: c for d, c in zip(self.market.get("date", []), self.market.get("close", []))}
        if not mk:
            return None
        seq, last = [], None
        for d in dates:
            if d in mk:
                last = mk[d]
            seq.append(last)
        first = next((v for v in seq if v is not None), None)
        seq = [first if v is None else v for v in seq]
        ma = ind.sma(seq, self.mkt_ma)
        vol = _rolling_vol(seq, self.mkt_vol_n)     # 指数已实现年化波动率（组合层风控用）
        out = []
        for i, c in enumerate(seq):
            m = ma[i]
            out.append({"close": c, "ma": m, "above": None if m is None else (c > m),
                        "vol": vol[i]})
        return out

    def run(self):
        dates, idx = self._align()
        if not dates:
            raise ValueError("标的历史数据无共同交易日")

        # 预计算各标的完整历史切片
        cols = {}
        for sym, bars in self.data.items():
            cols[sym] = {
                "date": [b[0] for b in bars], "open": [b[1] for b in bars],
                "close": [b[2] for b in bars], "high": [b[3] for b in bars],
                "low": [b[4] for b in bars], "volume": [b[5] for b in bars],
            }

        # ---- 一次性预计算指标 + 市场状态（大样本性能关键） ----
        indc = self._prepare(cols)
        mkt = self._mkt_state(dates)

        cash = self.cash0
        pos = {s: Position() for s in self.data}
        state = {s: {} for s in self.data}
        equity_curve, trades, last_wk = [], [], None
        ctx = {"date": None, "t": 0, "equity": 0.0, "scan": None, "ind": None, "mkt": None}

        for t, d in enumerate(dates):
            # ---- 估值 ----
            prices = {s: cols[s]["close"][idx[s][t]] for s in self.data}
            equity = cash + sum(pos[s].shares * prices[s] for s in self.data)

            # ---- 是否需要调仓 ----
            if self.rebalance in ("weekly", "monthly", "quarterly", "semiannual"):
                if self.rebalance == "weekly":
                    key = _isoweek(d)
                elif self.rebalance == "monthly":
                    key = _monthkey(d)
                elif self.rebalance == "quarterly":
                    key = _quarterkey(d)
                else:
                    key = _halfkey(d)
                if last_wk is not None and key == last_wk:
                    equity_curve.append((d, equity))
                    continue
                last_wk = key

            # ---- 横截面预扫描（轮动类策略用） ----
            scan = self._call_prescan(cols, idx, t, indc) if self.prescan else {}
            ctx["date"] = d; ctx["t"] = t; ctx["equity"] = equity; ctx["scan"] = scan
            if mkt:
                ctx["mkt"] = mkt[t]

            # ---- 计算目标权重 ----
            targets = {}
            for s in self.data:
                j = idx[s][t]
                ctx["ind"] = indc[s]
                try:
                    w = self.strategy(s, cols[s], j, state[s], ctx)
                except Exception:
                    w = None
                if w is None:
                    w = (pos[s].shares * prices[s] / equity) if equity > 0 else 0.0
                targets[s] = max(0.0, float(w))

            # ---- 单标的最大权重（风控）----
            # 没有这条时，某只票单独金叉就会拿到 100% 仓位：
            # 既把风险集中到一只票上，又制造出年换手 15000% 这种荒谬数字。
            if self.max_weight > 0:
                targets = {k: min(v, self.max_weight) for k, v in targets.items()}

            tot = sum(targets.values())
            if tot > 1.0 and not self.allow_leverage:
                targets = {k: v / tot for k, v in targets.items()}

            # ---- 降换手：调仓带宽 + 最短持有期 ----
            # band 是"相对偏离比例"：|目标-当前| < band × max(目标,当前) 就不动。
            #   用相对值而非绝对值，是因为 300 只组合里单只权重只有 0.3%，
            #   绝对带宽会把所有调仓都误判为"不值得动"，甚至让建仓永远无法发生。
            # 两条豁免：空仓->建仓、以及清仓指令，都视为必须执行（后者是风控）。
            frozen = set()
            if self.band > 0 or self.min_hold > 0:
                for s in self.data:
                    tgt = targets[s]
                    cur_w = pos[s].shares * prices[s] / equity if equity > 0 else 0.0
                    if tgt <= 1e-12 or cur_w <= 1e-12:
                        continue
                    if self.min_hold > 0 and pos[s].buy_day is not None \
                            and (t - pos[s].buy_day) < self.min_hold and tgt < cur_w:
                        frozen.add(s)
                    elif abs(tgt - cur_w) < self.band * max(tgt, cur_w):
                        frozen.add(s)

            # ---- 先卖 ----
            for s in self.data:
                if s in frozen:
                    continue
                p = prices[s]
                want_val = equity * targets[s]
                want_sh = self._lots(want_val / p / self.lot) * self.lot
                cur = pos[s].shares
                if cur > want_sh:
                    if self.t_plus_1 and pos[s].buy_day == t:
                        continue          # T+1：当日买入不可卖
                    sell_sh = cur - want_sh
                    amount = sell_sh * p
                    fee = self.cost.sell(amount)
                    pnl = (p - pos[s].cost) * sell_sh - fee
                    pos[s].realized += pnl
                    if pnl >= 0:
                        pos[s].wins += 1; pos[s].gross_win += pnl
                    else:
                        pos[s].losses += 1; pos[s].gross_loss += -pnl
                    trades.append({"date": d, "sym": s, "side": "SELL", "shares": sell_sh,
                                   "price": p, "amount": amount, "fee": fee, "pnl": pnl})
                    cash += amount - fee
                    pos[s].shares -= sell_sh
                    if pos[s].shares == 0:
                        pos[s].cost = 0.0

            # ---- 再买 ----
            for s in self.data:
                if s in frozen:
                    continue
                p = prices[s]
                want_sh = self._lots(equity * targets[s] / p / self.lot) * self.lot
                cur = pos[s].shares
                if want_sh > cur:
                    add = want_sh - cur
                    amount = add * p
                    fee = self.cost.buy(amount)
                    if amount + fee > cash:
                        add = int((cash * 0.999) / p / self.lot) * self.lot
                        amount = add * p
                        fee = self.cost.buy(amount)
                        if add <= 0:
                            continue
                    cash -= amount + fee
                    old = pos[s].shares
                    pos[s].cost = (pos[s].cost * old + amount) / (old + add) if (old + add) > 0 else p
                    pos[s].shares += add
                    pos[s].buy_day = t
                    trades.append({"date": d, "sym": s, "side": "BUY", "shares": add,
                                   "price": p, "amount": amount, "fee": fee, "pnl": 0.0})

            equity = cash + sum(pos[s].shares * prices[s] for s in self.data)
            equity_curve.append((d, equity))

        bench = self._benchmark(cols, idx, dates)
        return self._result(equity_curve, trades, pos, prices, bench)

    def _benchmark(self, cols, idx, dates):
        """等权买入持有基准"""
        syms = list(self.data.keys())
        if not syms or len(dates) < 2:
            return None
        cash = self.cash0
        shares = {}
        per = cash / len(syms)
        for s in syms:
            p = cols[s]["close"][idx[s][0]]
            sh = int(per / p / self.lot) * self.lot
            fee = self.cost.buy(sh * p)
            cash -= sh * p + fee
            shares[s] = sh
        curve = []
        for t, d in enumerate(dates):
            v = cash + sum(shares[s] * cols[s]["close"][idx[s][t]] for s in syms)
            curve.append((d, v))
        eq = [v for _, v in curve]
        mdd, dd = ind.max_drawdown(eq)
        rets = [eq[i] / eq[i - 1] - 1 for i in range(1, len(eq))]
        if rets:
            mu = sum(rets) / len(rets)
            sd = (sum((r - mu) ** 2 for r in rets) / len(rets)) ** 0.5
            sharpe = (mu * 252 - 0.02) / (sd * 252 ** 0.5) if sd > 0 else 0.0
        else:
            sharpe = 0.0
        tr = eq[-1] / self.cash0 - 1
        yrs = len(eq) / 252.0
        return {"name": f"等权买入持有({len(syms)}只)", "ret": tr,
                "ann": (1 + tr) ** (1 / yrs) - 1 if yrs > 0 else 0,
                "mdd": mdd, "sharpe": sharpe, "curve": eq, "drawdown": dd}

    def _result(self, curve, trades, pos, last_prices, bench=None):
        eq = [v for _, v in curve]
        dates = [d for d, _ in curve]
        n = len(eq)
        total_ret = eq[-1] / self.cash0 - 1.0
        years = n / 252.0 if n else 0
        ann = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0
        mdd, dd = ind.max_drawdown(eq)

        rets = [eq[i] / eq[i - 1] - 1 for i in range(1, n)]
        if rets:
            mu = sum(rets) / len(rets)
            sd = (sum((r - mu) ** 2 for r in rets) / len(rets)) ** 0.5
            sharpe = (mu * 252 - 0.02) / (sd * 252 ** 0.5) if sd > 0 else 0.0
        else:
            sharpe = 0.0

        closed = [t for t in trades if t["side"] == "SELL"]
        wins = sum(1 for t in closed if t["pnl"] > 0)
        win_rate = wins / len(closed) if closed else 0.0
        gw = sum(t["pnl"] for t in closed if t["pnl"] > 0)
        gl = -sum(t["pnl"] for t in closed if t["pnl"] < 0)
        pf = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)

        return {
            "dates": dates, "equity": eq, "drawdown": dd,
            "total_return": total_ret, "annual": ann, "max_drawdown": mdd,
            "sharpe": sharpe, "win_rate": win_rate, "profit_factor": pf,
            "trades": trades, "n_trades": len(trades), "n_closed": len(closed),
            "final_positions": {s: {"shares": p.shares, "cost": p.cost,
                                    "value": p.shares * last_prices.get(s, 0)}
                                for s, p in pos.items() if p.shares > 0},
            "benchmark": bench,
        }


def _rolling_vol(seq, n):
    """指数**日收益**的滚动年化波动率（用于组合层波动率目标风控）。

    注意与逐票 vol_target 的区别：这里算的是**指数**的已实现波动，
    产出一个**全组合共用**的标量，不改变各票相对权重 —— 因此与选股层正交。
    前 n 根不足则记 None（此时风控层退化为满仓）。
    """
    if n is None or n <= 1:
        return [None] * len(seq)
    rets = [None]
    for i in range(1, len(seq)):
        a, b = seq[i - 1], seq[i]
        rets.append((b / a - 1.0) if (a and b and a > 0) else None)
    out = [None] * len(seq)
    for i in range(len(seq)):
        lo = i - n + 1
        if lo < 1:
            continue
        win = rets[lo:i + 1]
        if any(x is None for x in win):
            continue
        m = sum(win) / len(win)
        var = sum((x - m) ** 2 for x in win) / len(win)
        out[i] = (var ** 0.5) * (252 ** 0.5)
    return out


def _isoweek(datestr):
    import datetime
    y, m, dd = [int(x) for x in datestr.split("-")]
    return datetime.date(y, m, dd).isocalendar()[:2]


def _monthkey(datestr):
    y, m, _ = datestr.split("-")
    return (int(y), int(m))


def _quarterkey(datestr):
    y, m, _ = datestr.split("-")
    return (int(y), (int(m) - 1) // 3)


def _halfkey(datestr):
    y, m, _ = datestr.split("-")
    return (int(y), (int(m) - 1) // 6)
