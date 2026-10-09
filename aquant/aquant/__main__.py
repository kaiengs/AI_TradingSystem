"""命令行入口：python -m aquant [--signal DATE | --backtest]"""
from __future__ import annotations

import argparse

from aquant import run_backtest, target_weights


def _main() -> None:
    ap = argparse.ArgumentParser(prog="aquant", description="AQuant 量化框架 CLI")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--backtest", action="store_true", help="跑一次回测（默认）")
    mode.add_argument("--signal", metavar="DATE", default=None,
                      help="给一个日期则只输出该日目标权重")
    ap.add_argument("--factor", default="bp")
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--floor", type=float, default=0.0)
    ap.add_argument("--freq", default="monthly")
    ap.add_argument("--pool", type=int, default=300)
    args = ap.parse_args()

    if args.signal:
        w = target_weights(args.signal, factor=args.factor,
                           tau=args.tau, floor=args.floor, pool_top=args.pool)
        as_of = w.attrs.get("as_of", args.signal)
        cov = w.attrs.get("coverage", len(w))
        print(f"# 目标权重 {as_of} | factor={args.factor} tau={args.tau} "
              f"floor={args.floor} | 覆盖 {cov} 只")
        if len(w) == 0:
            print("  (覆盖率不足，返回空信号)")
            return
        for _, row in w.head(30).iterrows():
            print(f"  {row['symbol']}  {row['weight'] * 100:6.3f}%")
    else:
        r = run_backtest(factor=args.factor, tau=args.tau,
                         floor=args.floor, freq=args.freq, pool_top=args.pool)
        print(f"# 回测 factor={args.factor} tau={args.tau} "
              f"floor={args.floor} freq={args.freq}")
        print(f"  总收益   {r.total_return * 100:+.2f}%")
        print(f"  基准     {r.benchmark_return * 100:+.2f}%")
        print(f"  超额     {r.excess_pp:+.2f}pp")
        print(f"  年换手   {r.annual_turnover * 100:.0f}%")
        print(f"  成交笔数 {r.n_trades}")


if __name__ == "__main__":
    _main()
