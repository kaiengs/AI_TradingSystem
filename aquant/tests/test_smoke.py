"""冒烟测试：验证包能正常 import、回测与信号两条路径都能跑通。

因依赖联网行情（首次会拉取并缓存），本测试默认在缓存已存在时快速通过；
若完全离线且缓存不存在，会因无法拉取数据而跳过相关用例。
"""
from __future__ import annotations

import os

import pytest

from aquant import BacktestConfig, run_backtest
from aquant import SignalConfig, target_weights


def _have_cache() -> bool:
    # 引擎缓存目录：aquant/_engine/.cache
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.isdir(os.path.join(here, "aquant", "_engine", ".cache"))


@pytest.mark.skipif(not _have_cache(), reason="无行情缓存且离线，跳过联网用例")
def test_run_backtest_returns_structured_result():
    r = run_backtest(BacktestConfig(factor="bp", tau=3.0, freq="monthly"))
    assert r.n_trades > 0
    assert abs((r.total_return - r.benchmark_return) - r.excess_return) < 1e-9
    df = r.to_frame()
    assert list(df.columns) == ["equity", "benchmark"]
    assert len(df) == len(r.dates)


@pytest.mark.skipif(not _have_cache(), reason="无行情缓存且离线，跳过联网用例")
def test_target_weights_returns_normalized_dataframe():
    w = target_weights("2026-10-08", SignalConfig(factor="bp", tau=3.0))
    if len(w) == 0:
        pytest.skip("覆盖率不足，返回空信号（合理行为）")
    assert abs(w["weight"].sum() - 1.0) < 1e-6
    assert list(w.columns) == ["symbol", "weight"]
    assert w["weight"].is_monotonic_decreasing


def test_overrides_reject_unknown_param():
    with pytest.raises(AttributeError):
        run_backtest(nonsense=1)
