# AQuant

A 股量化回测与策略研究框架的**可安装封装层**。把自包含的研究引擎（`aquant._engine.aq`，纯标准库、零第三方依赖）包成带类型注解、可返回 `pandas.DataFrame` 的干净接口。

> 设计原则：引擎层（`_engine/aq`）保持零改动、零外部依赖；所有类型注解、DataFrame 转换、配置对象都在外层封装里，便于替换与测试。

## 安装

```bash
pip install -e .          # 可编辑安装（开发用）
# 或
pip install .             # 正式安装
```

安装后在任意项目里：

```python
from aquant import run_backtest, target_weights
```

## 两种核心用法

### 1. 跑回测

```python
from aquant import run_backtest

r = run_backtest(factor="bp", tau=3.0, floor=0.0, freq="monthly")
print(r.total_return, r.benchmark_return, r.excess_pp, r.annual_turnover)

# 净值 / 基准曲线（索引为交易日）
df = r.to_frame()
```

`run_backtest` 接受 `BacktestConfig` 或任意关键字覆盖：

```python
from aquant import run_backtest, BacktestConfig

r = run_backtest(BacktestConfig(pool_top=200), factor="ep", tau=6.0)
```

### 2. 生成目标权重信号（不跑回测）

```python
from aquant import target_weights

w = target_weights("2026-10-08", factor="bp", tau=3.0)
# w: DataFrame, columns=['symbol', 'weight'], 按权重降序, sum(weight)==1.0
# w.attrs['as_of'] 为实际生效交易日
```

每天收盘后调用一次即可拿到调仓目标权重。

## 命令行

```bash
python -m aquant --backtest --factor bp --tau 3.0 --freq monthly
python -m aquant --signal 2026-10-08 --factor bp --tau 3.0
```

## 参数速查

| 参数 | 含义 | 默认 |
|---|---|---|
| `factor` | 因子名（`FACTORS` 中，P2 验证用 `bp`） | `"bp"` |
| `tau` | 倾斜锐度，越大越接近等权 | `3.0` |
| `floor` | 与等权混合比例（0=纯倾斜） | `0.0` |
| `freq` | 调仓频率 `daily/weekly/monthly/quarterly/semiannual` | `"monthly"` |
| `pool_top` | 股票池规模（随机抽样，避免前视偏差） | `300` |
| `cost_slippage` | 双边滑点 | `0.001` |

## 目录结构

```
aquant/
├── pyproject.toml
├── README.md
├── aquant/                  # 导入包
│   ├── __init__.py          # 公共 API
│   ├── py.typed
│   ├── config.py            # 全局常量
│   ├── data.py              # 行情/股票池加载
│   ├── backtest.py          # run_backtest + 结果数据类
│   ├── signal.py            # target_weights
│   ├── __main__.py          # CLI
│   └── _engine/             # vendor 进来的自包含引擎（勿改）
│       └── aq/ ...
└── tests/
    └── test_smoke.py
```

## 注意事项

- **首次运行会联网**拉取行情（东方财富 HTTPS），之后落本地 `.cache` 目录；公司内网需放行对应域名。
- 这是**研究框架**而非生产交易系统；实盘前需补涨跌停成交、分红除权、真实盘口冲击成本，以及更严格的样本外验证。
- 引擎层（`_engine/aq`）保持纯净、可独立测试；如需升级引擎，整体替换该目录即可。
