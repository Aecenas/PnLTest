# PnLTest

期权损益绘图工具 (Options Profit and Loss Plotter)

## 简介

这是一个用于计算和绘制期权策略到期损益图的Python工具。它可以帮助交易者可视化不同期权策略的潜在收益和风险，包括单个期权、现货以及复杂组合策略。

## 功能特性

- 计算单个期权（看涨/看跌）或现货的到期损益
- 计算多个期权组合的到期损益
- 绘制损益图，包括盈亏平衡点和拐点
- 支持多种期权策略：
  - 单个期权（买入/卖出看涨/看跌期权）
  - 现货交易
  - 组合策略（如跨式、宽跨式、价差等）
  - 策略比较
- 自动计算并标注关键点：
  - 盈亏平衡点 (Break-even points)
  - 拐点 (Turning points)
  - 行权价 (Strike prices)

## 安装

```bash
pip install numpy matplotlib
```

## 使用方法

### 基本用法

```python
import option_pnl_plotter

# 绘制单个期权损益图
option_pnl_plotter.plot_single_option_pnl(
    option_type='call',
    strike_price=100,
    premium=5,
    direction='buy',
    quantity=1,
    filename='single_option.png'
)
```

### 组合策略

```python
# 定义期权组合
options = [
    ('call', 100, 5, 'buy', 1),   # 买入行权价100的看涨期权
    ('put', 100, 3, 'buy', 1)     # 买入行权价100的看跌期权
]

# 绘制组合损益图
option_pnl_plotter.plot_combined_pnl(
    options=options,
    filename='straddle.png',
    show_individual_legs=True
)
```

### 策略比较

```python
# 定义两个不同的策略
strategy1 = [
    ('call', 100, 5, 'buy', 1),
    ('put', 100, 3, 'buy', 1)
]

strategy2 = [
    ('call', 110, 2, 'buy', 1),
    ('put', 90, 2, 'buy', 1)
]

# 比较两个策略
option_pnl_plotter.plot_combined_pnl2(
    options1=strategy1,
    options2=strategy2,
    labels=['Straddle', 'Strangle'],
    filename='strategy_comparison.png'
)
```

## 主要函数

### `calculate_option_pnl`
计算单个期权或现货的到期损益

### `calculate_combined_pnl`
计算多个期权或现货组合的到期损益

### `plot_single_option_pnl`
绘制单个期权或现货到期损益图

### `plot_combined_pnl`
绘制多个期权组合的到期损益图

### `plot_combined_pnl2`
在同一张图上绘制两个期权组合的到期损益图

### `find_break_even_points`
找到损益曲线与x轴的交点（盈亏平衡点）

## 示例

运行`main()`函数可以查看各种策略的示例图表，包括：
- 单个期权损益图
- 组合策略损益图
- 策略比较图

## 输出

工具会生成高质量的PNG图像文件，显示期权策略的损益图，包括关键点标注和图例说明。

## 许可证

MIT