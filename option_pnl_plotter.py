import numpy as np
import matplotlib.pyplot as plt


def calculate_option_pnl(option_type, strike_price, premium, direction, quantity, underlying_prices, spot_price=0):
    """
    计算期权或现货的到期损益
    
    参数:
    option_type: 期权类型 ('call', 'put' 或 'spot')
    strike_price: 行权价（现货交易时可为0）
    premium: 权利金（现货交易时为0）
    direction: 方向 ('buy' 或 'sell')
    quantity: 数量
    underlying_prices: 标的资产价格列表
    spot_price: 现货价格（用于现货交易，默认为0）
    
    返回:
    到期损益列表
    """
    pnl = []
    
    # 确定符号：买入为1，卖出为-1
    sign = 1 if direction.lower() == 'buy' else -1 if direction.lower() == 'sell' else None
    if sign is None:
        raise ValueError("方向必须是 'buy' 或 'sell'")
    
    for price in underlying_prices:
        if option_type.lower() == 'call':
            # 看涨期权到期损益
            # 买入：max(标的资产价格 - 行权价, 0) - 权利金
            # 卖出：权利金 - max(标的资产价格 - 行权价, 0)
            payoff = sign * (max(price - strike_price, 0) - premium)
        elif option_type.lower() == 'put':
            # 看跌期权到期损益
            # 买入：max(行权价 - 标的资产价格, 0) - 权利金
            # 卖出：权利金 - max(行权价 - 标的资产价格, 0)
            payoff = sign * (max(strike_price - price, 0) - premium)
        elif option_type.lower() == 'spot':
            # 现货交易损益
            # 买入：标的资产价格 - 现货价格
            # 卖出：现货价格 - 标的资产价格
            payoff = sign * (price - spot_price)
        else:
            raise ValueError("期权类型必须是 'call', 'put' 或 'spot'")
        
        # 应用数量参数
        payoff = payoff * quantity
        pnl.append(payoff)
    
    return pnl


def find_break_even_points(underlying_prices, pnl):
    """
    找到损益曲线与x轴的交点（盈亏平衡点）
    
    参数:
    underlying_prices: 标的资产价格列表
    pnl: 损益列表
    
    返回:
    交点的坐标列表
    """
    break_even_points = []
    
    for i in range(len(pnl) - 1):
        # 检查是否穿过x轴
        if pnl[i] * pnl[i+1] < 0:  # 符号不同，说明穿过x轴
            # 线性插值计算精确的交点
            x1, x2 = underlying_prices[i], underlying_prices[i+1]
            y1, y2 = pnl[i], pnl[i+1]
            
            # 线性插值公式: x = x1 + (0-y1) * (x2-x1) / (y2-y1)
            x_cross = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
            break_even_points.append((x_cross, 0))
    return break_even_points


def find_turning_points(underlying_prices, pnl):
    """
    找到损益曲线的拐点（行权价处）
    
    参数:
    underlying_prices: 标的资产价格列表
    pnl: 损益列表
    
    返回:
    拐点的坐标列表
    """
    turning_points = []
    
    # 对于期权损益曲线，拐点通常出现在行权价处
    # 这里我们简化处理，认为曲线在行权价处有拐点
    # 在实际实现中，可以通过计算二阶导数来找拐点
    
    # 由于我们是在计算特定期权的损益，拐点就是行权价
    # 这个函数主要是为了保持接口的一致性
    return turning_points


def calculate_combined_pnl(options, underlying_prices):
    """
    计算多个期权或现货组合的到期损益
    
    参数:
    options: 期权或现货列表，每个元素为 (option_type, strike_price, premium, direction, quantity) 或 
             (option_type, spot_price, premium, direction, quantity) 的元组
             对于现货，option_type为'spot'，spot_price为实际现货价格，premium为0
    underlying_prices: 标的资产价格列表
    
    返回:
    组合到期损益列表
    """
    # 初始化组合损益为0
    combined_pnl = np.zeros_like(underlying_prices)
    
    # 计算每个期权的损益并累加
    for option_type, strike_price, premium, direction, quantity in options:
        # 对于现货交易，strike_price实际上是spot_price
        if option_type.lower() == 'spot':
            option_pnl = calculate_option_pnl(option_type, 0, 0, direction, quantity, underlying_prices, strike_price)
        else:
            option_pnl = calculate_option_pnl(option_type, strike_price, premium, direction, quantity, underlying_prices)
        combined_pnl += np.array(option_pnl)
    
    return combined_pnl.tolist()


def plot_single_option_pnl(option_type, strike_price, premium, direction, quantity, filename=None):
    """
    绘制单个期权或现货到期损益图
    
    参数:
    option_type: 期权类型 ('call' 或 'put' 或 'spot')
    strike_price: 行权价/现价
    premium: 权利金
    direction: 方向 ('buy' 或 'sell')
    quantity: 数量
    filename: 保存图片的文件名，默认为None则自动生成
    """
    # 生成标的资产价格范围（行权价的0.1倍到2倍）
    underlying_prices = np.linspace(strike_price * 0.5, strike_price * 1.5, 400)
    
    # 计算对应的损益
    if option_type.lower() == 'spot':
        pnl = calculate_option_pnl(option_type, 0, 0, direction, quantity, underlying_prices, strike_price)
    else:
        pnl = calculate_option_pnl(option_type, strike_price, premium, direction, quantity, underlying_prices)
    
    # 找到与x轴的交点
    break_even_points = find_break_even_points(underlying_prices, pnl)
    
    # 找到拐点
    turning_points = [(strike_price, calculate_option_pnl(option_type, strike_price, premium, direction, quantity, [strike_price])[0])]
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(underlying_prices, pnl, 'b-', linewidth=2, label=f'{direction.upper()} {option_type.upper()} Option (Qty: {quantity})')
    
    # 标注与x轴的交点
    for point in break_even_points:
        plt.plot(point[0], point[1], 'ro', markersize=5, label=f'Break-even ({point[0]:.4f}, {point[1]:.4f})')
    
    # 标注拐点
    for point in turning_points:
        plt.plot(point[0], point[1], 'go', markersize=5, label=f'Turning Point ({point[0]:.4f}, {point[1]:.4f})')
    
    # 标注行权价
    plt.axvline(x=strike_price, color='r', linestyle='--', alpha=0.7, label=f'Strike Price ({strike_price})')
    
    # 标注盈亏平衡点
    if direction.lower() == 'buy':
        if option_type.lower() == 'call':
            breakeven = strike_price + premium
        else:  # put
            breakeven = strike_price - premium
    else:  # sell
        # 对于卖出期权，盈亏平衡点计算相同，但图形表现不同
        if option_type.lower() == 'call':
            breakeven = strike_price + premium
        else:  # put
            breakeven = strike_price - premium
    
    plt.axvline(x=breakeven, color='c', linestyle='-.', alpha=0.7, label=f'Theoretical BE ({breakeven:.4f})')
    
    # 添加零线
    plt.axhline(y=0, color='k', linewidth=0.5)
    # 移除默认的y轴线
    # plt.axvline(x=0, color='k', linewidth=0.5)
    
    # 图表美化
    plt.grid(True, alpha=0.3)
    plt.xlabel('Underlying Price')
    plt.ylabel('Profit/Loss')
    plt.title(f'{direction.upper()} {option_type.upper()} Option PnL\n(Strike: {strike_price}, Premium: {premium}, Quantity: {quantity})')
    
    # 避免重复标签
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # 调整y轴显示范围，使图形更靠近x轴
    y_max = max(abs(max(pnl)), abs(min(pnl)))
    y_margin = y_max * 0.1  # 添加10%的边距
    plt.ylim(-y_max - y_margin, y_max + y_margin)
    
    # 将y轴移动到图形最左边
    ax = plt.gca()
    ax.spines['left'].set_position(('data', underlying_prices[0]))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data', 0))
    
    # 保存图表而不是显示
    plt.tight_layout()
    
    # 如果没有指定文件名，则根据期权类型和参数生成文件名
    if filename is None:
        filename = f'{direction.lower()}_{option_type.lower()}_option_pnl_k{strike_price}_p{premium}_q{quantity}.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"图片已保存为: {filename}")
    plt.close()  # 关闭图形以释放内存


def plot_combined_pnl(options, filename=None, show_individual_legs=True):
    """
    绘制多个期权组合的到期损益图
    
    参数:
    options: 期权列表，每个元素为 (option_type, strike_price, premium, direction, quantity) 的元组
    filename: 保存图片的文件名，默认为None则自动生成
    show_individual_legs: 是否显示单腿曲线，默认为True
    """
    # 确定价格范围（基于所有期权的行权价）
    all_strikes = [strike for _, strike, _, _, _ in options]
    min_strike = min(all_strikes)
    max_strike = max(all_strikes)
    
    # 生成标的资产价格范围
    underlying_prices = np.linspace(min_strike * 0.5, max_strike * 1.5, 400)
    
    # 计算组合损益
    combined_pnl = calculate_combined_pnl(options, underlying_prices)
    
    # 找到组合损益与x轴的交点
    combined_break_even_points = find_break_even_points(underlying_prices, combined_pnl)
    
    # 找到组合损益的拐点（行权价处）
    combined_turning_points = []
    for type, strike_price, _, _, _ in options:
        if type == 'spot':
            continue
        # 计算在行权价处的组合损益值
        strike_pnl = calculate_combined_pnl(options, [strike_price])
        combined_turning_points.append((strike_price, strike_pnl[0]))
    
    # 去除重复的拐点
    unique_turning_points = list(set(combined_turning_points))
    
    # 绘图
    plt.figure(figsize=(12, 7))
    
    # 绘制组合损益
    plt.plot(underlying_prices, combined_pnl, 'b-', linewidth=2.5, label='Combined PnL')
    
    # 标注组合损益与x轴的交点
    for point in combined_break_even_points:
        plt.plot(point[0], point[1], 'ro', markersize=5, label=f'Combined BE ({point[0]:.4f}, {point[1]:.4f})')
    
    # 标注组合损益的拐点
    for point in unique_turning_points:
        plt.plot(point[0], point[1], 'go', markersize=5, label=f'Combined TP ({point[0]:.4f}, {point[1]:.4f})')
    
    # 如果需要显示单腿曲线
    if show_individual_legs:
        # 绘制每个单独期权的损益
        for i, (option_type, strike_price, premium, direction, quantity) in enumerate(options):
            if option_type == 'spot':
                option_pnl = calculate_option_pnl(option_type, 0, premium, direction, quantity, underlying_prices, strike_price)
            else:
                option_pnl = calculate_option_pnl(option_type, strike_price, premium, direction, quantity, underlying_prices)
            plt.plot(underlying_prices, option_pnl, '--', alpha=0.7, 
                    label=f'Leg {i+1}: {direction.upper()} {option_type.upper()} (K={strike_price}, P={premium}, Q={quantity})')
    
    # 添加零线
    plt.axhline(y=0, color='k', linewidth=0.5)
    # 移除默认的y轴线
    # plt.axvline(x=0, color='k', linewidth=0.5)
    
    # 标注各个期权的行权价
    for i, (option_type, strike_price, premium, direction, quantity) in enumerate(options):
        plt.axvline(x=strike_price, linestyle=':', alpha=0.6)
    
    # 图表美化
    plt.grid(True, alpha=0.3)
    plt.xlabel('Underlying Price')
    plt.ylabel('Profit/Loss')
    
    # 构建标题
    title = "Options Strategy PnL\n"
    for i, (option_type, strike_price, premium, direction, quantity) in enumerate(options):
        title += f"Leg {i+1}: {direction.upper()} {option_type.upper()} K={strike_price} P={premium} Q={quantity}\n"
    
    plt.title(title)
    
    # 避免重复标签
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # 调整y轴显示范围，使图形更靠近x轴
    y_max = max(abs(max(combined_pnl)), abs(min(combined_pnl)))
    y_margin = y_max * 0.1  # 添加10%的边距
    plt.ylim(-y_max - y_margin, y_max + y_margin)
    
    # 将y轴移动到图形最左边
    ax = plt.gca()
    ax.spines['left'].set_position(('data', underlying_prices[0]))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data', 0))
    
    # 保存图表而不是显示
    plt.tight_layout()
    
    # 如果没有指定文件名，则根据期权参数生成文件名
    if filename is None:
        filename = 'combined_option_pnl.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"组合图片已保存为: {filename}")
    plt.close()  # 关闭图形以释放内存


def plot_combined_pnl2(options1, options2, labels=None, filename=None):
    """
    在同一张图上绘制两个期权组合的到期损益图
    
    参数:
    options1: 第一个期权组合列表，每个元素为 (option_type, strike_price, premium, direction, quantity) 的元组
    options2: 第二个期权组合列表，每个元素为 (option_type, strike_price, premium, direction, quantity) 的元组
    labels: 包含两个组合标签的列表，默认为None
    filename: 保存图片的文件名，默认为None则自动生成
    """
    # 确定价格范围（基于所有期权的行权价）
    all_strikes = [strike for _, strike, _, _, _ in options1] + [strike for _, strike, _, _, _ in options2]
    min_strike = min(all_strikes)
    max_strike = max(all_strikes)
    
    # 生成标的资产价格范围
    underlying_prices = np.linspace(min_strike * 0.7, max_strike * 1.2, 400)
    
    # 计算两个组合的损益
    combined_pnl1 = calculate_combined_pnl(options1, underlying_prices)
    combined_pnl2 = calculate_combined_pnl(options2, underlying_prices)
    
    # 获取标签
    label1 = labels[0] if labels and len(labels) > 0 else 'Strategy 1'
    label2 = labels[1] if labels and len(labels) > 1 else 'Strategy 2'
    
    # 计算两个组合的盈亏平衡点
    break_even_points1 = find_break_even_points(underlying_prices, combined_pnl1)
    break_even_points2 = find_break_even_points(underlying_prices, combined_pnl2)
    
    # 计算两个组合的拐点（行权价处）
    turning_points1 = []
    for type, strike_price, _, _, _ in options1:
        if type == 'spot':
            continue
        # 计算在行权价处的组合损益值
        strike_pnl = calculate_combined_pnl(options1, [strike_price])
        turning_points1.append((strike_price, strike_pnl[0]))
    
    # 去除重复的拐点
    unique_turning_points1 = list(set(turning_points1))
    
    turning_points2 = []
    for type, strike_price, _, _, _ in options2:
        if type == 'spot':
            continue
        # 计算在行权价处的组合损益值
        strike_pnl = calculate_combined_pnl(options2, [strike_price])
        turning_points2.append((strike_price, strike_pnl[0]))
    
    # 去除重复的拐点
    unique_turning_points2 = list(set(turning_points2))
    
    # 绘图
    plt.figure(figsize=(12, 7))
    
    # 绘制两个组合的损益
    line1, = plt.plot(underlying_prices, combined_pnl1, 'b-', linewidth=2.5, label=label1)
    line2, = plt.plot(underlying_prices, combined_pnl2, 'r-', linewidth=2.5, label=label2)
    
    # 准备图例标签
    legend_labels = []
    legend_handles = []
    
    # 添加原始线条到图例
    legend_handles.append(line1)
    legend_labels.append(label1)
    legend_handles.append(line2)
    legend_labels.append(label2)
    
    # 标注第一个组合的盈亏平衡点
    for i, point in enumerate(break_even_points1):
        marker, = plt.plot(point[0], point[1], 'bo', markersize=6, markeredgecolor='white', markeredgewidth=0.5)
        plt.annotate(f'P{i+1}', xy=point, xytext=(-5, 5), textcoords='offset points', fontsize=9, color=line1.get_color())
        legend_handles.append(marker)
        legend_labels.append(f'{label1} BE P{i+1} ({point[0]:.4f}, {point[1]:.4f})')
    
    # 标注第二个组合的盈亏平衡点
    for i, point in enumerate(break_even_points2):
        marker, = plt.plot(point[0], point[1], 'ro', markersize=6, markeredgecolor='white', markeredgewidth=0.5)
        plt.annotate(f'P{i+1+len(break_even_points1)}', xy=point, xytext=(-5, 5), textcoords='offset points', fontsize=9, color=line2.get_color())
        legend_handles.append(marker)
        legend_labels.append(f'{label2} BE P{i+1+len(break_even_points1)} ({point[0]:.4f}, {point[1]:.4f})')
    
    # 标注第一个组合的拐点
    start_index = len(break_even_points1) + len(break_even_points2) + 1
    for i, point in enumerate(unique_turning_points1):
        marker, = plt.plot(point[0], point[1], 'b^', markersize=6, markeredgecolor='white', markeredgewidth=0.5)
        plt.annotate(f'P{i+start_index}', xy=point, xytext=(-5, 5), textcoords='offset points', fontsize=9, color=line1.get_color())
        legend_handles.append(marker)
        legend_labels.append(f'{label1} TP P{i+start_index} ({point[0]:.4f}, {point[1]:.4f})')
    
    # 标注第二个组合的拐点
    start_index = len(break_even_points1) + len(break_even_points2) + len(unique_turning_points1) + 1
    for i, point in enumerate(unique_turning_points2):
        marker, = plt.plot(point[0], point[1], 'r^', markersize=6, markeredgecolor='white', markeredgewidth=0.5)
        plt.annotate(f'P{i+start_index}', xy=point, xytext=(-5, 5), textcoords='offset points', fontsize=9, color=line2.get_color())
        legend_handles.append(marker)
        legend_labels.append(f'{label2} TP P{i+start_index} ({point[0]:.4f}, {point[1]:.4f})')
    
    # 添加零线
    plt.axhline(y=0, color='k', linewidth=0.5)
    
    # 图表美化
    plt.grid(True, alpha=0.3)
    plt.xlabel('Underlying Price', horizontalalignment='right', position=(1,0))
    plt.ylabel('Profit/Loss', verticalalignment='top', position=(0,1))
    plt.title(f'Comparison of Two Options Strategies')
    
    # 创建图例
    plt.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # 调整y轴显示范围，使图形更靠近x轴
    all_pnl_values = combined_pnl1 + combined_pnl2
    y_max = max(abs(max(all_pnl_values)), abs(min(all_pnl_values)))
    y_min = min(all_pnl_values)
    y_margin = y_max * 0.1  # 添加10%的边距
    plt.ylim(y_min - y_margin, y_max + y_margin)
    
    # 将y轴移动到图形最左边
    ax = plt.gca()
    ax.spines['left'].set_position(('data', underlying_prices[0]))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data', 0))
    
    # 保存图表而不是显示
    plt.tight_layout()
    
    # 如果没有指定文件名，则根据期权参数生成文件名
    if filename is None:
        filename = 'combined_option_pnl_comparison.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"组合比较图片已保存为: {filename}")
    plt.close()  # 关闭图形以释放内存

def plot_combined_pnl3(options_list, labels=None, filename=None):
    """
    在同一张图上绘制多个期权组合的到期损益图
    
    参数:
    options_list: 包含多个期权组合的列表，每个元素都是一个期权组合列表
                 每个期权组合列表包含多个元组 (option_type, strike_price, premium, direction, quantity)
    labels: 包含组合标签的列表，默认为None
    filename: 保存图片的文件名，默认为None则自动生成
    """
    # 确定价格范围（基于所有期权的行权价）
    all_strikes = []
    for options in options_list:
        all_strikes.extend([strike for _, strike, _, _, _ in options])
    
    min_strike = min(all_strikes)
    max_strike = max(all_strikes)
    
    # 生成标的资产价格范围
    underlying_prices = np.linspace(min_strike * 0.7, max_strike * 1.2, 400)
    
    # 计算每个组合的损益
    combined_pnls = []
    for options in options_list:
        combined_pnl = calculate_combined_pnl(options, underlying_prices)
        combined_pnls.append(combined_pnl)
    
    # 获取标签
    strategy_labels = []
    if labels:
        strategy_labels = labels[:len(options_list)]
    
    # 如果标签数量不足，补充默认标签
    while len(strategy_labels) < len(options_list):
        strategy_labels.append(f'Strategy {len(strategy_labels) + 1}')
    
    # 计算每个组合的盈亏平衡点
    break_even_points_list = []
    for combined_pnl in combined_pnls:
        break_even_points = find_break_even_points(underlying_prices, combined_pnl)
        break_even_points_list.append(break_even_points)
    
    # 计算每个组合的拐点（行权价处）
    unique_turning_points_list = []
    for options in options_list:
        turning_points = []
        for type, strike_price, _, _, _ in options:
            if type == 'spot':
                continue
            # 计算在行权价处的组合损益值
            strike_pnl = calculate_combined_pnl(options, [strike_price])
            turning_points.append((strike_price, strike_pnl[0]))
        
        # 去除重复的拐点
        unique_turning_points = list(set(turning_points))
        unique_turning_points_list.append(unique_turning_points)
    
    # 定义颜色和标记样式
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # 绘图
    plt.figure(figsize=(12, 7))
    
    # 绘制每个组合的损益
    lines = []
    for i, (combined_pnl, label) in enumerate(zip(combined_pnls, strategy_labels)):
        color = colors[i % len(colors)]
        line, = plt.plot(underlying_prices, combined_pnl, color=color, linewidth=2.5, label=label)
        lines.append(line)
    
    # 准备图例标签
    legend_labels = []
    legend_handles = []
    
    # 添加原始线条到图例
    for line, label in zip(lines, strategy_labels):
        legend_handles.append(line)
        legend_labels.append(label)
    
    # 标注每个组合的盈亏平衡点和拐点
    point_index = 1
    for i, (break_even_points, unique_turning_points, line, label) in enumerate(
            zip(break_even_points_list, unique_turning_points_list, lines, strategy_labels)):
        color = colors[i % len(colors)]
        marker_style = markers[i % len(markers)]
        
        # 标注盈亏平衡点
        for point in break_even_points:
            marker, = plt.plot(point[0], point[1], marker=marker_style, color=color, markersize=6, 
                             markeredgecolor='white', markeredgewidth=0.5)
            plt.annotate(f'P{point_index}', xy=point, xytext=(0, 5), textcoords='offset points', 
                        fontsize=9, color=color)
            legend_handles.append(marker)
            legend_labels.append(f'{label} BE P{point_index} ({point[0]:.4f}, {point[1]:.4f})')
            point_index += 1
        
        # 标注拐点
        for point in unique_turning_points:
            marker, = plt.plot(point[0], point[1], marker='^', color=color, markersize=6, 
                             markeredgecolor='white', markeredgewidth=0.5)
            plt.annotate(f'P{point_index}', xy=point, xytext=(0, 5), textcoords='offset points', 
                        fontsize=9, color=color)
            legend_handles.append(marker)
            legend_labels.append(f'{label} TP P{point_index} ({point[0]:.4f}, {point[1]:.4f})')
            point_index += 1
    
    # 添加零线
    plt.axhline(y=0, color='k', linewidth=0.5)
    
    # 图表美化
    plt.grid(True, alpha=0.3)
    plt.xlabel('Underlying Price', horizontalalignment='right', position=(1,0))
    plt.ylabel('Profit/Loss', verticalalignment='top', position=(0,1))
    plt.title(f'Comparison of Multiple Options Strategies')
    
    # 创建图例
    plt.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # 调整y轴显示范围，使图形更靠近x轴
    all_pnl_values = []
    for combined_pnl in combined_pnls:
        all_pnl_values.extend(combined_pnl)
    
    y_max = max(abs(max(all_pnl_values)), abs(min(all_pnl_values)))
    y_min = min(all_pnl_values)
    y_margin = y_max * 0.1  # 添加10%的边距
    plt.ylim(y_min - y_margin, y_max + y_margin)
    
    # 将y轴移动到图形最左边
    ax = plt.gca()
    ax.spines['left'].set_position(('data', underlying_prices[0]))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data', 0))
    
    # 保存图表而不是显示
    plt.tight_layout()
    
    # 如果没有指定文件名，则根据期权参数生成文件名
    if filename is None:
        filename = 'combined_option_pnl_comparison_multiple.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"组合比较图片已保存为: {filename}")
    plt.close()  # 关闭图形以释放内存

def SamplePlot_LongCallVsSpot():
    print("\n绘制long call和spot的比较图:")
    # ATM Call vs Spot
    # 第一个组合
    options_longcall = [
        ('call', 1.100, 0.0242, 'buy', 1),  # 买入看涨期权
    ]
    # 第二个组合
    options_spot = [
        ('spot', 1.091, 0, 'buy', 1),  # 买入现货
    ]
    # 绘制并保存组合比较图表
    plot_combined_pnl2(options_longcall, options_spot, labels=['ATMCall', 'Spot'], 
                      filename='./plot_res/atm_call_vs_spot.png')

    # ITM Call vs Spot
    options_itm_list = [
        [('call', 0.850, 0.2448, 'buy', 1)],
        [('call', 0.900, 0.1941, 'buy', 1)],
        [('call', 0.950, 0.1428, 'buy', 1)],
        [('call', 1.000, 0.0944, 'buy', 1)],
        [('call', 1.050, 0.0511, 'buy', 1)]
    ]
    # 生成标签数组，格式为 LCxxx，其中 xxx 是行权价
    labels = [f'ITM{options[0][1]:.3f}' for options in options_itm_list]
    options_itm_list.append([('spot', 1.091, 0, 'buy', 1)])
    labels.append('Spot')
    # 绘制并保存组合比较图表
    plot_combined_pnl3(options_itm_list, labels=labels, 
                      filename='./plot_res/itm_call_vs_spot.png')
    
    # OTM Call vs Spot
    options_otm_list = [
        [('call', 1.150, 0.0112, 'buy', 1)],
        [('call', 1.200, 0.0050, 'buy', 1)],
        [('call', 1.250, 0.0022, 'buy', 1)],
        [('call', 1.300, 0.0014, 'buy', 1)],
        [('call', 1.350, 0.0009, 'buy', 1)],
    ]
    # 生成标签数组，格式为 LCxxx，其中 xxx 是行权价
    labels = [f'OTM{options[0][1]:.3f}' for options in options_otm_list]
    options_otm_list.append([('spot', 1.091, 0, 'buy', 1)])
    labels.append('Spot')
    # 绘制并保存组合比较图表
    plot_combined_pnl3(options_otm_list, labels=labels, 
                      filename='./plot_res/otm_call_vs_spot.png')

def SamplePlot_LongCallVsSpread():
    print("\n绘制long call和spread的比较图:")
    # ATM Call vs spread
    # 第一个组合
    options_longcall = [
        ('call', 2.300, 0.0555, 'buy', 1),  # 买入看涨期权
    ]
    # 第二个组合
    options_spread = [
       ('call', 2.300, 0.0555, 'buy', 1),
       ('call', 2.350, 0.0361, 'sell', 1),
    ]
    # 绘制并保存组合比较图表
    plot_combined_pnl2(options_longcall, options_spread, labels=['ATMCall', 'ATMSpread'], 
                      filename='./plot_res/atm_call_vs_atm_spread.png')

    # 第二个组合
    options_spread = [
       ('call', 2.300, 0.0555, 'buy', 1),
       ('call', 2.350, 0.0361, 'sell', 1),
    ]
    options_over_spread = [
       ('call', 2.300, 0.0555, 'buy', 1),
       ('call', 2.350, 0.0361, 'sell', 1.5),
    ]
    # 绘制并保存组合比较图表
    plot_combined_pnl2(options_over_spread, options_spread, labels=['ATMCall', 'ATMOverSpread'], 
                      filename='./plot_res/atm_call_vs_atm_over_spread.png')
    
    # more otm spread
    options_otm_list = [
        [('call', 2.300, 0.0555, 'buy', 1),
       ('call', 2.350, 0.0361, 'sell', 1)],
        [('call', 2.300, 0.0555, 'buy', 1),
       ('call', 2.400, 0.0233, 'sell', 1)],
        [('call', 2.300, 0.0555, 'buy', 1),
       ('call', 2.450, 0.0142, 'sell', 1)],
        [('call', 2.300, 0.0555, 'buy', 1),
       ('call', 2.500, 0.0089, 'sell', 1)],
    ]
    # 生成标签数组，格式为 LCxxx，其中 xxx 是行权价
    labels = [f'Spread {options[0][1]:.3f}/{options[1][1]:.3f}' for options in options_otm_list]
    # 绘制并保存组合比较图表
    plot_combined_pnl3(options_otm_list, labels=labels, 
                      filename='./plot_res/more_bull_spread.png')

    # more otm spread
    options_otm_list = [
        [('call', 2.300, 0.0555, 'buy', 1),
       ('call', 2.350, 0.0361, 'sell', 1)],
        [('call', 2.300, 0.0555, 'buy', 1),
       ('call', 2.400, 0.0233, 'sell', 1)],
        [('call', 2.300, 0.0555, 'buy', 1),
       ('call', 2.450, 0.0142, 'sell', 1)],
        [('call', 2.300, 0.0555, 'buy', 1),
       ('call', 2.500, 0.0089, 'sell', 1)],
    ]
    # 生成标签数组，格式为 LCxxx，其中 xxx 是行权价
    labels = [f'Spread {options[0][1]:.3f}/{options[1][1]:.3f}' for options in options_otm_list]
    options_otm_list.append([('call', 2.300, 0.0555, 'buy', 1)])
    labels.append('ATMCall')
    # 绘制并保存组合比较图表
    plot_combined_pnl3(options_otm_list, labels=labels, 
                      filename='./plot_res/more_bull_spread_with_atm_call.png')

def SamplePlot_FixedGapSpread():
    options_otm_list = [
        [('call', 2.150, 0.1603, 'buy', 1),
       ('call', 2.200, 0.1180, 'sell', 1)],
        [('call', 2.200, 0.1180, 'buy', 1),
       ('call', 2.250, 0.0830, 'sell', 1)],
        [('call', 2.250, 0.0830, 'buy', 1),
       ('call', 2.300, 0.0555, 'sell', 1)],
        [('call', 2.300, 0.0555, 'buy', 1),
       ('call', 2.350, 0.0361, 'sell', 1)],
       [('call', 2.350, 0.0361, 'buy', 1),
       ('call', 2.400, 0.0223, 'sell', 1)],
       [('call', 2.400, 0.0223, 'buy', 1),
       ('call', 2.450, 0.0142, 'sell', 1)],
       [('call', 2.450, 0.0142, 'buy', 1),
       ('call', 2.500, 0.0089, 'sell', 1)],
    ]
    # 生成标签数组，格式为 LCxxx，其中 xxx 是行权价
    labels = [f'Spread {options[0][1]:.3f}/{options[1][1]:.3f}' for options in options_otm_list]
    # 绘制并保存组合比较图表
    plot_combined_pnl3(options_otm_list, labels=labels, 
                      filename='./plot_res/fixed_gap_bull_spread.png')

    options_otm_list = [
        [('call', 2.300, 0.0555, 'buy', 1),
       ('call', 2.350, 0.0361, 'sell', 1)],
       [('put', 2.300, 0.0565, 'buy', 1),
       ('put', 2.350, 0.0878, 'sell', 1)]
    ]
    # 生成标签数组，格式为 LCxxx，其中 xxx 是行权价
    labels = [f'{options[0][0]}-Spread {options[0][1]:.3f}/{options[1][1]:.3f}' for options in options_otm_list]
    # 绘制并保存组合比较图表
    plot_combined_pnl3(options_otm_list, labels=labels, 
                      filename='./plot_res/2.3_2.35_call_put_bull_spread.png')

    options_otm_list = [
        [('call', 2.250, 0.0830, 'buy', 1),
       ('call', 2.300, 0.0555, 'sell', 1)],
       [('put', 2.250, 0.0344, 'buy', 1),
       ('put', 2.300, 0.0565, 'sell', 1)]
    ]
    # 生成标签数组，格式为 LCxxx，其中 xxx 是行权价
    labels = [f'{options[0][0]}-Spread {options[0][1]:.3f}/{options[1][1]:.3f}' for options in options_otm_list]
    # 绘制并保存组合比较图表
    plot_combined_pnl3(options_otm_list, labels=labels, 
                      filename='./plot_res/2.25_2.3_call_put_bull_spread.png')

    options_otm_list = [
        [('call', 2.350, 0.0361, 'buy', 1),
       ('call', 2.400, 0.0223, 'sell', 1)],
       [('put', 2.350, 0.0878, 'buy', 1),
       ('put', 2.400, 0.1229, 'sell', 1)]
    ]
    # 生成标签数组，格式为 LCxxx，其中 xxx 是行权价
    labels = [f'{options[0][0]}-Spread {options[0][1]:.3f}/{options[1][1]:.3f}' for options in options_otm_list]
    # 绘制并保存组合比较图表
    plot_combined_pnl3(options_otm_list, labels=labels, 
                      filename='./plot_res/2.35_2.4_call_put_bull_spread.png')

    options_otm_list = [
        [('call', 622.0, 12.50, 'buy', 1),
       ('call', 635.0, 5.32, 'sell', 1)],
       [('put', 622.0, 10.23, 'buy', 1),
       ('put', 635.0, 16.22, 'sell', 1)]
    ]
    # 生成标签数组，格式为 LCxxx，其中 xxx 是行权价
    labels = [f'{options[0][0]}-Spread {options[0][1]:.3f}/{options[1][1]:.3f}' for options in options_otm_list]
    # 绘制并保存组合比较图表
    plot_combined_pnl3(options_otm_list, labels=labels, 
                      filename='./plot_res/spy_call_put_bull_spread.png')

def main():
    """
    主函数：使用预设参数绘制期权到期损益图
    """
    print("期权到期损益图绘制工具")
    print("=" * 30)
    
    # 示例1: 单个期权（买入看涨期权）
    print("\n绘制单个期权/现货损益图:")
    option_type = 'call'    # 期权类型: 'call' 或 'put' 或 'spot'
    strike_price = 1.100    # 行权价
    premium = 0.0242        # 权利金
    direction = 'buy'       # 方向: 'buy' 或 'sell'
    quantity = 1            # 数量
    
    print(f"期权/现货类型: {option_type}")
    print(f"行权价: {strike_price}")
    print(f"权利金: {premium}")
    print(f"方向: {direction}")
    print(f"数量: {quantity}")
    
    # 绘制并保存单个期权图表
    plot_single_option_pnl(option_type, strike_price, premium, direction, quantity, 
                          filename='./plot_res/buy_call_option_pnl.png')

    plot_single_option_pnl('spot', 1.091, 0, direction, quantity, 
                          filename='./plot_res/spot_pnl.png')
    
    # 示例2: 多个期权组合（跨式组合）
    print("\n绘制多个期权组合损益图 (显示单腿曲线):")
    # 期权组合: (option_type, strike_price, premium, direction, quantity)
    options = [
        ('call', 1.100, 0.0242, 'buy', 1),  # 买入看涨期权
        ('put', 1.100, 0.0307, 'buy', 1),    # 买入看跌期权
        ('spot', 1.091, 0 ,'buy', 1)
    ]
    
    # 打印期权信息
    for i, (opt_type, strike, prem, direc, qty) in enumerate(options):
        print(f"第{i+1}腿: {direc} {opt_type} 期权, 行权价: {strike}, 权利金: {prem}, 数量: {qty}")
    
    # 绘制并保存组合期权图表（显示单腿曲线）
    plot_combined_pnl(options, filename='./plot_res/combined_option_pnl_with_legs.png', show_individual_legs=True)
    
    # 示例3: 组合期权（不显示单腿曲线）
    print("\n绘制多个期权组合损益图 (不显示单腿曲线):")
    # 绘制并保存组合期权图表（不显示单腿曲线）
    plot_combined_pnl(options, filename='./plot_res/combined_option_pnl_without_legs.png', show_individual_legs=False)
    
    # 示例4: 卖出期权示例
    print("\n绘制卖出期权损益图:")
    option_type_sell = 'call'    # 期权类型: 'call' 或 'put'
    strike_price_sell = 1.100    # 行权价
    premium_sell = 0.0242        # 权利金
    direction_sell = 'sell'      # 方向: 'buy' 或 'sell'
    quantity_sell = 1            # 数量
    
    print(f"期权类型: {option_type_sell}")
    print(f"行权价: {strike_price_sell}")
    print(f"权利金: {premium_sell}")
    print(f"方向: {direction_sell}")
    print(f"数量: {quantity_sell}")
    
    # 绘制并保存卖出期权图表
    plot_single_option_pnl(option_type_sell, strike_price_sell, premium_sell, direction_sell, quantity_sell, 
                          filename='./plot_res/sell_call_option_pnl.png')
    
    # 示例5: 多个期权组合（不同数量）
    print("\n绘制多个期权组合损益图 (不同数量):")
    # 期权组合: (option_type, strike_price, premium, direction, quantity)
    options_diff_qty = [
        ('call', 1.100, 0.0242, 'buy', 2),  # 买入2份看涨期权
        ('put', 1.100, 0.0307, 'buy', 1)    # 买入1份看跌期权
    ]
    
    # 打印期权信息
    for i, (opt_type, strike, prem, direc, qty) in enumerate(options_diff_qty):
        print(f"第{i+1}腿: {direc} {opt_type} 期权, 行权价: {strike}, 权利金: {prem}, 数量: {qty}")
    
    # 绘制并保存组合期权图表（不同数量）
    plot_combined_pnl(options_diff_qty, filename='./plot_res/combined_option_pnl_diff_qty.png', show_individual_legs=True)
    
    # 示例6: 比较两个期权组合
    print("\n绘制两个期权组合的比较图:")
    # 第一个组合: 跨式组合
    options_straddle = [
        ('call', 1.100, 0.0242, 'buy', 1),  # 买入看涨期权
        ('put', 1.100, 0.0307, 'buy', 1)    # 买入看跌期权
    ]
    
    # 第二个组合: 宽跨式组合
    options_strangle = [
        ('call', 1.150, 0.0150, 'buy', 1),  # 买入较高行权价的看涨期权
        ('put', 1.050, 0.0200, 'buy', 1)    # 买入较低行权价的看跌期权
    ]
    
    # 打印期权信息
    print("策略1 (跨式组合):")
    for i, (opt_type, strike, prem, direc, qty) in enumerate(options_straddle):
        print(f"  第{i+1}腿: {direc} {opt_type} 期权, 行权价: {strike}, 权利金: {prem}, 数量: {qty}")
    
    print("策略2 (宽跨式组合):")
    for i, (opt_type, strike, prem, direc, qty) in enumerate(options_strangle):
        print(f"  第{i+1}腿: {direc} {opt_type} 期权, 行权价: {strike}, 权利金: {prem}, 数量: {qty}")
    
    # 绘制并保存组合比较图表
    plot_combined_pnl2(options_straddle, options_strangle, labels=['Straddle', 'Strangle'], 
                      filename='./plot_res/combined_option_pnl_comparison.png')

if __name__ == "__main__":
    # main()

    # long call vs spot
    SamplePlot_LongCallVsSpot()
    # long call vs spread
    SamplePlot_LongCallVsSpread()
    # fixed gap spread
    SamplePlot_FixedGapSpread()

    # 胜率数组
    win_rates = [74.80, 68.11, 61.88, 54.34, 47.08, 39.35, 32.30, 25.75]
    # 赔率数组
    odds = [38.0/462.0, 50.0/450.0, 150.0/350.0, 215.0/285.0, 305.0/195.0, 362.0/138.0, 417.0/83.0, 446.0/54.0]
    # 绘制胜率图表
    plt.plot(win_rates, odds, 'o-', label='Win Rate vs Odds')
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Odds')
    plt.title('Win Rate vs Odds')
    plt.legend()
    plt.grid(True)
    plt.savefig('./plot_res/win_rate.png')
    # 再画一个对数的，y轴是对数坐标
    plt.figure()
    plt.plot(win_rates, odds, 'o-', label='Win Rate vs Odds')
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Odds')
    plt.title('Win Rate vs Odds (Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('./plot_res/win_rate_log.png')
