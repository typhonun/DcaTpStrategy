from freqtrade.strategy import IStrategy
from pandas import DataFrame
from freqtrade.persistence import Trade
import numpy as np

class LimitGrid(IStrategy):
    """
    示例策略：分批加减仓逻辑
    1) 初次进场：以 100 USDT 挂限价买单，价格 = current_rate * 1.0 （市价附近）
    2) 价格下跌到 0.995× 时，挂第二笔限价买单，加仓 50%（50 USDT）
    3) 持仓后，如果价格上涨到 1.01×，先卖出 30% 持仓
    4) 如果价格再上涨到 1.015×，卖出剩余 70% 持仓
    """

    # ———— 策略基础参数 ————
    timeframe = '5m'
    minimal_roi = {"0": 0.0}     # ROI 逻辑暂时不启用，完全靠 exit_order 来控制
    stoploss = -0.2             # 设一个较宽的止损，降低意外爆仓干扰
    trailing_stop = False

    can_short = False           # 只做多，示例只写多头部分
    use_exit_signal = False     # 统一出口由 exit_order 实现，不用 populate_exit_trend
    use_entry_signal = False    # 统一入口由 entry_order 实现，不用 populate_entry_trend

    # ———— 一些自定义参数 ————
    # 初次用 100 USDT，第二笔加仓用 50 USDT
    initial_stake_amount = 100
    add_stake_amount     = 50

    # 加仓价 = 买入均价 * 0.995；如已有持仓时触发
    add_price_ratio = 0.999
    # 减仓价分两步：第一步 1.01，第二步 1.015
    reduce_price_1_ratio = 1.001
    reduce_price_2_ratio = 1.015

    def leverage(self, pair: str, **kwargs) -> float:
        # 举例：全仓 1 倍杠杆
        return 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 这里无指标
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 所有 K 线都触发信号，让 entry_order() 自行判断是“初次进场”或“加仓”
        dataframe['enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 用 exit_order() 来挂部分减仓单
        dataframe['exit_long'] = 0
        return dataframe

    def entry_order(self, pair: str, current_rate: float, direction,
                    trade: Trade = None, **kwargs):
        """
        entry_order 会在“enter_long = 1”触发后被调用。
        当没有 trade（说明是第一次进场），挂 initial_stake_amount。
        如果已持仓，则判断是否满足“回调加仓”条件，再挂第二笔 add_stake_amount。
        """

        # 只做多：direction == 'long'
        if direction != 'long':
            return []

        orders = []

        # ———— 1. 初次进场：无持仓时挂 100 USDT 限价买单
        if (trade is None) or (not trade.is_open):
            # 买入价格：用市价附近或 current_rate，如 current_rate * 1.0
            entry_price = current_rate * 1.0
            buy_amount  = self.initial_stake_amount / entry_price
            orders.append({
                "order_type": "limit",
                "price": round(entry_price, 8),
                "amount": round(buy_amount, 8)
            })
            return orders

        # 如果已经有持仓，则 trade.amount 表示当前持有的基础币数量，trade.open_rate 表示买入均价
        if trade.is_open:
            avg_price   = float(trade.open_rate)     # 假设这是已持仓的买入成本价
            current_qty = float(trade.amount)        # 已持仓的币数量

            # ———— 2. 回调加仓：如果价格 <= avg_price * 0.995，则挂第二笔 50 USDT 限价买单
            add_price = avg_price * self.add_price_ratio
            if current_rate <= add_price:
                add_amount = self.add_stake_amount / add_price
                orders.append({
                    "order_type": "limit",
                    "price": round(add_price, 8),
                    "amount": round(add_amount, 8)
                })
                return orders

        return orders  # 其他情况下不加仓

    def exit_order(self, pair: str, current_rate: float, direction,
                   trade: Trade = None, **kwargs):
        """
        exit_order 会在“满足 exit 条件”后被调用，但我们这里用返回空信号，
        而在 entry 和 exit 里手动判断时机并挂单，所以只要 trade 存在且 is_open，就去检查“部分减仓”触发条件。
        """

        orders = []
        if direction != 'long' or trade is None or not trade.is_open:
            return orders

        avg_price   = float(trade.open_rate)
        current_qty = float(trade.amount)

        # ———— 3. 第一阶段减仓：价格 >= avg_price * 1.01 时，卖出 30% 持仓
        tp1_price = avg_price * self.reduce_price_1_ratio
        tp1_amount = current_qty * 0.3
        # 如果当前价格触及 TP1 且尚未挂 TP1 卖单（我们需要控制不会重复挂单）
        # 这里用 custom_info 存一个状态，标记“TP1 已经挂过”
        if not self.custom_info.get(f"{pair}_tp1_done", False):
            if current_rate >= tp1_price:
                orders.append({
                    "order_type": "limit",
                    "price": round(tp1_price, 8),
                    "amount": round(tp1_amount, 8),
                })
                # 标记第一阶段已挂单过，不要重复挂
                self.custom_info[f"{pair}_tp1_done"] = True
                return orders

        # ———— 4. 第二阶段减仓：价格 >= avg_price * 1.015 时，卖出剩余 70%
        tp2_price = avg_price * self.reduce_price_2_ratio
        # 如果第一阶段已经执行过，且第二阶段尚未执行
        if self.custom_info.get(f"{pair}_tp1_done", False) and not self.custom_info.get(f"{pair}_tp2_done", False):
            if current_rate >= tp2_price:
                # 由于 TP1 已经卖掉 30%，剩余 70% 在 trade.amount 里可能已经被框架更新
                # 为保险，可重新fetch当前持仓数量：这里先假设 current_qty 是最新数据
                # 也可以通过 self.wallets 或者 ccxt_api 去 fetch_balance() 精确计算
                remaining_qty = current_qty  # 注意：如果 TP1 在同一个 K 线内成交，trade.amount 会立即减
                orders.append({
                    "order_type": "limit",
                    "price": round(tp2_price, 8),
                    "amount": round(remaining_qty, 8),
                })
                self.custom_info[f"{pair}_tp2_done"] = True
                return orders

        return orders
