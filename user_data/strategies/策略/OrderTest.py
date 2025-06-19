from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


class SimpleLimitEntry(IStrategy):
    # 策略基础设置
    timeframe = '3m'
    can_short = False
    can_long = True
    position_adjustment_enable = False
    minimal_roi = {"0": 0.01}
    stoploss = -0.05

    # 限价订单类型
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
    }

    # time_in_force 必须为 dict！
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc',
        'stoploss': 'gtc',
        'force_entry': 'gtc',
        'force_exit': 'gtc',
        'cancel': 'gtc',
    }

    # 下单价格侧 - Freqtrade 默认使用 bid/ask（仅在 limit 有效）
    order_price = 'ask'  # buy单挂 ask（卖一）

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = (
            (dataframe['close'] > dataframe['close'].shift(1))  # 简单上涨判断
            & (dataframe['volume'] > 0)
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = True  # 简单设定：只要进了就卖出
        return dataframe
