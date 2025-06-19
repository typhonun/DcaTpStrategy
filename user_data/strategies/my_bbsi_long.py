from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
import logging
import numpy as np
import pandas as pd
import talib.abstract as ta

#强震荡策略15m  CAGR %300% 适合小波动，稳定的币

class OptimizedDcaLong(IStrategy):
    timeframe = '15m'
    stoploss = -0.8  # 动态止损由 ATR 控制
    can_short = False  # 禁止做空
    minimal_roi = {
        "150": 0.01,
        "140": 0.01267,
        "130": 0.01533,
        "120": 0.018,
        "110": 0.02067,
        "100": 0.02333,
        "90": 0.026,
        "80": 0.02867,
        "70": 0.03133,
        "60": 0.034,
        "50": 0.03667,
        "40": 0.03933,
        "30": 0.042,
        "20": 0.04467,
        "10": 0.04733,
        "0": 0.05
    }

    position_adjustment_enable = True  # 允许加仓
    max_safety_orders = 3  # 最多加仓 3 次
    max_active_safety_orders = 3  # 允许最多 3 个活跃加仓
    safety_order_volume_scale = 1.2  # 每次加仓数量增加 1.2 倍
    safety_order_price_deviation = 0.01  # 每次价格下降 1.5% 触发加仓
    safety_order_step_scale = 1  # 下次加仓触发价格增加 1.1 倍

    def leverage(self, pair: str, current_leverage: float = 1.0, max_leverage: float = 5.0, side: str = "long",
                 **kwargs) -> float:
        return 5

    # 强制设置 5 倍杠杆

    def custom_exit_check(self, trade, current_profit, **kwargs):
        if current_profit < 0.10:
            self._rpc.send_msg(f"⚠️ 警告：{trade.pair} 当前收益率低于10%（{current_profit * 100:.2f}%）")
        return None

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 布林带
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_lowerband'] = bollinger['lowerband']

        # 趋势 & 动量指标
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # ADX + DI
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (dataframe['close'] <= dataframe['bb_lowerband']) &  # 价格触及布林带下轨
            (dataframe['rsi'] < 30),  # # 根据行情，可以适当减小rsi31-36-41，适应极端行情
            'buy'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (dataframe['close'] > dataframe['bb_upperband']),  # 只在价格触及布林带上轨时卖出
            'sell'
        ] = 1
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time, current_rate, current_profit, **kwargs):
        if trade.nr_of_successful_exits >= self.max_safety_orders:
            return None

        dataframe_tuple = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if not dataframe_tuple or len(dataframe_tuple) < 1:
            return None

        dataframe = dataframe_tuple[0]
        if dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1]

        # 计算加仓触发价格
        safety_order_index = trade.nr_of_successful_exits
        safety_order_price = trade.open_rate * (
                1 - (self.safety_order_price_deviation * (self.safety_order_step_scale ** safety_order_index))
        )

        # 固定加仓为当前仓位的 30%
        safety_order_size = trade.amount * 0.3

        # 满足条件时加仓
        if current_rate <= safety_order_price and last_candle['rsi'] < 25:
            return safety_order_size

        return None

    def custom_stoploss(self, pair: str, trade: Trade, current_time, current_rate, current_profit, **kwargs):
        return -1
