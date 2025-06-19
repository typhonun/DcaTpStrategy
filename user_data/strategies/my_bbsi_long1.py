from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade
from pandas import DataFrame
import talib.abstract as ta
from datetime import datetime

class BollingerRsiDCA1(IStrategy):
    """
    1) 浮亏 DCA：只要价格 ≤ 开仓价 * 0.99 且 RSI < 35，就按当前持仓 30% 加仓（不限次数）。
    2) 分批止盈：当达到 minimal_roi 定义的任意阶段 ROI 时，先卖 30%，再买回剩余 70%，
       并立即将“已止盈”标志重置，让策略回到“刚开仓”状态，刷新下次 ROI 条件。
    """

    timeframe = '1m'
    minimal_roi_user_defined = {
        "190": 0.005, "180": 0.010, "170": 0.015, "160": 0.020, "150": 0.025,
        "140": 0.030, "130": 0.035, "120": 0.040, "110": 0.045, "100": 0.050,
        "90": 0.055, "80": 0.060, "70": 0.065, "60": 0.070, "50": 0.075,
        "40": 0.080, "30": 0.085, "20": 0.090, "10": 0.095, "0": 0.100
    }
    trailing_stop = False
    stoploss = -0.9
    use_exit_signal = False

    position_adjustment_enable = True
    # 不限制 DCA 次数

    def leverage(self, pair: str, **kwargs) -> float:
        return 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        upperband, middleband, lowerband = ta.BBANDS(
            dataframe['close'],
            timeperiod=20,
            nbdevup=2.0,
            nbdevdn=2.0,
            matype=0
        )
        dataframe['bb_upperband'] = upperband
        dataframe['bb_midband']   = middleband
        dataframe['bb_lowerband'] = lowerband
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe.loc[
            (dataframe['close'] < dataframe['bb_lowerband']) &
            (dataframe['rsi'] < 30),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float | None,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs
    ) -> float | tuple[float, str] | None:
        """
        1) 浮亏 DCA（不限次数）：若价格 ≤ 开仓价 * 0.99 且 RSI < 35，买入当前持仓 30%。
        2) 分批止盈：根据交易已有时长选定对应的 minimal_roi 阈值。
           第一阶段：未卖出 30% 且 current_profit ≥ 阈值 → 卖 30%。
           第二阶段：已卖出 30% 且未买回 70% → 买回剩余 70%，并立即重置标志。
        """

        # 若有未成交订单，则跳过
        if trade.has_open_orders:
            return None

        # 读取止盈两步标志
        flag1 = bool(trade.get_custom_data(key='flag_step1'))
        flag2 = bool(trade.get_custom_data(key='flag_step2'))

        # 获取最新 RSI
        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df.empty:
            return None
        last_rsi = df['rsi'].iloc[-1]

        # —— 步骤1：浮亏 DCA（不限次数） —— #
        avg_entry = float(trade.open_rate)
        if (current_rate <= avg_entry * 0.99) and (last_rsi < 35):
            add_amount = float(trade.amount) * 0.5
            buy_stake = add_amount * current_rate
            return buy_stake, "dca_add_30%"

        # —— 步骤2：分批止盈 —— #
        # 计算交易已开仓时长（分钟）
        elapsed_min = (current_time - trade.open_date_utc).total_seconds() / 60
        # 按 minimal_roi 映射选取合适阈值
        applicable_roi = 0.0
        for key, val in sorted(self.minimal_roi.items(), key=lambda x: int(x[0]), reverse=True):
            if elapsed_min >= int(key):
                applicable_roi = val
                break

        # 阶段一：尚未卖出 30%，且盈利达到阈值 → 卖 30%
        if not flag1 and (current_profit >= applicable_roi):
            sell_stake = -0.30 * trade.stake_amount
            trade.set_custom_data(key='flag_step1', value=True)
            return sell_stake, "take_profit_30%"

        # 阶段二：已卖出 30%，且尚未买回 70% → 买回剩余 70%，并重置标志
        if flag1 and not flag2:
            remaining_amount = float(trade.amount)
            buy_amount = 0.70 * remaining_amount
            buy_stake = buy_amount * current_rate
            # 重置，使策略回到初始状态，下次可再次用 “卖 30% → 买 70%”
            trade.set_custom_data(key='flag_step1', value=False)
            trade.set_custom_data(key='flag_step2', value=False)
            return buy_stake, "add_position_70%"

        return None
