from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
import pandas as pd
import talib.abstract as ta

class Partial(IStrategy):
    timeframe = '1m'
    stoploss = -0.1       # 如跌 10% 再全部止损（可不启用）
    can_short = False

    # 盈利达到 8% 时触发 exit_order()
    minimal_roi = {
        "0": 0.05
    }

    def leverage(self, pair: str, current_leverage: float = 1.0, max_leverage: float = 5.0, side: str = "long",
                 **kwargs) -> float:
        return 10

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        boll = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_lower'] = boll['lowerband']
        dataframe['bb_upper'] = boll['upperband']
        dataframe['rsi']      = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 简单示：触及布林下轨且 RSI<30 时做多
        dataframe.loc[
            (dataframe['close'] <= dataframe['bb_lower']) &
            (dataframe['rsi'] < 30), 'buy'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 由 minimal_roi 触发，不写传统 sell 信号
        dataframe['sell'] = 0
        return dataframe

    def exit_order(self, pair: str, current_rate: float, direction: str,
                   trade: Trade = None, current_profit: float = 0.0, **kwargs):
        """
        当盈利 ≥ 8% 时，只卖出 30% 仓位
        """
        orders = []
        if direction != 'long' or trade is None or not trade.is_open:
            return orders

        # 仅执行一次“卖 30%”，用 custom_info 做标记
        flag = f"{pair}_partial_done"
        if current_profit >= 0.08 and not self.custom_info.get(flag, False):
            current_qty = float(trade.amount)
            qty_to_sell = round(current_qty * 0.3, 8)
            if qty_to_sell > 0:
                orders.append({
                    "order_type": "limit",
                    "price": round(current_rate, 8),
                    "amount": qty_to_sell,
                })
                self.custom_info[flag] = True
                return orders

        return []

    def on_trade_closed(self, trade: Trade, **kwargs):
        # 当 trade 完全退出（触发止损或手动平）时，清除标记
        self.custom_info.pop(f"{trade.pair}_partial_done", None)
