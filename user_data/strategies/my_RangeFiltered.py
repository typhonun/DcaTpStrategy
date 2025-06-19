from freqtrade.strategy import IStrategy
from freqtrade.strategy import merge_informative_pair
from freqtrade.persistence import Trade
from pandas import DataFrame
import numpy as np
import talib.abstract as ta

class RangeFiltered(IStrategy):
    timeframe = '15m'
    startup_candle_count = 200
    stoploss = -0.10  # 或者你想要的任意止损百分比，例如 -10%

    # Kalman & Supertrend 参数
    kalman_alpha = 0.01
    kalman_beta = 0.1
    kalman_period = 77
    deviation = 1.2
    supertrend_atr_period = 7
    supertrend_factor = 0.7

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # === Kalman Filter 模拟 ===
        df['k'] = df['close'].ewm(span=self.kalman_period, adjust=False).mean()

        # === Vola for Upper/Lower Range ===
        df['vola'] = df['high'].rolling(window=200).mean() - df['low'].rolling(window=200).mean()
        df['upper'] = df['k'] + self.deviation * df['vola']
        df['lower'] = df['k'] - self.deviation * df['vola']
        df['mid'] = (df['open'] + df['close']) / 2

        # === 简化 Supertrend ===
        atr = ta.ATR(df, timeperiod=self.supertrend_atr_period)
        hl2 = (df['high'] + df['low']) / 2
        upperband = hl2 + (self.supertrend_factor * atr)
        lowerband = hl2 - (self.supertrend_factor * atr)
        supertrend = [np.nan] * len(df)
        direction = [0] * len(df)

        for i in range(1, len(df)):
            if df['close'][i] > upperband[i - 1]:
                direction[i] = 1
                supertrend[i] = lowerband[i]
            elif df['close'][i] < lowerband[i - 1]:
                direction[i] = -1
                supertrend[i] = upperband[i]
            else:
                direction[i] = direction[i - 1]
                supertrend[i] = supertrend[i - 1]

        df['direction'] = direction
        df['trend'] = np.where(df['close'] > df['upper'], 1, np.where(df['close'] < df['lower'], -1, 0))
        df['ktrend'] = np.where(df['direction'] < 0, 1, np.where(df['direction'] > 0, -1, 0))
        df['range_mode'] = df['ktrend'] * df['trend'] == -1

        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (df['range_mode']) &
            (df['close'] <= df['lower']),
            'enter_long'
        ] = 1

        df.loc[
            (df['range_mode']) &
            (df['close'] >= df['upper']),
            'enter_short'
        ] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (df['range_mode']) &
            (df['close'] >= df['mid']),
            'exit_long'
        ] = 1

        df.loc[
            (df['range_mode']) &
            (df['close'] <= df['mid']),
            'exit_short'
        ] = 1

        return df
