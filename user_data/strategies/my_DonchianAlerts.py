from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, CategoricalParameter
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib.abstract as ta

class DonchianAlerts(IStrategy):
    """
    Freqtrade implementation of Donchian Channel Alerts R1 by JustUncleL
    - Entry: Price breaks basis (center) or channel extremes, filtered by MA trend
    - Exit: Cross MA, cross center, or bar color reversal
    """
    timeframe = '15m'
    minimal_roi = {"0": 0.1}
    stoploss = -0.1
    trailing_stop = False

    # --- Hyperopt parameters ---
    ma_len = IntParameter(5, 20, default=8, space='buy')
    don_length = IntParameter(10, 50, default=30, space='buy')
    exit_mode = CategoricalParameter([0,1,2], default=0, space='sell')  # 0=Cross MA,1=Cross Centre,2=Bar Color

    def leverage(self, pair: str, current_leverage: float = 1.0, max_leverage: float = 5.0, side: str = "long",
                 **kwargs) -> float:
        return 5

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Moving average
        df['ma'] = ta.EMA(df['close'], timeperiod=int(self.ma_len.value))
        # Donchian channel
        length = int(self.don_length.value)
        df['dc_upper'] = df['high'].rolling(window=length).max()
        df['dc_lower'] = df['low'].rolling(window=length).min()
        df['dc_basis'] = (df['dc_upper'] + df['dc_lower']) / 2
        # Price action flags
        df['break_above_basis'] = (df['close'] > df['dc_basis'].shift(1)).astype(int)
        df['break_below_basis'] = (df['close'] < df['dc_basis'].shift(1)).astype(int)
        df['break_above_chan'] = (df['high'] > df['dc_upper'].shift(1)).astype(int)
        df['break_below_chan'] = (df['low'] < df['dc_lower'].shift(1)).astype(int)
        # Count consecutive channel breaks
        df['hh_count'] = df['break_above_chan'] * (df['break_above_chan'] * df['break_above_chan'].shift(1) + 1)
        df['ll_count'] = df['break_below_chan'] * (df['break_below_chan'] * df['break_below_chan'].shift(1) + 1)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Trend filter: price vs MA
        df['ma_trend'] = np.where(df['close'] > df['ma'], 1, -1)

        # Price Action signal
        long_pa = (
            (df['close'] > df['open']) &
            ((df['break_above_basis'] == 1) | (df['hh_count'] > 1))
        )
        short_pa = (
            (df['close'] < df['open']) &
            ((df['break_below_basis'] == 1) | (df['ll_count'] > 1))
        )

        # Final entries
        df.loc[long_pa & (df['ma_trend'] == 1), 'buy'] = 1
        df.loc[short_pa & (df['ma_trend'] == -1), 'sell'] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        mode = self.exit_mode.value
        # MA cross
        exit_ma_long = df['close'] < df['ma']
        exit_ma_short = df['close'] > df['ma']
        # Centre cross
        exit_center_long = df['break_below_basis'] == 1
        exit_center_short = df['break_above_basis'] == 1
        # Bar color
        exit_bar_long = (df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1))
        exit_bar_short = (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1))

        if mode == 0:
            df.loc[exit_ma_long, 'sell'] = 1
            df.loc[exit_ma_short, 'buy']  = 1
        elif mode == 1:
            df.loc[exit_center_long, 'sell'] = 1
            df.loc[exit_center_short, 'buy'] = 1
        else:
            df.loc[exit_bar_long, 'sell'] = 1
            df.loc[exit_bar_short, 'buy']  = 1
        return df
