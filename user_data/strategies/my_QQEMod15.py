from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib.abstract as ta


class QQEMod15(IStrategy):
    timeframe = '15m'
    stoploss = -0.5
    use_custom_exit = True
    trailing_stop = False

    minimal_roi = {
        "60": 0.03,
        "30": 0.05,
        "0": 0.08,
    }
    def leverage(self, pair: str, current_leverage: float = 1.0, max_leverage: float = 5.0, side: str = "long",
                 **kwargs) -> float:
        return 10

    rsi_len_p = IntParameter(6, 10, default=8, space="buy", optimize=True)
    smooth_p = IntParameter(4, 8, default=5, space="buy", optimize=True)
    qqe_fac_p = DecimalParameter(1.0, 1.8, default=1.3, decimals=2, space="buy", optimize=True)

    rsi_len_s = IntParameter(4, 8, default=6, space="buy", optimize=True)
    smooth_s = IntParameter(2, 6, default=4, space="buy", optimize=True)
    qqe_fac_s = DecimalParameter(0.8, 1.5, default=1.0, decimals=2, space="buy", optimize=True)

    thresh_s = DecimalParameter(1.5, 4.0, default=2.5, decimals=2, space="buy", optimize=True)

    bb_len = IntParameter(20, 60, default=40, space="buy", optimize=True)
    bb_mult = DecimalParameter(0.2, 0.5, default=0.3, decimals=2, space="buy", optimize=True)

    atr_len = IntParameter(10, 20, default=14, space="buy", optimize=True)
    atr_thresh = DecimalParameter(0.002, 0.008, default=0.0035, decimals=4, space="buy", optimize=True)

    def _calc_qqe(self, rsi_series: pd.Series, length: int, smooth_len: int, factor: float):
        wild_len = length * 2 - 1

        # 确保是 pd.Series
        if not isinstance(rsi_series, pd.Series):
            rsi_series = pd.Series(rsi_series)

        sm_rsi = pd.Series(ta.EMA(rsi_series, smooth_len), index=rsi_series.index)
        atr_rsi = sm_rsi.diff().abs().fillna(0)
        sm_atr = pd.Series(ta.EMA(atr_rsi, wild_len), index=rsi_series.index)
        dyn_atr = sm_atr * factor

        lb, sb, dirn = pd.Series(index=rsi_series.index), pd.Series(index=rsi_series.index), pd.Series(0,
                                                                                                       index=rsi_series.index)
        for i in range(len(rsi_series)):
            if i == 0:
                lb.iat[i] = sm_rsi.iat[i] - dyn_atr.iat[i]
                sb.iat[i] = sm_rsi.iat[i] + dyn_atr.iat[i]
            else:
                prev_lb, prev_sb, prev_dir = lb.iat[i - 1], sb.iat[i - 1], dirn.iat[i - 1]
                new_lb, new_sb = sm_rsi.iat[i] - dyn_atr.iat[i], sm_rsi.iat[i] + dyn_atr.iat[i]
                lb.iat[i] = max(prev_lb, new_lb) if sm_rsi.iat[i - 1] > prev_lb and sm_rsi.iat[i] > prev_lb else new_lb
                sb.iat[i] = min(prev_sb, new_sb) if sm_rsi.iat[i - 1] < prev_sb and sm_rsi.iat[i] < prev_sb else new_sb
                dirn.iat[i] = 1 if sm_rsi.iat[i] > prev_sb else -1 if sm_rsi.iat[i] < prev_lb else prev_dir

        trend_line = np.where(dirn == 1, lb, sb)
        return pd.Series(trend_line, index=rsi_series.index), sm_rsi, dirn

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        rsi_p = ta.RSI(df['close'], int(self.rsi_len_p.value))
        trend_p, sm_rsi_p, dirn_p = self._calc_qqe(rsi_p, int(self.rsi_len_p.value), int(self.smooth_p.value), float(self.qqe_fac_p.value))
        df['primary_trend'], df['primary_rsi'], df['dirn_p'] = trend_p, sm_rsi_p, dirn_p

        rsi_s = ta.RSI(df['close'], int(self.rsi_len_s.value))
        trend_s, sm_rsi_s, dirn_s = self._calc_qqe(rsi_s, int(self.rsi_len_s.value), int(self.smooth_s.value), float(self.qqe_fac_s.value))
        df['secondary_trend'], df['secondary_rsi'], df['dirn_s'] = trend_s, sm_rsi_s, dirn_s

        # Bollinger Band on RSI deviation
        mid = df['primary_trend'] - 50
        df['bb_mid'] = ta.SMA(mid, int(self.bb_len.value))
        df['bb_dev'] = float(self.bb_mult.value) * ta.STDDEV(mid, int(self.bb_len.value))
        df['bb_upper'] = df['bb_mid'] + df['bb_dev']
        df['bb_lower'] = df['bb_mid'] - df['bb_dev']

        df['atr'] = ta.ATR(df, int(self.atr_len.value))

        df.dropna(inplace=True)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (df['secondary_rsi'] > 50 + self.thresh_s.value) &
            (df['primary_rsi'] > df['bb_upper']) &
            (df['close'] > df['close'].shift(1)),
            'buy'
        ] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (df['secondary_rsi'] < 50 - self.thresh_s.value) |
            (df['primary_rsi'] < df['bb_lower']),
            'sell'
        ] = 1
        return df

    def custom_exit(self, pair: str, trade, current_time, current_rate: float,
                    current_profit: float, **kwargs) -> str:
        dataframe: DataFrame = kwargs.get("dataframe", None)
        if dataframe is None or len(dataframe) < 2:
            return None
        last_row, prev_row = dataframe.iloc[-1], dataframe.iloc[-2]

        # 趋势反转
        if prev_row['dirn_p'] == 1 and last_row['dirn_p'] == -1:
            return "Early_QQE_Reversal"

        # 动态止盈
        peak = trade.max_profit if hasattr(trade, 'max_profit') else current_profit
        strength = abs(last_row['primary_rsi'] - 50)
        dyn_roi = 0.015 + min(strength / 100, 0.05)
        dyn_trail = 0.004 + min(strength / 100, 0.015)

        if current_profit > dyn_roi and current_profit < (peak - dyn_trail):
            return f"Trailing_Exit_{dyn_trail:.3f}"

        return None
