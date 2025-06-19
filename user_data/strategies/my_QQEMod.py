from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib.abstract as ta


class QQEMod(IStrategy):
    timeframe = '30m'
    minimal_roi = {
        "150": 0.01,
        "140": 0.015,
        "130": 0.02,
        "120": 0.025,
        "110": 0.03,
        "100": 0.035,
        "90": 0.04,
        "80": 0.045,
        "70": 0.05,
        "60": 0.055,
        "50": 0.06,
        "40": 0.065,
        "30": 0.07,
        "20": 0.075,
        "10": 0.08,
        "0": 0.085,
    }
    stoploss = -0.9
    use_custom_exit = True
    trailing_stop = False  # 使用自定义 trailing

    def leverage(self, pair: str, current_leverage: float = 1.0, max_leverage: float = 5.0, side: str = "long",
                 **kwargs) -> float:
        return 10

    # === Hyperopt 参数 ===
    rsi_len_p = IntParameter(2, 14, default=6, space="buy", optimize=True)
    smooth_p = IntParameter(2, 10, default=5, space="buy", optimize=True)
    qqe_fac_p = DecimalParameter(1.0, 5.0, default=3.0, decimals=2, space="buy", optimize=True)
    thresh_s = DecimalParameter(1.0, 10.0, default=3.0, decimals=2, space="buy", optimize=True)

    rsi_len_s = IntParameter(2, 14, default=6, space="buy", optimize=True)
    smooth_s = IntParameter(2, 10, default=5, space="buy", optimize=True)
    qqe_fac_s = DecimalParameter(1.0, 3.0, default=1.61, decimals=2, space="buy", optimize=True)

    bb_len = IntParameter(20, 100, default=50, space="buy", optimize=True)
    bb_mult = DecimalParameter(0.1, 1.0, default=0.35, decimals=2, space="buy", optimize=True)

    def _calc_qqe(self, rsi_series: pd.Series, length: int, smooth_len: int, factor: float):
        wild_len = length * 2 - 1
        sm_rsi_arr = ta.EMA(rsi_series.values, timeperiod=smooth_len)
        sm_rsi = pd.Series(sm_rsi_arr, index=rsi_series.index)
        atr_rsi = sm_rsi.diff().abs().fillna(0)
        sm_atr_arr = ta.EMA(atr_rsi.values, timeperiod=wild_len)
        sm_atr = pd.Series(sm_atr_arr, index=rsi_series.index)
        dyn_atr = sm_atr * factor

        lb = pd.Series(np.nan, index=rsi_series.index)
        sb = pd.Series(np.nan, index=rsi_series.index)
        dirn = pd.Series(0, index=rsi_series.index)

        for i in range(len(rsi_series)):
            if i == 0:
                lb.iat[i] = sm_rsi.iat[i] - dyn_atr.iat[i]
                sb.iat[i] = sm_rsi.iat[i] + dyn_atr.iat[i]
            else:
                new_lb = sm_rsi.iat[i] - dyn_atr.iat[i]
                new_sb = sm_rsi.iat[i] + dyn_atr.iat[i]
                prev_lb = lb.iat[i-1]
                prev_sb = sb.iat[i-1]
                prev_dir = dirn.iat[i-1]
                lb.iat[i] = max(prev_lb, new_lb) if sm_rsi.iat[i-1] > prev_lb and sm_rsi.iat[i] > prev_lb else new_lb
                sb.iat[i] = min(prev_sb, new_sb) if sm_rsi.iat[i-1] < prev_sb and sm_rsi.iat[i] < prev_sb else new_sb
                dirn.iat[i] = 1 if sm_rsi.iat[i] > prev_sb else -1 if sm_rsi.iat[i] < prev_lb else prev_dir

        trend_line = pd.Series(np.where(dirn == 1, lb, sb), index=rsi_series.index)
        return trend_line, sm_rsi, dirn

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        rsi_p = pd.Series(ta.RSI(df['close'], timeperiod=int(self.rsi_len_p.value)), index=df.index)
        trend_p, sm_rsi_p, dirn_p = self._calc_qqe(
            rsi_p,
            length=int(self.rsi_len_p.value),
            smooth_len=int(self.smooth_p.value),
            factor=float(self.qqe_fac_p.value)
        )
        df['primary_trend'] = trend_p
        df['primary_rsi'] = sm_rsi_p
        df['dirn_p'] = dirn_p

        rsi_s = pd.Series(ta.RSI(df['close'], timeperiod=int(self.rsi_len_s.value)), index=df.index)
        trend_s, sm_rsi_s, dirn_s = self._calc_qqe(
            rsi_s,
            length=int(self.rsi_len_s.value),
            smooth_len=int(self.smooth_s.value),
            factor=float(self.qqe_fac_s.value)
        )
        df['secondary_trend'] = trend_s
        df['secondary_rsi'] = sm_rsi_s
        df['dirn_s'] = dirn_s

        mid = df['primary_trend'] - 50
        df['bb_mid'] = ta.SMA(mid, timeperiod=int(self.bb_len.value))
        df['bb_dev'] = float(self.bb_mult.value) * ta.STDDEV(mid, timeperiod=int(self.bb_len.value))
        df['bb_upper'] = df['bb_mid'] + df['bb_dev']
        df['bb_lower'] = df['bb_mid'] - df['bb_dev']

        df.dropna(inplace=True)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (df['secondary_rsi'] - 50 > self.thresh_s.value) &
            (df['primary_rsi'] - 50 > df['bb_upper']),
            'buy'
        ] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (df['secondary_rsi'] - 50 < -self.thresh_s.value) &
            (df['primary_rsi'] - 50 < df['bb_lower']),
            'sell'
        ] = 1
        return df

    def custom_exit(self, pair: str, trade, current_time, current_rate: float,
                    current_profit: float, **kwargs) -> str:

        dataframe: DataFrame = kwargs.get("dataframe", None)
        if dataframe is None or len(dataframe) < 2 or 'dirn_p' not in dataframe.columns:
            return None

        last_row = dataframe.iloc[-1]
        prev_row = dataframe.iloc[-2]

        # ✅ QQE趋势反转出场
        if prev_row['dirn_p'] == 1 and last_row['dirn_p'] == -1:
            return "QQE_Trend_Reversal"

        # ✅ 自定义 Trailing Stop
        if current_profit > 0.03:
            trailing_stop_pct = 0.01
            peak_profit = trade.max_profit
            if current_profit < (peak_profit - trailing_stop_pct):
                return "Trailing_Stop"

        return None
