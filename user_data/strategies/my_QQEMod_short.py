from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib.abstract as ta


class QQEModShort(IStrategy):
    """
    QQE‑Mod Short Strategy (纯做空) on 30m
    - QQE 主趋势 + 副趋势做空信号
    - 多重过滤避免在底部或震荡区做空
    - 动态止盈 + 趋势反转智能止盈
    """
    timeframe = '30m'
    can_short = True

    minimal_roi = {
        "150": 0.01, "140": 0.015, "130": 0.02, "120": 0.025,
        "110": 0.03, "100": 0.035, "90": 0.04, "80": 0.045,
        "70": 0.05, "60": 0.055, "50": 0.06, "40": 0.065,
        "30": 0.07, "20": 0.075, "10": 0.08, "0": 0.085,
    }

    stoploss = -0.2
    use_custom_exit = True
    trailing_stop = False

    def leverage(self, pair: str, current_leverage: float = 1.0,
                 max_leverage: float = 5.0, side: str = "short", **kwargs) -> float:
        return 10

    # === Hyperopt 参数 ===
    rsi_len_p = IntParameter(10, 14, default=12, space="buy")
    smooth_p = IntParameter(6, 10, default=8, space="buy")
    qqe_fac_p = DecimalParameter(1.2, 2.2, default=1.6, decimals=2, space="buy")

    rsi_len_s = IntParameter(8, 12, default=10, space="buy")
    smooth_s = IntParameter(4, 8, default=6, space="buy")
    qqe_fac_s = DecimalParameter(1.0, 1.8, default=1.3, decimals=2, space="buy")

    thresh_s = DecimalParameter(2.0, 6.0, default=3.5, decimals=2, space="buy")

    bb_len = IntParameter(40, 80, default=60, space="buy")
    bb_mult = DecimalParameter(0.2, 0.6, default=0.35, decimals=2, space="buy")

    atr_len = IntParameter(10, 25, default=14, space="buy")
    atr_thresh = DecimalParameter(0.002, 0.01, default=0.0045, decimals=4, space="buy")

    def _calc_qqe(self, rsi: pd.Series, length: int, smooth_len: int, factor: float):
        wild_len = length * 2 - 1
        rsi = pd.Series(rsi)
        sm_rsi = pd.Series(ta.EMA(rsi, timeperiod=smooth_len), index=rsi.index)
        atr_rsi = sm_rsi.diff().abs().fillna(0)
        sm_atr = ta.EMA(atr_rsi, timeperiod=wild_len)
        dyn_atr = sm_atr * factor

        lb, sb, dirn = rsi.copy() * np.nan, rsi.copy() * np.nan, pd.Series(0, index=rsi.index)
        for i in range(1, len(rsi)):
            new_lb = sm_rsi[i] - dyn_atr[i]
            new_sb = sm_rsi[i] + dyn_atr[i]
            lb[i] = max(lb[i - 1], new_lb) if sm_rsi[i - 1] > lb[i - 1] and sm_rsi[i] > lb[i - 1] else new_lb
            sb[i] = min(sb[i - 1], new_sb) if sm_rsi[i - 1] < sb[i - 1] and sm_rsi[i] < sb[i - 1] else new_sb
            dirn[i] = 1 if sm_rsi[i] > sb[i - 1] else -1 if sm_rsi[i] < lb[i - 1] else dirn[i - 1]

        trend_line = np.where(dirn == 1, lb, sb)
        return pd.Series(trend_line, index=rsi.index), sm_rsi, dirn

    def is_consolidating(self, df: DataFrame) -> pd.Series:
        bb_width = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        bb_width_mean = bb_width.rolling(20).mean()
        bb_narrow = bb_width < bb_width_mean * 0.8
        rsi_cons = df['primary_rsi'].between(45, 55)
        atr_low = df['atr'] < (df['close'] * self.atr_thresh.value)
        adx = ta.ADX(df)
        not_trending = adx < 20
        return bb_narrow & rsi_cons & atr_low & not_trending

    def is_potential_bottom(self, df: DataFrame) -> pd.Series:
        rsi_div = (df['close'] < df['close'].shift(1)) & (df['primary_rsi'] > df['primary_rsi'].shift(1))
        vol_low = df['volume'] < df['volume'].rolling(20).mean() * 0.8
        adx_falling = ta.ADX(df) < ta.ADX(df).shift(1)
        return rsi_div & vol_low & adx_falling

    def is_bottom_reversal(self, df: DataFrame) -> pd.Series:
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        body = (df['open'] - df['close']).abs()
        ratio = lower_shadow / (body + 1e-9)
        wick = ratio > 1.5
        vol_low = df['volume'] < df['volume'].rolling(10).mean() * 0.8
        return wick & vol_low

    def is_near_previous_low(self, df: DataFrame) -> pd.Series:
        prior_low = df['close'].rolling(20).min().shift(1)
        return df['close'] <= prior_low * 1.02

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        rsi_p = ta.RSI(df['close'], timeperiod=self.rsi_len_p.value)
        t_p, s_p, d_p = self._calc_qqe(rsi_p, self.rsi_len_p.value, self.smooth_p.value, self.qqe_fac_p.value)
        df['primary_trend'], df['primary_rsi'], df['dirn_p'] = t_p, s_p, d_p

        rsi_s = ta.RSI(df['close'], timeperiod=self.rsi_len_s.value)
        t_s, s_s, d_s = self._calc_qqe(rsi_s, self.rsi_len_s.value, self.smooth_s.value, self.qqe_fac_s.value)
        df['secondary_trend'], df['secondary_rsi'], df['dirn_s'] = t_s, s_s, d_s

        mid = df['primary_trend'] - 50
        df['bb_mid'] = ta.SMA(mid, timeperiod=self.bb_len.value)
        df['bb_dev'] = self.bb_mult.value * ta.STDDEV(mid, timeperiod=self.bb_len.value)
        df['bb_upper'] = df['bb_mid'] + df['bb_dev']
        df['bb_lower'] = df['bb_mid'] - df['bb_dev']
        df['atr'] = ta.ATR(df, timeperiod=self.atr_len.value)

        # 结构信号
        df['is_consolidating'] = self.is_consolidating(df)
        df['is_potential_bottom'] = self.is_potential_bottom(df)
        df['is_bottom_reversal'] = self.is_bottom_reversal(df)
        df['near_previous_low'] = self.is_near_previous_low(df)

        return df.dropna()

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (~df['is_consolidating']) &
            (~df['is_potential_bottom']) &
            (~df['is_bottom_reversal']) &
            (df['secondary_rsi'] - 50 < -self.thresh_s.value) &
            (df['primary_rsi'] - 50 < df['bb_lower']),
            'enter_short'
        ] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (df['secondary_rsi'] - 50 > self.thresh_s.value) &
            (df['primary_rsi'] - 50 > df['bb_upper']),
            'exit_short'
        ] = 1
        return df

    def custom_exit(self, pair: str, trade, current_time, current_rate: float,
                    current_profit: float, **kwargs) -> str:
        df = kwargs.get("dataframe", None)
        if df is None or len(df) < 2:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # QQE 趋势反转
        if prev['dirn_p'] == -1 and last['dirn_p'] == 1:
            return "QQE_Trend_Reversal"

        peak_profit = getattr(trade, 'max_profit', current_profit)
        strength = abs(last['primary_rsi'] - 50)
        roi_target = 0.02 + min(strength / 100, 0.06)
        trailing_offset = 0.005 + min(strength / 100, 0.02)

        if current_profit > roi_target and current_profit < (peak_profit - trailing_offset):
            return f"Dynamic_Trail({trailing_offset:.3f})"

        return None
