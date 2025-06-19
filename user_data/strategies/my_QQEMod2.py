from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib.abstract as ta


class QQEMod2(IStrategy):
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
    trailing_stop = False

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

    atr_len = IntParameter(5, 30, default=14, space="buy", optimize=True)
    atr_thresh = DecimalParameter(0.001, 0.02, default=0.01, decimals=4, space="buy", optimize=True)  # 波动率下限

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
                prev_lb = lb.iat[i - 1]
                prev_sb = sb.iat[i - 1]
                prev_dir = dirn.iat[i - 1]
                lb.iat[i] = max(prev_lb, new_lb) if sm_rsi.iat[i - 1] > prev_lb and sm_rsi.iat[i] > prev_lb else new_lb
                sb.iat[i] = min(prev_sb, new_sb) if sm_rsi.iat[i - 1] < prev_sb and sm_rsi.iat[i] < prev_sb else new_sb
                dirn.iat[i] = 1 if sm_rsi.iat[i] > prev_sb else -1 if sm_rsi.iat[i] < prev_lb else prev_dir

        trend_line = pd.Series(np.where(dirn == 1, lb, sb), index=rsi_series.index)
        return trend_line, sm_rsi, dirn

    def is_consolidating(self, df: DataFrame) -> pd.Series:
        bb_width = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

        # 更严格的 RSI 判断
        rsi_consolidating = (df['primary_rsi'] > 47) & (df['primary_rsi'] < 53)

        # 提高 ATR 阈值过滤力度
        atr_low = df['atr'] < (df['close'] * float(self.atr_thresh.value))

        # 更动态的布林带收窄判断（当前宽度 < 过去20根的平均宽度 * 0.8）
        bb_avg_width = bb_width.rolling(window=20).mean()
        bb_narrow = bb_width < (bb_avg_width * 0.8)

        # 可选 debug 输出震荡占比
        # print(f"震荡区间占比: {(rsi_consolidating & bb_narrow & atr_low).mean():.2%}")

        return rsi_consolidating & bb_narrow & atr_low

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

        df['atr'] = ta.ATR(df, timeperiod=int(self.atr_len.value))

        df['is_consolidating'] = self.is_consolidating(df)
        df.dropna(inplace=True)

        # 添加成交量过滤：计算最近 N 根的平均成交量，并判断当前是否过低
        df['avg_volume'] = df['volume'].rolling(window=20).mean()
        df['low_volume'] = df['volume'] < df['avg_volume'] * 0.5  # 你可以调整阈值 0.5，表示低于50%平均成交量

        df.dropna(inplace=True)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (~df['is_consolidating']) &
            (~df['low_volume']) &  # 新增的成交量过滤条件
            (df['secondary_rsi'] - 50 > self.thresh_s.value) &
            (df['primary_rsi'] - 50 > df['bb_upper']),
            'buy'
        ] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # RSI 斜率（主 RSI 下滑越快越有用）
        df['primary_rsi_slope'] = df['primary_rsi'].diff()

        # 顶部反转识别（局部高点 + RSI > 阈值）
        df['rsi_peak'] = (
                (df['primary_rsi'].shift(2) < df['primary_rsi'].shift(1)) &
                (df['primary_rsi'].shift(1) > df['primary_rsi']) &
                (df['primary_rsi'].shift(1) > 70)  # 可调
        )

        # 出场条件：任一信号触发
        df.loc[
            (
                    (df['secondary_rsi'] - 50 < -self.thresh_s.value) |  # 次级趋势弱化
                    (df['primary_rsi'] - 50 < df['bb_lower']) |  # RSI 跌穿布林下轨
                    (df['primary_rsi_slope'] < -2) |  # RSI 快速下滑
                    (df['dirn_p'] == -1) |  # 主趋势变空
                    (df['rsi_peak'])  # 顶部反转识别
            ),
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

        # QQE 主趋势反转出场
        if prev_row['dirn_p'] == 1 and last_row['dirn_p'] == -1:
            return "QQE_Trend_Reversal"

        # RSI 死叉 QQE 趋势线
        if (
                prev_row['primary_rsi'] > prev_row['primary_trend'] and
                last_row['primary_rsi'] < last_row['primary_trend']
        ):
            return "RSI_Cross_Trend_Down"

        # 动态止盈（根据趋势强度和回撤）
        peak_profit = trade.max_profit if hasattr(trade, 'max_profit') else current_profit
        trend_strength = abs(last_row['primary_rsi'] - 50)
        dyn_roi_threshold = 0.01 + min(trend_strength / 100, 0.04)
        dyn_trailing_offset = 0.004 + min(trend_strength / 100, 0.015)

        if current_profit > dyn_roi_threshold:
            if current_profit < (peak_profit - dyn_trailing_offset):
                return f"Trailing_Exit_{dyn_trailing_offset:.3f}"

        return None