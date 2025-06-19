from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np

#1h: CAGR %170

class TrailingStop(IStrategy):
    """
    基于 LuxAlgo Statistical Trailing Stop 的 Freqtrade 策略示例
    - 自动翻多/翻空：当价格突破 `level` 时翻转
    - 自定义止损：使用统计波幅分布动态生成追踪止损线
    """
    timeframe = '1h'
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
    stoploss = -0.8
    use_custom_stoploss = True

    # 参数
    data_length = 10
    distribution_length = 100
    base_level = 2  # 对应 Lux 的 Level2

    def leverage(self, pair: str, current_leverage: float = 1.0, max_leverage: float = 5.0, side: str = "long",
                 **kwargs) -> float:
        return 10

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 计算 HLC3（为后续使用做准备）
        dataframe['hlc3'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3

        # 计算真实波幅
        prev_close = dataframe['close'].shift(1)
        tr = np.maximum(
            dataframe['high'] - dataframe['low'],
            np.maximum(
                abs(dataframe['high'] - prev_close),
                abs(dataframe['low'] - prev_close)
            )
        )

        # 对数波幅
        dataframe['log_tr'] = np.log(tr.replace(0, np.nan)).fillna(0)

        # 计算 log 波幅分布的均值与标准差
        dataframe['tr_mean'] = dataframe['log_tr'].rolling(self.distribution_length).mean()
        dataframe['tr_std'] = dataframe['log_tr'].rolling(self.distribution_length).std()

        # 计算 delta
        dataframe['delta'] = np.exp(
            dataframe['tr_mean'] + self.base_level * dataframe['tr_std']
        )

        # 初始化 bias 与 anchor 等值
        dataframe['bias'] = 0
        dataframe['anchor'] = 0.0
        dataframe['level'] = np.nan
        dataframe['extreme'] = np.nan

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 当价格突破止损线时翻多
        df = dataframe.copy()
        for i in range(1, len(df)):
            if np.isnan(df.at[i-1, 'level']):
                # 初始 bias 设为空头
                df.at[i, 'bias'] = -1
                df.at[i, 'anchor'] = df.at[i-1, 'close']
                df.at[i, 'extreme'] = df.at[i-1, 'low']
                df.at[i, 'level'] = df.at[i, 'anchor'] + df.at[i, 'delta']
            else:
                prev = df.at[i-1, 'bias']
                anchor = df.at[i-1, 'anchor']
                level  = df.at[i-1, 'level']
                extreme = df.at[i-1, 'extreme']
                price = df.at[i, 'close']
                delta = df.at[i, 'delta']
                # 更新 extreme
                if prev == 1:
                    extreme = max(extreme, df.at[i, 'high'])
                else:
                    extreme = min(extreme, df.at[i, 'low'])
                # 更新 level
                if prev == 1:
                    level = max(level, max(df.at[i, 'hlc3'] - delta, 0))
                else:
                    level = min(level, df.at[i, 'hlc3'] + delta)
                # 判断翻转
                if (prev == -1 and price > level):
                    bias = 1
                    anchor = price
                    extreme = df.at[i, 'high']
                    level = max(df.at[i, 'hlc3'] - delta, 0)
                    dataframe.loc[dataframe.index[i], 'buy'] = 1
                elif (prev == 1 and price < level):
                    bias = -1
                    anchor = price
                    extreme = df.at[i, 'low']
                    level = df.at[i, 'hlc3'] + delta
                    dataframe.loc[dataframe.index[i], 'sell'] = 1
                else:
                    bias = prev
                df.at[i, 'bias'] = bias
                df.at[i, 'anchor'] = anchor
                df.at[i, 'extreme'] = extreme
                df.at[i, 'level'] = level
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 不使用 exit 信号，由 custom_stoploss 处理止损
        return dataframe

    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs) -> float:
        # 获取当前指标行
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        row = df[df['date'] == current_time]
        if row.empty:
            return 1.0
        level = row['level'].values[0]
        # 生成 stoploss %
        if current_rate <= level:
            return 0.01  # 跌破时以 1% 止损
        return 1.0  # 否则不开启止损
