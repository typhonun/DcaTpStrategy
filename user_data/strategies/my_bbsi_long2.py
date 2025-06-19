from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade, Order
from pandas import DataFrame, Timestamp
import pandas as pd
import talib.abstract as ta
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BollingerRsiDCA2(IStrategy):
    """
    核心思路：
    1) 把 minimal_roi 设为永不触发（“0”: 999），屏蔽框架全仓 ROI 退出。
    2) 在 adjust_trade_position() 里：
       a. 浮亏 DCA：price <= avg_entry*0.99 且 RSI<35，每根 K 线只加一次仓，30% 的持仓，u+=1。
       b. 自定义的 ROI 判断：current_profit >= applicable_roi 时
          - 如果 u>0：一次性卖 (30% + 5%*u)，重置 u 与标志，不加仓。
          - 如果 u==0：第一次卖 30%（标记 flag_step1=True），在下一根 K 线里再市价买回 70%（rebuy 流程），重置标志。
    """

    # 基础配置
    timeframe = '1m'
    stoploss = -5           # 仍然保留一个止损，但可以根据需要自行修改
    use_exit_signal = False     # 我们不通过 populate_exit_trend() 产生 exit_long
    trailing_stop = False

    # 把 minimal_roi 设置得永远触发不到
    minimal_roi = {
        "0": 999.0
    }

    # 下面这张表只是给 strategy 里拿来算 “applicable_roi” 之用
    # 注意：因为我们把 minimal_roi 全部屏蔽了，这里就用一张自定义表
    minimal_roi_user_defined = {
        "190": 0.005,
        "180": 0.010,
        "170": 0.015,
        "160": 0.020,
        "150": 0.025,
        "140": 0.030,
        "130": 0.035,
        "120": 0.040,
        "110": 0.045,
        "100": 0.050,
        "90": 0.055,
        "80": 0.060,
        "70": 0.065,
        "60": 0.070,
        "50": 0.075,
        "40": 0.080,
        "30": 0.085,
        "20": 0.090,
        "10": 0.095,
        "0": 0.100
    }

    position_adjustment_enable = True

    def leverage(self, pair: str, **kwargs) -> float:
        return 20

    def on_trade_open(self, trade: Trade, **kwargs) -> None:
        """
        每次新开仓时，清空所有 custom_data 标志：
        - dca_count：浮亏 DCA 的累计次数 u
        - flag_step1/flag_step2：分批止盈的标志
        - dca_done：本根 K 线是否已经做过 DCA
        - need_rebuy70：分批止盈第一步(tp30)成交后，下一根 K 线是否要立刻买回 70%
        - last_dca_candle：上次执行 DCA 时对应的 K 线时间戳（秒级 integer）
        """
        trade.set_custom_data(key='dca_count',       value=0)
        trade.set_custom_data(key='flag_step1',      value=False)
        trade.set_custom_data(key='flag_step2',      value=False)
        trade.set_custom_data(key='dca_done',        value=False)
        trade.set_custom_data(key='need_rebuy70',    value=False)
        trade.set_custom_data(key='last_dca_candle', value=None)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算 20 周期的布林带和 14 周期的 RSI，作为浮亏 DCA 的依据
        """
        upper, mid, lower = ta.BBANDS(
            dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
        )
        dataframe['bb_upperband'] = upper
        dataframe['bb_midband']   = mid
        dataframe['bb_lowerband'] = lower
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        进场信号：当 收盘价 < 布林下轨 且 RSI < 30 时做多
        """
        dataframe['enter_long'] = 0
        dataframe.loc[
            (dataframe['close'] < dataframe['bb_lowerband']) &
            (dataframe['rsi'] < 30),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        我们不让 populate_exit_trend 产生任何退出信号，一切退出都交给 adjust_trade_position()
        """
        dataframe['exit_long'] = 0
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
        核心逻辑：

        1) 浮亏 DCA：
           - 先根据 df 索引拿到“当前 K 线”的时间戳 current_candle_ts（floor 到分钟）。
             如果 custom_data["last_dca_candle"] 不存在或与 current_candle_ts 不同，就说明是“新 K 线”，
             这时把 custom_data["dca_done"]=False，以便本根新的 K 线可能做一次 DCA。
           - 只有在本根 K 线 dca_done=False 且 current_rate <= avg_entry×0.99 且 RSI<35 时，
             才做一次“卖掉 30% 仓位对应的成本（即加仓当前持仓 30% 的基础币）”，并：
             · trade.set_custom_data("dca_count", u+1)
             · trade.set_custom_data("dca_done", True)
             · trade.set_custom_data("last_dca_candle", current_candle_ts.timestamp())
           - 返回一个正数 buy_stake → 代表“市价买入 30% 持仓对应的 USDT 数量”。

        2) Rebuy70（分批止盈 Step2）：
           - 如果 custom_data["need_rebuy70"] == True，说明上一根 K 线的“卖 30%”单(tp30)已经在 order_filled 里
             把 need_rebuy70 置为 True，到了这一根新的 K 线就要立刻市价 “买回剩余 70% 基础币”，
             并设置 flag_step2=True，清除 need_rebuy70、flag_step1。
           - 返回一个正数 buy_stake → 代表“市价买入 70% 剩余基础币所需的 USDT 数量”。

        3) 自定义分批止盈（自定义 ROI）：
           - 先计算 elapsed_min = (current_time - trade.open_date_utc).total_seconds()/60，
             然后在 self.minimal_roi_user_defined 中，找到“最接近且 elapsed_min >= key” 的 ROI 值。
           - 如果 current_profit >= applicable_roi：
             A) 若 u>0：直接一次性卖出 (30% + 5%*u) 的持仓，对应 sell_stake = -sell_pct * trade.stake_amount。
                并重置 custom_data["dca_count"]=0、flag_step1=False、flag_step2=False。返回 (sell_stake, tag)。
             B) 若 u==0：
                · 如果 flag_step1=False：卖出 30% (sell_stake=-0.30*trade.stake_amount)，ft_order_tag="tp30"，同时
                  trade.set_custom_data("flag_step1", True)。等这笔 limit/market 卖单在 order_filled() 里把
                  need_rebuy70 置 True，然后等下一根 K 线再做 Rebuy70。
                · flag_step1=True 时，不在此处再返回，等“rebuy70”逻辑去处理买回70%。

        * 无需 return None 的时候就不下任何单。
        """
        # 1) 有未成交挂单时，跳过所有逻辑
        if trade.has_open_orders:
            return None

        # 2) 读取 custom_data 标志
        flag1      = bool(trade.get_custom_data(key='flag_step1'))
        flag2      = bool(trade.get_custom_data(key='flag_step2'))
        u          = int(trade.get_custom_data(key='dca_count') or 0)
        dca_done   = bool(trade.get_custom_data(key='dca_done'))
        need_rebuy = bool(trade.get_custom_data(key='need_rebuy70'))

        # 3) 拿到最新 K 线的收盘价/RSI/时间戳
        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df.empty:
            return None

        # 3.1 取最后一根 K 线的索引
        last_index = df.index[-1]
        current_candle_ts = pd.Timestamp(last_index)
        if current_candle_ts.tzinfo is not None:
            current_candle_ts = current_candle_ts.tz_convert(None)
        current_candle_ts = current_candle_ts.floor('T')

        last_rsi = df['rsi'].iloc[-1]

        # 4) 检查是否“进入新 K 线”：如果 last_dca_candle 为空或与 current_candle_ts 不同，就重置 dca_done
        last_dca_ts_int = trade.get_custom_data(key='last_dca_candle')
        last_dca_candle = None
        if last_dca_ts_int is not None:
            last_dca_candle = Timestamp(last_dca_ts_int, unit='s')
        if (last_dca_candle is None) or (last_dca_candle != current_candle_ts):
            trade.set_custom_data(key='dca_done', value=False)
            dca_done = False

        # ─── 步骤 1：浮亏 DCA ───
        avg_entry = float(trade.open_rate)
        if (not dca_done) and (current_rate <= avg_entry * 0.99) and (last_rsi < 35):
            add_amount = float(trade.amount) * 0.30
            buy_stake  = add_amount * current_rate

            trade.set_custom_data(key='dca_count',      value=u + 1)
            trade.set_custom_data(key='dca_done',       value=True)
            trade.set_custom_data(key='last_dca_candle', value=int(current_candle_ts.timestamp()))

            logger.info(
                f"[{trade.pair}][浮亏 DCA 加仓] u = ({u} → {u+1}), "
                f"当前持仓(USDT) = {trade.stake_amount:.8f}, 加仓(USDT) = {buy_stake:.8f}"
            )
            return buy_stake, f"dca_u={u+1}"

        # ─── 步骤 2：rebuy70 ───
        if need_rebuy:
            remaining_amount = float(trade.amount)  # 剩余基础币数
            if remaining_amount > 0:
                buy_amount = 0.70 * remaining_amount
                buy_stake  = buy_amount * current_rate

                logger.info(
                    f"[{trade.pair}][浮盈加仓： Step2 买回70%] u=0, "
                    f"当前持仓(USDT) = {trade.stake_amount:.8f}, 买回(USDT) = {buy_stake:.8f}"
                )
                trade.set_custom_data(key='need_rebuy70', value=False)
                trade.set_custom_data(key='flag_step2',    value=True)
                trade.set_custom_data(key='flag_step1',    value=False)
                return buy_stake, "rebuy70"

        # ─── 步骤 3：自定义分批止盈 ───
        elapsed_min    = (current_time - trade.open_date_utc).total_seconds() / 60
        applicable_roi = 0.0
        for key, val in sorted(self.minimal_roi_user_defined.items(), key=lambda x: int(x[0]), reverse=True):
            if elapsed_min >= int(key):
                applicable_roi = val
                break

        if current_profit >= applicable_roi:
            # A) 曾经做过浮亏 DCA（u > 0）
            if u > 0:
                sell_pct   = 0.30 + 0.05 * u
                sell_pct   = min(sell_pct, 1.0)
                sell_stake = -sell_pct * trade.stake_amount

                logger.info(
                    f"[{trade.pair}][浮亏 DCA 止盈： ] u={u}, 当前持仓(USDT) = {trade.stake_amount:.8f}, "
                    f"卖出比例={sell_pct*100:.1f}%, 卖出(USDT)={abs(sell_stake):.8f}"
                )
                trade.set_custom_data(key='dca_count',  value=0)
                trade.set_custom_data(key='flag_step1', value=False)
                trade.set_custom_data(key='flag_step2', value=False)
                return sell_stake, f"tp_afterDCA_{int(sell_pct*100)}%"

            # B) 从未做过浮亏 DCA（u == 0）
            if not flag1:
                sell_stake = -0.30 * trade.stake_amount
                trade.set_custom_data(key='flag_step1', value=True)

                logger.info(
                    f"[{trade.pair}][浮盈减仓： 卖30%] u=0, 当前持仓(USDT) = {trade.stake_amount:.8f}, "
                    f"卖出30%(USDT) = {abs(sell_stake):.8f}"
                )
                return sell_stake, "tp30"
            # → 等待 order_filled() 设置 need_rebuy70，下一根 K 线买回70%

        # 否则，不做任何操作
        return None

    def order_filled(
        self,
        pair: str,
        trade: Trade,
        order: Order,
        current_time: datetime,
        **kwargs
    ) -> None:
        """
        当“tp30” 卖单完全成交后（即分批止盈的第 1 步），
        把 custom_data['need_rebuy70'] = True，让下一根 K 线立刻买回70%。
        """
        tag = getattr(order, "ft_order_tag", None)
        if tag == "tp30" and order.side == "sell":
            trade.set_custom_data(key='need_rebuy70', value=True)
            logger.info(
                f"[{pair}][浮盈加仓： Step1 “tp30” 已成交] 标记 need_rebuy70=True，下一根 K 线买回70%。"
            )

    def custom_stoploss(
        self, pair: str, trade: Trade, current_time, current_rate, current_profit, **kwargs
    ) -> float | None:
        # 我们自己已经有 DCA+ROI 控制止损/止盈，不让框架再额外做 stoploss
        return None
