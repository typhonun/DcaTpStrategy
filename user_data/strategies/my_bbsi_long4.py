from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade, Order
from pandas import DataFrame, Timestamp
import pandas as pd
import talib.abstract as ta
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BollingerRsiDCA4(IStrategy):
    """
    完整做空版，将原做多逻辑对称地转为做空，并把 u<->n 互换：

    1) 浮亏 DCA（均以 USDT 保证金为单位）：
       - 当 price ≥ avg_entry * 1.01 且 RSI > 65 → 本根 K 线只触发一次，加仓当前仓位 USDT 保证金的 30%，n += 1，同时清零 u。
       - 用 custom_data["last_dca_candle"] 存储上次触发 DCA 的 K 线时间戳，保证同根 K 线不重复触发。

    2) 自定义分批止盈 & 止盈次数 u（均以 USDT 保证金为单位）：
       - 如果 n > 0 且 current_profit ≥ applicable_roi → 一次性“买回”USDT 保证金的 (30% + 5% × n)，重置 n=0, u=0, dca_done=False。
       - 如果 n == 0 且 current_profit ≥ applicable_roi → 本根 K 线只触发一次“买回”USDT 保证金的 30%（ft_order_tag="tp30"），u += 1，dca_done=False；
         用 custom_data["last_tp_candle"] 存储本根 K 线的 timestamp，防止同根 K 线重复执行。
         卖单（平空）成交后在 order_filled() 标记 need_rebuy30=True，等待下一根 K 线由 adjust_trade_position() 返回“做空加仓 70% USDT 保证金”。

    3) 止盈后回撤 3%（u > 0，仍以 USDT 保证金为单位）：
       - 如果 u > 0 且 current_profit < 0.03 → “买回”USDT 保证金的 (20% + 3% × u)（平空止盈），重置 n=0, u=0, dca_done=False，不再加仓。

    4) on_trade_open()：新开仓时初始化所有 custom_data 标志，包括 last_tp_candle。

    5) order_filled()：当“tp30” 卖单（平空 30%）完全成交后，把 custom_data["need_rebuy30"]=True，交给下一根 K 线由 adjust_trade_position() 触发“做空加仓 70% USDT 保证金”。
    """

    # ———— 策略基础配置 ————
    timeframe = '1m'
    stoploss = -7  # 下跌 7% 则强制平仓
    use_exit_signal = False
    trailing_stop = False
    can_short = True
    can_long = False

    # 屏蔽框架自带的 ROI 退出
    minimal_roi = {"0": 999.0}

    # 自定义 ROI 表，用于计算 “elapsed_min” 对应的 ROI 阈值
    minimal_roi_user_defined = {
        "720": -0.1, "480": -0.05, "360": 0.01, "180": 0.020, "170": 0.050,
        "160": 0.080, "150": 0.110, "140": 0.140, "130": 0.170, "120": 0.200,
        "110": 0.230, "100": 0.260, "90": 0.290, "80": 0.320, "70": 0.350,
        "60": 0.380, "50": 0.400, "40": 0.420, "30": 0.440, "20": 0.460,
        "10": 0.480, "0": 0.500
    }

    position_adjustment_enable = True

    def leverage(self, pair: str, **kwargs) -> float:
        return 50

    def on_trade_open(self, trade: Trade, **kwargs) -> None:
        """
        新开仓时初始化所有 custom_data 标志：
        - dca_count：原 u，此处记录“止盈次数” u
        - tp_count：原 n，此处记录“浮亏 DCA”次数 n
        - dca_done：本根 K 线是否已触发过 DCA
        - last_dca_candle：上次触发浮亏 DCA 的 K 线 timestamp（秒级 int）
        - last_tp_candle：上次触发“买回30%”所在的 K 线 timestamp（秒级 int），用于避免在同根 K 线上重复累加 u
        - need_rebuy30：当“tp30”平空 30% 完成后，下一根 K 线是否要立刻再做空 70%
        """
        trade.set_custom_data(key='dca_count', value=0)
        trade.set_custom_data(key='tp_count', value=0)
        trade.set_custom_data(key='dca_done', value=False)
        trade.set_custom_data(key='last_dca_candle', value=None)
        trade.set_custom_data(key='last_tp_candle', value=None)
        trade.set_custom_data(key='need_rebuy30', value=False)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算布林带（20 周期）和 RSI（14 周期）。做空时用 RSI > 65 配合 DCA 触发。
        """
        upper, mid, lower = ta.BBANDS(
            dataframe['close'],
            timeperiod=20,
            nbdevup=2.0,
            nbdevdn=2.0,
            matype=0
        )
        dataframe['bb_upperband'] = upper
        dataframe['bb_midband'] = mid
        dataframe['bb_lowerband'] = lower
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        做空进场信号：当收盘价 > 布林带上轨 且 RSI > 70 时做空
        """
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe.loc[
            (dataframe['close'] > dataframe['bb_upperband']) &
            (dataframe['rsi'] > 70),
            'enter_short'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        不使用指标产生退出信号，一律在 adjust_trade_position + order_filled 中处理
        """
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
        核心做空逻辑（全部以 USDT 保证金为单位）：
        1) 浮亏 DCA（每根新 K 线只触发一次）：n += 1, u = 0
        2) rebuy30（分批止盈 Step2）：当“tp30”平空 30% 完成后，下一根 K 线立刻做空 70%
        3) 止盈后回撤 3%（u > 0）：买回 (20% + 3% × u)，u=n=0
        4) 自定义分批止盈：if current_profit ≥ applicable_roi，则按照 n>0 或 n==0 分两种情况处理，
           对于 n==0 的情况，需要做“本根 K 线只触发一次 u += 1”的判断，使用 last_tp_candle 避免同根 K 线重复执行
        """

        # 0) 若当前有未成交挂单，跳过
        if trade.has_open_orders:
            return None

        # 1) 读取 custom_data
        tp_count = int(trade.get_custom_data(key='tp_count') or 0)  # 原 n → 浮亏 DCA 次数
        dca_count = int(trade.get_custom_data(key='dca_count') or 0)  # 原 u → 止盈次数
        dca_done = bool(trade.get_custom_data(key='dca_done'))
        need_rebuy = bool(trade.get_custom_data(key='need_rebuy30'))
        last_tp_ts_int = trade.get_custom_data(key='last_tp_candle')

        # 2) 获取最新 K 线 DataFrame、RSI、时间戳
        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df.empty:
            return None

        last_index = df.index[-1]
        current_candle_ts = pd.Timestamp(last_index)
        if current_candle_ts.tzinfo is not None:
            current_candle_ts = current_candle_ts.tz_convert(None)
        current_candle_ts = current_candle_ts.floor('T')
        last_rsi = df['rsi'].iat[-1]

        # 3) 读取上次浮亏 DCA 触发时的 K 线 timestamp
        last_dca_ts_int = trade.get_custom_data(key='last_dca_candle')
        last_dca_candle = None
        if last_dca_ts_int is not None:
            last_dca_candle = Timestamp(last_dca_ts_int, unit='s')

        # 4) 如果“进入新 K 线”，把 dca_done 设为 False；同理，如果 new candle，也把 last_tp_candle 清除
        if (last_dca_candle is None) or (last_dca_candle != current_candle_ts):
            trade.set_custom_data(key='dca_done', value=False)
            dca_done = False

        if last_tp_ts_int is not None:
            last_tp_candle = Timestamp(last_tp_ts_int, unit='s')
        else:
            last_tp_candle = None
        if (last_tp_candle is None) or (last_tp_candle != current_candle_ts):
            # 如果进入新 K 线，则清除标记，下根 K 线才能再次触发 u += 1
            trade.set_custom_data(key='last_tp_candle', value=None)
            last_tp_candle = None

        avg_entry = float(trade.open_rate)

        # ─── 步骤 1：浮亏 DCA ───
        # 如果当前 K 线还没做过 DCA，且 price ≥ entry*1.01 且 RSI > 65，就触发浮亏 DCA
        if (not dca_done) and (current_rate >= avg_entry * 1.01) and (last_rsi > 65):
            usdt_stake = float(trade.stake_amount)
            sell_stake = 0.30 * usdt_stake  # 再做空 30% 保证金对应的基础币

            new_n = tp_count + 1
            trade.set_custom_data(key='tp_count', value=new_n)  # n += 1
            trade.set_custom_data(key='dca_done', value=True)
            trade.set_custom_data(key='last_dca_candle', value=int(current_candle_ts.timestamp()))
            trade.set_custom_data(key='dca_count', value=0)  # 清零 u
            # 清除 last_tp_candle，让本根 K 线只触发一次 "u += 1"
            trade.set_custom_data(key='last_tp_candle', value=None)

            logger.info(
                f"[{trade.pair}][浮亏 DCA 加仓] n=({tp_count} → {new_n}), u=0, "
                f"保证金(USDT)={usdt_stake:.8f}, 再做空(USDT)={sell_stake:.8f}"
            )
            # 返回一个正值，Freqtrade 识别为卖单 (sell)
            return sell_stake, f"dca_n={new_n}"

        # ─── 步骤 2：rebuy30（分批止盈 Step2）───
        if need_rebuy:
            usdt_stake = float(trade.stake_amount)
            sell_stake = 0.70 * usdt_stake  # 做空剩余 70% 保证金对应的基础币

            logger.info(
                f"[{trade.pair}][分批止盈 Step2 做空70%] n={tp_count}, u={dca_count}, "
                f"保证金(USDT)={usdt_stake:.8f}, 再做空(USDT)={sell_stake:.8f}"
            )
            trade.set_custom_data(key='need_rebuy30', value=False)
            trade.set_custom_data(key='dca_done', value=False)
            # 保留 tp_count 和 dca_count 以便“止盈后回撤3%”检查
            return sell_stake, "rebuy30"

        # ─── 步骤 3：止盈后回撤 3% ───
        if (dca_count > 0) and (current_profit < 0.03):
            usdt_stake = float(trade.stake_amount)
            sell_pct = 0.20 + 0.05 * dca_count
            if sell_pct > 1.0:
                sell_pct = 1.0
            # 平空止盈时，要买回基础币，返回正数
            sell_stake = sell_pct * usdt_stake

            logger.info(
                f"[{trade.pair}][止盈后回撤3%] n={tp_count}, u={dca_count}, "
                f"保证金(USDT)={usdt_stake:.8f}, 买回比例={sell_pct * 100:.1f}%, 买回(USDT)={sell_stake:.8f}"
            )
            trade.set_custom_data(key='tp_count', value=0)
            trade.set_custom_data(key='dca_count', value=0)
            trade.set_custom_data(key='dca_done', value=False)
            # 清除 last_tp_candle，下根 K 线才可重新触发 u++
            trade.set_custom_data(key='last_tp_candle', value=None)
            return sell_stake, f"tp_fallback3%_{int(sell_pct * 100)}%"

        # ─── 步骤 4：自定义分批止盈（applicable_roi） ───
        elapsed_min = (current_time - trade.open_date_utc).total_seconds() / 60
        applicable_roi = 0.0
        for k, v in sorted(self.minimal_roi_user_defined.items(), key=lambda x: int(x[0]), reverse=True):
            if elapsed_min >= int(k):
                applicable_roi = v
                break

        if current_profit >= applicable_roi:
            usdt_stake = float(trade.stake_amount)

            # —— 情况 A：曾做过浮亏 DCA（n > 0）——
            if tp_count > 0:
                sell_pct = 0.30 + 0.07 * tp_count
                if sell_pct > 1.0:
                    sell_pct = 1.0
                sell_stake = sell_pct * usdt_stake  # 平空止盈，买回基础币

                logger.info(
                    f"[{trade.pair}][浮亏 DCA 后止盈] n={tp_count}, u={dca_count}, "
                    f"保证金(USDT)={usdt_stake:.8f}, 买回比例={sell_pct * 100:.1f}%, 买回(USDT)={sell_stake:.8f}"
                )
                trade.set_custom_data(key='tp_count', value=0)
                trade.set_custom_data(key='dca_count', value=0)
                trade.set_custom_data(key='dca_done', value=False)
                # 清除 last_tp_candle，下根 K 线才可重新触发 u++
                trade.set_custom_data(key='last_tp_candle', value=None)
                return sell_stake, f"tp_afterDCA_{int(sell_pct * 100)}%"

            # —— 情况 B：从未做过浮亏 DCA（n == 0）——
            # 本根 K 线只执行一次“买回30%”（u += 1），需要判断 last_tp_candle
            if (last_tp_candle is None) or (last_tp_candle != current_candle_ts):
                # 本根 K 线尚未触发过“买回30%” ⇒ u += 1
                sell_pct = 0.30
                sell_stake = sell_pct * usdt_stake
                new_u = dca_count + 1  # u += 1

                trade.set_custom_data(key='dca_count', value=new_u)
                trade.set_custom_data(key='tp_count', value=0)
                trade.set_custom_data(key='dca_done', value=False)
                # 记录本根 K 线已执行“买回30%”
                trade.set_custom_data(key='last_tp_candle', value=int(current_candle_ts.timestamp()))

                logger.info(
                    f"[{trade.pair}][浮盈减仓 平空30%→后续做空70%] n=0, u=({dca_count} → {new_u}), "
                    f"保证金(USDT)={usdt_stake:.8f}, 平空30%(USDT)={sell_stake:.8f}"
                )
                # 平空 30% 后，order_filled() 会标记 need_rebuy30=True，下一根 K 线再做空 70%
                return sell_stake, "tp30"

            # 如果 last_tp_candle == current_candle_ts，说明本根K线已经执行过 u += 1 返仓逻辑，跳过
            return None

        # 其它情况不下单
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
        当“tp30” 卖单（平空 30% USDT 保证金）完全成交后，将 need_rebuy30 置 True，让下一根 K 线触发“做空 70%（rebuy30）”
        """
        tag = getattr(order, "ft_order_tag", None)
        if tag == "tp30" and order.side == "sell":
            trade.set_custom_data(key='need_rebuy30', value=True)
            tp_count = int(trade.get_custom_data(key='tp_count') or 0)
            dca_count = int(trade.get_custom_data(key='dca_count') or 0)
            logger.info(
                f"[{pair}][分批止盈 Step1 “tp30” 已成交] n={tp_count}, u={dca_count}，"
                f"标记 need_rebuy30=True，下一根 K 线再做空 70% USDT 保证金。"
            )

    def custom_stoploss(
            self, pair: str, trade: Trade, current_time, current_rate, current_profit, **kwargs
    ) -> float | None:
        """
        禁用框架默认止损，策略仅依赖浮亏 DCA + 自定义 ROI 逻辑进行止损/止盈。
        """
        return None
