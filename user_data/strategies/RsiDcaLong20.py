from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade, Order
from pandas import DataFrame, Timestamp
import pandas as pd
import talib.abstract as ta
from datetime import datetime, timedelta
import logging

RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
RESET = "\033[0m"
logger = logging.getLogger(__name__)


class RsiDCALong20(IStrategy):
    """
    DCA + 分批止盈策略，新增：
      - 当触发浮亏 DCA（u>0）后，若 8 小时内未触发任何止盈，则减仓 10%（只执行一次）。
      - 步骤4：自定义分批止盈 & ROI 重置，从上次止盈或开仓时开始计时。
    """

    timeframe = '1m'
    stoploss = -7
    use_exit_signal = False
    trailing_stop = False
    can_short = False
    can_long = True

    minimal_roi = {"0": 777.0}
    minimal_roi_user_defined = {
        "300": 0.010, "290": 0.020, "280": 0.030, "270": 0.040, "260": 0.050,
        "250": 0.060, "240": 0.080, "230": 0.090, "220": 0.100, "210": 0.110,
        "200": 0.120, "190": 0.130, "180": 0.140, "160": 0.145, "140": 0.150,
        "120": 0.155, "100": 0.160, "80": 0.165, "60": 0.170, "50": 0.175,
        "40": 0.180, "30": 0.185, "20": 0.190, "10": 0.195, "0": 0.200,
    }

    position_adjustment_enable = True

    def leverage(self, pair: str, **kwargs) -> float:
        return 20

    def on_trade_open(self, trade: Trade, **kwargs) -> None:
        # 初始化所有 custom_data
        trade.set_custom_data('dca_count', 0)
        trade.set_custom_data('tp_count', 0)
        trade.set_custom_data('dca_done', False)
        trade.set_custom_data('last_dca_candle', None)
        trade.set_custom_data('last_dca_time', None)
        trade.set_custom_data('dca_reduce_done', False)  # for u>0 6h reduction
        trade.set_custom_data('open_reduce_done', False)  # for u==0 8h reduction
        trade.set_custom_data('need_rebuy70', False)
        trade.set_custom_data('last_tp_time', None)
        trade.set_custom_data('low_margin_start', None)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 初始化指标
        upper, mid, lower = ta.BBANDS(
            dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
        )
        dataframe['bb_upperband'] = upper
        dataframe['bb_midband'] = mid
        dataframe['bb_lowerband'] = lower
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe.loc[
            (dataframe['close'] < dataframe['bb_lowerband']) &
            (dataframe['rsi'] < 30),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe

    def adjust_trade_position(
            self,
            trade: Trade,
            current_time: datetime,
            current_rate: float,
            current_profit: float,
            **kwargs
    ) -> tuple[float, str] | None:
        # 去掉时区
        if hasattr(current_time, "tzinfo") and current_time.tzinfo is not None:
            current_time = current_time.replace(tzinfo=None)
        open_time = trade.open_date_utc
        if hasattr(open_time, "tzinfo") and open_time.tzinfo is not None:
            open_time = open_time.replace(tzinfo=None)

        # 跳过未成交订单
        if trade.has_open_orders:
            return None

        # ===== 新增 低保证金持续4h加仓逻辑 =====
        margin = float(trade.stake_amount)
        low_start = trade.get_custom_data('low_margin_start')
        if margin < 5.00:
            if not low_start:
                trade.set_custom_data('low_margin_start', float(current_time.timestamp()))
            else:
                start_time = datetime.fromtimestamp(float(low_start))
                if current_time >= start_time + timedelta(hours=3):
                    buy_amt = 6 - margin
                    logger.info(
                        f"{YELLOW}保证金={trade.stake_amount:.8f}{RESET},"
                        f"{GREEN}保证金小于5usdt已持续3h，加仓至6usdt{RESET}"
                    )
                    trade.set_custom_data('low_margin_start', None)
                    return buy_amt, 'add_to_6_usdt'
        else:
            trade.set_custom_data('low_margin_start', None)

        u = int(trade.get_custom_data('dca_count') or 0)
        # 1) u>0 且 6h 后未止盈减仓20%
        last_dca_time = trade.get_custom_data('last_dca_time')
        reduce6_done = bool(trade.get_custom_data('dca_reduce_done'))
        if u > 0 and last_dca_time and not reduce6_done:
            dca_dt = datetime.fromtimestamp(int(last_dca_time))
            if current_time >= dca_dt + timedelta(hours=16):
                sell_amt = -0.20 * float(trade.stake_amount)
                logger.info(
                    f"[{trade.pair}][6h DCA 后减仓20%] DCA持续 已超16h,减仓20%"
                    f"{YELLOW}保证金={trade.stake_amount:.8f}{RESET},"
                    f"{RED}卖出 {abs(sell_amt):.8f} USDT{RESET}"
                )
                trade.set_custom_data('dca_reduce_done', True)
                return sell_amt, "reduce20%_postDCA"

        # 2) u==0 且开仓后8h未触发DCA减仓10%
        reduce8_done = bool(trade.get_custom_data('open_reduce_done'))
        if u == 0 and not reduce8_done:
            if current_time >= open_time + timedelta(hours=24):
                sell_amt = -0.10 * float(trade.stake_amount)
                logger.info(
                    f"[{trade.pair}][8h 未DCA减仓10%] 未触发Dca 已超24h,减仓10%"
                    f"{YELLOW}保证金={trade.stake_amount:.8f}{RESET},"
                    f"{GREEN}卖出 {abs(sell_amt):.8f} USDT{RESET}"
                )
                trade.set_custom_data('open_reduce_done', True)
                return sell_amt, "reduce10%_postOpen"

        # —— 3) 浮亏 DCA ——
        n = int(trade.get_custom_data('tp_count') or 0)
        dca_done = bool(trade.get_custom_data('dca_done'))
        need_rebuy = bool(trade.get_custom_data('need_rebuy70'))

        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df.empty:
            return None
        last_idx = df.index[-1]
        candle_ts = pd.Timestamp(last_idx)
        if candle_ts.tzinfo:
            candle_ts = candle_ts.tz_convert(None)
        candle_ts = candle_ts.floor('T')
        last_rsi = df['rsi'].iat[-1]

        last_dca = trade.get_custom_data('last_dca_candle')
        last_dca_ts = Timestamp(last_dca, unit='s') if last_dca else None
        if last_dca_ts != candle_ts:
            trade.set_custom_data('dca_done', False)
            dca_done = False

        # 获取动态平均开仓价
        avg_entry = float(trade.get_custom_data('dynamic_avg_entry') or trade.open_rate)
        u = int(trade.get_custom_data('dca_count') or 0)

        # 计算下次加仓阈值：0.99, 0.985, 0.98, … 基于动态均价
        threshold = avg_entry * (1 - 0.005 * (u + 2))

        if not dca_done and current_rate <= threshold and last_rsi < 30:
            # 加仓量：仓位*0.3
            buy_amt = 0.3 * float(trade.stake_amount)
            # 计算新增数量与新的平均开仓价
            leverage = self.leverage(trade.pair)
            prev_qty = float(trade.amount)
            prev_cost = prev_qty * avg_entry
            added_qty = (buy_amt * leverage) / current_rate
            new_qty = prev_qty + added_qty
            new_cost = prev_cost + buy_amt * leverage
            new_avg_entry = new_cost / new_qty
            # 更新 custom_data
            trade.set_custom_data('dca_count', u + 1)
            trade.set_custom_data('dca_done', True)
            trade.set_custom_data('last_dca_candle', int(candle_ts.timestamp()))
            trade.set_custom_data('last_dca_time', int(current_time.timestamp()))
            trade.set_custom_data('dca_reduce_done', False)
            trade.set_custom_data('open_reduce_done', False)
            trade.set_custom_data('tp_count', 0)
            trade.set_custom_data('dynamic_avg_entry', new_avg_entry)
            logger.info(
                f"[{trade.pair}][浮亏 DCA 加仓] {RED}u=({u}→{u + 1}){RESET}, "
                f"{YELLOW}保证金={trade.stake_amount:.8f}{RESET}, {RED}加仓={buy_amt:.8f}{RESET}, "
                f"{BLUE}新均价={new_avg_entry:.8f}{RESET}"
            )
            return buy_amt, f"dca_u={u + 1}"

        # 4) rebuy70
        if need_rebuy:
            buy_amt = 0.70 * float(trade.stake_amount)
            logger.info(
                f"[{trade.pair}][分批止盈 Step2 加仓70%], u={u}, n={n}，"
                f"{YELLOW}保证金={trade.stake_amount:.8f}{RESET}, {GREEN}加仓={buy_amt:.8f}{RESET}"
            )
            trade.set_custom_data('need_rebuy70', False)
            trade.set_custom_data('dca_done', False)
            return buy_amt, f"rebuy70"

        # 5) 止盈后回撤1%
        if n > 0 and current_profit < 0.01:
            pct = min(1.0, 0.20 + 0.05 * n)
            sell_amt = - pct * float(trade.stake_amount)
            logger.info(
                f"[{trade.pair}][止盈后回撤1%] u={u}, n={n}, "
                f"{YELLOW}保证金={trade.stake_amount:.8f}{RESET}, {GREEN}减仓={abs(sell_amt):.8f}{RESET}"
            )
            trade.set_custom_data('dca_count', 0)
            trade.set_custom_data('tp_count', 0)
            trade.set_custom_data('dca_done', False)
            trade.set_custom_data('last_tp_time', int(current_time.timestamp()))
            return sell_amt, f"tp_fallback1%_{int(pct * 100)}%"

        # 6) 自定义分批止盈 & ROI 重置
        last_tp = trade.get_custom_data('last_tp_time')
        if last_tp:
            base_time = Timestamp(last_tp, unit='s')
            if base_time.tzinfo:
                base_time = base_time.tz_localize(None)
        else:
            base_time = open_time

        elapsed = (current_time - base_time).total_seconds() / 60
        applicable_roi = 0.0
        for k, v in sorted(self.minimal_roi_user_defined.items(),
                           key=lambda x: int(x[0]), reverse=True):
            if elapsed >= int(k):
                applicable_roi = v
                break

        if current_profit >= applicable_roi:
            # A) u>0
            if u > 0:
                pct = min(1.0, 0.30 + 0.07 * u)
                sell_amt = - pct * float(trade.stake_amount)
                logger.info(
                    f"[{trade.pair}][浮亏 DCA 后止盈] u={u}, n={n}, "
                    f"{YELLOW}保证金={trade.stake_amount:.8f}{RESET}, {GREEN}减仓={abs(sell_amt):.8f}{RESET}"
                )
                trade.set_custom_data('dca_count', 0)
                trade.set_custom_data('tp_count', 0)
                trade.set_custom_data('dca_done', False)
                trade.set_custom_data('last_tp_time', int(current_time.timestamp()))
                return sell_amt, f"tp_afterDCA_{int(pct * 100)}%"

            # B) u==0
            if not last_tp or Timestamp(last_tp, unit='s').floor('T') != candle_ts:
                pct = 0.30
                sell_amt = - pct * float(trade.stake_amount)
                new_n = n + 1
                logger.info(
                    f"[{trade.pair}][浮盈减仓 卖30%→后续买70%] u=0,"
                    f"{GREEN}n=({n}→{new_n}){RESET}, {YELLOW}保证金={trade.stake_amount:.8f}{RESET}, "
                    f"{GREEN}减仓={abs(sell_amt):.8f}{RESET}"
                )
                trade.set_custom_data('tp_count', new_n)
                trade.set_custom_data('dca_count', 0)
                trade.set_custom_data('dca_done', False)
                trade.set_custom_data('last_tp_time', int(current_time.timestamp()))
                return sell_amt, "tp30"

        return None

    def order_filled(
            self, pair: str, trade: Trade, order: Order,
            current_time: datetime, **kwargs
    ) -> None:
        if getattr(order, 'ft_order_tag', None) == "tp30" and order.side == "sell":
            trade.set_custom_data('need_rebuy70', True)
            u = int(trade.get_custom_data('dca_count') or 0)
            n = int(trade.get_custom_data('tp_count') or 0)
            logger.info(
                f"[{pair}][分批止盈 Step1 tp30 已成交] u={u}, n={n}，标记 need_rebuy70=True"
            )

    def custom_stoploss(self, *args, **kwargs) -> float | None:
        return None
