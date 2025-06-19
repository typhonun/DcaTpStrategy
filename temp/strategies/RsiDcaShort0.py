from typing import Tuple
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


class RsiDCAShort0(IStrategy):
    timeframe = '1m'
    stoploss = -7
    use_exit_signal = False
    trailing_stop = False
    can_short = True
    can_long = False

    minimal_roi = {"0": 777.0}
    minimal_roi_user_defined = {
        "210": 0.00, "200": 0.020, "190": 0.060, "180": 0.080, "170": 0.100,
        "160": 0.120, "150": 0.140, "140": 0.160, "130": 0.180, "120": 0.200,
        "110": 0.230, "100": 0.260, "90": 0.290, "80": 0.320, "70": 0.350,
        "60": 0.380, "50": 0.400, "40": 0.420, "30": 0.440, "20": 0.460,
        "10": 0.480, "0": 0.500
    }

    position_adjustment_enable = True

    def leverage(self, pair: str, **kwargs) -> float:
        return 50

    def on_trade_open(self, trade: Trade, **kwargs) -> None:
        trade.set_custom_data('dca_count', 0)
        trade.set_custom_data('tp_count', 0)
        trade.set_custom_data('dca_done', False)
        trade.set_custom_data('last_dca_candle', None)
        trade.set_custom_data('last_dca_time', None)
        trade.set_custom_data('dca_reduce_done', False)
        trade.set_custom_data('open_reduce_done', False)
        trade.set_custom_data('need_rebuy70', False)
        trade.set_custom_data('last_tp_time', None)
        trade.set_custom_data('low_margin_start', None)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        upper, mid, lower = ta.BBANDS(
            dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
        )
        dataframe['bb_upperband'] = upper
        dataframe['bb_midband'] = mid
        dataframe['bb_lowerband'] = lower
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_short'] = 0
        dataframe.loc[
            (dataframe['close'] > dataframe['bb_upperband']) &
            (dataframe['rsi'] > 70),
            'enter_short'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_short'] = 0
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):
        if hasattr(current_time, "tzinfo") and current_time.tzinfo is not None:
            current_time = current_time.replace(tzinfo=None)
        open_time = trade.open_date_utc
        if hasattr(open_time, "tzinfo") and open_time.tzinfo is not None:
            open_time = open_time.replace(tzinfo=None)
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
                if current_time >= start_time + timedelta(hours=4):
                    buy_amt = 6 - margin
                    logger.info(
                        f"{YELLOW}保证金={trade.stake_amount:.8f}{RESET},"
                        f"{GREEN}保证金小于5usdt已持续4h，加仓至5usdt{RESET}"
                    )
                    trade.set_custom_data('low_margin_start', None)
                    return buy_amt, 'add_to_5_usdt'
        else:
            trade.set_custom_data('low_margin_start', None)

        n = int(trade.get_custom_data('dca_count') or 0)
        last_dca_time = trade.get_custom_data('last_dca_time')
        reduce6_done = bool(trade.get_custom_data('dca_reduce_done'))
        if n > 0 and last_dca_time and not reduce6_done:
            dca_dt = datetime.fromtimestamp(int(last_dca_time))
            if current_time >= dca_dt + timedelta(hours=8):
                sell_amt = -0.20 * float(trade.stake_amount)
                logger.info(
                    f"[{trade.pair}][6h DCA 后减仓20%] DCA持续 已超6h,减仓20%"
                    f"{YELLOW}保证金={trade.stake_amount:.8f}{RESET},"
                    f"{GREEN}减仓 {abs(sell_amt):.8f} USDT{RESET}"
                )
                trade.set_custom_data('dca_reduce_done', True)
                return sell_amt, "cover20%_postDCA"

        reduce8_done = bool(trade.get_custom_data('open_reduce_done'))
        if n == 0 and not reduce8_done:
            if current_time >= open_time + timedelta(hours=8):
                sell_amt = -0.10 * float(trade.stake_amount)
                logger.info(
                    f"[{trade.pair}][8h 未DCA减仓10%] 未触发Dca 已超8h,减仓10%"
                    f"{YELLOW}保证金={trade.stake_amount:.8f}{RESET},"
                    f"{RESET}减仓 {abs(sell_amt):.8f} USDT{RESET}"
                )
                trade.set_custom_data('open_reduce_done', True)
                return sell_amt, "cover10%_postOpen"

        u = int(trade.get_custom_data('tp_count') or 0)
        dca_done = bool(trade.get_custom_data('dca_done'))
        need_rebuy = bool(trade.get_custom_data('need_rebuy70'))

        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df.empty:
            return None
        last_idx = df.index[-1]
        candle_ts = pd.Timestamp(last_idx).tz_localize('UTC').tz_convert(None).floor('T')

        last_rsi = df['rsi'].iat[-1]

        last_dca = trade.get_custom_data('last_dca_candle')
        last_dca_ts = Timestamp(last_dca, unit='s') if last_dca else None
        if last_dca_ts != candle_ts:
            trade.set_custom_data('dca_done', False)
            dca_done = False

        avg_entry = float(trade.get_custom_data('dynamic_avg_entry') or trade.open_rate)
        n = int(trade.get_custom_data('dca_count') or 0)
        threshold = avg_entry * (1 + 0.005 * (n + 2))

        if not dca_done and current_rate >= threshold and last_rsi > 70:
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
            trade.set_custom_data('dca_count', n + 1)
            trade.set_custom_data('dca_done', True)
            trade.set_custom_data('last_dca_candle', int(candle_ts.timestamp()))
            trade.set_custom_data('last_dca_time', int(current_time.timestamp()))
            trade.set_custom_data('dca_reduce_done', False)
            trade.set_custom_data('open_reduce_done', False)
            trade.set_custom_data('tp_count', 0)
            trade.set_custom_data('dynamic_avg_entry', new_avg_entry)
            logger.info(
                f"[{trade.pair}][浮亏 DCA 加仓] {RED}n=({n}→{n + 1}){RESET}, u=0, "
                f"{YELLOW}保证金={trade.stake_amount:.8f}{RESET}, {GREEN}加仓={buy_amt:.8f}{RESET}"
            )
            return buy_amt, f"dca_n={n + 1}"

        if need_rebuy:
            buy_amt = 0.70 * float(trade.stake_amount)
            logger.info(
                f"[{trade.pair}][分批止盈 Step2 买回70%], u={u}, n={n}，"
                f"{YELLOW}保证金={trade.stake_amount:.8f}{RESET}, {RED}加仓={buy_amt:.8f}{RESET}"
            )
            trade.set_custom_data('need_rebuy70', False)
            trade.set_custom_data('dca_done', False)
            return buy_amt, "resell70"

        if u > 0 and current_profit < 0.01:
            pct = -min(1.0, 0.20 + 0.05 * u)
            sell_amt = pct * float(trade.stake_amount)
            logger.info(
                f"[{trade.pair}][止盈后回撤1%] n={n}, u={u}, "
                f"{YELLOW}保证金={trade.stake_amount:.8f}{RESET}, {RED}减仓={abs(sell_amt):.8f}{RESET}"
            )
            trade.set_custom_data('dca_count', 0)
            trade.set_custom_data('tp_count', 0)
            trade.set_custom_data('dca_done', False)
            trade.set_custom_data('last_tp_time', int(current_time.timestamp()))
            return sell_amt, f"tp_fallback1%_{int(pct * 100)}%"

        last_tp = trade.get_custom_data('last_tp_time')
        base_time = Timestamp(last_tp, unit='s') if last_tp else open_time
        base_time = base_time.tz_localize(None) if base_time.tzinfo else base_time

        elapsed = (current_time - base_time).total_seconds() / 60
        applicable_roi = 0.0
        for k, v in sorted(self.minimal_roi_user_defined.items(), key=lambda x: int(x[0]), reverse=True):
            if elapsed >= int(k):
                applicable_roi = v
                break

        if current_profit >= applicable_roi:
            if n > 0:
                pct = -min(1.0, 0.30 + 0.07 * n)
                sell_amt = pct * float(trade.stake_amount)
                logger.info(
                    f"[{trade.pair}][浮亏 DCA 后止盈] n={n}, u={u}, "
                    f"{YELLOW}保证金={trade.stake_amount:.8f}{RESET}, {RED}卖出={abs(sell_amt):.8f}{RESET}"
                )
                trade.set_custom_data('dca_count', 0)
                trade.set_custom_data('tp_count', 0)
                trade.set_custom_data('dca_done', False)
                trade.set_custom_data('last_tp_time', int(current_time.timestamp()))
                return sell_amt, f"tp_afterDCA_{int(pct * 100)}%"
            if not last_tp or Timestamp(last_tp, unit='s').floor('T') != candle_ts:
                pct = 0.30
                sell_amt = -pct * float(trade.stake_amount)
                new_u = u + 1
                logger.info(
                    f"[{trade.pair}][浮盈减仓 卖30%→后续买70%] n=0,"
                    f"{RED}u=({u}→{new_u}){RESET}, {YELLOW}保证金={trade.stake_amount:.8f}{RESET}, "
                    f"{RED}卖出={abs(sell_amt):.8f}{RESET}"
                )
                trade.set_custom_data('tp_count', new_u)
                trade.set_custom_data('dca_count', 0)
                trade.set_custom_data('dca_done', False)
                trade.set_custom_data('last_tp_time', int(current_time.timestamp()))
                return sell_amt, "tp30"

        return None

    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> None:
        if getattr(order, 'ft_order_tag', None) == "tp30" and order.side == "buy":
            trade.set_custom_data('need_rebuy70', True)
            n = int(trade.get_custom_data('dca_count') or 0)
            u = int(trade.get_custom_data('tp_count') or 0)
            logger.info(
                f"[{pair}][分批止盈 Step1 tp30 已成交] n={n}, u={u}，标记 need_rebuy70=True"
            )

    def custom_stoploss(self, *args, **kwargs) -> float | None:
        return None
