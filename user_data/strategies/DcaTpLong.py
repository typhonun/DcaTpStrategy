from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade, Order
from pandas import DataFrame, Series, Timestamp
import pandas as pd
import talib.abstract as ta
from datetime import datetime, timedelta
import logging
import math

RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
RESET = "\033[0m"
logger = logging.getLogger(__name__)


class DcaTpLong(IStrategy):
    timeframe = '30m'  # 时间周期
    stoploss = -7
    can_short = False
    can_long = True
    use_exit_signal = False
    trailing_stop = False
    position_adjustment_enable = True

    minimal_roi = {"0": 777.0}
    minimal_roi_user_defined = {  # 止盈参数
        "300": 0.010, "290": 0.020, "280": 0.030, "270": 0.040, "260": 0.050,
        "250": 0.060, "240": 0.080, "230": 0.090, "220": 0.100, "210": 0.110,
        "200": 0.120, "190": 0.130, "180": 0.140, "160": 0.145, "140": 0.150,
        "120": 0.155, "100": 0.160, "80": 0.165, "60": 0.170, "50": 0.175,
        "40": 0.180, "30": 0.185, "20": 0.190, "10": 0.195, "0": 0.200,
    }

    def leverage(self, pair: str, **kwargs) -> float:
        return 20

    def on_trade_open(self, trade: Trade, **kwargs) -> None:
        flags = {
            'dca_count': 0, 'tp_count': 0, 'dca_done': False,
            'last_dca_candle': None, 'last_dca_time': None,
            'dca_reduce_done': False, 'open_reduce_done': False,
            'need_rebuy': False, 'last_tp_time': None,
            'low_margin_start': None, 'trend_level': 0,
            'bottom_added': False, 'top_reduced': False,
            'bb_added': False, 'pullback_ready': True,
            'reset_needed': False, 'last_trend_side': None,
            'tp_repull_done': False
        }
        for k, v in flags.items():
            trade.set_custom_data(k, v)
        trade.set_custom_data('dynamic_avg_entry', trade.open_rate)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        upper, mid, lower = ta.BBANDS(dataframe['close'], timeperiod=20)
        dataframe['bb_upperband'] = upper
        dataframe['bb_midband'] = mid
        dataframe['bb_lowerband'] = lower
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)

        df30 = self.dp.get_pair_dataframe(metadata['pair'], '30m')
        if not df30.empty:
            # MACD 参数
            macd, macdsignal, macdhist = ta.MACD(
                df30['close'], fastperiod=8, slowperiod=21, signalperiod=5
            )
            # KDJ 参数
            k, d = ta.STOCH(
                df30['high'], df30['low'], df30['close'],
                fastk_period=5,
                slowk_period=3, slowk_matype=0,
                slowd_period=3, slowd_matype=0
            )
            j = 3 * k - 2 * d
            # EMA 参数
            ema9 = ta.EMA(df30['close'], timeperiod=9)
            ema21 = ta.EMA(df30['close'], timeperiod=21)
            ema99 = ta.EMA(df30['close'], timeperiod=99)
            adx = ta.ADX(df30['high'], df30['low'], df30['close'])

            series_map = {
                'macd_30': macd, 'macdsig_30': macdsignal,
                'macdhist_30': macdhist,
                'k_30': k, 'd_30': d, 'j_30': j,
                'ema9_30': ema9, 'ema21_30': ema21, 'ema99_30': ema99,
                'adx_30': adx
            }
            for name, series in series_map.items():
                dataframe[name] = pd.Series(series, index=df30.index) \
                    .reindex(dataframe.index).ffill()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 趋势入场
        long_cond1 = (
                (dataframe['macd_30'] > dataframe['macdsig_30']) &
                (dataframe['k_30'] > dataframe['d_30']) &
                (dataframe['adx_30'] > 25) &
                (dataframe['ema9_30'] > dataframe['ema21_30']) &
                (dataframe['ema21_30'] > dataframe['ema99_30'])
        )
        # 抄底入场
        long_cond2 = (
                (dataframe['close'] < dataframe['bb_lowerband']) &
                (dataframe['rsi'] < 35)
        )
        dataframe['enter_long'] = (long_cond1 | long_cond2).astype(int)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float, current_profit: float,
                              **kwargs) -> tuple[float, str] | None:
        if trade.has_open_orders:
            return None
        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df.empty:
            return None
        last = df.iloc[-1]
        margin = float(trade.stake_amount)
        if current_time.tzinfo:
            current_time = current_time.replace(tzinfo=None)
        open_time = trade.open_date_utc
        if open_time.tzinfo:
            open_time = open_time.replace(tzinfo=None)
        candle_ts = pd.Timestamp(df.index[-1]).tz_localize(None).floor('T')
        df30, _ = self.dp.get_analyzed_dataframe(trade.pair, '30m')
        if df30.empty:
            return None
        last30_ts = pd.Timestamp(df30.index[-1]).tz_localize(None).floor('T')
        if candle_ts != last30_ts:
            return None

        price = last['close']
        lower = last['bb_lowerband']
        upper = last['bb_upperband']
        mid = last['bb_midband']

        # -- 止盈回落加仓 --
        last_tp_price = trade.get_custom_data('last_tp_price')
        repull_done = bool(trade.get_custom_data('tp_repull_done'))
        if last_tp_price and not repull_done and price <= last_tp_price * 0.99:
            amt = 0.10 * margin
            trade.set_custom_data('tp_repull_done', True)
            logger.info(
                f"{GREEN}[{trade.pair}] 止盈回落加仓10%: "
                f"止盈价={last_tp_price:.4f}, 当前价={price:.4f}, "
                f"保证金={margin:.4f}, 加仓={abs(amt):.4f} USDT{RESET}"
            )
            return amt, 'tp_repull10'

        # -- 趋势加多 --
        level = int(trade.get_custom_data('trend_level') or 0)
        reset_needed = bool(trade.get_custom_data('trend_reset'))
        # 多头信号
        is_bullish_trend = (
                last['macd_30'] > last['macdsig_30'] and
                last['k_30'] > last['d_30'] and
                last['adx_30'] > 25 and
                last['ema9_30'] > last['ema21_30'] > last['ema99_30'] and
                current_profit > 0
        )
        if level == 0 and not reset_needed and is_bullish_trend:
            trade.set_custom_data('trend_level', 2)
            trade.set_custom_data('last_trend_side', 'long')
            amt = 0.6 * margin  # 趋势加仓参数
            logger.info(f"{GREEN}[{trade.pair}] 多头趋势加仓60%{RESET} 保证金={margin:.4f}, 加仓={abs(amt):.4f} USDT")
            return amt, 'trend_add60_bull'
        # KDJ 衰弱减仓
        if level == 2 and last['k_30'] < last['d_30']:
            trade.set_custom_data('trend_level', 0)
            trade.set_custom_data('last_trend_side', 'long')
            amt = -0.3 * margin  # KDJ死叉减仓参数
            logger.info(f"{RED}[{trade.pair}] KDJ 衰弱减仓30%{RESET} 保证金={margin:.4f}, 减仓={abs(amt):.4f} USDT")
            return amt, 'kdj_reduce30_long'

        # # -- 空头信号止损 --
        # last_side = trade.get_custom_data('last_trend_side') or 'none'
        # is_bearish_trend = (
        #         last['macd_30'] < last['macdsig_30'] and
        #         last['k_30'] < last['d_30'] and
        #         last['adx_30'] > 25 and
        #         last['ema9_30'] < last['ema21_30'] < last['ema99_30']
        # )
        # if last_side == 'long' and is_bearish_trend:
        #     trade.set_custom_data('last_trend_side', 'short')
        #     amt = -0.5 * margin
        #     logger.info(
        #         f"{BLUE}[{trade.pair}] 空头信号，多头减仓50%{RESET} 保证金={margin:.4f}, 减仓={abs(amt):.4f} USDT"
        #     )
        #     return amt, 'trend_stop50_long'

        # -- 趋势回撤加仓 --
        high14 = df['close'].rolling(14).max().iat[-1]
        if (last['ema9_30'] > last['ema21_30'] > last['ema99_30']
                and last['close'] == high14):
            trade.set_custom_data('ref_high', float(high14))
            trade.set_custom_data('pullback_done', False)
        ref = trade.get_custom_data('ref_high')
        pb_done = bool(trade.get_custom_data('pullback_done'))
        ready = bool(trade.get_custom_data('pullback_ready'))
        if (ref is not None and ready and not pb_done
                and current_rate <= ref * 0.99
                and last['ema9_30'] > last['ema21_30']):
            # 回撤到高点 99% 且 EMA9 仍在 EMA21 之上时，加仓 30%
            amt = 0.20 * margin  # 回撤加仓参数
            trade.set_custom_data('pullback_done', True)
            trade.set_custom_data('pullback_ready', False)
            logger.info(
                f"{BLUE}[{trade.pair}] 回撤加仓20%: "
                f"高点={ref:.4f}, 当前价={current_rate:.4f} "
                f"保证金={margin:.4f}, 加仓={amt:.4f} USDT{RESET}"
            )
            return amt, 'pullback_dca20'

        # -- 浮亏 DCA 加仓 --
        if df.empty:
            return None
        last_idx = df.index[-1]
        candle_ts = pd.Timestamp(last_idx).tz_localize(None).floor('min')
        last_rsi = df['rsi'].iat[-1]
        u = int(trade.get_custom_data('dca_count') or 0)
        last_dca = trade.get_custom_data('last_dca_candle')
        last_dca_ts = Timestamp(last_dca, unit='s') if last_dca else None
        if last_dca_ts is None or last_dca_ts != candle_ts:
            trade.set_custom_data('dca_done', False)
        dca_done = bool(trade.get_custom_data('dca_done'))
        avg_entry = float(trade.get_custom_data('dynamic_avg_entry') or trade.open_rate)
        threshold = avg_entry * (1 - 0.01 - 0.01 * u)  # Dca加仓价格参数
        # 触发加仓
        rsi_thresh = max(0, 35)  # RSI参数
        if not dca_done and current_rate <= threshold and last_rsi < rsi_thresh:
            buy_amt = 0.2 * margin  # DCA加仓参数
            leverage = self.leverage(trade.pair)
            prev_qty = float(trade.amount)
            prev_cost = prev_qty * avg_entry
            added_qty = (buy_amt * leverage) / current_rate
            new_avg_entry = (prev_cost + buy_amt * leverage) / (prev_qty + added_qty)
            trade.set_custom_data('dca_count', u + 1)
            trade.set_custom_data('dca_done', True)
            trade.set_custom_data('last_dca_candle', int(candle_ts.timestamp()))
            trade.set_custom_data('last_dca_time', int(current_time.timestamp()))
            trade.set_custom_data('dca_reduce_done', False)
            trade.set_custom_data('open_reduce_done', False)
            trade.set_custom_data('tp_count', 0)
            trade.set_custom_data('dynamic_avg_entry', new_avg_entry)
            logger.info(
                f"[{trade.pair}][浮亏 DCA 加仓] {RED}u=({u}→{u + 1}){RESET},RSI<{rsi_thresh} "
                f"{YELLOW}保证金={trade.stake_amount:.8f}{RESET}, {RED}加仓={buy_amt:.8f}{RESET}, "
                f"{BLUE}成交价={threshold:.8f}, 新均价={new_avg_entry:.8f}{RESET}"
            )
            return buy_amt, f"dca_u={u + 1}"

        # -- 浮盈加仓 --
        need_rebuy = bool(trade.get_custom_data('need_rebuy'))
        n = int(trade.get_custom_data('tp_count') or 0)
        if need_rebuy:
            # 连续 2 次及以上的 30% 分批止盈，则本次加仓80%
            if n >= 2:
                buy_amt = 0.8 * margin  # 连续浮盈加仓参数
                tag = 'rebuy80'
                logger.info(
                    f"[{trade.pair}][分批止盈 Step{n + 1} 加仓80%] u={u}, n={n}, "
                    f"{YELLOW}保证金={margin:.4f}{RESET}, {GREEN}加仓={buy_amt:.4f} USDT{RESET}"
                )
            else:
                # 第一次加仓按 60%
                buy_amt = 0.60 * margin  # 首次浮盈加仓参数
                tag = 'rebuy60'
                logger.info(
                    f"[{trade.pair}][分批止盈 Step{n + 1} 加仓60%] u={u}, n={n}, "
                    f"{YELLOW}保证金={margin:.4f}{RESET}, {GREEN}加仓={buy_amt:.4f}{RESET}"
                )
            trade.set_custom_data('need_rebuy', False)
            trade.set_custom_data('dca_done', False)
            return buy_amt, tag

        # -- 止盈后回撤减仓 --
        if n > 0 and current_profit < 0.01:
            pct = min(1.0, 0.20 + 0.05 * n)  # 回撤减仓卖出参数
            sell_amt = -pct * margin
            logger.info(
                f"[{trade.pair}][止盈后回撤1%] u={u}, n={n}, {YELLOW}保证金={margin:.2f}{RESET},"
                f"{GREEN}减仓={abs(sell_amt):.2f}{RESET}")
            trade.set_custom_data('dca_count', 0)
            trade.set_custom_data('tp_count', 0)
            trade.set_custom_data('dca_done', False)
            trade.set_custom_data('last_tp_time', int(current_time.timestamp()))
            return sell_amt, f"tp_fallback1%_{int(pct * 100)}%"

        # -- 分批止盈 --
        last_tp = trade.get_custom_data('last_tp_time')
        base_time = datetime.fromtimestamp(last_tp) if last_tp else open_time
        elapsed = (current_time - base_time).total_seconds() / 60
        roi_target = 0.0
        for k, v in sorted(self.minimal_roi_user_defined.items(), key=lambda x: int(x[0]), reverse=True):
            if elapsed >= int(k):
                roi_target = v
                break
        if current_profit >= roi_target:
            if u > 0:
                pct = min(1.0, 0.3 + 0.1 * u)  # 浮亏止盈卖出参数
                sell_amt = -pct * margin
                logger.info(
                    f"[{trade.pair}][浮亏 DCA 后止盈] u={u}, n={n}, {YELLOW}保证金={margin:.2f}{RESET},"
                    f"{GREEN}减仓={abs(sell_amt):.2f}{RESET}"
                )
                trade.set_custom_data('dca_count', 0)
                trade.set_custom_data('tp_count', 0)
                trade.set_custom_data('dca_done', False)
                trade.set_custom_data('last_tp_time', int(current_time.timestamp()))
                return sell_amt, f"tp_afterDCA_{int(pct * 100)}%"
            else:
                if not last_tp or Timestamp(last_tp, unit='s').floor('T') != candle_ts:
                    sell_amt = -0.30 * margin  # 浮盈止盈卖出参数
                    logger.info(
                        f"[{trade.pair}][浮盈减仓 卖30%→后续加仓60%] u=0, n={n}->{n + 1}, "
                        f"{YELLOW}保证金={margin:.2f}{RESET}, {GREEN}减仓={abs(sell_amt):.2f}{RESET}"
                    )
                    trade.set_custom_data('tp_count', n + 1)
                    trade.set_custom_data('dca_count', 0)
                    trade.set_custom_data('dca_done', False)
                    trade.set_custom_data('last_tp_time', int(current_time.timestamp()))
                    return sell_amt, "tp30"

        # -- 抄底逃顶 --
        if trade.get_custom_data('bottom_added') and price > upper:
            trade.set_custom_data('bottom_added', False)
        if trade.get_custom_data('top_reduced') and price < lower:
            trade.set_custom_data('top_reduced', False)
        # 抄底
        if not trade.get_custom_data('bottom_added') and last['j_30'] < 0 and last['rsi'] < 35:  # KDJ_J&Rsi参数
            trade.set_custom_data('bottom_added', True)
            amt = 0.5 * margin  # 抄底加仓参数
            logger.info(
                f"{BLUE}[{trade.pair}] 抄底加仓50%: J={last['j_30']:.2f}, RSI={last['rsi']:.1f}, "
                f"保证金={margin:.4f}, 加仓={amt:.4f} USDT{RESET}"
            )
            return amt, 'bottom_add50'
        # 逃顶
        if not trade.get_custom_data('top_reduced') and last['j_30'] > 100 and last['rsi'] > 70:  # KDJ_J&Rsi参数
            trade.set_custom_data('top_reduced', True)
            amt = -0.6 * margin  # 逃顶卖出参数
            logger.info(
                f"{RED}[{trade.pair}] 逃顶减仓60%: J={last['j_30']:.2f}, RSI={last['rsi']:.1f}, "
                f"保证金={margin:.4f}, 减仓={abs(amt):.4f} USDT{RESET}"
            )
            return amt, 'top_reduce60'

        # -- 16hDCA减仓 --
        u = int(trade.get_custom_data('dca_count') or 0)
        last_dca_time = trade.get_custom_data('last_dca_time')
        reduce6_done = bool(trade.get_custom_data('dca_reduce_done'))
        if u > 0 and last_dca_time and not reduce6_done:
            dca_dt = datetime.fromtimestamp(int(last_dca_time))
            if current_time >= dca_dt + timedelta(hours=16):  # Dca持续时间参数
                # 价格突破布林带上轨
                if price > upper:
                    amt = -0.30 * margin  # 布林上轨卖出参数
                    logger.info(
                        f"{YELLOW}[{trade.pair}][16h DCA后 · 突破上轨减仓30%] "
                        f"当前价={price:.4f}, 上轨={upper:.4f}, 保证金={margin:.2f}, 减仓={abs(amt):.2f} USDT{RESET}"
                    )
                    trade.set_custom_data('dca_reduce_done', True)
                    return amt, 'reduce300%_postDCA_long'

        if not self.wallets:
            return None
        total_usdt = self.wallets.get_total('USDT')
        margin = float(trade.stake_amount)
        # -- 仓位过小加仓 --
        if total_usdt > 0 and margin < total_usdt * 0.01:  # 仓位下限
            buy_amt = 1.0 * margin  # 小仓位加仓参数
            logger.info(
                f"{GREEN}[{trade.pair}] 保证金过低，当前保证金={margin:.4f} USDT, "
                f"总资产={total_usdt:.4f} USDT，加仓→{buy_amt:.4f} USDT{RESET}"
            )
            return buy_amt, 'add50_low_margin'
        # -- 仓位过大减仓 --
        if total_usdt > 0 and margin > total_usdt * 0.30:  # 仓位上限
            sell_amt = -0.20 * margin  # 大仓位减仓参数
            logger.info(
                f"{YELLOW}[{trade.pair}] 保证金过大，当前保证金={margin:.4f} USDT, "
                f"总资产={total_usdt:.4f} USDT，减仓→{abs(sell_amt):.4f} USDT{RESET}"
            )
            return sell_amt, 'reduce20_over_collateral'

        return None

    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> None:
        if getattr(order, 'ft_order_tag', None) == "tp30" and order.side == "sell":
            trade.set_custom_data('need_rebuy', True)
            trade.set_custom_data('pullback_ready', True)
            trade.set_custom_data('last_tp_price', order.price)
            trade.set_custom_data('tp_repull_done', False)

    def custom_stoploss(self, *args, **kwargs) -> float | None:
        return None
