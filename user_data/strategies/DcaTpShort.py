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


class DcaTpShort1(IStrategy):
    timeframe = '30m'
    stoploss = -7
    can_short = True
    can_long = False
    use_exit_signal = False
    trailing_stop = False
    position_adjustment_enable = True

    minimal_roi = {"0": 777.0}
    minimal_roi_user_defined = {
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
            'top_added': False, 'bottom_reduced': False,
            'bb_added': False, 'pullback_ready_short': True,
            'trend_reset': False, 'last_trend_side': 'none',
            'last_fallback_price_short': None, 'fallback_repull_done_short': False,
            'tp_repull_done': False, 'last_floating_tp_price': None,
            'floating_tp_repull_done': False, 'top_add_state': None,
            'bb24h_first_done': False, 'dca_block_repull': False,
            'last_action_min': int(trade.open_date_utc.timestamp() // 60),
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
        dataframe['vol_ma20'] = dataframe['volume'].rolling(20).mean()
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['atr_ma'] = dataframe['atr'].rolling(14).mean()

        df30 = self.dp.get_pair_dataframe(metadata['pair'], '30m')
        if not df30.empty:
            # MACD 参数
            macd, macdsig, macdhist = ta.MACD(
                df30['close'], fastperiod=8, slowperiod=21, signalperiod=5
            )
            # KDJ 参数
            k, d = ta.STOCH(
                df30['high'], df30['low'], df30['close'],
                fastk_period=5, slowk_period=3, slowd_period=3
            )
            j = 3 * k - 2 * d
            # EMA 参数
            ema9 = ta.EMA(df30['close'], timeperiod=9)
            ema21 = ta.EMA(df30['close'], timeperiod=21)
            ema99 = ta.EMA(df30['close'], timeperiod=99)
            adx = ta.ADX(df30['high'], df30['low'], df30['close'])

            series_map = {
                'macd_30': macd, 'macdsig_30': macdsig,
                'k_30': k, 'd_30': d, 'j_30': j,
                'ema9_30': ema9, 'ema21_30': ema21, 'ema99_30': ema99,
                'adx_30': adx
            }
            for name, series in series_map.items():
                dataframe[name] = pd.Series(series, index=df30.index).reindex(dataframe.index).ffill()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 趋势入场
        short_cond1 = (
                (dataframe['macd_30'] < dataframe['macdsig_30']) &
                (dataframe['k_30'] < dataframe['d_30']) &
                (dataframe['adx_30'] > 25) &
                (dataframe['ema9_30'] < dataframe['ema21_30']) &
                (dataframe['ema21_30'] < dataframe['ema99_30'])
        )
        # vol_ok = dataframe['volume'] > dataframe['vol_ma20']
        # atr_ok = dataframe['atr'] > dataframe['atr_ma']
        # 抄顶入场
        short_cond2 = (
                (dataframe['close'] > dataframe['bb_upperband']) &
                (dataframe['rsi'] > 65)
        )
        dataframe['enter_short'] = (short_cond1 | short_cond2).astype(int)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_short'] = 0
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, **kwargs) -> tuple[float, str] | None:
        if current_time.tzinfo:
            current_time = current_time.replace(tzinfo=None)
        open_time = trade.open_date_utc
        if open_time.tzinfo:
            open_time = open_time.replace(tzinfo=None)

        if trade.has_open_orders:
            return None
        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df.empty:
            return None
        candle_ts = pd.Timestamp(df.index[-1]).tz_localize(None).floor('min')
        last = df.iloc[-1]
        margin = float(trade.stake_amount)

        if not self.wallets:
            return None
        collateral = self.wallets.get_total('USDT')

        def collateral_add(frac: float) -> float:
            return collateral * frac

        # -- 趋势加仓 --
        level = int(trade.get_custom_data('trend_level') or 0)
        reset_needed = bool(trade.get_custom_data('reset_needed'))
        # vol = last['volume']
        # vol_ma20 = last['vol_ma20']
        # atr = last['atr']
        # atr_ma = last['atr_ma']
        # 空头信号
        is_bearish_trend = (
                last['macd_30'] < last['macdsig_30'] and
                last['k_30'] < last['d_30'] and
                last['adx_30'] > 25 and
                last['ema9_30'] < last['ema21_30'] < last['ema99_30'] and
                current_profit > 0
        )
        if level == 0 and not reset_needed and is_bearish_trend:
            trade.set_custom_data('trend_level', 2)
            trade.set_custom_data('last_trend_side', 'short')
            amt = collateral_add(0.02)  # 趋势加仓参数
            logger.info(
                f"{GREEN}[{trade.pair}] 空头趋势加仓 2% {RESET}"
                f"保证金={margin:.4f}, 加仓={abs(amt):.4f} USDT{RESET}"
            )
            return amt, 'trend_add20_bear'
        # KDJ 衰弱减仓
        if level == 2 and last['k_30'] > last['d_30']:
            trade.set_custom_data('trend_level', 0)
            trade.set_custom_data('trend_reset', True)
            trade.set_custom_data('last_trend_side', 'short')
            amt = -0.4 * margin  # KDJ死叉减仓参数
            logger.info(
                f"{RED}[{trade.pair}] KDJ 衰弱减仓{RESET}"
                f"保证金={margin:.4f}, 减仓={abs(amt):.4f} USDT{RESET}"
            )
            return amt, 'kdj_reduce40_short'

        # # -- 多头信号止损 --
        # last_side = trade.get_custom_data('last_trend_side') or 'none'
        # is_bullish_trend = (
        #         last['macd_30'] > last['macdsig_30'] and
        #         last['k_30'] > last['d_30'] and
        #         last['adx_30'] > 25 and
        #         last['ema9_30'] > last['ema21_30'] > last['ema99_30']
        # )
        # if last_side == 'short' and is_bullish_trend:
        #     trade.set_custom_data('last_trend_side', 'long')
        #     amt = -0.5 * margin
        #     logger.info(
        #         f"{BLUE}[{trade.pair}] 多头信号，空头减仓{RESET} 保证金={margin:.4f}, 减仓={abs(amt):.4f} USDT"
        #     )
        #     return amt, 'trend_stop50_short'

        # -- 趋势反弹加仓 --
        low14 = df['close'].rolling(14).min().iat[-1]
        if (last['ema9_30'] < last['ema21_30'] < last['ema99_30']
                and last['close'] == low14):
            trade.set_custom_data('ref_low', float(low14))
            trade.set_custom_data('pullback_done_short', False)
        ref_low = trade.get_custom_data('ref_low')
        done = bool(trade.get_custom_data('pullback_done_short'))
        pullback_ready = bool(trade.get_custom_data('pullback_ready_short'))
        # 当前价回升到低点 101% 时，且 EMA9 仍在 EMA21 之下
        if (ref_low is not None and pullback_ready and not done
                and current_rate >= ref_low * 1.01
                and last['ema9_30'] < last['ema21_30']):
            amt = collateral_add(0.02)  # 反弹加仓参数
            trade.set_custom_data('pullback_done_short', True)
            trade.set_custom_data('pullback_ready_short', False)
            logger.info(
                f"{RED}[{trade.pair}] 空头回撤加仓 2%"
                f"低点={ref_low:.4f}, 当前价={current_rate:.4f}, "
                f"保证金={margin:.4f}, 加仓={abs(amt):.4f} USDT{RESET}"
            )
            return amt, 'short_pullback_dca20'

        # -- 浮亏 DCA 加仓 --
        last_rsi2 = last['rsi']  # same df

        def get_cd(key, default=None):
            v = trade.get_custom_data(key)
            return default if v is None or (isinstance(v, str) and v.lower() == 'null') else v

        last_dca = get_cd('last_dca_candle')
        last_dca_ts = Timestamp(last_dca, unit='s') if isinstance(last_dca, (int, float)) else None
        if last_dca_ts != candle_ts:
            trade.set_custom_data('dca_done', False)
        dca_done = bool(get_cd('dca_done', False))
        avg = float(get_cd('dynamic_avg_entry', trade.open_rate))
        u = int(get_cd('dca_count', 0))
        threshold = avg * (1 + 0.01 + 0.01 * u)  # Dca加仓价格参数
        rsi_thresh = max(0, 65)  # RSI参数
        # 触发加仓
        if not dca_done and current_rate >= threshold and last_rsi2 > rsi_thresh:
            amt = collateral_add(0.02)  # DCA加仓参数
            leverage = self.leverage(trade.pair)
            prev_qty = abs(float(trade.amount))
            prev_cost = prev_qty * avg
            added_qty = (abs(amt) * leverage) / current_rate
            new_avg = (prev_cost + abs(amt) * leverage) / (prev_qty + added_qty)
            trade.set_custom_data('dca_count', u + 1)
            trade.set_custom_data('dca_done', True)
            trade.set_custom_data('last_dca_candle', int(candle_ts.timestamp()))
            trade.set_custom_data('last_dca_time', int(current_time.timestamp()))
            trade.set_custom_data('dca_reduce_done', False)
            trade.set_custom_data('open_reduce_done', False)
            trade.set_custom_data('tp_count', 0)
            trade.set_custom_data('dynamic_avg_entry', new_avg)
            trade.set_custom_data('dca_block_repull', True)
            trade.set_custom_data('last_action_min', int(current_time.timestamp() // 60))
            trade.set_custom_data('bb24h_first_done', False)
            logger.info(
                f"[{trade.pair}][浮亏 DCA 加仓] u={u}->{u + 1}, "
                f"{YELLOW}保证金={trade.stake_amount:.8f}{RESET}{RED}加仓={abs(amt):.8f}{RESET}, "
                f"{BLUE}新均价={new_avg:.8f}{RESET}"
            ),
            return amt, f"dca_u_short={u + 1}"

        # -- 浮盈加仓 --
        price = last['close']
        ema20_arr = ta.EMA(df['close'], timeperiod=20)
        ema20 = float(ema20_arr[-1])
        need_rebuy = bool(trade.get_custom_data('need_rebuy'))
        n = int(trade.get_custom_data('tp_count') or 0)
        if need_rebuy:
            base_frac = 0.03  # 浮盈加仓参数
            # 如果价格偏离 20EMA 超过 5%，加仓量减半
            if price < ema20 * 0.95:
                base_frac /= 2
                tag = 'rebuy15_ema'
                logger.info(f"价格过于偏离，可能触底，浮盈加仓量减半")
            else:
                tag = 'rebuy30'
            amt = collateral_add(base_frac)
            logger.info(
                f"[{trade.pair}][分批止盈加仓] base={base_frac * 100:.1f}% "
                f"(ema20={ema20:.4f}), 当前价={price:.4f},"
                f"{YELLOW}保证金={margin:.4f}{RESET}, {GREEN}加仓={amt:.4f}{RESET}"
            )
            trade.set_custom_data('need_rebuy', False)
            trade.set_custom_data('dca_done', False)
            return amt, tag

        # -- 止盈后回撤减仓逻辑 --
        if n > 0 and current_profit < 0.01:
            if n >= 6:
                pct = 0.8
            else:
                pct = min(1.0, 0.50 + 0.05 * n)
            amt = -pct * margin
            trade.set_custom_data('last_fallback_price_short', price)
            trade.set_custom_data('fallback_ready_short', True)
            trade.set_custom_data('fallback_repull_done_short', False)
            logger.info(
                f"{YELLOW}[{trade.pair}] 止盈后回撤减仓{int(abs(pct) * 100)}%: "
                f"回撤价={price:.4f}, 保证金={margin:.4f}, 减仓={abs(amt):.4f} USDT{RESET}"
            )
            trade.set_custom_data('dca_count', 0)
            trade.set_custom_data('tp_count', 0)
            trade.set_custom_data('dca_done', False)
            trade.set_custom_data('last_tp_time', int(current_time.timestamp()))
            return amt, f"tp_fallback1%_{int(abs(pct) * 100)}%_short"

        # -- 止盈反弹加仓 --
        last_tp_price = trade.get_custom_data('last_tp_price')
        repull_done = bool(trade.get_custom_data('tp_repull_done'))
        if last_tp_price and not repull_done and price >= last_tp_price * 1.01:
            amt = collateral_add(0.02)
            trade.set_custom_data('tp_repull_done', True)
            logger.info(
                f"{GREEN}[{trade.pair}] 止盈反弹加仓总资金 2%: "
                f"止盈价={last_tp_price:.4f}, 当前价={price:.4f}, "
                f"保证金={margin:.4f}, 加仓={abs(amt):.4f} USDT{RESET}"
            )
            return amt, 'tp_repull20_tp'

        # -- 回撤价反弹加仓 --
        last_fb = trade.get_custom_data('last_fallback_price_short')
        fallback_ready = bool(trade.get_custom_data('fallback_ready_short'))
        done = bool(trade.get_custom_data('fallback_repull_done_short'))
        # 当价格反弹到回撤价的 1.01 时加仓
        if last_fb and fallback_ready and not done and price >= last_fb * 1.01:  # 反弹价格参数
            amt = collateral_add(0.02)  # 反弹加仓参数
            trade.set_custom_data('fallback_repull_done_short', True)
            trade.set_custom_data('fallback_ready_short', False)
            logger.info(
                f"{GREEN}[{trade.pair}] 回撤价反弹加仓 2%: "
                f"回撤价={last_fb:.4f}, 当前价={price:.4f}, "
                f"保证金={margin:.4f}, 加仓={abs(amt):.4f} USDT{RESET}"
            )
            return amt, 'tp_repull20_short'

        # -- 浮亏止盈后反弹加仓 --
        last_floating_tp_price = trade.get_custom_data('last_floating_tp_price')
        floating_done = bool(trade.get_custom_data('floating_tp_repull_done'))
        if last_floating_tp_price and not floating_done and price >= last_floating_tp_price * 1.01:
            amt = collateral_add(0.02)
            trade.set_custom_data('floating_tp_repull_done', True)
            logger.info(
                f"{GREEN}[{trade.pair}] 浮亏止盈反弹加仓 2%: "
                f"浮亏止盈价={last_floating_tp_price:.4f}, 当前价={price:.4f}, "
                f"保证金={margin:.4f}, 加仓={abs(amt):.4f} USDT{RESET}"
            )
            return amt, 'floating_tp_repull20'

        # -- 分批止盈 --
        last_tp = trade.get_custom_data('last_tp_time')
        base = datetime.fromtimestamp(last_tp) if last_tp else open_time
        elapsed = (current_time - base).total_seconds() / 60
        roi = 0.0
        for k, v in sorted(self.minimal_roi_user_defined.items(), key=lambda x: int(x[0]), reverse=True):
            if elapsed >= int(k):
                roi = v
                break
        if current_profit >= roi:
            if u > 0:
                if u >= 6 and n >= 6:
                    pct = 0.8
                else:
                    pct = min(1.0, 0.50 + 0.05 * u)  # 浮亏止盈卖出参数
                amt = -pct * margin
                logger.info(
                    f"[{trade.pair}][浮亏 DCA 后止盈] u={u}, n={n}, {YELLOW}保证金={margin:.2f}{RESET},"
                    f"{GREEN}减仓={abs(amt):.2f}{RESET}"
                )
                trade.set_custom_data('last_floating_tp_price', current_rate)
                trade.set_custom_data('floating_tp_repull_done', False)
                trade.set_custom_data('dca_count', 0)
                trade.set_custom_data('tp_count', 0)
                trade.set_custom_data('dca_done', False)
                trade.set_custom_data('last_tp_time', int(current_time.timestamp()))
                trade.set_custom_data('last_action_min', int(current_time.timestamp() // 60))
                trade.set_custom_data('bb24h_first_done', False)
                return amt, f"tp_afterDCA_short_u{u}"
            else:
                if not last_tp or Timestamp(last_tp, unit='s').floor('T') != candle_ts:
                    amt = -0.30 * margin  # 浮盈止盈卖出参数
                    logger.info(
                        f"[{trade.pair}][浮盈减仓 卖30%→后续加仓 3%] u=0, n={n}->{n + 1}, "
                        f"{YELLOW}保证金={margin:.2f}{RESET}, {GREEN}减仓={abs(amt):.2f}{RESET}"
                    )
                    trade.set_custom_data('tp_count', n + 1)
                    trade.set_custom_data('dca_count', 0)
                    trade.set_custom_data('dca_done', False)
                    trade.set_custom_data('last_tp_time', int(current_time.timestamp()))
                    trade.set_custom_data('last_action_min', int(current_time.timestamp() // 60))
                    trade.set_custom_data('bb24h_first_done', False)
                    return amt, 'tp30'

        # --24h无止盈或加仓，触及布林下轨时，加仓--
        lower = last['bb_lowerband']
        upper = last['bb_upperband']
        mid = last['bb_midband']

        last_action_min = int(trade.get_custom_data('last_action_min') or (open_time.timestamp() // 60))
        current_min = int(current_time.timestamp() // 60)
        elapsed_minutes = current_min - last_action_min
        bb24h_first_done = bool(trade.get_custom_data('bb24h_first_done'))

        # 当超过 1440 分钟 且 触及布林上轨时触发加仓
        if elapsed_minutes >= 1440 and price >= upper:
            if not bb24h_first_done:
                amt = collateral_add(0.02)
                trade.set_custom_data('bb24h_first_done', True)
                tag = 'bb24h_add20'
                note = "首次 0.02"
            else:
                amt = collateral_add(0.01)
                tag = 'bb24h_add10'
                note = "后续 0.01"
            trade.set_custom_data('last_action_min', current_min)
            logger.info(
                f"{BLUE}[{trade.pair}] 24h 无止盈或加仓，布林上轨加仓 ({note}): lower={lower:.4f}, price={price:.4f}, "
                f"保证金={margin:.4f}, 加仓={amt:.4f} USDT, tag={tag}{RESET}"
            )
            return amt, tag

        # -- 抄顶逃底 --
        if trade.get_custom_data('top_added') and price < lower:
            trade.set_custom_data('top_added', False)
        if trade.get_custom_data('bottom_reduced') and price > upper:
            trade.set_custom_data('bottom_reduced', False)

        top_state = trade.get_custom_data('top_add_state')
        if (not trade.get_custom_data('top_added')) and last['j_30'] > 100 and last['rsi'] > 65:
            if top_state in ('first', 'repeat'):
                amt = collateral_add(0.01)
                trade.set_custom_data('top_add_state', 'repeat')
                tag = 'top_add10'
                info_note = "连续抄顶加仓 0.01"
            else:
                amt = collateral_add(0.02)
                trade.set_custom_data('top_add_state', 'first')
                tag = 'top_add20'
                info_note = "首次抄顶加仓 0.02"

            trade.set_custom_data('top_added', True)
            trade.set_custom_data('last_action_min', int(current_time.timestamp() // 60))
            logger.info(
                f"{BLUE}[{trade.pair}] {info_note}: J={last['j_30']:.2f}, RSI={last['rsi']:.1f}, "
                f"保证金={margin:.4f}, 加仓={abs(amt):.4f} USDT{RESET}"
            )
            return amt, tag
        # 逃底（触发后重置抄顶状态，下一次抄顶加仓0.02）
        if (not trade.get_custom_data('bottom_reduced')) and current_profit > 0 and last['j_30'] < 0 and last[
            'rsi'] < 35:
            trade.set_custom_data('bottom_reduced', True)
            trade.set_custom_data('top_add_state', None)
            amt = -0.5 * margin
            logger.info(
                f"{RED}[{trade.pair}] 逃底减仓50%: J={last['j_30']:.2f}, RSI={last['rsi']:.1f}, "
                f"保证金={margin:.4f}, 回补={abs(amt):.4f} USDT{RESET}"
            )
            return amt, 'bottom_cover50'
        # 逃底后反弹至布林中轨，加仓
        if trade.get_custom_data('bottom_reduced') and price >= mid:
            trade.set_custom_data('bottom_reduced', False)
            amt = collateral_add(0.02)
            logger.info(
                f"{GREEN}[{trade.pair}] 逃底后反弹至中轨，加仓 2%: 当前价={price:.4f}, 布林中轨={mid:.4f}, "
                f"保证金={margin:.4f}, 加仓={abs(amt):.4f} USDT{RESET}"
            )
            return amt, 'rebound_add20_short'

        # # -- 24hDCA减仓 --
        # u = int(trade.get_custom_data('dca_count') or 0)
        # last_dca_time = trade.get_custom_data('last_dca_time')
        # reduce6_done = bool(trade.get_custom_data('dca_reduce_done'))
        # if u > 0 and last_dca_time and not reduce6_done:
        #     dca_dt = datetime.fromtimestamp(int(last_dca_time))
        #     # 已超过24h
        #     if current_time >= dca_dt + timedelta(hours=24):  # Dca持续时间参数
        #         lower = last['bb_lowerband']
        #         price = last['close']
        #         # 价格跌破布林带下轨
        #         if price < lower:
        #             amt = -0.30 * margin  # 布林下轨卖出参数
        #             logger.info(
        #                 f"{YELLOW}[{trade.pair}][24h DCA后 · 跌破下轨减仓30%] "
        #                 f"当前价={price:.4f}, 下轨={lower:.4f}, 保证金={margin:.2f}, 减仓={abs(amt):.2f} USDT{RESET}"
        #             )
        #             trade.set_custom_data('dca_reduce_done', True)
        #             return amt, 'reduce30%_postDCA_short'

        # # -- 仓位过小加仓 --
        # if collateral > 0 and margin < collateral * 0.01:  # 仓位下限
        #     buy_amt = 1.0 * margin  # 小仓位加仓参数
        #     logger.info(
        #         f"{GREEN}[{trade.pair}] 保证金过低，当前保证金={margin:.4f} USDT, "
        #         f"总资产={collateral:.4f} USDT，加仓→{buy_amt:.4f} USDT{RESET}"
        #     )
        #     return buy_amt, 'add50_low_margin'
        # # -- 仓位过大减仓 --
        # if collateral > 0 and margin > collateral * 0.30:  # 仓位上限
        #     sell_amt = -0.30 * margin  # 大仓位减仓参数
        #     logger.info(
        #         f"{YELLOW}[{trade.pair}] 保证金过大，当前保证金={margin:.4f} USDT, "
        #         f"总资产={collateral:.4f} USDT，减仓→{abs(sell_amt):.4f} USDT{RESET}"
        #     )
        #     return sell_amt, 'reduce20_over_collateral'

        return None

    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> None:
        tag = getattr(order, 'ft_order_tag', None)
        if tag == "tp30":
            trade.set_custom_data('need_rebuy', True)
            trade.set_custom_data('last_tp_price', order.price)
            trade.set_custom_data('tp_repull_done', False)
            trade.set_custom_data('fallback_repull_done_short', False)
            trade.set_custom_data('pullback_ready_short', True)

    def custom_stoploss(self, *args, **kwargs) -> float | None:
        return None