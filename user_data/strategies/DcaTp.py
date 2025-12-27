from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade, Order
from pandas import DataFrame, Timestamp
import pandas as pd
from typing import Any, Optional
import talib.abstract as ta
from datetime import datetime
import logging

RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
RESET = "\033[0m"
logger = logging.getLogger(__name__)


class DcaTp(IStrategy):
    timeframe = '30m'
    stoploss = -9
    can_short = False
    can_long = True
    use_exit_signal = False
    trailing_stop = False
    position_adjustment_enable = True

    minimal_roi = {"0": 999.0}
    minimal_roi_user_defined = {
        "300": 0.05, "270": 0.06, "240": 0.07, "210": 0.08,
        "180": 0.09, "150": 0.10, "120": 0.11, "90": 0.12,
        "60": 0.13, "30": 0.14, "0": 0.15,
    }

    def leverage(self, pair: str, **kwargs) -> float:
        return 10

    def on_trade_open(self, trade: Trade, **kwargs) -> None:
        flags = {
            'dca_count': 0, 'tp_count': 0, 'dca_done': False,
            'last_dca_candle': None, 'last_dca_time': None,
            'need_rebuy': False, 'last_tp_time': None,
            'trend_level': 0,
            'reset_needed': False,
        }

        for k, v in flags.items():
            self._set(trade, k, v)
        self._set(trade, 'dynamic_avg_entry', float(trade.open_rate or 0.0))
        base_avg = float(trade.open_rate or 0.0)
        self._set(trade, 'grid_upper', float(base_avg * 1.02))
        self._set(trade, 'grid_lower', float(base_avg * 0.98))
        self._set(trade, 'last_grid_action_time', None)
        self._set(trade, 'grid_count', 0)
        self._set(trade, 'grid_added_total_usdt', 0.0)
        self._set(trade, 'grid_added_total_qty', 0.0)
        self._set(trade, 'grid_reduced_total_usdt', 0.0)

    def _serial(self, v):
        if v is None:
            return None

        if isinstance(v, (pd.Timestamp, Timestamp, datetime)):
            return int(pd.Timestamp(v).timestamp())

        try:
            import numpy as np
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            if isinstance(v, np.bool_):
                return bool(v)
        except ImportError:
            pass

        if isinstance(v, (int, float, bool)):
            return v

        if isinstance(v, str):
            s = v.strip()
            if s.lower() in ("", "null", "none", "nan", "na", "n/a"):
                return None
            return s
        return str(v)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        upper, mid, lower = ta.BBANDS(dataframe['close'], timeperiod=20)
        dataframe['bb_upperband'] = upper
        dataframe['bb_midband'] = mid
        dataframe['bb_lowerband'] = lower
        # rsi
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)

        df30 = self.dp.get_pair_dataframe(metadata['pair'], '30m')

        if not df30.empty:
            # macd
            macd, macdsignal, macdhist = ta.MACD(df30['close'], fastperiod=8, slowperiod=21, signalperiod=5)
            # kdj
            k, d = ta.STOCH(df30['high'], df30['low'], df30['close'],
                            fastk_period=5, slowk_period=3, slowk_matype=0,
                            slowd_period=3, slowd_matype=0)
            j = 3 * k - 2 * d
            # ema
            ema9 = ta.EMA(df30['close'], timeperiod=9)
            ema21 = ta.EMA(df30['close'], timeperiod=21)
            ema99 = ta.EMA(df30['close'], timeperiod=99)
            adx = ta.ADX(df30['high'], df30['low'], df30['close'])

            series_map = {
                'macd_30': macd, 'macdsig_30': macdsignal, 'macdhist_30': macdhist,
                'k_30': k, 'd_30': d, 'j_30': j,
                'ema9_30': ema9, 'ema21_30': ema21, 'ema99_30': ema99,
                'adx_30': adx
            }
            for name, series in series_map.items():
                dataframe[name] = pd.Series(series, index=df30.index).reindex(dataframe.index).ffill()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_cond = (
                (dataframe['close'] < dataframe['bb_lowerband']) &
                (dataframe['rsi'] < 35)
        )
        dataframe['enter_long'] = long_cond.astype(int)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe

    def _clean_val(self, v: Any) -> Optional[Any]:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            if not s or s.lower() in ("null", "none", "nan", "na", "n/a"):
                return None
            return s
        try:
            import numpy as np
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            if isinstance(v, np.bool_):
                return bool(v)
        except ImportError:
            pass
        if isinstance(v, (int, float, bool)):
            return v
        return str(v)

    def _get(self, trade: Trade, key: str, default: Any = None) -> Any:
        try:
            raw = trade.get_custom_data(key)
        except Exception:
            raw = None
        val = self._clean_val(raw)
        return default if val is None else val

    def _get_float(self, trade: Trade, key: str, default: float = 0.0) -> float:
        v = self._get(trade, key, None)
        if v is None:
            return default
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        try:
            f = float(v)
            if not (float('-inf') < f < float('inf')) or pd.isna(f):
                return default
            return f
        except Exception:
            return default

    def _get_int(self, trade: Trade, key: str, default: int = 0) -> int:
        v = self._get(trade, key, None)
        if v is None:
            return default
        if isinstance(v, bool):
            return int(v)
        try:
            return int(float(v))
        except Exception:
            return default

    def _get_bool(self, trade: Trade, key: str, default: bool = False) -> bool:
        v = self._get(trade, key, None)
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        s = str(v).strip().lower()
        if s in ("1", "true", "t", "yes", "y"):
            return True
        if s in ("0", "false", "f", "no", "n"):
            return False
        try:
            return bool(float(s))
        except Exception:
            return default

    def _set(self, trade: Trade, key: str, value: Any) -> None:
        trade.set_custom_data(key, self._serial(value))

    def _collateral_total(self) -> float:
        try:
            return float(self.wallets.get_total('USDT'))
        except Exception:
            return 0.0

    def _qty_from_usdt(self, usdt: float, price: float, lev: float) -> float:
        try:
            return (usdt * float(lev)) / float(price) if price and lev else 0.0
        except Exception:
            return 0.0

    def _record_add_usdt(self, trade: Trade, key: str, usdt: float) -> None:
        prev = self._get_float(trade, key, 0.0)
        try:
            add = float(usdt)
            if pd.isna(add) or not (float('-inf') < add < float('inf')):
                add = 0.0
        except Exception:
            add = 0.0
        self._set(trade, key, prev + add)

    def _record_add_qty(self, trade: Trade, key: str, qty: float) -> None:
        prev = self._get_float(trade, key, 0.0)
        try:
            add = float(qty)
            if pd.isna(add) or not (float('-inf') < add < float('inf')):
                add = 0.0
        except Exception:
            add = 0.0
        self._set(trade, key, prev + add)

    def _record_add(self, trade: Trade, key: str, value: float) -> None:
        prev = self._get_float(trade, key, 0.0)
        try:
            add = float(value)
            if pd.isna(add) or not (float('-inf') < add < float('inf')):
                add = 0.0
        except Exception:
            add = 0.0
        self._set(trade, key, prev + add)

    def trend_add(self, trade: Trade, last: dict, margin: float, collateral: float, current_time: datetime):
        level = self._get_int(trade, 'trend_level', 0)
        reset_needed = self._get_bool(trade, 'reset_needed', False)
        u = self._get_int(trade, 'dca_count', 0)

        is_bullish_trend = (
                last.get('macd_30', 0) > last.get('macdsig_30', 0) and
                last.get('k_30', 0) > last.get('d_30', 0) and
                last.get('adx_30', 0) > 25 and
                last.get('ema9_30', 0) > last.get('ema21_30', 0) > last.get('ema99_30', 0)
        )
        price = last.get('close', None)

        def _add_trend_total_usdt(x):
            self._record_add_usdt(trade, 'trend_total_added_usdt', x)

        if level == 0 and u == 0 and (not reset_needed) and price is not None and is_bullish_trend:
            self._set(trade, 'trend_level', 1)
            self._set(trade, 'kdj_reduced', False)
            amt = collateral * 0.02  # 趋势加仓
            _add_trend_total_usdt(abs(amt))

            avg_price = getattr(trade, 'open_rate', None)
            if avg_price is None:
                avg_price = 0.0

            logger.info(f"[{trade.pair}] {GREEN}[多头趋势加仓 2%]{RESET}, "
                        f"{YELLOW} 保证金={margin:.2f}{RESET}, "f"当前价={price:.4f}, "
                        f"加仓={abs(amt):.4f} USDT, 新均价={avg_price:.4f}")
            return float(amt), 'trend_add20_bull'

        if level == 1:
            kdj_reduced = self._get_bool(trade, 'kdj_reduced', False)
            if (not kdj_reduced) and last.get('k_30', 0) < last.get('d_30', 0):
                total_trend_usdt = self._get_float(trade, 'trend_total_added_usdt', 0.0) or 0.0
                sell_usdt = min(total_trend_usdt, margin)
                amt = -float(sell_usdt)
                self._set(trade, 'trend_total_added_usdt', 0.0)
                self._set(trade, 'trend_level', 0)
                self._set(trade, 'kdj_reduced', True)

                logger.info(f"[{trade.pair}] {RED}[KDJ 减仓]{RESET}, "
                            f"{YELLOW}保证金={margin:.2f}{RESET}, 当前价={last.get('close'):.4f}, "
                            f"减仓={abs(amt):.4f} USDT")
                return float(amt), 'kdj_reduce_by_trend_added'

    def dca_add(self, trade: Trade, df: DataFrame, last: dict, current_time: datetime, current_rate: float):
        last_idx = df.index[-1]
        candle_ts = pd.Timestamp(last_idx).tz_localize(None).floor('min')
        last_rsi = df['rsi'].iat[-1] if 'rsi' in df.columns else None
        last_j = df['j_30'].iat[-1] if 'j_30' in df.columns else None

        u = self._get_int(trade, 'dca_count', 0)
        last_dca = self._get_int(trade, 'last_dca_candle', 0)
        last_dca_ts = Timestamp(int(last_dca), unit='s') if last_dca else None
        if last_dca_ts is None or last_dca_ts != candle_ts:
            self._set(trade, 'dca_done', False)
        dca_done = self._get_bool(trade, 'dca_done', False)

        within_dca_cooldown = False
        last_dca_time = self._get_int(trade, 'last_dca_time', 0)
        if last_dca_time:
            try:
                last_dca_dt = datetime.fromtimestamp(int(last_dca_time))
                if (current_time - last_dca_dt).total_seconds() < 60 * 60:
                    within_dca_cooldown = True
            except Exception:
                within_dca_cooldown = False

        max_dca = 5  # dca 上限
        if u >= max_dca:
            logger.debug(f"[{trade.pair}] 已达 DCA 次数上限 u={u}，跳过浮亏 DCA")
            return None

        avg_entry = self._get_float(trade, 'dynamic_avg_entry', float(trade.open_rate or 0.0)) or float(
            trade.open_rate or 0.0)
        try:
            lev = float(self.leverage(trade.pair) or 1.0)
        except Exception:
            lev = 1.0

        threshold = avg_entry * (1 - 0.01 - 0.02 * u)  # dca 阈值
        cond_j_lt_0 = (last_j is not None and last_j < 0)
        cond_rsi_lt_20 = (last_rsi is not None and last_rsi < 20)
        cond_j_lt_20_and_rsi_lt_35 = (last_j is not None and last_j < 20 and last_rsi is not None and last_rsi < 35)
        triggered_cond = None
        if cond_j_lt_0:
            triggered_cond = "J<0"
        elif cond_rsi_lt_20:
            triggered_cond = "RSI<20"
        elif cond_j_lt_20_and_rsi_lt_35:
            triggered_cond = "J<20 & RSI<35"

        try:
            candle_close = float(last.get('close')) if last.get('close') is not None else None
        except Exception:
            candle_close = None

        logger.debug(
            f"[{trade.pair}] DCA check: u={u}, avg_entry={avg_entry:.8f}, threshold={threshold:.8f}, candle_close={candle_close}, triggered_cond={triggered_cond}")

        if within_dca_cooldown:
            logger.debug(f"[{trade.pair}] 浮亏 DCA 冷却中")
            return None
        if dca_done:
            logger.debug(f"[{trade.pair}] 本根 K 已执行 DCA，跳过")
            return None

        if (candle_close is not None) and (triggered_cond is not None) and (candle_close <= threshold):
            collateral = self._collateral_total()
            try:
                margin = float(trade.stake_amount)
            except Exception:
                margin = 0.0

            pct = 0.02 - 0.002 * u
            buy_amt = collateral * pct  # dca 加仓
            try:
                prev_qty = float(trade.amount)
            except Exception:
                prev_qty = 0.0
            prev_cost = prev_qty * avg_entry
            added_qty = self._qty_from_usdt(buy_amt, candle_close, lev)
            new_avg_entry = avg_entry
            if (prev_qty + added_qty) > 0:
                new_avg_entry = (prev_cost + buy_amt * lev) / (prev_qty + added_qty)

            self._set(trade, 'dca_count', u + 1)
            self._set(trade, 'dca_done', True)
            self._set(trade, 'last_dca_candle', int(candle_ts.timestamp()))
            self._set(trade, 'last_dca_time', int(current_time.timestamp()))
            self._set(trade, 'tp_count', 0)
            self._set(trade, 'dynamic_avg_entry', float(new_avg_entry))

            logger.info(
                f"[{trade.pair}] {RED}[浮亏 DCA 加仓 2%]{RESET}, u=({u}->{u + 1}), "
                f"触发条件={triggered_cond}, 触发价={threshold:.4f}, 收盘价={candle_close:.4f}, "
                f"{YELLOW}保证金={margin:.4f}{RESET}, 加仓={buy_amt:.4f} USDT, 新均价={new_avg_entry:.4f}"
            )
            return float(buy_amt), f"dca_u={u + 1}"
        return None

    def profit_rebuy(self, trade: Trade, last: dict, collateral: float):
        need_rebuy = self._get_bool(trade, 'need_rebuy', False)
        if not need_rebuy:
            return None

        price = last.get('close', None)
        if price is None:
            return None

        base_frac = 0.05  # 浮盈加仓
        final_frac = base_frac
        margin = float(trade.stake_amount)
        buy_amt = collateral * final_frac
        n = self._get_int(trade, 'tp_count', 0)

        self._set(trade, 'need_rebuy', False)
        self._set(trade, 'dca_done', False)
        self._set(trade, 'tp_count', n + 1)
        avg_price = getattr(trade, 'open_rate', None)
        if avg_price is None:
            avg_price = 0.0

        logger.info(
            f"[{trade.pair}] {GREEN}[浮盈加仓 5%]{RESET}, n={n}->{n + 1}, "
            f"{YELLOW}保证金={margin:.2f}{RESET}, 当前价={price:.4f}, "
            f"加仓={buy_amt:.4f} USDT, 新均价={avg_price:.4f}"
        )
        return float(buy_amt), 'rebuy_merged'

    def fallback_reduce(self, trade: Trade, last: dict, current_profit: float, margin: float,
                        current_rate: float, current_time: datetime):
        n = self._get_int(trade, 'tp_count', 0)
        if not (n > 0 and current_profit < 0.01):  # 回撤阈值
            return None

        if getattr(current_time, 'tzinfo', None):
            current_time = current_time.replace(tzinfo=None)

        collateral = self._collateral_total()
        try:
            current_margin = float(getattr(trade, 'stake_amount', margin or 0.0))
        except Exception:
            current_margin = float(margin or 0.0)

        try:
            current_qty = float(trade.amount)
        except Exception:
            current_qty = 0.0

        try:
            price = float(current_rate) if current_rate is not None else None
        except Exception:
            price = None

        try:
            lev = float(self.leverage(trade.pair) or 1.0)
        except Exception:
            lev = 1.0

        if price is None or price <= 0 or lev <= 0:
            logger.debug(f"[{trade.pair}] 回撤减仓: 无效 price/lev (price={price}, lev={lev}), 跳过")
            return None

        try:
            current_value_usdt = (current_qty * price) / lev
        except Exception:
            current_value_usdt = 0.0

        target_usdt = max(0.0, float(collateral) * 0.05)  # 剩余仓位

        if current_value_usdt > target_usdt:
            sell_usdt = current_value_usdt - target_usdt
            sell_usdt_capped = min(sell_usdt, max(0.0, current_margin))

            if sell_usdt_capped <= 0:
                logger.debug(
                    f"[{trade.pair}] 回撤减仓: sell_usdt_capped={sell_usdt_capped:.4f} 无效或被保证金限制, 跳过")
                return None
            self._set(trade, 'dca_count', 0)
            self._set(trade, 'tp_count', 0)
            self._set(trade, 'dca_done', False)
            self._set(trade, 'last_tp_time', int(current_time.timestamp()))

            logger.info(
                f"[{trade.pair}] {RED}[止盈回撤减仓至 5%]{RESET}, "
                f"{YELLOW}保证金={current_margin:.2f}{RESET}, 当前价={price:.4f}, "
                f"卖出={sell_usdt_capped:.4f} USDT"
            )
            return float(-sell_usdt_capped), "tp_fallback_reduce_to_5pct"

        if current_value_usdt < target_usdt:
            buy_usdt = target_usdt - current_value_usdt
            buy_usdt_capped = min(buy_usdt, max(0.0, collateral))

            if buy_usdt_capped <= 0:
                logger.debug(f"[{trade.pair}] 回撤加仓: buy_usdt_capped={buy_usdt_capped:.4f} 无效或无可用余额, 跳过")
                return None
            added_qty = self._qty_from_usdt(buy_usdt_capped, price, lev)

            try:
                prev_qty = float(trade.amount)
            except Exception:
                prev_qty = 0.0
            prev_avg = self._get_float(trade, 'dynamic_avg_entry', float(trade.open_rate or 0.0)) or float(
                trade.open_rate or 0.0)
            prev_cost = prev_qty * prev_avg
            new_avg = prev_avg
            if (prev_qty + added_qty) > 0:
                new_avg = (prev_cost + buy_usdt_capped * lev) / (prev_qty + added_qty)

            self._set(trade, 'dca_count', 0)
            self._set(trade, 'tp_count', 0)
            self._set(trade, 'dca_done', False)
            self._set(trade, 'dynamic_avg_entry', float(new_avg))
            self._set(trade, 'last_tp_time', int(current_time.timestamp()))

            logger.info(
                f"[{trade.pair}] {RED}[止盈回撤加仓至 5%]{RESET}, "
                f"{YELLOW}保证金={current_margin:.2f}{YELLOW}, 当前价={price:.4f}, "
                f"买入={buy_usdt_capped:.4f} USDT, 新均价={new_avg:.4f}"
            )
            return float(buy_usdt_capped), "tp_fallback_buy_to_5pct"

    def tp_reduce(self, trade: Trade, last: dict, current_profit: float, margin: float, current_rate: float,
                  current_time: datetime):
        last_tp = self._get_int(trade, 'last_tp_time', None)
        if last_tp is not None:
            base_time = datetime.fromtimestamp(int(last_tp))
        else:
            base_time = trade.open_date_utc
            if getattr(base_time, 'tzinfo', None):
                base_time = base_time.replace(tzinfo=None)
        elapsed = (current_time - base_time).total_seconds() / 60
        roi_target = 0.0
        for k, v in sorted(self.minimal_roi_user_defined.items(), key=lambda x: int(x[0]), reverse=True):
            if elapsed >= int(k):
                roi_target = v
                break

        if current_profit < roi_target:
            return None
        u = self._get_int(trade, 'dca_count', 0)

        # -- 浮亏 DCA 止盈 --
        if u > 0:
            try:
                current_qty = float(trade.amount)
            except Exception:
                current_qty = 0.0

            current_qty_abs = abs(current_qty)
            try:
                price = float(current_rate) if current_rate is not None else None
            except Exception:
                price = None

            try:
                lev = float(self.leverage(trade.pair) or 1.0)
            except Exception:
                lev = 1.0

            if price is None or price <= 0 or lev <= 0:
                logger.warning(f"[{trade.pair}] [浮亏 DCA 止盈跳过] 无效 price/lev (price={price}, lev={lev})")
                return None

            margin_val = margin
            collateral_total = self._collateral_total()
            target_usdt = float(collateral_total) * 0.05
            target_qty = (target_usdt * lev) / price
            sell_qty = max(0.0, current_qty_abs - target_qty)

            if sell_qty <= 0:
                logger.warning(
                    f"[{trade.pair}] [浮亏 DCA 止盈跳过] 当前持仓 base_qty={current_qty_abs:.6f} <= target_qty={target_qty:.6f}, 不做减仓")
                return None

            try:
                sell_usdt = (sell_qty * price) / lev if lev and price else 0.0
            except Exception:
                sell_usdt = 0.0

            sell_usdt_capped = min(sell_usdt, max(0.0, margin_val))

            if sell_usdt_capped <= 0:
                logger.warning(
                    f"[{trade.pair}] [浮亏 DCA 止盈跳过] 计算到的 sell_usdt={sell_usdt_capped:.6f} 无效或被保证金限制, 跳过")
                return None

            self._set(trade, 'dca_count', 0)
            self._set(trade, 'dca_done', False)
            self._set(trade, 'last_tp_time', int(current_time.timestamp()))

            logger.info(
                f"[{trade.pair}]{RED}[浮亏止盈减仓至 5%]{RESET}, "
                f"{YELLOW}保证金={margin:.2f} USDT, 当前价={last.get('close'):.4f}, "
                f"减仓={sell_usdt_capped:.4f} USDT"
            )
            return float(-sell_usdt_capped), "tp_afterDCA_sell_keep_5pct"

        # -- 浮盈 TP 减仓 --
        sell_amt = -0.30 * margin
        self._set(trade, 'dca_done', False)
        self._set(trade, 'last_tp_time', int(current_time.timestamp()))
        self._set(trade, 'last_tp_margin', float(margin))
        self._set(trade, 'need_rebuy', True)

        logger.info(f"[{trade.pair}] {GREEN}[浮盈减仓 卖30%]{RESET}, "
                    f"{YELLOW}保证金={margin:.2f}{RESET}, 当前价={last.get('close'):.4f}, "
                    f"减仓={abs(sell_amt):.4f}")
        return float(sell_amt), "tp30"

    def grid_single(self, trade: Trade, last: dict, current_time: datetime, current_rate: float,
                    margin: float = None):
        try:
            price = float(current_rate) if current_rate is not None else None
        except Exception:
            price = None

        if price is None:
            price = last.get('close', None)

        if price is None:
            return None

        if getattr(current_time, 'tzinfo', None):
            current_time = current_time.replace(tzinfo=None)

        anchor = self._get_float(trade, 'grid_anchor_price', None)
        if anchor is None or anchor <= 0:
            try:
                anchor = self._get_float(trade, 'dynamic_avg_entry', None)
            except Exception:
                anchor = None
        if anchor is None or anchor <= 0:
            try:
                anchor = float(trade.open_rate or 0.0)
            except Exception:
                anchor = None
        if anchor is None or anchor <= 0:
            return None

        # 网格阈值
        grid_lower = anchor * 0.98
        grid_upper = anchor * 1.02

        collateral = self._collateral_total()
        try:
            cur_margin = float(trade.stake_amount) if margin is None else float(margin)
        except Exception:
            cur_margin = 0.0
        if collateral <= 0 and cur_margin <= 0:
            return None

        last_grid_ts = self._get_int(trade, 'last_grid_action_time', None)
        if last_grid_ts is not None:
            try:
                last_dt = datetime.fromtimestamp(int(last_grid_ts))
                if (current_time - last_dt).total_seconds() < 30 * 60:  # 网格cd（30 分钟）
                    logger.debug(f"[{trade.pair}] 网格冷却中（30min），上次动作 ts={last_grid_ts}")
                    return None
            except Exception:
                pass

        w = int(self._get_int(trade, 'grid_w', 0))
        max_w = 5  # 加仓上限

        try:
            lev = float(self.leverage(trade.pair) or 1.0)
            if not (lev > 0):
                lev = 1.0
        except Exception:
            lev = 1.0
        price_source = "market" if current_rate is not None else "candle_close"

        if price <= grid_lower:
            if max_w > 0 and w >= max_w:
                logger.debug(f"[{trade.pair}] 网格加仓达到上限 w={w} >= {max_w}，跳过")
                return None

            u = self._get_int(trade, 'dca_count', 0)
            if u >= 1:
                grid_buy_frac = 0.01  # 网格加仓
                buy_usdt = collateral * grid_buy_frac
            else:
                grid_buy_frac = 0.02
                buy_usdt = collateral * grid_buy_frac

            if buy_usdt <= 0:
                return None

            added_qty = self._qty_from_usdt(buy_usdt, price, lev)
            if added_qty <= 0:
                logger.debug(f"[{trade.pair}] 计算到的加仓基础数量为0，跳过")
                return None

            try:
                prev_qty = float(trade.amount)
            except Exception:
                prev_qty = 0.0
            prev_avg = self._get_float(trade, 'dynamic_avg_entry', float(trade.open_rate or 0.0)) or float(
                trade.open_rate or 0.0)
            prev_cost = prev_qty * prev_avg

            new_avg = prev_avg
            if (prev_qty + added_qty) > 0:
                new_avg = (prev_cost + buy_usdt * lev) / (prev_qty + added_qty)

            new_w = min(w + 1, max_w) if max_w > 0 else w + 1
            new_anchor = float(price)
            next_buy = new_anchor * 0.98
            next_sell = new_anchor * 1.02

            self._set(trade, 'dynamic_avg_entry', float(new_avg))
            self._set(trade, 'grid_anchor_price', float(new_anchor))
            self._set(trade, 'grid_upper', float(next_sell))
            self._set(trade, 'grid_lower', float(next_buy))
            self._set(trade, 'last_grid_action_time', int(current_time.timestamp()))
            self._record_add_usdt(trade, 'grid_added_total_usdt', buy_usdt)
            self._record_add_qty(trade, 'grid_added_total_qty', added_qty)
            self._set(trade, 'grid_count', int(self._get_int(trade, 'grid_count', 0) + 1))
            self._set(trade, 'grid_w', int(new_w))

            logger.info(
                f"[{trade.pair}] {RED}[网格加仓 {int(grid_buy_frac * 100)}%]{RESET}, 当前价={price:.4f}, 触发价={grid_lower:.4f}, "
                f"{YELLOW}保证金={cur_margin:.2f}{RESET}, 加仓={buy_usdt:.4f} USDT, 新均价={new_avg:.4f}, w={new_w}{RESET}, "
                f"下次网格 买价={next_buy:.4f}, 卖价={next_sell:.4f}"
            )
            tag = "grid_buy_1pct" if grid_buy_frac == 0.01 else "grid_buy_2pct"
            return float(buy_usdt), tag

        if price >= grid_upper:
            try:
                current_qty = float(trade.amount)
            except Exception:
                current_qty = 0.0

            sell_qty = max(0.0, current_qty * 0.20)  # 网格减仓
            if sell_qty <= 0:
                logger.debug(f"[{trade.pair}] 网格减仓触发，但当前持仓={current_qty:.8f} 无法卖出")
                return None

            try:
                sell_usdt = (sell_qty * float(
                    current_rate if current_rate is not None else last.get('close'))) / lev if lev else 0.0
            except Exception:
                sell_usdt = 0.0

            sell_usdt = min(sell_usdt, cur_margin) if cur_margin > 0 else sell_usdt
            if sell_usdt <= 0:
                return None

            new_w = max(0, w - 1)
            new_anchor = float(price)
            next_buy = new_anchor * 0.98
            next_sell = new_anchor * 1.02

            self._record_add_usdt(trade, 'grid_reduced_total_usdt', float(sell_usdt))
            self._set(trade, 'grid_anchor_price', float(new_anchor))
            self._set(trade, 'grid_upper', float(next_sell))
            self._set(trade, 'grid_lower', float(next_buy))
            self._set(trade, 'last_grid_action_time', int(current_time.timestamp()))
            self._set(trade, 'grid_count', int(self._get_int(trade, 'grid_count', 0) + 1))
            self._set(trade, 'grid_w', int(new_w))

            logger.info(
                f"[{trade.pair}] {RED}[网格减仓 20%]{RESET} (triggered by {price_source}), 当前价={price:.4f}, 触发价={grid_upper:.4f}, "
                f"{YELLOW}保证金={cur_margin:.2f}{RESET}, 卖出={sell_usdt:.4f}, w={new_w}{RESET}, "
                f"下次网格 买价={next_buy:.4f}, 卖价={next_sell:.4f}"
            )
            return float(-sell_usdt), "grid_sell_20pct"
        return None

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float, current_profit: float,
                              **kwargs) -> tuple[float, str] | None:
        if trade.has_open_orders:
            return None

        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df.empty:
            return None
        last = df.iloc[-1]

        try:
            margin = float(trade.stake_amount)
        except Exception:
            margin = 0.0

        if current_time.tzinfo:
            current_time = current_time.replace(tzinfo=None)
        open_time = trade.open_date_utc
        if getattr(open_time, 'tzinfo', None):
            open_time = open_time.replace(tzinfo=None)
        candle_ts = pd.Timestamp(df.index[-1]).tz_localize(None).floor('min')
        df30, _ = self.dp.get_analyzed_dataframe(trade.pair, '30m')
        if df30.empty:
            return None
        last30_ts = pd.Timestamp(df30.index[-1]).tz_localize(None).floor('T')
        if candle_ts != last30_ts:
            return None

        collateral = self._collateral_total()
        if collateral <= 0:
            return None

        # 1) 趋势加仓
        res = self.trend_add(trade, last, margin, collateral, current_time)
        if res:
            return res

        # 2) DCA 加仓
        res = self.dca_add(trade, df, last, current_time, current_rate)
        if res:
            return res

        # 3) 浮盈加仓
        res = self.profit_rebuy(trade, last, collateral)
        if res:
            return res

        # 4) 止盈回撤减仓
        res = self.fallback_reduce(trade, last, current_profit, margin, current_rate, current_time)
        if res:
            return res

        # 5) tp 止盈
        res = self.tp_reduce(trade, last, current_profit, margin, current_rate, current_time)
        if res:
            return res

        # 6) 网格
        res = self.grid_single(trade, last, current_time, current_rate, margin)
        if res:
            return res
        return None

    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> None:
        tag = getattr(order, 'ft_order_tag', '') or ''

        if tag in "tp30" and order.side == "sell":
            self._set(trade, 'need_rebuy', True)

    def custom_stoploss(self, *args, **kwargs) -> float | None:
        return None
