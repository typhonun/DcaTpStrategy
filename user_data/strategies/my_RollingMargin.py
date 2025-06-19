from typing import Optional, Dict
from decimal import Decimal

import ta
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade

class RollingMargin(IStrategy):
    timeframe = '15m'
    can_short = False
    can_long = True
    stoploss = -0.99

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._states: Dict[int, dict] = {}

    def leverage(self, pair: str, current_leverage: float = 1.0,
                 max_leverage: float = 5.0, side: str = "long", **kwargs) -> float:
        return 50

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_20'] = ta.trend.ema_indicator(dataframe['close'], window=20)
        dataframe['rsi'] = ta.momentum.rsi(dataframe['close'], window=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = (
            (dataframe['close'] > dataframe['close'].shift(1)) &
            (dataframe['volume'] > 0)
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = False
        return dataframe

    def adjust_trade_position(
            self,
            trade: Trade,
            current_time: datetime,
            current_rate: float,
            current_profit: float,
            amount: Optional[float] = None,
            pair: Optional[str] = None,
            **kwargs
    ) -> Optional[float]:

        state = self._states.get(trade.id, {
            'profit_marks': 0,
            'loss_marks': 0,
            'in_profit_phase': False,
            'in_loss_phase': False,
            'last_action': None
        })
        profit = (Decimal(str(current_rate)) / Decimal(str(trade.open_rate))) - Decimal('1')

        next_profit_marker = (state['profit_marks'] + 1) * Decimal('0.012')

        if profit >= next_profit_marker:
            state['profit_marks'] += 1
            add_amt = trade.amount
            state['last_action'] = 'add'
            state['in_profit_phase'] = True
            self._states[trade.id] = state
            return float(add_amt)

        if state.get('last_action') == 'add':
            state['last_action'] = 'reduce'
            reduce_amt = Decimal(str(trade.amount)) * Decimal('0.3')
            self._states[trade.id] = state
            return float(-reduce_amt)

        if state['in_profit_phase'] and profit <= Decimal('0.001'):
            state['in_profit_phase'] = False
            reduce_amt = Decimal(str(trade.amount)) * Decimal('0.7')
            self._states[trade.id] = state
            return float(-reduce_amt)

        next_loss_marker = -(state['loss_marks'] + 1) * Decimal('0.01')
        if profit <= next_loss_marker:
            state['loss_marks'] += 1
            state['in_loss_phase'] = True
            add_amt = Decimal(str(trade.amount)) * Decimal('0.5')
            self._states[trade.id] = state
            return float(add_amt)

        if state['in_loss_phase'] and profit >= Decimal('0.01'):
            state['in_loss_phase'] = False
            reduce_amt = Decimal(str(trade.amount)) * Decimal('0.6')
            self._states[trade.id] = state
            return float(-reduce_amt)

        self._states[trade.id] = state
        return None

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> float:
        return 1.0
