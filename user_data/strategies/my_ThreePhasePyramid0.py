from typing import Optional, Dict, Any
from decimal import Decimal
from datetime import datetime
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade


class MakerPyramid(IStrategy):
    timeframe = '3m'
    can_short = False
    can_long = True
    stoploss = -1.0

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._avg_map: Dict[str, Decimal] = {}
        self._state: Dict[str, Dict[str, Any]] = {}
        self._open_orders: Dict[str, list] = {}

    def leverage(self, pair: str, **kwargs) -> float:
        return 50.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = (dataframe['close'] > dataframe['close'].shift(1)) & (dataframe['volume'] > 0)
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
        **kwargs
    ) -> Optional[Dict[str, Any]]:

        pair = trade.pair
        cur_avg = Decimal(str(trade.open_rate))
        cur_sz = Decimal(str(trade.amount))

        st = self._state.setdefault(pair, {'stage': 'none', 'n': 0, 'u': 0, 'avg': cur_avg, 'size': cur_sz})
        if cur_avg != st['avg'] or cur_sz != st['size']:
            st['avg'] = cur_avg
            st['size'] = cur_sz

        avg = st['avg']
        size = st['size']

        # 如果有挂单，先撤销（每次只撤一笔）
        open_list = self._open_orders.get(pair, [])
        if open_list:
            return {'type': 'cancel', 'order_id': open_list.pop(0)}

        def record(order):
            cid = order.get("custom_id")
            if cid:
                self._open_orders.setdefault(pair, []).append(cid)
            return order

        # 初始阶段下初始挂单
        if st['stage'] == 'none':
            st['stage'] = 'await'
            return record({
                "order_type": "limit",
                "side": "sell",
                "amount": float(size * Decimal('0.5')),
                "price": float((avg * Decimal('1.012')).quantize(Decimal('0.0001'))),
                "post_only": True,
                "custom_id": f"init-sell-{trade.id}"
            })

        # 浮盈加仓
        trigger_price = float((avg * Decimal('1.012') * Decimal('0.9999')).quantize(Decimal('0.0001')))
        if st['stage'] in ('await', 'profit') and current_rate >= trigger_price:
            st['stage'] = 'profit'
            st['n'] += 1
            return record({
                "order_type": "market",
                "side": "buy",
                "amount": float(size * Decimal('0.7')),
                "custom_id": f"profit-buy-{trade.id}-{st['n']}"
            })

        # 浮盈退出
        profit_exit_price = float((avg * Decimal('1.001')).quantize(Decimal('0.0001')))
        if st['stage'] == 'profit' and current_rate >= profit_exit_price:
            st.update({'stage': 'none', 'n': 0})
            return record({
                "order_type": "limit",
                "side": "sell",
                "amount": float(size * (Decimal('0.5') + Decimal('0.03') * st['n'])),
                "price": profit_exit_price,
                "post_only": True,
                "custom_id": f"profit-exit-{trade.id}-{st['n']}"
            })

        # 浮亏加仓
        dip_price = float((avg * Decimal('0.99') - Decimal('0.01') * st['u']).quantize(Decimal('0.0001')))
        if st['stage'] in ('await', 'loss') and current_rate <= dip_price:
            st['stage'] = 'loss'
            st['u'] += 1
            return record({
                "order_type": "limit",
                "side": "buy",
                "amount": float(size * Decimal('0.3')),
                "price": dip_price,
                "post_only": True,
                "custom_id": f"loss-buy-{trade.id}-{st['u']}"
            })

        # 浮亏退出
        loss_exit_price = float((avg * (Decimal('1.01') - Decimal('0.002') * st['u'])).quantize(Decimal('0.0001')))
        if st['stage'] == 'loss' and current_rate >= loss_exit_price:
            st.update({'stage': 'none', 'u': 0})
            return record({
                "order_type": "limit",
                "side": "sell",
                "amount": float(size * (Decimal('0.3') + Decimal('0.05') * st['u'])),
                "price": loss_exit_price,
                "post_only": True,
                "custom_id": f"loss-exit-{trade.id}-{st['u']}"
            })

        return None
