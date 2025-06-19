from typing import Optional, Dict, Any
from decimal import Decimal
from datetime import datetime
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade
from freqtrade.rpc.rpc_manager import RPCManager

class MakerPyramid1(IStrategy):
    timeframe = '15m'
    can_short = False
    can_long = True
    stoploss = -1.0  # 永不触发系统止损

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.rpc: Optional[RPCManager] = None
        self._states: Dict[int, Dict[str, Any]] = {}

    def ft_bot_start(self, **kwargs) -> None:
        freqtrade = kwargs.get("freqtrade")
        if freqtrade is None:
            raise ValueError("Missing freqtrade instance.")
        self.rpc = RPCManager(freqtrade)

    def leverage(self, pair: str, **kwargs) -> float:
        return 50.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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

    def _rpc_order(self, trade: Trade, side: str, amount: float, price: Optional[float], custom_id: str, order_type: str = 'limit'):
        if self.rpc is None:
            return
        payload = {
            "type": "create_custom_order",
            "pair": trade.pair,
            "order_type": order_type,
            "side": side,
            "amount": amount,
            "custom_id": custom_id
        }
        if price is not None:
            payload["price"] = price
            payload["post_only"] = True
        self.rpc.send_msg(payload)

    def _rpc_cancel(self, custom_id: str):
        if self.rpc is not None:
            self.rpc.send_msg({"type": "cancel_order", "order_id": custom_id})

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[Dict[str, Any]]:

        st = self._states.setdefault(trade.id, {
            'stage': 'none', 'n': 0, 'u': 0,
            'avg': Decimal(str(trade.open_rate)),
            'size': Decimal(str(trade.amount)),
            'orders': []
        })

        cur_avg = Decimal(str(trade.open_rate))
        cur_sz = Decimal(str(trade.amount))
        if cur_avg != st['avg'] or cur_sz != st['size']:
            for oid in st['orders']:
                self._rpc_cancel(oid)
            st.update({'stage': 'none', 'n': 0, 'u': 0, 'avg': cur_avg, 'size': cur_sz, 'orders': []})

        avg = st['avg']
        size = st['size']
        profit = (Decimal(str(current_rate)) / avg) - Decimal('1')

        if st['stage'] == 'none':
            st['stage'] = 'await'
            sell_price = float((avg * Decimal('1.012')).quantize(Decimal('0.0001')))
            dip_price = float((avg * Decimal('0.99')).quantize(Decimal('0.0001')))
            self._rpc_order(trade, 'buy', float(size * Decimal('0.7')), None, f"init-buy-{trade.id}", order_type='market')
            self._rpc_order(trade, 'sell', float(size * Decimal('0.3')), sell_price, f"init-sell-{trade.id}")
            self._rpc_order(trade, 'buy', float(size * Decimal('0.3')), dip_price, f"init-dip-{trade.id}")
            st['orders'] = [f"init-buy-{trade.id}", f"init-sell-{trade.id}", f"init-dip-{trade.id}"]
            return None

        if st['stage'] == 'await' and profit >= Decimal('0.012'):
            for oid in st['orders']:
                self._rpc_cancel(oid)
            st['stage'] = 'profit'
            st['n'] += 1
            base = avg * Decimal('1.012')
            sell_price = float(base.quantize(Decimal('0.0001')))
            self._rpc_order(trade, 'buy', float(size * Decimal('0.7')), None, f"profit-buy-{trade.id}-{st['n']}", order_type='market')
            self._rpc_order(trade, 'sell', float(size * Decimal('0.3')), sell_price, f"profit-sell-{trade.id}-{st['n']}")
            st['orders'] = [f"profit-buy-{trade.id}-{st['n']}", f"profit-sell-{trade.id}-{st['n']}"]
            return None

        if st['stage'] == 'profit' and profit <= Decimal('0.001'):
            for oid in st['orders']:
                self._rpc_cancel(oid)
            st['stage'] = 'none'
            volume = float(size * (Decimal('0.5') + Decimal('0.03') * st['n']))
            price = float((avg * Decimal('1.001')).quantize(Decimal('0.0001')))
            self._rpc_order(trade, 'sell', volume, price, f"profit-exit-{trade.id}-{st['n']}")
            st['orders'] = [f"profit-exit-{trade.id}-{st['n']}"]
            return None

        if st['stage'] in ['await', 'profit_exit'] and profit <= -Decimal('0.01'):
            for oid in st['orders']:
                self._rpc_cancel(oid)
            st['stage'] = 'loss'
            st['u'] += 1
            price = float((avg * (Decimal('0.99') - Decimal('0.01') * st['u'])).quantize(Decimal('0.0001')))
            self._rpc_order(trade, 'buy', float(size * Decimal('0.3')), price, f"loss-buy-{trade.id}-{st['u']}")
            st['orders'] = [f"loss-buy-{trade.id}-{st['u']}"]
            return None

        if st['stage'] == 'loss' and profit >= Decimal('0.01'):
            for oid in st['orders']:
                self._rpc_cancel(oid)
            st['stage'] = 'none'
            volume = float(size * (Decimal('0.3') + Decimal('0.05') * st['u']))
            price = float((avg * (Decimal('1.01') - Decimal('0.002') * st['u'])).quantize(Decimal('0.0001')))
            self._rpc_order(trade, 'sell', volume, price, f"loss-exit-{trade.id}-{st['u']}")
            st['orders'] = [f"loss-exit-{trade.id}-{st['u']}"]
            return None

        return None
