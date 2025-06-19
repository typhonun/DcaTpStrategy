from typing import Optional, Dict, Any
from decimal import Decimal
from pandas import DataFrame
from datetime import datetime
import logging

from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade


class RollingLimitMargin(IStrategy):
    """
    限价滚仓策略示例
      - 开仓后通过 adjust_trade_position 做多阶段滚仓
      - position_adjustment_enable=True + "order_types":{"position_adjustment":"limit"}
      - adjust_trade_position 返回 dict 来挂限价单
    """
    timeframe = '15m'
    can_short = False
    can_long = True
    stoploss = -0.99  # 由 custom_stoploss 全权控制

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        # 每个 trade.id 存储：stage, avg_price, position_size, pyramid_count
        self._states: Dict[int, Dict[str, Any]] = {}

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

    def leverage(self, pair: str, side: str, **kwargs) -> float:
        # 固定 50 倍杠杆
        return 50.0

    def adjust_trade_position(
            self,
            trade: Trade,
            current_time: datetime,
            current_rate: float,
            current_profit: float,
            **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        多阶段滚仓逻辑，返回 dict 触发限价挂单，或 None:
          {
            "order_type":"limit",
            "side":      "buy"/"sell",
            "amount":    <float>,
            "price":     <float>
          }
        """
        # --- 状态初始化或读取 ---
        st = self._states.setdefault(trade.id, {
            'stage': 'none',
            'avg_price': Decimal(str(trade.open_rate)),
            'position_size': Decimal(str(trade.amount)),
            'pyr': 0,
        })

        # 若检测到手动或系统已成交（均价 or 数量变了），就重置阶段
        cur_size = Decimal(str(trade.amount))
        cur_avg = Decimal(str(trade.open_rate))
        if cur_size != st['position_size'] or cur_avg != st['avg_price']:
            self.logger.info(f"[{trade.id}] 检测到成交，重置滚仓阶段")
            st.update({
                'stage': 'none',
                'avg_price': cur_avg,
                'position_size': cur_size,
                'pyr': 0,
            })

        avg = st['avg_price']
        size = st['position_size']
        profit = (Decimal(str(current_rate)) / avg) - Decimal('1')

        # 1) 初始阶段：挂第一组浮盈买单
        if st['stage'] == 'none':
            st['stage'] = 'await'
            base_price = (avg * Decimal('1.012') * Decimal('0.999')).quantize(Decimal('0.01'))
            return {
                "order_type": "limit",
                "side": "buy",
                "amount": float(size),
                "price": float(base_price)
            }

        # 2) 浮盈金字塔：每涨 +1.2%，最多 9 次
        if st['stage'] == 'await' and profit >= Decimal('0.012') and st['pyr'] < 9:
            st['pyr'] += 1
            st['stage'] = 'profit_pyramid'
            return {
                "order_type": "limit",
                "side": "sell",
                "amount": float(size * Decimal('0.3')),
                "price": float((avg * Decimal('1.012')).quantize(Decimal('0.01')))
            }

        # 3) 浮盈结束：回撤到 +0.1%
        if st['stage'] == 'profit_pyramid' and profit <= Decimal('0.001'):
            st['stage'] = 'profit_exit'
            return {
                "order_type": "limit",
                "side": "sell",
                "amount": float(size * Decimal('0.7')),
                "price": float((avg * Decimal('1.001')).quantize(Decimal('0.01')))
            }

        # 4) 浮亏金字塔：每跌 -1%，最多 5 次
        if st['stage'] in ['await', 'profit_exit'] and profit <= -Decimal('0.01') and st['pyr'] < 5:
            st['pyr'] += 1
            st['stage'] = 'loss_pyramid'
            return {
                "order_type": "limit",
                "side": "buy",
                "amount": float(size * Decimal('0.5')),
                "price": float((avg * Decimal('0.99')).quantize(Decimal('0.01')))
            }

        # 5) 浮亏结束：反弹到 +1%
        if st['stage'] == 'loss_pyramid' and profit >= Decimal('0.01'):
            st['stage'] = 'loss_exit'
            return {
                "order_type": "limit",
                "side": "sell",
                "amount": float(size * Decimal('0.6')),
                "price": float((avg * Decimal('1.01')).quantize(Decimal('0.01')))
            }

        # 其余情况：不下单
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
        # 禁用系统止损，由调整函数全权控制
        return 1.0
