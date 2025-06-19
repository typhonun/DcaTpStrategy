# -*- coding: utf-8 -*-
from typing import Optional, Dict, Any
from decimal import Decimal
from datetime import datetime
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade


class LimitBuy(IStrategy):
    """
    每当有单子开仓后，Freqtrade 会调用 adjust_trade_position() 一次。
    我们在持仓平均价的 0.999 倍挂一笔 limit-buy，并确保返回值里的 `stake_amount` 是 float。
    """
    timeframe = '1m'
    can_short = False
    can_long = True
    stoploss = -1.0

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # 用来标记：某笔 Trade.id 是否已经挂过加仓单
        self._has_added_limit: Dict[int, bool] = {}

    def leverage(self, pair: str, **kwargs) -> float:
        # 举例：全仓 1 倍杠杆
        return 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 这里我们不需要任何额外指标
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        任意开仓条件：只要这根K线收盘价 > 前一根，就触发一次 enter_long。
        """
        df = dataframe.copy()
        df['enter_long'] = False
        if len(df) >= 2 and df['close'].iloc[-1] > df['close'].iloc[-2]:
            df.at[df.index[-1], 'enter_long'] = True
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        暂时不做自动平仓，exit_long 恒为 False。
        """
        df = dataframe.copy()
        df['exit_long'] = False
        return df

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        当 Freqtrade 检测到有持仓（Trade）时，会立即调用这个方法一次。
        我们只在第一次调用时挂一笔限价买单，价格 = avg_price * 0.999。
        一定要保证返回的 dict 里 "stake_amount" 是数值型 (float/int)，
        而“订单详情”放到 "order" 下面。
        """

        # —— 1) 如果这笔 trade 已经挂过加仓单，就跳过
        if self._has_added_limit.get(trade.id, False):
            return None

        # —— 2) 取出当前持仓的平均开仓价
        avg_price = Decimal(str(trade.open_rate))

        # —— 3) 计算限价买单价格：avg_price * 0.999
        limit_price = float((avg_price * Decimal("0.999")).quantize(Decimal("0.0001")))

        # —— 4) 下单数量 = 当前持仓手数
        buy_amount = float(trade.amount)

        # —— 5) 生成一个唯一 custom_id，便于未来撤单或追踪
        cid = f"addbuy-{trade.id}-{int(datetime.utcnow().timestamp())}"

        # —— 6) 标记这笔 trade 已经挂过加仓单，防止重复挂
        self._has_added_limit[trade.id] = True

        # —— 7) 返回格式：一定要包含数值型的 stake_amount，以及一个完整的 order 子字典
        return {
            # —— !!!! 这里一定要是 float 或 int，否则 Freqtrade 内部会把它
            #         跟 0.0 做大小比较，从而报 “dict > float” 的错误。
            "stake_amount": 0.0,

            # —— 把限价挂单的所有细节放在 "order" 子字典里
            "order": {
                "order_type": "limit",
                "side": "buy",
                "amount": buy_amount,
                "price": limit_price,
                "custom_id": cid
            }
        }
