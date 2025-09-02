## 概述

此项目是基于[freqtrade](https://www.freqtrade.io/en/stable/)框架研发的量化交易策略，配置默认有三個个交易机器人（添加机器人请在监听端口API Url中登录主机端口）,docker-compose设置端口号（默认为8000，8001和8002）。可以同時運行不同交易对（单向持仓下买入开多和买入平空会冲突）的DcaTpLong和DcaTpShort多空双向机器人，依靠波动获得高低差利润，类似中性网格，也可以只运行DcaTp做多机器人(根据DcaTpLong优化了参数以减小回撤风险)。


如果要使用，请将DcaTpLong复制到strategies目录下，config配置可供参考，可同时持有不同交易对的多空仓位。该版本为开发版，后续将会更新完整的实盘数据。 leverage：杠杆大小，stake_amount：初始资金，tradable_balance_ratio：资金占用率，pair_whitelist：交易对白名单。

优点：

初始资金利用率低，根据总资金进行加仓，风险较小，可以控制仓位大小。

可根据风险偏好调整杠杆，止盈条件和仓位大小。

扩展性高，可优化MACD,KDJ,EMA,RSI指标，加减仓参数和成交逻辑。

缺点：

无固定止损。

无法判断支撑与阻力位。

有滑点，参数过拟合的风险，指标设置不一定最优。




## 安装(以docker为例)

#### 详情参考[freqtrade官方文档](https://www.freqtrade.io/en/stable/docker_quickstart/)

```
mkdir ft_userdata
cd ft_userdata/
# 克隆yml文件
curl https://raw.githubusercontent.com/freqtrade/freqtrade/stable/docker-compose.yml -o docker-compose.yml
# 拉取docker镜像
docker compose pull
# 启动交易机器人
docker compose up -d
# 建立使用者目录
docker compose run --rm freqtrade create-userdir --userdir user_data
# 建立config配置
docker compose run --rm freqtrade new-config --config user_data/config.json
```

## 使用
```
# 查看可下载的时间戳
docker-compose run --rm freqtrade list-timeframes
# 下载 OHLCV 数据（以30m为例）
docker-compose run --rm freqtrade download-data --exchange binance --timeframe 30m
# 列出可用数据
docker-compose run --rm freqtrade list-data --exchange binance
# 回测数据
docker-compose run --rm freqtrade backtesting --datadir user_data/data/binance --export trades --stake-amount 10 -s DcaTpLong -i 30m --timerange=20250510-20250701
```


## 介绍

策略是以30m时间周期k线的收盘价为基础计算的，杠杆默认为20X，初始仓位大小为总资金的0.05，根据总资金collateral加仓，仓位大小margin减仓。

#### 趋势/抄底入场

MACD、KDJ 金叉、ADX>25、EMA9>EMA21>EMA99时或者布林下轨且rsi<35时入场。

#### 趋势加仓(可选)

MACD、KDJ 金叉、ADX>25、EMA9>EMA21>EMA99时，加仓总资金 2%，KDJ死叉减仓 40%。

#### 空头信号止损(可选)

MACD、KDJ 死叉、ADX>25、EMA9<EMA21<EMA99时，减仓 50%。

#### 趋势回撤加仓(可选)

价格回落到14根k线中收盘价最高的k线的0.99且EMA9>EMA21>EMA99时，加仓总资金 2%。

#### 浮亏 DCA 加仓

基于动态平均入场价，跌破 1−(0.01+dca_count×0.01)（dca_count 浮虧为加仓次数），且 RSI<35，加仓总资金 2%，触发浮亏止盈重置dca_count。

#### 浮盈 TP 加仓

触发浮盈止盈后无回撤 tp_count>=1（tp_count为浮盈加仓次数）加仓总资金 3%，觸發回撤止盈重置tp_count。


#### 浮盈后回撤减仓

已有 TP（tp_count>0），且持仓浮盈跌回至 1% 以下时，按 0.5+0.5×tp_count 保本减仓。

#### 止盈回落加仓

价格回落至止盈价的 0.99时，加仓总资金 2%。

#### 回撤回落加仓

价格回落至回撤价的 0.99时，加仓总资金 2%。

####  浮亏止盈后回落加仓(可选)

价格回落到浮亏止盈价的0.99时，加仓总资金 2%。

#### 分批止盈

已有 DCA（dca_count>0），触发止盈按 0.5+0.05×dca_count 减仓。

已有 TP（tp_count>0），触发止盈减仓 30%，标记可浮盈加仓。

#### 24h无成交加仓(可选)

24h无加仓和止盈布林下轨加仓总资金的 2%，连续触发加仓 1%，触发止盈重置。

#### 抄底/逃顶

抄底： KDJ_J<0 且 RSI<35 时，加仓总资金 2%，连续触发加仓1%。
逃顶： KDJ_J>100 且 RSI>65 时，减仓 50%，回落中轨加仓总资金 2%。
突破布林上轨重置抄底，跌破下轨重置逃顶，避免反复触发。

#### DCA 24h 后减仓(可选)

DCA 持续 24 小时以上，价格突破布林上轨减仓 30%。

#### 小仓位加仓(可选)

持仓量小于总资金1%时加仓100%。

#### 大仓位减仓(可选)

持仓量大于总资金30%时减仓30%


## Telegram RPC 

#### 更多指令请参阅[文件](https://www.freqtrade.io/en/latest/telegram-usage/)

- `/start`: 启动交易
- `/stop`: 关闭交易
- `/stopentry`: 停止新的交易
- `/reload_config`: 加载config配置
- `/forcelong`: 立即开多
- `/forceshort`: 立即开空
- `/forceexit`: 立即退出



## 免责声明

本策略仅供参考用途，勿将担心损失的资金用于冒险，使用本策略的风险由您自行承担。强烈建议先在 Dry-run 中运行交易机器人，在了解其工作原理以及您应该预期的利润/损失之前，不要投入资金。

#### 不足和待完善之处请谅解！源码仅供学习建议


