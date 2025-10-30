## 概述

此项目是基于[freqtrade](https://www.freqtrade.io/en/stable/)框架研发的量化交易策略，配置默认有三個个交易机器人（添加机器人请在监听端口API Url中登录主机端口）,docker-compose设置端口号（默认为8000，8001和8002）。可以同時运行不同交易对（单向持仓下同一币种买入开多和买入平空会冲突）的DcaTpLong和DcaTpShort多空双向机器人，依靠波动获得高低差利润，类似中性网格，也可以只运行DcaTp做多机器人(根据DcaTpLong优化了参数以减小回撤风险)。


如果要使用，请将DcaTpLong复制到strategies目录下，config配置可供参考。若要手动下单请先 /stopentry，/stop完全停止机器人后下单，最后/start恢复机器人。该版本为开发版，后续将会更新完整的实盘数据。 leverage：杠杆大小，max_open_trades：交易对数量上限，stake_amount：初始资金，tradable_balance_ratio：资金占用率，pair_whitelist：交易对白名单。


## 介绍

策略是以30m时间周期k线的收盘价为基础计算的，杠杆默认为20X，初始仓位大小为总资金的5%，根据总资金collateral加仓，仓位大小margin减仓。

#### 趋势/抄底入场

MACD、KDJ 金叉、ADX>25、EMA9>EMA21>EMA99时或者布林下轨且rsi<35时入场。

#### 趋势加仓

MACD、KDJ 金叉、ADX>25、EMA9>EMA21>EMA99时，加仓总资金 2%，KDJ死叉减掉加仓量。

#### 浮亏 DCA 加仓

基于动态平均入场价，跌破 1−(0.01+dca_count×0.02)，且 KDJ_J小于0或RSI<20或KDJ_J小于20和RSI小于35，加仓总资金 2%，dca_count+1，上限5次，冷却时间60m，触发浮亏止盈重置dca_count。

#### 浮盈 TP 加仓

触发浮盈止盈后加仓总资金 5%，tp_count+1，触发回撤重置tp_count。

#### 止盈后回撤减仓

已有 TP（tp_count>0），且持仓浮盈跌回至 1% 以下时，减至总资金 5%。

#### 分批止盈

已有 DCA（dca_count>0），触发止盈减至总资金 5%。

已有 TP（tp_count>0），触发止盈减仓 30%，标记可浮盈加仓。

#### 网格
首次买单为持仓均价的 0.98，后续买单为成交价的 0.98，加仓总资金 2%，上限 5次，冷却时间30m；
卖单为成交价的 1.02，减仓 20%。


### 优点：

24h自动化交易，结合了网格和马丁的部分逻辑。

初始资金利用率低，根据总资金加仓，仓位上限20%左右。

可以根据风险偏好调整杠杆，止盈条件和仓位大小。

扩展性高，可优化和新增MACD,KDJ,EMA,RSI指标，加减仓参数和成交逻辑。

### 缺点：

无固定止损，无法判断支撑与阻力位。

有滑点，参数过拟合的风险，指标设置不一定最优。

频繁交易可能产生较高的手续费。


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
