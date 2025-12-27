## 概述

本项目是一个基于 [freqtrade](https://www.freqtrade.io/en/stable/) 框架开发的量化交易策略。默认配置包含三个交易机器人（如需添加机器人，请登录监听端口 API URL 中的主机端口）。端口号在 docker-compose 中设置（默认值为 8000、8001 和 8002）。它可以同时运行 DcaTpLong 和 DcaTpShort 两款多空机器人，分别针对不同的交易对（在单向持仓的情况下，对同一货币对同时进行多头平仓和空头操作会相互冲突）。它基于波动率利用价格差异获利，类似于中性网格策略。您也可以单独运行 DcaTpLong 机器人（DcaTpLong 的参数已优化，以降低回撤风险）。

要使用本项目，请将 DcaTpLong 复制到 strategies 目录。配置文件已提供，供您参考。要手动下单，请先使用 `/stopentry` 命令，然后使用 `/stop` 命令完全停止机器人，最后使用 `/start` 命令恢复机器人运行。杠杆：杠杆等级，最大持仓交易数：最大交易对数量，初始资金：初始资金，可交易资金利用率：资金利用率，交易对白名单：交易对白名单。

## 简介

该策略基于 30 分钟 K 线图的收盘价计算。默认杠杆为 20 倍。初始仓位规模为总资金的 5%。仓位根据总资金以保证金形式增加，并以保证金减少。

#### 趋势/抄底入场

入场时机为：MACD 和 KDJ 形成黄金交叉，ADX > 25，EMA9 > EMA21 > EMA99，或布林带下轨触及且 RSI < 35。

#### 基于趋势的加仓

当 MACD 和 KDJ 形成黄金交叉，ADX > 25，EMA9 > EMA21 > EMA99 时，加仓金额为总资金的 2%。当 KDJ 形成死亡交叉时，减去加仓金额。

#### 未实现亏损期间加仓（定投）

基于动态平均入场价格，如果价格下跌至 1 − (0.01 + dca_count × 0.02)，且 KDJ_J < 0 或 RSI < 20，或 KDJ_J < 20 且 RSI < 35，则加仓总资金的 2%，并将 dca_count 加 1，最多加仓 5 次，每次加仓后有 60 分钟的冷却时间。

#### 未实现盈利期间加仓（止盈）

触发止盈止损后，加仓总资金的 5%，并将 tp_count 加 1。

#### 止盈后减仓

如果已设置止盈（tp_count > 0），且未实现盈利小于 1%，则将仓位减至总资金的 5%，并将 tp_count 设置为 0。

#### 分批止盈

如果已触发定投（dca_count > 0），则在止盈时将仓位减至总资金的 5%，并将 dca_count 设置为 0。

如果未触发止盈，则在止盈时将仓位减少 30%，并标记以备后续盈利。

#### 网格交易

初始买单价格为平均持仓价格的 0.98 倍，后续买单价格为交易价格的 0.98 倍，每次加仓 2%。如果触发 DCA（分布式校准），则每次加仓 1%，最多加仓 5 次，每次加仓后有 30 分钟的冷却时间。

卖单价格为交易价格的 1.02 倍，每次减仓 20%。

### 优势：

24 小时自动交易，结合了网格交易和美元成本平均法的部分逻辑。

初始资金要求低，根据总资金加仓，最大仓位约为总资金的 25%。

可根据风险承受能力调整杠杆、止盈条件和仓位大小。

可优化 MACD、KDJ、EMA 和 RSI 指标，并可修改加仓/减仓参数。

缺点：

没有固定止损位，无法确定支撑位和阻力位。

存在滑点风险、参数过拟合风险，且指标设置可能并非最优。频繁交易可能导致更高的交易费用。 ## 安装（以 Docker 为例）

#### 详情请参阅 [freqtrade 官方文档](https://www.freqtrade.io/en/stable/docker_quickstart/)

```

mkdir ft_userdata

cd ft_userdata/

# 克隆 yml 文件

curl https://raw.githubusercontent.com/freqtrade/freqtrade/stable/docker-compose.yml -o docker-compose.yml

# 拉取 Docker 镜像

docker compose pull

# 启动交易机器人

docker compose up -d

# 创建用户目录

docker compose run --rm freqtrade create-userdir --userdir user_data

# 创建配置文件

docker compose run --rm freqtrade new-config --config user_data/config.json

```

## 使用方法

```

# 查看可下载的时间戳

docker-compose run --rm freqtrade列出交易时段

# 下载 OHLCV 数据（以 30 分钟周期为例）

docker-compose run --rm freqtrade download-data --exchange binance --timeframe 30m

# 列出可用数据

docker-compose run --rm freqtrade list-data --exchange binance

# 回测数据

docker-compose run --rm freqtrade backtesting --datadir user_data/data/binance --export trades --stake-amount 10 -s DcaTpLong -i 30m --time