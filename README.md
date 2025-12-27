## Overview


[中文版本](./README-cn.md)

This project is a quantitative trading strategy developed based on the [freqtrade](https://www.freqtrade.io/en/stable/) framework. The default configuration includes three trading bots (to add bots, please log in to the host port in the listening port API URL). The port numbers are set in docker-compose (defaults are 8000, 8001, and 8002). It can run DcaTpLong and DcaTpShort long/short bots simultaneously on different trading pairs (buying long and closing short positions on the same currency pair will conflict under a one-way position). It profits from the price difference based on volatility, similar to a neutral grid. Alternatively, you can run only the DcaTp long bot (parameters have been optimized for DcaTpLong to reduce drawdown risk).

To use this project, please copy DcaTpLong to the strategies directory. The config file is provided for reference. To place orders manually, first use `/stopentry` and then `/stop` to completely stop the bot before placing an order, and finally `/start` to resume the bot. Leverage: Leverage level, max_open_trades: Maximum number of trading pairs, stake_amount: Initial capital, tradeable_balance_ratio: Capital utilization rate, pair_whitelist: Whitelist of trading pairs.

## Introduction

The strategy is calculated based on the closing price of the 30-minute candlestick chart. The default leverage is 20X. The initial position size is 5% of the total capital. Positions are added collaterally based on the total capital, and reduced by margin.

#### Trend/Bottom-Fishing Entry

Entry occurs when MACD and KDJ form a golden cross, ADX > 25, EMA9 > EMA21 > EMA99, or when the lower Bollinger Band is reached and RSI < 35.

#### Trend-Based Position Addition

Add to the position by 2% of the total capital when MACD and KDJ form a golden cross, ADX > 25, EMA9 > EMA21 > EMA99. Subtract the amount added when KDJ forms a death cross.

#### Adding to Position During Unrealized Losses (DCA)

Based on the dynamic average entry price, if the price falls to 1 − (0.01 + dca_count × 0.02), and KDJ_J < 0 or RSI < 20, or both KDJ_J < 20 and RSI < 35, add 2% of the total capital, increment dca_count by 1, up to a maximum of 5 times, with a 60-minute cooldown.

#### Adding to Position During Unrealized Profits (TP)

After triggering a profit-taking stop-loss, add 5% of the total capital, increment tp_count by 1.

#### Reducing Position After Profit-Taking on Drawdown

If there is already TP (tp_count > 0), and the unrealized profit is < 1%, reduce the position to 5% of the total capital, and tp_count = 0.

#### Taking Profit in Batches

If DCA (dca_count > 0) has been triggered, reduce the position to 5% of the total capital when taking profit, and dca_count = 0.

If TP (Take Profit) is not triggered, reduce position by 30% upon taking profit, and mark for potential additional profit.

#### Grid Trading
The initial buy order is at 0.98 of the average holding price, subsequent buy orders are at 0.98 of the transaction price, adding 2% of total capital. If DCA (Distributed Calibration) is triggered, only 1% is added, up to a maximum of 5 times, with a 30-minute cooldown.
The sell order price is 1.02 of the transaction price, reducing position by 20%.

### Advantages:

24-hour automated trading, combining some logic from grid trading and dollar-cost averaging.

Low initial capital requirement, adding positions based on total capital, with a maximum position size of approximately 25%.

Leverage, take-profit conditions, and position size can be adjusted according to risk appetite.

MACD, KDJ, EMA, and RSI indicators can be optimized, and position addition/reduction parameters can be modified.

### Disadvantages:

No fixed stop-loss, unable to determine support and resistance levels.

Risk of slippage, parameter overfitting, and indicator settings may not be optimal. Frequent trading may result in higher transaction fees. ## Installation (using Docker as an example)

#### For details, refer to the [freqtrade official documentation](https://www.freqtrade.io/en/stable/docker_quickstart/)

```
mkdir ft_userdata
cd ft_userdata/

# Clone the yml file
curl https://raw.githubusercontent.com/freqtrade/freqtrade/stable/docker-compose.yml -o docker-compose.yml

# Pull the Docker image
docker compose pull

# Start the trading bot
docker compose up -d

# Create the user directory
docker compose run --rm freqtrade create-userdir --userdir user_data

# Create the config
docker compose run --rm freqtrade new-config --config user_data/config.json

```

## Usage
```
# View downloadable timestamps
docker-compose run --rm freqtrade list-timeframes

# Download OHLCV Data (30m example)

docker-compose run --rm freqtrade download-data --exchange binance --timeframe 30m

# List available data

docker-compose run --rm freqtrade list-data --exchange binance

# Backtest data

docker-compose run --rm freqtrade backtesting --datadir user_data/data/binance --export trades --stake-amount 10 -s DcaTpLong -i 30m --timerange=20250510-20250701

```

## Telegram RPC

#### For more commands, please refer to [file](https://www.freqtrade.io/en/latest/telegram-usage/)

- `/start`: Start a trade

- `/stop`: Stop a trade

- `/stopentry`: Stop a new trade

- `/reload_config`: Load the configuration

- `/forcelong`: Open a long position immediately

`/forceshort`: Open a short position immediately

`/forceexit`: Exit immediately

## Disclaimer

This strategy is a development version and is for reference only. Do not risk your money. You assume all risks associated with using this strategy. It is strongly recommended to run the trading robot in Dry-run first and not invest any funds until you understand how it works and what profits/losses you should expect.

#### Please forgive any shortcomings and areas for improvement! The source code is for learning and suggestion only.

