## Overview

This project is a quantitative trading strategy developed based on the [Freqtrade](https://www.freqtrade.io/en/stable/) framework. The default configuration includes three trading bots (to add bots, please log in to the host port in the listening port API URL). The port numbers are set in docker-compose (defaults are 8000, 8001, and 8002). Because the strategy is limited to one-way positions, opening long and closing short positions on the same currency pair will conflict. You can run both long and short bots (DcaTpLong and DcaTpShort) on different trading pairs simultaneously to profit from the price difference. Alternatively, you can run only the DcaTp long bot (parameters have been optimized for DcaTpLong to reduce drawdown risk).

To use this strategy, please copy DcaTpLong to the strategies directory. The config settings are for reference: leverage: leverage size, max_open_trades: maximum number of trading pairs, stake_amount: initial capital, tradeable_balance_ratio: capital utilization rate, pair_whitelist: whitelist of trading pairs. ## Introduction

The strategy is calculated based on the closing price of the 30-minute candlestick chart. The default leverage is 10x. The initial position size is 5% of the total capital, and the position size is adjusted based on the total capital collateral and position size margin.

#### Trend/Bottom-Fishing Entry

Enter when MACD and KDJ form a golden cross, ADX > 25, EMA9 > EMA21 > EMA99, or when the lower Bollinger Band and RSI < 35.

#### Trend-Based Add-on Position

Add to the position by 2% of the total capital when MACD and KDJ form a golden cross, ADX > 25, EMA9 > EMA21 > EMA99. Reduce the position by 2% of the total capital when KDJ forms a death cross.

#### Adding to Position During Floating Loss (DCA)

Based on the dynamic average closing price, if the price falls to 1 − (0.01 - DCA addition count × 0.02), and KDJ_J < 0 or RSI < 20, or both KDJ_J < 20 and RSI < 35, the addition amount is (2% - 0.2% * DCA addition count), with a maximum of 7 times and a cooldown time of 60 minutes.

#### Adding to Position During Floating Profit (TP)

After triggering the floating profit stop-loss, add to the position up to 5% of the total capital. The addition amount is (5% of the total capital - position size).

#### Taking Profit in Batches

After triggering the floating loss DCA, reduce the position to 5% of the total capital when taking profit.

When the TP take-profit is triggered, reduce the position by 30% and mark it as eligible for adding to the floating profit position.

#### Grid Trading
The initial grid buy order price is 0.98 of the average holding price; subsequent buy orders are at 0.98 of the transaction price. The position-adding amount is (2% - 0.2% * number of grid add-ons), with a maximum of 5 add-ons and a 30-minute cooldown.
The sell order price is 1.02 of the transaction price, reducing the total capital by 2%.

### Advantages:

24/7 automated trading, combining some logic from grid and DCA strategies.

Initial capital utilization is 5%, leverage is 10x, and automatic compounding is calculated.

Leverage, profit-taking conditions, and position size can be adjusted according to risk appetite.

MACD, KDJ, EMA, and RSI indicators can be optimized, and position-adding/reducing parameters can be modified.

### Disadvantages:

No fixed stop-loss; in extreme market conditions, accumulated add-ons can lead to a position size of approximately 25% of total capital.

There is a risk of slippage, parameter overfitting, and indicator settings may not be optimal.


Frequent trading may result in higher transaction fees.

## Installation (using Docker as an example)

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

- `/forceshort`: Open a short position immediately

- `/forceexit`: Exit immediately

## Disclaimer

This strategy is a development version and is for reference only. Do not risk your money. You assume all risks associated with using this strategy. It is strongly recommended to run the trading robot in Dry-run first and not invest any funds until you understand how it works and what profits/losses you should expect.

#### Please forgive any shortcomings and areas for improvement! The source code is for learning and suggestion only.
