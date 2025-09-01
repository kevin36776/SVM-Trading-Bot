# SVM Trading Bot

A machine learning trading bot that uses Support Vector Machine (SVM) models to predict cryptocurrency price movements and execute automated trades.


### 1. Data Collection (`Pull_Data/`)
- **bybit_data_downloader.py**: Downloads historical OHLCV data from Bybit API
- Supports multiple timeframes (15m, 1h, 4h, etc.)
- Automatically handles API rate limits and data pagination

### 2. Model Training (`SVM_Bot/`)
- **SVM_Model_Gen.py**: Creates and trains SVM models for price prediction
- Features technical indicators: EMA, MACD, RSI, Bollinger Bands, ATR, OBV
- Supports different SVM kernels (RBF, Linear, Polynomial, Sigmoid)
- Saves trained models and scalers for later use

### 3. Backtesting (`SVM_backtest/`)
- **Backtesting.py**: Tests strategy performance on historical data
- **Backtest_MoreTrades.py**: Extended backtesting with more trading opportunities
- Generates performance charts and statistics
- Compares strategy returns vs buy-and-hold

### 4. Utility Functions (`nice_funcs.py`)
- Close all positions on Bybit exchange
- Fetch wallet holdings and portfolio value
- Colored terminal output for better visibility

## Quick Start

### 1. Download Data
```python
# Edit Pull_Data/bybit_data_downloader.py
SYMBOL = "BTCUSDT"
INTERVAL_MINUTES = "60"
START_DATE_STR = "2024-01-01 00:00:00"
END_DATE_STR = "2025-01-01 00:00:00"

# Run the script
python Pull_Data/bybit_data_downloader.py
```

### 2. Train SVM Model
```python
# Edit SVM_Bot/SVM_Model_Gen.py
CSV_FILE_PATH = "path/to/your/data.csv"
SVM_KERNEL = 'rbf'  # or 'linear', 'poly', 'sigmoid'
SVM_C = 10
SVM_GAMMA = 0.01

# Run training
python SVM_Bot/SVM_Model_Gen.py
```

### 3. Backtest Strategy
```python
# Edit SVM_backtest/Backtesting.py
CSV_FILE = "path/to/your/data.csv"
INITIAL_CAPITAL = 10000
TRADE_VALUE_USD = 500

# Run backtest
python SVM_backtest/Backtesting.py
```

## Features

- **Multiple SVM Kernels**: RBF, Linear, Polynomial, Sigmoid
- **Technical Indicators**: 10+ indicators including EMA, MACD, RSI, Bollinger Bands
- **Automated Backtesting**: Performance analysis with charts and statistics
- **Risk Management**: Position sizing and commission handling
- **Exchange Integration**: Direct integration with Bybit API
- **Data Management**: Automated data collection and storage

## Key Files

- `SVM_Model_Gen.py`: Train new SVM models
- `Backtesting.py`: Test strategy performance
- `bybit_data_downloader.py`: Download market data
- `nice_funcs.py`: Trading utility functions

## Model Performance

The system generates detailed backtest results including:
- Total return vs buy-and-hold
- Sharpe ratio and maximum drawdown
- Win rate and profit factor
- Visual equity curves and volume charts

## Requirements

- Python 3.7+
- pandas, numpy, scikit-learn
- ccxt (for exchange integration)
- ta (technical analysis library)
- backtesting.py library
- matplotlib (for charts)

## Configuration

Each script contains user-configurable parameters at the top:
- File paths for data and models
- SVM hyperparameters (C, gamma, kernel)
- Trading parameters (capital, position size)
- API credentials (for live trading)

## Note

This is a research and educational project. MIT License.
