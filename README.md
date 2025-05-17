# Quant-Bot: Algorithmic Trading Pipeline

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture & Pipeline](#architecture--pipeline)
    - [Directory Structure](#directory-structure)
    - [Pipeline Stages](#pipeline-stages)
3. [Technologies Used](#technologies-used)
    - [Why Local Models? (FinBERT vs API)](#why-local-models-finbert-vs-api)
4. [Detailed Pipeline Walkthrough](#detailed-pipeline-walkthrough)
    - [1. Data Collection](#1-data-collection)
    - [2. Data Preprocessing & Feature Engineering](#2-data-preprocessing--feature-engineering)
    - [3. Model Training](#3-model-training)
    - [4. Model Evaluation](#4-model-evaluation)
    - [5. Backtesting](#5-backtesting)
    - [6. Paper Trading (Execution)](#6-paper-trading-execution)
    - [7. Monitoring & Performance Tracking](#7-monitoring--performance-tracking)
5. [How to Run the Project](#how-to-run-the-project)
6. [What’s Missing / Future Improvements](#whats-missing--future-improvements)
7. [References](#references)

---

## Project Overview

Quant-Bot is a modular, end-to-end algorithmic trading system designed for research, backtesting, and paper trading of equity strategies. It leverages technical indicators, financial news sentiment (via FinBERT), and machine learning (XGBoost) to predict short-term price movements and execute trades using the Alpaca API. The pipeline is fully automated and can be run from a single entry point, making it easy to experiment, retrain, and deploy new strategies.

---

## Architecture & Pipeline

### Directory Structure

```
quant-bot/
├── main.py                  # Pipeline entry point
├── requirements.txt         # Python dependencies
├── data/                    # Raw & processed data, feature engineering
│   ├── data_collection.py
│   ├── data_preprocessing.py
│   └── sentiment_fetcher.py
├── models/                  # Model training & evaluation
│   ├── model_training.py
│   ├── model_eval.py
│   └── <ticker>_model.pkl
├── backtesting/             # Backtest engine & results
│   └── backtest.py
├── execution/               # Paper/live trading logic
│   └── paper_trading.py
├── monitoring/              # Logging & performance tracking
│   ├── logger.py
│   └── performance_tracker.py
└── ...
```

### Pipeline Stages

1. **Data Collection**: Download historical OHLCV data for a given ticker.
2. **Data Preprocessing**: Add technical indicators, lagged features, and news sentiment.
3. **Model Training**: Train an XGBoost classifier to predict next-day price direction.
4. **Model Evaluation**: Evaluate model performance on a holdout set.
5. **Backtesting**: Simulate trading with risk management on historical data.
6. **Paper Trading**: Run the strategy in real-time using Alpaca’s paper trading API.
7. **Monitoring**: Log trades and track portfolio performance.

---

## Technologies Used

- **Python 3.x**: Main programming language.
- **Pandas, NumPy**: Data manipulation and analysis.
- **yfinance**: Download historical stock data.
- **ta**: Technical analysis indicators (SMA, RSI, etc).
- **transformers (FinBERT)**: Local financial news sentiment analysis.
- **xgboost**: Machine learning model for classification.
- **alpaca-trade-api**: Broker API for paper/live trading.
- **scikit-learn**: Model evaluation metrics.
- **dotenv**: Securely load API keys from environment.

#### Why Local Models? (FinBERT vs API)
- **Local FinBERT**: The project uses a locally cached FinBERT model for sentiment analysis. This avoids API rate limits, privacy issues, and recurring costs, and allows for faster, large-scale batch processing. It also ensures reproducibility and independence from third-party service outages.
- **API-based Sentiment**: While APIs (e.g., Finnhub, Alpha Vantage) can provide news and sentiment, they often have strict rate limits, require paid plans for scale, and may not be as customizable or transparent as running your own model.

---

## Detailed Pipeline Walkthrough

### 1. Data Collection
- **Script**: `data/data_collection.py`
- **Functionality**: Downloads historical OHLCV data for a given ticker using yfinance. Supports custom date ranges and tickers via command-line arguments.
- **Output**: CSV file in `data/<ticker>_data.csv`.

### 2. Data Preprocessing & Feature Engineering
- **Script**: `data/data_preprocessing.py`
- **Functionality**:
    - Adds technical indicators (SMA, RSI) using `ta`.
    - Adds lagged features (e.g., previous day’s close).
    - Fetches financial news and computes sentiment scores using a local FinBERT model (`sentiment_fetcher.py`).
    - Merges all features into a single DataFrame.
- **Output**: CSV file in `data/<ticker>_features.csv`.

### 3. Model Training
- **Script**: `models/model_training.py`
- **Functionality**:
    - Loads processed features.
    - Defines the target: 1 if next day’s close > today’s close, else 0.
    - Trains an XGBoost classifier with a time-based split (train on earlier data, test on later data).
    - Supports both technical and sentiment features.
    - Saves the trained model to `models/<ticker>_model.pkl`.

### 4. Model Evaluation
- **Script**: `models/model_eval.py`
- **Functionality**:
    - Loads the trained model and features.
    - Evaluates on the test set (last 20% of data).
    - Prints accuracy, precision, recall, and confusion matrix.

### 5. Backtesting
- **Script**: `backtesting/backtest.py`
- **Functionality**:
    - Loads the trained model and test data.
    - Simulates trading with position sizing, stop-loss, and risk management.
    - Calculates daily and cumulative returns, Sharpe ratio, and compares to buy-and-hold.
    - Saves results to `backtesting/<ticker>_backtest_results.csv`.

### 6. Paper Trading (Execution)
- **Script**: `execution/paper_trading.py`
- **Functionality**:
    - Connects to Alpaca’s paper trading API using credentials from `.env`.
    - Fetches recent market data.
    - Computes features (including real-time sentiment if enabled).
    - Makes predictions and places simulated trades.
    - Logs trades for later analysis.

### 7. Monitoring & Performance Tracking
- **Scripts**: `monitoring/logger.py`, `monitoring/performance_tracker.py`
- **Functionality**:
    - Logs all trades to a text file.
    - Calculates portfolio value, total profit, and return from the trade log.
    - Prints performance summary.

---

## How to Run the Project

1. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
2. **Set up Alpaca API keys**:
   - Create a `.env` file or use `secrets.env` with your Alpaca paper trading credentials:
     ```env
     APCA-API-KEY-ID=your_key
     APCA-API-SECRET-KEY=your_secret
     ```
3. **Run the full pipeline**:
   ```sh
   python main.py --ticker AAPL --days 1825
   ```
   - Use `--skip-data-fetch`, `--no-backtest`, `--no-retrain`, `--no-test`, or `--live` to control pipeline stages.

---

## What’s Missing / Future Improvements

- **Live Trading**: The current pipeline is set up for paper trading. Live trading is possible but should be used with caution.
- **News API Integration**: The news fetching in `sentiment_fetcher.py` is a placeholder. Integrate a real news API (e.g., Finnhub, NewsAPI) for production use.
- **Model Selection & Hyperparameter Tuning**: Add grid search or Bayesian optimization for model parameters.
- **Advanced Risk Management**: Implement more sophisticated position sizing, trailing stops, and portfolio-level risk controls.
- **Strategy Diversification**: Support for multiple tickers, asset classes, and strategy types.
- **Backtest Robustness**: Add slippage, transaction costs, and more realistic order simulation.
- **Visualization**: Add plots for equity curve, drawdown, and feature importance.
- **Unit Tests & CI**: Add automated tests and continuous integration.
- **Dockerization**: Containerize the pipeline for reproducibility.

---

## References
- [FinBERT: Financial Sentiment Analysis](https://github.com/ProsusAI/finBERT)
- [Alpaca API Documentation](https://alpaca.markets/docs/api-references/trading-api/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
- [TA-Lib / ta](https://technical-analysis-library-in-python.readthedocs.io/en/latest/)
- [yfinance](https://github.com/ranaroussi/yfinance)

---
