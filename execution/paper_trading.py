import pandas as pd
import pickle
import numpy as np
from alpaca_trade_api import REST
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("APCA-API-KEY-ID")
SECRET_KEY = os.environ.get("APCA-API-SECRET-KEY")
BASE_URL = "https://paper-api.alpaca.markets"

if not API_KEY or not SECRET_KEY:
    raise ValueError("API_KEY or SECRET_KEY not found in the environment")

def load_model(model_file):
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    return model

def fetch_recent_data(api, ticker, days=20):
    """Fetch recent OHLCV data from Alpaca, ending a week ago"""
    end_date = (datetime.now().date() - timedelta(days=7))
    start_date = end_date - timedelta(days=days)
    bars = api.get_bars(
        ticker,
        "1D",
        start=start_date.isoformat(),
        end=end_date.isoformat()
    ).df
    bars = bars.reset_index()
    bars["Date"] = pd.to_datetime(bars["timestamp"]).dt.date
    bars = bars[["Date", "open", "high", "low", "close", "volume"]]
    bars.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    return bars

def compute_features(data):
    """Compute features for prediction."""
    sma = SMAIndicator(close=data['Close'], window=14)
    rsi = RSIIndicator(close=data['Close'], window=14)
    data['SMA_14'] = sma.sma_indicator()
    data['RSI_14'] = rsi.rsi()
    data['Close_Lag1'] = data['Close'].shift(1)
    return data.dropna()

def get_prediction(model, data):
    """Make a prediction using the latest data."""
    X = data.tail(1)[['SMA_14', 'RSI_14', 'Close_Lag1']]
    prediction = model.predict(X)[0]
    return prediction

def execute_trade(api, ticker, prediction, capital=10000, risk_per_trade=0.01):
    """Execute a trade based on prediction."""
    account = api.get_account()
    cash = float(account.cash)

    if prediction == 1:
        price  = float(api.get_latest_bar(ticker).close)
        position_size = (capital * risk_per_trade) / price  # Shares to buy
        if cash >= position_size * price:
            api.submit_order(
                symbol=ticker,
                qty=str(position_size),
                side='buy',
                type='market',
                time_in_force='day'
            )
            print(f"Placed buy order for {position_size:.2f} shares of {ticker} at ${price:.2f}")
        else:
            print("Insufficient cash to place buy order.")
    else:
        # Sell if holding a position
        try:
            position = api.get_position(ticker)
            qty = float(position.qty)
            api.submit_order(
                symbol=ticker,
                qty=str(qty),
                side='sell',
                type='market',
                time_in_force='day'
            )
            print(f"Placed sell order for {qty:.2f} shares of {ticker}")
        except:
            print(f"No position in {ticker} to sell.")

def main(test_mode=True):
    ticker = "AAPL"
    model_file = f"models/{ticker.lower()}_model.pkl"

    api = REST(API_KEY, SECRET_KEY, BASE_URL)
    
    print(f"Loading model for {ticker}...")
    model = load_model(model_file)
    
    if test_mode:
        # Run once immediately for testing
        print(f"Running in test mode at {datetime.now()}...")
        data = fetch_recent_data(api, ticker)
        data = compute_features(data)
        prediction = get_prediction(model, data)
        print(f"Prediction for {ticker} tomorrow: {'Up' if prediction == 1 else 'Down'}")
        execute_trade(api, ticker, 1) # Change 1 for prediction, currently overides prediction
    else:
        # LIVE MODE: run daily at 3 PM EST
        while True:
            now = datetime.now()
            if now.hour == 15 and now.minute == 0:
                print(f"Checking market at {now}")

                # Fetch and process data
                print(f"Fetching recent data for {ticker}...")
                data = fetch_recent_data(api, ticker)
                data = compute_features(data)

                # Predict
                prediction = get_prediction(model, data)
                print(f"Prediction for {ticker} tomorrow: {'Up' if prediction == 1 else 'Down'}")
                
                # Execute trade
                execute_trade(api, ticker, prediction)
                
                # Wait until next day
                time.sleep(60)  # Sleep 1 minute to avoid rapid looping
            else:
                print(f"Waiting for 3 PM EST... Current time: {now}")
                time.sleep(60)

if __name__ == "__main__":
    if not os.path.exists("execution"):
        os.makedirs("execution")
    main()