import pandas as pd
import pickle
import numpy as np
from alpaca_trade_api import REST
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from monitoring.logger import log_trade
from data.sentiment_fetcher import FinancialNewsFetcher

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

def compute_features(data, ticker=None, include_sentiment=True):
    """Compute features for prediction."""
    # Technical indicators
    sma = SMAIndicator(close=data['Close'], window=14)
    rsi = RSIIndicator(close=data['Close'], window=14)
    data['SMA_14'] = sma.sma_indicator()
    data['RSI_14'] = rsi.rsi()
    data['Close_Lag1'] = data['Close'].shift(1)
    
    # Add sentiment features if requested
    if include_sentiment and ticker:
        try:
            # Get the most recent date in the data
            latest_date = pd.to_datetime(data['Date']).max()
            
            # Define a date range that includes today for fetching the most recent sentiment
            start_date = (latest_date - timedelta(days=5)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Initialize the news fetcher
            news_fetcher = FinancialNewsFetcher()
            
            # Fetch and analyze sentiment
            sentiment_data = news_fetcher.fetch_and_analyze(ticker, start_date, end_date)
            
            if not sentiment_data.empty:
                # Get only the most recent sentiment data
                daily_sentiment = news_fetcher.aggregate_daily_sentiment(sentiment_data)
                
                # Get the most recent sentiment (last row)
                if not daily_sentiment.empty:
                    latest_sentiment = daily_sentiment.iloc[-1]
                    
                    # Add sentiment features to the data
                    data['sentiment_score'] = latest_sentiment['sentiment_score']
                    data['news_count'] = latest_sentiment['news_count']
                    data['negative'] = latest_sentiment['negative']
                    data['neutral'] = latest_sentiment['neutral']
                    data['positive'] = latest_sentiment['positive']
                else:
                    # Add neutral sentiment if none found
                    data['sentiment_score'] = 0
                    data['news_count'] = 0
                    data['negative'] = 0.333
                    data['neutral'] = 0.333
                    data['positive'] = 0.333
            else:
                # Add neutral sentiment if none found
                data['sentiment_score'] = 0
                data['news_count'] = 0
                data['negative'] = 0.333
                data['neutral'] = 0.333
                data['positive'] = 0.333
                
        except Exception as e:
            print(f"Error fetching sentiment data: {e}")
            # Add neutral sentiment in case of error
            data['sentiment_score'] = 0
            data['news_count'] = 0
            data['negative'] = 0.333
            data['neutral'] = 0.333
            data['positive'] = 0.333
    
    return data.dropna()

def get_prediction(model, data):
    """Make a prediction using the latest data."""
    # Check for available features in the model
    model_features = model.get_booster().feature_names
    available_cols = [col for col in model_features if col in data.columns]
    
    # Extract only the features used by the model
    X = data.tail(1)[available_cols]
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Get prediction probability
    probability = model.predict_proba(X)[0][1]  # Probability of positive class
    
    return prediction, probability

def execute_trade(api, ticker, prediction, probability=None, threshold=0.55):
    """Execute a trade based on the prediction."""
    # Get account information to determine position size
    account = api.get_account()
    buying_power = float(account.buying_power)
    position_size = buying_power * 0.02  # Using 2% of buying power
    
    # Check current position
    try:
        position = api.get_position(ticker)
        has_position = True
    except:
        has_position = False
    
    # Get latest market price
    latest_trade = api.get_latest_trade(ticker)
    price = latest_trade.price
    
    # Calculate number of shares
    shares = int(position_size / price)
    
    # Make trading decision
    # Only buy with sufficient confidence if probability is provided
    if probability is not None:
        buy_signal = prediction == 1 and probability > threshold
    else:
        buy_signal = prediction == 1
    
    sell_signal = prediction == 0
    
    # Execute trade
    if buy_signal and not has_position and shares > 0:
        print(f"Buying {shares} shares of {ticker} at ${price:.2f}")
        api.submit_order(
            symbol=ticker,
            qty=shares,
            side="buy",
            type="market",
            time_in_force="day"
        )
        log_trade(ticker, "buy", shares, price)
        return "BUY"
    
    elif sell_signal and has_position:
        print(f"Selling all shares of {ticker} at ${price:.2f}")
        api.submit_order(
            symbol=ticker,
            qty=position.qty,
            side="sell",
            type="market",
            time_in_force="day"
        )
        log_trade(ticker, "sell", int(float(position.qty)), price)
        return "SELL"
    
    else:
        action = "HOLD"
        if buy_signal:
            reason = "but already have position" if has_position else "but unable to buy shares"
            print(f"Buy signal for {ticker} {reason}")
        elif sell_signal:
            reason = "but no position to sell"
            print(f"Sell signal for {ticker} {reason}")
        else:
            print(f"No clear signal for {ticker}, holding")
        
        return action

def main(test_mode=True):
    ticker = "AAPL"
    model_file = f"models/{ticker.lower()}_model.pkl"

    api = REST(API_KEY, SECRET_KEY, BASE_URL)
    
    print(f"Loading model for {ticker}...")
    model = load_model(model_file)
    
    # TEST MODE, PLACE ORDERS AT RUNTIME
    if test_mode:
        # Run once immediately for testing
        print(f"Running in test mode at {datetime.now()}...")
        data = fetch_recent_data(api, ticker)
        data = compute_features(data, ticker=ticker, include_sentiment=True)
        
        # Test buy first
        print("Testing buy order...")
        prediction = 1  # Force buy
        print(f"Prediction for {ticker} tomorrow: {'Up' if prediction == 1 else 'Down'}")
        execute_trade(api, ticker, prediction)
        
        print("Waiting 10 seconds to simulate next day...")
        time.sleep(10)
        
        # Test sell
        print("Testing sell order...")
        prediction = 0  # Force sell
        print(f"Prediction for {ticker} tomorrow: {'Up' if prediction == 1 else 'Down'}")
        execute_trade(api, ticker, prediction)
    else:
        # LIVE MODE: run daily at 3 PM EST
        while True:
            now = datetime.now()
            if now.hour == 15 and now.minute == 0:
                print(f"Checking market at {now}")

                # Fetch and process data
                print(f"Fetching recent data for {ticker}...")
                data = fetch_recent_data(api, ticker)
                data = compute_features(data, ticker=ticker, include_sentiment=True)

                # Predict
                prediction, probability = get_prediction(model, data)
                print(f"Prediction for {ticker} tomorrow: {'Up' if prediction == 1 else 'Down'} with {probability:.2%} confidence")
                
                # Execute trade
                execute_trade(api, ticker, prediction, probability)
                
                # Wait until next day
                time.sleep(60)  # Sleep 1 minute to avoid rapid looping
            else:
                # Check every 30 seconds
                time.sleep(30)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run paper trading bot")
    parser.add_argument("--live", action="store_true", help="Run in live mode")
    args = parser.parse_args()
    
    main(test_mode=not args.live)