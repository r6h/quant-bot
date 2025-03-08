import pandas as pd
import pickle
import numpy as np
import os

def load_model_and_data(model_file, features_file):
    """Load the trained model and feature data."""
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    data = pd.read_csv(features_file)
    data["Date"] = pd.to_datetime(data["Date"])
    return model, data

def prepare_backtest_data(data):
    """Prepare data for backtesting."""
    # Recreate Target for reference (not used in trading, just for comparison)
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    X = data[["SMA_14", "RSI_14", "Close_Lag1"]]
    return data, X

def run_backtest(model, data, X, test_size=0.2):
    """Run the backtest on the test period."""
    # Split into test set (same as training/evaluation)
    split_index = int(len(X) * (1 - test_size))
    test_data = data.iloc[split_index:].copy() # Use .copy() to avoid SettingWithCopyWarning
    X_test = X.iloc[split_index:]

    # Predict on test data
    test_data["Prediction"] = model.predict(X_test)

    # Calculate daily returns (next day's close - current close) / current close
    test_data["Next_Close"] = test_data["Close"].shift(-1)
    test_data["Daily_Return"] = (test_data["Next_Close"] - test_data["Close"]) / test_data["Close"]

    # Strategy: Buy if predict 1, hold 1 day, sell next day; otherwise do nothing
    test_data["Strategy_Return"] = np.where(test_data["Prediction"] == 1, test_data["Daily_Return"], 0)

    # Drop last row (no next day to sell)
    test_data = test_data.dropna()

    return test_data

def calculate_metrics(test_data):
    """Calculate backtest performance metrics."""
    # Cumulative returns
    test_data["Cumulative_Strategy_Return"] = (1 + test_data["Strategy_Return"]).cumprod() - 1
    test_data['Cumulative_Market_Return'] = (1 + test_data['Daily_Return']).cumprod() - 1

    # Total profit/loss
    total_strategy_return = test_data["Cumulative_Strategy_Return"].iloc[-1]
    total_market_return = test_data["Cumulative_Market_Return"].iloc[-1]

    # Annualized Sharpe Ratio (assuming 252 trading days/year)
    strategy_mean = test_data['Strategy_Return'].mean() * 252
    strategy_std = test_data['Strategy_Return'].std() * np.sqrt(252)
    sharpe_ratio = strategy_mean / strategy_std if strategy_std != 0 else 0
    
    print(f"Total Strategy Return: {total_strategy_return:.4f}")
    print(f"Total Market Return (Buy & Hold): {total_market_return:.4f}")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.4f}")
    
    return test_data

def main():
    ticker = "AAPL"  # Hardcoded for now
    model_file = f"models/{ticker.lower()}_model.pkl"
    features_file = f"data/{ticker.lower()}_features.csv"
    
    print(f"Loading model and data for {ticker}...")
    model, data = load_model_and_data(model_file, features_file)
    
    print("Preparing backtest data...")
    data, X = prepare_backtest_data(data)
    
    print("Running backtest...")
    test_data = run_backtest(model, data, X)
    
    print("Calculating performance metrics...")
    test_data = calculate_metrics(test_data)
    
    # Optional: Save results
    output_file = f"backtesting/{ticker.lower()}_backtest_results.csv"
    test_data.to_csv(output_file, index=False)
    print(f"Backtest results saved to {output_file}")

if __name__ == "__main__":
    # Create backtesting directory if it doesn't exist
    if not os.path.exists("backtesting"):
        os.makedirs("backtesting")
    main()