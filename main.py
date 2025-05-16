#!/usr/bin/env python
import os
import sys
import argparse
import subprocess
from datetime import datetime, timedelta

def run_command(command, description):
    """Run a system command and print status"""
    print(f"\n{description}...")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(f"✓ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        print(f"Error: {e.stderr}")
        if input("Continue pipeline? (y/n): ").lower() != 'y':
            sys.exit(1)
        return None

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        "data",
        "data/models",
        "models",
        "monitoring/logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    print("✓ All required directories created")

def full_pipeline(ticker, days=None, skip_data_fetch=False, backtest=True, 
                  retrain=True, test_trading=True, live_trading=False):
    """
    Run the complete pipeline from data fetching to execution.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL")
        days (int): Number of days to fetch (None for all available)
        skip_data_fetch (bool): Skip fetching new data (use existing)
        backtest (bool): Run backtesting after model training
        retrain (bool): Retrain the model (if False, use existing model)
        test_trading (bool): Run paper trading in test mode
        live_trading (bool): Run paper trading in live mode
    """
    # Ensure all required directories exist
    ensure_directories()
    
    ticker = ticker.upper()
    ticker_lower = ticker.lower()
    
    # Step 1: Fetch historical price data
    if not skip_data_fetch:
        days_param = f"--days {days}" if days else ""
        run_command(
            f"python data/data_collection.py --ticker {ticker} {days_param}",
            f"Fetching historical data for {ticker}"
        )
    
    # Step 2: Preprocess data with technical indicators and sentiment analysis
    run_command(
        f"python data/data_preprocessing.py --ticker {ticker}",
        f"Preprocessing data for {ticker}"
    )
    
    # Step 3: Train or load the model
    if retrain:
        run_command(
            f"python models/model_training.py --ticker {ticker}",
            f"Training model for {ticker}"
        )
    else:
        print(f"\nSkipping model training, using existing model for {ticker}")
    
    # Step 4: Run backtesting on historical data
    if backtest:
        run_command(
            f"python backtesting/backtest.py --ticker {ticker}",
            f"Running backtesting for {ticker}"
        )
    
    # Step 5: Run paper trading bot (test or live mode)
    if test_trading:
        run_command(
            f"python execution/paper_trading.py --ticker {ticker}",
            f"Running test paper trading for {ticker}"
        )
    
    if live_trading:
        confirmation = input(f"\n⚠️ Are you sure you want to run LIVE paper trading for {ticker}? (yes/no): ")
        if confirmation.lower() == "yes":
            run_command(
                f"python execution/paper_trading.py --ticker {ticker} --live",
                f"Running LIVE paper trading for {ticker}"
            )
        else:
            print("Live trading skipped.")
    
    print("\n✓ Pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete trading bot pipeline")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--days", type=int, help="Number of days to fetch historical data")
    parser.add_argument("--skip-data-fetch", action="store_true", help="Skip fetching new data")
    parser.add_argument("--no-backtest", action="store_true", help="Skip backtesting")
    parser.add_argument("--no-retrain", action="store_true", help="Skip model retraining")
    parser.add_argument("--no-test", action="store_true", help="Skip test trading")
    parser.add_argument("--live", action="store_true", help="Run live paper trading")
    
    args = parser.parse_args()
    
    full_pipeline(
        ticker=args.ticker,
        days=args.days,
        skip_data_fetch=args.skip_data_fetch,
        backtest=not args.no_backtest,
        retrain=not args.no_retrain,
        test_trading=not args.no_test,
        live_trading=args.live
    )