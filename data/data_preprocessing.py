import os
import pandas as pd
import argparse
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from sentiment_fetcher import FinancialNewsFetcher

def load_data(filename):
    """
    Load the raw OHLCV data from a CSV file.
    
    Args:
        filename (str): Path to the CSV file.
    
    Returns:
        pandas.DataFrame: Loaded data.
    """
    data = pd.read_csv(filename)
    data["Date"] = pd.to_datetime(data["Date"]) # Ensure "Date" is in datetime format
    return data

def add_technical_indicators(data):
    """
    Add technical indicators (SMA, RSI) to the data.
    
    Args:
        data (pandas.DataFrame): OHLCV data.
    
    Returns:
        pandas.DataFrame: Data with added features.
    """
    # Simple Moving Average (14-day window)
    sma = SMAIndicator(close=data["Close"], window=14)
    data["SMA_14"] = sma.sma_indicator()
    
    # Relative Strength Index (14-day window)
    rsi = RSIIndicator(close=data["Close"], window=14)
    data["RSI_14"] = rsi.rsi()

    return data

def add_lagged_features(data):
    """
    Add lagged price features (previous day's close).
    
    Args:
        data (pandas.DataFrame): Data with OHLCV.
    
    Returns:
        pandas.DataFrame: Data with lagged features.
    """
    data["Close_Lag1"] = data["Close"].shift(1)
    return data

def add_sentiment_features(data, ticker, cache_dir="data/models/finbert", skip_sentiment=False):
    """
    Add sentiment features from financial news for the ticker.
    
    Args:
        data (pandas.DataFrame): OHLCV data with Date column
        ticker (str): Stock ticker symbol (e.g., "AAPL")
        cache_dir (str): Directory to cache the HuggingFace model
        skip_sentiment (bool): Skip sentiment analysis and use neutral values
    
    Returns:
        pandas.DataFrame: Data with added sentiment features
    """
    if skip_sentiment:
        print("Skipping sentiment analysis, using neutral values")
        data["sentiment_score"] = 0.0
        data["news_count"] = 0
        data["negative"] = 0.333
        data["neutral"] = 0.333
        data["positive"] = 0.333
        return data
        
    # Check if sentiment data already exists to avoid re-processing
    sentiment_file = f"data/{ticker.lower()}_sentiment.csv"
    
    if os.path.exists(sentiment_file):
        print(f"Loading existing sentiment data from {sentiment_file}")
        sentiment_data = pd.read_csv(sentiment_file)
        sentiment_data["date"] = pd.to_datetime(sentiment_data["date"])
    else:
        print(f"Fetching and analyzing sentiment for {ticker}...")
        # Get date range from price data
        start_date = data["Date"].min().strftime("%Y-%m-%d")
        end_date = data["Date"].max().strftime("%Y-%m-%d")
        
        # Initialize news fetcher
        news_fetcher = FinancialNewsFetcher(cache_dir=cache_dir)
        
        # Fetch and analyze sentiment
        news_sentiment = news_fetcher.fetch_and_analyze(ticker, start_date, end_date)
        
        if news_sentiment.empty:
            print("No sentiment data found. Continuing without sentiment features.")
            # Add empty sentiment columns to avoid errors in downstream processing
            data["sentiment_score"] = 0.0
            data["news_count"] = 0
            data["negative"] = 0.333
            data["neutral"] = 0.333
            data["positive"] = 0.333
            return data
        
        # Aggregate daily sentiment
        sentiment_data = news_fetcher.aggregate_daily_sentiment(news_sentiment)
        
        # Save sentiment data for future use
        sentiment_data.to_csv(sentiment_file, index=False)
        print(f"Saved sentiment data to {sentiment_file}")
    
    # Merge sentiment data with price data
    data_merged = pd.merge(
        data, 
        sentiment_data,
        left_on="Date",
        right_on="date",
        how="left"
    )
    
    # Drop duplicate date column from sentiment data
    data_merged.drop("date", axis=1, inplace=True)
    
    # Fill missing sentiment data with neutral values
    if "sentiment_score" in data_merged.columns:
        data_merged["sentiment_score"].fillna(0, inplace=True)
    else:
        data_merged["sentiment_score"] = 0.0
        
    if "news_count" in data_merged.columns:
        data_merged["news_count"].fillna(0, inplace=True)
    else:
        data_merged["news_count"] = 0
        
    # Optionally fill other sentiment columns
    for col in ["negative", "neutral", "positive"]:
        if col in data_merged.columns:
            data_merged[col].fillna(0.333, inplace=True)  # Neutral default
        else:
            data_merged[col] = 0.333  # Neutral default
    
    return data_merged

def save_processed_data(data, filename):
    """
    Save the processed data to a CSV file.
    
    Args:
        data (pandas.DataFrame): Processed data.
        filename (str): Path to the output CSV file.
    """
    data.to_csv(filename, index=False)
    print(f"Processed data saved to {filename}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess stock data with technical indicators and sentiment")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--skip-sentiment", action="store_true", help="Skip sentiment analysis")
    
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    ticker_lower = ticker.lower()
    
    # Load the raw data
    input_file = f"data/{ticker_lower}_data.csv"
    print(f"Loading data from {input_file}...")
    data = load_data(input_file)

    # Add features
    print("Adding technical indicators...")
    data = add_technical_indicators(data)

    print("Adding lagged features...")
    data = add_lagged_features(data)
    
    print("Adding sentiment features...")
    data = add_sentiment_features(data, ticker, skip_sentiment=args.skip_sentiment)
    
    # Drop rows with NaN values (due to indicators needing a window)
    data = data.dropna()
    print(f"Processed {len(data)} rows of data.")
    print("First few rows with features:")
    print(data.head())

    # Save processed data
    output_file = f"data/{ticker_lower}_features.csv"
    save_processed_data(data, output_file)

if __name__ == "__main__":
    main()