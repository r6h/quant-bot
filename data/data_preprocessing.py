import os
import pandas as pd
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator

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
    # Load the raw data
    input_file = "data/aapl_data.csv"
    print(f"Loading data from {input_file}...")
    data = load_data(input_file)

    # Add features
    print("Adding technical indicators...")
    data = add_technical_indicators(data)

    print("Adding lagged features...")
    data = add_lagged_features(data)
    
    # Drop rows with NaN values (due to indicators needing a window)
    data = data.dropna()
    print(f"Processed {len(data)} rows of data.")
    print("First few rows with features:")
    print(data.head())

    # Save processed data
    base_name = os.path.splitext(os.path.basename(input_file))[0].replace("_data", "")
    output_file = f"{base_name}_features.csv"
    save_processed_data(data, output_file)

if __name__ == "__main__":
    main()