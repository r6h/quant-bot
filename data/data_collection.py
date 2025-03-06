import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL" for Apple).
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
    
    Returns:
        pandas.DataFrame: OHLCV data (Open, High, Low, Close, Volume).
    """
    # Create Ticker for the stock
    try:
        stock = yf.Ticker(ticker)

        # Fetch historical data
        data = stock.history(start=start_date, end=end_date)

        if data.empty:
            raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}.")

        # Reset index to make 'Date' a column instead of the index
        data = data.reset_index()

        # Select only the columns we are interested in
        data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]

        # Convert Date to just the date part (remove time if present)
        data["Date"] = pd.to_datetime(data["Date"]).dt.date

        return data
    
    except Exception as e:
        print(f"Erroe fetching data: {e}")
        return None

def save_data_to_csv(data, filename):
    """
    Save the fetched data to a CSV file.
    
    Args:
        data (pandas.DataFrame): The data to save.
        filename (str): Path to the output CSV file.
    """
    if data is not None:
        data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    else:
        print("No data to save.")

def main():
    # Define the stock ticker and date range
    ticker = "AAPL"

    # Define the date range (e.g., last 5 years up to today)
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=5*365)).strftime("%Y-%m-%d")  # Approx 5 years ago

    # Fetch the data
    print(f"Fetching the data for {ticker} from {start_date} to {end_date}")
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    # Check and display data
    if stock_data is not None:
        print(f"Fetched {len(stock_data)} rows of data.")
        print("First few rows:")
        print(stock_data.head())
        
        # Check for missing values
        missing_values = stock_data.isnull().sum()
        print("\nMissing values in each column:")
        print(missing_values)
        
        # Save the data
        filename = f"data/{ticker.lower()}_data.csv"  # Dynamic filename based on ticker
        save_data_to_csv(stock_data, filename)
    else:
        print("Failed to fetch data. Check ticker, dates, or connection.")

if __name__ == "__main__":
    main()