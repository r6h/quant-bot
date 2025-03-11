import pandas as pd
import os

def load_trade_log(log_file="monitoring/logs/trade_log.txt"):
    """Load trade log into a DataFrame."""
    if not os.path.exists(log_file):
        return pd.DataFrame(columns=['Timestamp', 'Ticker', 'Action', 'Qty', 'Price'])
    return pd.read_csv(log_file, names=['Timestamp', 'Ticker', 'Action', 'Qty', 'Price'])

def calculate_performance(trades, initial_capital=10000):
    """Calculate performance metrics from trade log."""
    if trades.empty:
        print("No trades to analyze.")
        return None
    
    trades['Timestamp'] = pd.to_datetime(trades['Timestamp'])
    trades['Dollar_Value'] = trades['Qty'] * trades['Price'] * (-1 if trades['Action'] == 'sell' else 1)
    portfolio_value = initial_capital + trades['Dollar_Value'].cumsum()
    
    total_profit = portfolio_value.iloc[-1] - initial_capital
    total_return = (total_profit / initial_capital) * 100
    
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Portfolio Value: ${portfolio_value.iloc[-1]:.2f}")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    
    return portfolio_value

def main():
    trades = load_trade_log()
    if not trades.empty:
        portfolio_value = calculate_performance(trades)
    else:
        print("No trades logged yet.")

if __name__ == "__main__":
    if not os.path.exists("monitoring"):
        os.makedirs("monitoring")
    main()