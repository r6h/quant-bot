import os
from datetime import datetime

def log_trade(ticker, action, qty, price, timestamp=None):
    """Log a trade to a file."""
    if timestamp is None:
        timestamp = datetime.now()
    
    log_dir = "monitoring/logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = f"{log_dir}/trade_log.txt"
    with open(log_file, 'a') as f:
        f.write(f"{timestamp},{ticker},{action},{qty:.6f},${price:.2f}\n")