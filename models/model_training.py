import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle
import os
import argparse

def load_features(filename):
    """
    Load the processed features from a CSV file.
    
    Args:
        filename (str): Path to the features CSV file.
    
    Returns:
        pandas.DataFrame: Loaded data with features.
    """
    data = pd.read_csv(filename)
    data["Date"] = pd.to_datetime(data["Date"]) # Ensure Date is datetime
    return data

def prepare_data(data):
    """
    Prepare features and target for modeling.
    
    Args:
        data (pandas.DataFrame): Data with features.
    
    Returns:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Target variable.
    """
    # Define target: 1 if next day's Close > current Close, 0 otherwise
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    # Drop the last row (no next day to predict)
    data = data.dropna()
    
    # Define feature sets
    technical_features = ['SMA_14', 'RSI_14', 'Close_Lag1']
    
    # Check if sentiment features are available
    sentiment_features = ['sentiment_score', 'news_count', 'negative', 'neutral', 'positive']
    available_features = technical_features.copy()
    
    for feature in sentiment_features:
        if feature in data.columns:
            available_features.append(feature)
    
    print(f"Using features: {', '.join(available_features)}")
    
    # Use all available features
    X = data[available_features]
    y = data['Target']

    return X, y

def train_model(X, y, test_size=0.2, n_estimators=100, learning_rate=0.1, max_depth=6):
    """
    Train an XGBoost model with time-based split.
    
    Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Target.
        test_size (float): Fraction of data for testing.
        n_estimators (int): Number of trees for XGBoost.
        learning_rate (float): Learning rate for XGBoost.
        max_depth (int): Maximum tree depth for XGBoost.
    
    Returns:
        model: Trained XGBoost model.
        X_train, X_test, y_train, y_test: Split data.
    """
    # Time-based split (train on earlier data, test on later)
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Train XGBoost model
    model = xgb.XGBClassifier(
        random_state=42, 
        eval_metric="logloss",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )
    
    # Fit model with early stopping
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )

    # Print feature importance
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(importances)

    return model, X_train, X_test, y_train, y_test

def save_model(model, filename):
    """
    Save the trained model to a file.
    
    Args:
        model: Trained model.
        filename (str): Path to save the model.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train stock prediction model")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--test-size", type=float, default=0.2, 
                       help="Fraction of data to use for testing (default: 0.2)")
    parser.add_argument("--n-estimators", type=int, default=100,
                       help="Number of trees for XGBoost (default: 100)")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                       help="Learning rate for XGBoost (default: 0.1)")
    parser.add_argument("--max-depth", type=int, default=6,
                       help="Maximum tree depth for XGBoost (default: 6)")
    
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    ticker_lower = ticker.lower()
    
    # Load features
    input_file = f"data/{ticker_lower}_features.csv"
    print(f"Loading features from {input_file}...")
    data = load_features(input_file)

    # Prepare data
    print("Preparing data for modeling...")
    X, y = prepare_data(data)

    # Train model
    print("Training XGBoost model...")
    model, X_train, X_test, y_train, y_test = train_model(
        X, y, 
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth
    )

    # Basic evaluation
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")

    # Save model
    model_file = f"models/{ticker_lower}_model.pkl"
    save_model(model, model_file)

if __name__ == "__main__":
    main()