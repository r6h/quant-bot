import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle
import os

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
    
    # Features: SMA_14, RSI_14, Close_Lag1
    X = data[['SMA_14', 'RSI_14', 'Close_Lag1']]
    y = data['Target']

    return X, y

def train_model(X, y, test_size=0.2):
    """
    Train an XGBoost model with time-based split.
    
    Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Target.
        test_size (float): Fraction of data for testing.
    
    Returns:
        model: Trained XGBoost model.
        X_train, X_test, y_train, y_test: Split data.
    """
    # Time-based split (train on earlier data, test on later)
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Train XGBoost model
    model = xgb.XGBClassifier(random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

def save_model(model, filename):
    """
    Save the trained model to a file.
    
    Args:
        model: Trained model.
        filename (str): Path to save the model.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def main():
    # Load features
    input_file = "data/aapl_features.csv"
    print(f"Loading features from {input_file}...")
    data = load_features(input_file)

    # Prepare data
    print("Preparing data for modeling...")
    X, y = prepare_data(data)

    # Train model
    print("Training XGBoost model...")
    model, X_train, X_test, y_train, y_test = train_model(X, y)

    # Basic evaluation
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")

    # Save model
    base_name = os.path.splitext(os.path.basename(input_file))[0].replace("_features", "")
    model_file = f"models/{base_name}_model.pkl"
    save_model(model, model_file)

if __name__ == "__main__":
    main()