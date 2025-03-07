import pandas as pd
import pickle
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import os

def load_model_and_data(model_file, features_file):
    """Load the trained model and feature data."""
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    data = pd.read_csv(features_file)
    data['Date'] = pd.to_datetime(data['Date'])
    return model, data

def prepare_data(data):
    """Prepare features and target for evaluation."""
    # Recreate Target column same as in training
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data = data.dropna()  # Drop NaN from shift
    X = data[['SMA_14', 'RSI_14', 'Close_Lag1']]
    y = data['Target']
    return X, y

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    y_pred = model.predict(X_test)

    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (Correct 'up' predictions / All 'up' predictions)")
    print(f"Recall: {recall:.4f} (Correct 'up' predictions / All actual 'ups')")
    print("Confusion Matrix:")
    print(conf_matrix)
    # [True Negatives, False Positives]
    # [False Negatives, True Positives]

def main():
    ticker = "AAPL"
    model_file = f"models/{ticker.lower()}_model.pkl"
    features_file = f"data/{ticker.lower()}_features.csv"

    print(f"Loading model and data for {ticker}...")
    model, data = load_model_and_data(model_file, features_file)

    # Prepare data with Target
    print("Preparing data for evaluation...")
    X , y = prepare_data(data)
    # Prepare test data (same split as training)
    split_index = int(len(X) * 0.8)  # 80% train, 20% test
    X_test, y_test = X.iloc[split_index:], y.iloc[split_index:]

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()