import pandas as pd
import requests
import os
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class FinancialNewsFetcher:
    """
    A class to fetch financial news for specified ticker symbols and analyze sentiment.
    """
    def __init__(self, model_name="ProsusAI/finbert", cache_dir="data/models/finbert"):
        """
        Initialize the news fetcher and sentiment analyzer.
        
        Args:
            model_name (str): HuggingFace model to use for sentiment analysis.
            cache_dir (str): Directory to cache the downloaded model.
        """
        self.cache_dir = cache_dir
        self.model_name = model_name
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize finbert model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Set device (use GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # FinBERT label mapping
        self.labels = ["negative", "neutral", "positive"]
    
    def _fetch_news_api(self, ticker, start_date, end_date, api_key=None):
        """
        Fetch financial news for a ticker symbol using an API.
        """
        if api_key is None:
            api_key = os.getenv("NEWS_API_KEY")  # Changed from os.environ.get to os.getenv
            if not api_key:
                raise ValueError("API key required. Set NEWS_API_KEY in .env file or pass as parameter")
        
        URL = f"https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": start_date,
            "to": end_date,
            "token": api_key
        }
        response = requests.get(URL, params=params)
        if response.status_code == 200:
            news_json = response.json()
            # Finnhub returns a list of news dicts with 'headline', 'datetime', etc.
            news_data = []
            for item in news_json:
                news_data.append({
                    "date": datetime.utcfromtimestamp(item["datetime"]).strftime("%Y-%m-%d"),
                    "headline": item["headline"],
                    "source": item.get("source", ""),
                    "url": item.get("url", "")
                })
            return news_data
        else:
            print(f"Error fetching news: {response.status_code}")
            return None
    
    def analyze_sentiment(self, texts):
        """
        Analyze the sentiment of a list of news headlines or text snippets.
        
        Args:
            texts (list): List of text strings to analyze
            
        Returns:
            DataFrame with sentiment scores for each text
        """
        results = []
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            scores = torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]
            sentiment_score = scores[2] - scores[0]  # positive - negative
            
            label_idx = scores.argmax()
            sentiment_label = self.labels[label_idx]
            
            results.append({
                "text": text,
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score,
                "negative": scores[0],
                "neutral": scores[1],
                "positive": scores[2]
            })
        
        return pd.DataFrame(results)
    
    def fetch_and_analyze(self, ticker, start_date, end_date, api_key=None):
        """
        Fetch news and analyze sentiment for a ticker within a date range.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., "AAPL")
            start_date (str): Start date in "YYYY-MM-DD" format
            end_date (str): End date in "YYYY-MM-DD" format
            api_key (str, optional): API key for news service
            
        Returns:
            DataFrame with news and sentiment scores
        """
        # Fetch news data
        news_data = self._fetch_news_api(ticker, start_date, end_date, api_key)
        
        if not news_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        news_df = pd.DataFrame(news_data)
        
        # Extract headlines for sentiment analysis
        headlines = news_df["headline"].tolist()
        
        # Analyze sentiment
        sentiment_df = self.analyze_sentiment(headlines)
        
        # Combine news data with sentiment analysis
        result_df = pd.concat([news_df.reset_index(drop=True), 
                              sentiment_df.drop("text", axis=1).reset_index(drop=True)], 
                              axis=1)
        
        # Ensure date column is in datetime format
        result_df["date"] = pd.to_datetime(result_df["date"])
        
        return result_df
    
    def aggregate_daily_sentiment(self, sentiment_df):
        """
        Aggregate sentiment scores by day.
        
        Args:
            sentiment_df (DataFrame): DataFrame with sentiment scores
            
        Returns:
            DataFrame with daily aggregated sentiment scores
        """
        if sentiment_df.empty:
            return pd.DataFrame()
        
        # Group by date and calculate average sentiment and count
        daily_sentiment = sentiment_df.groupby("date").agg({
            "sentiment_score": "mean",
            "negative": "mean",
            "neutral": "mean", 
            "positive": "mean",
            "headline": "count"  # Count of news articles per day
        }).reset_index()
        
        # Rename count column
        daily_sentiment.rename(columns={"headline": "news_count"}, inplace=True)
        
        return daily_sentiment


def main():
    """
    Example usage of the FinancialNewsFetcher class.
    """
    # Define stock ticker and date range
    ticker = "AAPL"
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    print(f"Fetching news for {ticker} from {start_date} to {end_date}")
    
    # Initialize news fetcher
    news_fetcher = FinancialNewsFetcher()
    
    # Fetch and analyze news
    sentiment_df = news_fetcher.fetch_and_analyze(ticker, start_date, end_date)
    
    if not sentiment_df.empty:
        print(f"Fetched {len(sentiment_df)} news articles")
        print("\nSample of news with sentiment:")
        print(sentiment_df.head())
        
        # Aggregate daily sentiment
        daily_sentiment = news_fetcher.aggregate_daily_sentiment(sentiment_df)
        print("\nDaily sentiment aggregation:")
        print(daily_sentiment.head())
        
        # Save to CSV
        output_file = f"data/{ticker.lower()}_sentiment.csv"
        daily_sentiment.to_csv(output_file, index=False)
        print(f"\nSaved daily sentiment data to {output_file}")
    else:
        print("No news data found for the specified period.")


if __name__ == "__main__":
    main()