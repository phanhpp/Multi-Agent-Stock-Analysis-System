"""
Threshold calibration utilities for Alpha Agents.
Run this during development to set data-driven thresholds.
"""

import sys
from pathlib import Path

# Add src to path so we can import data loaders
sys.path.append(str(Path(__file__).parent.parent))

from data_collectors.news_loader import NewsDataLoader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment_distribution():
    """
    Analyze sentiment scores across all news data to calibrate thresholds.
    Run this once during development to set realistic BUY/SELL thresholds.
    """
    print("Calibrating sentiment thresholds based on actual news data...")
    
    news_loader = NewsDataLoader()
    analyzer = SentimentIntensityAnalyzer()
    
    all_scores = []
    
    for ticker in ['AAPL', 'MSFT', 'NVDA', 'TSLA']:
        try:
            articles = news_loader.load_news_data(ticker)
            print(f"Loaded {len(articles)} articles for {ticker}")
            
            for article in articles:
                text = f"{article['title']}. {article['snippet']}"
                score = analyzer.polarity_scores(text)['compound']
                all_scores.append({
                    'ticker': ticker,
                    'score': score,
                    'title': article['title'][:50] + "..." if len(article['title']) > 50 else article['title'],
                    'date': article['date']
                })
        except Exception as e:
            print(f"Warning: Could not load news for {ticker}: {e}")
    
    if not all_scores:
        print("Error: No sentiment scores calculated. Check your news data.")
        return
    
    scores_only = [item['score'] for item in all_scores]
    scores_only.sort()
    
    # Calculate statistics
    n = len(scores_only)
    mean_score = sum(scores_only) / n
    min_score = min(scores_only)
    max_score = max(scores_only)
    q25 = scores_only[n//4]
    q75 = scores_only[3*n//4]
    
    print(f"\n=== SENTIMENT SCORE DISTRIBUTION ===")
    print(f"Total articles analyzed: {n}")
    print(f"Score range: {min_score:.3f} to {max_score:.3f}")
    print(f"Mean score: {mean_score:.3f}")
    print(f"25th percentile: {q25:.3f}")
    print(f"75th percentile: {q75:.3f}")
    
    # Suggest thresholds
    suggested_buy = q75    # 0.477 - only top 25% positive news
    suggested_sell = q25   # -0.026 - only bottom 25% negative news

    print(f"\n=== SUGGESTED THRESHOLDS ===")
    print(f"buy_sentiment_threshold: {suggested_buy:.2f}")
    print(f"sell_sentiment_threshold: {suggested_sell:.2f}")
    print(f"Rationale: Based on quartile analysis of your news dataset")
    
    # Show examples
    all_scores.sort(key=lambda x: x['score'])
    
    print(f"\n=== MOST NEGATIVE ARTICLES ===")
    for item in all_scores[:3]:
        print(f"{item['ticker']}: {item['score']:.3f} - {item['title']} ({item['date']})")
    
    print(f"\n=== MOST POSITIVE ARTICLES ===")
    for item in all_scores[-3:]:
        print(f"{item['ticker']}: {item['score']:.3f} - {item['title']} ({item['date']})")
    

if __name__ == "__main__":
    analyze_sentiment_distribution()