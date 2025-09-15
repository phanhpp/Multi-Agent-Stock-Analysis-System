"""
News data loader for Alpha Agents system.
Loads hand-curated news data from existing local JSON files.

This handles the news/sentiment data requirement:
- Load from data/news/<ticker>.json files
- Filter by as-of date to prevent data leakage
- Fields: title, snippet, date, source, url
"""

import json
import os
import logging
from datetime import datetime
from typing import List, Dict
from pathlib import Path

# Configure logger for this module
logger = logging.getLogger(__name__)

class NewsDataLoader:
    """
    Loads news data from existing local JSON files.
    
    As required by assessment:
    - Small, local JSON of headlines/snippets per ticker (5–15 items each)
    - Hand-curated from reputable sources 
    - Store locally (data/news/<ticker>.json)
    - Fields: title, snippet, date, source, url
    """
    
    def __init__(self):
        self.news_dir = Path('data/news')
        self.supported_tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA']
    
    def load_news_data(self, ticker: str) -> List[Dict]:
        """
        Load all news data for a specific ticker from JSON file
        
        Args:
            ticker: Stock symbol (AAPL, MSFT, NVDA, TSLA)
            
        Returns:
            List of news articles with fields: title, snippet, date, source, url
            
        Raises:
            FileNotFoundError: If news file doesn't exist
            ValueError: If ticker not supported
        """
        if ticker not in self.supported_tickers:
            raise ValueError(f"Unsupported ticker: {ticker}. Supported: {self.supported_tickers}")
            
        news_file = self.news_dir / f'{ticker}.json'
        
        if not news_file.exists():
            raise FileNotFoundError(f"News data not found: {news_file}")
        
        try:
            with open(news_file, 'r') as f:
                articles = json.load(f)
            
            # Validate required fields
            required_fields = ['title', 'snippet', 'date', 'source']
            for i, article in enumerate(articles):
                missing_fields = [field for field in required_fields if field not in article]
                if missing_fields:
                    raise ValueError(f"Article {i} missing fields: {missing_fields}")
            
            return articles
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {news_file}: {e}")
        except Exception as e:
            raise Exception(f"Error loading news data for {ticker}: {e}")
    
    def get_news_for_as_of_date(self, ticker: str, as_of_date: str) -> List[Dict]:
        """
        Get news articles for a ticker up to (but not after) the as-of date.
        This prevents data leakage - agents only see news before their decision date.
        
        Args:
            ticker: Stock symbol
            as_of_date: Decision date in 'YYYY-MM-DD' format
            
        Returns:
            List of news articles published before or on the as-of date
        """
        articles = self.load_news_data(ticker)
        
        # Filter to prevent data leakage
        filtered_articles = []
        for article in articles:
            if article['date'] <= as_of_date:
                filtered_articles.append(article)
        
        return filtered_articles
    
    def get_news_for_date_range(self, ticker: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Get news articles within specific date range
        
        Args:
            ticker: Stock symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            List of news articles within the date range
        """
        articles = self.load_news_data(ticker)
        
        filtered_articles = []
        for article in articles:
            if start_date <= article['date'] <= end_date:
                filtered_articles.append(article)
        
        return filtered_articles
    
    def get_all_news_summary(self) -> Dict:
        """
        Get summary of all news data across tickers
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_articles': 0,
            'by_ticker': {},
            'date_range': {'earliest': None, 'latest': None},
            'sources': set()
        }
        
        all_dates = []
        
        for ticker in self.supported_tickers:
            try:
                articles = self.load_news_data(ticker)
                summary['by_ticker'][ticker] = {
                    'count': len(articles),
                    'date_range': {
                        'start': min(a['date'] for a in articles),
                        'end': max(a['date'] for a in articles)
                    }
                }
                summary['total_articles'] += len(articles)
                
                # Collect all dates and sources
                for article in articles:
                    all_dates.append(article['date'])
                    summary['sources'].add(article['source'])
                    
            except FileNotFoundError:
                summary['by_ticker'][ticker] = {'count': 0, 'error': 'File not found'}
        
        if all_dates:
            summary['date_range']['earliest'] = min(all_dates)
            summary['date_range']['latest'] = max(all_dates)
        
        summary['sources'] = list(summary['sources'])  # Convert set to list
        
        return summary
    
    def validate_news_data(self) -> tuple[bool, List[str]]:
        """
        Validate that all required news files exist and have proper format
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        for ticker in self.supported_tickers:
            news_file = self.news_dir / f'{ticker}.json'
            
            if not news_file.exists():
                issues.append(f"Missing news file: {news_file}")
                continue
            
            try:
                articles = self.load_news_data(ticker)
                
                if len(articles) == 0:
                    issues.append(f"{ticker}: No articles found")
                elif len(articles) < 5:
                    issues.append(f"{ticker}: Only {len(articles)} articles (minimum 5 recommended)")
                elif len(articles) > 15:
                    issues.append(f"{ticker}: {len(articles)} articles (maximum 15 recommended)")
                
                # Check date format
                for i, article in enumerate(articles):
                    try:
                        datetime.strptime(article['date'], '%Y-%m-%d')
                    except ValueError:
                        issues.append(f"{ticker} article {i}: Invalid date format '{article['date']}'")
                
            except Exception as e:
                issues.append(f"{ticker}: Error loading data - {e}")
        
        return len(issues) == 0, issues


def load_all_news_for_as_of_date(as_of_date: str) -> Dict[str, List[Dict]]:
    """
    Convenience function to load news for all tickers up to as-of date
    
    Args:
        as_of_date: Decision date in 'YYYY-MM-DD' format
        
    Returns:
        Dictionary mapping ticker -> list of news articles
    """
    loader = NewsDataLoader()
    news_data = {}
    
    for ticker in loader.supported_tickers:
        try:
            news_data[ticker] = loader.get_news_for_as_of_date(ticker, as_of_date)
        except Exception as e:
            logger.warning(f"Failed to load news for {ticker}: {e}")
            news_data[ticker] = []
    
    return news_data


if __name__ == "__main__":
    """Test the news loader"""
    # Configure logging for demo
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    loader = NewsDataLoader()
    
    print("News Data Validation:")
    is_valid, issues = loader.validate_news_data()
    
    if is_valid:
        print("✅ All news data valid")
        summary = loader.get_all_news_summary()
        print(f"📰 Total articles: {summary['total_articles']}")
        print(f"📅 Date range: {summary['date_range']['earliest']} to {summary['date_range']['latest']}")
        
        for ticker, info in summary['by_ticker'].items():
            print(f"   {ticker}: {info['count']} articles")
    else:
        print("❌ News data issues found:")
        for issue in issues:
            print(f"   - {issue}")