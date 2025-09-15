from src.agents.state import TickerAnalysisState, news_loader
from src.config import config
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def sentiment_agent(state: TickerAnalysisState) -> TickerAnalysisState:
    ticker = state['ticker']
    as_of_date = state['as_of_date']
    
    # Use your existing news loader
    news_articles = news_loader.get_news_for_as_of_date(ticker, as_of_date)
    
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    
    scores = []
    for article in news_articles:
        # Combine title and snippet for richer context
        text = f"{article['title']}. {article['snippet']}"
        sentiment = analyzer.polarity_scores(text)
        scores.append(sentiment['compound'])
    
    avg_sentiment = sum(scores) / len(scores) if scores else 0
    
    # Use config thresholds
    cfg = config.sentiment
    if avg_sentiment > cfg['buy_sentiment_threshold']:
        recommendation = "BUY"
    elif avg_sentiment < cfg['sell_sentiment_threshold']:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
    
    return {'sentiment_analysis': {
        # Rating (categorical)
        'recommendation': recommendation,
        
        # Score used for decision (for transparency)
        'decision_score': round(avg_sentiment, 3),  # -1 to 1 VADER scale used for decision
        
        # Metadata
        'metadata': {
            'article_count': len(news_articles),
            'individual_scores': scores[:5],  # Show first 5 for transparency
            'thresholds_used': {
                'buy_threshold': cfg['buy_sentiment_threshold'],
                'sell_threshold': cfg['sell_sentiment_threshold']
            }
        }
    }}
    
    

# Alternatively, we can use a simple tool for summarizing news articles
# use Langchain's WebBaseLoader to scrape the web
# then use LangGraph's map-reduce to summarize the articles
# https://python.langchain.com/docs/tutorials/summarization/#map-reduce
