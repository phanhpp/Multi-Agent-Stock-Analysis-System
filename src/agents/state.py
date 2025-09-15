from typing import TypedDict, Dict, Any, List, Annotated
from src.data_collectors.price_loader import PriceDataLoader
from src.data_collectors.news_loader import NewsDataLoader
from src.data_collectors.fundamental_loader import FundamentalDataLoader

def merge_ticker_analyses(left: Dict[str, Dict[str, Any]], right: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Merge ticker analyses from multiple nodes"""
    merged = left.copy()
    merged.update(right)
    return merged

class TickerAnalysisState(TypedDict):
    """State for individual ticker analysis (subgraph)"""
    ticker: str
    as_of_date: str
    valuation_analysis: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    fundamental_analysis: Dict[str, Any]
    consensus_rating: str
    analysis_summary: Dict[str, Any]


class PortfolioState(TypedDict):
    """State for portfolio management (parent graph)"""
    as_of_date: str
    tickers: List[str]
    
    # Store complete analysis for each ticker - use Annotated for parallel updates
    ticker_analyses: Annotated[Dict[str, Dict[str, Any]], merge_ticker_analyses]
    
    # Portfolio construction results
    portfolio_composition: List[str]  # List of BUY-rated tickers
    portfolio_weights: Dict[str, float]  # weights for selected tickers
    


# Initialize data loaders once
price_loader = PriceDataLoader()
news_loader = NewsDataLoader()
fundamental_loader = FundamentalDataLoader()