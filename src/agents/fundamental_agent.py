from src.agents.state import TickerAnalysisState, fundamental_loader
from src.config import config

def fundamental_agent(state: TickerAnalysisState) -> TickerAnalysisState:
    ticker = state['ticker']
    
    # Get thresholds from config
    cfg = config.fundamental
    
    # Use your fundamental loader
    fund_data = fundamental_loader.load_fundamental_data(ticker)
    
    # Calculate scores (your existing logic)
    total_score = 0
    factor_count = 0
    
    for metric_name, metric_data in fund_data.items():
        if isinstance(metric_data, dict) and 'score' in metric_data:
            score = metric_data['score']
            total_score += score
            factor_count += 1
    
    avg_score = total_score / factor_count if factor_count > 0 else 3
    
    # Determine rating
    if avg_score > cfg['buy_score_threshold']:
        recommendation = "BUY"
    elif avg_score < cfg['sell_score_threshold']:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
    
    return {'fundamental_analysis': {
        # Rating (categorical)
        'recommendation': recommendation,
        
        # Score used for decision (for transparency)
        'decision_score': round(avg_score, 2),  # Average score used for decision
        
        # Metadata
        'metadata': {
            'factors_analyzed': factor_count,
            'thresholds_used': {
                'buy_threshold': cfg['buy_score_threshold'],
                'sell_threshold': cfg['sell_score_threshold']
            }
        }
    }}