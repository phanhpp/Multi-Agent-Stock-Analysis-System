from src.agents.state import TickerAnalysisState
from src.config import config
def coordinator(state: TickerAnalysisState) -> TickerAnalysisState:
    """
    Coordinate analysis from all agents to produce final consensus rating.
    Uses configurable weighting for all three agents.
    """
    cfg = config.coordinator
    # Agent names and equal weights
    AGENTS = ['valuation', 'sentiment', 'fundamental']
    
    # Extract recommendations and decision scores from each agent
    votes = {agent: state[f'{agent}_analysis']['recommendation'] for agent in AGENTS}
    decision_scores = {agent: state[f'{agent}_analysis']['decision_score'] for agent in AGENTS}
    
    # Weighted voting (your existing logic)
    buy_weight = 0
    if votes['valuation'] == 'BUY': buy_weight += cfg['weights']['valuation']
    if votes['sentiment'] == 'BUY': buy_weight += cfg['weights']['sentiment']
    if votes['fundamental'] == 'BUY': buy_weight += cfg['weights']['fundamental']
    

    # Determine consensus rating based on buy weight
    if buy_weight >= cfg['buy_weight_threshold']:
        consensus = "BUY"
    elif buy_weight <= cfg['sell_weight_threshold']:
        consensus = "SELL"
    else:
        consensus = "HOLD"
    
    # No overall composite score - agents only provide buy/hold/sell decisions
    
    # Create analysis summary for CSV export with both ratings and decision scores
    analysis_summary = {
        'ticker': state['ticker'],
        'consensus_rating': consensus,
        **{f'{agent}_rating': votes[agent] for agent in AGENTS},
        **{f'{agent}_decision_score': decision_scores[agent] for agent in AGENTS}
    }
    
    return {
        'consensus_rating': consensus,
        'analysis_summary': analysis_summary
    }
    
    
    