from src.config import config
from src.agents.state import TickerAnalysisState, price_loader

def valuation_agent(state: TickerAnalysisState) -> TickerAnalysisState:
    ticker = state['ticker']
    as_of_date = state['as_of_date']
    
    # Get thresholds from config
    cfg = config.valuation
    
    # Get price data
    start_date, _ = price_loader.calculate_date_range(
        as_of_date, 
        lookback_days=cfg['lookback_days'], 
        forward_days=0
    )
    price_data = price_loader.get_price_data([ticker], start_date, as_of_date)
    
    # BlackRock paper formulas
    n_days = len(price_data)
    start_price = price_data['close'].iloc[0]
    end_price = price_data['close'].iloc[-1]
    R_cumulative = (end_price / start_price) - 1
    
    # Annualized return
    R_annualized = ((1 + R_cumulative) ** (252 / n_days)) - 1
    
    # Annualized volatility
    daily_returns = price_data['close'].pct_change().dropna()
    sigma_daily = daily_returns.std()
    sigma_annualized = sigma_daily * (252 ** 0.5)
    
    # Convert to percentages
    return_pct = R_annualized * 100
    vol_pct = sigma_annualized * 100
    
    # Use configurable thresholds
    if (return_pct > cfg['buy_return_threshold'] and 
        vol_pct < cfg['buy_volatility_threshold']):
        recommendation = "BUY"
    elif (return_pct < cfg['sell_return_threshold'] or 
          vol_pct > cfg['sell_volatility_threshold']):
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
    
    # To do: add LLM here to generate natural language analysis

    return {'valuation_analysis' : {
        # Rating (categorical)
        'recommendation': recommendation,
        
        # Scores used for decision (for transparency)
        'decision_score': {
            'annualized_return_pct': round(return_pct, 2),
            'annualized_volatility_pct': round(vol_pct, 2)
        },
        
        # Metadata (for transparency)
        'metadata': {
            'lookback_days': n_days,
            'thresholds_used': {
                'buy_return': cfg['buy_return_threshold'],
                'sell_return': cfg['sell_return_threshold'],
                'buy_volatility': cfg['buy_volatility_threshold'],
                'sell_volatility': cfg['sell_volatility_threshold']
            }
        }
    }}
    
   