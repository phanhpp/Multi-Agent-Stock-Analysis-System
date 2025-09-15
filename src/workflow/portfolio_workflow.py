from src.agents.state import PortfolioState
from typing import Dict, Any
from src.workflow.ticker_workflow import create_agent_workflow
from src.utils.graph_utils import show_workflow_graph
from langgraph.graph import StateGraph, START, END


def analyze_ticker(state: PortfolioState, ticker: str):
    """
    Run ticker analysis subgraph.
    Returns the analysis summary for portfolio construction.
    """
    # Prepare input for subgraph
    subgraph_input = {
        "ticker": ticker,
        "as_of_date": state["as_of_date"],
        "valuation_analysis": {},
        "sentiment_analysis": {},
        "fundamental_analysis": {},
        "consensus_rating": "",
        "analysis_summary": {}
    }
    
    # Run the subgraph
    ticker_analyzer = create_agent_workflow()
    analysis = ticker_analyzer.invoke(subgraph_input)

    #print(f"Analysis for {ticker}: {analysis}")
    
    # Extract the complete analysis
    return {
        "ticker_analyses": {
            f"{ticker}": analysis
        }
    }


def build_portfolio(state: PortfolioState):
    """Differential weighting: BUY gets full weight, HOLD gets reduced weight"""
    
    # Define weight multipliers
    WEIGHT_MAP = {
        "BUY": 1.0,    # Full conviction
        "HOLD": 0.5,   # Reduced conviction  
        "SELL": 0.0    # Exclude
    }
    
    # Calculate raw weights
    raw_weights = {}
    total_weight = 0
    
    for ticker in state["tickers"]:
        analysis = state["ticker_analyses"][ticker]
        rating = analysis["consensus_rating"]
        weight = WEIGHT_MAP.get(rating, 0)
        
        if weight > 0:
            raw_weights[ticker] = weight
            total_weight += weight
    
    # Normalize to sum to 1.0
    if total_weight > 0:
        portfolio_weights = {
            ticker: weight / total_weight 
            for ticker, weight in raw_weights.items()
        }
        
        return {
            "portfolio_weights": portfolio_weights,
            "portfolio_composition": list(portfolio_weights.keys())
        }
    else:
        return {"portfolio_weights": {}, "portfolio_composition": []}

def create_portfolio_graph():
    """Create portfolio graph with parallel ticker analysis."""
    graph = StateGraph(PortfolioState)
    
    # Add nodes using lambdas to call analyze_ticker with specific tickers
    graph.add_node("analyze_AAPL", lambda state: analyze_ticker(state, "AAPL"))
    graph.add_node("analyze_MSFT", lambda state: analyze_ticker(state, "MSFT"))
    graph.add_node("analyze_NVDA", lambda state: analyze_ticker(state, "NVDA"))
    graph.add_node("analyze_TSLA", lambda state: analyze_ticker(state, "TSLA"))
    
    # Portfolio builder node
    graph.add_node("build_portfolio", build_portfolio)
    
    # Parallel execution edges
    graph.add_edge(START, "analyze_AAPL")
    graph.add_edge(START, "analyze_MSFT")
    graph.add_edge(START, "analyze_NVDA")
    graph.add_edge(START, "analyze_TSLA")
    
    # Converge to portfolio builder
    graph.add_edge("analyze_AAPL", "build_portfolio")
    graph.add_edge("analyze_MSFT", "build_portfolio")
    graph.add_edge("analyze_NVDA", "build_portfolio")
    graph.add_edge("analyze_TSLA", "build_portfolio")
    
    graph.add_edge("build_portfolio", END)
    
    return graph.compile()


def show_portfolio_workflow_graph():
    """Display the portfolio analysis workflow graph"""
    portfolio_graph = create_portfolio_graph()
    show_workflow_graph(portfolio_graph, "outputs/workflow_diagrams/portfolio_construction.png")


# Usage
def run_portfolio_workflow(as_of_date: str = "2024-08-20") -> Dict[str, Any]:
    """
    Run portfolio analysis for multiple tickers.
    
    Args:
        as_of_date: Analysis date in YYYY-MM-DD format
        
    Returns:
        Dictionary containing portfolio analysis results
    """
    portfolio_graph = create_portfolio_graph()
    
    initial_state = {
        "as_of_date": as_of_date,
        "tickers": ["AAPL", "MSFT", "NVDA", "TSLA"],
        "ticker_analyses": {},
        "portfolio_composition": [],
        "portfolio_weights": {},
    }
    
    result = portfolio_graph.invoke(initial_state)
    return result


# Uncomment the line below to run the portfolio analysis
if __name__ == "__main__":
    run_portfolio_workflow()
