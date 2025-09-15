"""
Single ticker analysis workflow module.

This module provides the core ticker analysis workflow that coordinates
the three specialized agents (Valuation, Sentiment, Fundamental) and 
produces consensus ratings.

Usage:
    # Import functions (silent)
    from src.workflow.ticker_workflow import create_agent_workflow
    
    # Run demo analysis
    python -m src.workflow.ticker_workflow
"""

from src.agents.state import TickerAnalysisState
from src.agents.valuation_agent import valuation_agent
from src.agents.sentiment_agent import sentiment_agent
from src.agents.fundamental_agent import fundamental_agent
from src.agents.coordinator import coordinator
from src.utils.graph_utils import show_workflow_graph
from langgraph.graph import StateGraph, END, START

def create_agent_workflow():
    workflow = StateGraph(TickerAnalysisState)
    
    # Add nodes
    workflow.add_node("valuation", valuation_agent)
    workflow.add_node("sentiment", sentiment_agent)
    workflow.add_node("fundamental", fundamental_agent)
    workflow.add_node("coordinator", coordinator)
    
    # Add edges (parallel execution for agents)
    workflow.add_edge(START, "valuation")
    workflow.add_edge(START, "sentiment")
    workflow.add_edge(START, "fundamental")
    workflow.add_edge("valuation", "coordinator")
    workflow.add_edge("sentiment", "coordinator")
    workflow.add_edge("fundamental", "coordinator")
    workflow.add_edge("coordinator", END)
    
    return workflow.compile()

def show_ticker_workflow_graph():
    """Display the ticker analysis workflow graph"""
    app = create_agent_workflow()
    show_workflow_graph(app, "outputs/workflow_diagrams/ticker_analysis.png")


def run_single_ticker_analysis(ticker: str = "AAPL", as_of_date: str = "2024-08-20"):
    """
    Run analysis for a single ticker and print results.
    
    Args:
        ticker: Stock symbol to analyze
        as_of_date: Analysis date in YYYY-MM-DD format
        
    Returns:
        Dict containing analysis results
    """
    app = create_agent_workflow()
    
    result = app.invoke({
        "ticker": ticker,
        "as_of_date": as_of_date,
        "valuation_analysis": {},
        "sentiment_analysis": {},
        "fundamental_analysis": {},
        "consensus_rating": "",
        "analysis_summary": {},
    })

    # Print results to terminal
    print("\n" + "="*60)
    print("🎯 STOCK ANALYSIS RESULTS")
    print("="*60)
    print(f"📊 Ticker: {result['ticker']}")
    print(f"📅 Analysis Date: {result['as_of_date']}")
    print(f"🏆 Final Rating: {result['consensus_rating']}")

    print(f"\n📋 Individual Agent Results:")
    print("-" * 40)

    # Valuation Agent
    val_analysis = result['valuation_analysis']
    print(f"💰 Valuation Agent:")
    print(f"   Rating: {val_analysis['recommendation']}")
    print(f"   Decision Score - Return: {val_analysis['decision_score']['annualized_return_pct']:.2f}%")
    print(f"   Decision Score - Volatility: {val_analysis['decision_score']['annualized_volatility_pct']:.2f}%")

    # Sentiment Agent  
    sent_analysis = result['sentiment_analysis']
    print(f"\n📰 Sentiment Agent:")
    print(f"   Rating: {sent_analysis['recommendation']}")
    print(f"   Decision Score: {sent_analysis['decision_score']:.3f} (VADER scale)")
    print(f"   Articles: {sent_analysis['metadata']['article_count']}")

    # Fundamental Agent
    fund_analysis = result['fundamental_analysis']
    print(f"\n🏢 Fundamental Agent:")
    print(f"   Rating: {fund_analysis['recommendation']}")
    print(f"   Decision Score: {fund_analysis['decision_score']:.2f}/5")
    print(f"   Factors: {fund_analysis['metadata']['factors_analyzed']}")

    print("\n" + "="*60)

    # Also print the analysis summary if available
    if 'analysis_summary' in result and result['analysis_summary']:
        print("📊 Analysis Summary for CSV:")
        summary = result['analysis_summary']
        for key, value in summary.items():
            print(f"   {key}: {value}")
        print("="*60)
    
    return result


if __name__ == "__main__":
    # Only run when executed directly, not when imported
    print("Running single ticker analysis demo...")
    run_single_ticker_analysis()