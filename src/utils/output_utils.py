"""
Output utilities for Alpha Agents pipeline.

This module handles all CSV file generation and output formatting for the multi-agent
stock analysis system. It provides clean separation between analysis logic and 
output presentation, enabling picks.csv generation independent of backtesting.

Key Functions:
    - save_picks_csv: Agent ratings and consensus decisions per ticker
    - save_performance_csv: Portfolio vs benchmark performance metrics  
    - ensure_output_directory: Directory creation utility
    - format_agent_metadata: Consistent metadata formatting

Usage:
    from src.utils.output_utils import save_picks_csv, save_performance_csv
    
    # Save agent decisions (independent of backtesting)
    picks_path = save_picks_csv(portfolio_result)
    
    # Save performance metrics (requires backtest results)
    performance_path = save_performance_csv(backtest_result, portfolio_result)
"""

import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def ensure_output_directory(output_path: str) -> None:
    """
    Ensure the output directory exists, creating parent directories if needed.
    
    Args:
        output_path: Full path to output file including filename
        
    Raises:
        OSError: If directory creation fails due to permissions
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)


def save_picks_csv(portfolio_result: Dict[str, Any], 
                   output_path: str = "outputs/picks.csv") -> str:
    """
    Save per-ticker agent ratings and consensus decisions to CSV.
    
    This function extracts individual agent recommendations, decision scores,
    and final consensus ratings for each ticker in the analysis universe.
    Can be called independently of backtesting for rapid development cycles.
    
    Args:
        portfolio_result: Complete portfolio analysis results from workflow
        output_path: Output CSV file path (default: outputs/picks.csv)
        
    Returns:
        str: Absolute path to saved CSV file
        
    Raises:
        KeyError: If required keys missing from portfolio_result
        ValueError: If no ticker analyses found in results
        
    CSV Columns:
        - ticker: Stock symbol (AAPL, MSFT, etc.)
        - valuation_rating: BUY/HOLD/SELL from valuation agent
        - valuation_return_pct: Annualized return percentage 
        - valuation_volatility_pct: Annualized volatility percentage
        - sentiment_rating: BUY/HOLD/SELL from sentiment agent
        - sentiment_score: VADER compound score (-1 to 1)
        - sentiment_articles: Number of news articles analyzed
        - fundamental_rating: BUY/HOLD/SELL from fundamental agent
        - fundamental_score: Average fundamental score (1-5 scale)
        - fundamental_factors: Number of factors analyzed
        - consensus_rating: Final coordinated decision
        - portfolio_weight: Allocation in final portfolio (0.0-1.0)
    """
    
    if not portfolio_result.get("ticker_analyses"):
        raise ValueError("No ticker analyses found in portfolio result")
    
    picks_data = []
    tickers = portfolio_result.get("tickers", [])
    portfolio_weights = portfolio_result.get("portfolio_weights", {})
    
    for ticker in tickers:
        if ticker not in portfolio_result["ticker_analyses"]:
            continue
            
        analysis = portfolio_result["ticker_analyses"][ticker]
        
        # Extract agent-specific data with error handling
        try:
            valuation_analysis = analysis["valuation_analysis"]
            sentiment_analysis = analysis["sentiment_analysis"]
            fundamental_analysis = analysis["fundamental_analysis"]
            
            # Build row data
            row_data = {
                "ticker": ticker,
                
                # Valuation agent outputs
                "valuation_rating": valuation_analysis["recommendation"],
                "valuation_return_pct": round(
                    valuation_analysis["decision_score"]["annualized_return_pct"], 2
                ),
                "valuation_volatility_pct": round(
                    valuation_analysis["decision_score"]["annualized_volatility_pct"], 2
                ),
                
                # Sentiment agent outputs  
                "sentiment_rating": sentiment_analysis["recommendation"],
                "sentiment_score": round(sentiment_analysis["decision_score"], 3),
                "sentiment_articles": sentiment_analysis["metadata"]["article_count"],
                
                # Fundamental agent outputs
                "fundamental_rating": fundamental_analysis["recommendation"], 
                "fundamental_score": round(fundamental_analysis["decision_score"], 2),
                "fundamental_factors": fundamental_analysis["metadata"]["factors_analyzed"],
                
                # Coordinator outputs
                "consensus_rating": analysis["consensus_rating"],
                "portfolio_weight": round(portfolio_weights.get(ticker, 0.0), 4)
            }
            
            picks_data.append(row_data)
            
        except KeyError as e:
            print(f"Warning: Missing data for {ticker}, key {e}. Skipping.")
            continue
    
    if not picks_data:
        raise ValueError("No valid ticker data found for CSV generation")
    
    # Create DataFrame and save
    ensure_output_directory(output_path)
    picks_df = pd.DataFrame(picks_data)
    picks_df.to_csv(output_path, index=False)
    
    abs_path = Path(output_path).resolve()
    print(f"Agent picks saved to: {abs_path}")
    
    return str(abs_path)


def save_performance_csv(backtest_result: Dict[str, Any],
                        portfolio_result: Dict[str, Any],
                        output_path: str = "outputs/performance.csv") -> str:
    """
    Save portfolio vs benchmark performance metrics to CSV.
    
    This function compiles comprehensive performance analytics including
    returns, risk metrics, and portfolio composition statistics from
    both backtesting results and original portfolio analysis.
    
    Args:
        backtest_result: Performance metrics from backtesting engine
        portfolio_result: Original portfolio analysis for metadata
        output_path: Output CSV file path (default: outputs/performance.csv)
        
    Returns:
        str: Absolute path to saved CSV file
        
    Raises:
        KeyError: If required performance metrics missing
        ValueError: If backtest results are invalid
        
    CSV Columns:
        - decision_date: As-of date for agent analysis (YYYY-MM-DD)
        - end_date: Final date of backtest period (YYYY-MM-DD)
        - test_period_days: Actual number of trading days tested
        - portfolio_return_pct: Total portfolio return percentage
        - benchmark_return_pct: Equal-weight benchmark return percentage  
        - excess_return_pct: Portfolio outperformance (can be negative)
        - portfolio_volatility_pct: Annualized portfolio volatility
        - benchmark_volatility_pct: Annualized benchmark volatility
        - portfolio_sharpe: Risk-adjusted return ratio for portfolio
        - benchmark_sharpe: Risk-adjusted return ratio for benchmark
        - num_buy_ratings: Count of BUY consensus decisions
        - num_hold_ratings: Count of HOLD consensus decisions
        - num_sell_ratings: Count of SELL consensus decisions
        - portfolio_stocks: Number of stocks in final portfolio
        - portfolio_composition: Comma-separated list of included tickers
    """
    
    required_keys = ["portfolio_return", "benchmark_return", "as_of_date"]
    for key in required_keys:
        if key not in backtest_result:
            raise KeyError(f"Required backtest result key missing: {key}")
    
    # Count consensus ratings
    rating_counts = {"BUY": 0, "HOLD": 0, "SELL": 0}
    if "ticker_analyses" in portfolio_result:
        for analysis in portfolio_result["ticker_analyses"].values():
            rating = analysis.get("consensus_rating", "UNKNOWN")
            if rating in rating_counts:
                rating_counts[rating] += 1
    
    # Portfolio composition summary
    portfolio_weights = portfolio_result.get("portfolio_weights", {})
    portfolio_tickers = list(portfolio_weights.keys())
    composition_str = ", ".join(portfolio_tickers) if portfolio_tickers else "Empty"
    
    # Compile performance row
    performance_data = [{
        "decision_date": backtest_result["as_of_date"],
        "end_date": backtest_result.get("end_date", "N/A"),
        "test_period_days": backtest_result.get("test_period_days", 0),
        
        # Return metrics (convert to percentages)
        "portfolio_return_pct": round(backtest_result["portfolio_return"] * 100, 2),
        "benchmark_return_pct": round(backtest_result["benchmark_return"] * 100, 2), 
        "excess_return_pct": round(backtest_result.get("excess_return", 0) * 100, 2),
        
        # Risk metrics (convert to percentages)
        "portfolio_volatility_pct": round(backtest_result.get("portfolio_volatility", 0) * 100, 2),
        "benchmark_volatility_pct": round(backtest_result.get("benchmark_volatility", 0) * 100, 2),
        
        # Risk-adjusted returns
        "portfolio_sharpe": round(backtest_result.get("portfolio_sharpe", 0), 3),
        "benchmark_sharpe": round(backtest_result.get("benchmark_sharpe", 0), 3),
        
        # Portfolio composition metrics
        "num_buy_ratings": rating_counts["BUY"],
        "num_hold_ratings": rating_counts["HOLD"], 
        "num_sell_ratings": rating_counts["SELL"],
        "portfolio_stocks": len(portfolio_tickers),
        "portfolio_composition": composition_str
    }]
    
    # Create DataFrame and save
    ensure_output_directory(output_path)
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv(output_path, index=False)
    
    abs_path = Path(output_path).resolve()
    print(f"Performance metrics saved to: {abs_path}")
    
    return str(abs_path)


def format_agent_metadata(analysis: Dict[str, Any]) -> Dict[str, str]:
    """
    Format agent analysis metadata for consistent display.
    
    Standardizes metadata formatting across different agent types
    for debugging and transparency purposes.
    
    Args:
        analysis: Individual agent analysis dictionary
        
    Returns:
        Dict[str, str]: Formatted metadata with string values
    """
    
    metadata = analysis.get("metadata", {})
    formatted = {}
    
    # Common formatting rules
    for key, value in metadata.items():
        if isinstance(value, dict):
            # Convert nested dicts to readable strings
            formatted[key] = ", ".join(f"{k}={v}" for k, v in value.items())
        elif isinstance(value, (list, tuple)):
            # Convert sequences to comma-separated strings
            formatted[key] = ", ".join(str(item) for item in value)
        else:
            # Keep simple types as strings
            formatted[key] = str(value)
    
    return formatted


def create_output_summary(portfolio_result: Dict[str, Any], 
                         backtest_result: Optional[Dict[str, Any]] = None) -> str:
    """
    Create a text summary of pipeline outputs for logging/debugging.
    
    Generates a human-readable summary of analysis results and performance
    metrics that can be logged or displayed for validation purposes.
    
    Args:
        portfolio_result: Portfolio analysis results
        backtest_result: Optional backtest results for performance data
        
    Returns:
        str: Multi-line formatted summary string
    """
    
    lines = []
    lines.append("=" * 60)
    lines.append("ALPHA AGENTS OUTPUT SUMMARY")
    lines.append("=" * 60)
    
    # Analysis metadata
    as_of_date = portfolio_result.get("as_of_date", "Unknown")
    lines.append(f"Analysis Date: {as_of_date}")
    lines.append(f"Universe: {', '.join(portfolio_result.get('tickers', []))}")
    
    # Portfolio composition
    weights = portfolio_result.get("portfolio_weights", {})
    if weights:
        lines.append(f"Portfolio: {len(weights)} stocks")
        for ticker, weight in weights.items():
            lines.append(f"  {ticker}: {weight*100:.1f}%")
    else:
        lines.append("Portfolio: Empty (Cash)")
    
    # Performance summary (if available)
    if backtest_result:
        port_ret = backtest_result.get("portfolio_return", 0) * 100
        bench_ret = backtest_result.get("benchmark_return", 0) * 100
        excess_ret = backtest_result.get("excess_return", 0) * 100
        
        lines.append(f"Portfolio Return: {port_ret:+.2f}%")
        lines.append(f"Benchmark Return: {bench_ret:+.2f}%") 
        lines.append(f"Excess Return: {excess_ret:+.2f}%")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


# Module-level constants for standardization
DEFAULT_OUTPUT_DIR = "outputs"
PICKS_FILENAME = "picks.csv"
PERFORMANCE_FILENAME = "performance.csv"

# Export key functions for easy importing
__all__ = [
    "save_picks_csv",
    "save_performance_csv", 
    "ensure_output_directory",
    "format_agent_metadata",
    "create_output_summary"
]