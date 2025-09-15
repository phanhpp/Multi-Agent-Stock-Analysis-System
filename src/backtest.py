"""
Backtesting engine for Alpha Agents portfolio system.

This module provides pure performance calculation functionality for comparing
agent-selected portfolios against equal-weight benchmarks. It focuses solely
on financial metrics computation without handling output generation or
visualization, maintaining clean separation of concerns.

Core Functionality:
    - Portfolio vs benchmark return calculation
    - Risk metrics (volatility, Sharpe ratios) 
    - Temporal controls to prevent data leakage
    - Graceful handling of empty portfolios and missing data

Usage:
    from src.backtest import BacktestEngine
    from src.data_collectors.price_loader import PriceDataLoader
    
    engine = BacktestEngine(price_loader)
    results = engine.run_backtest(portfolio_result, forward_days=63)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import os
from pathlib import Path

from src.data_collectors.price_loader import PriceDataLoader


class BacktestEngine:
    """
    Core backtesting engine for portfolio performance evaluation.
    
    Handles all performance calculations including returns, risk metrics,
    and benchmark comparisons with proper temporal controls to prevent
    data leakage. Designed to work with any portfolio construction approach.
    """
    
    def __init__(self, price_loader: PriceDataLoader, risk_free_rate: float = 0.05):
        """
        Initialize backtest engine with data source and parameters.
        
        Args:
            price_loader: Initialized price data loader instance
            risk_free_rate: Annual risk-free rate for Sharpe calculation (default: 5%)
        """
        self.price_loader = price_loader
        self.risk_free_rate = risk_free_rate
        self.all_tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]
    
    def run_backtest(self, portfolio_result: Dict[str, Any], forward_days: int = 63) -> Dict[str, Any]:
        """
        Execute complete backtest comparing portfolio vs benchmark performance.
        
        Loads forward price data, calculates portfolio and benchmark returns,
        computes risk metrics, and returns comprehensive results dictionary
        for downstream processing.
        
        Args:
            portfolio_result: Portfolio analysis results containing weights and metadata
            forward_days: Forward testing window in trading days (~3 months = 63)
            
        Returns:
            Dict[str, Any]: Comprehensive backtest results including:
                - portfolio_return: Total portfolio return (decimal)
                - benchmark_return: Equal-weight benchmark return (decimal)
                - excess_return: Portfolio outperformance (decimal)
                - portfolio_volatility: Annualized portfolio volatility (decimal)
                - benchmark_volatility: Annualized benchmark volatility (decimal)
                - portfolio_sharpe: Risk-adjusted return ratio
                - benchmark_sharpe: Risk-adjusted return ratio
                - price_data: Raw price data for visualization
                - test_period_days: Actual trading days tested
                - as_of_date: Analysis date
                - end_date: Final backtest date
        """
        as_of_date = portfolio_result["as_of_date"]
        portfolio_weights = portfolio_result.get("portfolio_weights", {})
        
        print(f"\n{'='*60}")
        print("Running Performance Backtest")
        print('='*60)
        print(f"Decision Date: {as_of_date}")
        print(f"Forward Window: {forward_days} trading days")
        
        # Calculate forward date range
        start_date = datetime.strptime(as_of_date, "%Y-%m-%d")
        end_date = self._add_trading_days(start_date, forward_days)
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        # Load forward price data with leakage controls
        try:
            price_data = self.price_loader.get_price_data(
                self.all_tickers, 
                as_of_date, 
                end_date_str
            )
        except Exception as e:
            print(f"Error loading price data: {e}")
            return self._create_error_result(as_of_date, str(e))
        
        # Validate sufficient data
        if len(price_data) < 5:
            print(f"Insufficient price data ({len(price_data)} days). Skipping backtest.")
            return self._create_error_result(as_of_date, "Insufficient price data")
        
        actual_days = len(price_data)
        print(f"Loaded {actual_days} trading days of price data")
        
        # Calculate performance metrics directly from long format
        portfolio_metrics = self._calculate_portfolio_performance(price_data, portfolio_weights)
        benchmark_metrics = self._calculate_benchmark_performance(price_data)
        
        # Compile comprehensive results
        backtest_result = {
            # Metadata
            "as_of_date": as_of_date,
            "end_date": end_date_str,
            "test_period_days": actual_days,
            "forward_days": forward_days,  # Original forward period requested
            
            # Portfolio metrics
            "portfolio_return": portfolio_metrics["portfolio_return"],
            "portfolio_volatility": portfolio_metrics["portfolio_volatility"], 
            "portfolio_sharpe": portfolio_metrics["portfolio_sharpe"],
            "portfolio_composition": portfolio_metrics["portfolio_composition"],
            
            # Benchmark metrics
            "benchmark_return": benchmark_metrics["benchmark_return"],
            "benchmark_volatility": benchmark_metrics["benchmark_volatility"],
            "benchmark_sharpe": benchmark_metrics["benchmark_sharpe"],
            "benchmark_composition": "Equal-weight AAPL, MSFT, NVDA, TSLA (25% each)",
            
            # Performance comparison
            "excess_return": portfolio_metrics["portfolio_return"] - benchmark_metrics["benchmark_return"],
            
            # Raw data for downstream processing
            "price_data": price_data,
            "portfolio_weights": portfolio_weights,
        }
        
        self._print_results_summary(backtest_result)
        return backtest_result
    
    
    def _calculate_portfolio_performance(self, price_data: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate portfolio performance metrics from long format price data."""
        if not weights:  # Empty portfolio = cash position
            return {
                "portfolio_return": 0.0,
                "portfolio_volatility": 0.0,
                "portfolio_sharpe": 0.0,
                "portfolio_composition": "Cash (0 stocks)"
            }
        
        # Calculate total return for each stock
        stock_returns = {}
        daily_portfolio_returns = []
        
        # Get unique dates for daily return calculation
        dates = sorted(price_data['date'].unique())
        
        for ticker, weight in weights.items():
            # Get price data for this ticker
            ticker_data = price_data[price_data['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date')
            
            if len(ticker_data) < 2:
                continue  # Skip if insufficient data
                
            # Calculate total return for this stock
            start_price = ticker_data['close'].iloc[0]
            end_price = ticker_data['close'].iloc[-1]
            stock_return = (end_price / start_price) - 1
            stock_returns[ticker] = stock_return
        
        # Calculate portfolio total return
        total_return = sum(weight * stock_returns.get(ticker, 0) for ticker, weight in weights.items())
        
        # Calculate daily portfolio returns for volatility
        for i in range(1, len(dates)):
            daily_portfolio_return = 0.0
            for ticker, weight in weights.items():
                ticker_today = price_data[(price_data['ticker'] == ticker) & (price_data['date'] == dates[i])]
                ticker_yesterday = price_data[(price_data['ticker'] == ticker) & (price_data['date'] == dates[i-1])]
                
                if len(ticker_today) > 0 and len(ticker_yesterday) > 0:
                    daily_return = (ticker_today['close'].iloc[0] / ticker_yesterday['close'].iloc[0]) - 1
                    daily_portfolio_return += weight * daily_return
            
            daily_portfolio_returns.append(daily_portfolio_return)
        
        daily_returns = pd.Series(daily_portfolio_returns)
        
        # Risk metrics calculation
        daily_vol = daily_returns.std()
        annualized_vol = daily_vol * np.sqrt(252) if daily_vol > 0 else 0
        
        # Sharpe ratio calculation
        annualized_return = total_return * (252 / len(price_data))
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / annualized_vol if annualized_vol > 0 else 0
        
        # Portfolio composition description
        composition = ", ".join([f"{ticker}: {weight*100:.1f}%" 
                                for ticker, weight in weights.items()])
        
        return {
            "portfolio_return": total_return,
            "portfolio_volatility": annualized_vol,
            "portfolio_sharpe": sharpe_ratio,
            "portfolio_composition": composition
        }
    
    def _calculate_benchmark_performance(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate equal-weight benchmark performance from long format price data."""
        equal_weight = 1.0 / len(self.all_tickers)
        
        # Calculate total return for each stock
        stock_returns = {}
        daily_benchmark_returns = []
        
        # Get unique dates for daily return calculation
        dates = sorted(price_data['date'].unique())
        
        for ticker in self.all_tickers:
            # Get price data for this ticker
            ticker_data = price_data[price_data['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date')
            
            if len(ticker_data) < 2:
                continue  # Skip if insufficient data
                
            # Calculate total return for this stock
            start_price = ticker_data['close'].iloc[0]
            end_price = ticker_data['close'].iloc[-1]
            stock_return = (end_price / start_price) - 1
            stock_returns[ticker] = stock_return
        
        # Calculate benchmark total return (equal weight)
        total_return = sum(equal_weight * stock_returns.get(ticker, 0) for ticker in self.all_tickers)
        
        # Calculate daily benchmark returns for volatility
        for i in range(1, len(dates)):
            daily_benchmark_return = 0.0
            for ticker in self.all_tickers:
                ticker_today = price_data[(price_data['ticker'] == ticker) & (price_data['date'] == dates[i])]
                ticker_yesterday = price_data[(price_data['ticker'] == ticker) & (price_data['date'] == dates[i-1])]
                
                if len(ticker_today) > 0 and len(ticker_yesterday) > 0:
                    daily_return = (ticker_today['close'].iloc[0] / ticker_yesterday['close'].iloc[0]) - 1
                    daily_benchmark_return += equal_weight * daily_return
            
            daily_benchmark_returns.append(daily_benchmark_return)
        
        daily_returns = pd.Series(daily_benchmark_returns)
        
        # Risk metrics calculation
        daily_vol = daily_returns.std()
        annualized_vol = daily_vol * np.sqrt(252) if daily_vol > 0 else 0
        
        # Sharpe ratio calculation
        annualized_return = total_return * (252 / len(price_data))
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / annualized_vol if annualized_vol > 0 else 0
        
        return {
            "benchmark_return": total_return,
            "benchmark_volatility": annualized_vol,
            "benchmark_sharpe": sharpe_ratio
        }
    
    def _add_trading_days(self, start_date: datetime, trading_days: int) -> datetime:
        """
        Approximate addition of trading days to start date.
        
        Uses simple heuristic: 5 trading days per 7 calendar days.
        More sophisticated date handling could account for actual
        holidays and market closures.
        """
        calendar_days = int(trading_days * 7 / 5)
        return start_date + timedelta(days=calendar_days)
    
    def _print_results_summary(self, result: Dict[str, Any]) -> None:
        """Print backtest results summary."""
        print(f"\nResults Summary:")
        print(f"Portfolio Return: {result['portfolio_return']*100:+.2f}%")
        print(f"Benchmark Return: {result['benchmark_return']*100:+.2f}%")
        print(f"Excess Return: {result['excess_return']*100:+.2f}%")
        print(f"Portfolio Sharpe: {result['portfolio_sharpe']:.3f}")
        print(f"Benchmark Sharpe: {result['benchmark_sharpe']:.3f}")
        print('='*60)
    
    def _create_error_result(self, as_of_date: str, error_msg: str) -> Dict[str, Any]:
        """Create error result for failed backtests."""
        return {
            "as_of_date": as_of_date,
            "portfolio_return": 0.0,
            "benchmark_return": 0.0,
            "excess_return": 0.0,
            "portfolio_sharpe": 0.0,
            "benchmark_sharpe": 0.0,
            "portfolio_volatility": 0.0,
            "benchmark_volatility": 0.0,
            "error": error_msg,
            "test_period_days": 0
        }


def generate_performance_chart(backtest_result: Dict[str, Any], output_path: str = "outputs/portfolio_chart.png") -> None:
    """
    Generate portfolio vs benchmark growth chart from backtest results.
    
    Creates a professional-quality visualization showing cumulative performance
    of agent portfolio vs equal-weight benchmark over the testing period.
    
    Args:
        backtest_result: Complete backtest results dictionary
        output_path: Output file path for chart (default: outputs/portfolio_chart.png)
    """
    if "price_data" not in backtest_result or backtest_result.get("error"):
        print("Insufficient data for chart generation")
        return
    
    price_data = backtest_result["price_data"]
    portfolio_weights = backtest_result["portfolio_weights"]
    
    # Get unique dates
    dates = sorted(price_data['date'].unique())
    
    # Calculate cumulative performance series
    portfolio_values = [1.0]  # Start with $1
    benchmark_values = [1.0]
    
    # Calculate daily returns for portfolio and benchmark
    for i in range(1, len(dates)):
        # Portfolio daily return
        portfolio_daily_return = 0.0
        if portfolio_weights:
            for ticker, weight in portfolio_weights.items():
                ticker_today = price_data[(price_data['ticker'] == ticker) & (price_data['date'] == dates[i])]
                ticker_yesterday = price_data[(price_data['ticker'] == ticker) & (price_data['date'] == dates[i-1])]
                
                if len(ticker_today) > 0 and len(ticker_yesterday) > 0:
                    daily_return = (ticker_today['close'].iloc[0] / ticker_yesterday['close'].iloc[0]) - 1
                    portfolio_daily_return += weight * daily_return
        
        portfolio_values.append(portfolio_values[-1] * (1 + portfolio_daily_return))
        
        # Benchmark daily return (equal weight)
        benchmark_daily_return = 0.0
        equal_weight = 0.25
        for ticker in ["AAPL", "MSFT", "NVDA", "TSLA"]:
            ticker_today = price_data[(price_data['ticker'] == ticker) & (price_data['date'] == dates[i])]
            ticker_yesterday = price_data[(price_data['ticker'] == ticker) & (price_data['date'] == dates[i-1])]
            
            if len(ticker_today) > 0 and len(ticker_yesterday) > 0:
                daily_return = (ticker_today['close'].iloc[0] / ticker_yesterday['close'].iloc[0]) - 1
                benchmark_daily_return += equal_weight * daily_return
        
        benchmark_values.append(benchmark_values[-1] * (1 + benchmark_daily_return))
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    plt.plot(dates, portfolio_values, label='Agent Portfolio', linewidth=2, color='blue')
    plt.plot(dates, benchmark_values, label='Equal-Weight Benchmark', 
             linewidth=2, color='gray', linestyle='--')
    
    # Calculate forward period info
    forward_days = backtest_result.get("forward_days", "N/A")
    forward_months = round(forward_days / 21, 1) if forward_days != "N/A" else "N/A"
    
    plt.title(f'Portfolio Performance Comparison\n'
              f'{backtest_result["as_of_date"]} to {backtest_result["end_date"]} '
              f'({forward_days} trading days, ~{forward_months} months)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Growth of $1', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save chart
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance chart saved to: {output_path}")


def run_complete_backtest(portfolio_result: Dict[str, Any], 
                         price_loader: PriceDataLoader,
                         forward_days: int = 63) -> Dict[str, Any]:
    """
    Execute complete backtesting workflow including chart generation.
    
    Convenience function that runs backtest and generates visualization
    in a single call. This is the main integration point for the pipeline.
    
    Args:
        portfolio_result: Portfolio analysis results from agent workflow
        price_loader: Initialized price data loader instance
        forward_days: Forward testing window in trading days
        
    Returns:
        Dict[str, Any]: Complete backtest results with all metrics
    """
    # Run core backtest
    backtest_engine = BacktestEngine(price_loader)
    backtest_result = backtest_engine.run_backtest(portfolio_result, forward_days)
    
    # Generate visualization
    generate_performance_chart(backtest_result)
    
    return backtest_result


if __name__ == "__main__":
    print("Backtest module loaded. Use BacktestEngine class or run_complete_backtest() function.")