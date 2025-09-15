#!/usr/bin/env python3
"""
Lightweight tests for backtest math sanity checks.

Tests core mathematical calculations without requiring real market data:
- Trading days conversion logic
- Return calculations
- Portfolio vs benchmark math
- Risk metrics computation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging
import io
import contextlib

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.backtest import BacktestEngine


class MockPriceLoader:
    """Mock price loader for testing backtest calculations."""
    
    def get_price_data(self, tickers, start_date, end_date):
        """Generate synthetic price data for testing."""
        # Create date range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        dates = pd.date_range(start_dt, end_dt, freq='D')
        
        all_data = []
        
        for ticker in tickers:
            # Generate synthetic prices with different patterns
            if ticker == "WINNER":
                # Steadily increasing stock
                prices = np.linspace(100, 120, len(dates))
            elif ticker == "LOSER":
                # Steadily decreasing stock
                prices = np.linspace(100, 80, len(dates))
            elif ticker == "FLAT":
                # Flat stock with small fluctuations
                prices = np.full(len(dates), 100) + np.random.normal(0, 0.5, len(dates))
            else:
                # Default: slight upward trend
                prices = np.linspace(100, 110, len(dates)) + np.random.normal(0, 1, len(dates))
            
            ticker_data = pd.DataFrame({
                'date': dates.date,
                'ticker': ticker,
                'open': prices,
                'high': prices * 1.02,
                'low': prices * 0.98,
                'close': prices,
                'volume': 1000000
            })
            
            all_data.append(ticker_data)
        
        return pd.concat(all_data, ignore_index=True)


@pytest.fixture
def backtest_engine():
    """Fixture to create a BacktestEngine with MockPriceLoader."""
    # Suppress all output during tests for cleaner results
    logging.getLogger().setLevel(logging.CRITICAL)
    
    mock_loader = MockPriceLoader()
    engine = BacktestEngine(mock_loader, risk_free_rate=0.05)
    
    return engine


def run_quiet_backtest(engine, portfolio_result, forward_days):
    """Helper function to run backtest with suppressed output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return engine.run_backtest(portfolio_result, forward_days)


class TestBacktestMath:
    """Test backtest mathematical calculations."""
    
    def test_trading_days_conversion(self, backtest_engine):
        """Test trading days to calendar days conversion."""
        start_date = datetime(2024, 8, 20)  # Tuesday
        
        # Test basic conversion: 5 trading days should be ~7 calendar days
        result_date = backtest_engine._add_trading_days(start_date, 5)
        delta = (result_date - start_date).days
        assert abs(delta - 7) <= 2  # Allow some tolerance
        
        # Test 21 trading days (1 month) should be ~30 calendar days
        result_date = backtest_engine._add_trading_days(start_date, 21)
        delta = (result_date - start_date).days
        assert abs(delta - 30) <= 5
        
        # Test 63 trading days (3 months) should be ~90 calendar days
        result_date = backtest_engine._add_trading_days(start_date, 63)
        delta = (result_date - start_date).days
        assert abs(delta - 90) <= 10
        
        # Test zero trading days
        result_date = backtest_engine._add_trading_days(start_date, 0)
        assert result_date == start_date
    
    def test_return_calculation_simple(self, backtest_engine):
        """Test basic return calculation logic."""
        # Create simple test portfolio result
        portfolio_result = {
            "as_of_date": "2024-08-20",
            "portfolio_weights": {
                "WINNER": 0.5,  # Stock that goes up 20%
                "FLAT": 0.5     # Stock that stays flat
            },
            "ticker_analyses": {
                "WINNER": {"consensus_rating": "BUY"},
                "FLAT": {"consensus_rating": "HOLD"}
            }
        }
        
        # Override engine's all_tickers for this test
        backtest_engine.all_tickers = ["WINNER", "FLAT"]
        
        # Run backtest with short window
        result = run_quiet_backtest(backtest_engine, portfolio_result, forward_days=10)
        
        # Verify result structure
        assert 'portfolio_return' in result
        assert 'benchmark_return' in result
        assert 'excess_return' in result
        
        # Portfolio and benchmark should have same weights in this case (50% WINNER, 50% FLAT)
        # So returns should be equal or very close
        assert abs(result['portfolio_return'] - result['benchmark_return']) < 1e-6
        assert abs(result['excess_return'] - 0.0) < 1e-6
        
        # Returns should be reasonable (not crazy large/small)
        assert result['portfolio_return'] > -0.5  # Not less than -50%
        assert result['portfolio_return'] < 0.5   # Not more than +50%
    
    def test_empty_portfolio_behavior(self, backtest_engine):
        """Test backtest behavior with empty portfolio (100% cash)."""
        portfolio_result = {
            "as_of_date": "2024-08-20",
            "portfolio_weights": {},  # Empty portfolio
            "ticker_analyses": {}
        }
        
        backtest_engine.all_tickers = ["WINNER", "LOSER"]
        
        result = run_quiet_backtest(backtest_engine, portfolio_result, forward_days=10)
        
        # Empty portfolio should have zero return
        assert result['portfolio_return'] == 0.0
        
        # Benchmark return depends on the synthetic data - could be zero if flat
        # The important thing is that excess return reflects the difference
        assert isinstance(result['benchmark_return'], (int, float))
        
        # Excess return should equal benchmark_return - portfolio_return
        expected_excess = result['benchmark_return'] - result['portfolio_return']
        assert abs(result['excess_return'] - expected_excess) < 1e-6
    
    def test_single_stock_portfolio(self, backtest_engine):
        """Test portfolio with 100% allocation to single stock."""
        portfolio_result = {
            "as_of_date": "2024-08-20",
            "portfolio_weights": {
                "WINNER": 1.0  # 100% allocation
            },
            "ticker_analyses": {
                "WINNER": {"consensus_rating": "BUY"}
            }
        }
        
        backtest_engine.all_tickers = ["WINNER", "LOSER"]
        
        result = run_quiet_backtest(backtest_engine, portfolio_result, forward_days=10)
        
        # Portfolio return should equal WINNER's return
        # Benchmark return should be average of WINNER and LOSER
        assert result['portfolio_return'] > result['benchmark_return']
    
    def test_risk_metrics_sanity(self, backtest_engine):
        """Test that risk metrics are reasonable."""
        portfolio_result = {
            "as_of_date": "2024-08-20",
            "portfolio_weights": {
                "WINNER": 0.6,
                "FLAT": 0.4
            },
            "ticker_analyses": {
                "WINNER": {"consensus_rating": "BUY"},
                "FLAT": {"consensus_rating": "HOLD"}
            }
        }
        
        backtest_engine.all_tickers = ["WINNER", "FLAT"]
        
        result = run_quiet_backtest(backtest_engine, portfolio_result, forward_days=20)
        
        # Volatility should be non-negative
        assert result['portfolio_volatility'] >= 0
        assert result['benchmark_volatility'] >= 0
        
        # Volatility should be reasonable (not crazy high)
        assert result['portfolio_volatility'] < 2.0  # Less than 200% annualized
        assert result['benchmark_volatility'] < 2.0
        
        # Sharpe ratios should be finite
        assert not np.isnan(result['portfolio_sharpe'])
        assert not np.isnan(result['benchmark_sharpe'])
        assert not np.isinf(result['portfolio_sharpe'])
        assert not np.isinf(result['benchmark_sharpe'])
    
    def test_weight_normalization(self, backtest_engine):
        """Test that portfolio weights are handled correctly."""
        # Test with weights that don't sum to 1.0
        portfolio_result = {
            "as_of_date": "2024-08-20",
            "portfolio_weights": {
                "WINNER": 0.3,  # Only 60% allocated, 40% should be cash
                "FLAT": 0.3
            },
            "ticker_analyses": {
                "WINNER": {"consensus_rating": "BUY"},
                "FLAT": {"consensus_rating": "HOLD"}
            }
        }
        
        backtest_engine.all_tickers = ["WINNER", "FLAT"]
        
        # Should not raise an error
        result = run_quiet_backtest(backtest_engine, portfolio_result, forward_days=10)
        
        # Result should be valid
        assert isinstance(result['portfolio_return'], (int, float))
        assert isinstance(result['benchmark_return'], (int, float))
    
    def test_date_boundary_conditions(self, backtest_engine):
        """Test edge cases with date handling."""
        portfolio_result = {
            "as_of_date": "2024-08-20",
            "portfolio_weights": {"WINNER": 1.0},
            "ticker_analyses": {"WINNER": {"consensus_rating": "BUY"}}
        }
        
        backtest_engine.all_tickers = ["WINNER"]
        
        # Test very short window (1 day)
        result = run_quiet_backtest(backtest_engine, portfolio_result, forward_days=1)
        assert 'portfolio_return' in result
        
        # Test longer window
        result = run_quiet_backtest(backtest_engine, portfolio_result, forward_days=100)
        assert 'portfolio_return' in result
    
    def test_error_result_structure(self, backtest_engine):
        """Test error result structure when backtest fails."""
        error_result = backtest_engine._create_error_result("2024-08-20", "Test error")
        
        # Should have all required fields
        required_fields = [
            'as_of_date', 'portfolio_return', 'benchmark_return', 'excess_return',
            'portfolio_sharpe', 'benchmark_sharpe', 'portfolio_volatility',
            'benchmark_volatility', 'error', 'test_period_days'
        ]
        
        for field in required_fields:
            assert field in error_result
        
        # Numeric fields should be zero
        assert error_result['portfolio_return'] == 0.0
        assert error_result['benchmark_return'] == 0.0
        assert error_result['excess_return'] == 0.0
        
        # Error message should be preserved
        assert error_result['error'] == "Test error"


# Pytest will automatically discover and run tests
# To run: pytest tests/test_backtest_math.py -v
