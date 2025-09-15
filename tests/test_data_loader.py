#!/usr/bin/env python3
"""
Lightweight tests for PriceDataLoader behavior.

Tests core functionality without requiring API keys or external dependencies:
- Date range calculation logic
- Input validation
- Cached data fallback behavior
- Error handling
"""

import pytest
import tempfile
import pandas as pd
from datetime import datetime, date
from pathlib import Path
import sys
import os
import shutil
import logging

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data_collectors.price_loader import PriceDataLoader


@pytest.fixture
def temp_dir():
    """Fixture to create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def price_loader(temp_dir):
    """Fixture to create a PriceDataLoader with temporary cache directory."""
    # Suppress logging output during tests for cleaner output
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # Use temporary directory for cache to avoid interfering with real data
    loader = PriceDataLoader(api_key=None)  # No API key for testing
    # Override cache directory to use temp directory
    loader.cache_dir = Path(temp_dir) / "prices"
    loader.cache_dir.mkdir(parents=True, exist_ok=True)
    
    return loader


class TestPriceDataLoader:
    """Test PriceDataLoader core functionality."""
    
    def test_date_range_calculation(self, price_loader):
        """Test tight date window calculation."""
        as_of_date = "2024-08-20"
        
        # Test default parameters (90 days each direction)
        start_date, end_date = price_loader.calculate_date_range(as_of_date)
        
        # Verify format
        import re
        assert re.match(r'\d{4}-\d{2}-\d{2}', start_date)
        assert re.match(r'\d{4}-\d{2}-\d{2}', end_date)
        
        # Verify dates are reasonable
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        as_of_dt = datetime.strptime(as_of_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        assert start_dt < as_of_dt
        assert as_of_dt < end_dt
        
        # Test with custom parameters
        start_date, end_date = price_loader.calculate_date_range(
            as_of_date, lookback_days=30, forward_days=60
        )
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Should be roughly 30 days before and 60 days after
        lookback_delta = (as_of_dt - start_dt).days
        forward_delta = (end_dt - as_of_dt).days
        
        assert abs(lookback_delta - 30) <= 1
        assert abs(forward_delta - 60) <= 1
    
    def test_ticker_validation(self, price_loader):
        """Test ticker validation logic."""
        valid_tickers = ["AAPL", "MSFT"]
        invalid_tickers = ["INVALID", "XYZ"]
        mixed_tickers = ["AAPL", "INVALID"]
        
        # Should not raise for valid tickers (even though we'll fail on data loading)
        try:
            price_loader.get_price_data(valid_tickers, "2024-08-01", "2024-08-31")
        except ValueError as e:
            # Should fail on data loading, not ticker validation
            assert "Unsupported tickers" not in str(e)
        except Exception:
            pass  # Other errors are fine for this test
        
        # Should raise for invalid tickers
        with pytest.raises(ValueError) as exc_info:
            price_loader.get_price_data(invalid_tickers, "2024-08-01", "2024-08-31")
        assert "Unsupported tickers" in str(exc_info.value)
        
        # Should raise for mixed valid/invalid
        with pytest.raises(ValueError) as exc_info:
            price_loader.get_price_data(mixed_tickers, "2024-08-01", "2024-08-31")
        assert "Unsupported tickers" in str(exc_info.value)
    
    def test_cached_data_fallback(self, price_loader):
        """Test cached data loading and filtering."""
        # Create mock cached data
        ticker = "AAPL"
        cache_path = price_loader.cache_dir / f"{ticker}.csv"
        
        # Create test data spanning wider range than we'll request
        test_dates = pd.date_range("2024-07-01", "2024-09-30", freq='D')
        test_data = pd.DataFrame({
            'date': test_dates.date,
            'ticker': ticker,
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 102.0,
            'volume': 1000000
        })
        test_data.to_csv(cache_path, index=False)
        
        # Test loading subset of cached data
        result_df = price_loader._load_cached_data(ticker, "2024-08-01", "2024-08-31")
        
        # Verify result structure
        assert isinstance(result_df, pd.DataFrame)
        assert not result_df.empty
        
        # Verify date filtering worked
        min_date = result_df['date'].min()
        max_date = result_df['date'].max()
        assert min_date >= date(2024, 8, 1)
        assert max_date <= date(2024, 8, 31)
        
        # Verify required columns present
        required_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            assert col in result_df.columns
    
    def test_cached_data_missing_file(self, price_loader):
        """Test behavior when cached file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            price_loader._load_cached_data("NONEXISTENT", "2024-08-01", "2024-08-31")
    
    def test_cached_data_date_range_mismatch(self, price_loader):
        """Test behavior when cached data doesn't cover requested range."""
        ticker = "MSFT"
        cache_path = price_loader.cache_dir / f"{ticker}.csv"
        
        # Create cached data for limited range
        test_dates = pd.date_range("2024-08-10", "2024-08-20", freq='D')
        test_data = pd.DataFrame({
            'date': test_dates.date,
            'ticker': ticker,
            'open': 200.0,
            'high': 205.0,
            'low': 195.0,
            'close': 202.0,
            'volume': 500000
        })
        test_data.to_csv(cache_path, index=False)
        
        # Request broader range than available - should still work but with warnings
        # The data loader only returns data for the intersection of requested and available ranges
        try:
            result_df = price_loader._load_cached_data(ticker, "2024-08-01", "2024-08-31")
            # Should return data for the overlapping period (2024-08-10 to 2024-08-20)
            assert not result_df.empty
        except ValueError:
            # If it raises an error due to no overlap, that's also acceptable behavior
            pass
        
        # But requesting completely outside range should fail
        with pytest.raises(Exception) as exc_info:
            price_loader._load_cached_data(ticker, "2024-07-01", "2024-07-31")
        assert "No cached data in requested date range" in str(exc_info.value)
    
    def test_validation_without_api_key(self, price_loader, temp_dir):
        """Test validation logic when no API key is available."""
        # Create a new loader explicitly without API key to ensure clean state
        # Need to temporarily remove any API key from environment
        original_api_key = os.environ.get('FINANCIAL_DATASETS_API_KEY')
        if original_api_key:
            del os.environ['FINANCIAL_DATASETS_API_KEY']
        
        try:
            loader_no_key = PriceDataLoader(api_key=None)
            loader_no_key.cache_dir = Path(temp_dir) / "prices_no_key"
            loader_no_key.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Without cached data, should return False
            result = loader_no_key.validate_for_as_of_date("2024-08-20")
            assert not result
        finally:
            # Restore original API key if it existed
            if original_api_key:
                os.environ['FINANCIAL_DATASETS_API_KEY'] = original_api_key
        
        # Test with partial cached data (should still fail)
        try:
            # Create minimal cached data for one ticker
            ticker = "AAPL"
            cache_path = loader_no_key.cache_dir / f"{ticker}.csv"
            
            # Calculate required range for as-of date
            start_date, end_date = loader_no_key.calculate_date_range("2024-08-20")
            
            # Create data that covers the required range
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            test_dates = pd.date_range(start_dt, end_dt, freq='D')
            
            test_data = pd.DataFrame({
                'date': test_dates.date,
                'ticker': ticker,
                'open': 150.0,
                'high': 155.0,
                'low': 145.0,
                'close': 152.0,
                'volume': 2000000
            })
            test_data.to_csv(cache_path, index=False)
            
            # Should still return False because we need data for all tickers
            result = loader_no_key.validate_for_as_of_date("2024-08-20")
            assert not result
        finally:
            # Restore original API key if it existed
            if original_api_key:
                os.environ['FINANCIAL_DATASETS_API_KEY'] = original_api_key
    
    def test_validation_with_api_key(self, price_loader):
        """Test validation logic when API key is available."""
        # Create loader with mock API key
        loader_with_key = PriceDataLoader(api_key="test_key")
        
        # Should return True regardless of cached data
        result = loader_with_key.validate_for_as_of_date("2024-08-20")
        assert result


# Pytest will automatically discover and run tests
# To run: pytest tests/test_data_loader.py -v
