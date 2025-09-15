"""
Fundamental data loader for Alpha Agents system.
Loads fundamental analysis data from existing local JSON files.

This handles the fundamental data requirement:
- Load from data/fundamentals/<ticker>.json files
- All data is Q2 2024 10-Q data (filed by May-August 2024, available before as-of dates)
- No date filtering needed - data is already time-appropriate to prevent leakage
- Fields: revenue_growth_ttm, operating_margin, net_cash_position, free_cash_flow, capex_intensity
"""

import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

# Configure logger for this module
logger = logging.getLogger(__name__)

class FundamentalDataLoader:
    """
    Loads fundamental data from existing local JSON files.
    
    As required by assessment:
    - Small, local JSON factsheets with 3-6 key fundamental signals per ticker
    - Hand-curated from reputable sources (SEC filings, etc.)
    - Store locally (data/fundamentals/<ticker>.json)
    - All data is Q2 2024 10-Q data, filed by May-August 2024 (no leakage)
    - Fields: revenue_growth_ttm, operating_margin, net_cash_position, free_cash_flow, capex_intensity
    """
    
    def __init__(self):
        self.fundamentals_dir = Path('data/fundamentals')
        self.supported_tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA']
    
    def load_fundamental_data(self, ticker: str) -> Dict:
        """
        Load all fundamental data for a specific ticker from JSON file
        
        Args:
            ticker: Stock symbol (AAPL, MSFT, NVDA, TSLA)
            
        Returns:
            Dictionary of fundamental metrics with structure:
            {
                "metric_name": {
                    "value": float,
                    "direction": str,
                    "score": int,
                    "explanation": str,
                    "source_url": str,
                    "source_type": str,
                    "data_date": str,
                    "derivation": str
                }
            }
            
        Raises:
            FileNotFoundError: If fundamental file doesn't exist
            ValueError: If ticker not supported
        """
        if ticker not in self.supported_tickers:
            raise ValueError(f"Unsupported ticker: {ticker}. Supported: {self.supported_tickers}")
            
        fundamental_file = self.fundamentals_dir / f'{ticker}.json'
        
        if not fundamental_file.exists():
            raise FileNotFoundError(f"Fundamental data file not found: {fundamental_file}")
        
        try:
            with open(fundamental_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in fundamental file {fundamental_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading fundamental file {fundamental_file}: {e}")
    
    def load_all_fundamentals(self) -> Dict[str, Dict]:
        """
        Load fundamental data for all supported tickers
        
        Returns:
            Dictionary with ticker as key and fundamental data as value
            {
                "AAPL": {...fundamental_metrics...},
                "MSFT": {...fundamental_metrics...},
                ...
            }
        """
        all_data = {}
        
        for ticker in self.supported_tickers:
            try:
                all_data[ticker] = self.load_fundamental_data(ticker)
            except (FileNotFoundError, ValueError, RuntimeError) as e:
                logger.warning(f"Could not load fundamental data for {ticker}: {e}")
                continue
                
        return all_data
    
    def get_metric_value(self, ticker: str, metric_name: str) -> Optional[float]:
        """
        Get the value of a specific fundamental metric for a ticker
        
        Args:
            ticker: Stock symbol
            metric_name: Name of the metric (e.g., 'revenue_growth_ttm', 'operating_margin')
            
        Returns:
            The metric value as float, or None if not found
        """
        try:
            data = self.load_fundamental_data(ticker)
            if metric_name in data and 'value' in data[metric_name]:
                return data[metric_name]['value']
            return None
        except Exception:
            return None
    
    def get_metric_score(self, ticker: str, metric_name: str) -> Optional[int]:
        """
        Get the score of a specific fundamental metric for a ticker
        
        Args:
            ticker: Stock symbol
            metric_name: Name of the metric
            
        Returns:
            The metric score as int (1-5), or None if not found
        """
        try:
            data = self.load_fundamental_data(ticker)
            if metric_name in data and 'score' in data[metric_name]:
                return data[metric_name]['score']
            return None
        except Exception:
            return None
    
    def get_available_metrics(self, ticker: str) -> List[str]:
        """
        Get list of available fundamental metrics for a ticker
        
        Args:
            ticker: Stock symbol
            
        Returns:
            List of metric names available for the ticker
        """
        try:
            data = self.load_fundamental_data(ticker)
            return list(data.keys())
        except Exception:
            return []
    
    def get_fundamentals_summary(self, ticker: str) -> Dict[str, float]:
        """
        Get a summary of all fundamental metrics (values only) for a ticker
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary mapping metric names to their values
        """
        try:
            data = self.load_fundamental_data(ticker)
            summary = {}
            for metric_name, metric_data in data.items():
                if 'value' in metric_data:
                    summary[metric_name] = metric_data['value']
            return summary
        except Exception:
            return {}
    
    def validate_fundamental_data(self) -> tuple[bool, List[str]]:
        """
        Validate that fundamental data files exist and are readable
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if fundamentals directory exists
        if not self.fundamentals_dir.exists():
            issues.append(f"Fundamentals directory not found: {self.fundamentals_dir}")
            return False, issues
        
        # Check each ticker file
        for ticker in self.supported_tickers:
            file_path = self.fundamentals_dir / f"{ticker}.json"
            if not file_path.exists():
                issues.append(f"Missing fundamental data file: {file_path}")
            else:
                try:
                    # Try to load the file to check if it's valid JSON
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if not data:
                        issues.append(f"Empty fundamental data file: {file_path}")
                except json.JSONDecodeError:
                    issues.append(f"Invalid JSON in fundamental data file: {file_path}")
                except Exception as e:
                    issues.append(f"Error reading fundamental data file {file_path}: {e}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_fundamental_score(self, ticker: str) -> float:
        """
        Calculate an overall fundamental score for a ticker (1-5 scale)
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Overall score from 1.0 to 5.0 based on fundamental metrics
        """
        try:
            data = self.load_fundamental_data(ticker)
            if not data:
                return 3.0  # Neutral score if no data
            
            scores = []
            for metric_name, metric_data in data.items():
                if 'score' in metric_data and isinstance(metric_data['score'], (int, float)):
                    scores.append(float(metric_data['score']))
            
            if not scores:
                return 3.0  # Neutral score if no scores found
            
            # Average the scores
            avg_score = sum(scores) / len(scores)
            # Clamp to 1-5 range
            return max(1.0, min(5.0, avg_score))
            
        except Exception:
            return 3.0  # Neutral score on error

def main():
    """
    Simple usage demonstration of FundamentalDataLoader
    """
    # Configure logging for demo
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("=== Fundamental Data Loader Demo ===\n")
    
    # Initialize loader
    loader = FundamentalDataLoader()
    
    # Basic usage example
    print("Loading fundamental data for AAPL:")
    aapl_data = loader.load_fundamental_data('AAPL')
    print(f"Available metrics: {list(aapl_data.keys())}")
    
    # Show key metrics comparison
    print(f"\nKey metrics comparison:")
    print(f"{'Ticker':<8} {'Rev Growth':<12} {'Op Margin':<12} {'FCF':<12}")
    print("-" * 50)
    
    for ticker in ['AAPL', 'MSFT', 'NVDA', 'TSLA']:
        rev_growth = loader.get_metric_value(ticker, 'revenue_growth_ytd') or 0
        op_margin = loader.get_metric_value(ticker, 'operating_margin') or 0
        fcf = loader.get_metric_value(ticker, 'free_cash_flow') or 0
        print(f"{ticker:<8} {rev_growth:<12.1f} {op_margin:<12.1f} {fcf:<12.1f}")
    
    print(f"\nDemo complete. Use FundamentalDataLoader().load_fundamental_data(ticker) for access.")


if __name__ == '__main__':
    main()
