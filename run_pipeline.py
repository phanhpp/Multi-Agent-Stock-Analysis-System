#!/usr/bin/env python3
"""
Alpha Agents Multi-Agent Stock Analysis Pipeline

This is the main entry point for the Alpha Agents system - a multi-agent framework
for systematic stock analysis and portfolio construction. The pipeline orchestrates
three specialized AI agents (Valuation, Sentiment, Fundamental) plus a Coordinator
to analyze stocks and evaluate performance through backtesting.

Architecture Overview:
    1. Agent Analysis: Parallel execution of specialized agents on target universe
    2. Coordination: Weighted consensus mechanism for final BUY/HOLD/SELL decisions  
    3. Portfolio Construction: Differential weighting based on conviction levels
    4. Backtesting: Forward-looking performance evaluation vs equal-weight benchmark
    5. Output Generation: CSV reports and performance visualization

Key Features:
    - No data leakage: Strict temporal controls prevent future information usage
    - Configurable parameters: All agent thresholds externally configurable
    - Modular design: Independent agent testing and incremental development
    - Comprehensive logging: Full audit trail of decisions and reasoning
    - Error resilience: Graceful handling of missing data and API failures

Usage Examples:
    python run_pipeline.py                           # Default analysis (2024-08-20)
    python run_pipeline.py --date 2024-09-15        # Custom analysis date
    python run_pipeline.py --forward-days 45        # Shorter backtest window
    python run_pipeline.py --no-backtest            # Skip performance evaluation
    python run_pipeline.py --config config/test.yaml # Alternative configuration

Output Files:
    - outputs/picks.csv: Individual agent ratings and consensus decisions
    - outputs/performance.csv: Portfolio vs benchmark performance metrics
    - outputs/portfolio_chart.png: Growth of $1 visualization

Dependencies:
    - LangGraph: Multi-agent workflow orchestration
    - VADER: Sentiment analysis for news processing
    - financialdatasets.ai: Price data API (with CSV fallback)
    - PyYAML: Configuration management
    - pandas/matplotlib: Data processing and visualization

Assignment Context:
    This implementation adapts concepts from BlackRock's "AlphaAgents" research
    into a practical, auditable prototype focusing on system design and 
    reproducible workflows rather than alpha generation.
"""

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Core pipeline imports
from src.workflow.portfolio_workflow import run_portfolio_workflow
from src.backtest import run_complete_backtest
from src.data_collectors.price_loader import PriceDataLoader
from src.config import config
from src.utils.output_utils import (
    save_picks_csv, 
    save_performance_csv,
    create_output_summary,
    ensure_output_directory
)


def validate_date(date_string: str) -> str:
    """Validate date string in YYYY-MM-DD format."""
    try:
        parsed_date = datetime.strptime(date_string, "%Y-%m-%d")
        
        # Enforce data availability constraint
        # News and fundamental data only available for August 2024
        min_date = datetime(2024, 8, 1)   # Aug 1, 2024
        max_date = datetime(2024, 8, 31)  # Aug 31, 2024
        
        if parsed_date < min_date:
            raise ValueError(f"Date too early: {date_string}. "
                           f"News and fundamental data only available from 2024-08-01")
        if parsed_date > max_date:
            raise ValueError(f"Date too late: {date_string}. "
                           f"News and fundamental data only available until 2024-08-31")
        if parsed_date > datetime.now():
            raise ValueError(f"Future date not allowed: {date_string}")
            
        return date_string
        
    except ValueError as e:
        if "does not match format" in str(e):
            raise ValueError(f"Invalid date format: {date_string}. Use YYYY-MM-DD format.")
        else:
            raise  # Re-raise other ValueError types


def setup_pipeline_environment() -> None:
    """Initialize pipeline environment and output directories."""
    try:
        # Ensure output directory exists
        ensure_output_directory("outputs/dummy.txt")  # Creates outputs/ dir
        
        # Validate configuration directory exists
        config_dir = Path("config")
        if not config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {config_dir}")
            
        print("✅ Pipeline environment initialized")
        
    except Exception as e:
        raise RuntimeError(f"Failed to setup pipeline environment: {e}")


def print_pipeline_header(as_of_date: str, forward_days: int, config_file: str) -> None:
    """
    Display formatted pipeline execution header with key parameters.
    
    Provides clear visibility into pipeline configuration and execution
    parameters for logging and debugging purposes.
    
    Args:
        as_of_date: Analysis date being used
        forward_days: Backtest window length
        config_file: Configuration file path
    """
    separator = "=" * 70
    print(f"\n{separator}")
    print("🚀 ALPHA AGENTS PIPELINE - Multi-Agent Stock Analysis")
    print(separator)
    print(f"📅 Analysis Date: {as_of_date}")
    print(f"🎯 Universe: AAPL, MSFT, NVDA, TSLA (4 stocks)")
    print(f"🤖 Agents: Valuation, Sentiment, Fundamental + Coordinator")
    print(f"📊 Backtest: {forward_days} trading days (~{forward_days//21:.1f} months)")
    print(f"⚙️  Configuration: {config_file}")
    print(separator)


def print_pipeline_summary(portfolio_result: Dict[str, Any], 
                         backtest_result: Optional[Dict[str, Any]] = None) -> None:
    """
    Display comprehensive pipeline execution summary.
    
    Provides final results overview including portfolio composition,
    performance metrics, and output file locations for user review.
    
    Args:
        portfolio_result: Portfolio analysis results from agent workflow
        backtest_result: Optional backtest performance results
    """
    separator = "=" * 70
    print(f"\n{separator}")
    print("✅ PIPELINE EXECUTION COMPLETE")
    print(separator)
    
    # Portfolio composition summary
    weights = portfolio_result.get("portfolio_weights", {})
    if weights:
        print("🎯 Final Portfolio Allocation:")
        for ticker, weight in weights.items():
            consensus = portfolio_result["ticker_analyses"][ticker]["consensus_rating"]
            print(f"   • {ticker}: {weight*100:.1f}% (Rating: {consensus})")
        print(f"   Total Stocks: {len(weights)}")
    else:
        print("🎯 Final Portfolio: Empty (100% Cash)")
    
    # Performance summary (if backtesting was performed)
    if backtest_result and "portfolio_return" in backtest_result:
        print(f"\n📈 Performance Summary ({backtest_result.get('test_period_days', 'N/A')} days):")
        port_ret = backtest_result['portfolio_return'] * 100
        bench_ret = backtest_result['benchmark_return'] * 100
        excess_ret = backtest_result.get('excess_return', 0) * 100
        
        print(f"   • Portfolio Return: {port_ret:+.2f}%")
        print(f"   • Benchmark Return: {bench_ret:+.2f}%")
        print(f"   • Excess Return: {excess_ret:+.2f}%")
        
        # Risk metrics
        port_sharpe = backtest_result.get('portfolio_sharpe', 0)
        bench_sharpe = backtest_result.get('benchmark_sharpe', 0)
        print(f"   • Portfolio Sharpe: {port_sharpe:.3f}")
        print(f"   • Benchmark Sharpe: {bench_sharpe:.3f}")
    
    # Output files generated
    print(f"\n📁 Output Files Generated:")
    print(f"   • outputs/picks.csv - Agent ratings and consensus decisions")
    print(f"   • outputs/performance.csv - Portfolio vs benchmark metrics")
    print(f"   • outputs/portfolio_chart.png - Performance visualization")
    
    print(f"\n💡 Next Steps:")
    print(f"   • Review agent reasoning in picks.csv")
    print(f"   • Analyze performance attribution in outputs/")
    print(f"   • Adjust agent thresholds if needed in config/")
    
    print(separator)


def run_agent_analysis(as_of_date: str) -> Dict[str, Any]:
    """
    Execute multi-agent stock analysis workflow.
    
    Orchestrates parallel execution of specialized agents (Valuation, Sentiment,
    Fundamental) followed by coordinator consensus and portfolio construction.
    This is the core analytical component of the pipeline.
    
    Args:
        as_of_date: Analysis date in YYYY-MM-DD format
        
    Returns:
        Dict[str, Any]: Complete portfolio analysis results including:
            - Individual agent ratings and scores
            - Consensus decisions per ticker
            - Portfolio weights and composition
            - Detailed analysis metadata
            
    Raises:
        RuntimeError: If agent analysis fails or returns invalid results
        KeyError: If required data is missing from agent responses
    """
    print("\n🤖 Executing Multi-Agent Analysis...")
    try:
        portfolio_result = run_portfolio_workflow(as_of_date)
        
        # Validate results structure
        required_keys = ["ticker_analyses", "portfolio_weights", "as_of_date"]
        for key in required_keys:
            if key not in portfolio_result:
                raise KeyError(f"Missing required key in portfolio result: {key}")
        
        # Validate we have analysis for expected tickers
        expected_tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]
        analyzed_tickers = list(portfolio_result["ticker_analyses"].keys())
        
        if not all(ticker in analyzed_tickers for ticker in expected_tickers):
            missing = set(expected_tickers) - set(analyzed_tickers)
            raise ValueError(f"Missing analysis for tickers: {missing}")
        
        print("✅ Multi-agent analysis completed successfully")
        return portfolio_result
        
    except Exception as e:
        print(f"❌ Multi-agent analysis failed: {e}")
        raise RuntimeError(f"Agent analysis workflow failed: {e}")


def run_performance_backtest(portfolio_result: Dict[str, Any], 
                           forward_days: int,
                           price_loader: PriceDataLoader) -> Optional[Dict[str, Any]]:
    """
    Execute portfolio performance backtesting against benchmark.
    
    Compares agent-selected portfolio performance vs equal-weight benchmark
    over specified forward window with proper leakage controls and risk
    adjustment. This validates the effectiveness of agent decisions.
    
    Args:
        portfolio_result: Portfolio analysis results from agent workflow
        forward_days: Forward testing window in trading days
        price_loader: Initialized price data loader instance
        
    Returns:
        Optional[Dict[str, Any]]: Backtest results including:
            - Portfolio and benchmark returns
            - Risk metrics (volatility, Sharpe ratios)
            - Performance attribution data
            - Price data for visualization
        Returns None if backtesting fails.
        
    Raises:
        RuntimeError: If backtesting fails with unrecoverable error
    """
    print(f"\n📊 Running Performance Backtest ({forward_days} day window)...")
    # print("   • Loading forward price data with leakage controls")
    # print("   • Calculating portfolio vs benchmark returns")
    # print("   • Computing risk-adjusted performance metrics")
    # print("   • Generating performance attribution analysis")
    
    try:
        backtest_result = run_complete_backtest(
            portfolio_result=portfolio_result,
            price_loader=price_loader,
            forward_days=forward_days
        )
        
        # Validate backtest results
        if not backtest_result:
            raise ValueError("Backtest returned empty results")
            
        required_metrics = ["portfolio_return", "benchmark_return"]
        for metric in required_metrics:
            if metric not in backtest_result:
                raise KeyError(f"Missing required backtest metric: {metric}")
        
        print("✅ Performance backtesting completed successfully")
        return backtest_result
        
    except Exception as e:
        print(f"⚠️  Backtesting failed: {e}")
        print("   Pipeline will continue with analysis results only")
        return None


def save_pipeline_outputs(portfolio_result: Dict[str, Any],
                         backtest_result: Optional[Dict[str, Any]] = None) -> None:
    """
    Generate and save all required pipeline output files.
    
    Creates CSV reports and performance visualizations as specified by
    the assignment requirements. Handles both success and failure cases
    gracefully to ensure partial results are always preserved.
    
    Args:
        portfolio_result: Portfolio analysis results
        backtest_result: Optional backtest performance results
        
    Raises:
        RuntimeError: If critical output generation fails
    """
    print("\n💾 Generating Output Files...")
    
    try:
        # Always save agent picks (independent of backtesting)
        picks_path = save_picks_csv(portfolio_result)
        print(f"   • Agent decisions: {Path(picks_path).name}")
        
        # Save performance metrics if available
        if backtest_result:
            performance_path = save_performance_csv(backtest_result, portfolio_result)
            print(f"   • Performance metrics: {Path(performance_path).name}")
            print(f"   • Performance chart: portfolio_chart.png")
        else:
            print("   • Performance outputs skipped (no backtest results)")
        
        # Create summary for logging
        summary = create_output_summary(portfolio_result, backtest_result)
        summary_path = "outputs/pipeline_summary.txt"
        ensure_output_directory(summary_path)
        
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"   • Execution summary: {Path(summary_path).name}")
        
        print("✅ All output files generated successfully")
        
    except Exception as e:
        print(f"❌ Output generation failed: {e}")
        raise RuntimeError(f"Failed to save pipeline outputs: {e}")


def main() -> int:
    """
    Main pipeline orchestration function with comprehensive error handling.
    
    Coordinates the complete Alpha Agents workflow from configuration loading
    through final output generation. Implements robust error handling and
    logging to support debugging and operational monitoring.
    
    Returns:
        int: Exit code (0=success, 1=failure) for shell integration
        
    The pipeline follows this execution sequence:
        1. Parse and validate command line arguments
        2. Initialize environment and load configuration  
        3. Execute multi-agent stock analysis
        4. Run performance backtesting (optional)
        5. Generate output files and reports
        6. Display execution summary
    """
    
    # Configure command line argument parsing
    parser = argparse.ArgumentParser(
        description="Alpha Agents Multi-Agent Stock Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py                           # Default execution
    python run_pipeline.py --date 2024-09-15        # Custom analysis date  
    python run_pipeline.py --forward-days 45        # Shorter backtest window
    python run_pipeline.py --no-backtest            # Skip performance evaluation
    python run_pipeline.py --config config/test.yaml # Alternative configuration

For more information, see README.md or visit the project repository.
        """
    )
    
    parser.add_argument(
        "--date",
        type=str,
        default="2024-08-20",
        help="Analysis as-of date in YYYY-MM-DD format. Must be in August 2024 (default: %(default)s)"
    )
    
    parser.add_argument(
        "--forward-days",
        type=int,
        default=63,
        help="Backtest window in trading days, ~3 months (default: %(default)s)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/agent_config.yaml",
        help="Agent configuration file path (default: %(default)s)"
    )
    
    parser.add_argument(
        "--no-backtest",
        action="store_true",
        help="Skip backtesting and only run agent analysis"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging and debug output"
    )
    
    # Parse arguments and validate inputs
    args = parser.parse_args()
    
    try:
        # Input validation
        as_of_date = validate_date(args.date)
        forward_days = args.forward_days
        config_file = args.config
        
        if forward_days <= 0:
            raise ValueError(f"Forward days must be positive, got: {forward_days}")
        
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        # Environment setup
        setup_pipeline_environment()
        print_pipeline_header(as_of_date, forward_days, config_file)
        
        # Load configuration
        print("⚙️  Loading agent configuration...")
        # Reinitialize config with custom path if provided
        if config_file != "config/agent_config.yaml":
            from src.config import AgentConfig
            config = AgentConfig(config_file)
        print(f"✅ Configuration loaded successfully")
        
        # Initialize data loaders
        print("\n📥 Initializing data infrastructure...")
        price_loader = PriceDataLoader()
        print("✅ Price data loader initialized")
        
        # Step 1: Multi-agent analysis (core pipeline)
        portfolio_result = run_agent_analysis(as_of_date)
        
        # Step 2: Performance backtesting (optional)
        backtest_result = None
        if not args.no_backtest:
            backtest_result = run_performance_backtest(
                portfolio_result, forward_days, price_loader
            )
        else:
            print("\n📊 Backtesting skipped (--no-backtest flag)")
        
        # Step 3: Output generation
        save_pipeline_outputs(portfolio_result, backtest_result)
        
        # Step 4: Execution summary
        print_pipeline_summary(portfolio_result, backtest_result)
        
        return 0  # Success exit code
        
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user (Ctrl+C)")
        print("   Partial results may be available in outputs/ directory")
        return 1
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("   Verify all required files exist and paths are correct")
        return 1
        
    except ValueError as e:
        print(f"\n❌ Invalid input parameter: {e}")
        print("   Check command line arguments and try again")
        return 1
        
    except RuntimeError as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        print("   Check logs above for detailed error information")
        if args.verbose:
            print("\nDetailed traceback:")
            traceback.print_exc()
        return 1
        
    except Exception as e:
        print(f"\n❌ Unexpected error occurred: {e}")
        print("   This may indicate a bug or system issue")
        print("\nFor debugging assistance:")
        print("1. Run with --verbose flag for detailed error traces")
        print("2. Check that all dependencies are installed correctly")
        print("3. Verify data files exist in data/ directory")
        print("4. Review configuration file for syntax errors")
        
        if args.verbose:
            print("\nFull traceback:")
            traceback.print_exc()
            
        return 1


def run_quick_test(as_of_date: str = "2024-08-20") -> Optional[Dict[str, Any]]:
    """Quick test execution for development and debugging."""
    try:
        print("🧪 Running quick test execution...")
        
        # Use minimal setup
        setup_pipeline_environment()
        # Config is already loaded by default
        
        # Run core pipeline
        price_loader = PriceDataLoader()
        portfolio_result = run_portfolio_workflow(as_of_date)
        backtest_result = run_complete_backtest(portfolio_result, price_loader, 63)
        
        # Save outputs quietly
        save_picks_csv(portfolio_result)
        if backtest_result:
            save_performance_csv(backtest_result, portfolio_result)
        
        print("✅ Quick test completed successfully")
        
        return {
            "portfolio": portfolio_result,
            "backtest": backtest_result,
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return None


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)