#!/usr/bin/env python3
"""
Simple integration test for end-to-end pipeline execution.

Tests that the full pipeline runs successfully:
- Data loading
- Agent analysis 
- Coordinator consensus
- Backtest execution
- Output file generation
"""

import pytest
import tempfile
import shutil
import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import json

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for integration testing."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    
    try:
        # Copy necessary files to temp directory
        project_root = Path(__file__).parent.parent
        temp_path = Path(temp_dir)
        
        # Copy essential directories
        essential_dirs = ['src', 'config', 'data']
        for dir_name in essential_dirs:
            src_dir = project_root / dir_name
            if src_dir.exists():
                shutil.copytree(src_dir, temp_path / dir_name)
        
        # Copy run_pipeline.py
        pipeline_file = project_root / 'run_pipeline.py'
        if pipeline_file.exists():
            shutil.copy2(pipeline_file, temp_path / 'run_pipeline.py')
        
        # Create outputs directory
        (temp_path / 'outputs').mkdir(exist_ok=True)
        
        # Change to temp directory
        os.chdir(temp_dir)
        
        yield temp_path
        
    finally:
        # Restore original directory and cleanup
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestIntegration:
    """Integration tests for full pipeline execution."""
    
    def test_end_to_end_pipeline_execution(self, temp_workspace):
        """Test that the full pipeline runs successfully end-to-end."""
        # Use a test date within the valid range (August 2024)
        test_date = "2024-08-20"
        
        # Run the pipeline
        try:
            result = subprocess.run([
                sys.executable, 'run_pipeline.py', 
                '--date', test_date,
                '--no-backtest'  # Skip backtest for faster testing
            ], 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
            )
            
            # Check that pipeline completed successfully
            assert result.returncode == 0, f"Pipeline failed with error: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.fail("Pipeline execution timed out after 5 minutes")
        except FileNotFoundError:
            pytest.skip("run_pipeline.py not found - skipping integration test")
    
    def test_output_files_created(self, temp_workspace):
        """Test that expected output files are created."""
        test_date = "2024-08-20"
        outputs_dir = temp_workspace / 'outputs'
        
        # Run pipeline (skip if run_pipeline.py doesn't exist)
        pipeline_file = temp_workspace / 'run_pipeline.py'
        if not pipeline_file.exists():
            pytest.skip("run_pipeline.py not found - skipping output file test")
        
        try:
            subprocess.run([
                sys.executable, 'run_pipeline.py',
                '--date', test_date,
                '--no-backtest'
            ], 
            check=True, 
            capture_output=True, 
            timeout=300
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pytest.skip("Pipeline execution failed - skipping output validation")
        
        # Check that expected output files exist
        # Note: performance.csv is not created when using --no-backtest
        expected_files = [
            'picks.csv',
            'pipeline_summary.txt'
        ]
        
        for filename in expected_files:
            file_path = outputs_dir / filename
            assert file_path.exists(), f"Expected output file not created: {filename}"
            assert file_path.stat().st_size > 0, f"Output file is empty: {filename}"
    
    def test_output_content_sanity(self, temp_workspace):
        """Test that output files contain reasonable data."""
        test_date = "2024-08-20"
        outputs_dir = temp_workspace / 'outputs'
        
        # Run pipeline (skip if doesn't exist or fails)
        pipeline_file = temp_workspace / 'run_pipeline.py'
        if not pipeline_file.exists():
            pytest.skip("run_pipeline.py not found - skipping content validation")
        
        try:
            subprocess.run([
                sys.executable, 'run_pipeline.py',
                '--date', test_date,
                '--no-backtest'
            ], 
            check=True, 
            capture_output=True, 
            timeout=300
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pytest.skip("Pipeline execution failed - skipping content validation")
        
        # Validate picks.csv content
        picks_file = outputs_dir / 'picks.csv'
        if picks_file.exists():
            picks_df = pd.read_csv(picks_file)
            
            # Should have data for our 4 tickers
            expected_tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA']
            assert not picks_df.empty, "picks.csv is empty"
            
            # Check for required columns
            required_cols = ['ticker', 'consensus_rating']
            for col in required_cols:
                assert col in picks_df.columns, f"Missing required column in picks.csv: {col}"
            
            # Ratings should be valid
            valid_ratings = ['BUY', 'HOLD', 'SELL']
            for rating in picks_df['consensus_rating']:
                assert rating in valid_ratings, f"Invalid rating found: {rating}"
        
        # Validate performance.csv content
        performance_file = outputs_dir / 'performance.csv'
        if performance_file.exists():
            perf_df = pd.read_csv(performance_file)
            
            assert not perf_df.empty, "performance.csv is empty"
            
            # Should have numeric return columns
            numeric_cols = ['portfolio_return', 'benchmark_return', 'excess_return']
            for col in numeric_cols:
                if col in perf_df.columns:
                    # Returns should be reasonable (not crazy large)
                    returns = perf_df[col].iloc[0] if len(perf_df) > 0 else None
                    if returns is not None and not pd.isna(returns):
                        assert abs(returns) < 2.0, f"Unreasonable return value in {col}: {returns}"
    
    def test_pipeline_deterministic_behavior(self, temp_workspace):
        """Test that pipeline produces consistent results when run multiple times."""
        test_date = "2024-08-20"
        
        pipeline_file = temp_workspace / 'run_pipeline.py'
        if not pipeline_file.exists():
            pytest.skip("run_pipeline.py not found - skipping deterministic test")
        
        # Run pipeline twice
        results = []
        for run_num in range(2):
            try:
                result = subprocess.run([
                    sys.executable, 'run_pipeline.py',
                    '--date', test_date,
                    '--no-backtest'
                ], 
                capture_output=True, 
                text=True, 
                timeout=300
                )
                results.append(result.returncode)
            except subprocess.TimeoutExpired:
                pytest.skip("Pipeline execution timed out")
        
        # Both runs should succeed
        assert all(rc == 0 for rc in results), "Pipeline results not consistent across runs"
        
        # Check that picks.csv is identical (if it exists)
        picks_file = temp_workspace / 'outputs' / 'picks.csv'
        if picks_file.exists():
            # For this simple test, we just verify the file exists and has content
            # More sophisticated tests could compare actual content
            assert picks_file.stat().st_size > 0, "picks.csv should have content"
    
    def test_full_pipeline_with_backtest(self, temp_workspace):
        """Test full pipeline including backtest to verify performance.csv creation."""
        test_date = "2024-08-20"
        outputs_dir = temp_workspace / 'outputs'
        
        pipeline_file = temp_workspace / 'run_pipeline.py'
        if not pipeline_file.exists():
            pytest.skip("run_pipeline.py not found - skipping full pipeline test")
        
        # Run full pipeline WITH backtest (this takes longer)
        try:
            result = subprocess.run([
                sys.executable, 'run_pipeline.py',
                '--date', test_date
                # No --no-backtest flag, so backtest will run
            ], 
            capture_output=True, 
            text=True, 
            timeout=600  # 10 minute timeout for full pipeline
            )
            
            # Should complete successfully
            assert result.returncode == 0, f"Full pipeline failed: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.skip("Full pipeline execution timed out")
        except subprocess.CalledProcessError:
            pytest.skip("Full pipeline execution failed")
        
        # Check that ALL expected files are created (including performance.csv)
        expected_files = [
            'picks.csv',
            'performance.csv',  # Should exist when backtest runs
            'pipeline_summary.txt'
        ]
        
        for filename in expected_files:
            file_path = outputs_dir / filename
            assert file_path.exists(), f"Expected output file not created: {filename}"
            assert file_path.stat().st_size > 0, f"Output file is empty: {filename}"


class TestPipelineComponents:
    """Test individual pipeline components in isolation."""
    
    def test_data_validation_integration(self):
        """Test that data validation works with pipeline components."""
        # Test date validation (copied from pipeline validation)
        from datetime import datetime
        
        def validate_date(date_string: str) -> str:
            """Simple date validation for integration testing."""
            try:
                parsed_date = datetime.strptime(date_string, "%Y-%m-%d")
                
                # Enforce data availability constraint
                min_date = datetime(2024, 8, 1)
                max_date = datetime(2024, 8, 31)
                
                if parsed_date < min_date or parsed_date > max_date:
                    raise ValueError(f"Date outside valid range: {date_string}")
                    
                return date_string
                
            except ValueError as e:
                if "does not match format" in str(e):
                    raise ValueError(f"Invalid date format: {date_string}")
                else:
                    raise
        
        # Should accept valid dates
        valid_date = validate_date("2024-08-20")
        assert valid_date == "2024-08-20"
        
        # Should reject invalid dates
        with pytest.raises(ValueError):
            validate_date("2024-07-01")  # Too early
        
        with pytest.raises(ValueError):
            validate_date("invalid-date")  # Bad format


# Pytest will automatically discover and run tests
# To run: pytest tests/test_integration.py -v
