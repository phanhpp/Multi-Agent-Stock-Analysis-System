# Alpha Agents - Multi-Agent Stock Analysis System

A multi-agent framework for systematic stock analysis and portfolio construction, implementing concepts from BlackRock's "AlphaAgents" research. The system orchestrates three specialized AI agents (Valuation, Sentiment, Fundamental) plus a Coordinator to analyze stocks and evaluate performance through backtesting.

## Setup

### Environment & Dependencies

1. **Python Requirements**: Python 3.8+ with pip
2. **Create Virtual Environment & Install Dependencies**:
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **financialdatasets.ai Integration**:
   - **API Key**: Set environment variable `FINANCIAL_DATASETS_API_KEY` with your API key
   - **Fallback Behavior**: If API fails or key missing, automatically falls back to cached CSV files. 

4. **Environment Setup**:
   ```bash
   # Option 1: Set environment variable
   export FINANCIAL_DATASETS_API_KEY="your_api_key_here"
   
   # Option 2: Create .env file in project root
   echo "FINANCIAL_DATASETS_API_KEY=your_api_key_here" > .env
   ```

### How to Run

**Note**: Make sure your virtual environment is activated before running:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

```bash
# Default execution (2024-08-20, 63-day backtest)
python run_pipeline.py

# Custom analysis date (must be in August 2024)
python run_pipeline.py --date 2024-08-15

# Shorter backtest window
python run_pipeline.py --forward-days 45

# Skip backtesting (analysis only)
python run_pipeline.py --no-backtest

# Custom configuration
python run_pipeline.py --config config/custom.yaml
```

**Important**: The pipeline only allows decision date (the “as-of” date) from 2024-08-01 to 2024-08-31 and will reject other dates.

## Design

### Agent Logic

**Valuation Agent**:
- Calculates annualized return and volatility from 90-day price history
- BUY: Return >15% AND volatility <30%
- SELL: Return <-5% OR volatility >50%
- Uses BlackRock paper formulas for risk-adjusted momentum

**Sentiment Agent**:
- Processes news articles using VADER sentiment analysis
- Combines title and snippet for richer context
- BUY: Average sentiment >0.4 (~75th percentile)
- SELL: Average sentiment <-0.03 (below neutral)
- Analyzes 5-10 news articles per ticker per date

**Fundamental Agent**:
- Evaluates fundamental metrics (1-5 scale)
- Averages scores across all available factors
- BUY: Average score >4.0 (strong fundamentals)
- SELL: Average score <2.0 (weak fundamentals)
- Covers valuation, growth, profitability, financial strength

**⚙️ Configuration**: All agent decision thresholds are fully configurable via `config/agent_config.yaml` - no code changes required to adjust BUY/SELL/HOLD criteria. For example, change `buy_return_threshold: 15.0` to `10.0` to make the valuation agent more aggressive (buy stocks with lower returns).

### Coordinator Rule

**Weighted Consensus**:
- Valuation: 40% weight, Sentiment: 30%, Fundamental: 30%
- BUY: Weighted vote ≥0.4 threshold
- SELL: Weighted vote ≤0.2 threshold
- HOLD: Between thresholds (0.2-0.4)
- Portfolio construction: BUY stocks get full weight (1.0), HOLD stocks get half weight (0.5), SELL stocks excluded (0.0)


### Multi-Agent Workflow Architecture

**Individual Stock Analysis** (`ticker_workflow.py`):
- **LangGraph Orchestration**: Uses StateGraph to coordinate 3 specialized agents + coordinator
- **Parallel Execution**: Valuation, Sentiment, and Fundamental agents run **simultaneously** for each stock
- **Workflow**: `START → [Valuation || Sentiment || Fundamental] → Coordinator → END`
- **Output**: Single BUY/SELL/HOLD decision per stock with detailed reasoning

**Portfolio Analysis** (`portfolio_workflow.py`):
- **Subgraph Architecture**: Uses ticker_workflow as a reusable subgraph component
- **Parallel Stock Analysis**: Runs 4 ticker subgraphs **simultaneously** (AAPL || MSFT || NVDA || TSLA)
- **Workflow**: `START → [AAPL_subgraph || MSFT_subgraph || NVDA_subgraph || TSLA_subgraph] → Portfolio_Builder → END`
- **Output**: Complete portfolio allocation with individual stock decisions

### Cached Data for API Fallback

The system implements an **API-first strategy** with cached data as reliable fallback, supporting all "as-of" dates from August 1-31, 2024.

📅 **Cache Date Range**: May 3, 2024 to November 29, 2024
- **Start**: 2024-05-03 (90 days before Aug 1 for technical indicators)
- **End**: 2024-11-29 (90 days after Aug 31 for backtesting)
 
**Cache Locations**:
- **Primary fallback**: `data/raw/prices/[TICKER].csv` (individual ticker files)
- **Combined backup**: `data/raw/cached_prices_combined.csv` (all tickers combined)

**API Behavior**:
1. Always tries financialdatasets.ai API first (fresh data when available)
2. Falls back to cached data if API fails
3. Cache files remain protected during normal operations (preserve_cache=True)

### Leakage Controls

**Temporal Boundaries**:
- **Supported Analysis Dates**: August 1-31, 2024 only (enforced in `run_pipeline.py`)

- **News Data Coverage**: May 1, 2024 to August 31, 2024 (provides sufficient lookback)

- **News Filtering**: Sentiment agent only uses articles published **before or on** the as-of date.

- **Fundamental Data**: All fundamental data - SEC 10-Q filings - was publicly available **before August 1, 2024**, ensuring no future information is used in analysis.

- **Price Data**: Strict cutoff at analysis date for indicators


## Backtest

### Window Choice
- **Default**: 63 trading days (~3 months forward performance)
- **Rationale**: Long enough to capture meaningful performance differences, short enough to avoid regime changes
- **Trading Days Adjustment**: Converts trading days to calendar days using 7/5 ratio (accounts for weekends)
- **Example**: 63 trading days = 88 calendar days

### Metric Definitions

**Return Metrics**:
- **Portfolio Return**: Each stock's return multiplied by its conviction-based weight (BUY=full, HOLD=half, SELL=zero), then summed
- **Benchmark Return**: Equal-weight (25% each) return of all four tickers
- **Excess Return**: Portfolio return minus benchmark return

**Risk Metrics**:
- **Volatility**: Annualized standard deviation of daily returns (252 trading days)
- **Sharpe Ratio**: (Annualized return - 5% risk-free rate) / Annualized volatility
- **Risk-free Rate**: 5% annual rate for Sharpe calculation

**Portfolio Construction**:
- Empty portfolio → 100% cash (0% return)
- Selected stocks → Equal weight within selected universe
- No leverage or short positions

## Assumptions & Limitations

### Simplifications Made

**Data Constraints**:
- No transaction costs, slippage, or liquidity constraints
- Daily rebalancing assumed (unrealistic for real trading)

**Model Simplifications**:
- **Agent thresholds**: Decision thresholds (e.g., 15% return for BUY, 0.4 sentiment score) are intuitive/trial-based, which is reasonable for this prototype given the 20-hour time constraint and focus on system design over statistical optimization.
- Equal weighting within selected universe (no position sizing optimization)
- Simple linear combination for coordinator (no machine learning)
- No dynamic threshold adjustment based on market conditions
- No sector/factor exposure controls 

**Backtesting Limitations**:
- Single time period (August-November 2024)
- Survivorship bias (only analyzing current constituents)
- No regime change detection

## Time Accounting

**Rough Development Breakdown** (~8-10 hours total):

- **Architecture & Setup** (1.5-2 hours): Project structure, LangGraph integration, configuration system
- **Data Infrastructure** (2-2.5 hours): API integration, fallback mechanisms, data loaders, validation
- **Agent Development** (2-2.5 hours): Individual agent logic, VADER sentiment, fundamental scoring
- **Coordination & Portfolio** (1-1.5 hours): Weighted voting, portfolio construction, consensus logic
- **Backtesting Engine** (1-1.5 hours): Performance calculations, risk metrics, temporal controls
- **Testing & Validation** (0.5-1 hour): Unit tests, pipeline validation, edge case handling
- **Documentation & Polish** (0.5-1 hour): README, code comments, output formatting

**Time Distribution**:
- 40% Core functionality (agents, coordination, backtest)
- 30% Data handling (API, validation, caching)
- 20% System architecture (workflows, configuration)
- 10% Testing and documentation

## AI-Tool Usage

**Cursor AI Assistant**: Used throughout development for:
- Debugging API integration issues
- Writing comprehensive docstrings and comments
- Generating test cases and validation logic
- README documentation and formatting

**Anthropic Claude** (Better at following instructions):
- Code structure and architectural decisions
- Generating prompts for curating news and fundamental data

**ChatGPT** (Good at web search):
- Curating news and fundamental data using web search capabilities

### Output Files
- **picks.csv**: Individual agent ratings and consensus decisions
- **performance.csv**: Portfolio vs benchmark performance metrics  
- **portfolio_chart.png**: Growth of $1 visualization
- **pipeline_summary.txt**: Execution summary and metadata

## Tests

**Comprehensive test suite** (31 tests) covering core functionality:
- **Unit Tests** (26 tests):
  - **Data Loading**: API integration, fallback mechanisms, temporal filtering
  - **Backtest Math**: Return calculations, risk metrics, portfolio construction logic  
  - **Pipeline Validation**: Date validation, input sanitization, error handling
- **Integration Tests** (5 tests):
  - **End-to-End Pipeline**: Full workflow execution with real components
  - **Output Validation**: Verifies `picks.csv`, `performance.csv`, and summary files
  - **Content Sanity**: Validates rating consistency and reasonable return values
  - **Deterministic Behavior**: Ensures consistent results across runs

Run tests: `python -m pytest tests/` (requires activated `.venv`)


