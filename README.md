# Vibe Trader

## Overview
Vibe Trader is a cryptocurrency trading agent that watches Telegram for signals, checks them on Kraken, and makes portfolio decisions with help from OpenAI. It favors medium‑term moves, typically holding up to 4 hours, and uses careful order execution to get in and out efficiently.

### Notes on Vibe Coding

This project was built with an exploratory “vibe coding” approach. That means we prioritized learning quickly and stitching together real APIs into something useful. Along the way, we proved that Telegram sentiment, OpenAI analysis, Kraken trading, and simple learning signals can work together as a practical trading loop.

If you plan to run this in a team setting, a fresh implementation is recommended. Treat this codebase as a design reference that validated the core ideas and integrations. In production, invest in thorough test coverage from day one so changes remain safe and predictable over time.

## Getting Started

### Environment Variables

#### Required Variables
| Variable | Description | Example |
|----------|-------------|---------|
| `TELEGRAM_BOT_TOKEN` | Telegram Bot API token for message monitoring | `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz` |
| `TELEGRAM_CHAT_ID` | Telegram chat ID to monitor for messages | `-1001234567890` |
| `OPENAI_API_KEY` | OpenAI API key for sentiment analysis and portfolio decisions | `sk-...` |
| `KRAKEN_API_KEY` | Kraken API key for trading operations | `your_kraken_api_key` |
| `KRAKEN_SECRET` | Kraken API secret for authentication | `your_kraken_secret` |

#### Optional Variables
| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `MAX_TRADE_LIFETIME_HOURS` | Hours before opportunities expire | `4` | Controls position holding duration |
| `STOP_LOSS_PERCENTAGE` | Stop-loss protection percentage below entry price | `20` | **Note**: Currently implemented as hardcoded values (10% or 20%) in code, environment variable not yet integrated |
| `LIVE_MODE` | Enable live trading (vs paper trading) | `false` | Set to `true` for actual trade execution |

### API Setup
Set up the required API accounts and keys according to each provider’s documentation, then export the environment variables listed above. That’s all the system needs to run.

### Quick Start
The easiest way to get Vibe Trader running is with these simple steps:

```bash
# 1. Create and activate virtual environment (recommended Python 3.8+)
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set required environment variables
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
export OPENAI_API_KEY="your_openai_key"
export KRAKEN_API_KEY="your_kraken_key"
export KRAKEN_SECRET="your_kraken_secret"

# 4. Run the trading system
python run_system.py
```

**Note**: Keep the virtual environment activated for all subsequent runs. The system will run in paper trading mode by default (`LIVE_MODE=false`). Set `LIVE_MODE=true` only when you're ready for live trading.

### Detailed Installation and Execution

For more control over configuration and environment setup:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (add to your shell profile or .env file)
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
export OPENAI_API_KEY="your_openai_key"
export KRAKEN_API_KEY="your_kraken_key"
export KRAKEN_SECRET="your_kraken_secret"

# Optional: Configure trading parameters
export MAX_TRADE_LIFETIME_HOURS=4
export LIVE_MODE=false

# Run the trading system
python run_system.py
```

#### Virtual Environment Commands (macOS/Linux)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Deactivate when done
deactivate
```

**Python Version Requirements:**
- **Recommended**: Python 3.8 or higher
- **Minimum**: Python 3.7 (may require package version adjustments)
- **Tested with**: Python 3.9+ (compatible with all packages in requirements.txt)

## Implementation Strategy

### Geographic & Regulatory Considerations
- **Trading Jurisdiction**: Operates in Texas, where Kraken offers broad spot coverage and solid liquidity
- **Position Restrictions**: Long-only, aligned with local restrictions on shorting for non‑institutional traders
- **Sentiment Utilization**: Uses Telegram sentiment as directional context, aiming to ride sustained moves rather than scalp entries/exits

### OpenAI Prompt Strategies
- **Sentiment Analysis**: Structured prompts blend historical performance context, technicals, and source trust to interpret messages
- **Portfolio Analysis**: RAG evaluates current holdings against new ideas with indicators and trust scores to recommend hold vs switch
- **Trade Decision Logic**: Considers sentiment strength, technical confirmation, costs, and how current positions are doing
- **Holding Bias**: Prefers holding until `MAX_TRADE_LIFETIME_HOURS` unless a clearly better, high‑trust opportunity appears

## Architecture
- **Main Loop** (`main.py`): Service-oriented orchestrator running every 15 minutes via `run_system.py`
- **Telegram Service** (`services/telegram.py`): Handles message polling and update tracking
- **Kraken Service** (`services/kraken.py`): Manages market data, balance queries, and order execution
- **OpenAI Service** (`services/openai.py`): Processes sentiment analysis and portfolio recommendations
- **Opportunity Service** (`services/opportunity.py`): Manages trading opportunities, technical analysis, and bandit model
- **APIs**: Integrates with Telegram Bot API, OpenAI GPT, Kraken REST API

## Workflow

### 1. Opportunity Lifecycle Management
- **Expired Opportunity Cleanup**: Automatically clears opportunities older than `MAX_TRADE_LIFETIME_HOURS` (default: 4 hours)
- **Position Consistency Bias**: System maintains existing positions until full expiry period to capture medium-term bullish sentiment trends
- **PnL Recording**: Expired opportunities trigger buy-and-hold PnL calculations based on entry price vs current market price
- **State Management**: Removes expired opportunities from active tracking while preserving performance data

### 2. Bandit Model Updates
- **Incremental Processing**: Only processes new outcomes from `state_performance.json` to avoid redundant calculations
- **Trust Score Calculation**: Computed as historical performance from inception to `MAX_TRADE_LIFETIME_HOURS` based on buy-and-hold returns, weighted by outcome frequency
- **Source Performance Tracking**: Maintains per-source statistics including success rate, average profit/loss, and total trades processed
- **Live Learning**: Updated exclusively during live trading cycles (no offline training due to Telegram API historical data limitations)

### 3. Telegram Message Processing
- **Polling Strategy**: Queries Telegram Bot API for messages since last update using `state_last_update_id.json`
- **Real-time Detection**: Processes only new messages since previous cycle to avoid duplicate processing
- **Asset Extraction**: Identifies cryptocurrency mentions using OpenAI analysis of message content
- **Sentiment Integration**: Combines message sentiment with technical confirmation before opportunity creation

### 4. Technical Analysis Integration
**Technical Indicators Used:**
- **Price Momentum**: SMA (Simple Moving Average), EMA (Exponential Moving Average)
- **Trend Strength**: MACD (Moving Average Convergence Divergence) with signal line
- **Volatility**: Bollinger Bands (20-period, 2 standard deviations)
- **Overbought/Oversold**: RSI (Relative Strength Index, 14-period)
- **Price Change**: 24-hour percentage change from Kraken ticker data
- **Volume Metrics**: 24-hour volume and price-volume trends

### 5. Portfolio Analysis & Position Management
- **RAG Decision Framework**: OpenAI analyzes current portfolio against new opportunities using technical indicators and trust scores
- **Holding Preference**: Explicit bias towards maintaining positions until `MAX_TRADE_LIFETIME_HOURS` expires unless high-trust opportunities emerge
- **Switching Criteria**: Requires combination of high trust score (>0.8), strong conviction, and advantageous technical indicators
- **Cost Awareness**: Considers transaction fees and spread costs when evaluating position changes

### 6. Advanced Order Execution Strategy

#### Selling Phase (Complete Liquidation)
- **Order Cancellation**: Cancels any outstanding orders holding funds to maximize available balance
- **Iterative Limit Orders**: Attempts 3 progressive limit sell orders:
  - **Iteration 1**: 0.5% below current price, 2-minute polling
  - **Iteration 2**: 0.6% below current price, 2-minute polling
  - **Iteration 3**: 0.7% below current price, 2-minute polling
- **Market Fallback**: Executes market order if all limit attempts timeout
- **Balance Tracking**: Monitors available vs held balances throughout process

#### Buying Phase (Complete Reallocation)
- **Capital Deployment**: Uses 95% of post-liquidation USD balance for new position
- **Iterative Limit Orders**: Attempts 3 progressive limit buy orders:
  - **Iteration 1**: 0.5% above current price, 2-minute polling
  - **Iteration 2**: 0.6% above current price, 2-minute polling
  - **Iteration 3**: 0.7% above current price, 2-minute polling
- **Market Fallback**: Executes market order if all limit attempts timeout
- **Stop-Loss Protection**: Configurable via `STOP_LOSS_PERCENTAGE` environment variable (default: 20% below entry price)

#### Stop-Loss Configuration
- **Environment Variable**: `STOP_LOSS_PERCENTAGE` (default: 20)
- **Calculation**: `(current_price * (1 - stop_loss_percentage / 100))`
- **Coverage**: Applied to entire position size for comprehensive risk management
- **Kraken Integration**: Uses conditional close orders for automatic execution

## Bandit Reward Model
- **Training Limitation**: Cannot be trained offline due to Telegram Bot API restrictions on historical message access
- **Live Learning**: Exclusively updated during live trading cycles with real P&L outcomes
- **Performance Tracking**: Records buy-and-hold returns from opportunity entry to `MAX_TRADE_LIFETIME_HOURS` expiry
- **Trust Score Evolution**: Source reliability scores improve over time based on actual trading performance
- **Outcome Processing**: Handles both successful (profitable) and failed (unprofitable) trades to refine future signal filtering

## Components

### Service Architecture
- **TelegramService**: Message polling, update ID management, and real-time message processing
- **KrakenService**: Balance queries, order execution, market data retrieval, and price validation
- **OpenAIService**: Sentiment analysis, technical indicator processing, and portfolio recommendations
- **OpportunityService**: Technical analysis, bandit model management, and opportunity lifecycle tracking

### Key Configuration Variables
- `MAX_TRADE_LIFETIME_HOURS`: Opportunity expiry time (default: 4)
- `STOP_LOSS_PERCENTAGE`: Stop-loss protection level (default: 20)
- `TELEGRAM_BOT_TOKEN`: Telegram Bot API authentication
- `KRAKEN_API_KEY`: Kraken API authentication
- `OPENAI_API_KEY`: OpenAI API authentication

## Trade Logging
All trade factors are comprehensively logged including:
- Message content, sender, and timestamp
- Sentiment score and detected cryptocurrency assets
- Technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands, volume metrics)
- Bandit trust score and confidence metrics
- Trade parameters (size, direction, stop-loss percentage, entry price)
- Order execution details (limit vs market, partial fills, fees)
- Actual P&L outcomes (buy-and-hold returns, duration held)
- Bandit model updates and performance statistics
