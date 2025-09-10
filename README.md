# Vibe Trader

A sophisticated cryptocurrency trading system that uses AI-powered sentiment analysis and technical indicators to make automated trading decisions.

## Vibe Coding

This project was built with an exploratory “vibe coding” approach. That means we prioritized learning quickly and stitching together real APIs into something useful. Along the way, we proved that Telegram sentiment, OpenAI analysis, Kraken trading, and simple learning signals can work together as a practical trading loop.

If you plan to run this in a team setting, a fresh implementation is recommended. Treat this codebase as a design reference that validated the core ideas and integrations. In production, invest in thorough test coverage from day one so changes remain safe and predictable over time.

## Performance

The live-trading results of vibe trader show potential, but improvements need to be made.  Specifically: vibe trader is far too confident in it's ability to pick the most relevant opportunity cycle-to-cycle, and loses money consistently on transaction costs.  The fundamental hypothesis of telegram signals providing a market edge over medium-term time horizons looks promising, but we still have not collected enough data to make a statistically significant determination.

If you plan to run this in a live setting, backtestable evaluations on historical telegram and market data are recommended.  In production, it is neccessary to validate changes to the prompts or their struture to ensure agent decisions trend towards making profitable financial decisions.

## Getting Started

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

### Environment Variables
- `LIVE_MODE=false` - Enable live trading
- `CLOUD_MODE=true` - Enable cloud mode
- `TELEGRAM_BOT_TOKEN` - Telegram bot token
- `OPENAI_API_KEY` - OpenAI API key
- `KRAKEN_API_KEY` - Kraken API key
- `KRAKEN_API_SECRET` - Kraken API secret
- `MAX_TRADE_LIFETIME_HOURS` - Trade lifetime
- `STOP_LOSS_PERCENTAGE` - Stop loss

### Google Cloud Setup
1. Create GCP project
2. Enable required APIs (Cloud Run, Secret Manager, Storage)
3. Create storage bucket
4. Set up secrets in Secret Manager
5. Deploy using Cloud Build

### Docker

Build Image
```bash
docker build -t vibe-trader .
```

Run Locally
```bash
docker run -p 8080:8080 -e CLOUD_MODE=true vibe-trader
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

## Workflow

It is recommended that it runs every hour to check new opportunities and rebalance it's single holding.  At any given cycle it chooses a single cryptocurrency based on sentiment and techicals.  The MAX_TRADE_LIFETIME_HOURS determines how long an opportunity exists, as well as how exit time for PnL is calculated for the performance model.

### 1. Opportunity Lifecycle Management
- **Expired Opportunity Cleanup**: Automatically clears opportunities older than `MAX_TRADE_LIFETIME_HOURS` (default: 6 hours)
- **Position Consistency Bias**: System maintains existing positions until full expiry period to capture medium-term bullish sentiment trends
- **PnL Recording**: Expired opportunities trigger buy-and-hold PnL calculations based on entry price vs current market price
- **State Management**: Removes expired opportunities from active tracking while preserving performance data

### 2. Performance Model Updates
- **Incremental Processing**: Only processes new outcomes from `state_performance.json` to avoid redundant calculations
- **Trust Score Calculation**: Computed as historical performance from inception to `MAX_TRADE_LIFETIME_HOURS` based on buy-and-hold returns, weighted by outcome frequency
- **Source Performance Tracking**: Maintains per-source statistics including success rate, average profit/loss, and total trades processed
- **Live Learning**: Updated exclusively during live trading cycles (no offline training due to Telegram API historical data limitations)

### 3. Check Opportunities on Telegram
- **Polling Strategy**: Queries Telegram Bot API for messages since last update using `state_last_update_id.json`
- **Real-time Detection**: Processes only new messages since previous cycle to avoid duplicate processing
- **Asset Extraction**: Identifies cryptocurrency mentions using OpenAI analysis of message content
- **Sentiment Integration**: Combines message sentiment with technical confirmation before opportunity creation

**Technical Indicators Calculated:**
- **Price Momentum**: SMA (Simple Moving Average), EMA (Exponential Moving Average)
- **Trend Strength**: MACD (Moving Average Convergence Divergence) with signal line
- **Volatility**: Bollinger Bands (20-period, 2 standard deviations)
- **Overbought/Oversold**: RSI (Relative Strength Index, 14-period)
- **Price Change**: 24-hour percentage change from Kraken ticker data
- **Volume Metrics**: 24-hour volume and price-volume trends

### 4. Portfolio Analysis & Position Management
- **RAG Decision Framework**: OpenAI analyzes current portfolio against new opportunities using technical indicators and trust scores
- **Holding Preference**: Explicit bias towards maintaining positions until `MAX_TRADE_LIFETIME_HOURS` expires unless high-trust opportunities emerge
- **Switching Criteria**: Requires combination of high trust score (>0.8), strong conviction, and advantageous technical indicators
- **Cost Awareness**: Considers transaction fees and spread costs when evaluating position changes

### 5. Order Execution Strategy

#### Reallocation Phase
- **Order Cancellation**: Cancels any outstanding orders holding funds to maximize available balance
- **Iterative Limit Orders**: Attempts limit orders near market price
- **Market Fallback**: Executes market order if all limit attempts timeout
- **Balance Tracking**: Monitors available vs held balances throughout process
- **Stop-Loss Protection**: Configurable via `STOP_LOSS_PERCENTAGE` environment variable (default: 20% below entry price)

#### Stop-Loss Configuration
- **Environment Variable**: `STOP_LOSS_PERCENTAGE` (default: 20)
- **Calculation**: `(current_price * (1 - stop_loss_percentage / 100))`
- **Coverage**: Applied to entire position size for comprehensive risk management
- **Kraken Integration**: Uses conditional close orders for automatic execution
