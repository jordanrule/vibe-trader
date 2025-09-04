# Vibe Trader

## Overview
An intelligent cryptocurrency trading agent that monitors Telegram messages for crypto sentiment, validates trade opportunities on Kraken, and executes trades with ML-guided decision making. The system operates with a medium-term bias, focusing on maintaining positions for up to 4 hours to capture sustained bullish sentiment while utilizing advanced order execution strategies for optimal trade execution.

### Notes on Vibe Coding

Vibe Trader was developed through an exploratory "vibe coding" approach, which is reflected in the code quality and current test coverage. This development methodology excels at rapidly prototyping complex integrations across multiple APIs to create a functional, cohesive system. The approach successfully demonstrated how Telegram sentiment monitoring, OpenAI analysis, Kraken trading, and machine learning can be combined into a working cryptocurrency trading agent.

For production deployment with team maintenance, a complete rewrite from scratch would be recommended. The current implementation serves as an excellent design document, having validated key architectural decisions and API integrations. Future development should prioritize comprehensive test coverage from the initial implementation phase, as extensive automated testing is essential for maintaining system reliability when multiple developers are involved.

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

### API Setup Instructions

#### Telegram Bot Setup
1. **Create a Telegram Bot**: Message [@BotFather](https://t.me/botfather) on Telegram with `/newbot` to create your bot and receive your `TELEGRAM_BOT_TOKEN`
2. **Get Chat ID**: Add your bot to the desired Telegram channel/group and send a message, then use `https://api.telegram.org/bot<YourBOTToken>/getUpdates` to find the chat ID
3. **Forward Messages to Bot**: Telegram channels must be forwarded to your bot using [@junction_bot](https://t.me/junction_bot) or similar forwarding service, as bots cannot directly access channel messages

#### OpenAI API Setup
1. **Create Account**: Sign up at [OpenAI Platform](https://platform.openai.com/)
2. **Generate API Key**: Navigate to API Keys section and create a new secret key
3. **Billing Setup**: Configure billing to enable API usage
4. **Environment Variable**: Set `OPENAI_API_KEY` with your generated key

#### Kraken API Setup
1. **Create Account**: Sign up at [Kraken Exchange](https://www.kraken.com/)
2. **Enable API**: Go to Settings → API → Generate New Key
3. **Configure Permissions**: Enable necessary permissions (Query Funds, Query Open Orders, Query Closed Orders, Create & Modify Orders)
4. **Environment Variables**: Set both `KRAKEN_API_KEY` and `KRAKEN_SECRET`

### Quick Start
The easiest way to get Vibe Trader running is with these simple steps:

```bash
# 1. Create and activate virtual environment (recommended Python 3.8+)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

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

#### Virtual Environment Commands by Operating System

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Deactivate when done
deactivate
```

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Deactivate when done
deactivate
```

**Python Version Requirements:**
- **Recommended**: Python 3.8 or higher
- **Minimum**: Python 3.7 (may require package version adjustments)
- **Tested with**: Python 3.9+ (compatible with all packages in requirements.txt)

## Implementation Strategy

### Geographic & Regulatory Considerations
- **Trading Jurisdiction**: Operates in Texas, where Kraken maintains the largest quantity of available cryptocurrencies, providing optimal market depth and liquidity for trading operations
- **Position Restrictions**: Adheres to Texas regulations prohibiting short-selling of cryptocurrencies except for institutional investors, focusing exclusively on long positions
- **Sentiment Utilization**: Leverages Telegram sentiment for medium-term directional guidance rather than explicit entry/exit point marking, allowing the system to capture sustained market movements

### OpenAI Prompt Strategies
- **Sentiment Analysis**: Uses structured prompts with historical performance context, technical indicators, and source trust scores to evaluate message sentiment and asset identification
- **Portfolio Analysis**: Employs RAG (Retrieval-Augmented Generation) with position-specific context, including current holdings, technical indicators, and trust scores to make holding vs switching decisions
- **Trade Decision Logic**: Arrives at current trade decisions by weighing sentiment strength against technical confirmation, transaction costs, and existing position performance
- **Holding Bias**: Prompts include explicit preferences to maintain positions until MAX_TRADE_LIFETIME_HOURS expires unless compelling new opportunities with high trust scores emerge

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
