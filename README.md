# Live Trading Bot

Autonomous crypto trading bot for Hyperliquid perpetuals, powered by DeepSeek AI.

## Overview

This bot combines AI-driven decision making with algorithmic trading strategies to trade cryptocurrency perpetuals on the Hyperliquid exchange. It features multi-coin trading capabilities, comprehensive risk management, and full transparency through detailed logging of trades and AI reasoning.

### Key Features

- **AI-Driven Decisions**: Uses DeepSeek's language model to analyze market conditions and make trading decisions
- **Multi-Coin Trading**: Supports trading multiple perpetual contracts simultaneously
- **Risk Management**: Built-in position sizing, leverage controls, and drawdown protection
- **Full Transparency**: Logs all trades and AI reasoning for auditing and analysis
- **Modular Architecture**: Pluggable strategies and easy-to-extend design
- **Real-Time Monitoring**: Dashboard for tracking positions, trades, and performance

## Architecture

The bot is organized into the following modules:

- **`src/exchange/`**: Hyperliquid API integration

  - Client wrapper for authentication, market data, and order management

- **`src/ai/`**: DeepSeek decision engine

  - Uses OpenAI-compatible API to generate trading decisions based on market data

- **`src/strategies/`**: Trading algorithms

  - Momentum trading
  - Mean-reversion strategies
  - ML-based classifiers

- **`src/risk/`**: Position sizing and risk controls

  - Leverage limits
  - Stop-loss enforcement
  - Drawdown kill-switches

- **`src/logging/`**: Trade and AI reasoning logs

  - `TradeLogger` class for structured logging to SQLite database
    - AI decisions with reasoning, confidence, and execution status
    - Trade executions with entry/exit prices, P&L, holding time
    - Query methods for analysis and performance metrics
    - JSON/CSV export for external analysis
  - `ChatLogger` class for human-readable message logging
    - AI reasoning and plan/action messages
    - Trade event summaries
    - Risk management alerts
    - Automatic log rotation

- **`src/ui/`**: Dashboard for monitoring bot activity

  - `dashboard.py`: Main dashboard module with CLI and web implementations
    - `DashboardData` class: Data aggregation from loggers and exchange
    - `CLIDashboard` class: Terminal-based dashboard using Rich library
    - `WebDashboard` class: FastAPI web dashboard with REST API
  - `templates/dashboard.html`: Jinja2 template for web UI
  - CLI mode: One-time or live-updating terminal display
  - Web mode: Browser-based dashboard with real-time updates (polling or WebSocket)
  - REST API: JSON endpoints for external integrations

- **`src/main_loop.py`**: Main trading loop orchestration
  - `TradingBot` class that coordinates all modules
  - Continuous loop with configurable interval (default: 3 minutes)
  - State management for position metadata (stop-loss, profit targets)
  - Graceful shutdown with signal handling
  - Error recovery and retry logic
  - Integration with all modules: exchange, AI, strategies, risk, logging

## Testnet Setup

**IMPORTANT: Always test on testnet before using real funds!**

Hyperliquid provides a testnet environment where you can test the trading bot with mock USDC without risking real money. This is the recommended way to:

- Verify the bot works correctly
- Test your trading strategies
- Understand the bot's behavior
- Tune risk parameters safely

### Step 1: Create or Use a Testnet Wallet

**Option A: Create a new wallet (recommended)**

```bash
# Generate a new Ethereum wallet (you can use any Ethereum wallet tool)
# Example using Python:
python -c "from eth_account import Account; acc = Account.create(); print(f'Address: {acc.address}\nPrivate Key: {acc.key.hex()}')"
```

**Option B: Use an existing wallet**

- You can use any Ethereum wallet (MetaMask, hardware wallet, etc.)
- **NEVER use your mainnet wallet with real funds for testnet testing**
- Create a separate wallet specifically for testnet

**Important:** Save your testnet wallet credentials securely. You'll need:

- Wallet address (public key)
- Private key (keep this secret)

### Step 2: Get Testnet USDC

1. **Visit the Hyperliquid testnet faucet:**

   - URL: https://app.hyperliquid-testnet.xyz/drip
   - This is the official Hyperliquid testnet faucet

2. **Claim mock USDC:**

   - Enter your testnet wallet address
   - Click "Drip" or "Claim" to receive testnet USDC
   - You should receive a sufficient amount for testing (typically 10,000+ USDC)
   - You can claim multiple times if needed

3. **Verify your balance:**
   - Visit https://app.hyperliquid-testnet.xyz
   - Connect your testnet wallet
   - Check that you have testnet USDC in your account

### Step 3: Configure Environment for Testnet

1. **Copy the environment template:**

   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` file with your testnet credentials:**

   ```bash
   # Hyperliquid testnet configuration
   HYPERLIQUID_WALLET_ADDRESS=0xYourTestnetWalletAddress
   HYPERLIQUID_PRIVATE_KEY=your_testnet_private_key_here
   HYPERLIQUID_TESTNET=true  # MUST be true for testnet

   # DeepSeek API key (same for testnet and mainnet)
   DEEPSEEK_API_KEY=your_deepseek_api_key_here

   # Conservative settings for testing
   MAX_LEVERAGE=3
   RISK_PER_TRADE_PERCENT=1.0
   MAX_DRAWDOWN_PERCENT=10.0
   ```

3. **Verify the configuration:**
   - Ensure `HYPERLIQUID_TESTNET=true` is set
   - Use conservative risk settings for initial testing
   - Start with lower leverage (3-5x) to understand behavior

### Step 4: Run the Bot on Testnet

1. **Start the bot:**

   ```bash
   uv run python main.py
   ```

2. **Verify testnet connection:**

   - Check the startup logs for:
     ```
     ============================================================
     RUNNING ON TESTNET WITH MOCK BALANCE
     This is a safe testing environment with no real funds at risk
     ============================================================
     ```
   - If you see "RUNNING ON MAINNET WITH REAL FUNDS", STOP immediately and check your `.env` file

3. **Monitor the bot:**
   - Watch the console logs for trading activity
   - Use the dashboard to view positions and trades:
     ```bash
     python -m src.ui.dashboard --mode web
     ```
   - Check the database for logged trades:
     ```bash
     sqlite3 ./data/trades.db "SELECT * FROM trade_executions;"
     ```

### Step 5: Test Thoroughly

**Recommended testing checklist:**

- [ ] Bot starts successfully and connects to testnet
- [ ] AI generates trading decisions
- [ ] Orders are placed and filled on testnet
- [ ] Stop-losses are enforced correctly
- [ ] Risk limits are respected (leverage, drawdown)
- [ ] Positions are tracked accurately
- [ ] Logs are written correctly (database and chat log)
- [ ] Dashboard displays correct information
- [ ] Bot handles errors gracefully (network issues, API failures)
- [ ] Graceful shutdown works (Ctrl+C)

**Testing duration:**

- Run for at least 24-48 hours on testnet
- Execute at least 10-20 trades
- Test different market conditions (trending, ranging, volatile)
- Verify P&L calculations match expectations

### Step 6: Switch to Mainnet (When Ready)

**Only proceed when:**

- ✅ Bot has run successfully on testnet for 24+ hours
- ✅ All features work as expected
- ✅ You understand the bot's behavior and risk management
- ✅ You have reviewed and are comfortable with the AI's decisions
- ✅ You have adequate funds and can afford potential losses

**To switch to mainnet:**

1. **Update `.env` file:**

   ```bash
   # Hyperliquid mainnet configuration
   HYPERLIQUID_WALLET_ADDRESS=0xYourMainnetWalletAddress
   HYPERLIQUID_PRIVATE_KEY=your_mainnet_private_key_here
   HYPERLIQUID_TESTNET=false  # MUST be false for mainnet

   # Start with conservative settings
   MAX_LEVERAGE=5
   RISK_PER_TRADE_PERCENT=1.0
   MAX_DRAWDOWN_PERCENT=15.0
   ```

2. **Use a mainnet wallet with funds:**

   - Ensure your mainnet wallet has sufficient USDC
   - Keep private keys secure (never commit to git)
   - Consider using a hardware wallet for added security

3. **Start with small positions:**

   - Use low `RISK_PER_TRADE_PERCENT` (0.5-1%)
   - Use conservative leverage (3-5x)
   - Monitor closely for the first few days

4. **Verify mainnet connection:**
   - Check startup logs for:
     ```
     ============================================================
     RUNNING ON MAINNET WITH REAL FUNDS
     All trades will use real money - ensure risk settings are correct
     ============================================================
     ```
   - This warning should appear in yellow/orange (WARNING level)

### Troubleshooting Testnet Issues

**Bot connects to mainnet instead of testnet:**

- Check `.env` file: `HYPERLIQUID_TESTNET=true` (not "True" or "TRUE")
- Restart the bot after changing `.env`
- Check startup logs for testnet confirmation

**Faucet not working:**

- Ensure you're using a valid Ethereum address
- Try again after a few minutes (rate limiting)
- Check Hyperliquid Discord for testnet status
- Alternative: Use a different wallet address

**Orders not filling on testnet:**

- Testnet liquidity may be lower than mainnet
- Use market orders or wider limit prices
- Check Hyperliquid testnet UI to see order book depth

**Testnet balance shows zero:**

- Verify you claimed from the faucet successfully
- Check the correct wallet address in `.env`
- Visit https://app.hyperliquid-testnet.xyz to verify balance

### Testnet vs Mainnet Differences

| Aspect        | Testnet                     | Mainnet                     |
| ------------- | --------------------------- | --------------------------- |
| **Funds**     | Mock USDC (no value)        | Real USDC (real money)      |
| **Risk**      | Zero financial risk         | Full financial risk         |
| **Liquidity** | Lower, may have slippage    | Higher, better fills        |
| **API URL**   | `constants.TESTNET_API_URL` | `constants.MAINNET_API_URL` |
| **Faucet**    | Free mock USDC available    | Must deposit real funds     |
| **Purpose**   | Testing and development     | Live trading                |
| **Data**      | Separate from mainnet       | Production data             |

**Note:** The bot automatically selects the correct API URL based on the `HYPERLIQUID_TESTNET` environment variable. The implementation in `src/exchange/hyperliquid_client.py` (line 54) handles this:

```python
base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
```

### Best Practices

1. **Always start on testnet** - Never skip testnet testing
2. **Use separate wallets** - Never mix testnet and mainnet wallets
3. **Test thoroughly** - Run for at least 24-48 hours before mainnet
4. **Start conservative** - Use low leverage and risk on mainnet initially
5. **Monitor closely** - Watch the bot carefully for the first week on mainnet
6. **Keep testnet running** - Use testnet for testing new strategies or settings
7. **Document changes** - Note any configuration changes and their effects
8. **Regular backups** - Backup your `.env` and state files regularly

## Trading Strategies

The bot includes three pluggable trading strategies that can be used independently or consulted by the AI engine:

### 1. Momentum Strategy (`src/strategies/momentum.py`)

**Approach:** Time-series momentum with volatility targeting

**Key Features:**

- EMA crossover detection (20-period vs 50-period)
- MACD confirmation for trend direction
- Breakout detection using ATR-based volatility bands
- Volatility targeting to adjust position confidence
- Multi-timeframe analysis (3-minute and 4-hour)

**Best for:** Trending markets with clear directional moves

**Usage:**

```python
from src.strategies import MomentumStrategy

strategy = MomentumStrategy()
signal = strategy.generate_signal(market_data)
print(f"Signal: {signal['signal']}, Confidence: {signal['confidence']:.2f}")
print(f"Reasoning: {signal['reasoning']}")
```

### 2. Mean Reversion Strategy (`src/strategies/mean_reversion.py`)

**Approach:** RSI-based oversold/overbought signals with Bollinger Band mean reversion

**Key Features:**

- RSI analysis (7-period and 14-period) for oversold/overbought detection
- Bollinger Bands (EMA20 ± 2\*STD) for price extremes
- Multi-timeframe RSI confirmation (3-minute and 4-hour)
- Reversal confirmation using recent price action
- Volatility-adjusted confidence (works best in low-medium volatility)

**Best for:** Range-bound markets with clear support/resistance levels

**Usage:**

```python
from src.strategies import MeanReversionStrategy

strategy = MeanReversionStrategy()
signal = strategy.generate_signal(market_data)
```

### 3. ML Classifier Strategy (`src/strategies/ml_classifier.py`)

**Approach:** Gradient boosting classifier for trend prediction using OHLCV + indicators

**Key Features:**

- Lightweight GradientBoostingClassifier (no GPU required)
- Feature engineering from price, RSI, MACD, EMA, ATR, volume, funding rate
- Rolling window training (last 100 candles)
- Automatic retraining every 20 invocations
- Feature importance analysis for interpretability
- Probability-based confidence scores

**Best for:** Complex market conditions where traditional indicators are ambiguous

**Usage:**

```python
from src.strategies import MLClassifierStrategy

strategy = MLClassifierStrategy()
signal = strategy.generate_signal(market_data)
```

### Strategy Interface

All strategies implement a standardized interface:

```python
def generate_signal(market_data: dict) -> dict:
    """
    Args:
        market_data: Dict containing:
            - indicators_3m: 3-minute indicators (prices, ema20, ema50, macd, rsi7, rsi14, atr, volume)
            - indicators_4h: 4-hour indicators (same structure)
            - open_interest: Current open interest
            - funding_rate: Current funding rate

    Returns:
        {
            "signal": "buy" | "sell" | "hold",
            "confidence": float (0.0 to 1.0),
            "reasoning": str (detailed explanation)
        }
    """
```

### Integration with AI Engine

The DeepSeek AI engine can optionally consult these strategies for additional signal confirmation:

```python
from src.ai import DeepSeekEngine
from src.strategies import MomentumStrategy, MeanReversionStrategy

engine = DeepSeekEngine()
momentum = MomentumStrategy()
mean_reversion = MeanReversionStrategy()

# Get AI decision
ai_decisions = engine.generate_trading_decision(market_data, positions, account_info)

# Get strategy signals for validation
momentum_signal = momentum.generate_signal(market_data["BTC"])
mean_reversion_signal = mean_reversion.generate_signal(market_data["BTC"])

# Compare signals and adjust confidence
if ai_decisions[0]["signal"] == momentum_signal["signal"]:
    print("AI and momentum strategy agree!")
```

### Strategy Selection Guidelines

- **High volatility, strong trends:** Use Momentum Strategy
- **Low volatility, range-bound:** Use Mean Reversion Strategy
- **Uncertain conditions:** Use ML Classifier Strategy
- **Best practice:** Combine multiple strategies and require agreement for higher confidence

### Customization

Each strategy exposes configurable parameters in its constructor:

```python
# Customize momentum strategy
momentum = MomentumStrategy()
momentum.ema_fast_period = 15  # Default: 20
momentum.ema_slow_period = 40  # Default: 50
momentum.breakout_atr_multiplier = 2.5  # Default: 2.0

# Customize mean reversion strategy
mean_reversion = MeanReversionStrategy()
mean_reversion.rsi_oversold_threshold = 25  # Default: 30
mean_reversion.rsi_overbought_threshold = 75  # Default: 70

# Customize ML classifier
ml_classifier = MLClassifierStrategy()
ml_classifier.lookback_window = 150  # Default: 100
ml_classifier.prediction_horizon = 10  # Default: 5
```

## Risk Management

The bot includes comprehensive risk management through the `RiskManager` class (`src/risk/risk_manager.py`), which enforces position sizing, leverage limits, stop-loss automation, and drawdown protection.

### Key Features

- **Dynamic Position Sizing**: Calculates optimal position size based on account risk percentage and stop-loss distance
- **Leverage Controls**: Validates both per-position and aggregate leverage limits
- **Automatic Stop-Loss**: Monitors positions and closes them automatically when stop-loss triggers
- **Drawdown Kill-Switch**: Tracks account drawdown and halts trading when threshold is exceeded
- **Minimum Position Size**: Prevents dust trades below minimum USD threshold
- **Maximum Single Position**: Limits individual position size as percentage of account

### Configuration Parameters

Configure risk parameters in `.env`:

| Parameter                     | Description                   | Default | Example                             |
| ----------------------------- | ----------------------------- | ------- | ----------------------------------- |
| `MAX_LEVERAGE`                | Maximum leverage per position | `10`    | `5` for 5x max                      |
| `RISK_PER_TRADE_PERCENT`      | Risk per trade (% of account) | `2.0`   | `1.5` for 1.5% risk                 |
| `MAX_DRAWDOWN_PERCENT`        | Kill-switch threshold         | `20.0`  | `15.0` for 15% drawdown limit       |
| `MAX_TOTAL_EXPOSURE_PERCENT`  | Maximum aggregate exposure    | `200.0` | `150.0` for 1.5x aggregate leverage |
| `MIN_POSITION_SIZE_USD`       | Minimum position size         | `10.0`  | `25.0` for $25 minimum              |
| `MAX_SINGLE_POSITION_PERCENT` | Max single position size      | `50.0`  | `30.0` for 30% of account           |

### Usage

```python
from src.risk import RiskManager
from src.exchange import HyperliquidClient

# Initialize
client = HyperliquidClient()
risk_manager = RiskManager(client)

# Calculate position size
account_value = 10000.0  # $10k account
current_price = 50000.0  # BTC at $50k
stop_loss_price = 49000.0  # Stop at $49k

sizing = risk_manager.calculate_position_size(
    signal={
        "coin": "BTC",
        "signal": "buy",
        "confidence": 0.8,
        "entry_price": current_price,
        "stop_loss": stop_loss_price,
        "take_profit": 52000.0
    },
    account_value=account_value,
    risk_per_trade=2.0  # Optional, defaults to RISK_PER_TRADE_PERCENT
)

if sizing["approved"]:
    print(f"Position size: {sizing['position_size']:.4f} BTC")
    print(f"Notional value: ${sizing['notional_value']:.2f}")
    print(f"Leverage: {sizing['leverage_used']:.2f}x")
else:
    print(f"Position rejected: {sizing['rejection_reason']}")

# Check leverage compliance
positions = client.get_open_positions()
leverage_check = risk_manager.check_max_leverage(positions)  # Fetches account value internally
if not leverage_check["within_limits"]:
    for violation in leverage_check["violations"]:
        print(f"Leverage violation: {violation}")

# Monitor stop-losses
# First, register stop-loss prices
risk_manager.set_position_stops({
    "BTC": 49000.0,
    "ETH": 2900.0
})
current_prices = {"BTC": 48500.0, "ETH": 3100.0}
stop_loss_result = risk_manager.enforce_stop_loss(positions, current_prices)
print(f"Positions closed: {stop_loss_result['positions_closed']}")
print(f"Total realized P&L: ${stop_loss_result['total_realized_pnl']:.2f}")
for attempt in stop_loss_result['close_attempts']:
    print(f"Close attempt for {attempt['symbol']}: {'success' if attempt['success'] else 'failed'}")

# Check drawdown
account_state = client.get_user_state()
account_value = float(account_state["marginSummary"]["accountValue"])
drawdown_check = risk_manager.check_drawdown_limit(account_value, peak_value=12000.0)  # Optional peak_value
if drawdown_check["kill_switch"]:
    print(f"KILL SWITCH ACTIVE: Drawdown {drawdown_check['current_drawdown_pct']:.2f}%")
    print(f"Peak: ${drawdown_check['peak_value']:,.2f}, Current: ${drawdown_check['current_value']:,.2f}")
    # Halt all trading operations
```

### Risk Controls Explained

**Position Sizing Algorithm:**

1. Calculates risk amount: `risk_amount = account_value × (risk_percent / 100)`
2. Calculates stop distance: `stop_distance = |entry_price - stop_loss_price|`
3. Calculates position size: `position_size = risk_amount / stop_distance`
4. Validates against leverage and position size limits

**Drawdown Kill-Switch:**

- Tracks peak account value (high-water mark)
- Calculates current drawdown: `drawdown = (peak - current) / peak × 100`
- Activates kill-switch if drawdown ≥ `MAX_DRAWDOWN_PERCENT`
- Requires manual reset via `reset_kill_switch()` after review

**Leverage Validation:**

- Per-position leverage: `leverage = (position_size × price) / account_value`
- Aggregate leverage: `total_leverage = Σ(all_position_values) / account_value`
- Both must be ≤ configured limits

**Stop-Loss Automation:**

- Monitors all open positions every iteration
- Triggers when: `current_price ≤ stop_loss` (long) or `current_price ≥ stop_loss` (short)
- Executes market orders (IOC) for immediate closure
- Logs all stop-loss executions with P&L

### Best Practices

1. **Conservative Risk Settings**: Start with `RISK_PER_TRADE_PERCENT ≤ 2%` and `MAX_LEVERAGE ≤ 5x`
2. **Always Set Stops**: Every signal should include `stop_loss` price
3. **Monitor Drawdown**: Regularly check `get_risk_summary()` for risk metrics
4. **Test on Testnet**: Validate risk calculations on testnet before live trading
5. **Review Kill-Switch**: Analyze trades and market conditions before resetting kill-switch
6. **Avoid Over-Leverage**: Keep `MAX_TOTAL_EXPOSURE_PERCENT` ≤ 200% (2x aggregate)

### Risk Summary

Get real-time risk metrics:

```python
summary = risk_manager.get_risk_summary(positions, account_value)
print(f"Total Positions: {summary['total_positions']}")
print(f"Total Notional: ${summary['total_notional']:,.2f}")
print(f"Effective Leverage: {summary['effective_leverage']:.2f}x")
print(f"Total Unrealized P&L: ${summary['total_unrealized_pnl']:,.2f}")
print(f"Current Drawdown: {summary['current_drawdown_pct']:.2f}%")
print(f"Available Margin: ${summary['available_margin']:,.2f}")
print(f"Kill Switch: {'ACTIVE' if summary['kill_switch_active'] else 'Inactive'}")
```

## Logging and Transparency

The bot provides comprehensive logging for full transparency and auditability of all trading activities:

### Trade Logger (Structured Data)

**Storage:** SQLite database + JSON export

**What's logged:**

- **AI Decisions:** Every trading decision from the AI engine
  - Timestamp, coin, signal (buy/sell/hold), confidence, reasoning
  - Profit target, stop loss, invalidation condition
  - Leverage, risk amount
  - Execution status (whether the trade was actually placed)
- **Trade Executions:** Every position entry and exit
  - Entry: timestamp, coin, side (long/short), price, quantity, notional value
  - Exit: timestamp, exit price, holding time, net P&L, exit reason
  - Linkage to related AI decision (if applicable)

**Database Schema:**

```sql
-- AI Decisions Table
CREATE TABLE ai_decisions (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    invocation_number INTEGER,
    coin TEXT,
    signal TEXT,
    quantity REAL,
    confidence REAL,
    reasoning TEXT,
    profit_target REAL,
    stop_loss REAL,
    invalidation_condition TEXT,
    leverage INTEGER,
    risk_usd REAL,
    executed INTEGER  -- 0 or 1
);

-- Trade Executions Table
CREATE TABLE trade_executions (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    coin TEXT,
    side TEXT,
    entry_price REAL,
    entry_time TEXT,
    exit_price REAL,
    exit_time TEXT,
    quantity REAL,
    notional_entry REAL,
    notional_exit REAL,
    holding_time_minutes REAL,
    net_pnl REAL,
    exit_reason TEXT,
    related_decision_id INTEGER
);
```

**Usage Example:**

```python
from src.logging import TradeLogger

# Initialize logger
trade_logger = TradeLogger()

# Log an AI decision
decision = {
    "coin": "BTC",
    "signal": "buy",
    "quantity": 0.1,
    "confidence": 0.75,
    "reasoning": "Strong bullish momentum with RSI oversold bounce",
    "profit_target": 115000,
    "stop_loss": 105000,
    "invalidation_condition": "If price closes below 104000",
    "leverage": 10,
    "risk_usd": 500
}
decision_id = trade_logger.log_ai_decision(decision, invocation_number=42)

# Log trade entry
trade_entry = {
    "coin": "BTC",
    "side": "long",
    "entry_price": 110000,
    "quantity": 0.1,
    "notional_entry": 11000
}
trade_logger.log_trade_execution(trade_entry, related_decision_id=decision_id)
trade_logger.mark_decision_executed(decision_id)

# Later: log trade exit
trade_exit = {
    "coin": "BTC",
    "exit_price": 115000,
    "exit_time": "2025-10-30T15:30:00",
    "net_pnl": 500,
    "exit_reason": "take_profit"
}
trade_logger.log_trade_execution(trade_exit)

# Query performance
summary = trade_logger.get_performance_summary()
print(f"Total trades: {summary['total_trades']}")
print(f"Win rate: {summary['win_rate']:.2%}")
print(f"Total P&L: ${summary['total_pnl']:,.2f}")
print(f"Profit factor: {summary['profit_factor']:.2f}")

# Export to JSON for analysis
trade_logger.export_to_json("./exports/trades_2025-10.json", start_date="2025-10-01")
```

**Query Examples:**

```python
# Get all decisions from last 7 days
from datetime import datetime, timedelta
start = (datetime.now() - timedelta(days=7)).isoformat()
decisions = trade_logger.get_decisions_by_date(start)

# Get all BTC trades
btc_trades = trade_logger.get_trades_by_coin("BTC", limit=50)

# Get currently open positions
open_positions = trade_logger.get_open_positions()
```

### Chat Logger (Human-Readable Messages)

**Storage:** Text file with automatic rotation

**What's logged:**

- **AI Reasoning:** Natural language explanations of trading decisions
- **Plan and Action:** What the AI is thinking and planning to do
- **Trade Events:** Human-readable summaries of position entries/exits
- **Risk Events:** Warnings, alerts, and risk management actions

**Log Format:**

```
[2025-10-30 12:34:56] [AI_REASONING]
Markets are oversold across the board with many coins showing very low RSI values,
so I'm holding all my current positions in ETH, SOL, XRP, BTC, DOGE, and BNB as
none of their invalidation conditions have been met, and I'm already in all the
available trades.

---

[2025-10-30 12:45:23] [TRADE_EVENT] EXIT - BNB
Bot completed a long trade on BNB!
Entry: $1,140.60 → Exit: $1,083.20
Quantity: 5.64
Notional: $6,433 → $6,109
Holding time: 43H 44M
Net P&L: -$327.04
Reason: Stop loss hit

---

[2025-10-30 13:00:00] [RISK_EVENT] DRAWDOWN_ALERT
Account drawdown: 18.5% (threshold: 20.0%)
Peak value: $12,500 → Current: $10,188
Distance to kill-switch: 1.5%

---
```

**Usage Example:**

```python
from src.logging import ChatLogger

# Initialize logger
chat_logger = ChatLogger()

# Log AI reasoning
chat_logger.log_message(
    "Markets are oversold across the board with many coins showing very low RSI values, "
    "so I'm holding all my current positions...",
    category="AI_REASONING"
)

# Log AI decision summary
decisions = [...]  # List of decisions from DeepSeekEngine
chat_logger.log_ai_decision_summary(decisions, invocation_number=42)

# Log trade event
chat_logger.log_trade_event(
    event_type="EXIT",
    coin="BNB",
    details={
        "entry_price": 1140.60,
        "exit_price": 1083.20,
        "quantity": 5.64,
        "notional_entry": 6433,
        "notional_exit": 6109,
        "holding_time_minutes": 2624,  # 43H 44M
        "net_pnl": -327.04,
        "reason": "Stop loss hit"
    }
)

# Log risk event
chat_logger.log_risk_event(
    event_type="DRAWDOWN_ALERT",
    details={
        "current_drawdown_pct": 18.5,
        "threshold_pct": 20.0,
        "peak_value": 12500,
        "current_value": 10188,
        "distance_to_threshold": 1.5
    }
)

# Read recent messages (for dashboard)
recent = chat_logger.get_recent_messages(count=20)
for msg in recent:
    print(msg)

# Search for specific events
stop_losses = chat_logger.search_messages("stop loss", max_results=50)
```

**Log Rotation:**

Chat logs automatically rotate when they exceed the configured size (default: 10 MB):

- Current log: `chat_log.txt`
- Rotated logs: `chat_log_2025-10-30_12-34-56.txt`
- Old logs are automatically deleted (keeps last 10 by default)

### Configuration

Logging parameters in `.env`:

| Variable                   | Description                    | Default               |
| -------------------------- | ------------------------------ | --------------------- |
| `DATABASE_PATH`            | SQLite database path           | `./data/trades.db`    |
| `CHAT_LOG_PATH`            | Chat log file path             | `./data/chat_log.txt` |
| `CHAT_LOG_MAX_SIZE_MB`     | Max log size before rotation   | 10 MB                 |
| `CHAT_LOG_RETENTION_COUNT` | Number of rotated logs to keep | 10                    |
| `LOG_LEVEL`                | Console logging verbosity      | INFO                  |

### Querying and Analysis

**Using SQLite directly:**

```bash
# Open database
sqlite3 ./data/trades.db

# Query AI decisions
SELECT coin, signal, confidence, reasoning
FROM ai_decisions
WHERE timestamp >= '2025-10-01'
ORDER BY timestamp DESC;

# Query trade performance by coin
SELECT
    coin,
    COUNT(*) as total_trades,
    SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
    SUM(net_pnl) as total_pnl,
    AVG(net_pnl) as avg_pnl,
    AVG(holding_time_minutes) as avg_holding_time
FROM trade_executions
WHERE exit_price IS NOT NULL
GROUP BY coin;

# Find best and worst trades
SELECT coin, entry_price, exit_price, net_pnl, exit_reason
FROM trade_executions
WHERE exit_price IS NOT NULL
ORDER BY net_pnl DESC
LIMIT 10;
```

**Using Python for analysis:**

```python
import sqlite3
import pandas as pd

# Load data into pandas
conn = sqlite3.connect('./data/trades.db')
df_decisions = pd.read_sql_query("SELECT * FROM ai_decisions", conn)
df_trades = pd.read_sql_query("SELECT * FROM trade_executions WHERE exit_price IS NOT NULL", conn)
conn.close()

# Analyze decision accuracy
df_decisions['correct'] = (
    ((df_decisions['signal'] == 'buy') & (df_trades['net_pnl'] > 0)) |
    ((df_decisions['signal'] == 'sell') & (df_trades['net_pnl'] > 0))
)
accuracy = df_decisions['correct'].mean()
print(f"AI decision accuracy: {accuracy:.2%}")

# Analyze by confidence level
df_decisions['confidence_bucket'] = pd.cut(df_decisions['confidence'], bins=[0, 0.5, 0.7, 0.9, 1.0])
accuracy_by_confidence = df_decisions.groupby('confidence_bucket')['correct'].mean()
print(accuracy_by_confidence)
```

### Best Practices

1. **Regular backups:** Backup the SQLite database regularly (it's just a file)
2. **Monitor log sizes:** Check chat log rotation is working properly
3. **Periodic exports:** Export to JSON/CSV for long-term archival
4. **Query optimization:** Use indices for common queries (already created)
5. **Privacy:** Logs may contain sensitive trading data; secure them appropriately
6. **Retention policy:** Decide how long to keep historical logs (disk space)

### Troubleshooting

**Database locked errors:**

- SQLite uses file-level locking; ensure only one writer at a time
- The TradeLogger uses context managers to properly close connections
- If errors persist, check for zombie processes holding locks

**Chat log not rotating:**

- Check file permissions on log directory
- Verify `CHAT_LOG_MAX_SIZE_MB` is set correctly
- Check disk space availability

**Missing data in logs:**

- Verify loggers are initialized in main loop
- Check for exceptions during logging (logged to console)
- Ensure database path is writable

## Dashboard UI

The bot includes both CLI and web-based dashboards for monitoring trading activity:

### CLI Dashboard

**Terminal-based monitoring using Rich library**

**Features:**

- Current positions with unrealized P&L
- Completed trades summary
- Performance metrics (win rate, total P&L, profit factor)
- Recent AI reasoning messages
- Account summary (value, cash, leverage, drawdown)
- Color-coded tables (green for profits, red for losses)

**Usage:**

```bash
# One-time render (snapshot)
python -m src.ui.dashboard --mode cli --once

# Live updates (refreshes every 30 seconds)
python -m src.ui.dashboard --mode cli --refresh 30

# Custom refresh interval (60 seconds)
python -m src.ui.dashboard --mode cli --refresh 60
```

**Example output:**

```
╭─────────────────────────────────────────────────────────────╮
│                    Account Summary                          │
│  Account Value: $10,250.00  |  Available Cash: $2,150.00   │
│  Effective Leverage: 6.2x   |  Drawdown: 8.5%              │
╰─────────────────────────────────────────────────────────────╯

┏━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Symbol ┃ Quantity ┃ Entry Price ┃ Current     ┃ Unrealized    ┃
┃        ┃          ┃             ┃ Price       ┃ P&L           ┃
┡━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ BTC    │ 0.1000   │ $107,343.00 │ $110,396.50 │ +$305.35      │
│ ETH    │ 4.9800   │ $4,001.10   │ $3,907.05   │ -$468.37      │
│ SOL    │ 124.64   │ $192.80     │ $195.41     │ +$324.30      │
└────────┴──────────┴─────────────┴─────────────┴───────────────┘
```

**Keyboard shortcuts:**

- `Ctrl+C`: Exit dashboard
- In live mode, dashboard auto-refreshes at configured interval

### Web Dashboard

**Browser-based monitoring with FastAPI**

**Features:**

- Real-time updates (polling or WebSocket)
- Responsive design (works on mobile/tablet/desktop)
- Interactive tables with sorting
- Detailed trade history
- AI chat log with search
- Performance charts (future enhancement)
- REST API for external integrations

**Usage:**

```bash
# Start web dashboard (default: http://0.0.0.0:8000)
python -m src.ui.dashboard --mode web

# Custom host and port
python -m src.ui.dashboard --mode web --host 127.0.0.1 --port 8080

# Run in background
nohup python -m src.ui.dashboard --mode web > dashboard.log 2>&1 &
```

**Access:**

- Open browser: `http://localhost:8000`
- From other devices: `http://<your-ip>:8000`
- API documentation: `http://localhost:8000/docs` (automatic OpenAPI docs)

**API Endpoints:**

| Endpoint           | Method    | Description                    |
| ------------------ | --------- | ------------------------------ |
| `/`                | GET       | Main dashboard HTML page       |
| `/api/data`        | GET       | All dashboard data (JSON)      |
| `/api/positions`   | GET       | Current positions only         |
| `/api/trades`      | GET       | Completed trades (paginated)   |
| `/api/performance` | GET       | Performance metrics            |
| `/api/chat`        | GET       | AI chat log                    |
| `/ws`              | WebSocket | Real-time updates (if enabled) |

**Example API usage:**

```bash
# Get all dashboard data
curl http://localhost:8000/api/data

# Get current positions
curl http://localhost:8000/api/positions

# Get last 10 trades
curl "http://localhost:8000/api/trades?limit=10"

# Get performance metrics
curl http://localhost:8000/api/performance
```

**Real-time updates:**

The web dashboard supports two modes for real-time updates:

1. **Polling (default)**: Frontend fetches data every 30 seconds

   - Simple, reliable, works everywhere
   - Set `DASHBOARD_ENABLE_WEBSOCKET=false` in `.env`

2. **WebSocket (optional)**: Server pushes updates to frontend
   - More efficient, lower latency
   - Set `DASHBOARD_ENABLE_WEBSOCKET=true` in `.env`
   - Requires WebSocket support in browser (all modern browsers)

### Configuration

Dashboard parameters in `.env`:

| Variable                     | Description                    | Default |
| ---------------------------- | ------------------------------ | ------- |
| `DASHBOARD_HOST`             | Web server host                | 0.0.0.0 |
| `DASHBOARD_PORT`             | Web server port                | 8000    |
| `DASHBOARD_REFRESH_INTERVAL` | CLI refresh interval (seconds) | 30      |
| `DASHBOARD_ENABLE_WEBSOCKET` | Use WebSocket for updates      | false   |

### Running Alongside Trading Bot

**Option 1: Separate terminals**

```bash
# Terminal 1: Run trading bot
uv run python main.py

# Terminal 2: Run web dashboard
python -m src.ui.dashboard --mode web
```

**Option 2: Background processes**

```bash
# Start bot in background
nohup uv run python main.py > bot.log 2>&1 &

# Start dashboard in background
nohup python -m src.ui.dashboard --mode web > dashboard.log 2>&1 &

# View logs
tail -f bot.log
tail -f dashboard.log
```

**Option 3: Screen/tmux sessions**

```bash
# Create screen session for bot
screen -S trading-bot
uv run python main.py
# Detach: Ctrl+A, D

# Create screen session for dashboard
screen -S dashboard
python -m src.ui.dashboard --mode web
# Detach: Ctrl+A, D

# Reattach to sessions
screen -r trading-bot
screen -r dashboard
```

### Screenshots

**CLI Dashboard:**

```
[Screenshot of terminal with Rich-formatted tables]
```

**Web Dashboard:**

```
[Screenshot of browser showing web dashboard]
```

### Troubleshooting

**CLI dashboard not displaying colors:**

- Ensure terminal supports ANSI colors
- Try setting `TERM=xterm-256color`
- Use a modern terminal (iTerm2, Windows Terminal, etc.)

**Web dashboard not accessible:**

- Check if port 8000 is already in use: `lsof -i :8000` (Mac/Linux) or `netstat -ano | findstr :8000` (Windows)
- Change port: `--port 8080`
- Check firewall settings if accessing from other devices
- Verify `DASHBOARD_HOST` is set to `0.0.0.0` for network access

**Dashboard shows "No data available":**

- Ensure trading bot has been running and logging data
- Check database path: `./data/trades.db` exists
- Verify chat log file: `./data/chat_log.txt` exists
- Check HyperliquidClient credentials are valid

**WebSocket connection fails:**

- Fallback to polling mode: Set `DASHBOARD_ENABLE_WEBSOCKET=false`
- Check browser console for errors
- Verify WebSocket endpoint: `ws://localhost:8000/ws`
- Some proxies/firewalls block WebSocket connections

**High memory usage:**

- Limit trade history: Modify query limits in `DashboardData`
- Reduce refresh frequency: Increase `DASHBOARD_REFRESH_INTERVAL`
- Clear old log files: Rotate chat logs more frequently

### Security Considerations

**Web dashboard security:**

- **No authentication by default**: Anyone with network access can view dashboard
- **Recommendation**: Use firewall rules to restrict access
- **For production**: Add authentication (FastAPI supports OAuth2, JWT)
- **Sensitive data**: Dashboard displays account balances and positions
- **Network exposure**: Set `DASHBOARD_HOST=127.0.0.1` for local-only access

**Best practices:**

1. Run dashboard on localhost only (`127.0.0.1`) unless needed remotely
2. Use SSH tunneling for remote access: `ssh -L 8000:localhost:8000 user@server`
3. Add authentication if exposing to internet (not included in basic implementation)
4. Use HTTPS/WSS for encrypted connections (requires reverse proxy like nginx)
5. Regularly review access logs

### Future Enhancements

Potential improvements (not in current implementation):

- Authentication and user management
- Performance charts (candlestick, P&L over time)
- Trade execution from dashboard (manual override)
- Alert notifications (email, Telegram, Discord)
- Mobile app (React Native, Flutter)
- Multi-bot monitoring (manage multiple bot instances)
- Backtesting visualization
- Strategy comparison dashboard

## Running the Bot

### Prerequisites

1. **Environment setup**: Copy `.env.example` to `.env` and fill in your credentials:

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

2. **Install dependencies**:

   ```bash
   uv sync
   ```

3. **Test on testnet first**: Ensure `HYPERLIQUID_TESTNET=true` in your `.env` file before running with real funds.

### Starting the Bot

**Basic usage:**

```bash
uv run python main.py
```

**With custom log level:**

```bash
LOG_LEVEL=DEBUG uv run python main.py
```

**Running in background (Linux/Mac):**

```bash
nohup uv run python main.py > bot.log 2>&1 &
```

**Using screen/tmux for persistent sessions:**

```bash
screen -S trading-bot
uv run python main.py
# Detach with Ctrl+A, D
# Reattach with: screen -r trading-bot
```

### Bot Lifecycle

**Startup:**

1. Loads environment variables and validates credentials
2. Initializes all modules (exchange client, AI engine, risk manager, strategies)
3. Loads persisted state (position metadata) from `./data/bot_state.json`
4. Sets up signal handlers for graceful shutdown
5. Enters main trading loop

**Main Loop (every 3 minutes by default):**

1. **Fetch market data**: Prices, candles, indicators, funding rates
2. **Get positions**: Current open positions and account state
3. **Risk checks**: Drawdown monitoring, leverage validation
4. **Enforce stop-losses**: Auto-close positions that hit stop-loss or near liquidation
5. **Generate AI decisions**: Call DeepSeek engine with market data
6. **Optional strategy consultation**: Validate AI decisions with momentum/mean-reversion/ML strategies
7. **Execute trades**: Place orders for approved signals
8. **Check invalidations**: Close positions if invalidation conditions met
9. **Save state**: Persist position metadata to disk
10. **Sleep**: Wait until next interval

**Shutdown:**

1. Receives shutdown signal (Ctrl+C or SIGTERM)
2. Completes current iteration
3. Saves state to disk
4. Logs final statistics
5. Exits cleanly

### Monitoring

**Console output:**
The bot logs all activities to console with timestamps:

```
2025-10-30 12:34:56 - main_loop - INFO - Starting iteration #42
2025-10-30 12:34:57 - main_loop - INFO - Fetched market data for 6 symbols
2025-10-30 12:34:58 - main_loop - INFO - Current positions: 3, Account value: $10,250
2025-10-30 12:35:00 - main_loop - INFO - AI decision: BTC buy (confidence: 0.75)
2025-10-30 12:35:01 - main_loop - INFO - Trade executed: BTC long, size: 0.1, price: $110,000
```

**Log files:**
Optionally configure file logging in `main.py` to write to `./logs/bot.log`.

**Database queries:**
Query the SQLite database for trade history:

```bash
sqlite3 ./data/trades.db "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10;"
```

**Chat log:**
View AI reasoning messages:

```bash
tail -f ./data/chat_log.txt
```

### Stopping the Bot

**Graceful shutdown (recommended):**

```bash
# Press Ctrl+C in the terminal
# Or send SIGTERM:
kill -TERM <pid>
```

The bot will:

- Complete the current iteration
- Save state to disk
- Log final statistics
- Exit cleanly

**Force stop (not recommended):**

```bash
kill -9 <pid>
```

This may leave state unsaved and positions unmanaged.

### Configuration

**Key parameters in `.env`:**

| Variable                       | Description                   | Default                  | Impact                         |
| ------------------------------ | ----------------------------- | ------------------------ | ------------------------------ |
| `TRADING_INTERVAL_MINUTES`     | Loop frequency                | 3                        | Higher = less frequent trading |
| `TRADING_SYMBOLS`              | Coins to trade                | BTC,ETH,SOL,BNB,XRP,DOGE | Comma-separated list           |
| `MAX_LEVERAGE`                 | Max leverage per position     | 10                       | Lower = safer                  |
| `RISK_PER_TRADE_PERCENT`       | Max loss per trade            | 2.0                      | Lower = smaller positions      |
| `MAX_DRAWDOWN_PERCENT`         | Kill-switch threshold         | 20.0                     | Lower = earlier shutdown       |
| `ENABLE_STRATEGY_CONSULTATION` | Use strategies for validation | false                    | true = more conservative       |

### Troubleshooting

**Bot won't start:**

- Check `.env` file exists and has all required variables
- Verify API keys are correct
- Check network connectivity to Hyperliquid and DeepSeek APIs
- Review logs for specific error messages

**No trades being executed:**

- Check if kill-switch is active (drawdown exceeded)
- Verify account has sufficient balance
- Check if AI is generating "hold" signals (market conditions)
- Review risk manager rejection reasons in logs

**Positions not closing at stop-loss:**

- Verify position metadata is being saved (check `./data/bot_state.json`)
- Check if stop-loss prices are reasonable
- Review logs for stop-loss enforcement attempts
- Ensure bot is running continuously (not stopped/restarted frequently)

**High API errors:**

- Check network stability
- Verify API rate limits not exceeded
- Review Hyperliquid/DeepSeek API status
- Consider increasing `TRADING_INTERVAL_MINUTES` to reduce API calls

**State file corrupted:**

- Delete `./data/bot_state.json` to start fresh
- Bot will lose position metadata (stop-losses, profit targets)
- Manually review open positions and set new stop-losses

### Best Practices

1. **Always test on testnet first**: Set `HYPERLIQUID_TESTNET=true` and verify bot behavior
2. **Start with small positions**: Use low `RISK_PER_TRADE_PERCENT` (0.5-1%) initially
3. **Monitor regularly**: Check logs and positions frequently, especially in first days
4. **Use conservative leverage**: Start with 3-5x, not 10x
5. **Set tight drawdown limits**: Use 10-15% `MAX_DRAWDOWN_PERCENT` for safety
6. **Keep bot running**: Avoid frequent restarts to ensure stop-losses are enforced
7. **Backup state file**: Periodically backup `./data/bot_state.json`
8. **Review AI decisions**: Check chat log to understand bot's reasoning
9. **Maintain adequate margin**: Keep account balance above minimum for all positions
10. **Have a manual override plan**: Know how to manually close positions if needed

### Emergency Procedures

**If bot crashes or becomes unresponsive:**

1. Manually check open positions on Hyperliquid
2. Review last known state in `./data/bot_state.json`
3. Manually set stop-losses on exchange if needed
4. Restart bot after investigating crash cause

**If kill-switch activates:**

1. Bot will stop opening new positions
2. Existing positions remain open (not auto-closed)
3. Review account state and market conditions
4. Manually close positions if needed
5. Reset kill-switch: requires code modification or manual intervention

**If account is liquidated:**

1. Bot will continue running but fail to place orders
2. Check logs for liquidation events
3. Add funds to account
4. Restart bot with fresh state

## Setup

### Prerequisites

- **Python 3.9 or higher**
- **uv package manager** (recommended) or pip
- **Hyperliquid account** with API credentials
- **DeepSeek API key** from [platform.deepseek.com](https://platform.deepseek.com)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd live_trading_bot
   ```

2. **Install dependencies**

   ```bash
   uv sync
   ```

   Or if using pip:

   ```bash
   pip install -e .
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and fill in your API credentials:

   - `HYPERLIQUID_WALLET_ADDRESS`: Your wallet address
   - `HYPERLIQUID_PRIVATE_KEY`: Your private key (keep secret!)
   - `HYPERLIQUID_TESTNET`: Set to `true` for testnet, `false` for mainnet
   - `DEEPSEEK_API_KEY`: Your DeepSeek API key

   **Important**: Always test on testnet first before using real funds!

### Configuration

Key environment variables in `.env`:

| Variable                   | Description              | Example               |
| -------------------------- | ------------------------ | --------------------- |
| `TRADING_INTERVAL_MINUTES` | How often the bot runs   | `3` (3 minutes)       |
| `MAX_LEVERAGE`             | Maximum leverage allowed | `10` (10x)            |
| `RISK_PER_TRADE_PERCENT`   | Max risk per trade       | `2.0` (2% of account) |
| `MAX_DRAWDOWN_PERCENT`     | Kill-switch threshold    | `20.0` (stop at -20%) |
| `LOG_LEVEL`                | Logging verbosity        | `INFO`                |

4. **Run the bot**:

   ```bash
   uv run python main.py
   ```

   See "Running the Bot" section for detailed instructions.

5. **View dashboard** (optional):

   ```bash
   # CLI dashboard
   python -m src.ui.dashboard --mode cli

   # Web dashboard
   python -m src.ui.dashboard --mode web
   # Open http://localhost:8000 in browser
   ```

   See "Dashboard UI" section for detailed instructions.

## Usage

### Monitoring

Access the dashboard to monitor bot activity:

```bash
# CLI dashboard (to be implemented)
uv run python -m src.ui.dashboard

# Or view logs directly
tail -f data/trades.log
```

### Viewing Trade History

Trade logs are stored in SQLite database:

```bash
sqlite3 data/trades.db "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10;"
```

## Development

### Adding New Strategies

1. Create a new file in `src/strategies/`
2. Implement the `generate_signal()` method
3. Register your strategy in the main trading loop

Example:

```python
# src/strategies/my_strategy.py
class MyStrategy:
    def generate_signal(self, market_data):
        # Your logic here
        return {"action": "BUY", "confidence": 0.8}
```

### Managing Dependencies

Add new dependencies using uv:

```bash
uv add <package-name>
```

This automatically updates `pyproject.toml` and `uv.lock`.

### Running Tests

```bash
# Tests to be implemented
uv run pytest
```

## Safety and Disclaimers

**� WARNING: Trading cryptocurrency is highly risky**

- **Start with testnet**: Always test thoroughly on Hyperliquid testnet before using real funds
- **Use small amounts**: When going live, start with small position sizes
- **Monitor closely**: Regularly check bot performance and logs
- **No guarantees**: This bot is provided as-is with no guarantees of profit
- **Your responsibility**: You are solely responsible for any losses incurred

**Recommended precautions:**

- Set conservative risk limits (`RISK_PER_TRADE_PERCENT` d 2%)
- Use the `MAX_DRAWDOWN_PERCENT` kill-switch
- Keep `MAX_LEVERAGE` low (d 5x for beginners)
- Review AI decisions regularly to ensure they make sense
- Keep your private keys secure and never commit them to version control

## License

[To be determined - specify your license here, e.g., MIT, Apache 2.0, or proprietary]

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description of changes

## Support

For issues or questions:

- Open an issue on GitHub
- Check Hyperliquid docs: [hyperliquid.gitbook.io](https://hyperliquid.gitbook.io)
- Review DeepSeek docs: [platform.deepseek.com/docs](https://platform.deepseek.com/docs)
