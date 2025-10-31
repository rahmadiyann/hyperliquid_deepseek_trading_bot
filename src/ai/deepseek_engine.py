"""DeepSeek AI engine for generating trading decisions."""

import os
import json
import logging
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import openai
from hyperliquid.info import Info
from hyperliquid.utils import constants
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import requests.exceptions
import ta.momentum
import ta.trend
import ta.volatility


class DeepSeekEngine:
    """AI-powered trading decision engine using DeepSeek LLM.

    Integrates market data, technical analysis, and LLM reasoning to generate
    trading decisions with profit targets, stop losses, and confidence scores.
    """

    def __init__(self):
        """Initialize DeepSeek engine with API credentials and clients.

        Loads environment variables and sets up OpenAI client (DeepSeek-compatible)
        and Hyperliquid Info client for historical data.
        """
        # Load environment variables
        load_dotenv()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Read DeepSeek API key
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Missing required environment variable: DEEPSEEK_API_KEY")

        # Initialize OpenAI client with DeepSeek base URL
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

        # Initialize Hyperliquid Info client for historical data
        testnet = os.getenv("HYPERLIQUID_TESTNET", "true").lower() == "true"
        base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self.info = Info(base_url, skip_ws=True)

        # Initialize tracking variables
        self.invocation_count = 0
        self.start_time = datetime.now()

        self.logger.info("DeepSeek engine initialized successfully")

    def generate_trading_decision(
        self,
        market_data: dict,
        positions: list[dict],
        account_info: dict,
    ) -> list[dict]:
        """Generate AI-driven trading decisions for multiple symbols.

        Args:
            market_data: Dict mapping symbols to their market data:
                {
                    "BTC": {
                        "candles_3m": [...],
                        "candles_4h": [...],
                        "indicators_3m": {...},
                        "indicators_4h": {...},
                        "open_interest": float,
                        "funding_rate": float,
                    },
                    ...
                }
            positions: List of current positions (from HyperliquidClient.get_open_positions())
            account_info: Account state (from HyperliquidClient.get_user_state())

        Returns:
            List of trading decisions:
            [
                {
                    "coin": str,
                    "signal": str,  # "buy", "sell", or "hold"
                    "quantity": float,
                    "confidence": float,  # 0-1
                    "reasoning": str,
                    "profit_target": float,
                    "stop_loss": float,
                    "invalidation_condition": str,
                    "leverage": int,
                    "risk_usd": float,
                },
                ...
            ]
        """
        try:
            # Extract symbols from market data
            symbols = list(market_data.keys())

            # Increment invocation counter
            self.invocation_count += 1
            self.logger.info(
                f"Generating trading decisions (invocation #{self.invocation_count}) "
                f"for symbols: {symbols}"
            )

            # Construct prompt
            prompt = self._construct_prompt(symbols, market_data, positions, account_info)

            # Call DeepSeek API
            response_json = self._call_deepseek_api(prompt)

            # Parse and validate response
            decisions = self._parse_and_validate_response(response_json, symbols)

            self.logger.info(f"Generated {len(decisions)} trading decisions")
            for decision in decisions:
                self.logger.info(
                    f"  {decision['coin']}: {decision['signal']} "
                    f"(confidence: {decision['confidence']:.2f})"
                )

            return decisions

        except Exception as e:
            self.logger.error(f"Error generating trading decisions: {str(e)}")
            raise

    def _fetch_historical_candles(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> list[dict]:
        """Fetch historical OHLCV candles for a symbol.

        Args:
            symbol: Coin symbol (e.g., "BTC")
            interval: Candle interval (e.g., "3m", "1h", "4h")
            limit: Number of candles to fetch

        Returns:
            List of candle dicts:
            [
                {
                    "t": timestamp_ms,
                    "o": open,
                    "h": high,
                    "l": low,
                    "c": close,
                    "v": volume,
                },
                ...
            ]
        """
        try:
            # Calculate time range
            # Map interval to milliseconds
            interval_ms_map = {
                "1m": 60 * 1000,
                "3m": 3 * 60 * 1000,
                "5m": 5 * 60 * 1000,
                "15m": 15 * 60 * 1000,
                "1h": 60 * 60 * 1000,
                "4h": 4 * 60 * 60 * 1000,
                "1d": 24 * 60 * 60 * 1000,
            }

            interval_ms = interval_ms_map.get(interval)
            if not interval_ms:
                self.logger.error(f"Invalid interval: {interval}")
                return []

            # Calculate start and end times
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = end_time - (limit * interval_ms)

            # Fetch candles using Hyperliquid SDK
            candles = self.info.candle_snapshot(
                coin=symbol,
                interval=interval,
                startTime=start_time,
                endTime=end_time,
            )

            self.logger.info(f"Fetched {len(candles)} candles for {symbol} ({interval})")
            return candles

        except Exception as e:
            self.logger.warning(
                f"Error fetching candles for {symbol} ({interval}): {str(e)}"
            )
            return []

    def calculate_indicators(self, candles: list[dict]) -> dict:
        """Calculate technical indicators from candle data.

        Public method for computing technical indicators from OHLCV candles.

        Args:
            candles: List of OHLCV candles

        Returns:
            Dict containing indicator series:
            {
                "prices": [...],
                "ema20": [...],
                "ema50": [...],
                "macd": [...],
                "rsi7": [...],
                "rsi14": [...],
                "atr3": [...],
                "atr14": [...],
                "volume": [...],
            }
        """
        try:
            if not candles or len(candles) < 50:
                self.logger.warning(
                    f"Insufficient candles for indicator calculation: {len(candles) if candles else 0}"
                )
                return {}

            # Convert to DataFrame
            df = pd.DataFrame(candles)

            # Rename columns to standard names
            df = df.rename(columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            })

            # Convert to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Set timestamp as index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")

            # Calculate indicators
            close = df["close"]
            high = df["high"]
            low = df["low"]

            # RSI (7 and 14 period)
            rsi7 = ta.momentum.RSIIndicator(close=close, window=7).rsi()
            rsi14 = ta.momentum.RSIIndicator(close=close, window=14).rsi()

            # MACD
            macd = ta.trend.MACD(
                close=close,
                window_slow=26,
                window_fast=12,
                window_sign=9,
            ).macd()

            # EMA (20 and 50 period)
            ema20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
            ema50 = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()

            # ATR (3 and 14 period)
            atr3 = ta.volatility.AverageTrueRange(
                high=high,
                low=low,
                close=close,
                window=3,
            ).average_true_range()

            atr14 = ta.volatility.AverageTrueRange(
                high=high,
                low=low,
                close=close,
                window=14,
            ).average_true_range()

            # Return indicators (drop NaN values)
            result = {
                "prices": close.dropna().tolist(),
                "ema20": ema20.dropna().tolist(),
                "ema50": ema50.dropna().tolist(),
                "macd": macd.dropna().tolist(),
                "rsi7": rsi7.dropna().tolist(),
                "rsi14": rsi14.dropna().tolist(),
                "atr3": atr3.dropna().tolist(),
                "atr14": atr14.dropna().tolist(),
                "volume": df["volume"].dropna().tolist(),
            }

            return result

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return {}

    def _construct_prompt(
        self,
        symbols: list[str],
        market_data: dict,
        positions: list[dict],
        account_info: dict,
    ) -> str:
        """Construct detailed prompt for DeepSeek API.

        Args:
            symbols: List of symbols to analyze
            market_data: Dict of market data and indicators per symbol
            positions: Current positions
            account_info: Account state

        Returns:
            Formatted prompt string
        """
        # Calculate elapsed time
        elapsed_minutes = (datetime.now() - self.start_time).total_seconds() / 60

        # Start building prompt
        lines = []

        # Header
        lines.append(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"Invocation #{self.invocation_count} (running for {elapsed_minutes:.1f} minutes)")
        lines.append("")
        lines.append("ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST")
        lines.append("Intraday series are provided at 3-minute intervals")
        lines.append("")

        # Per-symbol data
        for symbol in symbols:
            data = market_data.get(symbol, {})
            indicators_3m = data.get("indicators_3m", {})
            indicators_4h = data.get("indicators_4h", {})
            open_interest = data.get("open_interest", 0)
            funding_rate = data.get("funding_rate", 0)

            lines.append(f"=" * 60)
            lines.append(f"ALL {symbol} DATA")
            lines.append(f"=" * 60)
            lines.append("")

            # Current state
            prices = indicators_3m.get("prices", []) if indicators_3m else []
            ema20 = indicators_3m.get("ema20", []) if indicators_3m else []
            macd = indicators_3m.get("macd", []) if indicators_3m else []
            rsi7 = indicators_3m.get("rsi7", []) if indicators_3m else []

            if prices and len(prices) > 0:
                lines.append(f"Current State:")
                lines.append(f"  Price: {prices[-1]:.2f}")
                if ema20 and len(ema20) > 0:
                    lines.append(f"  EMA(20): {ema20[-1]:.2f}")
                if macd and len(macd) > 0:
                    lines.append(f"  MACD: {macd[-1]:.4f}")
                if rsi7 and len(rsi7) > 0:
                    lines.append(f"  RSI(7): {rsi7[-1]:.2f}")
                if open_interest > 0:
                    lines.append(f"  Open Interest: {open_interest:,.2f}")
                if funding_rate != 0:
                    lines.append(f"  Funding Rate: {funding_rate:.6%}")
                lines.append("")
            else:
                lines.append(f"Current State: Insufficient data for {symbol}")
                lines.append("")

            # Intraday series (last N points, up to 10)
            prices_full = indicators_3m.get("prices", []) if indicators_3m else []
            if len(prices_full) >= 3:
                # Dynamically adjust number of points to show
                num_points = min(10, len(prices_full))
                lines.append(f"Intraday Series (3-minute intervals, last {num_points} points):")

                # Get last N values
                prices_slice = prices_full[-num_points:]
                ema20_full = indicators_3m.get("ema20", [])
                macd_full = indicators_3m.get("macd", [])
                rsi7_full = indicators_3m.get("rsi7", [])
                rsi14_full = indicators_3m.get("rsi14", [])

                lines.append(f"  Prices: {[f'{p:.2f}' for p in prices_slice]}")
                if ema20_full and len(ema20_full) >= 3:
                    ema20_slice = ema20_full[-num_points:]
                    lines.append(f"  EMA(20): {[f'{e:.2f}' for e in ema20_slice]}")
                if macd_full and len(macd_full) >= 3:
                    macd_slice = macd_full[-num_points:]
                    lines.append(f"  MACD: {[f'{m:.4f}' for m in macd_slice]}")
                if rsi7_full and len(rsi7_full) >= 3:
                    rsi7_slice = rsi7_full[-num_points:]
                    lines.append(f"  RSI(7): {[f'{r:.2f}' for r in rsi7_slice]}")
                if rsi14_full and len(rsi14_full) >= 3:
                    rsi14_slice = rsi14_full[-num_points:]
                    lines.append(f"  RSI(14): {[f'{r:.2f}' for r in rsi14_slice]}")
                lines.append("")
            else:
                lines.append("Intraday Series: Insufficient data (need at least 3 points)")
                lines.append("")

            # Longer-term context (4-hour)
            ema20_4h_full = indicators_4h.get("ema20", []) if indicators_4h else []
            if len(ema20_4h_full) >= 3:
                # Dynamically adjust number of points to show
                num_points_4h = min(10, len(ema20_4h_full))
                lines.append(f"Longer-term Context (4-hour timeframe, last {num_points_4h} points):")

                ema50_4h_full = indicators_4h.get("ema50", [])
                atr3_4h_full = indicators_4h.get("atr3", [])
                atr14_4h_full = indicators_4h.get("atr14", [])
                macd_4h_full = indicators_4h.get("macd", [])
                rsi14_4h_full = indicators_4h.get("rsi14", [])
                volume_4h_full = indicators_4h.get("volume", [])

                if ema20_4h_full and len(ema20_4h_full) >= 3:
                    ema20_4h_slice = ema20_4h_full[-num_points_4h:]
                    lines.append(f"  EMA(20): {[f'{e:.2f}' for e in ema20_4h_slice]}")
                if ema50_4h_full and len(ema50_4h_full) >= 3:
                    ema50_4h_slice = ema50_4h_full[-num_points_4h:]
                    lines.append(f"  EMA(50): {[f'{e:.2f}' for e in ema50_4h_slice]}")
                if atr3_4h_full and len(atr3_4h_full) >= 3:
                    atr3_4h_slice = atr3_4h_full[-num_points_4h:]
                    lines.append(f"  ATR(3): {[f'{a:.2f}' for a in atr3_4h_slice]}")
                if atr14_4h_full and len(atr14_4h_full) >= 3:
                    atr14_4h_slice = atr14_4h_full[-num_points_4h:]
                    lines.append(f"  ATR(14): {[f'{a:.2f}' for a in atr14_4h_slice]}")
                if macd_4h_full and len(macd_4h_full) >= 3:
                    macd_4h_slice = macd_4h_full[-num_points_4h:]
                    lines.append(f"  MACD: {[f'{m:.4f}' for m in macd_4h_slice]}")
                if rsi14_4h_full and len(rsi14_4h_full) >= 3:
                    rsi14_4h_slice = rsi14_4h_full[-num_points_4h:]
                    lines.append(f"  RSI(14): {[f'{r:.2f}' for r in rsi14_4h_slice]}")

                # Average volume
                if volume_4h_full and len(volume_4h_full) >= 3:
                    volume_4h_slice = volume_4h_full[-num_points_4h:]
                    avg_volume = sum(volume_4h_slice) / len(volume_4h_slice)
                    current_volume = volume_4h_slice[-1]
                    lines.append(f"  Current Volume: {current_volume:.2f} (avg: {avg_volume:.2f})")

                lines.append("")
            else:
                lines.append("Longer-term Context: Insufficient data (need at least 3 points)")
                lines.append("")

        # Account information
        lines.append(f"=" * 60)
        lines.append("ACCOUNT INFORMATION")
        lines.append(f"=" * 60)
        lines.append("")

        # Extract account details
        margin_summary = account_info.get("marginSummary", {})
        account_value = float(margin_summary.get("accountValue", 0))
        total_margin_used = float(margin_summary.get("totalMarginUsed", 0))
        available_cash = account_value - total_margin_used

        lines.append(f"Account Value: ${account_value:,.2f}")
        lines.append(f"Available Cash: ${available_cash:,.2f}")
        lines.append(f"Margin Used: ${total_margin_used:,.2f}")
        lines.append("")

        # Current positions
        if positions:
            lines.append("Current Positions:")
            for pos in positions:
                symbol = pos.get("symbol", "")
                size = pos.get("size", 0)
                entry_price = pos.get("entry_price", 0)
                current_price = pos.get("current_price", 0)
                liquidation_price = pos.get("liquidation_price", 0)
                unrealized_pnl = pos.get("unrealized_pnl", 0)
                leverage = pos.get("leverage", 1)

                lines.append(f"  {symbol}:")
                lines.append(f"    Quantity: {size}")
                lines.append(f"    Entry Price: ${entry_price:.2f}")
                lines.append(f"    Current Price: ${current_price:.2f}")
                lines.append(f"    Liquidation Price: ${liquidation_price:.2f}")
                lines.append(f"    Unrealized P&L: ${unrealized_pnl:.2f}")
                lines.append(f"    Leverage: {leverage}x")
                lines.append("")
        else:
            lines.append("No current positions")
            lines.append("")

        # Instructions
        lines.append(f"=" * 60)
        lines.append("INSTRUCTIONS")
        lines.append(f"=" * 60)
        lines.append("")
        lines.append("Based on the above data, provide trading decisions for each coin in JSON format.")
        lines.append("")
        lines.append("Return a JSON object with a 'decisions' key containing an array of decision objects.")
        lines.append("Each decision object must have:")
        lines.append("  - coin (string): The coin symbol")
        lines.append("  - signal (string): \"buy\", \"sell\", or \"hold\"")
        lines.append("  - quantity (float): Amount to trade (0 for hold)")
        lines.append("  - confidence (float): Confidence level 0-1")
        lines.append("  - reasoning (string): Detailed explanation of the decision")
        lines.append("  - profit_target (float): Target price for taking profit")
        lines.append("  - stop_loss (float): Stop loss price")
        lines.append("  - invalidation_condition (string): Condition that invalidates the thesis")
        lines.append("  - leverage (int): Recommended leverage (1-20)")
        lines.append("  - risk_usd (float): Dollar amount at risk")
        lines.append("")
        lines.append("Example:")
        lines.append('{"decisions": [{"coin": "BTC", "signal": "buy", "quantity": 0.1, "confidence": 0.75, "reasoning": "Strong bullish momentum with RSI oversold bounce", "profit_target": 115000, "stop_loss": 105000, "invalidation_condition": "If price closes below 104000", "leverage": 10, "risk_usd": 500}]}')

        return "\n".join(lines)

    @retry(
        retry=retry_if_exception_type((
            requests.exceptions.RequestException,
            openai.APIError,
            openai.RateLimitError,
            openai.APIConnectionError,
        )),
        wait=wait_random_exponential(multiplier=0.5, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    def _call_deepseek_api(self, prompt: str) -> dict:
        """Call DeepSeek API with retry logic.

        Args:
            prompt: The prompt to send to the API

        Returns:
            Parsed JSON response from the API
        """
        try:
            self.logger.info(f"Calling DeepSeek API (prompt length: {len(prompt)} chars)")

            # Call API
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert crypto trading AI. Analyze market data and provide trading decisions in JSON format.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=8000,
            )

            # Extract content
            content = response.choices[0].message.content

            # Log token usage
            if hasattr(response, "usage"):
                self.logger.info(
                    f"API call successful - Tokens: "
                    f"prompt={response.usage.prompt_tokens}, "
                    f"completion={response.usage.completion_tokens}, "
                    f"total={response.usage.total_tokens}"
                )

            # Parse JSON
            response_json = json.loads(content)

            return response_json

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {str(e)}")
            self.logger.error(f"Response content: {content[:500]}")
            raise
        except Exception as e:
            self.logger.error(f"Error calling DeepSeek API: {str(e)}")
            raise

    def _parse_and_validate_response(
        self,
        response_json: dict,
        symbols: list[str],
    ) -> list[dict]:
        """Parse and validate API response.

        Args:
            response_json: Raw JSON response from API
            symbols: Expected symbols for filtering

        Returns:
            List of validated trading decisions
        """
        try:
            # Extract decisions list
            decisions = None

            # Check if response is dict with "decisions" key (primary format)
            if isinstance(response_json, dict) and "decisions" in response_json:
                if isinstance(response_json["decisions"], list):
                    decisions = response_json["decisions"]

            # Fallback: check if response is already a list (backward compatibility)
            elif isinstance(response_json, list):
                decisions = response_json

            # Fallback: check other common keys
            elif isinstance(response_json, dict):
                for key in ["trades", "signals", "data", "results"]:
                    if key in response_json and isinstance(response_json[key], list):
                        decisions = response_json[key]
                        break

                # If still not found, check if there's only one key
                if decisions is None and len(response_json) == 1:
                    value = list(response_json.values())[0]
                    if isinstance(value, list):
                        decisions = value

            if decisions is None:
                self.logger.error(f"Could not extract decisions list from response: {response_json}")
                return []

            # Validate each decision
            validated_decisions = []
            for decision in decisions:
                try:
                    # Check required fields
                    required_fields = [
                        "coin",
                        "signal",
                        "quantity",
                        "confidence",
                        "reasoning",
                        "profit_target",
                        "stop_loss",
                    ]

                    for field in required_fields:
                        if field not in decision:
                            self.logger.warning(f"Missing required field '{field}' in decision: {decision}")
                            raise ValueError(f"Missing field: {field}")

                    # Validate signal
                    signal = decision["signal"].lower()
                    if signal not in ["buy", "sell", "hold"]:
                        self.logger.warning(f"Invalid signal '{signal}' for {decision['coin']}")
                        continue

                    # Validate confidence
                    confidence = float(decision["confidence"])
                    if not 0 <= confidence <= 1:
                        self.logger.warning(f"Invalid confidence {confidence} for {decision['coin']}")
                        confidence = max(0, min(1, confidence))  # Clamp to [0, 1]

                    # Validate quantity
                    quantity = float(decision["quantity"])
                    if quantity < 0:
                        self.logger.warning(f"Negative quantity {quantity} for {decision['coin']}")
                        quantity = 0

                    # Add defaults for optional fields
                    validated_decision = {
                        "coin": decision["coin"],
                        "signal": signal,
                        "quantity": quantity,
                        "confidence": confidence,
                        "reasoning": decision["reasoning"],
                        "profit_target": float(decision["profit_target"]),
                        "stop_loss": float(decision["stop_loss"]),
                        "invalidation_condition": decision.get("invalidation_condition", ""),
                        "leverage": int(decision.get("leverage", 1)),
                        "risk_usd": float(decision.get("risk_usd", 0)),
                    }

                    validated_decisions.append(validated_decision)

                except (ValueError, TypeError, KeyError) as e:
                    self.logger.warning(f"Failed to validate decision: {str(e)}")
                    continue

            self.logger.info(f"Validated {len(validated_decisions)}/{len(decisions)} decisions")

            # Normalize symbols for case-insensitive comparison
            normalized_symbols = {s.upper() for s in symbols}

            # Filter decisions to only requested symbols
            filtered_decisions = []
            seen_symbols = set()

            for decision in validated_decisions:
                coin_upper = decision["coin"].upper()
                if coin_upper in normalized_symbols:
                    # Normalize the coin symbol in the decision
                    decision["coin"] = coin_upper
                    filtered_decisions.append(decision)
                    seen_symbols.add(coin_upper)
                else:
                    self.logger.warning(
                        f"Filtering out decision for unrequested symbol: {decision['coin']}"
                    )

            # Add default "hold" decisions for missing symbols
            missing_symbols = normalized_symbols - seen_symbols
            for symbol in missing_symbols:
                self.logger.warning(f"Model did not provide decision for {symbol}, adding default 'hold'")
                filtered_decisions.append({
                    "coin": symbol,
                    "signal": "hold",
                    "quantity": 0,
                    "confidence": 0.5,
                    "reasoning": "No decision provided by model, defaulting to hold",
                    "profit_target": 0,
                    "stop_loss": 0,
                    "invalidation_condition": "",
                    "leverage": 1,
                    "risk_usd": 0,
                })

            self.logger.info(
                f"Returning {len(filtered_decisions)} decisions "
                f"(filtered from {len(validated_decisions)})"
            )
            return filtered_decisions

        except Exception as e:
            self.logger.error(f"Error parsing response: {str(e)}")
            return []
