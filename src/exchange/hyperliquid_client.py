"""Hyperliquid exchange client wrapper with retry logic and error handling."""

import os
import logging
from typing import Optional, Any
import eth_account
from dotenv import load_dotenv
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
    before_sleep_log,
)
import requests.exceptions


class HyperliquidClient:
    """Wrapper for Hyperliquid SDK with authentication and retry logic.

    Handles:
    - Environment variable loading and authentication
    - Automatic retries for transient errors (network, 429, 5xx)
    - Structured error responses
    - Logging of all operations
    """

    def __init__(self):
        """Initialize Hyperliquid client with credentials from environment.

        Loads .env file and creates authenticated Info and Exchange clients.
        """
        # Load environment variables
        load_dotenv()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Read credentials from environment
        wallet_address = os.getenv("HYPERLIQUID_WALLET_ADDRESS")
        private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
        testnet = os.getenv("HYPERLIQUID_TESTNET", "true").lower() == "true"

        if not wallet_address or not private_key:
            raise ValueError(
                "Missing required environment variables: "
                "HYPERLIQUID_WALLET_ADDRESS and HYPERLIQUID_PRIVATE_KEY"
            )

        # Determine base URL
        base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self.logger.info(f"Initializing Hyperliquid client (testnet={testnet})")

        # Initialize Info client (public data queries)
        self.info = Info(base_url, skip_ws=True)

        # Create signing wallet
        self.wallet = eth_account.Account.from_key(private_key)

        # Initialize Exchange client (authenticated trading)
        self.exchange = Exchange(self.wallet, base_url)

        # Store wallet address for queries
        self.address = wallet_address

        self.logger.info(f"Hyperliquid client initialized for address {wallet_address[:8]}...")

    @staticmethod
    def _is_transient_error(exception: Exception) -> bool:
        """Check if an error is retriable (network/rate limit/server error).

        Args:
            exception: The exception to check

        Returns:
            True if the error should be retried, False otherwise
        """
        # Network errors are always retriable
        if isinstance(exception, (
            requests.exceptions.RequestException,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        )):
            return True

        # Check error message for rate limiting and server errors
        error_msg = str(exception).lower()
        if any(code in error_msg for code in ["429", "500", "502", "503", "504"]):
            return True

        # Deterministic trading errors are not retriable
        deterministic_errors = [
            "insufficient margin",
            "invalid tick",
            "invalid size",
            "position does not exist",
            "order does not exist",
            "price too far from oracle",
        ]
        if any(err in error_msg for err in deterministic_errors):
            return False

        # Default to not retrying unknown errors
        return False

    @staticmethod
    def _parse_order_response(response: dict) -> dict:
        """Parse SDK order response to extract order ID and status.

        Args:
            response: Raw response from Exchange.order()

        Returns:
            Structured dict with success, oid, status, and error
        """
        try:
            # Check if response indicates success
            if "response" in response and "data" in response["response"]:
                data = response["response"]["data"]

                # Extract order status
                if "statuses" in data and len(data["statuses"]) > 0:
                    status = data["statuses"][0]

                    # Check if order is resting (successfully placed)
                    if "resting" in status:
                        oid = status["resting"].get("oid")
                        return {
                            "success": True,
                            "oid": oid,
                            "status": "resting",
                            "error": None,
                            "raw_response": response,
                        }

                    # Check if order filled immediately
                    if "filled" in status:
                        return {
                            "success": True,
                            "oid": None,
                            "status": "filled",
                            "error": None,
                            "raw_response": response,
                        }

                    # Check for error status
                    if "error" in status:
                        return {
                            "success": False,
                            "oid": None,
                            "status": "error",
                            "error": status["error"],
                            "raw_response": response,
                        }

            # Check for top-level error
            if "error" in response:
                return {
                    "success": False,
                    "oid": None,
                    "status": "error",
                    "error": response["error"],
                    "raw_response": response,
                }

            # Unknown response format
            return {
                "success": False,
                "oid": None,
                "status": "unknown",
                "error": "Unknown response format",
                "raw_response": response,
            }

        except (KeyError, IndexError, TypeError) as e:
            return {
                "success": False,
                "oid": None,
                "status": "parse_error",
                "error": f"Failed to parse response: {str(e)}",
                "raw_response": response,
            }

    @staticmethod
    def _parse_cancel_response(response: dict) -> dict:
        """Parse SDK cancel response to extract status.

        Args:
            response: Raw response from Exchange.cancel()

        Returns:
            Structured dict with success, oid, status, and error
        """
        try:
            # Check for response.response.data.statuses structure
            if "response" in response and "data" in response["response"]:
                data = response["response"]["data"]

                if "statuses" in data and len(data["statuses"]) > 0:
                    status_entry = data["statuses"][0]

                    # Check if status is the string "success"
                    if status_entry == "success":
                        return {
                            "success": True,
                            "status": "cancelled",
                            "error": None,
                        }

                    # Check if status is a dict with error
                    if isinstance(status_entry, dict) and "error" in status_entry:
                        return {
                            "success": False,
                            "status": "error",
                            "error": status_entry["error"],
                        }

            # Check for top-level status (ok/err)
            top_status = response.get("status")
            if top_status == "ok":
                return {
                    "success": True,
                    "status": "cancelled",
                    "error": None,
                }
            elif top_status == "err":
                error_msg = response.get("error", "Unknown error")
                return {
                    "success": False,
                    "status": "error",
                    "error": error_msg,
                }

            # Check for top-level error
            if "error" in response:
                return {
                    "success": False,
                    "status": "error",
                    "error": response["error"],
                }

            # Unknown response
            return {
                "success": False,
                "status": "unknown",
                "error": "Unknown response format",
            }

        except (KeyError, TypeError) as e:
            return {
                "success": False,
                "status": "parse_error",
                "error": f"Failed to parse response: {str(e)}",
            }

    @retry(
        retry=retry_if_exception(_is_transient_error.__func__),
        wait=wait_random_exponential(multiplier=0.5, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    def get_market_data(self, symbols: list[str]) -> dict:
        """Fetch current market data for specified symbols.

        Args:
            symbols: List of coin symbols (e.g., ["BTC", "ETH"])

        Returns:
            Dict mapping symbols to market data:
            {
                "BTC": {
                    "mid": float,      # Mid price
                    "mark": float,     # Mark price
                    "funding": float,  # Current funding rate
                    "volume_24h": float,  # 24h volume (if available)
                },
                ...
            }
        """
        try:
            self.logger.info(f"Fetching market data for symbols: {symbols}")

            # Fetch mid prices for all markets
            all_mids = self.info.all_mids()

            # Fetch metadata and asset contexts (includes mark, funding, open interest)
            meta_and_ctx = self.info.meta_and_asset_ctxs()

            # Extract universe (list of coin names)
            universe = meta_and_ctx[0]["universe"]
            asset_ctxs = meta_and_ctx[1]

            # Build mapping of coin name to asset context
            coin_to_ctx = {}
            for i, coin_info in enumerate(universe):
                coin = coin_info["name"]
                if i < len(asset_ctxs):
                    coin_to_ctx[coin] = asset_ctxs[i]

            # Build result for requested symbols
            result = {}
            for symbol in symbols:
                # Get mid price
                mid = float(all_mids.get(symbol, 0))

                # Get asset context
                ctx = coin_to_ctx.get(symbol, {})
                mark = float(ctx.get("markPx", mid))  # Use mid as fallback
                funding = float(ctx.get("funding", 0))

                # Volume might not be available in all contexts
                volume_24h = float(ctx.get("dayNtlVlm", 0))

                result[symbol] = {
                    "mid": mid,
                    "mark": mark,
                    "funding": funding,
                    "volume_24h": volume_24h,
                }

            self.logger.info(f"Successfully fetched market data for {len(result)} symbols")
            return result

        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            raise

    @retry(
        retry=retry_if_exception(_is_transient_error.__func__),
        wait=wait_random_exponential(multiplier=0.5, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    def get_user_state(self) -> dict:
        """Fetch current user state including positions and margin info.

        Returns:
            Dict containing:
            - assetPositions: List of current positions
            - marginSummary: Account margin summary
            - crossMarginSummary: Cross margin details
            - withdrawable: Withdrawable balance
            - time: Timestamp
        """
        try:
            self.logger.info(f"Fetching user state for {self.address[:8]}...")

            state = self.info.user_state(self.address)

            self.logger.info("Successfully fetched user state")
            return state

        except Exception as e:
            self.logger.error(f"Error fetching user state: {str(e)}")
            raise

    @retry(
        retry=retry_if_exception(_is_transient_error.__func__),
        wait=wait_random_exponential(multiplier=0.5, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: dict,
    ) -> dict:
        """Place an order on Hyperliquid.

        Args:
            symbol: Coin symbol (e.g., "BTC")
            side: "buy" or "sell"
            quantity: Order size
            price: Limit price
            order_type: Order type dict (e.g., {"limit": {"tif": "Gtc"}})

        Returns:
            Dict with:
            - success: bool
            - oid: Order ID (if success)
            - status: Order status
            - error: Error message (if failed)
            - raw_response: Full SDK response
        """
        try:
            # Convert side to boolean
            is_buy = side.lower() == "buy"

            self.logger.info(
                f"Placing {side} order: {quantity} {symbol} @ {price} "
                f"(type: {order_type})"
            )

            # Place order
            response = self.exchange.order(symbol, is_buy, quantity, price, order_type)

            # Parse response
            result = self._parse_order_response(response)

            if result["success"]:
                self.logger.info(
                    f"Order placed successfully: oid={result['oid']}, "
                    f"status={result['status']}"
                )
            else:
                self.logger.error(
                    f"Order failed: {result['error']}"
                )

            return result

        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            # Check if this is a deterministic error that shouldn't be retried
            if not self._is_transient_error(e):
                return {
                    "success": False,
                    "oid": None,
                    "status": "error",
                    "error": str(e),
                    "raw_response": None,
                }
            raise

    @retry(
        retry=retry_if_exception(_is_transient_error.__func__),
        wait=wait_random_exponential(multiplier=0.5, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    def cancel_order(self, oid: int) -> dict:
        """Cancel an open order.

        Args:
            oid: Order ID to cancel

        Returns:
            Dict with:
            - success: bool
            - oid: Order ID
            - status: Cancellation status
            - error: Error message (if failed)
        """
        try:
            self.logger.info(f"Cancelling order {oid}")

            # Fetch open orders to resolve the symbol for this oid
            open_orders = self.info.open_orders(self.address)

            # Find the symbol for this order ID
            symbol = None
            for order in open_orders:
                if order.get("oid") == oid:
                    symbol = order.get("coin")
                    break

            if symbol is None:
                self.logger.error(f"Order {oid} not found in open orders")
                return {
                    "success": False,
                    "oid": oid,
                    "status": "error",
                    "error": f"Order {oid} not found in open orders",
                }

            self.logger.info(f"Resolved order {oid} to symbol {symbol}")

            # Cancel order
            response = self.exchange.cancel(symbol, oid)

            # Parse response
            result = self._parse_cancel_response(response)
            result["oid"] = oid

            if result["success"]:
                self.logger.info(f"Order {oid} cancelled successfully")
            else:
                self.logger.error(f"Failed to cancel order {oid}: {result['error']}")

            return result

        except Exception as e:
            self.logger.error(f"Error cancelling order {oid}: {str(e)}")
            # Check if this is a deterministic error that shouldn't be retried
            if not self._is_transient_error(e):
                return {
                    "success": False,
                    "oid": oid,
                    "status": "error",
                    "error": str(e),
                }
            raise

    @retry(
        retry=retry_if_exception(_is_transient_error.__func__),
        wait=wait_random_exponential(multiplier=0.5, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    def get_open_positions(self) -> list[dict]:
        """Get all open positions for the account.

        Returns:
            List of position dicts:
            [
                {
                    "symbol": str,
                    "size": float,              # Positive for long, negative for short
                    "entry_price": float,
                    "current_price": float,     # Mark price
                    "liquidation_price": float,
                    "leverage": int,
                    "margin_type": str,         # "cross" or "isolated"
                    "position_value": float,
                    "unrealized_pnl": float,
                    "margin_used": float,
                },
                ...
            ]
        """
        try:
            self.logger.info("Fetching open positions")

            # Get user state
            state = self.info.user_state(self.address)

            # Fetch metadata and asset contexts to get mark prices
            meta_and_ctx = self.info.meta_and_asset_ctxs()
            universe = meta_and_ctx[0]["universe"]
            asset_ctxs = meta_and_ctx[1]

            # Build mapping of coin to mark price
            coin_to_mark_price = {}
            for i, coin_info in enumerate(universe):
                coin = coin_info["name"]
                if i < len(asset_ctxs):
                    mark_px = float(asset_ctxs[i].get("markPx", 0))
                    coin_to_mark_price[coin] = mark_px

            # Extract positions
            asset_positions = state.get("assetPositions", [])

            positions = []
            for pos in asset_positions:
                # Get position details
                position = pos.get("position", {})
                coin = position.get("coin", "")
                szi = float(position.get("szi", 0))

                # Skip if no position
                if szi == 0:
                    continue

                # Extract position details
                entry_price = float(position.get("entryPx", 0))
                liquidation_price = float(position.get("liquidationPx", 0))
                margin_used = float(position.get("marginUsed", 0))
                unrealized_pnl = float(position.get("unrealizedPnl", 0))

                # Get leverage and margin type
                leverage_info = position.get("leverage", {})
                leverage = int(leverage_info.get("value", 1))
                margin_type = leverage_info.get("type", "cross")

                # Get current (mark) price
                current_price = coin_to_mark_price.get(coin, entry_price)

                # Calculate position value using current price
                position_value = abs(szi * current_price)

                positions.append({
                    "symbol": coin,
                    "size": szi,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "liquidation_price": liquidation_price,
                    "leverage": leverage,
                    "margin_type": margin_type,
                    "position_value": position_value,
                    "unrealized_pnl": unrealized_pnl,
                    "margin_used": margin_used,
                })

            self.logger.info(f"Found {len(positions)} open positions")
            return positions

        except Exception as e:
            self.logger.error(f"Error fetching open positions: {str(e)}")
            raise

    @retry(
        retry=retry_if_exception(_is_transient_error.__func__),
        wait=wait_random_exponential(multiplier=0.5, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    def get_candles(
        self,
        symbol: str,
        interval: str,
        limit: int = 100
    ) -> list[dict]:
        """Fetch historical candles for a symbol.

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
            self.logger.info(f"Fetching {limit} candles for {symbol} ({interval})")

            # Call SDK candle_snapshot
            candles = self.info.candle_snapshot(
                coin=symbol,
                interval=interval,
                limit=limit
            )

            self.logger.info(f"Successfully fetched {len(candles)} candles for {symbol}")
            return candles

        except Exception as e:
            self.logger.error(f"Error fetching candles for {symbol}: {str(e)}")
            raise

    @retry(
        retry=retry_if_exception(_is_transient_error.__func__),
        wait=wait_random_exponential(multiplier=0.5, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    def get_funding_rates(self, symbols: list[str]) -> dict:
        """Get current funding rates for specified symbols.

        Args:
            symbols: List of coin symbols (e.g., ["BTC", "ETH"])

        Returns:
            Dict mapping symbols to funding rates (as hourly fraction):
            {
                "BTC": 0.0000125,  # 0.00125% per hour
                "ETH": -0.0000050,
                ...
            }
        """
        try:
            self.logger.info(f"Fetching funding rates for symbols: {symbols}")

            # Fetch metadata and asset contexts
            meta_and_ctx = self.info.meta_and_asset_ctxs()

            # Extract universe and asset contexts
            universe = meta_and_ctx[0]["universe"]
            asset_ctxs = meta_and_ctx[1]

            # Build mapping of coin name to funding rate
            funding_rates = {}
            for i, coin_info in enumerate(universe):
                coin = coin_info["name"]
                if coin in symbols and i < len(asset_ctxs):
                    funding = float(asset_ctxs[i].get("funding", 0))
                    funding_rates[coin] = funding

            # Add zeros for symbols not found
            for symbol in symbols:
                if symbol not in funding_rates:
                    funding_rates[symbol] = 0.0

            self.logger.info(f"Successfully fetched funding rates for {len(funding_rates)} symbols")
            return funding_rates

        except Exception as e:
            self.logger.error(f"Error fetching funding rates: {str(e)}")
            raise
