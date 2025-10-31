"""Risk management module for position sizing, leverage control, and drawdown protection."""

import os
import logging
from typing import Optional, Dict, List
from dotenv import load_dotenv


class RiskManager:
    """Comprehensive risk management for trading bot.

    Provides:
    - Position sizing based on account risk parameters
    - Leverage validation and enforcement
    - Automatic stop-loss monitoring and execution
    - Drawdown kill-switch protection
    """

    def __init__(self, hyperliquid_client):
        """Initialize risk manager with Hyperliquid client and configuration.

        Args:
            hyperliquid_client: Instance of HyperliquidClient for executing trades
        """
        # Load environment variables
        load_dotenv()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Store client reference
        self.client = hyperliquid_client

        # Load risk parameters from environment
        self.max_leverage = float(os.getenv("MAX_LEVERAGE", "10"))
        self.risk_per_trade_percent = float(os.getenv("RISK_PER_TRADE_PERCENT", "2.0"))
        self.max_drawdown_percent = float(os.getenv("MAX_DRAWDOWN_PERCENT", "20.0"))
        self.max_total_exposure_percent = float(os.getenv("MAX_TOTAL_EXPOSURE_PERCENT", "200.0"))
        self.min_position_size_usd = float(os.getenv("MIN_POSITION_SIZE_USD", "10.0"))
        self.max_single_position_percent = float(os.getenv("MAX_SINGLE_POSITION_PERCENT", "50.0"))

        # Validate configuration
        self._validate_configuration()

        # Initialize state tracking
        self.peak_account_value = 0.0
        self.kill_switch_active = False

        self.logger.info(
            f"Risk Manager initialized - "
            f"max_leverage: {self.max_leverage}x, "
            f"risk_per_trade: {self.risk_per_trade_percent}%, "
            f"max_drawdown: {self.max_drawdown_percent}%"
        )

    def _validate_configuration(self):
        """Validate risk configuration parameters."""
        # Validate MAX_LEVERAGE
        if not 1 <= self.max_leverage <= 20:
            self.logger.warning(
                f"MAX_LEVERAGE ({self.max_leverage}) outside recommended range [1-20]"
            )

        # Validate RISK_PER_TRADE_PERCENT
        if not 0.1 <= self.risk_per_trade_percent <= 10:
            self.logger.warning(
                f"RISK_PER_TRADE_PERCENT ({self.risk_per_trade_percent}) "
                f"outside recommended range [0.1-10]"
            )

        # Validate MAX_DRAWDOWN_PERCENT
        if not 5 <= self.max_drawdown_percent <= 50:
            self.logger.warning(
                f"MAX_DRAWDOWN_PERCENT ({self.max_drawdown_percent}) "
                f"outside recommended range [5-50]"
            )

    def calculate_position_size(
        self,
        signal: dict,
        account_value: float,
        risk_per_trade: Optional[float] = None
    ) -> dict:
        """Calculate appropriate position size based on risk parameters.

        Args:
            signal: Trading signal dict with coin, signal, and optionally stop_loss, leverage, risk_usd, entry_price, current_price
            account_value: Current account value
            risk_per_trade: Risk per trade as percentage (e.g., 2.0 for 2%), defaults to self.risk_per_trade_percent

        Returns:
            Dict with position_size, notional_value, leverage_used, risk_amount,
            approved, rejection_reason
        """
        try:
            # Input validation
            if not self._validate_signal(signal):
                return {
                    "position_size": 0,
                    "notional_value": 0,
                    "leverage_used": 0,
                    "risk_amount": 0,
                    "stop_distance_percent": 0,
                    "approved": False,
                    "rejection_reason": "Invalid signal format"
                }

            if account_value <= 0:
                return {
                    "position_size": 0,
                    "notional_value": 0,
                    "leverage_used": 0,
                    "risk_amount": 0,
                    "stop_distance_percent": 0,
                    "approved": False,
                    "rejection_reason": "Invalid account value"
                }

            # Get current_price from signal (try entry_price first, then current_price)
            current_price = signal.get("entry_price", signal.get("current_price", 0))
            if current_price <= 0:
                return {
                    "position_size": 0,
                    "notional_value": 0,
                    "leverage_used": 0,
                    "risk_amount": 0,
                    "stop_distance_percent": 0,
                    "approved": False,
                    "rejection_reason": "Invalid or missing price in signal (need entry_price or current_price)"
                }

            # Check kill-switch
            if self.kill_switch_active:
                return {
                    "position_size": 0,
                    "notional_value": 0,
                    "leverage_used": 0,
                    "risk_amount": 0,
                    "stop_distance_percent": 0,
                    "approved": False,
                    "rejection_reason": "Kill-switch active (drawdown limit exceeded)"
                }

            # Extract stop-loss and calculate distance
            stop_loss = signal.get("stop_loss", 0)
            stop_distance = self._calculate_stop_distance(
                current_price, stop_loss, signal.get("signal", "hold")
            )

            # If stop distance is invalid, use default (5% of price)
            if stop_distance == 0:
                stop_distance = current_price * 0.05
                self.logger.warning(
                    f"Using default stop distance (5%) for {signal.get('coin')}"
                )

            stop_distance_percent = (stop_distance / current_price) * 100

            # Calculate risk amount from provided risk_per_trade or default
            if risk_per_trade is None:
                risk_per_trade = self.risk_per_trade_percent

            risk_amount = account_value * (risk_per_trade / 100)

            # Override with signal's risk_usd if provided and lower
            signal_risk = signal.get("risk_usd", float('inf'))
            if signal_risk < risk_amount:
                risk_amount = signal_risk
                self.logger.debug(
                    f"Using signal risk_usd: ${risk_amount:.2f} "
                    f"(lower than account risk)"
                )

            # Calculate base position size
            position_size = risk_amount / stop_distance

            # Apply leverage
            requested_leverage = signal.get("leverage", 1)
            actual_leverage = min(requested_leverage, self.max_leverage)

            if actual_leverage != requested_leverage:
                self.logger.warning(
                    f"Capping leverage from {requested_leverage}x to {actual_leverage}x "
                    f"for {signal.get('coin')}"
                )

            # Calculate notional value
            notional_value = position_size * current_price

            # Additional checks
            # Check minimum position size
            if notional_value < self.min_position_size_usd:
                return {
                    "position_size": position_size,
                    "notional_value": notional_value,
                    "leverage_used": actual_leverage,
                    "risk_amount": risk_amount,
                    "stop_distance_percent": stop_distance_percent,
                    "approved": False,
                    "rejection_reason": f"Position too small (${notional_value:.2f} < ${self.min_position_size_usd})"
                }

            # Check maximum single position size
            max_single_position = account_value * (self.max_single_position_percent / 100)
            if notional_value > max_single_position:
                return {
                    "position_size": position_size,
                    "notional_value": notional_value,
                    "leverage_used": actual_leverage,
                    "risk_amount": risk_amount,
                    "stop_distance_percent": stop_distance_percent,
                    "approved": False,
                    "rejection_reason": f"Position too large (${notional_value:.2f} > ${max_single_position:.2f}, {self.max_single_position_percent}% limit)"
                }

            # Approved
            self.logger.info(
                f"Position sizing approved for {signal.get('coin')}: "
                f"size={position_size:.4f}, notional=${notional_value:.2f}, "
                f"risk=${risk_amount:.2f}, leverage={actual_leverage}x"
            )

            return {
                "position_size": position_size,
                "notional_value": notional_value,
                "leverage_used": actual_leverage,
                "risk_amount": risk_amount,
                "stop_distance_percent": stop_distance_percent,
                "approved": True,
                "rejection_reason": ""
            }

        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return {
                "position_size": 0,
                "notional_value": 0,
                "leverage_used": 0,
                "risk_amount": 0,
                "stop_distance_percent": 0,
                "approved": False,
                "rejection_reason": f"Calculation error: {str(e)}"
            }

    def check_max_leverage(
        self,
        positions: List[dict]
    ) -> dict:
        """Validate that positions don't exceed leverage limits.

        Args:
            positions: List of position dicts from HyperliquidClient

        Returns:
            Dict with within_limits, total_notional, effective_leverage,
            violations, positions_to_reduce
        """
        try:
            # Fetch account_value from client
            try:
                user_state = self.client.get_user_state()
                account_value = float(user_state.get("marginSummary", {}).get("accountValue", 0))
            except Exception as e:
                self.logger.error(f"Failed to fetch account value: {str(e)}")
                return {
                    "within_limits": False,
                    "total_notional": 0,
                    "effective_leverage": 0,
                    "max_position_leverage": 0,
                    "violations": [f"Failed to fetch account value: {str(e)}"],
                    "positions_to_reduce": []
                }

            if account_value <= 0:
                return {
                    "within_limits": False,
                    "total_notional": 0,
                    "effective_leverage": 0,
                    "max_position_leverage": 0,
                    "violations": ["Invalid account value"],
                    "positions_to_reduce": []
                }

            # Calculate aggregate metrics
            total_notional = sum(abs(pos.get("position_value", 0)) for pos in positions)
            effective_leverage = total_notional / account_value if account_value > 0 else 0
            max_position_leverage = max(
                (pos.get("leverage", 0) for pos in positions),
                default=0
            )

            violations = []
            positions_to_reduce = []

            # Check per-position leverage
            for pos in positions:
                pos_leverage = pos.get("leverage", 0)
                if pos_leverage > self.max_leverage:
                    symbol = pos.get("symbol", "UNKNOWN")
                    violations.append(
                        f"{symbol} leverage {pos_leverage}x exceeds limit {self.max_leverage}x"
                    )
                    positions_to_reduce.append(symbol)

            # Check effective leverage
            if effective_leverage > self.max_leverage:
                violations.append(
                    f"Effective leverage {effective_leverage:.2f}x exceeds limit {self.max_leverage}x"
                )
                # Add all positions for potential reduction
                positions_to_reduce.extend([
                    pos.get("symbol", "UNKNOWN") for pos in positions
                    if pos.get("symbol") not in positions_to_reduce
                ])

            # Check total exposure
            max_total_exposure = account_value * (self.max_total_exposure_percent / 100)
            if total_notional > max_total_exposure:
                violations.append(
                    f"Total exposure ${total_notional:,.2f} exceeds limit "
                    f"${max_total_exposure:,.2f} ({self.max_total_exposure_percent}%)"
                )
                # Add largest positions for reduction
                sorted_positions = sorted(
                    positions,
                    key=lambda p: abs(p.get("position_value", 0)),
                    reverse=True
                )
                for pos in sorted_positions[:3]:  # Top 3 largest
                    symbol = pos.get("symbol", "UNKNOWN")
                    if symbol not in positions_to_reduce:
                        positions_to_reduce.append(symbol)

            within_limits = len(violations) == 0

            if not within_limits:
                self.logger.warning(
                    f"Leverage violations detected: {', '.join(violations)}"
                )

            self.logger.info(
                f"Leverage check: total_notional=${total_notional:,.2f}, "
                f"effective_leverage={effective_leverage:.2f}x, "
                f"within_limits={within_limits}"
            )

            return {
                "within_limits": within_limits,
                "total_notional": total_notional,
                "effective_leverage": effective_leverage,
                "max_position_leverage": max_position_leverage,
                "violations": violations,
                "positions_to_reduce": positions_to_reduce
            }

        except Exception as e:
            self.logger.error(f"Error checking leverage: {str(e)}")
            return {
                "within_limits": False,
                "total_notional": 0,
                "effective_leverage": 0,
                "max_position_leverage": 0,
                "violations": [f"Error: {str(e)}"],
                "positions_to_reduce": []
            }

    # State management for position stops and invalidations
    def set_position_stops(self, position_stops: Dict[str, float]):
        """Set stop-loss prices for positions.

        Args:
            position_stops: Dict mapping symbol to stop-loss price
        """
        if not hasattr(self, '_position_stops'):
            self._position_stops = {}
        self._position_stops.update(position_stops)

    def set_position_invalidations(self, position_invalidations: Dict[str, any]):
        """Set invalidation conditions for positions.

        Args:
            position_invalidations: Dict mapping symbol to invalidation condition data
        """
        if not hasattr(self, '_position_invalidations'):
            self._position_invalidations = {}
        self._position_invalidations.update(position_invalidations)

    def enforce_stop_loss(
        self,
        positions: List[dict],
        current_prices: Dict[str, float],
        position_stops: Dict[str, float] = None
    ) -> dict:
        """Monitor and automatically close positions that hit stop-loss or invalidations.

        Args:
            positions: List of position dicts
            current_prices: Dict mapping symbol to current price
            position_stops: Dict mapping symbol to stop-loss price (optional, uses internal state if not provided)

        Returns:
            Dict with positions_closed, close_attempts, failed_closes, total_realized_pnl
        """
        try:
            # Get stop-loss data from parameter or internal state
            if position_stops is None:
                position_stops = getattr(self, '_position_stops', {})

            # Get invalidation data from internal state
            position_invalidations = getattr(self, '_position_invalidations', {})

            # Get IOC cross percentage from env or use default
            ioc_cross_pct = float(os.getenv("IOC_CROSS_PCT", "0.005"))  # 0.5% default

            close_attempts = []
            failed_closes = []
            positions_closed = 0
            total_realized_pnl = 0.0

            for pos in positions:
                symbol = pos.get("symbol", "")
                size = pos.get("size", 0)
                current_price = current_prices.get(symbol, pos.get("current_price", 0))
                liquidation_price = pos.get("liquidation_price", 0)
                stop_loss = position_stops.get(symbol)

                if size == 0:
                    continue

                # Guard against division by zero
                if current_price <= 0:
                    self.logger.warning(f"Invalid current_price ({current_price}) for {symbol}, skipping")
                    continue

                # Check liquidation proximity (within 5%)
                near_liquidation = False
                if liquidation_price > 0:
                    if self._is_position_long(size):
                        price_to_liq = (current_price - liquidation_price) / current_price
                        near_liquidation = price_to_liq < 0.05
                    else:
                        price_to_liq = (liquidation_price - current_price) / current_price
                        near_liquidation = price_to_liq < 0.05

                # Check stop-loss trigger
                stop_loss_hit = False
                if stop_loss is not None:
                    if self._is_position_long(size) and current_price <= stop_loss:
                        stop_loss_hit = True
                    elif not self._is_position_long(size) and current_price >= stop_loss:
                        stop_loss_hit = True

                # Check invalidation conditions
                invalidation_triggered = False
                invalidation_data = position_invalidations.get(symbol)
                if invalidation_data:
                    # Simple predicate evaluation (can be extended for more complex conditions)
                    invalidation_triggered = self._evaluate_invalidation(
                        symbol, invalidation_data, current_price
                    )

                # Determine if we need to close
                should_close = False
                close_reason = ""

                if near_liquidation:
                    should_close = True
                    close_reason = "near_liquidation"
                    self.logger.critical(
                        f"EMERGENCY: {symbol} near liquidation! "
                        f"Current: {current_price}, Liquidation: {liquidation_price}"
                    )
                elif stop_loss_hit:
                    should_close = True
                    close_reason = "stop_loss_hit"
                    self.logger.warning(
                        f"Stop-loss hit for {symbol}: "
                        f"Current: {current_price}, Stop: {stop_loss}"
                    )
                elif invalidation_triggered:
                    should_close = True
                    close_reason = "invalidation"
                    self.logger.warning(
                        f"Invalidation condition triggered for {symbol}: "
                        f"Current: {current_price}"
                    )

                # Execute close
                if should_close:
                    side = self._calculate_close_side(size)
                    quantity = abs(size)

                    # Calculate aggressive IOC price to ensure fill
                    # For closing longs (sell), price below market
                    # For closing shorts (buy), price above market
                    if side == "sell":
                        ioc_price = current_price * (1 - ioc_cross_pct)
                    else:  # buy
                        ioc_price = current_price * (1 + ioc_cross_pct)

                    try:
                        # Place immediate-or-cancel order with aggressive pricing
                        order_response = self.client.place_order(
                            symbol=symbol,
                            side=side,
                            quantity=quantity,
                            price=ioc_price,
                            order_type={"limit": {"tif": "Ioc"}}
                        )

                        success = order_response.get("success", False)
                        order_id = order_response.get("oid")
                        error = order_response.get("error")

                        # If first attempt failed or not filled, try with larger offset
                        if not success:
                            larger_offset = ioc_cross_pct * 2  # Double the offset
                            if side == "sell":
                                ioc_price_retry = current_price * (1 - larger_offset)
                            else:
                                ioc_price_retry = current_price * (1 + larger_offset)

                            self.logger.warning(
                                f"First IOC attempt failed for {symbol}, retrying with larger offset"
                            )

                            order_response = self.client.place_order(
                                symbol=symbol,
                                side=side,
                                quantity=quantity,
                                price=ioc_price_retry,
                                order_type={"limit": {"tif": "Ioc"}}
                            )

                            success = order_response.get("success", False)
                            order_id = order_response.get("oid")
                            error = order_response.get("error")

                        if success:
                            positions_closed += 1
                            unrealized_pnl = pos.get("unrealized_pnl", 0)
                            total_realized_pnl += unrealized_pnl
                            self.logger.info(
                                f"Successfully closed {symbol} position: "
                                f"size={quantity}, P&L=${unrealized_pnl:.2f}"
                            )
                        else:
                            failed_closes.append({
                                "symbol": symbol,
                                "reason": close_reason,
                                "error": error
                            })
                            self.logger.error(
                                f"Failed to close {symbol}: {error}"
                            )

                        close_attempts.append({
                            "symbol": symbol,
                            "reason": close_reason,
                            "trigger_price": current_price,
                            "position_size": quantity,
                            "success": success,
                            "order_id": order_id,
                            "error": error
                        })

                    except Exception as e:
                        self.logger.error(
                            f"Exception closing {symbol}: {str(e)}"
                        )
                        failed_closes.append({
                            "symbol": symbol,
                            "reason": close_reason,
                            "error": str(e)
                        })
                        close_attempts.append({
                            "symbol": symbol,
                            "reason": close_reason,
                            "trigger_price": current_price,
                            "position_size": quantity,
                            "success": False,
                            "order_id": None,
                            "error": str(e)
                        })

            return {
                "positions_closed": positions_closed,
                "close_attempts": close_attempts,
                "failed_closes": failed_closes,
                "total_realized_pnl": total_realized_pnl
            }

        except Exception as e:
            self.logger.error(f"Error enforcing stop-loss: {str(e)}")
            return {
                "positions_closed": 0,
                "close_attempts": [],
                "failed_closes": [],
                "total_realized_pnl": 0
            }

    def check_drawdown_limit(
        self,
        account_value: float,
        peak_value: Optional[float] = None
    ) -> dict:
        """Monitor account drawdown and activate kill-switch if exceeded.

        Args:
            account_value: Current account value
            peak_value: Optional peak account value to override/initialize internal tracking

        Returns:
            Dict with kill_switch, current_drawdown_pct, peak_value, current_value,
            threshold_pct, distance_to_threshold
        """
        try:
            # If peak_value is provided, use it to override/initialize
            if peak_value is not None:
                self.peak_account_value = peak_value
                self.logger.info(f"Peak account value set to: ${peak_value:,.2f}")

            # Update peak value
            if account_value > self.peak_account_value:
                self.peak_account_value = account_value
                self.logger.info(f"New peak account value: ${account_value:,.2f}")

            # Handle first run
            if self.peak_account_value == 0:
                self.peak_account_value = account_value

            # Calculate drawdown
            if self.peak_account_value > 0:
                drawdown = (self.peak_account_value - account_value) / self.peak_account_value
                drawdown_pct = drawdown * 100
            else:
                drawdown_pct = 0

            distance_to_threshold = self.max_drawdown_percent - drawdown_pct

            # Check threshold
            if drawdown_pct >= self.max_drawdown_percent and not self.kill_switch_active:
                self.kill_switch_active = True
                self.logger.critical(
                    f"KILL-SWITCH ACTIVATED! Drawdown {drawdown_pct:.2f}% "
                    f"exceeds threshold {self.max_drawdown_percent:.2f}%. "
                    f"Peak: ${self.peak_account_value:,.2f}, "
                    f"Current: ${account_value:,.2f}"
                )

            # Log status
            if self.kill_switch_active:
                self.logger.warning(
                    f"Kill-switch ACTIVE - Drawdown: {drawdown_pct:.2f}%"
                )
            else:
                self.logger.info(
                    f"Drawdown: {drawdown_pct:.2f}% "
                    f"(threshold: {self.max_drawdown_percent:.2f}%, "
                    f"distance: {distance_to_threshold:.2f}%)"
                )

            return {
                "kill_switch": self.kill_switch_active,
                "current_drawdown_pct": drawdown_pct,
                "peak_value": self.peak_account_value,
                "current_value": account_value,
                "threshold_pct": self.max_drawdown_percent,
                "distance_to_threshold": distance_to_threshold
            }

        except Exception as e:
            self.logger.error(f"Error checking drawdown: {str(e)}")
            # Return safe default (kill-switch active)
            return {
                "kill_switch": True,
                "current_drawdown_pct": 0,
                "peak_value": self.peak_account_value,
                "current_value": account_value,
                "threshold_pct": self.max_drawdown_percent,
                "distance_to_threshold": 0
            }

    def reset_kill_switch(self):
        """Manually reset the kill-switch after review."""
        self.kill_switch_active = False
        self.logger.warning("Kill-switch manually RESET")

    def is_kill_switch_active(self) -> bool:
        """Check if kill-switch is active.

        Returns:
            True if kill-switch is active
        """
        return self.kill_switch_active

    def get_risk_summary(
        self,
        positions: List[dict],
        account_value: float
    ) -> dict:
        """Generate comprehensive risk summary.

        Args:
            positions: List of position dicts
            account_value: Current account value

        Returns:
            Dict with risk metrics
        """
        try:
            # Calculate metrics
            total_positions = len(positions)
            total_notional = sum(abs(pos.get("position_value", 0)) for pos in positions)
            effective_leverage = total_notional / account_value if account_value > 0 else 0
            total_unrealized_pnl = sum(pos.get("unrealized_pnl", 0) for pos in positions)

            # Find largest position
            largest_position = {}
            if positions:
                largest_pos = max(
                    positions,
                    key=lambda p: abs(p.get("position_value", 0))
                )
                largest_position = {
                    "symbol": largest_pos.get("symbol", ""),
                    "notional": abs(largest_pos.get("position_value", 0)),
                    "leverage": largest_pos.get("leverage", 0)
                }

            # Calculate drawdown
            if self.peak_account_value > 0:
                current_drawdown_pct = (
                    (self.peak_account_value - account_value) / self.peak_account_value * 100
                )
            else:
                current_drawdown_pct = 0

            # Calculate available margin
            available_margin = account_value - total_notional

            # Calculate risk utilization
            max_total_exposure = account_value * (self.max_total_exposure_percent / 100)
            risk_utilization_pct = (total_notional / max_total_exposure * 100) if max_total_exposure > 0 else 0

            summary = {
                "total_positions": total_positions,
                "total_notional": total_notional,
                "effective_leverage": effective_leverage,
                "largest_position": largest_position,
                "total_unrealized_pnl": total_unrealized_pnl,
                "current_drawdown_pct": current_drawdown_pct,
                "available_margin": available_margin,
                "risk_utilization_pct": risk_utilization_pct,
                "kill_switch_active": self.kill_switch_active
            }

            self.logger.info(
                f"Risk Summary: positions={total_positions}, "
                f"notional=${total_notional:,.2f}, "
                f"leverage={effective_leverage:.2f}x, "
                f"pnl=${total_unrealized_pnl:,.2f}, "
                f"drawdown={current_drawdown_pct:.2f}%"
            )

            return summary

        except Exception as e:
            self.logger.error(f"Error generating risk summary: {str(e)}")
            return {
                "total_positions": 0,
                "total_notional": 0,
                "effective_leverage": 0,
                "largest_position": {},
                "total_unrealized_pnl": 0,
                "current_drawdown_pct": 0,
                "available_margin": 0,
                "risk_utilization_pct": 0,
                "kill_switch_active": self.kill_switch_active
            }

    # Private helper methods

    def _validate_signal(self, signal: dict) -> bool:
        """Validate signal dict has required fields."""
        required_fields = ["coin", "signal"]
        return all(field in signal for field in required_fields)

    def _calculate_stop_distance(
        self,
        entry_price: float,
        stop_loss: float,
        signal_type: str
    ) -> float:
        """Calculate stop-loss distance with validation."""
        if stop_loss <= 0 or entry_price <= 0:
            return 0

        distance = abs(entry_price - stop_loss)

        # Validate reasonable stop distance (not more than 50% of price)
        if distance > entry_price * 0.5:
            self.logger.warning(
                f"Stop distance ({distance}) > 50% of price ({entry_price})"
            )
            return 0

        return distance

    def _is_position_long(self, size: float) -> bool:
        """Determine if position is long (positive size) or short (negative)."""
        return size > 0

    def _calculate_close_side(self, position_size: float) -> str:
        """Determine order side to close position."""
        # Long position (size > 0) needs sell order to close
        # Short position (size < 0) needs buy order to close
        return "sell" if position_size > 0 else "buy"

    def enforce_take_profit(
        self,
        positions: List[dict],
        current_prices: Dict[str, float],
        position_targets: Dict[str, float]
    ) -> dict:
        """Monitor and automatically close positions that hit take-profit targets.

        Args:
            positions: List of position dicts from HyperliquidClient
            current_prices: Dict mapping symbol to current price
            position_targets: Dict mapping symbol to profit target price

        Returns:
            Dict with positions_closed, close_attempts, failed_closes, total_realized_pnl
        """
        try:
            # Get IOC cross percentage from env or use default
            ioc_cross_pct = float(os.getenv("IOC_CROSS_PCT", "0.005"))  # 0.5% default

            close_attempts = []
            failed_closes = []
            positions_closed = 0
            total_realized_pnl = 0.0

            for pos in positions:
                symbol = pos.get("symbol", "")
                size = pos.get("size", 0)
                current_price = current_prices.get(symbol, pos.get("current_price", 0))
                profit_target = position_targets.get(symbol)

                if size == 0:
                    continue

                # Skip if no profit target set
                if profit_target is None or profit_target <= 0:
                    continue

                # Guard against invalid current price
                if current_price <= 0:
                    self.logger.warning(f"Invalid current_price ({current_price}) for {symbol}, skipping TP check")
                    continue

                # Check take-profit trigger
                take_profit_hit = False
                if self._is_position_long(size):
                    # For longs, trigger when price >= profit_target
                    take_profit_hit = current_price >= profit_target
                else:
                    # For shorts, trigger when price <= profit_target
                    take_profit_hit = current_price <= profit_target

                # Execute close if take-profit hit
                if take_profit_hit:
                    side = self._calculate_close_side(size)
                    quantity = abs(size)

                    self.logger.info(
                        f"Take-profit hit for {symbol}: "
                        f"Current: {current_price}, Target: {profit_target}"
                    )

                    # Calculate aggressive IOC price to ensure fill
                    # For closing longs (sell), price below market
                    # For closing shorts (buy), price above market
                    if side == "sell":
                        ioc_price = current_price * (1 - ioc_cross_pct)
                    else:  # buy
                        ioc_price = current_price * (1 + ioc_cross_pct)

                    try:
                        # Place immediate-or-cancel order with aggressive pricing
                        order_response = self.client.place_order(
                            symbol=symbol,
                            side=side,
                            quantity=quantity,
                            price=ioc_price,
                            order_type={"limit": {"tif": "Ioc"}}
                        )

                        success = order_response.get("success", False)
                        order_id = order_response.get("oid")
                        error = order_response.get("error")

                        # If first attempt failed, retry with larger offset
                        if not success:
                            larger_offset = ioc_cross_pct * 2  # Double the offset
                            if side == "sell":
                                ioc_price_retry = current_price * (1 - larger_offset)
                            else:
                                ioc_price_retry = current_price * (1 + larger_offset)

                            self.logger.warning(
                                f"First IOC attempt failed for {symbol}, retrying with larger offset"
                            )

                            order_response = self.client.place_order(
                                symbol=symbol,
                                side=side,
                                quantity=quantity,
                                price=ioc_price_retry,
                                order_type={"limit": {"tif": "Ioc"}}
                            )

                            success = order_response.get("success", False)
                            order_id = order_response.get("oid")
                            error = order_response.get("error")

                        if success:
                            positions_closed += 1
                            unrealized_pnl = pos.get("unrealized_pnl", 0)
                            total_realized_pnl += unrealized_pnl
                            self.logger.info(
                                f"Successfully closed {symbol} position via take-profit: "
                                f"size={quantity}, P&L=${unrealized_pnl:.2f}"
                            )
                        else:
                            failed_closes.append({
                                "symbol": symbol,
                                "reason": "take_profit",
                                "error": error
                            })
                            self.logger.error(
                                f"Failed to close {symbol} via take-profit: {error}"
                            )

                        close_attempts.append({
                            "symbol": symbol,
                            "reason": "take_profit",
                            "trigger_price": current_price,
                            "target_price": profit_target,
                            "position_size": quantity,
                            "success": success,
                            "order_id": order_id,
                            "error": error
                        })

                    except Exception as e:
                        self.logger.error(
                            f"Exception closing {symbol} via take-profit: {str(e)}"
                        )
                        failed_closes.append({
                            "symbol": symbol,
                            "reason": "take_profit",
                            "error": str(e)
                        })
                        close_attempts.append({
                            "symbol": symbol,
                            "reason": "take_profit",
                            "trigger_price": current_price,
                            "target_price": profit_target,
                            "position_size": quantity,
                            "success": False,
                            "order_id": None,
                            "error": str(e)
                        })

            return {
                "positions_closed": positions_closed,
                "close_attempts": close_attempts,
                "failed_closes": failed_closes,
                "total_realized_pnl": total_realized_pnl
            }

        except Exception as e:
            self.logger.error(f"Error enforcing take-profit: {str(e)}")
            return {
                "positions_closed": 0,
                "close_attempts": [],
                "failed_closes": [],
                "total_realized_pnl": 0
            }

    def _evaluate_invalidation(
        self,
        symbol: str,
        invalidation_data: any,
        current_price: float
    ) -> bool:
        """Evaluate if invalidation condition is triggered.

        Args:
            symbol: Position symbol
            invalidation_data: Invalidation condition data (can be dict, threshold, etc.)
            current_price: Current price

        Returns:
            True if invalidation triggered
        """
        try:
            # If invalidation_data is a dict, check for common condition types
            if isinstance(invalidation_data, dict):
                # Price below threshold
                if "price_below" in invalidation_data:
                    return current_price < invalidation_data["price_below"]

                # Price above threshold
                if "price_above" in invalidation_data:
                    return current_price > invalidation_data["price_above"]

                # Candle close below (would need candle data, simplified here)
                if "close_below" in invalidation_data:
                    return current_price < invalidation_data["close_below"]

                # Time-based invalidation (would need timestamp comparison)
                if "expires_at" in invalidation_data:
                    from datetime import datetime
                    expiry = datetime.fromisoformat(invalidation_data["expires_at"])
                    return datetime.now() > expiry

            # If invalidation_data is a simple threshold number
            elif isinstance(invalidation_data, (int, float)):
                return current_price < invalidation_data

            return False

        except Exception as e:
            self.logger.error(f"Error evaluating invalidation for {symbol}: {str(e)}")
            return False
