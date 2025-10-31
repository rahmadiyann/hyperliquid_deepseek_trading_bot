"""Momentum-based trading strategy using time-series momentum with volatility targeting."""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np


class MomentumStrategy:
    """Time-series momentum strategy with EMA crossover, MACD confirmation, and breakout detection.

    This strategy identifies trending markets using multiple momentum indicators:
    - EMA crossover (fast vs slow) for trend direction
    - MACD for momentum confirmation
    - Breakout detection using ATR-based volatility bands
    - Volatility targeting to adjust position confidence
    """

    def __init__(self):
        """Initialize momentum strategy with default parameters."""
        self.logger = logging.getLogger(__name__)

        # Configurable parameters
        self.ema_fast_period = 20  # Fast EMA for crossover
        self.ema_slow_period = 50  # Slow EMA for crossover
        self.macd_threshold = 0  # MACD must be above this for buy
        self.breakout_atr_multiplier = 2.0  # ATR multiplier for breakout detection
        self.volatility_target = 0.02  # 2% target volatility
        self.min_confidence = 0.5  # Minimum confidence threshold
        self.max_confidence = 0.9  # Maximum confidence cap

        self.logger.info("Momentum strategy initialized")

    def generate_signal(self, market_data: dict) -> dict:
        """Generate momentum-based trading signal.

        Args:
            market_data: Dict containing:
                - indicators_3m: 3-minute indicators
                - indicators_4h: 4-hour indicators
                - open_interest: Current open interest (optional)
                - funding_rate: Current funding rate (optional)

        Returns:
            Dict with:
            - signal: "buy", "sell", or "hold"
            - confidence: float (0.0 to 1.0)
            - reasoning: str (detailed explanation)
        """
        try:
            # Input validation
            if not self._validate_input(market_data):
                return {
                    "signal": "hold",
                    "confidence": 0.0,
                    "reasoning": "Insufficient data for momentum analysis"
                }

            # Extract indicators
            indicators_3m = market_data.get("indicators_3m", {})
            indicators_4h = market_data.get("indicators_4h", {})

            prices_3m = indicators_3m.get("prices", [])
            ema20_3m = indicators_3m.get("ema20", [])
            ema50_3m = indicators_3m.get("ema50", [])
            macd_3m = indicators_3m.get("macd", [])
            atr_3m = indicators_3m.get("atr3", [])
            volume_3m = indicators_3m.get("volume", [])

            ema20_4h = indicators_4h.get("ema20", [])
            ema50_4h = indicators_4h.get("ema50", [])
            macd_4h = indicators_4h.get("macd", [])

            # EMA Crossover Detection
            ema_score, ema_reasoning = self._detect_ema_crossover(
                ema20_3m, ema50_3m, ema20_4h, ema50_4h
            )

            # MACD Confirmation
            macd_score, macd_reasoning = self._analyze_macd(macd_3m, macd_4h)

            # Breakout Detection
            breakout_score, breakout_reasoning = self._detect_breakout(
                prices_3m, atr_3m, volume_3m
            )

            # Volatility Targeting
            volatility_adjustment, vol_reasoning = self._calculate_volatility_adjustment(
                prices_3m[-1] if prices_3m else 0,
                atr_3m[-1] if atr_3m else 0,
                self.volatility_target
            )

            # Aggregate signals
            total_score = ema_score + macd_score + breakout_score

            # Determine signal
            if total_score >= 2:
                signal = "buy"
            elif total_score <= -2:
                signal = "sell"
            else:
                signal = "hold"

            # Calculate confidence
            base_confidence = abs(total_score) / 3.0  # Normalize to 0-1
            confidence = base_confidence * volatility_adjustment

            # Apply confidence clamping only for actionable signals
            if signal == "hold":
                confidence = 0.0
            else:
                confidence = max(self.min_confidence, min(self.max_confidence, confidence))

            # Generate reasoning
            reasoning_parts = [
                f"Momentum Analysis (score: {total_score}/3):",
                f"  {ema_reasoning}",
                f"  {macd_reasoning}",
                f"  {breakout_reasoning}",
                f"  {vol_reasoning}",
                f"Signal: {signal} (confidence: {confidence:.2f})"
            ]
            reasoning = "\n".join(reasoning_parts)

            self.logger.info(
                f"Momentum signal: {signal}, confidence: {confidence:.2f}, "
                f"scores: EMA={ema_score}, MACD={macd_score}, Breakout={breakout_score}"
            )

            return {
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning
            }

        except Exception as e:
            self.logger.error(f"Error generating momentum signal: {str(e)}")
            return {
                "signal": "hold",
                "confidence": 0.0,
                "reasoning": f"Error in momentum analysis: {str(e)}"
            }

    def _validate_input(self, market_data: dict) -> bool:
        """Validate input data has sufficient information.

        Args:
            market_data: Market data dict to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(market_data, dict):
            self.logger.warning("Market data is not a dict")
            return False

        indicators_3m = market_data.get("indicators_3m", {})
        indicators_4h = market_data.get("indicators_4h", {})

        # Check 3m indicators
        if not indicators_3m:
            self.logger.warning("No 3-minute indicators")
            return False

        prices_3m = indicators_3m.get("prices", [])
        ema20_3m = indicators_3m.get("ema20", [])
        ema50_3m = indicators_3m.get("ema50", [])

        # Need at least 50 points for EMA50
        if len(prices_3m) < 50 or len(ema20_3m) < 50 or len(ema50_3m) < 50:
            self.logger.warning(
                f"Insufficient 3m data: prices={len(prices_3m)}, "
                f"ema20={len(ema20_3m)}, ema50={len(ema50_3m)}"
            )
            return False

        # Check 4h indicators (optional but preferred)
        if indicators_4h:
            ema20_4h = indicators_4h.get("ema20", [])
            ema50_4h = indicators_4h.get("ema50", [])
            if len(ema20_4h) < 10 or len(ema50_4h) < 10:
                self.logger.debug("Limited 4h data, will proceed with 3m only")

        return True

    def _detect_ema_crossover(
        self,
        ema20_3m: list,
        ema50_3m: list,
        ema20_4h: list,
        ema50_4h: list
    ) -> Tuple[int, str]:
        """Detect EMA crossover and determine score.

        Args:
            ema20_3m: Fast EMA on 3-minute timeframe
            ema50_3m: Slow EMA on 3-minute timeframe
            ema20_4h: Fast EMA on 4-hour timeframe
            ema50_4h: Slow EMA on 4-hour timeframe

        Returns:
            Tuple of (score, reasoning)
            Score: +1 (bullish), -1 (bearish), 0 (neutral)
        """
        try:
            # Current crossover state
            current_bullish_3m = ema20_3m[-1] > ema50_3m[-1]

            # Detect recent crossover (last 5 candles)
            lookback = min(5, len(ema20_3m) - 1)
            crossed_recently = False
            for i in range(1, lookback + 1):
                prev_bullish = ema20_3m[-i-1] > ema50_3m[-i-1]
                if current_bullish_3m != prev_bullish:
                    crossed_recently = True
                    break

            # Check 4-hour confirmation
            has_4h_confirmation = False
            if len(ema20_4h) > 0 and len(ema50_4h) > 0:
                current_bullish_4h = ema20_4h[-1] > ema50_4h[-1]
                has_4h_confirmation = (current_bullish_3m == current_bullish_4h)

            # Calculate score
            score = 0
            if current_bullish_3m:
                score = 1
                direction = "bullish"
            else:
                score = -1
                direction = "bearish"

            # Build reasoning
            reasoning = f"EMA(20) vs EMA(50): {direction}"
            if crossed_recently:
                reasoning += " (recent crossover)"
            if has_4h_confirmation:
                reasoning += ", confirmed on 4h"
                reasoning += f" [score: {score}]"
            else:
                reasoning += ", no 4h confirmation"
                reasoning += f" [score: {score}]"

            return score, reasoning

        except Exception as e:
            self.logger.error(f"Error in EMA crossover detection: {str(e)}")
            return 0, "EMA crossover analysis failed"

    def _analyze_macd(self, macd_3m: list, macd_4h: list) -> Tuple[int, str]:
        """Analyze MACD for momentum confirmation.

        Args:
            macd_3m: MACD values on 3-minute timeframe
            macd_4h: MACD values on 4-hour timeframe

        Returns:
            Tuple of (score, reasoning)
            Score: +1 (bullish), -1 (bearish), 0 (neutral)
        """
        try:
            if not macd_3m or len(macd_3m) < 3:
                return 0, "MACD: insufficient data"

            # Current MACD direction
            current_macd = macd_3m[-1]
            is_positive = current_macd > self.macd_threshold

            # MACD momentum (is it increasing?)
            prev_macd = macd_3m[-3] if len(macd_3m) >= 3 else macd_3m[-2]
            is_increasing = current_macd > prev_macd

            # Check 4h MACD confirmation
            has_4h_confirmation = False
            if len(macd_4h) > 0:
                macd_4h_positive = macd_4h[-1] > self.macd_threshold
                has_4h_confirmation = (is_positive == macd_4h_positive)

            # Calculate score
            if is_positive and is_increasing:
                score = 1
                description = "positive and increasing"
            elif not is_positive and not is_increasing:
                score = -1
                description = "negative and decreasing"
            else:
                score = 0
                description = "mixed signals"

            # Build reasoning
            reasoning = f"MACD: {description} ({current_macd:.4f})"
            if has_4h_confirmation:
                reasoning += ", 4h confirms"
            reasoning += f" [score: {score}]"

            return score, reasoning

        except Exception as e:
            self.logger.error(f"Error in MACD analysis: {str(e)}")
            return 0, "MACD analysis failed"

    def _detect_breakout(
        self,
        prices: list,
        atr: list,
        volume: list
    ) -> Tuple[int, str]:
        """Detect price breakouts with volume confirmation.

        Args:
            prices: Price series
            atr: ATR values
            volume: Volume series

        Returns:
            Tuple of (score, reasoning)
            Score: +1 (upside breakout), -1 (downside breakout), 0 (no breakout)
        """
        try:
            if not prices or len(prices) < 20 or not atr:
                return 0, "Breakout: insufficient data"

            # Calculate recent high/low (last 20 candles)
            lookback_prices = prices[-20:]
            recent_high = max(lookback_prices)
            recent_low = min(lookback_prices)

            current_price = prices[-1]
            current_atr = atr[-1]

            # Calculate breakout thresholds
            upside_threshold = recent_high + (current_atr * self.breakout_atr_multiplier)
            downside_threshold = recent_low - (current_atr * self.breakout_atr_multiplier)

            # Check for volume surge
            has_volume_confirmation = False
            if volume and len(volume) >= 20:
                recent_volume = volume[-20:]
                avg_volume = np.mean(recent_volume)
                current_volume = volume[-1]
                has_volume_confirmation = current_volume > avg_volume

            # Detect breakout
            score = 0
            if current_price > upside_threshold:
                score = 1
                direction = "upside"
            elif current_price < downside_threshold:
                score = -1
                direction = "downside"
            else:
                return 0, "Breakout: price within range [score: 0]"

            # Build reasoning
            reasoning = f"Breakout: {direction} (price: {current_price:.2f}"
            if direction == "upside":
                reasoning += f", threshold: {upside_threshold:.2f})"
            else:
                reasoning += f", threshold: {downside_threshold:.2f})"

            if has_volume_confirmation:
                reasoning += " with volume surge"
            else:
                reasoning += " without volume confirmation"

            reasoning += f" [score: {score}]"

            return score, reasoning

        except Exception as e:
            self.logger.error(f"Error in breakout detection: {str(e)}")
            return 0, "Breakout detection failed"

    def _calculate_volatility_adjustment(
        self,
        current_price: float,
        current_atr: float,
        target_volatility: float
    ) -> Tuple[float, str]:
        """Calculate confidence adjustment based on volatility.

        Args:
            current_price: Current price
            current_atr: Current ATR value
            target_volatility: Target volatility (as fraction)

        Returns:
            Tuple of (adjustment_multiplier, reasoning)
            Multiplier: 0.5 to 1.0
        """
        try:
            if current_price <= 0 or current_atr <= 0:
                return 0.8, "Volatility: invalid data"

            # Calculate current volatility as percentage
            current_volatility = current_atr / current_price

            # Calculate volatility ratio
            vol_ratio = current_volatility / target_volatility

            # Determine adjustment
            if 0.8 <= vol_ratio <= 1.2:
                # Optimal volatility range
                adjustment = 1.0
                description = "optimal"
            elif vol_ratio < 0.5:
                # Too low volatility (not enough movement)
                adjustment = 0.7
                description = "too low"
            elif vol_ratio > 3.0:
                # Too high volatility (too risky)
                adjustment = 0.5
                description = "too high"
            elif vol_ratio < 0.8:
                adjustment = 0.85
                description = "low"
            elif vol_ratio > 1.5:
                adjustment = 0.75
                description = "high"
            else:
                adjustment = 0.9
                description = "moderate"

            reasoning = (
                f"Volatility: {current_volatility:.2%} "
                f"(target: {target_volatility:.2%}, {description}), "
                f"confidence adjusted by {adjustment:.2f}x"
            )

            return adjustment, reasoning

        except Exception as e:
            self.logger.error(f"Error calculating volatility adjustment: {str(e)}")
            return 0.8, "Volatility adjustment failed"
