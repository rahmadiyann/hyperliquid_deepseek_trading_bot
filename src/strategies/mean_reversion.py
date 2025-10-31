"""Mean reversion trading strategy using RSI and Bollinger Bands."""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np


class MeanReversionStrategy:
    """Mean reversion strategy using RSI oversold/overbought and Bollinger Bands.

    This strategy identifies price extremes and potential reversions using:
    - RSI (7-period and 14-period) for overbought/oversold conditions
    - Bollinger Bands (EMA20 ± 2*STD) for price extremes
    - Multi-timeframe confirmation (3-minute and 4-hour)
    - Reversal confirmation from price action
    """

    def __init__(self):
        """Initialize mean reversion strategy with default parameters."""
        self.logger = logging.getLogger(__name__)

        # Configurable parameters
        self.rsi_oversold_threshold = 30  # RSI below = oversold
        self.rsi_overbought_threshold = 70  # RSI above = overbought
        self.rsi_extreme_oversold = 20  # Extreme oversold
        self.rsi_extreme_overbought = 80  # Extreme overbought
        self.bollinger_std_multiplier = 2.0  # Std dev multiplier
        self.min_confidence = 0.5  # Minimum confidence
        self.max_confidence = 0.95  # Maximum confidence
        self.lookback_period = 20  # Period for Bollinger calculation

        self.logger.info("Mean reversion strategy initialized")

    def generate_signal(self, market_data: dict) -> dict:
        """Generate mean reversion trading signal.

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
                    "reasoning": "Insufficient data for mean reversion analysis"
                }

            # Extract indicators
            indicators_3m = market_data.get("indicators_3m", {})
            indicators_4h = market_data.get("indicators_4h", {})

            prices_3m = indicators_3m.get("prices", [])
            ema20_3m = indicators_3m.get("ema20", [])
            rsi7_3m = indicators_3m.get("rsi7", [])
            rsi14_3m = indicators_3m.get("rsi14", [])
            atr_3m = indicators_3m.get("atr3", [])

            ema20_4h = indicators_4h.get("ema20", [])
            rsi14_4h = indicators_4h.get("rsi14", [])
            atr_4h = indicators_4h.get("atr14", [])

            # RSI Analysis
            rsi_score, rsi_zone = self._calculate_rsi_score(
                rsi7_3m[-1] if rsi7_3m else 50,
                rsi14_3m[-1] if rsi14_3m else 50,
                rsi14_4h[-1] if rsi14_4h else 50
            )

            # Bollinger Band Analysis
            upper_band, middle_band, lower_band = self._calculate_bollinger_bands(
                prices_3m,
                ema20_3m[-1] if ema20_3m else 0,
                self.lookback_period,
                self.bollinger_std_multiplier
            )

            bollinger_score, band_position = self._calculate_bollinger_score(
                prices_3m[-1] if prices_3m else 0,
                upper_band,
                lower_band,
                atr_3m[-1] if atr_3m else 0
            )

            # Check reversal confirmation
            signal_direction = "buy" if (rsi_score + bollinger_score) > 0 else "sell"
            reversal_confirmed = self._check_reversal_confirmation(
                prices_3m, signal_direction
            )
            reversal_bonus = 0.5 if reversal_confirmed else 0

            # Volatility context
            current_volatility = 0
            band_width = 0
            if prices_3m and ema20_3m and atr_3m:
                current_volatility = atr_3m[-1] / prices_3m[-1]
                if upper_band > 0 and lower_band > 0:
                    band_width = (upper_band - lower_band) / middle_band

            volatility_adjustment = self._calculate_volatility_adjustment(
                current_volatility, band_width
            )

            # Aggregate signals
            total_score = rsi_score + bollinger_score + reversal_bonus

            # Determine signal
            if total_score >= 2:
                signal = "buy"
            elif total_score <= -2:
                signal = "sell"
            else:
                signal = "hold"

            # Calculate confidence
            base_confidence = min(abs(total_score) / 4.0, 1.0)

            # Boost for extreme RSI
            extreme_boost = 0
            if rsi7_3m:
                if rsi7_3m[-1] < self.rsi_extreme_oversold and signal == "buy":
                    extreme_boost = 0.15
                elif rsi7_3m[-1] > self.rsi_extreme_overbought and signal == "sell":
                    extreme_boost = 0.15

            confidence = (base_confidence + extreme_boost) * volatility_adjustment

            # Apply confidence clamping only for actionable signals
            if signal == "hold":
                confidence = 0.0
            else:
                confidence = max(self.min_confidence, min(self.max_confidence, confidence))

            # Generate reasoning
            # Prepare display variables for RSI
            rsi7_disp = rsi7_3m[-1] if rsi7_3m else None
            rsi14_disp = rsi14_3m[-1] if rsi14_3m else None
            rsi14_4h_disp = rsi14_4h[-1] if rsi14_4h else None

            rsi7_str = f"{rsi7_disp:.1f}" if rsi7_disp is not None else "N/A"
            rsi14_str = f"{rsi14_disp:.1f}" if rsi14_disp is not None else "N/A"
            rsi14_4h_str = f"{rsi14_4h_disp:.1f}" if rsi14_4h_disp is not None else "N/A"

            # Prepare display variable for price
            price_disp = prices_3m[-1] if prices_3m else 0.0

            reasoning_parts = [
                f"Mean Reversion Analysis (score: {total_score:.1f}/4):",
                f"  RSI: {rsi_zone} (7p: {rsi7_str}, 14p: {rsi14_str}, 4h: {rsi14_4h_str}) [score: {rsi_score}]",
                f"  Bollinger: {band_position} (price: {price_disp:.2f}, bands: {lower_band:.2f}/{middle_band:.2f}/{upper_band:.2f}) [score: {bollinger_score}]",
                f"  Reversal: {'confirmed' if reversal_confirmed else 'not confirmed'} [bonus: {reversal_bonus}]",
                f"  Volatility: {current_volatility:.2%} (band width: {band_width:.2%}), adjustment: {volatility_adjustment:.2f}x",
                f"Signal: {signal} (confidence: {confidence:.2f})"
            ]
            reasoning = "\n".join(reasoning_parts)

            self.logger.info(
                f"Mean reversion signal: {signal}, confidence: {confidence:.2f}, "
                f"scores: RSI={rsi_score}, Bollinger={bollinger_score}"
            )

            return {
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning
            }

        except Exception as e:
            self.logger.error(f"Error generating mean reversion signal: {str(e)}")
            return {
                "signal": "hold",
                "confidence": 0.0,
                "reasoning": f"Error in mean reversion analysis: {str(e)}"
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

        # Check 3m indicators
        if not indicators_3m:
            self.logger.warning("No 3-minute indicators")
            return False

        prices_3m = indicators_3m.get("prices", [])
        rsi7_3m = indicators_3m.get("rsi7", [])
        rsi14_3m = indicators_3m.get("rsi14", [])

        # Need at least 20 points for Bollinger Bands
        if len(prices_3m) < 20:
            self.logger.warning(f"Insufficient price data: {len(prices_3m)}")
            return False

        if not rsi7_3m or not rsi14_3m:
            self.logger.warning("Missing RSI indicators")
            return False

        return True

    def _calculate_rsi_score(
        self,
        rsi7: float,
        rsi14: float,
        rsi_4h: float
    ) -> Tuple[float, str]:
        """Calculate RSI-based score and zone description.

        Args:
            rsi7: 7-period RSI
            rsi14: 14-period RSI
            rsi_4h: 4-hour RSI

        Returns:
            Tuple of (score, zone_description)
        """
        try:
            # Determine zone for 3-minute RSI
            if rsi7 < self.rsi_oversold_threshold:
                zone_3m = "oversold"
                score_3m = 1
            elif rsi7 > self.rsi_overbought_threshold:
                zone_3m = "overbought"
                score_3m = -1
            else:
                zone_3m = "neutral"
                score_3m = 0

            # Check 14-period confirmation
            if rsi14 < self.rsi_oversold_threshold and score_3m > 0:
                # Both oversold
                pass  # Keep score
            elif rsi14 > self.rsi_overbought_threshold and score_3m < 0:
                # Both overbought
                pass  # Keep score
            elif score_3m == 0:
                # 7-period neutral, check 14-period
                if rsi14 < self.rsi_oversold_threshold:
                    score_3m = 0.5
                    zone_3m = "mildly oversold"
                elif rsi14 > self.rsi_overbought_threshold:
                    score_3m = -0.5
                    zone_3m = "mildly overbought"

            # Check 4-hour confirmation
            score = score_3m
            if rsi_4h < self.rsi_oversold_threshold and score_3m > 0:
                # 4h also oversold - strong signal
                score = 2
                zone_description = f"{zone_3m} (multi-timeframe confirmed)"
            elif rsi_4h > self.rsi_overbought_threshold and score_3m < 0:
                # 4h also overbought - strong signal
                score = -2
                zone_description = f"{zone_3m} (multi-timeframe confirmed)"
            else:
                score = score_3m
                zone_description = zone_3m

            return score, zone_description

        except Exception as e:
            self.logger.error(f"Error calculating RSI score: {str(e)}")
            return 0.0, "RSI calculation failed"

    def _calculate_bollinger_bands(
        self,
        prices: list,
        ema: float,
        lookback: int,
        std_multiplier: float
    ) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands.

        Args:
            prices: Price series
            ema: Current EMA value (middle band)
            lookback: Period for standard deviation calculation
            std_multiplier: Standard deviation multiplier

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        try:
            if not prices or len(prices) < lookback or ema <= 0:
                return 0, 0, 0

            # Get recent prices
            recent_prices = prices[-lookback:]

            # Calculate standard deviation
            std_dev = np.std(recent_prices)

            # Calculate bands
            middle_band = ema
            upper_band = middle_band + (std_dev * std_multiplier)
            lower_band = middle_band - (std_dev * std_multiplier)

            return upper_band, middle_band, lower_band

        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return 0, 0, 0

    def _calculate_bollinger_score(
        self,
        price: float,
        upper: float,
        lower: float,
        atr: float
    ) -> Tuple[int, str]:
        """Calculate Bollinger Band score and position description.

        Args:
            price: Current price
            upper: Upper Bollinger Band
            lower: Lower Bollinger Band
            atr: Current ATR

        Returns:
            Tuple of (score, position_description)
        """
        try:
            if price <= 0 or upper <= 0 or lower <= 0:
                return 0, "invalid data"

            # Check position relative to bands
            if price < lower:
                # Below lower band (oversold)
                distance_in_atr = abs(price - lower) / atr if atr > 0 else 0
                if distance_in_atr > 1:
                    score = 2
                    position = "significantly below lower band"
                else:
                    score = 1
                    position = "below lower band"
            elif price > upper:
                # Above upper band (overbought)
                distance_in_atr = abs(price - upper) / atr if atr > 0 else 0
                if distance_in_atr > 1:
                    score = -2
                    position = "significantly above upper band"
                else:
                    score = -1
                    position = "above upper band"
            else:
                # Within bands
                score = 0
                position = "within bands"

            return score, position

        except Exception as e:
            self.logger.error(f"Error calculating Bollinger score: {str(e)}")
            return 0, "calculation failed"

    def _check_reversal_confirmation(
        self,
        prices: list,
        signal_direction: str
    ) -> bool:
        """Check if price is starting to reverse in expected direction.

        Args:
            prices: Price series
            signal_direction: Expected direction ("buy" or "sell")

        Returns:
            True if reversal is confirmed
        """
        try:
            if not prices or len(prices) < 2:
                return False

            current_price = prices[-1]
            prev_price = prices[-2]

            if signal_direction == "buy":
                # For buy signal, check if price is starting to bounce up
                return current_price > prev_price
            else:
                # For sell signal, check if price is starting to drop
                return current_price < prev_price

        except Exception as e:
            self.logger.error(f"Error checking reversal: {str(e)}")
            return False

    def _calculate_volatility_adjustment(
        self,
        volatility: float,
        band_width: float
    ) -> float:
        """Calculate confidence adjustment based on volatility.

        Args:
            volatility: Current volatility (ATR / price)
            band_width: Bollinger Band width (normalized)

        Returns:
            Adjustment multiplier (0.5 to 1.0)
        """
        try:
            # Mean reversion works better in low-medium volatility
            if volatility > 0.03:  # > 3% volatility
                adjustment = 0.5  # High volatility, less confident
            elif volatility > 0.02:
                adjustment = 0.7
            elif band_width > 0.15:  # Wide bands
                adjustment = 0.6  # High volatility environment
            else:
                adjustment = 1.0  # Normal volatility, full confidence

            return adjustment

        except Exception as e:
            self.logger.error(f"Error calculating volatility adjustment: {str(e)}")
            return 0.8
