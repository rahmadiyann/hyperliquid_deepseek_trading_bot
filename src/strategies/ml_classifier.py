"""Machine learning-based trading strategy using gradient boosting classifier."""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError


class MLClassifierStrategy:
    """ML-based trading strategy using gradient boosting for trend prediction.

    This strategy uses machine learning to classify trend direction based on:
    - Recent price changes and momentum
    - Technical indicators (RSI, MACD, EMA ratios, ATR)
    - Volume patterns
    - Cross-timeframe features
    - Market context (funding rate, open interest)
    """

    def __init__(self):
        """Initialize ML classifier strategy with default parameters."""
        self.logger = logging.getLogger(__name__)

        # Configurable parameters
        self.lookback_window = 100  # Historical candles for training
        self.feature_window = 10  # Recent candles for features
        self.prediction_horizon = 5  # Predict N candles ahead
        self.price_change_threshold = 0.01  # 1% threshold for buy/sell
        self.min_training_samples = 50  # Minimum samples to train
        self.min_confidence = 0.55  # Minimum probability threshold
        self.retrain_interval = 20  # Retrain every N invocations

        # Model state
        self.model = None  # GradientBoostingClassifier
        self.scaler = None  # StandardScaler
        self.invocation_count = 0
        self.last_training_data_hash = None

        self.logger.info("ML Classifier strategy initialized")

    def generate_signal(self, market_data: dict) -> dict:
        """Generate ML-based trading signal.

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
            - reasoning: str (ML prediction explanation)
        """
        try:
            # Input validation
            if not self._validate_input(market_data):
                return {
                    "signal": "hold",
                    "confidence": 0.0,
                    "reasoning": "Insufficient data for ML classification"
                }

            # Increment invocation counter
            self.invocation_count += 1

            # Extract indicators
            indicators_3m = market_data.get("indicators_3m", {})
            indicators_4h = market_data.get("indicators_4h", {})
            funding_rate = market_data.get("funding_rate", 0)
            open_interest = market_data.get("open_interest", 0)

            prices_3m = indicators_3m.get("prices", [])

            # Feature engineering
            features, feature_names = self._engineer_features(
                indicators_3m, indicators_4h, funding_rate, open_interest
            )

            if features is None or len(features) == 0:
                return {
                    "signal": "hold",
                    "confidence": 0.0,
                    "reasoning": "Failed to engineer features"
                }

            # Generate labels for training
            labels = self._generate_labels(
                prices_3m, self.prediction_horizon, self.price_change_threshold
            )

            # Check if we need to train/retrain
            current_data_hash = self._calculate_data_hash(prices_3m)
            should_retrain = self._should_retrain(current_data_hash)

            if should_retrain:
                # Train the model
                train_success = self._train_model(features, labels, feature_names, prices_3m)
                if not train_success:
                    if self.model is None:
                        return {
                            "signal": "hold",
                            "confidence": 0.0,
                            "reasoning": "Model training failed, no existing model"
                        }
                    # Use existing model
                    self.logger.warning("Training failed, using existing model")

            # Make prediction
            if self.model is None or self.scaler is None:
                return {
                    "signal": "hold",
                    "confidence": 0.0,
                    "reasoning": "Model not trained"
                }

            # Get current features (last row)
            current_features = features[-1:, :]
            current_features_scaled = self.scaler.transform(current_features)

            # Predict
            probabilities = self.model.predict_proba(current_features_scaled)[0]
            classes = self.model.classes_

            # Map probabilities to signal
            signal, confidence = self._map_probabilities_to_signal(
                probabilities, classes, self.min_confidence
            )

            # Feature importance analysis
            top_features = self._get_top_features(
                self.model.feature_importances_, feature_names, top_n=3
            )

            # Generate reasoning
            reasoning_parts = [
                f"ML Prediction (confidence: {confidence:.2%}):",
                f"  Predicted signal: {signal}",
                f"  Probabilities: " + ", ".join([
                    f"{cls}: {prob:.2%}" for cls, prob in zip(classes, probabilities)
                ]),
                f"  Top features: " + ", ".join([
                    f"{name}={imp:.3f}" for name, imp in top_features
                ]),
                f"  Model: trained on {len(labels)} samples",
                f"  Prediction horizon: {self.prediction_horizon} candles"
            ]
            reasoning = "\n".join(reasoning_parts)

            self.logger.info(
                f"ML signal: {signal}, confidence: {confidence:.2f}, "
                f"probs: {probabilities}"
            )

            return {
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning
            }

        except Exception as e:
            self.logger.error(f"Error generating ML signal: {str(e)}")
            return {
                "signal": "hold",
                "confidence": 0.0,
                "reasoning": f"Error in ML classification: {str(e)}"
            }

    def _validate_input(self, market_data: dict) -> bool:
        """Validate input data has sufficient information.

        Args:
            market_data: Market data dict to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(market_data, dict):
            return False

        indicators_3m = market_data.get("indicators_3m", {})
        if not indicators_3m:
            return False

        prices = indicators_3m.get("prices", [])
        if len(prices) < self.min_training_samples:
            self.logger.warning(
                f"Insufficient data: {len(prices)} < {self.min_training_samples}"
            )
            return False

        return True

    def _engineer_features(
        self,
        indicators_3m: dict,
        indicators_4h: dict,
        funding_rate: float,
        open_interest: float
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Create feature matrix from indicators.

        Args:
            indicators_3m: 3-minute indicators
            indicators_4h: 4-hour indicators
            funding_rate: Current funding rate
            open_interest: Current open interest

        Returns:
            Tuple of (features array, feature names)
        """
        try:
            # Extract data
            prices = np.array(indicators_3m.get("prices", []))
            ema20 = np.array(indicators_3m.get("ema20", []))
            ema50 = np.array(indicators_3m.get("ema50", []))
            macd = np.array(indicators_3m.get("macd", []))
            rsi7 = np.array(indicators_3m.get("rsi7", []))
            rsi14 = np.array(indicators_3m.get("rsi14", []))
            atr = np.array(indicators_3m.get("atr3", []))
            volume = np.array(indicators_3m.get("volume", []))

            # Extract 4h indicators for cross-timeframe features
            ema20_4h = np.array(indicators_4h.get("ema20", []))
            rsi14_4h = np.array(indicators_4h.get("rsi14", []))
            macd_4h = np.array(indicators_4h.get("macd", []))

            # Get last available 4h values (or defaults)
            ema20_4h_last = ema20_4h[-1] if len(ema20_4h) > 0 else 0
            rsi14_4h_last = rsi14_4h[-1] if len(rsi14_4h) > 0 else 50
            macd_4h_last = macd_4h[-1] if len(macd_4h) > 0 else 0

            # Extract open interest history if available
            open_interest_history = np.array(indicators_3m.get("open_interest", []))
            if len(open_interest_history) == 0:
                # Try 4h timeframe
                open_interest_history = np.array(indicators_4h.get("open_interest", []))

            if len(prices) < self.feature_window:
                return None, []

            # Initialize feature list
            features_list = []
            feature_names = []

            # Calculate features for each candle
            num_samples = len(prices) - self.feature_window + 1
            for i in range(self.feature_window - 1, len(prices)):
                features = []

                # Price features
                # Price changes
                if i >= 1:
                    price_change_1 = (prices[i] - prices[i-1]) / prices[i-1]
                    features.append(price_change_1)
                else:
                    features.append(0)

                if i >= 3:
                    price_change_3 = (prices[i] - prices[i-3]) / prices[i-3]
                    features.append(price_change_3)
                else:
                    features.append(0)

                if i >= 5:
                    price_change_5 = (prices[i] - prices[i-5]) / prices[i-5]
                    features.append(price_change_5)
                else:
                    features.append(0)

                if i >= 10:
                    price_momentum = (prices[i] - prices[i-10]) / prices[i-10]
                    features.append(price_momentum)
                else:
                    features.append(0)

                # Distance from EMAs
                if len(ema20) > i and ema20[i] > 0:
                    dist_ema20 = (prices[i] - ema20[i]) / ema20[i]
                    features.append(dist_ema20)
                else:
                    features.append(0)

                if len(ema50) > i and ema50[i] > 0:
                    dist_ema50 = (prices[i] - ema50[i]) / ema50[i]
                    features.append(dist_ema50)
                else:
                    features.append(0)

                # Indicator features
                if len(rsi7) > i:
                    features.append(rsi7[i] / 100.0)  # Normalize
                else:
                    features.append(0.5)

                if len(rsi14) > i:
                    features.append(rsi14[i] / 100.0)  # Normalize
                else:
                    features.append(0.5)

                if len(macd) > i:
                    features.append(macd[i])
                else:
                    features.append(0)

                # MACD change
                if len(macd) > i and i >= 1:
                    macd_change = macd[i] - macd[i-1]
                    features.append(macd_change)
                else:
                    features.append(0)

                # ATR as percentage
                if len(atr) > i and prices[i] > 0:
                    atr_pct = atr[i] / prices[i]
                    features.append(atr_pct)
                else:
                    features.append(0)

                # Volume features
                if len(volume) > i and i >= 1 and volume[i-1] > 0:
                    volume_change = (volume[i] - volume[i-1]) / volume[i-1]
                    features.append(volume_change)
                else:
                    features.append(0)

                if len(volume) > i and i >= 20:
                    avg_volume = np.mean(volume[i-20:i])
                    if avg_volume > 0:
                        volume_ratio = volume[i] / avg_volume
                        features.append(volume_ratio)
                    else:
                        features.append(1.0)
                else:
                    features.append(1.0)

                # Cross-timeframe features (4h)
                # Distance from 4h EMA20
                if ema20_4h_last > 0:
                    dist_4h_ema20 = (prices[i] - ema20_4h_last) / ema20_4h_last
                    features.append(dist_4h_ema20)
                else:
                    features.append(0)

                # 4h RSI normalized
                features.append(rsi14_4h_last / 100.0)

                # 4h MACD
                features.append(macd_4h_last)

                # Open interest change
                if len(open_interest_history) > i and i >= 1:
                    if open_interest_history[i-1] > 0:
                        oi_change = (open_interest_history[i] - open_interest_history[i-1]) / open_interest_history[i-1]
                        features.append(oi_change)
                    else:
                        features.append(0)
                else:
                    # No history available, use current value vs baseline approximation
                    if i >= 1 and open_interest > 0:
                        # Approximate as small change
                        features.append(0.0)
                    else:
                        features.append(0)

                # Market context
                features.append(funding_rate)
                features.append(open_interest / 1e9 if open_interest > 0 else 0)

                features_list.append(features)

            # Feature names (for first iteration)
            if not feature_names:
                feature_names = [
                    "price_change_1", "price_change_3", "price_change_5",
                    "price_momentum_10", "dist_ema20", "dist_ema50",
                    "rsi7_norm", "rsi14_norm", "macd", "macd_change",
                    "atr_pct", "volume_change", "volume_ratio",
                    "dist_4h_ema20", "rsi14_4h_norm", "macd_4h",
                    "open_interest_change", "funding_rate", "open_interest"
                ]

            features_array = np.array(features_list)
            return features_array, feature_names

        except Exception as e:
            self.logger.error(f"Error engineering features: {str(e)}")
            return None, []

    def _generate_labels(
        self,
        prices: list,
        prediction_horizon: int,
        threshold: float
    ) -> np.ndarray:
        """Generate training labels from future price movements.

        Args:
            prices: Price series
            prediction_horizon: How many candles ahead to predict
            threshold: Price change threshold for buy/sell classification

        Returns:
            Label array (-1: sell, 0: hold, 1: buy)
        """
        try:
            labels = []
            for i in range(len(prices) - prediction_horizon):
                future_price = prices[i + prediction_horizon]
                current_price = prices[i]

                price_change = (future_price - current_price) / current_price

                if price_change > threshold:
                    labels.append(1)  # Buy
                elif price_change < -threshold:
                    labels.append(-1)  # Sell
                else:
                    labels.append(0)  # Hold

            return np.array(labels)

        except Exception as e:
            self.logger.error(f"Error generating labels: {str(e)}")
            return np.array([])

    def _should_retrain(self, current_data_hash: int) -> bool:
        """Determine if model should be retrained.

        Args:
            current_data_hash: Hash of current price data

        Returns:
            True if retraining needed
        """
        # First time
        if self.model is None:
            return True

        # Periodic retraining
        if self.invocation_count % self.retrain_interval == 0:
            self.logger.info(f"Periodic retrain at invocation {self.invocation_count}")
            return True

        # Data changed significantly
        if self.last_training_data_hash is not None and self.last_training_data_hash != current_data_hash:
            self.logger.info("Data hash changed, triggering retrain")
            return True

        return False

    def _calculate_data_hash(self, prices: list) -> int:
        """Calculate hash of price data.

        Args:
            prices: Price series

        Returns:
            Hash value
        """
        try:
            # Use last 50 prices for hash
            recent_prices = prices[-50:] if len(prices) >= 50 else prices
            return hash(tuple(recent_prices))
        except Exception:
            return 0

    def _train_model(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: list,
        prices: list
    ) -> bool:
        """Train the gradient boosting model.

        Args:
            features: Feature matrix
            labels: Label array
            feature_names: Names of features
            prices: Price series for hash calculation

        Returns:
            True if training successful
        """
        try:
            # Align features and labels
            # Features start at index (feature_window - 1), so align labels accordingly
            labels_aligned = labels[self.feature_window - 1:]
            X = features[:len(labels_aligned)]
            y = labels_aligned

            # Check minimum samples after alignment
            if len(X) < self.min_training_samples:
                self.logger.warning(
                    f"Insufficient aligned samples for training: {len(X)}"
                )
                return False

            # Check for class imbalance
            unique_labels, counts = np.unique(y, return_counts=True)
            if len(unique_labels) < 2:
                self.logger.warning(
                    f"Only one class in labels: {unique_labels[0]}"
                )
                return False

            class_dist = dict(zip(unique_labels, counts))
            self.logger.info(f"Training on {len(y)} samples, distribution: {class_dist}")

            # Initialize model and scaler
            if self.model is None:
                self.model = GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42
                )

            if self.scaler is None:
                self.scaler = StandardScaler()

            # Fit scaler and transform features
            X_scaled = self.scaler.fit_transform(X)

            # Train model
            self.model.fit(X_scaled, y)

            # Store data hash from prices
            self.last_training_data_hash = self._calculate_data_hash(prices)

            self.logger.info("Model trained successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            return False

    def _map_probabilities_to_signal(
        self,
        probabilities: np.ndarray,
        classes: np.ndarray,
        min_confidence: float
    ) -> Tuple[str, float]:
        """Map model probabilities to trading signal.

        Args:
            probabilities: Probability array from model
            classes: Class labels
            min_confidence: Minimum confidence threshold

        Returns:
            Tuple of (signal, confidence)
        """
        try:
            # Find class with highest probability
            max_prob_idx = np.argmax(probabilities)
            predicted_class = classes[max_prob_idx]
            confidence = probabilities[max_prob_idx]

            # Map class to signal
            if predicted_class == 1 and confidence >= min_confidence:
                signal = "buy"
            elif predicted_class == -1 and confidence >= min_confidence:
                signal = "sell"
            else:
                signal = "hold"
                # For hold, confidence is certainty of not trading
                if predicted_class == 0:
                    confidence = probabilities[max_prob_idx]
                else:
                    # Low confidence on buy/sell -> hold
                    confidence = 1.0 - confidence

            return signal, confidence

        except Exception as e:
            self.logger.error(f"Error mapping probabilities: {str(e)}")
            return "hold", 0.0

    def _get_top_features(
        self,
        importances: np.ndarray,
        feature_names: list,
        top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """Get top N most important features.

        Args:
            importances: Feature importance array
            feature_names: Feature names
            top_n: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        try:
            if len(importances) != len(feature_names):
                return []

            # Get indices of top features
            top_indices = np.argsort(importances)[-top_n:][::-1]

            # Return feature names and importances
            top_features = [
                (feature_names[i], importances[i])
                for i in top_indices
            ]

            return top_features

        except Exception as e:
            self.logger.error(f"Error getting top features: {str(e)}")
            return []
