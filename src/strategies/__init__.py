"""Trading strategy modules with standardized signal generation interface.

All strategies expose a generate_signal(market_data) method that returns:
{
    "signal": "buy" | "sell" | "hold",
    "confidence": float (0.0 to 1.0),
    "reasoning": str (detailed explanation)
}
"""

from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.ml_classifier import MLClassifierStrategy

__all__ = ["MomentumStrategy", "MeanReversionStrategy", "MLClassifierStrategy"]
