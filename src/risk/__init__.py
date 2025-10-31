"""Risk management module for position sizing, leverage control, and drawdown protection.

Provides:
- Position sizing based on account risk parameters
- Leverage validation and enforcement
- Automatic stop-loss monitoring and execution
- Drawdown kill-switch protection
"""

from src.risk.risk_manager import RiskManager

__all__ = ["RiskManager"]
