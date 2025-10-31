"""Logging module for trade execution and AI reasoning transparency.

Provides:
- TradeLogger: Structured logging of AI decisions and trade executions to SQLite/JSON
- ChatLogger: Human-readable logging of AI reasoning and plan/action messages

Note: This module is named 'logging' which conflicts with Python's built-in logging module.
Always use absolute imports: `from src.logging import TradeLogger, ChatLogger`
"""

from src.logging.trade_logger import TradeLogger
from src.logging.chat_logger import ChatLogger

__all__ = ["TradeLogger", "ChatLogger"]
