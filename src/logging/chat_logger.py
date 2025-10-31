import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from dotenv import load_dotenv


class ChatLogger:
    """Human-readable logging of AI reasoning and plan/action messages to text files.

    Provides structured logging with automatic rotation, search, and export capabilities.
    """

    def __init__(self):
        """Initialize ChatLogger with configuration from environment variables."""
        load_dotenv()

        self.log_path = os.getenv("CHAT_LOG_PATH", "./data/chat_log.txt")
        max_size_mb = float(os.getenv("CHAT_LOG_MAX_SIZE_MB", 10))
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.logger = logging.getLogger(__name__)

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.file_handle = None

        self.logger.info(f"ChatLogger initialized with log path: {self.log_path}")

    def log_message(self, message: str, category: str = "AI_REASONING") -> bool:
        """Log an AI reasoning or action message to the text file.

        Args:
            message: The text message to log
            category: Category tag (e.g., "AI_REASONING", "RISK_EVENT")

        Returns:
            True if successful, False otherwise
        """
        try:
            self._check_and_rotate_log()
            with open(self.log_path, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] [{category}]\n")
                f.write(message + "\n\n")
                f.write("---\n")
                f.flush()
            return True
        except Exception as e:
            self.logger.error(f"Error logging message: {e}")
            return False

    def log_ai_decision_summary(self, decisions: list[dict], invocation_number: int) -> bool:
        """Log a summary of AI decisions in a human-readable format.

        Args:
            decisions: List of decision dicts from DeepSeekEngine
            invocation_number: Current invocation number

        Returns:
            True if successful, False otherwise
        """
        try:
            lines = [f"Invocation #{invocation_number}\n"]
            for decision in decisions:
                coin = decision['coin']
                signal = decision['signal'].upper()
                confidence = decision['confidence']
                reasoning = decision['reasoning']
                if len(reasoning) > 100:
                    reasoning = reasoning[:100] + "..."
                profit_target = decision.get('profit_target')
                stop_loss = decision.get('stop_loss')

                lines.append(f"{coin}: {signal} (confidence: {confidence:.2f})")
                lines.append(f"  Reasoning: {reasoning}")
                if profit_target is not None and stop_loss is not None:
                    lines.append(f"  Target: ${profit_target:,.0f} | Stop: ${stop_loss:,.0f}")
                lines.append("")

            message = "\n".join(lines)
            return self.log_message(message, "AI_DECISION_SUMMARY")
        except Exception as e:
            self.logger.error(f"Error logging AI decision summary: {e}")
            return False

    def log_trade_event(self, event_type: str, coin: str, details: dict) -> bool:
        """Log a trade-related event.

        Args:
            event_type: Type of event ("ENTRY", "EXIT", etc.)
            coin: Coin symbol
            details: Event-specific details

        Returns:
            True if successful, False otherwise
        """
        try:
            lines = [f"{event_type} - {coin}"]
            if event_type == "ENTRY":
                side = details.get('side', 'long')
                entry_price = details.get('entry_price', 0)
                quantity = details.get('quantity', 0)
                notional = details.get('notional', entry_price * quantity)

                lines.append(f"Bot opened a {side} position on {coin}!")
                lines.append(f"Entry price: ${entry_price:.2f}")
                lines.append(f"Quantity: {quantity}")
                lines.append(f"Notional: ${notional:,.0f}")
                if 'leverage' in details:
                    lines.append(f"Leverage: {details['leverage']}x")

            elif event_type == "EXIT":
                lines.append(f"Bot completed a long trade on {coin}!")
                entry_price = details.get('entry_price', 0)
                exit_price = details.get('exit_price', 0)
                quantity = details.get('quantity', 0)
                notional_entry = entry_price * quantity
                notional_exit = exit_price * quantity
                holding_time = self._format_holding_time(details.get('holding_time_minutes', 0))
                net_pnl = details.get('net_pnl', 0)
                reason = details.get('reason', '')

                lines.append(f"Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f}")
                lines.append(f"Quantity: {quantity}")
                lines.append(f"Notional: ${notional_entry:,.0f} → ${notional_exit:,.0f}")
                lines.append(f"Holding time: {holding_time}")
                lines.append(f"Net P&L: ${net_pnl:.2f}")
                lines.append(f"Reason: {reason}")

            message = "\n".join(lines)
            return self.log_message(message, "TRADE_EVENT")
        except Exception as e:
            self.logger.error(f"Error logging trade event: {e}")
            return False

    def log_risk_event(self, event_type: str, details: dict) -> bool:
        """Log a risk management event.

        Args:
            event_type: Type of risk event ("DRAWDOWN_ALERT", etc.)
            details: Event-specific details

        Returns:
            True if successful, False otherwise
        """
        try:
            lines = [event_type]
            if event_type == "DRAWDOWN_ALERT":
                current_drawdown_pct = details.get('current_drawdown_pct', 0)
                threshold_pct = details.get('threshold_pct', 0)
                peak_value = details.get('peak_value', 0)
                current_value = details.get('current_value', 0)
                distance_to_threshold = details.get('distance_to_threshold', 0)

                lines.append(f"Account drawdown: {current_drawdown_pct:.1f}% (threshold: {threshold_pct:.1f}%)")
                lines.append(f"Peak value: ${peak_value:,.0f} → Current: ${current_value:,.0f}")
                lines.append(f"Distance to kill-switch: {distance_to_threshold:.1f}%")

            message = "\n".join(lines)
            return self.log_message(message, "RISK_EVENT")
        except Exception as e:
            self.logger.error(f"Error logging risk event: {e}")
            return False

    def _check_and_rotate_log(self) -> None:
        """Check if log file needs rotation and perform rotation if necessary."""
        try:
            if os.path.exists(self.log_path):
                size = os.path.getsize(self.log_path)
                if size > self.max_size_bytes:
                    # Derive directory, base name, and extension
                    log_dir = os.path.dirname(self.log_path)
                    log_filename = os.path.basename(self.log_path)
                    base, ext = os.path.splitext(log_filename)

                    # Rotate log
                    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    rotated_filename = f"{base}_{timestamp}{ext}"
                    rotated_path = os.path.join(log_dir, rotated_filename)
                    os.rename(self.log_path, rotated_path)

                    # Create new log file with rotation message
                    with open(self.log_path, 'w', encoding='utf-8') as f:
                        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [SYSTEM]\n")
                        f.write("Log rotated\n\n---\n")

                    # Clean up old rotated logs
                    # List files matching pattern: base_*ext
                    log_files = [
                        f for f in os.listdir(log_dir)
                        if f.startswith(f"{base}_") and f.endswith(ext)
                    ]
                    log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
                    retention_count = int(os.getenv("CHAT_LOG_RETENTION_COUNT", 10))
                    while len(log_files) > retention_count:
                        oldest = log_files.pop(0)
                        os.remove(os.path.join(log_dir, oldest))
        except Exception as e:
            self.logger.warning(f"Error during log rotation: {e}")

    def get_recent_messages(self, count: int = 50) -> list[str]:
        """Read the last N messages from the log file.

        Args:
            count: Number of recent messages to return

        Returns:
            List of message strings
        """
        try:
            if not os.path.exists(self.log_path):
                return []
            with open(self.log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            messages = content.split('---\n')
            if messages and messages[-1].strip() == '':
                messages.pop()
            return messages[-count:] if len(messages) > count else messages
        except Exception as e:
            self.logger.error(f"Error reading recent messages: {e}")
            return []

    def search_messages(self, keyword: str, max_results: int = 100) -> list[str]:
        """Search log file for messages containing a keyword.

        Args:
            keyword: Keyword to search for
            max_results: Maximum number of results to return

        Returns:
            List of matching message strings
        """
        try:
            if not os.path.exists(self.log_path):
                return []
            with open(self.log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            messages = content.split('---\n')
            if messages and messages[-1].strip() == '':
                messages.pop()
            keyword_lower = keyword.lower()
            matching = [msg for msg in messages if keyword_lower in msg.lower()]
            return matching[:max_results]
        except Exception as e:
            self.logger.error(f"Error searching messages: {e}")
            return []

    def export_to_json(self, output_path: str, start_date: str = None, end_date: str = None) -> bool:
        """Export chat log to structured JSON format.

        Args:
            output_path: Path to output JSON file
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)

        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.log_path):
                return False
            with open(self.log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            messages = content.split('---\n')
            if messages and messages[-1].strip() == '':
                messages.pop()
            parsed_messages = []
            for msg in messages:
                parsed = self._parse_log_entry(msg)
                if parsed:
                    if start_date or end_date:
                        msg_date = parsed['timestamp'][:10]  # YYYY-MM-DD
                        if start_date and msg_date < start_date:
                            continue
                        if end_date and msg_date > end_date:
                            continue
                    parsed_messages.append(parsed)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_messages, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {e}")
            return False

    def _parse_log_entry(self, entry: str) -> dict:
        """Parse a log entry string into structured dict.

        Args:
            entry: Raw log entry string

        Returns:
            Parsed dict with timestamp, category, message
        """
        try:
            lines = entry.strip().split('\n')
            if len(lines) < 2:
                return None
            first_line = lines[0]
            if not first_line.startswith('[') or ']' not in first_line:
                return None
            timestamp_str = first_line.split(']')[0][1:]
            category_str = first_line.split(']')[1].strip()[1:-1]  # Remove brackets
            message = '\n'.join(lines[1:]).strip()
            return {
                "timestamp": timestamp_str.replace(' ', 'T'),
                "category": category_str,
                "message": message
            }
        except Exception:
            return None

    def _format_timestamp(self, dt: datetime) -> str:
        """Format datetime for log entries.

        Args:
            dt: Datetime object

        Returns:
            Formatted timestamp string
        """
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    def _format_holding_time(self, minutes: float) -> str:
        """Format holding time as human-readable string.

        Args:
            minutes: Holding time in minutes

        Returns:
            Formatted string like "43H 44M"
        """
        days = int(minutes // 1440)
        hours = int((minutes % 1440) // 60)
        mins = int(minutes % 60)
        parts = []
        if days > 0:
            parts.append(f"{days}D")
        if hours > 0:
            parts.append(f"{hours}H")
        if mins > 0 or not parts:
            parts.append(f"{mins}M")
        return ' '.join(parts)

    def _format_currency(self, amount: float) -> str:
        """Format currency amounts.

        Args:
            amount: Amount to format

        Returns:
            Formatted string like "$1,234.56"
        """
        return f"${amount:,.2f}"