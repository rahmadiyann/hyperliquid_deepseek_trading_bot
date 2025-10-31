import os
import json
import csv
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dotenv import load_dotenv


class TradeLogger:
    """Logger for AI trading decisions and trade executions using SQLite database.

    Provides structured logging of AI decisions, trade executions, and performance analysis.
    Supports JSON/CSV export for external analysis.
    """

    def __init__(self):
        """Initialize TradeLogger with database connection and table setup."""
        # Load environment variables
        load_dotenv()

        # Read database path from environment
        self.db_path = os.getenv("DATABASE_PATH", "./data/trades.db")

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Initialize database (create tables if needed)
        self._initialize_database()

        self.logger.info(f"TradeLogger initialized with database: {self.db_path}")

    def _initialize_database(self) -> None:
        """Create database tables and indices on first run."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create ai_decisions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ai_decisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        invocation_number INTEGER NOT NULL,
                        coin TEXT NOT NULL,
                        signal TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        confidence REAL NOT NULL,
                        reasoning TEXT NOT NULL,
                        profit_target REAL,
                        stop_loss REAL,
                        invalidation_condition TEXT,
                        leverage INTEGER,
                        risk_usd REAL,
                        executed INTEGER DEFAULT 0
                    )
                """)

                # Create trade_executions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trade_executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        coin TEXT NOT NULL,
                        side TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        entry_time TEXT NOT NULL,
                        exit_price REAL,
                        exit_time TEXT,
                        quantity REAL NOT NULL,
                        notional_entry REAL NOT NULL,
                        notional_exit REAL,
                        holding_time_minutes REAL,
                        net_pnl REAL,
                        exit_reason TEXT,
                        related_decision_id INTEGER,
                        FOREIGN KEY (related_decision_id) REFERENCES ai_decisions (id)
                    )
                """)

                # Create indices for common queries
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_coin ON ai_decisions(coin)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON ai_decisions(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_executions_coin ON trade_executions(coin)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_executions_timestamp ON trade_executions(timestamp)")

                conn.commit()

            self.logger.info("Database tables and indices created successfully")

        except sqlite3.Error as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def log_ai_decision(self, decision: dict, invocation_number: int) -> int:
        """Log an AI trading decision to the database.

        Args:
            decision: Dict containing decision details
            invocation_number: Sequential invocation number

        Returns:
            Decision ID if successful, -1 on failure
        """
        try:
            # Validate decision
            if not self._validate_decision(decision):
                self.logger.warning("Invalid decision structure, skipping log")
                return -1

            # Extract fields
            coin = decision["coin"]
            signal = decision["signal"]
            confidence = decision["confidence"]

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO ai_decisions (
                        timestamp, invocation_number, coin, signal, quantity, confidence,
                        reasoning, profit_target, stop_loss, invalidation_condition,
                        leverage, risk_usd, executed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """, (
                    datetime.now().isoformat(),
                    invocation_number,
                    coin,
                    signal,
                    decision.get("quantity", 0),
                    confidence,
                    decision.get("reasoning", ""),
                    decision.get("profit_target"),
                    decision.get("stop_loss"),
                    decision.get("invalidation_condition"),
                    decision.get("leverage"),
                    decision.get("risk_usd")
                ))

                decision_id = cursor.lastrowid
                conn.commit()

            self.logger.info(f"Logged AI decision: {coin} {signal} (confidence: {confidence:.2f})")
            return decision_id

        except sqlite3.Error as e:
            self.logger.error(f"Database error logging AI decision: {str(e)}")
            return -1
        except Exception as e:
            self.logger.error(f"Error logging AI decision: {str(e)}")
            return -1

    def log_trade_execution(self, trade: dict, related_decision_id: int = None) -> int:
        """Log a trade execution (entry or exit) to the database.

        Args:
            trade: Dict containing trade details
            related_decision_id: Optional ID of related AI decision

        Returns:
            Trade execution ID if successful, -1 on failure
        """
        try:
            # Validate trade
            if not self._validate_trade(trade):
                self.logger.warning("Invalid trade structure, skipping log")
                return -1

            coin = trade["coin"]

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if "entry_price" in trade:
                    # New position entry
                    entry_price = trade["entry_price"]
                    quantity = trade["quantity"]
                    side = trade["side"]
                    notional_entry = trade.get("notional_entry", entry_price * quantity)

                    cursor.execute("""
                        INSERT INTO trade_executions (
                            timestamp, coin, side, entry_price, entry_time, quantity,
                            notional_entry, related_decision_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now().isoformat(),
                        coin,
                        side,
                        entry_price,
                        datetime.now().isoformat(),
                        quantity,
                        notional_entry,
                        related_decision_id
                    ))

                    trade_id = cursor.lastrowid
                    self.logger.info(f"Logged trade entry: {coin} {side} @ {entry_price}")

                elif "exit_price" in trade:
                    # Position exit - find open position for this coin (and side if provided)
                    side = trade.get("side")
                    position_id = trade.get("position_id")

                    if position_id is not None:
                        # Use provided position_id for exact matching
                        cursor.execute("""
                            SELECT id, entry_time FROM trade_executions
                            WHERE id = ? AND exit_price IS NULL
                        """, (position_id,))
                    elif side is not None:
                        # Match by coin and side
                        cursor.execute("""
                            SELECT id, entry_time FROM trade_executions
                            WHERE coin = ? AND side = ? AND exit_price IS NULL
                            ORDER BY entry_time DESC LIMIT 1
                        """, (coin, side))
                    else:
                        # Fallback to coin-only matching
                        cursor.execute("""
                            SELECT id, entry_time FROM trade_executions
                            WHERE coin = ? AND exit_price IS NULL
                            ORDER BY entry_time DESC LIMIT 1
                        """, (coin,))

                    open_position = cursor.fetchone()
                    if not open_position:
                        self.logger.warning(f"No open position found for {coin}, skipping exit log")
                        return -1

                    position_id, entry_time_str = open_position
                    exit_price = trade["exit_price"]
                    exit_time_str = trade["exit_time"]
                    net_pnl = trade["net_pnl"]
                    exit_reason = trade["exit_reason"]

                    # Calculate holding time
                    entry_time = datetime.fromisoformat(entry_time_str)
                    exit_time = datetime.fromisoformat(exit_time_str)
                    holding_time_minutes = (exit_time - entry_time).total_seconds() / 60

                    # Calculate notional exit
                    cursor.execute("SELECT quantity FROM trade_executions WHERE id = ?", (position_id,))
                    quantity_row = cursor.fetchone()
                    quantity = quantity_row[0] if quantity_row else 0
                    notional_exit = exit_price * abs(quantity)

                    # Update position with exit details
                    cursor.execute("""
                        UPDATE trade_executions SET
                            exit_price = ?, exit_time = ?, notional_exit = ?,
                            holding_time_minutes = ?, net_pnl = ?, exit_reason = ?
                        WHERE id = ?
                    """, (
                        exit_price, exit_time_str, notional_exit,
                        holding_time_minutes, net_pnl, exit_reason, position_id
                    ))

                    trade_id = position_id
                    self.logger.info(f"Logged trade exit: {coin} @ {exit_price}, P&L: {net_pnl}")

                    # Mark related decision as executed if provided
                    if related_decision_id is not None:
                        cursor.execute(
                            "UPDATE ai_decisions SET executed = 1 WHERE id = ?",
                            (related_decision_id,)
                        )

                conn.commit()
                return trade_id

        except sqlite3.Error as e:
            self.logger.error(f"Database error logging trade execution: {str(e)}")
            return -1
        except Exception as e:
            self.logger.error(f"Error logging trade execution: {str(e)}")
            return -1

    def mark_decision_executed(self, decision_id: int) -> bool:
        """Mark an AI decision as executed.

        Args:
            decision_id: ID of the decision to mark

        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE ai_decisions SET executed = 1 WHERE id = ?",
                    (decision_id,)
                )
                conn.commit()

            self.logger.info(f"Marked decision {decision_id} as executed")
            return True

        except sqlite3.Error as e:
            self.logger.error(f"Database error marking decision executed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error marking decision executed: {str(e)}")
            return False

    def get_decisions_by_date(self, start_date: str, end_date: str = None) -> list[dict]:
        """Query AI decisions within a date range.

        Args:
            start_date: Start date in ISO format
            end_date: End date in ISO format (optional, defaults to now)

        Returns:
            List of decision dicts
        """
        try:
            if end_date is None:
                end_date = datetime.now().isoformat()

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM ai_decisions
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp DESC
                """, (start_date, end_date))

                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        except sqlite3.Error as e:
            self.logger.error(f"Database error querying decisions: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"Error querying decisions: {str(e)}")
            return []

    def get_trades_by_coin(self, coin: str, limit: int = 100) -> list[dict]:
        """Query trade executions for a specific coin.

        Args:
            coin: Coin symbol
            limit: Maximum number of results

        Returns:
            List of trade dicts
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM trade_executions
                    WHERE coin = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (coin, limit))

                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        except sqlite3.Error as e:
            self.logger.error(f"Database error querying trades for {coin}: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"Error querying trades for {coin}: {str(e)}")
            return []

    def get_open_positions(self) -> list[dict]:
        """Query all currently open positions.

        Returns:
            List of open position dicts
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM trade_executions
                    WHERE exit_price IS NULL
                    ORDER BY entry_time DESC
                """)

                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        except sqlite3.Error as e:
            self.logger.error(f"Database error querying open positions: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"Error querying open positions: {str(e)}")
            return []

    def get_all_trades(self, limit: int = None) -> list[dict]:
        """Query all trade executions without coin filter.

        Args:
            limit: Optional maximum number of results

        Returns:
            List of trade dicts
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                if limit is not None:
                    cursor.execute("""
                        SELECT * FROM trade_executions
                        ORDER BY entry_time DESC
                        LIMIT ?
                    """, (limit,))
                else:
                    cursor.execute("""
                        SELECT * FROM trade_executions
                        ORDER BY entry_time DESC
                    """)

                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        except sqlite3.Error as e:
            self.logger.error(f"Database error querying all trades: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"Error querying all trades: {str(e)}")
            return []

    def get_completed_trades(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Query completed trades with UI-normalized fields.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of normalized trade dicts with UI-friendly fields
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM trade_executions
                    WHERE exit_price IS NOT NULL
                    ORDER BY exit_time DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))

                rows = cursor.fetchall()
                trades = []

                for row in rows:
                    trade = dict(row)

                    # Normalize fields for UI consumption
                    # Map holding_time_minutes to holding_time string
                    holding_time_minutes = trade.get('holding_time_minutes', 0)
                    if holding_time_minutes is not None:
                        if holding_time_minutes < 60:
                            trade['holding_time'] = f"{int(holding_time_minutes)} min"
                        else:
                            hours = holding_time_minutes / 60
                            trade['holding_time'] = f"{hours:.1f} hr"
                    else:
                        trade['holding_time'] = "N/A"

                    # Add date field from exit_time for JSON serialization
                    exit_time = trade.get('exit_time')
                    if exit_time:
                        trade['date'] = exit_time  # ISO 8601 format, JSON-serializable
                    else:
                        trade['date'] = None

                    trades.append(trade)

                return trades

        except sqlite3.Error as e:
            self.logger.error(f"Database error querying completed trades: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"Error querying completed trades: {str(e)}")
            return []

    def get_performance_summary(self, start_date: str = None, end_date: str = None) -> dict:
        """Calculate performance metrics from trade history.

        Args:
            start_date: Start date in ISO format (optional)
            end_date: End date in ISO format (optional)

        Returns:
            Dict with performance metrics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Build query for closed trades
                query = """
                    SELECT net_pnl, holding_time_minutes
                    FROM trade_executions
                    WHERE exit_price IS NOT NULL
                """
                params = []

                if start_date:
                    query += " AND exit_time >= ?"
                    params.append(start_date)
                if end_date:
                    query += " AND exit_time <= ?"
                    params.append(end_date)

                cursor.execute(query, params)
                trades = cursor.fetchall()

                if not trades:
                    return {
                        "total_trades": 0,
                        "winning_trades": 0,
                        "losing_trades": 0,
                        "win_rate": 0.0,
                        "total_pnl": 0.0,
                        "avg_pnl_per_trade": 0.0,
                        "avg_holding_time": 0.0,
                        "best_trade": 0.0,
                        "worst_trade": 0.0,
                        "profit_factor": 0.0
                    }

                # Calculate metrics
                pnls = [row[0] for row in trades]
                holding_times = [row[1] for row in trades if row[1] is not None]

                total_trades = len(pnls)
                winning_trades = sum(1 for pnl in pnls if pnl > 0)
                losing_trades = sum(1 for pnl in pnls if pnl < 0)
                win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
                total_pnl = sum(pnls)
                avg_pnl_per_trade = total_pnl / total_trades
                avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0.0
                best_trade = max(pnls) if pnls else 0.0
                worst_trade = min(pnls) if pnls else 0.0

                # Profit factor
                winning_pnl = sum(pnl for pnl in pnls if pnl > 0)
                losing_pnl = abs(sum(pnl for pnl in pnls if pnl < 0))
                profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')

                summary = {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": win_rate,
                    "total_pnl": total_pnl,
                    "avg_pnl_per_trade": avg_pnl_per_trade,
                    "avg_holding_time": avg_holding_time,
                    "best_trade": best_trade,
                    "worst_trade": worst_trade,
                    "profit_factor": profit_factor
                }

                self.logger.info(f"Performance summary: {total_trades} trades, win rate {win_rate:.2%}, total P&L ${total_pnl:.2f}")
                return summary

        except sqlite3.Error as e:
            self.logger.error(f"Database error calculating performance: {str(e)}")
            return {}
        except Exception as e:
            self.logger.error(f"Error calculating performance: {str(e)}")
            return {}

    def export_to_json(self, output_path: str, start_date: str = None, end_date: str = None) -> bool:
        """Export database records to JSON file.

        Args:
            output_path: Path to output JSON file
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get data
            decisions = self.get_decisions_by_date(start_date or "1970-01-01", end_date)
            trades = self.get_all_trades()  # Get all trades
            performance = self.get_performance_summary(start_date, end_date)

            # Filter trades by date if needed
            if start_date or end_date:
                filtered_trades = []
                for trade in trades:
                    trade_time = trade.get("exit_time") or trade.get("entry_time")
                    if trade_time:
                        if start_date and trade_time < start_date:
                            continue
                        if end_date and trade_time > end_date:
                            continue
                    filtered_trades.append(trade)
                trades = filtered_trades

            # Prepare export data
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "date_range": {
                    "start": start_date,
                    "end": end_date
                },
                "ai_decisions": decisions,
                "trade_executions": trades,
                "performance_summary": performance
            }

            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)

            self.logger.info(f"Exported {len(decisions)} decisions and {len(trades)} trades to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {str(e)}")
            return False

    def export_to_csv(self, output_dir: str, start_date: str = None, end_date: str = None) -> bool:
        """Export database records to CSV files.

        Args:
            output_dir: Directory to save CSV files
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Export decisions
            decisions = self.get_decisions_by_date(start_date or "1970-01-01", end_date)
            decisions_file = os.path.join(output_dir, "ai_decisions.csv")
            if decisions:
                with open(decisions_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=decisions[0].keys())
                    writer.writeheader()
                    writer.writerows(decisions)

            # Export trades
            trades = self.get_all_trades()  # Get all trades
            if start_date or end_date:
                filtered_trades = []
                for trade in trades:
                    trade_time = trade.get("exit_time") or trade.get("entry_time")
                    if trade_time:
                        if start_date and trade_time < start_date:
                            continue
                        if end_date and trade_time > end_date:
                            continue
                    filtered_trades.append(trade)
                trades = filtered_trades

            trades_file = os.path.join(output_dir, "trade_executions.csv")
            if trades:
                with open(trades_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=trades[0].keys())
                    writer.writeheader()
                    writer.writerows(trades)

            self.logger.info(f"Exported to CSV: {decisions_file}, {trades_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {str(e)}")
            return False

    def _validate_decision(self, decision: dict) -> bool:
        """Validate decision dict structure.

        Args:
            decision: Decision dict to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["coin", "signal", "confidence", "reasoning"]
        for field in required_fields:
            if field not in decision:
                self.logger.warning(f"Missing required field '{field}' in decision")
                return False

        signal = decision["signal"].lower()
        if signal not in ["buy", "sell", "hold"]:
            self.logger.warning(f"Invalid signal '{signal}' in decision")
            return False

        confidence = decision.get("confidence", 0)
        if not (0 <= confidence <= 1):
            self.logger.warning(f"Invalid confidence {confidence} in decision")
            return False

        return True

    def _validate_trade(self, trade: dict) -> bool:
        """Validate trade dict structure.

        Args:
            trade: Trade dict to validate

        Returns:
            True if valid, False otherwise
        """
        if "coin" not in trade:
            self.logger.warning("Missing 'coin' field in trade")
            return False

        if "entry_price" in trade:
            required_entry = ["side", "quantity"]
            for field in required_entry:
                if field not in trade:
                    self.logger.warning(f"Missing required field '{field}' in entry trade")
                    return False
        elif "exit_price" in trade:
            required_exit = ["exit_time", "net_pnl", "exit_reason"]
            for field in required_exit:
                if field not in trade:
                    self.logger.warning(f"Missing required field '{field}' in exit trade")
                    return False
        else:
            self.logger.warning("Trade must contain either 'entry_price' or 'exit_price'")
            return False

        return True