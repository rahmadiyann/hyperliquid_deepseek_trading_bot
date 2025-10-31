import os
import json
import time
import signal
import logging
from datetime import datetime
from dotenv import load_dotenv
from src.exchange.hyperliquid_client import HyperliquidClient
from src.ai.deepseek_engine import DeepSeekEngine
from src.risk.risk_manager import RiskManager
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.ml_classifier import MLClassifierStrategy
# Try to import loggers (may not be implemented yet)
try:
    from src.logging.trade_logger import TradeLogger
    from src.logging.chat_logger import ChatLogger
except ImportError:
    TradeLogger = None
    ChatLogger = None


class TradingBot:
    """Main trading bot that orchestrates all components in a continuous loop."""

    def __init__(self):
        """Initialize the trading bot with all components and configuration."""
        # Load environment variables
        load_dotenv()

        # Read configuration
        self.interval_minutes = int(os.getenv("TRADING_INTERVAL_MINUTES", "3"))
        symbols_str = os.getenv("TRADING_SYMBOLS", "BTC,ETH,SOL,BNB,XRP,DOGE")
        self.symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        self.state_file_path = os.getenv("STATE_FILE_PATH", "./data/bot_state.json")
        self.heartbeat_path = os.getenv("HEARTBEAT_PATH", "./data/health/heartbeat")

        # Order execution configuration
        self.order_execution_type = os.getenv("ORDER_EXECUTION_TYPE", "market").lower()
        self.entry_cross_pct = float(os.getenv("ENTRY_CROSS_PCT", "0.001"))  # 0.1% default

        # Validate order execution type
        valid_execution_types = ["market", "limit_ioc", "limit_gtc"]
        if self.order_execution_type not in valid_execution_types:
            self.logger.warning(f"Invalid ORDER_EXECUTION_TYPE '{self.order_execution_type}', defaulting to 'market'")
            self.order_execution_type = "market"

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize all modules
        self.client = HyperliquidClient()
        self.ai_engine = DeepSeekEngine()
        self.risk_manager = RiskManager(self.client)
        
        # Initialize strategies (optional - may not be implemented)
        try:
            self.momentum_strategy = MomentumStrategy()
        except Exception as e:
            self.logger.warning(f"Could not initialize MomentumStrategy: {e}")
            self.momentum_strategy = None
        
        try:
            self.mean_reversion_strategy = MeanReversionStrategy()
        except Exception as e:
            self.logger.warning(f"Could not initialize MeanReversionStrategy: {e}")
            self.mean_reversion_strategy = None
        
        try:
            self.ml_strategy = MLClassifierStrategy()
        except Exception as e:
            self.logger.warning(f"Could not initialize MLClassifierStrategy: {e}")
            self.ml_strategy = None

        # Initialize loggers with graceful fallback
        self.trade_logger = None
        self.chat_logger = None
        if TradeLogger is not None:
            try:
                self.trade_logger = TradeLogger()
            except Exception as e:
                self.logger.warning(f"Could not initialize TradeLogger: {e}")
        else:
            self.logger.warning("TradeLogger not available, trade logging disabled")
        
        if ChatLogger is not None:
            try:
                self.chat_logger = ChatLogger()
            except Exception as e:
                self.logger.warning(f"Could not initialize ChatLogger: {e}")
        else:
            self.logger.warning("ChatLogger not available, chat logging disabled")

        # Initialize state
        self.position_metadata = {}  # symbol -> {stop_loss, profit_target, invalidation_condition, entry_time, decision_id}
        self.running = False
        self.iteration_count = 0
        self.start_time = datetime.now()

        # Create health directory for heartbeat
        heartbeat_dir = os.path.dirname(self.heartbeat_path)
        if heartbeat_dir:
            os.makedirs(heartbeat_dir, exist_ok=True)

        # Load persisted state
        self._load_state()

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info("TradingBot initialized successfully")
        self.logger.info(f"Configuration: symbols={self.symbols}, interval={self.interval_minutes}min, state_file={self.state_file_path}")
        self.logger.info(f"Order execution: type={self.order_execution_type}, entry_cross_pct={self.entry_cross_pct*100:.2f}%")

        # Write initial heartbeat
        self._update_heartbeat()

    def run(self) -> None:
        """Main entry point to start the trading loop."""
        self.running = True
        self.start_time = datetime.now()
        
        self.logger.info("Starting trading bot...")
        self.logger.info(f"Configuration summary: {len(self.symbols)} symbols, {self.interval_minutes}min intervals")
        
        try:
            while self.running:
                iteration_start = time.time()
                
                try:
                    self._run_iteration()
                except Exception as e:
                    self.logger.error(f"Error in iteration {self.iteration_count}: {str(e)}", exc_info=True)
                    # Continue to next iteration
                
                # Calculate sleep time
                elapsed_time = time.time() - iteration_start
                sleep_time = (self.interval_minutes * 60) - elapsed_time
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    self.logger.warning(f"Iteration {self.iteration_count} took longer than interval ({elapsed_time:.1f}s > {self.interval_minutes*60}s)")
                
                self.iteration_count += 1
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        
        # Shutdown
        self._shutdown()

    def _run_iteration(self) -> None:
        """Execute one complete trading cycle."""
        self.logger.info(f"Starting iteration #{self.iteration_count}")
        
        # Step 1: Fetch market data
        market_data = {}
        try:
            # Get current prices and funding rates
            market_info = self.client.get_market_data(self.symbols)
            
            for symbol in self.symbols:
                # Fetch candles
                candles_data = self._fetch_candles_for_symbol(symbol)
                
                # Calculate indicators
                indicators_3m = self._calculate_indicators_for_candles(candles_data.get("candles_3m", []))
                indicators_4h = self._calculate_indicators_for_candles(candles_data.get("candles_4h", []))
                
                # Get open interest and funding rate
                open_interest = 0.0  # Not available in current API
                funding_rate = market_info.get(symbol, {}).get("funding", 0.0)
                
                market_data[symbol] = {
                    "candles_3m": candles_data.get("candles_3m", []),
                    "candles_4h": candles_data.get("candles_4h", []),
                    "indicators_3m": indicators_3m,
                    "indicators_4h": indicators_4h,
                    "open_interest": open_interest,
                    "funding_rate": funding_rate,
                }
            
            self.logger.info(f"Fetched market data for {len(market_data)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            return
        
        # Step 2: Get current positions and account state
        try:
            positions = self.client.get_open_positions()
            account_info = self.client.get_user_state()
            account_value = float(account_info.get("marginSummary", {}).get("accountValue", 0))

            self.logger.info(f"Current positions: {len(positions)}, Account value: ${account_value:,.2f}")

            # Reconcile position_metadata with live positions
            live_symbols = {pos.get("symbol") for pos in positions}
            stale_symbols = set(self.position_metadata.keys()) - live_symbols

            for symbol in stale_symbols:
                self.logger.info(f"Removing stale position_metadata entry for {symbol}")
                del self.position_metadata[symbol]

        except Exception as e:
            self.logger.error(f"Error getting positions/account state: {str(e)}")
            return
        
        # Step 3: Risk checks
        try:
            drawdown_result = self.risk_manager.check_drawdown_limit(account_value)
            kill_switch_active = drawdown_result["kill_switch"]
            
            if kill_switch_active:
                self.logger.critical("Kill-switch active - skipping trading decisions")
            
            leverage_result = self.risk_manager.check_max_leverage(positions, account_value)
            if not leverage_result["within_limits"]:
                self.logger.warning(f"Leverage violations: {leverage_result['violations']}")
            
            risk_summary = self.risk_manager.get_risk_summary(positions, account_value)
            self.logger.info(f"Risk summary: effective_leverage={risk_summary['effective_leverage']:.2f}x, drawdown={risk_summary['current_drawdown_pct']:.2f}%")
            
        except Exception as e:
            self.logger.error(f"Error in risk checks: {str(e)}")
            kill_switch_active = True  # Be safe
        
        # Step 4: Enforce stop-losses
        try:
            current_prices = {symbol: market_info.get(symbol, {}).get("mark", 0) for symbol in self.symbols}
            position_stops = {symbol: meta.get("stop_loss", 0) for symbol, meta in self.position_metadata.items()}

            stop_loss_result = self.risk_manager.enforce_stop_loss(positions, current_prices, position_stops)

            for close_attempt in stop_loss_result["close_attempts"]:
                if close_attempt["success"]:
                    symbol = close_attempt["symbol"]
                    if symbol in self.position_metadata:
                        del self.position_metadata[symbol]

                    # Log to TradeLogger
                    if self.trade_logger:
                        try:
                            self.trade_logger.log_trade_exit(symbol, close_attempt, "stop_loss")
                        except Exception as e:
                            self.logger.warning(f"Failed to log trade exit: {e}")

                    # Log to ChatLogger
                    if self.chat_logger:
                        try:
                            message = f"Stop-loss triggered for {symbol}: closed position at ${close_attempt['trigger_price']:.2f}"
                            self.chat_logger.log_message(category="TRADE", message=message)
                        except Exception as e:
                            self.logger.warning(f"Failed to log chat message: {e}")

                    self.logger.info(f"Position closed via stop-loss: {symbol}")

            self.logger.info(f"Stop-loss enforcement: {stop_loss_result['positions_closed']} positions closed")

        except Exception as e:
            self.logger.error(f"Error enforcing stop-losses: {str(e)}")

        # Step 4b: Enforce take-profit targets
        try:
            position_targets = {symbol: meta.get("profit_target", 0) for symbol, meta in self.position_metadata.items()}

            take_profit_result = self.risk_manager.enforce_take_profit(positions, current_prices, position_targets)

            for close_attempt in take_profit_result["close_attempts"]:
                if close_attempt["success"]:
                    symbol = close_attempt["symbol"]
                    if symbol in self.position_metadata:
                        del self.position_metadata[symbol]

                    # Log to TradeLogger
                    if self.trade_logger:
                        try:
                            self.trade_logger.log_trade_exit(symbol, close_attempt, "take_profit")
                        except Exception as e:
                            self.logger.warning(f"Failed to log trade exit: {e}")

                    # Log to ChatLogger
                    if self.chat_logger:
                        try:
                            message = f"Take-profit triggered for {symbol}: closed position at ${close_attempt['trigger_price']:.2f} (target: ${close_attempt['target_price']:.2f})"
                            self.chat_logger.log_message(category="TRADE", message=message)
                        except Exception as e:
                            self.logger.warning(f"Failed to log chat message: {e}")

                    self.logger.info(f"Position closed via take-profit: {symbol}")

            self.logger.info(f"Take-profit enforcement: {take_profit_result['positions_closed']} positions closed")

        except Exception as e:
            self.logger.error(f"Error enforcing take-profit: {str(e)}")
        
        # Step 5: Generate AI trading decisions
        try:
            decisions = self.ai_engine.generate_trading_decision(market_data, positions, account_info)
            
            # Log decisions
            for decision in decisions:
                self.logger.info(f"AI Decision: {decision['coin']} {decision['signal']} (confidence: {decision['confidence']:.2f})")
                
                if self.trade_logger:
                    try:
                        self.trade_logger.log_ai_decision(decision)
                    except Exception as e:
                        self.logger.warning(f"Failed to log AI decision: {e}")
                
                if self.chat_logger:
                    try:
                        summary = f"AI Decision for {decision['coin']}: {decision['signal']} ({decision['confidence']:.2f}) - {decision['reasoning'][:100]}..."
                        self.chat_logger.log_message(category="AI_DECISION", message=summary)
                    except Exception as e:
                        self.logger.warning(f"Failed to log AI decision to chat: {e}")
            
        except Exception as e:
            self.logger.error(f"Error generating AI decisions: {str(e)}")
            decisions = []
        
        # Step 6: Optional strategy consultation
        enable_consultation = os.getenv("ENABLE_STRATEGY_CONSULTATION", "false").lower() == "true"
        if enable_consultation and decisions:
            try:
                for decision in decisions:
                    coin = decision["coin"]
                    ai_signal = decision["signal"]
                    
                    strategy_signals = {}
                    
                    if self.momentum_strategy and coin in market_data:
                        momentum_signal = self.momentum_strategy.generate_signal(market_data[coin])
                        strategy_signals["momentum"] = momentum_signal
                    
                    if self.mean_reversion_strategy and coin in market_data:
                        mean_rev_signal = self.mean_reversion_strategy.generate_signal(market_data[coin])
                        strategy_signals["mean_reversion"] = mean_rev_signal
                    
                    if self.ml_strategy and coin in market_data:
                        ml_signal = self.ml_strategy.generate_signal(market_data[coin])
                        strategy_signals["ml"] = ml_signal
                    
                    # Log agreement/disagreement
                    agreements = sum(1 for sig in strategy_signals.values() if sig["signal"] == ai_signal)
                    total_strategies = len(strategy_signals)
                    
                    self.logger.info(f"Strategy consultation for {coin}: AI={ai_signal}, strategies agree: {agreements}/{total_strategies}")
                    
                    for name, sig in strategy_signals.items():
                        if sig["signal"] != ai_signal:
                            self.logger.warning(f"  {name} disagrees: {sig['signal']} vs AI {ai_signal}")
                    
            except Exception as e:
                self.logger.error(f"Error in strategy consultation: {str(e)}")
        
        # Step 7: Execute trades
        trades_executed = 0
        try:
            for decision in decisions:
                coin = decision["coin"]
                signal = decision["signal"]
                
                if signal == "hold":
                    continue
                
                if kill_switch_active:
                    self.logger.warning(f"Skipping trade for {coin} - kill-switch active")
                    continue
                
                # Check if already have position
                existing_position = next((p for p in positions if p["symbol"] == coin), None)
                if existing_position:
                    # For now, skip if already have position (no pyramiding)
                    # Stop-loss logic handles closing opposite positions
                    self.logger.info(f"Skipping {coin} - already have position")
                    continue
                
                # Get current price
                current_price = market_info.get(coin, {}).get("mark", 0)
                if current_price <= 0:
                    self.logger.warning(f"Invalid price for {coin}: {current_price}")
                    continue
                
                # Calculate position size
                position_calc = self.risk_manager.calculate_position_size(decision, account_value, current_price)
                if not position_calc["approved"]:
                    self.logger.warning(f"Position size rejected for {coin}: {position_calc['rejection_reason']}")
                    continue
                
                quantity = position_calc["position_size"]
                side = "buy" if signal == "buy" else "sell"

                # Determine order type and price based on execution configuration
                if self.order_execution_type == "market":
                    # Emulate market order with aggressive limit IOC
                    if side == "buy":
                        order_price = current_price * (1 + self.entry_cross_pct)
                    else:  # sell
                        order_price = current_price * (1 - self.entry_cross_pct)
                    order_type = {"limit": {"tif": "Ioc"}}
                elif self.order_execution_type == "limit_ioc":
                    # Limit IOC with configurable offset
                    if side == "buy":
                        order_price = current_price * (1 + self.entry_cross_pct)
                    else:  # sell
                        order_price = current_price * (1 - self.entry_cross_pct)
                    order_type = {"limit": {"tif": "Ioc"}}
                else:  # limit_gtc
                    # GTC at mark price (current behavior)
                    order_price = current_price
                    order_type = {"limit": {"tif": "Gtc"}}

                # Place order
                order_response = self.client.place_order(coin, side, quantity, order_price, order_type)
                
                if order_response["success"]:
                    # Store position metadata
                    self.position_metadata[coin] = {
                        "stop_loss": decision["stop_loss"],
                        "profit_target": decision["profit_target"],
                        "invalidation_condition": decision["invalidation_condition"],
                        "entry_time": datetime.now().isoformat(),
                        "decision_id": f"decision_{self.iteration_count}_{coin}"
                    }
                    
                    trades_executed += 1
                    
                    # Log trade entry
                    if self.trade_logger:
                        try:
                            self.trade_logger.log_trade_entry(coin, decision, order_response, position_calc)
                        except Exception as e:
                            self.logger.warning(f"Failed to log trade entry: {e}")
                    
                    if self.chat_logger:
                        try:
                            message = f"Trade executed: {side.upper()} {quantity:.4f} {coin} @ ${order_price:.2f} (type: {self.order_execution_type})"
                            self.chat_logger.log_message(category="TRADE", message=message)
                        except Exception as e:
                            self.logger.warning(f"Failed to log trade to chat: {e}")

                    self.logger.info(f"Trade executed: {side.upper()} {quantity:.4f} {coin} @ ${order_price:.2f} (type: {self.order_execution_type}, mark: ${current_price:.2f})")
                else:
                    self.logger.error(f"Order failed for {coin}: {order_response['error']}")
            
            self.logger.info(f"Trade execution summary: {trades_executed} trades executed")
            
        except Exception as e:
            self.logger.error(f"Error executing trades: {str(e)}")
        
        # Step 8: Check invalidation conditions
        try:
            invalidations_triggered = 0
            for symbol, metadata in list(self.position_metadata.items()):
                condition = metadata.get("invalidation_condition", "")
                current_price = market_info.get(symbol, {}).get("mark", 0)
                
                if self._check_invalidation_condition(condition, current_price, symbol):
                    # Close position
                    position = next((p for p in positions if p["symbol"] == symbol), None)
                    if position:
                        side = "sell" if position["size"] > 0 else "buy"
                        quantity = abs(position["size"])
                        
                        order_response = self.client.place_order(symbol, side, quantity, current_price, {"limit": {"tif": "Ioc"}})
                        
                        if order_response["success"]:
                            del self.position_metadata[symbol]
                            invalidations_triggered += 1
                            
                            self.logger.warning(f"Invalidation triggered for {symbol}: {condition}")
                            
                            if self.chat_logger:
                                try:
                                    message = f"Invalidation closed {symbol}: {condition}"
                                    self.chat_logger.log_message(category="INVALIDATION", message=message)
                                except Exception as e:
                                    self.logger.warning(f"Failed to log invalidation: {e}")
                        else:
                            self.logger.error(f"Failed to close invalidated position {symbol}: {order_response['error']}")
            
            self.logger.info(f"Invalidation checks: {invalidations_triggered} positions closed")
            
        except Exception as e:
            self.logger.error(f"Error checking invalidations: {str(e)}")
        
        # Step 9: Save state
        try:
            self._save_state()
            self.logger.info("State saved to disk")
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
        
        # Step 10: Iteration complete
        self.logger.info(f"Iteration #{self.iteration_count} complete - positions: {len(positions)}, account: ${account_value:,.2f}")

        # Update heartbeat for liveness monitoring
        self._update_heartbeat()

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals for graceful exit."""
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        self.logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        self.running = False

    def _shutdown(self) -> None:
        """Perform graceful shutdown procedures."""
        self.logger.info("Starting shutdown procedure...")
        
        try:
            self._save_state()
            self.logger.info("State saved during shutdown")
        except Exception as e:
            self.logger.error(f"Error saving state during shutdown: {str(e)}")
        
        uptime = datetime.now() - self.start_time
        self.logger.info(f"Shutdown complete - total iterations: {self.iteration_count}, uptime: {uptime}")

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        try:
            if os.path.exists(self.state_file_path):
                with open(self.state_file_path, 'r') as f:
                    data = json.load(f)
                    self.position_metadata = data.get("position_metadata", {})
                self.logger.info(f"State loaded from {self.state_file_path}")
            else:
                self.logger.info("No state file found, starting with empty state")
        except Exception as e:
            self.logger.warning(f"Error loading state, starting fresh: {str(e)}")
            self.position_metadata = {}

    def _save_state(self) -> None:
        """Save current state to disk atomically."""
        try:
            # Create directory if needed
            dirpath = os.path.dirname(self.state_file_path)
            if dirpath:  # Only create directory if path has a directory component
                os.makedirs(dirpath, exist_ok=True)

            # Prepare data
            data = {
                "position_metadata": self.position_metadata,
                "last_updated": datetime.now().isoformat(),
                "iteration_count": self.iteration_count
            }

            # Atomic write
            temp_file = self.state_file_path + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            os.rename(temp_file, self.state_file_path)

            self.logger.debug(f"State saved to {self.state_file_path}")

        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
            raise

    def _update_heartbeat(self) -> None:
        """Update heartbeat file with current timestamp for liveness monitoring."""
        try:
            current_time = int(time.time())
            with open(self.heartbeat_path, 'w') as f:
                f.write(str(current_time))
            self.logger.debug(f"Heartbeat updated: {current_time}")
        except Exception as e:
            self.logger.warning(f"Failed to update heartbeat: {str(e)}")

    def _fetch_candles_for_symbol(self, symbol: str) -> dict:
        """Fetch historical candles for a single symbol."""
        try:
            candles_3m = self.client.get_candles(symbol, "3m", limit=100)
            candles_4h = self.client.get_candles(symbol, "4h", limit=50)

            return {
                "candles_3m": candles_3m,
                "candles_4h": candles_4h
            }
        except Exception as e:
            self.logger.warning(f"Error fetching candles for {symbol}: {str(e)}")
            return {"candles_3m": [], "candles_4h": []}

    def _calculate_indicators_for_candles(self, candles: list) -> dict:
        """Calculate technical indicators from candle data."""
        try:
            if not candles:
                return {}
            return self.ai_engine.calculate_indicators(candles)
        except Exception as e:
            self.logger.warning(f"Error calculating indicators: {str(e)}")
            return {}

    def _check_invalidation_condition(self, condition: str, current_price: float, symbol: str) -> bool:
        """Evaluate invalidation condition string."""
        try:
            if not condition or current_price <= 0:
                return False
            
            # Simple parser for conditions like "If price closes below 105000"
            condition_lower = condition.lower()
            
            if "below" in condition_lower and "price" in condition_lower:
                # Extract number after "below"
                parts = condition_lower.split("below")
                if len(parts) > 1:
                    threshold_str = parts[1].strip().split()[0]
                    try:
                        threshold = float(threshold_str.replace(",", "").replace("$", ""))
                        return current_price < threshold
                    except ValueError:
                        pass
            
            elif "above" in condition_lower and "price" in condition_lower:
                # Extract number after "above"
                parts = condition_lower.split("above")
                if len(parts) > 1:
                    threshold_str = parts[1].strip().split()[0]
                    try:
                        threshold = float(threshold_str.replace(",", "").replace("$", ""))
                        return current_price > threshold
                    except ValueError:
                        pass
            
            # If we can't parse, assume not invalidated
            self.logger.debug(f"Could not parse invalidation condition for {symbol}: {condition}")
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking invalidation for {symbol}: {str(e)}")
            return False

    def _format_trade_summary(self, decision: dict, order_response: dict, position_info: dict) -> str:
        """Format a human-readable trade summary."""
        try:
            coin = decision.get("coin", "UNKNOWN")
            signal = decision.get("signal", "hold")
            quantity = decision.get("quantity", 0)
            confidence = decision.get("confidence", 0)
            price = position_info.get("price", 0)
            
            return f"{signal.upper()} {quantity:.4f} {coin} @ ${price:.2f} (confidence: {confidence:.2f})"
        except Exception as e:
            return "Error formatting trade summary"