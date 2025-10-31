import os
import argparse
import logging
import time
import asyncio
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout

# Try to import internal modules (may not be available in all environments)
try:
    from src.logging import TradeLogger, ChatLogger
    from src.exchange import HyperliquidClient
except ImportError as e:
    logging.warning(f"Failed to import modules: {e}")
    TradeLogger = None
    ChatLogger = None
    HyperliquidClient = None


class DashboardData:
    """Helper class to fetch and aggregate all dashboard data."""

    def __init__(self, trade_logger: Optional[TradeLogger] = None, 
                 chat_logger: Optional[ChatLogger] = None, 
                 hyperliquid_client: Optional[HyperliquidClient] = None):
        self.trade_logger = trade_logger
        self.chat_logger = chat_logger
        self.hyperliquid_client = hyperliquid_client
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = 10  # seconds

    def _is_cached(self, key: str) -> bool:
        if key not in self._cache_time:
            return False
        return (datetime.now() - self._cache_time[key]).total_seconds() < self._cache_ttl

    def _normalize_position(self, pos: dict) -> dict:
        """Normalize position fields to ensure consistent UI keys.

        Args:
            pos: Raw position dict from client

        Returns:
            Normalized position dict with UI-friendly fields
        """
        # Map common client fields to UI expected keys
        normalized = {
            'coin': pos.get('symbol', pos.get('coin', '')),
            'quantity': abs(pos.get('size', pos.get('quantity', 0))),
            'entry_price': pos.get('entry_price', pos.get('entryPrice', 0)),
            'current_price': pos.get('current_price', pos.get('mark', pos.get('markPrice', 0))),
            'leverage': pos.get('leverage', 1),
        }

        # Calculate unrealized P&L if not provided
        if 'unrealized_pnl' in pos:
            normalized['unrealized_pnl'] = pos['unrealized_pnl']
        elif 'size' in pos and 'entry_price' in normalized and 'current_price' in normalized:
            size = pos['size']
            entry = normalized['entry_price']
            current = normalized['current_price']
            if entry > 0 and current > 0:
                # For longs (size > 0): PnL = size * (current - entry)
                # For shorts (size < 0): PnL = size * (current - entry) [negative size handles sign]
                normalized['unrealized_pnl'] = abs(size) * (current - entry) if size > 0 else abs(size) * (entry - current)
            else:
                normalized['unrealized_pnl'] = 0
        else:
            normalized['unrealized_pnl'] = 0

        # Calculate holding time if entry_time provided
        if 'entry_time' in pos:
            try:
                entry_time = datetime.fromisoformat(pos['entry_time'])
                holding_minutes = (datetime.now() - entry_time).total_seconds() / 60
                if holding_minutes < 60:
                    normalized['holding_time'] = f"{int(holding_minutes)} min"
                else:
                    hours = holding_minutes / 60
                    normalized['holding_time'] = f"{hours:.1f} hr"
            except:
                normalized['holding_time'] = "N/A"
        else:
            normalized['holding_time'] = pos.get('holding_time', 'N/A')

        return normalized

    def get_current_positions(self) -> list[dict]:
        """Fetch live positions from HyperliquidClient with current prices."""
        cache_key = 'positions'
        if self._is_cached(cache_key):
            return self._cache[cache_key]

        try:
            if self.hyperliquid_client:
                positions = self.hyperliquid_client.get_open_positions()
                # Normalize all positions for consistent UI fields
                normalized_positions = [self._normalize_position(pos) for pos in positions]
                self._cache[cache_key] = normalized_positions
                self._cache_time[cache_key] = datetime.now()
                return normalized_positions
        except Exception as e:
            logging.warning(f"Failed to fetch current positions: {e}")
        return []

    def get_completed_trades(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Fetch closed trades from TradeLogger.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of normalized trade dicts
        """
        # Include parameters in cache key
        cache_key = f'trades:{limit}:{offset}'
        if self._is_cached(cache_key):
            return self._cache[cache_key]

        try:
            if self.trade_logger:
                # Use TradeLogger API instead of direct SQLite
                trades = self.trade_logger.get_completed_trades(limit=limit, offset=offset)
                self._cache[cache_key] = trades
                self._cache_time[cache_key] = datetime.now()
                return trades
        except Exception as e:
            logging.warning(f"Failed to fetch completed trades: {e}")
        return []

    def get_performance_metrics(self) -> dict:
        """Get performance summary from TradeLogger."""
        cache_key = 'performance'
        if self._is_cached(cache_key):
            return self._cache[cache_key]

        try:
            if self.trade_logger:
                metrics = self.trade_logger.get_performance_summary()
                self._cache[cache_key] = metrics
                self._cache_time[cache_key] = datetime.now()
                return metrics
        except Exception as e:
            logging.warning(f"Failed to fetch performance metrics: {e}")
        return {}

    def get_ai_chat_log(self, count: int = 20) -> list[str]:
        """Get recent AI messages from ChatLogger.

        Args:
            count: Number of messages to retrieve

        Returns:
            List of message strings
        """
        # Include parameter in cache key
        cache_key = f'chat_log:{count}'
        if self._is_cached(cache_key):
            return self._cache[cache_key]

        try:
            if self.chat_logger:
                messages = self.chat_logger.get_recent_messages(count)
                self._cache[cache_key] = messages
                self._cache_time[cache_key] = datetime.now()
                return messages
        except Exception as e:
            logging.warning(f"Failed to fetch AI chat log: {e}")
        return []

    def get_account_summary(self) -> dict:
        """Get account value, cash, leverage from HyperliquidClient."""
        if self._is_cached('account'):
            return self._cache['account']
        
        try:
            if self.hyperliquid_client:
                state = self.hyperliquid_client.get_user_state()
                # Assume state has 'value', 'cash', 'leverage', 'drawdown'
                summary = {
                    'value': state.get('value', 0),
                    'cash': state.get('cash', 0),
                    'leverage': state.get('leverage', 0),
                    'drawdown': state.get('drawdown', 0)
                }
                self._cache['account'] = summary
                self._cache_time['account'] = datetime.now()
                return summary
        except Exception as e:
            logging.warning(f"Failed to fetch account summary: {e}")
        return {'value': 0, 'cash': 0, 'leverage': 0, 'drawdown': 0}

    def get_all_data(self) -> dict:
        """Aggregate all data into single dict for dashboard rendering."""
        return {
            'positions': self.get_current_positions(),
            'trades': self.get_completed_trades(),
            'performance': self.get_performance_metrics(),
            'chat_log': self.get_ai_chat_log(),
            'account': self.get_account_summary(),
            'last_updated': datetime.now().isoformat()
        }


class CLIDashboard:
    """Terminal-based dashboard using Rich library."""

    def __init__(self, data: DashboardData):
        self.data = data
        self.console = Console()

    def build_layout(self) -> Layout:
        """Build layout with current data.

        Returns:
            Rich Layout object ready for display
        """
        layout = Layout()
        layout.split_column(
            Layout(name="header"),
            Layout(name="body", ratio=3),
            Layout(name="footer")
        )

        layout["header"].update(self._render_account_summary(self.data.get_account_summary()))
        layout["body"].split_row(
            Layout(name="positions"),
            Layout(name="trades")
        )
        layout["body"]["positions"].update(self._render_positions_table(self.data.get_current_positions()))
        layout["body"]["trades"].split_column(
            Layout(name="performance"),
            Layout(name="chat")
        )
        layout["body"]["trades"]["performance"].update(self._render_performance_metrics(self.data.get_performance_metrics()))
        layout["body"]["trades"]["chat"].update(self._render_ai_chat_log(self.data.get_ai_chat_log()))
        layout["footer"].update(self._render_completed_trades_table(self.data.get_completed_trades(10)))

        return layout

    def _render_account_summary(self, data: dict) -> Panel:
        """Create Rich Panel with account info."""
        content = f"Account Value: ${data.get('value', 0):,.2f}\n"
        content += f"Available Cash: ${data.get('cash', 0):,.2f}\n"
        content += f"Effective Leverage: {data.get('leverage', 0):.2f}x\n"
        content += f"Current Drawdown: {data.get('drawdown', 0):.2f}%"
        return Panel(content, title="Account Summary")

    def _render_positions_table(self, positions: list[dict]) -> Table:
        """Create Rich Table with current positions."""
        table = Table(title="Current Positions")
        table.add_column("Symbol", style="cyan")
        table.add_column("Quantity", justify="right")
        table.add_column("Entry Price", justify="right")
        table.add_column("Current Price", justify="right")
        table.add_column("Unrealized P&L", justify="right")
        table.add_column("Leverage", justify="right")
        table.add_column("Holding Time", justify="right")
        
        for pos in positions:
            pnl = pos.get('unrealized_pnl', 0)
            pnl_style = "green" if pnl > 0 else "red"
            table.add_row(
                pos.get('coin', ''),
                f"{pos.get('quantity', 0):.4f}",
                f"${pos.get('entry_price', 0):.2f}",
                f"${pos.get('current_price', 0):.2f}",
                f"[bold {pnl_style}]${pnl:.2f}[/bold {pnl_style}]",
                f"{pos.get('leverage', 1)}x",
                pos.get('holding_time', '')
            )
        return table

    def _render_completed_trades_table(self, trades: list[dict]) -> Table:
        """Create Rich Table with completed trades."""
        table = Table(title="Recent Completed Trades")
        table.add_column("Symbol", style="cyan")
        table.add_column("Entry Price", justify="right")
        table.add_column("Exit Price", justify="right")
        table.add_column("Holding Time", justify="right")
        table.add_column("Net P&L", justify="right")
        table.add_column("Exit Reason", justify="left")
        
        for trade in trades[:10]:
            pnl = trade.get('net_pnl', 0)
            pnl_style = "green" if pnl > 0 else "red"
            table.add_row(
                trade.get('coin', ''),
                f"${trade.get('entry_price', 0):.2f}",
                f"${trade.get('exit_price', 0):.2f}",
                trade.get('holding_time_minutes', 0),
                f"[bold {pnl_style}]${pnl:.2f}[/bold {pnl_style}]",
                trade.get('exit_reason', '')
            )
        return table

    def _render_performance_metrics(self, metrics: dict) -> Panel:
        """Create Rich Panel with performance stats."""
        content = f"Total Trades: {metrics.get('total_trades', 0)}\n"
        content += f"Win Rate: {metrics.get('win_rate', 0):.2%}\n"
        content += f"Total P&L: ${metrics.get('total_pnl', 0):,.2f}\n"
        content += f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
        content += f"Best Trade: ${metrics.get('best_trade', 0):,.2f}\n"
        content += f"Worst Trade: ${metrics.get('worst_trade', 0):,.2f}"
        return Panel(content, title="Performance Metrics")

    def _render_ai_chat_log(self, messages: list[str]) -> Panel:
        """Create Rich Panel with recent AI messages."""
        content = "\n\n".join(messages[-5:])
        return Panel(content, title="Recent AI Messages")

    def run_once(self) -> None:
        """Render dashboard once and exit."""
        self.console.print(self.build_layout())

    def run_live(self, refresh_interval: int = 30) -> None:
        """Continuously update dashboard every N seconds."""
        try:
            with Live(self.build_layout(), refresh_per_second=1, screen=True) as live:
                while True:
                    time.sleep(refresh_interval)
                    live.update(self.build_layout())
        except KeyboardInterrupt:
            self.console.print("[bold red]Exiting dashboard...[/bold red]")


class WebDashboard:
    """FastAPI-based web dashboard."""

    def __init__(self, data: DashboardData, enable_websocket: bool = False):
        self.data = data
        self.enable_websocket = enable_websocket
        self.app = FastAPI(title="Trading Bot Dashboard", version="1.0.0")
        self.templates = create_jinja2_templates()

    def setup_routes(self) -> None:
        """Register all FastAPI routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def root(request: Request):
            data = self.data.get_all_data()
            return self.templates.TemplateResponse("dashboard.html", {
                "request": request,
                "enable_websocket": self.enable_websocket,
                **data
            })
        
        @self.app.get("/api/data", response_class=JSONResponse)
        async def api_data():
            return self.data.get_all_data()
        
        @self.app.get("/api/positions", response_class=JSONResponse)
        async def api_positions():
            return {"positions": self.data.get_current_positions()}
        
        @self.app.get("/api/trades", response_class=JSONResponse)
        async def api_trades(limit: int = 50, offset: int = 0):
            trades = self.data.get_completed_trades(limit + offset)
            return {"trades": trades[offset:offset + limit]}
        
        @self.app.get("/api/performance", response_class=JSONResponse)
        async def api_performance():
            return {"performance": self.data.get_performance_metrics()}
        
        @self.app.get("/api/chat", response_class=JSONResponse)
        async def api_chat(count: int = 20):
            return {"chat_log": self.data.get_ai_chat_log(count)}

        # Only register WebSocket route if enabled
        if self.enable_websocket:
            @self.app.websocket("/ws")
            async def websocket_endpoint(websocket: WebSocket):
                await websocket.accept()
                try:
                    # Send initial data
                    await websocket.send_json(self.data.get_all_data())
                    while True:
                        await asyncio.sleep(5)
                        await websocket.send_json(self.data.get_all_data())
                except WebSocketDisconnect:
                    pass

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start uvicorn server."""
        logging.info(f"Starting web dashboard on http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


def create_jinja2_templates() -> Jinja2Templates:
    """Set up Jinja2 template environment."""
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    return Jinja2Templates(directory=template_dir)


def main() -> None:
    """Main entry point with argument parsing."""
    # Load environment variables for defaults
    load_dotenv()

    # Read defaults from environment
    default_host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    default_port = int(os.getenv("DASHBOARD_PORT", "8000"))
    default_refresh = int(os.getenv("DASHBOARD_REFRESH_INTERVAL", "30"))
    default_enable_websocket = os.getenv("DASHBOARD_ENABLE_WEBSOCKET", "false").lower() in {"1", "true", "yes"}

    parser = argparse.ArgumentParser(description="Trading Bot Dashboard")
    parser.add_argument("--mode", choices=["cli", "web"], default="cli", help="Dashboard mode")
    parser.add_argument("--refresh", type=int, default=default_refresh, help="Refresh interval for CLI live mode (seconds)")
    parser.add_argument("--host", default=default_host, help="Web server host")
    parser.add_argument("--port", type=int, default=default_port, help="Web server port")
    parser.add_argument("--once", action="store_true", help="CLI: render once and exit")
    parser.add_argument("--enable-websocket", action="store_true", help="Web: enable WebSocket for real-time updates")
    parser.add_argument("--disable-websocket", action="store_true", help="Web: disable WebSocket, use polling instead")

    args = parser.parse_args()

    # Resolve WebSocket enable flag (CLI overrides env)
    if args.enable_websocket and args.disable_websocket:
        parser.error("Cannot specify both --enable-websocket and --disable-websocket")

    if args.enable_websocket:
        enable_websocket = True
    elif args.disable_websocket:
        enable_websocket = False
    else:
        enable_websocket = default_enable_websocket

    # Initialize data sources with error handling
    trade_logger = None
    if TradeLogger:
        try:
            trade_logger = TradeLogger()
            logging.info("TradeLogger initialized successfully")
        except Exception as e:
            logging.warning(f"Failed to initialize TradeLogger: {e}")

    chat_logger = None
    if ChatLogger:
        try:
            chat_logger = ChatLogger()
            logging.info("ChatLogger initialized successfully")
        except Exception as e:
            logging.warning(f"Failed to initialize ChatLogger: {e}")

    hyperliquid_client = None
    if HyperliquidClient:
        try:
            hyperliquid_client = HyperliquidClient()
            logging.info("HyperliquidClient initialized successfully")
        except Exception as e:
            logging.warning(f"Failed to initialize HyperliquidClient: {e}")

    data = DashboardData(trade_logger, chat_logger, hyperliquid_client)

    if args.mode == "cli":
        cli_dashboard = CLIDashboard(data)
        if args.once:
            cli_dashboard.run_once()
        else:
            cli_dashboard.run_live(args.refresh)
    elif args.mode == "web":
        web_dashboard = WebDashboard(data, enable_websocket=enable_websocket)
        web_dashboard.setup_routes()
        logging.info(f"WebSocket {'enabled' if enable_websocket else 'disabled'} - using {'real-time push' if enable_websocket else 'polling'} updates")
        web_dashboard.run(args.host, args.port)


if __name__ == "__main__":
    main()