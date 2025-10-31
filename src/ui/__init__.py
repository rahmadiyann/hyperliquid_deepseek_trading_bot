"""Dashboard UI module for monitoring trading bot activity.

Provides:
- DashboardData: Data aggregation from loggers and exchange client
- CLIDashboard: Terminal-based dashboard using Rich library
- WebDashboard: FastAPI web dashboard with real-time updates
- main: Entry point for running dashboard (CLI or web mode)

Usage:
    # CLI mode (one-time render)
    python -m src.ui.dashboard --mode cli --once
    
    # CLI mode (live updates)
    python -m src.ui.dashboard --mode cli --refresh 30
    
    # Web mode
    python -m src.ui.dashboard --mode web --port 8000
"""

from src.ui.dashboard import DashboardData, CLIDashboard, WebDashboard, main

__all__ = ["DashboardData", "CLIDashboard", "WebDashboard", "main"]
