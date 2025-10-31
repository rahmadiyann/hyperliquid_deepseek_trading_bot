"""Entry point for running dashboard as a module.

Usage:
    python -m src.ui.dashboard --mode cli
    python -m src.ui.dashboard --mode web --port 8000
"""

from src.ui.dashboard import main

if __name__ == "__main__":
    main()