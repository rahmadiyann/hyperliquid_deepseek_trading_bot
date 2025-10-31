#!/bin/bash
set -e

# Determine run mode (both, bot-only, dashboard-only)
RUN_MODE=${RUN_MODE:-both}

if [ "$RUN_MODE" = "bot-only" ]; then
    echo "Running bot only..."
    exec python main.py
elif [ "$RUN_MODE" = "dashboard-only" ]; then
    echo "Running dashboard only..."
    exec python -m src.ui.dashboard --mode web --host 0.0.0.0 --port ${PORT:-8000}
elif [ "$RUN_MODE" = "both" ]; then
    # Start trading bot in background
    echo "Starting trading bot..."
    python main.py &
    BOT_PID=$!

    # Start dashboard in background
    echo "Starting dashboard..."
    python -m src.ui.dashboard --mode web --host 0.0.0.0 --port ${PORT:-8000} &
    DASH_PID=$!

    # Log PIDs for debugging
    echo "Bot PID: $BOT_PID"
    echo "Dashboard PID: $DASH_PID"

    # Trap SIGTERM and SIGINT for graceful shutdown
    shutdown() {
        echo 'Shutting down...'
        kill -TERM $BOT_PID 2>/dev/null || true
        kill -TERM $DASH_PID 2>/dev/null || true
        wait
    }
    trap shutdown SIGTERM SIGINT

    # Wait for both processes, exit when first one exits
    echo "Both services started. Waiting for processes..."
    wait -n

    # If we get here, one process exited - shut down the other
    EXIT_CODE=$?
    echo "One process exited with code $EXIT_CODE, shutting down remaining process..."
    kill -TERM $BOT_PID 2>/dev/null || true
    kill -TERM $DASH_PID 2>/dev/null || true
    wait
    exit $EXIT_CODE
else
    echo "Error: Invalid RUN_MODE '$RUN_MODE'. Valid values: both, bot-only, dashboard-only"
    exit 1
fi
