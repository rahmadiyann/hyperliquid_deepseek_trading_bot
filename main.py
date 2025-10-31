import sys
import logging
from dotenv import load_dotenv
import os
from src.main_loop import TradingBot


def main():
    # Load environment
    load_dotenv()
    
    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    normalized_level = log_level.upper()
    level_value = getattr(logging, normalized_level, None)

    # Validate log level - must be a valid logging constant (int)
    if level_value is None or not isinstance(level_value, int):
        print(f"WARNING: Invalid LOG_LEVEL '{log_level}', falling back to INFO")
        level_value = logging.INFO

    logging.basicConfig(
        level=level_value,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            # Optional: logging.FileHandler("./logs/bot.log")
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Live Trading Bot...")
    
    # Validate environment
    required_vars = [
        "HYPERLIQUID_WALLET_ADDRESS",
        "HYPERLIQUID_PRIVATE_KEY",
        "DEEPSEEK_API_KEY"
    ]
    for var in required_vars:
        if not os.getenv(var):
            logger.error(f"Missing required environment variable: {var}")
            logger.error("Please copy .env.example to .env and fill in your credentials")
            sys.exit(1)
    
    # Log configuration summary
    testnet = os.getenv("HYPERLIQUID_TESTNET", "true").lower() == "true"

    # Parse trading interval with validation
    try:
        trading_interval = int(os.getenv("TRADING_INTERVAL_MINUTES", "3"))
    except ValueError:
        logger.warning(f"Invalid TRADING_INTERVAL_MINUTES value '{os.getenv('TRADING_INTERVAL_MINUTES')}', falling back to 3")
        trading_interval = 3

    symbols = os.getenv("TRADING_SYMBOLS", "BTC,ETH,SOL,BNB,XRP,DOGE")
    logger.info(f"Configuration: testnet={testnet}, interval={trading_interval}min, symbols={symbols}")
    
    # Log prominent testnet/mainnet warning
    if testnet:
        logger.info("="*60)
        logger.info("RUNNING ON TESTNET WITH MOCK BALANCE")
        logger.info("This is a safe testing environment with no real funds at risk")
        logger.info("="*60)
    else:
        logger.warning("="*60)
        logger.warning("RUNNING ON MAINNET WITH REAL FUNDS")
        logger.warning("All trades will use real money - ensure risk settings are correct")
        logger.warning("="*60)
    
    try:
        # Initialize and run bot
        bot = TradingBot()
        bot.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
    
    logger.info("Bot shutdown complete")


if __name__ == "__main__":
    main()
