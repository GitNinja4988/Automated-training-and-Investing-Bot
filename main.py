import os
from dotenv import load_dotenv
from stock_screener import AssetScreener
from models import PricePredictor
from trading_bot import TradingBot
from notifications import NotificationSystem
import logging
from config import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize notification system
    notification_system = NotificationSystem()
    
    try:
        # Get user input
        budget = float(input("Enter your investment budget in USD: "))
        risk_tolerance = input("Enter your risk tolerance (conservative/aggressive/balanced): ").lower()
        
        # Screen assets
        logging.info("Starting asset screening process...")
        screener = AssetScreener(budget, risk_tolerance)
        screened_assets = screener.screen_assets()
        
        # Create trading pairs list from screened assets
        trading_pairs = [asset['symbol'] for asset in screened_assets]
        
        # Initialize and run trading bot
        bot = TradingBot(budget, trading_pairs, risk_tolerance)
        bot.run()
        
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        notification_system.send_alert("Fatal Error", str(e))

if __name__ == "__main__":
    main() 