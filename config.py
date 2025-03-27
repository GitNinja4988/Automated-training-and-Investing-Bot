import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Trading Parameters
DEFAULT_INVESTMENT_AMOUNT = 1000  # USD
MAX_POSITION_SIZE = 0.1  # Maximum position size as percentage of portfolio
MAX_DAILY_LOSS = 100  # USD
MAX_OPEN_POSITIONS = 3
MAX_DRAWDOWN = 0.1  # 10%
STOP_LOSS_PERCENTAGE = 0.02  # 2%
TAKE_PROFIT_PERCENTAGE = 0.05  # 5%

# Technical Analysis Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
MA_FAST = 9
MA_SLOW = 21

# Trading Pairs and Timeframes
CRYPTO_PAIRS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
STOCK_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT']
TRADING_PAIRS = CRYPTO_PAIRS + STOCK_SYMBOLS
TIMEFRAME = '1h'
UPDATE_INTERVAL = 60  # seconds

# Position Sizing
POSITION_SIZES = {
    # Crypto positions
    'BTCUSDT': 0.1,  # 10% of portfolio
    'ETHUSDT': 0.1,  # 10% of portfolio
    'BNBUSDT': 0.1,  # 10% of portfolio
    'ADAUSDT': 0.1,  # 10% of portfolio
    'SOLUSDT': 0.1,  # 10% of portfolio
    
    # Stock positions
    'AAPL': 0.1,  # 10% of portfolio
    'MSFT': 0.1,  # 10% of portfolio
    'GOOGL': 0.1,  # 10% of portfolio
    'AMZN': 0.1,  # 10% of portfolio
    'META': 0.1,  # 10% of portfolio
    'NVDA': 0.1,  # 10% of portfolio
    'TSLA': 0.1,  # 10% of portfolio
    'JPM': 0.1,   # 10% of portfolio
    'V': 0.1,     # 10% of portfolio
    'WMT': 0.1    # 10% of portfolio
}

# Model Architecture Parameters
SEQUENCE_LENGTH = 60  # Number of time steps to use for prediction
LSTM_UNITS = 50      # Number of LSTM units
GRU_UNITS = 50       # Number of GRU units
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Notification Settings
ENABLE_TELEGRAM_NOTIFICATIONS = True
ENABLE_EMAIL_NOTIFICATIONS = False
NOTIFICATION_INTERVAL = 3600  # seconds (1 hour)

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'trading_bot.log'

# Database
DATABASE_URL = 'sqlite:///trading_bot.db'

# Risk Management
POSITION_SIZING = {
    # Crypto positions
    'BTCUSDT': 0.2,  # 20% of portfolio
    'ETHUSDT': 0.15,  # 15% of portfolio
    'BNBUSDT': 0.15,  # 15% of portfolio
    'ADAUSDT': 0.1,   # 10% of portfolio
    'SOLUSDT': 0.1,   # 10% of portfolio
    
    # Stock positions
    'AAPL': 0.1,  # 10% of portfolio
    'MSFT': 0.1,  # 10% of portfolio
    'GOOGL': 0.1,  # 10% of portfolio
    'AMZN': 0.1,  # 10% of portfolio
    'META': 0.1,  # 10% of portfolio
    'NVDA': 0.1,  # 10% of portfolio
    'TSLA': 0.1,  # 10% of portfolio
    'JPM': 0.1,   # 10% of portfolio
    'V': 0.1,     # 10% of portfolio
    'WMT': 0.1    # 10% of portfolio
} 