# Automated Trading & Investing Bot

An intelligent trading bot that uses deep learning models (LSTM and GRU) along with technical analysis to automate cryptocurrency trading on Binance.

## Features

- Multi-platform support (currently Binance, expandable to other platforms)
- Deep learning models (LSTM and GRU) for price prediction
- Technical analysis indicators (RSI, MACD, Moving Averages)
- Risk management with stop-loss and take-profit orders
- Real-time portfolio tracking
- Telegram notifications for trades and portfolio updates
- Automated trading based on combined signals

## Prerequisites

### Windows Setup
1. Install Python 3.8-3.10 (recommended for best compatibility)
   - Download from: https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"

2. Install Visual Studio Build Tools
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - During installation, select "Desktop development with C++"

3. Install Git (optional, for cloning the repository)
   - Download from: https://git-scm.com/download/win

### Required Accounts
- Binance account with API access
- Telegram bot token (for notifications)

## Installation

1. Clone the repository or download the source code:
```bash
git clone <repository-url>
cd automated-trading-bot
```

2. Run the setup script:
```bash
python setup.py
```

This will:
- Check Python version compatibility
- Install all required packages
- Create a template .env file
- Verify Visual C++ Build Tools installation

3. Edit the `.env` file with your API keys:
```
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

## Usage

1. Run the bot:
```bash
python trading_bot.py
```

2. Enter your investment amount when prompted.

3. The bot will:
   - Train LSTM and GRU models on historical data
   - Start monitoring market conditions
   - Execute trades based on combined signals
   - Send notifications via Telegram

## Troubleshooting

### Common Issues

1. TensorFlow DLL Error
   - Ensure you have Visual C++ Build Tools installed
   - Try using Python 3.8-3.10 instead of newer versions
   - Run `pip install --upgrade tensorflow-cpu`

2. API Connection Issues
   - Verify your Binance API keys are correct
   - Check your internet connection
   - Ensure your Binance account has sufficient permissions

3. Telegram Notifications Not Working
   - Verify your Telegram bot token and chat ID
   - Check if the bot is added to your Telegram chat

## Trading Strategy

The bot uses a combination of:
- Deep learning predictions (LSTM and GRU models)
- Technical indicators (RSI, MACD, Moving Averages)
- Risk management rules (stop-loss, take-profit, position sizing)

## Risk Warning

Trading cryptocurrencies involves significant risk. This bot is for educational purposes only. Always:
- Start with small amounts
- Monitor the bot's performance
- Understand the trading strategy
- Never invest more than you can afford to lose

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 