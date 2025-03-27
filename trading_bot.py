import time
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from technical_analysis import TechnicalAnalyzer
from risk_manager import RiskManager
from notifications import NotificationSystem
from config import *
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

class TradingBot:
    def __init__(self, investment_amount=DEFAULT_INVESTMENT_AMOUNT, trading_pairs=None, risk_tolerance='balanced'):
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        self.technical_analyzer = TechnicalAnalyzer()
        self.risk_manager = RiskManager(investment_amount)
        self.notification_system = NotificationSystem()
        
        # Create analysis directory if it doesn't exist
        self.analysis_dir = 'analysis'
        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)
        
        # Use provided trading pairs or default to config
        self.symbols = trading_pairs if trading_pairs else TRADING_PAIRS
        
        # Set risk parameters based on tolerance
        self.risk_tolerance = risk_tolerance
        self._set_risk_parameters()
        
        logging.info(f"Trading Bot initialized with investment amount: ${investment_amount}")
        logging.info(f"Risk tolerance: {risk_tolerance}")
        logging.info(f"Trading pairs: {self.symbols}")
    
    def _set_risk_parameters(self):
        """Set risk parameters based on risk tolerance"""
        if self.risk_tolerance == 'conservative':
            self.stop_loss = 0.01  # 1% stop loss
            self.take_profit = 0.03  # 3% take profit
            self.max_daily_loss = 0.02  # 2% max daily loss
            self.max_positions = 3
        elif self.risk_tolerance == 'aggressive':
            self.stop_loss = 0.03  # 3% stop loss
            self.take_profit = 0.08  # 8% take profit
            self.max_daily_loss = 0.05  # 5% max daily loss
            self.max_positions = 5
        else:  # balanced
            self.stop_loss = 0.02  # 2% stop loss
            self.take_profit = 0.05  # 5% take profit
            self.max_daily_loss = 0.03  # 3% max daily loss
            self.max_positions = 4
    
    def get_historical_data(self, symbol, interval=TIMEFRAME, limit=100):
        """Get historical data for both stocks and crypto"""
        try:
            if symbol.endswith('USDT'):  # Crypto pair
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
                
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Convert string values to float
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                return df
            else:  # Stock
                ticker = yf.Ticker(symbol)
                df = ticker.history(period='1d', interval=interval)
                return df
            
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def execute_trade(self, symbol, action, quantity):
        """Execute trade on appropriate platform"""
        try:
            if symbol.endswith('USDT'):  # Crypto trade
                # Get current price
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                
                # Place order
                order = self.client.create_order(
                    symbol=symbol,
                    side=action,
                    type='MARKET',
                    quantity=quantity
                )
            else:  # Stock trade
                # Get current price
                ticker = yf.Ticker(symbol)
                current_price = ticker.info.get('currentPrice', 0)
                
                # Here you would integrate with your stock broker's API
                # For now, we'll just log the trade
                logging.info(f"Stock trade: {action} {quantity} shares of {symbol} at ${current_price}")
            
            # Update risk manager
            self.risk_manager.update_position(symbol, current_price, quantity, action)
            
            # Send notification
            trade_info = {
                'symbol': symbol,
                'action': action,
                'price': current_price,
                'quantity': quantity,
                'timestamp': datetime.now()
            }
            self.notification_system.send_trade_notification(trade_info)
            
            logging.info(f"Executed {action} order for {quantity} {symbol} at ${current_price}")
            return True
            
        except Exception as e:
            logging.error(f"Error executing trade for {symbol}: {str(e)}")
            return False
    
    def create_analysis_graph(self, df, symbol):
        """Create interactive analysis graph using Plotly"""
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.5, 0.25, 0.25])

        # Candlestick chart
        fig.add_trace(go.Candlestick(x=df.index,
                                    open=df['open'],
                                    high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    name='OHLC'),
                     row=1, col=1)

        # Add Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'],
                               line=dict(color='gray', width=1),
                               name='BB Upper'),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'],
                               line=dict(color='gray', width=1),
                               name='BB Lower'),
                     row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'],
                               line=dict(color='purple', width=1),
                               name='RSI'),
                     row=2, col=1)
        fig.add_hline(y=RSI_OVERBOUGHT, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=RSI_OVERSOLD, line_dash="dash", line_color="green", row=2, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],
                               line=dict(color='blue', width=1),
                               name='MACD'),
                     row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'],
                               line=dict(color='orange', width=1),
                               name='Signal'),
                     row=3, col=1)

        # Update layout
        fig.update_layout(
            title=f'{symbol} Analysis',
            yaxis_title='Price',
            yaxis2_title='RSI',
            yaxis3_title='MACD',
            xaxis_rangeslider_visible=False,
            height=1000
        )

        # Save the graph
        fig.write_html(f'{self.analysis_dir}/{symbol}_analysis.html')
        return fig

    def generate_analysis_report(self, symbol, df, signals, analysis):
        """Generate detailed analysis report"""
        current_price = float(df['close'].iloc[-1])
        current_signal = signals['Trade_signal'].iloc[-1]
        
        report = f"""
=== {symbol} Analysis Report ===
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Current Price: ${current_price:.2f}
Signal: {'BUY' if current_signal == 1 else 'SELL' if current_signal == -1 else 'HOLD'}
Market Condition: {analysis['market_condition']}

Technical Indicators:
- RSI: {df['RSI'].iloc[-1]:.2f}
- MACD: {df['MACD'].iloc[-1]:.2f}
- MACD Signal: {df['MACD_signal'].iloc[-1]:.2f}
- BB Upper: ${df['BB_upper'].iloc[-1]:.2f}
- BB Lower: ${df['BB_lower'].iloc[-1]:.2f}

Support/Resistance:
- Support Level: ${analysis['support_level']:.2f}
- Resistance Level: ${analysis['resistance_level']:.2f}

Risk Management:
- Stop Loss: ${current_price * (1 - self.stop_loss):.2f}
- Take Profit: ${current_price * (1 + self.take_profit):.2f}
- Max Daily Loss: ${self.max_daily_loss * self.risk_manager.investment_amount:.2f}

Recommendation:
"""
        if current_signal == 1:
            report += "BUY - Strong buy signal based on technical indicators"
        elif current_signal == -1:
            report += "SELL - Strong sell signal based on technical indicators"
        else:
            report += "HOLD - No clear signal, maintain current position"

        # Save report to file
        with open(f'{self.analysis_dir}/{symbol}_report.txt', 'w') as f:
            f.write(report)

        return report

    def analyze_and_trade(self, symbol):
        """Analyze market and execute trades"""
        # Get historical data
        df = self.get_historical_data(symbol)
        if df is None:
            return
        
        # Calculate technical indicators
        df = self.technical_analyzer.calculate_indicators(df)
        
        # Generate signals
        signals = self.technical_analyzer.generate_signals(df)
        
        # Get detailed analysis
        analysis = self.technical_analyzer.get_market_condition(df)
        
        # Create analysis graph
        self.create_analysis_graph(df, symbol)
        
        # Generate and print analysis report
        report = self.generate_analysis_report(symbol, df, signals, analysis)
        print(report)
        
        # Check stop loss and take profit
        action, reason = self.risk_manager.check_stop_loss_take_profit(symbol, float(df['close'].iloc[-1]))
        if action:
            position = self.risk_manager.positions[symbol]
            self.execute_trade(symbol, action, position['quantity'])
            logging.info(f"Executed {action} for {symbol}: {reason}")
            return
        
        # Execute new trades based on signals
        if signals['Trade_signal'].iloc[-1] == 1:  # Buy signal
            can_open, message = self.risk_manager.can_open_position(symbol, float(df['close'].iloc[-1]), 1)
            if can_open:
                quantity = self.risk_manager.calculate_position_size(symbol, float(df['close'].iloc[-1]))
                self.execute_trade(symbol, 'BUY', quantity)
                logging.info(f"Opened BUY position for {symbol}")
            else:
                logging.info(f"Cannot open position for {symbol}: {message}")
        
        elif signals['Trade_signal'].iloc[-1] == -1 and symbol in self.risk_manager.positions:  # Sell signal
            position = self.risk_manager.positions[symbol]
            self.execute_trade(symbol, 'SELL', position['quantity'])
            logging.info(f"Closed position for {symbol} based on sell signal")
    
    def run(self):
        """Main bot loop"""
        logging.info("Starting Trading Bot...")
        
        while True:
            try:
                # Reset daily stats at market open
                if datetime.now().hour == 0 and datetime.now().minute == 0:
                    self.risk_manager.reset_daily_stats()
                    logging.info("Reset daily statistics")
                
                # Analyze and trade each symbol
                for symbol in self.symbols:
                    self.analyze_and_trade(symbol)
                
                # Get and send portfolio update
                portfolio_summary = self.risk_manager.get_portfolio_summary()
                self.notification_system.send_portfolio_update(
                    portfolio_summary['total_value'],
                    pd.DataFrame(self.risk_manager.daily_trades)
                )
                
                # Wait for next iteration
                time.sleep(UPDATE_INTERVAL)
                
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                self.notification_system.send_alert("Error", str(e))
                time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    try:
        # Get investment amount from user
        investment_amount = float(input("Enter investment amount in USD: "))
        
        # Create and run bot
        bot = TradingBot(investment_amount)
        bot.run()
        
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}") 