import numpy as np
import pandas as pd
from config import *
from binance.client import Client
from datetime import datetime

class TradingStrategy:
    def __init__(self, binance_client, investment_amount):
        self.client = binance_client
        self.investment_amount = investment_amount
        self.positions = {}
        self.trade_history = []
        
    def analyze_signals(self, df, predictions):
        """Analyze technical indicators and model predictions for trading signals"""
        signals = pd.DataFrame(index=df.index)
        
        # Technical Analysis Signals
        signals['RSI_signal'] = np.where(
            df['RSI'] < RSI_OVERSOLD, 1,
            np.where(df['RSI'] > RSI_OVERBOUGHT, -1, 0)
        )
        
        signals['MACD_signal'] = np.where(
            df['MACD'] > df['MACD_signal'], 1,
            np.where(df['MACD'] < df['MACD_signal'], -1, 0)
        )
        
        signals['MA_signal'] = np.where(
            df['SMA_20'] > df['SMA_50'], 1,
            np.where(df['SMA_20'] < df['SMA_50'], -1, 0)
        )
        
        # Model Prediction Signals
        signals['Model_signal'] = np.where(
            predictions['ensemble'] > df['close'], 1,
            np.where(predictions['ensemble'] < df['close'], -1, 0)
        )
        
        # Combine signals
        signals['Combined_signal'] = (
            signals['RSI_signal'] +
            signals['MACD_signal'] +
            signals['MA_signal'] +
            signals['Model_signal']
        )
        
        return signals
    
    def execute_trade(self, symbol, signal, current_price):
        """Execute trade based on signal"""
        try:
            if signal > 2 and symbol not in self.positions:  # Strong buy signal
                # Calculate position size
                position_size = min(
                    self.investment_amount * MAX_POSITION_SIZE,
                    self.investment_amount
                )
                
                # Place buy order
                order = self.client.create_order(
                    symbol=symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=position_size / current_price
                )
                
                # Record position
                self.positions[symbol] = {
                    'entry_price': current_price,
                    'quantity': position_size / current_price,
                    'stop_loss': current_price * (1 - STOP_LOSS_PERCENTAGE),
                    'take_profit': current_price * (1 + TAKE_PROFIT_PERCENTAGE)
                }
                
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current_price,
                    'quantity': position_size / current_price
                })
                
            elif signal < -2 and symbol in self.positions:  # Strong sell signal
                position = self.positions[symbol]
                
                # Place sell order
                order = self.client.create_order(
                    symbol=symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=position['quantity']
                )
                
                # Record trade
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': current_price,
                    'quantity': position['quantity']
                })
                
                # Remove position
                del self.positions[symbol]
                
        except Exception as e:
            print(f"Error executing trade: {str(e)}")
    
    def check_stop_loss_take_profit(self, symbol, current_price):
        """Check and execute stop loss or take profit orders"""
        if symbol in self.positions:
            position = self.positions[symbol]
            
            if current_price <= position['stop_loss']:
                # Execute stop loss
                self.execute_trade(symbol, -3, current_price)
                print(f"Stop loss triggered for {symbol}")
                
            elif current_price >= position['take_profit']:
                # Execute take profit
                self.execute_trade(symbol, -3, current_price)
                print(f"Take profit triggered for {symbol}")
    
    def get_portfolio_value(self):
        """Calculate current portfolio value"""
        total_value = 0
        for symbol, position in self.positions.items():
            try:
                current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
                total_value += position['quantity'] * current_price
            except Exception as e:
                print(f"Error getting price for {symbol}: {str(e)}")
        return total_value
    
    def get_trade_history(self):
        """Return trade history"""
        return pd.DataFrame(self.trade_history) 