import pandas as pd
from datetime import datetime, timedelta
from config import *

class RiskManager:
    def __init__(self, initial_balance):
        self.investment_amount = initial_balance
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}
        self.daily_pnl = 0
        self.daily_trades = []
        self.max_drawdown = 0
        self.peak_balance = initial_balance
    
    def can_open_position(self, symbol, price, size):
        """Check if a new position can be opened"""
        # Check maximum number of open positions
        if len(self.positions) >= MAX_OPEN_POSITIONS:
            return False, "Maximum number of open positions reached"
        
        # Check if symbol already has an open position
        if symbol in self.positions:
            return False, "Position already open for this symbol"
        
        # Calculate position size
        position_value = price * size
        
        # Check position size limits
        max_position_size = self.current_balance * POSITION_SIZING.get(symbol, MAX_POSITION_SIZE)
        if position_value > max_position_size:
            return False, "Position size exceeds maximum allowed"
        
        # Check daily loss limit
        if self.daily_pnl < -self.initial_balance * MAX_DAILY_LOSS:
            return False, "Daily loss limit reached"
        
        # Check maximum drawdown
        if self.max_drawdown > MAX_DRAWDOWN:
            return False, "Maximum drawdown reached"
        
        return True, "Position can be opened"
    
    def calculate_position_size(self, symbol, price):
        """Calculate appropriate position size based on risk management rules"""
        # Get maximum position size for symbol
        max_position_size = self.current_balance * POSITION_SIZING.get(symbol, MAX_POSITION_SIZE)
        
        # Calculate position size based on risk
        risk_amount = self.current_balance * MAX_POSITION_SIZE
        stop_loss_points = price * STOP_LOSS_PERCENTAGE
        
        # Calculate position size that would result in max risk
        position_size = risk_amount / stop_loss_points
        
        # Ensure position size doesn't exceed maximum allowed
        position_size = min(position_size, max_position_size / price)
        
        return position_size
    
    def update_position(self, symbol, price, quantity, action):
        """Update position and calculate P&L"""
        if action == 'BUY':
            self.positions[symbol] = {
                'entry_price': price,
                'quantity': quantity,
                'entry_time': datetime.now(),
                'stop_loss': price * (1 - STOP_LOSS_PERCENTAGE),
                'take_profit': price * (1 + TAKE_PROFIT_PERCENTAGE)
            }
        elif action == 'SELL' and symbol in self.positions:
            position = self.positions[symbol]
            pnl = (price - position['entry_price']) * position['quantity']
            
            # Update daily P&L
            self.daily_pnl += pnl
            self.daily_trades.append({
                'symbol': symbol,
                'entry_price': position['entry_price'],
                'exit_price': price,
                'quantity': position['quantity'],
                'pnl': pnl,
                'time': datetime.now()
            })
            
            # Update current balance
            self.current_balance += pnl
            
            # Update peak balance and drawdown
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
            else:
                current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Remove position
            del self.positions[symbol]
    
    def check_stop_loss_take_profit(self, symbol, current_price):
        """Check if stop loss or take profit levels are hit"""
        if symbol in self.positions:
            position = self.positions[symbol]
            
            if current_price <= position['stop_loss']:
                return 'SELL', "Stop loss triggered"
            elif current_price >= position['take_profit']:
                return 'SELL', "Take profit triggered"
        
        return None, None
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_pnl = 0
        self.daily_trades = []
    
    def get_portfolio_summary(self):
        """Get current portfolio summary"""
        total_value = self.current_balance
        open_positions_value = 0
        
        for symbol, position in self.positions.items():
            open_positions_value += position['quantity'] * position['entry_price']
        
        return {
            'total_value': total_value,
            'open_positions_value': open_positions_value,
            'cash_balance': self.current_balance - open_positions_value,
            'daily_pnl': self.daily_pnl,
            'max_drawdown': self.max_drawdown,
            'number_of_positions': len(self.positions)
        }
    
    def get_position_summary(self, symbol):
        """Get summary for a specific position"""
        if symbol in self.positions:
            position = self.positions[symbol]
            return {
                'entry_price': position['entry_price'],
                'current_price': position['entry_price'],  # Update this with current price
                'quantity': position['quantity'],
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'time_in_position': datetime.now() - position['entry_time']
            }
        return None 