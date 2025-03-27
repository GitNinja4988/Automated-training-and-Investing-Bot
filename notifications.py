from telegram.ext import Application
import pandas as pd
import asyncio
from config import *

class NotificationSystem:
    def __init__(self):
        if ENABLE_TELEGRAM_NOTIFICATIONS:
            self.application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
    
    def send_trade_notification(self, trade_info):
        """Send notification about executed trade"""
        if ENABLE_TELEGRAM_NOTIFICATIONS:
            message = (
                f"🤖 Trade Executed\n\n"
                f"Symbol: {trade_info['symbol']}\n"
                f"Action: {trade_info['action']}\n"
                f"Price: ${trade_info['price']:.2f}\n"
                f"Quantity: {trade_info['quantity']:.4f}\n"
                f"Time: {trade_info['timestamp']}"
            )
            self._send_message_sync(message)
    
    def send_portfolio_update(self, portfolio_value, trade_history):
        """Send portfolio value update"""
        if ENABLE_TELEGRAM_NOTIFICATIONS:
            # Calculate daily P&L
            if not trade_history.empty:
                daily_trades = trade_history[
                    trade_history['timestamp'].dt.date == pd.Timestamp.now().date()
                ]
                daily_pnl = self._calculate_daily_pnl(daily_trades)
            else:
                daily_pnl = 0
            
            message = (
                f"📊 Portfolio Update\n\n"
                f"Total Value: ${portfolio_value:.2f}\n"
                f"Daily P&L: ${daily_pnl:.2f}\n"
                f"Time: {pd.Timestamp.now()}"
            )
            self._send_message_sync(message)
    
    def send_alert(self, alert_type, message):
        """Send custom alert"""
        if ENABLE_TELEGRAM_NOTIFICATIONS:
            formatted_message = f"⚠️ {alert_type}\n\n{message}"
            self._send_message_sync(formatted_message)
    
    def _send_message_sync(self, message):
        """Synchronously send message via Telegram"""
        try:
            self.loop.run_until_complete(self._send_telegram_message(message))
        except Exception as e:
            print(f"Error sending Telegram message: {str(e)}")
    
    async def _send_telegram_message(self, message):
        """Send message via Telegram"""
        try:
            await self.application.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            print(f"Error sending Telegram message: {str(e)}")
    
    def _calculate_daily_pnl(self, trades):
        """Calculate daily profit/loss from trades"""
        pnl = 0
        for _, trade in trades.iterrows():
            if trade['action'] == 'SELL':
                # Find corresponding BUY trade
                buy_trade = trades[
                    (trades['symbol'] == trade['symbol']) &
                    (trades['action'] == 'BUY') &
                    (trades['timestamp'] < trade['timestamp'])
                ].iloc[0]
                
                pnl += (trade['price'] - buy_trade['price']) * trade['quantity']
        return pnl 