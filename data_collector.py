import yfinance as yf
import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import ta
from config import *
from sklearn.preprocessing import MinMaxScaler

class DataCollector:
    def __init__(self):
        self.binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        
    def get_stock_data(self, symbol, period='1y', interval='1d'):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)
            return self._add_technical_indicators(df)
        except Exception as e:
            print(f"Error fetching stock data for {symbol}: {str(e)}")
            return None
    
    def get_crypto_data(self, symbol, interval='1d', limit=500):
        """Fetch cryptocurrency data from Binance"""
        try:
            klines = self.binance_client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return self._add_technical_indicators(df)
        except Exception as e:
            print(f"Error fetching crypto data for {symbol}: {str(e)}")
            return None
    
    def _add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(
            df['close'],
            window=RSI_PERIOD
        ).rsi()
        
        # MACD
        macd = ta.trend.MACD(
            df['close'],
            window_slow=MACD_SLOW,
            window_fast=MACD_FAST,
            window_sign=MACD_SIGNAL
        )
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['BB_high'] = bollinger.bollinger_hband()
        df['BB_low'] = bollinger.bollinger_lband()
        df['BB_mid'] = bollinger.bollinger_mavg()
        
        return df
    
    def prepare_data_for_model(self, df, sequence_length=SEQUENCE_LENGTH):
        """Prepare data for LSTM/GRU models"""
        features = ['open', 'high', 'low', 'close', 'volume', 'RSI', 
                   'MACD', 'MACD_signal', 'MACD_diff', 'SMA_20', 'SMA_50',
                   'EMA_20', 'BB_high', 'BB_low', 'BB_mid']
        
        # Normalize the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[features])
        
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
            y.append(scaled_data[i + sequence_length, 3])  # Predict close price
            
        return np.array(X), np.array(y), scaler 