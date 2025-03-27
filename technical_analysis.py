import pandas as pd
import numpy as np
import ta
from config import *

class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                # Rename columns if needed
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
            
            # Calculate RSI
            df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=RSI_PERIOD).rsi()
            
            # Calculate MACD
            macd = ta.trend.MACD(df['close'], 
                                window_slow=MACD_SLOW,
                                window_fast=MACD_FAST,
                                window_sign=MACD_SIGNAL)
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_hist'] = macd.macd_diff()
            
            # Calculate Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'],
                                                    window=BB_PERIOD,
                                                    window_dev=BB_STD)
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_middle'] = bollinger.bollinger_mavg()
            df['BB_lower'] = bollinger.bollinger_lband()
            
            # Calculate Moving Averages
            df['MA_fast'] = ta.trend.SMAIndicator(df['close'], window=MA_FAST).sma_indicator()
            df['MA_slow'] = ta.trend.SMAIndicator(df['close'], window=MA_SLOW).sma_indicator()
            
            # Calculate Volume indicators
            df['Volume_MA'] = ta.trend.SMAIndicator(df['volume'], window=20).sma_indicator()
            
            # Calculate ATR for volatility
            df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return df
    
    def generate_signals(self, df):
        """Generate trading signals based on technical indicators"""
        try:
            signals = pd.DataFrame(index=df.index)
            signals['Trade_signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
            
            # RSI signals
            signals.loc[df['RSI'] < RSI_OVERSOLD, 'Trade_signal'] = 1
            signals.loc[df['RSI'] > RSI_OVERBOUGHT, 'Trade_signal'] = -1
            
            # MACD signals
            signals.loc[df['MACD'] > df['MACD_signal'], 'Trade_signal'] = 1
            signals.loc[df['MACD'] < df['MACD_signal'], 'Trade_signal'] = -1
            
            # Bollinger Bands signals
            signals.loc[df['close'] < df['BB_lower'], 'Trade_signal'] = 1
            signals.loc[df['close'] > df['BB_upper'], 'Trade_signal'] = -1
            
            # Moving Average signals
            signals.loc[df['MA_fast'] > df['MA_slow'], 'Trade_signal'] = 1
            signals.loc[df['MA_fast'] < df['MA_slow'], 'Trade_signal'] = -1
            
            # Volume confirmation
            volume_signal = df['volume'] > df['Volume_MA']
            signals.loc[~volume_signal, 'Trade_signal'] = 0
            
            return signals
            
        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            return pd.DataFrame({'Trade_signal': [0] * len(df)}, index=df.index)
    
    def calculate_support_resistance(self, df):
        """Calculate support and resistance levels"""
        try:
            # Use recent price action to identify levels
            recent_highs = df['high'].rolling(window=20).max()
            recent_lows = df['low'].rolling(window=20).min()
            
            # Get the most recent levels
            resistance = recent_highs.iloc[-1]
            support = recent_lows.iloc[-1]
            
            return {
                'support_level': support,
                'resistance_level': resistance
            }
            
        except Exception as e:
            print(f"Error calculating support/resistance: {str(e)}")
            return {
                'support_level': df['close'].iloc[-1] * 0.95,
                'resistance_level': df['close'].iloc[-1] * 1.05
            }
    
    def analyze_volume(self, df):
        """Analyze volume patterns"""
        try:
            # Calculate volume metrics
            volume_ma = df['volume'].rolling(window=20).mean()
            current_volume = df['volume'].iloc[-1]
            
            # Determine volume trend
            if current_volume > volume_ma.iloc[-1] * 1.5:
                return "High volume - Strong trend"
            elif current_volume < volume_ma.iloc[-1] * 0.5:
                return "Low volume - Weak trend"
            else:
                return "Normal volume"
                
        except Exception as e:
            print(f"Error analyzing volume: {str(e)}")
            return "Volume analysis unavailable"
    
    def get_market_condition(self, df):
        """Determine overall market condition"""
        try:
            # Calculate trend strength
            price_change = df['close'].pct_change()
            trend_strength = price_change.rolling(window=20).std()
            
            # Calculate volatility
            volatility = df['ATR'] / df['close']
            
            # Initialize market condition series
            market_condition = pd.Series(index=df.index, dtype=str)
            
            # Determine market condition for each period
            for i in range(len(df)):
                if trend_strength.iloc[i] > 0.02:  # Strong trend
                    if price_change.iloc[i] > 0:
                        market_condition.iloc[i] = 'Strong Uptrend'
                    else:
                        market_condition.iloc[i] = 'Strong Downtrend'
                elif volatility.iloc[i] > 0.02:  # High volatility
                    market_condition.iloc[i] = 'High Volatility'
                else:
                    market_condition.iloc[i] = 'Sideways'
            
            # Get current market condition
            current_condition = market_condition.iloc[-1]
            
            # Get support/resistance levels
            levels = self.calculate_support_resistance(df)
            
            # Get volume analysis
            volume_analysis = self.analyze_volume(df)
            
            return {
                'market_condition': current_condition,
                'support_level': levels['support_level'],
                'resistance_level': levels['resistance_level'],
                'volume_analysis': volume_analysis,
                'volatility': volatility.iloc[-1],
                'trend_strength': trend_strength.iloc[-1]
            }
            
        except Exception as e:
            print(f"Error determining market condition: {str(e)}")
            return {
                'market_condition': 'Unknown',
                'support_level': df['close'].iloc[-1] * 0.95,
                'resistance_level': df['close'].iloc[-1] * 1.05,
                'volume_analysis': 'Volume analysis unavailable',
                'volatility': 0.0,
                'trend_strength': 0.0
            } 