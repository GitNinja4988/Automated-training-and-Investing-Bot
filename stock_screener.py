import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from config import *
from binance.client import Client
from binance.exceptions import BinanceAPIException

class AssetScreener:
    def __init__(self, budget, risk_tolerance='balanced'):
        self.budget = budget
        self.risk_tolerance = risk_tolerance
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        self.logger = logging.getLogger(__name__)
        
        # Set risk parameters based on tolerance
        if risk_tolerance == 'conservative':
            self.min_market_cap = 10000000000  # $10B
            self.min_volume = 1000000  # $1M daily volume
            self.max_volatility = 0.02  # 2% daily volatility
        elif risk_tolerance == 'aggressive':
            self.min_market_cap = 1000000000  # $1B
            self.min_volume = 500000  # $500K daily volume
            self.max_volatility = 0.05  # 5% daily volatility
        else:  # balanced
            self.min_market_cap = 5000000000  # $5B
            self.min_volume = 750000  # $750K daily volume
            self.max_volatility = 0.03  # 3% daily volatility

    def get_sp500_stocks(self):
        """Get list of S&P 500 stocks"""
        try:
            # Using pandas to get S&P 500 stocks from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            stocks = df['Symbol'].tolist()
            self.logger.info(f"Successfully fetched {len(stocks)} S&P 500 stocks")
            return stocks
        except Exception as e:
            self.logger.error(f"Error fetching S&P 500 stocks: {str(e)}")
            return []

    def get_top_cryptos(self):
        """Get list of top cryptocurrencies by market cap"""
        try:
            # Get 24hr ticker for all symbols
            tickers = self.client.get_ticker()
            
            # Filter for USDT pairs and sort by volume
            usdt_pairs = [ticker for ticker in tickers if ticker['symbol'].endswith('USDT')]
            sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['volume']), reverse=True)
            
            # Return top 100 cryptocurrencies
            return [pair['symbol'] for pair in sorted_pairs[:100]]
        except Exception as e:
            self.logger.error(f"Error fetching crypto list: {str(e)}")
            return []

    def screen_stocks(self, stocks):
        """Screen stocks based on criteria"""
        screened_stocks = []
        total_stocks = len(stocks)
        
        self.logger.info(f"Starting to screen {total_stocks} stocks...")
        
        for i, symbol in enumerate(stocks):
            try:
                # Add delay to avoid rate limiting
                time.sleep(1)
                
                if (i + 1) % 50 == 0:
                    self.logger.info(f"Screened {i + 1}/{total_stocks} stocks...")
                
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Skip if required data is missing
                if not all(key in info for key in ['marketCap', 'averageVolume', 'regularMarketPrice']):
                    continue
                
                market_cap = info['marketCap']
                volume = info['averageVolume']
                price = info['regularMarketPrice']
                
                # Calculate volatility
                hist = ticker.history(period='1d')
                if not hist.empty:
                    volatility = hist['Close'].pct_change().std()
                else:
                    continue
                
                # Apply screening criteria
                if (market_cap >= self.min_market_cap and
                    volume >= self.min_volume and
                    volatility <= self.max_volatility and
                    price * 100 <= self.budget):  # Ensure we can buy at least 100 shares
                    
                    screened_stocks.append({
                        'symbol': symbol,
                        'market_cap': market_cap,
                        'volume': volume,
                        'price': price,
                        'volatility': volatility,
                        'type': 'stock',
                        'name': info.get('longName', ''),
                        'sector': info.get('sector', ''),
                        'pe_ratio': info.get('forwardPE', 0),
                        'dividend_yield': info.get('dividendYield', 0)
                    })
            
            except Exception as e:
                self.logger.error(f"Error screening stock {symbol}: {str(e)}")
                continue
        
        return screened_stocks

    def screen_cryptos(self, cryptos):
        """Screen cryptocurrencies based on criteria"""
        screened_cryptos = []
        total_cryptos = len(cryptos)
        
        self.logger.info(f"Starting to screen {total_cryptos} cryptocurrencies...")
        
        for i, symbol in enumerate(cryptos):
            try:
                # Add delay to avoid rate limiting
                time.sleep(1)
                
                if (i + 1) % 20 == 0:
                    self.logger.info(f"Screened {i + 1}/{total_cryptos} cryptocurrencies...")
                
                # Get 24hr ticker
                ticker = self.client.get_ticker(symbol=symbol)
                
                # Get symbol info
                symbol_info = self.client.get_symbol_info(symbol)
                
                # Calculate volatility from klines
                klines = self.client.get_klines(symbol=symbol, interval='1d', limit=2)
                if len(klines) >= 2:
                    volatility = (float(klines[1][4]) - float(klines[1][1])) / float(klines[1][1])
                else:
                    continue
                
                price = float(ticker['lastPrice'])
                volume = float(ticker['volume'])
                market_cap = float(ticker['quoteVolume'])  # Using 24h volume as proxy for market cap
                
                # Apply screening criteria
                if (market_cap >= self.min_market_cap and
                    volume >= self.min_volume and
                    volatility <= self.max_volatility and
                    price * 0.1 <= self.budget):  # Ensure we can buy at least 0.1 units
                    
                    screened_cryptos.append({
                        'symbol': symbol,
                        'market_cap': market_cap,
                        'volume': volume,
                        'price': price,
                        'volatility': volatility,
                        'type': 'crypto',
                        'name': symbol_info.get('baseAsset', ''),
                        'change_24h': float(ticker['priceChangePercent'])
                    })
            
            except Exception as e:
                self.logger.error(f"Error screening crypto {symbol}: {str(e)}")
                continue
        
        return screened_cryptos

    def screen_assets(self):
        """Screen both stocks and cryptocurrencies"""
        self.logger.info("Starting asset screening process...")
        
        # Get lists of assets to screen
        stocks = self.get_sp500_stocks()
        cryptos = self.get_top_cryptos()
        
        # Screen both asset types
        screened_stocks = self.screen_stocks(stocks)
        screened_cryptos = self.screen_cryptos(cryptos)
        
        # Combine and sort results
        all_screened = screened_stocks + screened_cryptos
        all_screened.sort(key=lambda x: x['market_cap'], reverse=True)
        
        # Log results
        self.logger.info(f"Screening complete. Found {len(screened_stocks)} stocks and {len(screened_cryptos)} cryptocurrencies meeting criteria.")
        
        # Print detailed results
        print("\nScreened Assets:")
        print("===============")
        
        # Print stocks
        print("\nSTOCKS:")
        print("-------")
        for asset in screened_stocks:
            print(f"\nSymbol: {asset['symbol']}")
            print(f"Name: {asset['name']}")
            print(f"Sector: {asset['sector']}")
            print(f"Price: ${asset['price']:.2f}")
            print(f"Market Cap: ${asset['market_cap']:,.2f}")
            print(f"24h Volume: ${asset['volume']:,.2f}")
            print(f"Volatility: {asset['volatility']:.2%}")
            print(f"P/E Ratio: {asset['pe_ratio']:.2f}")
            print(f"Dividend Yield: {asset['dividend_yield']:.2%}")
            print("-" * 50)
        
        # Print cryptocurrencies
        print("\nCRYPTOCURRENCIES:")
        print("----------------")
        for asset in screened_cryptos:
            print(f"\nSymbol: {asset['symbol']}")
            print(f"Name: {asset['name']}")
            print(f"Price: ${asset['price']:.2f}")
            print(f"Market Cap: ${asset['market_cap']:,.2f}")
            print(f"24h Volume: ${asset['volume']:,.2f}")
            print(f"Volatility: {asset['volatility']:.2%}")
            print(f"24h Change: {asset['change_24h']:.2f}%")
            print("-" * 50)
        
        return all_screened

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get user input
    budget = float(input("Enter your investment budget in USD: "))
    risk_tolerance = input("Enter your risk tolerance (conservative/aggressive/balanced): ")
    
    # Create and run screener
    screener = AssetScreener(budget, risk_tolerance)
    screened_assets = screener.screen_assets() 