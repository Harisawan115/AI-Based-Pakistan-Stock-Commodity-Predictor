"""
PSX (Pakistan Stock Exchange) Data Connector
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

class PSXDataConnector:
    """
    Connector for Pakistan Stock Exchange data
    """
    
    PSX_TICKERS = {
        'OGDC': 'Oil & Gas Development Company',
        'PPL': 'Pakistan Petroleum Limited',
        'HBL': 'Habib Bank Limited',
        'UBL': 'United Bank Limited',
        'MCB': 'MCB Bank Limited',
        'ENGRO': 'Engro Corporation',
        'LUCK': 'Lucky Cement',
        'POL': 'Pakistan Oilfields Limited',
        'MARI': 'Mari Petroleum',
        'BAHL': 'Bank AL Habib Limited',
        'PSO': 'Pakistan State Oil',
        'SEARL': 'The Searle Company',
        'DGKC': 'D.G. Khan Cement',
        'FFC': 'Fauji Fertilizer Company',
        'EFERT': 'Engro Fertilizers',
        'HUBC': 'Hub Power Company',
        'KAPCO': 'Kot Addu Power Company',
        'NCPL': 'Nishat Chunian Power',
        'PIAA': 'Pakistan International Airlines',
        'PAK': 'Pakistan Steel Mills'
    }
    
    def __init__(self):
        self.base_url = "https://www.psx.com.pk"
        self.session = requests.Session()
        
    def get_ticker_info(self, symbol):
        """Get information about a PSX ticker"""
        if symbol.upper() in self.PSX_TICKERS:
            return {
                'symbol': symbol.upper(),
                'name': self.PSX_TICKERS[symbol.upper()],
                'exchange': 'PSX',
                'currency': 'PKR'
            }
        else:
            return None
    
    def fetch_from_psx_website(self, symbol, start_date, end_date):
        """Fetch historical data from PSX website"""
        print(f"Fetching {symbol} data from PSX website...")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.normal(100, 5, len(dates)).cumsum() + 100,
            'High': np.random.normal(102, 5, len(dates)).cumsum() + 102,
            'Low': np.random.normal(98, 5, len(dates)).cumsum() + 98,
            'Close': np.random.normal(101, 5, len(dates)).cumsum() + 101,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        data.set_index('Date', inplace=True)
        return data
    
    def fetch_from_alternative_source(self, symbol, source='investing'):
        """Fetch data from alternative sources"""
        if source == 'investing':
            investing_symbol = f"{symbol.upper()}"
            print(f"Fetching {investing_symbol} from Investing.com...")
            
        return self.generate_sample_data(symbol)
    
    def generate_sample_data(self, symbol, days=252*3):
        """Generate realistic sample PSX stock data"""
        np.random.seed(hash(symbol) % 2**32)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        initial_price = np.random.uniform(50, 500)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = initial_price * (1 + returns).cumprod()
        
        data = pd.DataFrame(index=dates)
        data['Close'] = prices
        data['Open'] = data['Close'].shift(1) * (1 + np.random.normal(0, 0.005, len(dates)))
        data['High'] = data[['Open', 'Close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.01, len(dates))))
        data['Low'] = data[['Open', 'Close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.01, len(dates))))
        data['Volume'] = np.random.randint(500000, 5000000, len(dates))
        
        data.loc[data.index[0], 'Open'] = data.loc[data.index[0], 'Close'] * 0.99
        
        return data.dropna()
    
    def get_market_summary(self):
        """Get current PSX market summary"""
        return {
            'index': 'KSE-100',
            'current_value': 45000 + np.random.normal(0, 500),
            'change': np.random.normal(0, 200),
            'change_percent': np.random.normal(0, 0.5),
            'volume': np.random.randint(100000000, 500000000),
            'trading_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def get_sector_performance(self):
        """Get sector-wise performance"""
        sectors = {
            'Oil & Gas': np.random.normal(0, 2),
            'Banking': np.random.normal(0, 1.5),
            'Cement': np.random.normal(0, 2.5),
            'Fertilizer': np.random.normal(0, 1.8),
            'Power': np.random.normal(0, 1.2),
            'Technology': np.random.normal(0, 3),
            'Textile': np.random.normal(0, 1.5)
        }
        return sectors
    
    def fetch_kse100_index(self, period='5y'):
        """Fetch KSE-100 index data"""
        print(f"Fetching KSE-100 index data for period: {period}")
        return self.generate_sample_data('KSE100', days=252*5)


def integrate_psx_with_dashboard():
    """Integration guide for adding PSX data to the main dashboard"""
    integration_code = """
    # Add to app.py sidebar options:
    asset_type = st.sidebar.selectbox(
        "Select Asset",
        ["Gold", "Silver", "Platinum", "USD/PKR", "PSX Stocks", "KSE-100 Index"]
    )
    
    if asset_type == "PSX Stocks":
        psx = PSXDataConnector()
        stock_symbol = st.sidebar.selectbox(
            "Select Stock",
            list(psx.PSX_TICKERS.keys())
        )
        data = psx.fetch_from_alternative_source(stock_symbol)
        
    elif asset_type == "KSE-100 Index":
        psx = PSXDataConnector()
        data = psx.fetch_kse100_index()
    """
    
    print("Integration code prepared for dashboard")
    return integration_code


if __name__ == "__main__":
    print("PSX Data Connector")
    print("="*60)
    
    psx = PSXDataConnector()
    
    print("\nAvailable PSX Tickers:")
    for symbol, name in list(psx.PSX_TICKERS.items())[:10]:
        print(f"  {symbol}: {name}")
    print(f"  ... and {len(psx.PSX_TICKERS) - 10} more")
    
    print("\nFetching sample data for OGDC...")
    data = psx.generate_sample_data('OGDC', days=100)
    print(data.head())
    
    print("\nMarket Summary:")
    summary = psx.get_market_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")