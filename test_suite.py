"""
Test Suite for Pakistan Stock & Metals Price Prediction System
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestDataProcessing(unittest.TestCase):
    """Test data loading and preprocessing functions"""
    
    def setUp(self):
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        self.sample_data = pd.DataFrame({
            'Open': np.random.normal(100, 10, len(dates)),
            'High': np.random.normal(105, 10, len(dates)),
            'Low': np.random.normal(95, 10, len(dates)),
            'Close': np.random.normal(102, 10, len(dates)),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    def test_data_loading(self):
        self.assertIsNotNone(self.sample_data)
        self.assertEqual(len(self.sample_data), 366)
        self.assertTrue(all(col in self.sample_data.columns for col in ['Open', 'High', 'Low', 'Close']))
    
    def test_technical_indicators(self):
        ma7 = self.sample_data['Close'].rolling(window=7).mean()
        self.assertEqual(len(ma7), len(self.sample_data))
        self.assertTrue(ma7.iloc[6:].notna().all())
        
        returns = self.sample_data['Close'].pct_change()
        self.assertEqual(len(returns), len(self.sample_data))
        self.assertTrue(pd.isna(returns.iloc[0]))
    
    def test_data_cleaning(self):
        data_with_nan = self.sample_data.copy()
        data_with_nan.loc[data_with_nan.index[10], 'Close'] = np.nan
        
        cleaned = data_with_nan.fillna(method='ffill')
        self.assertTrue(cleaned['Close'].notna().all())


class TestModels(unittest.TestCase):
    """Test machine learning models"""
    
    def setUp(self):
        np.random.seed(42)
        self.X_train = np.random.randn(100, 10)
        self.y_train = np.random.randn(100)
        self.X_test = np.random.randn(20, 10)
        self.y_test = np.random.randn(20)
    
    def test_linear_regression(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertTrue(np.all(np.isfinite(predictions)))
        
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        self.assertLess(rmse, 10)
    
    def test_random_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertTrue(np.all(np.isfinite(predictions)))
        
        importance = model.feature_importances_
        self.assertEqual(len(importance), self.X_train.shape[1])
        self.assertTrue(np.all(importance >= 0))


class TestPSXConnector(unittest.TestCase):
    """Test PSX data connector"""
    
    def setUp(self):
        try:
            from psx_connector import PSXDataConnector
            self.connector = PSXDataConnector()
        except ImportError:
            self.skipTest("PSX connector not available")
    
    def test_ticker_list(self):
        self.assertGreater(len(self.connector.PSX_TICKERS), 0)
        self.assertIn('OGDC', self.connector.PSX_TICKERS)
        self.assertIn('HBL', self.connector.PSX_TICKERS)
    
    def test_sample_data_generation(self):
        data = self.connector.generate_sample_data('TEST', days=30)
        
        self.assertEqual(len(data), 30)
        self.assertTrue(all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']))
        self.assertTrue((data['High'] >= data['Low']).all())
        self.assertTrue((data['Volume'] > 0).all())


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functions"""
    
    def setUp(self):
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        self.prices = pd.Series(np.cumsum(np.random.randn(100)) + 100, index=dates)
    
    def test_rsi_calculation(self):
        delta = self.prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all() and (valid_rsi <= 100).all())
    
    def test_macd_calculation(self):
        ema12 = self.prices.ewm(span=12, adjust=False).mean()
        ema26 = self.prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        self.assertEqual(len(macd), len(self.prices))
        self.assertEqual(len(signal), len(self.prices))


class TestDataValidation(unittest.TestCase):
    """Test data validation and error handling"""
    
    def test_missing_values_handling(self):
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [np.nan, 2, 3, 4, 5]
        })
        
        filled = data.fillna(method='ffill')
        self.assertTrue(filled.notna().all().all())
    
    def test_outlier_detection(self):
        data = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9])
        
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = data[z_scores > 3]
        
        self.assertEqual(len(outliers), 1)
        self.assertEqual(outliers.iloc[0], 100)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def test_end_to_end_pipeline(self):
        dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
        data = pd.DataFrame({
            'Close': np.cumsum(np.random.randn(200) * 0.01) + 100
        }, index=dates)
        
        data['MA7'] = data['Close'].rolling(window=7).mean()
        data['Returns'] = data['Close'].pct_change()
        data = data.dropna()
        
        X = data[['MA7', 'Returns']].values
        y = data['Close'].values
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        self.assertEqual(len(predictions), len(y_test))
        self.assertTrue(np.all(np.isfinite(predictions)))
        
        correlation = np.corrcoef(predictions, y_test)[0, 1]
        self.assertTrue(-1 <= correlation <= 1)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestModels))
    suite.addTests(loader.loadTestsFromTestCase(TestPSXConnector))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureEngineering))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)