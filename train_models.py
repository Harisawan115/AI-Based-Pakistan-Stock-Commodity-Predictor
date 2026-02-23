"""
Model Training Script for Pakistan Stock & Metals Price Prediction
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def fetch_data():
    """Fetch latest data from Yahoo Finance"""
    print("Fetching data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    gold = yf.download("GC=F", start=start_date, end=end_date, progress=False)
    silver = yf.download("SI=F", start=start_date, end=end_date, progress=False)
    platinum = yf.download("PL=F", start=start_date, end=end_date, progress=False)
    usd_pkr = yf.download("PKR=X", start=start_date, end=end_date, progress=False)
    
    data = pd.DataFrame({
        'Gold_Close': gold['Close'].squeeze(),
        'Silver_Close': silver['Close'].squeeze(),
        'Platinum_Close': platinum['Close'].squeeze(),
        'USD_PKR': usd_pkr['Close'].squeeze(),
    })
    
    return data.fillna(method='ffill').dropna()

def add_features(df):
    """Add technical indicators"""
    for col in ['Gold_Close', 'Silver_Close', 'Platinum_Close', 'USD_PKR']:
        prefix = col.split('_')[0]
        df[f'{prefix}_MA7'] = df[col].rolling(window=7).mean()
        df[f'{prefix}_MA30'] = df[col].rolling(window=30).mean()
        df[f'{prefix}_Returns'] = df[col].pct_change()
        df[f'{prefix}_Volatility'] = df[col].rolling(window=7).std()
        
        delta = df[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df[f'{prefix}_RSI'] = 100 - (100 / (1 + rs))
    
    return df

def create_lags(df, target, lags=5):
    """Create lag features"""
    for i in range(1, lags + 1):
        df[f'{target}_lag_{i}'] = df[target].shift(i)
    df['Target'] = df[target].shift(-1)
    return df.dropna()

def train_and_evaluate(X, y, model_name):
    """Train model and return metrics"""
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    if model_name == 'LinearRegression':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    mape = mean_absolute_percentage_error(y_test, pred) * 100
    
    return model, scaler, {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def main():
    print("="*60)
    print("PAKISTAN STOCK & METALS PRICE PREDICTION - MODEL TRAINING")
    print("="*60)
    
    data = fetch_data()
    print(f"Data loaded: {len(data)} records")
    
    data = add_features(data)
    
    assets = {
        'Gold': 'Gold_Close',
        'Silver': 'Silver_Close',
        'Platinum': 'Platinum_Close',
        'USD_PKR': 'USD_PKR'
    }
    
    results = {}
    
    for asset_name, target_col in assets.items():
        print(f"\nTraining models for {asset_name}...")
        
        df_ml = create_lags(data.copy(), target_col)
        feature_cols = [c for c in df_ml.columns if c not in ['Target', target_col] and not c.endswith('_lag_')]
        lag_cols = [c for c in df_ml.columns if '_lag_' in c]
        all_features = feature_cols + lag_cols
        
        X = df_ml[all_features]
        y = df_ml['Target']
        
        lr_model, lr_scaler, lr_metrics = train_and_evaluate(X, y, 'LinearRegression')
        rf_model, rf_scaler, rf_metrics = train_and_evaluate(X, y, 'RandomForest')
        
        results[asset_name] = {
            'LinearRegression': {'model': lr_model, 'scaler': lr_scaler, 'metrics': lr_metrics},
            'RandomForest': {'model': rf_model, 'scaler': rf_scaler, 'metrics': rf_metrics}
        }
        
        print(f"  Linear Regression - RMSE: {lr_metrics['RMSE']:.2f}, MAPE: {lr_metrics['MAPE']:.2f}%")
        print(f"  Random Forest     - RMSE: {rf_metrics['RMSE']:.2f}, MAPE: {rf_metrics['MAPE']:.2f}%")
        
        joblib.dump(lr_model, f'model_{asset_name}_LR.pkl')
        joblib.dump(lr_scaler, f'scaler_{asset_name}_LR.pkl')
        joblib.dump(rf_model, f'model_{asset_name}_RF.pkl')
        joblib.dump(rf_scaler, f'scaler_{asset_name}_RF.pkl')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED - Models saved")
    print("="*60)
    
    print("\nPerformance Summary:")
    print(f"{'Asset':<12} {'Model':<18} {'RMSE':<10} {'MAE':<10} {'MAPE':<10}")
    print("-"*60)
    for asset in assets.keys():
        for model_type in ['LinearRegression', 'RandomForest']:
            m = results[asset][model_type]['metrics']
            print(f"{asset:<12} {model_type:<18} {m['RMSE']:<10.2f} {m['MAE']:<10.2f} {m['MAPE']:<10.2f}%")

if __name__ == "__main__":
    main()