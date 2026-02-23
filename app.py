import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Pakistan Stock & Metals Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    """Load historical data for commodities and USD/PKR"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # Fetch data
    gold = yf.download("GC=F", start=start_date, end=end_date, progress=False)
    silver = yf.download("SI=F", start=start_date, end=end_date, progress=False)
    platinum = yf.download("PL=F", start=start_date, end=end_date, progress=False)
    usd_pkr = yf.download("PKR=X", start=start_date, end=end_date, progress=False)
    
    # Process data - FIX: Properly extract single column and handle MultiIndex
    data = pd.DataFrame({
        'Gold_Close': gold['Close'].squeeze(),
        'Gold_Open': gold['Open'].squeeze(),
        'Gold_High': gold['High'].squeeze(),
        'Gold_Low': gold['Low'].squeeze(),
        'Gold_Volume': gold['Volume'].squeeze(),
        'Silver_Close': silver['Close'].squeeze(),
        'Silver_Open': silver['Open'].squeeze(),
        'Silver_High': silver['High'].squeeze(),
        'Silver_Low': silver['Low'].squeeze(),
        'Platinum_Close': platinum['Close'].squeeze(),
        'Platinum_Open': platinum['Open'].squeeze(),
        'Platinum_High': platinum['High'].squeeze(),
        'Platinum_Low': platinum['Low'].squeeze(),
        'USD_PKR_Close': usd_pkr['Close'].squeeze(),
        'USD_PKR_Open': usd_pkr['Open'].squeeze(),
        'USD_PKR_High': usd_pkr['High'].squeeze(),
        'USD_PKR_Low': usd_pkr['Low'].squeeze(),
    })
    
    # Ensure all columns are 1-dimensional
    for col in data.columns:
        if isinstance(data[col], pd.DataFrame):
            data[col] = data[col].iloc[:, 0]
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data = data.fillna(method='ffill').dropna()
    return data

def add_technical_indicators(df, prefix=''):
    """Add technical indicators"""
    df = df.copy()
    close_col = f'{prefix}Close' if prefix else 'Close'
    
    if close_col in df.columns:
        df[f'{prefix}MA7'] = df[close_col].rolling(window=7).mean()
        df[f'{prefix}MA30'] = df[close_col].rolling(window=30).mean()
        df[f'{prefix}Returns'] = df[close_col].pct_change()
        df[f'{prefix}Volatility'] = df[close_col].rolling(window=7).std()
        
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df[f'{prefix}RSI'] = 100 - (100 / (1 + rs))
        
        ema12 = df[close_col].ewm(span=12, adjust=False).mean()
        ema26 = df[close_col].ewm(span=26, adjust=False).mean()
        df[f'{prefix}MACD'] = ema12 - ema26
        df[f'{prefix}Signal'] = df[f'{prefix}MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def prepare_ml_data(df, target_col, feature_cols, prediction_days=5):
    """Prepare data for machine learning"""
    df_ml = df.copy()
    
    for i in range(1, prediction_days + 1):
        df_ml[f'lag_{i}'] = df_ml[target_col].shift(i)
    
    df_ml['Target'] = df_ml[target_col].shift(-1)
    df_ml = df_ml.dropna()
    
    all_features = feature_cols + [f'lag_{i}' for i in range(1, prediction_days + 1)]
    X = df_ml[all_features]
    y = df_ml['Target']
    
    return X, y, all_features

def train_models(X, y):
    """Train multiple models"""
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    def metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        return rmse, mae, mape
    
    lr_metrics = metrics(y_test, lr_pred)
    rf_metrics = metrics(y_test, rf_pred)
    
    return {
        'Linear Regression': {'model': lr_model, 'scaler': scaler, 'metrics': lr_metrics, 'pred': lr_pred, 'actual': y_test},
        'Random Forest': {'model': rf_model, 'scaler': scaler, 'metrics': rf_metrics, 'pred': rf_pred, 'actual': y_test}
    }

def main():
    st.markdown('<h1 class="main-header">📈 Pakistan Stock & Metals Price Prediction</h1>', unsafe_allow_html=True)
    
    st.sidebar.title("⚙️ Configuration")
    
    asset_type = st.sidebar.selectbox(
        "Select Asset",
        ["Gold", "Silver", "Platinum", "USD/PKR"]
    )
    
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["Linear Regression", "Random Forest"]
    )
    
    prediction_days = st.sidebar.slider("Prediction Window (Days)", 1, 30, 7)
    show_technical = st.sidebar.checkbox("Show Technical Indicators", True)
    
    with st.spinner("Loading market data..."):
        data = load_data()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Prediction", "📈 Analysis", "📋 Data"])
    
    with tab1:
        st.header(f"{asset_type} Market Dashboard")
        
        # FIX: Correct column mapping for each asset type
        if asset_type == "Gold":
            current_price = float(data['Gold_Close'].iloc[-1])
            prev_price = float(data['Gold_Close'].iloc[-2])
            col_prefix = 'Gold_'
        elif asset_type == "Silver":
            current_price = float(data['Silver_Close'].iloc[-1])
            prev_price = float(data['Silver_Close'].iloc[-2])
            col_prefix = 'Silver_'
        elif asset_type == "Platinum":
            current_price = float(data['Platinum_Close'].iloc[-1])
            prev_price = float(data['Platinum_Close'].iloc[-2])
            col_prefix = 'Platinum_'
        else:  # USD/PKR
            current_price = float(data['USD_PKR_Close'].iloc[-1])
            prev_price = float(data['USD_PKR_Close'].iloc[-2])
            col_prefix = 'USD_PKR_'
        
        change = ((current_price - prev_price) / prev_price) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${current_price:,.2f}", f"{change:.2f}%")
        with col2:
            high_val = float(data[f'{col_prefix}High'].iloc[-1])
            st.metric("24h High", f"${high_val:,.2f}")
        with col3:
            low_val = float(data[f'{col_prefix}Low'].iloc[-1])
            st.metric("24h Low", f"${low_val:,.2f}")
        with col4:
            if f'{col_prefix}Volume' in data.columns:
                volume = float(data[f'{col_prefix}Volume'].iloc[-1])
                st.metric("Volume", f"{volume:,.0f}")
            else:
                st.metric("Volume", "N/A")
        
        fig = go.Figure()
        close_values = data[f'{col_prefix}Close'].values
        fig.add_trace(go.Scatter(
            x=list(data.index),
            y=close_values,
            mode='lines',
            name=f'{asset_type} Price',
            line=dict(color='#FFD700' if asset_type == 'Gold' else '#C0C0C0' if asset_type == 'Silver' else '#E5E4E2', width=2)
        ))
        
        if show_technical:
            ma7 = data[f'{col_prefix}Close'].rolling(window=7).mean().values
            ma30 = data[f'{col_prefix}Close'].rolling(window=30).mean().values
            fig.add_trace(go.Scatter(x=list(data.index), y=ma7, mode='lines', name='MA7', line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=list(data.index), y=ma30, mode='lines', name='MA30', line=dict(color='blue', width=1)))
        
        fig.update_layout(
            title=f'{asset_type} Price Trend',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_white',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if asset_type != "USD/PKR":
            st.subheader("Correlation with USD/PKR Exchange Rate")
            correlation = data[f'{col_prefix}Close'].corr(data['USD_PKR_Close'])
            
            fig2 = go.Figure()
            x_vals = data['USD_PKR_Close'].values
            y_vals = data[f'{col_prefix}Close'].values
            fig2.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                name=f'{asset_type} vs USD/PKR',
                marker=dict(color='purple', size=4, opacity=0.6)
            ))
            
            z = np.polyfit(data['USD_PKR_Close'].dropna(), data[f'{col_prefix}Close'].dropna(), 1)
            p = np.poly1d(z)
            sorted_x = sorted(x_vals)
            fig2.add_trace(go.Scatter(
                x=sorted_x,
                y=p(sorted_x),
                mode='lines',
                name=f'Trend (r={correlation:.3f})',
                line=dict(color='red', dash='dash')
            ))
            
            fig2.update_layout(
                title=f'{asset_type} vs USD/PKR (Correlation: {correlation:.3f})',
                xaxis_title='USD/PKR Rate',
                yaxis_title=f'{asset_type} Price',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.header("🔮 Price Prediction")
        
        if asset_type == "Gold":
            target = 'Gold_Close'
            features = ['Gold_MA7', 'Gold_RSI', 'Gold_Volatility', 'Silver_Close', 'Platinum_Close', 'USD_PKR_Close']
        elif asset_type == "Silver":
            target = 'Silver_Close'
            features = ['Silver_MA7', 'Gold_Close', 'Platinum_Close', 'USD_PKR_Close']
        elif asset_type == "Platinum":
            target = 'Platinum_Close'
            features = ['Platinum_MA7', 'Gold_Close', 'Silver_Close', 'USD_PKR_Close']
        else:
            target = 'USD_PKR_Close'
            features = ['USD_PKR_MA7', 'Gold_Close', 'Silver_Close']
        
        data_with_indicators = add_technical_indicators(data, 'Gold_')
        data_with_indicators = add_technical_indicators(data_with_indicators, 'Silver_')
        data_with_indicators = add_technical_indicators(data_with_indicators, 'Platinum_')
        data_with_indicators = add_technical_indicators(data_with_indicators, 'USD_PKR_')
        
        try:
            X, y, all_features = prepare_ml_data(data_with_indicators, target, features, prediction_days)
            
            if len(X) > 100:
                models = train_models(X, y)
                selected_model = models[model_type]
                
                st.subheader("Model Performance")
                rmse, mae, mape = selected_model['metrics']
                
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.markdown(f'<div class="metric-card"><h3>RMSE</h3><h2>${rmse:,.2f}</h2></div>', unsafe_allow_html=True)
                with m_col2:
                    st.markdown(f'<div class="metric-card"><h3>MAE</h3><h2>${mae:,.2f}</h2></div>', unsafe_allow_html=True)
                with m_col3:
                    st.markdown(f'<div class="metric-card"><h3>MAPE</h3><h2>{mape:.2f}%</h2></div>', unsafe_allow_html=True)
                
                # FIX: Convert range to list for plotly
                actual_values = list(selected_model['actual'])
                pred_values = list(selected_model['pred'])
                x_range = list(range(len(actual_values)))
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=x_range,
                    y=actual_values,
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue', width=2)
                ))
                fig_pred.add_trace(go.Scatter(
                    x=x_range,
                    y=pred_values,
                    mode='lines',
                    name='Predicted',
                    line=dict(color='red', width=2, dash='dash')
                ))
                fig_pred.update_layout(
                    title=f'{model_type} Predictions vs Actual',
                    xaxis_title='Time',
                    yaxis_title='Price',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                
                st.subheader("📅 Next Day Prediction")
                last_row = X.iloc[-1:].values
                last_row_scaled = selected_model['scaler'].transform(last_row)
                next_pred = selected_model['model'].predict(last_row_scaled)[0]
                
                current = float(data[target].iloc[-1])
                pred_change = ((next_pred - current) / current) * 100
                
                st.markdown(f'<div class="prediction-box">Predicted {asset_type} Price: ${next_pred:,.2f}<br><small>Expected Change: {pred_change:+.2f}%</small></div>', unsafe_allow_html=True)
                
            else:
                st.warning("Insufficient data for prediction. Please check data sources.")
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
    
    with tab3:
        st.header("📈 Technical Analysis")
        
        if asset_type != "USD/PKR":
            close_series = data[f'{col_prefix}Close']
            delta = close_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=list(data.index), y=rsi.values, mode='lines', name='RSI', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig_rsi.update_layout(title='RSI (Relative Strength Index)', height=300, template='plotly_white')
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            volatility = close_series.rolling(window=7).std()
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(x=list(data.index), y=volatility.values, name='Volatility', marker_color='orange'))
            fig_vol.update_layout(title='7-Day Rolling Volatility', height=300, template='plotly_white')
            st.plotly_chart(fig_vol, use_container_width=True)
    
    with tab4:
        st.header("📋 Raw Data")
        st.dataframe(data.tail(100))
        
        csv = data.to_csv().encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f'market_data_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()