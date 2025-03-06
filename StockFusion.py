import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import os
import yfinance as yf
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from snowflake.snowpark.context import get_active_session


# Set up yfinance cache directory
os.environ["YFINANCE_CACHE_DIR"] = "/tmp"

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

def add_meta_tag():
    meta_tag = """
    <head>
    <meta name="google-site-verification" content="QBiAoAo1GAkCBe1QoWq-dQ1RjtPHeFPyzkqJqsrqW-s" />
    </head>
    """
    st.markdown(meta_tag, unsafe_allow_html=True)

st.write('''# StockFusion ''')
st.sidebar.write('''# StockFusion ''')

with st.sidebar:
    selected = option_menu("Utilities", ["Stocks Performance Comparison", "Real-Time Stock Price", "Stock Prediction"])
    start = st.sidebar.date_input('Start', datetime.date(2015, 1, 1))
    end = st.sidebar.date_input('End', datetime.date.today())

def fetch_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(25, return_sequences=True, input_shape=input_shape),
        LSTM(25, return_sequences=False),
        Dense(10, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(data, look_back=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def predict_stock_price_lstm(data, n_days):
    X, y, scaler = prepare_data(data)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = create_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, batch_size=32, epochs=30, validation_split=0.1, verbose=0)
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform predictions
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform([y_test])
    
    # Calculate accuracy metrics
    train_rmse = np.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
    test_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
    train_mae = mean_absolute_error(y_train[0], train_predict[:,0])
    test_mae = mean_absolute_error(y_test[0], test_predict[:,0])
    train_r2 = r2_score(y_train[0], train_predict[:,0])
    test_r2 = r2_score(y_test[0], test_predict[:,0])
    
    # Future predictions
    last_60_days = data[-60:]
    future_predictions = []
    
    for _ in range(n_days):
        X_future = scaler.transform(last_60_days.reshape(-1, 1))
        X_future = np.reshape(X_future, (1, X_future.shape[0], 1))
        prediction = model.predict(X_future)
        future_predictions.append(scaler.inverse_transform(prediction)[0][0])
        last_60_days = np.append(last_60_days[1:], prediction)
    
    return future_predictions, {
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'Train R2': train_r2,
        'Test R2': test_r2
    }

def predict_stock_price_prophet(data, n_days):
    df = pd.DataFrame({'ds': data.index, 'y': data.values})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=n_days)
    forecast = model.predict(future)
    
    # Calculate accuracy metrics
    train_predictions = forecast['yhat'][:len(df)]
    train_rmse = np.sqrt(mean_squared_error(df['y'], train_predictions))
    train_mae = mean_absolute_error(df['y'], train_predictions)
    train_r2 = r2_score(df['y'], train_predictions)
    
    return forecast['yhat'].tail(n_days).values, {
        'Train RMSE': train_rmse,
        'Train MAE': train_mae,
        'Train R2': train_r2
    }

if selected == 'Stocks Performance Comparison':
    st.subheader("Stocks Performance Comparison")
    ticker_input = st.text_input("Enter ticker symbols (comma-separated, e.g., AAPL,MSFT,GOOGL)")
    if ticker_input:
        tickers = [ticker.strip().upper() for ticker in ticker_input.split(',')]
        with st.spinner('Loading...'):
            data = fetch_data(tickers, start, end)
            data = data.sort_index(ascending=False)  # Sort by index in descending order
            if not data.empty:
                st.subheader('Raw Data')
                st.write(data)

                # Combined Close Price Graph
                st.subheader('Close Price')
                fig_close = go.Figure()
                for ticker in tickers:
                    fig_close.add_trace(go.Scatter(x=data.index, y=data[f'Close_{ticker}'], name=f'{ticker} Close'))
                st.plotly_chart(fig_close)

                # Combined Open Price Graph
                st.subheader('Open Price')
                fig_open = go.Figure()
                for ticker in tickers:
                    fig_open.add_trace(go.Scatter(x=data.index, y=data[f'Open_{ticker}'], name=f'{ticker} Open'))
                st.plotly_chart(fig_open)

                # Combined Volume Graph
                st.subheader('Volume')
                fig_volume = go.Figure()
                for ticker in tickers:
                    fig_volume.add_trace(go.Bar(x=data.index, y=data[f'Volume_{ticker}'], name=f'{ticker} Volume'))
                st.plotly_chart(fig_volume)
            else:
                st.error("No data found for these ticker symbols")
    else:
        st.write('Please enter at least one ticker symbol')

elif selected == 'Real-Time Stock Price':
    st.subheader("Real-Time Stock Price")
    ticker = st.text_input("Enter ticker symbol (e.g., AAPL)")
    if st.button("Search") and ticker:
        with st.spinner('Loading...'):
            data = fetch_data([ticker], start, end)
            data = data.sort_index(ascending=False)  # Sort by index in descending order
            if not data.empty:
                data.reset_index(inplace=True)
                st.subheader(f'Raw Data of {ticker}')
                st.write(data)

                def plot_raw_data():
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data[f'Open_{ticker}'], name="stock_open"))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data[f'Close_{ticker}'], name="stock_close"))
                    fig.layout.update(title_text=f' Line Chart of {ticker}', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig)

                def plot_candle_data():
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=data['Date'], open=data[f'Open_{ticker}'], high=data[f'High_{ticker}'], low=data[f'Low_{ticker}'], close=data[f'Close_{ticker}'], name='market data'))
                    fig.update_layout(title=f'Candlestick Chart of {ticker}', yaxis_title='Stock Price', xaxis_title='Date')
                    st.plotly_chart(fig)

                chart = ('Candle Stick')
                dropdown1 = st.selectbox('Pick your chart', chart)
                if dropdown1 == 'Candle Stick':
                    plot_candle_data()
            else:
                st.error("No data found for this ticker symbol")

elif selected == 'Stock Prediction':
    st.subheader("Stock Prediction")
    ticker = st.text_input("Enter ticker symbol (e.g., AAPL)")
    
    if ticker:
        with st.spinner('Loading...'):
            data = fetch_data([ticker], start, end)
            
            if not data.empty:
                data.reset_index(inplace=True)
                st.subheader(f'Raw Data of {ticker}')
                st.write(data)

                def plot_raw_data():
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data[f'Open_{ticker}'], name="Stock Open"))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data[f'Close_{ticker}'], name="Stock Close"))
                    fig.layout.update(title_text=f'Time Series Data of {ticker}', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig)

                plot_raw_data()
                
                n_days = st.slider('Days of prediction:', 1, 365)
                
                model_choice = st.radio("Choose prediction model:", ("LSTM", "Prophet"))

                close_prices = data[f'Close_{ticker}'].values
                if model_choice == 'LSTM':
                    with st.spinner("Training LSTM model..."):
                        predictions, metrics = predict_stock_price_lstm(close_prices, n_days)
                    
                    st.subheader("LSTM Prediction Results")
                    st.write(f"Predicted prices for the next {n_days} days:")
                                    
                    # Create a DataFrame with actual and predicted prices
                    prediction_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_days)
                    pred_df = pd.DataFrame({
                        'Date': prediction_dates,
                        'Predicted Close': predictions,
                        'Actual Close': [None] * n_days  # We don't have actual values for future dates
                    })
                    # Display the prediction table
                    st.write(pred_df)
                    
                    st.subheader("Model Performance Metrics")
                    for metric, value in metrics.items():
                        st.write(f"{metric}: {value:.4f}")
                    # Calculate overall accuracy
                    accuracy = (metrics['Train R2'] + metrics['Test R2']) / 2 * 100
                    st.write(f"Overall Accuracy: {accuracy:.2f}%")
                    

                    # Plot actual vs predicted
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data[f'Close_{ticker}'], name='Actual Close Price', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=pd.date_range(start=data['Date'].iloc[-1], periods=n_days+1, freq='D')[1:], 
                                             y=predictions, name='Predicted Close Price', line=dict(color='red', dash='dot')))
                    fig.update_layout(title=f'{ticker} Stock Price Prediction', xaxis_title='Date', yaxis_title='Close Price')
                    st.plotly_chart(fig)

                elif model_choice == 'Prophet':
                    with st.spinner("Training Prophet model..."):
                        predictions, metrics = predict_stock_price_prophet(pd.Series(close_prices, index=data['Date']), n_days)
                    
                    st.subheader("Prophet Prediction Results")
                    st.write(f"Predicted prices for the next {n_days} days:")
                     # Create a DataFrame with actual and predicted prices
                    prediction_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_days)
                    pred_df = pd.DataFrame({
                        'Date': prediction_dates,
                        'Predicted Close': predictions,
                        'Actual Close': [None] * n_days  # We don't have actual values for future dates
                    })
                    # Display the prediction table
                    st.write(pred_df)

                    
                    st.subheader("Model Performance Metrics")
                    for metric, value in metrics.items():
                        st.write(f"{metric}: {value:.4f}")
                     # Calculate overall accuracy
                    accuracy = metrics['Train R2'] * 100
                    st.write(f"Overall Accuracy: {accuracy:.2f}%")
                    
                    
                    # Plot actual vs predicted
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data[f'Close_{ticker}'], name='Actual Price', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=pd.date_range(start=data['Date'].iloc[-1], periods=n_days+1, freq='D')[1:], 
                                             y=predictions, name='Predicted Price', line=dict(color='red', dash='dot')))
                    fig.update_layout(title=f'{ticker} Stock Price Prediction', xaxis_title='Date', yaxis_title='Close Price', )
                    st.plotly_chart(fig)

            else:
                st.error("No data found for this ticker symbol")
                    
# Footer
st.markdown("---")
st.markdown("<h2 style='text-align: center; color: #1E88E5;'>About the Developers</h2>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 3])
st.markdown("---")
with col1:
    st.image("https://img.icons8.com/color/240/000000/user-male-circle--v1.png", width=150)
with col2:
    st.markdown("<h3 style='color: #1E88E5;'>Ahmad Saraj and Arbaz Shafiq</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-style: italic;'>Financial Data Analyst & Machine Learning Enthusiast</p>", unsafe_allow_html=True)
    st.markdown("Passionate about leveraging data science and machine learning to unlock insights in financial markets.")

st.markdown("<h4 style='text-align: center; color: #1E88E5; margin-top: 20px;'>Connect with Us</h4>", unsafe_allow_html=True)

social_media = {
    "Ahmad Saraj": {
        "LinkedIn": {"url": "https://www.linkedin.com/in/ahmad-saraj-69a73a292/", "icon": "https://img.icons8.com/color/48/000000/linkedin.png"},
        "GitHub": {"url": "https://github.com/ahmadsaraj246", "icon": "https://img.icons8.com/fluent/48/000000/github.png"}
    },
    "Arbaz Shafiq": {
        "LinkedIn": {"url": "https://www.linkedin.com/in/arbaz-shafiq-802374267/", "icon": "https://img.icons8.com/color/48/000000/linkedin.png"},
        "GitHub": {"url": "https://github.com/malik087", "icon": "https://img.icons8.com/fluent/48/000000/github.png"}
    }
}

for developer, links in social_media.items():
    st.markdown(f"<h4 style='text-align: center; color: #1E88E5; margin-top: 20px;'>{developer}</h4>", unsafe_allow_html=True)
    cols = st.columns(len(links))
    for index, (platform, info) in enumerate(links.items()):
        with cols[index]:
            st.markdown(f"""
            <a href="{info['url']}" target="_blank">
                <img src="{info['icon']}" width="40" height="40" style="display: block; margin: auto;">
                <p style="text-align: center; font-size: 0.8em; margin-top: 5px;">{platform}</p>
            </a>
            """, unsafe_allow_html=True)

st.markdown("<p style='text-align: center; font-size: 0.8em; margin-top: 30px;'>© 2025 Ahmad Saraj. All rights reserved.</p>", unsafe_allow_html=True)
