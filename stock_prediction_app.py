import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import requests

# Load the pre-trained model and scaler
svr_model = joblib.load('svr_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load predefined dataset available in the backend
def load_predefined_dataset():
    return pd.read_csv('stock_market_data.csv')

# Function to fetch data from Polygon.io
def fetch_stock_data(api_key, symbol, timespan="day", from_date="2023-01-01", to_date="2023-08-01"):
    url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}/{from_date}/{to_date}?adjusted=true&sort=asc&limit=120&apiKey={api_key}'
    response = requests.get(url)
    data = response.json()
    if "results" in data:
        dates = [item['t'] for item in data['results']]
        closing_prices = [item['c'] for item in data['results']]
        df = pd.DataFrame({
            'date': pd.to_datetime(dates, unit='ms'),
            'closing_price': closing_prices
        })
        return df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if there's no data

# Streamlit UI
st.title('Future Stock Price Prediction using Pre-trained SVR')

# User input for API key, CSV file, or predefined dataset
api_key = st.text_input('Enter your Polygon.io API key (optional)')
symbol = st.text_input('Enter Stock Symbol', value='AAPL')
uploaded_file = st.file_uploader("Upload a CSV file (optional)", type="csv")
predefined_dataset = st.selectbox("Select a predefined dataset (optional)", ["None", "stock_market_data.csv"])

# Number of days to predict
num_days = st.slider('Select number of days to predict', 1, 30, 7)

# Button to trigger prediction
if st.button('Fetch and Predict'):
    df = pd.DataFrame()

    if api_key:
        # Fetch the stock data using API
        df = fetch_stock_data(api_key, symbol)
        if df.empty:
            st.warning("No data returned from API. Please check your API key and symbol.")
    elif uploaded_file:
        # Load data from uploaded CSV
        df = pd.read_csv(uploaded_file)
    elif predefined_dataset == "stock_market_data.csv":
        # Load predefined dataset
        df = load_predefined_dataset()

    if not df.empty:
        # Verify if the required columns exist
        if 'date' not in df.columns or 'closing_price' not in df.columns:
            st.error("The dataset must contain 'date' and 'closing_price' columns.")
        else:
            # Ensure 'date' is datetime type
            df['date'] = pd.to_datetime(df['date'])

            # Data Preparation
            df['price_change'] = df['closing_price'].pct_change()
            df['moving_average'] = df['closing_price'].rolling(window=5).mean()
            df = df.dropna()

            # Features
            X = df[['price_change', 'moving_average']].values

            # Feature Scaling
            X_scaled = scaler.transform(X)

            # Predict future prices
            last_date = df['date'].max()
            future_dates = pd.date_range(last_date, periods=num_days + 1, freq='B')[1:]

            # Generate future predictions
            last_moving_average = df['moving_average'].iloc[-1]
            last_price_change = df['price_change'].iloc[-1]
            future_predictions = []
            for _ in range(num_days):
                X_future = np.array([[last_price_change, last_moving_average]])
                X_future_scaled = scaler.transform(X_future)
                future_price = svr_model.predict(X_future_scaled)[0]
                future_predictions.append(future_price)
                # Update moving average and price change for next prediction
                last_price_change = (future_price - df['closing_price'].iloc[-1]) / df['closing_price'].iloc[-1]
                last_moving_average = (last_moving_average * 4 + future_price) / 5

            # Append future predictions to the original DataFrame
            future_df = pd.DataFrame({
                'date': future_dates,
                'closing_price': future_predictions
            })

            # Combine with existing data
            combined_df = pd.concat([df[['date', 'closing_price']], future_df])

            # Visualization
            plt.figure(figsize=(12, 6))
            plt.plot(df['date'], df['closing_price'], label='Historical Prices', color='blue')
            plt.plot(future_df['date'], future_df['closing_price'], label='Future Predictions', color='red')
            plt.axvline(x=last_date, color='green', linestyle='--', label='Prediction Start')
            plt.xlabel('Date')
            plt.ylabel('Closing Price')
            plt.title(f'Future Stock Price Prediction for {symbol}')
            plt.legend()
            st.pyplot(plt.gcf())

            st.write(f"Predicted Prices for the next {num_days} days:")
            st.write(future_df)
    else:
        st.error("No data available to make predictions.")




