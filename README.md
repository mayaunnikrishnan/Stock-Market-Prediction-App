# Stock Price Prediction App

## Overview

The **Stock Price Prediction App** is designed to forecast future stock prices using a pre-trained Support Vector Regressor (SVR) model. The application is built using **Streamlit** for the user interface, **scikit-learn** for machine learning, and **Polygon.io** for stock data retrieval. Users can fetch stock data via an API, upload their own CSV files, or use a predefined dataset provided within the app.
To Visit the app [Follow Me](https://stock-market-prediction-appgit-62ykqnsh6j9bcfhlhx3twa.streamlit.app/)
## Features

- **Data Retrieval Options**:
  - Fetch stock data from the Polygon.io API using your API key and stock symbol.
  - Upload a CSV file containing historical stock prices.
  - Use a predefined dataset (`stock_prediction_app.csv`) available within the app.
 
### For Uploaded CSV Files

If you choose to upload your own CSV file, ensure that the file contains the following columns:

- **date**: The date of the stock data (in any standard date format).
- **closing_price**: The closing price of the stock on that date.

The column names must be exactly as specified (`date` and `closing_price`). The file should be in CSV format.

### For Predefined Dataset

The app includes a predefined dataset (`stock_market_data.csv`) that already contains the required columns. You can select this option if you do not wish to provide your own data.

- **Prediction Capabilities**:
  - Predict stock prices for a specified number of future days.
  - Visualize historical stock prices and future predictions on a graph.

## Requirements

Ensure you have the following Python packages installed:

- Python 3.7+
- Streamlit
- Pandas
- scikit-learn
- Matplotlib
- Requests
- Joblib

You can install these dependencies by running:

```bash
pip install -r requirements.txt
