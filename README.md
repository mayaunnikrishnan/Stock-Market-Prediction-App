# Stock Price Prediction App

## Overview

The **Stock Price Prediction App** is designed to forecast future stock prices using a pre-trained Support Vector Regressor (SVR) model. The application is built using **Streamlit** for the user interface, **scikit-learn** for machine learning, and **Polygon.io** for stock data retrieval. Users can fetch stock data via an API, upload their own CSV files, or use a predefined dataset provided within the app.
To Visit the app [Follow Me]()
## Features

- **Data Retrieval Options**:
  - Fetch stock data from the Polygon.io API using your API key and stock symbol.
  - Upload a CSV file containing historical stock prices.
  - Use a predefined dataset (`stock_prediction_app.csv`) available within the app.

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
