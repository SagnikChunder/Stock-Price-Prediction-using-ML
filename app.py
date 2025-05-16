import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import gradio as gr
import tempfile
import os

# Load model
model = load_model("Stock Predictions Model.keras")

# Plot function helpers
def plot_ma(data, ma_days, color, label):
    return data.Close.rolling(ma_days).mean(), color, label

def plot_figure(data, lines_info, title):
    plt.figure(figsize=(8,6))
    for line, color, label in lines_info:
        plt.plot(line, color, label=label)
    plt.plot(data.Close, 'g', label='Close Price')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    tmpfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(tmpfile.name)
    plt.close()
    return tmpfile.name

# Main prediction function
def stock_predictor(stock_symbol):
    start = '2012-01-01'
    end = '2024-10-01'
    
    data = yf.download(stock_symbol, start, end)
    
    if data.empty:
        return "Invalid stock symbol or no data found", None, None, None, None

    # Train-test split
    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

    # Scaling
    scaler = MinMaxScaler(feature_range=(0,1))
    pas_100_days = data_train.tail(100)
    data_test_full = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test_full)

    # Moving Averages
    ma_50, _, _ = plot_ma(data, 50, 'r', 'MA50')
    ma_100, _, _ = plot_ma(data, 100, 'b', 'MA100')
    ma_200, _, _ = plot_ma(data, 200, 'y', 'MA200')

    # Prepare test data
    x, y = [], []
    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i,0])
    x, y = np.array(x), np.array(y)

    # Predict
    predictions = model.predict(x)
    scale = 1 / scaler.scale_[0]
    predictions = predictions * scale
    y = y * scale

    # Plots
    ma50_fig = plot_figure(data, [(ma_50, 'r', 'MA50')], 'Price vs MA50')
    ma100_fig = plot_figure(data, [(ma_50, 'r', 'MA50'), (ma_100, 'b', 'MA100')], 'Price vs MA50 vs MA100')
    ma200_fig = plot_figure(data, [(ma_100, 'r', 'MA100'), (ma_200, 'b', 'MA200')], 'Price vs MA100 vs MA200')

    # Prediction vs Original plot
    plt.figure(figsize=(8,6))
    plt.plot(predictions, 'r', label='Predicted Price')
    plt.plot(y, 'g', label='Original Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.title("Original vs Predicted Price")
    pred_plot_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(pred_plot_file.name)
    plt.close()

    return "Data fetched and processed successfully!", ma50_fig, ma100_fig, ma200_fig, pred_plot_file.name

# Gradio interface
interface = gr.Interface(
    fn=stock_predictor,
    inputs=gr.Textbox(label="Enter Stock Symbol", placeholder="e.g. GOOG"),
    outputs=[
        gr.Textbox(label="Status"),
        gr.Image(label="Price vs MA50"),
        gr.Image(label="Price vs MA50 vs MA100"),
        gr.Image(label="Price vs MA100 vs MA200"),
        gr.Image(label="Original vs Predicted Price"),
    ],
    title="Stock Market Price Predictor",
    description="Predicts stock prices using a pre-trained deep learning model."
)

interface.launch()
