import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# --- Load and Preprocess Data ---
data = pd.read_csv(
    './cleaned_data.csv',
    skiprows=1,
    header=0,
    names=[
        'Date', 'Close_BTC-USD', 'Close_ETH-USD', 'Close_LTC-USD',
        'High_BTC-USD', 'High_ETH-USD', 'High_LTC-USD',
        'Low_BTC-USD', 'Low_ETH-USD', 'Low_LTC-USD',
        'Open_BTC-USD', 'Open_ETH-USD', 'Open_LTC-USD',
        'Volume_BTC-USD', 'Volume_ETH-USD', 'Volume_LTC-USD'
    ]
)

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data.ffill().dropna()

# --- Normalize Data ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# --- Create Sequences ---
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(len(data) - window_size - 1):
        X.append(data[i:(i + window_size), :])
        y.append(data[i + window_size, :3])  # Target: Close prices for BTC/ETH/LTC
    return np.array(X), np.array(y)

window_size = 60
X, y = create_sequences(scaled_data, window_size)

# Split train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- Build and Train Model ---
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(window_size, X_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1, callbacks=[early_stop])

# --- Evaluate Model Accuracy ---
def calculate_metrics(actual, predicted):
    metrics = {
        'MAE': np.mean(np.abs(actual - predicted)),
        'RMSE': np.sqrt(np.mean((actual - predicted)**2)),
        'MAPE': np.mean(np.abs((actual - predicted) / actual)) * 100
    }
    # Directional accuracy (up/down movement)
    direction_actual = np.sign(actual[1:] - actual[:-1])
    direction_pred = np.sign(predicted[1:] - predicted[:-1])
    metrics['Direction_Accuracy'] = np.mean(direction_actual == direction_pred) * 100
    return metrics

# Generate predictions
predictions = model.predict(X_test)

# Inverse scaling for predictions
dummy_features = np.zeros((len(predictions), scaled_data.shape[1]))
dummy_features[:, :3] = predictions
predictions = scaler.inverse_transform(dummy_features)[:, :3]

# Get actual values
test_dates = data.index[window_size + 1 : window_size + 1 + len(y_test)]
actual = data[['Close_BTC-USD', 'Close_ETH-USD', 'Close_LTC-USD']].loc[test_dates].values

# Calculate metrics for each cryptocurrency
cryptos = ['BTC-USD', 'ETH-USD', 'LTC-USD']
for i, crypto in enumerate(cryptos):
    print(f"\nAccuracy Metrics for {crypto}:")
    metrics = calculate_metrics(actual[:, i], predictions[:, i])
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}{'%' if k in ['MAPE', 'Direction_Accuracy'] else ''}")

# --- Visualize Predictions vs. Actual ---
# plt.figure(figsize=(15, 10))
# for i, crypto in enumerate(cryptos):
#     plt.subplot(3, 1, i+1)
#     plt.plot(test_dates, actual[:, i], label='Actual', linewidth=2)
#     plt.plot(test_dates, predictions[:, i], label='Predicted', linestyle='--')
#     plt.title(f'{crypto} Price Prediction')
#     plt.xlabel('Date')
#     plt.ylabel('Price (USD)')
#     plt.legend()
# plt.tight_layout()
# plt.show()

# --- Prediction Function (Same as Before) ---
def predict_for_date(target_date_str):
    last_date = data.index[-1]
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    
    if target_date <= last_date:
        raise ValueError("Target date must be after the last available date.")
    
    days_ahead = (target_date - last_date).days
    current_sequence = scaled_data[-window_size:]
    
    for _ in range(days_ahead):
        input_seq = current_sequence.reshape(1, window_size, -1)
        next_pred = model.predict(input_seq)
        new_row = np.concatenate([next_pred[0], current_sequence[-1, 3:]])
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    dummy_row = np.zeros((1, scaled_data.shape[1]))
    dummy_row[:, :3] = next_pred
    predicted_closes = scaler.inverse_transform(dummy_row)[0, :3]
    return {
        'btc': predicted_closes[0],
        'eth': predicted_closes[1],
        'ltc': predicted_closes[2],
    }

# # --- User Interaction ---
# if __name__ == "__main__":
#     target_date = input("\nEnter the date (YYYY-MM-DD) to predict prices: ")
    
#     try:
#         predictions = predict_for_date(target_date)
#         print(f"\nPredicted Close Prices for {target_date}:")
#         for crypto, price in zip(cryptos, predictions):
#             print(f"{crypto}: ${price:.2f}")
#     except ValueError as e:
#         print(f"Error: {e}")
