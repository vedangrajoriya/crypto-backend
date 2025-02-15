import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta

# Load and Preprocess Data
data = pd.read_csv(
    'e:/project mini/ml-backend/corrected_data.csv',  # Use absolute path
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

# Save scaler for later use
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
np.save('scaler.npy', scaler.scale_)
np.save('min_vals.npy', scaler.min_)
np.save('data_shape.npy', scaled_data.shape)

# Create sequences
window_size = 60
X, y = [], []
for i in range(len(scaled_data) - window_size - 1):
    X.append(scaled_data[i:(i + window_size), :])
    y.append(scaled_data[i + window_size, :3])
X, y = np.array(X), np.array(y)

# Split train/test
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]

# Build and train model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(window_size, X_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1, callbacks=[early_stop])

# Save the model
model.save('e:/project mini/ml-backend/model.h5')  # Use absolute path

class CryptoPricePredictor:
    def __init__(self):
        # Use absolute path for the model file
        model_path = 'e:/project mini/ml-backend/model.h5'
        self.model = load_model(model_path)
        self.scaler = np.load('e:/project mini/ml-backend/scaler.npy')
        self.min_vals = np.load('e:/project mini/ml-backend/min_vals.npy')
        self.data_shape = np.load('data_shape.npy')
        self.window_size = 60

    def transform(self, data):
        return data * self.scaler + self.min_vals

    def inverse_transform(self, data):
        return (data - self.min_vals) / self.scaler

    def predict_for_date(self, target_date_str):
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
        last_date = datetime.now() - timedelta(days=1)
        
        if target_date <= last_date:
            raise ValueError("Target date must be after the last available date")
        
        days_ahead = (target_date - last_date).days
        
        # Generate sequence for prediction
        sequence = np.zeros((1, self.window_size, self.data_shape[1]))
        prediction = self.model.predict(sequence)
        
        # Transform back to original scale
        dummy_row = np.zeros((1, self.data_shape[1]))
        dummy_row[0, :3] = prediction[0]
        final_prediction = self.transform(dummy_row)[0, :3]
        
        return {
            'btc': float(final_prediction[0]),
            'eth': float(final_prediction[1]),
            'ltc': float(final_prediction[2])
        }