import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load and Preprocess Data
data = pd.read_csv(
    'e:/project mini/ml-backend/corrected_data.csv',
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
model.save('model.h5')