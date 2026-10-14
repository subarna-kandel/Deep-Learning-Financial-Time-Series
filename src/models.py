"""Deep learning neural network architectures (MLP, LSTM, CNN-GAF)."""

import numpy as np


def build_mlp_regressor(input_dim, output_dim, hidden_units=(64, 32)):
    """Build multi-layer perceptron for time series regression using Keras."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    model = Sequential()
    model.add(Dense(hidden_units[0], activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.2))
    for units in hidden_units[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(output_dim))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_lstm_model(input_shape, output_dim=1, units=64):
    """Build LSTM architecture for sequential return forecasting."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(units, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_cnn_gaf_model(input_shape, output_dim=1):
    """Build 2D CNN model accepting Gramian Angular Field input representations."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
