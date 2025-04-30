import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import math


class WeatherForecastModel:
    def __init__(self, model_dir='models/saved_models'):
        """Initialize weather forecasting model."""
        self.model_dir = model_dir
        self.model = None
        self.scalers = {}
        self.target_cols = ['temperature', 'rain_analog', 'aqi']  # Define target columns explicitly
        self.lookback = 24  # Default lookback window
        self.forecast_horizon = 7  # Default forecast horizon

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

    def preprocess_data(self, df, feature_cols=None, target_cols=None):
        """Preprocess data for model training."""
        # Default feature and target columns if not specified
        if feature_cols is None:
            feature_cols = [
                'temperature', 'humidity', 'pressure',
                'rain_analog', 'light_intensity',
                'nh3', 'co', 'co2', 'aqi',
                'timestamp'  # Will be converted to time features
            ]

        if target_cols is None:
            target_cols = self.target_cols

        # Create time-based features
        df = self._create_time_features(df)

        # Add all time features to feature_cols
        time_features = ['hour', 'day_of_week', 'month', 'day_of_month', 'day_of_year']
        for feat in time_features:
            if feat not in feature_cols and feat in df.columns:
                feature_cols.append(feat)

        # Scale the data
        X, y = self._scale_data(df, feature_cols, target_cols)

        # Split into training and test sets (chronological split)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, X_test, y_train, y_test, split_idx

    def _create_time_features(self, df):
        """Create time-based features from datetime or timestamp."""
        df_copy = df.copy()

        # Convert timestamp to datetime if needed
        if 'datetime' not in df_copy and 'timestamp' in df_copy:
            df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'], unit='s')
        elif 'datetime' in df_copy and isinstance(df_copy['datetime'].iloc[0], str):
            df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])

        if 'datetime' in df_copy:
            # Extract time features
            df_copy['hour'] = df_copy['datetime'].dt.hour
            df_copy['day_of_week'] = df_copy['datetime'].dt.dayofweek
            df_copy['month'] = df_copy['datetime'].dt.month
            df_copy['day_of_month'] = df_copy['datetime'].dt.day
            df_copy['day_of_year'] = df_copy['datetime'].dt.dayofyear

            # Calculate cyclical time features
            df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
            df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
            df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
            df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
            df_copy['day_of_week_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
            df_copy['day_of_week_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)

        return df_copy

    def _scale_data(self, df, feature_cols, target_cols):
        """Scale features and targets."""
        # Create a scaler for each feature and target column
        for col in feature_cols + target_cols:
            if col not in self.scalers:
                self.scalers[col] = MinMaxScaler()
                # Reshape to ensure 2D array for scaler
                self.scalers[col].fit(df[col].values.reshape(-1, 1))

        # Scale features
        X = np.zeros((len(df), len(feature_cols)))
        for i, col in enumerate(feature_cols):
            X[:, i] = self.scalers[col].transform(df[col].values.reshape(-1, 1)).flatten()

        # Scale targets
        y = np.zeros((len(df), len(target_cols)))
        for i, col in enumerate(target_cols):
            y[:, i] = self.scalers[col].transform(df[col].values.reshape(-1, 1)).flatten()

        return X, y

    def create_sequences(self, X, y, lookback=None, forecast_horizon=None):
        """Create sequences for time series prediction."""
        # Use instance variables if not specified
        if lookback is None:
            lookback = self.lookback
        if forecast_horizon is None:
            forecast_horizon = self.forecast_horizon

        X_seq, y_seq = [], []

        for i in range(len(X) - lookback - forecast_horizon + 1):
            # Input sequence
            X_seq.append(X[i:i + lookback])

            # Target sequence (multiple steps ahead)
            y_seq.append(y[i + lookback:i + lookback + forecast_horizon])

        return np.array(X_seq), np.array(y_seq)

    def build_model(self, input_shape, output_shape):
        """Build an LSTM model for weather forecasting."""
        model = Sequential()

        # Input layer
        model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        # Hidden layers
        model.add(LSTM(64, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        # Output layer - reshape to match output_shape
        model.add(Dense(output_shape[0] * output_shape[1]))
        model.add(tf.keras.layers.Reshape(output_shape))

        # Compile the model - use the actual function object
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.MeanSquaredError()  # Use actual function object
        )

        return model

    def predict(self, current_data, lookback_data=None, days=7):
        """Make weather predictions for the next N days."""
        # Load the model if not already loaded
        if self.model is None:
            if not self.load_saved_model():
                return None

        # Convert current_data to DataFrame if it's a dict
        if isinstance(current_data, dict):
            current_data = pd.DataFrame([current_data])

        # Add time features if they don't exist
        current_data_processed = self._create_time_features(current_data)

        # Get feature columns from the scalers
        feature_cols = [col for col in self.scalers.keys()
                        if col in current_data_processed.columns]

        # Use the predefined target columns
        target_cols = self.target_cols

        # Scale the current data
        X_current = np.zeros((len(current_data_processed), len(feature_cols)))
        for i, col in enumerate(feature_cols):
            if col in current_data_processed.columns:
                X_current[:, i] = self.scalers[col].transform(
                    current_data_processed[col].values.reshape(-1, 1)
                ).flatten()

        # Use the instance variables for lookback and forecast_horizon
        lookback = self.lookback
        forecast_horizon = self.forecast_horizon

        # Create a sequence using lookback data if available
        if lookback_data is not None and len(lookback_data) >= lookback:
            # Process lookback data
            lookback_processed = self._create_time_features(lookback_data)

            # Scale lookback data
            X_lookback = np.zeros((len(lookback_processed), len(feature_cols)))
            for i, col in enumerate(feature_cols):
                if col in lookback_processed.columns:
                    X_lookback[:, i] = self.scalers[col].transform(
                        lookback_processed[col].values.reshape(-1, 1)
                    ).flatten()

            # Use the last 'lookback' timesteps
            X_seq = X_lookback[-lookback:].reshape(1, lookback, -1)
        else:
            # If no lookback data, repeat current data
            X_seq = np.repeat(X_current, lookback, axis=0).reshape(1, lookback, -1)

        # Get initial prediction
        y_pred = self.model.predict(X_seq)

        # For multi-day forecasts beyond model horizon, use iterative prediction
        all_predictions = [y_pred[0]]

        # If we need more days than the forecast horizon
        remaining_days = max(0, days - forecast_horizon)

        if remaining_days > 0:
            # Iterative prediction (use previous predictions as input for next)
            for _ in range(remaining_days // forecast_horizon):
                # Update input sequence:
                # Remove oldest timesteps and add predictions
                y_pred_flat = y_pred[0].reshape(forecast_horizon, -1)
                X_seq = np.concatenate([X_seq[0, forecast_horizon:], y_pred_flat])
                X_seq = X_seq.reshape(1, lookback, -1)

                # Generate next prediction
                y_pred = self.model.predict(X_seq)
                all_predictions.append(y_pred[0])

        # Combine all predictions
        combined_predictions = np.concatenate(all_predictions)[:days]

        # Determine the shape of the predictions for safe access
        pred_shape = combined_predictions.shape

        # Inverse transform predictions
        predictions_dict = {}
        for i, col in enumerate(target_cols):
            if len(pred_shape) >= 3 and i < pred_shape[2]:  # 3D array with sufficient depth
                predictions_dict[col] = self.scalers[col].inverse_transform(
                    combined_predictions[:, :, i].reshape(-1, 1)
                ).flatten()
            elif len(pred_shape) == 2 and i < pred_shape[1]:  # 2D array
                predictions_dict[col] = self.scalers[col].inverse_transform(
                    combined_predictions[:, i].reshape(-1, 1)
                ).flatten()

        # Create a DataFrame with daily predictions
        current_date = pd.to_datetime(current_data_processed['datetime'].iloc[-1])
        dates = [current_date + pd.Timedelta(days=i + 1) for i in range(days)]

        # Create the prediction DataFrame
        pred_df = pd.DataFrame({
            'date': dates,
        })

        # Add predictions for each target
        for col in target_cols:
            if col in predictions_dict:
                values = predictions_dict[col]
                # If we have enough values, average them by day
                if len(values) >= days:
                    day_values = []
                    values_per_day = len(values) // days
                    for i in range(days):
                        start_idx = i * values_per_day
                        end_idx = start_idx + values_per_day
                        day_values.append(np.mean(values[start_idx:end_idx]))
                    pred_df[col] = day_values
                else:
                    # Not enough values, just use what we have
                    pred_df[col] = values[:days]

        # Add additional processed columns
        if 'rain_analog' in pred_df.columns:
            # Convert rain_analog to precipitation probability
            pred_df['rain_probability'] = (100 - pred_df['rain_analog']) / 100
            pred_df['rain_probability'] = pred_df['rain_probability'].clip(0, 1)

        return pred_df

    def train(self, X_train, y_train, X_val, y_val, lookback=24, forecast_horizon=7, epochs=50, batch_size=32):
        """Train the weather forecasting model."""
        # Save lookback and forecast_horizon as instance variables
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon

        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, lookback, forecast_horizon)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val, lookback, forecast_horizon)

        # Define input and output shapes
        input_shape = (lookback, X_train.shape[1])
        output_shape = (forecast_horizon, y_train.shape[1])

        # Build the model
        self.model = self.build_model(input_shape, output_shape)

        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=f"{self.model_dir}/weather_forecast_model.h5",
                save_best_only=True
            )
        ]

        # Train the model
        history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Save the scalers and model parameters
        joblib.dump(self.scalers, f"{self.model_dir}/scalers.pkl")
        model_params = {
            'lookback': self.lookback,
            'forecast_horizon': self.forecast_horizon,
            'target_cols': self.target_cols
        }
        joblib.dump(model_params, f"{self.model_dir}/model_params.pkl")

        return history

    def load_saved_model(self):
        """Load a saved model and scalers."""
        model_path = f"{self.model_dir}/weather_forecast_model.h5"
        scalers_path = f"{self.model_dir}/scalers.pkl"
        params_path = f"{self.model_dir}/model_params.pkl"

        if os.path.exists(model_path) and os.path.exists(scalers_path):
            # Define custom objects to help with model loading
            custom_objects = {
                'MeanSquaredError': tf.keras.losses.MeanSquaredError
            }

            try:
                # Load with custom objects
                self.model = load_model(model_path, custom_objects=custom_objects)
                self.scalers = joblib.load(scalers_path)

                # Load model parameters if available
                if os.path.exists(params_path):
                    params = joblib.load(params_path)
                    self.lookback = params.get('lookback', self.lookback)
                    self.forecast_horizon = params.get('forecast_horizon', self.forecast_horizon)
                    self.target_cols = params.get('target_cols', self.target_cols)

                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            print("Saved model or scalers not found. Please train the model first.")
            return False