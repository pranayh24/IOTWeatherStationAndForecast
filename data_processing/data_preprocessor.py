import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataPreprocessor:
    def __init__(self):
        """Initialize the data preprocessor."""
        pass

    def process_raw_data(self, raw_data):
        """Process raw sensor data into a format suitable for the model.

        Args:
            raw_data: Dictionary of sensor readings

        Returns:
            dict: Processed sensor data with derived features
        """
        processed_data = raw_data.copy()

        # Calculate average temperature from both sensors
        if 'temp_dht' in raw_data and 'temp_bmp' in raw_data:
            processed_data['temperature'] = (raw_data['temp_dht'] + raw_data['temp_bmp']) / 2

        # Derive hour of day (useful for time-based patterns)
        if 'datetime' in processed_data:
            dt = datetime.fromisoformat(processed_data['datetime'])
            processed_data['hour'] = dt.hour
            processed_data['day_of_week'] = dt.weekday()
            processed_data['month'] = dt.month

        # Calculate rain intensity based on analog value and boolean detection
        if 'rain_analog' in processed_data and 'rain_detected' in processed_data:
            if processed_data['rain_detected']:
                # Map the analog value to a rain intensity scale (0-10)
                # Assuming rain_analog ranges from approximately 0-100
                # Lower value means more rain (sensor behavior)
                rain_intensity = max(0, 10 - (processed_data['rain_analog'] / 10))
            else:
                rain_intensity = 0
            processed_data['rain_intensity'] = rain_intensity

        # Create composite air quality feature
        if all(gas in processed_data for gas in ['nh3', 'co', 'co2']):
            # Simple weighted sum of gas concentrations
            # This is a simplified approach; actual AQI calculation is more complex
            processed_data['composite_air_quality'] = (
                    0.3 * processed_data['nh3'] +
                    0.5 * processed_data['co'] +
                    0.2 * processed_data['co2']
            )

        return processed_data

    def prepare_for_model(self, processed_data, lookback_window=24):
        """Prepare data for the ML model.

        Args:
            processed_data: DataFrame of processed sensor readings
            lookback_window: Number of past observations to include

        Returns:
            tuple: X (features) and y (targets) for the model
        """
        # Feature selection - variables we want to use for prediction
        feature_cols = [
            'temperature', 'humidity', 'pressure',
            'rain_intensity', 'light_intensity',
            'aqi', 'nh3', 'co', 'co2',
            'hour', 'day_of_week', 'month'
        ]

        # Target variables - what we want to predict
        target_cols = ['temperature', 'rain_intensity', 'aqi']

        # Create sequences for time series forecasting
        X, y = [], []

        for i in range(len(processed_data) - lookback_window):
            X.append(processed_data[feature_cols].iloc[i:i + lookback_window].values)

            # For targets, we use multiple future values (next 7 days)
            # In a real implementation, you'd need proper aggregation for daily forecasts
            future_window = min(7, len(processed_data) - i - lookback_window)
            y.append(processed_data[target_cols].iloc[i + lookback_window:i + lookback_window + future_window].values)

        return np.array(X), np.array(y)