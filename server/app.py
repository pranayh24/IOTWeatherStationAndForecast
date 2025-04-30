from flask import Flask, jsonify, request
import pandas as pd
import json
import os
import sys
from datetime import datetime, timedelta
from flask_cors import CORS



# Add root directory to path to be able to import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.sensor_reader import SensorReader
from data_processing.data_preprocessor import DataPreprocessor
from models.weather_model import WeatherForecastModel
from utils.config import Config

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize components
config = Config()
sensor_reader = SensorReader(port=config.SERIAL_PORT)
data_preprocessor = DataPreprocessor()
weather_model = WeatherForecastModel()

# In-memory storage for recent readings
recent_readings = []
MAX_RECENT_READINGS = 100  # Store up to 100 recent readings

# Sample data for simulation mode
sample_data = []


def load_sample_data():
    """Load sample data from file or use default data."""
    global sample_data

    try:
        if os.path.exists('data/sample_readings.json'):
            with open('data/sample_readings.json', 'r') as f:
                sample_data = json.load(f)
        else:
            # Use the data provided in the user's request
            sample_data = [
                {"timestamp": 50, "temp_dht": 35.2, "humidity": 46, "temp_bmp": 36.2, "pressure": 981.8,
                 "rain_analog": 58, "rain_detected": True, "light_intensity": 213, "nh3": 2.7,
                 "co": 3.7, "co2": 0.5, "alcohol": 1.3, "lpg": 244.1, "ch4": 244.1, "mq9_co": 141, "aqi": 43},
                # Add more sample data here...
            ]
    except Exception as e:
        print(f"Error loading sample data: {e}")
        # Fallback to empty list
        sample_data = []


def get_current_reading(use_real_sensor=True):
    """Get current sensor reading (real or simulated)."""
    if use_real_sensor:
        return sensor_reader.read_data()
    else:
        # Simulation mode
        if not sample_data:
            load_sample_data()
        return sensor_reader.simulate_reading(sample_data)


@app.route('/api/current', methods=['GET'])
def get_current_weather():
    """API endpoint to get current weather data."""
    try:
        # Get current reading (from hardware or simulation)
        use_real_sensor = config.USE_REAL_SENSOR
        raw_data = get_current_reading(use_real_sensor)

        if not raw_data:
            return jsonify({"error": "Failed to read sensor data"}), 500

        # Process the data
        processed_data = data_preprocessor.process_raw_data(raw_data)

        # Store in recent readings
        recent_readings.append(processed_data)
        if len(recent_readings) > MAX_RECENT_READINGS:
            recent_readings.pop(0)  # Remove oldest reading

        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "data": processed_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """API endpoint to get weather forecast."""
    try:
        # Get forecast days from query parameter (default to 7)
        days = int(request.args.get('days', 7))

        # Get current reading
        use_real_sensor = config.USE_REAL_SENSOR
        current_data = get_current_reading(use_real_sensor)

        if not current_data:
            return jsonify({"error": "Failed to read current sensor data"}), 500

        # Process current data
        processed_data = data_preprocessor.process_raw_data(current_data)

        # Make prediction using recent readings as lookback data
        predictions = weather_model.predict(
            processed_data,
            lookback_data=pd.DataFrame(recent_readings) if recent_readings else None,
            days=days
        )

        if predictions is None:
            return jsonify({"error": "Failed to generate forecast"}), 500

        # Convert predictions to JSON-serializable format
        forecast_data = predictions.to_dict(orient='records')

        # Convert dates to ISO format strings
        for day in forecast_data:
            day['date'] = day['date'].isoformat().split('T')[0]  # Get just the date part

            # Round numeric values
            for key, value in day.items():
                if isinstance(value, float):
                    day[key] = round(value, 2)

        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "forecast": forecast_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """API endpoint to get recent sensor readings history."""
    try:
        # Get limit from query parameter (default to 10)
        limit = int(request.args.get('limit', 10))
        limit = min(limit, len(recent_readings))  # Ensure limit doesn't exceed available data

        # Return the most recent readings
        history_data = recent_readings[-limit:]

        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "count": len(history_data),
            "history": history_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint for health check."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    })


if __name__ == '__main__':
    # Load sample data for simulation mode
    if not config.USE_REAL_SENSOR:
        load_sample_data()

    # Make sure the model is loaded
    weather_model.load_saved_model()

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=config.DEBUG_MODE)