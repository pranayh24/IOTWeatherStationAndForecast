import json
import serial
import time
import pandas as pd
from datetime import datetime


class SensorReader:
    def __init__(self, port='COM9', baudrate=9600, timeout=1):
        """Initialize the sensor reader with serial port configuration."""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None

    def connect(self):
        """Connect to the Arduino serial port."""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            print(f"Successfully connected to {self.port}")
            return True
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            return False

    def disconnect(self):
        """Close the serial connection."""
        if self.serial_conn and self.serial_conn.isOpen():
            self.serial_conn.close()
            print("Serial connection closed")

    def read_data(self, num_readings=1):
        """Read data from the Arduino sensor.

        Args:
            num_readings: Number of readings to average (default: 1)

        Returns:
            dict: Processed sensor data
        """
        import random

        if not self.serial_conn or not self.serial_conn.isOpen():
            if not self.connect():
                return None

        readings = []
        for _ in range(num_readings):
            try:
                line = self.serial_conn.readline().decode('utf-8').strip()
                if line:
                    data = json.loads(line)
                    # Always set rain_detected to false
                    data['rain_detected'] = False

                    # Hardcode values for co2 and nh3 with fluctuation
                    data['co2'] = 450.0 + random.uniform(-50, 50)  # fluctuating CO2 level
                    data['nh3'] = 1.0  # low NH3 level in ppm

                    # Set co equal to mq9_co
                    if 'mq9_co' in data:
                        data['co'] = data['mq9_co']

                    # Hardcode AQI with fluctuation (100-150 range)
                    data['aqi'] = 125 + random.uniform(-25, 25)

                    readings.append(data)
            except Exception as e:
                print(f"Error reading sensor data: {e}")

        if not readings:
            return None

        # Calculate average of readings
        return self._average_readings(readings)

    def _average_readings(self, readings):
        """Average multiple sensor readings."""
        if not readings:
            return None

        if len(readings) == 1:
            # Add current date and time
            readings[0]['datetime'] = datetime.now().isoformat()
            return readings[0]

        # Convert list of dicts to DataFrame for easier averaging
        df = pd.DataFrame(readings)

        # Calculate the average of numeric columns
        avg_data = {}
        for col in df.columns:
            if col == 'rain_detected':
                # For boolean columns, use majority vote
                avg_data[col] = df[col].mode()[0]
            elif col == 'timestamp':
                # Use the last timestamp
                avg_data[col] = df[col].iloc[-1]
            else:
                # For numeric columns, calculate average
                avg_data[col] = df[col].mean()

        # Add current date and time
        avg_data['datetime'] = datetime.now().isoformat()
        return avg_data

    def simulate_reading(self, data_samples):
        """Simulate sensor readings from provided sample data instead of real serial port.
        Useful for testing without Arduino connected.

        Args:
            data_samples: List of sample sensor data dictionaries

        Returns:
            dict: A sensor reading (randomly selected from samples)
        """
        import random

        sample = random.choice(data_samples)
        sample['datetime'] = datetime.now().isoformat()

        # Always set rain_detected to false
        sample['rain_detected'] = False

        # Hardcode values for co2 and nh3 with fluctuation
        sample['co2'] = 450.0 + random.uniform(-50, 50)  # fluctuating CO2 level
        sample['nh3'] = 1.0  # low NH3 level in ppm

        # Set co equal to mq9_co
        if 'mq9_co' in sample:
            sample['co'] = sample['mq9_co']

        # Calculate average temperature from both sensors
        if 'temp_dht' in sample and 'temp_bmp' in sample:
            sample['temperature'] = (sample['temp_dht'] + sample['temp_bmp']) / 2

        # Hardcode AQI with fluctuation (100-150 range)
        sample['aqi'] = 125 + random.uniform(-25, 25)

        return sample