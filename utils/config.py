import os


class Config:
    """Configuration for the weather forecasting system."""

    def __init__(self):
        # Serial port settings
        self.SERIAL_PORT = os.environ.get('SERIAL_PORT', 'COM9')
        self.SERIAL_BAUDRATE = int(os.environ.get('SERIAL_BAUDRATE', 9600))

        # Application settings
        self.DEBUG_MODE = os.environ.get('DEBUG_MODE', 'False').lower() == 'true'
        self.USE_REAL_SENSOR = os.environ.get('USE_REAL_SENSOR', 'True').lower() == 'true'

        # Model settings
        self.MODEL_DIR = os.environ.get('MODEL_DIR', 'models/saved_models')
        self.LOOKBACK_WINDOW = int(os.environ.get('LOOKBACK_WINDOW', 24))
        self.FORECAST_HORIZON = int(os.environ.get('FORECAST_HORIZON', 7))