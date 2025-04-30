import os
import sys
import argparse
import pandas as pd
from datetime import datetime

from data_processing.sensor_reader import SensorReader
from data_processing.data_preprocessor import DataPreprocessor
from models.dataset_generator import WeatherDatasetGenerator
from models.weather_model import WeatherForecastModel
from utils.config import Config


def setup_directories():
    """Create necessary directories for the application."""
    directories = [
        'data',
        'models/saved_models',
        'logs'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory {directory} created or already exists.")


def generate_dataset(days=365, readings_per_day=10):
    """Generate synthetic dataset for training."""
    print(f"Generating synthetic dataset with {days} days of data...")

    # Create dataset generator
    generator = WeatherDatasetGenerator(
        start_date=datetime.now().strftime("%Y-%m-%d"),
        days_back=days
    )

    # Generate dataset
    dataset = generator.generate_dataset(readings_per_day=readings_per_day)

    # Save to CSV
    dataset.to_csv('data/synthetic_weather_data.csv', index=False)
    print(f"Dataset saved with {len(dataset)} entries.")

    return dataset


def train_model(dataset=None, epochs=50, batch_size=32):
    """Train the weather forecasting model."""
    print("Training weather forecasting model...")

    # Load dataset if not provided
    if dataset is None:
        if os.path.exists('data/synthetic_weather_data.csv'):
            dataset = pd.read_csv('data/synthetic_weather_data.csv')
            # Convert datetime column to datetime type if it exists
            if 'datetime' in dataset.columns:
                dataset['datetime'] = pd.to_datetime(dataset['datetime'])
        else:
            print("No dataset found. Generating synthetic data...")
            dataset = generate_dataset()

    # Initialize model
    model = WeatherForecastModel()

    # Preprocess data
    X_train, X_test, y_train, y_test, _ = model.preprocess_data(dataset)

    # Train the model
    history = model.train(
        X_train, y_train,
        X_test, y_test,
        epochs=epochs,
        batch_size=batch_size
    )

    print("Model training completed.")
    return model


def run_server():
    """Run the Flask server."""
    print("Starting Flask server...")
    # Import here to avoid circular imports
    from server.app import app

    # Run the app
    app.run(host='0.0.0.0', port=5000)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Weather Forecasting System")
    parser.add_argument('--generate-data', action='store_true', help='Generate synthetic dataset')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--serve', action='store_true', help='Start the Flask server')
    parser.add_argument('--days', type=int, default=365, help='Number of days for synthetic data')
    parser.add_argument('--readings-per-day', type=int, default=10, help='Readings per day for synthetic data')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')

    args = parser.parse_args()

    # Setup directory structure
    setup_directories()

    if args.generate_data:
        dataset = generate_dataset(days=args.days, readings_per_day=args.readings_per_day)

        if args.train:
            train_model(dataset, epochs=args.epochs, batch_size=args.batch_size)
    elif args.train:
        train_model(epochs=args.epochs, batch_size=args.batch_size)

    if args.serve:
        run_server()

    # If no arguments, show help
    if not (args.generate_data or args.train or args.serve):
        parser.print_help()


if __name__ == "__main__":
    main()