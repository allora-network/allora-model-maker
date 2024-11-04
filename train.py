# conda create --name modelmaker python=3.9
# conda activate modelmaker
# pip install setuptools==72.1.0 Cython==3.0.11 numpy==1.24.3
# pip install -r requirements.txt
import os
import sys
from datetime import datetime, timedelta

# pylint: disable=no-name-in-module
from configs import models
from data.csv_loader import CSVLoader
from data.tiingo_data_fetcher import DataFetcher
from data.utils.data_preprocessing import preprocess_data
from models.model_factory import ModelFactory
from utils.common import print_colored


def select_data(fetcher, default_selection=None, file_path=None):
    """Provide an interface to choose between Tiingo stock, Tiingo crypto, or CSV data."""

    default_end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    if default_selection is None:
        print("Select the data source:")
        print("1. Tiingo Stock Data")
        print("2. Tiingo Crypto Data")
        print("3. Load data from CSV file")

        selection = input("Enter your choice (1/2/3): ").strip()
    else:
        selection = default_selection

    if selection == "1":
        print("You selected Tiingo Stock Data.")
        symbol = input("Enter the stock symbol (default: AAPL): ").strip() or "AAPL"
        frequency = (
            input(
                "Enter the frequency (daily/weekly/monthly/annually, default: daily): "
            ).strip()
            or "daily"
        )
        start_date = (
            input("Enter the start date (YYYY-MM-DD, default: 2021-01-01): ").strip()
            or "2021-01-01"
        )
        end_date = (
            input(
                f"Enter the end date (YYYY-MM-DD, default: {default_end_date}): "
            ).strip()
            or default_end_date
        )

        print(
            f"Fetching Tiingo Stock Data for {symbol} from {start_date} to {end_date} with {frequency} frequency..."
        )
        return fetcher.fetch_tiingo_stock_data(symbol, start_date, end_date, frequency)

    if selection == "2":
        print("You selected Tiingo Crypto Data.")
        symbol = (
            input("Enter the crypto symbol (default: btcusd): ").strip() or "btcusd"
        )
        frequency = (
            input("Enter the frequency (1min/5min/4hour/1day, default: 1day): ").strip()
            or "1day"
        )
        start_date = (
            input("Enter the start date (YYYY-MM-DD, default: 2021-01-01): ").strip()
            or "2021-01-01"
        )
        end_date = (
            input(
                f"Enter the end date (YYYY-MM-DD, default: {default_end_date}): "
            ).strip()
            or default_end_date
        )

        print(
            f"Fetching Tiingo Crypto Data for {symbol} from {start_date} to {end_date} with {frequency} frequency..."
        )
        return fetcher.fetch_tiingo_crypto_data(symbol, start_date, end_date, frequency)

    if selection == "3":
        print("You selected to load data from a CSV file.")

        # Add numbered directory listing functionality
        sets_dir = os.path.join(os.path.dirname(__file__), "data", "sets")
        if os.path.exists(sets_dir):
            print("\nAvailable files in data/sets:")
            csv_files = [f for f in os.listdir(sets_dir) if f.endswith('.csv')]

            if not csv_files:
                print("No CSV files found in data/sets directory.")
            else:
                for idx, file in enumerate(csv_files, 1):
                    print(f"{idx}. {file}")
                print()

                if file_path is None:
                    file_selection = input("Enter the number of the file or provide a custom path: ").strip()
                    try:
                        file_idx = int(file_selection)
                        if 1 <= file_idx <= len(csv_files):
                            file_path = os.path.join(sets_dir, csv_files[file_idx - 1])
                        else:
                            print_colored("Invalid file number. Using provided path as-is.", "warning")
                    except ValueError:
                        # If input is not a number, treat it as a custom path
                        file_path = file_selection

        if file_path is None:
            file_path = input("Enter the CSV file path: ").strip()
        return CSVLoader.load_csv(file_path)

    # Exit the program if the user enters an invalid choice
    print_colored("Invalid choice", "error")
    sys.exit(1)


def model_selection_input():
    print("Select the models to train:")
    print("1. All models")
    print("2. Custom selection")

    model_selection = input("Enter your choice (1/2): ").strip()

    if model_selection == "1":
        model_types = models
    elif model_selection == "2":
        available_models = {str(i + 1): model for i, model in enumerate(models)}
        print("Available models to train:")
        for key, value in available_models.items():
            print(f"{key}. {value}")

        selected_models = input(
            "Enter the numbers of the models to train (e.g., 1,3,5): "
        ).strip()
        model_types = [
            available_models[num.strip()]
            for num in selected_models.split(",")
            if num.strip() in available_models
        ]
    else:
        print_colored("Invalid choice, defaulting to all models.", "error")
        model_types = models

    return model_types


def main():
    fetcher = DataFetcher()

    # Select data dynamically based on user input
    data = select_data(fetcher)  # example testing defaults , "4", "data/sets/eth.csv"

    # Normalize and preprocess the data
    data = preprocess_data(data)

    # Initialize ModelFactory
    factory = ModelFactory()

    # Select models to train
    model_types = model_selection_input()

    # Train and save the selected models
    for model_type in model_types:
        print(f"Training {model_type} model...")
        model = factory.create_model(model_type)
        model.train(data)

    print_colored("Model training and saving complete!", "success")


if __name__ == "__main__":
    main()
