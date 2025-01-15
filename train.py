# conda create --name modelmaker python=3.9
# conda activate modelmaker
# pip install setuptools==72.1.0 Cython==3.0.11 numpy==1.24.3
# pip install -r requirements.txt
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import pandas as pd

# pylint: disable=no-name-in-module
from configs import models
from data.csv_loader import CSVLoader
from data.tiingo_data_fetcher import DataFetcher
from data.utils.data_preprocessing import preprocess_data
from models.model_factory import ModelFactory
from utils.common import print_colored

def split_date_ranges(start_date, end_date, frequency):
    """
    Splits the date range between start_date and end_date into subranges based on the specified frequency.

    Args:
        start_date (str): Start date in the format 'YYYY-MM-DD'.
        end_date (str): End date in the format 'YYYY-MM-DD'.
        frequency (str): Frequency to split the range. Options are '1min', '5min', '1hour', '4hour', '1day'.

    Returns:
        list of tuples: A list of (start_date, end_date) tuples.
    """

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    date_ranges = []
    current_date = start_date

    # Define the time delta based on frequency
    if frequency == '1min':
        delta = timedelta(days=1)
    elif frequency == '5min':
        delta = timedelta(days=15)
    elif frequency == '1hour':
        delta = timedelta(days=182)  # Approx. 6 months
    elif frequency in ['4hour', '1day']:
        delta = timedelta(days=365)  # Approx. 1 year
    else:
        raise ValueError("Unsupported frequency. Choose from '1min', '5min', '1hour', '4hour', '1day'.")

    while current_date < end_date:
        next_date = current_date + delta
        if next_date > end_date:
            next_date = end_date
        date_ranges.append((current_date.strftime("%Y-%m-%d"), next_date.strftime("%Y-%m-%d")))
        current_date = next_date

    return date_ranges

def select_data(fetcher, default_selection=None, file_path=None, dir_path=None):
    """Provide an interface to choose between Tiingo stock, Tiingo crypto, or CSV data."""

    default_end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    if default_selection is None:
        print("Select the data source:")
        print("1. Tiingo Stock Data")
        print("2. Tiingo Crypto Data")
        print("3. Load data from CSV file")
        print("4. Load data from a directory of CSV files")

        selection = input("Enter your choice (1/2/3/4): ").strip()
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
        stock_data = pd.DataFrame()
        date_ranges = split_date_ranges(start_date, end_date, frequency)
        for start, end in date_ranges:
            fetched_data = fetcher.fetch_tiingo_stock_data(symbol, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), frequency)
            if fetched_data is not None and not fetched_data.empty:
                stock_data = pd.concat([stock_data, fetched_data], ignore_index=True)

        return stock_data

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
        crypto_data = pd.DataFrame()
        date_ranges = split_date_ranges(start_date, end_date, frequency)
        for start, end in date_ranges:
            fetched_data = fetcher.fetch_tiingo_crypto_data(symbol, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), frequency)
            if fetched_data is not None and not fetched_data.empty:
                crypto_data = pd.concat([crypto_data, fetched_data], ignore_index=True)

        return crypto_data

    if selection == "3":
        print("You selected to load data from a CSV file.")
        if file_path is None:
            file_path = input("Enter the CSV file path: ").strip()
        return CSVLoader.load_csv(file_path)

    if selection == "4":
        print("You selected to load data from a directory of CSV files.")
        if dir_path is None:
            dir_path = input("Enter the directory path: ").strip()

        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            print("Invalid directory path.")
            sys.exit(1)

        combined_data = pd.DataFrame()

        for filename in os.listdir(dir_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(dir_path, filename)
                print(f"Loading data from {file_path}...")
                try:
                    data = CSVLoader.load_csv(file_path)
                    combined_data = pd.concat([combined_data, data], ignore_index=True)
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

        combined_data.drop_duplicates(inplace=True)

        print(f"Loaded and combined data from directory: {dir_path}")
        print(f"Total rows after removing duplicates: {len(combined_data)}")

        return combined_data

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
