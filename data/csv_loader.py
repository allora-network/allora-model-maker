import pandas as pd


class CSVLoader:
    """
    A class to load and preprocess CSV files.
    Provides functionality to load a CSV file, and ensure it's preprocessed.
    """

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """Load and preprocess CSV data."""
        try:
            data = pd.read_csv(file_path, parse_dates=["date"])
            # Ensure 'date' is a datetime object and sorted
            data = data.sort_values(by="date")
            print(f"Data loaded successfully from {file_path}")
            return data
        except (
            FileNotFoundError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as e:
            print(f"Error loading CSV: {e}")
            return None  # type: ignore
