# pylint: disable=R0801
#  Description: Configuration class for Random Forest Regressor Volatility model.
class RfregressorConfig:
    """
    Configuration class for the Random Forest Regressor Volatility model.
    Stores hyperparameters for the model and settings for data preprocessing.
    """

    def __init__(self):
        # Random Forest hyperparameters
        self.params = {
            "objective": "reg:squarederror",  # Regression objective
            "eval_metric": "rmse"
        }

        # Feature calculation parameters
        self.candle_interval = '1min'        # Base timeframe for data
        self.volatility_window = '5min'    # Window for volatility calculation
        self.n_lags = 5                      # Number of lags for features
        self.ma_window = 10                  # Window for moving averages
        self.target_shift = 5                # Prediction horizon

        # Data preprocessing
        self.fillna_method = 'ffill'         # Method for handling missing values
        self.technical_indicators = [         # List of technical indicators to use
            'SMA_10',
            'EMA_10'
        ]

        #Model parameters
        self.n_estimators = 10
        self.random_state = 42
        self.n_jobs = 25

    def display(self):
        """Prints out the current configuration."""
        print("Random Forest Regressor Volatility Configuration:")
        print(f"  params: {self.params}")
        print(f"  candle_interval: {self.candle_interval}")

        print(f"  volatility_window: {self.volatility_window}")
        print(f"  n_lags: {self.n_lags}")
        print(f"  ma_window: {self.ma_window}")
        print(f"  target_shift: {self.target_shift}")
        print(f"  fillna_method: {self.fillna_method}")
        print(f"  technical_indicators: {self.technical_indicators}")
