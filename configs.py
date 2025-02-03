# Description: Configuration file for the project
#

# Define the models to be used
models = [
    "regression",
    "regression_time_series",
    "random_forest",
    "random_forest_time_series",
    "xgboost",
    "xgboost_time_series",
    "prophet",
    "arima",
    "lstm",
    "rfregressor_volatility"
]

# Define the metrics to be used
metrics = [
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "maximum_drawdown",
    "mdd_duration",
    "cagr",
    "volatility",
    "value_at_risk",
    "expected_shortfall",
]
