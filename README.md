# Allora Model Maker

<div style="text-align: center;">
<img src="https://cdn.prod.website-files.com/667c44f051907593fdb7e7fe/667c789fa233d4f02c1d8cfa_allora-webclip.png" alt="Allora Logo" width="200"/>
</div>

Allora Model Maker is a comprehensive machine learning framework designed for time series forecasting, specifically optimized for financial market data like cryptocurrency prices, stock prices, and more. It supports multiple models, including traditional statistical models like ARIMA and machine learning models like LSTM, XGBoost, and more.

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
   - [3.1 Model Training](#model-training)
   - [3.2 Model & Metric Testing](#model-testing)
   - [3.3 Model Inference](#model-inference)
   - [3.4 Model Forecasting](#model-forecasting)
   - [3.5 Metrics Calculation](#metrics-calculation)
4. [Supported Models](#supported-models)
5. [Configuration](#configuration)
   - [5.1 Model Configurations](#model-configurations)
   - [5.2 Interval Configuration](#interval-configuration)
6. [Directory Structure](#directory-structure)
7. [Makefile](#makefile)
   - [7.1 Lint](#lint)
   - [7.2 Format](#format)
   - [7.3 Test](#test)
   - [7.4 Clean](#clean)
   - [7.5 Run Training](#run-training)
   - [7.6 Run Inference Tests](#run-inference-tests)
   - [7.7 Format Again](#format-again)
   - [7.8 Package](#package)
8. [Packaging Models for Allora Worker](#packaging-models-for-allora-worker)
   - [8.1 How to Use](#how-to-use)
   - [8.2 Next Steps](#next-steps)
9. [Data](#data)
   - [9.1 Data Provider](#data-provider)
   - [9.2 How We Use Tiingo](#how-we-use-tiingo)
   - [9.3 Setting Up Tiingo](#setting-up-tiingo)
10. [Contributing](#contributing)
11. [General Best Practices](#general-best-practices)
12. [License](#license)
## Features
- Multiple model support (ARIMA, LSTM, XGBoost, Random Forest, etc.)
- Configurable time intervals (`5M`, `H`, `D`, etc.) for time series data
- Built-in metrics for evaluating model performance (e.g., CAGR, Sortino Ratio, Expected Shortfall)
- Easy model saving and loading mechanism
- Scalable for large datasets

## Installation

1. Clone the repository:
    ```bash
   git clone https://github.com/allora-network/allora-model-maker.git
   cd allora-model-maker
    ```

   #### Don't have conda?
      On Mac simply use brew
      ```bash
      brew install miniconda
      ```
      On Windows go to the official [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html)

2. Create a conda environment:
    ```bash
    conda env create -f environment.yml
    ```
    If you want to manually do it:
    ```bash
    conda create --name modelmaker python=3.9 && conda activate modelmaker
    ```
   Preinstall setuptools, cython and numpy
   ```bash
   pip install setuptools==72.1.0 Cython==3.0.11 numpy==1.24.3
   ```
   Install dependencies:
    ```bash
   pip install -r requirements.txt
    ```

## Usage

### Model Training

You can train models by running the `train.py` script. It supports multiple model types and interval resampling.
We provided an eth.csv dataset that you can use for training, select option 3 and use data/sets/eth.csv otherwise setup [Tiingo](#data)!

Example for training:
 ```bash
make train
 ```

### Model Testing

You can test models by running the `test.py` script. It supports multiple model types and metrics.

Example for testing:
 ```bash
make eval
 ```

During runtime, you will be prompted to select if you want to test models, metrics or both. The testing data is currently synthetic.

### Model Inference

To make predictions using a trained model, you can use the `inference()` method on the desired model.

Example:
 ```python
from models.lstm.model import LstmModel
model = LstmModel()
predictions = model.inference(input_data)
print(predictions)
 ```

### Model Forecasting

Forecast future data based on past time series data using the `forecast()` method:

 ```python
forecast_data = model.forecast(steps=10, last_known_data=input_data)
print(forecast_data)
 ```

### Metrics Calculation

Metrics can be calculated using the provided `metrics` module:

 ```python
from metrics.sortino_ratio.metric import SortinoRatioMetric
metric = SortinoRatioMetric()
result = metric.calculate(input_data)
print(result)
 ```

## Supported Models

The following models are supported out-of-the-box:
- **ARIMA**: Auto-Regressive Integrated Moving Average
- **LSTM**: Long Short-Term Memory Networks
- **Random Forest**: Random Forest for time series and regression
- **XGBoost**: Gradient Boosting for time series and regression
- **Prophet**: Facebook's Prophet for time series forecasting
- **Regression**: Basic regression models

## Configuration

### Model Configurations

Each model has its own configuration class located in its corresponding folder. For example, `LstmConfig` can be found in `models/lstm/configs.py`. Configurations include parameters for training, architecture, and data preprocessing.

You can modify configurations as needed:
 ```python
config = LstmConfig()
config.learning_rate = 0.001
 ```

### Interval Configuration

By default, the system uses daily (`D`) intervals for time series resampling. You can modify this in the configuration files for each model by setting the `interval` parameter (e.g., `5M`, `H`, `D`, etc.).

## Directory Structure

 ```
allora-model-maker/
│
├── models/                   # Directory containing different models
│   ├── lstm/                  # LSTM model files
│   ├── arima/                 # ARIMA model files
│   ├── random_forest/         # Random Forest model files
│   └── ...                    # Other model directories
│
├── metrics/                   # Metrics for evaluating model performance
│   ├── cagr/                  # CAGR metric implementation
│   ├── sortino_ratio/         # Sortino Ratio metric implementation
│   └── ...                    # Other metrics
│
├── data/                      # Data loading and preprocessing utilities
│
├── configs.py                 # Global configuration files
├── Makefile                   # Makefile for streamlined operations
├── package_model_worker.py    # Script for packaging a model for allora worker
├── requirements.txt           # Python dependencies
├── train.py                   # Script for training models
├── test.py                    # Script for testing models and metrics
 ```

## Makefile

This project includes a Makefile to simplify common tasks such as linting, formatting, testing, and running scripts. Below are the available commands:

#### Lint

 ```
make lint
 ```

This command runs pylint on all Python files in the project to check for coding errors, stylistic errors, and other issues. It will scan through all .py files.

#### Format

 ```
make format
 ```

This command formats all Python files using black, a widely used code formatter. It automatically reformats code to follow the best practices and standards.

#### Test

 ```
make test
 ```

This command runs the unit tests using pytest. By default, it will search for tests under the tests/ directory.

#### Clean

 ```
make clean
 ```

This command removes common build artifacts and directories, such as Python caches, test logs, and generated model files. Specifically, it will remove:

	•	__pycache__
	•	.pytest_cache
	•	.coverage
	•	trained_models/
	•	packaged_models/
	•	logs/
	•	test_results/

#### Run Training

 ```
make train
 ```

This command executes the training script train.py and starts the model training process.

#### Run Inference Tests

 ```
make eval
 ```

This command runs the script test.py, allowing you to test the model prediction and validation.

#### Format

 ```
make format
 ```

This command will format the entire codebase using black. Use this before committing code to ensure consistency and readability.

#### Package

 ```
make package-[model name]
 ```

This command will package your model for use in an allora worker, remember to replace [model name] with your actual model, ex: "arima"


## Packaging Models for Allora Worker

The purpose of the package_model_worker.py script is to export a trained model along with its configurations in a format that can be deployed into the allora-worker repository.

This script packages the model files, dependencies, and configuration into a structure that allows seamless integration with the allora-worker setup.

#### How to Use

Run the following command to package your model for the Allora worker:

``` make package-arima ```

Replace arima with the name of the model you’d like to package (e.g., lstm, arima, etc.).

This will:

 - Copy the model’s files and dependencies into the packaged_models/package folder.
 - Run test's for inference and training to validate functionality in a worker
 - Generate a configuration file, config.py, that contains the active model information.

#### Next Steps

After running the packaging command:

1.	Navigate to the packaged_models folder in your allora-model-maker repo.
2.	Copy the **package** folder into the src folder of your allora-worker repository.
3. If you did this right in your allora-worker repo you'll now have **allora-worker/src/package**

By following these steps, your packaged model will be ready for deployment in the allora-worker environment.


## Data
### Data Provider

<img src="https://www.tiingo.com/dist/images/tiingo/logos/tiingo_full_light_color.svg" alt="Tiingo Logo" width="200"/>

We are proud to incorporate **Tiingo** as the primary data provider for our framework. Tiingo is a powerful financial data platform that offers a wide range of market data, including:

- **Stock Prices** (historical and real-time)
- **Crypto Prices** (historical and real-time)
- **Fundamental Data**
- **News Feeds**
- **Alternative Data Sources**

By integrating Tiingo, our framework ensures that you have access to high-quality, reliable data for various financial instruments, empowering you to make informed decisions based on up-to-date market information.

### How We Use Tiingo

Our framework uses the Tiingo API to fetch and process data seamlessly within the system. This integration provides efficient and real-time data access to enable advanced analytics, backtesting, and more. Whether you're developing trading strategies, conducting financial analysis, or creating investment models, Tiingo powers the data behind our features.

To use Tiingo data with our framework, you'll need to obtain a Tiingo API key. You can sign up for an API key by visiting [Tiingo's website](https://www.tiingo.com) and following their documentation for API access.

### Setting Up Tiingo

To configure Tiingo within the framework:

1. **Get your API key** from Tiingo:
   Visit [Tiingo's API](https://api.tiingo.com/) to sign up and retrieve your API key.

2. **Set the API key** in your environment:
   Add the following environment variable to your `.env` file or pass it directly in your configuration:
   ```bash
   TIINGO_API_KEY=your_api_key_here
   ```

3. **Start using Tiingo data** in your projects:
   Our framework will automatically fetch data from Tiingo using your API key, ensuring that you have the most accurate and up-to-date market data for your application.

For more detailed information on how to use Tiingo's services, please refer to their [official API documentation](https://api.tiingo.com/documentation).


## Contributing

Contributions are welcome! To ensure a smooth contribution process, please follow these steps:

1. **Fork the repository.**
2. **Create a new branch:**
    ```git checkout -b feature-branch ```
3. **Make your changes.**
4. **Before committing your changes, run the following Makefile commands to ensure code quality and consistency:**

   - **Lint your code:**
      ```make lint ```

   - **Format your code:**
      ```make format ```

   - **Run tests to ensure everything works:**
      ```make test ```

5. **Commit your changes:**
    ```git commit -am 'Add new feature' ```

6. **Push to your branch:**
    ```git push origin feature-branch ```

7. **Create a pull request.**

### General Best Practices

- Ensure your code follows the project's coding standards by using `pylint` for linting and `black` for formatting.
- Test your changes thoroughly before pushing by running the unit tests.
- Use meaningful commit messages that clearly describe your changes.
- Make sure your branch is up-to-date with the latest changes from the main branch.
- Avoid including unnecessary files in your pull request, such as compiled or cache files. The `make clean` command can help with that:

   ```make clean ```

By following these practices, you help maintain the quality and consistency of the project.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
