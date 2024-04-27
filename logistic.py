import pandas as pd
import datetime
import MetaTrader5 as mt5
from datetime import datetime
import pytz
import pandas_ta as ta
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time
import shutil
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

def fetch_fx_data_mt5(symbol, timeframe_str, start_date, end_date):

    # Define your MetaTrader 5 account number
    account_number = 530064788
    # Define your MetaTrader 5 password
    password = 'fe5@6YV*'
    # Define the server name associated with your MT5 account
    server_name ='FTMO-Server3'

    # Initialize MT5 connection; if it fails, print error message and exit
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
    
    # Attempt to log in with the given account number, password, and server
    authorized = mt5.login(account_number, password=password, server=server_name)
    # If login fails, print error message, shut down MT5 connection, and exit
    if not authorized:
        print("login failed, error code =", mt5.last_error())
        mt5.shutdown()
        quit()
    # On successful login, print a confirmation message
    else:
        print("Connected to MetaTrader 5")

    # Set the timezone to Berlin, as MT5 times are in UTC
    timezone = pytz.timezone("Europe/Berlin")

    # Convert start and end dates to datetime objects, considering the timezone
    start_date = start_date.replace(tzinfo=timezone)
    end_date = end_date.replace(hour=23, minute=59, second=59, tzinfo=timezone)

    # Define a mapping from string representations of timeframes to MT5's timeframe constants
    timeframes = {
        '1H': mt5.TIMEFRAME_H1,
        'DAILY': mt5.TIMEFRAME_D1,
        '12H': mt5.TIMEFRAME_H12,
        '2H': mt5.TIMEFRAME_H2,
        '3H': mt5.TIMEFRAME_H3,
        '4H': mt5.TIMEFRAME_H4,
        '6H': mt5.TIMEFRAME_H6,
        '8H': mt5.TIMEFRAME_H8,
        '1M': mt5.TIMEFRAME_M1,
        '10M': mt5.TIMEFRAME_M10,
        '12M': mt5.TIMEFRAME_M12,
        '15M': mt5.TIMEFRAME_M15,
        '2M': mt5.TIMEFRAME_M2,
        '20M': mt5.TIMEFRAME_M20,
        '3M': mt5.TIMEFRAME_M3,
        '30M': mt5.TIMEFRAME_M30,
        '4M': mt5.TIMEFRAME_M4,
        '5M': mt5.TIMEFRAME_M5,
        '6M': mt5.TIMEFRAME_M6,
        '1MN': mt5.TIMEFRAME_MN1,
        '1W': mt5.TIMEFRAME_W1
    }

    # Retrieve the MT5 constant for the requested timeframe
    timeframe = timeframes.get(timeframe_str)
    # If the requested timeframe is invalid, print error message, shut down MT5, and return None
    if timeframe is None:
        print(f"Invalid timeframe: {timeframe_str}")
        mt5.shutdown()
        return None

    # Fetch the rates for the given symbol and timeframe within the start and end dates
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

    # If no rates were fetched, print error message, shut down MT5, and return None
    if rates is None:
        print("No rates retrieved, error code =", mt5.last_error())
        mt5.shutdown()
        return None
    
    # Convert the fetched rates into a Pandas DataFrame
    rates_frame = pd.DataFrame(rates)
    # Convert the 'time' column from UNIX timestamps to human-readable dates
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')

    # Set the 'time' column as the DataFrame index and ensure its format is proper for datetime
    rates_frame.set_index('time', inplace=True)
    rates_frame.index = pd.to_datetime(rates_frame.index, format="%Y-%m-%d")

    # Check if 'tick_volume' column is present in the fetched data
    if 'tick_volume' not in rates_frame.columns:
        print("tick_volume is not in the fetched data. Ensure it's included in the API call.")
    else:
        print("tick_volume is included in the data.")
    
    # Shut down the MT5 connection before returning the data
    mt5.shutdown()
    
    # Return the prepared DataFrame containing the rates
    return rates_frame

class IndicatorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sma_short_length=50, sma_long_length=200, ema_medium_length=50, ema_long_length=200,
                 ema_short_length=9, ema_fast_length=21, original_bollinger_length=20,
                 original_bollinger_std=2, bollinger_length=12, bollinger_std_dev=1.5, sma_trend_length=50, window=9):
        self.sma_short_length = sma_short_length
        self.sma_long_length = sma_long_length
        self.ema_medium_length = ema_medium_length
        self.ema_long_length = ema_long_length
        self.ema_short_length = ema_short_length
        self.ema_fast_length = ema_fast_length
        self.original_bollinger_length = original_bollinger_length
        self.original_bollinger_std = original_bollinger_std
        self.bollinger_length = bollinger_length
        self.bollinger_std_dev = bollinger_std_dev
        self.sma_trend_length = sma_trend_length
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Implementing SMA, EMA, and Bollinger Bands
        X[f'SMA_{self.sma_short_length}'] = ta.sma(X['close'], length=self.sma_short_length)
        X[f'SMA_{self.sma_long_length}'] = ta.sma(X['close'], length=self.sma_long_length)
        X[f'EMA_{self.ema_medium_length}'] = ta.ema(X['close'], length=self.ema_medium_length)
        X[f'EMA_{self.ema_long_length}'] = ta.ema(X['close'], length=self.ema_long_length)
        X[f'EMA_{self.ema_short_length}'] = ta.ema(X['close'], length=self.ema_short_length)
        X[f'EMA_{self.ema_fast_length}'] = ta.ema(X['close'], length=self.ema_fast_length)
        original_bollinger = ta.bbands(X['close'], length=self.original_bollinger_length, std=self.original_bollinger_std)
        X['Upper Band'] = original_bollinger['BBU_20_2.0']
        X['Lower Band'] = original_bollinger['BBL_20_2.0']
        updated_bollinger = ta.bbands(X['close'], length=self.bollinger_length, std=self.bollinger_std_dev)
        X['Lower Band Scalping'] = updated_bollinger['BBL_12_1.5']
        X['Middle Band Scalping'] = updated_bollinger['BBM_12_1.5']
        X['Upper Band Scalping'] = updated_bollinger['BBU_12_1.5']
        macd = ta.macd(X['close'])
        X['MACD'] = macd['MACD_12_26_9']
        X['Signal_Line'] = macd['MACDs_12_26_9']
        X[f'RSI_{self.window}'] = ta.rsi(X['close'], length=self.window).round(2)
        X.dropna(inplace=True)
        return X

def get_data():
    # Retrieve and store the current date
    current_date = str(datetime.now().date())
    current_date_datetime = datetime.now().date()

    # Calculate the date for one month before the current date
    one_month_before = current_date_datetime - relativedelta(months=1)
    # Convert to string if needed
    one_month_before_str = str(one_month_before)

    # Hardcoded start date for strategy evaluation
    strategy_start_date_all = "1971-01-04"
    # Use the current date as the end date for strategy evaluation
    strategy_end_date_all = current_date

    # Convert string representation of dates to datetime objects for further processing
    start_date_all = datetime.strptime(strategy_start_date_all, "%Y-%m-%d")
    end_date_all = datetime.strptime(strategy_end_date_all, "%Y-%m-%d")

    # Prompt the user for the desired timeframe for analysis and standardize the input
    timeframe_str = input("Enter the currency pair (e.g., Daily, 1H): ").strip().upper()
    # Prompt the user for the currency pair they're interested in and standardize the input
    Pair = input("Enter the currency pair (e.g., GBPUSD, EURUSD): ").strip().upper()

    training_start_date = "2023-01-01"
    training_end_date = current_date

    # Fetch and prepare the FX data for the specified currency pair and timeframe
    eur_usd_data = fetch_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)

    calculate_target(eur_usd_data)

    # Filter the EUR/USD data for the in-sample training period
    dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]

    dataset = dataset.fillna(0)

    return dataset, timeframe_str, Pair

def get_data_multiple(Pair, timeframe_str):
    # Retrieve and store the current date
    current_date = str(datetime.now().date())
    current_date_datetime = datetime.now().date()

    # Calculate the date for one month before the current date
    one_month_before = current_date_datetime - relativedelta(months=1)
    # Convert to string if needed
    one_month_before_str = str(one_month_before)

    # Hardcoded start date for strategy evaluation
    strategy_start_date_all = "1971-01-04"
    # Use the current date as the end date for strategy evaluation
    strategy_end_date_all = current_date

    # Convert string representation of dates to datetime objects for further processing
    start_date_all = datetime.strptime(strategy_start_date_all, "%Y-%m-%d")
    end_date_all = datetime.strptime(strategy_end_date_all, "%Y-%m-%d")

    training_start_date = "2023-01-01"
    training_end_date = current_date

    # Fetch and prepare the FX data for the specified currency pair and timeframe
    eur_usd_data = fetch_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)

    eur_usd_data = calculate_target(eur_usd_data) 

    # Filter the EUR/USD data for the in-sample training period
    dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]

    dataset = dataset.fillna(0)

    return dataset

def split_and_save_dataset(dataset, timeframe, pair):
    """
    Splits the dataset into training and validation sets with an 80/20 split,
    cleans up old related CSV files, and saves the new splits to CSV files.

    Args:
    dataset (pd.DataFrame): The full dataset to split.
    timeframe (str): Description of the timeframe, used in file naming.
    pair (str): Currency pair or dataset identifier, used in file naming.

    Returns:
    tuple: A tuple containing two DataFrames (training_set, validation_set).
    """
    if len(dataset) < 10:
        raise ValueError("Dataset is too small to split effectively.")

    # Calculate the split index for an 80/20 split
    split_index = int(len(dataset) * 0.8)
    
    # Split the dataset into training and validation sets
    training_set = dataset.iloc[:split_index]
    validation_set = dataset.iloc[split_index:]

    # Clean up existing CSV files related to previous runs
    file_patterns = [f'Full_data_{pair}_{timeframe}.csv', 
                     f'training_{pair}_{timeframe}_data.csv', 
                     f'validation_{pair}_{timeframe}_data.csv']
    for pattern in file_patterns:
        for file in glob.glob(pattern):
            os.remove(file)

    # Save the datasets into CSV files
    dataset.to_csv(f'Full_data_{pair}_{timeframe}.csv', index=True)
    training_set.to_csv(f'training_{pair}_{timeframe}_data.csv', index=True)
    validation_set.to_csv(f'validation_{pair}_{timeframe}_data.csv', index=True)
    
    return training_set, validation_set

def multiple():
    forex_pairs = [
    'GBPUSD', 'USDCHF', 'USDCAD', 'AUDUSD', 'AUDNZD', 'AUDCAD',
    'AUDCHF', 'GBPCAD', 'NZDUSD', 'EURGBP', 'EURAUD',
    'EURCHF', 'EURNZD', 'EURCAD', 'GBPCAD', 'GBPCHF',
    'CADCHF', 'GBPAUD', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'EURUSD'
    ]

    timeframe_str = input("Enter the currency pair (e.g., Daily, 1H): ").strip().upper()

    for pair in forex_pairs:
        print(f"Processing {pair} on {timeframe_str}")
        try:
            data = get_data_multiple(pair, timeframe_str)
            train_and_evaluate_models(data, timeframe_str, pair)
        except Exception as e:
            print(f"Failed to process {pair}: {str(e)}")

def move_directory(source_directory, destination_directory):
    # Ensure the destination directory exists where the source directory needs to be moved
    destination_path = os.path.join(destination_directory, os.path.basename(source_directory))
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Move the source directory to the new location
    shutil.move(source_directory, destination_path)

def calculate_target(data):
    data['close_price_next'] = data['close'].shift(-1)
    data['Actual Movement'] = np.where(data['close_price_next'] > data['close'], 1,
                                    np.where(data['close_price_next'] < data['close'], -1, 0))
    data.drop(columns=['close_price_next'], inplace=True)
    data = data.dropna()

def train_and_evaluate_models(data, timeframe, Pair):
    # Prepare data
    X = data.drop('Actual Movement', axis=1)
    y = data['Actual Movement']

    # Define the pipeline with the IndicatorTransformer
    pipeline = Pipeline([
        ('indicators', IndicatorTransformer()),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])

    # Define parameter grid for grid search
    param_grid = [
        {
            'indicators__sma_short_length': [30, 50, 70],
            'indicators__sma_long_length': [150, 200, 250],
            'indicators__ema_medium_length': [30, 50, 70],
            'indicators__ema_long_length': [150, 200, 250],
            'indicators__ema_short_length': [5, 9, 12],
            'indicators__ema_fast_length': [18, 21, 24],
            'indicators__original_bollinger_length': [15, 20, 25],
            'indicators__original_bollinger_std': [1.5, 2, 2.5],
            'indicators__bollinger_length': [10, 12, 15],
            'indicators__bollinger_std_dev': [1, 1.5, 2],
            'indicators__window': [7, 9, 12],
            'classifier': [LogisticRegression()],
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__penalty': ['l2']
        },
        {
            'indicators__sma_short_length': [30, 50, 70],
            'indicators__sma_long_length': [150, 200, 250],
            'indicators__ema_medium_length': [30, 50, 70],
            'indicators__ema_long_length': [150, 200, 250],
            'indicators__ema_short_length': [5, 9, 12],
            'indicators__ema_fast_length': [18, 21, 24],
            'indicators__original_bollinger_length': [15, 20, 25],
            'indicators__original_bollinger_std': [1.5, 2, 2.5],
            'indicators__bollinger_length': [10, 12, 15],
            'indicators__bollinger_std_dev': [1, 1.5, 2],
            'indicators__window': [7, 9, 12],
            'classifier': [RandomForestClassifier()],
            'classifier__n_estimators': [100, 200],
            'classifier__max_features': ['sqrt', 'log2']
        },
        {
            'indicators__sma_short_length': [30, 50, 70],
            'indicators__sma_long_length': [150, 200, 250],
            'indicators__ema_medium_length': [30, 50, 70],
            'indicators__ema_long_length': [150, 200, 250],
            'indicators__ema_short_length': [5, 9, 12],
            'indicators__ema_fast_length': [18, 21, 24],
            'indicators__original_bollinger_length': [15, 20, 25],
            'indicators__original_bollinger_std': [1.5, 2, 2.5],
            'indicators__bollinger_length': [10, 12, 15],
            'indicators__bollinger_std_dev': [1, 1.5, 2],
            'indicators__window': [7, 9, 12],
            'classifier': [SVC()],
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        }
    ]

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=TimeSeriesSplit(n_splits=5), scoring='accuracy', verbose=1)
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    best_model_params = grid_search.best_params_
    best_model_score = grid_search.best_score_

    # Create a DataFrame
    best_model_details = pd.DataFrame([{
        'Best Model Parameters': str(best_model_params),
        'Best Model Score': best_model_score
    }])

    # Save to CSV
    csv_path = 'best_model_details.csv'
    best_model_details.to_csv(csv_path, index=False)

    # Split the data into training and validation sets
    training_set, validation_set = split_and_save_dataset(data, timeframe, Pair)
    
    X_train = training_set.drop('Actual Movement', axis=1)
    y_train = training_set['Actual Movement']
    X_test = validation_set.drop('Actual Movement', axis=1)
    y_test = validation_set['Actual Movement']

    # Evaluate the best model on the validation set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy for evaluation on validation data: {accuracy:.2%}')

    # Save the best model
    joblib.dump(best_model, f'best_model_{Pair}_{timeframe}.joblib')

    # Export results
    results_df = pd.DataFrame({
        'Date': X_test.index,
        'Close Price': X_test['close'],
        'Actual Movement': y_test,
        'Predicted Movement': y_pred
    })
    results_df.to_csv('predicted_movements.csv', index=False)
    print("Results exported to 'predicted_movements.csv'.")

    # Optionally, save and display feature importances if your model supports it
    if hasattr(best_model.named_steps['classifier'], 'coef_'):
        importances = best_model.named_steps['classifier'].coef_[0]
        features = X_train.columns
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
        feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
        print("Feature importances:\n", feature_importance)

def plot_feature_importances(feature_importances):
    # Sort features by their importance
    feature_importances = feature_importances.sort_values(by='Importance', ascending=True)
    
    # Create barh plot
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    plt.close()

def main_menu():
    while True:
        print("\nMain Menu:")
        print("1 - Single Pair and Timeframe")
        print("2 - Multiple Pair and Timeframe")

        choice = input("Enter your choice (1/2/3/4/5): ")

        if choice == '1':
            data, timeframe, Pair = get_data()  # Fetch and prepare data
            train_and_evaluate_models(data, timeframe, Pair)
            break
        elif choice == '2':
            multiple()
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4 or 5.")

main_menu()