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

    calculate_indicators(eur_usd_data)

    # Filter the EUR/USD data for the in-sample training period
    dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]

    dataset = dataset.fillna(0)

    return dataset

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

    eur_usd_data = calculate_indicators(eur_usd_data) 

    # Filter the EUR/USD data for the in-sample training period
    dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]

    dataset = dataset.fillna(0)

    return dataset

def calculate_indicators(data, bollinger_length=12, bollinger_std_dev=1.5, sma_trend_length=50, window=9):
    # Calculate the 50-period simple moving average of the 'close' price
    data['SMA_50'] = ta.sma(data['close'], length=50)
    # Calculate the 200-period simple moving average of the 'close' price
    data['SMA_200'] = ta.sma(data['close'], length=200)
    
    # Calculate the 50-period exponential moving average of the 'close' price
    data['EMA_50'] = ta.ema(data['close'], length=50)
    # Calculate the 200-period exponential moving average of the 'close' price
    data['EMA_200'] = ta.ema(data['close'], length=200)

    data['previous_close'] = data['close'].shift(1)

    # Calculate the 9-period exponential moving average for scalping strategies
    data['EMA_9'] = ta.ema(data['close'], length=9)
    # Calculate the 21-period exponential moving average for scalping strategies
    data['EMA_21'] = ta.ema(data['close'], length=21)
    
    # Generate original Bollinger Bands with a 20-period SMA and 2 standard deviations
    original_bollinger = ta.bbands(data['close'], length=20, std=2)
    # The 20-period simple moving average for the middle band
    data['SMA_20'] = ta.sma(data['close'], length=20)
    # Upper and lower bands from the original Bollinger Bands calculation
    data['Upper Band'] = original_bollinger['BBU_20_2.0']
    data['Lower Band'] = original_bollinger['BBL_20_2.0']

    # Generate updated Bollinger Bands for scalping with custom length and standard deviation
    updated_bollinger = ta.bbands(data['close'], length=bollinger_length, std=bollinger_std_dev)
    # Assign lower, middle, and upper bands for scalping
    data['Lower Band Scalping'], data['Middle Band Scalping'], data['Upper Band Scalping'] = updated_bollinger['BBL_'+str(bollinger_length)+'_'+str(bollinger_std_dev)], ta.sma(data['close'], length=bollinger_length), updated_bollinger['BBU_'+str(bollinger_length)+'_'+str(bollinger_std_dev)]
    
    # Calculate the MACD indicator and its signal line
    macd = ta.macd(data['close'])
    data['MACD'] = macd['MACD_12_26_9']
    data['Signal_Line'] = macd['MACDs_12_26_9']
    
    # Calculate the Relative Strength Index (RSI) with the specified window length
    data[f'RSI_{window}'] = ta.rsi(data['close'], length=window).round(2)

    # Calculate a 5-period RSI for scalping strategies
    data[f'RSI_5 Scalping'] = ta.rsi(data['close'], length=5).round(2)

    # Calculate a simple moving average for trend analysis in scalping strategies
    data[f'SMA_{sma_trend_length}'] = ta.sma(data['close'], length=sma_trend_length)

    # Calculate the Stochastic Oscillator
    stoch = ta.stoch(data['high'], data['low'], data['close'])
    data['Stoch_%K'] = stoch['STOCHk_14_3_3']
    data['Stoch_%D'] = stoch['STOCHd_14_3_3']

    # Return the data with added indicators
    return data

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
            logist_regression(data)
        except Exception as e:
            print(f"Failed to process {pair}: {str(e)}")

def logist_regression(data):
    # https://chat.openai.com/g/g-cKXjWStaE-python/c/62d2f9bb-7a53-469c-bc37-8312a4175155
    # Feature engineering
    data['close_price_next'] = data['close'].shift(-1)
    data['Actual Movement'] = np.where(data['close_price_next'] > data['close'], 1,
                                       np.where(data['close_price_next'] < data['close'], -1, 0))
    data.drop(columns=['close_price_next'], inplace=True)

    data.dropna(inplace=True)  # Clean NaN values
    X = data.drop('Actual Movement', axis=1)  # Features
    y = data['Actual Movement']  # Labels

    # Model pipeline with L2 regularization
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, penalty='l2', C=0.5))

    # Initialize time series cross-validator
    tscv = TimeSeriesSplit(n_splits=10)

    # Implement cross-validation specifically for time series data
    scores = cross_val_score(model, X, y, cv=tscv)
    print("Cross-validated accuracy scores:", scores)
    print("Mean cross-validation score: %.2f" % scores.mean())

    # Fit the model on the full dataset for evaluation (might vary depending on the use case)
    model.fit(X, y)
    y_pred = model.predict(X)

    print(f'Model Accuracy for evaluation: {accuracy_score(y, y_pred):.2%}')

    # Extract coefficients from the LogisticRegression object within the pipeline
    # Since it's the last step of the pipeline, we access it with -1 index
    coefficients = model.named_steps['logisticregression'].coef_[0]
    features = X.columns
    feature_importance = pd.DataFrame({'Feature': features, 'Importance': coefficients})

    # Normalize the coefficients to positive values for better interpretation and visualization
    feature_importance['Importance'] = np.abs(feature_importance['Importance'])
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    print("Feature importances:\n", feature_importance)

    plot_feature_importances(feature_importance)

    return model

def plot_feature_importances(feature_importances):
    # Sort features by their importance
    feature_importances = feature_importances.sort_values(by='Importance', ascending=True)
    
    # Create barh plot
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.show()

def main_menu():
    while True:
        print("\nMain Menu:")
        print("1 - Single Pair and Timeframe")
        print("2 - Multiple Pair and Timeframe")

        choice = input("Enter your choice (1/2/3/4/5): ")

        if choice == '1':
            data = get_data()  # Fetch and prepare data
            logist_regression(data)
            break
        elif choice == '2':
            multiple()
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4 or 5.")

main_menu()