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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import optuna

class ForexTradingSimulator:
    def __init__(self, initial_balance, leverage, transaction_cost, lot_size):
        self.current_balance = initial_balance
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.transaction_cost = transaction_cost
        self.lot_size = lot_size
        self.trade_history = []
        self.position = 'neutral'
        self.entry_price = None
        self.is_open_position = False
        self.worst_case_pnl = 0
        self.best_case_pnl = 0
        self.worst_balance = 0
        self.last_processed_date = None  # Keep track of the last processed date
        self.monetary_loss = -(self.initial_balance * 0.005)  # Stop loss threshold in percentage (0.5%)
        self.monetary_gain = self.initial_balance * 0.01  # Take profit threshold in percentage (1%)
        self.stop_loss = 0
        self.take_profit = 0
        self.stop_loss_percent = 0.5
        self.take_profit_percent = 1.5

    def open_position(self, current_price, position_type, time):
        if not self.is_open_position:
            cost = self.transaction_cost
            self.current_balance -= cost  # Transaction cost
            self.entry_price = current_price
            self.position = position_type
            self.is_open_position = True

            # Calculate stop loss and take profit prices based on whether the trade is a buy or sell
            if position_type == 'long':
                # For buy orders, use the ask price
                self.stop_loss = current_price * (1 - self.stop_loss_percent / 100)  # Stop loss below the ask price
                self.take_profit = current_price * (1 + self.take_profit_percent / 100)  # Take profit above the ask price
            elif position_type == 'short':
                # For sell orders, use the bid price
                self.stop_loss = current_price  * (1 + self.stop_loss_percent / 100)  # Stop loss above the bid price
                self.take_profit = current_price * (1 - self.take_profit_percent / 100)  # Take profit below the bid price

            # Log the trade
            self.log_trade('open', current_price, time,)

    def close_position(self, current_price=None, time=None, profit=None):
        if current_price != None:
            if self.is_open_position:
                if self.position == 'long':
                    profit = (current_price - self.entry_price) * self.lot_size
                elif self.position == 'short':
                    profit = (self.entry_price - current_price) * self.lot_size

                self.current_balance += profit - self.transaction_cost
                self.is_open_position = False
                self.position = 'neutral'
                self.take_profit = 0
                self.stop_loss = 0
        else:
            self.current_balance += profit - self.transaction_cost
            self.is_open_position = False
            self.position = 'neutral'
            self.take_profit = 0
            self.stop_loss = 0

        # Log the trade
        self.log_trade('close', current_price, time , profit)

    def log_trade(self, action, price, time, profit=None):
        self.trade_history.append({
            'current_price': price,
            'time': time,
            'action': action,
            'position': self.position,
            'entry_price': self.entry_price if action == 'open' else None,
            'close_price': price if action == 'close' else None,
            'profit': profit,
            'balance': self.current_balance,
            'worst_case_pnl': self.worst_case_pnl,
            'best_cast_pnl': self.best_case_pnl
        })

    def update_current_pnl(self, high_price, low_price):
        if self.position == 'long':
            # Worst-case loss for a long position at the lowest price
            self.worst_case_pnl = (low_price - self.entry_price) * self.lot_size - self.transaction_cost
            # Best-case profit for a long position at the highest price
            self.best_case_pnl = (high_price - self.entry_price) * self.lot_size - self.transaction_cost
        elif self.position == 'short':
            # Worst-case loss for a short position at the highest price
            self.worst_case_pnl = (self.entry_price - high_price) * self.lot_size - self.transaction_cost
            # Best-case profit for a short position at the lowest price
            self.best_case_pnl = (self.entry_price - low_price) * self.lot_size - self.transaction_cost
        else:
            # If there is no open position, set both PnLs to 0
            self.worst_case_pnl = 0
            self.best_case_pnl = 0

    def simulate_trading(self, data):
        for _, row in data.iterrows():
            current_date = row['time'].date()  # Assuming 'time' is a datetime object

            # Update P&L after potentially resetting it for a new day
            self.update_current_pnl(row['high'], row['low'])
            
            # Check if there is a change in predicted movement and the current position status
            if row['Predicted Movement'] == 1 and self.position != 'long':
                if self.position:  # If there is an existing position, close it
                    self.close_position(current_price=row['close'], time=row['time'], profit=None)
                self.open_position(row['close'], 'long', row['time'])
            elif row['Predicted Movement'] == -1 and self.position != 'short':
                if self.position:  # If there is an existing position, close it
                    self.close_position(current_price=row['close'], time=row['time'], profit=None)
                self.open_position(row['close'], 'short', row['time'])
            else:
                self.log_trade('Holding', row['close'], row['time'] ,profit=None)

    def plot_balance_over_time(self, folder_name):
        # Create a DataFrame from the trade history
        trades = pd.DataFrame(self.trade_history)
        trades.set_index('time', inplace=True)  # Set time as the index

        # Plotting
        plt.figure(figsize=(10, 5))
        trades['balance'].plot(title='Balance Over Time', color='blue', marker='o')
        plt.xlabel('Time')
        plt.ylabel('Balance ($)')
        plt.grid(True)
        # Save the plot as a PNG file
        plot_path = os.path.join(folder_name, 'balance_over_time_backtest.png')
        plt.savefig(plot_path)
        plt.close()  # Close the plot figure to free up memory

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
                 ema_short_length=9, ema_fast_length=21, window=9):
        self.sma_short_length = sma_short_length
        self.sma_long_length = sma_long_length
        self.ema_medium_length = ema_medium_length
        self.ema_long_length = ema_long_length
        self.ema_short_length = ema_short_length
        self.ema_fast_length = ema_fast_length
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Implementing SMA, EMA, and Bollinger Bands
        X[f'SMA_Short'] = ta.sma(X['close'], length=self.sma_short_length)
        X[f'SMA_Long'] = ta.sma(X['close'], length=self.sma_long_length)
        X[f'EMA_Medium'] = ta.ema(X['close'], length=self.ema_medium_length)
        X[f'EMA_Long'] = ta.ema(X['close'], length=self.ema_long_length)
        X[f'EMA_Short'] = ta.ema(X['close'], length=self.ema_short_length)
        X[f'EMA_Fast'] = ta.ema(X['close'], length=self.ema_fast_length)

        macd = ta.macd(X['close'])
        X['MACD'] = macd['MACD_12_26_9']
        X['Signal_Line'] = macd['MACDs_12_26_9']
        X[f'RSI'] = ta.rsi(X['close'], length=self.window).round(2)

        # Ensure all data manipulations have been completed before dropping NA
        pd.set_option('future.no_silent_downcasting', True)
        X.fillna(0, inplace=True)

        return X

def calculate_indicators(data, **kwargs):
        data = data.copy()
        # Assuming your indicators function can handle these keyword arguments:
        sma_short_length = kwargs.get('sma_short_length') 
        sma_long_length = kwargs.get('sma_long_length')
        ema_medium_length = kwargs.get('ema_medium_length')
        ema_long_length = kwargs.get('ema_long_length')
        ema_short_length = kwargs.get('ema_short_length')
        ema_fast_length = kwargs.get('ema_fast_length')
        window = kwargs.get('window')

        data = data.copy()
        # Implementing SMA, EMA, and Bollinger Bands
        data[f'SMA_Short'] = ta.sma(data['close'], length=sma_short_length)
        data[f'SMA_Long'] = ta.sma(data['close'], length=sma_long_length)
        data[f'EMA_Medium'] = ta.ema(data['close'], length=ema_medium_length)
        data[f'EMA_Long'] = ta.ema(data['close'], length=ema_long_length)
        data[f'EMA_Short'] = ta.ema(data['close'], length=ema_short_length)
        data[f'EMA_Fast'] = ta.ema(data['close'], length=ema_fast_length)

        macd = ta.macd(data['close'])
        data['MACD'] = macd['MACD_12_26_9']
        data['Signal_Line'] = macd['MACDs_12_26_9']
        data[f'RSI'] = ta.rsi(data['close'], length=window).round(2)

        # Ensure all data manipulations have been completed before dropping NA
        pd.set_option('future.no_silent_downcasting', True)
        data.fillna(0, inplace=True)

        return data

def get_data(choice=None, symbol=None,timeframe=None):
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

    if choice == '2':
        Pair = symbol
        timeframe_str = timeframe
        # Fetch and prepare the FX data for the specified currency pair and timeframe
        eur_usd_data = fetch_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)
    else:
        # Prompt the user for the desired timeframe for analysis and standardize the input
        timeframe_str = input("Enter the currency pair (e.g., Daily, 1H): ").strip().upper()
        # Prompt the user for the currency pair they're interested in and standardize the input
        Pair = input("Enter the currency pair (e.g., GBPUSD, EURUSD): ").strip().upper()
            # Fetch and prepare the FX data for the specified currency pair and timeframe
        eur_usd_data = fetch_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)

    calculate_target(eur_usd_data)

    # Filter the EUR/USD data for the in-sample training period
    dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]

    dataset = dataset.fillna(0)

    return dataset, timeframe_str, Pair

def split_and_save_dataset(dataset, timeframe, pair):
    if len(dataset) < 10:
        raise ValueError("Dataset is too small to split effectively.")

    # Calculate the split index for an 80/20 split
    split_index = int(len(dataset) * 0.8)
    
    # Split the dataset into training and validation sets
    training_set = dataset.iloc[:split_index]
    testing_set = dataset.iloc[split_index:]

    # Clean up existing CSV files related to previous runs
    file_patterns = [f'Full_data.csv', 
                     f'training.csv', 
                     f'testing.csv']
    for pattern in file_patterns:
        for file in glob.glob(pattern):
            os.remove(file)

    # Save the datasets into CSV files
    dataset.to_csv(f'Full_data.csv', index=True)
    training_set.to_csv(f'training.csv', index=True)
    testing_set.to_csv(f'testing.csv', index=True)
    
    return training_set, testing_set

def multiple(choice):
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
            data,_,_ = get_data(choice,pair, timeframe_str)
            train_and_evaluate_models_optuna(data, timeframe_str, pair)
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
    training_set, _ =  split_and_save_dataset(data, timeframe, Pair)
    # Prepare data
    X = training_set.drop('Actual Movement', axis=1)
    y = training_set['Actual Movement']

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
            'indicators__window': [7, 9, 12],
            'scaler': [StandardScaler(), MinMaxScaler()],
            'classifier': [LogisticRegression()],
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__penalty': ['l2'],
            'classifier__max_iter': [100, 200, 500, 1000, 1500, 2000]  # Varying numbers of iterations
        },
        {
            'indicators__sma_short_length': [30, 50, 70],
            'indicators__sma_long_length': [150, 200, 250],
            'indicators__ema_medium_length': [30, 50, 70],
            'indicators__ema_long_length': [150, 200, 250],
            'indicators__ema_short_length': [5, 9, 12],
            'indicators__ema_fast_length': [18, 21, 24],
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
            'indicators__window': [7, 9, 12],
            'scaler': [StandardScaler(), MinMaxScaler()],
            'classifier': [SVC()],
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        }
    ]

    # Perform grid search # GridSearchCV # RandomizedSearchCV
    grid_search = RandomizedSearchCV(pipeline, param_grid, cv=TimeSeriesSplit(n_splits=5), scoring='accuracy', verbose=1)
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

    data_indicators = calculate_indicators(data, **best_model_params)

    _, testing_set =  split_and_save_dataset(data_indicators, timeframe, Pair)
    
    X_test = testing_set.drop('Actual Movement', axis=1)
    y_test = testing_set['Actual Movement']

    # Evaluate the best model on the validation set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Model Accuracy for evaluation on validation data: {accuracy:.2%}')

    scaler_path = f'scaler_{Pair}_{timeframe}.joblib'

    # Save the best model
    joblib.dump(best_model, f'best_model_{Pair}_{timeframe}.joblib')
    joblib.dump(best_model.named_steps['scaler'], scaler_path)

    # Export results
    results_df = pd.DataFrame({
        'Date': X_test.index,
        'Close Price': X_test['close'],
        'Actual Movement': y_test,
        'Predicted Movement': y_pred
    })
    results_df.to_csv('predicted_movements.csv', index=False)

    test_trade(accuracy, Pair, timeframe)

def train_and_evaluate_models_optuna(data, timeframe, Pair):
    def objective(trial):
        # Split data
        training_set, _ = split_and_save_dataset(data, timeframe, Pair)
        X = training_set.drop('Actual Movement', axis=1)
        y = training_set['Actual Movement']

        # Hyperparameters for IndicatorTransformer
        sma_short_length = trial.suggest_categorical('sma_short_length', [30, 50, 70])
        sma_long_length = trial.suggest_categorical('sma_long_length', [150, 200, 250])
        ema_medium_length = trial.suggest_categorical('ema_medium_length', [30, 50, 70])
        ema_long_length = trial.suggest_categorical('ema_long_length', [150, 200, 250])
        ema_short_length = trial.suggest_categorical('ema_short_length', [5, 9, 12])
        ema_fast_length = trial.suggest_categorical('ema_fast_length', [18, 21, 24])
        window = trial.suggest_categorical('window', [7, 9, 12])

        # Choose a scaler
        scaler = trial.suggest_categorical('scaler', ['StandardScaler', 'MinMaxScaler'])
        if scaler == 'StandardScaler':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        # Hyperparameters to tune for classifiers
        classifier_name = trial.suggest_categorical('classifier', ['LogisticRegression', 'RandomForest', 'SVC'])

        if classifier_name == 'LogisticRegression':
            C = trial.suggest_loguniform('lr_C', 1e-3, 1e3)
            max_iter = trial.suggest_int('lr_max_iter', 100, 2000)
            classifier = LogisticRegression(C=C, max_iter=max_iter)
        elif classifier_name == 'RandomForest':
            n_estimators = trial.suggest_int('rf_n_estimators', 100, 1000)
            max_features = trial.suggest_categorical('rf_max_features', ['sqrt', 'log2'])
            classifier = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features)
        elif classifier_name == 'SVC':
            C = trial.suggest_loguniform('svc_C', 1e-3, 1e3)
            kernel = trial.suggest_categorical('svc_kernel', ['linear', 'rbf'])
            classifier = SVC(C=C, kernel=kernel)

        # Setup the pipeline
        pipeline = Pipeline([
            ('indicators', IndicatorTransformer(sma_short_length=sma_short_length, sma_long_length=sma_long_length,
                                                ema_medium_length=ema_medium_length, ema_long_length=ema_long_length,
                                                ema_short_length=ema_short_length, ema_fast_length=ema_fast_length,
                                                window=window)),
            ('scaler', scaler),
            ('classifier', classifier)
        ])

        # Perform cross-validation
        score = cross_val_score(pipeline, X, y, cv=TimeSeriesSplit(n_splits=5), scoring='accuracy')
        trial.set_user_attr('pipeline', pipeline)
        return score.mean()

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # Retrieve the best model
    best_params = study.best_params
    best_score = study.best_value
    best_pipeline = study.best_trial.user_attrs['pipeline']
    best_scaler = best_pipeline.named_steps['scaler']

    # Convert the best parameters dictionary to a DataFrame
    best_params_df = pd.DataFrame([best_params])

    # Add the best score to the DataFrame
    best_params_df['Best Score'] = best_score

    # Define a path for the CSV file
    csv_path = 'best_model_params.csv'

    # Save the DataFrame to a CSV file
    best_params_df.to_csv(csv_path, index=False)

    # Re-train the best model on the entire dataset
    training_set, testing_set = split_and_save_dataset(data, timeframe, Pair)
    X_train = training_set.drop('Actual Movement', axis=1)
    y_train = training_set['Actual Movement']

    best_pipeline.fit(X_train, y_train)  # Correctly reference the training data

    X_test = testing_set.drop('Actual Movement', axis=1)
    y_test = testing_set['Actual Movement']

    # Evaluate the best model on the validation set
    y_pred = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Model Accuracy for evaluation on validation data: {accuracy:.2%}')

    # Save the best model
    joblib.dump(best_pipeline, f'best_model_{Pair}_{timeframe}.joblib')
    # Save the scaler
    joblib.dump(best_scaler, f'scaler_{Pair}_{timeframe}.joblib')

    # Export results
    results_df = pd.DataFrame({
        'Date': X_test.index,
        'Close Price': X_test['close'],
        'Actual Movement': y_test,
        'Predicted Movement': y_pred
    })
    results_df.to_csv('predicted_movements.csv', index=False)

    # Additional function that may be defined elsewhere
    test_trade(accuracy, Pair, timeframe)

def analyze_pair_data(df):
    """ Analyze data for a single currency pair and return key metrics. """
    # Make a copy of the DataFrame to avoid SettingWithCopyWarning when modifying data
    df_copy = df.copy()

    # Ensure 'time' column is in datetime format
    df_copy['time'] = pd.to_datetime(df_copy['time'])

    # Calculate daily profit
    daily_profit = df_copy.groupby('time')['profit'].sum()

    worst_daily_pnl = df_copy.groupby('time')['worst_case_pnl'].sum()

    # Initial settings for simulation
    initial_balance = 10000
    upper_reset_threshold = 11000
    lower_reset_threshold = 9000
    upper_reset_count = 0
    lower_reset_count = 0

    daily_max_loss_reset_threshold = 9500
    daily_max_loss_count = 0

    # Calculate cumulative balance with resets
    balances = [initial_balance]
    for profit in daily_profit:
        new_balance = balances[-1] + profit
        if new_balance >= upper_reset_threshold:
            balances.append(initial_balance)  # Reset to initial balance
            upper_reset_count += 1
        elif new_balance <= lower_reset_threshold:
            balances.append(initial_balance)  # Reset to initial balance
            lower_reset_count += 1
        else:
            balances.append(new_balance)

    balances_2 = [initial_balance]

    for worst_daily in worst_daily_pnl:
        new_balance = balances_2[-1] + worst_daily
        if new_balance <= daily_max_loss_reset_threshold:
            balances_2.append(initial_balance)
            daily_max_loss_count += 1
        elif new_balance <= lower_reset_threshold:
            balances.append(initial_balance)  # Reset to initial balance
            lower_reset_count += 1
        else:
            balances_2.append(new_balance)

    total_resets = upper_reset_count + lower_reset_count + daily_max_loss_count
    if total_resets > 0:
        probability_of_passing = (upper_reset_count / total_resets) * 100
    else:
        probability_of_passing = 0  # Set probability to 0 (or another appropriate value) when no resets have occurred

    return {
        'TotalCumulativeProfit': daily_profit.sum(),
        'ProbabilityOfPassing': probability_of_passing,
        'PositiveResets': upper_reset_count,
        'NegativeResets': lower_reset_count,
        'daily_max_loss_count': daily_max_loss_count
    }

def compute_probabilities(df):
    result = []
    analysis_results = analyze_pair_data(df)
    result.append({
        'Probability': analysis_results['ProbabilityOfPassing'],
        'TotalProfit': analysis_results['TotalCumulativeProfit'],
        'CountPositiveReset': analysis_results['PositiveResets'],
        'CountNegativeReset': analysis_results['NegativeResets'],
        'CountDailyLossReset': analysis_results['daily_max_loss_count']
    })
    
    result_df = pd.DataFrame(result)
    return result_df.sort_values(by='Probability', ascending=False)

def perform_analysis():
    combined_df = pd.read_csv('trade_history_backtest.csv')
    probability_rankings = compute_probabilities(combined_df)
    # Get the current working directory
    current_directory = os.getcwd()
    # List all subdirectories in the current directory
    all_subdirs = [os.path.join(current_directory, d) for d in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, d))]

    # Find the most recent directory
    most_recent_dir = max(all_subdirs, key=os.path.getmtime)
    
    probability_rankings.to_csv('forex_pair_probability_rankings.csv', index=False)

    highest_probability = probability_rankings['Probability'].max()

    return highest_probability

def test_trade(accuracy, Pair, timeframe):
    predicted_movements = pd.read_csv('predicted_movements.csv', index_col='Date', parse_dates=True)
    # Reading the second CSV and setting the date column as index
    testing = pd.read_csv('testing.csv', index_col='time', parse_dates=True)

    columns_from_predicted_movements = predicted_movements[['Predicted Movement']]
    columns_testing = testing[['high', 'close', 'low']]

    # Concatenating the selected columns along the axis=1 (side by side), matching by index (date)
    concatenated_columns = pd.concat([columns_from_predicted_movements, columns_testing], axis=1)

    concatenated_columns_reset = concatenated_columns.reset_index()

    backtest_data = concatenated_columns_reset.rename(columns={'index': 'time'})

    initial_balance=10000
    leverage=30
    transaction_cost=0.0002
    lot_size = 10000

    folder_name = os.getcwd()

    data_csv_filename = os.path.join(folder_name, 'data_backtest.csv')
    backtest_data.to_csv(data_csv_filename)

    trader = ForexTradingSimulator(
        initial_balance,
        leverage,
        transaction_cost,
        lot_size,
    )
    
    trader.simulate_trading(backtest_data)

    trader.plot_balance_over_time(folder_name)

    trade_history_df = pd.DataFrame(trader.trade_history)

    trade_history_filename = os.path.join(folder_name, 'trade_history_backtest.csv')
    trade_history_df.to_csv(trade_history_filename)

    highest_probability = perform_analysis()

    print(highest_probability)

    save_directory = f'model_{accuracy:.2%}_{Pair}_{timeframe}'

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Save all files except the specified ones
    exclude_files = ['things to do.txt', 'MLP.py', 'test_1.py', 'Chart.csv', 'Chart_1h.csv', 'Chart_Latest.csv', 'LSTM.py', 'RNN.py', 'XGboost.py', 'manual.py', 'logistic.py']
    for file in os.listdir('.'):
        if file not in exclude_files and os.path.isfile(file):
            shutil.move(file, os.path.join(save_directory, file))

def main_menu():
    while True:
        print("\nMain Menu:")
        print("1 - Single Pair and Timeframe")
        print("2 - Multiple Pair and Timeframe")

        choice = input("Enter your choice (1/2/3/4/5): ")

        if choice == '1':
            data, timeframe, Pair = get_data()  # Fetch and prepare data
            train_and_evaluate_models_optuna(data, timeframe, Pair)
            break
        elif choice == '2':
            multiple(choice)
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4 or 5.")

main_menu()