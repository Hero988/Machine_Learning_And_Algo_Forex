import pandas as pd
import datetime
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pytz
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import time
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse

# On support and resistance section: https://www.youtube.com/watch?v=kzRsEU3M7zY&ab_channel=TheTradingGeek

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

def identify_flag_patterns(data):
    # Calculating necessary columns for pole detection
    data['price_change'] = data['close'].diff()
    data['abs_price_change'] = data['price_change'].abs()

    # Rolling window to detect sharp movements (pole)
    rolling_max_change = data['abs_price_change'].rolling(window=10).max()

    # Identifying potential poles for bullish and bearish flags
    threshold = rolling_max_change.quantile(0.90)  # Using 90th percentile as a threshold for significant movement
    data['potential_pole'] = rolling_max_change > threshold

    # Identifying flag formation
    data['high_slope'] = data['high'].rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    data['low_slope'] = data['low'].rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])

    # Bullish flag: highs descending, lows ascending
    data['bullish_flag'] = (data['high_slope'] < 0) & (data['low_slope'] > 0) & data['potential_pole']

    # Bearish flag: highs ascending, lows descending
    data['bearish_flag'] = (data['high_slope'] > 0) & (data['low_slope'] < 0) & data['potential_pole']

    # Identifying breakout/breakdown
    # Bullish breakout when the close price goes above the last high of the flag
    data['bullish_breakout'] = (data['close'] > data['high'].shift(1)) & data['bullish_flag']

    # Bearish breakdown when the close price goes below the last low of the flag
    data['bearish_breakdown'] = (data['close'] < data['low'].shift(1)) & data['bearish_flag']

    # Simplifying the DataFrame to the necessary columns
    data = data[['close', 'high', 'low', 'potential_pole', 'bullish_flag', 'bearish_flag', 'bullish_breakout', 'bearish_breakdown']]

    return data

def trend_identification_live(data, window_size):
    # Calculate rolling maximum of highs and minimum of lows
    data['rolling_max_high'] = data['high'].rolling(window=window_size, min_periods=1).max()
    data['rolling_min_low'] = data['low'].rolling(window=window_size, min_periods=1).min()

    # Identify if the current high is higher or lower than the rolling max from the previous period
    data['higher_highs'] = data['high'] > data['rolling_max_high'].shift(1)
    data['lower_highs'] = data['high'] < data['rolling_max_high'].shift(1)

    # Identify if the current low is higher or lower than the rolling min from the previous period
    data['higher_lows'] = data['low'] > data['rolling_min_low'].shift(1)
    data['lower_lows'] = data['low'] < data['rolling_min_low'].shift(1)

    # Aggregate counts of conditions within the window
    data['count_higher_highs'] = data['higher_highs'].rolling(window=window_size, min_periods=1).sum()
    data['count_higher_lows'] = data['higher_lows'].rolling(window=window_size, min_periods=1).sum()
    data['count_lower_highs'] = data['lower_highs'].rolling(window=window_size, min_periods=1).sum()
    data['count_lower_lows'] = data['lower_lows'].rolling(window=window_size, min_periods=1).sum()

    # Define trend based on majority vote
    data['uptrend'] = (data['count_higher_highs'] + data['count_higher_lows']) > \
                      (data['count_lower_highs'] + data['count_lower_lows'])

    data['downtrend'] = (data['count_lower_highs'] + data['count_lower_lows']) > \
                        (data['count_higher_highs'] + data['count_higher_lows'])

    data['consolidation'] = ~data['uptrend'] & ~data['downtrend']

    # Store last higher low and lower high for retest
    data['last_higher_low'] = data['low'].where(data['higher_lows']).ffill()
    data['last_lower_high'] = data['high'].where(data['lower_highs']).ffill()

    # Initialize the columns if they are not present in the DataFrame
    if 'breakdown_retest_persistence' not in data:
        data['breakdown_retest_persistence'] = False
    if 'breakout_retest_persistence' not in data:
        data['breakout_retest_persistence'] = False

    # Evaluate potential retests
    data['potential_breakdown_retest'] = data['low'] < data['last_higher_low']
    data['potential_breakout_retest'] = data['high'] > data['last_lower_high']

    # Applying the 5-period persistence for retests
    persistence_period = 5
    # Iterate through the DataFrame using integer index with .iloc or use the actual DateTime index with .at
    for i in range(len(data)):
        current_time = data.index[i]
        # Use current_time for accessing rows with .at
        if data.at[current_time, 'potential_breakdown_retest']:
            # Set the persistence flag for the current and next 5 periods
            end_index = min(i + persistence_period, len(data) - 1)
            data.loc[data.index[i:end_index], 'breakdown_retest_persistence'] = True
        
        if data.at[current_time, 'potential_breakout_retest']:
            end_index = min(i + persistence_period, len(data) - 1)
            data.loc[data.index[i:end_index], 'breakout_retest_persistence'] = True

    # Removing future-looking conditions
    # Determine the current breakdown and breakout confirmation based on the most recent data only
    data['confirm_breakdown_retest'] = data['potential_breakdown_retest'] & \
                                       (data['high'] >= data['last_higher_low']) & \
                                       (data['high'].shift(1) < data['high'])

    data['confirm_breakout_retest'] = data['potential_breakout_retest'] & \
                                      (data['low'] <= data['last_lower_high']) & \
                                      (data['low'].shift(1) > data['low'])

    return data

def calculate_movement(data):
    # Initialize candlestick pattern encoding
    data['candlestick_pattern'] = 0  # Default to 'No pattern'

    # Identify Bull, Bear, or Neutral candle
    data['candle_type'] = np.where(data['close'] > data['open'], 1,
                                   np.where(data['close'] < data['open'], -1, 0))
    
    # Doji
    doji_mask = ((data['close'] == data['open']) | 
                 (abs(data['close'] - data['open']) < (data['high'] - data['low']) * 0.1))
    data.loc[doji_mask, 'candlestick_pattern'] = 1
    
    # Improved Bullish Engulfing
    bullish_engulfing_mask = ((data['open'] < data['close']) &  # Current is a bull candle
                              (data['open'].shift(1) > data['close'].shift(1)) &  # Previous is a bear candle
                              (data['open'] < data['close'].shift(1)) &  # Current open is less than previous close
                              (data['close'] > data['open'].shift(1)))  # Current close is more than previous open
    data.loc[bullish_engulfing_mask, 'candlestick_pattern'] = 2
    
    # Improved Bearish Engulfing
    bearish_engulfing_mask = ((data['open'] > data['close']) &  # Current is a bear candle
                              (data['open'].shift(1) < data['close'].shift(1)) &  # Previous is a bull candle
                              (data['open'] > data['close'].shift(1)) &  # Current open is higher than previous close
                              (data['close'] < data['open'].shift(1)))  # Current close is lower than previous open
    data.loc[bearish_engulfing_mask, 'candlestick_pattern'] = 3

    # Bullish Pin Bar
    bullish_pin_bar_mask = ((data['close'] > data['open']) &  # Bull candle
                            ((data['open'] - data['low']) > 2 * (data['close'] - data['open'])) &  # Long lower shadow
                            ((data['high'] - data['close']) < (data['close'] - data['open'])))  # Small upper shadow
    data.loc[bullish_pin_bar_mask, 'candlestick_pattern'] = 4

    # Bearish Pin Bar
    bearish_pin_bar_mask = ((data['close'] < data['open']) &  # Bear candle
                            ((data['high'] - data['open']) > 2 * (data['open'] - data['close'])) &  # Long upper shadow
                            ((data['close'] - data['low']) < (data['open'] - data['close'])))  # Small lower shadow
    data.loc[bearish_pin_bar_mask, 'candlestick_pattern'] = 5

    # Morning Star pattern detection
    morning_star_mask = (
        (data['close'].shift(2) > data['open'].shift(2)) &  # First candle is bearish
        ((data['open'].shift(1) < data['close'].shift(2)) |  # Second candle gaps down
         (data['close'].shift(1) < data['close'].shift(2))) &
        (data['close'].shift(1) > data['open'].shift(1)) &  # Second candle is bullish
        (data['open'] < data['close']) &  # Third candle is bullish
        (data['close'] > (data['open'].shift(2) + (data['close'].shift(2) - data['open'].shift(2)) / 2)) &  # Third candle closes above the midpoint of the first
        ((data['close'] - data['open']) >= (data['open'].shift(2) - data['close'].shift(2)))  # Third candle is as large or larger than the first
    )
    data.loc[morning_star_mask, 'candlestick_pattern'] = 6

    # Evening Star pattern detection
    evening_star_mask = (
        (data['close'].shift(2) < data['open'].shift(2)) &  # First candle is bullish
        ((data['open'].shift(1) > data['close'].shift(2)) |  # Second candle gaps up
         (data['close'].shift(1) > data['close'].shift(2))) &
        (data['close'].shift(1) < data['open'].shift(1)) &  # Second candle is bearish or doji
        (data['open'] > data['close']) &  # Third candle is bearish
        (data['close'] < (data['open'].shift(2) + (data['close'].shift(2) - data['open'].shift(2)) / 2))  # Third candle closes below the midpoint of the first
    )
    data.loc[evening_star_mask, 'candlestick_pattern'] = 7  # Assign a unique identifier for the Evening Star pattern

    return data

def split_and_save_dataset(dataset, timeframe='4h', pair='6B'):
    # Calculate the split index for training and testing
    split_index_train_test = int(len(dataset) * 0.8)
    
    # Split the dataset into training and testing
    training_set = dataset.iloc[:split_index_train_test]
    testing_set = dataset.iloc[split_index_train_test:]
    
    # Further split the training set to create a validation set (80% training, 20% validation of the training set)
    split_index_train_val = int(len(training_set) * 0.8)
    final_training_set = training_set.iloc[:split_index_train_val]
    validation_set = training_set.iloc[split_index_train_val:]

    # Search for CSV files starting with "validation"
    validation_files = glob.glob('validation*.csv')

    # Search for CSV files starting with "training"
    training_files = glob.glob('training*.csv')

    # Search for CSV files starting with "testing"
    testing_files = glob.glob('testing*.csv')

    for file in validation_files:
        os.remove(file)

    for file in training_files:
        os.remove(file)

    for file in testing_files:
        os.remove(file)

    # Save the sets into CSV files
    final_training_set.to_csv(f'training_{pair}_{timeframe}_data.csv', index=True)
    validation_set.to_csv(f'validation_{pair}_{timeframe}_data.csv', index=True)
    testing_set.to_csv(f'testing_{pair}_{timeframe}_data.csv', index=True)
    
    # Return the split datasets
    return final_training_set, validation_set, testing_set

def get_backtest_data(dataset, timeframe='4h', pair='6B'):
    # Get today's date
    today = datetime.now()
    
    # Calculate the date three months ago
    three_months_ago = today - relativedelta(months=3)

    date_three_months_ago = three_months_ago.strftime('%Y-%m-%d')

    # Retrieve and store the current date
    current_date_string = str(datetime.now().date())

    dataset = dataset[(dataset.index >= date_three_months_ago) & (dataset.index <= current_date_string)]

    # Search for CSV files starting with "testing"
    testing_files = glob.glob('backtest_*.csv')

    for file in testing_files:
        os.remove(file)

    dataset.to_csv(f'backtest_{pair}_{timeframe}_data.csv', index=True)

    return dataset

def read_csv_to_dataframe(file_path):

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    if 'DateTime' in df.columns:
        # Rename the column
        df.rename(columns={'DateTime': 'time'}, inplace=True)
    
    if 'Close' in df.columns:
        # Rename the column
        df.rename(columns={'Close': 'close'}, inplace=True)

    # Conditionally strip timezone information if it ends with a pattern like +00:00 or -00:00
    df['time'] = df['time'].str.replace(r"\s+\+\d{2}:\d{2}$", "", regex=True)
    df['time'] = df['time'].str.replace(r"\s+\-\d{2}:\d{2}$", "", regex=True)

    # List of possible date formats
    date_formats = [
        '%m/%d/%Y %H:%M:%S',  # Format 1: Month/Day/Year Hour:Minute:Second
        '%d/%m/%Y %H:%M:%S',  # Format 2: Day/Month/Year Hour:Minute:Second
        '%Y-%m-%d %H:%M:%S', # Format 3: Year-Month-Day Hour:Minute:Second
        '%Y-%m-%d' # Format 4: Year-Month-Day
    ]

    # Attempt to convert the 'time' column to datetime
    for date_format in date_formats:
        try:
            df['time'] = pd.to_datetime(df['time'], format=date_format)
            break
        except ValueError:
            continue
    else:
        # If all formats fail, raise an error
        raise ValueError("Date format not supported. Please check the 'time' data.")

    # Set the 'time' column as the DataFrame index
    df.set_index('time', inplace=True)

    # Remove any columns that only contain NaN values
    df.dropna(axis=1, how='all', inplace=True)

    # Drop rows where any of the data is missing
    df = df.dropna()

    return df

def main_training_loop_multiple_pairs():
    # Define a list of forex pairs
    forex_pairs = [
        'EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY', 'USDCAD',
        'AUDUSD', 'AUDNZD', 'AUDCAD', 'AUDCHF', 'AUDJPY',
        'GBPCAD', 'NZDUSD', 'CHFJPY', 'EURGBP', 'EURAUD',
        'EURCHF', 'EURJPY', 'EURNZD', 'EURCAD', 'GBPCAD',
        'GBPCHF', 'GBPJPY', 'CADCHF', 'CADJPY', 'GBPAUD',
        'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY'
    ]
    
    # Ask user for the timeframe
    timeframe = input("Enter the timeframe (e.g., Daily, 1H): ").strip().upper()

    # Strategy Selection
    strategies = {
        '1': 'Reversal'
        # Add more strategies as needed
    }

    choice = '1'

    # Loop through each strategy and forex pair combination
    for strategy_key in strategies:
        for pair in forex_pairs:
            print(f"Training for {pair} on {timeframe} using {strategies[strategy_key]}")
            multiple_manual_trading(choice, pair, timeframe, strategy_key)

def multiple_manual_trading(choice, Pair, timeframe_str, choice_strategy):
    # Retrieve and store the current date
    current_date = str(datetime.now().date())

    # Hardcoded start date for strategy evaluation
    strategy_start_date_all = "1971-01-04"
    # Use the current date as the end date for strategy evaluation
    strategy_end_date_all = current_date

    # Convert string representation of dates to datetime objects for further processing
    start_date_all = datetime.strptime(strategy_start_date_all, "%Y-%m-%d")
    end_date_all = datetime.strptime(strategy_end_date_all, "%Y-%m-%d")

    training_start_date = "2000-01-01"
    training_end_date = current_date

    window_size = 5

    # Fetch and prepare the FX data for the specified currency pair and timeframe
    eur_usd_data = fetch_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)

    if choice_strategy == '1':
        trend_identification_live(eur_usd_data, window_size)
    elif choice_strategy == '2':
        identify_flag_patterns(eur_usd_data)

    # Filter the EUR/USD data for the in-sample training period
    dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]

    # Drop rows where any of the data is missing
    dataset = dataset.dropna()

    backtest_set = get_backtest_data(dataset, timeframe_str, Pair)

    initial_balance=10000
    leverage=30
    transaction_cost=0.0002

    folder_name = os.getcwd()

    if 'JPY' in Pair:
        lot_size = 100  # Smaller lot size for pairs including JPY
    else:
        lot_size = 1000  # Default lot size for other pairs

    trader = ForexTradingSimulator(
        initial_balance,
        leverage,
        transaction_cost,
        lot_size,
    )
    backtest_set = backtest_set.reset_index()
    trader.simulate_trading(backtest_set, choice_strategy)

    trader.plot_balance_over_time(folder_name)

    trade_history_df = pd.DataFrame(trader.trade_history)

    trade_history_filename = os.path.join(folder_name, 'trade_history_backtest.csv')
    trade_history_df.to_csv(trade_history_filename)

    data_csv_filename = os.path.join(folder_name, 'data_backtest.csv')
    backtest_set.to_csv(data_csv_filename)

    highest_probability = perform_analysis(choice)

    if choice_strategy == '1':
        save_directory = f'bactest_reversal_patterns_{highest_probability:.2f}%_{Pair}_{timeframe_str}'
    elif choice_strategy == '2':
        save_directory = f'bactest_flag_patterns_{highest_probability:.2f}%_{Pair}_{timeframe_str}'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save all files except the specified ones
    exclude_files = ['things to do.txt', 'MLP.py', 'test_1.py', 'Chart.csv', 'Chart_1h.csv', 'Chart_Latest.csv', 'LSTM.py', 'RNN.py', 'XGboost.py', 'manual.py']

    for file in os.listdir('.'):
        if file not in exclude_files and os.path.isfile(file):
            shutil.move(file, os.path.join(save_directory, file))

def different():
    """
    identify_flag_patterns()
    # Check if there is a change in predicted movement
    if row['bullish_breakout'] == True and (not self.is_open_position or self.position == 'short'):
        self.open_position(row['close'], 'long', row['time'])
    elif row['bearish_breakdown'] == True and (not self.is_open_position or self.position == 'long'):
        self.open_position(row['close'], 'short', row['time'])
    # Log the trade as hold if there is no change in position
    elif self.is_open_position:
        self.log_trade('hold', row['close'], row['time'])

    trend_identification(data, window_size)
    # Check if there is a change in predicted movement
    if row['trend_change'] == True and row['uptrend'] == True and (not self.is_open_position or self.position == 'short'):
        self.open_position(row['close'], 'long', row['time'])
    elif row['trend_change'] == True and row['uptrend'] == True and (not self.is_open_position or self.position == 'long'):
        self.open_position(row['close'], 'short', row['time'])
    # Log the trade as hold if there is no change in position
    elif self.is_open_position:
        self.log_trade('hold', row['close'], row['time'])
    """

def manual_trading(choice):
    # Retrieve and store the current date
    current_date = str(datetime.now().date())

    # Hardcoded start date for strategy evaluation
    strategy_start_date_all = "1971-01-04"
    # Use the current date as the end date for strategy evaluation
    strategy_end_date_all = current_date

    # Convert string representation of dates to datetime objects for further processing
    start_date_all = datetime.strptime(strategy_start_date_all, "%Y-%m-%d")
    end_date_all = datetime.strptime(strategy_end_date_all, "%Y-%m-%d")

    # Prompt the user to decide if they want real-time data updates and store the boolean result
    #enable_real_time = input("Do you want to enable real-time data updates? (yes/no): ").lower().strip() == 'yes'

    # Prompt the user for the desired timeframe for analysis and standardize the input
    timeframe_str = input("Enter the currency pair (e.g., Daily, 1H): ").strip().upper()
    # Prompt the user for the currency pair they're interested in and standardize the input
    Pair = input("Enter the currency pair (e.g., GBPUSD, EURUSD): ").strip().upper()

    training_start_date = "2000-01-01"
    training_end_date = current_date

    # Fetch and prepare the FX data for the specified currency pair and timeframe
    eur_usd_data = fetch_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)

    window_size = 5

    while True:
        print("\nPlease choose strategy you would like to use:")
        print("1 - Reversal")
        print("2 - Flag")

        choice_strategy = input("Enter your choice (1/2/3): ")

        if choice_strategy == '1':
            trend_identification_live(eur_usd_data, window_size)
            break
        elif choice_strategy == '2':
            identify_flag_patterns(eur_usd_data)
            break

    # Filter the EUR/USD data for the in-sample training period
    dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]

    # Drop rows where any of the data is missing
    dataset = dataset.dropna()

    backtest_set = get_backtest_data(dataset, timeframe_str, Pair)

    initial_balance=10000
    leverage=30
    transaction_cost=0.0002

    folder_name = os.getcwd()

    if 'JPY' in Pair:
        lot_size = 100  # Smaller lot size for pairs including JPY
    else:
        lot_size = 1000  # Default lot size for other pairs

    trader = ForexTradingSimulator(
        initial_balance,
        leverage,
        transaction_cost,
        lot_size,
    )
    backtest_set = backtest_set.reset_index()
    trader.simulate_trading(backtest_set, choice_strategy)

    trader.plot_balance_over_time(folder_name)

    trade_history_df = pd.DataFrame(trader.trade_history)

    trade_history_filename = os.path.join(folder_name, 'trade_history_backtest.csv')
    trade_history_df.to_csv(trade_history_filename)

    data_csv_filename = os.path.join(folder_name, 'data_backtest.csv')
    backtest_set.to_csv(data_csv_filename)

    highest_probability = perform_analysis(choice)

    if choice_strategy == '1':
        save_directory = f'bactest_reversal_patterns_{highest_probability:.2f}%_{Pair}_{timeframe_str}'
    elif choice_strategy == '2':
        save_directory = f'bactest_flag_patterns_{highest_probability:.2f}%_{Pair}_{timeframe_str}'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save all files except the specified ones
    exclude_files = ['things to do.txt', 'MLP.py', 'test_1.py', 'Chart.csv', 'Chart_1h.csv', 'Chart_Latest.csv', 'LSTM.py', 'RNN.py', 'XGboost.py', 'manual.py']

    for file in os.listdir('.'):
        if file not in exclude_files and os.path.isfile(file):
            shutil.move(file, os.path.join(save_directory, file))

def perform_analysis(choice):
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
        self.daily_worst_pnl = 0  # Initialize daily profit and loss
        self.daily_best_pnl = 0 


    def open_position(self, current_price, position_type, time):
        if not self.is_open_position:
            cost = self.transaction_cost
            self.current_balance -= cost  # Transaction cost
            self.entry_price = current_price
            self.position = position_type
            self.is_open_position = True

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
        else:
            self.current_balance += profit - self.transaction_cost
            self.is_open_position = False
            self.position = 'neutral'

        # Log the trade
        self.log_trade('close', current_price, time , profit)

    def log_trade(self, action, price, time, profit=None):
        self.trade_history.append({
            'time': time,
            'action': action,
            'position': self.position,
            'entry_price': self.entry_price if action == 'open' else None,
            'close_price': price if action == 'close' else None,
            'profit': profit,
            'balance': self.current_balance,
            'worst_daily_pnl': self.daily_worst_pnl,
            'best_daily_pnl': self.daily_best_pnl
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

    def simulate_trading(self, data, choice_strategy):
        for _, row in data.iterrows():
            self.update_current_pnl(row['high'], row['low'])

            current_date = row['time'].date()

            # Check for date change
            if self.last_processed_date is None or self.last_processed_date != current_date:
                # Reset daily P&L if it's a new day
                if self.last_processed_date is not None:  # Ensure it's not the first step
                    self.daily_worst_pnl = 0
                    self.daily_best_pnl = 0

            # Update last processed date
            self.last_processed_date = current_date

            self.daily_worst_pnl += self.worst_case_pnl

            self.daily_best_pnl += self.best_case_pnl

            # Check profit and close the trade if profit/loss threshold is crossed
            if self.is_open_position:
                # Determine closing logic based on position type
                if self.position == 'long':
                    # Close long position at the highest price if profit is good or at the lowest if loss is too high
                    if self.daily_best_pnl >= 100:
                        self.close_position(None, row['time'], profit=100)  # Close at high for maximum profit
                    elif self.daily_worst_pnl <= -50:
                        self.close_position(None, row['time'], profit=-50)  # Close at low to stop further loss
                elif self.position == 'short':
                    # Close short position at the lowest price if profit is good or at the highest if loss is too high
                    if self.daily_best_pnl >= 100:
                        self.close_position(None, row['time'],  profit=100)  # Close at low for maximum profit
                    elif self.daily_worst_pnl <= -50:
                        self.close_position(None, row['time'], profit=-50)  # Close at high to stop further loss

            if choice_strategy == '1':
                # Check if there is a change in predicted movement
                if row['confirm_breakout_retest'] == True and (not self.is_open_position or self.position == 'short'):
                    self.open_position(row['close'], 'long', row['time'])
                elif row['confirm_breakdown_retest'] == True and (not self.is_open_position or self.position == 'long'):
                    self.open_position(row['close'], 'short', row['time'])
                # Log the trade as hold if there is no change in position
                elif self.is_open_position:
                    self.log_trade('hold', row['close'], row['time'])
            elif choice_strategy == '2':
                # Check if there is a change in predicted movement
                if row['bullish_breakout'] == True and (not self.is_open_position or self.position == 'short'):
                    self.open_position(row['close'], 'long', row['time'])
                elif row['bearish_breakdown'] == True and (not self.is_open_position or self.position == 'long'):
                    self.open_position(row['close'], 'short', row['time'])
                # Log the trade as hold if there is no change in position
                elif self.is_open_position:
                    self.log_trade('hold', row['close'], row['time'])

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

def analyze_pair_data(df):
    """ Analyze data for a single currency pair and return key metrics. """
    # Make a copy of the DataFrame to avoid SettingWithCopyWarning when modifying data
    df_copy = df.copy()

    # Ensure 'time' column is in datetime format
    df_copy['time'] = pd.to_datetime(df_copy['time'])

    # Calculate daily profit
    daily_profit = df_copy.groupby('time')['profit'].sum()

    worst_daily_pnl = df_copy.groupby('time')['worst_daily_pnl'].sum()

    best_daily_pnl = df_copy.groupby('time')['best_daily_pnl'].sum()

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

    print(f"Probability of passing: {probability_of_passing:.2f}%")

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

def extract_info_from_folder_name(folder_name):
    # Split the folder name by underscores
    parts = folder_name.split('_')
    
    # Extract currency pair and timeframe assuming the format includes them
    if len(parts) >= 4:
        pair = parts[3]
        timeframe_str = parts[4]
        return pair, timeframe_str
    else:
        raise ValueError("Folder name format is incorrect or missing essential information.")

def fetch_forex_agent_folders():
    """ Fetch all folder names that start with 'agent_forex' from the current directory. """
    # Get the current working directory
    base_dir = os.getcwd()

    # Ensure the base directory exists and is a directory
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"The specified base directory does not exist: {base_dir}")
    if not os.path.isdir(base_dir):
        raise NotADirectoryError(f"The specified path is not a directory: {base_dir}")

    # List all items in the base directory
    dirs = os.listdir(base_dir)

    # Filter out directories that start with 'agent_forex'
    forex_dirs = [d for d in dirs if d.startswith('bactest_reversal') and os.path.isdir(os.path.join(base_dir, d))]

    return forex_dirs

def fetch_and_aggregate_results():
    folders = fetch_forex_agent_folders()
    combined_df = pd.DataFrame()
    
    for folder in folders:
        file_path = os.path.join(folder, 'trade_history_backtest.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            Pair, timeframe_str = extract_info_from_folder_name(folder)
            df['Pair'] = Pair
            df['Timeframe'] = timeframe_str  # Assuming folder names include this info
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

def plot_multi_forex_data(data):
    # Ensure the 'time' column is a datetime type
    data['time'] = pd.to_datetime(data['time'])

    # Plotting
    plt.figure(figsize=(14, 7))
    
    # Group by 'Pair' and 'Timeframe' to plot each series separately
    grouped = data.groupby(['Pair', 'Timeframe'])
    for (pair, timeframe), group in grouped:
        plt.plot(group['time'], group['balance'], label=f'{pair} {timeframe}')

    plt.title('Forex Trading Balance Over Time')
    plt.xlabel('Time')
    plt.ylabel('Balance')
    plt.legend()
    plt.grid(True)
    plt.savefig('multi_backtest.png')

def perform_combined_analysis():
    combined_df = fetch_and_aggregate_results()
    probability_rankings = compute_probabilities(combined_df)
    # Get the current working directory
    current_directory = os.getcwd()
    # List all subdirectories in the current directory
    all_subdirs = [os.path.join(current_directory, d) for d in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, d))]

    # Find the most recent directory
    most_recent_dir = max(all_subdirs, key=os.path.getmtime)

    probability_rankings.to_csv('forex_pair_probability_rankings.csv', index=False)

    plot_multi_forex_data(combined_df)

    print("Analysis complete. Results saved.")

def fetch_live_data(symbol, timeframe_str, number_of_bars):
    timezone = pytz.timezone("Europe/Berlin")

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

    timeframe = timeframes.get(timeframe_str, None)
    if timeframe is None:
        print(f"Invalid timeframe: {timeframe_str}")
        return None

    # Get the last 10 bars of the given symbol and timeframe
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, number_of_bars)
    if rates is None:
        print("No rates retrieved, error code =", mt5.last_error())
        return None

    rates_frame = pd.DataFrame(rates)
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s', utc=True)
    rates_frame.set_index('time', inplace=True)
    rates_frame.index = rates_frame.index.tz_convert(timezone)

    return rates_frame

def countdown(t):
    print("\n")  # Ensure the countdown starts on a new line
    while t:
        mins, secs = divmod(t, 60)
        time_format = '{:02d}:{:02d}'.format(mins, secs)
        print(f"The next candle will form in: {time_format}", end='\r')
        time.sleep(1)
        t -= 1
    print("Next candle is now live!           ")

def calculate_time_to_next_candle(latest_time_index, timeframe):
    # Convert latest_time_index to datetime if it's a string
    if isinstance(latest_time_index, str):
        latest_time_index = parse(latest_time_index)

    timeframe_mapping = {
        '1H': 1,
        '4H': 4,
        '1D': 24
    }
    
    hours = timeframe_mapping.get(timeframe, 4)  # Default to 4 hours if not specified
    next_candle_time = latest_time_index + timedelta(hours=hours, minutes=10)
    now = datetime.now() 

    # Format to just include the hour and minute
    next_candle_time_str = next_candle_time.strftime("%H:%M")

    # Access the hour and minute
    current_hour = now.hour
    current_minute = now.minute

    time_string = f"{current_hour:02d}:{current_minute:02d}"
    
    # Calculate difference in seconds
    delta_seconds = (next_candle_time - now).total_seconds()
    return max(0, int(delta_seconds)), next_candle_time_str, time_string  # Ensure non-negative

def initialize_mt5():
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

def get_latest_data():
    # Retrieve and store the current date
    current_date = str(datetime.now().date())

    # Hardcoded start date for strategy evaluation
    strategy_start_date_all = "1971-01-04"
    # Use the current date as the end date for strategy evaluation
    strategy_end_date_all = current_date

    # Convert string representation of dates to datetime objects for further processing
    start_date_all = datetime.strptime(strategy_start_date_all, "%Y-%m-%d")
    end_date_all = datetime.strptime(strategy_end_date_all, "%Y-%m-%d")

    # Prompt the user to decide if they want real-time data updates and store the boolean result
    #enable_real_time = input("Do you want to enable real-time data updates? (yes/no): ").lower().strip() == 'yes'

    # Prompt the user for the desired timeframe for analysis and standardize the input
    timeframe_str = input("Enter the currency pair (e.g., Daily, 1H): ").strip().upper()
    # Prompt the user for the currency pair they're interested in and standardize the input
    Pair = input("Enter the currency pair (e.g., GBPUSD, EURUSD): ").strip().upper()
    while True:
        training_start_date = "2000-01-01"
        training_end_date = current_date

        current_date_time  = datetime.now()

        # Fetch and prepare the FX data for the specified currency pair and timeframe
        eur_usd_data = fetch_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)

        # Filter the EUR/USD data for the in-sample training period
        dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]

        latest_dataset_entry = dataset.index[-1]

        # Calculate the difference in time
        time_difference = current_date_time - latest_dataset_entry

        # Convert the time difference to hours
        hours_difference = int(time_difference.total_seconds() / 3600) + 3

        window_size = 5

        initialize_mt5()
        data = fetch_live_data(Pair, timeframe_str, hours_difference)

        if data is not None:
            data = data[:-1]  # Slices off the last row
            data.index = data.index - pd.Timedelta(hours=2)  # Adjust the time by subtracting 2 hours
            data.index = data.index.tz_localize(None)  # Remove the timezone information
            
            # Calculate indicators for the updated dataset
            # Concatenate new data
            updated_dataset = pd.concat([dataset, data]).drop_duplicates()

            print(updated_dataset)
            
            updated_dataset = trend_identification_live(updated_dataset, window_size)
            
            # Get the latest time index
            latest_time_index = updated_dataset.index[-1].strftime('%Y-%m-%d %H:%M:%S')

            updated_dataset.to_csv(f'Full_data_{latest_time_index}.csv', index=True)

            # Calculate time to sleep until the next candle plus 5 minutes
            sleep_time, next_candle_time_str, time_string = calculate_time_to_next_candle(latest_time_index, timeframe_str)
            print(f"Next candle will form at: {next_candle_time_str} and the current time is {time_string}")
            countdown(sleep_time)  # Display countdown

def main_menu():
    while True:
        print("\nMain Menu:")
        print("1 - Forex Trading Strategy (single)")
        print("2 - Forex Trading Strategy (multiple)")
        print("3 - Analysis (combined)")
        print("4 - Live Trading")

        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            manual_trading(choice)
            break
        elif choice == '2':
            main_training_loop_multiple_pairs()
            break
        elif choice == '3':
            perform_combined_analysis()
            break
        elif choice == '4':
            get_latest_data()
            break

if __name__ == "__main__":
    main_menu()