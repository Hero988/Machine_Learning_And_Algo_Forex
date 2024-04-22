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
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import shutil
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
import pickle

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

def calculate_indicators(data, choice):
    bollinger_length=12
    bollinger_std_dev=1.5
    sma_trend_length=50
    window=9
    print("Doing calculate_indicators function")
    # Calculate the 50-period simple moving average of the 'close' price
    data['SMA_50'] = ta.sma(data['close'], length=50)
    # Calculate the 200-period simple moving average of the 'close' price
    data['SMA_200'] = ta.sma(data['close'], length=200)
    
    # Calculate the 50-period exponential moving average of the 'close' price
    data['EMA_50'] = ta.ema(data['close'], length=50)
    # Calculate the 200-period exponential moving average of the 'close' price
    data['EMA_200'] = ta.ema(data['close'], length=200)

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

    # Shift the 'close' column up to get the next 'close' price in the future
    data['close_price_next'] = data['close'].shift(-1)

    data['close_price_previous'] = data['close'].shift(1)

    # Calculate the 'target' based on the future price movement
    data['target'] = np.where(data['close_price_next'] > data['close'], 1,
                                       np.where(data['close_price_next'] < data['close'], -1, 0))

    # Drop the 'close_price_next' column if you no longer need it
    data.drop(columns=['close_price_next'], inplace=True)

    data['close_price_percentage_change'] = data['close'].pct_change().fillna(0) * 100
    data['previous_close_price_percentage_change'] = data['close_price_percentage_change'].shift(1)

    # Adding lag features for 'target'
    number_of_lags = 3
    for lag in range(1, number_of_lags + 1):
        data[f'Actual Movement Lag_{lag}'] = data['target'].shift(lag)

    if choice == '1':
        # Remove the last row of the DataFrame
        data = data.drop(data.tail(1).index)

    horizons = [2, 5, 60, 250, 1000]

    new_predictors = []

    for horizon in horizons:
        rolling_average = data.rolling(horizon).mean()

        ratio_column = f'Close_Ratio_{horizon}'

        data[ratio_column] = data['close'] / rolling_average['close']

        trend_column = f'Trend_{horizon}'

        data[trend_column] = data.shift(1).rolling(horizon).sum()['target']

        new_predictors += [ratio_column, trend_column]

    # Return the data with added indicators
    return data

def calculate_movement(data):
    # Ensure proper column names
    if 'DateTime' in data.columns:
        data.rename(columns={'DateTime': 'time'}, inplace=True)
    if 'Close' in data.columns:
        data.rename(columns={'Close': 'close'}, inplace=True)

    # Calculate future price movement
    data['close_price_next'] = data['close'].shift(-1)
    data['target'] = np.where(data['close_price_next'] > data['close'], 1,
                              np.where(data['close_price_next'] < data['close'], -1, 0))
    data.drop(columns=['close_price_next'], inplace=True)

    # Adding rolling features, ratios, and more complex indicators
    horizons = [2, 5, 60, 250, 1000]
    for horizon in horizons:
        rolling_mean = data['close'].rolling(window=horizon).mean()
        data[f'Close_Ratio_{horizon}'] = data['close'] / rolling_mean
        data[f'Trend_{horizon}'] = data['target'].shift(1).rolling(window=horizon).sum()

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
    return final_training_set, validation_set

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

def preprocess_data_no_scale(data, columns_to_drop):
    
    # Assuming 'time' and 'target' are non-numeric columns we want to drop for training
    data_numeric = data.drop(columns=columns_to_drop).values

    data_non_numeric = data.drop(columns=columns_to_drop)
    
    # Prepare labels if 'target' exists in the data
    if 'target' in data.columns:
        labels = np.where(data['target'].values == -1, 0, 1)
        return data_numeric, labels, data_non_numeric
    
    return data_numeric, data_non_numeric

def find_optimal_threshold_accuracy(y_true, y_probs):
    # Define a range of possible thresholds from 0 to 1 with a small step
    thresholds = np.linspace(0, 1, num=100)
    # Initialize best accuracy and threshold
    best_accuracy = 0
    best_threshold = 0.5
    
    # Iterate over thresholds to find the best one
    for thresh in thresholds:
        # Predict labels based on the threshold
        y_pred = (y_probs >= thresh).astype(int)
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        # If the current accuracy is better, update the best accuracy and threshold
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh

    print(f"Optimal threshold based on accuracy is: {best_threshold:.4f}, with an accuracy of: {best_accuracy:.4f}")
    return best_threshold

def save_optimal_threshold_json(threshold):
    data = {'optimal_threshold': threshold}
    with open('optimal_threshold.json', 'w') as f:
        json.dump(data, f)

def load_optimal_threshold_json(model_path):
    # Derive the directory from the model path
    directory = os.path.dirname(model_path)
    
    # Assuming the optimal threshold is stored in a JSON file in the same directory
    json_path = os.path.join(directory, 'optimal_threshold.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No optimal threshold JSON file found at {json_path}")
    
    # Load the optimal threshold from the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data['optimal_threshold']

def evaluate(Pair='N/A', timeframe_str='N/A', selector=None):
    # Your existing code
    testing_files = glob.glob('testing*.csv')
    for file in testing_files:
        # Reading the CSV with the default settings to preserve the data as is
        testing_set = pd.read_csv(file)
        # Set the 'time' column as the DataFrame index
        testing_set.set_index('time', inplace=True)

    testing_set = testing_set.reset_index()

    columns_to_drop = ['time', 'target', 'spread', 'low', 'close', 'high', 'open']

    X_test, y_test, X_test_non_numeric = preprocess_data_no_scale(testing_set, columns_to_drop)

    # Apply the same feature selector to the test data
    X_test_selected = selector.transform(X_test)

    # Get the mask of selected features
    selected_features_mask = selector.get_support()

    # Apply this mask to the columns of the original data (before dropping any columns)
    selected_feature_names = X_test_non_numeric.columns[selected_features_mask].tolist()

    # Now 'selected_feature_names' contains the names of the features selected by the model
    # and 'X_test_selected' contains the actual data for those selected features

    dtest = xgb.DMatrix(X_test_selected, label=y_test, feature_names=selected_feature_names)

    # Create a new model object
    bst_loaded = xgb.Booster()

    # Load the model from the file
    bst_loaded.load_model('xgboost_model.json')

    # Make predictions (probabilities)
    preds = bst_loaded.predict(dtest)

    # Find the optimal threshold
    optimal_threshold = find_optimal_threshold_accuracy(y_test, preds)

    save_optimal_threshold_json(optimal_threshold)

    # Convert probabilities to final predictions based on the optimal threshold
    predictions = (preds >= optimal_threshold).astype(int)
    adjusted_predictions = np.where(predictions == 0, -1, 1)

    # Calculate accuracy with the optimal threshold
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    results_df = testing_set[['time', 'target', 'close']].reset_index(drop=True)
    results_df['Predictions'] = preds
    results_df['Predicted'] = adjusted_predictions

    results_df.to_csv('predicted_classification_with_Actual_Movement_and_close_MLP.csv', index=False)

    save_directory = f'agent_forex_{accuracy*100:.2f}%_{Pair}_{timeframe_str}'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Calculate metrics
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    # Accuracy: (TP + TN) / (TP + TN + FP + FN)
    accuracy = (TP + TN) / np.sum(conf_matrix)

    # Precision: TP / (TP + FP)
    precision = TP / (TP + FP)

    # Recall: TP / (TP + FN)
    recall = TP / (TP + FN)

    # Specificity: TN / (TN + FP)
    specificity = TN / (TN + FP)

    fig, ax = plt.subplots(figsize=(8, 6))  # Larger figure size
    cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Set the ticks first before setting labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])

    # Annotate each cell with the numeric value
    for (i, j), val in np.ndenumerate(conf_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', color='black')

    # Accuracy (How often the model is correct)
    # Precision (When it predicts yes, how often is it correct?)
    # Recall or Sensitivity (How often it correctly predicts yes when it is actually yes)
    # Specificity (How often it correctly predicts no when it is actually no)

    # Include metrics in the title, properly formatted as percentages
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix\n'
            f'Accuracy: {accuracy*100:.2f}%, Precision: {precision*100:.2f}%, Recall or Sensitivity: {recall*100:.2f}%, Specificity: {specificity*100:.2f}%')
    plt.savefig('confusion_matrix.png')  # Save to the file system of this environment
    # Save all files except the specified ones
    exclude_files = ['things to do.txt', 'MLP.py', 'test_1.py', 'Chart.csv', 'Chart_1h.csv', 'Chart_Latest.csv', 'LSTM.py', 'RNN.py', 'XGboost.py']

    for file in os.listdir('.'):
        if file not in exclude_files and os.path.isfile(file):
            shutil.move(file, os.path.join(save_directory, file))

    print(f'The accuracy of the model predictions is: {accuracy * 100:.2f}%')
    print(f'Confusion Matrix:\n{conf_matrix}')
    return results_df, accuracy, conf_matrix

def get_model_path(folder_name):
    # Construct the path to the model file within the specified folder
    model_path = os.path.join(folder_name, 'xgboost_model.json')

    print(model_path)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found in directory {folder_name}")
    
    return model_path

def find_recent_forex_agent_dir(pair='EURUSD', timeframe='DAILY', base_dir=None):
    # Use the current working directory if no base_dir is provided
    if base_dir is None:
        base_dir = os.getcwd()

    # Ensure the base_dir exists and is a directory
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"The specified base directory does not exist: {base_dir}")
    if not os.path.isdir(base_dir):
        raise NotADirectoryError(f"The specified path is not a directory: {base_dir}")

    # List all items in the base directory
    dirs = os.listdir(base_dir)

    # Filter out directories that contain the specific pair and timeframe, and start with 'agent_forex'
    forex_dirs = [d for d in dirs if d.startswith('agent_forex') and pair in d and timeframe in d and os.path.isdir(os.path.join(base_dir, d))]

    # Sort directories by name, assuming names include sortable dates or increment numbers
    forex_dirs.sort(reverse=True)

    if forex_dirs:
        return forex_dirs[0]  # Return the most recent directory
    else:
        return None  # Return None if no directory found

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

def predict_next_forex(choice):
    # Retrieve and store the current date
    current_date = str(datetime.now().date())

    # Hardcoded start date for strategy evaluation
    strategy_start_date_all = "1971-01-04"
    # Use the current date as the end date for strategy evaluation
    strategy_end_date_all = current_date

    # Convert string representation of dates to datetime objects for further processing
    start_date_all = datetime.strptime(strategy_start_date_all, "%Y-%m-%d")
    end_date_all = datetime.strptime(strategy_end_date_all, "%Y-%m-%d")

    timeframe = input("Enter the currency pair (e.g., Daily, 1H): ").strip().upper()

    pair = input("Enter the currency pair (e.g., GBPUSD, EURUSD): ").strip().upper()

    folder_name = find_recent_forex_agent_dir(pair, timeframe)

    Pair, timeframe_str = extract_info_from_folder_name(folder_name)

    training_start_date = "2000-01-01"
    training_end_date = current_date

    # Fetch and prepare the FX data for the specified currency pair and timeframe
    eur_usd_data = fetch_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)

    # Apply technical indicators to the data using the 'calculate_indicators' function
    eur_usd_data = calculate_indicators(eur_usd_data, choice) 

    #calculate_movement(eur_usd_data)

    # Drop rows where any of the data is missing
    eur_usd_data = eur_usd_data.dropna()

    eur_usd_data = eur_usd_data.iloc[:-1]  # Drops the last row

    eur_usd_data = eur_usd_data.reset_index()

    latest_row = eur_usd_data.tail(1)

    print(latest_row)

    columns_to_drop = ['time', 'target', 'spread', 'low', 'close', 'high', 'open']

    # Since this is a prediction for the future, we assume no 'target' available
    # Prepare data (you need to ensure your preprocessing function can handle single row)
    X, _, X_non_numeric = preprocess_data_no_scale(latest_row, columns_to_drop)

    # Create the full file path
    file_path = os.path.join(folder_name, 'selector.pkl')

    # Load the selector from the specified folder
    with open(file_path, 'rb') as file:
        loaded_selector = pickle.load(file)

    # Apply the same feature selector to the test data
    X_selected = loaded_selector.transform(X)

    # Get the mask of selected features
    selected_features_mask = loaded_selector.get_support()

    # Apply this mask to the columns of the original data (before dropping any columns)
    selected_feature_names = X_non_numeric.columns[selected_features_mask].tolist()

    # Convert to DMatrix
    dtest_single = xgb.DMatrix(X_selected, _, feature_names=selected_feature_names)

    # Create a new model object
    bst_loaded = xgb.Booster()

    model_path = get_model_path(folder_name)

    # Load the model from the file
    bst_loaded.load_model(model_path)

    # Make predictions (probabilities)
    pred = bst_loaded.predict(dtest_single)

    # Load the optimal threshold (make sure you have this value saved from your training phase)
    optimal_threshold = load_optimal_threshold_json(model_path)  # Implement this function to read the saved threshold

    # Use the threshold to determine the predicted class
    predicted_class = (pred >= optimal_threshold)

    print(f'Output: {pred}, Optimal Threshold: {optimal_threshold}')

    # Convert the tensor to a numpy array for easier handling if necessary
    adjusted_prediction = np.where(predicted_class == 0, -1, 1)

    print(f'Predicted Movement is {adjusted_prediction}')

    # Extract necessary information from the latest row
    time = latest_row['time'].iloc[0]
    close = latest_row['close'].iloc[0]
    predicted = adjusted_prediction

    # Prepare the data dictionary
    data = {
        'Time': [time],
        'Pair': [pair],
        'Close': [close],
        'Predicted': [predicted]
    }

    # Create DataFrame
    result_df = pd.DataFrame(data)

    # Define the path for the new CSV file within the specified folder
    csv_file_path = os.path.join(folder_name, f'predictions_{timeframe_str}.csv')

    # Check if the file already exists
    if not os.path.exists(csv_file_path):
        result_df.to_csv(csv_file_path, index=False)
        print(f"File created: {csv_file_path}")
    else:
        result_df.to_csv(csv_file_path, mode='a', header=False, index=False)
        print(f"Data appended to: {csv_file_path}")

    # Optionally, return the prediction or handle it as needed
    return adjusted_prediction

def try_parse_datetime(input_str):
    try:
        # Try parsing as datetime first
        return datetime.strptime(input_str, "%Y-%m-%d %H:%M:%S"), True
    except ValueError:
        try:
            # If it fails, try parsing as date
            return datetime.strptime(input_str, "%Y-%m-%d"), False
        except ValueError:
            return None, False

def predict_specific(choice):
    # Retrieve and store the current date
    current_date = str(datetime.now().date())

    # Hardcoded start date for strategy evaluation
    strategy_start_date_all = "1971-01-04"
    # Use the current date as the end date for strategy evaluation
    strategy_end_date_all = current_date

    # Convert string representation of dates to datetime objects for further processing
    start_date_all = datetime.strptime(strategy_start_date_all, "%Y-%m-%d")
    end_date_all = datetime.strptime(strategy_end_date_all, "%Y-%m-%d")

    timeframe = input("Enter the currency pair (e.g., Daily, 1H): ").strip().upper()

    pair = input("Enter the currency pair (e.g., GBPUSD, EURUSD): ").strip().upper()

    folder_name = find_recent_forex_agent_dir(pair, timeframe)

    Pair, timeframe_str = extract_info_from_folder_name(folder_name)

    training_start_date = "2000-01-01"
    training_end_date = current_date

    # Fetch and prepare the FX data for the specified currency pair and timeframe
    eur_usd_data = fetch_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)

    # Apply technical indicators to the data using the 'calculate_indicators' function
    #eur_usd_data = calculate_indicators(eur_usd_data, choice) 

    calculate_movement(eur_usd_data)

    # Filter the EUR/USD data for the in-sample training period
    dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]

    # Drop rows where any of the data is missing
    dataset = dataset.dropna()

    dataset = dataset.drop(dataset.tail(1).index)  # Assuming the last row is always dropped

    # Requesting a specific date from the user
    user_date_str = input("Enter the date you want to predict for (YYYY-MM-DD): ")

    user_date, is_datetime = try_parse_datetime(user_date_str)

    if user_date is None:
        print("Invalid format. Please enter the date in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format.")
    else:
        if user_date in dataset.index:
            # Extract as DataFrame instead of Series
            specific_row = dataset.loc[[user_date]]
            specific_row = specific_row.reset_index()
            print("Data on selected date:")
            print(specific_row)
        else:
            print("The specified date or datetime is not available in the dataset. Please choose another within the range.")

    columns_to_drop = ['time', 'target', 'spread', 'low', 'close', 'high', 'open']

    # Since this is a prediction for the future, we assume no 'target' available
    # Prepare data (you need to ensure your preprocessing function can handle single row)
    X, _, X_non_numeric = preprocess_data_no_scale(specific_row, columns_to_drop)

    # Create the full file path
    file_path = os.path.join(folder_name, 'selector.pkl')

    # Load the selector from the specified folder
    with open(file_path, 'rb') as file:
        loaded_selector = pickle.load(file)

    # Apply the same feature selector to the test data
    X_selected = loaded_selector.transform(X)

    # Get the mask of selected features
    selected_features_mask = loaded_selector.get_support()

    # Apply this mask to the columns of the original data (before dropping any columns)
    selected_feature_names = X_non_numeric.columns[selected_features_mask].tolist()

    # Convert to DMatrix
    dtest_single = xgb.DMatrix(X_selected, _, feature_names=selected_feature_names)

    # Create a new model object
    bst_loaded = xgb.Booster()

    model_path = get_model_path(folder_name)

    # Load the model from the file
    bst_loaded.load_model(model_path)

    # Make predictions (probabilities)
    pred = bst_loaded.predict(dtest_single)

    # Load the optimal threshold (make sure you have this value saved from your training phase)
    optimal_threshold = load_optimal_threshold_json(model_path)  # Implement this function to read the saved threshold

    # Use the threshold to determine the predicted class
    predicted_class = (pred >= optimal_threshold)

    print(f'Output: {pred}, Optimal Threshold: {optimal_threshold}')

    # Convert the tensor to a numpy array for easier handling if necessary
    adjusted_prediction = np.where(predicted_class == 0, -1, 1)

    print(f'Predicted Movement is {adjusted_prediction}')

def training_forex_multiple(choice, Pair, timeframe_str):
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

    # Fetch and prepare the FX data for the specified currency pair and timeframe
    eur_usd_data = fetch_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)

    # Apply technical indicators to the data using the 'calculate_indicators' function
    #eur_usd_data = calculate_indicators(eur_usd_data, choice)

    calculate_movement(eur_usd_data)

    # Filter the EUR/USD data for the in-sample training period
    dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]

    # Drop rows where any of the data is missing
    dataset = dataset.dropna()

    training_set, testing_set = split_and_save_dataset(dataset, timeframe_str, Pair)

    training_set = training_set.reset_index()
    testing_set = testing_set.reset_index()

    columns_to_drop = ['time', 'target', 'spread', 'low', 'close', 'high', 'open']

    X_train, y_train, _ = preprocess_data_no_scale(training_set, columns_to_drop)
    X_val, y_val, _ = preprocess_data_no_scale(testing_set, columns_to_drop)

    feature_coloumns = testing_set.drop(columns=columns_to_drop).columns.tolist()
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_coloumns)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_coloumns)
    
    # Define the model
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', early_stopping_rounds=50)

    # Define the grid of parameters to search
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 2, 5, 10],
        'learning_rate': [0.01, 0.1, 0.3],  # 'eta' in xgb.train corresponds to 'learning_rate' in XGBClassifier
    }

    # Setup the grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)

    # Convert DMatrix to NumPy arrays (if using DMatrix objects)
    X_train_numpy, y_train_numpy = dtrain.get_data(), dtrain.get_label()
    X_val_numpy, y_val_numpy = dval.get_data(), dval.get_label()

    # Fit grid search to the data
    best_model = grid_search.fit(X_train_numpy, y_train_numpy, eval_set=[(X_val_numpy, y_val_numpy)], verbose=True)

    # Retrain using the best parameters found
    bst = xgb.XGBClassifier(**best_model.best_params_, objective='binary:logistic', eval_metric='logloss')
    bst.fit(X_train_numpy, y_train_numpy, eval_set=[(X_val_numpy, y_val_numpy)], verbose=True)

    # Save the model
    bst.save_model('xgboost_model.json')

    # Use SelectFromModel to select features based on importance
    selector = SelectFromModel(estimator=bst, threshold='mean', prefit=True)

    with open('selector.pkl', 'wb') as file:
        pickle.dump(selector, file)

    # Get feature importance
    importance = bst.feature_importances_

    # Convert importance scores and feature names into a DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_coloumns,
        'Importance': importance
    })

    # Sort the DataFrame by importance scores
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plotting feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()  # Invert y axis to have the most important at the top

    # Optionally, you can also save this plot to a file
    plt.savefig('Feature_Importance.png')  # Save to the file system of this environment

    evaluate(Pair, timeframe_str, selector)

    backtest_trades_with_dataframe(choice, timeframe_str, Pair)
    
def fetch_forex_pairs():
    account_number = 530064788
    password = 'fe5@6YV*'
    server_name = 'FTMO-Server3'

    # Initialize MT5 connection
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return None

    # Attempt to log in with the given account number, password, and server
    authorized = mt5.login(account_number, password=password, server=server_name)
    if not authorized:
        print("login failed, error code =", mt5.last_error())
        mt5.shutdown()
        return None
    else:
        print("Connected to MetaTrader 5")

    # Fetch all symbols
    symbols = mt5.symbols_get()

    # Define a function to check if the symbol name matches the typical Forex pair format
    def is_forex_pair(name):
        return len(name) == 6 and name.isalpha()

    # Filter to get only forex pairs, based on the new criteria
    forex_pairs = [symbol.name for symbol in symbols if is_forex_pair(symbol.name)]

    # Shutdown MT5 connection
    mt5.shutdown()

    return forex_pairs

def main_training_loop_multiple_pairs():
    """
    forex_pairs = fetch_forex_pairs()
    if forex_pairs is None:
        print("Failed to fetch forex pairs or no forex pairs available.")
        return
    """

    forex_pairs = ['EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY', 'USDCAD', 'AUDUSD', 'AUDNZD', 'AUDCAD', 'AUDCHF', 'AUDJPY', 'GBPCAD', 'NZDUSD', 'CHFJPY', 'EURGBP','EURAUD', 'EURCHF', 'EURJPY', 'EURNZD', 'EURCAD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'CADCHF', 'CADJPY', 'GBPAUD', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY']
    
    timeframe = input("Enter the currency pair (e.g., Daily, 1H): ").strip().upper()

    choice = '1'  # Assuming '1' is for Forex, as per your original function setup

    for pair in forex_pairs:
        print(f"Training for {pair} on {timeframe}")
        training_forex_multiple(choice, pair, timeframe)

def training(choice):
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

    # Apply technical indicators to the data using the 'calculate_indicators' function
    #eur_usd_data = calculate_indicators(eur_usd_data, choice)

    calculate_movement(eur_usd_data)

    # Filter the EUR/USD data for the in-sample training period
    dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]

    # Drop rows where any of the data is missing
    dataset = dataset.dropna()

    training_set, testing_set = split_and_save_dataset(dataset, timeframe_str, Pair)

    training_set = training_set.reset_index()
    testing_set = testing_set.reset_index()

    columns_to_drop = ['time', 'target', 'spread', 'low', 'close', 'high', 'open']

    X_train, y_train, _ = preprocess_data_no_scale(training_set, columns_to_drop)
    X_val, y_val, _ = preprocess_data_no_scale(testing_set, columns_to_drop)

    feature_coloumns = testing_set.drop(columns=columns_to_drop).columns.tolist()
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_coloumns)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_coloumns)
    
    # Define the model
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', early_stopping_rounds=50)

    # Define the grid of parameters to search
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 2, 5, 10],
        'learning_rate': [0.01, 0.1, 0.3],  # 'eta' in xgb.train corresponds to 'learning_rate' in XGBClassifier
    }

    # Setup the grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)

    # Convert DMatrix to NumPy arrays (if using DMatrix objects)
    X_train_numpy, y_train_numpy = dtrain.get_data(), dtrain.get_label()
    X_val_numpy, y_val_numpy = dval.get_data(), dval.get_label()

    # Fit grid search to the data
    best_model = grid_search.fit(X_train_numpy, y_train_numpy, eval_set=[(X_val_numpy, y_val_numpy)], verbose=True)

    # Retrain using the best parameters found
    bst = xgb.XGBClassifier(**best_model.best_params_, objective='binary:logistic', eval_metric='logloss')
    bst.fit(X_train_numpy, y_train_numpy, eval_set=[(X_val_numpy, y_val_numpy)], verbose=True)

    # Save the model
    bst.save_model('xgboost_model.json')

    # Get feature importance
    importance = bst.feature_importances_

    # Use SelectFromModel to select features based on importance
    selector = SelectFromModel(estimator=bst, threshold='mean', prefit=True)

    with open('selector.pkl', 'wb') as file:
        pickle.dump(selector, file)

    """
    with open('selector.pkl', 'rb') as file:
    loaded_selector = pickle.load(file)
    """

    # Convert importance scores and feature names into a DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_coloumns,
        'Importance': importance
    })

    # Sort the DataFrame by importance scores
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plotting feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()  # Invert y axis to have the most important at the top

    # Optionally, you can also save this plot to a file
    plt.savefig('Feature_Importance.png')  # Save to the file system of this environment

    evaluate(Pair, timeframe_str, selector)

    backtest_trades_with_dataframe(choice, timeframe_str, Pair)
    
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

    def simulate_trading(self, data):
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
                    if self.best_case_pnl >= 100:
                        self.close_position(None, row['time'], profit=100)  # Close at high for maximum profit
                    elif self.worst_case_pnl <= -100:
                        self.close_position(None, row['time'], profit=-100)  # Close at low to stop further loss
                elif self.position == 'short':
                    # Close short position at the lowest price if profit is good or at the highest if loss is too high
                    if self.best_case_pnl >= 100:
                        self.close_position(None, row['time'],  profit=100)  # Close at low for maximum profit
                    elif self.worst_case_pnl <= -100:
                        self.close_position(None, row['time'], profit=-100)  # Close at high to stop further loss

            # Check if there is a change in predicted movement
            if row['Predicted'] == 1 and (not self.is_open_position or self.position == 'short'):
                # Close the current short position before opening a new long position
                if self.is_open_position: 
                    self.close_position(row['close'], row['time'])
                self.open_position(row['close'], 'long', row['time'])
            elif row['Predicted'] == -1 and (not self.is_open_position or self.position == 'long'):
                # Close the current long position before opening a new short position
                if self.is_open_position: 
                    self.close_position(row['close'], row['time'])
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

def backtest_trades_with_dataframe(choice, timeframe_new=None, pair_new=None):

    if choice == '7':
        timeframe = input("Enter the currency pair (e.g., Daily, 1H): ").strip().upper()

        pair = input("Enter the currency pair (e.g., GBPUSD, EURUSD): ").strip().upper()

        folder_name = find_recent_forex_agent_dir(pair, timeframe)
    elif choice == '1' or '2' or '6':
        folder_name = find_recent_forex_agent_dir(pair_new, timeframe_new)
        pair = pair_new
        timeframe = timeframe_new

    # Construct the path to the model file within the specified folder
    predicted_movement_path = os.path.join(folder_name, 'predicted_classification_with_Actual_Movement_and_close_MLP.csv')

    predictions_df = pd.read_csv(predicted_movement_path)

    main_df = os.path.join(folder_name, f'testing_{pair}_{timeframe}_data.csv')

    prices_df = pd.read_csv(main_df)  

    data = predictions_df.merge(prices_df, left_index=True, right_index=True, how='left')

    data.drop(columns=['target_y'], inplace=True)
    data.rename(columns={'target_x': 'target'}, inplace=True)

    data.drop(columns=['time_x'], inplace=True)
    data.rename(columns={'time_y': 'time'}, inplace=True)

    data.rename(columns={'close_x': 'close'}, inplace=True)

    # Convert time column to datetime if not already
    data['time'] = pd.to_datetime(data['time'])
    data = data.sort_values('time')  # Ensure data is sorted by time

    data['close_price_next'] = data['close'].shift(-1)

    # Define the columns to keep
    columns_to_keep = ['time', 'target', 'Predictions', 'Predicted', 'close', 'close_price_next', 'high', 'low']

    # Select these columns in the DataFrame
    data = data[columns_to_keep]

    initial_balance=10000
    leverage=30
    transaction_cost=0.0002

    if 'JPY' in pair:
        lot_size = 100  # Smaller lot size for pairs including JPY
    else:
        lot_size = 10000  # Default lot size for other pairs

    trader = ForexTradingSimulator(
        initial_balance,
        leverage,
        transaction_cost,
        lot_size,
    )
    trader.simulate_trading(data)

    trader.plot_balance_over_time(folder_name)

    trade_history_df = pd.DataFrame(trader.trade_history)

    trade_history_filename = os.path.join(folder_name, 'trade_history_backtest.csv')
    trade_history_df.to_csv(trade_history_filename)

    data_csv_filename = os.path.join(folder_name, 'data_backtest.csv')
    data.to_csv(data_csv_filename)

    perform_analysis(choice)

def train_magnitude(choice):
    timeframe = input("Enter the currency pair (e.g., Daily, 1H): ").strip().upper()

    pair = input("Enter the currency pair (e.g., GBPUSD, EURUSD): ").strip().upper()

    folder_name = find_recent_forex_agent_dir(pair, timeframe)

    # Define the paths for the CSV files
    csv_paths = {
        'training': os.path.join(folder_name, f'training_{pair}_{timeframe}_data.csv'),
        'testing': os.path.join(folder_name, f'testing_{pair}_{timeframe}_data.csv'),
        'validation': os.path.join(folder_name, f'validation_{pair}_{timeframe}_data.csv')
    }

    # Dictionary to store the average normalized_change for each file
    averages = {}

    # Loop over each CSV file type
    for file_type, path in csv_paths.items():
        # Read the CSV file
        df = pd.read_csv(path)
        
        # Convert 'time' column to datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Reset index (if needed)
        df = df.reset_index(drop=True)
        
        # Drop 'target' column, if it exists
        if 'target' in df.columns:
            df.drop(columns=['target'], inplace=True)
        
        # Calculate the absolute percentage change in the 'close' price from the previous day
        df['close_price_percentage_change'] = df['close'].pct_change().abs()
        
        # Normalize the percentage change to a scale of 0 to 1
        df['normalized_change'] = df['close_price_percentage_change'].clip(upper=1)
        
        # Calculate the average normalized change
        average_normalized_change = df['normalized_change'].mean()
        averages[file_type] = average_normalized_change
        
        # Save the modified DataFrame back to CSV
        output_path = os.path.join(folder_name, f'{file_type}_processed_{pair}_{timeframe}.csv')
        df.to_csv(output_path, index=False)
        print(f'Processed {file_type} data saved to {output_path}')

    # Compute the overall average from the stored averages
    overall_average = sum(averages.values()) / len(averages)

    # Save the averages to a CSV file
    averages['overall'] = overall_average
    averages_df = pd.DataFrame(list(averages.items()), columns=['File Type', 'Average Normalized Change'])
    average_csv_path = os.path.join(folder_name, 'averages_normalized_change.csv')
    averages_df.to_csv(average_csv_path, index=False)
    print(f'Averages saved to {average_csv_path}')

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
    forex_dirs = [d for d in dirs if d.startswith('agent_forex') and os.path.isdir(os.path.join(base_dir, d))]

    return forex_dirs

def combine_backtest_results():
    """ Combine backtest results from multiple folders. """
    folders = fetch_forex_agent_folders()
    combined_df = pd.DataFrame()

    for folder in folders:
        file_path = os.path.join(folder, 'trade_history_backtest.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            Pair, timeframe_str = extract_info_from_folder_name(folder)
            df['Pair'] = Pair  # Assuming folder name contains the pair info at a specific index
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df

def analyze_combined_data(combined_df):
    """ Perform analysis on the combined DataFrame. """
    # Convert 'time' column to datetime if not already
    if combined_df['time'].dtype != 'datetime64[ns]':
        combined_df['time'] = pd.to_datetime(combined_df['time'])

    # Group by 'time' and calculate sum of 'profit' for each day
    daily_profit = combined_df.groupby('time')['profit'].sum()

    worst_daily_pnl = combined_df.groupby('time')['worst_daily_pnl'].sum()

    highest_daily_loss = worst_daily_pnl.min()

    print(highest_daily_loss)

    # Initial settings
    initial_balance = 10000
    upper_reset_threshold = 11000
    lower_reset_threshold = 9000
    upper_reset_count = 0
    lower_reset_count = 0

    daily_max_loss_reset_threshold = 9500
    daily_max_loss_count = 0

    # Prepare cumulative balance calculation with reset logic
    balances = [initial_balance]
    for profit in daily_profit:
        new_balance = balances[-1] + profit
        if new_balance >= upper_reset_threshold:
            balances.append(initial_balance)  # Reset balance to initial when exceeding upper threshold
            upper_reset_count += 1  # Increment the upper reset counter
        elif new_balance <= lower_reset_threshold:
            balances.append(initial_balance)  # Reset balance to initial when dropping below lower threshold
            lower_reset_count += 1  # Increment the lower reset counter
        else:
            balances.append(new_balance)

    balances_2 = [initial_balance]

    for worst_daily in worst_daily_pnl:
        new_balance = balances_2[-1] + worst_daily
        if new_balance <= daily_max_loss_reset_threshold:
            balances_2.append(initial_balance)
            daily_max_loss_count += 1
        else:
            balances_2.append(new_balance)

    # Starting balance
    initial_balance_graph = 10000

    # Compute cumulative balance by adding the daily profit to the initial balance
    cumulative_balance = daily_profit.cumsum() + initial_balance_graph

    # Sum of daily profits
    total_profit = daily_profit.sum()
    print(f"Total Cumulative Profit: {total_profit}")
    
    total_resets = upper_reset_count + lower_reset_count + daily_max_loss_count
    if total_resets > 0:
        probability_of_passing = (upper_reset_count / total_resets) * 100
    else:
        probability_of_passing = 0  # Set probability to 0 (or another appropriate value) when no resets have occurred

    print(f"Probability of passing: {probability_of_passing:.2f}%")

    # Plotting
    plt.figure(figsize=(14, 7))

    # Plotting daily profit
    plt.subplot(1, 2, 1)  # subplot for daily profits
    plt.plot(daily_profit.index, daily_profit.values, marker='o', linestyle='-', color='blue')
    plt.title('Daily Profit')
    plt.xlabel('Date')
    plt.ylabel('Daily Profit')
    plt.grid(True)
    plt.xticks(rotation=45)

    # Plotting cumulative balance with resets
    plt.subplot(1, 2, 2)  # subplot for cumulative balance
    plt.plot(cumulative_balance.index, cumulative_balance.values, marker='o', linestyle='-', color='green')
    plt.title('Cumulative Balance Over Time')
    plt.xlabel('Date')
    plt.ylabel('Balance')
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.savefig('balance_over_time_combined.png')  # Save the figure

def save_combined_results(combined_df, filename='combined_backtest_results.csv'):
    """ Save the combined DataFrame to a CSV file in the current directory. """
    # Define the output path using the current directory and the filename
    output_path = os.path.join(os.getcwd(), filename)

    # Save the DataFrame to a CSV file
    combined_df.to_csv(output_path, index=False)
    print(f"Combined backtest results saved to {output_path}")

def combine_forex_backtest():
    result = []
    combined_df = combine_backtest_results()
    analyze_combined_data(combined_df)
    save_combined_results(combined_df, 'combined_backtest_results.csv')
    analysis_results = analyze_pair_data(combined_df)
    result.append({
        'Probability': analysis_results['ProbabilityOfPassing'],
        'TotalProfit': analysis_results['TotalCumulativeProfit'],
        'CountPositiveReset': analysis_results['PositiveResets'],
        'CountNegativeReset': analysis_results['NegativeResets'],
        'CountDailyLossReset': analysis_results['daily_max_loss_count']
    })
    result_df = pd.DataFrame(result)
    result_df.sort_values(by='Probability', ascending=False)
    result_df.to_csv('probability_using_all_pairs.csv', index=False)

def fetch_and_aggregate_results(choice):
    print(choice)
    if choice in ('9', '11'):
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
    elif choice == '1':
        # Get the current working directory
        current_directory = os.getcwd()
        # List all subdirectories in the current directory
        all_subdirs = [os.path.join(current_directory, d) for d in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, d))]

        # Find the most recent directory
        most_recent_dir = max(all_subdirs, key=os.path.getmtime)
        combined_df = pd.DataFrame()

        file_path = os.path.join(most_recent_dir, 'trade_history_backtest.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            Pair, timeframe_str = extract_info_from_folder_name(most_recent_dir)
            df['Pair'] = Pair
            df['Timeframe'] = timeframe_str  # Assuming folder names include this info
            combined_df = pd.concat([combined_df, df], ignore_index=True) 
    else:
        timeframe = input("Enter the currency pair (e.g., Daily, 1H): ").strip().upper()

        pair = input("Enter the currency pair (e.g., GBPUSD, EURUSD): ").strip().upper()

        folder = find_recent_forex_agent_dir(pair, timeframe)

        combined_df = pd.DataFrame()

        file_path = os.path.join(folder, 'trade_history_backtest.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            Pair, timeframe_str = extract_info_from_folder_name(folder)
            df['Pair'] = Pair
            df['Timeframe'] = timeframe_str  # Assuming folder names include this info
            combined_df = pd.concat([combined_df, df], ignore_index=True) 
    return combined_df

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
    pairs = df['Pair'].unique()
    
    for pair in pairs:
        sub_df = df[df['Pair'] == pair]
        analysis_results = analyze_pair_data(sub_df)
        result.append({
            'Pair': pair,
            'Probability': analysis_results['ProbabilityOfPassing'],
            'TotalProfit': analysis_results['TotalCumulativeProfit'],
            'CountPositiveReset': analysis_results['PositiveResets'],
            'CountNegativeReset': analysis_results['NegativeResets'],
            'CountDailyLossReset': analysis_results['daily_max_loss_count']
        })
    
    result_df = pd.DataFrame(result)
    return result_df.sort_values(by='Probability', ascending=False)

def perform_analysis(choice):
    combined_df = fetch_and_aggregate_results(choice)
    probability_rankings = compute_probabilities(combined_df)
    # Get the current working directory
    current_directory = os.getcwd()
    # List all subdirectories in the current directory
    all_subdirs = [os.path.join(current_directory, d) for d in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, d))]

    # Find the most recent directory
    most_recent_dir = max(all_subdirs, key=os.path.getmtime)

    if choice not in ('9','6', '11'):
        # Create the full path for the new CSV file
        file_path = os.path.join(most_recent_dir, 'forex_pair_probability_rankings.csv')

        # Save the DataFrame to CSV in the most recently created folder
        probability_rankings.to_csv(file_path, index=False)
    else:
        probability_rankings.to_csv('forex_pair_probability_rankings.csv', index=False)

    print("Analysis complete. Results saved.")

def main_menu():
    while True:
        print("\nMain Menu:")
        print("1 - Train model with latest data - Forex")
        print("4 - Predict Next- forex")
        print("5 - Predict Specific Date- forex")
        print("6 - Train Multiple - forex")
        print("7 - Backtest - forex")
        print("8 - Train model with latest data (Magnitude) - forex")
        print("9 - Combine Forex Data and see what is the probability of suceeding (combined) - forex")
        print("10 - Combine Forex Data and see what is the probability of suceeding (single) - forex")
        print("11 - Combine Forex Data and see what is the probability of suceeding (single of multiple) - forex")

        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            training(choice)
            break
        elif choice == '4':
            predict_next_forex(choice)
            break
        elif choice == '5':
            predict_specific(choice)
            break
        elif choice == '6':
            main_training_loop_multiple_pairs()
            break
        elif choice == '7':
            backtest_trades_with_dataframe(choice)
            break
        elif choice == '8':
            train_magnitude(choice)
            break
        elif choice == '9':
            combine_forex_backtest()
            break
        elif choice == '10':
            perform_analysis(choice)
            break
        elif choice == '11':
            perform_analysis(choice)
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main_menu()