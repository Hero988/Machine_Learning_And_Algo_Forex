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
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from torch.nn import BCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import shutil
import matplotlib.pyplot as plt
import joblib

class MLPModel(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes=[100, 50], dropout_rates=[0.2, 0.5], output_size=1):
        super(MLPModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_sizes[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rates[0]),
            torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rates[1]),
            torch.nn.Linear(hidden_sizes[1], output_size),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

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

def get_user_date_input(prompt):
    # Specify the expected date format
    date_format = '%Y-%m-%d'
    # Prompt the user for a date
    date_str = input(prompt)
    # Loop until a valid date format is entered
    while True:
        try:
            # Attempt to parse the date; if successful, it's valid, so return it
            pd.to_datetime(date_str, format=date_format)
            return date_str
        except ValueError:
            # If parsing fails, notify the user and prompt again
            print("The date format is incorrect. Please enter the date in 'YYYY-MM-DD' format.")
            date_str = input(prompt)

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

    data['close_price_previous'] = data['close'].shift(1).fillna(0)
    data['close_price_percentage_change'] = data['close'].pct_change().fillna(0) * 100
    data['close_price_previous_percentage_change'] = data['close_price_percentage_change'].shift(1).fillna(0)
    data['Actual Movement'] = np.where(data['close'] > data['close_price_previous'], 1, np.where(data['close'] < data['close_price_previous'], -1, 0))

    if choice == '1':
        # Remove the last row of the DataFrame
        data = data.drop(data.tail(1).index)

    # Return the data with added indicators
    return data

def calculate_movement(data):

    if 'DateTime' in data.columns:
        # Rename the column
        data.rename(columns={'DateTime': 'time'}, inplace=True)
    
    if 'Close' in data.columns:
        # Rename the column
        data.rename(columns={'Close': 'close'}, inplace=True)

    data['close_price_percentage_change'] = data['close'].pct_change().fillna(0) * 100
    data['close_price_previous_percentage_change'] = data['close_price_percentage_change'].shift(1).fillna(0)
    data['close_price_previous'] = data['close'].shift(1).fillna(0)
    data['Actual Movement'] = np.where(data['close'] > data['close_price_previous'], 1, np.where(data['close'] < data['close_price_previous'], -1, 0))

    # Return the data with added indicators
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

def preprocess_data(data, scaler_path='scaler.pkl'):
    # Load the pre-fitted scaler
    scaler = joblib.load(scaler_path)
    
    # Assuming 'time' and 'Actual Movement' are non-numeric columns we want to drop for training
    data_numeric = data.drop(columns=['time', 'Actual Movement']).values
    
    # Transform the data using the loaded scaler
    data_scaled = scaler.transform(data_numeric)
    
    # Prepare labels if 'Actual Movement' exists in the data
    if 'Actual Movement' in data.columns:
        labels = np.where(data['Actual Movement'].values == -1, 0, 1)
        return data_scaled, labels
    
    return data_scaled

def fit_and_save_scaler(data):
    # Drop non-numeric or target columns
    data_numeric = data.drop(columns=['time', 'Actual Movement']).values
    
    # Initialize the scaler
    scaler = MinMaxScaler()
    
    # Fit the scaler on the data
    scaler.fit(data_numeric)
    
    # Save the scaler for later use
    joblib.dump(scaler, 'scaler.pkl')

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

def evaluate(choice, Pair='N/A', timeframe_str='N/A'):
    # Your existing code
    testing_files = glob.glob('testing*.csv')
    for file in testing_files:
        testing_set = read_csv_to_dataframe(file)

    testing_set = testing_set.reset_index()
    X_test, y_test = preprocess_data(testing_set)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # Adjust shape for BCELoss compatibility

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = MLPModel(input_size=X_test.shape[1])
    model.load_state_dict(torch.load('mlp_model.pth'))
    model.eval()

    predictions, true_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            output = model(inputs)
            predictions.extend(output.view(-1).numpy())
            true_labels.extend(labels.view(-1).numpy())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    optimal_threshold = find_optimal_threshold_accuracy(true_labels, predictions)
    save_optimal_threshold_json(optimal_threshold)
    final_predictions = (predictions >= optimal_threshold).astype(int)
    adjusted_predictions = np.where(final_predictions == 0, -1, 1)
    adjusted_true_labels = np.where(true_labels == 0, -1, 1)

    results_df = testing_set[['time', 'Actual Movement']].reset_index(drop=True)
    results_df['Predictions'] = predictions
    results_df['Predicted'] = adjusted_predictions

    results_df.to_csv('predicted_classification_with_Actual_Movement_and_close_MLP.csv', index=False)

    accuracy = accuracy_score(adjusted_true_labels, adjusted_predictions)
    conf_matrix = confusion_matrix(adjusted_true_labels, adjusted_predictions)

    if choice == '1':
        # Directory for saved files
        save_directory = f'agent_forex_{accuracy*100:.2f}%_{Pair}_{timeframe_str}'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
    elif choice == '2':
        # Directory for saved files
        save_directory = f'agent_futures_{accuracy*100:.2f}%_{Pair}_{timeframe_str}'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

    # Plotting the confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_xticklabels([''] + ['Negative', 'Positive'])
    ax.set_yticklabels([''] + ['Negative', 'Positive'])

    # Annotate each cell with the numeric value
    for (i, j), val in np.ndenumerate(conf_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', color='black')

    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # Save to the file system of this environment

    # Save all files except the specified ones
    exclude_files = ['things to do.txt', 'MLP.py', 'test_1.py', 'Chart.csv', 'Chart_1h.csv', 'Chart_Latest.csv']
    for file in os.listdir('.'):
        if file not in exclude_files and os.path.isfile(file):
            shutil.move(file, os.path.join(save_directory, file))

    print(f'The accuracy of the model predictions is: {accuracy * 100:.2f}%')
    print(f'Confusion Matrix:\n{conf_matrix}')
    return results_df, accuracy, conf_matrix

def get_model_path(folder_name):
    # Construct the path to the model file within the specified folder
    model_path = os.path.join(folder_name, 'mlp_model.pth')

    print(model_path)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found in directory {folder_name}")
    
    return model_path

def get_scaler_path(folder_name):
    # Construct the path to the model file within the specified folder
    scaler_path = os.path.join(folder_name, 'scaler.pkl')
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model file not found in directory {folder_name}")
    
    return scaler_path

def predict_next_futures():

    # Load the latest chart data
    dataset = read_csv_to_dataframe('Chart_Latest.csv')

    # Calculate movement for the dataset
    calculate_movement(dataset)

    # Select the last five rows but remove the very last one
    last_five_rows = dataset.tail(5).drop(dataset.tail(1).index)

    # Select only the last row (latest data point)
    latest_row = last_five_rows.tail(1)

    latest_row = latest_row.reset_index()

    print(latest_row)

    # Since this is a prediction for the future, we assume no 'Actual Movement' available
    # Prepare data (you need to ensure your preprocessing function can handle single row)
    X, _ = preprocess_data(latest_row)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)

    # Load the model (ensure it's already trained and the state dict is available)
    model = MLPModel(input_size=X.shape[1])
    model_path = get_model_path(base_dir='agent_futures')  # Optionally pass accuracy if known
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Make predictions
    with torch.no_grad():
        output = model(X_tensor)
    
    print(output)

    # Load the optimal threshold (make sure you have this value saved from your training phase)
    optimal_threshold = load_optimal_threshold_json(model_path)  # Implement this function to read the saved threshold

    # Use the threshold to determine the predicted class
    predicted_class = (output >= optimal_threshold).float()

    # Convert the tensor to a numpy array for easier handling if necessary
    predicted_movement = np.where(predicted_class.numpy().flatten() == 0, -1, 1)[0]  # [0] to get a single value

    print(f'Predicted Movement is {predicted_movement}')

    # Optionally, return the prediction or handle it as needed
    return predicted_movement

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

    # Drop rows where any of the data is missing
    eur_usd_data = eur_usd_data.dropna()

    eur_usd_data = eur_usd_data.iloc[:-1]  # Drops the last row

    eur_usd_data = eur_usd_data.reset_index()

    latest_row = eur_usd_data.tail(1)

    print(latest_row)

    scaler_path = get_scaler_path(folder_name)

    # Since this is a prediction for the future, we assume no 'Actual Movement' available
    # Prepare data (you need to ensure your preprocessing function can handle single row)
    X, _ = preprocess_data(latest_row, scaler_path)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)

    # Load the model (ensure it's already trained and the state dict is available)
    model = MLPModel(input_size=X.shape[1])
    model_path = get_model_path(folder_name)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Make predictions
    with torch.no_grad():
        output = model(X_tensor)
    
    # Load the optimal threshold (make sure you have this value saved from your training phase)
    optimal_threshold = load_optimal_threshold_json(model_path)  # Implement this function to read the saved threshold

    # Use the threshold to determine the predicted class
    predicted_class = (output >= optimal_threshold).float()

    print(f'Output: {output}, Optimal Threshold: {optimal_threshold}')

    # Convert the tensor to a numpy array for easier handling if necessary
    predicted_movement = np.where(predicted_class.numpy().flatten() == 0, -1, 1)[0]  # [0] to get a single value

    print(f'Predicted Movement is {predicted_movement}')

    # Extract necessary information from the latest row
    time = latest_row['time'].iloc[0]
    close = latest_row['close'].iloc[0]
    predicted = predicted_movement

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
    return predicted_movement

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
    eur_usd_data = calculate_indicators(eur_usd_data, choice) 

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
        scaler_path = get_scaler_path(folder_name)

    # Since this is a prediction for the future, we assume no 'Actual Movement' available
    # Prepare data (you need to ensure your preprocessing function can handle single row)
    X, _ = preprocess_data(specific_row, scaler_path)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)

    # Load the model (ensure it's already trained and the state dict is available)
    model = MLPModel(input_size=X.shape[1])
    model_path = get_model_path(folder_name)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Make predictions
    with torch.no_grad():
        output = model(X_tensor)
    
    # Load the optimal threshold (make sure you have this value saved from your training phase)
    optimal_threshold = load_optimal_threshold_json(model_path)  # Implement this function to read the saved threshold

    # Use the threshold to determine the predicted class
    predicted_class = (output >= optimal_threshold).float()

    print(f'Output: {output}, Optimal Threshold: {optimal_threshold}')

    # Convert the tensor to a numpy array for easier handling if necessary
    predicted_movement = np.where(predicted_class.numpy().flatten() == 0, -1, 1)[0]  # [0] to get a single value

    print(f'Predicted Movement is {predicted_movement}')

def training_forex_multiple(choice, Pair, timeframe_str):
    current_date = str(datetime.now().date())
    strategy_start_date_all = "1971-01-04"
    strategy_end_date_all = current_date

    start_date_all = datetime.strptime(strategy_start_date_all, "%Y-%m-%d")
    end_date_all = datetime.strptime(strategy_end_date_all, "%Y-%m-%d")

    training_start_date = "2000-01-01"
    training_end_date = current_date

    eur_usd_data = fetch_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)
    eur_usd_data = calculate_indicators(eur_usd_data, choice)
    dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]
    dataset = dataset.dropna()

    training_set, testing_set = split_and_save_dataset(dataset, timeframe_str, Pair)

    batch_size=16

    training_set = training_set.reset_index()
    testing_set = testing_set.reset_index()

    fit_and_save_scaler(training_set)

    X_train, y_train = preprocess_data(training_set)
    X_test, y_test = preprocess_data(testing_set)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # BCELoss expects the same shape for input and target
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    # Create DataLoader instances
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = MLPModel(input_size=X_train.shape[1])
    loss_function = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    epochs = 100  # Adjust number of epochs based on your dataset and early stopping criteria

    # For storing metrics
    train_losses = []
    val_losses = []

    patience = 10
    min_delta = 0.01
    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        scheduler.step(val_loss)  # Adjust learning rate based on the validation loss

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improvement_count = 0
            torch.save(model.state_dict(), 'mlp_model.pth')
            print("Validation loss decreased, saving model...")
        else:
            no_improvement_count += 1
            print(f"No improvement in validation loss for {no_improvement_count} epochs")
            if no_improvement_count >= patience:
                print("Early stopping triggered")
                break

    print("Training completed. Best model saved as 'mlp_model.pth'.")

    evaluate(choice, Pair, timeframe_str)

def main_training_loop_multiple_pairs():
    pairs = ['EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY', 'USDCAD', 'AUDUSD', 'XAUUSD', 'AUDCAD', 'NZDUSD', 'GBPCAD']
    timeframe = input("Enter the currency pair (e.g., Daily, 1H): ").strip().upper()

    choice = '1'  # Assuming '1' is for Forex, as per your original function setup

    for pair in pairs:
        print(f"Training for {pair} on {timeframe}")
        training_forex_multiple(choice, pair, timeframe)

def training(choice):
    if choice == '1':
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
        eur_usd_data = calculate_indicators(eur_usd_data, choice)

        # Filter the EUR/USD data for the in-sample training period
        dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]

        # Drop rows where any of the data is missing
        dataset = dataset.dropna()
    
        training_set, testing_set = split_and_save_dataset(dataset, timeframe_str, Pair)
    elif choice == '2':
        dataset = read_csv_to_dataframe('Chart_1h.csv')

        Pair = '6B'

        timeframe_str = '1H'

        # Remove the last row of the DataFrame
        dataset = dataset.drop(dataset.tail(1).index)

        calculate_movement(dataset)

        training_set, testing_set = split_and_save_dataset(dataset, timeframe_str, Pair)    

    batch_size=16

    training_set = training_set.reset_index()
    testing_set = testing_set.reset_index()

    fit_and_save_scaler(training_set)

    X_train, y_train = preprocess_data(training_set)
    X_test, y_test = preprocess_data(testing_set)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # BCELoss expects the same shape for input and target
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    # Create DataLoader instances
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = MLPModel(input_size=X_train.shape[1])
    loss_function = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    epochs = 100  # Adjust number of epochs based on your dataset and early stopping criteria

    # For storing metrics
    train_losses = []
    val_losses = []

    patience = 10
    min_delta = 0.01
    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        scheduler.step(val_loss)  # Adjust learning rate based on the validation loss

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improvement_count = 0
            torch.save(model.state_dict(), 'mlp_model.pth')
            print("Validation loss decreased, saving model...")
        else:
            no_improvement_count += 1
            print(f"No improvement in validation loss for {no_improvement_count} epochs")
            if no_improvement_count >= patience:
                print("Early stopping triggered")
                break

    print("Training completed. Best model saved as 'mlp_model.pth'.")

    evaluate(choice, Pair, timeframe_str)

def main_menu():
    while True:
        print("\nMain Menu:")
        print("1 - Train model with latest data - Forex")
        print("2 - Train model with csv file data - futures")
        print("3 - Predict Next- futures")
        print("4 - Predict Next- forex")
        print("5 - Predict Specific Date- forex")
        print("6 - Train Multiple - forex")

        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            training(choice)
            break
        elif choice == '2':
            training(choice)
            break
        elif choice == '3':
            predict_next_futures()
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
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main_menu()