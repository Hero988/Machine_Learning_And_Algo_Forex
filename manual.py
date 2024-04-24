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
from sklearn.model_selection import ParameterGrid
import time
import pandas_ta as ta
from sklearn.linear_model import LinearRegression

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

def trend_identification_live(data, window_size, persistence_period):
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

def identify_fair_value_gaps(data, window):

    data = data.copy()
    # Calculate the percentage change in closing prices
    data['Close_pct_change'] = data['close'].pct_change().abs()
    
    # Identify Bull, Bear, or Neutral candle
    data['candle_type'] = np.where(data['close'] > data['open'], 1,
                                   np.where(data['close'] < data['open'], -1, 0))
    
    # Calculate rolling maximum of highs and minimum of lows
    data['rolling_max_high'] = data['high'].rolling(window, min_periods=1).max()
    data['rolling_min_low'] = data['low'].rolling(window, min_periods=1).min()

    # Calculate global IQR for gap detection
    Q1 = data['Close_pct_change'].quantile(0.25)
    Q3 = data['Close_pct_change'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR
    
    # Initialize columns for storing gap information
    data['Gap'] = False
    data['Gap_Valid'] = False
    data['Gap_Direction'] = None
    data['Gap_Zone_High_Low'] = None
    data['Gap_Touched'] = False
    data['Gap_Closed_Inside'] = False
    data['Gap_Zone_High'] = None 
    data['Gap_Zone_Low'] = None 
    data['Gap_Closed_Inside_Direction'] = None

    # List to track ongoing gaps
    active_gaps = []

    # Iterate through each row in the DataFrame after the first few
    for i in range(2, len(data)):
        current_index = data.index[i]
        previous_index = data.index[i-1]
        two_candles_ago_index = data.index[i-2]
        
        previous_high = data.at[previous_index, 'high']
        previous_low = data.at[previous_index, 'low']

        latest_rolling_max_high = data['rolling_max_high'].iloc[-1]
        latest_rolling_min_low = data['rolling_min_low'].iloc[-1]

        # Check for gap in the previous candle
        if data.at[previous_index, 'Close_pct_change'] > outlier_threshold and (previous_high > latest_rolling_max_high or previous_low <  latest_rolling_min_low):
            data.at[previous_index, 'Gap'] = True
            
            # Determine the direction of the gap based on the candle type
            gap_direction = 'Bearish' if data.at[previous_index, 'candle_type'] == -1 else 'Bullish' if data.at[previous_index, 'candle_type'] == 1 else 'Neutral'

            Gap_Closed_Inside = False

            # High from two candles ago (before the gap)
            high_before_gap = data.at[two_candles_ago_index, 'high']
            # Low from the current candle (immediately after the gap)
            low_after_gap = data.at[current_index, 'low']
            
            # Initialize gap validity and zone for the current candle if it's not neutral
            if gap_direction != 'Neutral':
                data.at[current_index, 'Gap_Zone_High_Low'] = (high_before_gap, low_after_gap)
                data.at[current_index, 'Gap_Direction'] = gap_direction
                data.at[current_index, 'Gap_Valid'] = True
                active_gaps.append((current_index, high_before_gap, low_after_gap, gap_direction, Gap_Closed_Inside))

        # Update validity for active gaps
        for gap in list(active_gaps):
            gap_index, gap_high, gap_low, gap_dir, Gap_Closed_Inside = gap
            # Check if current price action touches the gap zone
            if (data.at[current_index, 'high'] >= gap_low and data.at[current_index, 'high'] <= gap_high):
                data.at[current_index, 'Gap_Touched'] = True
            # Check if the candle closed inside the gap zone
            if data.at[current_index, 'close'] >= gap_low and data.at[current_index, 'close'] <= gap_high and data.at[previous_index, 'Gap_Zone_High_Low'] is None and data.at[current_index, 'Gap_Zone_High_Low'] is None and Gap_Closed_Inside == False:
                Gap_Closed_Inside = True
                data.at[current_index, 'Gap_Closed_Inside'] = True
                if data.at[gap_index, 'Gap_Valid'] == True:
                    data.at[current_index, 'Gap_Zone_High'] = gap_high
                    data.at[current_index, 'Gap_Zone_Low'] = gap_low
                    data.at[current_index, 'Gap_Closed_Inside_Direction'] = gap_dir 
            # Check if the candle closed through the gap zone (in the direction of the gap)
            if (gap_dir == 'Bearish' and data.at[current_index, 'close'] > gap_high) or \
                (gap_dir == 'Bullish' and data.at[current_index, 'close'] < gap_low):
                data.at[gap_index, 'Gap_Valid'] = False
                active_gaps.remove(gap)  # Invalidate the gap if closed through
            else:
                # Extend validity if not touched
                data.at[gap_index, 'Gap_Valid'] = True

    return data

def forex_scalping_strategy_1(data):

    data = data.copy()

    # Assuming 'date' is already a datetime column or data.index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data['date'] = pd.to_datetime(data['date'])  # convert date column to datetime if not already
        data.set_index('date', inplace=True)  # Set date as index if it isn't already
        
    # Calculate the 200-day EMA
    data['EMA_500'] = ta.ema(data['close'], length=500)
    
    # Determine the trend
    data['Trend'] = data.apply(lambda row: 'UpTrend' if row['close'] > row['EMA_500'] else 'DownTrend', axis=1)
    
    # Calculate the percentage change in closing prices
    data['Close_pct_change'] = data['close'].pct_change().abs()
    
    # Identify Bull, Bear, or Neutral candle
    data['candle_type'] = np.where(data['close'] > data['open'], 'Bull',
                                   np.where(data['close'] < data['open'], 'Bear', 'Neutral'))
    
    data['candlestick_pattern'] = 'No Pattern'  # Default to 'No pattern'
    
    # Bullish Pin Bar
    bullish_pin_bar_mask = ((data['close'] > data['open']) &  # Bull candle
                            ((data['open'] - data['low']) > 2 * (data['close'] - data['open'])) &  # Long lower shadow
                            ((data['high'] - data['close']) < (data['close'] - data['open'])))  # Small upper shadow
    data.loc[bullish_pin_bar_mask, 'candlestick_pattern'] = 'Bullish_Pin_Bar'

    # Bearish Pin Bar
    bearish_pin_bar_mask = ((data['close'] < data['open']) &  # Bear candle
                            ((data['high'] - data['open']) > 2 * (data['open'] - data['close'])) &  # Long upper shadow
                            ((data['close'] - data['low']) < (data['open'] - data['close'])))  # Small lower shadow
    data.loc[bearish_pin_bar_mask, 'candlestick_pattern'] = 'Bearish_Pin_Bar'


    # Calculate global IQR for gap detection
    Q1 = data['Close_pct_change'].quantile(0.25)
    Q3 = data['Close_pct_change'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR
    
    # Initialize columns for storing gap information
    data['Demand_and_Supply_Zone'] = None
    data['Gap_Zone_High_Low'] = None
    data['Demand_and_Supply_Zone_ID'] = None
    data['Demand_or_Supply'] = None

    # List to track ongoing zones
    active_zones = []
    data['buy_or_sell'] = None  # Initialize the new column
    
    # Iterate through each row in the DataFrame after the first few
    for i in range(2, len(data)):
        current_index = data.index[i]
        previous_index = data.index[i-1]
        two_candles_ago_index = data.index[i-2]
        current_date = current_index
        previous_date = current_date - pd.DateOffset(days=1)
        two_candles_ago_date = current_date - pd.DateOffset(days=2)

        # Check for a significant change
        if data.at[previous_index, 'Close_pct_change'] > outlier_threshold:
            data.at[previous_index, 'Demand_and_Supply_Zone'] = True
            demand_or_supply = 'Supply' if data.at[previous_index, 'candle_type'] == 'Bear' else 'Demand' if data.at[previous_index, 'candle_type'] == 'Bull' else 'Neutral'
            
            # High and Low from relevant candles
            high_of_zone = data.at[two_candles_ago_index, 'high']
            low_of_zone = data.at[current_index, 'low']
            
            if demand_or_supply != 'Neutral':
                zone_id = f"{demand_or_supply}_{previous_date.strftime('%Y-%m-%d')}"
                data.at[previous_index, 'Gap_Zone_High_Low'] = (high_of_zone, low_of_zone)
                data.at[previous_index, 'Demand_or_Supply'] = demand_or_supply
                data.at[previous_index, 'Demand_and_Supply_Zone_ID'] = zone_id
                active_zones.append((zone_id, high_of_zone, low_of_zone, demand_or_supply, previous_date))

        # Remove zones older than 1 year
        active_zones = [zone for zone in active_zones if zone[4] >= current_date - pd.DateOffset(days=15)]

        updated_zones = []

        for zone in active_zones:
            zone_id, zone_high, zone_low, zone_type, zone_date = zone
            current_price = data.at[current_index, 'close']
            if zone_date >= current_date - pd.DateOffset(days=365):
                if (zone_type == 'Supply' and current_price > zone_high) or (zone_type == 'Demand' and current_price < zone_low):
                    # Switch zone type from Supply to Demand or vice versa
                    new_zone_type = 'Demand' if zone_type == 'Supply' else 'Supply'
                    zone_id = f"{new_zone_type}_{zone_date.strftime('%Y-%m-%d')}"
                    updated_zones.append((zone_id, zone_high, zone_low, new_zone_type, zone_date))
                else:
                    updated_zones.append(zone)
        active_zones = updated_zones
        
        # Check if current price enters any active zone
        current_price = data.at[current_index, 'close']
        current_zone = None
        for zone in active_zones:
            zone_id, zone_high, zone_low, zone_type, zone_date = zone
            if zone_low <= current_price <= zone_high:
                data.at[current_index, 'Entered_Zone_ID'] = zone_id
                data.at[current_index, 'zone_high'] = zone_high
                data.at[current_index, 'zone_low'] = zone_low
                data.at[current_index, 'zone_type'] = zone_type
                current_zone = zone
                break

        if current_zone:
            # Determine if the current candle direction aligns with the zone type
            candle_type = data.at[current_index, 'candle_type']
            trend_type = data.at[current_index, 'Trend']
            candle_stick_pattern_type = data.at[current_index, 'candlestick_pattern']
            if (zone_type == 'Demand' and candle_type == 'Bull' and candle_stick_pattern_type == 'Bullish_Pin_Bar') or (zone_type == 'Supply' and candle_type == 'Bear' and trend_type == 'DownTrend' and candle_stick_pattern_type == 'Bearish_Pin_Bar'):
                data.at[current_index, 'buy_or_sell'] = 'Buy' if zone_type == 'Demand' else 'Sell'

            next_zone_type = 'Demand' if current_zone[3] == 'Supply' else 'Supply'
            # Find the next non-overlapping zone
            next_zone = next((z for z in active_zones if z[3] == next_zone_type and z[1] > current_zone[1] and z[2] < current_zone[2]), None)
            if next_zone:
                data.at[current_index, 'Next_Zone_Type'] = next_zone_type
                data.at[current_index, 'Next_Zone_High'] = next_zone[1]
                data.at[current_index, 'Next_Zone_Low'] = next_zone[2]

    return data

def linear_regression_strategy(data):
    # https://www.youtube.com/watch?v=qHISgnYkB7Y&ab_channel=QuantProgram
    data = data.copy()
    # Prepare the data
    # Convert date index to a numerical value, days since start
    data['Numerical Index'] = (data.index - data.index.min()).days

    # Initialize the columns for predictions and signals
    data['Predicted'] = np.nan
    data['Signal'] = 0

    # Model initialization
    model = LinearRegression()

    # Calculate the 200-day EMA
    data['EMA_500'] = ta.ema(data['close'], length=500)
    
    # Determine the trend
    data['Trend'] = data.apply(lambda row: 'UpTrend' if row['close'] > row['EMA_500'] else 'DownTrend', axis=1)

   # Iterate through each row in the DataFrame after the first few
    for i in range(2, len(data)):
        current_index = data.index[i]
        X = data.loc[data.index[:i], 'Numerical Index'].values.reshape(-1, 1)  # Use data up to the current point
        y = data.loc[data.index[:i], 'close'].values

        # Fit the model
        model.fit(X, y)

        # Make a prediction for the next day
        if i < len(data) - 1:
            next_index = np.array([[data.loc[current_index, 'Numerical Index']]])
            data.loc[current_index, 'Predicted'] = model.predict(next_index)[0]

            # Determine the trend by the slope of the regression line
            slope = model.coef_[0]

            # Create signals based on the prediction
            if data.loc[current_index, 'close'] < data.loc[current_index, 'Predicted'] and data.loc[current_index, 'Trend'] == 'UpTrend':
                data.loc[current_index, 'Signal'] = 1  # Buy signal
            elif data.loc[current_index, 'close'] > data.loc[current_index, 'Predicted'] and data.loc[current_index, 'Trend'] == 'DownTrend':
                data.loc[current_index, 'Signal'] = -1  # Sell signal

    return data

def ema_corssover_strategy(data, short_ema_length, long_ema_length, trend_ema_long):

    # Initialize default values if not already set
    if short_ema_length is None:
        short_ema_length = 5
    if long_ema_length is None:
        long_ema_length = 10
    if trend_ema_long is None:
        trend_ema_long = 50

    data = data.copy()
    data['short_ema'] = ta.ema(data['close'], length=short_ema_length)  # Short-term EMA ta.ema(data['close'], length=500)
    data['long_ema'] = ta.ema(data['close'], length=long_ema_length)   # Medium-term EMA
    data['trend_ema'] = ta.ema(data['close'], length=trend_ema_long)  # Long-term EMA for trend direction

    # Create signals based on crossovers
    data['buy_signal'] = (data['short_ema'] > data['long_ema']) & (data['short_ema'].shift(1) <= data['long_ema'].shift(1)) & (data['close'] > data['trend_ema'])
    data['sell_signal'] = (data['short_ema'] < data['long_ema']) & (data['short_ema'].shift(1) >= data['long_ema'].shift(1)) & (data['close'] < data['trend_ema'])

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
        'EURUSD', 'GBPUSD', 'USDCHF', 'USDCAD',
        'AUDUSD', 'AUDNZD', 'AUDCAD', 'AUDCHF',
        'GBPCAD', 'NZDUSD', 'EURGBP', 'EURAUD',
        'EURCHF', 'EURNZD', 'EURCAD', 'GBPCAD',
        'GBPCHF', 'CADCHF', 'GBPAUD',
        'GBPNZD', 'NZDCAD', 'NZDCHF',
    ]
    
    # Ask user for the timeframe
    timeframe = input("Enter the timeframe (e.g., Daily, 1H): ").strip().upper()

    use_param_grid = input("Do you want to use grid search? (yes/no): ").strip().lower()

    while True:
        print("\nPlease choose strategy you would like to use:")
        print("1 - Reversal")
        print("2 - Flag")
        print("3 - Fair Value Gaps")
        print("4 - Demand and Supply Zones")
        print("5 - EMA corssover")
        print("6 - Linar Regression")

        choice_strategy = input("Enter your choice (1/2/3): ")
        break

    choice = '1'
    for pair in forex_pairs:
        print(f"Training for {pair} on {timeframe}")
        multiple_manual_trading(choice, pair, timeframe, choice_strategy, use_param_grid)

def multiple_manual_trading(choice, Pair, timeframe_str, choice_strategy, use_param_grid):
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

    if use_param_grid == 'yes':
        if choice_strategy != '5':
            # Setting up parameter grid
            param_grid = {
                'window_size': range(5, 21, 5),  # Example: testing window sizes 5, 10, 15, 20
                'persistence_period': range(3, 13, 3)  # Example: testing persistence periods 3, 6, 9, 12
            }
            # DataFrame to store results
            results_df = pd.DataFrame(columns=['window_size', 'persistence_period'])
        else:
            # Setting up parameter grid
            param_grid = {
                'short_ema_length': [5, 10, 15],  # Short EMA to capture quick movements
                'long_ema_length': [20, 30, 40],  # Long EMA for more significant trend changes
                'trend_ema_long': [50, 100, 200]  # Very long EMA for overall trend direction
            }
            results_df = pd.DataFrame(columns=['short_ema_length', 'long_ema_length', 'trend_ema_long'])

        # Grid search over parameters
        for params in ParameterGrid(param_grid):
            if choice_strategy != '5':
                highest_probability = evaluate_parameters(params['window_size'], params['persistence_period'], choice_strategy, eur_usd_data, training_start_date, training_end_date, timeframe_str, Pair, choice)
                # Create a DataFrame for the current results and concatenate it
                current_results = pd.DataFrame([{
                    'window_size': params['window_size'],
                    'persistence_period': params['persistence_period'],
                    'highest_probability': highest_probability
                }])
                results_df = pd.concat([results_df, current_results], ignore_index=True)
            else:
                window_size = 10
                persistence_period = 5
                highest_probability = evaluate_parameters(window_size, persistence_period, choice_strategy, eur_usd_data, training_start_date, training_end_date, timeframe_str, Pair, choice, params['short_ema_length'], params['long_ema_length'], params['trend_ema_long'])
                # Create a DataFrame for the current results and concatenate it
                current_results = pd.DataFrame([{
                    'short_ema_length': params['short_ema_length'],
                    'long_ema_length': params['long_ema_length'],
                    'trend_ema_long': params['trend_ema_long'],
                    'highest_probability': highest_probability
                }])
                results_df = pd.concat([results_df, current_results], ignore_index=True)

            # Save the current results to a CSV file
            current_results.to_csv(f'current_parameters.csv', index=False)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Managing directories for results
            if choice_strategy == '1':
                save_directory = f'backtest_reversal_patterns_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
                destination_directory = f'backtest_reversal_patterns'
            elif choice_strategy == '2':
                save_directory = f'backtest_flag_patterns_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
                destination_directory = f'backtest_flag_patterns'
            elif choice_strategy == '3':
                save_directory = f'fair_value_gap_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
                destination_directory = f'fair_value_gap'
            elif choice_strategy == '4':
                save_directory = f'demand_and_supply_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
                destination_directory = f'demand_and_supply'
            elif choice_strategy == '5':
                save_directory = f'ema_crossover_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
                destination_directory = f'ema_crossover'
            elif choice_strategy == '6':
                save_directory = f'linear_regression_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
                destination_directory = f'linear_regression'
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)

            # Save all files except the specified ones
            exclude_files = ['things to do.txt', 'MLP.py', 'test_1.py', 'Chart.csv', 'Chart_1h.csv', 'Chart_Latest.csv', 'LSTM.py', 'RNN.py', 'XGboost.py', 'manual.py']
            for file in os.listdir('.'):
                if file not in exclude_files and os.path.isfile(file):
                    shutil.move(file, os.path.join(save_directory, file))

            if highest_probability == 0:
                shutil.rmtree(save_directory)
            else:
                move_directory(save_directory, destination_directory)
    else:
        window_size = 10
        persistence_period = 5
        highest_probability = evaluate_parameters(window_size, persistence_period, choice_strategy, eur_usd_data, training_start_date, training_end_date, timeframe_str, Pair, choice)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Managing directories for results
        if choice_strategy == '1':
            save_directory = f'backtest_reversal_patterns_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
            destination_directory = f'backtest_reversal_patterns'
        elif choice_strategy == '2':
            save_directory = f'backtest_flag_patterns_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
            destination_directory = f'backtest_flag_patterns'
        elif choice_strategy == '3':
            save_directory = f'fair_value_gap_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
            destination_directory = f'fair_value_gap'
        elif choice_strategy == '4':
            save_directory = f'demand_and_supply_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
            destination_directory = f'demand_and_supply'
        elif choice_strategy == '5':
            save_directory = f'ema_crossover_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
            destination_directory = f'ema_crossover'
        elif choice_strategy == '6':
            save_directory = f'linear_regression_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
            destination_directory = f'linear_regression'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)  
        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)

        # Save all files except the specified ones
        exclude_files = ['things to do.txt', 'MLP.py', 'test_1.py', 'Chart.csv', 'Chart_1h.csv', 'Chart_Latest.csv', 'LSTM.py', 'RNN.py', 'XGboost.py', 'manual.py']
        for file in os.listdir('.'):
            if file not in exclude_files and os.path.isfile(file):
                shutil.move(file, os.path.join(save_directory, file))
        move_directory(save_directory, destination_directory)

def evaluate_parameters(window_size, persistence_period, choice_strategy, eur_usd_data, training_start_date, training_end_date, timeframe_str, Pair, choice, short_ema_length=None, long_ema_length=None, trend_ema_long=None):
    if choice_strategy == '1':
        trend_identification_live(eur_usd_data, window_size, persistence_period)
        dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]
        # Drop rows where any of the data is missing
        dataset = dataset.dropna()
        backtest_set = get_backtest_data(dataset, timeframe_str, Pair)
    elif choice_strategy == '2':
        identify_flag_patterns(eur_usd_data)
        dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]
        # Drop rows where any of the data is missing
        dataset = dataset.dropna()
        backtest_set = get_backtest_data(dataset, timeframe_str, Pair)
    elif choice_strategy == '3':
        # Filter the EUR/USD data for the in-sample training period
        dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]
        backtest_set_copy = get_backtest_data(dataset, timeframe_str, Pair)
        backtest_set = identify_fair_value_gaps(backtest_set_copy, window_size)
    elif choice_strategy == '4':
        # Filter the EUR/USD data for the in-sample training period
        dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]
        backtest_set_copy = get_backtest_data(dataset, timeframe_str, Pair)
        backtest_set = forex_scalping_strategy_1(backtest_set_copy)
    elif choice_strategy == '5':
        # Filter the EUR/USD data for the in-sample training period
        dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]
        backtest_set_copy = get_backtest_data(dataset, timeframe_str, Pair)
        backtest_set = ema_corssover_strategy(backtest_set_copy, short_ema_length, long_ema_length, trend_ema_long)
    elif choice_strategy == '6':
        # Filter the EUR/USD data for the in-sample training period
        dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]
        backtest_set_none = get_backtest_data(dataset, timeframe_str, Pair)
        backtest_set = linear_regression_strategy(backtest_set_none)
        
    initial_balance=10000
    leverage=30
    transaction_cost=0.0002

    folder_name = os.getcwd()

    data_csv_filename = os.path.join(folder_name, 'data_backtest.csv')
    backtest_set.to_csv(data_csv_filename)

    if 'JPY' in Pair:
        lot_size = 100  # Smaller lot size for pairs including JPY
    else:
        lot_size = 10000  # Default lot size for other pairs

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

    highest_probability = perform_analysis(choice)

    return highest_probability

def move_directory(source_directory, destination_directory):
    # Ensure the destination directory exists where the source directory needs to be moved
    destination_path = os.path.join(destination_directory, os.path.basename(source_directory))
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Move the source directory to the new location
    shutil.move(source_directory, destination_path)

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

    # Prompt the user for the desired timeframe for analysis and standardize the input
    timeframe_str = input("Enter the currency pair (e.g., Daily, 1H): ").strip().upper()
    # Prompt the user for the currency pair they're interested in and standardize the input
    Pair = input("Enter the currency pair (e.g., GBPUSD, EURUSD): ").strip().upper()

    training_start_date = "2000-01-01"
    training_end_date = current_date

    # Fetch and prepare the FX data for the specified currency pair and timeframe
    eur_usd_data = fetch_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)

    while True:
        print("\nPlease choose strategy you would like to use:")
        print("1 - Reversal")
        print("2 - Flag")
        print("3 - Fair Value Gaps")
        print("4 - Demand and Supply Zones")
        print("5 - EMA corssover")
        print("6 - Linar Regression")

        choice_strategy = input("Enter your choice (1/2/3): ")
        break

    use_param_grid = input("Do you want to use grid search? (yes/no): ").strip().lower()

    if use_param_grid == 'yes':
        if choice_strategy != '5':
            # Setting up parameter grid
            param_grid = {
                'window_size': range(5, 21, 5),  # Example: testing window sizes 5, 10, 15, 20
                'persistence_period': range(3, 13, 3)  # Example: testing persistence periods 3, 6, 9, 12
            }
            # DataFrame to store results
            results_df = pd.DataFrame(columns=['window_size', 'persistence_period'])
        else:
            # Setting up parameter grid
            param_grid = {
                'short_ema_length': [5, 10, 15],  # Short EMA to capture quick movements
                'long_ema_length': [20, 30, 40],  # Long EMA for more significant trend changes
                'trend_ema_long': [50, 100, 200]  # Very long EMA for overall trend direction
            }
            results_df = pd.DataFrame(columns=['short_ema_length', 'long_ema_length', 'trend_ema_long'])

        # Grid search over parameters
        for params in ParameterGrid(param_grid):
            if choice_strategy != '5':
                highest_probability = evaluate_parameters(params['window_size'], params['persistence_period'], choice_strategy, eur_usd_data, training_start_date, training_end_date, timeframe_str, Pair, choice)
                # Create a DataFrame for the current results and concatenate it
                current_results = pd.DataFrame([{
                    'window_size': params['window_size'],
                    'persistence_period': params['persistence_period'],
                    'highest_probability': highest_probability
                }])
                results_df = pd.concat([results_df, current_results], ignore_index=True)
            else:
                window_size = 10
                persistence_period = 5
                highest_probability = evaluate_parameters(window_size, persistence_period, choice_strategy, eur_usd_data, training_start_date, training_end_date, timeframe_str, Pair, choice, params['short_ema_length'], params['long_ema_length'], params['trend_ema_long'])
                # Create a DataFrame for the current results and concatenate it
                current_results = pd.DataFrame([{
                    'short_ema_length': params['short_ema_length'],
                    'long_ema_length': params['long_ema_length'],
                    'trend_ema_long': params['trend_ema_long'],
                    'highest_probability': highest_probability
                }])
                results_df = pd.concat([results_df, current_results], ignore_index=True)

            # Save the current results to a CSV file
            current_results.to_csv(f'current_parameters.csv', index=False)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Managing directories for results
            if choice_strategy == '1':
                save_directory = f'backtest_reversal_patterns_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
                destination_directory = f'backtest_reversal_patterns'
            elif choice_strategy == '2':
                save_directory = f'backtest_flag_patterns_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
                destination_directory = f'backtest_flag_patterns'
            elif choice_strategy == '3':
                save_directory = f'fair_value_gap_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
                destination_directory = f'fair_value_gap'
            elif choice_strategy == '4':
                save_directory = f'demand_and_supply_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
                destination_directory = f'demand_and_supply'
            elif choice_strategy == '5':
                save_directory = f'ema_crossover_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
                destination_directory = f'ema_crossover'
            elif choice_strategy == '6':
                save_directory = f'linear_regression_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
                destination_directory = f'linear_regression'
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)

            # Save all files except the specified ones
            exclude_files = ['things to do.txt', 'MLP.py', 'test_1.py', 'Chart.csv', 'Chart_1h.csv', 'Chart_Latest.csv', 'LSTM.py', 'RNN.py', 'XGboost.py', 'manual.py']
            for file in os.listdir('.'):
                if file not in exclude_files and os.path.isfile(file):
                    shutil.move(file, os.path.join(save_directory, file))

            if highest_probability == 0:
                shutil.rmtree(save_directory)
            else:
                move_directory(save_directory, destination_directory)
    else:
        window_size = 10
        persistence_period = 5
        highest_probability = evaluate_parameters(window_size, persistence_period, choice_strategy, eur_usd_data, training_start_date, training_end_date, timeframe_str, Pair, choice)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Managing directories for results
        if choice_strategy == '1':
            save_directory = f'backtest_reversal_patterns_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
            destination_directory = f'backtest_reversal_patterns'
        elif choice_strategy == '2':
            save_directory = f'backtest_flag_patterns_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
            destination_directory = f'backtest_flag_patterns'
        elif choice_strategy == '3':
            save_directory = f'fair_value_gap_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
            destination_directory = f'fair_value_gap'
        elif choice_strategy == '4':
            save_directory = f'demand_and_supply_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
            destination_directory = f'demand_and_supply'
        elif choice_strategy == '5':
            save_directory = f'ema_crossover_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
            destination_directory = f'ema_crossover'
        elif choice_strategy == '6':
            save_directory = f'linear_regression_{highest_probability:.2f}%_{Pair}_{timeframe_str}_{timestamp}'
            destination_directory = f'linear_regression'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)  
        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)

        # Save all files except the specified ones
        exclude_files = ['things to do.txt', 'MLP.py', 'test_1.py', 'Chart.csv', 'Chart_1h.csv', 'Chart_Latest.csv', 'LSTM.py', 'RNN.py', 'XGboost.py', 'manual.py']
        for file in os.listdir('.'):
            if file not in exclude_files and os.path.isfile(file):
                shutil.move(file, os.path.join(save_directory, file))
        move_directory(save_directory, destination_directory)

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

    def simulate_trading(self, data, choice_strategy):
        for _, row in data.iterrows():
            current_date = row['time'].date()  # Assuming 'time' is a datetime object

            # Update P&L after potentially resetting it for a new day
            self.update_current_pnl(row['high'], row['low'])

            if self.is_open_position:
                if self.position == 'long' and row['high'] >= self.take_profit:
                    self.close_position(self.take_profit, row['time'])
                elif self.position == 'long' and row['low'] <= self.stop_loss:
                    self.close_position(self.stop_loss, row['time'])
                elif self.position == 'short' and row['high'] >= self.stop_loss:
                    self.close_position(self.stop_loss, row['time'])
                elif self.position == 'short' and row['low'] <= self.take_profit:
                    self.close_position(self.take_profit, row['time'])

                if self.worst_case_pnl <= self.monetary_loss:
                    self.close_position(None, row['time'], self.monetary_loss)
                elif self.best_case_pnl >= self.monetary_gain:
                    self.close_position(None, row['time'], self.monetary_gain)
            
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
            elif choice_strategy == '3':
                if row['Gap_Closed_Inside'] == True and row['Gap_Closed_Inside_Direction'] == 'Bullish' and (not self.is_open_position or self.position == 'short'):
                    self.open_position(row['close'], 'long', row['time'])
                elif row['Gap_Closed_Inside'] == True and row['Gap_Closed_Inside_Direction'] == 'Bearish' and (not self.is_open_position or self.position == 'long'):
                    self.open_position(row['close'], 'short', row['time'])
                # Log the trade as hold if there is no change in position
                elif self.is_open_position:
                    self.log_trade('hold', row['close'], row['time'])
            elif choice_strategy == '4':
                if row['buy_or_sell'] == 'Buy' and (not self.is_open_position or self.position == 'short'):
                    self.open_position(row['close'], 'long', row['time'])
                elif row['buy_or_sell'] == 'Sell' and (not self.is_open_position or self.position == 'long'):
                    self.open_position(row['close'], 'short', row['time'])
                # Log the trade as hold if there is no change in position
                elif self.is_open_position:
                    self.log_trade('hold', row['close'], row['time'])
            elif choice_strategy == '5':
                if row['buy_signal'] == True and (not self.is_open_position or self.position == 'short'):
                    self.open_position(row['close'], 'long', row['time'])
                elif row['sell_signal'] == True and (not self.is_open_position or self.position == 'long'):
                    self.open_position(row['close'], 'short', row['time'])
                # Log the trade as hold if there is no change in position
                elif self.is_open_position:
                    self.log_trade('hold', row['close'], row['time'])
            elif choice_strategy == '6':
                if row['Signal'] == 1 and (not self.is_open_position or self.position == 'short'):
                    self.open_position(row['close'], 'long', row['time'])
                elif row['Signal'] == -1 and (not self.is_open_position or self.position == 'long'):
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
    
    hours = timeframe_mapping.get(timeframe, 1)  # Default to 4 hours if not specified
    hours_correct  = hours - 1
    next_candle_time = latest_time_index + timedelta(hours=hours_correct, minutes=2)
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

def execute_trade(symbol, trade_type, stop_loss_percent, take_profit_percent):
    in_trade = False
    # Initialize MT5 connection
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return

    # Check if we are already in a trade for the specified symbol
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        print("Failed to get positions, error code =", mt5.last_error())
    elif len(positions) > 0:
        print("Already in a trade on", symbol)
        in_trade = True
    else:
        # Get the current market price
        price_info = mt5.symbol_info_tick(symbol)
        print(price_info)
        if price_info is None:
            print("Failed to get price for symbol", symbol)
        else:
            ask = price_info.ask  # price for buy orders
            bid = price_info.bid  # price for sell orders
            current_price = price_info.last

            # Define the order type based on trade type
            order_type = mt5.ORDER_TYPE_BUY if trade_type == 'buy' else mt5.ORDER_TYPE_SELL

            # Calculate stop loss and take profit prices based on whether the trade is a buy or sell
            if order_type == mt5.ORDER_TYPE_BUY:
                # For buy orders, use the ask price
                sl_price = ask * (1 - stop_loss_percent / 100)  # Stop loss below the ask price
                tp_price = ask * (1 + take_profit_percent / 100)  # Take profit above the ask price
            else:
                # For sell orders, use the bid price
                sl_price = bid * (1 + stop_loss_percent / 100)  # Stop loss above the bid price
                tp_price = bid * (1 - take_profit_percent / 100)  # Take profit below the bid price

            # Create trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": 0.1,
                "type": order_type,
                "price": current_price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 10,
                "magic": 234000,
                "comment": "python script open",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Send the trade request
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print("Failed to send order :", result)
            else:
                print("Order executed, ", result)

    # Disconnect from MT5
    mt5.shutdown()

    return in_trade

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

    folder_name = 'latest_data'

    while True:
        current_date_time  = datetime.now()

        # Fetch and prepare the FX data for the specified currency pair and timeframe
        eur_usd_data = fetch_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)
        
        # Calculate the date three months ago
        three_months_ago = current_date_time - relativedelta(months=3)

        date_three_months_ago = three_months_ago.strftime('%Y-%m-%d')

        # Retrieve and store the current date
        current_date_string = str(datetime.now().date())

        dataset = eur_usd_data[(eur_usd_data.index >= date_three_months_ago) & (eur_usd_data.index <= current_date_string)]

        latest_dataset_entry = dataset.index[-1]

        # Calculate the difference in time
        time_difference = current_date_time - latest_dataset_entry

        # Convert the time difference to hours
        hours_difference = int(time_difference.total_seconds() / 3600) + 3

        window_size = 5
        persistence_period = 5

        initialize_mt5()
        data = fetch_live_data(Pair, timeframe_str, hours_difference)

        if data is not None:
            data = data[:-1]  # Slices off the last row
            data.index = data.index - pd.Timedelta(hours=2)  # Adjust the time by subtracting 2 hours
            data.index = data.index.tz_localize(None)  # Remove the timezone information
            
            # Calculate indicators for the updated dataset
            # Concatenate new data
            updated_dataset = pd.concat([dataset, data]).drop_duplicates()

            forex_scalping_strategy_1(updated_dataset)
            
            #updated_dataset = trend_identification_live(updated_dataset, window_size, persistence_period)
            
            # Get the latest time index
            latest_time_index = updated_dataset.index[-1].strftime('%Y-%m-%d %H:%M:%S')

            latest_time_index_file_name = updated_dataset.index[-1].strftime('%Y-%m-%d %H-%M-%S')

            # Ensure the folder exists
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)  # Create the folder if it does not exist

            # Define the full path for the CSV file
            file_path = os.path.join(folder_name, f'Full_data_{latest_time_index_file_name}_{Pair}.csv')

            # Save the DataFrame to CSV in the specified folder
            updated_dataset.to_csv(file_path, index=True)

            latest_row = updated_dataset.iloc[-1]

            latest_close_price = latest_row['close']

            print(f'Latest Row Time: {latest_time_index}, Latest Row Close Price: {latest_close_price}')

            # Decision making based on the latest row analysis
            action = 'no buy or sell'
            if latest_row['Gap_Closed_Inside'] == True and latest_row['Gap_Closed_Inside_Direction'] == 'Bullish':
                action = 'buy'
                in_trade = execute_trade(Pair, 'buy', 0.5, 1)
                if in_trade:
                    action = 'holding'
            elif latest_row['Gap_Closed_Inside'] == True and latest_row['Gap_Closed_Inside_Direction'] == 'Bearish':
                action = 'sell'
                in_trade = execute_trade(Pair, 'sell', 0.5, 1)
                if in_trade:
                    action = 'holding'

            # Prepare the data to be saved
            output_data = {
                'Time': [latest_time_index],
                'Close Price': [latest_close_price],
                'Action': [action]
            }

            # Convert the dictionary to DataFrame
            output_df = pd.DataFrame(output_data)

            output_file_path = os.path.join(folder_name, 'latest_actions.csv')

            # Check if the file exists to determine if headers should be written
            file_exists = os.path.isfile(output_file_path)

            # Save the DataFrame to CSV in the specified folder
            output_df.to_csv(output_file_path, mode='a', index=False, header=not file_exists)

            print(f"Data saved to {output_file_path}")

            # Calculate time to sleep until the next candle plus 5 minutes
            sleep_time, next_candle_time_str, time_string = calculate_time_to_next_candle(latest_time_index, timeframe_str)
            print(f"Next candle will form at: {next_candle_time_str} and the current time is {time_string}")
            countdown(sleep_time)  # Display countdown

def test_data():
    # Retrieve and store the current date
    current_date = str(datetime.now().date())

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

    # Fetch and prepare the FX data for the specified currency pair and timeframe
    eur_usd_data_none = fetch_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)

    eur_usd_data = linear_regression_strategy(eur_usd_data_none)

    training_start_date = "2000-01-01"
    training_end_date = current_date

    # Filter the EUR/USD data for the in-sample training period
    dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]

    # Drop rows where any of the data is missing
    dataset = dataset.fillna(0)

    backtest_set = get_backtest_data(dataset, timeframe_str, Pair)
   
def main_menu():
    while True:
        print("\nMain Menu:")
        print("1 - Forex Trading Strategy (single)")
        print("2 - Forex Trading Strategy (multiple)")
        print("3 - Analysis (combined)")
        print("4 - Live Trading")
        print("5 - Test Data")

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
        elif choice == '5':
            test_data()
            break

if __name__ == "__main__":
    main_menu()