import requests
import pandas as pd
import time
import os
from datetime import datetime, timezone

# --- Configuration ---
SYMBOL = "ETHUSDT"  # Trading pair (e.g., BTCUSDT, ETHUSDT)
INTERVAL_MINUTES = "60"  # Candlestick interval: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, "D", "W", "M"
START_DATE_STR = "2024-01-01 00:00:00"  # Start date (YYYY-MM-DD HH:MM:SS)
END_DATE_STR = "2025-01-01 00:00:00"    # End date (YYYY-MM-DD HH:MM:SS)
# Extract just the date part for the filename
START_DATE_FILENAME = START_DATE_STR.split()[0].replace('-', '')
END_DATE_FILENAME = END_DATE_STR.split()[0].replace('-', '')

# Define DATA directory path and ensure it exists
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "DATA")
os.makedirs(DATA_DIR, exist_ok=True)

# Create output filename with full path
OUTPUT_FILENAME = os.path.join(DATA_DIR, f"{SYMBOL}_{INTERVAL_MINUTES}m_{START_DATE_FILENAME}_to_{END_DATE_FILENAME}.csv")
API_ENDPOINT = "https://api.bybit.com/v5/market/kline"
REQUEST_LIMIT = 1000  # Max candles per API request (Bybit's limit)
# Delay between API requests in seconds to respect rate limits
API_DELAY_SECONDS = 0.2

# --- Helper Functions ---
def datestr_to_milliseconds(date_str):
    """Converts a YYYY-MM-DD HH:MM:SS string to UTC milliseconds timestamp."""
    dt_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    return int(dt_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)

def milliseconds_to_datestr(ms):
    """Converts UTC milliseconds timestamp to YYYY-MM-DD HH:MM:SS string."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

# --- Main Data Fetching Logic ---
def fetch_bybit_ohlcv(symbol, interval, start_time_ms, end_time_ms, limit):
    all_data = []
    current_start_time_ms = start_time_ms
    # Calculate interval in milliseconds once
    interval_val = 0
    if interval.isdigit(): # For 1, 5, 15, 60 etc. (minutes)
        interval_val = int(interval) * 60 * 1000
    elif interval == "D":
        interval_val = 24 * 60 * 60 * 1000
    elif interval == "W":
        interval_val = 7 * 24 * 60 * 60 * 1000
    elif interval == "M": # Approximation, actual month length varies (use 30 days)
        interval_val = 30 * 24 * 60 * 60 * 1000 
    
    if interval_val == 0:
        print("Error: Could not determine interval in milliseconds. Exiting.")
        return []

    print(f"Fetching data for {symbol} at {interval} interval...")
    print(f"Desired Range: {milliseconds_to_datestr(start_time_ms)} To: {milliseconds_to_datestr(end_time_ms)}")

    while True: # Loop indefinitely until break condition is met
        params = {
            "category": "linear",  # Or "spot" or "inverse" depending on the symbol
            "symbol": symbol,
            "interval": interval,
            "start": current_start_time_ms,
            # "end": end_time_ms, # REMOVED end parameter from loop request
            "limit": limit
        }
        
        print(f"Requesting data starting from: {milliseconds_to_datestr(current_start_time_ms)}...")
        
        try:
            headers = { # Added User-Agent just in case it helps stability
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(API_ENDPOINT, params=params, headers=headers) # Added headers
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") == 0 and data.get("result") and data["result"].get("list"):
                kline_list = data["result"]["list"]
                if not kline_list:
                    print("No more data returned by API.")
                    break # Exit loop if API returns empty list

                # Bybit returns data in reverse chronological order (newest first)
                kline_list.reverse() # Reverse to process chronologically

                # Filter received data to stay within the desired end_time_ms
                new_candles_in_range = 0
                for kline in kline_list:
                    candle_start_time = int(kline[0])
                    # Only add candles that start AT or BEFORE the overall end time
                    if candle_start_time <= end_time_ms:
                         # Also ensure we don't add data before the initial start (edge case for first request)
                        if candle_start_time >= start_time_ms:
                            all_data.append({
                                "timestamp_ms": candle_start_time,
                                "open_time": milliseconds_to_datestr(candle_start_time),
                                "open": float(kline[1]),
                                "high": float(kline[2]),
                                "low": float(kline[3]),
                                "close": float(kline[4]),
                                "volume": float(kline[5])
                                # Removed turnover field
                            })
                            new_candles_in_range += 1
                    else:
                        # Stop processing this batch if we exceed the end_time_ms
                        break 

                print(f"Received {len(kline_list)} candles, added {new_candles_in_range} within target range.")

                # Determine the timestamp of the last candle processed in this batch
                last_processed_candle_timestamp = int(kline_list[-1][0])

                # --- Loop termination checks ---
                # 1. If the last candle received is already at or past the overall end time
                if last_processed_candle_timestamp >= end_time_ms:
                    print("Reached end of specified date range.")
                    break
                # 2. If API returned fewer candles than requested (means no more data available)
                if len(kline_list) < limit:
                    print("API returned fewer candles than limit, assuming end of available data.")
                    break

                # Update start time for the next request (start after the last received candle)
                current_start_time_ms = last_processed_candle_timestamp + interval_val
                
                # --- Safety check for infinite loop ---
                if current_start_time_ms <= last_processed_candle_timestamp:
                     print("Warning: Next start time calculation did not advance. Stopping to prevent infinite loop.")
                     break

            elif data.get("retCode") != 0:
                print(f"Bybit API Error: {data.get('retMsg')} (Code: {data.get('retCode')})")
                if data.get('retCode') == 10006: # Rate limit error
                    print("Rate limit hit. Waiting for 60 seconds...")
                    time.sleep(60)
                    continue # Retry the same request
                break # Other critical API error
            else:
                print("No data in 'list' or unexpected response structure.")
                break
        
        except requests.exceptions.RequestException as e:
            print(f"HTTP Request Error: {e}")
            print("Waiting for 60 seconds before retrying...")
            time.sleep(60)
            continue
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

        time.sleep(API_DELAY_SECONDS) # Respect API rate limits

    return all_data

# --- Script Execution ---
if __name__ == "__main__":
    start_ms = datestr_to_milliseconds(START_DATE_STR)
    end_ms = datestr_to_milliseconds(END_DATE_STR)

    ohlcv_data = fetch_bybit_ohlcv(SYMBOL, INTERVAL_MINUTES, start_ms, end_ms, REQUEST_LIMIT)

    if ohlcv_data:
        df = pd.DataFrame(ohlcv_data)
        # Remove duplicates just in case, based on timestamp
        df.drop_duplicates(subset=['timestamp_ms'], keep='first', inplace=True)
        # Ensure data is sorted chronologically before saving
        df.sort_values(by='timestamp_ms', inplace=True) 
        
        # Optional: Filter dataframe again just to be absolutely sure it's within exact start/end ms
        df = df[(df['timestamp_ms'] >= start_ms) & (df['timestamp_ms'] <= end_ms)]

        df.to_csv(OUTPUT_FILENAME, index=False)
        print(f"\nData successfully saved to {OUTPUT_FILENAME}")
        print(f"Total candles fetched: {len(df)}")
        if not df.empty:
            print(f"Final data range in CSV: {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}")
    else:
        print("\nNo data was fetched or saved.")