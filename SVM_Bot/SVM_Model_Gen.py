import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import ta

CSV_FILE_PATH = r"C:\Users\kevin\OneDrive\CRYPTO_BOTS\DATA\BTCUSDT_60m_20240101_to_20250101.csv"  # Example: r"C:\data\BTCUSDT_15min_data.csv"
MODEL_SAVE_DIR = r"C:\Users\kevin\OneDrive\CRYPTO_BOTS\SVM_Bot\SVM_models"
CRYPTO_ASSET = "BTCUSDT"  # For naming saved files

TIMEFRAME = "1H"
N_FUTURE_PERIODS = 1  # How many periods ahead to predict

SVM_KERNEL = 'sigmoid'  # Options: 'rbf', 'linear', 'poly', 'sigmoid'
SVM_C = 50
SVM_GAMMA = .01  # Kernel coefficient for 'rbf', 'poly', 'sigmoid'. Options: 'scale', 'auto', or a float value
SVM_DEGREE = 2      # Degree for 'poly' kernel. Ignored by other kernels.

def load_and_prepare_data(csv_path):
    """
    Loads data from a CSV file. Assumes data is already in the correct timeframe.
    Assumes the input CSV has 'open_time' column for timestamp.
    """
    print(f"Loading data: {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  Error: CSV file not found at {csv_path}")
        return None

    if 'open_time' not in df.columns:
        print("  Error: 'open_time' column not found in CSV.")
        return None
    df['open_time'] = pd.to_datetime(df['open_time'])
    df.set_index('open_time', inplace=True)

    df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    }, inplace=True)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"  Warning: Column '{col}' not found for numeric conversion.")

    df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

    if df.empty:
        print("  Error: Dataframe is empty after loading and initial cleaning. Check CSV content.")
        return None
        
    print("  Data loaded and prepared successfully.")
    return df

def calculate_features(df):
    """
    Calculates the specified technical features.
    """
    print("Calculating features...")
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            print(f"  Error: Required column '{col}' not found for feature calculation.")
            return None
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"  Error: Column '{col}' is not numeric.")
            return None

    df['Return_1bar'] = df['Close'].pct_change()
    df['EMA_12'] = ta.trend.EMAIndicator(close=df['Close'], window=12, fillna=False).ema_indicator()
    df['EMA_26'] = ta.trend.EMAIndicator(close=df['Close'], window=26, fillna=False).ema_indicator()
    
    macd_indicator = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    df['MACD_Hist'] = macd_indicator.macd_diff()
    
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14, fillna=False).rsi()
    
    stoch_indicator = ta.momentum.StochasticOscillator(
        high=df['High'], low=df['Low'], close=df['Close'], 
        window=14, smooth_window=3, fillna=False
    )
    df['Stoch_K_14'] = stoch_indicator.stoch()
    
    bb_indicator = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2, fillna=False)
    df['BB_Upper'] = bb_indicator.bollinger_hband()
    df['BB_Lower'] = bb_indicator.bollinger_lband()
    df['BB_Width_20_2'] = df['BB_Upper'] - df['BB_Lower']
    
    df['ATR_14'] = ta.volatility.AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Close'], window=14, fillna=False
    ).average_true_range()
    
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'], fillna=False).on_balance_volume()
    df['OBV_Change'] = df['OBV'].diff()
    
    df['Volume_Change'] = df['Volume'].diff()

    feature_columns = [
        'Return_1bar', 'EMA_12', 'EMA_26', 'MACD_Hist', 'RSI_14', 
        'Stoch_K_14', 'BB_Width_20_2', 'ATR_14', 'OBV_Change', 'Volume_Change'
    ]
    
    df_features = df[feature_columns].copy()
    
    print("  Features calculated successfully.")
    return df_features

def define_target_variable(df_full, n_future_periods):
    """
    Defines the target variable: 1 if price moves up, 0 otherwise.
    """
    print("Defining target variable...")
    if 'Close' not in df_full.columns:
        print("  Error: 'Close' column not found for target definition.")
        return None
        
    df_full['price_future'] = df_full['Close'].shift(-n_future_periods)
    df_full['target'] = (df_full['price_future'] > df_full['Close']).astype(int)
    print("  Target variable defined successfully.")
    return df_full['target']

def main():
    """
    Main function to run the SVM model training pipeline.
    """
    print("--- SVM Model Training Initialized ---")

    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        print(f"Created directory: {MODEL_SAVE_DIR}")

    # 1. Load and Prepare Data
    df_full = load_and_prepare_data(CSV_FILE_PATH)
    if df_full is None:
        print("--- Script terminated due to data loading error ---")
        return

    # 2. Calculate Features
    df_features = calculate_features(df_full.copy())
    if df_features is None:
        print("--- Script terminated due to feature calculation error ---")
        return
    
    # 3. Define Target Variable
    df_target = define_target_variable(df_full, N_FUTURE_PERIODS)
    if df_target is None:
        print("--- Script terminated due to target definition error ---")
        return

    # 4. Combine features and target, and clean NaNs
    print("Combining features and target, and removing NaN values...")
    df_model_data = pd.concat([df_features, df_target], axis=1)
    
    initial_rows = len(df_model_data)
    df_model_data.dropna(inplace=True)
    final_rows = len(df_model_data)
    print(f"  Original rows: {initial_rows}, Rows after NaN removal: {final_rows}")

    if df_model_data.empty:
        print("  Error: No data left after NaN removal. Check feature calculation or input data.")
        print("--- Script terminated ---")
        return

    # 5. Prepare data for SVM
    print("Preparing data for SVM (splitting X and y, scaling features)...")
    X = df_model_data.drop('target', axis=1)
    y = df_model_data['target']

    if X.empty or y.empty:
        print("  Error: Feature set (X) or target (y) is empty after processing.")
        print("--- Script terminated ---")
        return
    
    print(f"  Feature columns for training: {X.columns.tolist()}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("  Features scaled successfully.")

    # 6. Train SVM Model
    print("\n--- Training SVM Model ---")
    print(f"  Kernel: {SVM_KERNEL}, C: {SVM_C}, Gamma: {SVM_GAMMA}")
    if SVM_KERNEL == 'poly':
        print(f"  Degree: {SVM_DEGREE}")
        model = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, degree=SVM_DEGREE, class_weight='balanced', probability=True, random_state=42)
    else:
        model = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, class_weight='balanced', probability=True, random_state=42)
    
    class_counts = np.bincount(y)
    print(f"  Class distribution in target (0s and 1s): {class_counts}")
    if len(class_counts) < 2:
        print("  Error: Target variable has only one class. Cannot train SVM for classification.")
        print("  This might happen if all future prices move in one direction or N_FUTURE_PERIODS is too large.")
        print("--- Script terminated ---")
        return
    
    try:
        print("  Starting model fitting...")
        model.fit(X_scaled, y)
        print("  SVM Model training completed successfully.")
    except Exception as e:
        print(f"  Error during SVM training: {e}")
        print("--- Script terminated ---")
        return

    # 7. Save Model and Scaler
    print("\n--- Saving Model and Scaler ---")
    degree_str = f"_Degree-{SVM_DEGREE}" if SVM_KERNEL == 'poly' else ""
    base_filename = f"{CRYPTO_ASSET}_{TIMEFRAME}_Kernel-{SVM_KERNEL}{degree_str}_C-{SVM_C}_Gamma-{str(SVM_GAMMA).replace('.', 'p')}_Target-{N_FUTURE_PERIODS}pds"
    
    model_filename = os.path.join(MODEL_SAVE_DIR, f"{base_filename}_MODEL.joblib")
    scaler_filename = os.path.join(MODEL_SAVE_DIR, f"{base_filename}_SCALER.joblib")

    try:
        joblib.dump(model, model_filename)
        print(f"  Model saved to: {model_filename}")
        joblib.dump(scaler, scaler_filename)
        print(f"  Scaler saved to: {scaler_filename}")
    except Exception as e:
        print(f"  Error saving model/scaler: {e}")

    print("\n--- SVM Model Training Script Finished ---")

if __name__ == "__main__":
    main()

