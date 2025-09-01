# ===== easy settings =================================================
CSV_FILE            = r"C:\Users\kevin\OneDrive\CRYPTO_BOTS\DATA\BTCUSDT_60m_20250101_to_20250519.csv"
INITIAL_CAPITAL     = 10000
TRADE_VALUE_USD     = 250
BTC_PER_SHARE       = 0.00001
COMMISSION_FRACTION = 0.001
MAX_OPEN_TRADES     = 25  # how many trades can be open at one time
SVM_MODELS_FOLDER   = r"C:\Users\kevin\OneDrive\CRYPTO_BOTS\SVM_Bot\SVM_models"
PNG_FOLDER          = r"C:\Users\kevin\OneDrive\CRYPTO_BOTS\SVM_backtest\Backtest_Results"
# ====================================================================

import os, glob, warnings
import pandas as pd, numpy as np, joblib, ta
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt, matplotlib.dates as mdates
warnings.filterwarnings('ignore', category=UserWarning)

# ---------- find model + scaler ----------

def load_joblibs(folder):
    files = glob.glob(os.path.join(folder, "*.joblib"))
    if len(files) != 2:
        raise FileNotFoundError("Need exactly two *.joblib files in SVM_models.")
    scaler_file = next((f for f in files if "scaler" in f.lower()), None)
    model_file  = next((f for f in files if "model"  in f.lower()), None)
    if not scaler_file or not model_file:
        scaler_file, model_file = sorted(files)
    return model_file, scaler_file

MODEL_FILE, SCALER_FILE = load_joblibs(SVM_MODELS_FOLDER)

def model_tag(path):
    base = os.path.splitext(os.path.basename(path))[0]
    tags = [p for p in base.split('_')
            if p.lower().startswith(("kernel-", "c-", "gamma-"))]
    return '_'.join(tags) if tags else 'model'

# ---------- data helpers ----------

def load_csv(path):
    df = pd.read_csv(path)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df.set_index('open_time', inplace=True)
    df.rename(columns={'open':'Open','high':'High','low':'Low',
                       'close':'Close','volume':'Volume'}, inplace=True)
    df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].apply(pd.to_numeric)
    return df.dropna()

def add_features(df):
    d = df.copy()
    d['Return_1bar']   = d['Close'].pct_change()
    d['EMA_12']        = ta.trend.EMAIndicator(d['Close'], 12).ema_indicator()
    d['EMA_26']        = ta.trend.EMAIndicator(d['Close'], 26).ema_indicator()
    d['MACD_Hist']     = ta.trend.MACD(d['Close']).macd_diff()
    d['RSI_14']        = ta.momentum.RSIIndicator(d['Close'], 14).rsi()
    d['Stoch_K_14']    = ta.momentum.StochasticOscillator(d['High'], d['Low'], d['Close']).stoch()
    d['BB_Width_20_2'] = ta.volatility.BollingerBands(d['Close']).bollinger_wband()
    d['ATR_14']        = ta.volatility.AverageTrueRange(d['High'], d['Low'], d['Close']).average_true_range()
    obv                = ta.volume.OnBalanceVolumeIndicator(d['Close'], d['Volume'])
    d['OBV_Change']    = obv.on_balance_volume().diff()
    d['Volume_Change'] = d['Volume'].diff()
    return d.dropna()

def scale_prices(df, unit):
    d = df.copy()
    for col in ['Open','High','Low','Close']:
        d[col] = d[col] * unit
    d['Volume'] = d['Volume'] / unit
    return d

# ---------- strategy ----------
class SvmStrategy(Strategy):
    feature_names = ['Return_1bar','EMA_12','EMA_26','MACD_Hist','RSI_14',
                     'Stoch_K_14','BB_Width_20_2','ATR_14','OBV_Change','Volume_Change']

    def init(self):
        self.model  = joblib.load(MODEL_FILE)
        self.scaler = joblib.load(SCALER_FILE)

    def next(self):
        feats = [getattr(self.data, n)[-1] for n in self.feature_names]
        if np.isnan(feats).any():
            return

        pred = self.model.predict(
            self.scaler.transform(pd.DataFrame([feats], columns=self.feature_names))
        )[0]
        price = self.data.Close[-1]
        cash  = self._broker._cash
        open_trades = len(self.trades)  # current number of open trades

        if pred == 1:
            if open_trades < MAX_OPEN_TRADES and cash >= TRADE_VALUE_USD:
                shares = int(TRADE_VALUE_USD / price)
                if shares > 0:
                    self.buy(size=shares)
        elif self.position.is_long:
            self.position.close()

# ---------- plotting ----------

def make_plot(stats, price_df, png_name):
    equity = stats['_equity_curve']['Equity']
    close_actual = price_df['Close'] / BTC_PER_SHARE
    bh_equity = INITIAL_CAPITAL * (close_actual / close_actual.iloc[0])

    alpha_val = stats.get('Alpha [%]', None)
    title_str = 'Equity vs Buy & Hold'
    if isinstance(alpha_val, float):
        title_str += f'   (Alpha {alpha_val:.2f}%)'

    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                           gridspec_kw={'height_ratios': [3, 1]})
    ax[0].set_title(title_str)
    ax[0].plot(equity.index, equity.values, label='Strategy Equity', color='tab:blue')
    ax[0].plot(bh_equity.index, bh_equity.values, label='Buy & Hold',
               color='gray', alpha=0.7)
    ax[0].set_ylabel('Equity ($)')
    ax[0].grid(True, linestyle=':')
    ax[0].legend(loc='upper right')         # legend moved

    ax[1].bar(price_df.index, price_df['Volume'], color='purple', width=0.02)
    ax[1].set_ylabel('Volume')
    ax[1].grid(True, linestyle=':')
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    os.makedirs(PNG_FOLDER, exist_ok=True)
    full_path = os.path.join(PNG_FOLDER, png_name)
    plt.savefig(full_path)
    plt.close()
    print('Chart saved to', full_path)

# ---------- summary ----------

def print_summary(s):
    keys = ['Equity Final [$]', 'Return [%]', 'Buy & Hold Return [%]',   # CAGR removed
            'Sharpe Ratio', 'Alpha [%]',
            'Max. Drawdown [%]', 'Win Rate [%]', 'Profit Factor', '# Trades']
    print('--- summary ---')
    for k in keys:
        v = s.get(k, 'n/a')
        if isinstance(v, float):
            print(f'{k:<25} {v:.2f}{"%" if "%" in k else ""}')
        else:
            print(f'{k:<25} {v}')
    print('---------------')

# ---------- main ----------

def main():
    raw   = load_csv(CSV_FILE)
    feat  = add_features(raw)
    trade = scale_prices(feat, BTC_PER_SHARE)

    bt = Backtest(trade, SvmStrategy,
                  cash=INITIAL_CAPITAL,
                  commission=COMMISSION_FRACTION,
                  exclusive_orders=False)  # allow many trades at once

    stats = bt.run()
    print_summary(stats)

    csv_tag   = os.path.splitext(os.path.basename(CSV_FILE))[0]
    modelinfo = model_tag(MODEL_FILE)
    png_name  = f"{csv_tag}_{modelinfo}_backtest.png"
    make_plot(stats, trade, png_name)

if __name__ == '__main__':
    main()
