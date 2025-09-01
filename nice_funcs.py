import time
from datetime import datetime
from typing import List, Dict, Any
import ccxt
import pandas as pd
import os
import requests

def close_all_bybit_positions(api_key: str, api_secret: str, do_not_trade_list: List[str] = None) -> None:
    """
    Close all positions in Bybit except for tokens in DO_NOT_TRADE_LIST
    """
    if do_not_trade_list is None:
        do_not_trade_list = []
        
    cprint(f'🌙 Closing all Bybit positions...', 'cyan', attrs=['bold'])
    
    # Initialize Bybit exchange
    try:
        exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        cprint(f'✅ Connected to Bybit', 'green')
    except Exception as e:
        cprint(f'❌ Failed to connect to Bybit: {str(e)}', 'red')
        return
    
    # Get all positions
    try:
        positions = exchange.fetch_positions()
        
        if not positions:
            cprint('🔍 No positions to close', 'yellow')
            return
            
        cprint(f'📊 Found {len(positions)} open positions', 'blue')
    except Exception as e:
        cprint(f'❌ Error fetching positions: {str(e)}', 'red')
        return
    
    # Loop through all positions and close them
    closed_count = 0
    skipped_count = 0
    
    for position in positions:
        if not float(position['contracts']) or float(position['contracts']) == 0:
            continue
            
        symbol = position['symbol']
        side = position['side']
        amount = abs(float(position['contracts']))
        
        # Skip protected tokens
        if any(protected_token in symbol for protected_token in do_not_trade_list):
            cprint(f'🔒 Skipping protected token: {symbol}', 'yellow')
            skipped_count += 1
            continue
        
        try:
            # Close position - create order in opposite direction
            close_side = 'sell' if side == 'long' else 'buy'
            cprint(f'🔄 Closing {side} position for {symbol}...', 'magenta')
            
            exchange.create_market_order(
                symbol=symbol,
                side=close_side,
                amount=amount,
                params={'reduceOnly': True}
            )
            
            cprint(f'✅ Closed {side} position for {symbol}', 'green')
            closed_count += 1
            time.sleep(1)  # Small delay between closures
            
        except Exception as e:
            cprint(f'❌ Error closing {symbol}: {str(e)}', 'red')
    
    # Print summary
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cprint(f'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'white')
    cprint(f'📋 Position Closure Summary ({timestamp})', 'cyan', attrs=['bold'])
    cprint(f'✅ Successfully closed: {closed_count}', 'green')
    cprint(f'🔒 Protected/skipped: {skipped_count}', 'yellow')
    cprint(f'💹 Total processed: {closed_count + skipped_count}', 'blue')
    cprint(f'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'white')
    
    if closed_count > 0:
        cprint(f'✨ Finished closing all eligible positions', 'green', attrs=['bold'])
    else:
        cprint(f'ℹ️ No positions were closed', 'yellow')

def fetch_wallet_holdings_bybit(api_key: str = None, api_secret: str = None) -> pd.DataFrame:
    """
    Fetch wallet holdings from Bybit and return as a DataFrame
    """
    cprint(f'🔍 Fetching Bybit wallet holdings...', 'cyan', attrs=['bold'])
    
    # Use provided API keys or try to get from environment
    if api_key is None:
        api_key = os.getenv('BYBIT_API_KEY')
        if not api_key:
            cprint('❌ No API key provided or found in environment', 'red')
            return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value'])
    
    if api_secret is None:
        api_secret = os.getenv('BYBIT_API_SECRET')
        if not api_secret:
            cprint('❌ No API secret provided or found in environment', 'red')
            return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value'])
    
    try:
        # Initialize Bybit exchange
        exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        
        # Fetch balances
        balances = exchange.fetch_balance()
        
        # Extract relevant data
        assets = []
        
        for currency, data in balances['total'].items():
            if float(data) > 0:
                # Get ticker info for USD value calculation
                try:
                    ticker = None
                    if currency != 'USDT' and currency != 'USD':
                        try:
                            ticker = exchange.fetch_ticker(f'{currency}/USDT')
                        except:
                            try:
                                ticker = exchange.fetch_ticker(f'{currency}/USD')
                            except:
                                pass
                    
                    usd_value = 0
                    if ticker:
                        usd_value = float(data) * ticker['last']
                    elif currency == 'USDT' or currency == 'USD':
                        usd_value = float(data)
                    
                    assets.append({
                        'Mint Address': currency,
                        'Amount': float(data),
                        'USD Value': usd_value
                    })
                except Exception as e:
                    cprint(f'⚠️ Error getting price for {currency}: {str(e)}', 'yellow')
                    assets.append({
                        'Mint Address': currency,
                        'Amount': float(data),
                        'USD Value': 0
                    })
        
        # Create DataFrame
        df = pd.DataFrame(assets)
        
        # Filter out assets with very small USD values (dust)
        if not df.empty:
            df = df[df['USD Value'] > 0.05]
        
        # Sort by USD value
        if not df.empty:
            df = df.sort_values(by='USD Value', ascending=False)
        
        # Print summary
        if not df.empty:
            total_usd = df['USD Value'].sum()
            cprint(f'💰 Total portfolio value: ${total_usd:.2f}', 'green')
            cprint(f'🪙 Found {len(df)} assets with value > $0.05', 'blue')
        else:
            cprint('⚠️ No assets found with significant value', 'yellow')
        
        return df
    
    except Exception as e:
        cprint(f'❌ Error fetching wallet holdings: {str(e)}', 'red', 'on_white')
        return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value'])

# Helper function for colored terminal output
def cprint(text, color, attrs=None):
    """
    Print colored text in terminal
    Wrapper around termcolor's cprint function, with fallback if not available
    """
    try:
        from termcolor import colored, cprint as termcolor_cprint
        if attrs:
            print(colored(text, color, attrs=attrs))
        else:
            print(colored(text, color))
    except ImportError:
        # Fallback if termcolor is not available
        print(text)
