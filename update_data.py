import os
import pandas as pd
from datetime import datetime
from download_data import download_ohlcv, save_ohlcv_to_csv
from config import proxies
import ccxt

# 配置
DATA_DIR = 'data'
EXCHANGE_NAME = 'binance'
TIMEFRAME = '1d'
MARKET_TYPE = 'spot'

def get_latest_timestamp(symbol, exchange_name=EXCHANGE_NAME, timeframe=TIMEFRAME, market_type=MARKET_TYPE):
    file_path = os.path.join(DATA_DIR, market_type, exchange_name, timeframe, f"{symbol.replace('/', '_')}.csv")
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    if df.empty or 'timestamp' not in df.columns:
        return None
    return df['timestamp'].max()

def update_symbol_data(exchange, symbol, timeframe=TIMEFRAME, market_type=MARKET_TYPE):
    latest_ts = get_latest_timestamp(symbol, exchange.id, timeframe, market_type)
    if latest_ts is not None:
        # 下载新数据
        new_ohlcv = download_ohlcv(exchange, symbol, timeframe, since=latest_ts + 1)
        if new_ohlcv:
            # 追加保存
            save_ohlcv_to_csv(new_ohlcv, symbol, timeframe, exchange.id, market_type)
            print(f"{symbol} updated from {latest_ts}")
        else:
            print(f"No new data for {symbol}")
    else:
        print(f"No existing data for {symbol}, skipping.")

def main():
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'proxies': proxies
    })
    # 遍历所有已下载的币对
    symbols_dir = os.path.join(DATA_DIR, MARKET_TYPE, EXCHANGE_NAME, TIMEFRAME)
    if not os.path.exists(symbols_dir):
        print("No data directory found.")
        return
    symbols = [f.replace('.csv', '').replace('_', '/') for f in os.listdir(symbols_dir) if f.endswith('.csv')]
    for symbol in symbols:
        update_symbol_data(exchange, symbol, TIMEFRAME, MARKET_TYPE)

if __name__ == "__main__":
    main()