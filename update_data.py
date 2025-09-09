import os
import pandas as pd
from datetime import datetime
from download_data import download_ohlcv, save_ohlcv_to_csv
from config import proxies
import ccxt

# 配置
DATA_DIR = 'data'

def get_latest_timestamp(symbol, exchange_name, timeframe, market_type):
    file_path = os.path.join(DATA_DIR, market_type, exchange_name, timeframe, f"{symbol.replace('/', '_')}.csv")
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    if df.empty or 'timestamp' not in df.columns:
        return None
    return df['timestamp'].max()

def update_symbol_data(exchange, symbol, timeframe, exchange_name, market_type):
    latest_ts = get_latest_timestamp(symbol, exchange_name, timeframe, market_type)
    if latest_ts is not None:
        # 下载新数据
        new_ohlcv = download_ohlcv(exchange, symbol, timeframe, since=latest_ts + 1)
        if new_ohlcv:
            # 追加保存
            save_ohlcv_to_csv(new_ohlcv, symbol, timeframe, exchange_name, market_type)
            print(f"{exchange_name} {market_type} {timeframe} {symbol} updated from {latest_ts}")
        else:
            print(f"No new data for {exchange_name} {market_type} {timeframe} {symbol}")
    else:
        print(f"No existing data for {exchange_name} {market_type} {timeframe} {symbol}, skipping.")

def main():
    if not os.path.exists(DATA_DIR):
        print("No data directory found.")
        return
    # 使用os.walk递归遍历目录结构
    for root, dirs, files in os.walk(DATA_DIR):
        # 只处理包含csv文件的目录
        if not files:
            continue
        # 检查目录深度，确保结构为data/market_type/exchange_name/timeframe/
        rel_path = os.path.relpath(root, DATA_DIR)
        parts = rel_path.split(os.sep)
        if len(parts) != 3:
            continue
        market_type, exchange_name, timeframe = parts
        # 初始化exchange对象
        exchange = getattr(ccxt, exchange_name)({
            'enableRateLimit': True,
            'proxies': proxies
        })
        for file in files:
            if file.endswith('.csv'):
                symbol = file.replace('.csv', '').replace('_', '/')
                update_symbol_data(exchange, symbol, timeframe, exchange_name, market_type)

if __name__ == "__main__":
    main()