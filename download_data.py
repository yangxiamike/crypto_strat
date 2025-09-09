import os
import logging
import pandas as pd
from datetime import datetime
from config import proxies
import asyncio
import sys
import ccxt
import time
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    filename='data/download_log.txt',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def find_earliest_ohlcv(exchange, symbol, timeframe):
    left = int(datetime(2000, 1, 1).timestamp() * 1000)
    right = exchange.milliseconds()
    earliest = None
    last_earliest = None
    max_attempts = 100
    attempts = 0

    while left < right and attempts < max_attempts:
        mid = (left + right) // 2
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=mid, limit=1)
        if ohlcv:
            earliest = ohlcv[0][0]
            if last_earliest == earliest:
                break
            right = earliest - 1
            last_earliest = earliest
        else:
            left = mid + 1
        attempts += 1
    return earliest

def download_ohlcv(exchange, symbol, timeframe, since, limit=1000):
    exchange_name = exchange.id
    if not exchange.has['fetchOHLCV']:
        raise ValueError(f"{exchange_name} does not support fetching OHLCV data.")

    if isinstance(since, str):
        since_dt = datetime.strptime(since, "%Y-%m-%d")
        since_ms = int(since_dt.timestamp() * 1000)
    else:
        since_ms = since
    
    try:
        earliest = find_earliest_ohlcv(exchange, symbol, timeframe)
        if earliest and earliest > since_ms:
            since_ms = earliest
    except Exception:
        pass

    all_ohlcv = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since_ms, limit)
        except Exception as e:
            logging.error(f"fetch_ohlcv error: {e}")
            return all_ohlcv
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        last_timestamp = ohlcv[-1][0]
        print(f"最新K线时间: {datetime.utcfromtimestamp(last_timestamp / 1000)}")
        if since_ms >= last_timestamp:
            break
        since_ms = last_timestamp + 1
        if len(ohlcv) < limit:
            break
    return all_ohlcv

def filter_usdt_crypto_symbols(markets, market_type='all'):
    fiat_currencies = {
        'USD', 'EUR', 'CNY', 'JPY', 'KRW', 'GBP', 'AUD', 'CAD', 'CHF', 'SGD', 'HKD', 'NZD', 'THB', 'MYR', 'IDR', 'INR', 'RUB', 'TRY', 'BRL', 'ZAR', 'MXN', 'PHP', 'VND', 'PLN', 'UAH', 'AED',
        'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'PAX', 'SUSD', 'GUSD', 'EURT', 'USDP'
    }
    spot_list = []
    swap_list = []
    for symbol, info in markets.items():
        base = info.get('base')
        quote = info.get('quote')
        # 现货：只要USDT结尾的交易对，排除法币和稳定币，只保留加密货币
        if symbol.endswith('/USDT'):
            if base not in fiat_currencies and quote == 'USDT':
                if info.get('spot'):
                    spot_list.append(symbol)
        # 永续合约：OKX等格式通常为 "BTC/USDT:USDT"
        if info.get('swap'):
            if base not in fiat_currencies and quote == 'USDT':
                swap_list.append(symbol)
    return spot_list, swap_list

def save_ohlcv_to_csv(ohlcv, symbol, timeframe, exchange_name, market_type=None):
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # 抛弃最后一行，最新一行没有更新完成
    df = df.iloc[:-1]

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    if market_type:
        dir_path = f"data/{market_type}/{exchange_name}/{timeframe}"
    else:
        dir_path = f"data/{exchange_name}/{timeframe}"
    os.makedirs(dir_path, exist_ok=True)
    file_path = f"{dir_path}/{symbol.replace('/', '_')}.csv"
    df.to_csv(file_path, index=False)
    print(f"Saved OHLCV data to {file_path}")

def download_all_data(exchange, timeframes, since, limit=1000, market_type='all'):
    exchange.load_markets()
    spot_list, swap_list = filter_usdt_crypto_symbols(exchange.markets, market_type=market_type)
    print(f"筛选后现货USDT交易对数量: {len(spot_list)}")
    print(f"筛选后永续USDT交易对数量: {len(swap_list)}")

    if market_type == 'spot':
        symbol_list = spot_list
    elif market_type == 'swap':
        symbol_list = swap_list
    else:
        symbol_list = spot_list + swap_list

    failed = []

    total = len(symbol_list) * len(timeframes)
    with tqdm(total=total, desc="Downloading OHLCV") as pbar:
        for symbol in symbol_list:
            info = exchange.markets[symbol]
            mtype = 'spot' if info.get('spot') else 'swap' if info.get('swap') else 'unknown'
            for tf in timeframes:
                try:
                    ohlcv = download_ohlcv(exchange, symbol, tf, since, limit)
                    if ohlcv:
                        save_ohlcv_to_csv(ohlcv, symbol, tf, exchange.id, mtype)
                    else:
                        msg = f"NO DATA: {exchange.id} {mtype} {symbol} {tf}"
                        logging.warning(msg)
                        failed.append((symbol, tf, mtype, "no data"))
                except Exception as e:
                    msg = f"FAILED: {exchange.id} {mtype} {symbol} {tf}: {e}"
                    logging.error(msg)
                    failed.append((symbol, tf, mtype, str(e)))
                pbar.update(1)

    if failed:
        print("\n下载失败的币对/周期：")
        for item in failed:
            print(item)
        logging.error(f"FAILED SUMMARY: {failed}")
    else:
        print("\n全部下载任务完成，无失败。")
        logging.info("All downloads completed successfully.")

def download_failed_data(since, limit=1000):
    # 读取日志文件，提取失败的symbol, tf, mtype, exchange
    failed = []
    if not os.path.exists('download_log.txt'):
        print("No log file found.")
        return

    with open('download_log.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if "FAILED:" in line:
                # 例如: FAILED: binance spot BTC/USDT 1h: error message
                try:
                    parts = line.strip().split()
                    idx = parts.index("FAILED:")
                    exchange = parts[idx + 1]
                    mtype = parts[idx + 2]
                    symbol = parts[idx + 3]
                    tf = parts[idx + 4].replace(":", "")
                    failed.append((exchange, mtype, symbol, tf))
                except Exception:
                    continue

    if not failed:
        print("No failed records found in log.")
        return

    for exchange_name, mtype, symbol, tf in failed:
        try:
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({
                'enableRateLimit': True,
                'proxies': proxies
            })
            exchange.load_markets()
            if symbol not in exchange.markets:
                print(f"{exchange_name} {symbol} not in markets, skip.")
                continue
            ohlcv = download_ohlcv(exchange, symbol, tf, since, limit)
            if ohlcv:
                save_ohlcv_to_csv(ohlcv, symbol, tf, exchange_name, mtype)
                logging.info(f"RETRY SUCCESS: {exchange_name} {mtype} {symbol} {tf}")
            else:
                logging.warning(f"RETRY NO DATA: {exchange_name} {mtype} {symbol} {tf}")
            exchange.close()
        except Exception as e:
            logging.error(f"RETRY FAILED: {exchange_name} {mtype} {symbol} {tf}: {e}")
            print(f"Retry failed: {exchange_name} {mtype} {symbol} {tf}: {e}")

if __name__ == "__main__":
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'proxies': proxies
    })
    # download_all_data(
    #     exchange,
    #     timeframes=['1h', '1d'],
    #     since=int(datetime(2017, 1, 1).timestamp() * 1000),
    #     limit=1000,
    #     market_type='spot'
    # )

    ohlcv = download_ohlcv(
        exchange,
        symbol='BTC/USDT',
        timeframe='15m',
        since=int(datetime(2021, 1, 1).timestamp() * 1000),
        limit=1000,
    )

    save_ohlcv_to_csv(ohlcv, 'BTC/USDT', '15m', exchange.id, 'spot')