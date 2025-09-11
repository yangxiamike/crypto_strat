import talib as ta
import pandas as pd

def _trend_ratio(df: pd.DataFrame, window: int = 14) -> pd.Series:
    is_up = df['close'] > df['close'].shift(1)
    up_count = is_up.rolling(window=window).sum()
    up_ratio = up_count / window
    trend_ratio = up_ratio
    return trend_ratio

def _consecutive_count(series: pd.Series, direct = 'up') -> pd.Series:
    count = []
    curr = 0
    for i in range(len(series)):
        if i == 0:
            count.append(0)
            continue
        if direct == 'up':
            if series.iloc[i] > series.iloc[i - 1]:
                curr += 1
            else:
                curr = 0
        else:
            if series.iloc[i] < series.iloc[i - 1]:
                curr += 1
            else:
                curr = 0
        count.append(curr)
    return pd.Series(count, index=series.index)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'close', 'high', 'low', and 'volume' columns.

    Returns:
    pd.DataFrame: DataFrame with added technical indicators.
    """
    required_columns = ['close', 'high', 'low', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")

    # Add Exponential Moving Averages (EMA)
    df['EMA_20'] = ta.EMA(df['close'], timeperiod=20)
    df['EMA_60'] = ta.EMA(df['close'], timeperiod=60)
    df['EMA_100'] = ta.EMA(df['close'], timeperiod=100)

    # Add Relative Strength Index (RSI)
    df['RSI_7'] = ta.RSI(df['close'], timeperiod=7)
    df['RSI_14'] = ta.RSI(df['close'], timeperiod=14)

    RSI_PRICE_DIVERG = df['RSI_7'].rolling(window=10).mean().pct_change(periods=10)
    RSI_PRICE_DIVERG *= df['close'].rolling(window=10).mean().pct_change(periods=10)
    df['RSI_PRICE_DIVERG'] = RSI_PRICE_DIVERG

    # 实体比例 (Balance of Power)
    df['BOP'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
    df['BOP_EMA_3'] = ta.EMA(df['BOP'], timeperiod=3)

    # 变化率，趋势速度
    df['ROCP'] = ta.ROCP(df['close'], timeperiod=5)
    df['ROCP'] = ta.ROCR(df['close'], timeperiod=10)

    # 多空k比例
    df['trend_ratio_5'] = _trend_ratio(df, window=5)
    df['trend_ratio_20'] = _trend_ratio(df, window=20)
    df['trend_ratio_60'] = _trend_ratio(df, window=60)

    # 斜率 (用线性回归斜率)
    df['slope_14'] = ta.LINEARREG_SLOPE(df['close'], timeperiod=14)
    df['slope_14_roc'] = df['slope_14'].pct_change(periods=10)

    # Add Moving Average Convergence Divergence (MACD)
    macd, macdsignal, macdhist = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['MACD_Diff'] = macdhist

    # Add Bollinger Bands 判断震荡
    upper, middle, lower = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['Bollinger_Width'] = (upper - lower) / middle
    df['Bollinger_Width_Percent'] = df['Bollinger_Width'] / df['Bollinger_Width'].rolling(window=60).mean()

    # Add Average True Range (ATR)
    df['ATR_20'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=20)
    df['ATR_20_Percent'] = df['ATR_20'] / df['ATR_20'].rolling(window=60).mean()

    # ADX
    df['ADX_14'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)

    # Add On-Balance Volume (OBV)
    df['OBV'] = ta.OBV(df['close'], df['volume'])
    df['obv_momentum_20'] = df['OBV'].pct_change(periods=20)

    # Add Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # 连续上涨下跌计数
    df['consecutive_up'] = _consecutive_count(df['close'], direct='up')
    df['consecutive_down'] = _consecutive_count(df['close'], direct='down')

    df['consecutive_down_low'] = _consecutive_count(df['low'], direct='down')
    df['consecutive_up_low'] = _consecutive_count(df['low'], direct='up')

    df['consecutive_down_high'] = _consecutive_count(df['high'], direct='down')
    df['consecutive_up_high'] = _consecutive_count(df['high'], direct='up')

    # prior high prior low
    df['prior_high'] = df['high'].rolling(window=100).max().shift(1)
    df['prior_low'] = df['low'].rolling(window=100).min().shift(1)

    # distance
    df['ema20_dist'] = (df['close'] - df['EMA_20']) / df['EMA_20']
    df['ema60_dist'] = (df['close'] - df['EMA_60']) / df['EMA_60']
    df['ema100_dist'] = (df['close'] - df['EMA_100']) / df['EMA_100']
    df['prior_high_dist'] = (df['close'] - df['prior_high']) / df['prior_high']
    df['prior_low_dist'] = (df['close'] - df['prior_low']) / df['prior_low']
    df['vwap_dist'] = (df['close'] - df['VWAP']) / df['VWAP']

    return df
