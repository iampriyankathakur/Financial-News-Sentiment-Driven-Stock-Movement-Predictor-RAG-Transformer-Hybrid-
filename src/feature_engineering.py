import ta

def add_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    df['sma'] = df['Close'].rolling(10).mean()
    return df.dropna()
