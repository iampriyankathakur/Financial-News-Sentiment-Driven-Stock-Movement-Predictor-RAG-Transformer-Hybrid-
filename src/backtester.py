import pandas as pd

def backtest(predictions, returns):

    strategy_returns = predictions.shift(1) * returns
    cumulative_strategy = (1 + strategy_returns).cumprod()
    cumulative_market = (1 + returns).cumprod()

    return cumulative_strategy, cumulative_market
