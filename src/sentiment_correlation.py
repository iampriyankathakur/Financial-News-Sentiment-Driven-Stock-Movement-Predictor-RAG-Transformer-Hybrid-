import pandas as pd

def sentiment_correlation(sentiment_dict):

    df = pd.DataFrame(sentiment_dict)
    corr = df.corr()
    return corr
