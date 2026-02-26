import yfinance as yf
from newsapi import NewsApiClient
import pandas as pd

newsapi = NewsApiClient(api_key="YOUR_API_KEY")

def fetch_news(query="Reliance"):
    articles = newsapi.get_everything(q=query,
                                      language='en',
                                      sort_by='publishedAt')
    return [a['title'] + ". " + str(a['description']) for a in articles['articles']]

def fetch_stock(symbol="RELIANCE.NS"):
    df = yf.download(symbol, start="2020-01-01")
    return df
