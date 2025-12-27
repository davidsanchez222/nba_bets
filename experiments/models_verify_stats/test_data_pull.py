import numpy as np
import pandas as pd

#Data Source
import matplotlib.pyplot as plt
import yfinance as yf
import time

#Data viz
# import plotly.graph_objs as go

# Get Bitcoin data to stantiate
data = yf.download(tickers='BTC-USD', period = '7d', interval = '1m')
end_date = data.index[0]
new_start_date = end_date - pd.DateOffset(days=6)
combined = pd.DataFrame()
new_fetch = yf.download(tickers='BTC-USD', start=new_start_date, end=end_date, interval='1m')
combined = pd.concat([new_fetch,data], axis=0)
combined.sort_index(inplace=True)
combined = combined.drop_duplicates()


while (combined.shape[0] < 100000):
    print(combined.shape[0])
    end_date = pd.to_datetime(combined.index[0])
    new_start_date = end_date - pd.DateOffset(days = 6)
    new_fetch = yf.download(tickers='BTC-USD',start = new_start_date,end=end_date , interval = '1m')
    combined = pd.concat([new_fetch,combined],axis=0)
    combined.sort_index(inplace=True)
    combined = combined.drop_duplicates()
    time.sleep(10)


combined.to_csv('/home/danny/data/btc.csv')


combined = pd.read_csv('/home/danny/data/btc.csv',index_col=0)

