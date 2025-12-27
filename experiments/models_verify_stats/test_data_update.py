import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import time

# Get Bitcoin data to stantiate
data = yf.download(tickers='BTC-USD', period = '4d', interval = '1m')
combined = pd.read_csv('/home/danny/data/btc.csv',index_col=0)
combined.to_csv('/home/danny/data/btcBAK.csv')

newcombined = pd.concat([combined,data], axis=0)
newcombined = newcombined.drop_duplicates()
newcombined.index = pd.to_datetime(newcombined.index, utc=True)
newcombined = newcombined.drop_duplicates()
newcombined.sort_index(inplace=True)


newcombined.to_csv('/home/danny/data/btc.csv')

