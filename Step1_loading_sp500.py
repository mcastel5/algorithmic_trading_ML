# importing libraries
import pandas as pd
import yfinance as yf
import os


# setting up table - only download content once
file_path = os.path.join(os.path.dirname(__file__), 'sp500_data.csv')
print(file_path)
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    df = df.set_index(['Date','Ticker'])
    df.index.names = ['date', 'ticker']
    df.columns = df.columns.str.lower()
    df.to_csv(file_path)
else:
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sp500['Symbol'] = sp500['Symbol'].str.replace('.','-')
    symbols_list = sp500['Symbol'].unique().tolist() # list of all s&p500 stocks
    end_date = '2023-09-27'
    start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8) # 8 years ago
    df = yf.download(tickers=symbols_list, start=start_date, end=end_date).stack()
    # Save the data to a file
    df.to_csv(file_path)


