import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


portfolio_df = pd.read_csv('strategy_performance.csv',parse_dates=['date'])
portfolio_df = portfolio_df.set_index('date')

spy = yf.download(tickers='SPY',
                  start='2015-01-01',
                  end=dt.date.today())
    # Downloading benchmark data for S&P500

spy_ret = np.log(spy[['Adj Close']] #takes natural log of the adjusted closing prices
                 ).diff( # computes difference between consecutive log prices to get log returns
                 ).dropna().rename({'Adj Close':'SPY Buy&Hold'}, axis=1)
    # calculating returns -- 1. calculates log returns 
portfolio_df = portfolio_df.merge(spy_ret,
                                  left_index=True,
                                  right_index=True)
    # merges strategy and benchmark data

portfolio_df.to_csv('strategy_vs_benchmark.csv')

# plotting results
plt.style.use('ggplot')
portfolio_cumulative_return = np.exp(np.log1p(portfolio_df #  # natural log of 1 plus each return - transforms % returns to log returns
                                              ).cumsum() #computes comulative sum of logs
                                              )-1  # converts comulative returns back to percentages
portfolio_cumulative_return[:'2023-09-29'].plot(figsize=(16,6)) # plots cumulative returns up to Sept. 29,2023
plt.title('Unsupervised Learning Trading Strategy Returns Over Time')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1)) # formats y-axis to display percentages
plt.ylabel('Return')
plt.savefig('Final_Performance.png') # save file
plt.show()