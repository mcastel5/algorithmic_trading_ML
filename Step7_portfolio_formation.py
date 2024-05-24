from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings


data = pd.read_csv('clusters_data.csv',parse_dates=['date'])
data = data.set_index(['date','ticker'])
# selecting stocks from cluster 3
filtered_df = data[data['cluster']==3].copy()
filtered_df = filtered_df.reset_index(level=1)
filtered_df.index = filtered_df.index+pd.DateOffset(1) #add one day, going to next month
filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])
dates = filtered_df.index.get_level_values('date').unique().tolist()

fixed_dates = {}
for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()
# list of stocks to invest in for next month

# portfolio optimization function - assigning weight to stocks
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

def optimize_weights(prices,lower_bound=0):
    returns = expected_returns.mean_historical_return(prices=prices, frequency=252,)
        # mean historical returns of stocks using daily prices, annualizing them (252 trading days per year)                                      
    cov = risk_models.sample_cov(prices=prices, frequency=252) 
        # calculates sample covarience matrix of stock prices, also annualized
    ef = EfficientFrontier(expected_returns=returns, # initializes Efficient Frontier Object
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, .1), # restricts weights of stocks withon domain
                           solver='SCS') # finds weights
    
    weights = ef.max_sharpe() # optimizes to maximize Sharpe ratio (measures risk-adjusted return)
    return ef.clean_weights()

# Download fresh daily prices data for shortenned list of stocks

stocks = data.index.get_level_values('ticker').unique().tolist()

new_df = yf.download(tickers=stocks,
                     start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                     end=data.index.get_level_values('date').unique()[-1])

# Calculate weights for stocks each day
returns_dataframe = np.log(new_df['Adj Close']).diff()

portfolio_df = pd.DataFrame() # empty frame to store performance data

for start_date in fixed_dates.keys(): # loop through fixed dates
    try:
        end_date = (pd.to_datetime(start_date)+pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d') # calculates end date for each month
        cols = fixed_dates[start_date] # retrieves columns to be used for current start date
        optimization_start_date = (pd.to_datetime(start_date)-pd.DateOffset(months=12)).strftime('%Y-%m-%d')
        # defines start and end days for the optimization period
        optimization_end_date = (pd.to_datetime(start_date)-pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        optimization_df = new_df[optimization_start_date:optimization_end_date]['Adj Close'][cols]
            # Filters adjusted closing prices for optimization period and selected columns
        success = False
        

# optimizing weights
        try:
            weights = optimize_weights(prices=optimization_df, # Attempts to optimize portfolio weights
                                        lower_bound=round(1/(len(optimization_df.columns)*2),3)) # sets lower bound based on number of columns
            weights = pd.DataFrame(weights, index=pd.Series(0))
            # returns list with percentage of each stock in portfolio
            success = True
        except:
            print(f'Max Sharpe Optimization failed for {start_date}, Continuing with Equal-Weights')
        if success==False:
            weights = pd.DataFrame([1/len(optimization_df.columns) for i in range(len(optimization_df.columns))],
                                     index=optimization_df.columns.tolist(),
                                     columns=pd.Series(0)).T # applies equal weights when max sharpe optimization failed
        
        #calculating portfolio returns
        temp_df = returns_dataframe[start_date:end_date] # filters daily returns data for the current month
        temp_df = temp_df.stack().to_frame('return').reset_index(level=0)\
                   .merge(weights.stack().to_frame('weight').reset_index(level=0, drop=True), # merges weights dataframe with returns dataframe
                          left_index=True,
                          right_index=True)\
                   .reset_index().set_index(['Date', 'Ticker']).unstack().stack() #sets multi-index
        # returns table with daily return of each stock plus its weight for the month
        
        temp_df.index.names = ['date', 'ticker'] # renames indexes
        temp_df['weighted_return'] = temp_df['return']*temp_df['weight'] # adds column of weighted return
        temp_df = temp_df.groupby(level=0)['weighted_return'].sum().to_frame('Strategy Return') # groups by date and sums weighted returns to get total return for the strtegy
        portfolio_df = pd.concat([portfolio_df, temp_df], axis=0) # concatenates results for the current month to the portfolio dataframe
    
    except Exception as e: # handles exceptions to avoid error
        print("")

portfolio_df = portfolio_df.drop_duplicates()
portfolio_df.to_csv('strategy_performance.csv')
