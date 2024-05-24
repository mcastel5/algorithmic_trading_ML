# refer to Explain 1 for better understanding
from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import statsmodels.api as sm
import pandas as pd



data = pd.read_csv('features.csv',parse_dates=['date'])
data = data.set_index(['date','ticker'])

factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                             'famafrench',
                               start='2010')[0].drop('RF', axis=1) # [0] indicates first key which is for monthly factors, [1] is for yearly; do not consider RF (risk free return)

    #pandas-datareader: library to access historical financial data from various sources

# Fama French Factors
factor_data.index = factor_data.index.to_timestamp()
factor_data = factor_data.resample('ME').last().div(100) # convert factors from percentages to decimal
factor_data.index.name = 'date'
factor_data = factor_data.join(data['return_1m']).sort_index()
    # add the return at the end of the month for for each month's fama french factors

# remove stoks with less than 10 months of data (to avoid future error in model that requires certain amount of values)
observations = factor_data.groupby(level=1).size()
valid_stocks = observations[observations >= 10]
factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]


# calculate betas
betas = (factor_data.groupby(level=1, 
                            group_keys=False) # groups data by ticker and prefents group neames to be aded as keys
         .apply(lambda x: RollingOLS(   #rollingOLS performs rolling linear regression
                                    endog=x['return_1m'], # sets dependant variable (y) as one month return
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)), # independent variables (x) set as fama-french factors, with added constant
                                     window=min(24, x.shape[0]), #defines window size for rolling reg, smaller between 24 periods and available observations
                                     min_nobs=len(x.columns)+1) #ensures min observations is greater than number of independnat variables + 1
         .fit(params_only=True) # fits rrm, returns only regression parameters (betas)
         .params #extracts regression parameters from fitted model
         .drop('const', axis=1))) # drops constant term from regression parameters to retain only factor betas

# betas represent predictions for next month

# join betas with features data
data = (data.join(betas.groupby('ticker').shift())) # shift to account for next month phenomena
factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean())) # for each ticker apply factors and replace NA with averages
data = data.drop('adj close', axis=1) #adjusted close no longer needed
data = data.dropna() # drop any na
data.info()

# Ready to apply machine learning models!

data.to_csv('MLdata.csv')