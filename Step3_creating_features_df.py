import pandas as pd

df = pd.read_csv('calculations.csv',parse_dates=['date'])
df = df.set_index(['date','ticker'])

last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open','high', 'low', 'close']]
    # we only want the columns to input in machine learning model (calculations and adj close)

dollar_volume_list = df.unstack('ticker')['dollar_volume'].resample('ME').mean().stack('ticker',future_stack = True).to_frame('dollar_volume')
    # list of avg dollar volume for stocks over each month

features_df = df.unstack()[last_cols].resample('ME').last().stack('ticker',future_stack = True)

data = pd.concat([dollar_volume_list,features_df],axis=1).dropna() #concatonate list and drop na

data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack(future_stack = True))
    # list of aggregated dollar volume over 5 years for each stock

data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))
    # rank according to monthly dollar volume

data = data[data['dollar_vol_rank']<150]
    # only select 150 most liquid stocks

data = data.drop(['dollar_volume', 'dollar_vol_rank'], axis=1)
    #drop dollar_volume and dollar_vol_rank columns

# Calculating monthly returns for different time horizons
    # increases roubustness of data and identify momentum patterns for each stock
def calculate_returns(df):

    outlier_cutoff = 0.005  #cutoff for outliers (99.5 percentile)

    lags = [1, 2, 3, 6, 9, 12] # month lags to calculate momentum patterns

    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close']  # return column for each lag
                              .pct_change(lag) # calculates percentage change between current value and value at specifies number of periods (lags) before
                              .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                     upper=x.quantile(1-outlier_cutoff))) # remove outliers
                              .add(1) # +1 to percentage change for compounded growth rate calculations (5% = 1.05)
                              .pow(1/lag) # anualizes returns for lags in months
                              .sub(1)) # returns to percentage
    return df
data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna() #group by ticker, after applying function

data.to_csv('features.csv')

