import pandas as pd
import numpy as np
import pandas_ta

df = pd.read_csv('sp500_data.csv')
df = df.set_index(['date','ticker'])
# Garman Klass Volatility
df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

# RSI
df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20)) # length indicates 20 period window for calculations
    # average gains or losses over specified period of time
# Bollinger bands (low, middle, high, with normalized data)
#   adj_close is an adjusted closing price for the stock that provides a more accurate eval of its value 
df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0]) # middle band minus (typically 2) stdev --> identify oversold conditions (low price = buy)                                        
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1]) # simple moving average over 20 periods --> trend direction                                      
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2]) # middle band plus (typically 2) stdev --> identifies overbought conditions (high price = sell)

# calculate ATR with own function
def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        length=14)
    return atr.sub(atr.mean()).div(atr.std())
'''
        last line normalizes the values
        sub(atr.mean()) subtracts the mean from each value, centering them around zero
        div(atr.std) divifes the result by the stdev of the atr values, such that they have a unit variance
'''
df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)
    # standardized measure of volatility by averaging the true range
    # Helps determine stop close price (price that, if exceeded, should result in an order execution)
    # Ex. stock = $50, stop close = $44, if $44 reached, then sold to avoid further loss
# calculate MACD indicator
def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std()) # normalize for ML model
df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)
    # difference between short-term exponential moving average and long term ex.mov.avg (short term-long term)
    # if short term > long term --> buy (short-term price movement is becoming stronger than the long-term average, suggesting a new upward trend.)
    # if short term < long term --> sell (short-term price movement is weaker than the long-term average, signaling a potential downward trend)

# Dollar volume 
df['dollar_volume'] = (df['adj close']*df['volume'])/1e6 #1million stocks created per day
    # price of stock times how much is traded = market activity for stock, indicating attractiveness

print(df)
df.to_csv('calculations.csv')
