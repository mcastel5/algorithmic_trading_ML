# Read Explain 2 for explanation
#ML to predict which stocks to inlude in the portfolio and predict weight
# using clustering

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

data = pd.read_csv('MLdata.csv',parse_dates=['date'])
data = data.set_index(['date','ticker'])

target_rsi_values = [30,45,55,70]
initial_centroids = np.zeros((len(target_rsi_values),18)) # we have 18 columns of features
initial_centroids[:, 1] = target_rsi_values  #6th colum is rsi values

def get_clusters(df):
    df['cluster'] = KMeans(n_clusters=4,
                           random_state=0,
                           init=initial_centroids).fit(df).labels_ 
    return df
# Problem with init = 'random': clusters are random for their position changes over the course of time
# We are basing our predictions on stock momentum, so we will base our initial centroid posions on ATR metrics (related to momentum)
# manually assigned centroids keep the low one cluster one, middle cluster 2 etc

data = data.dropna().groupby('date',group_keys=False).apply(get_clusters)

data.to_csv('clusters_data.csv')
