import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('clusters_data.csv',parse_dates=['date'])
data = data.set_index(['date','ticker'])

def plot_clusters(data):
    # dataframe for each cluster
    cluster_0 = data[data['cluster']==0]
    cluster_1 = data[data['cluster']==1]
    cluster_2 = data[data['cluster']==2]
    cluster_3 = data[data['cluster']==3]

    # create plot for each cluster, using ATR and RSI
    plt.scatter(cluster_0.iloc[:,5] , cluster_0.iloc[:,1] , color = 'red', label='cluster 0')
    plt.scatter(cluster_1.iloc[:,5] , cluster_1.iloc[:,1] , color = 'green', label='cluster 1')
    plt.scatter(cluster_2.iloc[:,5] , cluster_2.iloc[:,1] , color = 'blue', label='cluster 2')
    plt.scatter(cluster_3.iloc[:,5] , cluster_3.iloc[:,1] , color = 'black', label='cluster 3')
    
    # display graph
    plt.legend()
    plt.show()
    return

plt.style.use('ggplot')

for i in data.index.get_level_values('date').unique().tolist():
    g = data.xs(i, level=0)
    plt.title(f'Date {i}') # i indicates the month
    plot_clusters(g)

# Problem: clusters are random for their position changes over the course of time
# cluster one lower, the in the middle, then higher
    
# Observe cluster 3 i  the one with highest RSI