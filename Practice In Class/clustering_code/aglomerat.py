import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt


data = pd.read_csv('Mall_Customers.csv')
##########extracting two features for clustering, we can pass any number of features to perfom clustering
X=data.iloc[:,[3,4]].values
#########calling agglomerative clustering package in sklearn
model=AgglomerativeClustering(n_clusters=5,affinity="euclidean")
label=model.fit_predict(X)
#print(label)
########visualisation of the clusters generated
#########in the x, y plot , we pass x values which represents the cluster number
#cluster number  specifed by first coloumn in x axis and second coloumn in y axis is the feature we extracted
##0 is the first feature and 1 is the second
plt.scatter(X[label==0, 0], X[label==0, 1], s=50, marker='o', color='red')
plt.scatter(X[label==1, 0], X[label==1, 1], s=50, marker='o', color='blue')
plt.scatter(X[label==2, 0], X[label==2, 1], s=50, marker='o', color='green')
plt.scatter(X[label==3, 0], X[label==3, 1], s=50, marker='o', color='purple')
plt.scatter(X[label==4, 0], X[label==4, 1], s=50, marker='o', color='orange')
plt.show()

