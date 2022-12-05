import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


###read the data to pandas dataframe
data=pd.read_csv('Mall_Customers.csv')
#plt.show()
data=data.drop(['Genre'],axis='columns')
cost=[]
############using elbow method to obtain the optimal value for the number of clusters
for i in range(1,15):
    model=KMeans(i)
    model.fit(data)
    cost.append(model.inertia_)
plt.plot(cost)
#plt.show()
###########after cluster size 4 there is no much gain in cost from elbow method
model=KMeans(4)
label=model.fit_predict(data)
print(label)
data['cluster']=label
##########count the number of datapoints in each cluster
print(data['cluster'].value_counts())
#########visualising how well the clusters are implemented for each feature value
sns.pairplot(data,hue='cluster',palette='crest')
plt.show()

