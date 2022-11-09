# call the appropriate libraries
import numpy as np
from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot
from numpy import cov
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import scipy.stats as stats
import pandas as pd

# generate data
# seed random number generator
seed(1)
# prepare data
data1 = 20 * randn(1000) + 100
data2 = data1 + (10 * randn(1000) + 50)
# print mean and std
print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# plot the sample data
pyplot.scatter(data1, data2)
pyplot.show()

# calculate covariance
covariance = cov(data1, data2)
print('The covariance matrix')
print(covariance)

# calculate Pearson's correlation
corr, _ = pearsonr(data1, data2)
print('Pearsons correlation: %.3f' % corr)
##calculate spearman correlation
corr, _ = spearmanr(data1, data2)
print('Spearmans correlation: %.3f' % corr)
######################chi square test
####Input as contigency table
####consider different pets bought by male and female
#######dog   cat  bird  total
##men   207  282  241   730
##women 234  242  232   708
##total 441   524  473  1438

###The aim of the test is to conclude whether the two variables( gender and choice of pet ) are related to each other.


from scipy.stats import chi2_contingency
print("CHI SQUARE TEST with HYPOTHESIS TESTING")
# defining the table
data = [[207, 282, 241], [234, 242, 232]]
stat, p, dof, expected = chi2_contingency(data)
######hyposthesis testing
# interpret p-value
alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
    print('reject H0 - have correlation with 95% confidence level')
else:
    print('accept H0 - Independent no correlation with 95% confidence level')


np.random.seed(6)
####generate possion distribution with lowest x value 18 and mean given by mu of the distribution and size of sample
population_ages1 = stats.poisson.rvs(loc=18, mu=35, size=150000)
population_ages2 = stats.poisson.rvs(loc=18, mu=10, size=100000)
population_ages = np.concatenate((population_ages1, population_ages2))

minnesota_ages1 = stats.poisson.rvs(loc=18, mu=30, size=30)
minnesota_ages2 = stats.poisson.rvs(loc=18, mu=10, size=20)
minnesota_ages = np.concatenate((minnesota_ages1, minnesota_ages2))

print( population_ages.mean(), ' is the population mean' )
print( minnesota_ages.mean(), ' is the sample mean' )
### we know that both samples comes from different distribution
#####Let's conduct a t-test at a 95% confidence level and see if it correctly rejects the null hypothesis
# that the sample comes from the same distribution as the population.
s, p = stats.ttest_1samp(a = minnesota_ages,               # Sample data
                 popmean = population_ages.mean())
print(s, 'is the test statistics')

### interpret the t-statistics
if s >= 0:
   print('Sample mean is larger than the population mean')
else:
   print('Sample mean is smaller than the population mean')

###if p value is less than 0.05 we reject the null hypothesis that both samples are same
if p < 0.05:
    print('This observation is statistically significant with 95% confidence.')
else:
    print('This observation is not statistically significant with 95% confidence.')



###performing min max normalization

data = {'weight':[300, 250, 800],
        'price':[3, 2, 5]}
df = pd.DataFrame(data)

print(df)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)
print('Normalized data')
print(normalized_data)

###standardization is process of converting data to z score value and spread is across median 0

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
standardized_data = scaler.fit_transform(df)
print('standardized data value')
print(standardized_data)

####normality check using Q-Q plot

np.random.seed(0)
data = np.random.normal(0,1, 1000)

import statsmodels.api as sm
import matplotlib.pyplot as plt

#create Q-Q plot with 45-degree line added to plot
fig = sm.qqplot(data, line='45')
plt.show()