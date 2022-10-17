## 2022
Materials for Applied Data Analysis CS-401, Fall 2022


## Vocab list for Python

# Initialization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, date, time       ==> managing dates
from dateutil.parser import parse               ==> extracting dates from strings imports

# Load data
.read_csv('file_name', sep = ',')
.read_csv('file_name', decimal=',')     ==> permet de convertir 2,5 (string) en 2.5 (float)
.read_csv('file_name')
.read_csv('file_name', parse_dates = True )
NE PAS OUBLIER DE CONVERTIR LES DATAS NUMERIQUE POUR PAS QU'ELLES SOIENT DES STRINGS !!!! 
DATA_FOLDER = "MovieSummaries/folder1/"

# Loading dates and time
from datetime import datetime, date, time       ==> managing dates
from dateutil.parser import parse               ==> extracting dates from strings imports

datetime.strptime(var1, 'format')
var1 : db.col.iloc[0]
format :    '%m/%d/%y %H:%M'                   ==> works if the format is all the same in each row
            https://www.geeksforgeeks.org/python-datetime-strptime-function/
db.col.apply(lambda d: datetime.strptime(d, 'format'))
pd.to_datetime(db.col) 
pd.to_datetime(db.col, errors = "coerce")      ==> "coerce" set the errors to NaT (can be remove by dropping the .isnull() values)

# Quick Viz
db.index
db.columns
db.dtypes
db['column_name']
db['column_name1','column_name2']
db.head(10)     ==> attention different de db.head sans les ()
db.sample(10)
db[:3]          ==> shows only the 3 first rows
db['col1'][:3]
len(db)         ==> returns the rows size
db.size         ==> numbers of values (rows x columns) ? 
.isnull()       ==> renvoie un boolean TRUE/FALSE

# Sample the data
db.sample(n = 10,replace = True)                        ==> with replacement (2* le même datapoint dans le sample possible)
.sample(n = 10, replace = False)
df.sample(n = 10, replace = False, weights = df['col']) ==> unbalanced sampling

# Bootstrap Function
def bootstrap(data, n): # n number of bootstrap samples 
    means = ''
    sample_size = len(data)
    for i in range(n):
        x = data.sample(n=sample_size, replacement = 'True')
        means[i] = x.mean()
    means.sort_values(ascending = True)
    conf_int = ''
    conf_int[0] = means[int(n*0.025)+1]
    conf_int[1] = means[int(n*0.975)]
    return conf_int

# Rearrange
.drop("column", axis=1)
.drop("index", axis=0)
.drop("index")
.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')

.sort_values("column", ascending=False)
db[db.col1=='value'][what_to_show]          ==> select one the rows where col1 == value
db[db.col1=='value']['col2']                ==> returns a Series
db[db.col1=='value'][['col2','col3']]       ==> attention [[]], returns a DataFrame
db.loc[df['col1'] == 'value']['col2']

db.query(expr)                  ==> 

# Groups 
arrgh
db.groupby('column_name')               ==> this creates a 'DataFrameGroupBy' object
db.groupby(db.columname)                ==> this creates a 'DataFrameGroupBy' object
db.groupby('column_name')["column2"]    ==> this creates a 'SeriesGroupBy' object 
.get_group('name')

Aggregations Functions (gdb = 'DataFrameGroupBy' object)
gdb.agg('mean')                ==> returns a DataFrame 
gdb.agg(['mean']) 
gdb.agg(['mean','count']) 
count => returns the numbers of non-null values, here in each group
https://datascientyst.com/list-aggregation-functions-aggfunc-groupby-pandas/ 


# Stats
db.describe()       => returns a DataFrame (drops non-numerical columns)
db.info()           ==> non-null stuff
db.count()          ==> non-null counts
db.mean()           => returns a Series (drops non-numerical columns)
db.col.mean()       => returns usually a float
db['col'].hist(bins = 50)       ==> histogram

from statsmodels.stats import diagnostic
from scipy import stats
diagnostic.kstest_normal(db['col'].values, dist = 'norm')
        ==> H0 = the data is {‘norm’, ‘exp’} distributed
        ==> dist{‘norm’, ‘exp’} (which distribution you want to try to fit)
        ==> returns (ksstat,pvalues) i.e two differents tests
        ==> p_value < 0.05 -> we can reject the null hypothesis 

*CORRELATION*
stats.spearmanr(Serie1,Serie2)      ==> Spearman's correlation
stats.pearsonr(Serie1,Serie2)       ==> Pearson's correlation
        ==> H0 = the data is not correlated
        ==> p_value < 0.05 -> we can reject H0 -> the data IS correlated
        ==> the first value gives the correlation coeff (positive or negative correlation)

*T TEST*
stats.ttest_ind(Serie1,Serie2)
        ==> H0 = the two independent samples have identical average (expected) values

*BINOMIAL TEST*
from statsmodels.stats import proportion
statsmodels.stats.proportion.binom_test(count {==># of successes}, nobs (==># of observations), prop=0.5, alternative='two-sided')
        ==> H0 = probability of success = prop
        ==> p_value < 0.05 -> we can reject H0 -> probalility of success isn't prop
        ==> alternative [‘two-sided’, ‘smaller’, ‘larger’] (one-tailed test where we test is p < prop {smaller} or p> prop {larger} )

# String
.str.lower()    #case sensitivity

# Apply
db['column2'] = db['column1'].apply(lambda r: r.function()) 
r will refer to the successive element of column1 row by row 
db.apply(lambda r : f(r.col1,r.col2), axis = 1)     ==> apply functions to each rows (axis = 1)


# Merge
db1.merge(db2, on="column")
pd.merge(db1, db2, left_on = 'column1', right_on = 'column2')

# Boolean
x == y              ==> will return true of false
x==a | x==b         ==> OR

## Print
var1 = 1
var2 = 2
print('hello world', var1, "I'm happy", var2)       ==> hello world 1 I'm happy 2
print("hello {} world".format(var1))                ==> hello 1 world
You can choose a specific formattage pour les variables dans print
Ex : 2 decimales et en %
print("hello {:.2%} world".format(var1))            ==> hello 100.00% world
https://www.pyblog.in/programming/print-formmating-in-python/ 

# Iterate over the rows of a DataFrame
can be used to print stuff
.iterrows()                     => in a for loop, returns a tuple (index, row)
for x in db.iterrows():         => called this way, x will be a tuple (inusable :/ )
    do stuff
for idx, row in db.iterrows():  => here idx is a string and row a Series, where elements are callable
    do stuff
    print(row.col1)             => for ex you can call the column element 


## Plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.lmplot(data = df, x="clo1", y="col2")
sns.barplot(x="col1", y="col2", data=df)               ==> includes errors bars, which ones? 
plt.xlabel('label')
plt.ylabel()

.scatter(x='col1', y='col2')    ==> doesn't work
.plot(x="col1", y="col2", kind="scatter")
plt.ylim([range])
https://seaborn.pydata.org/tutorial.html 