## Vocab list for Python

# if needed
pip install openpyxl

# Initialization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

from datetime import datetime, date, time       ==> managing dates
from dateutil.parser import parse               ==> extracting dates from strings imports

## LOAD DATA
data_folder = './Data/'
filename = ./folder/filename.csv        ==> interactive pathway (attention / et pas \ )
finename = data_folder + 'filename'
.read_csv('file_name', sep = ',')
.read_csv('file_name', decimal=',')     ==> permet de convertir 2,5 (string) en 2.5 (float)
.read_csv('file_name')
.read_csv('file_name', parse_dates = True )
NE PAS OUBLIER DE CONVERTIR LES DATAS NUMERIQUE POUR PAS QU'ELLES SOIENT DES STRINGS !!!! 
DATA_FOLDER = "MovieSummaries/folder1/"

pd.read_excel('file',skiprows=5,skipfooter=7, sheet_name='',names=[',,'])


pd.read_csv("filename", compression = 'gzip')

z = zipfile.ZipFile('filename.gz')
pd.read_csv(z.open("filename.csv")


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

# Loading and formatting
db['col'] = movies['col'].str.replace('$', '').astype(float)
db.copy()

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

def bootstrap_CI(data, nbr_draws):
    means = np.zeros(nbr_draws)
    data = np.array(data)

    for n in range(nbr_draws):
        indices = np.random.randint(0, len(data), len(data))
        data_tmp = data[indices] 
        means[n] = np.nanmean(data_tmp)

    return [np.nanpercentile(means, 2.5),np.nanpercentile(means, 97.5)]

## Rearrange
.drop("column", axis=1)
.drop("index", axis=0)
.drop("index")
.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
.reset_index(inplace=True)          ==> puts the index back into columns and create a new index 0,1,2... 
                                    ==> inplace = same variable, not creating a new one)

.sort_values("column", ascending=False)
db[db.col1=='value'][what_to_show]          ==> select one the rows where col1 == value
db[[var.startswith('') for var in db.col]]
db[db.col1=='value']['col2']                ==> returns a Series
db[db.col1=='value'][['col2','col3']]       ==> attention [[]], returns a DataFrame
db.loc[(df['col1'] == 'value')& ()]['col2']

db.query(expr)                  ==> 

# Change stuff dynamically according to the name of the column
for col in df.columns:
    if 'string_stuck' in col:
        df[col[:-1]] = task_4_df[col[:-1] + 'some_string'] + task_4_df[col]
    ==> col[:-1] is the col string truncated of the last character (col[:-3] => truncated of 3 characters, etc)

# Pivot
db.pivot_table(values='', index='', columns='', aggfunc='count',fill_value='NaN')   ==> will count the number of occurences in each nod
db.pivot(values='', index='', columns='')                                           ==> does not support multiple indexes

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

np.nanpercentile(db.col, 2.5)

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

# Multiple testing
https://multithreaded.stitchfix.com/blog/2015/10/15/multiple-hypothesis-testing/ 


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
&

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

# Mise en page
-- NEVER FORGET (label+units)
plt.xlabel('label')
plt.ylabel()
plt.title()

plt.ylim([range])
plt.xticks([])

*kwargs* 
figsize=(8, 8))

# Subplots
fig, ax = plt.subplots(nrows,ncols,sharex='all', sharey='all')  ==> creates the subplots spaces
fig.suptitle('')        ==> overall legends 
fig.supxlabel('')
fig.supylabel('')

plt.subplot(nrows,ncols,idx+1)          ==> active subplot is idx+1 (left upper to right lower)
plt.graph(data)                         ==> fill in with whaterver you want
    


# Graphs types
plt.hist(db,bins = 100)
db.hist(bins=100)

plt.boxplot()

sns.jointplot(x= , y= , kind="hex")
    ==> gives 2D map (scatter) of (x,y) plus histogramms
    ==> kind = hex, kde, reg

sns.boxplot(x=,y=,data)
sns.barplot()
sns.violinplot()

plt.errorbar(x= , y =, yerr='length', xerr=None)  => line plot with error bars a each point

.scatter(x='col1', y='col2')    ==> doesn't work
.plot(x="col1", y="col2", kind="scatter")

# Colors in pyplot
kwarg =  color
(you can pass the name or directly hte HEX value #xxxxxx)
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
...more https://www.delftstack.com/howto/python/colors-in-python/ 


https://seaborn.pydata.org/tutorial.html 


# Variables
dict = {}
dict['label'] = content
for idx,key in enumerate(dict)


# Linear regression
import statsmodels.formula.api as smf
mod = smf.ols(formula='time ~ C(a) + C(b)', data=df) 
        ==> ordinary least squares linear regression 
        ==> (C(a) is a categorical data)
        ==> terms are columns in dataframe
        ==> a*b = a + b + a:b (interaction term)
np.random.seed(2)       ==> for consistency
res = mod.fit()         ==> tries to fit the model 
res.summary()