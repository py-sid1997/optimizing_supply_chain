#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:27:34 2021

@author: siddharth
"""

#Analysing the new data provided in first week of Oct 2021

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime

import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics 

df_new=pd.read_excel("Final Sheet-5 year Data.xlsx")

df_total_orders = pd.read_excel("Total Orders data.xlsx")

df_prod = pd.read_excel("Prodution Data.xlsx")
df_loc = pd.read_excel("loom Data with location.xlsx")

df_prod=df_prod.merge(df_loc,on='Loom ID',how='left')




visits=df_prod.groupby(['OTN']).agg({'Running Length':['max'],'Length (Inches)':['max'],'Pending Work':['min'],'Posting Date':['max']})

visits=visits.reset_index()
visits.columns=['OTN', 'run_length','length','work','latest_date']

visits['dif']=visits['length']-visits['run_length']

in_work=visits[visits['dif']>10]


a=in_work[(in_work['latest_date']<'2022-01-01 00:00:00')&(in_work['latest_date']>='2021-01-01 00:00:00')]
b=in_work[(in_work['latest_date']<'2021-01-01 00:00:00')&(in_work['latest_date']>='2020-01-01 00:00:00')]
c=in_work[(in_work['latest_date']<'2020-01-01 00:00:00')&(in_work['latest_date']>='2019-01-01 00:00:00')]





ct=df_prod.groupby(['OTN','QS']).agg({'Running Length':['count']})

ct.columns=['OTN','QS','count']

ct=ct.reset_index()

ct_nex=ct.merge(df_new[['OTN No_','days']],left_on='OTN',right_on='OTN No_',how='left')


df_columns=df_total_orders.columns


# og_data=df_new[['Sales Order Date',
#                          'Actual Store Issue Date',
#                          'Actual Branch Issue Date',
#                          'Actual Weaver Issue Date',
#                          'Actual Off Loom Date',
#                          'Actual HO Receipt Date',
#                          'Ramgarh Receipt Date',
#                          'Actual Carpet Finish Date',
#                          'Original Ex India',
#                          'Rev_Ex India',
#                          'Original Ex Factory',
#                          'Rev_Ex Factory',
#                          'Posting Date',
#                          'Last Swapping Date',
#                          ]]

# og_data=df_total_orders[['Sales Order Date',
#                          'Actual Store Issue Date',
#                          'Actual Branch Issue Date',
#                          'Actual Weaver Issue Date',
#                          'Actual Off Loom Date',
#                          'Actual HO Receipt Date',
#                          'Ramgarh Receipt Date',
#                          'Actual Carpet Finish Date',
#                          'Original Ex India',
#                          'Rev_Ex India',
#                          'Original Ex Factory',
#                          'Rev_Ex Factory',
#                          'Posting Date',
#                          'Last Swapping Date',
#                          ]]


# mod_data=df_total_orders[['Order Priority',
#                          'Quality',
#                          'Design',
#                          'Size',
#                          'Shape',
#                          'Weaver Name',
#                          'Primary Style',
#                          'Quality Check',
#                          'Original Ex India',
#                          'Rev_Ex India',
#                          'Original Ex Factory',
#                          'Rev_Ex Factory',
#                          'Posting Date',
#                          'Last Swapping Date',
#                          ]]


df_new['weave time']=(df_new['Actual Off Loom Date'] )-(df_new['Actual Weaver Issue Date'] )
test = [str(x).split(' ')[0] for x in df_new['weave time']]
df_new['days']=test
df_new['days'] = (df_new['days'].astype(str).replace({'NaT': None})) 
df_new['days']=pd.to_numeric(df_new['days'])

df=df_new[['Quality','Design','Prod_ Cubage','Ground Color','days']]
df.dropna(inplace=True)
df=df[(df['days']>0)&(df['days']<400)]

#'Quality','Design','Prod_ Cubage','Ground Color'
X=df[['Ground Color']]
X = pd.get_dummies(X, columns=['Ground Color'])
#X=X.reset_index()

y=df['days']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training and 20% test

# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)


fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(X_test, predictions, color='k', label='Regression model')
ax.scatter(X, y, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
ax.set_ylabel('Weave time', fontsize=9)
ax.set_xlabel('Order Priority', fontsize=14)
ax.legend(facecolor='white', fontsize=11)
ax.text(0.55, 0.55, '$y = %.3f x_1 - %.2f $' % (model.coef_[0], abs(model.intercept_)), fontsize=17, transform=ax.transAxes)


plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
#Predict the response for test dataset
y_pred = lm.predict(X_test)













# a=df_new[(df_new['Actual Off Loom Date']<'2022-01-01 00:00:00')&(df_new['Actual Off Loom Date']>='2021-01-01 00:00:00')]
# b=df_new[(df_new['Actual Off Loom Date']<'2021-01-01 00:00:00')&(df_new['Actual Off Loom Date']>='2020-01-01 00:00:00')]
# c=df_new[(df_new['Actual Off Loom Date']<'2020-01-01 00:00:00')&(df_new['Actual Off Loom Date']>='2019-01-01 00:00:00')]
# d=df_new[(df_new['Actual Off Loom Date']<'2019-01-01 00:00:00')&(df_new['Actual Off Loom Date']>='2018-01-01 00:00:00')]
# e=df_new[(df_new['Actual Off Loom Date']<'2018-01-01 00:00:00')&(df_new['Actual Off Loom Date']>='2017-01-01 00:00:00')]
# f=df_new[(df_new['Actual Off Loom Date']<'2017-01-01 00:00:00')&(df_new['Actual Off Loom Date']>='2016-01-01 00:00:00')]
# g=df_new[(df_new['Actual Off Loom Date']<'2016-01-01 00:00:00')&(df_new['Actual Off Loom Date']>='2015-01-01 00:00:00')]
# h=df_new[(df_new['Actual Off Loom Date']<'2015-01-01 00:00:00')&(df_new['Actual Off Loom Date']>='2014-01-01 00:00:00')]
# i=df_new[(df_new['Actual Off Loom Date']<'2014-01-01 00:00:00')&(df_new['Actual Off Loom Date']>='2013-01-01 00:00:00')]
# j=df_new[(df_new['Actual Off Loom Date']<'2013-01-01 00:00:00')&(df_new['Actual Off Loom Date']>='2012-01-01 00:00:00')]
# k=df_new[(df_new['Actual Off Loom Date']<'2012-01-01 00:00:00')&(df_new['Actual Off Loom Date']>='2011-01-01 00:00:00')]
# l=df_new[(df_new['Actual Off Loom Date']<'2011-01-01 00:00:00')&(df_new['Actual Off Loom Date']>='2010-01-01 00:00:00')]
# m=df_new[(df_new['Actual Off Loom Date']<'2010-01-01 00:00:00')&(df_new['Actual Off Loom Date']>='2009-01-01 00:00:00')]


# col=df_new.columns

status=pd.Series(df_new['Border Color'].value_counts())

test_1=df_new['Quality'].value_counts()
test_1=test_1.reset_index()
sns.countplot(data=df_new,y='Quality',order=df_new.Quality.value_counts().iloc[:25].index)


check=df_new.groupby('Quality').days.mean()

df_new[df_new['days']>0].days.hist(bins=100)
plt.xticks(range(0,500,50))
plt.grid(False)

# for x in ["Actual Off Loom Date",'Actual Weaver Issue Date']:
#     og_data[x]=og_data[x].dt.strftime('%d/%m/%Y')


# for x in range(len(og_data['Actual Weaver Issue Date'])):
#     #(og_data.loc[x,'Sales Order Date'])
#     if ((og_data.loc[x,"Actual Off Loom Date"]=='01/01/1753') or(og_data.loc[x,"Actual Weaver Issue Date"]=='01/01/1753')):
#         og_data.loc[x,'days']=np.nan

# for x in range(len(og_data['Actual Weaver Issue Date'])):
#     #(og_data.loc[x,'Sales Order Date'])
#     if (not (og_data.loc[x,"Actual Off Loom Date"]=='01/01/1753') and(og_data.loc[x,"Actual Weaver Issue Date"]=='01/01/1753')):
#         og_data.loc[x,'weave time']=datetime.strptime(og_data.loc[x,'Actual Off Loom Date'] , '%d/%m/%Y')-datetime.strptime(og_data.loc[x,'Actual Weaver Issue Date'] , '%d/%m/%Y')





    

# og_data['weave time']= pd.Series()
# for x in range(len(og_data['Actual Weaver Issue Date'])):
#     #(og_data.loc[x,'Sales Order Date'])
#     if ( (og_data.loc[x,"Actual Off Loom Date"]=='01/01/1753') and(og_data.loc[x,"Actual Weaver Issue Date"]=='01/01/1753')):
#         og_data.loc[x,'weave time']=np.nan



#og_data[x]= og_data[x].replace('01/01/1753',np.NaN)

#og_data['Sales Order Date']=pd.strptime(og_data['Sales Order Date'], format="%d/%m/%y")
for y in og_data.columns:
    for x in range(len(og_data[y])):
        #(og_data.loc[x,'Sales Order Date'])
        if not (og_data.loc[x,y]=='01/01/1753'):
            og_data.loc[x,y]=datetime.strptime(og_data.loc[x,y] , '%d/%m/%Y')
        else:
            og_data.loc[x,y]=np.nan

plt.scatter(y=og_data['days'],xlim=(0,1000),kind='scatter', color='black')

# for y in og_data.columns:
#     for x in range(len(og_data[y])):
#         #(og_data.loc[x,'Sales Order Date'])
#         og_data.loc[x,y]=datetime.strptime(og_data.loc[x,y] , '%d/%m/%Y')

# Missing value count

# percent_missing = og_data.isnull().sum() * 100 / len(og_data)
# missing_value_df = pd.DataFrame({'column_name': og_data.columns,
#                                   'percent_missing': percent_missing})

# percent_missing = df_total_orders.isnull().sum() * 100 / len(df_total_orders)
# missing_value_df = pd.DataFrame({'column_name': df_total_orders.columns,
#                                   'percent_missing': percent_missing})



og_data['weave time']=og_data['Actual Off Loom Date']-og_data['Actual Weaver Issue Date']
test = [str(x).split(' ')[0] for x in og_data['weave time']]
og_data['days']=test
og_data['days'] = (og_data['days'].astype(str).replace({'NaT': None})) 
og_data['days']=pd.to_numeric(og_data['days'])

# status=pd.Series(df_total_orders['Current Location'].unique())

