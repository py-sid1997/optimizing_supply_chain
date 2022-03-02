#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: siddharth
"""

#Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn import linear_model



df_total_orders = pd.read_csv("total ORder Rug List for Share.csv")

#df_columns=df_total_orders.columns
#tab_new=(df_total_orders.replace('01-01-1753 00:00:00',np.NaN)).describe(include='all')


og_data= df_total_orders[[
    'Sales Order Date','Actual Weaver Issue Date','Actual Off Loom Date',
    'Ramgarh Receipt Date','Actual Carpet Finish Date', 'OTN No','Order Priority',
    'Shape','Size','Quality','Design','Fiber Content','Primary Style',
    'Location Code','Actual WareHouse Date','Rev Ex Factory']]

og_data['Sales Order Date'] = og_data['Sales Order Date'].str.split(' ').str[0]
og_data['Sales Order Date']= og_data['Sales Order Date'].replace('01-01-1753',np.NaN)
og_data['Sales Order Date']= pd.to_datetime(og_data['Sales Order Date'], format="%d/%m/%y")

og_data['Year'] = og_data['Sales Order Date'].dt.year
og_data['Order Month'] = og_data['Sales Order Date'].dt.month
#og_data['Month Name'] = og_data['Sales Order Date'].dt.month_name()
og_data=og_data.replace('01-01-1753 00:00:00',np.NaN)

#og_data['Rev Ex Factory'] = og_data['Rev Ex Factory'].str.split(' ').str[0]
#og_data['Rev Ex Factory']= og_data['Rev Ex Factory'].replace('01-01-1753',np.NaN)
og_data['Rev Ex Factory']=pd.to_datetime(og_data['Rev Ex Factory'], format="%d/%m/%y %H:%M")
og_data['Actual Weaver Issue Date']=pd.to_datetime(og_data['Actual Weaver Issue Date'], format="%d/%m/%y %H:%M")
og_data['Month Name'] = og_data['Actual Weaver Issue Date'].dt.month_name()

#tab=og_data.describe(include='all')

rand=og_data[og_data['Rev Ex Factory']>og_data['Actual Weaver Issue Date']]

# percent_missing = og_data.isnull().sum() * 100 / len(og_data)
# missing_value_df = pd.DataFrame({'column_name': og_data.columns,
#                                  'percent_missing': percent_missing})




order_seven_years= rand[rand['Year'].isin([2013,2014,2015,2016,2017,2018,2019])]
order_seven_years=order_seven_years.replace('01-01-1753 00:00:00',np.NaN)

order_seven_years=order_seven_years[order_seven_years['Quality']=='8/8 RWB']
# order_seven_years=order_seven_years[order_seven_years['Location Code']=='LOC-007']


#weave time= 'Actual Weaver Issue Date'-'Rev Ex Factory'
weave_time = pd.DataFrame()
weave_time['OTN No']= order_seven_years['OTN No']
weave_time[['Actual Weaver Issue Date']]= order_seven_years['Actual Weaver Issue Date']
weave_time['Rev Ex Factory']= order_seven_years['Rev Ex Factory']
weave_time['difference']= weave_time['Rev Ex Factory']-weave_time['Actual Weaver Issue Date']


test = [str(x).split(' ')[0] for x in weave_time['difference']]
weave_time['days']=test
weave_time['days'] = (weave_time['days'].astype(str).replace({'NaT': None})) 
weave_time['days']=pd.to_numeric(weave_time['days'])



df=order_seven_years[[
    'Order Priority','Shape','Size','Fiber Content','Primary Style',
    'Month Name', 'Location Code','Design'
    ]]

#making area
df['Size'] = df['Size'].str.replace("’", "'") 
df[['length','width']] = df['Size'].str.split('X',expand=True)
df[['length_feet','length_inch']] = df['length'].str.split("'",expand=True)
df['length_inch'] = df['length_inch'].fillna(0)
df[['width_feet','width_inch']] = df['width'].str.split("'",expand=True)
df['width_inch'] = df['width_inch'].fillna(0)
df['width_feet'] = pd.to_numeric(df['width_feet'])*12
df['length_feet'] = pd.to_numeric(df['length_feet'])*12
df['l_i'] = df['length_feet'] + pd.to_numeric(df['length_inch'])
df['w_i'] = df['width_feet'] + pd.to_numeric(df['width_inch'])
df['a_i'] = df['l_i'] *df['w_i']
df=df.drop(columns=['l_i','w_i','Size','length','width','length_feet','length_inch','width_feet','width_inch'])



df['days']=weave_time[['days']]


#Plotting the data

df_plot=df.groupby('Fiber Content')['days']

mean = df_plot.mean()
mean=mean.drop('60% Handcarded Wool 40% Hand Spun Silk')
#mean=mean.drop('Contemporary')
#mean=mean.reindex(['January',"February", 'March','April','May','June','July','August','September','October','November','December'])
# mean=mean.drop(['LOC-129', 'LOC-168', 'LOC-113',
#         'CON-GJ', 'LOC-059', 'CON-MH', 'LOC-200', 'LOC-300', 'CON-RJ',
#         'LOC-191', 'CON-KA'])
#mean=mean.drop(['Runner','Round','Oval'])

p025 = df_plot.quantile(0.025)
#p025=p025.drop('Contemporary')
p025=p025.drop('60% Handcarded Wool 40% Hand Spun Silk')
#p025=p025.reindex(['January',"February", 'March','April','May','June','July','August','September','October','November','December'])
# p025 =p025.drop(['LOC-129', 'LOC-168', 'LOC-113',
        # 'CON-GJ', 'LOC-059', 'CON-MH', 'LOC-200', 'LOC-300', 'CON-RJ',
        # 'LOC-191', 'CON-KA'])
#p025=p025.drop(['Runner','Round','Oval'])

p975 = df_plot.quantile(0.975)
#p975=p975.drop('Contemporary')
p975=p975.drop('60% Handcarded Wool 40% Hand Spun Silk')
#p975=p975.drop(['Runner','Round','Oval'])
#p975=p975.reindex(['January',"February", 'March','April','May','June','July','August','September','October','November','December'])
# p975=p975.drop(['LOC-129', 'LOC-168', 'LOC-113',
#         'CON-GJ', 'LOC-059', 'CON-MH', 'LOC-200', 'LOC-300', 'CON-RJ',
#         'LOC-191', 'CON-KA'])

plt.errorbar(
    mean,
    mean.index,
    xerr=[mean - p025, p975 - mean],
    linestyle='',
    fmt='o',
    ecolor=('green')
)
plt.title('Mean and 95% Confidence Interval Standard deviation')
plt.show()



x_i=['January',
      "February", 'March','April','May','June','July','August','September',
      'October','November','December']
sns.set(rc={'figure.figsize':(5,3.5)})

ax=sns.countplot(x="Month Name", 
  data=df,
  palette="Greens_d",order=x_i)

plt.title('Orders in Weaver Issue Date (by Month)')


for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()+100))



df = pd.get_dummies(df, columns=['Shape','Fiber Content','Primary Style','Month Name','Location Code','Design'])

# df=df[df['days'].isna()==False]



features = ['Order Priority']
target = 'days'

X = df[features].values.reshape(-1, len(features))
y = df[target].values


ols = linear_model.LinearRegression()
model = ols.fit(X, y)




x_pred = df['Order Priority'].unique()          
x_pred = x_pred.reshape(-1, len(features))  # preprocessing required by scikit-learn functions

y_pred = model.predict(x_pred)

plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(7, 3.5))

ax.plot(x_pred, y_pred, color='k', label='Regression model')
ax.scatter(X, y, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
ax.set_ylabel('Rev Ex Factory - Actual Weaver Issue Date', fontsize=9)
ax.set_xlabel('Order Priority', fontsize=14)
ax.legend(facecolor='white', fontsize=11)
ax.text(0.55, 0.55, '$y = %.3f x_1 - %.2f $' % (model.coef_[0], abs(model.intercept_)), fontsize=17, transform=ax.transAxes)

fig.tight_layout()


