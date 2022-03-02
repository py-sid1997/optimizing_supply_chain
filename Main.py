#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: siddharth
"""
#Importing necessary libraries
import pandas as pd
#from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

#df_production = pd.read_excel('Prodution Data.xlsx')
df_total_orders = pd.read_csv("total ORder Rug List for Share.csv")
df_columns=df_total_orders.columns

#Selecting specific columns from the dataframe
# imp_dates = df_total_orders[['First Order Date','Map Planning Date', 
#                               'Actual MAP Completion Date', 'Actual Branch Issue Date',
#                               'Actual Weaver Issue Date', 'Weaver On Loom Date',
#                               'Weaver Off Loom Date', 'Actual Off Loom Date',
#                               'Actual Repairing Issue Date', 'Actual Finishing Issue Date',
#                               'Actual Carpet Finish Date', 'Ramgarh Receipt Date', 
#                               'Actual WareHouse Date', 'OTN No']]

og_data= df_total_orders[['Sales Order Date',
'Posting Date',
'Actual RPO Creation Date',
'Actual MAP Completion Date',
'Actual Store Issue Date',
'Actual Branch Issue Date',
'Actual Weaver Issue Date',
'Revised Order Due Date',
'PPR Date',
'Actual HO Receipt Date',
'Actual Repairing Issue Date',
'Actual Finishing Issue Date',
'Ramgarh Receipt Date',
'Actual WareHouse Date',
'Shipment Date',
'PO Date',
'Purchase Due Date',
'Actual Purchase Receipt Date', 'OTN No','Prod  Cubage',
                              'Std  Length','Std  Width','Std Cubage','Order Priority',
                              'Shape','Size','Quality','Design', 'Border Color',
                              'Ground Color','Fiber Content','Primary Style','Location Code',
                              'Actual Length (in ft)']]
                                        

# imp_dates.to_csv('imp_dates.csv')

# #Converting the type of First Order Date to date and fetching the year from the same
# imp_dates['First Order Date'] = imp_dates['First Order Date'].str.split(' ').str[0]
# imp_dates['First Order Date']=imp_dates['First Order Date'].replace('01-01-1753',np.NaN)
# #imp_dates['First Order Date'] = imp_dates['First Order Date'].str.replace('-','/')
# imp_dates['First Order Date']= pd.to_datetime(imp_dates['First Order Date'], format="%d/%m/%y")

# imp_dates['Year'] = imp_dates['First Order Date'].dt.year


#Converting the type of First Order Date to date and fetching the year from the same
og_data['Sales Order Date'] = og_data['Sales Order Date'].str.split(' ').str[0]
og_data['Sales Order Date']= og_data['Sales Order Date'].replace('01-01-1753',np.NaN)
#imp_dates['First Order Date'] = imp_dates['First Order Date'].str.replace('-','/')
og_data['Sales Order Date']= pd.to_datetime(og_data['Sales Order Date'], format="%d/%m/%y")

og_data['Year'] = og_data['Sales Order Date'].dt.year
og_data['Month'] = og_data['Sales Order Date'].dt.month


# order_two_years= og_data[og_data['Year'].isin([2020,2021])]

#First order date between 2013-2019
order_seven_years= og_data[og_data['Year'].isin([2013,2014,2015,2016,2017,2018,2019])]
order_seven_years=order_seven_years.replace('01-01-1753 00:00:00',np.NaN)


##Time taken to GET to the different steps from the First Order date

#'Sales Order Date'-'Posting Date'
phase_one= pd.DataFrame()
phase_one[['Sales Order Date','OTN No']]= order_seven_years[['Sales Order Date', 'OTN No']]
phase_one['Posting Date']= pd.to_datetime(order_seven_years['Posting Date'], format="%d/%m/%y %H:%M")
phase_one['difference']=phase_one['Posting Date'] - phase_one['Sales Order Date']

test_new = [str(x).split(' ')[0] for x in phase_one['difference']]
phase_one['days']=test_new
phase_one['days'] = (phase_one['days'].astype(str).replace({'NaT': None})) 
                         

phase_one['days']=pd.to_numeric(phase_one['days'])

x=phase_one[(phase_one['days']<1000) &(phase_one['days']>0)].describe()


#Time taken to issue a branch to a order
phase_two= pd.DataFrame()
phase_two[['Sales Order Date','OTN No']]= order_seven_years[['Sales Order Date', 'OTN No']]
phase_two['Actual RPO Creation Date']= pd.to_datetime(order_seven_years['Actual RPO Creation Date'], format="%d/%m/%y %H:%M")
phase_two['difference']=phase_two['Actual RPO Creation Date'] - phase_two['Sales Order Date']

test_new = [str(x).split(' ')[0] for x in phase_two['difference']]
phase_two['days']=test_new
phase_two['days'] = (phase_two['days'].astype(str).replace({'NaT': None})) 
                         

phase_two['days']=pd.to_numeric(phase_two['days'])

x=phase_two[(phase_two['days']<1000) &(phase_two['days']>0)].describe()


#Time taken to issue a weaver to a order
phase_three= pd.DataFrame()
phase_three[['Sales Order Date','OTN No']]= order_seven_years[['Sales Order Date', 'OTN No']]
phase_three['Actual MAP Completion Date']= pd.to_datetime(order_seven_years['Actual MAP Completion Date'], format="%d/%m/%y %H:%M")
phase_three['difference']=phase_three['Actual Weaver Issue Date'] - phase_three['Sales Order Date']

#Time taken to get a weaver on loom for the particular order
time_on_loom= pd.DataFrame()
time_on_loom[['First Order Date','OTN No']]= order_seven_years[['First Order Date', 'OTN No']]
time_on_loom['Weaver On Loom Date']= pd.to_datetime(order_seven_years['Weaver On Loom Date'], format="%d/%m/%y %H:%M")
time_on_loom['difference']=time_on_loom['Weaver On Loom Date'] - time_on_loom['First Order Date']

#Time taken to get a weaver off loom for the particular order
time_off_loom= pd.DataFrame()
time_off_loom[['First Order Date','OTN No']]= order_seven_years[['First Order Date', 'OTN No']]
time_off_loom['Actual Off Loom Date']= pd.to_datetime(order_seven_years['Actual Off Loom Date'], format="%d/%m/%y %H:%M")
time_off_loom['difference']=time_off_loom['Actual Off Loom Date'] - time_off_loom['First Order Date']

#Time taken to get to the reapiring phase for the particular order
time_repair_issue= pd.DataFrame()
time_repair_issue[['First Order Date','OTN No']]= order_seven_years[['First Order Date', 'OTN No']]
time_repair_issue['Actual Repairing Issue Date']= pd.to_datetime(order_seven_years['Actual Repairing Issue Date'], format="%d/%m/%y %H:%M")
time_repair_issue['difference']=time_repair_issue['Actual Repairing Issue Date'] - time_repair_issue['First Order Date']

#Time taken to get to the finishing issue for the particular order
time_finish_issue= pd.DataFrame()
time_finish_issue[['First Order Date','OTN No']]= order_seven_years[['First Order Date', 'OTN No']]
time_finish_issue['Actual Finishing Issue Date']= pd.to_datetime(order_seven_years['Actual Finishing Issue Date'], format="%d/%m/%y %H:%M")
time_finish_issue['difference']=time_finish_issue['Actual Finishing Issue Date'] - time_finish_issue['First Order Date']

#Time taken to get to the finished product for the particular order
time_carpet_finish= pd.DataFrame()
time_carpet_finish[['First Order Date','OTN No']]= order_seven_years[['First Order Date', 'OTN No']]
time_carpet_finish['Actual Carpet Finish Date']= pd.to_datetime(order_seven_years['Actual Carpet Finish Date'], format="%d/%m/%y %H:%M")
time_carpet_finish['difference']=time_carpet_finish['Actual Carpet Finish Date'] - time_carpet_finish['First Order Date']


##The time taken IN a step by subtracting date from the previous step

#Branch Issue after map completion
branch_issue_after_map = pd.DataFrame()
branch_issue_after_map['OTN No']= order_seven_years['OTN No']
branch_issue_after_map['Actual MAP Completion Date']= time_to_map['Actual MAP Completion Date']
branch_issue_after_map['Actual Branch Issue Date']= time_to_branch['Actual Branch Issue Date']
branch_issue_after_map['difference']= branch_issue_after_map['Actual Branch Issue Date'] - branch_issue_after_map['Actual MAP Completion Date']


#Time between Weaver Issue Date and Branch issue date
weaver_issue_after_branch = pd.DataFrame()
weaver_issue_after_branch['OTN No']= order_seven_years['OTN No']
weaver_issue_after_branch['Actual Branch Issue Date']= time_to_branch['Actual Branch Issue Date']
weaver_issue_after_branch['Actual Weaver Issue Date']= time_to_issue_weaver['Actual Weaver Issue Date']
weaver_issue_after_branch['difference']= weaver_issue_after_branch['Actual Weaver Issue Date'] - weaver_issue_after_branch['Actual Branch Issue Date']

#Time taken by weaver to start once issued
on_loom_after_issue = pd.DataFrame()
on_loom_after_issue['OTN No']= order_seven_years['OTN No']
on_loom_after_issue['Weaver On Loom Date']=time_on_loom['Weaver On Loom Date']
on_loom_after_issue['Actual Weaver Issue Date']=pd.to_datetime(order_seven_years['Actual Weaver Issue Date'], format="%d/%m/%y %H:%M")
on_loom_after_issue['difference']= on_loom_after_issue['Weaver On Loom Date']-on_loom_after_issue['Actual Weaver Issue Date']

#Expected Time for Weaver to complete the carpet
planned_time_to_weave = pd.DataFrame()
planned_time_to_weave['OTN No']= order_seven_years['OTN No']
planned_time_to_weave['Weaver On Loom Date']=time_on_loom['Weaver On Loom Date']
planned_time_to_weave['Weaver Off Loom Date']=pd.to_datetime(order_seven_years['Weaver Off Loom Date'], format="%d/%m/%y %H:%M")
planned_time_to_weave['difference']= planned_time_to_weave['Weaver Off Loom Date']-planned_time_to_weave['Weaver On Loom Date']

# 
# 
# 
# 
# Revised Order Due Date- Actual weaver Issue
real_time_to_weave = pd.DataFrame()
real_time_to_weave[['OTN No','Prod  Cubage',
                   'Std  Length','Std  Width','Std Cubage','Order Priority',
                   'Shape','Size','Quality','Design', 'Border Color',
                   'Ground Color','Fiber Content','Primary Style','Location Code',
                   'Actual Length (in ft)']]= order_seven_years[['OTN No','Prod  Cubage',
                   'Std  Length','Std  Width','Std Cubage','Order Priority',
                   'Shape','Size','Quality','Design', 'Border Color',
                   'Ground Color','Fiber Content','Primary Style','Location Code',
                   'Actual Length (in ft)']]
real_time_to_weave[['Actual Weaver Issue Date']]= pd.to_datetime(order_seven_years['Actual Weaver Issue Date'], format="%d/%m/%y %H:%M")
real_time_to_weave['Revised Order Due Date']= pd.to_datetime(order_seven_years['Revised Order Due Date'], format="%d/%m/%y %H:%M")
real_time_to_weave['difference']= real_time_to_weave['Revised Order Due Date']-real_time_to_weave['Actual Weaver Issue Date']

test = [str(x).split(' ')[0] for x in real_time_to_weave['difference']]
real_time_to_weave['days']=test
real_time_to_weave['days'] = (real_time_to_weave['days'].astype(str).replace({'NaT': None})) 
                         

real_time_to_weave['days']=pd.to_numeric(real_time_to_weave['days'])

x=real_time_to_weave[(real_time_to_weave['days']<1000) &(real_time_to_weave['days']>0)].describe()




# PPR Date- Actual weaver Issue

system_entry = pd.DataFrame()
system_entry['OTN No']= order_seven_years['OTN No']
system_entry[['Actual Weaver Issue Date']]= pd.to_datetime(order_seven_years['Actual Weaver Issue Date'], format="%d/%m/%y %H:%M")
system_entry['PPR Date']= pd.to_datetime(order_seven_years['PPR Date'], format="%d/%m/%y %H:%M")
system_entry['difference']= system_entry['PPR Date']-system_entry['Actual Weaver Issue Date']


test = [str(x).split(' ')[0] for x in system_entry['difference']]
system_entry['days']=test
system_entry['days'] = (system_entry['days'].astype(str).replace({'NaT': None})) 
system_entry['days']=pd.to_numeric(system_entry['days'])
y=system_entry[(system_entry['days']<1000) &(system_entry['days']>0)].describe()

# 'Actual WareHouse Date'- Actual weaver Issue
warehouse = pd.DataFrame()
warehouse['OTN No']= order_seven_years['OTN No']
warehouse[['Actual Weaver Issue Date']]= system_entry[['Actual Weaver Issue Date']]
warehouse['Actual WareHouse Date']= pd.to_datetime(order_seven_years['Actual WareHouse Date'], format="%d/%m/%y %H:%M")
warehouse['difference']= warehouse['Actual WareHouse Date']-warehouse['Actual Weaver Issue Date']


test = [str(x).split(' ')[0] for x in warehouse['difference']]
warehouse['days']=test
warehouse['days'] = (warehouse['days'].astype(str).replace({'NaT': None})) 
warehouse['days']=pd.to_numeric(warehouse['days'])
y=warehouse[(warehouse['days']<1000) &(warehouse['days']>0)].describe()






# 'Shipment Date',
shipment = pd.DataFrame()
shipment['OTN No']= order_seven_years['OTN No']
shipment[['Actual Weaver Issue Date']]= system_entry[['Actual Weaver Issue Date']]
shipment['Shipment Date']= pd.to_datetime(order_seven_years['Shipment Date'], format="%d/%m/%y %H:%M")
shipment['difference']= shipment['Shipment Date']-shipment['Actual Weaver Issue Date']


test = [str(x).split(' ')[0] for x in shipment['difference']]
shipment['days']=test
shipment['days'] = (shipment['days'].astype(str).replace({'NaT': None})) 
shipment['days']=pd.to_numeric(shipment['days'])
y=shipment[(shipment['days']<1000) &(shipment['days']>0)].describe()


# 'PO Date',
# 'Purchase Due Date',
# 'Actual Purchase Receipt Date',








z=order_seven_years.describe(include='all')











test=real_time_to_weave[(real_time_to_weave['days']<1000) &(real_time_to_weave['days']>0)]
# test = dataset[(dataset['days']<400) & (dataset['days']>0)]


sns.set(rc={'figure.figsize':(12,7.27)})
ax=sns.countplot(y="Quality", 
 data=test,
 palette="Greens_d")

plt.title('Count of Quality')


for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+5))



sns.boxplot(y=test['Quality'],x=test[test['days']<500]['days'], color='green', 
            showmeans=True,
           meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"5"},
           fliersize=5)


sns.set(rc={'figure.figsize':(20,7.27)})
# sns.catplot(x='days',data=test,y='Order Priority')

sns.set(rc={'figure.figsize':(12,8.27)})
sns.catplot(x='Location Code',hue='Shape',data=test,kind='count',height=7, aspect=3)

sns.set(rc={'figure.figsize':(10,7.27)})
sns.pointplot(x="Order Priority", 
  y="days", 
 hue="Shape", 
 data=test,
 #palette={"male":"g",
 #"female":"m"},
 #markers=["^","o"],
 linestyles=["-","--","-.","dotted",":","solid"])


# sns.boxplot(data=test,orient="h") 


# sns.set(rc={'figure.figsize':(15,7.27)})
# sns.violinplot(x="Primary Style", 
#  y="days",
#  hue="Order Priority",
#  data=test)

sns.jointplot("days",
 "Order Priority",
 data=test,
 kind='kde')

sns.jointplot("days",
 "Primary Style",
 data=test,
 kind='scatter')




sns.set(rc={'figure.figsize':(50,20.27)})
sns.stripplot(x='days',y='Order Priority',data=test)

test['Primary Style']=test['Primary Style'].astype(str)
x=test['days']
y=test['Order Priority']
#plt.figure(figsize=(20,20))
plt.scatter(test['days'], test['Order Priority'])
plt.xlabel("days")
plt.ylabel("Order Priority")
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")

plt.show()











#Time taken by Weaver to complete the carpet
real_time_to_weave = pd.DataFrame()
real_time_to_weave[['OTN No',
                   'First Order Date', 
                   'Actual MAP Completion Date', 'Actual Branch Issue Date',
                   'Actual Weaver Issue Date',
                   'Weaver Off Loom Date',
                   'Actual Repairing Issue Date', 'Actual Finishing Issue Date',
                   'Actual Carpet Finish Date', 'Ramgarh Receipt Date', 
                   'Actual WareHouse Date','Prod  Cubage',
                   'Std  Length','Std  Width','Std Cubage','Order Priority',
                   'Shape','Size','Quality','Design', 'Border Color',
                   'Ground Color','Fiber Content','Primary Style','Location Code',
                   'Actual Length (in ft)']]= order_seven_years[['OTN No','First Order Date', 
                              'Actual MAP Completion Date', 'Actual Branch Issue Date',
                              'Actual Weaver Issue Date',
                              'Weaver Off Loom Date', 
                              'Actual Repairing Issue Date', 'Actual Finishing Issue Date',
                              'Actual Carpet Finish Date', 'Ramgarh Receipt Date', 
                              'Actual WareHouse Date','Prod  Cubage',
                              'Std  Length','Std  Width','Std Cubage','Order Priority',
                              'Shape','Size','Quality','Design', 'Border Color',
                              'Ground Color','Fiber Content','Primary Style','Location Code',
                              'Actual Length (in ft)']]
real_time_to_weave['Weaver On Loom Date']=time_on_loom['Weaver On Loom Date']
real_time_to_weave['Actual Off Loom Date']=time_off_loom['Actual Off Loom Date']
real_time_to_weave['difference']= real_time_to_weave['Actual Off Loom Date']-real_time_to_weave['Weaver On Loom Date']

# # real_time_to_weave.to_csv('real_time_to_weave.csv')
#details=dataset.describe(include='all')

#Time taken to finish the reaparing once issued
time_to_repair = pd.DataFrame()
time_to_repair['OTN No']= order_seven_years['OTN No']
time_to_repair['Actual Repairing Issue Date']=time_repair_issue['Actual Repairing Issue Date']
time_to_repair['Actual Finishing Issue Date']=time_finish_issue['Actual Finishing Issue Date']
time_to_repair['difference']= time_to_repair['Actual Finishing Issue Date']-time_to_repair['Actual Repairing Issue Date']

#Time taken to finish the carpet ready once finishing is done
carpet_ready = pd.DataFrame()
carpet_ready['OTN No']= order_seven_years['OTN No']
carpet_ready['Actual Carpet Finish Date']= time_carpet_finish['Actual Carpet Finish Date']
carpet_ready['Actual Finishing Issue Date']=time_to_repair['Actual Finishing Issue Date']
carpet_ready['difference']= carpet_ready['Actual Carpet Finish Date']-carpet_ready['Actual Finishing Issue Date']

# df_production_2=df_production[['OTN','Loom ID','Pending Work','Weaver','Branch Location New']]
weave_time= real_time_to_weave[real_time_to_weave['difference'].notna()]
#weave_time.to_csv('weave_time.csv')



# dataset = pd.merge(weave_time,df_production_2,left_on='OTN No',right_on='OTN', how='left')

# test = [str(x).split(' ')[0] for x in dataset['difference']]
# dataset['days']=test
# dataset['days']=pd.to_numeric(dataset['days'])

# #desc=dataset.describe(include='all')

# test=dataset[(dataset['days']<1000)& (dataset['days']>0)]
# test = dataset[(dataset['days']<400) & (dataset['days']>0)]

# ax=sns.scatterplot(x="days",y='Actual Length (in ft)', data=test)
# ax.xaxis.set_major_locator(ticker.MultipleLocator(50))

# sns.set(rc={'figure.figsize':(11.7,8.27)})
# ax=sns.scatterplot(x="days",y='Prod  Cubage', data=test)
# ax.xaxis.set_major_locator(ticker.MultipleLocator(30))

sns.boxplot(x=test['Shape'],y=test['days'], color='lime', 
            showmeans=True,
           meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"5"},
           fliersize=5)



test['Primary Style']=test['Primary Style'].astype(str)
x=test['days']
y=test['Primary Style']
#plt.figure(figsize=(20,20))
plt.scatter(test['days'], test['Primary Style'])
plt.xlabel("days")
plt.ylabel("Primary Style")
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")

plt.show()


# ax=sns.lmplot(x="days",y='Std  Length', data=test)
# ax.xaxis.set_major_locator(ticker.MultipleLocator(30))




# dataset.to_csv('weave_time_with_loomID.csv')


location=[]
for x in test["Quality"].unique():
    location=location+[x]
np.mean, np.std
test_new=test.groupby(['Quality'], as_index=False).agg([np.mean,np.std])
test_days=test_new['days']
test_days.plot(kind = "barh", y = "mean", legend = False,
          xerr = "std", title = "Quality", color='blue')




# # Build the plot
# location=[]
# for x in test["Fiber Content"].unique():
#     location=location+[x]
# # np.mean, np.std
# #plt.figure(figsize=(40,40))
# test_new=test.groupby(['Fiber Content'], as_index=False).agg([np.mean,np.std])
# test_days=test_new['days']
# ax=test_days.plot(kind = "barh", y = "mean", legend = False,
#           xerr = "std", title = "Mean and standard deviation", color='blue')
# ax.set_xlabel('Days')
# plt.show()





# fig, ax = plt.subplots()
# ax.bar(test['Location Code'], location, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
# ax.set_ylabel('Coefficient of Thermal Expansion ($\degree C^{-1}$)')
# ax.set_xticks(x_pos)
# ax.set_xticklabels(materials)
# ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
# ax.yaxis.grid(True)

# # Save the figure and show
# plt.tight_layout()
# plt.savefig('bar_plot_with_error_bars.png')
# plt.show()



# #test code to manipulate with dates and for converting the extreme date to nan
# test=pd.DataFrame()
# test['First Order Date'] = imp_dates['First Order Date'].str.split(' ').str[0]
# test['First Order Date']=test['First Order Date'].replace('01-01-1753',np.NaN)
# test['First Order Date'] = test['First Order Date'].str.replace('1753','53')
# test['First Order Date'] = test['First Order Date'].str.replace('-','/')

# test['First Order Date']=pd.to_datetime(test['First Order Date'], format="%d/%m/%y").dt.strftime("%Y-%m-%d")
