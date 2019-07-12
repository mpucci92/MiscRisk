#!/usr/bin/env python
# coding: utf-8

# ### Section 1: Libraries and Data Imports

# ### Importing Libraries

# In[393]:


import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt
import sys, os
import lightgbm as lgbm


# In[394]:


with open("C:\Python\Algo\Model\lgbm_2019-04-16", 'rb') as file:
    x = joblib.load(file)


# ### All possible signals generated

# In[395]:


trades_new = pd.read_csv("Algo/Data/Raw trades Data/EURUSD_trades.csv")


# ### RAW DATA FOR EURUSD

# In[396]:


raw_trades = pd.read_csv("Algo/Data/Data/EURUSD.csv")


# ### Scaled_new all signals generated with features from the model

# In[397]:


scaled_new = pd.read_csv("Algo/Data/Raw Scaled/EURUSD_scaled.csv")


# ### All our trades signals YES and NO from the model

# In[398]:


scaled_new['trades'] = x.predict(scaled_new.iloc[:,3:].values)


# ### Data Merged to display positively predicted trades

# In[399]:


merged_set = scaled_new.merge(trades_new[['Datetime']], how="outer",on="Datetime").dropna() # merge scaled new with trades_new 


# In[400]:


filtered_merge = merged_set[merged_set['trades'] == 1.0]


# ### Filtering the dataframe for predicted trades for the years 2012 to 2016

# In[401]:



filtered_merge['year'] = filtered_merge.Datetime.apply(pd.to_datetime)
filtered_merge['year'] = filtered_merge.year.dt.year

filtered_merge_12_16 = filtered_merge[(filtered_merge['year']>=2012)&(filtered_merge['year']<=2016)]


# ### Section 2: Data Analysis

# In[402]:


idx_list = []

dataset_train = filtered_merge_12_16.loc[0:(len(filtered_merge))]      # dataset_train grabs all the rows of filteredmerge

                                  # All the indexes of our trades corresponding to the raw data


# In[403]:


for datetime in dataset_train.Datetime:
    idx = raw_trades[raw_trades.Datetime == datetime].index
    idx_list.append((idx[0]))                   


# ### Direction list - fetches the direction of all the trades we predicted to take

# In[404]:


array = []
for i in dataset_train.Direction:
    array.append(i)


# In[405]:


close_values = []
df_drawdown = []
ttc_list = []

tick_increment = 0.00005
fixed_dollar = 5

### Original Code Trade Progession TTC ###
for ttc in dataset_train.TTC:     # Fetches all time to close values for every trade 
    ttc_list.append(int(ttc))

# Close price of the signal candle
for idx in idx_list:
    close_value = raw_trades.get_value(idx, 'Close')
    close_values.append(close_value)
    


# close value has to stop at start: start + TTC value for the trade. 
for i in range(len(idx_list)):
    
    close = raw_trades.Close[(idx_list[i]+1):((idx_list[i]+1)+(ttc_list[i]-1)):1]
    direction = array[i]
    
    if direction == 1.0:
        change = 1*(close_values[i] - close)
    elif direction == -1.0:
        change = -1*(close_values[i] - close)
    
    df_drawdown.append(change) # df_drawdown contains price movemen changes compared to the close of the signal candle 


# ### Rate of Change Function

# In[406]:


def ROC (data,timeframe):
    price_change = data.diff()
    time = timeframe #specify time in minutes 
    roc = price_change/time
    
    return roc 


# ### Contain the difference in percentage for each Trade, each candle in the trade pegged against the close of the signal (First 5 candles)

# In[407]:


average_list5 = [] 

n=6 # Always add 1 to your orginal input since the signal candle is included

for i in range(len(df_drawdown)):
    local_average= np.mean(df_drawdown[i][:n])
    average_list5.append(local_average)

# VIEW OUTPUT
# print (average_list5)


# ### Contain the difference in percentage for each Trade, each candle in the trade pegged against the close of the signal (First 10 candles)

# In[408]:


average_list10 = [] 

n=11 # Always add 1 to your orginal input since the signal candle is included

for i in range(len(df_drawdown)):
    local_average= np.mean(df_drawdown[i][:n])
    average_list10.append(local_average)

# VIEW OUTPUT
# print (average_list10)


# ### OUTLIER DRAWDOWN BLOCK: Bottom 10% of the curve in terms of drawdowns

# In[409]:


# Assuming Gaussian Distribution - Outlier definition ROUGH METRIC 

outliers=[]

def detect_outlier(data_1):
    
    threshold= -1.7
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if (z_score) < threshold:
            outliers.append(y)
    return outliers


# ### Applying the function to detect the outlier value amounts

# In[410]:


drawdown_data = ((filtered_merge_12_16.Drawdown))
#print(drawdown_data)
outlier_set = detect_outlier(drawdown_data)
#print (outlier_set)
max_ = max(outlier_set)
print(max_)
print(len(outlier_set))


# ### TESTING BLOCK

# In[411]:


filtered_merge_copy = filtered_merge_12_16.copy() 
val = (filtered_merge_copy['Drawdown'])
idx_list1 = []

dataset_train1 = filtered_merge_copy[filtered_merge_copy['Drawdown'] < max_] # All the outlier trade losers
#print(dataset_train1)
for datetime in dataset_train1.Datetime:
    idx = raw_trades[raw_trades.Datetime == datetime].index
    idx_list1.append((idx[0]))


# In[412]:


len(dataset_train1[dataset_train1.Amer == 1 ])/len(dataset_train1)


# In[413]:


dataset_train1.columns


# In[414]:


array = []
for i in dataset_train1.Direction:
    array.append(i)


# In[415]:


close_values = []
df_drawdown = []
ttc_list = []

tick_increment = 0.00005
fixed_dollar = 5

### Original Code Trade Progession TTC ###
for ttc in dataset_train1.TTC:     # Fetches all time to close values for every trade 
    ttc_list.append(int(ttc))

# Close price of the signal candle
for idx in idx_list:
    close_value = raw_trades.get_value(idx, 'Close')
    close_values.append(close_value)

# close value has to stop at start: start + TTC value for the trade. 
for i in range(len(idx_list1)):
    
    close = raw_trades.Close[(idx_list1[i]+1):((idx_list1[i]+1)+(ttc_list[i]-1)):1]
    direction = array[i]
    
    if direction == 1.0:
        change = 1*(close_values[i] - close)
    elif direction == -1.0:
        change = -1*(close_values[i] - close)
    
    df_drawdown.append(change) # df_drawdown contains price movemen changes compared to the close of the signal candle 


# ### Contain the difference in percentage for each Trade, each candle in the trade pegged against the close of the signal (First 5 candles)

# In[416]:


average_list5 = [] 

n=6 # Always add 1 to your orginal input since the signal candle is included

for i in range(len(df_drawdown)):
    local_average= np.mean(df_drawdown[i][:n])
    average_list5.append(local_average)


# ### Contain the difference in percentage for each Trade, each candle in the trade pegged against the close of the signal (First 10 candles)

# In[417]:


average_list10 = [] 

n=11 # Always add 1 to your orginal input since the signal candle is included

for i in range(len(df_drawdown)):
    local_average= np.mean(df_drawdown[i][:n])
    average_list10.append(local_average)
    


# In[418]:


n=11
roc_list = []
for i in range(len(df_drawdown)):
    val = (df_drawdown[i][:n])
    roc_list.append(roc(val,5))
print(roc_list)


# ### Separation of Distributions

# In[328]:


n=11
diff_distribution_neg = []
diff_distribution_pos = []

for i in range(len(df_drawdown)):
    average_difference = np.mean(df_drawdown[i][:n])
    if average_difference > 0:
        diff_distribution_pos.append(average_difference)
    elif average_difference < 0:
        diff_distribution_neg.append(average_difference)
    #plt.plot(df_drawdown[i][:n])
    #plt.show()


# ### Negative 10 candle averages

# In[329]:


sns.distplot(diff_distribution_neg)


# In[391]:


neg_diff = pd.DataFrame(diff_distribution_neg)
cutoff_val = np.percentile(neg_diff,1) # Retrieve the 1st Quartile -check trades success rate which have an average less than this. 
print(cutoff_val)


# ### Positive 10 candle averages

# In[331]:


sns.distplot(diff_distribution_pos)


# ### Positives and Negatives grouped all together

# In[332]:


n=11
diff_distribution = []

for i in range(len(df_drawdown)):
    average_difference = np.mean(df_drawdown[i][:n])
    diff_distribution.append(average_difference)
    #plt.plot(df_drawdown[i][:n])
    #plt.show()


# In[333]:


sns.distplot(diff_distribution)


# ### Distribution of the TTC for all the outlier trades

# In[334]:


plt.hist(dataset_train1.TTC,bins=50)


# In[335]:


sns.kdeplot(dataset_train1.TTC)


# ### Suggesting that after 3rd quartile, Salvage mode should be considered Majority of the large loser trade exhibit long TTCs

# ### TTC of the whole dataset of trades, you see the 3rd quartile, and outliers tend to be longer trades -> hinting that after a threshold its time to cut when possible.
# 

# In[336]:


filtered_merge_12_16.TTC.describe()


# ### Rate of Change Function

# In[337]:


def ROC (data,timeframe):
    price_change = data.diff()
    time = timeframe #specify time in minutes 
    roc = price_change/time
    
    return roc 


# In[338]:


#n=11
#for i in range(len(df_drawdown)):
    #plt.plot(ROC(df_drawdown[i][:n],5))
    #plt.show()


# ### Non-Outlier Trade TTC distribution within the top right of 90% of the middle of the distribution

# In[339]:


outliers=[]

def detect_outlier(data_1):
    
    threshold= -1.7
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if (z_score) > threshold:
            outliers.append(y)
    return outliers


# In[340]:


drawdown_data = ((filtered_merge_12_16.Drawdown))
#print(drawdown_data)
outlier_set = detect_outlier(drawdown_data)
#print (outlier_set)
min_ = min(outlier_set)


# In[341]:


filtered_merge_copy = filtered_merge_12_16.copy() 
dataset_train2 = filtered_merge_copy[filtered_merge_copy['Drawdown'] > min_] # All the non loser outliers.


# In[342]:


plt.hist(dataset_train2.TTC,bins=100)


# ### TESTING BLOCK SECTION 3

# ### Trades above 20 TTC - relative percentage to their max drawdown 

# In[353]:


dataset_train3 = filtered_merge_copy[filtered_merge_copy['TTC'] > 20]


# In[354]:


array = []
for i in dataset_train3.Direction:
    array.append(i)


# In[355]:


idx_list = []

for datetime in dataset_train3.Datetime:
    idx = raw_trades[raw_trades.Datetime == datetime].index
    idx_list.append((idx[0]))


# In[356]:


close_values = []
df_drawdown = []
ttc_list = []
open_values = []
candle_size = []

tick_increment = 0.00005
fixed_dollar = 5

### Original Code Trade Progession TTC ###
for ttc in dataset_train3.TTC:     # Fetches all time to close values for every trade 
    ttc_list.append(int(ttc))

    # Close price of the signal candle
for idx in idx_list:
    close_value = raw_trades.get_value(idx, 'Close')
    close_values.append(close_value)

    # Open price of the signal candle 
    open_value = raw_trades.get_value(idx,'Open')
    open_values.append(open_value)


# Take first value of the list close - first value of list open -> get change in terms of price of the candle 
    
    
    
    
    # close value has to stop at start: start + TTC value for the trade. 
for i in range(len(idx_list)):
      
    close = raw_trades.Close[(idx_list[i]+1):((idx_list[i]+1)+(ttc_list[i]-1)):1]
    direction = array[i]
    
    if direction == 1.0:
        change = 1*(close_values[i] - close)
    elif direction == -1.0:
        change = -1*(close_values[i] - close)
    
    df_drawdown.append(change) # df_drawdown contains price movement changes compared to the close of the signal candle 


# ### Candle Size List

# In[371]:


# Take first value of the list close - first value of list open -> get change in terms of price of the candle 

candle_size = []
stop_loss = []


for i in range(len(idx_list)):
    candle = abs((close_values[i]-open_values[i]))
    candle_size.append(candle)   


# ### Stop Loss denominator 

# In[372]:


stop_loss = []
risk = 2.5

for i in range(len(idx_list)):
    direction=array[i]
    stop_value = (risk*abs(candle_size[i]))
    stop_loss.append(stop_value)


# ### Slicing 20 candles after for the drawdowns

# In[373]:


percentage_rdd = [] # percentage realized drawdown 

#for i in stop_loss:
    

for i in range(len(df_drawdown)):
    values = df_drawdown[i][20:]/stop_loss[i]
    percentage_rdd.append(values)
    


# ### TESTING BLOCK - Realized Drawdowns for each trade after 20 TTC

# In[374]:


#drawdown_array = []
#for i in dataset_train3.Drawdown:
#   drawdown_array.append(i)


# In[375]:


# print(dataset_train3.Drawdown)


# In[376]:


percentage_rdd = [x.values.tolist() for x in percentage_rdd] # converting series to Arrays 


# In[377]:


value = [[inner_list[i] for inner_list in percentage_rdd if len(inner_list) > i] for i in range(100)]


# In[385]:





# In[378]:


max([len(x) for x in percentage_drawdown])


# In[420]:


plt.figure(figsize=(20,14))
plt.boxplot(value)
plt.ylim(-1,3)
plt.axhline(1,color='r')
plt.show()


# ### Modification to the above image where candles are stopped out once they reach the 1.0 mark 
# 

# In[426]:


dataset_train4 = filtered_merge_copy[filtered_merge_copy['TTC'] > 20]


# In[427]:


array = []
for i in dataset_train4.Direction:
    array.append(i)


# In[428]:


idx_list = []

for datetime in dataset_train4.Datetime:
    idx = raw_trades[raw_trades.Datetime == datetime].index
    idx_list.append((idx[0]))


# In[429]:


close_values = []
df_drawdown = []
ttc_list = []
open_values = []
candle_size = []

tick_increment = 0.00005
fixed_dollar = 5

### Original Code Trade Progession TTC ###
for ttc in dataset_train4.TTC:     # Fetches all time to close values for every trade 
    ttc_list.append(int(ttc))

    # Close price of the signal candle
for idx in idx_list:
    close_value = raw_trades.get_value(idx, 'Close')
    close_values.append(close_value)

    # Open price of the signal candle 
    open_value = raw_trades.get_value(idx,'Open')
    open_values.append(open_value)


# Take first value of the list close - first value of list open -> get change in terms of price of the candle 
    
    
    
    
    # close value has to stop at start: start + TTC value for the trade. 
for i in range(len(idx_list)):
      
    close = raw_trades.Close[(idx_list[i]+1):((idx_list[i]+1)+(ttc_list[i]-1)):1]
    direction = array[i]
    
    if direction == 1.0:
        change = 1*(close_values[i] - close)
    elif direction == -1.0:
        change = -1*(close_values[i] - close)
    
    df_drawdown.append(change) # df_drawdown contains price movement changes compared to the close of the signal candle 


# In[430]:


# Take first value of the list close - first value of list open -> get change in terms of price of the candle 

candle_size = []
stop_loss = []


for i in range(len(idx_list)):
    candle = abs((close_values[i]-open_values[i]))
    candle_size.append(candle)   


# In[431]:


stop_loss = []
risk = 2.5

for i in range(len(idx_list)):
    direction=array[i]
    stop_value = (risk*abs(candle_size[i]))
    stop_loss.append(stop_value)


# In[463]:


percentage_rdd_modification = [] # percentage realized drawdown 
index_list = []
#for i in stop_loss:
    

for i in range(len(df_drawdown)):
    values = df_drawdown[i][20:]/stop_loss[i]
    print(values)


# In[459]:


percentage_rdd_modification = [x.values.tolist() for x in percentage_rdd_modification] # converting series to Arrays 


# In[460]:


value = [[inner_list[i] for inner_list in percentage_rdd_modification if len(inner_list) > i] for i in range(100)]


# In[461]:


plt.figure(figsize=(20,14))
plt.boxplot(value)
plt.ylim(-1,3)
plt.axhline(1,color='r')
plt.show()


# In[ ]:




