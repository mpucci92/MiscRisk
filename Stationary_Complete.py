#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Import all libraries ### 

import pandas as pd
import numpy as np
from sklearn import preprocessing

# Visualizations # 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.stattools import adfuller


# In[2]:


eurusd = pd.read_csv('C:\\Users\\mpucci\\Desktop\\EURUSD.csv')


# In[3]:


eurusd['Candle'] = 100*((eurusd['Close'].values - eurusd['Open'].values)/(eurusd['Open'].values))


# In[4]:


normalcandle = eurusd['Candle']
candle_scaled = preprocessing.scale(eurusd['Candle'])
eurusd['CandleSTD'] = candle_scaled
value = eurusd['CandleSTD'] 
#plt.hist(candle_scaled,bins=500)
#plt.xlim(left=-20)
#plt.xlim(right=20)
#plt.show()


# In[5]:


tester = eurusd['CandleSTD']
a = tester.isnull().values.any() # No NaN Values


# In[6]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries,start,end,jump):
    array=[]
    
    for x in range(int(len(timeseries)/(jump))):
        data = timeseries[start:end]
        #Perform Dickey-Fuller test:
        #print ('Results of Dickey-Fuller Test:')
        dftest = adfuller(data, autolag = 'AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
    
        if (dfoutput['Test Statistic'] < dfoutput['Critical Value (5%)'] or np.isnan(dfoutput['Test Statistic'])):
            dfoutput['Stationary'] = 0 
            #print ('S')
        else:
            dfoutput['Stationary'] = 1 
            #print ('NS')
    
        array.append(dfoutput['Stationary'])
    
        start = start + jump
        end = end+ jump
    
    return (np.nan_to_num(array))


# In[8]:


val = test_stationarity(value,0,20,20)
plt.hist(val,bins=2)
plt.show()


# In[ ]:





# In[9]:


# Output for the autocorrelation 

autocorrelation = np.nan_to_num(autocorrelation_array)
print(autocorrelation)
plt.hist(autocorrelation)
plt.show()


# In[ ]:


# Output for the Stationarity 

stationarity = np.nan_to_num(stationarity_array)
print (stationarity)
plt.hist(stationarity)
plt.show()


# In[ ]:




