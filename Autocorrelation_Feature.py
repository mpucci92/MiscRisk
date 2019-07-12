#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

def autocorrelation (data,start_period,end_period,lag):
    array = []
    jump = 50
    
    for x in range(int(len(value)/(jump))):
        data = data[start_period:end_period]
    
        autocorr_value = data.autocorr(lag)
        array.append(autocorr_value)
        
        start_period= start_period + jump
        end_period = end_period + jump
        
    return (np.nan_to_num(array))


# In[20]:


eurusd = pd.read_csv('C:\\Users\\mpucci\\Desktop\\EURUSD.csv')
eurusd['Candle'] = 100*((eurusd['Close'].values - eurusd['Open'].values)/(eurusd['Open'].values))

candle_scaled = preprocessing.scale(eurusd['Candle'])
eurusd['CandleSTD'] = candle_scaled
value = eurusd['CandleSTD']


# In[21]:


result = autocorrelation(value,0,50,11)
print (results)


# In[12]:


plt.hist(results,bins=100)
plt.show()


# In[ ]:




