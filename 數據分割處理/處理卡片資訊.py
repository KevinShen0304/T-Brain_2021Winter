# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 00:05:37 2021

@author: shen
"""
import pandas as pd
import numpy as np

# In[chid_info]
df = pd.read_csv('Data//simple_data//chid_info.csv')
df_head = df[0:200]

df_without_duplicates = df.drop_duplicates(subset=None, keep='first')
df_without_duplicates.to_csv('Data//simple_data//chid_info_without_duplicates.csv', index=False)

# In[]
df_without_duplicates = pd.read_csv('Data//simple_data//chid_info_without_duplicates.csv')

df_group = df_without_duplicates.groupby(['chid']).agg({'dt': [np.max]})
df_group.reset_index(inplace=True)
df_group.columns = ['chid', 'dt']

df_group = pd.merge(df_group,df_without_duplicates, on=['chid', 'dt'], how='left')
df_group.to_csv('Data//simple_data//chid_info_new.csv', index=False)

