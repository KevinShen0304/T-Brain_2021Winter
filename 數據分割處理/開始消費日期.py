# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 20:05:12 2021

@author: shen
"""

import pandas as pd 

df = pd.read_csv('Data//simple_data//train.csv')

chid_min_dt = df.groupby(by=['chid']).agg({'dt':'min'})
chid_min_dt = chid_min_dt.reset_index()
chid_min_dt = chid_min_dt.rename(columns={'dt':'begin_dt'})
chid_min_dt.to_csv('Data//simple_data//begin_dt.csv',index=False)