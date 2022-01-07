# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 22:52:48 2021

@author: shen
"""
import pandas as pd

filename = 'data_bydt//train_dt24.csv'

df = pd.read_csv(filename)

df_head = df[0:200]

df.loc[0:4]
df = df[['dt','chid','shop_tag','txn_cnt','txn_amt']]

df_pivot = pd.pivot_table(df, values='txn_amt', index=['dt', 'chid'],columns=['shop_tag'])
df_pivot_head = df_pivot[0:200]

df_pivot.to_csv('data_bydt//train_pivot_dt24.csv')


#df.to_excel('data_bydt//train_dt24.xlsx')
