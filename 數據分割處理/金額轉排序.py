# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 00:18:22 2021

@author: shen
"""

import pandas as pd 

df = pd.read_csv('Data//simple_data//train.csv')

txn_amt = df.groupby(by=['txn_amt']).agg({'dt':'first'})
txn_amt = txn_amt.reset_index()
del txn_amt['dt']
txn_amt['txn_amt_rank'] = txn_amt['txn_amt'].rank()

txn_amt.to_csv('Data//simple_data//txn_amt_rank.csv',index=False)

# 
df = df.merge(txn_amt,on='txn_amt')
df.to_csv('Data//simple_data//train_with_rank.csv',index=False)
