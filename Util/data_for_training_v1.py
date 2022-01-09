# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 20:29:54 2021
訓練前處理
@author: shen
"""

import numpy as np

# 轉換type，減少空間浪費
def change_type(X_df):
    # category
    X_df['shop_tag'] = X_df['shop_tag'].astype('category')
    # prop
    prop_col =[col for col in X_df.columns if col.find('prop')!=-1]
    for col in prop_col:
        X_df[col] = X_df[col]*100
    # txn
    txn_col =[col for col in X_df.columns if col.find('txn')!=-1]
    for col in txn_col:
        X_df[col] = X_df[col].astype('int',errors='ignore')
    return(X_df)

# 獲得訓練col
def get_train_columns(X_df):
    train_columns = [c for c in X_df.columns if c not in ['chid']]
    return(train_columns)

# 刪除皆為零的chid(不影響訓練)
def del_zero(X_df, Y_df):
    con_zero = Y_df.groupby(by=['chid']).agg({'txn_amt':'sum'})
    con_zero.reset_index(inplace=True)
    con_zero = con_zero[con_zero['txn_amt'] == 0]
    con_zero = list(con_zero['chid'])

    X_df = X_df[~X_df['chid'].isin(con_zero)]
    Y_df = Y_df[~Y_df['chid'].isin(con_zero)]
    return(X_df, Y_df)
    
# 產生qids
def get_qids(Y_df):
    qids= np.array([16]*(len(Y_df)//16))
    return(qids)
