# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 21:45:52 2021

@author: shen
"""

# 轉換type，減少空間浪費
def change_type2(X_df):
    # category
    X_df['shop_tag'] = X_df['shop_tag'].astype('category')
    # prop
    prop_col =[col for col in X_df.columns if col.find('prop')!=-1]
    for col in prop_col:
        X_df[col] = X_df[col]*1000000 #100->1000000
    # txn_cnt
    txn_cnt_col =[col for col in X_df.columns if col.find('txn_cnt')!=-1]
    for col in txn_cnt_col:
        X_df[col] = X_df[col]*1000000
    
    # pct
    pct_cnt_col =[col for col in X_df.columns if col.find('pct')!=-1]
    for col in pct_cnt_col:
        X_df[col] = X_df[col]*1000000
        X_df[col] = X_df[col].fillna(0)
        X_df[col] = X_df[col].astype('int',errors='ignore')
        
    # txn
    txn_col =[col for col in X_df.columns if col.find('txn')!=-1]
    for col in txn_col:
        X_df[col] = X_df[col].fillna(0)
        X_df[col] = X_df[col].astype('int',errors='ignore')
    
    # dt_gap
    dt_gap_col =[col for col in X_df.columns if col.find('dt_gap')!=-1]
    for col in dt_gap_col:
        X_df[col] = X_df[col].fillna(33)
        X_df[col] = X_df[col].astype('int',errors='ignore')

    return(X_df)
    

# 轉換type，減少空間浪費
def total_float2int(X_df):
    df_type = X_df.dtypes
    df_float64 = df_type[df_type=='float64']
    for col in df_float64.index:
        X_df[col] = X_df[col].fillna(0)
        X_df[col] = X_df[col].astype('int32',errors='ignore')
    return(X_df)
