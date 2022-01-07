# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 18:56:39 2021
資料處理函數
@author: shen
"""

import gc
gc.collect()
import pandas as pd
import numpy as np
from itertools import combinations

# In[資料讀取相關]
def read_train_df(filename='Data//simple_data//train.csv'):
    print('read df')
    df = pd.read_csv(filename)
    df['txn_amt'] = df['txn_amt'].astype('int')
    return(df)

# 計算總消費額&消費百分比
def cal_txn_amt_prop(df):
    df_dtsum = df.groupby(by=['chid','dt']).agg({'txn_amt': 'sum', 'txn_cnt': 'sum'})
    df_dtsum.reset_index(inplace=True)
    df_dtsum = df_dtsum.rename(columns={'txn_amt':'txn_amt_dt_sum', 'txn_cnt':'txn_cnt_dt_sum'})
    df = pd.merge(df,df_dtsum, on=['chid','dt'], how='left')
    df['txn_amt_prop'] = df['txn_amt']/df['txn_amt_dt_sum']
    return(df)

def read_begin_dt(filename='Data//simple_data//begin_dt.csv'):
    df = pd.read_csv(filename)
    return(df)

# total_chid 
def read_chid_df(filename='Data//simple_data//chid.csv'):
    df_chid_total = pd.read_csv(filename)
    return(df_chid_total)

#減少資料數(by chid)
def reduce_df_num(start, end, df, df_chid_total): #start
    from_ = int(start*len(df_chid_total))
    to_ = int(end*len(df_chid_total))
    df_chid = df_chid_total[from_:to_] 
    df = df[df['chid'].isin(df_chid['chid'])]
    return(df, df_chid)

def read_shop_tag():
    # predict_shop_tag
    with open('Data//simple_data//predict_shop_tag.txt') as f:
        predict_shop_tag = f.read().splitlines()
        predict_shop_tag.sort()
    # total_shop_tag
    with open('Data//simple_data//total_shop_tag.txt') as f:
        total_shop_tag = f.read().splitlines()
        total_shop_tag.sort()
    return(predict_shop_tag, total_shop_tag)

def read_chid_info(filename='Data//simple_data//chid_info_new.csv'):
    # chid_info_new,只包含最新的info
    df_chid_info = pd.read_csv('Data//simple_data//chid_info_new.csv')
    del df_chid_info['dt']
    # 定義類別資料
    cate_col = ['masts','educd','trdtp','naty','poscd','cuorg','gender_code','age','primary_card']
    for col in cate_col:
        df_chid_info[col] = df_chid_info[col].astype('category')
    return(df_chid_info)

def read_chid_info2(filename='Data//simple_data//chid_info_new.csv'):
    # chid_info_new,只包含最新的info
    df_chid_info = pd.read_csv('Data//simple_data//chid_info_new.csv')
    del df_chid_info['dt']
    # 定義類別資料
    for col in ['masts','trdtp','naty','poscd','cuorg','gender_code','primary_card']:
        df_chid_info[col] = df_chid_info[col].fillna(-99)
        df_chid_info[col] = df_chid_info[col].astype('int')
        df_chid_info[col] = df_chid_info[col].astype('category')

    for col in ['educd','age']:
        df_chid_info[col] = df_chid_info[col].fillna(-99)
        df_chid_info[col] = df_chid_info[col].astype('int')
    return(df_chid_info)
    
def gene_combo_chid_shoptag(df_chid, shop_tag):
    X1 = pd.DataFrame({'shop_tag':shop_tag, 'tmp':0})
    X2 = df_chid.copy()
    X2['tmp'] = 0
    combo = pd.merge(X2,X1,on='tmp')
    del combo['tmp']
    return(combo)

def filter_df_by_predict_shop_tag(df, predict_shop_tag):
    df_filter = df.copy()
    df_filter = df_filter[df_filter['shop_tag'].isin(predict_shop_tag)] #篩選predict_shop_tag
    return(df_filter)
    
# In[資料特徵處理相關]
# 計算所有tag上次消費時間
def gene_total_tag_last_dt(df,combo_total, predict_dt, delete_col=['dt_gap', 'txn_amt']):
    print('GeneWholeTagLastdt')
    dt_df = combo_total.copy()
    dt_df['dt'] = predict_dt
    dt_df['txn_amt'] = 0
    Lastdt = df[['dt', 'chid', 'shop_tag','txn_amt']][df['dt']<predict_dt]
    Lastdt = pd.concat([Lastdt, dt_df], ignore_index=True)
    
    for i in [1]:
        Lastdt[f'shift_{i}_dt'] = Lastdt.groupby(['chid','shop_tag'])['dt'].shift(i)
        Lastdt[f'shift_{i}_dt_gap'] = Lastdt['dt'] - Lastdt[f'shift_{i}_dt']
        Lastdt[f'shift_{i}_txn_amt'] = Lastdt.groupby(['chid','shop_tag'])['txn_amt'].shift(i)
        Lastdt[f'shift_{i}_txn_amt_divide_dt'] = Lastdt[f'shift_{i}_txn_amt']/Lastdt[f'shift_{i}_dt_gap']
        del Lastdt[f'shift_{i}_dt'], 
        for col in delete_col:
            del Lastdt[f'shift_{i}_{col}']
    Lastdt = Lastdt[Lastdt['dt']==predict_dt]
    del Lastdt['dt'], Lastdt['txn_amt']
    
    result = Lastdt.pivot_table(Lastdt, index=['chid'], columns=['shop_tag'], aggfunc='first')
    result.reset_index(inplace=True)
    result.columns = ['_'.join(col).strip() for col in result.columns.values]
    result.rename(columns={'chid_': 'chid'}, inplace=True)

    return(result)

# 計算上次消費時間
def gene_predict_tag_last_dt(df,combo_predict, predict_dt, delete_col=['dt_gap'], shift_list=[1,2,3]):
    print('GeneLastdt')
    dt_df = combo_predict.copy()
    dt_df['dt'] = predict_dt
    dt_df['txn_amt'] = 0
    Lastdt = df[['dt', 'chid', 'shop_tag','txn_amt']][df['dt']<predict_dt]
    Lastdt = pd.concat([Lastdt, dt_df], ignore_index=True)
    
    for i in shift_list:
        Lastdt[f'shift_{i}_dt'] = Lastdt.groupby(['chid','shop_tag'])['dt'].shift(i)
        Lastdt[f'shift_{i}_dt_gap'] = Lastdt['dt'] - Lastdt[f'shift_{i}_dt']
        Lastdt[f'shift_{i}_txn_amt'] = Lastdt.groupby(['chid','shop_tag'])['txn_amt'].shift(i)
        Lastdt[f'shift_{i}_txn_amt_divide_dt'] = Lastdt[f'shift_{i}_txn_amt']/Lastdt[f'shift_{i}_dt_gap']
        del Lastdt[f'shift_{i}_dt']
        if len(delete_col)>0:
            for col in delete_col:
                del Lastdt[f'shift_{i}_{col}']
    Lastdt = Lastdt[Lastdt['dt']==predict_dt]
    del Lastdt['dt'], Lastdt['txn_amt']
    return(Lastdt)
    
# XTrain
agg_func = {'txn_amt': ['sum','nunique'],
            'txn_cnt': ['sum'],
            'txn_amt_dt_sum': ['sum'],
            'txn_amt_prop': ['sum']}

# groupby(cal_txn_amt_prop後執行)
def gene_groupby_df(df, combo_predict, predict_dt, month_range=[-1,-3,-6,-12], agg_func_set=agg_func):
    print(f'predict_dt:{predict_dt}')
    X_train = combo_predict.copy()
    
    for month in month_range:
        dt = month+predict_dt
        print(dt)
        df_sub_train = df[(df['dt']>=dt)&(df['dt']<predict_dt)]
        # groupby
        if month == -1:
            agg_func = {'txn_amt': ['sum'],
                        'txn_cnt': ['sum'],
                        'txn_amt_dt_sum': ['sum'],
                        'txn_amt_prop': ['sum']}
        elif month >= -6 :
            agg_func = agg_func_set

        result = df_sub_train.groupby(by=['chid','shop_tag']).agg(agg_func)
        result.columns = [f'_{month}_'.join(col).strip() for col in result.columns.values]
        result.reset_index(inplace=True)

        result = result.fillna(0)
        
        X_train = pd.merge(X_train,result, on=['chid','shop_tag'], how='left')
    X_train = X_train.fillna(0)
    return(X_train)

# 計算趨勢(gene_groupby_df後執行)
def cal_trend(df, month_range=[-1,-3,-6,-12],txn_cols=['txn_amt']):
    print(f'cal trend {month_range}')
    for txn_col in txn_cols:
        for month in month_range:
            col = f'{txn_col}_{month}_sum'
            df[col] = df[col]/abs(month) # txn_amt除以月數
                
        combo = list(combinations(month_range, 2))  #兩兩一組，計算趨勢
        for m_new,m_old in combo:
            col_new = f'{txn_col}_{m_new}_sum'
            col_old = f'{txn_col}_{m_old}_sum'
            col_trend = f'{txn_col}_sum_{m_old}to{m_new}'
            df[col_trend] = df[col_new] - df[col_old]
    return(df)


# In[merge]
def merge_total_df(combo_predict, by_chid_list, by_tag_list):
    X_df = combo_predict.copy()
    for by_chid in by_chid_list:
        X_df = pd.merge(X_df,by_chid, on=['chid'], how='left')
    for by_tag in by_tag_list:
        X_df = pd.merge(X_df,by_tag, on=['chid','shop_tag'], how='left')
    
    return(X_df)
    
# In[Y]
def gene_y_data(df_filter, combo_predict, predict_dt):
    y = df_filter[df_filter['dt']==predict_dt]
    y = y[['chid','shop_tag','txn_amt']]
    y = pd.merge(combo_predict,y, on=["chid",'shop_tag'], how='left')
    y = y.fillna(0)
    y['txn_amt'] = y['txn_amt'].astype('int')
    return(y)