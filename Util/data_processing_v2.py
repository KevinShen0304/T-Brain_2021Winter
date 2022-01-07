# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:21:05 2021

@author: shen
"""
import gc
gc.collect()
import pandas as pd
from itertools import combinations
from Util.data_processing_v1 import *

# groupby(cal_txn_amt_prop後執行) v2
def gene_groupby_df2(df, combo_predict, predict_dt, month_range=[-1,-3,-6,-12]):
    print(f'predict_dt:{predict_dt}')
    X_train = combo_predict.copy()
    
    for month in month_range:
        dt = month+predict_dt
        df_sub_train = df[(df['dt']>=dt)&(df['dt']<predict_dt)]
        # groupby
        if month == -1:
            agg_func = {'txn_amt': ['sum'],
                        'txn_cnt': ['sum'],
                        'txn_amt_prop': ['sum']}
        elif month >= -3 :
            agg_func = {'txn_amt': ['sum','nunique'],
                        'txn_cnt': ['sum'],
                        'txn_amt_prop': ['sum']}

        result = df_sub_train.groupby(by=['chid','shop_tag']).agg(agg_func)
        result = result/abs(month) 
        result.columns = [f'_{month}_'.join(col).strip() for col in result.columns.values]
        result.reset_index(inplace=True)

        result = result.fillna(0)
        
        X_train = pd.merge(X_train,result, on=['chid','shop_tag'], how='left')
    X_train = X_train.fillna(0)
    return(X_train)
    
# gene_avg_txn_amt
def gene_avg_txn_amt(df):
    df['avg_txn_amt'] = df['txn_amt']/df['txn_cnt']
    return(df)


# 計算總消費額 # 11/29新增
def cal_total_txn_amt(df, predict_shop_tag, combo_predict, predict_dt, month_range=[-1,-3,-6,-12]):
    print(f'predict_dt:{predict_dt}')
    X_train = combo_predict.copy()
    
    for month in month_range:
        dt = month+predict_dt
        df_sub_train = df[(df['dt']>=dt)&(df['dt']<predict_dt)]
        # groupby
        df_total = df_sub_train.groupby(by=['chid']).agg({'txn_amt': 'sum'})
        df_total = df_total/abs(month) # 月數平均
        df_total.reset_index(inplace=True)
        df_total = df_total.rename(columns={'txn_amt':f'txn_amt_total_{month}_sum'})
        
        df_filter = filter_df_by_predict_shop_tag(df_sub_train, predict_shop_tag)
        df_filter_total = df_filter.groupby(by=['chid']).agg({'txn_amt': 'sum'})
        df_filter_total = df_filter_total/abs(month) # 月數平均
        df_filter_total.reset_index(inplace=True)
        df_filter_total = df_filter_total.rename(columns={'txn_amt':f'txn_amt_pt_total_{month}_sum'})
        
        X_train = pd.merge(X_train,df_total, on=['chid'], how='left')
        X_train = pd.merge(X_train,df_filter_total, on=['chid'], how='left')
    X_train = X_train.fillna(0)
    return(X_train)
    
# In[]
# 計算該tag消費額佔總消費額比例 # 11/29新增
def cal_group_prop(X_df, month_range=[-1,-3,-6,-12]):
    for month in month_range:
        X_df[f'txn_amt_group_prop_{month}_sum'] = X_df[f'txn_amt_{month}_sum']/X_df[f'txn_amt_total_{month}_sum']
        X_df[f'txn_amt_pt_group_prop_{month}_sum'] = X_df[f'txn_amt_{month}_sum']/X_df[f'txn_amt_pt_total_{month}_sum']
    return(X_df)

# 計算趨勢(gene_groupby_df、cal_group_prop後執行)
def cal_trend2(df, month_range=[-1,-3,-6,-12],txn_cols=['txn_amt','txn_cnt','txn_amt_prop','txn_amt_group_prop','txn_amt_pt_group_prop']):
    print(f'cal trend {month_range}')
    for txn_col in txn_cols:
        combo = list(combinations(month_range, 2))  #兩兩一組，計算趨勢
        for m_new,m_old in combo:
            col_new = f'{txn_col}_{m_new}_sum'
            col_old = f'{txn_col}_{m_old}_sum'
            col_trend = f'{txn_col}_sum_{m_old}to{m_new}'
            df[col_trend] = df[col_new] - df[col_old]
    return(df)

# In[資料特徵處理相關]
# 計算所有tag上次消費時間
def gene_total_tag_last_dt2(df,combo_total, predict_dt, delete_col=['dt_gap', 'txn_amt'],filter_tag=''):
    print('GeneWholeTagLastdt')
    dt_df = combo_total.copy()
    dt_df['dt'] = predict_dt
    dt_df['txn_amt'] = 0
    Lastdt = df[['dt', 'chid', 'shop_tag','txn_amt']][df['dt']<predict_dt]
    if filter_tag!='':
        Lastdt = Lastdt[Lastdt['shop_tag'].isin(filter_tag)]
    Lastdt = pd.concat([Lastdt, dt_df], ignore_index=True)
    
    for i in [1]:
        Lastdt[f'shift_{i}_dt'] = Lastdt.groupby(['chid','shop_tag'])['dt'].shift(i)
        Lastdt[f'shift_{i}_dt_gap'] = Lastdt['dt'] - Lastdt[f'shift_{i}_dt']
        Lastdt[f'shift_{i}_txn_amt'] = Lastdt.groupby(['chid','shop_tag'])['txn_amt'].shift(i)
        #Lastdt[f'shift_{i}_txn_amt_divide_dt'] = Lastdt[f'shift_{i}_txn_amt']/Lastdt[f'shift_{i}_dt_gap']
        Lastdt[f'shift_{i}_txn_amt_and_dt'] = Lastdt[f'shift_{i}_dt_gap']*10000000 + Lastdt[f'shift_{i}_txn_amt']
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

# In[國內國外、線上線下資料]
def get_train_dt_txn_info(predict_dt):
    dt = str(predict_dt-1)
    dt_txn_info = pd.read_csv(f'Data//data_bydt//train_dt{dt}.csv')
    dt_txn_info['domestic_amt_pct'] = dt_txn_info['domestic_offline_amt_pct'] + dt_txn_info['domestic_online_amt_pct']
    dt_txn_info['overseas_amt_pct'] = dt_txn_info['overseas_offline_amt_pct'] + dt_txn_info['overseas_online_amt_pct']
    dt_txn_info['offline_amt_pct'] = dt_txn_info['domestic_offline_amt_pct'] + dt_txn_info['overseas_offline_amt_pct']
    dt_txn_info['online_amt_pct'] = dt_txn_info['domestic_online_amt_pct'] + dt_txn_info['overseas_online_amt_pct']
    
    type_amt_list = list()
    for type_amt in ['domestic','overseas','offline','online']:
        dt_txn_info[f'{type_amt}_txn_amt'] = dt_txn_info[f'{type_amt}_amt_pct']*dt_txn_info['txn_amt']
        type_amt_list += [f'{type_amt}_txn_amt',f'{type_amt}_amt_pct']
    
    dt_txn_info = dt_txn_info[['chid','shop_tag',
             'domestic_offline_cnt','domestic_online_cnt', 'overseas_offline_cnt', 'overseas_online_cnt',
             'domestic_offline_amt_pct', 'domestic_online_amt_pct','overseas_offline_amt_pct', 'overseas_online_amt_pct']
             + type_amt_list ]
    
    return(dt_txn_info)
# In[]
# groupby(cal_txn_amt_prop後執行) v3
def gene_groupby_df3(df, combo_predict, predict_dt, month_range=[-1,-3,-6,-12]):
    print(f'predict_dt:{predict_dt}')
    X_train = combo_predict.copy()
    
    for month in month_range:
        dt = month+predict_dt
        df_sub_train = df[(df['dt']>=dt)&(df['dt']<predict_dt)]
        # groupby
        if month == -1:
            agg_func = {'txn_amt': ['sum'],
                        'txn_cnt': ['sum'],
                        'txn_amt_prop': ['sum'],
                        'avg_txn_amt': ['sum']}
        elif month != -1 :
            agg_func = {'txn_amt': ['sum','var','count','mean'],
                        'txn_cnt': ['sum','mean'],
                        'txn_amt_prop': ['sum','var','mean'],
                        'avg_txn_amt': ['sum','mean']}

        result = df_sub_train.groupby(by=['chid','shop_tag']).agg(agg_func)
        result.columns = [f'_{month}_'.join(col).strip() for col in result.columns.values]
        for col in result.columns:
            if col.find('sum')!=-1:
                result[col] = result[col]/abs(month) 
            
        result.reset_index(inplace=True)

        result = result.fillna(0)
        
        X_train = pd.merge(X_train,result, on=['chid','shop_tag'], how='left')
    X_train = X_train.fillna(0)
    return(X_train)
    
# groupby(cal_txn_amt_prop後執行) v4
def gene_groupby_df4(df, combo_predict, predict_dt, month_range=[-1,-3,-6,-12]):
    print(f'predict_dt:{predict_dt}')
    X_train = combo_predict.copy()
    
    for month in month_range:
        dt = month+predict_dt
        df_sub_train = df[(df['dt']>=dt)&(df['dt']<predict_dt)]
        # groupby
        if month == -1:
            agg_func = {'txn_amt': ['sum'],
                        'txn_cnt': ['sum'],
                        'txn_amt_prop': ['sum'],
                        'slam':['sum'],
                        'txn_amt_slam':['sum']}
        elif month != -1 :
            agg_func = {'txn_amt': ['sum','std'],
                        'txn_cnt': ['sum'],
                        'txn_amt_prop': ['sum'],
                        'slam':['sum'],
                        'txn_amt_slam':['sum']}

        result = df_sub_train.groupby(by=['chid','shop_tag']).agg(agg_func)
        result = result/abs(month) 
        result.columns = [f'_{month}_'.join(col).strip() for col in result.columns.values]
        result.reset_index(inplace=True)

        result = result.fillna(0)
        
        X_train = pd.merge(X_train,result, on=['chid','shop_tag'], how='left')
    X_train = X_train.fillna(0)
    return(X_train)

def gene_groupby_df5(df, combo_predict, predict_dt, month_range=[-1,-3,-6,-12]):
    print(f'predict_dt:{predict_dt}')
    X_train = combo_predict.copy()
    
    for month in month_range:
        dt = month+predict_dt
        df_sub_train = df[(df['dt']>=dt)&(df['dt']<predict_dt)]
        # groupby
        if month == -1:
            agg_func = {'txn_amt': ['sum'],
                        'txn_cnt': ['sum'],
                        'txn_amt_prop': ['sum']}
        elif month != -1 :
            agg_func = {'txn_amt': ['sum','var','count','mean'],
                        'txn_cnt': ['sum','var','mean'],
                        'txn_amt_prop': ['sum','var','mean']}

        result = df_sub_train.groupby(by=['chid','shop_tag']).agg(agg_func)
        result.columns = [f'_{month}_'.join(col).strip() for col in result.columns.values]
        for col in result.columns:
            if col.find('sum')!=-1:
                result[col] = result[col]/abs(month) 
            
        result.reset_index(inplace=True)

        result = result.fillna(0)
        
        X_train = pd.merge(X_train,result, on=['chid','shop_tag'], how='left')
    X_train = X_train.fillna(0)
    return(X_train)

def gene_groupby_df6(df, combo_predict, predict_dt, month_range=[-1,-3,-6,-12]):
    print(f'predict_dt:{predict_dt}')
    X_train = combo_predict.copy()
    
    for month in month_range:
        dt = month+predict_dt
        df_sub_train = df[(df['dt']>=dt)&(df['dt']<predict_dt)]
        # groupby
        if month == -1:
            agg_func = {'txn_amt': ['sum'],
                        'txn_cnt': ['sum'],
                        'txn_amt_prop': ['sum'],
                        'txn_amt_dt_rank_score': ['sum']}
        elif month != -1 :
            agg_func = {'txn_amt': ['sum','var','count','mean'],
                        'txn_cnt': ['sum','mean'],
                        'txn_amt_prop': ['sum','var','mean'],
                        'txn_amt_dt_rank_score': ['sum']}

        result = df_sub_train.groupby(by=['chid','shop_tag']).agg(agg_func)
        result.columns = [f'_{month}_'.join(col).strip() for col in result.columns.values]
        for col in result.columns:
            if col.find('sum')!=-1:
                result[col] = result[col]/abs(month) 
            
        result.reset_index(inplace=True)

        result = result.fillna(0)
        
        X_train = pd.merge(X_train,result, on=['chid','shop_tag'], how='left')
    X_train = X_train.fillna(0)
    return(X_train)

# In[group by total tag sum avg]
def gene_groupby_total_tag_df(df, combo_total, predict_dt, month_range=[-18]):
    X_train = combo_total.copy()
    
    for month in month_range:
        dt = month+predict_dt
        df_sub_train = df[(df['dt']>=dt)&(df['dt']<predict_dt)]

        agg_func = {'txn_amt': ['sum','count']}

        result = df_sub_train.groupby(by=['chid','shop_tag']).agg(agg_func)
        #result = result/abs(month)
        result.columns = [f'_{month}_'.join(col).strip() for col in result.columns.values]
        result[f'txn_amt_{month}_sum_count'] = result[f'txn_amt_{month}_sum']/result[f'txn_amt_{month}_count']
        del result[f'txn_amt_{month}_count'], result[f'txn_amt_{month}_sum']
        result.reset_index(inplace=True)
        result = result.fillna(0)
        X_train = pd.merge(X_train,result, on=['chid','shop_tag'], how='left')
        
    X_train = X_train.pivot_table(X_train, index=['chid'], columns=['shop_tag'], aggfunc='first')
    X_train.reset_index(inplace=True)
    X_train.columns = ['_'.join(col).strip() for col in X_train.columns.values]
    X_train.rename(columns={'chid_': 'chid'}, inplace=True)
    X_train = X_train.fillna(0)
    
    cols = [c for c in X_train.columns if c not in ['chid']]
    for col in cols:
        X_train[col] = X_train[col].astype('int',errors='ignore')

    return(X_train)


# In[]
def gene_trend_groupby_chid(X_total):
    trend_col = [col for col in X_total.columns if col.find('to-')!=-1]
    
    X_total_groupby_trend_std = X_total.groupby('chid')[trend_col].std()
    X_total_groupby_trend_std.columns = [f'{col}_chid_std' for col in X_total_groupby_trend_std.columns.values]
    X_total_groupby_trend_std = X_total_groupby_trend_std.reset_index()
    
    #X_total_groupby_trend_abssum = X_total.groupby('chid')[trend_col].apply(lambda c: c.abs().sum())
    #X_total_groupby_trend_abssum.columns = [f'{col}_chid_abssum' for col in X_total_groupby_trend_abssum.columns.values]
    #X_total_groupby_trend_abssum = X_total_groupby_trend_abssum.reset_index()
    
    X_total = X_total.merge(X_total_groupby_trend_std, on='chid')
    #X_total = X_total.merge(X_total_groupby_trend_abssum, on='chid')
    return(X_total)