# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 20:05:38 2021

@author: shen
"""
import pandas as pd 
from Util.data_for_training_v1 import *
from Util.data_processing_v1 import *

df = read_train_df()
predict_shop_tag, total_shop_tag = read_shop_tag()
df_filter = filter_df_by_predict_shop_tag(df, predict_shop_tag)
del df_filter['txn_cnt']

# In[]
df_filter['dt'] = df_filter['dt'].astype('str')

result = pd.pivot_table(df_filter, index=['chid','shop_tag'], columns=['dt'], aggfunc='first')

result = result.fillna(0)

result.reset_index(inplace=True)
result.columns = ['_dt'.join(col).strip() for col in result.columns.values]
result.rename(columns={'chid_dt': 'chid',
                       'shop_tag_dt': 'shop_tag'}, inplace=True)

result.to_csv('train_pivot_dt.csv',index=False)

#result_head = result[:300]

# In[]
df_filter['txn_amt_dt_rank'] = df_filter.groupby(['chid','dt'])['txn_amt'].rank("dense", ascending=False)
df_filter['txn_amt_dt_rank'] = df_filter['txn_amt_dt_rank'].astype('int')
del df_filter['txn_amt']

df_filter = df_filter[df_filter['txn_amt_dt_rank']<=3]

df_filter['txn_amt_dt_rank_score'] = df_filter['txn_amt_dt_rank'].replace([1, 2, 3], [3, 2, 1])
del df_filter['txn_amt_dt_rank']

df = df.merge(df_filter, on=['chid','dt','shop_tag'], how='left')
df['txn_amt_dt_rank_score'] = df['txn_amt_dt_rank_score'].fillna(0)

df.to_csv('train_addrank.csv',index=False)