# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 00:06:33 2021

@author: shen
"""

import pandas as pd

filename = 'Data//simple_data//train.csv'

df = pd.read_csv(filename)

# predict_shop_tag
with open('Data//simple_data//predict_shop_tag.txt') as f:
    predict_shop_tag = f.read().splitlines()
    
df = df[df['shop_tag'].isin(predict_shop_tag)]

df = df[df['dt']>=18]

# groupby
agg_func = {'txn_amt': ['sum','nunique']} #,'mean'
result = df.groupby(by=['chid','shop_tag']).agg(agg_func)
result.columns = ['_'.join(col).strip() for col in result.columns.values]
result.reset_index(inplace=True)

# 
result["sum_nunique"] = result["txn_amt_sum"] * result["txn_amt_nunique"]
result["rank"] = result.groupby("chid")["sum_nunique"].rank(ascending=False) #"dense"

# 測試平均猜測結果
result_rank3 = result[result["rank"]<=3.0]

result_rank3['rank'] = result_rank3['rank'].astype('int').astype('str')
result_rank3['chid'] = result_rank3['chid'].astype('str')
result_rank3['chid'] = result_rank3['chid'].astype('str')

test = pd.pivot_table(result_rank3, values='shop_tag', index=['chid'], columns=['rank'], aggfunc='first')
test.reset_index(inplace=True)
test.columns = ['chid','top1','top2','top3']
# 補空
submission = pd.read_csv('Data//submission//submission.csv')
submission['chid'] = submission['chid'].astype('str')
submission = pd.merge(submission['chid'].to_frame(),test, on="chid", how='left')
#submission.to_csv('simple_data//submission_1029avg.csv',index=False)

# 
#submission = pd.read_csv('simple_data//submission_1029avg.csv')
for col in ['top1','top2','top3']:
    submission[col] = submission[col].fillna(submission[col].mode()[0])
    submission[col] = submission[col].astype('int')
    
submission.to_csv('Data//simple_data//submission_1101_sum_nunique_dt-6_fillna.csv',index=False)
