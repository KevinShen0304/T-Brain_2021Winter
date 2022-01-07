# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:45:25 2021
# https://stackoverflow.com/questions/45870220/using-pandas-to-efficiently-read-in-a-large-csv-file-without-crashing
@author: shen
"""

import pandas as pd
#df = pd.read_csv('org_data//需預測的顧客名單及提交檔案範例//需預測的顧客名單及提交檔案範例.csv')

filename = 'Data//org_data//tbrain_cc_training_48tags_hash_final.csv'

# In[]
chunksize = 20000 # *1649
dt = 1
df = pd.DataFrame()
chunks = pd.read_csv(filename, chunksize=chunksize, iterator=True)
for i, chunk in enumerate(chunks):
    print(i, dt)
    if (len(chunk['dt']) - (chunk['dt']==dt).sum()) ==0: #判斷是否全部皆為dt
        df = pd.concat([df,chunk], ignore_index=True)
    else:
        # 把剩餘的加入
        chunk_old = chunk[chunk['dt']==dt]
        df = pd.concat([df,chunk_old], ignore_index=True)
        df.to_csv(f'data_bydt//train_dt{dt}.csv', index=False)
        # 開啟新的
        df = pd.DataFrame() #重製
        dt+=1
        chunk_new = chunk[chunk['dt']==dt]
        df = pd.concat([df,chunk_new], ignore_index=True)
        
df.to_csv(f'data_bydt//train_dt{dt}.csv', index=False)


# dt最大為24

# In[]
chunksize = 1000000
chunks = pd.read_csv(filename, chunksize=chunksize, iterator=True)

df = pd.DataFrame()
for i, chunk in enumerate(chunks):
    print(i)
    chunk = chunk[['dt','chid','shop_tag','txn_cnt','txn_amt']]
    df = pd.concat([df,chunk], ignore_index=True)

df.to_csv('simple_data//train.csv', index=False)

# In[]
chunksize = 1000000
chunks = pd.read_csv(filename, chunksize=chunksize, iterator=True)

df = pd.DataFrame()
for i, chunk in enumerate(chunks):
    print(i)
    chunk = chunk[['dt','chid','masts','educd','trdtp','naty','poscd','cuorg','slam','gender_code','age','primary_card']]
    df = pd.concat([df,chunk], ignore_index=True)

df.to_csv('simple_data//chid_info.csv', index=False)

# In[]
chunksize = 1000000
chunks = pd.read_csv(filename, chunksize=chunksize, iterator=True)

df = pd.DataFrame()
for i, chunk in enumerate(chunks):
    print(i)
    chunk = chunk[['dt','chid','shop_tag','txn_cnt','txn_amt','slam']]
    df = pd.concat([df,chunk], ignore_index=True)

df.to_csv('Data//simple_data//train_slam.csv', index=False)