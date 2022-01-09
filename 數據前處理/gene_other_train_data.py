import pandas as pd 
import numpy as np

train_path = 'Data//simple_data//train.csv'

# In[開始消費日期]
df = pd.read_csv(train_path)

chid_min_dt = df.groupby(by=['chid']).agg({'dt':'min'})
chid_min_dt = chid_min_dt.reset_index()
chid_min_dt = chid_min_dt.rename(columns={'dt':'begin_dt'})
chid_min_dt.to_csv('Data//simple_data//begin_dt.csv',index=False)

# In[卡片資訊]
df = pd.read_csv('Data//simple_data//chid_info.csv')
df_head = df[0:200]

df_without_duplicates = df.drop_duplicates(subset=None, keep='first')
df_without_duplicates.to_csv('Data//simple_data//chid_info_without_duplicates.csv', index=False)

df_without_duplicates = pd.read_csv('Data//simple_data//chid_info_without_duplicates.csv')

df_group = df_without_duplicates.groupby(['chid']).agg({'dt': [np.max]})
df_group.reset_index(inplace=True)
df_group.columns = ['chid', 'dt']

df_group = pd.merge(df_group,df_without_duplicates, on=['chid', 'dt'], how='left')
df_group.to_csv('Data//simple_data//chid_info_new.csv', index=False)

