import pandas as pd 

# In[]
filename = 'Data//simple_data//tbrain_cc_training_48tags_hash_final_simple.csv'

# In[]
chunksize = 1000000
chunks = pd.read_csv(filename, chunksize=chunksize, iterator=True)

df = pd.DataFrame()
for i, chunk in enumerate(chunks):
    print(i)
    chunk = chunk[['dt','chid','shop_tag','txn_cnt','txn_amt']]
    df = pd.concat([df,chunk], ignore_index=True)

df.to_csv('Data//simple_data//train.csv', index=False)

# In[]
chunksize = 1000000
chunks = pd.read_csv(filename, chunksize=chunksize, iterator=True)

df = pd.DataFrame()
for i, chunk in enumerate(chunks):
    print(i)
    chunk = chunk[['dt','chid','masts','educd','trdtp','naty','poscd','cuorg','slam','gender_code','age','primary_card']]
    df = pd.concat([df,chunk], ignore_index=True)

df.to_csv('Data//simple_data//chid_info.csv', index=False)