import pandas as pd

filename = 'tbrain_cc_training_48tags_hash_final.csv'

# In[]
sub_id = [str(10000000+i) for i in range(0,200)]

chunksize = 20000 # *1649
df = pd.DataFrame()
chunks = pd.read_csv(filename, chunksize=chunksize, iterator=True)
for i, chunk in enumerate(chunks):
    print(i)
    df_sub = chunk[chunk['chid'].isin(sub_id)]
    df = pd.concat([df,df_sub], ignore_index=True)

df.to_csv(f'tbrain_cc_training_48tags_hash_final_simple.csv', index=False)

