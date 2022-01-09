import gc
import pandas as pd 
from Util.data_for_training_v1 import *
from Util.data_processing_v1 import *
from Util.train_model_v1 import *
from Util.data_processing_v2 import *
from Util.data_for_training_v2 import *

# In[para]
month_range=[-1,-3,-8,-12,-18]
save_path = r'./'
gene_df = pd.read_excel(f'{save_path}/gene_df.xlsx')

# gene feature
def gene_XY_df(dt, Y_bool=True): #使用上方數據
    total_tag_last_dt = gene_total_tag_last_dt2(df,combo_total, predict_dt=dt)
    groupby_total_tag_df = gene_groupby_total_tag_df(df, combo_total, predict_dt=dt, month_range=[-18])
    
    df_filter = filter_df_by_predict_shop_tag(df, predict_shop_tag)
    
    df_groupby = gene_groupby_df3(df_filter, combo_predict, predict_dt=dt, month_range=month_range)
    
    df_total_txn_amt = cal_total_txn_amt(df, predict_shop_tag, combo_predict, predict_dt=dt, month_range=month_range)

    predict_tag_last_dt = gene_predict_tag_last_dt(df_filter, combo_predict, predict_dt=dt, delete_col=['txn_amt'], shift_list=[1,2,3,4,5]) # 修改
    
    print('merge start')
    X_df = merge_total_df(combo_predict, 
                          by_chid_list=[total_tag_last_dt,groupby_total_tag_df,df_chid_info,begin_dt], 
                          by_tag_list=[df_groupby,df_total_txn_amt,predict_tag_last_dt])
    print('merge end')   
    
    X_df= cal_group_prop(X_df, month_range=month_range)

    for month in month_range:
        X_df[f'txn_amt_{month}_rank_desc'] = X_df.groupby('chid')[f'txn_amt_{month}_sum'].rank("dense", ascending=False)
        X_df[f'txn_amt_{month}_rank_asc'] = X_df.groupby('chid')[f'txn_amt_{month}_sum'].rank("dense", ascending=True)
    
    X_df = change_type2(X_df)
    
    if Y_bool:
        Y_df = gene_y_data(df_filter, combo_predict, predict_dt=dt)
        X_df, Y_df = del_zero(X_df, Y_df)
        return(X_df, Y_df)
    else:
        return(X_df)

def X_df_cal_trend2(X_df): #使用上方數據
    X_df = cal_trend2(X_df,month_range=month_range,txn_cols=['txn_amt','txn_amt_prop','avg_txn_amt',
                                                             'txn_amt_group_prop','txn_amt_pt_group_prop']) # 修改
    X_df = gene_trend_groupby_chid(X_df)
    return(X_df)

# In[]
predict_shop_tag, total_shop_tag = read_shop_tag()

df = read_train_df()
df['txn_cnt'][df['txn_cnt']==0] = 1

df = cal_txn_amt_prop(df)
df = gene_avg_txn_amt(df)
df_chid_total = read_chid_df()

df, df_chid = reduce_df_num(0, 1, df, df_chid_total)

df_chid_info = read_chid_info2()
begin_dt = read_begin_dt()

combo_total = gene_combo_chid_shoptag(df_chid, total_shop_tag)
combo_predict = gene_combo_chid_shoptag(df_chid, predict_shop_tag)

# In[loop]
for i, row in gene_df.iterrows():
    train_dt = int(row['train_dt'])
    name = f'dt{train_dt}'
    
    if train_dt==25:
        # gene_X_df
        X_train= gene_XY_df(train_dt, Y_bool=False)
        X_train = X_df_cal_trend2(X_train)
        X_train = total_float2int(X_train)
        
        X_train.to_pickle(f"{save_path}/data/X_{name}.pkl")
        X_train_head = X_train[0:1600]
        X_train_head.to_excel(f"{save_path}/data/X_{name}_head.xlsx")
        del X_train
    else:
        # gene_XY_df
        X_train, Y_train = gene_XY_df(train_dt)
        X_train = X_df_cal_trend2(X_train)
        X_train = total_float2int(X_train)
        
        X_train.to_pickle(f"{save_path}/data/X_{name}.pkl")
        Y_train.to_pickle(f"{save_path}/data/Y_{name}.pkl")
        X_train_head = X_train[0:1600]
        X_train_head.to_excel(f"{save_path}/data/X_{name}_head.xlsx")
        del X_train, Y_train
    
    gc.collect()

del df
gc.collect()
