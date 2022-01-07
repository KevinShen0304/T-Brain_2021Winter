# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 20:22:07 2021

@author: shen
"""

import pandas as pd 
import numpy as np 
from Util.data_for_training_v1 import *
from Util.data_processing_v1 import *
from Util.train_model_v1 import *
from Util.data_processing_v2 import *
from Util.data_for_training_v2 import *
import lightgbm as lgb
import glob
import os 
import gc

# para
save_path = r'./RunFlow/20211210_more_feature/'

# In[read data]
test_dt = int(25)
X_test = pd.read_pickle(f"{save_path}/data/X_dt{test_dt}.pkl")

train_columns = get_train_columns(X_test)
#train_columns = get_head_feature(f'{save_path}/features_importance.csv', 10000)

# In[predict]
model_path = f'{save_path}\model\*.txt'
model_list = glob.glob(model_path)
model_names = [os.path.basename(x) for x in model_list]

predict_shop_tag, total_shop_tag = read_shop_tag()
df_chid = read_chid_df()
combo_predict = gene_combo_chid_shoptag(df_chid, predict_shop_tag)
Y_test = combo_predict.copy()
    
for i, model_name in enumerate(model_list):
    print(model_name)
    clf = lgb.Booster(model_file=model_name)
    
    # 批次預測
    cut_num = 10
    le = len(X_test)//cut_num 
    test_predict = list()
    for j in range(cut_num):
        print(j)
        test_predict_sub = clf.predict(X_test[train_columns][(j*le):(j*le+le)], num_iteration=clf.best_iteration)
        test_predict.extend(list(test_predict_sub))
        
    Y_test[f'{model_names[i]}'] = test_predict

Y_test.to_csv(f"{save_path}\Y_test.csv", index=False)

# In[]
Y_test['txn_amt']=0
for name in model_names:
    Y_test['txn_amt'] += Y_test[name]
    del Y_test[name]

#
Y_test = Y_test.pivot_table(index=['chid'],columns='shop_tag',values='txn_amt')
Y_test.reset_index(inplace=True)

def GetTop3(df):
    df = df.copy()
    chid = df['chid']
    del df['chid']
    Tops =pd.DataFrame(df.apply(lambda x:list(df.columns[np.array(x).argsort()[::-1][:3]]), axis=1).values.tolist(),  columns=['top1', 'top2', 'top3'])
    
    Tops = pd.concat([chid, Tops], axis=1)
    return(Tops)

Top3 = GetTop3(Y_test)
Top3.to_csv(f'{save_path}\submission_lgbmrank.csv', index=False)
