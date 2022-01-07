# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 23:13:31 2021

@author: shen
"""

import pandas as pd 
from Util.data_for_training_v1 import *
from Util.data_processing_v1 import *
from Util.train_model_v1 import *
from Util.data_processing_v2 import *
from Util.data_for_training_v2 import *

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


# In[para]
save_path = r'./RunFlow/20211210_more_feature/'

# In[loop]
# XY_df
dts = [24,23,22] #[24,23,22]

for dt in dts:
    X_total = pd.read_pickle(f"{save_path}/data/X_dt{dt}.pkl")
    Y_total = pd.read_pickle(f"{save_path}/data/Y_dt{dt}.pkl")
    train_columns = get_train_columns(X_total)
    
    #folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4500)
    folds = KFold(n_splits=5, shuffle=True, random_state=4500)
    
    feature_importance_df = pd.DataFrame()
    
    df_chid_info = read_chid_info()
    
    #for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_chid_info['chid'],df_chid_info['trdtp'].values)):
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_chid_info['chid'])):
        if fold_>=0:
            print(fold_)
            trn_chid = df_chid_info['chid'][trn_idx]
            val_chid = df_chid_info['chid'][val_idx]
            
# =============================================================================
#             # XY_df
#             X_train = X_total[X_total['chid'].isin(trn_chid)]
#             Y_train = Y_total[Y_total['chid'].isin(trn_chid)]
#             X_val =  X_total[X_total['chid'].isin(val_chid)]
#             Y_val = Y_total[Y_total['chid'].isin(val_chid)]
# =============================================================================
        
            qids_train = get_qids(Y_total[Y_total['chid'].isin(trn_chid)])
            qids_val = get_qids(Y_total[Y_total['chid'].isin(val_chid)])
            
            label_gain = get_label_gain(Y_total[Y_total['chid'].isin(trn_chid)], Y_total[Y_total['chid'].isin(val_chid)])
            
            param = {'num_leaves': 750, 
                     'min_data_in_leaf': 16*200, 
                     'objective':'lambdarank', # 'lambdarank', 'regression'
                     'max_depth': 22,
                     'max_bin':300, # 默認 =255 #特徵值將被切割的最大份數
                     #'min_data_in_bin':3, #默認 =3 #一個 bin 內的最少數據數
                     'learning_rate': 0.0025, #0.05
                     "boosting": "gbdt",
                     "feature_fraction": 0.80,
                     "bagging_freq": 1,
                     "bagging_fraction": 0.75 ,
                     #"lambda_l1": 0.015,
                     #"lambda_l2": 0.015,
                     "bagging_seed": 11,
                     "metric": 'ndcg', 
                     'eval_at':[3], 
                     'lambdarank_truncation_level':16,
                     'label_gain': label_gain,
                     "random_state": 6666,
                     'bin_construct_sample_cnt': 2000, 
                     'histogram_pool_size':1000,
                     "verbosity": -1}

            trn_data = lgb.Dataset(X_total[X_total['chid'].isin(trn_chid)][train_columns], label=Y_total[Y_total['chid'].isin(trn_chid)]['txn_amt'])
            val_data = lgb.Dataset(X_total[X_total['chid'].isin(val_chid)][train_columns], label=Y_total[Y_total['chid'].isin(val_chid)]['txn_amt'])
            trn_data.set_group(qids_train)
            val_data.set_group(qids_val)
            
            num_round = 100000
            clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=25, early_stopping_rounds = 100)
            
            clf.save_model(f'{save_path}/model/dt{dt}_model{fold_}.txt', num_iteration=clf.best_iteration) 
            
            score = clf.best_score
            with open(f'{save_path}/log.txt', 'a') as f:
                f.write(f'{fold_}\n')
                f.write(f'{str(score)}\n')
            
            # save_features_importance
            save_features_importance(train_columns, clf.feature_importance(), f'{save_path}/features_importance_{fold_}')
        
        
