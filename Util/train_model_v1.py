# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 20:43:27 2021

@author: shen
"""
import pandas as pd
import numpy as np

def save_features_importance(train_columns, feature_importance, filename):
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = train_columns
    fold_importance_df["importance"] = feature_importance
    fold_importance_df.to_csv(f"{filename}.csv", index=False)
    
def get_label_gain(Y_train, Y_val):
    max_txn_amt = np.max([np.max(Y_train['txn_amt']),np.max(Y_val['txn_amt'])])
    label_gain = list(range(0, max_txn_amt+1))
    return(label_gain)

def get_head_feature(path, head=80):
    feature_top = pd.read_csv(path)
    feature_top = feature_top.sort_values('importance',ascending=False).reset_index()
    feature_top = feature_top[0:head]
    train_columns = list(feature_top['Feature'])
    return(train_columns)