# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 22:19:37 2021

@author: shen
"""


def GetTop3(df):
    df = df.copy()
    chid = df['chid']
    del df['chid']
    Tops =pd.DataFrame(df.apply(lambda x:list(df.columns[np.array(x).argsort()[::-1][:3]]), axis=1).values.tolist(),  columns=['top1', 'top2', 'top3'])
    
    Tops = pd.concat([chid, Tops], axis=1)
    return(Tops)