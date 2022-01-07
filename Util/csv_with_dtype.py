# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 23:34:03 2021

@author: shen
"""
import pandas as pd

def to_csv(df, path):
    # Prepend dtypes to the top of df (from https://stackoverflow.com/a/43408736/7607701)
    df.loc[-1] = df.dtypes
    df.index = df.index + 1
    df.sort_index(inplace=True)
    # Then save it to a csv
    df.to_csv(path, index=False)

def read_csv(path):
    # Read types first line of csv
    dtypes = pd.read_csv('tmp.csv', nrows=1).iloc[0].to_dict()
    # Read the rest of the lines with the types from above
    return pd.read_csv('tmp.csv', dtype=dtypes, skiprows=[1])

if __name__=='__main__':
    
    # Create Example data with types
    df = pd.DataFrame({
        'words': ['foo', 'bar', 'spam', 'eggs'],
        'nums': [1, 2, 3, 4]
    }).astype(dtype={
        'words': 'object',
        'nums': 'int8'
    })
    
    print('Before: \n{}\n'.format(df.dtypes))

    to_csv(df, 'tmp.csv')
    df = read_csv('tmp.csv')
    
    print('After: \n{}\n'.format(df.dtypes))

    