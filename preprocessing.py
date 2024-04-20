import argparse
import pandas as pd
import numpy as np
import sklearn.preprocessing as sk


#olivia kim

def minmax_range(xTrain, xTest,columns):
    scaler = sk.MinMaxScaler()
    scaler.fit(xTrain[columns])
    xTrain[columns] = scaler.transform(xTrain[columns])
    xTest[columns] = scaler.transform(xTest[columns])
    return xTrain, xTest

def remove_na(df):
    return df.dropna(axis=1,how='all')

def fill_na(df):
    column_means = df.mean()
    df = df.fillna(column_means)
    return df

def onehot(df, columns):
    return pd.get_dummies(data = df, columns=columns)

