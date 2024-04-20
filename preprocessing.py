import argparse
import pandas as pd
import numpy as np
import sklearn.preprocessing as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



#olivia kim

def minmax_range(x, columns):
    scaler = sk.MinMaxScaler()
    scaler.fit(x[columns])
    x[columns] = scaler.transform(x[columns])
    return x

def remove_na(df):
    return df.dropna(axis=1,how='all')

def fill_na(df):
    column_means = df.mean()
    df = df.fillna(column_means)
    return df

def onehot(df, columns):
    return pd.get_dummies(data = df, columns=columns)

def main():
    df = pd.read_csv("data.csv")
    df = fill_na(df)
    df = remove_na(df)
    print(df)
    df = onehot(df, ['label', 'proto', 'service', 'conn_state', 'history'])
    df['id.orig_h'] = LabelEncoder().fit_transform(df['id.orig_h'])
    df['id.resp_h'] = LabelEncoder().fit_transform(df['id.resp_h'])

    print(df)
    #checking to see if all the columns are numeric
    #do we need scaling? 
    df.drop(columns=['label_Malicious'], inplace=True)

    prep_columns = ["id.orig_p",
                    "duration",
                    "orig_bytes",
                    "resp_bytes",
                    
                    "orig_pkts",
                    "orig_ip_bytes",
                    "resp_pkts",
                    "resp_ip_bytes"]
    
    df = minmax_range(df, prep_columns)

    y = df['label_Benign']
    x = df.drop(columns=['label_Benign'])
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)
    xTrain.to_csv('xTrain.csv', index=False)
    xTest.to_csv('xTest.csv', index=False)
    yTrain.to_csv('yTrain.csv', index=False)
    yTest.to_csv('yTest.csv', index=False)


if __name__ == "__main__":
    main()