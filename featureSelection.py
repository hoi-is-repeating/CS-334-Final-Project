import pandas as pd
import numpy as np
import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split




def select_features(xTrain, xTest):
    corr_matrix  = xTrain.corr().abs()
       
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.60)]
    xTrain = xTrain.drop(xTrain[to_drop], axis=1)
    xTest = xTest.drop(xTest[to_drop], axis=1)
    return xTrain, xTest

def selection():
<<<<<<< HEAD
    df = pd.read_csv("data.csv")
    y = df['label']
    y = preprocessing.onehot(y,'label')
    y = y.drop(columns=['Malicious'],inplace=True)
=======
    
    df = pd.read_csv("data.csv")
    y = df['label']
    y = preprocessing.onehot(y,'label')
>>>>>>> parent of 3e417bc (new csvs to prevent overfitting)
    df = df.drop(columns=['index','id.orig_h',"id.resp_h","service","missed_bytes","history","label"])
    df = preprocessing.remove_na(df)
    encode_columns = ["id.resp_p",
                      "proto",
                      "conn_state"
                      ]   
    df = preprocessing.onehot(df,encode_columns)
    df = preprocessing.fill_na(df)
<<<<<<< HEAD
    df = preprocessing.minmax_range(df,df.columns)
    xTrain, xTest, yTrain, yTest = train_test_split(df,y, test_size=0.3)

    
=======
    xTrain, xTest, yTrain, yTest = split(df, y)

    xTrain, xTest = preprocessing.minmax_range(xTrain,xTest,df.columns)
>>>>>>> parent of 3e417bc (new csvs to prevent overfitting)
    plt.figure()
    sns.heatmap(xTrain.corr(), annot=False, cmap='coolwarm',annot_kws={"size": 5})
    plt.title('Pearson Correlation Matrix')
    plt.show()
    xTrain, xTest = select_features(xTrain, xTest)   
    sns.heatmap(xTrain.corr(), annot=True, cmap='coolwarm',annot_kws={"size": 5})
    plt.title('Pearson Correlation Matrix')
    plt.show() 
    xTrain.to_csv("xTrain.csv", index=False)
    xTest.to_csv("xTest.csv", index=False)
    yTrain.to_csv("yTrain.csv", index=False)
    yTest.to_csv("yTest.csv", index=False)
<<<<<<< HEAD
    
    return xTrain,xTest,yTrain,yTest
=======
    return xTrain,xTest,yTrain,yTest
>>>>>>> parent of 3e417bc (new csvs to prevent overfitting)
