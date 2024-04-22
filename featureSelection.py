import pandas as pd
import numpy as np
import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#olivia kim
def split(xFeat,y):
    xTrain, xTest, yTrain, yTest = train_test_split(xFeat,y, test_size=0.3)
    return xTrain, xTest, yTrain, yTest


def select_features(xTrain, xTest):
    corr_matrix  = xTrain.corr().abs()
       
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.60)]
    xTrain = xTrain.drop(xTrain[to_drop], axis=1)
    xTest = xTest.drop(xTest[to_drop], axis=1)
    return xTrain, xTest

def main():
    
    # df = pd.read_csv("data.csv")
    # y = df['label']
    # y = preprocessing.onehot(y,'label')
    # df = df.drop(columns=['index','id.orig_h',"id.resp_h","service","missed_bytes","history","label"])
    # df = preprocessing.remove_na(df)
    # encode_columns = ["id.resp_p",
    #                   "proto",
    #                   "conn_state"
    #                   ]   
    # df = preprocessing.onehot(df,encode_columns)
    # df = preprocessing.fill_na(df)
    # xTrain, xTest, yTrain, yTest = split(df, y)
    # prep_columns = ["id.orig_p",
    #                 "duration",
    #                 "orig_bytes",
    #                 "resp_bytes",
                    
    #                 "orig_pkts",
    #                 "orig_ip_bytes",
    #                 "resp_pkts",
    #                 "resp_ip_bytes"
    #                 ]
    # xTrain, xTest = preprocessing.minmax_range(xTrain,xTest,prep_columns)
    # plt.figure()
    # sns.heatmap(xTrain.corr(), annot=False, cmap='coolwarm',annot_kws={"size": 5})
    # plt.title('Pearson Correlation Matrix')
    # plt.show()
    xTrain = pd.read_csv("xTrain.csv")
    yTrain = pd.read_csv("yTrain.csv")
    xTest = pd.read_csv("xTest.csv")
    yTest = pd.read_csv("yTest.csv")

    xTrain, xTest = select_features(xTrain, xTest)   
    sns.heatmap(xTrain.corr(), annot=True, cmap='coolwarm',annot_kws={"size": 5})
    plt.title('Pearson Correlation Matrix')
    plt.show() 
    xTrain.to_csv("xTrain_dropped.csv", index=False)
    xTest.to_csv("xTest_dropped.csv", index=False)
    

if __name__ == "__main__":
    main()