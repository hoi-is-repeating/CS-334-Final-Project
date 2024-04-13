import argparse
import pandas as pd
#import seaborn as sb
import numpy as np
import extractFeat
#import matplotlib.pyplot as plt


def cal_corr(df):
    """
    Compute the Pearson correlation matrix
    
    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    corrDF : pandas.DataFrame
        The correlation between the different columns
    """
    corrDF = df.corr(method = 'pearson')
    """
    btrain,btest = extractFeat.extract_binary(trainDF['TWTOKEN'],testDF['TWTOKEN'],3)
    bcorr = cal_corr(btrain)
    plt.figure()
    sb.heatmap(bcorr,annot=True,cmap='coolwarm')
    plt.title('Correlation Matrix for Binary Representation')
    plt.show()
    
    tfidf_train, tfidf_test = extractFeat.extract_tfidf(trainDF['TWTOKEN'],testDF['TWTOKEN'],3)
    tfidf_corr  = cal_corr(tfidf_train)
    plt.figure()
    sb.heatmap(tfidf_corr,annot=True,cmap='coolwarm')
    plt.title('Correlation Matrix for TF-IDF Representation')
    plt.show()
    
    trainDF = pd.concat([trainDF, btrain, tfidf_train], axis=1)
    testDF = pd.concat([testDF, btest, tfidf_test], axis=1)
    
    """
    return corrDF

def transform(trainDF, testDF, delta, gamma, columns):
    corr_matrix = trainDF[columns].corr().abs()
    to_drop = []
    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[1]):
            if abs(corr_matrix.iloc[i, j]) > delta:
                # Add the feature with the higher index to the drop list
                to_drop.append(corr_matrix.columns[j])
    # Drop highly correlated features from the training data
    trainDF.drop(columns=to_drop, inplace=True)

    # Drop the same features from the test data
    testDF.drop(columns=to_drop, inplace=True)

    # Identify features with low correlation to the target
    target_corr = trainDF.corr().abs().iloc[:-1, -1]
    low_corr_features = target_corr[target_corr < gamma].index.tolist()

    # Drop low correlation features from the training data
    trainDF.drop(columns=low_corr_features, inplace=True)

    # Drop the same features from the test data
    testDF.drop(columns=low_corr_features, inplace=True)

    return trainDF, testDF
    
def select_features(trainDF, testDF):
    """
    Preprocess the features
    
    Parameters
    ----------
    trainDF : pandas.DataFrame
        the training dataframe
    testDF : pandas.DataFrame
        the test dataframe

    Returns
    -------
    trainDF : pandas.DataFrame
        return the feature-selected trainDF dataframe
    testDF : pandas.DataFrame
        return the feature-selected testDT dataframe
    """
    k = 3
    #binary first
    train, test = extractFeat.extract_binary(trainDF["TWTOKEN"], testDF["TWTOKEN"],k)
    delta = 0.2
    gamma = 0.1
    trainDF.drop(columns=["TWTOKEN"], inplace=True)
    testDF.drop(columns=["TWTOKEN"], inplace=True)

    trainDF, testDF = transform(trainDF, testDF, delta, gamma, columns=train.columns.tolist())

    return trainDF, testDF



def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("inTrain",
                        help="filename of the training data")
    parser.add_argument("inTest",
                        help="filename of the test data")
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")

    args = parser.parse_args()
    # load the train and test data
    train_df = pd.read_csv(args.inTrain)
    test_df = pd.read_csv(args.inTest)

    print("Original Training Shape:", train_df.shape)
    # calculate the training correlation
    train_df, test_df = select_features(train_df,
                                        test_df)
    print("Transformed Training Shape:", train_df.shape)
    # save it to csv
    train_df.to_csv(args.outTrain, index=False)
    test_df.to_csv(args.outTest, index=False)


if __name__ == "__main__":
    main()



