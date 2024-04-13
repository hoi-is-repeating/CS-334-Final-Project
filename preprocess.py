import argparse
import pandas as pd
import numpy as np
import sklearn.preprocessing as sk


def standard_scale(xTrain, xTest):
    """
    Preprocess the training data to have zero mean and unit variance.
    The same transformation should be used on the test data. For example,
    if the mean and std deviation of feature 1 is 2 and 1.5, then each
    value of feature 1 in the test set is standardized using (x-2)/1.5.

    Parameters
    ----------
    xTrain : numpy.nd-array with shape (n, d)
        Training data 
    xTest : nd-array with shape (m, d)
        Test data 

    Returns
    -------
    xTrain : nd-array with shape (n, d)
        Transformed training data with mean 0 and unit variance 
    xTest : nd-array with  shape (m, d)
        Transformed test data using same process as training.
    """
    scaler = sk.StandardScaler()
    xTrain_scaled = scaler.fit_transform(xTrain)
    xTest_scaled = scaler.transform(xTest)
    
    return xTrain_scaled, xTest_scaled


def minmax_range(xTrain, xTest):
    """
    Preprocess the data to have minimum value of 0 and maximum
    value of 1.T he same transformation should be used on the test data.
    For example, if the minimum and maximum of feature 1 is 0.5 and 2, then
    then feature 1 of test data is calculated as:
    (1 / (2 - 0.5)) * x - 0.5 * (1 / (2 - 0.5))

    Parameters
    ----------
    xTrain : numpy.nd-array with shape (n, d)
        Training data 
    xTest : nd-array with shape (m, d)
        Test data 

    Returns
    -------
    xTrain : nd-array with shape (n, d)
        Transformed training data with min 0 and max 1.
    xTest : nd-array with  shape (m, d)
        Transformed test data using same process as training.
    """
    scaler = sk.MinMaxScaler()
    xTrain_scaled = scaler.fit_transform(xTrain)
    xTest_scaled = scaler.transform(xTest)
    return xTrain_scaled, xTest_scaled


def add_irr_feature(xTrain, xTest):
    """
    Add 2 features using Gaussian distribution with 0 mean,
    standard deviation of 50.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    xTest : nd-array with shape m x d
        Test data 

    Returns
    -------
    xTrain : nd-array with shape n x (d+2)
        Training data with 2 new noisy Gaussian features
    xTest : nd-array with shape m x (d+2)
        Test data with 2 new noisy Gaussian features
    """
    def gauss(X):
        custom = np.random.normal(loc = 0,scale = 50,size=(X.shape[0],2))
        return np.concatenate((X,custom),axis = 1)
    scaler = sk.FunctionTransformer(gauss)
    xTrain_scaled = scaler.transform(xTrain)
    xTest_scaled = scaler.transform(xTest)
    return xTrain_scaled, xTest_scaled    
