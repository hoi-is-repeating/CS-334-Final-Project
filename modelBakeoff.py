import argparse
import json
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def preprocess_data(xTrain, xTest):
    """
    Preprocess the features
    
    Parameters
    ----------
    xTrain : nd-array with shape (n, d)
        Training data
    xTest : nd-array with shape (m, d)
        Test data

    Returns
    -------
    xTrain : nd-array with shape (n, d)
        return the transformed training matrix
    xTest : nd-array with shape (m, d)
        return the transformed test matrix
    """
    xTrain = pd.DataFrame(xTrain)
    xTest = pd.DataFrame(xTest)
    #1. One-hot encoding - unnecessary
    #2. dropna using pandas
    xTrain.dropna(how = 'any',inplace = True)
    #3. min-max scaling
    scaler = MinMaxScaler()
    xTrain_scaled = scaler.fit_transform(xTrain)
    xTest_scaled = scaler.transform(xTest)
    
    return xTrain_scaled, xTest_scaled


def eval_gridsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn classifier and a parameter grid to search,
    choose the optimal parameters from pgrid using Grid Search CV
    and train the model using the training dataset and evaluate the
    performance on the test dataset.

    Parameters
    ----------
    clf : sklearn.ClassifierMixin
        The sklearn classifier model 
    pgrid : dict
        The dictionary of parameters to tune for in the model
    xTrain : nd-array with shape (n, d)
        Training data
    yTrain : 1d array with shape (n, )
        Array of labels associated with training data
    xTest : nd-array with shape (m, d)
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    resultDict: dict
        A Python dictionary with the following 4 keys,
        "AUC", "AUPRC", "F1", "Time" and the values are the floats
        associated with them for the test set.
    roc : dict
        A Python dictionary with 2 keys, fpr, and tpr, where
        each of the values are 1-d numpy arrays of the fpr and tpr
        associated with different thresholds. You should be able to use 
        this to plot the ROC for the model performance on the test curve.
    bestParams: dict
        A Python dictionary with the best parameters chosen by your
        GridSearch. The values in the parameters should be something
        that was in the original pgrid.
    """
    resultDict = {}
    roc = {}
    bestParams = {}
    start = time.time()
    #step 1. use grid search cross validation
    grid = GridSearchCV(clf,pgrid,cv=10)
    grid.fit(xTrain,yTrain)
    
    #step 2. train the classifiers with best parameters using all training data
    classifier = grid.best_estimator_
    classifier.fit(xTrain,yTrain)
    bestParams = grid.best_params_
    
    #step 3. evaluate the trained classifier using metrics
    #a. auc
    yHat_prob = classifier.predict_proba(xTest)[:,1]
    resultDict["AUC"] = roc_auc_score(yTest,yHat_prob)
    roc["fpr"], roc["tpr"], _ = roc_curve(yTest,yHat_prob)
    
    #b. auprc
    precision, recall, _ = precision_recall_curve(yTest, yHat_prob)
    resultDict["AUPRC"] = auc(recall, precision)
    
    #c. f1
    resultDict["F1"] = f1_score(yTest,classifier.predict(xTest))
    
    #d. time
    resultDict["Time"] = time.time() - start
    return resultDict, roc, bestParams


def eval_randomsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn classifier and a parameter grid to search,
    choose the optimal parameters from pgrid using Random Search CV
    and train the model using the training dataset and evaluate the
    performance on the test dataset. The random search cv should try
    at most 33% of the possible combinations.

    Parameters
    ----------
    clf : sklearn.ClassifierMixin
        The sklearn classifier model 
    pgrid : dict
        The dictionary of parameters to tune for in the model
    xTrain : nd-array with shape (n, d)
        Training data
    yTrain : 1d array with shape (n, )
        Array of labels associated with training data
    xTest : nd-array with shape (m, d)
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    resultDict: dict
        A Python dictionary with the following 4 keys,
        "AUC", "AUPRC", "F1", "Time" and the values are the floats
        associated with them for the test set.
    roc : dict
        A Python dictionary with 2 keys, fpr, and tpr, where
        each of the values are lists of the fpr and tpr associated
        with different thresholds. You should be able to use this
        to plot the ROC for the model performance on the test curve.
    bestParams: dict
        A Python dictionary with the best parameters chosen by your
        GridSearch. The values in the parameters should be something
        that was in the original pgrid.
    """
    resultDict = {}
    roc = {}
    bestParams = {}
    
    start = time.time()
    permutations = np.prod([len(v) for v in pgrid.values()])
    iter = int(0.33 * permutations)
    
    #1. find optimal param
    random = RandomizedSearchCV(clf,pgrid,n_iter=iter,cv = 10)
    random.fit(xTrain,yTrain)
    #2. train classifier
    classifier = random.best_estimator_
    classifier.fit(xTrain,yTrain)
    bestParams = random.best_params_
    #3. eval scores
    #a. auc
    yHat_prob = classifier.predict_proba(xTest)[:,1]
    resultDict["AUC"] = roc_auc_score(yTest,yHat_prob)
    roc["fpr"], roc["tpr"], _ = roc_curve(yTest,yHat_prob)
    
    #b. auprc
    precision, recall, _ = precision_recall_curve(yTest, yHat_prob)
    resultDict["AUPRC"] = auc(recall, precision)
    
    #c. f1
    resultDict["F1"] = f1_score(yTest,classifier.predict(xTest))
    
    #d. time
    resultDict["Time"] = time.time() - start
    return resultDict, roc, bestParams


def eval_searchcv(clfName, clf, clfGrid,
                  xTrain, yTrain, xTest, yTest,
                  perfDict, rocDF, bestParamDict):
    # evaluate grid search and add to perfDict
    cls_perf, cls_roc, gs_p  = eval_gridsearch(clf, clfGrid, xTrain,
                                               yTrain, xTest, yTest)
    perfDict[clfName + " (Grid)"] = cls_perf
    # add to ROC DF
    rocRes = pd.DataFrame(cls_roc)
    rocRes["model"] = clfName
    rocDF = pd.concat([rocDF, rocRes], ignore_index=True)
    # evaluate random search and add to perfDict
    clfr_perf, _, rs_p  = eval_randomsearch(clf, clfGrid, xTrain,
                                            yTrain, xTest, yTest)
    perfDict[clfName + " (Random)"] = clfr_perf
    bestParamDict[clfName] = {"Grid": gs_p, "Random": rs_p}
    return perfDict, rocDF, bestParamDict


def your_model():
    """
    Return an instance of the optimal model based on your
    model selection and assessment strategy. This can be any
    of the 6 models tested in the homework and should be a
    parameter combination you tested.
    """
    pGrid = {
            'penalty':['l1'],
            'solver':['saga'],
            'max_iter':[1000],
            'C': [0.01, 0.1, 1, 10],
            'tol':[1e-4,1e-5,1e-6],
            'fit_intercept':[True,False]
        }
    clf = LogisticRegression()
    permutations = np.prod([len(v) for v in pGrid.values()])
    iter = int(0.33 * permutations)
    model = RandomizedSearchCV(clf,pGrid,n_iter=iter,cv = 10)

    return model
    



def get_parameter_grid(mName):
    """
    Given a model name, return the parameter grid associated with it

    Parameters
    ----------
    mName : string
        name of the model (e.g., DT, KNN, LR (None))

    Returns
    -------
    pGrid: dict
        A Python dictionary with the appropriate parameters for the model.
        The dictionary should have at least 2 keys and each key should have
        at least 2 values to try.
    """
    pGrid = {}
    if mName=="DT":
        pGrid = {
        'max_depth': [1, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'criterion': ['gini', 'entropy']
        }
    elif mName == "LR (None)":
        pGrid = {
            'tol':[1e-4,1e-5,1e-6],
            'fit_intercept':[True,False]
        }
    elif mName == "LR (L1)":
        pGrid = {
            'C': [0.01, 0.1, 1, 10],
            'tol':[1e-4,1e-5,1e-6],
            'fit_intercept':[True,False]
        }
    elif mName == "LR (L2)":
        pGrid = {
            'C': [0.01, 0.1, 1, 10],
            'tol':[1e-4,1e-5,1e-6],
            'fit_intercept':[True,False]
        }
    elif mName == "KNN":
        pGrid = {
            'n_neighbors': [3, 5, 10],
            'weights': ['uniform', 'distance'],
            'p':[1,2],
            'metric': ['euclidean', 'manhattan','minkowski']
        }
    elif mName == "NN":
        pGrid = {
            'hidden_layer_sizes': [(30,),(50,)],
            'batch_size':[128,256],
            'activation': ['relu', 'tanh'],
            'alpha':[0.001, 0.01, 0.1],
        }
    else:
        raise ValueError("Try models between DT, KNN, LR (None)...etc")
    return pGrid


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="space_trainx.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="space_trainy.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="space_testx.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="space_testy.csv",
                        help="filename for labels associated with the test data")
    parser.add_argument("rocOutput",
                         help="csv filename for ROC curves")
    parser.add_argument("bestParamOutput",
                         help="json filename for best parameter")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()

    # preprocess the data
    xTrain, xTest = preprocess_data(xTrain, xTest)

    perfDict = {}
    rocDF = pd.DataFrame()
    bestParamDict = {}

    print("Tuning Decision Tree --------")
    # Compare Decision Tree
    dtName = "DT"
    dtGrid = get_parameter_grid(dtName)
    # fill in
    dtClf = DecisionTreeClassifier()
    perfDict, rocDF, bestParamDict = eval_searchcv(dtName, dtClf, dtGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, rocDF, bestParamDict)
    print("Tuning Unregularized Logistic Regression --------")
    # logistic regression (unregularized)
    unregLrName = "LR (None)"
    unregLrGrid = get_parameter_grid(unregLrName)
    # fill in
    lrClf = LogisticRegression(solver='lbfgs',max_iter = 1000)
    perfDict, rocDF, bestParamDict = eval_searchcv(unregLrName, lrClf, unregLrGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, rocDF, bestParamDict)
    # logistic regression (L1)
    print("Tuning Logistic Regression (Lasso) --------")
    lassoLrName = "LR (L1)"
    lassoLrGrid = get_parameter_grid(lassoLrName)
    # fill in
    lassoClf = LogisticRegression(penalty='l1',solver='saga',max_iter=1000)
    perfDict, rocDF, bestParamDict = eval_searchcv(lassoLrName, lassoClf, lassoLrGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, rocDF, bestParamDict)
    # Logistic regression (L2)
    print("Tuning Logistic Regression (Ridge) --------")
    ridgeLrName = "LR (L2)"
    ridgeLrGrid = get_parameter_grid(ridgeLrName)
    # fill in
    ridgeClf = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=1000)
    perfDict, rocDF, bestParamDict = eval_searchcv(ridgeLrName, ridgeClf, ridgeLrGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, rocDF, bestParamDict)
    # k-nearest neighbors
    print("Tuning K-nearest neighbors --------")
    knnName = "KNN"
    knnGrid = get_parameter_grid(knnName)
    # fill in
    knnClf = KNeighborsClassifier()
    perfDict, rocDF, bestParamDict = eval_searchcv(knnName, knnClf, knnGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, rocDF, bestParamDict)
    # neural networks
    print("Tuning neural networks --------")
    nnName = "NN"
    nnGrid = get_parameter_grid(nnName)
    # fill in
    nnClf = MLPClassifier(max_iter=500,early_stopping=True,solver='sgd',n_iter_no_change=10)
    perfDict, rocDF, bestParamDict = eval_searchcv(nnName, nnClf, nnGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, rocDF, bestParamDict)
    perfDF = pd.DataFrame.from_dict(perfDict, orient='index')
    print(perfDF)
    # save roc curves to data
    rocDF.to_csv(args.rocOutput, index=False)
    # store the best parameters
    with open(args.bestParamOutput, 'w') as f:
        json.dump(bestParamDict, f)


if __name__ == "__main__":
    main()
