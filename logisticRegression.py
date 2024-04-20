import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def gridSearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
    grid = GridSearchCV(clf,pgrid,cv=10)
    grid.fit(xTrain,yTrain)
#olivia kim
def lasso():
    lassoLrName = "LR (L1)"
    lassoLrGrid = get_parameter_grid(lassoLrName)
    lassoClf = LogisticRegression(penalty='l1',solver='saga',max_iter=1000)
    perfDict, rocDF, bestParamDict = eval_searchcv(lassoLrName, lassoClf, lassoLrGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, rocDF, bestParamDict)
def ridge():
    print("Tuning Logistic Regression (Ridge) --------")
    ridgeLrName = "LR (L2)"
    ridgeLrGrid = get_parameter_grid(ridgeLrName)
    # fill in
    ridgeClf = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=1000)
    perfDict, rocDF, bestParamDict = eval_searchcv(ridgeLrName, ridgeClf, ridgeLrGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, rocDF, bestParamDict)