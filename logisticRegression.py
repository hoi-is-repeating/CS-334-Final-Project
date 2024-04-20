import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def findParams(clf, pgrid, xTrain, yTrain):
    grid = GridSearchCV(clf,pgrid,cv=10)
    grid.fit(xTrain,yTrain)
    classifier = grid.best_estimator_
    classifier.fit(xTrain,yTrain)
    bestParams = grid.best_params_
    return bestParams
#olivia kim
def lasso(xTrain, yTrain):
    lassoLrGrid= {
            'C': [0.01, 0.1, 1, 10],
            'tol':[1e-4,1e-5,1e-6],
            'fit_intercept':[True,False]
        }
    lassoClf = LogisticRegression(penalty='l1',solver='saga',max_iter=1000)
    bestParams = findParams(lassoClf, lassoLrGrid,xTrain, yTrain)
    model = GridSearchCV(lassoClf,bestParams,1000,cv=10)
    return model
def ridge(xTrain, yTrain):
    ridgeLrGrid= {
            'C': [0.01, 0.1, 1, 10],
            'tol':[1e-4,1e-5,1e-6],
            'fit_intercept':[True,False]
        }
    ridgeClf = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=1000)
    bestParams = findParams(ridgeClf, ridgeLrGrid,xTrain, yTrain)
    model = GridSearchCV(ridgeClf,bestParams,1000,cv=10)
    return model
