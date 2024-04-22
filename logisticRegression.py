import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


#olivia kim
def lasso(xTrain, yTrain,pGrid):
    lassoClf = LogisticRegression(penalty='l1',solver='saga',max_iter=3000)
    
    return lassoClf
def ridge(xTrain, yTrain, pGrid):

    ridgeClf = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=3000)
    
    return ridge
