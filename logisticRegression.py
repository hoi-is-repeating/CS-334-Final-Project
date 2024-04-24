import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


#olivia kim
def lasso():
    lasso = LogisticRegression(penalty='l1',solver='saga',max_iter=1000)
    
    return lasso

def ridge():
    ridge = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=1000)
    
    return ridge
