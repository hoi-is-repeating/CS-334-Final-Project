import featureSelection
from pro import knn, nn
from naiveBayes import nb
from logisticRegression import lasso, ridge
from dt import dt
from rf import randf
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, auc


def get_parameter_grid(mName):
    pGrid = {}
    if mName=="DT":
        pGrid = {
        'max_depth': [1, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'criterion': ['gini', 'entropy']
        }

    elif mName == "LR (L1)":
        pGrid = {
            'C': [0.01, 0.1, 1],
            'tol':[1e-4,1e-5],
            'fit_intercept':[True,False]
        }
    elif mName == "LR (L2)":
        pGrid = {
            'C': [0.01, 0.1, 1],
            'tol':[1e-4,1e-5],
            'fit_intercept':[True,False]
        }
    elif mName == "KNN":
        pGrid = {
            'n_neighbors': [3, 5, 10],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan','minkowski']
        }
    elif mName == "NN":
        pGrid = {
            'hidden_layer_sizes': [(30,),(50,)],
            'activation': ['relu', 'tanh'],
            'alpha':[0.01, 0.1],
        }
    elif mName == "NB":
        pGrid = {
            'var_smoothing': [1e-9]
        }
    else:
        raise ValueError("Try models between DT, KNN, LR (None)...etc")
    return pGrid

def findParams(clf, pGrid, xTrain, yTrain):
    grid = GridSearchCV(clf,pGrid,cv=10)
    grid.fit(xTrain,yTrain)
    classifier = grid.best_estimator_
    classifier.fit(xTrain,yTrain)
    bestParams = grid.best_params_
    return classifier,bestParams

def predict(clfName,clf,pGrid,metricsMap,roc,bestParamsMap, xTrain,yTrain, xTest, yTest):
    metrics = {}
    rocDF = {} 
    bestClf, bestParams = findParams(clf,pGrid,xTrain,yTrain)
    
    yHat_prob = bestClf.predict_proba(xTest)[:,1]
    metrics["AUC"] = roc_auc_score(yTest,yHat_prob)
    rocDF["fpr"], rocDF["tpr"], _ = roc_curve(yTest,yHat_prob)
    precision, recall, _ = precision_recall_curve(yTest, yHat_prob)
    metrics["AUPRC"] = auc(recall, precision)
    metrics["F1"] = f1_score(yTest,bestClf.predict(xTest))
    
    metricsMap[clfName] = metrics
    
    rocRes = pd.DataFrame(rocDF)
    roc = pd.DataFrame(roc)
    rocRes["model"] = clfName
    roc = pd.concat([roc, rocRes], ignore_index=True)
    
    bestParamsMap[clfName] = bestParams
    
    
    return metricsMap,roc,bestParamsMap
    
    
def main():
    xTrain, xTest, yTrain, yTest = featureSelection.selection()
    yTrain = yTrain.values.ravel()
    yTest = yTest.values.ravel()
    
    metricsMap = {}
    roc = {}
    bestParamsMap = {}

    # k-nearest neighbors
    print("Tuning KNN --------")
    knnName = "KNN"
    knnClf = knn(xTrain,yTrain)
    knnGrid = get_parameter_grid(knnName)
    metricsMap,roc,bestParamsMap=predict(knnName,knnClf,knnGrid,metricsMap,roc,bestParamsMap, xTrain,yTrain, xTest, yTest)
 

    # neural networks
    print("Tuning NN --------")
    nnName = "NN"
    nnClf = nn(xTrain,yTrain)
    nnGrid = get_parameter_grid(nnName)
    metricsMap,roc,bestParamsMap=predict(nnName,nnClf,nnGrid,metricsMap,roc,bestParamsMap, xTrain,yTrain, xTest, yTest)
    
    # naive bayes
    print("Tuning NB --------")
    nbName = "NB"
    nbClf = nb(xTrain,yTrain)
    nbGrid = get_parameter_grid(nbName)
    metricsMap,roc,bestParamsMap=predict(nbName,nbClf,nbGrid,metricsMap,roc,bestParamsMap, xTrain,yTrain, xTest, yTest)
    # decision tree
    print("Tuning DT --------")
    dtName = "DT"
    dtClf = dt(xTrain,yTrain)
    dtGrid = get_parameter_grid(dtName)
    metricsMap,roc,bestParamsMap=predict(dtName,dtClf,dtGrid,metricsMap,roc,bestParamsMap, xTrain,yTrain, xTest, yTest)

    # logistic regression (L1)
    print("Tuning Logistic Regression (Lasso) --------")
    lassoLrName = "LR (L1)"
    lassoClf = lasso()
    lassoLrGrid = get_parameter_grid(lassoLrName)
    metricsMap,roc,bestParamsMap=predict(lassoLrName,lassoClf,lassoLrGrid,metricsMap,roc,bestParamsMap, xTrain,yTrain, xTest, yTest)


    # Logistic regression (L2)
    print("Tuning Logistic Regression (Ridge) --------")
    ridgeLrName = "LR (L2)"
    ridgeClf = ridge()
    ridgeLrGrid = get_parameter_grid(ridgeLrName)
    
    metricsMap,roc,bestParamsMap=predict(ridgeLrName,ridgeClf,ridgeLrGrid,metricsMap,roc,bestParamsMap, xTrain,yTrain, xTest, yTest)

    
    metricsMap = pd.DataFrame(metricsMap)
    roc = pd.DataFrame(roc)
    bestParamsMap = pd.DataFrame(bestParamsMap)
    print(metricsMap)
    print(bestParamsMap)
    # save roc curves to data
    metricsMap.to_csv("metrics.csv", index=False)
    roc.to_csv("rocOutput.csv", index=False)
    bestParamsMap.to_csv("bestParams.csv", index=False)


if __name__ == "__main__":
    main()
