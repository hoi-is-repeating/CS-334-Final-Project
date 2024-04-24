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

def predict(clf, xTest, yTest):
    metricsMap = {}
    roc = {}
    
    yHat_prob = clf.predict_proba(xTest)[:,1]
    metricsMap["AUC"] = roc_auc_score(yTest,yHat_prob)
    roc["fpr"], roc["tpr"], _ = roc_curve(yTest,yHat_prob)
    precision, recall, _ = precision_recall_curve(yTest, yHat_prob)
    metricsMap["AUPRC"] = auc(recall, precision)
    metricsMap["F1"] = f1_score(yTest,clf.predict(xTest))
    return metricsMap,roc
    
    
def main():
    xTrain, xTest, yTrain, yTest = featureSelection.selection()
    yTrain = yTrain.values.ravel()
    yTest = yTest.values.ravel()
    
    metricsMap = {}
    roc = {}
    bestParams = {}

    # k-nearest neighbors
    print("Tuning KNN --------")
    knnName = "KNN"
    knnGrid = get_parameter_grid(knnName)
    knnClf = knn(xTrain,yTrain)
    knnBestClf, knnBestParams = findParams(knnClf,knnGrid,xTrain,yTrain)
    metricsMap_knn,roc_knn=predict(knnBestClf,knnBestParams,xTrain,yTrain,xTest,yTest)
    
    metricsMap[knnName] = metricsMap_knn
    roc[knnName] = roc_knn
    bestParams[knnName] = knnBestParams

    # neural networks
    print("Tuning NN --------")
    nnName = "NN"
    nnGrid = get_parameter_grid(nnName)
    nnClf = nn(xTrain,yTrain)
    nnBestClf, nnBestParams = findParams(nnClf,nnGrid,xTrain,yTrain)
    metricsMap_nn,roc_nn=predict(nnBestClf,nnBestParams,xTrain,yTrain,xTest,yTest)
    
    metricsMap[nnName] = metricsMap_nn
    roc[nnName] = roc_nn
    bestParams[nnName] = nnBestParams
    
    # naive bayes
    print("Tuning NB --------")
    nbName = "NB"
    nbGrid = get_parameter_grid(nbName)
    nbClf = nb(xTrain,yTrain)
    nbBestClf, nbBestParams = findParams(nbClf,nbGrid,xTrain,yTrain)
    metricsMap_nb,roc_nb=predict(nbBestClf,nbBestParams,xTrain,yTrain,xTest,yTest)
    
    metricsMap[nbName] = metricsMap_nb
    roc[nbName] = roc_nb
    bestParams[nbName] = nbBestParams
    
    # decision tree
    print("Tuning DT --------")
    dtName = "DT"
    dtGrid = get_parameter_grid(dtName)
    dtClf = dt(xTrain,yTrain)
    dtBestClf, dtBestParams = findParams(dtClf,dtGrid,xTrain,yTrain)
    metricsMap_dt,roc_dt=predict(dtBestClf,dtBestParams,xTrain,yTrain,xTest,yTest)
    
    metricsMap[dtName] = metricsMap_dt
    roc[dtName] = roc_dt
    bestParams[dtName] = dtBestParams
    
    # logistic regression (L1)
    print("Tuning Logistic Regression (Lasso) --------")
    lassoLrName = "LR (L1)"
    lassoLrGrid = get_parameter_grid(lassoLrName)
    lassoClf = lasso()
    lassoBestClf, lassoBestParams = findParams(lassoClf,lassoLrGrid,xTrain,yTrain)
    metricsMap_lasso,roc_lasso=predict(lassoBestClf,lassoBestParams,xTrain,yTrain,xTest,yTest)

    metricsMap[lassoLrName] = metricsMap_lasso
    roc[lassoLrName] = roc_lasso
    bestParams[lassoLrName] = lassoBestParams

    # Logistic regression (L2)
    print("Tuning Logistic Regression (Ridge) --------")
    ridgeLrName = "LR (L2)"
    ridgeLrGrid = get_parameter_grid(ridgeLrName)
    ridgeClf = ridge()
    ridgeBestClf, ridgeBestParams = findParams(ridgeClf,ridgeLrGrid,xTrain,yTrain)
    metricsMap_ridge,roc_ridge=predict(ridgeBestClf,ridgeBestParams,xTrain,yTrain,xTest,yTest)
    
    metricsMap[ridgeLrName] = metricsMap_ridge
    roc[ridgeLrName] = roc_ridge
    bestParams[ridgeLrName] = ridgeBestParams

    

    roc_df = pd.DataFrame({
        'FPR': roc['fpr'],
        'TPR': roc['tpr']
    })
    
    metricsMap = pd.DataFrame.from_dict(metricsMap, orient='index')
    bestParams = pd.DataFrame.from_dict(bestParams, orient='index')
    print(metricsMap)
    print(bestParams)
    # save roc curves to data
    metricsMap.to_csv("metrics.csv", index=False)
    roc_df.to_csv("rocOutput.csv", index=False)
    bestParams.to_csv("bestParams.csv", index=False)


if __name__ == "__main__":
    main()
