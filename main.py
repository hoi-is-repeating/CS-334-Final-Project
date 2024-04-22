import featureSelection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score



def rndmForest(xTrain, yTrain, xTest, yTest):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rF = RandomForestClassifier(random_state=42)
    rF.fit(xTrain, yTrain)
    grid_search = GridSearchCV(
        estimator= rF, param_grid=param_grid, cv=5, scoring='accuracy')
    #grid search
    grid_search.fit(xTrain, yTrain)
    #best hyperparameters 
    best_params = grid_search.best_params_

    # train weith best hyperparams
    best_rf_model = RandomForestClassifier(random_state=42, **best_params)
    best_rf_model.fit(xTrain, yTrain)
    #predict training and testing data
    yTrain_pred = best_rf_model.predict(xTrain)
    yTest_pred = best_rf_model.predict(xTest)


    #calculate accuracy 
    train_accuracy = accuracy_score(yTrain, yTrain_pred)
    test_accuracy = accuracy_score(yTest, yTest_pred)
    # print("Best Hyperparameters:", best_params)
    # print("Training Accuracy:", train_accuracy)
    # print("Testing Accuracy:", test_accuracy)
    return rF



def dt(train_X, train_y, min_samples_leaf, max_depth):
    tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth)
    tree.fit(train_X, train_y)
    return tree

def main():
    xTrain,xTest,yTrain,yTest = featureSelection.selection()
    
if __name__ == "__main__":
    main()