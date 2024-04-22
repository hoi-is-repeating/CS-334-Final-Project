import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score


# df = pd.read_csv("data.csv")
# #Sanity cehck 
# # print(df.shape)
# # print(df.head)
# # print(df.tail)
# # print(df.info)
# # print(df.describe)
# # df.isnull().sum() 


# print(df['label'].value_counts)
# X = df.iloc[:,:-1].values
# y = df.iloc[:,-1].values
# # print(X.shape)
# # print(y.shape)

# xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=1)

# clf = RandomForestClassifier(n_estimators=args.n_estimators,
#                                            criterion=args.criterion,
#                                            max_depth=args.md,
#                                            min_samples_leaf=args.mls,
#                                            random_state=42)
# clf.fit(xTrain, yTrain)
# #clf.feature_importances_
# ypred = clf.predict(xTest)
# print(ypred)
# confusion_matrix(yTest, ypred)
# accuracy_score(yTest, ypred)
# cross_val_score(clf, xTrain, yTrain, cv=10)
# print(classification_report(ypred, yTest))



#----------------------------
# df = pd.read_csv("data.csv")
# X = df.iloc[:,:-1].values
# y = df.iloc[:,-1].values
# # X = df.iloc[:,1:2].values  #features
# # y = df.iloc[:,2].values  # Target variable

# label_encoder = LabelEncoder()
# x_categorical = df.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
# x_numerical = df.select_dtypes(exclude=['object']).values
# x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values
 
# # Fitting Random Forest Regression to the dataset
# regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
 
# # Fit the regressor with x and y data
# regressor.fit(x, y)

# oob_score = regressor.oob_score_
# print(f'Out-of-Bag Score: {oob_score}')
 
# # Making predictions on the same data or new data
# predictions = regressor.predict(x)
 
# # Evaluating the model
# mse = mean_squared_error(y, predictions)
# print(f'Mean Squared Error: {mse}')
# r2 = r2_score(y, predictions)
# print(f'R-squared: {r2}')
# X_grid = np.arange(min(X),max(X),0.01)
# X_grid = X_grid.reshape(len(X_grid),1) 


def preprocess_data(data_file):
    # Load the dataset
    data = pd.read_csv(data_file)

    # Separate features and labels
    x = data.drop(columns=['label'])  # Features
    y = data['label']                 # Labels


    # Handle categorical features
    categorical_cols = x.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        # Use Label Encoding for simplicity (replace with One-Hot Encoding if needed)
        label_encoder = LabelEncoder()
        x[categorical_cols] = x[categorical_cols].apply(
            label_encoder.fit_transform)
    return x, y


def main(args):
    # Preprocess the data
    x, y = preprocess_data(args.data)

    # Split the dataset into training and testing sets
    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=0.3, random_state=42)

    """
     # Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=args.n_estimators,
                                           criterion=args.criterion,
                                           max_depth=args.md,
                                           min_samples_leaf=args.mls,
                                           random_state=42)
    """

    # Train the Random Forest model
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    #Initialize Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

    #Perform Grid Search
    grid_search.fit(xTrain, yTrain)

    #Get the best hyperparameters
    best_params = grid_search.best_params_

    #Train the model with the best hyperparameters
    best_rf_model = RandomForestClassifier(random_state=42, **best_params)
    best_rf_model.fit(xTrain, yTrain)

    #Predictions on training and testing data
    yTrain_pred = best_rf_model.predict(xTrain)
    yTest_pred = best_rf_model.predict(xTest)

    #Calculate accuracy
    train_accuracy = accuracy_score(yTrain, yTrain_pred)
    test_accuracy = accuracy_score(yTest, yTest_pred)

    print("Best Hyperparameters:", best_params)
    print("Training Accuracy:", train_accuracy)
    print("Testing Accuracy:", test_accuracy)


if __name__ == "__main__":
    # Set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.csv",
                        help="filename for the dataset")
    # parser.add_argument("n_estimators", type=int,
    #                     help="number of decision trees in the forest")
    # parser.add_argument("md", type=int,
    #                     help="maximum depth")
    # parser.add_argument("mls", type=int,
    #                     help="minimum leaf samples")
    # parser.add_argument("--criterion", default="gini", choices=["gini", "entropy"],
    #                     help="splitting criterion")
    args = parser.parse_args()

    main(args)

    # # Set up the program to take in arguments from the command line
    # parser = argparse.ArgumentParser()
    # parser.add_argument("n_estimators", type=int,
    #                     help="number of decision trees in the forest")
    # parser.add_argument("md", type=int,
    #                     help="maximum depth")
    # parser.add_argument("mls", type=int,
    #                     help="minimum leaf samples")
    # parser.add_argument("--data", default="data.csv",
    #                     help="filename for the dataset")
    # parser.add_argument("--criterion", default="gini", choices=["gini", "entropy"],
    #                     help="splitting criterion")
    # args = parser.parse_args()

    # main(args)
    


 



# import argparse
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sklearn
# import warnings
    
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler

# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import mean_squared_error, r2_score

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import KNNImputer
# from sklearn.tree import plot_tree
# warnings.filterwarnings('ignore')




# """
# X = df.iloc[:,1:2].values  #features
# y = df.iloc[:,2].values  # Target variable

# label_encoder = LabelEncoder()
# x_categorical = df.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
# x_numerical = df.select_dtypes(exclude=['object']).values
# x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values
 
# # Fitting Random Forest Regression to the dataset
# regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
 
# # Fit the regressor with x and y data
# regressor.fit(x, y)

# oob_score = regressor.oob_score_
# print(f'Out-of-Bag Score: {oob_score}')
 
# # Making predictions on the same data or new data
# predictions = regressor.predict(x)
 
# # Evaluating the model
# mse = mean_squared_error(y, predictions)
# print(f'Mean Squared Error: {mse}')
 
# r2 = r2_score(y, predictions)
# print(f'R-squared: {r2}')

# X_grid = np.arange(min(X),max(X),0.01)
# X_grid = X_grid.reshape(len(X_grid),1) 



# plt.scatter(X,y, color='blue') #plotting real points
# plt.plot(X_grid, regressor.predict(X_grid),color='green') #plotting for predict points
# plt.title("Random Forest Regression Results")
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()
# tree_to_plot = regressor.estimators_[0]
# # Plot the decision tree
# plt.figure(figsize=(20, 10))
# plot_tree(tree_to_plot, feature_names=df.columns.tolist(), filled=True, rounded=True, fontsize=10)
# plt.title("Decision Tree from Random Forest")
# plt.show()
# """



# def preprocess_data(data_file):
#     # Load the dataset
#     data = pd.read_csv(data_file)

#     # Separate features and labels
#     x = data.drop(columns=['label'])  # Features
#     y = data['label']                 # Labels

#     """
#     # Handle categorical features
#     categorical_cols = x.select_dtypes(include=['object']).columns
#     if not categorical_cols.empty:
#         # Use Label Encoding for simplicity (replace with One-Hot Encoding if needed)
#         label_encoder = LabelEncoder()
#         x[categorical_cols] = x[categorical_cols].apply(
#             label_encoder.fit_transform)
#     """

#     return x, y
    
# def main(args):
#     # Preprocess the data
#     x, y = preprocess_data(args.data)

#     # Split the dataset into training and testing sets
#     xTrain, xTest, yTrain, yTest = train_test_split(
#         x, y, test_size=0.3, random_state=42)

#     """
#      # Initialize the Random Forest classifier
#     rf_classifier = RandomForestClassifier(n_estimators=args.n_estimators,
#                                            criterion=args.criterion,
#                                            max_depth=args.md,
#                                            min_samples_leaf=args.mls,
#                                            random_state=42)
#     """
#     # Train the Random Forest model
#     param_grid = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [None, 10, 20],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4]
#     }
#     #Initialize Random Forest classifier
#     rf_classifier = RandomForestClassifier(random_state=42)
#     grid_search = GridSearchCV(
#         estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
#     #Perform Grid Search
#     grid_search.fit(xTrain, yTrain)

#     #Get the best hyperparameters
#     best_params = grid_search.best_params_

#     #Train the model with the best hyperparameters
#     best_rf_model = RandomForestClassifier(random_state=42, **best_params)
#     best_rf_model.fit(xTrain, yTrain)

#     #Predictions on training and testing data
#     yTrain_pred = best_rf_model.predict(xTrain)
#     yTest_pred = best_rf_model.predict(xTest)

#     #Calculate accuracy
#     train_accuracy = accuracy_score(yTrain, yTrain_pred)
#     test_accuracy = accuracy_score(yTest, yTest_pred)

#     print("Best Hyperparameters:", best_params)
#     print("Training Accuracy:", train_accuracy)
#     print("Testing Accuracy:", test_accuracy)

#     """
#     #rf_classifier.fit(xTrain, yTrain)

#     # Predictions on training and testing data
#     yTrain_pred = rf_classifier.predict(xTrain)
#     yTest_pred = rf_classifier.predict(xTest)

#     # Calculate accuracy
#     train_accuracy = accuracy_score(yTrain, yTrain_pred)
#     test_accuracy = accuracy_score(yTest, yTest_pred)

#     print("Training Accuracy:", train_accuracy)
#     print("Testing Accuracy:", test_accuracy)
#     """


# if __name__ == "__main__":
#     # Set up the program to take in arguments from the command line
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data", default="data.csv",
#                         help="filename for the dataset")
#     args = parser.parse_args()

#     main(args)
#     # Set up the program to take in arguments from the command line
#     parser = argparse.ArgumentParser()
#     parser.add_argument("n_estimators", type=int,
#                         help="number of decision trees in the forest")
#     parser.add_argument("md", type=int,
#                         help="maximum depth")
#     parser.add_argument("mls", type=int,
#                         help="minimum leaf samples")
#     parser.add_argument("--data", default="data.csv",
#                         help="filename for the dataset")
#     parser.add_argument("--criterion", default="gini", choices=["gini", "entropy"],
#                         help="splitting criterion")
#     args = parser.parse_args()

#     main(args)
    
