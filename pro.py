from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve, auc, roc_curve
import pandas as pd
from naiveBayes import nb

def knn(train_x, train_y, neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(train_x, train_y)
    
    return knn
    

def nn(train_x, train_y):
    nn = MLPClassifier()
    nn.fit(train_x, train_y)
    return nn

def metrics(y, test_y):
    accuracy = accuracy_score(y, test_y)
    f1 = f1_score(y, test_y, average='weighted')
    roc_auc = roc_auc_score(y, test_y)
    fpr, tpr, _ = roc_curve(test_y, y)
    auprc = auc(fpr, tpr)

    return accuracy, f1, roc_auc, auprc
    

def main(): 
    xTrain = pd.read_csv("xTrain.csv")
    yTrain = pd.read_csv("yTrain.csv")
    xTest = pd.read_csv("xTest.csv")
    yTest = pd.read_csv("yTest.csv")

    knn_model = knn(xTrain, yTrain.squeeze())
    nn_model = nn(xTrain, yTrain.squeeze())
    nb_model = nb(xTrain, yTrain.squeeze())

    knn_y = knn_model.predict(xTest)
    nn_y = nn_model.predict(xTest)
    nb_y = nb_model.predict(xTest)

    knn_metrics = metrics(knn_y, yTest)
    nn_metrics = metrics(nn_y, yTest)
    nb_metrics = metrics(nb_y, yTest)

    print("KNN: ", knn_metrics)
    print("NN: ", nn_metrics)
    print("NB: ", nb_metrics)

if __name__ == "__main__":
    main()
