from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve, auc, roc_curve


def knn(train_x, train_y, test_x, test_y, neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(train_x, train_y)
    y = knn.predict(test_x)

    accuracy, f1, auc, auprc = metrics(y, test_y)

    return accuracy, f1, auc, auprc
    

def nn(train_x, train_y, test_x, test_y, neighbors=5):
    nn = MLPClassifier()
    nn.fit(train_x, train_y)
    y = nn.predict(test_x)

    accuracy, f1, auc, auprc = metrics(y, test_y)

    return accuracy, f1, auc, auprc


def metrics(y, test_y):
    accuracy = accuracy_score(y, test_y)
    f1 = f1_score(y, test_y, average='weighted')
    auc = roc_auc_score(y, test_y)
    fpr, tpr, _ = roc_curve(test_y, y)
    auprc = auc(fpr, tpr)

    return accuracy, f1, auc, auprc
    

def main(): 
    print("ello")

if __name__ == "__main__":
    main()
