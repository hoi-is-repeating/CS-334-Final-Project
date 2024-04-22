import featureSelection
from pro import knn, nn, nb, metrics
from dt import dt
from rf import randf


def main():
    xTrain, xTest, yTrain, yTest = featureSelection.selection()
    knn_model = knn(xTrain, yTrain.squeeze())
    nn_model = nn(xTrain, yTrain.squeeze())
    nb_model = nb(xTrain, yTrain.squeeze())
    dt_model = dt(xTrain,yTrain,3,3)
    rf_model = randf(xTrain,yTrain,xTest,yTest)
    

    knn_y = knn_model.predict(xTest)
    nn_y = nn_model.predict(xTest)
    nb_y = nb_model.predict(xTest)
    dt_y = dt_model.predict(xTest)
    rf_y = rf_model.predict(xTest)

    knn_metrics = metrics(knn_y, yTest)
    nn_metrics = metrics(nn_y, yTest)
    nb_metrics = metrics(nb_y, yTest)
    dt_metrics = metrics(dt_y, yTest)
    rf_metrics = metrics(rf_y, yTest)

    print("KNN: ", knn_metrics)
    print("NN: ", nn_metrics)
    print("NB: ", nb_metrics)
    print("DT : ", dt_metrics)
    print("RF : ", rf_metrics)


if __name__ == "__main__":
    main()
