import argparse
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt


class Knn(object):
    k = 0              # number of neighbors to use
    nFeatures = 0      # number of features seen in training
    nSamples = 0       # number of samples seen in training
    isFitted = False  # has train been called on a dataset?
    X_train = np.array([])
    y_train = np.array([]) #apparently vectors are lowercase


    def mode(self, arr):
        arr = np.array(arr)
        iwanttogohome, counts = np.unique(arr, return_counts=True)
        max_cnt = np.argmax(counts)
        mode = iwanttogohome[max_cnt]
        return mode
   
    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k 
        
    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : numpy nd-array with shape (n, d)
            Training data 
        y : numpy 1d array with shape (n, )
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        self.X_train = xFeat
        self.y_train = y
        for row in xFeat:
            self.nSamples += 1
            self.nFeatures += np.size(row)
        self.isFitted = True
        
        return self


    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : numpy nd-array with shape (m, d)
            The data to predict.  

        Returns
        -------
        yHat : numpy.1d array with shape (m, )
            Predicted class label per sample
        """
        neighbors = []
        for x in xFeat: #goes thru all rows of XFeat
            distances = np.sqrt(np.sum((x - self.X_train)**2, axis=1)) #compares length with all elements of the matrix, including itself!
            sort_y = [y for _, y in sorted(zip(distances,self.y_train))] #skip first value since its always 0 (comparison with itself)
            #sort y values according to distance length
            neighbors.append(sort_y[:self.k])
        
        return np.array(list(map(self.mode,neighbors)))


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape (n,)
        Predicted class label for n samples
    yTrue : 1d-array with shape (n, )
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    acc = (np.sum(1 for true, pred in zip(yTrue, yHat) if true == pred))/(yTrue.size)
    return acc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="simxTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="simyTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="simxTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="simyTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    # assume the data is all numerical and 
    # no additional pre-processing is necessary
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain)
    # predict the training datase
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain)
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest)
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)
    
    """
    test
    
    accuracies = []
    ks = range(1, (int)(np.sqrt(len(xTrain))))
    print(ks)
    for k in ks:
        knn = Knn(k)
        knn.train(xTrain, yTrain)
        yHatTrain = knn.predict(xTrain)
        trainAcc = accuracy(yHatTrain, yTrain)    
        accuracies.append(trainAcc)
    # Visualize accuracy vs. k
    fig, ax = plt.subplots()
    ax.plot(ks, accuracies)
    ax.set(xlabel="k",
       ylabel="Accuracy",
       title="Performance of knn")
    plt.show()
    """

if __name__ == "__main__":
    main()
