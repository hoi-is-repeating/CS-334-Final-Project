import argparse
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt bruh this always making an error in gradescope annoys me


def calculate_score(y, criterion):
    """
    Given a numpy array of labels associated with a node, y, 
    calculate the score based on the crieterion specified.

    Parameters
    ----------
    y : numpy.1d array with shape (n, )
        Array of labels associated with a node
    criterion : String
        The function to measure the quality of a split.
        Supported criteria are "gini" for the Gini impurity
        and "entropy" for the information gain.
    Returns
    -------
    score : float
        The gini or entropy associated with a node
    """
    #find number of occurrences
    vals, counts = np.unique(y, return_counts=True)
    num = len(y)
    score = 0.0
    if criterion=="gini":
        for label in counts:
            probability = (float(label)/num)
            score+=probability**2
        score = 1-score
            
    elif criterion=="entropy":
        for label in counts:
            probability = (float(label)/num)  
            score+=probability*np.log2(probability)
        score*=-1
    else:
        raise ValueError
            

    return score       



def find_best_splitval(xcol, y, criterion, minLeafSample):
    """
    Given a feature column (i.e., measurements for feature d),
    and the corresponding labels, calculate the best split
    value, v, such that the data will be split into two subgroups:
    xcol <= v and xcol > v. If there is a tie (i.e., multiple values
    that yield the same split score), you can return any of the
    possible values.

    Parameters
    ----------
    xcol : numpy.1d array with shape (n, )
    y : numpy.1d array with shape (n, )
        Array of labels associated with a node
    criterion : string
        The function to measure the quality of a split.
        Supported criteria are "gini" for the Gini impurity
        and "entropy" for the information gain.
    minLeafSample : int
            The min
    Returns
    -------
    v:  float / int (depending on the column)
        The best split value to use to split into 2 subgroups.
    score : float
        The gini or entropy associated with the best split
    """
    v, score = float('inf'), float('inf')
    indices = np.argsort(xcol)
    values = xcol[indices] #xcol sorted
    labels = y[indices] #y sorted
    for i in range(1,len(xcol)):
        if values[i]==values[i-1]:
            continue #skip repeated values
        split = (values[i]+values[i-1])/2.0
        y_left = labels[values<=split]
        y_right = labels[values>split]
        if len(y_left) < minLeafSample or len(y_right) < minLeafSample:
            continue
        newScore = (len(y_left)/len(y))*calculate_score(y_left,criterion)+(len(y_right)/len(y))*calculate_score(y_right,criterion)
        if newScore<score:
            score = newScore
            v = split
        
    return v, score


class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    
    X_train = np.array([])
    y_train = np.array([]) #apparently vectors are lowercase
    treenode = {}


    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    def decision_tree(self,xFeat,y,depth):
        # Check stopping criteria (e.g. maximum depth), if it is met, return majority class of y
        if depth == self.maxDepth or len(y) <= self.minLeafSample and len(y)>0:
            return np.argmax(y)
        # Find the split: enumerate all possible splits (for each feature and each split value), compute the score(entropy or gini) for each split, find the best split feature and split value
        v,score = 0, float('inf')
        v_idx = 0
        for feature in range(xFeat.shape[1]):
            values = xFeat[:, feature]
            newV, newScore = find_best_splitval(values, y, self.criterion, self.minLeafSample)
            if newScore < score:
                v,score = newV,newScore
                v_idx = feature
        # Partition data using the split feature and split value into two sets: xFeatL, xFeatR, yL, yR
        #print("splits at: "+str(v_idx)+ "with value "+ str(v)+" from "+str(parent))
        idx = np.where(xFeat[:, v_idx]<=v)[0]
        xFeatL, xFeatR = xFeat[idx], xFeat[~idx]
        yL, yR = y[idx], y[~idx]
        # Recursive call of decision_tree()
        return {"left": self.decision_tree (xFeatL, yL, depth+1),
                "right": self.decision_tree(xFeatR, yR, depth+1),
                "split value index":v_idx,"split value":v}
    def train(self, xFeat, y):
        """
        Train the decision tree model.

        Parameters
        ----------
        xFeat : numpy nd-array with shape (n, d)
            Training data 
        y : numpy 1d array with shape (n, )
            Array of labels associated with training data.

        Returns
        -------
        self : DecisionTree
            The decision tree model instance
        """
        self.X_train = xFeat
        self.y_train = y
        self.treenode = self.decision_tree(xFeat,y,0)
        
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
        yHat : numpy 1d array with shape (m, )
            Predicted class label per sample
        """
        yHat = []
        for feature in xFeat:
            yHat.append(self.predict_rec(feature,self.treenode))
        return np.array(yHat)
                
    def predict_rec(self,feature,node):
        if not isinstance(node, dict):
            return node  
        if feature[node["split value index"]] <= node["split value"]:
            return self.predict_rec(feature, node["left"])
        else:
            return self.predict_rec(feature, node["right"])

def _accuracy(yTrue, yHat):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yTrue : 1d-array with shape (n, )
        True labels associated with the n samples
    yHat : 1d-array with shape (n,)
        Predicted class label for n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    acc = np.sum(yHat == yTrue) / len(yTrue)
    return acc

def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : numpy.nd-array with shape n x d
        Training data 
    yTrain : numpy.1d array with shape n
        Array of labels associated with training data.
    xTest : numpy.nd-array with shape m x d
        Test data 
    yTest : numpy.1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain)
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = _accuracy(yTrain, yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = _accuracy(yTest, yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
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
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)
    
    """
    leaves = list(range(1,11))
    trainAccs = []
    testAccs = []
    for leaf in leaves:
        dt = DecisionTree('gini', 5, leaf)
        trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
        trainAccs.append(trainAcc)
        testAccs.append(testAcc)
        
    plt.figure(figsize=(10, 6))
    plt.plot(leaves, trainAccs, label="Train Accuracy")
    plt.plot(leaves, testAccs, label="Test Accuracy", linestyle='--')

    plt.title("Train and Test Accuracy vs Min Leaf Sample")
    plt.xlabel("Min Leaf Sample")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.xticks(leaves)
    plt.show()
    
    depths = list(range(1,11))
    trainAccs = []
    testAccs = []
    for depth in depths:
        dt = DecisionTree('gini', depth, 5)
        trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
        trainAccs.append(trainAcc)
        testAccs.append(testAcc)
        
    plt.figure(figsize=(10, 6))
    plt.plot(depths, trainAccs, label="Train Accuracy")
    plt.plot(depths, testAccs, label="Test Accuracy", linestyle='--')

    plt.title("Train and Test Accuracy vs Max Depth Value")
    plt.xlabel("Max Depth Value")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.xticks(depths)
    plt.show()"""
        
    
 


if __name__ == "__main__":
    main()
