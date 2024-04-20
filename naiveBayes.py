
from sklearn.naive_bayes import GaussianNB

#olivia kim

def nb(xTrain, yTrain):
    gnb = GaussianNB()
    gnb.fit(xTrain,yTrain)
    
    return gnb