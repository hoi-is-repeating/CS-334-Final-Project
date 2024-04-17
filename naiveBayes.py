from preprocessing import standard_scale, minmax_range, add_irr_feature
from sklearn.naive_bayes import GaussianNB

#olivia kim

def nb(xTrain, yTrain, xTest, yTest):
    gnb = GaussianNB()
    gnb.fit(xTrain,yTrain)
    
    return gnb