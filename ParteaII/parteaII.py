import numpy as np
import pandas as pd
import sklearn.preprocessing as pk
import sklearn.tree as ptree
import sklearn.metrics as pm

""" Task 1 """

def removeOutliers(dataFrame, col = "Age"):
    [Q1, Q3] = dataFrame[col].quantile(q=[0.25, 0.75])
    IQR = Q3 - Q1
    return dataFrame.query('(@Q1 - 1.5 * @IQR) <= '+ col +' <= (@Q3 + 1.5 * @IQR)')

""" Task 2 """

def zScore(dataFrame, col = "Age", zFactor = 3):
    stdDev = dataFrame[col].std()
    mean = dataFrame[col].mean()
    maxZ = zFactor * stdDev + mean
    minZ = -zFactor * stdDev + mean
    return dataFrame.query('@minZ <= '+ col +' <= @maxZ')

""" Task 3 """

""" Task 4 """

""" Subtask 1 - split dataset """

colsTrain = ['Age', 'Sex', 'Pclass', 'Fare']
colsUsed = colsTrain + ['Survived']
unsplitDf = pd.read_csv('../DataForAll/train.csv')[colsUsed]
unsplitDf.replace({'male': 1, 'female' : 0}, inplace=True)
trainSet = unsplitDf.iloc[:int(0.8 * len(unsplitDf)),:]
testSet = unsplitDf.iloc[int(0.8 * len(unsplitDf)):,:]

""" Subtask 2- clean up the data: I want it cleaned up before normalization"""
trainSet = removeOutliers(trainSet)
trainSet = removeOutliers(trainSet, "Fare")
trainSet["Age"].fillna(trainSet.Age.mean(), inplace=True)

""" Normalize data """
normalizedDf = trainSet
normalizedDf['Age'] = pk.scale(trainSet['Age'])
normalizedDf['Fare'] = pk.scale(trainSet['Fare'])
testSet['Age'] = pk.scale(testSet['Age'])
testSet['Fare'] = pk.scale(testSet['Fare'])
print(normalizedDf)

""" Subtask 3 - """

X = normalizedDf[colsTrain]
y = normalizedDf['Survived']
y_truth = testSet['Survived']
X_truth = testSet[colsTrain]
clf = ptree.DecisionTreeClassifier()
clf.fit(X, y)
y_pred = clf.predict(X_truth)
accuracy = pm.accuracy_score(y_truth, y_pred)

print(accuracy)