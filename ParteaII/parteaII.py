import numpy as np
import pandas as pd
import sklearn.preprocessing as pk
import sklearn.tree as ptree
import sklearn.metrics as pm
import matplotlib.pyplot as plt
import sys
import sklearn.ensemble as ens
from collections import defaultdict
import seaborn as sn

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

def showHists(dataFrame):
    num_cols = dataFrame.select_dtypes(include=np.number)
    col = num_cols.columns
    j = 0
    for i in col:
        if i != 'PassengerId':
            j += 1
            plt.subplot(2, 3, j).set_title(i)
            plt.hist(num_cols[i])
    plt.show()

def showHeatMap(dataFrame):
    dataFrame.corr()
    corrMatrix = dataFrame.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()

""" Task 4 """

""" Subtask 1 - split dataset """

colsTrain = ['Age', 'Sex', 'Pclass', 'Fare', 'Parch', 'SibSp']
colsUsed = colsTrain + ['Survived']
unsplitDf = pd.read_csv('../DataForAll/train.csv')[colsUsed]
unsplitDf.replace({'male': 1, 'female' : 0}, inplace=True)
trainSet = unsplitDf.iloc[:int(0.8 * len(unsplitDf)),:]
testSet = unsplitDf.iloc[int(0.8 * len(unsplitDf)):,:]

""" Subtask 2- clean up the data: I want it cleaned up before normalization"""

""" showHists(trainSet[colsTrain]) """
trainSet = removeOutliers(trainSet)
trainSet = removeOutliers(trainSet, "Fare")
trainSet["Age"].fillna(trainSet.Age.mean(), inplace=True)
trainSet["Fare"].fillna(trainSet.Fare.mean(), inplace=True)
showHeatMap(trainSet)
""" showHists(trainSet[colsTrain]) """

""" Normalize data """
normalizedDf = trainSet
normalizedDf['Age'] = pk.scale(trainSet['Age'])
normalizedDf['Fare'] = pk.scale(trainSet['Fare'])
testSet.loc[:, 'Age'] = pk.scale(testSet[['Age']])
testSet.loc[:, 'Fare'] = pk.scale(testSet[['Fare']])

""" Subtask 3 """

X = normalizedDf[colsTrain]
y = normalizedDf['Survived']
y_truth = testSet['Survived']
X_truth = testSet[colsTrain]

if (len(sys.argv) == 3):
    param1 = sys.argv[1]
    param2 = sys.argv[2]
    clf = ptree.DecisionTreeClassifier(criterion=param1, max_depth=int(param2))
    clf.fit(X, y)
    y_pred = clf.predict(X_truth)
    accuracy = pm.accuracy_score(y_truth, y_pred)
    precision = pm.precision_score(y_truth, y_pred)
    recall = pm.recall_score(y_truth, y_pred)
    F1 = 2 * precision * recall / (precision + recall)
    print(accuracy)
elif (len(sys.argv) == 4):
    X_truth.loc[:, 'Age'] = X_truth["Age"].fillna(X_truth.Age.median())
    X_truth.loc[:, 'Fare'] = X_truth["Fare"].fillna(X_truth.Fare.median())
    clf = ens.RandomForestClassifier(int(sys.argv[1]), criterion=sys.argv[2], max_depth=int(sys.argv[3]), random_state=1, oob_score=True)
    clf.fit(X, y)
    y_pred = clf.predict(X_truth)
    accuracy = pm.accuracy_score(y_truth, y_pred)
    print(accuracy)
    scores = defaultdict(list)
    for _ in range(5):
        for column in X.columns:
            X_t = X_truth.copy()
            X_t[column] = np.random.permutation(X_t[column].values)
            shuff_acc = pm.accuracy_score(y_truth, clf.predict(X_t))
            scores[column].append(np.abs(accuracy - shuff_acc) / accuracy)
    importance, features = zip(*sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True))
    plt.figure(figsize=(10, 6))
    plt.barh(features, importance, color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()


