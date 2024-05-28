import numpy as np
import pandas as pd

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

""" Subtask 2- clean up the data: I want it cleaned up before splitting """
unsplitDf = pd.read_csv('../DataForAll/train.csv')
""" unsplitDf = unsplitDf.drop('Cabin', axis=1) """
unsplitDf.replace({'male': 0, 'female' : 1}, inplace=True)
unsplitDf.replace({'C': 0, 'Q' : 1, 'S': 2}, inplace=True)
unsplitDf["Age"].fillna(unsplitDf.Age.mean(), inplace=True)
unsplitDf["Embarked"].fillna(unsplitDf.Embarked.median(), inplace=True)

""" Subtask 1 - split dataset """
trainSet = unsplitDf.iloc[:int(0.8 * len(unsplitDf)),:]
testSet = unsplitDf.iloc[int(0.8 * len(unsplitDf)):,:]
print(trainSet, testSet)

""" Subtask 3 - """
