import numpy as np
import pandas as pd

""" Task 1 """

def removeOutliers(dataFrame, col):
    [Q1, Q3] = dataFrame[col].quantile(q=[0.25, 0.75])
    IQR = Q3 - Q1
    return dataFrame.query('(@Q1 - 1.5 * @IQR) <= '+ col +' <= (@Q3 + 1.5 * @IQR)')

""" Task 2"""

def zScore(dataFrame, col):
    stdDev = dataFrame[col].std()
    mean = dataFrame[col].mean()
    maxZ = 3 * stdDev + mean
    minZ = -3 * stdDev + mean
    return dataFrame.query('@minZ <= '+ col +' <= @maxZ')


dataFrame = pd.read_csv('../DataForAll/train.csv')
col = "Age"
newDataFrame = removeOutliers(dataFrame, col)
newerDataFrame = zScore(dataFrame, col)
