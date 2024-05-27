import numpy as np
import pandas as pd

""" Task 1 """

dataFrame = pd.read_csv('../DataForAll/train.csv')
[Q1, Q3] = dataFrame.Age.quantile(q=[0.25, 0.75])
IQR = Q3 - Q1
print (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
noOutDf = dataFrame.query('(@Q1 - 1.5 * @IQR) <= Age <= (@Q3 + 1.5 * @IQR)')
print(noOutDf["Age"].max(), noOutDf["Age"].min())
