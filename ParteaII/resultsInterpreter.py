import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

def showHeatMap(dataFrame):
    dataFrame.corr()
    corrMatrix = dataFrame.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()

df = pd.read_csv('./results.csv')

""" take results of randomForestParams """
df.replace({'entropy': 1, 'gini' : 0}, inplace=True)

""" take top 50 parameter combinations """
worldsFinest = df.nlargest(50, columns=['accuracy'], keep='all')
print(worldsFinest)
print("Best criterion: ", worldsFinest['criterion'].mean())
print("Best number of estimators: ", worldsFinest['nEstimators'].median())
print("Best max depth: ", worldsFinest['maxDepth'].median())
