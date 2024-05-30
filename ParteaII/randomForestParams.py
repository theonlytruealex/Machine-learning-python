import subprocess
import pandas as pd

outputPath = "accuracy.txt"
df = pd.DataFrame()
index = 0
for i in range(50, 210, 10):
    for k in range(5, 21):
        result = 0
        for j in range(0, 5):
            command = ["python3", "parteaII.py", str(i), "entropy", str(k)]
            result += float(subprocess.run(command, capture_output=True, text=True).stdout)
        df.loc[index, 'criterion'] = "entropy"
        df.loc[index, 'nEstimators'] = i
        df.loc[index, 'maxDepth'] = k
        df.loc[index, 'accuracy'] = result / 5 
        index += 1

for i in range(50, 210, 10):
    for k in range(5, 21):
        result = 0
        for j in range(0, 5):
            command = ["python3", "parteaII.py", str(i), "gini", str(k)]
            result += float(subprocess.run(command, capture_output=True, text=True).stdout)
        df.loc[index, 'criterion'] = "gini"
        df.loc[index, 'nEstimators'] = i
        df.loc[index, 'maxDepth'] = k
        df.loc[index, 'accuracy'] = result / 5 
        index += 1
df.to_csv('results.csv')
