import subprocess
import pandas as pd

""" test paramethers for rando forest function """
df = pd.DataFrame()
index = 0

""" test numbers of estimators """
for i in range(50, 210, 10):

    """ test max depth """
    for k in range(5, 21):
        result = 0

        """ take the average of 5 tests to have a more accurate picture """
        for j in range(0, 5):
            command = ["python3", "parteaII.py", str(i), "entropy", str(k)]
            result += float(subprocess.run(command, capture_output=True, text=True).stdout)
        df.loc[index, 'criterion'] = "entropy"
        df.loc[index, 'nEstimators'] = i
        df.loc[index, 'maxDepth'] = k
        df.loc[index, 'accuracy'] = result / 5 
        index += 1

""" try another criterion """
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
""" write the results in order not to compute them twice """
df.to_csv('results.csv')
