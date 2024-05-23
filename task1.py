import numpy as np
import pandas as pd

file_info = pd.read_csv('train.csv')
"""Task1"""
n, m = file_info.shape
print("Number of columns in this file:", n)
print("Column types: ")
print(file_info.dtypes)
print("Gaps in each column: ")
print(file_info.isnull().sum())
print("Number of rows in this file:", m)
cnt = 0
dup_lines = file_info.duplicated()
for line in dup_lines:
    if line == 1:
        cnt = cnt + 1
print("There are", cnt, "duplicate rows in this file")
    


