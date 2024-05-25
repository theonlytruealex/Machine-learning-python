import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gaps(file_info):
    gaps = file_info.isnull()
    gap_cols = gaps.columns
    gap_cols_list = ""
    for i in gap_cols:
        res = gaps[gaps[i] == 1]
        if res.empty == 0:
            gap_cols_list = gap_cols_list + i + " "
    print("There are gaps in these columns:", gap_cols_list)
    sum_list = gaps.sum()
    print("Number of gaps in each column:")
    print(sum_list[sum_list != 0])
    print("Percentage of gaps in each column:")
    print(sum_list[sum_list != 0] / n * 100)

file_info = pd.read_csv('train.csv')

"""Task1"""
n, m = file_info.shape
print("Number of columns in this file:", m)
print("Column types: ")
print(file_info.dtypes)
print("Gaps in each column: ")
print(file_info.isnull().sum())
print("Number of rows in this file:", n)
cnt = 0
dup_lines = file_info.duplicated()
for line in dup_lines:
    if line == 1:
        cnt = cnt + 1
print("There are", cnt, "duplicate rows in this file")
    
""" Task 2 """
survived = len(file_info[file_info['Survived'] == 1])
survived_proc = round(survived / n * 100, 2)
print(survived_proc,"% of the passengers survived")
not_survived_proc = round(100 - survived / n * 100, 2)
print(not_survived_proc,"% of the passengers did not survive")
class_no = np.zeros(3)
for i in range(1,4):
    class_no[i - 1] = len(file_info[file_info['Pclass'] == i])
    class_no[i - 1] = round(class_no[i - 1] / n * 100, 2)
    print("There are", class_no[i - 1],"% class", i, "passengers")
male = len(file_info[file_info['Sex'] == 'male'])
female = len(file_info[file_info['Sex'] == 'female'])
male_proc = round(male / n * 100, 2)
female_proc = round(female / n * 100, 2)
print("There are", male_proc, "% male passengers")
print("There are", female_proc, "% female passengers")
g_survived = np.array([survived_proc, not_survived_proc])
label_survived = ["Survived", "Died"]
plt.title("Survived pie chart")
plt.pie(g_survived, labels = label_survived)
plt.legend(g_survived)
plt.show()
label_class = ["1", "2", "3"]
plt.title("Class pie chart")
plt.pie(class_no, labels = label_class)
plt.legend(class_no)
plt.show()
g_gender = np.array([male_proc, female_proc])
label_gender = ["Male", "Female"]
plt.title("Gender pie chart")
plt.pie(g_gender, labels = label_gender)
plt.legend(g_gender)
plt.show()

""" Task 3 """
num_cols = file_info.select_dtypes(include=np.number)
col = num_cols.columns
for i in col:
    plt.title(i)
    plt.hist(num_cols[i])
    plt.show()

""" Task 4 """
print("All gaps:")
gaps(file_info)
s_file_info = file_info[file_info['Survived'] == 1]
print("Gaps for passengers who survived:")
gaps(s_file_info)
print("Gaps for passengers who did not survive:")
s_file_info = file_info[file_info['Survived'] == 0]
gaps(s_file_info)
