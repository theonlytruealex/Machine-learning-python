import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def gaps(file_info, n):
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
    if i != 'PassengerId':
        plt.title(i)
        plt.hist(num_cols[i])
        plt.show()

""" Task 4 """
print("All gaps:")
gaps(file_info, n)
s_file_info = file_info[file_info['Survived'] == 1]
print("Gaps for passengers who survived:")
gaps(s_file_info, n)
print("Gaps for passengers who did not survive:")
s_file_info = file_info[file_info['Survived'] == 0]
gaps(s_file_info, n)

""" Task 5 """
ages1 = file_info[file_info['Age'] <= 20]
nr1 = len(ages1)
print("[0, 20]:", nr1, "passengers")
ages2 = file_info[file_info['Age'] <= 40]
ages2 = ages2[ages2['Age'] > 20]
nr2 = len(ages2)
print("[21, 40]:", nr2, "passengers")
ages3 = file_info[file_info['Age'] <= 60]
ages3 = ages3[ages3['Age'] > 40]
nr3 = len(ages3)
print("[41, 60]:", nr3, "passengers")
ages4 = file_info[file_info['Age'] > 60]
nr4 = len(ages4)
print("[61, max]:", nr4, "passengers")


no_children = [random.randint(0, 3) for _ in range(n)]
file_info['Children'] = no_children
plt.title("Number of children per passenger")
plt.hist(file_info['Children'])
plt.show()

""" Task 6 """
ages1 = ages1[ages1['Sex'] == 'male']
men_ages1 = len(ages1)
ages1 = ages1[ages1['Survived'] == 1]
men_ages1 = len(ages1) / men_ages1 * 100
print("[0, 20]:", len(ages1), "men")
ages2 = ages2[ages2['Sex'] == 'male']
men_ages2 = len(ages2)
ages2 = ages2[ages2['Survived'] == 1]
men_ages2 = len(ages2) / men_ages2 * 100
print("[21, 40]:", len(ages2), "men")
ages3 = ages3[ages3['Sex'] == 'male']
men_ages3 = len(ages3)
ages3 = ages3[ages3['Survived'] == 1]
men_ages3 = len(ages3) / men_ages3 * 100
print("[41, 60]:", len(ages3), "men")
ages4 = ages4[ages4['Sex'] == 'male']
men_ages4 = len(ages4)
ages4 = ages4[ages4['Survived'] == 1]
men_ages4 = len(ages4) / men_ages4 * 100
print("[61, max]:", len(ages4), "men")
values = [men_ages1, men_ages2, men_ages3, men_ages4]
ranges = ['0, 20', '20, 40', '40, 60', '60, 100']
data = {ranges[0]: [values[0]], ranges[1]: [values[1]], ranges[2]: [values[2]], ranges[3]: [values[3]]}
new_df = pd.DataFrame(data)
new_df.plot(kind = 'bar')
plt.title("Men survival percentage based on age")
plt.show()

"""Task 7"""

children = file_info[file_info['Age'] < 18]
children_no = len(children)
children_no = children_no / n * 100
print("Percentage of children on board:", round(children_no, 2))
ch_surv = len(children[children['Survived'] == 1])
ch_surv = ch_surv / children_no * 100
adults = file_info[file_info['Age'] >= 18]
adults_no = len(adults)
ad_surv = len(adults[adults['Survived'] == 1])
ad_surv = ad_surv / adults_no * 100
data = {'Adults': [ad_surv], 'Children': [ch_surv]}
new_df = pd.DataFrame(data)
new_df.plot(kind = 'bar')
plt.title("Adults / Children survival percentage")
plt.show()
