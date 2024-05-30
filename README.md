# PCLP3
Our project for the PCLP3 course

Part II

Files:

ParteaII.py:

Funcitons:
- removeOutliers: removes values that are smaller than Q1 - 1.5 IQR or bigger than Q# + 1.5 IQR
  for a specific column, the default being the Age column
- zScore: removes outliers who's Z-score is bigger than 3 or smaller than -3
- showHists: shows the distribution of values in the columns of a dataframe
- showHeatMap: shows the correlation between the columns of a database, we are interested in 
  the correlation between survival and the other factors

Program:

The script reads the train.csv file, and creates a dataframe based on it containing only the
following columns: 'Age', 'Sex', 'Pclass', 'Fare', 'Parch', 'SibSp'.

It makes sure all columns are filled with numbers, then splits it randomly into a training and
a testing set (0.8 to 0.2). The training set is then cleaned up using the functions above, any
NaN values are filled in, all the while the showHeatMap is being called on it to show the value
distribution before and after. At this point the correltion heat map is also created.

For the cleanup process the function removeOutliers is used because it was shown to have a
better performance accross many tests. The gaps are filled with the mean and not the median for
the same reason. The columns on which the cleanup process focuses are Age and Fare because they
have the most NaN values and their distributions are the most "continuous".

Before starting the machine learning, the data is normalised and broken down into its X and y
components

After that, there is a choice between Decision Tree and Random Forest for the learning process,
both of them can be tested, although Random Forests is consistently the better option.

At the end some stats are calculated and shown: accuracy, precision, recall and F1, and for the
Random forest the importance of the variables is also shown, sex being consistently the most
important variable that determins the survivability of a person.

decisionTreeParams:

A script that runs parteaII.py with different settings for the Decision Tree algorithm and stores
the results into the file accuracy.txt.

randomForestParams:

A script that runs parteaII.py with different settings for the Random Forest algorithm and stores
the results into the file results.csv.

resultsInterpreter.py:

A script that reads the results.csv file and returns the best performing parameters.



