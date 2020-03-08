# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:49:20 2019

@author: Madalin
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# 1) LOAD THE DATASET

# Import the data from the research project folder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import pickle

diabetes = pd.read_csv('diabetes.csv')

# 2) INSPECT THE DATASET

#Diabetes dataset columns
print(diabetes)


# Diabetes dataset head
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

diabetes.head(n= 10)

# Diabetes dataset dimension of data
print("Diabetes dataset dimension of data: {}".format(diabetes.shape))

#Dependent variable of our dataset
print(diabetes.groupby('Outcome').size())
#Outcome plot
import seaborn as sns
sns.countplot(diabetes['Outcome'],label="Count")

#Diabates dataset information
diabetes.info()

# 3) DIABETES DATASET CORRELATION MATRIX
corr = diabetes.corr()
corr
%matplotlib inline
import seaborn as sns
sns.heatmap(corr, annot = True)

g = sns.heatmap(diabetes.corr(),cmap="Blues",annot=False)

corr = diabetes.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);



#VISUALIZE OUR DATASET
import matplotlib.pyplot as plt
diabetes.hist(bins=50, figsize=(20, 15))
plt.show()


# 4) DATA CLEANING AND TRANSFORMATION

# Calculate the median value for BMI
median_bmi = diabetes['BMI'].median()
# Substitute it in the BMI column of the
# dataset where values are 0
diabetes['BMI'] = diabetes['BMI'].replace(
    to_replace=0, value=median_bmi)


# Calculate the median value for BloodPressure
median_bloodpressure = diabetes['BloodPressure'].median()
# Substitute it in the BloodP column of the
# dataset where values are 0
diabetes['BloodPressure'] = diabetes['BloodPressure'].replace(
    to_replace=0, value=median_bloodpressure)



# Calculate the median value for Glucose
median_glucose = diabetes['Glucose'].median()
# Substitute it in the Glucose column of the
# dataset where values are 0
diabetes['Glucose'] = diabetes['Glucose'].replace(
    to_replace=0, value=median_glucose)



# Calculate the median value for SkinThickness
median_skinthickness = diabetes['SkinThickness'].median()
# Substitute it in the SkinThick column of the
# dataset where values are 0
diabetes['SkinThickness'] = diabetes['SkinThickness'].replace(
    to_replace=0, value=median_skinthickness)



# Calculate the median value for Insulin
median_insulin = diabetes['Insulin'].median()
# Substitute it in the Insulin column of the
# dataset where values are 0
diabetes['Insulin'] = diabetes['Insulin'].replace(
    to_replace=0, value=median_insulin)


#VISUALIZE THE DATASET
import matplotlib.pyplot as plt
diabetes.hist(bins=50, figsize=(20, 15))
plt.show()


# 5) SPLITTING THE DATASET

dataset_X = diabetes.iloc[:,[0, 1, 2, 3, 4, 5, 6, 7]].values
dataset_Y = diabetes.iloc[:,8].values


dataset_X

#FEATURE SCALING

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dataset_scaled = scaler.fit_transform(dataset_X)
# 7) SCALED VALUES
dataset_scaled = pd.DataFrame(dataset_scaled)

X = dataset_scaled
Y = dataset_Y

X

Y

# Split the training dataset in 75% / 25%

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42, stratify = diabetes['Outcome'] )




# 8) Data modelling

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

LR_model=LogisticRegression(solver = 'lbfgs', multi_class= 'auto')
LR_model.fit(X_train, Y_train)
LR_model.score(X_train, Y_train)
Y_pred = LR_model.predict(X_test)


pickle.dump(LR_model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))