"""
RTI Analytics Exercise 

@author: Aditya Anerao
"""
# Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Read in flatten.csv file into a Pandas dataframe

df = pd.read_csv("/Users/amane/Documents/GitHub/data-scientist-exercise01/flatten.csv")

# Examine first 10 rows of the dataframe

print(df.head(10))

# Remove ID columns from dataframe
#df_clean = df.drop('workclass_id', axis = 1)
#df_clean = df_clean.drop('education_level_id', axis = 1)
#df_clean = df_clean.drop('marital_status_id', axis = 1)
#df_clean = df_clean.drop('occupation_id', axis = 1)
#df_clean = df_clean.drop('relationship_id', axis = 1)
#df_clean = df_clean.drop('race_id', axis = 1)
#df_clean = df_clean.drop('sex_id', axis = 1)
#df_clean = df_clean.drop('country_id', axis = 1)

# Remove Descriptive columns for categorical variables from dataframe
df_clean = df.drop('id', axis = 1)
df_clean = df_clean.drop('workclass', axis = 1)
df_clean = df_clean.drop('education_level', axis = 1)
df_clean = df_clean.drop('marital_status', axis = 1)
df_clean = df_clean.drop('occupation', axis = 1)
df_clean = df_clean.drop('relationship', axis = 1)
df_clean = df_clean.drop('race', axis = 1)
df_clean = df_clean.drop('sex', axis = 1)
df_clean = df_clean.drop('country', axis = 1)

print(df_clean.head())

# Explore variable type
print(df_clean.dtypes)

# Convert object data type to category
df_clean['over_50k'] = df_clean['over_50k'].astype("category") #Target Variable
df_clean["workclass_id"] = df_clean["workclass_id"].astype("category")
df_clean["education_level_id"] = df_clean["education_level_id"].astype("category")
df_clean["marital_status_id"] = df_clean["marital_status_id"].astype("category")
df_clean["occupation_id"] = df_clean["occupation_id"].astype("category")
df_clean["relationship_id"] = df_clean["relationship_id"].astype("category")
df_clean["race_id"] = df_clean["race_id"].astype("category")
df_clean["sex_id"] = df_clean["sex_id"].astype("category")
df_clean["country_id"] = df_clean["country_id"].astype("category")


# Explore variable type
print(df_clean.dtypes)


# Summary Statistics



# Split the data into a 70/20/10 training, validation, and test data split
X_train, X_test, Y_train, Y_test = train_test_split(df_clean.drop('over_50k', axis = 1), df_clean['over_50k'], train_size = 0.7, random_state = 123)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, train_size = 0.33, random_state = 123)


#Train model using training data
LogReg = LogisticRegression()
LogReg.fit(X_train, Y_train)

#Scoring the model - percentage of time that model correctly predicts whwther person makes over $50,000 per year
LogReg.score(X_val, Y_val)



for i in range(0,1):
    print(df_clean.iloc[0])



#Plots in matplotlib reside within a figure object, use plt.figure to create new figure
fig=plt.figure()
#Create one or more subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)
#Variable
ax.hist(df_clean['age'],bins = 5)


#Labels and Tit
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('# of Customers')
plt.show()

for i in range (0,1):
#range(0,len(df)):
    print(df_clean.iloc[i])