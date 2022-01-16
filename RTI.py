"""
RTI Analytics Exercise 

@author: Aditya Anerao
"""
# Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
#import csv

# Read in flatten.csv file into a Pandas dataframe

df = pd.read_csv("/Users/amane/Documents/GitHub/data-scientist-exercise01/flatten.csv")

# Examine first 10 rows of the dataframe

print(df.head(10))

# Remove ID columns from dataframe
df_clean = df.drop('workclass_id', axis = 1)
df_clean = df_clean.drop('education_level_id', axis = 1)
df_clean = df_clean.drop('marital_status_id', axis = 1)
df_clean = df_clean.drop('occupation_id', axis = 1)
df_clean = df_clean.drop('relationship_id', axis = 1)
df_clean = df_clean.drop('race_id', axis = 1)
df_clean = df_clean.drop('sex_id', axis = 1)
df_clean = df_clean.drop('country_id', axis = 1)

print(df_clean.head())

# Explore variable type
print(df_clean.dtypes)

# Convert object data type to category
df_clean["workclass"] = df_clean["workclass"].astype("category")
df_clean["education_level"] = df_clean["education_level"].astype("category")
df_clean["marital_status"] = df_clean["marital_status"].astype("category")
df_clean["occupation"] = df_clean["occupation"].astype("category")
df_clean["relationship"] = df_clean["relationship"].astype("category")
df_clean["race"] = df_clean["race"].astype("category")
df_clean["sex"] = df_clean["sex"].astype("category")
df_clean["country"] = df_clean["country"].astype("category")


# Explore variable type
print(df_clean.dtypes)

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