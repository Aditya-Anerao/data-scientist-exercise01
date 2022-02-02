"""
RTI Analytics Exercise 

@author: Aditya Anerao
"""
# Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Read in flatten.csv file into a Pandas dataframe

df = pd.read_csv("/Users/amane/Documents/GitHub/data-scientist-exercise01/flatten.csv")

# Understand Table Dimensions - 48842 rows, 23 columns
print(df.shape)


# Examine first 10 rows of the dataframe

print(df.head(10))

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

# Check for Missing Values - No Missing Values
df.isnull().sum()

# Summary Statistics of variables
print(df_clean.describe(include ='all'))

# Print Count of Unique Values for each variable
ncol = df.shape[1]
for col in df.columns:
    print(df[col].value_counts())

# 76% of people earn less than $50000/year

# Split the data into a 70/20/10 training, validation, and test data split
X_train, X_test, Y_train, Y_test = train_test_split(df_clean.drop('over_50k', axis = 1), df_clean['over_50k'], train_size = 0.7, random_state = 123)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, train_size = 0.33, random_state = 123)


# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(df_clean.corr(),annot = True)
plt.show()



#Train model using training data
LogReg = LogisticRegression(solver='liblinear', random_state=0)
LogReg.fit(X_train, Y_train)

#Scoring the model - percentage of time that model correctly predicts whwther person makes over $50,000 per year
LogReg.score(X_val, Y_val)

#Predicted vs Actual Validation
pred = LogReg.predict(X_val)

plt.figure(figsize=(5, 7))
ax = sns.distplot(Y_val, hist=False, color="r", label="Actual Value")
sns.distplot(pred, hist=False, color="b", label="Predicted Values" , ax=ax)
plt.title('Actual vs Predicted people earning over $50,000/year')
plt.show()

# Create Confusion Matrix
cm = confusion_matrix(Y_train, LogReg.predict(X_train))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Less than or equal to $50,000/year (False Positive)', 'Predicted Greater than $50,000/year (True Negative)'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Less than or equal to $50,000/year (True Positive)', 'Actual Greater than $50,000/year (False Negative)'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


# Create Classificaiton Report
print(classification_report(Y_train, LogReg.predict(X_train)))



df2=pd.DataFrame({'Actual':Y_val, 'Predicted':pred})
df2



for i in range (0,1):
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

