
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


#######################################################################################################################
###########################################  // Data set // ###########################################################
#######################################################################################################################

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
#merge = [df_train, df_test]
print(df_train.columns.values)

#######################################################################################################################
###########################################  // Plots - curves // #####################################################
#######################################################################################################################
g = sns.FacetGrid(df_train, col='Survived')
g.map(plt.hist, 'Age', bins=20)

g = sns.FacetGrid(df_train, col='Survived')
g.map(plt.hist, 'Pclass', bins=10)

g = sns.FacetGrid(df_train, col='Survived')
g.map(plt.hist, 'Age', bins=10)

#######################################################################################################################
###########################################  // Data cleaning // ######################################################
#######################################################################################################################

#train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#df_train[['Pclass', 'Survived']].groupby(['Pclass']).mean()
#df_train[['Sex', 'Survived']].groupby(['Sex']).mean()

df_train = df_train.drop(['Ticket', 'Cabin'], axis=1)
df_test = df_test.drop(['Ticket', 'Cabin'], axis=1)
df_train.head()
### Age
df_train.Age = df_train.Age.fillna(df_train.Age.mean())
df_test.Age = df_train.Age.fillna(df_train.Age.mean())
### Fare
df_train.Fare = df_train.Fare.fillna(df_train.Fare.mean())
df_test.Fare = df_train.Fare.fillna(df_train.Fare.mean())
### Sex
df_train.Sex = df_train.Sex.fillna(1)
df_test.Sex = df_train.Sex.fillna(1)
### SibSp
df_train.SibSp = df_train.SibSp.fillna(0)
df_test.SibSp = df_train.SibSp.fillna(0)
### Parch
df_train.Parch = df_train.Parch.fillna(0.38)
df_test.Parch = df_train.Parch.fillna(0.38)

# Sex - Change male to 1 and female to 0
df_train['Sex'] = df_train['Sex'].map( {'female': 0, 'male': 1} )
df_test['Sex'] = df_test['Sex'].map( {'female': 0, 'male': 1} )

# Define X_train & Y_train & X_test
X_train=df_train.drop(['Embarked','Survived','PassengerId','Name'], axis=1)
Y_train = df_train["Survived"]
X_test=df_test.drop(['Embarked','PassengerId','Name'], axis=1)
X_train.head(5)

#######################################################################################################################
###########################################  // Models // #############################################################
#######################################################################################################################


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log

# SVC
svc = SVC()
svc.fit(X_train, Y_train)
Y_predsvc = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

# KNN - Nearest neigbour 
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_predknn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian

# SGDClassifier

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 3)
acc_sgd

# Random forest
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred_rf = random_forest.predict(X_test)

acc_rf=round(random_forest.score(X_train, Y_train)*100, 2)
acc_rf

#######################################################################################################################
###########################################  // Verifications - Model accuracy // #####################################
#######################################################################################################################





#######################################################################################################################
###########################################  // Submission // #########################################################
#######################################################################################################################
models = pd.DataFrame({
    'Model': [ 'Logistic Regression','SVC','RandomForest','KNeighborsClassifier'],
    'Score': [acc_log, acc_svc,acc_rf,acc_knn]})
models.sort_values(by='Score', ascending=False)



submission = pd.DataFrame({"PassengerId": df_test["PassengerId"],"Survived": Y_predsvc})
    
submission.head()

#######################################################################################################################
###########################################  // Differences - Models // ###############################################
#######################################################################################################################


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_predknn, Y_pred_rf)
print(mse)

submission.to_csv("kaggle.csv", index=False)


#######################################################################################################################
###########################################  // Plots // ##############################################################
#######################################################################################################################

"""
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train.drop(['Ticket','Name', 'Embarked','Survived','Pclass','Sex'], axis = 1)
def format_name(df):
    df['Lname'] = df.Name.apply(lambda x:x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x:x.split(' ')[1])
    return df

format_name(df_train)
"""
"""
g = sns.FacetGrid(df_train, col='Survived')
g.map(plt.hist, 'Age', bins=20)

g = sns.FacetGrid(df_train, col='Survived')
g.map(plt.hist, 'Pclass', bins=10)

g = sns.FacetGrid(df_train, col='Survived')
g.map(plt.hist, 'Age', bins=10)
"""

"""
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1,0,5,12,18,25,35,60,120)
    group_names = ['Unknown','Baby', 'Child', 'Teenager', 'Student','Young Adult', 'Adult','Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x:x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1,0,8,15,31,1000)
    group_names = ['UnKnown', '1_quartile','2_quartile','3_quartile','4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x:x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x:x.split(' ')[1])
    return df

def drop_features(df):
    return df.drop(['Ticket','Name', 'Embarked'], axis = 1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

data_train = transform_features(data_train)
data_test = transform_features(data_test)
data_train.head()
"""