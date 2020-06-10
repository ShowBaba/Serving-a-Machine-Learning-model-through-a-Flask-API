import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split

# %matplotlib inline

train = pd.read_csv('train.csv')
include = ['Age', 'Sex', 'Embarked', 'Survived']
train = train[include]
categories = []
for col, col_type in train.dtypes.iteritems():
    if col_type == 'O':
        categories.append(col)
    else:
        train[col].fillna(0, inplace=True)

# train.fillna(0, inplace = True)
# train.dropna(inplace = True)
# print(train.shape)
train = pd.get_dummies(train, columns=categories, dummy_na=True)
# print(train.head())
from sklearn.linear_model import LogisticRegression
X = train.drop(columns=['Survived'])
y = train['Survived']

lr= LogisticRegression()
lr.fit(X,y)

# Saving model: Serializing
# joblib from sklearn not working so i used joblib directly
import joblib
joblib.dump(lr, 'model.pkl')
print('Model dumped')
# model is now persisted
# to deserialized model
# lr = joblib.load('model.pkl')

# persist columns to handle null columns during request
joblib.dump(list(X.columns), 'model_columns.pkl')
print('Columns dumped')