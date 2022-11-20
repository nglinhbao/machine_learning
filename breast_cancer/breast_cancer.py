import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score

data = pd.read_csv('dataR2.csv')

x = data[['Age','BMI','Glucose','Insulin','HOMA','Leptin','Adiponectin','Resistin','MCP.1']].values
y = data['Classification'].values

x = preprocessing.StandardScaler().fit(x).transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=4)

LR = LogisticRegression(C=0.01,solver='liblinear').fit(x_train,y_train)

y_pred = LR.predict(x_test)


print(jaccard_score(y_test,y_pred))
