import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

data = pd.read_csv('Surveillance.csv')
x = data[['A01','A02','A03','A04','A05','A06','A07']].values

for i in range(0,7):
    le_A = pp.LabelEncoder()
    le_A.fit(['+','-'])
    x[:,i] = le_A.transform(x[:,i])

y = data['Categories']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=4)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(x_train,y_train)

y_pred = drugTree.predict(x_test)

print(f1_score(y_pred,y_test,average='weighted'))
