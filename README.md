# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:shaik sameer 
RegisterNumber:  212221240051
*/
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
dataset = pd.read_csv('Placement_Data.csv') 
dataset.head() 
dataset = dataset.drop('sl_no', axis=1) 
dataset = dataset.drop('salary', axis=1) 
dataset["gender"] = dataset["gender"].astype('category') 
dataset["ssc_b"] = dataset["ssc_b"].astype('category') 
dataset["hsc_b"] = dataset["hsc_b"].astype('category') 
dataset["degree_t"] = dataset["degree_t"].astype('category') 
dataset["workex"] = dataset["workex"].astype('category') 
dataset["specialisation"] = dataset["specialisation"].astype('category') 
dataset["status"] = dataset["status"].astype('category') 
dataset["hsc_s"] = dataset["hsc_s"].astype('category') 
dataset.dtypes 
dataset["gender"] = dataset["gender"].cat.codes 
dataset["ssc_b"] = dataset["ssc_b"].cat.codes 
dataset["hsc_b"] = dataset["hsc_b"].cat.codes 
dataset["degree_t"] = dataset["degree_t"].cat.codes 
dataset["workex"] = dataset["workex"].cat.codes 
dataset["specialisation"] = dataset["specialisation"].cat.codes 
dataset["status"] = dataset["status"].cat.codes 
dataset["hsc_s"] = dataset["hsc_s"].cat.codes 
dataset.head() 
X = dataset.iloc[:, :-1].values 
Y = dataset.iloc[:, -1].values 
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) 
from sklearn.linear_model import LogisticRegression 
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, Y_train) 
Y_pred = clf.predict(X_test) 
Y_pred 
from sklearn.metrics import confusion_matrix, accuracy_score 
print(confusion_matrix(Y_test, Y_pred)) 
print(accuracy_score(Y_test, Y_pred))
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
