# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:VISHAL R 
RegisterNumber:25004464

# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
Dataset:


<img width="243" height="774" alt="Screenshot 2025-11-18 103928" src="https://github.com/user-attachments/assets/1c5c6bd9-e6ab-4286-9fda-ede653c2c8bb" />


Head values:


<img width="242" height="209" alt="Screenshot 2025-11-18 103939" src="https://github.com/user-attachments/assets/85694b44-8a00-45b1-844f-0e73015c562d" />


Tail values:


<img width="238" height="210" alt="Screenshot 2025-11-18 103948" src="https://github.com/user-attachments/assets/8b64939c-d28c-4a47-ad15-ba6c3490fded" />


X and Y values:


<img width="966" height="787" alt="Screenshot 2025-11-18 104028" src="https://github.com/user-attachments/assets/c15611d3-acb2-4c0b-8340-ad992ba8cd2b" />


Predication values of X and Y:


<img width="972" height="117" alt="Screenshot 2025-11-18 104038" src="https://github.com/user-attachments/assets/3f0770a4-63e5-496f-b479-6ae48c51201c" />


MSE,MAE and RMSE:


<img width="359" height="144" alt="Screenshot 2025-11-18 104044" src="https://github.com/user-attachments/assets/24f1c3bc-fa68-42bb-884d-56859bd89a94" />


<img width="832" height="664" alt="Screenshot 2025-11-18 104051" src="https://github.com/user-attachments/assets/061bd8b3-104b-4405-a0fb-1b11ec479b8e" />


<img width="756" height="685" alt="Screenshot 2025-11-18 104100" src="https://github.com/user-attachments/assets/bef2a6f3-d966-4c8c-9475-10b794e7c6de" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
