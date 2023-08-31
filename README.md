# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
# Step 1 :
Import the standard libraries such as pandas module to read the corresponding csv file.
# Step 2 :
Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
# Step 3 :
Import LabelEncoder and encode the corresponding dataset values.
# Step 4 :
Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y.
# Step 5 :
Predict the values of array using the variable y_pred.
# Step 6 :
Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.
# Step 7 :
Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.
# Step 8:
End the program.

## Program:

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

/*
Developed by:YOGABHARATHI S  
RegisterNumber:212222230179  
*/
```
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
# HEAD OF THE DATA :
![image](https://github.com/Yogabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118899387/b4e7913f-e49e-4c66-8c1d-77879d8cae1d)

# COPY HEAD OF THE DATA :
![image](https://github.com/Yogabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118899387/74bca33d-99dd-4a38-9254-480e94022eea)

# NULL AND SUM :
![image](https://github.com/Yogabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118899387/6c4eb879-c0bb-43d8-aa59-06fbd2479e77)

# DUPLICATED :
![image](https://github.com/Yogabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118899387/6aff512a-e6b3-4deb-b740-dde7aab67745)

# X VALUE :
![image](https://github.com/Yogabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118899387/22995262-8589-48e3-b7a4-633979dcaabb)

# Y VALUE :
![image](https://github.com/Yogabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118899387/f3fa0ed6-e76a-40e2-8b08-03c2f0b62e89)

# PREDICTED VALUES :
![image](https://github.com/Yogabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118899387/cdbd19cf-eba7-43b5-b886-60582b949b99)

# ACCURACY :
![image](https://github.com/Yogabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118899387/3c2cd284-8581-4f2f-857a-d64bdd136f2e)

# CONFUSION MATRIX :
![image](https://github.com/Yogabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118899387/eedc2fc7-4019-4343-a877-b6f3865a9424)

# CLASSIFICATION REPORT :
![image](https://github.com/Yogabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118899387/16f224ee-e1fc-44c5-a0e9-e03edfe873a4)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
