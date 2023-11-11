# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: CHARUMATHI R 
RegisterNumber: 212222240021 
*/
```
```c
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
## Result output:
![281978212-1be2e57f-2501-41c0-862a-19bd02626dc6](https://github.com/charumathiramesh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120204455/28a5795a-2580-433a-9443-e2f07c687b5e)


## data.head():
![281978242-afdcd24d-f5f0-48e8-ac1f-6fa351dc640d](https://github.com/charumathiramesh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120204455/5cdf8c27-cb0a-43b9-86c0-a2db5671c78d)

## data.info():


![281978267-56281d06-3be6-42b9-b41c-6022904ee09f](https://github.com/charumathiramesh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120204455/6c9e9c39-9def-41c3-993e-73ef4e37ac30)

## data.isnull().sum():
![281978298-d5bb37c1-052e-46c8-b99f-ef52ad4996bb](https://github.com/charumathiramesh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120204455/8cc08474-d436-4d1f-b9fd-300e03f40aca)

## Y_prediction value:
![281978334-c709b158-e17a-497d-923b-122cff2eff12](https://github.com/charumathiramesh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120204455/4ae9ec2d-9001-432f-8bb9-9ae3de9e2311)

 ## Accuracy value:

![281978407-d2dbf4c8-9e19-4d3a-ab23-ecb68d490c99](https://github.com/charumathiramesh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120204455/7d5ffca2-ba4e-4690-b5d1-524d12659f1c)

 



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
