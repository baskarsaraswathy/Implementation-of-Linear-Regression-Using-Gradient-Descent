# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*

Program to implement the linear regression using gradient descent.
Developed by: BASKAR J
RegisterNumber: 212223040025

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
    return theta
data = pd.read_csv('50_Startups.csv',header=None)
data.head()
X = (data.iloc[1:, :-2].values)
X1=X.astype(float)
print(X1)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
theta = linear_regression(X1_Scaled, Y1_Scaled)
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

*/

```

## Output:

### Dataset :
![image](https://github.com/user-attachments/assets/e4269b9b-ebdb-4f99-8c11-18cb7b59a108)


### X nd Y values:
![image](https://github.com/user-attachments/assets/c472c270-b657-4b22-b9fb-8ea205bd574c)


![image](https://github.com/user-attachments/assets/13d30526-d09e-4c6a-9e6e-c31e70621f16)


### Predicted Value :
![image](https://github.com/user-attachments/assets/b791b2d4-3009-4fab-8d5c-18d4befb8c1f)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
