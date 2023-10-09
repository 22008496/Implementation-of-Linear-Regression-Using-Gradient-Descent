# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Upload the file to your compiler.
2. Type the required program.
3. Print the program.
4. End the program.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Magesh.N
RegisterNumber:212222040091  
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1 (2).txt", header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Popuation of city (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="purple")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions = np.dot(theta.transpose(),x)
    return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000 , we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000 , we predict a profit of $"+str(round(predict2,0)))

```

## Output:
![Screenshot 2023-10-09 103905](https://github.com/22008496/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119476113/5a3b212d-517c-4aea-a12b-faa2fdbf8fe5)


![Screenshot 2023-10-09 103930](https://github.com/22008496/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119476113/e142b373-6106-4810-8a24-69f2f7c4f851)


![Screenshot 2023-10-09 103951](https://github.com/22008496/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119476113/cc02f4e8-95d5-4558-9817-4d9486bd4a74)


![Screenshot 2023-10-09 104021](https://github.com/22008496/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119476113/0debf3e4-c3a7-4a62-99ee-c5274a699a0e)


![Screenshot 2023-10-09 104054](https://github.com/22008496/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119476113/234301af-d52e-43ec-992c-7c3e2f518e99)


![Screenshot 2023-10-09 104116](https://github.com/22008496/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119476113/3c751750-9333-46fa-a5ba-f6d2553a8545)


![Screenshot 2023-10-09 104133](https://github.com/22008496/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119476113/ec6fd3c9-79e8-4bf9-aa21-95becf12b0cf)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
