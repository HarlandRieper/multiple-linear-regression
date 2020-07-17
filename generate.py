from sympy import *
import random
import numpy as np

samples=10;
x1234 = np.zeros((samples,3))
for i in range(samples):
    for j in range(3):
        x1234[i][j]=random.uniform(0-i*5,10+i*10);
print(x1234)
w_hat=np.array([[4.9],[3.6],[2.5]])
b_hat = 5.5
print(w_hat)
y_hat = x1234.dot(w_hat)+b_hat
print(y_hat)
