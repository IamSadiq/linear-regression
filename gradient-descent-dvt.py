"""
------------------->GRADIENT DESCENT<--------------------
Using Gradient Descent to Bypass the Dummy Variable Trap
---------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt

# sample size 
N = 10

# feature dimension
D = 3

# inputs
X = np.zeros([N, D])

# set the bias term
X[:,0] = 1
X[:5,1] = 1
X[5:,2] = 1

# targets
Y = np.array([0]*5 + [1]*5)

costs = []
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.005

for t in range(1000):
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - learning_rate*X.T.dot(delta)
    mse = delta.dot(delta)
    costs.append(mse)

plt.plot(costs)
plt.show()