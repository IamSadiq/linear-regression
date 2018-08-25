import numpy as np
import matplotlib.pyplot as plt

def make_poly(x, deg):
    n = len(x)
    data = [np.ones(n)]

    for d in range(deg):
        data.append(x**(d+1))

    
    # data = np.array(data)
    # return data.T
    return np.vstack(data).T

def fit(X, Y):
    return np.linalg.solve(X.T.dot(X), X.T.dot(Y))

# test it
N = 100
X = np.linspace(0, 6*np.pi, N)
Y = np.sin(X)

# for i in range(0, int(6*np.pi)):
#     print(i)

plt.scatter(X, Y)
plt.plot(X, Y)
plt.show()

Xtrain = make_poly(X, 3)

# solution
w = fit(Xtrain, Y)

# Model
Y_hat = Xtrain.dot(w)

plt.scatter(Xtrain, Y)
plt.plot(Xtrain, Y_hat)
plt.show()