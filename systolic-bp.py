import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel('data/mlr02.xls')
X = df.as_matrix()

plt.scatter(X[:,1], X[:,0])
plt.plot(sorted(X[:,1]), sorted(X[:,0]))
plt.show()

plt.scatter(X[:,2], X[:,0])
plt.plot(sorted(X[:,2]), sorted(X[:,0]))
plt.show()

df['ones'] = 1

Y = df['X1']
X = df[['X1','X2', 'ones']]

X2only = df[['X2', 'ones']]
X3only = df[['X3', 'ones']]

def get_r2(X, Y):
    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    Yhat = X.dot(w)

    d1 = Y - Yhat
    d2 = Y - Y.mean()

    return 1 - d1.dot(d1)/d2.dot(d2)

print('r2 for X2only: ', get_r2(X2only, Y))
print('r2 for X3only: ', get_r2(X3only, Y))
print('r2 for both: ', get_r2(X, Y))