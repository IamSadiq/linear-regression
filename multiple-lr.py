import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# import re

# 3D vector
X = []
# column vector
Y = []

data = np.genfromtxt('data/moores.csv', skip_header=1, delimiter=';', names=['Year', 'Transistors', 'Frequency', 'PowerDensity', 'Cores'])

for line in data:
    year = line['Year']
    cores = line['Cores']
    clock = line['Frequency']
    transistors = line['Transistors']

    X.append([float(year), float(cores), float(clock), 1])
    Y.append(float(transistors))

X = np.array(X)
Y = np.array(Y)

# show the distribution in a graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.scatter(X[:,0], X[:,1], X[:,2], Y)
# plt.savefig('my-figures/3d-moores.png')
plt.show()

# take the Log of transistor counts (Y) to make it linear
Y = np.log(Y)

# show again
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.scatter(X[:,0], X[:,1], X[:,2], Y)
# plt.savefig('my-figures/3d-moores-made-linear.png')
plt.show()

# our solution: w = (xTX)^-1 * xTY
# Ax = b: x = A^-1 * b ----- x = np.linalg.solve(A, b)
# As such: w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
print('w: ', w)

# # our Model: Prediction
yhat = np.dot(X, w)

# # Calculated Squared Error (Squared - residual)
d1 = Y - yhat
SSres = d1.dot(d1)
# # OR: SSres = ((Y - yhat)**2).sum()

d2 = Y - Y.mean()
SStot = d2.dot(d2)
# # OR: SStot = ((Y - Y.mean())**2).sum()

r_squared = 1 - SSres/SStot

print('\nSquared Error (SSres): ', SSres)
print('Predicted Mean (SStot): ', SStot)
print('R-Squared: ', r_squared)

# Plot the Model to fit the distribution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.scatter(X[:,0], X[:,1], X[:,2], Y)
ax.plot(yhat, Y)
# plt.savefig('my-figures/fig2.png')
plt.show()