import numpy as np
import matplotlib.pyplot as plt
import math

X = []
Y = []

for line in open('data/linear-data.csv'):
    x,y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

xDotx = X.dot(X)
xDoty = X.dot(Y)

xMean = X.mean()
yMean = Y.mean()
xSum = X.sum()

# yhat = a*X + b
denominator =  xDotx - xMean * xSum
# print(denominator)

a = (xDoty - yMean * xSum) / denominator
print('a: ', a)

b = (yMean * xDotx - xMean * xDoty) / denominator
print('b: ', b)

# Model -- predicted Y -- yhat
yhat = a*X + b

# make prediction --- using untrained x value
someX = 100
predY = a*someX + b

print('\nTest X: ', someX)
print('Predicted Y:  ', predY)

# Calculated Squared Error (Squared - residual)
d1 = Y - yhat
SSres = d1.dot(d1)
# OR: SSres = ((Y - yhat)**2).sum()

d2 = Y - yMean
SStot = d2.dot(d2)
# OR: SStot = ((Y - yMean)**2).sum()

r_squared = 1 - SSres/SStot

print('\nSquared Error (SSres): ', SSres)
print('Predicted Mean (SStot): ', SStot)
print('R-Squared: ', r_squared)

plt.scatter(X, Y)
plt.plot(yhat)
plt.show()