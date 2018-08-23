import numpy as np
import matplotlib.pyplot as plt
import math

X = []
Y = []

data = np.genfromtxt('data/moores.csv', skip_header=1, delimiter=';', names=['Year', 'Transistors', 'Frequency', 'PowerDensity', 'Cores'])

for line in data:
    year = line['Year']
    transistors = line['Transistors']

    # NOTE: moores law is exponential but the log of it is linear
    X.append(float(year))
    Y.append(float(transistors))
    # Y.append(float(math.log(transistors)))

X = np.array(X)
Y = np.array(Y)

# plot it --- exponential graph
plt.scatter(X,Y)
plt.show()

# take the Log of transistor counts (Y) to make it linear
Y = np.log(Y)

# plot it -- linear graph
plt.scatter(X, Y)
plt.show()

xDotx = X.dot(X)
xDoty = X.dot(Y)

xMean = X.mean()
yMean = Y.mean()
xSum = X.sum()

denominator =  xDotx - xMean * xSum
a = (xDoty - yMean * xSum) / denominator
b = (yMean * xDotx - xMean * xDoty) / denominator

print('a: ', a)
print('b: ', b)

# Model -- predicted Y -- yhat
yhat = a*X + b

# make prediction --- using untrained x value
someX = 2020
predY = a*someX + b

print('\nTest X (year): ', someX)
print('Predicted Y (transistor count):  ', predY)

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

print('time to double: ', np.log(2)/a, ' years')

plt.scatter(X, Y)
plt.plot(yhat)
plt.show()