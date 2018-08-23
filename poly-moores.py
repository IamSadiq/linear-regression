import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

# load data
data = np.genfromtxt('data/moores.csv', skip_header=1, delimiter=';', names=['Year', 'Transistors', 'Frequency', 'PowerDensity', 'Cores'])

for line in data:
    x = line['Year']
    y = line['Transistors']

    x = float(x)
    X.append([1, x, x*x])
    # X.append(x)
    Y.append(float(y))

# convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

# plot the distribution
plt.scatter(X[:,1], Y)
plt.show()

# make Y linear
Y = np.log(Y)

# plot again
plt.scatter(X[:,1], Y)
plt.show()

# the solution
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
print('w: ', w)

# the Model --- prediction - Yhat
Yhat = np.dot(X, w)

# plot again adding line of best fit
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Yhat)
# plt.savefig('my-figures/fig5')
plt.show()

# Calculated R-squared
d1 = Y - Yhat
d2 = Y - Y.mean()

r_squared = 1 - d1.dot(d1)/d2.dot(d2)
print('R-Squared: ', r_squared) # r-squared of approximately One(1) means a very good Model