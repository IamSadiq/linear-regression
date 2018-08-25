import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []
for line in open('data/linear-data.csv'):
    x,y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

# random sampling --- sample = 5
train_idx = np.random.choice(len(X), 5)
Xtrain = X[train_idx]
Ytrain = Y[train_idx]

print('\nX: ', X)
print('Y: ', Y)

print('\nXtrain: ', Xtrain)
print('Ytrain: ', Ytrain)

plt.scatter(Xtrain, Ytrain)
plt.show()