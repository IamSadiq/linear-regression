import numpy as np
import matplotlib.pyplot as plt

def make_poly(X, deg):
    n = len(X)
    data = [np.ones(n)]

    for d in range(deg):
        data.append(X**(d+1))
        
    return np.vstack(data).T

def fit(X, Y):
    return np.linalg.solve(X.T.dot(X), X.T.dot(Y))

def calcuate_r2(Y, Yhat):
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    return 1 - d1.dot(d1)/d2.dot(d2)

def calculate_mse(Y, Yhat):
    d = Y - Yhat
    return d.dot(d)/len(d)



''' test program '''

# load data
data = np.genfromtxt('data/moores.csv', skip_header=1, delimiter=';', names=['yrs', 'trans', 'freq', 'pd', 'cores'])

X =[]
Y =[]

for line in data:
    x = line['yrs']
    y = line['trans']

    X.append(float(x))
    Y.append(float(y))

# make transistor count linear
Y = np.log(Y)

X = np.array(X)
Y = np.array(Y)

# view the distribution
plt.scatter(X,Y)
plt.show()

degree = 20
sample = 10

train_mse_list = []
test_mse_list = []

for deg in range(degree):
    # sample the data to extract training data
    train_idx = np.random.choice(len(X), sample)
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]

    # the solution
    Xtrain_poly = make_poly(Xtrain, deg)
    w = fit(Xtrain_poly, Ytrain)

    # training prediction
    Yhat_train = Xtrain_poly.dot(w)

    # calculate and append training mse to list
    train_mse = calculate_mse(Ytrain, Yhat_train)
    train_mse_list.append(train_mse)

    # retrieve remaining data as test data
    test_idx = [idx for idx in range(len(X)) if idx not in train_idx]
    Xtest = X[test_idx]
    Ytest = Y[test_idx]

    # test
    Xtest_poly = make_poly(Xtest, deg)
    Yhat_test = Xtest_poly.dot(w)

    # calculate test mse and append to list
    test_mse = calculate_mse(Ytest, Yhat_test)
    test_mse_list.append(test_mse)

    # calculate and compare r2 on train and test data samples
    print('r2 of training sample: ', calcuate_r2(Ytrain, Yhat_train))
    print('r2 of test sample: ', calcuate_r2(Ytest, Yhat_test))

plt.plot(train_mse_list, label='training mse')
plt.plot(test_mse_list, label='test mse')
plt.legend()
plt.show()