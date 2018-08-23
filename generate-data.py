import numpy as np
import matplotlib.pyplot as plt
import csv

with open('data/sample.csv', 'wb') as csv_file:
    fileWriter = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    fileWriter.writerow(['Hey', 'Hello'])

X = []
Y = []

# print('X\tY')
# file = open('data/sample.csv').read()

# for x in range(50):
#     y = 2*x+3

#     # file.writable = True
#     # file.write(x)
#     # file.write(',')
#     # file.write(y)

#     X.append(float(x))
#     Y.append(float(y))

#     print(x, '\t', 2*x+3)

# # for line in open('data/sample.csv'):
# #     print(line)

# X = np.array(X)
# Y = np.array(Y)

# plt.scatter(X, Y)
# # plt.plot(Y)
# plt.show()

# file.close()