import numpy as np
import matplotlib.pyplot as plt
import csv

N = 70
X = np.linspace(-1, 1, N)
Y = 3 * X + np.random.randn(N)

# with open('data/sample.csv', 'wb') as csv_file:
#     fileWriter = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     fileWriter.writerow([X, Y])

# for line in open('data/sample.csv'):
#     a, b = line.split(',')
#     print(a, b)


# file.writable = True
# file.write(x)

# X = np.array(X)
# Y = np.array(Y)

plt.scatter(X, Y)
# plt.plot(Y)
plt.show()

# file.close()