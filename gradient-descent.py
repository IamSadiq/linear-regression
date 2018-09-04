import numpy as np

w1 = 20
w2 = 20
learn_rate = 0.1

# for i in range(100):
#     w = w - lr*2*w
#     print(w)

# exercise
for i in range(100):
    w1 = w1 - 2*learn_rate*w1
    w2 = w2 - 4*learn_rate*(w2**3)

    print('w1: ', w1)
    print('w2: ', w2)