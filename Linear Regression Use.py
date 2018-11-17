#!python3
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random
plt.style.use('ggplot')


def create_data_set(size, variance, slope):
    x = []
    y = []
    for i in range(size):
        x.append(i)
        y.append(i * slope + random.randrange(-variance, variance))

    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    return x, y


x_train, y_train = create_data_set(50, 12, 3)


clf = LinearRegression()
clf.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

plt.scatter(x_train, y_train, color='r', label='data')
plt.plot(clf.predict(x_train.reshape(-1, 1)), color='c', label='Line of best fit')
plt.legend(loc=4)
plt.show()
