#!python3
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y_train = np.multiply(1.3, x_train) + 3

clf = LinearRegression()
clf.fit(x_train, y_train)

plt.scatter(x_train, y_train)
plt.plot(x_train, y_train)
plt.show()
