#!python3
# simple demonstration of Naive Bayes algorithm

from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib

x = np.array([[-3, 7], [1, 5], [1, 2], [-2, 0], [2, 3], [-4, 0], [-1, 1], [1, 1],
              [-2, 2], [2, 7], [-4, 1], [-2, 7]])
y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

clf = GaussianNB()
clf.fit(x, y)
