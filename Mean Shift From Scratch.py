#!python3
# Mean Shift algorithm from scratch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
plt.style.use('ggplot')

centers = 4

x, y = make_blobs(n_samples=50, centers=centers, n_features=2)


class MeanShift:
    def __init__(self, radius=4):
        self.radius = radius
        self.centroids = {}

    def fit(self, data):
        pass
