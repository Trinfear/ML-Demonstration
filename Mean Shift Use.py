#!python3
# Mean Shift Use

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn import preprocessing, model_selection
from sklearn.datasets.samples_generator import make_blobs
plt.style.use('ggplot')

centers = 4

x, y = make_blobs(n_samples=50, centers=centers, n_features=2)

colors = ['g', 'r', 'c', 'b']

clf = MeanShift()
clf.fit(x)

labels = clf.labels_
centroids = clf.cluster_centers_

for i in range(len(x)):
    plt.scatter(x[i][0], x[i][1], color=colors[labels[i]])
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='k', s=100, linewidths=8)
plt.show()
