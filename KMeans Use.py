#!python3
# Simple use case of KMeans algorithm

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
plt.style.use('ggplot')

x, y = make_blobs(n_samples=50, centers=4, n_features=2)

colors = ['g', 'r', 'c', 'b']

clf = KMeans(n_clusters=4)
clf.fit(x)

centroids = clf.cluster_centers_
labels = clf.labels_

for i in range(len(x)):
    plt.scatter(x[i][0], x[i][1], color=colors[labels[i]])
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='k', s=100, linewidths=8)

plt.show()
