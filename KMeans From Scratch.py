#!python3
# KMeans from scratch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
plt.style.use('ggplot')

centers = 4

x, y = make_blobs(n_samples=50, centers=centers, n_features=2)

# set of colors which may be used for group labelings, repeat if too many groups
colors = 5 * ['g', 'r', 'c', 'b']


class KMeans:

    def __init__(self, k=2, tol=0.001, max_iter=300):
        # k = number of groups to find, tol = amount of change before assumed correct, max iter = maximum iterations before return
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = {}
        self.classifications = {}

    def fit(self, data):
        # intakes a set of coordinates representing datapoints in n-dimmensional space
        # creates centroids at random and finds all members of their groups
        # move the centroids to the centers of the respective groups and recalculates
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for j in range(self.k):
                self.classifications[j] = []

            for feature_set in data:
                distances = [np.linalg.norm(feature_set - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(feature_set)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for centroid in self.centroids:
                original_centroid = prev_centroids[centroid]
                current_centroid = self.centroids[centroid]
                if np.sum((current_centroid - original_centroid)/original_centroid * 100) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self, features):
        # finds closest centroid and returns that group
        distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


clf = KMeans(k=centers)
clf.fit(x)

for c in clf.centroids:
    plt.scatter(clf.centroids[c][0], clf.centroids[c][1],
                marker='x', color='k', s=100, linewidths=8)

for c in clf.classifications:
    color = colors[c]
    for feature_Set in clf.classifications[c]:
        plt.scatter(feature_Set[0], feature_Set[1], color=color, s=100)


plt.show()
