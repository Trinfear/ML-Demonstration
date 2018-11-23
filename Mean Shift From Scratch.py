#!python3
# Mean Shift algorithm from scratch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
plt.style.use('ggplot')

centers = 4

x, y = make_blobs(n_samples=50, centers=centers, n_features=2)


class MeanShift:
    def __init__(self, radius=4, norm_step=30):
        self.radius = radius
        self.centroids = {}
        self.radius_norm_step = norm_step

    def fit(self, data):

        if self.radius is None:  # calculate a starting point if no bandwidth is given as a baseline
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)  # this could be condensed to one line?
            self.radius = all_data_norm / self.radius_norm_step

        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]

        weights = [i for i in range(self.radius_norm_step)][::-1]

        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for feature_set in data:
                    if np.linalg.norm(feature_set-centroid) < self.radius:
                        in_bandwidth.append(feature_set)

                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(new_centroids))

            prev_centroids = dict(centroids)

            centroids = {}

            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break

            if optimized:
                break

        self.centroids = centroids

    def predict(self):
        pass


clf = MeanShift()
clf.fit(x)

centroids = clf.centroids

plt.scatter(x[:, 0], x[:, 1], s=150)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='x', s=150)

plt.show()
