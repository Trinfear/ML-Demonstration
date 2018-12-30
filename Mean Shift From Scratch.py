#!python3
# Mean Shift algorithm from scratch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
plt.style.use('ggplot')

centers = 4

x, y = make_blobs(n_samples=50, centers=centers, n_features=2)


class MeanShift:
    # kMeans but no group number input required
    def __init__(self, radius=None, norm_step=30):
        # radius is max size of each group in euclidean distance
        # norm step is used to calculate radius if none is given, roughly the average step size on the graph
        self.radius = radius
        self.centroids = {}
        self.radius_norm_step = norm_step

    def fit(self, data):
        # makes each datapoint a center for its own group and find all points within the radius
        # move the centroid to the center of all the points
        # centroids within a certain distance of eachother become identical
        # continue until now change is made in the centroids
        
        if self.radius is None:  # calculate a starting point if no radius is given as a baseline
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)  # this could be condensed to one line? less clear what things are
            self.radius = all_data_norm / self.radius_norm_step

        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]

        weights = [i for i in range(self.radius_norm_step)][::-1]

        while True:
            # iterates through centroids until the new centroids are identical to previous ones
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

            optimized = True    # assumes optimized until it finds a point where it isn't

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break

            if optimized:
                break

        self.centroids = centroids

    def predict(self, features):
        # find closest centroid and return that group
        distances = [np.linalg.norm(features - centroid) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


clf = MeanShift()
clf.fit(x)

centroids = clf.centroids

plt.scatter(x[:, 0], x[:, 1], s=150)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='x', s=150)

plt.show()
