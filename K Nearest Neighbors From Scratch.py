#!python
# writing K Nearest Neighbors from scratch

import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random

# create a class that classifier object can be assigned to
def k_nearest_neighbors(data, predict, k=3):
    # intakes a set of coordinates of known order and a coordinate location to be predicted
    # k is the number of neighbours taken into account
    if len(data) >= k:
        warnings.warn('More groups than value k!')

    distances = []
    for group in data:
        for features in data[group]:
            dist = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([dist, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, confidence


# get the dataset, and convert unusable forms to outliers so some value can still be reached from them in other dimmensions
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

# select metaparameters and break up the data
test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0
confidence_total = 0

# calculate the average accuracy of the classifier across a large number of runs
for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        confidence_total += confidence
        total += 1

accuracy = correct/total
confidence_average = confidence_total / total
print(accuracy, confidence_average)
