#!python3
# simple use of Naive Bayes and NLTK processing

import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(file_id)), category)
             for category in movie_reviews.categories()
             for file_id in movie_reviews.fileids(category)]

random.shuffle(documents)

word_set = []

for word in movie_reviews.words():
    word_set.append(word)

word_set = nltk.FreqDist(word_set)

feature_set = list(word_set.keys())[:3000]


def get_features(doc):
    words = set(doc)
    features = {}
    for w in feature_set:
        features[w] = (w in words)

    return features


feature_sets = [(get_features(rev), category) for (rev, category) in documents]

train_set = feature_sets[:1600]  # 2000 examples in total
test_set = feature_sets[1600:]

clf = nltk.NaiveBayesClassifier.train(train_set)


print('Classifier Accuracy: ', (nltk.classify.accuracy(clf, test_set))*100)
clf.show_most_informative_features(15)
