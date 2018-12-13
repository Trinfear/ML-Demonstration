#!python3
# Naive Bayes Classifier

import random
import numpy as np
import math
from nltk.corpus import movie_reviews


def get_data():

    def most_freq(data_set):
        uniques, counts = np.unique(data_set, return_counts=True)
        data_tuples = [(uniques[i], counts[i]) for i in range(len(counts))]
        data_tuples = sorted(data_tuples, key=lambda x: x[1], reverse=True)
        data_tuples = data_tuples[:3000]        # optimal 3500-4500 or so? still high variation, even accross groups of 10, less variation at around 4500
        word_list = [data_tuples[i][0] for i in range(len(data_tuples))]
        return word_list

    def get_features(doc, word_list):
        words = set(doc)
        features = {}
        for word in word_list:
            features[word] = (word in words)
        return features

    documents = [(list(movie_reviews.words(file_id)), category)
                 for category in movie_reviews.categories()
                 for file_id in movie_reviews.fileids(category)]

    random.shuffle(documents)

    word_set = []
    for word in movie_reviews.words():
        word_set.append(word)

    word_set = most_freq(word_set)

    feature_sets = [(get_features(review, word_set), category) for (review, category) in documents]
    random.shuffle(feature_sets)

    train_set = feature_sets[:1600]
    test_set = feature_sets[1600:]

    return train_set, test_set, word_set


def standard_deviation(data_set):
    mean = 0
    for datum in data_set:
        mean += datum
    mean = mean / len(data_set)

    sqr_error = 0
    for datum in data_set:
        sqr_error += (datum - mean)**2
    std_dvt = math.sqrt(sqr_error/(len(data_set) -1))
    return std_dvt


class BayesClassifier():
    def __init__(self, words):
        self.word_pos_count = {}
        self.word_neg_count = {}
        self.word_count = {}
        self.word_pos_prior = {}
        self.word_neg_prior = {}
        for word in words:
            self.word_pos_count[word] = 0
            self.word_neg_count[word] = 0
            self.word_count[word] = 0
            self.word_pos_prior[word] = 0.000000001   # start these low so they dont have high values if never seen in a set
            self.word_neg_prior[word] = 0.000000001

    def train(self, data):
        for features in data:
            if features[1] == 'pos':
                for word, appears in features[0].items():
                    if appears:
                        self.word_pos_count[word] += 1
                        self.word_count[word] += 1
            else:
                for word, appears in features[0].items():
                    if appears:
                        self.word_neg_count[word] += 1
                        self.word_count[word] += 1
        
        for word in self.word_pos_prior:
            if self.word_pos_count[word] > 0:
                self.word_pos_prior[word] = self.word_pos_count[word] / self.word_count[word]
                # insure no values of zero, so something not showing up in training doesn't always lead to 0 possibility
                # could also set value equal to a small amount and divide by occurences, so chance still goes down the
                # more samples don't contain the word

        for word in self.word_neg_prior:
            if self.word_neg_count[word] > 0:
                self.word_neg_prior[word] = self.word_neg_count[word] / self.word_count[word]
                # look into multiplying occurences by something to avoid rounding to 0 errors and very small values
                # in the predict function

    def predict(self, features):
        pos_value = 1
        neg_value = 1
        for word, appears in features.items():
            if appears:
                pos_value *= self.word_pos_prior[word]
                neg_value *= self.word_neg_prior[word]
        if pos_value > neg_value:
            prediction = 'pos'
            confidence = pos_value / neg_value
        else:
            prediction = 'neg'
            confidence = neg_value / pos_value

        return prediction, confidence

    def test(self, data):
        accuracy = 0
        ave_confidence = 0
        for features in data:
            prediction, confidence = self.predict(features[0])
            ave_confidence += confidence
            if prediction == features[1]:
                accuracy += 1
        accuracy = accuracy / len(data)
        ave_confidence = ave_confidence / len(data)
        return accuracy, ave_confidence

    def most_useful_words(self):
        # find words with largest difference between pos and neg priors
        # return the higher over the lower as a value, ie x-1 chance of representing y
        pass


def test_classifier():
    clf_accuracy = 0
    ave_confidence = 0
    accuracy_dvt = []
    confidence_dvt = []
    cycles = 100
    for cycle in range(cycles):
        train_set, test_set, words = get_data()
        clf = BayesClassifier(words)
        clf.train(train_set)
        accuracy, confidence = clf.test(test_set)
        clf_accuracy += accuracy
        ave_confidence += confidence
        accuracy_dvt.append(accuracy)
        confidence_dvt.append(confidence)
        print(100 * cycle/cycles, '% done')
    
    clf_accuracy = clf_accuracy / cycles
    ave_confidence = ave_confidence / cycles
    accuracy_dvt = standard_deviation(accuracy_dvt)
    confidence_dvt = standard_deviation(confidence_dvt)
    return clf_accuracy, accuracy_dvt, ave_confidence, confidence_dvt

print(test_classifier())
    

