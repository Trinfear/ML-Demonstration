#!python3
# decision tree from scratch

import numpy as np
from collections import Counter


class Node():
    def __init__(self, divider):
        self.divider = divider      # tuple, (location of feature, is_numeric_bool, feature_value)
        self.child_one = None
        self.child_two = None
        pass

    def predict(self, features):
        # intake a full feature set
        # compare to question
        # pass to child one or two
        divider = self.divider
        feature = divider[0] - 1        # take into account lack of lable.  better way to do this?
        value = features[feature]
        if divider[1]:
            if value >= divider[2]:
                answer = self.child_one.predict(features)
            else:
                answer = self.child_two.predict(features)
        else:
            if value == divider[2]:
                answer = self.child_one.predict(features)
            else:
                answer = self.child_two.predict(features)
        return answer

    def draw_tree(self, jumps):
        # intake how many tabs
        # print that number of tabs and then the divider
        # send to both children with number of tabs plus 1
        branch = ''
        for jump in range(jumps):
            branch += '\t'          # this seems like an unclean way to do this
        branch += str(self.divider)
        print(branch)
        if self.child_one:
            self.child_one.draw_tree(jumps+1)
        if self.child_two:
            self.child_two.draw_tree(jumps+1)


class Leaf():
    def __init__(self, data):
        self.probabilities = self.calculate_probabilities(data)     # array, of tuples, [(object, probability of object)]

    def calculate_probabilities(self, data):
        labels = []
        for datum in data:
            labels.append(datum[0])
        probabilities = Counter(labels).most_common()
        return probabilities

    def predict(self, features):
        return self.probabilities
    

    def draw_tree(self, jumps):
        branch = ''
        for jump in range(jumps):
            branch += '\t'          # this seems like an unclean way to do this
        branch += str(self.probabilities)
        print(branch)


def get_gini(data):
    # find most common datum
    # return 1 - fraction of total data that most common datum makes up
    labels = []
    for datum in data:
        labels.append(datum[0])

    best_guess = Counter(labels).most_common(1)     # returns [(object, count)]
    gini = 1 - (best_guess[0][1] / len(labels))
    return gini


def adj_gini(group_one, group_two):     # just have this chunk of code within generate optimal question instead of calling it here?
    # take two datasets
    # get gini for both
    # return average gini weighted by group size
    size_one = len(group_one)
    size_two = len(group_two)
    size_total = size_one + size_two

    if group_one:
        gini_one = get_gini(group_one)
    else:
        gini_one = 1
    if group_two:
        gini_two = get_gini(group_two)
    else:
        gini_two = 1

    gini_avg = (size_one * gini_one + size_two * gini_two) / size_total     
    
    return gini_avg


def split_data(data, divider):
    group_one = []
    group_two = []
    for datum in data:
        value = datum[divider[0]]
        if divider[1]:
            if value >= divider[2]:
                group_one.append(datum)
            else:
                group_two.append(datum)
        else:
            if value == divider[2]:
                group_one.append(datum)
            else:
                group_two.append(datum)
    return(group_one, group_two)


def optimal_question(data):
    best_change = 0
    best_divider = None
    current_gini = get_gini(data)
    # get rid of labels?
    for feature in range(1, len(data[0])):         # this assumes all datums are the same length
        possible_values = set([data[n][feature] for n in range(len(data))])      # this is clunky, better way to do this?
        for value in possible_values:
            if type(value) == int or type(value) == float:
                is_num = True
            else:
                is_num = False
            question = (feature, is_num, value)
            group_one, group_two = split_data(data, question)
            
            new_gini = adj_gini(group_one, group_two)
            change = current_gini - new_gini            # this is getting a rounding error? return e-17 continously
            if change < 0.001:
                change = 0      # fixes weird rounding errors.  better way to do this?

            if change > best_change:
                best_change = change
                best_divider = question
    
    return best_divider, best_change



def generate_tree(data):
    question, change = optimal_question(data)
    if change == 0:
        leaf = Leaf(data)
        return leaf
    if change > 0:
        node = Node(question)
        group_one, group_two = split_data(data, question)
        if group_one:
            node.child_one = generate_tree(group_one)
        if group_two:
            node.child_two = generate_tree(group_two)
        return node



data_set = [['apple', 3, 'green', 'round'],
            ['lemon', 2, 'yellow', 'stretched'],
            ['pear', 3, 'green', 'stretched'],
            ['orange', 3, 'orange', 'round'],
            ['apple', 3, 'red', 'round'],
            ['apple', 2, 'yellow', 'round'],
            ['pear', 2, 'yellow', 'stretched']]

new_fruit = [2, 'red', 'round']

classifier = generate_tree(data_set)
print(classifier.predict(new_fruit))
classifier.draw_tree(0)



