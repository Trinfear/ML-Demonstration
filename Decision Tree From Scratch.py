#!python3
# decision tree from scratch

'''

node object
    contains a divider
    contains method to divide incoming data
    contains two children which are nodes or leaves
    contains a print tree method:

string question object??

numerical question object??
        

leaf object
    contains an answer
    contains a probability estimate for each type
    contains a method to generate the answer and probability dictionary
    doesn't really need to be an object, just a dictionairy the tree tacks on to the end of nodes?


generate question
    intakes a set of data
    find the answer which creates the most information gain
    returns answer

get gini
    finds the probability of getting a random guess correctly given a selection of data
    do this waited by one type being more common or assume all answers equally likely?
    gini = 1 - (num most likely response / total num data)

get adjusted gini
    intake two data sets
    get gini for both
    get average gini
    return average gini

generate tree
    intakes a dataset
    calculates gini
    gets question
    if no decrease in gini is possible,  generate a leaf node and return it
    if optimal question decreases gini, generate a node
    split data
    call generate tree on two datapoints, and set as children to current node
    return current node

test
    find all leaves and find the average gini?
    just print the probability spread of each leaf, assuming a small number of leaves?

data
    what will the data look like?
    label will be a string and always the first item
    values/features will be an array/ list of several values
    question picks one of the values at random and gets the type
    if the type is numerical then create a >= divider
    if the type is string then create a == divider
    if another type just return an error for now
    generate data procedurally or just right it out?

'''


class Node():
    def __init__(divider):
        self.divider = divider      # tuple, (location of feature, is_numeric_bool, feature_value)
        self.child_one = None
        self.child_two = None
        pass

    def predict(features):
        # intake a full feature set
        # compare to question
        # pass to child one or two
        value = features[divider[0]]
        if divider[1]:
            if value >= divider[2]:
                answer = child_one.predict(features)
            else:
                answer = child_two.predict(features)
        else:
            if value == divider[2]:
                answer = child_one.predict(features)
            else:
                answer = child_two.predict(features)
        return answer

    def draw_tree():
        # intake how many tabs
        # print that number of tabs and then the divider
        # send to both children with number of tabs plus 1
        pass


class leaf():
    def __init__():
        # all probality already calculated in generate tree, no spliting data, just generate probability table
        pass

    def predict():
        pass

    def draw_tree():
        pass


def get_gini():
    # find most common datum
    # return 1 - fraction of total data that most common datum makes up
    pass


def adj_gini():     # just have this chunk of code within generate optimal question instead of calling it here?
    # take two datasets
    # get gini for both
    # 
    pass


def optimal_question():
    # intake a full real dataset, not just posibilites
    pass


def generate_tree():
    # intake a full real dataset, not just posibilites
    pass



data_set = [['apple', 3, 'green', 'round'],
            ['lemon', 2, 'yellow', 'stretched'],
            ['pear', 3, 'green', 'stretched'],
            ['orange' 3, 'orange', 'round'],
            ['apple', 3, 'red', 'round'],
            ['apple', 2, 'yellow', 'round'],
            ['pear', 2, 'yellow', 'stretched']]

new_fruit = [2, 'red', 'round']

classifier = generate_tree(data_set)
print(classifier.predict(new_fruit))
classifier.draw_tree()



