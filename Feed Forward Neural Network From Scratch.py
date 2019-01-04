#!python
# write a simple FFNN

'''

single class for network
networks contains layers which is a list of layers
each layer is a list of nodes and a list of biases for each node
each node is a list weights for each node in the previous layer

use sigmoid activation function


training
calculate output error
for each output node, calculate cost and pass it along weights leading to it and its own bias
have each node store its cumalitive costs
once all weights are updated, have each node sum its costs and update it using the same pattern

does this require nodes to be their own objects? probably easiest... but most effecient?
create a node error list which is a 2d array of the errors for each layer

'''


import numpy as np
import matplotlib.pyplot as plt


class Network():
    
    def __init__(self, layers=[2, 2, 2, 1], rate=0.00005):
        # intakes list with dimmensions of each layer, assumes first layer is input vector
        # rate is training rate to be used later
        self.rate = rate
        self.values = []    # 2d array of the values each node propegated, used for training
        self.layers = []        # each layer is a list of nodes with last object being a list of biases
        for i in range(len(layers)):
            print('dong')
            layer = layers[i]
            if i == 0:      
                prev_layer = layer
                continue
            self.layers.append(self.generate_nodes(prev_layer, layer))
            prev_layer = layer
        self.layers = np.array(self.layers)

    def generate_nodes(self, in_size, size):
        nodes = []
        for i in range(size):
            node = np.random.rand(in_size)
            nodes.append(node)
        nodes.append(np.random.rand(size))
        return nodes

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_p(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def calculate(self, inputs):          # rename this predict?
        self.values = []
        self.values.append(inputs)
        
        for layer in self.layers:
            weights = layer[:-1]
            bias = layer[-1]
            outputs = []

            for i in range(len(weights)):
                outputs.append(np.dot(inputs, weights[i]) + bias[i])  # multiply by inputs and add weights  

            self.values.append(outputs)                               # add unactivated values for use in descent??
            inputs = np.apply_along_axis(self.sigmoid, 0, outputs)    # activate output and pass on as inputs to next layer

        return inputs

    def error(self, prediction, target):
        return (prediction - target) ** 2

    def error_p(self, prediction, target):
        return 2 * (prediction - target)

    def descend(self, targets):
        layers = self.layers
        values = self.values             # this will be one longer than layers, as it includes input values which have no weight matrixes
        layers = np.flip(layers)
        values = np.flip(values)
        target_values = targets          # the target predictions of the current layer (activated)

        for i in range(len(layers)):
            current_layer = layers[i]            # the current weight set and the biases
            weight_set = current_layer[:-1]      # 2d array of weights.     list of nodes, each node is a list of weights
            bias_set = current_layer[-1]         # 1d list of biases.       each bias corresponds to one node, and is just a float
            
            input_values = np.apply_along_axis(self.sigmoid, 0, values[i + 1])       # the output of the previous layer activated
            output_values = values[i]                                                # the predictions of the current layer (unactivated)
            predictions = np.apply_along_axis(self.sigmoid, 0, output_values)        # the activated predictions of the current layer
            
            new_errors = np.zeros((len(input_values), len(output_values)))           # a 2d array of derivatives for the previous set of nodes
            
            for j in range(len(weight_set)):                    # how much of this can be done just using numpy loops instead of looping here? would that be better?
                node_weights = weight_set[j]    # list of weights
                node_bias = bias_set[j]         # single float bias
                
                node_cost_p = 2 * (predictions[j] - target_values[j])       # the derivative of the cost function with respect to the prediction
                node_pred_p = self.sigmoid_p(output_values[j])              # the derivative of the activation function with respect to the output

                layers[i][-1][j] -= self.rate * node_cost_p * node_pred_p

                for k in range(len(node_weights)):                          # iterate through each weight.  can this also be done more effeciently?
                    weight_p = input_values[k]
                    layers[i][j][k] -= self.rate * node_cost_p * node_pred_p * weight_p
                    new_errors[k][j] == node_cost_p * node_pred_p * node_weights[k]         # is this the correct formula?  the correct position?

            target_values = []
            for j in new_errors:
                target = np.sum(j)
                target = target/len(j)
                target_values.append(target)
        
        layers = np.flip(np.array(layers))
        self.layers = layers

    def train(self, input_set, output_set, rounds=70000):
        avg_out_costs = []      # list of the average costs of the output at each iteration
        data_size = len(input_set)
        if not len(output_set) == data_size:
            print("Data sizes don't match, abborting.")
            return None
        for r in range(rounds):
            # insure data is in unique order but inputs and outputs match each round
            data = list(zip(input_set, output_set))
            np.random.shuffle(data)
            input_set, output_set = zip(*data)

            avg_costs = []
            
            for i in range(data_size):
                prediction = self.calculate(input_set[i])
                self.descend(output_set[i])

                # all of this is excess for graphing
                error = np.subtract(prediction, output_set[i])
                error = np.mean(error)
                avg_costs.append(error)

            avg_costs = np.mean(avg_costs)
            avg_out_costs.append(avg_costs)

        return avg_out_costs

    def test(self, input_set, output_set):
        # run through each input/output and find the output error
        # return average cost?
        pass


def generate_values():
    # generate data for an autoencoder?
    # generate some data akin to what the starcraft bots saw? ie a softmax decision maker?
    return [[0, 1], [1, 0], [1, 1], [0, 0]], [[1], [1], [0], [0]]


classifier = Network()

train_features, train_labels = generate_values()
costs = classifier.train(train_features, train_labels)

#test_features, test_labels = generate_values()
#classifier.test(test_features, test_labels)

plt.plot(costs)
plt.show()

