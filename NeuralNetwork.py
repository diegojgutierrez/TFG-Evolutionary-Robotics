import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def ReLU(x):
    return x * (x > 0)

def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

# -----------------------------------------------------------------------------
# Jordan recurrent network
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
# This is an implementation of the multi-layer perceptron with retropropagation
# learning.
# -----------------------------------------------------------------------------
class JordanNetwork:

    def __init__(self, *args):

        self.shape = args
        n = len(args)
        
        self.size = 0
        # Build layers
        self.layers = []
        # Input layer (+1 unit for bias
        #              +size of oputput layer)
        self.layers.append(np.ones(self.shape[0]+1+self.shape[-1]))
        # Hidden layer(s) + output layer
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))
        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))
            self.size += self.weights[i].size
        # Reset weights
        self.reset()

    def reset(self):
        ''' Reset weights '''

        for i in range(len(self.weights)):
            Z = np.random.rand( len(self.layers[i]), len(self.layers[i+1]) )
            self.weights[i][:] = (2*Z-1)*0.25

    # change weights from genes [0,1] -> [-5,5]
    def update_weights(self, weights):

        if len(weights) != self.size:
            raise ValueError("Wrong number of parameters")
        start_index = 0
        end_index = 0
        for i in range(len(self.weights)):
            end_index = start_index + self.weights[i].size
            self.weights[i] = ((weights[start_index:end_index].reshape(self.weights[i].shape))*2.0 - 1.0)*5.0
            start_index = end_index

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer with data
        self.layers[0][0:self.shape[0]] = data
        # and output layer
        self.layers[0][self.shape[0]:-1] = self.layers[-1]

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1,len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))

        # Return output
        return self.layers[-1]