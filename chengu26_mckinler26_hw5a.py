import numpy as np
import random

# CS 421 - Homework 5: Part A
# Date: Nov. 4, 2025
# Authors: Malissa Chen and Rhiannon McKinley

# step 1:
# network structure
n_input = 4
n_hidden = 8
n_output = 1

# initialize weights
np.random.seed(0)
w_hidden = np.random.uniform(-1.0, 1.0, (n_hidden, n_input + 1))  # +1 for bias
w_output = np.random.uniform(-1.0, 1.0, (n_output, n_hidden + 1)) # +1 for bias

# step 2:
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_matrix(x, w_hidden, w_output):

    # Do one forward pass through the neural network.
    # x: input array of 4 numbers 
    # w_hidden: weights from input -> hidden layer (8 rows x 5 cols)
    # w_output: weights from hidden -> output layer (1 row x 9 cols)

    # add bias input
    x_with_bias = np.append(x, 1)  # now x has 5 numbers

    # calculate hidden layer signals using matrix multiplication
    hidden_inputs = np.dot(w_hidden, x_with_bias)  

    # apply sigmoid to get hidden layer outputs 
    hidden_outputs = sigmoid(hidden_inputs)

    # add bias to hidden layer outputs (for the output layer’s bias)
    hidden_with_bias = np.append(hidden_outputs, 1)  # now has 9 numbers

    # Calculate final output
    final_input = np.dot(w_output, hidden_with_bias)  
    final_output = sigmoid(final_input)             

    return hidden_with_bias, final_output
