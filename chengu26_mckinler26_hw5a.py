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

# step 2
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_matrix(x, W_hidden, W_output):

    # Sourced from ChatGBT
    # Performs a forward pass using matrix multiplication.
    # x: 1D numpy array of shape (4,)
    # W_hidden: (8, 5)
    # W_output: (1, 9)
    # Returns: (hidden_output_with_bias, final_output)

    # add bias to input
    x_b = np.append(x, 1) # shape (5,)

    # hidden layer activation
    z_hidden = np.dot(W_hidden, x_b) # shape (8,)
    h = sigmoid(z_hidden)

    # add bias to hidden layer output
    h_b = np.append(h, 1) # shape (9,)

    # output layer
    z_output = np.dot(W_output, h_b) # shape (1,)
    y = sigmoid(z_output)

    return h_b, y
